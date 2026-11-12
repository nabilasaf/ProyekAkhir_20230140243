import csv
import json
import os
from itertools import combinations

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType, StructField, StringType

HDFS_URI = os.environ.get("HDFS_URI", "hdfs://localhost:9000").rstrip("/")
PROJ = os.environ["PROJ"]

IN_DATA = f"{HDFS_URI}/proyek_akhir/clean/hotspot_adm2"
OUT_DIR = f"{HDFS_URI}/proyek_akhir/features"
GEOJSON = os.path.join(PROJ, "data_raw", "indonesia_adm2.geojson")
RES_DIR = os.path.join(PROJ, "results")
REPORT = os.path.join(RES_DIR, "quality_report_stage3.json")

Y0, Y1 = 2003, 2025
COLS = ["shapeID", "shapeName", "skenario", "total_hotspot", "hotspot_per_tahun",
        "hotspot_per_bulan", "cv_antartahun", "proporsi_malam",
        "rata2_frp", "rata2_confidence"]
CORE = ["hotspot_per_bulan", "cv_antartahun", "proporsi_malam",
        "rata2_frp", "rata2_confidence"]

spark = (SparkSession.builder.appName("03_build_features")
         .master("local[*]")
         .config("spark.driver.memory", "3g")
         .config("spark.hadoop.dfs.replication", "1")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

df = (spark.read.parquet(IN_DATA)
      .filter(F.col("shapeID").isNotNull())
      .select("shapeID", "acq_date", "daynight", "frp", "confidence", "source")
      .cache())

with open(GEOJSON) as fh:
    g = json.load(fh)
regions = spark.createDataFrame(
    [(f["properties"]["shapeID"], f["properties"]["shapeName"]) for f in g["features"]],
    StructType([StructField("shapeID", StringType()),
                StructField("shapeName", StringType())]))
years = spark.range(Y0, Y1 + 1).select(F.col("id").cast("int").alias("tahun"))
grid = regions.select("shapeID").crossJoin(years)


def build(dfs, name):
    dmin, dmax = dfs.agg(F.min("acq_date"), F.max("acq_date")).first()
    months = ((dmax - dmin).days + 1) / (365.25 / 12)

    agg = dfs.groupBy("shapeID").agg(
        F.count("*").alias("total_hotspot"),
        F.sum(F.when(F.col("daynight") == "N", 1).otherwise(0)).alias("n_malam"),
        F.avg("frp").alias("rata2_frp"),
        F.avg("confidence").alias("rata2_confidence"))

    yearly = (dfs.withColumn("tahun", F.year("acq_date"))
              .filter(F.col("tahun").between(Y0, Y1))
              .groupBy("shapeID", "tahun").count())
    ytab = (grid.join(yearly, ["shapeID", "tahun"], "left")
            .na.fill(0, ["count"])
            .groupBy("shapeID")
            .agg(F.avg("count").alias("hotspot_per_tahun"),
                 F.stddev_samp("count").alias("sd_tahunan")))

    out = (regions.join(agg, "shapeID", "left")
           .join(ytab, "shapeID", "left")
           .na.fill(0, ["total_hotspot", "n_malam"])
           .withColumn("skenario", F.lit(name))
           .withColumn("hotspot_per_bulan", F.col("total_hotspot") / F.lit(months))
           .withColumn("cv_antartahun",
                       F.when(F.col("hotspot_per_tahun") > 0,
                              F.col("sd_tahunan") / F.col("hotspot_per_tahun")))
           .withColumn("proporsi_malam",
                       F.when(F.col("total_hotspot") > 0,
                              F.col("n_malam") / F.col("total_hotspot")))
           .select(*COLS)
           .cache())

    out.write.mode("overwrite").parquet(f"{OUT_DIR}/region_features_{name}")

    os.makedirs(RES_DIR, exist_ok=True)
    with open(os.path.join(RES_DIR, f"features_{name}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        for r in out.orderBy("shapeID").toLocalIterator():
            w.writerow([r[c] for c in COLS])

    rep = {
        "periode": f"{dmin} s/d {dmax}",
        "lama_periode_bulan": round(months, 3),
        "baris_hotspot_dipakai": dfs.count(),
        "jumlah_wilayah": out.count(),
        "wilayah_tanpa_hotspot": out.filter("total_hotspot = 0").count(),
    }
    rep["wilayah_dengan_total_kurang_dari"] = {
        str(t): out.filter(F.col("total_hotspot") < t).count()
        for t in [1, 10, 23, 50, 100, 250]}
    qs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    rep["kuantil_total_hotspot"] = dict(zip(
        [str(q) for q in qs], out.approxQuantile("total_hotspot", qs, 0.0)))
    ok = out.filter("total_hotspot > 0 AND cv_antartahun IS NOT NULL")
    rep["korelasi_pearson_inti"] = {
        f"{a} ~ {b}": round(ok.stat.corr(a, b), 4)
        for a, b in combinations(CORE, 2)}
    rep["korelasi_pearson_profil"] = {
        "hotspot_per_tahun ~ hotspot_per_bulan":
            round(ok.stat.corr("hotspot_per_tahun", "hotspot_per_bulan"), 4)}
    return rep


report = {}
report["A"] = build(df.filter("source = 'ARSIP'"), "A")
report["B"] = build(df, "B")

mon = (df.filter("source = 'ARSIP'")
       .filter(F.year("acq_date").between(Y0, Y1))
       .groupBy(F.month("acq_date").alias("bulan")).count().orderBy("bulan"))
report["profil_bulanan_nasional_A_2003_2025"] = {
    str(r["bulan"]): r["count"] for r in mon.collect()}

with open(REPORT, "w") as fh:
    json.dump(report, fh, indent=2, default=str)
print(json.dumps(report, indent=2, default=str))
spark.stop()
