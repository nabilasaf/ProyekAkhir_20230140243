import json
import os

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (StructType, StructField, DoubleType,
                               StringType, IntegerType)

HDFS_URI = os.environ.get("HDFS_URI", "hdfs://localhost:9000").rstrip("/")
PROJ = os.environ["PROJ"]

RAW = f"{HDFS_URI}/proyek_akhir/raw"
OUT = f"{HDFS_URI}/proyek_akhir/clean/hotspot_merged"
REPORT = os.path.join(PROJ, "results", "quality_report_stage1.json")

spark = (SparkSession.builder.appName("01_ingest_quality")
         .master("local[*]")
         .config("spark.driver.memory", "3g")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

base = [
    StructField("latitude", DoubleType()),
    StructField("longitude", DoubleType()),
    StructField("brightness", DoubleType()),
    StructField("scan", DoubleType()),
    StructField("track", DoubleType()),
    StructField("acq_date", StringType()),
    StructField("acq_time", StringType()),
    StructField("satellite", StringType()),
    StructField("instrument", StringType()),
    StructField("confidence", IntegerType()),
    StructField("version", StringType()),
    StructField("bright_t31", DoubleType()),
    StructField("frp", DoubleType()),
    StructField("daynight", StringType()),
]
schema_arsip = StructType(base + [StructField("type", IntegerType())])
schema_nrt = StructType(base)

arsip = (spark.read.csv(f"{RAW}/fire_archive_M-C61_820870.csv",
                        header=True, schema=schema_arsip)
         .withColumn("source", F.lit("ARSIP")))

nrt = (spark.read.csv(f"{RAW}/fire_nrt_M-C61_820870.csv",
                      header=True, schema=schema_nrt)
       .withColumn("type", F.lit(None).cast("int"))
       .withColumn("source", F.lit("NRT")))

df = (arsip.unionByName(nrt)
      .withColumn("acq_date", F.to_date("acq_date", "yyyy-MM-dd"))
      .cache())

rep = {}
rep["baris_arsip"] = arsip.count()
rep["baris_nrt"] = nrt.count()
rep["baris_gabungan"] = df.count()

rep["null_per_kolom"] = df.select(
    [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]
).first().asDict()

key = ["latitude", "longitude", "acq_date", "acq_time", "satellite"]
dup = df.groupBy(*key).count().filter("count > 1")
rep["kelompok_duplikat"] = dup.count()
rep["baris_berlebih_duplikat"] = dup.agg(F.sum(F.col("count") - 1)).first()[0] or 0

mx_arsip = df.filter("source = 'ARSIP'").agg(F.max("acq_date")).first()[0]
mn_nrt = df.filter("source = 'NRT'").agg(F.min("acq_date")).first()[0]
rep["max_tanggal_arsip"] = str(mx_arsip)
rep["min_tanggal_nrt"] = str(mn_nrt)
rep["tumpang_tindih_tanggal"] = bool(mn_nrt <= mx_arsip)

rng = df.agg(
    F.min("latitude").alias("lat_min"), F.max("latitude").alias("lat_max"),
    F.min("longitude").alias("lon_min"), F.max("longitude").alias("lon_max"),
    F.min("confidence").alias("conf_min"), F.max("confidence").alias("conf_max"),
    F.min("frp").alias("frp_min"), F.max("frp").alias("frp_max"),
    F.min("acq_date").alias("tgl_min"), F.max("acq_date").alias("tgl_max"),
).first().asDict()
rep["rentang"] = {k: str(v) for k, v in rng.items()}

rep["frp_negatif"] = df.filter("frp < 0").count()
rep["confidence_di_luar_0_100"] = df.filter(
    "confidence < 0 OR confidence > 100").count()


def sebaran(col):
    return {str(r[col]): r["count"] for r in df.groupBy(col).count().collect()}


rep["sebaran_daynight"] = sebaran("daynight")
rep["sebaran_satellite"] = sebaran("satellite")
rep["sebaran_instrument"] = sebaran("instrument")
rep["sebaran_type"] = sebaran("type")
rep["hotspot_per_tahun"] = {
    str(r["y"]): r["count"]
    for r in df.groupBy(F.year("acq_date").alias("y")).count().orderBy("y").collect()
}

df.write.mode("overwrite").parquet(OUT)

os.makedirs(os.path.dirname(REPORT), exist_ok=True)
with open(REPORT, "w") as fh:
    json.dump(rep, fh, indent=2, default=str)

print(json.dumps(rep, indent=2, default=str))
spark.stop()