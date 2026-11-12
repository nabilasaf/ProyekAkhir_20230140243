import json
import os

import geopandas as gpd
import numpy as np
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (StructType, StructField, DoubleType,
                               StringType)

HDFS_URI = os.environ.get("HDFS_URI", "hdfs://localhost:9000").rstrip("/")
PROJ = os.environ["PROJ"]

IN_PARQUET = f"{HDFS_URI}/proyek_akhir/clean/hotspot_merged"
OUT_MAP = f"{HDFS_URI}/proyek_akhir/clean/coord_adm2_map"
OUT_DATA = f"{HDFS_URI}/proyek_akhir/clean/hotspot_adm2"
GEOJSON = os.path.join(PROJ, "data_raw", "indonesia_adm2.geojson")
REPORT = os.path.join(PROJ, "results", "quality_report_stage2.json")

spark = (SparkSession.builder.appName("02_spatial_mapping")
         .master("local[*]")
         .config("spark.driver.memory", "4g")
         .config("spark.hadoop.dfs.replication", "1")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

df = spark.read.parquet(IN_PARQUET)

# 1) koordinat unik -> numpy (tanpa toPandas)
coords = df.select("latitude", "longitude").distinct()
lats, lons = [], []
for r in coords.toLocalIterator():
    lats.append(r["latitude"])
    lons.append(r["longitude"])
lats = np.array(lats, dtype="float64")
lons = np.array(lons, dtype="float64")
print("koordinat unik:", len(lats))

# 2) spatial join dengan poligon ADM2
polys = gpd.read_file(GEOJSON)[["shapeID", "shapeName", "geometry"]]
polys = polys.to_crs("EPSG:4326")
pts = gpd.GeoDataFrame(
    {"latitude": lats, "longitude": lons},
    geometry=gpd.points_from_xy(lons, lats), crs="EPSG:4326")

joined = gpd.sjoin(pts, polys, how="left", predicate="intersects")
n_multi = int(joined.index.duplicated().sum())
joined = joined[~joined.index.duplicated(keep="first")]
n_unmatched = int(joined["shapeID"].isna().sum())
print("titik unik tanpa wilayah:", n_unmatched, "| cocok >1 poligon:", n_multi)

# 3) tabel pemetaan -> Spark -> HDFS
rows = list(zip(joined["latitude"].tolist(), joined["longitude"].tolist(),
                [None if x != x else x for x in joined["shapeID"].tolist()]))
map_schema = StructType([
    StructField("latitude", DoubleType()),
    StructField("longitude", DoubleType()),
    StructField("shapeID", StringType()),
])
mp = spark.createDataFrame(rows, schema=map_schema)
mp.write.mode("overwrite").parquet(OUT_MAP)
mp = spark.read.parquet(OUT_MAP)

# 4) gabungkan ke seluruh data (left join, tidak ada baris hilang)
out = df.join(mp, on=["latitude", "longitude"], how="left").cache()
n_total = out.count()
out.write.mode("overwrite").parquet(OUT_DATA)

# 5) laporan
rep = {
    "baris_masuk": df.count(),
    "baris_keluar": n_total,
    "koordinat_unik": int(len(lats)),
    "koordinat_unik_tanpa_wilayah": n_unmatched,
    "koordinat_cocok_lebih_dari_satu_poligon": n_multi,
    "baris_tanpa_wilayah": out.filter(F.col("shapeID").isNull()).count(),
    "wilayah_dengan_hotspot": out.filter(F.col("shapeID").isNotNull())
                                  .select("shapeID").distinct().count(),
    "total_wilayah_geojson": int(len(polys)),
}
rep["tanpa_wilayah_per_source"] = {
    r["source"]: r["count"] for r in
    out.filter(F.col("shapeID").isNull()).groupBy("source").count().collect()}
rep["tanpa_wilayah_per_type"] = {
    str(r["type"]): r["count"] for r in
    out.filter(F.col("shapeID").isNull()).groupBy("type").count().collect()}
rep["dengan_wilayah_per_type"] = {
    str(r["type"]): r["count"] for r in
    out.filter(F.col("shapeID").isNotNull()).groupBy("type").count().collect()}
per_wil = (out.filter(F.col("shapeID").isNotNull()).groupBy("shapeID").count())
rep["hotspot_per_wilayah"] = {
    k: v for k, v in zip(
        ["min", "median", "maks"],
        [per_wil.agg(F.min("count")).first()[0],
         per_wil.approxQuantile("count", [0.5], 0.01)[0],
         per_wil.agg(F.max("count")).first()[0]])}
rep["wilayah_hotspot_kurang_dari_10"] = per_wil.filter("count < 10").count()

os.makedirs(os.path.dirname(REPORT), exist_ok=True)
with open(REPORT, "w") as fh:
    json.dump(rep, fh, indent=2, default=str)
print(json.dumps(rep, indent=2, default=str))
spark.stop()
