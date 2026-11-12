import os

import pandas as pd

PROJ = os.environ["PROJ"]
HDFS_URI = os.environ.get("HDFS_URI", "hdfs://localhost:9000").rstrip("/")
MIN_TOTAL = int(os.environ.get("MIN_TOTAL", "23"))
TAG = f"min{MIN_TOTAL}"
RES = os.path.join(PROJ, "results")
TYPE_CSV = os.path.join(RES, "region_type_share.csv")


def build_type_share():
    from pyspark.sql import SparkSession, functions as F
    spark = (SparkSession.builder.appName("07_type_profile")
             .master("local[*]")
             .config("spark.driver.memory", "3g")
             .config("spark.hadoop.dfs.replication", "1")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    df = (spark.read.parquet(f"{HDFS_URI}/proyek_akhir/clean/hotspot_adm2")
          .filter("shapeID IS NOT NULL AND source = 'ARSIP'"))
    g = df.groupBy("shapeID").agg(
        F.count("*").alias("n_arsip"),
        *[F.sum(F.when(F.col("type") == t, 1).otherwise(0)).alias(f"n_type{t}")
          for t in (0, 1, 2, 3)])
    g.write.mode("overwrite").parquet(f"{HDFS_URI}/proyek_akhir/features/region_type_share")
    pd.DataFrame([r.asDict() for r in g.collect()]).to_csv(TYPE_CSV, index=False)
    spark.stop()


if os.environ.get("SKIP_SPARK") != "1":
    build_type_share()

t = pd.read_csv(TYPE_CSV)
for k in (1, 2, 3):
    t[f"share_type{k}"] = t[f"n_type{k}"] / t["n_arsip"]
t["share_nonveg"] = (t["n_type1"] + t["n_type2"] + t["n_type3"]) / t["n_arsip"]

c = pd.read_csv(os.path.join(RES, f"clusters_all_methods_{TAG}.csv"))
f = pd.read_csv(os.path.join(RES, "features_A.csv"))[
    ["shapeID", "total_hotspot", "hotspot_per_bulan", "proporsi_malam"]]
d = c.merge(f, on="shapeID").merge(t, on="shapeID", how="left")
d["dbscan_noise_A"] = d["dbscan_A"] == -1
d.to_csv(os.path.join(RES, f"region_type_clusters_{TAG}.csv"), index=False)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
tot = d["n_arsip"].sum()
nonveg = (d["n_type1"] + d["n_type2"] + d["n_type3"]).sum()
print(f"wilayah dianalisis: {len(d)} | baris arsip: {int(tot)} | "
      f"type bukan 0: {int(nonveg)} ({nonveg / tot:.1%})")
print("wilayah dengan share_nonveg >= 0,2:", int((d['share_nonveg'] >= 0.2).sum()),
      "| >= 0,5:", int((d['share_nonveg'] >= 0.5).sum()))

for col, ttl in (("kmeans_k4_A", "K-Means k=4 (A)"), ("kmeans_k5_A", "K-Means k=5 (A)"),
                 ("dbscan_noise_A", "Noise DBSCAN (A)")):
    g = d.groupby(col).agg(n=("shapeID", "size"),
                           share_nonveg=("share_nonveg", "mean"),
                           share_type1=("share_type1", "mean"),
                           share_type2=("share_type2", "mean"),
                           share_type3=("share_type3", "mean"),
                           malam=("proporsi_malam", "mean")).round(3)
    print(f"\n=== porsi type bukan-vegetasi per {ttl} ===")
    print(g)

rho = d[["proporsi_malam", "share_nonveg"]].corr(method="spearman").iloc[0, 1]
print(f"\nkorelasi Spearman proporsi_malam ~ share_nonveg: {rho:.3f}")

cols = ["shapeName", "n_arsip", "share_type1", "share_type2", "share_type3",
        "proporsi_malam", "kmeans_k4_A", "dbscan_A"]
big = d[d["n_arsip"] >= 30].sort_values("share_nonveg", ascending=False)
print("\n=== 15 wilayah dengan share_nonveg tertinggi (n_arsip >= 30) ===")
print(big[cols].head(15).round(3).to_string(index=False))

