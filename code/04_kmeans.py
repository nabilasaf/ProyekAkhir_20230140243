import csv
import json
import os

import numpy as np
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.functions import vector_to_array

HDFS_URI = os.environ.get("HDFS_URI", "hdfs://localhost:9000").rstrip("/")
PROJ = os.environ["PROJ"]
MIN_TOTAL = int(os.environ.get("MIN_TOTAL", "23"))
SEED = 42
K_RANGE = list(range(2, 11))
TAG = f"min{MIN_TOTAL}"

FEAT_DIR = f"{HDFS_URI}/proyek_akhir/features"
RES_DIR = os.path.join(PROJ, "results")
os.makedirs(RES_DIR, exist_ok=True)

FEATS = ["hotspot_per_bulan", "cv_antartahun", "proporsi_malam",
         "rata2_frp", "rata2_confidence"]
LOG_FEATS = ["hotspot_per_bulan", "rata2_frp"]

spark = (SparkSession.builder.appName("04_kmeans")
         .master("local[*]")
         .config("spark.driver.memory", "3g")
         .config("spark.hadoop.dfs.replication", "1")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

fa = spark.read.parquet(f"{FEAT_DIR}/region_features_A")
fb = spark.read.parquet(f"{FEAT_DIR}/region_features_B")

ok = F.col("total_hotspot") >= MIN_TOTAL
for c in FEATS:
    ok = ok & F.col(c).isNotNull()
elig = fa.filter(ok).select("shapeID").join(fb.filter(ok).select("shapeID"), "shapeID")
n_elig = elig.count()

ex = (fa.select("shapeID", "shapeName", F.col("total_hotspot").alias("total_A"))
      .join(fb.select("shapeID", F.col("total_hotspot").alias("total_B")), "shapeID")
      .join(elig.withColumn("ok", F.lit(1)), "shapeID", "left")
      .filter("ok IS NULL")
      .withColumn("alasan", F.when(F.col("total_A") < MIN_TOTAL,
                                   F.lit(f"total_hotspot_A < {MIN_TOTAL}"))
                  .otherwise(F.lit("fitur kosong")))
      .orderBy("total_A"))
with open(os.path.join(RES_DIR, f"wilayah_dikeluarkan_{TAG}.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["shapeID", "shapeName", "total_A", "total_B", "alasan"])
    for r in ex.collect():
        w.writerow([r["shapeID"], r["shapeName"], r["total_A"], r["total_B"], r["alasan"]])
n_ex = ex.count()
print(f"wilayah masuk clustering: {n_elig} | dikeluarkan: {n_ex}")


def run(name):
    src = spark.read.parquet(f"{FEAT_DIR}/region_features_{name}").join(elig, "shapeID")
    for c in LOG_FEATS:
        src = src.withColumn(c + "_x", F.log1p(F.col(c)))
    cols = [c + "_x" if c in LOG_FEATS else c for c in FEATS]
    d = VectorAssembler(inputCols=cols, outputCol="raw").transform(src)
    d = (StandardScaler(inputCol="raw", outputCol="features", withMean=True, withStd=True)
         .fit(d).transform(d).cache())

    z = d.withColumn("z", vector_to_array("features"))
    z = z.select("shapeID", "shapeName",
                 *[z["z"][i].alias("z_" + c) for i, c in enumerate(FEATS)])
    z.write.mode("overwrite").parquet(f"{FEAT_DIR}/model_matrix_{name}_{TAG}")
    with open(os.path.join(RES_DIR, f"model_matrix_{name}_{TAG}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["shapeID", "shapeName"] + ["z_" + c for c in FEATS])
        for r in z.orderBy("shapeID").collect():
            w.writerow([r["shapeID"], r["shapeName"]] + [r["z_" + c] for c in FEATS])

    rows = d.orderBy("shapeID").select("shapeID", "shapeName", "features").collect()
    ids = [r["shapeID"] for r in rows]
    names = [r["shapeName"] for r in rows]
    X = np.array([r["features"].toArray() for r in rows])

    ev = ClusteringEvaluator(featuresCol="features", predictionCol="prediction",
                             metricName="silhouette", distanceMeasure="squaredEuclidean")
    table, labels, models = [], {}, {}
    for k in K_RANGE:
        model = KMeans(k=k, seed=SEED, featuresCol="features",
                       maxIter=100).fit(d)
        pred = model.transform(d)
        lab = [r[0] for r in pred.orderBy("shapeID").select("prediction").collect()]
        table.append({"k": k,
                      "wssse": float(model.summary.trainingCost),
                      "silhouette": float(ev.evaluate(pred)),
                      "davies_bouldin": float(davies_bouldin_score(X, lab))})
        labels[k] = lab
        models[k] = pred
    best_k = max(table, key=lambda t: t["silhouette"])["k"]

    with open(os.path.join(RES_DIR, f"kmeans_k_selection_{name}_{TAG}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["k", "wssse", "silhouette", "davies_bouldin"])
        for t in table:
            w.writerow([t["k"], t["wssse"], t["silhouette"], t["davies_bouldin"]])
    with open(os.path.join(RES_DIR, f"kmeans_labels_{name}_{TAG}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["shapeID", "shapeName"] + [f"k{k}" for k in K_RANGE])
        for i in range(len(ids)):
            w.writerow([ids[i], names[i]] + [labels[k][i] for k in K_RANGE])

    prof = (models[best_k].groupBy("prediction")
            .agg(F.count("*").alias("n_wilayah"),
                 *[F.round(F.avg(c), 3).alias(c) for c in FEATS + ["total_hotspot"]])
            .orderBy("prediction").collect())
    return {"tabel_k": table, "k_terbaik_silhouette": best_k,
            "profil_klaster_k_terbaik": [r.asDict() for r in prof]}, labels


rep = {"MIN_TOTAL": MIN_TOTAL, "wilayah_masuk": n_elig, "wilayah_dikeluarkan": n_ex,
       "seed": SEED}
rep["A"], lab_a = run("A")
rep["B"], lab_b = run("B")
rep["ARI_A_vs_B_per_k"] = {
    str(k): round(float(adjusted_rand_score(lab_a[k], lab_b[k])), 4) for k in K_RANGE}
kA, kB = rep["A"]["k_terbaik_silhouette"], rep["B"]["k_terbaik_silhouette"]
rep["ARI_A_vs_B_pada_k_terbaik_masing2"] = round(
    float(adjusted_rand_score(lab_a[kA], lab_b[kB])), 4)

with open(os.path.join(RES_DIR, f"quality_report_stage4_{TAG}.json"), "w") as fh:
    json.dump(rep, fh, indent=2, default=str)
print(json.dumps(rep, indent=2, default=str))
spark.stop()
