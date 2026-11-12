import os

import geopandas as gpd
import pandas as pd

PROJ = os.environ["PROJ"]
MIN_TOTAL = int(os.environ.get("MIN_TOTAL", "23"))
TAG = f"min{MIN_TOTAL}"
RES = os.path.join(PROJ, "results")
OUT_DIR = os.path.join(PROJ, "results", "dashboard")
os.makedirs(OUT_DIR, exist_ok=True)
SIMPLIFY_DEG = float(os.environ.get("SIMPLIFY_DEG", "0.005"))
NON_ADMIN = {"Danau", "Danau Toba", "Hutan", "Waduk Cirata", "Wadung Kedungombo"}

FEATS = ["total_hotspot", "hotspot_per_bulan", "cv_antartahun", "proporsi_malam",
         "rata2_frp", "rata2_confidence"]

fa = pd.read_csv(os.path.join(RES, "features_A.csv"))[["shapeID", "shapeName"] + FEATS]
fb = pd.read_csv(os.path.join(RES, "features_B.csv"))[["shapeID"] + FEATS]
fa = fa.rename(columns={c: f"{c}_A" for c in FEATS})
fb = fb.rename(columns={c: f"{c}_B" for c in FEATS})
base = fa.merge(fb, on="shapeID")

cl = pd.read_csv(os.path.join(RES, f"clusters_all_methods_{TAG}.csv")).drop(columns=["shapeName"])
ts = pd.read_csv(os.path.join(RES, "region_type_share.csv"))
for k in (1, 2, 3):
    ts[f"share_type{k}"] = ts[f"n_type{k}"] / ts["n_arsip"]
ts["share_nonveg"] = (ts["n_type1"] + ts["n_type2"] + ts["n_type3"]) / ts["n_arsip"]
ts = ts[["shapeID", "n_arsip", "share_type1", "share_type2", "share_type3", "share_nonveg"]]
ex = pd.read_csv(os.path.join(RES, f"wilayah_dikeluarkan_{TAG}.csv"))[["shapeID", "alasan"]]

d = base.merge(cl, on="shapeID", how="left").merge(ts, on="shapeID", how="left")
d = d.merge(ex, on="shapeID", how="left")


def status(r):
    if not pd.isna(r["kmeans_k4_A"]):
        return "Masuk clustering"
    if r["shapeName"] in NON_ADMIN:
        return "Bukan wilayah administratif"
    if r["total_hotspot_A"] == 0:
        return "Tanpa hotspot"
    return f"Hotspot terlalu sedikit (< {MIN_TOTAL})"


d["status"] = d.apply(status, axis=1)


def name_k4(df, s):
    sub = df[df["kmeans_k4_" + s].notna()]
    prof = sub.groupby("kmeans_k4_" + s).agg(
        hs=("hotspot_per_bulan_" + s, "mean"), cv=("cv_antartahun_" + s, "mean"),
        nv=("share_nonveg", "mean"))
    names, rest = {}, set(prof.index)
    hi = prof.loc[list(rest), "hs"].idxmax()
    names[hi] = "Intensitas tinggi"
    rest.discard(hi)
    nv = prof.loc[list(rest), "nv"].idxmax()
    names[nv] = "Dominan sumber non-vegetasi"
    rest.discard(nv)
    ep = prof.loc[list(rest), "cv"].idxmax()
    names[ep] = "Intensitas rendah, tidak stabil"
    rest.discard(ep)
    for r in rest:
        names[r] = "Intensitas sedang, stabil"
    return names, prof


for s in ("A", "B"):
    names, prof = name_k4(d, s)
    print(f"\n=== penamaan K-Means k=4 skenario {s} (cek dengan profil) ===")
    print(prof.assign(nama=prof.index.map(names)).round(3))
    d[f"kmeans_k4_nama_{s}"] = d[f"kmeans_k4_{s}"].map(names)
    d[f"kmeans_k5_nama_{s}"] = d[f"kmeans_k5_{s}"].map(
        lambda v: None if pd.isna(v) else f"Klaster {int(v) + 1}")
    d[f"dbscan_nama_{s}"] = d[f"dbscan_{s}"].map(
        lambda v: None if pd.isna(v) else ("Noise" if v == -1 else f"Padat {int(v) + 1}"))
    n_core = int(d.loc[d[f"dbscan_{s}"] >= 0, f"dbscan_{s}"].max()) + 1

    def hy_name(r, s=s, n_core=n_core):
        v = r[f"hybrid_{s}"]
        if pd.isna(v):
            return None
        return f"Padat {int(v) + 1}" if r[f"hybrid_asal_{s}"] == "DBSCAN" \
            else f"Noise-K {int(v) - n_core + 1}"

    d[f"hybrid_nama_{s}"] = d.apply(hy_name, axis=1)
    d[f"ensemble_k4_nama_{s}"] = d[f"ensemble_k4_{s}"].map(
        lambda v: None if pd.isna(v) else f"Konsensus {int(v) + 1}")

d["label_peta_A"] = d["kmeans_k4_nama_A"].fillna(d["status"])
d["label_peta_B"] = d["kmeans_k4_nama_B"].fillna(d["status"])

csv_path = os.path.join(OUT_DIR, f"dashboard_regions_{TAG}.csv")
d.to_csv(csv_path, index=False)

g = gpd.read_file(os.path.join(PROJ, "data_raw", "indonesia_adm2.geojson"))[["shapeID", "geometry"]]
g = g.to_crs("EPSG:4326")
g["geometry"] = g.geometry.simplify(SIMPLIFY_DEG, preserve_topology=True)
keep = ["shapeID", "shapeName", "status", "label_peta_A", "label_peta_B",
        "kmeans_k4_nama_A", "kmeans_k4_nama_B", "kmeans_k5_nama_A", "kmeans_k5_nama_B",
        "dbscan_nama_A", "dbscan_nama_B", "hybrid_nama_A", "hybrid_nama_B",
        "ensemble_k4_nama_A", "ensemble_k4_nama_B",
        "total_hotspot_A", "hotspot_per_bulan_A", "cv_antartahun_A", "proporsi_malam_A",
        "rata2_frp_A", "rata2_confidence_A", "share_nonveg"]
gj = g.merge(d[keep], on="shapeID", how="left")
gj_path = os.path.join(OUT_DIR, f"adm2_clusters_{TAG}.geojson")
try:
    gj.to_file(gj_path, driver="GeoJSON", COORDINATE_PRECISION=4)
except Exception:
    gj.to_file(gj_path, driver="GeoJSON")

print("\n=== ringkasan ===")
print(d["status"].value_counts().to_string())
print("\nK-Means k=4 A:", d["kmeans_k4_nama_A"].value_counts().to_dict())
print("K-Means k=4 B:", d["kmeans_k4_nama_B"].value_counts().to_dict())
print("Hybrid A:", d["hybrid_nama_A"].value_counts().to_dict())
print(f"\nCSV   : {csv_path} ({os.path.getsize(csv_path) / 1e3:.0f} KB, {len(d)} baris, {d.shape[1]} kolom)")
print(f"GeoJSON: {gj_path} ({os.path.getsize(gj_path) / 1e6:.1f} MB, {len(gj)} fitur)")
