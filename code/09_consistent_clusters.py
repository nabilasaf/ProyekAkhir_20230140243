import os
import shutil

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

PROJ = os.environ["PROJ"]
MIN_TOTAL = int(os.environ.get("MIN_TOTAL", "23"))
TAG = f"min{MIN_TOTAL}"
EPS = float(os.environ.get("DBSCAN_EPS", "0.9"))
MINPTS = int(os.environ.get("DBSCAN_MINPTS", "5"))
SEED = 42
K_LIST = (2, 4, 6)
RES = os.path.join(PROJ, "results")
OUT = os.path.join(RES, "dashboard")
SIMPLIFY_DEG = float(os.environ.get("SIMPLIFY_DEG", "0.005"))

HS_HIGH, HS_MID = 20.0, 3.0
CV_UNSTABLE = 1.3
NV_DOM = 0.40

base_path = os.path.join(OUT, f"dashboard_regions_{TAG}.csv")
geo_path = os.path.join(OUT, f"adm2_clusters_{TAG}.geojson")
for p in (base_path, geo_path):
    if os.path.exists(p) and not os.path.exists(p + ".prev"):
        shutil.copy(p, p + ".prev")

base = pd.read_csv(base_path)
FEAT = ["total_hotspot", "hotspot_per_bulan", "cv_antartahun", "proporsi_malam",
        "rata2_frp", "rata2_confidence"]


def same(lab, noise_val=None):
    M = (lab[:, None] == lab[None, :]).astype(float)
    if noise_val is not None:
        nz = lab == noise_val
        M[np.ix_(nz, nz)] = 0.0
    np.fill_diagonal(M, 1.0)
    return M


def hybrid(X, lab_db):
    noise = lab_db == -1
    hy = lab_db.copy()
    n_core = int(lab_db.max()) + 1 if (~noise).any() else 0
    if noise.sum() >= 4:
        best = None
        for kk in range(2, min(5, int(noise.sum()) - 1) + 1):
            km = KMeans(kk, n_init=10, random_state=SEED).fit(X[noise])
            s = silhouette_score(X[noise], km.labels_)
            if best is None or s > best[0]:
                best = (s, kk, km.labels_)
        hy[noise] = n_core + best[2]
    elif noise.sum() > 0:
        hy[noise] = n_core
    origin = np.where(noise, "KMEANS", "DBSCAN")
    return hy, origin


def profile_name(hs, cv, nv):
    if nv >= NV_DOM:
        return "Dominan sumber non-vegetasi"
    if hs >= HS_HIGH:
        return "Intensitas tinggi"
    tier = "sedang" if hs >= HS_MID else "rendah"
    return f"Intensitas {tier}, " + ("tidak stabil" if cv >= CV_UNSTABLE else "stabil")


def name_clusters(lab, raw, noise_val=None, extra=None):
    names, hs_of = {}, {}
    for c in sorted(set(lab)):
        if noise_val is not None and c == noise_val:
            names[c] = "Noise (wilayah ekstrem)"
            continue
        m = lab == c
        hs, cv, nv = raw["hs"][m].mean(), raw["cv"][m].mean(), raw["nv"][m].mean()
        nm = profile_name(hs, cv, nv)
        if extra and c in extra:
            nm += extra[c]
        names[c], hs_of[c] = nm, hs
    cnt = {}
    for c in hs_of:
        cnt[names[c]] = cnt.get(names[c], 0) + 1
    seen = {}
    for c in sorted(hs_of, key=lambda k: -hs_of[k]):
        n = names[c]
        if cnt[n] > 1:
            seen[n] = seen.get(n, 0) + 1
            names[c] = f"{n} ({seen[n]})"
    return names


def metr(X, lab, mask):
    l, Xm = lab[mask], X[mask]
    ncl = len(set(l))
    if ncl < 2 or len(l) <= ncl:
        return None, None, ncl
    return (round(float(silhouette_score(Xm, l)), 4),
            round(float(davies_bouldin_score(Xm, l)), 4), ncl)


metric_rows = []
name_frames = []
for s in ("A", "B"):
    mm = pd.read_csv(os.path.join(RES, f"model_matrix_{s}_{TAG}.csv"))
    km = pd.read_csv(os.path.join(RES, f"kmeans_labels_{s}_{TAG}.csv"))
    assert list(mm["shapeID"]) == list(km["shapeID"]), "urutan wilayah berbeda"
    zc = [c for c in mm.columns if c.startswith("z_")]
    X = mm[zc].to_numpy()
    b = base.set_index("shapeID").loc[mm["shapeID"]]
    raw = {"hs": b[f"hotspot_per_bulan_{s}"].fillna(0).reset_index(drop=True),
           "cv": b[f"cv_antartahun_{s}"].fillna(0).reset_index(drop=True),
           "nv": b["share_nonveg"].fillna(0).reset_index(drop=True)}
    sel_path = os.path.join(RES, f"kmeans_k_selection_{s}_{TAG}.csv")
    sel = (pd.read_csv(sel_path).set_index("k")["silhouette"]
           if os.path.exists(sel_path) else {})

    lab_db = DBSCAN(eps=EPS, min_samples=MINPTS).fit_predict(X)
    noise = lab_db == -1
    common = ~noise
    hy, origin = hybrid(X, lab_db)
    M_db, M_hy = same(lab_db, -1), same(hy)

    variants = {}
    for k in K_LIST:
        lab = km[f"k{k}"].to_numpy()
        variants[f"K-Means k={k}"] = (lab, name_clusters(lab, raw), f"kmeans_k{k}_nama_{s}", k)
    variants["DBSCAN"] = (lab_db, name_clusters(lab_db, raw, noise_val=-1),
                          f"dbscan_nama_{s}", None)
    extra = {c: " (ekstrem)" for c in set(hy[origin == "KMEANS"])}
    variants["Hybrid"] = (hy, name_clusters(hy, raw, extra=extra), f"hybrid_nama_{s}", None)
    for k in K_LIST:
        C = (same(km[f"k{k}"].to_numpy()) + M_db + M_hy) / 3.0
        ens = AgglomerativeClustering(n_clusters=k, metric="precomputed",
                                      linkage="average").fit_predict(1.0 - C)
        variants[f"Ensemble k={k}"] = (ens, name_clusters(ens, raw),
                                       f"ensemble_k{k}_nama_{s}", k)

    nf = pd.DataFrame({"shapeID": mm["shapeID"]})
    nf[f"hybrid_asal_{s}"] = origin
    for vname, (lab, names, col, k) in variants.items():
        nf[col] = [names[c] for c in lab]
        for ev, mask in (("semua wilayah", np.ones(len(lab), bool)),
                         ("non-noise DBSCAN", common)):
            if vname == "DBSCAN" and ev == "semua wilayah":
                continue
            sil, dbi, ncl = metr(X, lab, mask)
            metric_rows.append({
                "skenario": s, "metode": vname.split(" k=")[0], "k": k,
                "varian": vname, "evaluasi_pada": ev,
                "n_klaster": ncl, "n_wilayah_dievaluasi": int(mask.sum()),
                "noise_share": round(float(noise.mean()), 4) if vname == "DBSCAN" else 0.0,
                "silhouette": sil, "davies_bouldin": dbi,
                "silhouette_spark_k_selection":
                    (round(float(sel[k]), 4) if (vname.startswith("K-Means") and k in sel) else None)})
        print(f"\n[{s}] {vname}: " + ", ".join(
            f"{names[c]} = {int((lab == c).sum())}" for c in sorted(names)))
    name_frames.append(nf)

new_cols = set()
for nf in name_frames:
    new_cols |= set(nf.columns) - {"shapeID"}
base = base.drop(columns=[c for c in new_cols if c in base.columns])
for nf in name_frames:
    base = base.merge(nf, on="shapeID", how="left")
for s in ("A", "B"):
    base[f"label_peta_{s}"] = base[f"kmeans_k4_nama_{s}"].fillna(base["status"])
base.to_csv(base_path, index=False)

mt = pd.DataFrame(metric_rows)
mt.to_csv(os.path.join(OUT, f"method_metrics_{TAG}.csv"), index=False)

g = gpd.read_file(os.path.join(PROJ, "data_raw", "indonesia_adm2.geojson"))[["shapeID", "geometry"]]
g = g.to_crs("EPSG:4326")
g["geometry"] = g.geometry.simplify(SIMPLIFY_DEG, preserve_topology=True)
keep = ["shapeID", "shapeName", "status", "label_peta_A", "label_peta_B", "share_nonveg"]
for s in ("A", "B"):
    keep += [f"{c}_{s}" for c in FEAT]
    keep += [f"kmeans_k{k}_nama_{s}" for k in K_LIST]
    keep += [f"ensemble_k{k}_nama_{s}" for k in K_LIST]
    keep += [f"dbscan_nama_{s}", f"hybrid_nama_{s}", f"hybrid_asal_{s}"]
keep = [c for c in keep if c in base.columns]
gj = g.merge(base[keep], on="shapeID", how="left")
try:
    gj.to_file(geo_path, driver="GeoJSON", COORDINATE_PRECISION=4)
except Exception:
    gj.to_file(geo_path, driver="GeoJSON")

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
print("\n=== METRIK (satu definisi: jarak Euclid, scikit-learn) ===")
print(mt.to_string(index=False))
print(f"\nCSV    : {base_path} ({len(base)} baris, {base.shape[1]} kolom)")
print(f"Metrik : {os.path.join(OUT, f'method_metrics_{TAG}.csv')}")
print(f"GeoJSON: {geo_path} ({os.path.getsize(geo_path) / 1e6:.1f} MB, {len(gj)} fitur)")
