import json
import os

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score,
                             silhouette_score)

PROJ = os.environ["PROJ"]
MIN_TOTAL = int(os.environ.get("MIN_TOTAL", "23"))
TAG = f"min{MIN_TOTAL}"
EPS = float(os.environ.get("DBSCAN_EPS", "0.9"))
MINPTS = int(os.environ.get("DBSCAN_MINPTS", "5"))
K_LIST = [4, 5, 6]
SEED = 42
RES = os.path.join(PROJ, "results")
FEATS = ["hotspot_per_bulan", "cv_antartahun", "proporsi_malam",
         "rata2_frp", "rata2_confidence"]

ari = adjusted_rand_score


def load(s):
    m = pd.read_csv(os.path.join(RES, f"model_matrix_{s}_{TAG}.csv"))
    km = pd.read_csv(os.path.join(RES, f"kmeans_labels_{s}_{TAG}.csv"))
    ft = pd.read_csv(os.path.join(RES, f"features_{s}.csv"))
    assert list(m["shapeID"]) == list(km["shapeID"]), "urutan wilayah berbeda"
    z = [c for c in m.columns if c.startswith("z_")]
    return m, m[z].to_numpy(), km, ft


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
    k_noise = 0
    if noise.sum() >= 4:
        best = None
        for kk in range(2, min(5, int(noise.sum()) - 1) + 1):
            km = KMeans(kk, n_init=10, random_state=SEED).fit(X[noise])
            s = silhouette_score(X[noise], km.labels_)
            if best is None or s > best[0]:
                best = (s, kk, km.labels_)
        hy[noise] = n_core + best[2]
        k_noise = best[1]
    elif noise.sum() > 0:
        hy[noise] = n_core
        k_noise = 1
    origin = np.where(noise, "KMEANS", "DBSCAN")
    return hy, origin, k_noise


def metrics(X, lab, mask):
    l, Xm = lab[mask], X[mask]
    ncl = len(set(l))
    if ncl < 2 or len(l) <= ncl:
        return None, None, ncl
    return (round(float(silhouette_score(Xm, l)), 4),
            round(float(davies_bouldin_score(Xm, l)), 4), ncl)


data = {s: load(s) for s in ("A", "B")}
n = len(data["A"][0])
report = {"MIN_TOTAL": MIN_TOTAL, "n_wilayah": n,
          "dbscan": {"eps": EPS, "min_samples": MINPTS}}
rows, agree, allab = [], [], {}
labs_store = {}

for s in ("A", "B"):
    m, X, km, ft = data[s]
    lab_db = DBSCAN(eps=EPS, min_samples=MINPTS).fit_predict(X)
    noise = lab_db == -1
    hy, origin, k_noise = hybrid(X, lab_db)
    common = ~noise
    report["dbscan"][s] = {"n_klaster": int(len(set(lab_db[~noise]))),
                           "n_noise": int(noise.sum()),
                           "noise_share": round(float(noise.mean()), 4),
                           "k_klaster_noise_hybrid": k_noise}
    labs_store[s] = {"DBSCAN": lab_db, "Hybrid": hy}

    M_db, M_hy = same(lab_db, -1), same(hy)
    out = m[["shapeID", "shapeName"]].copy()
    out[f"dbscan_{s}"], out[f"hybrid_{s}"], out[f"hybrid_asal_{s}"] = lab_db, hy, origin

    for k in K_LIST:
        kml = km[f"k{k}"].to_numpy()
        C = (same(kml) + M_db + M_hy) / 3.0
        ens = AgglomerativeClustering(n_clusters=k, metric="precomputed",
                                      linkage="average").fit_predict(1.0 - C)
        out[f"kmeans_k{k}_{s}"], out[f"ensemble_k{k}_{s}"] = kml, ens
        labs_store[s][f"KMeans_k{k}"] = kml
        labs_store[s][f"Ensemble_k{k}"] = ens

        methods = {"K-Means": kml, "DBSCAN": lab_db, "Hybrid": hy, "Ensemble": ens}
        for name, lab in methods.items():
            for ev, mask in (("semua_wilayah", np.ones(n, bool)),
                             ("non_noise_DBSCAN", common)):
                if name == "DBSCAN" and ev == "semua_wilayah":
                    continue
                sil, dbi, ncl = metrics(X, lab, mask)
                rows.append({"skenario": s, "k_acuan": k, "metode": name,
                             "evaluasi_pada": ev,
                             "n_wilayah_dievaluasi": int(mask.sum()),
                             "n_klaster": ncl,
                             "noise_share": report["dbscan"][s]["noise_share"]
                             if name == "DBSCAN" else 0.0,
                             "silhouette": sil, "davies_bouldin": dbi})
        names = list(methods)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                agree.append({"skenario": s, "k_acuan": k,
                              "metode_1": names[i], "metode_2": names[j],
                              "ARI_non_noise": round(float(ari(
                                  methods[names[i]][common], methods[names[j]][common])), 4)})
    allab[s] = out

    prof = []
    merged = ft.set_index("shapeID").loc[m["shapeID"]].reset_index()
    for k in K_LIST:
        merged["_c"] = km[f"k{k}"].to_numpy()
        g = merged.groupby("_c")
        p = g[FEATS].mean().round(3)
        p["total_hotspot_median"] = g["total_hotspot"].median()
        p.insert(0, "n_wilayah", g.size())
        p.insert(0, "klaster", p.index)
        p.insert(0, "k", k)
        prof.append(p)
    pd.concat(prof).to_csv(os.path.join(RES, f"kmeans_profile_{s}_{TAG}.csv"), index=False)

ab = {name: round(float(ari(labs_store["A"][name], labs_store["B"][name])), 4)
      for name in labs_store["A"]}
report["ARI_A_vs_B_per_metode"] = ab

cmp_df = pd.DataFrame(rows)
cmp_df.to_csv(os.path.join(RES, f"comparison_metrics_{TAG}.csv"), index=False)
pd.DataFrame(agree).to_csv(os.path.join(RES, f"method_agreement_{TAG}.csv"), index=False)
both = allab["A"].merge(allab["B"].drop(columns=["shapeName"]), on="shapeID")
both.to_csv(os.path.join(RES, f"clusters_all_methods_{TAG}.csv"), index=False)

with open(os.path.join(RES, f"quality_report_stage6_{TAG}.json"), "w") as fh:
    json.dump(report, fh, indent=2, default=str)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print(json.dumps(report, indent=2, default=str))
for s in ("A", "B"):
    print(f"\n=== PERBANDINGAN METODE, skenario {s} ===")
    print(cmp_df[cmp_df["skenario"] == s].drop(columns=["skenario"]).to_string(index=False))
print("\n=== PROFIL K-MEANS skenario A ===")
print(pd.read_csv(os.path.join(RES, f"kmeans_profile_A_{TAG}.csv")).to_string(index=False))
