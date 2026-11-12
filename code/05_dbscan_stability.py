import itertools
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.neighbors import NearestNeighbors

PROJ = os.environ["PROJ"]
MIN_TOTAL = int(os.environ.get("MIN_TOTAL", "23"))
TAG = f"min{MIN_TOTAL}"
RES = os.path.join(PROJ, "results")
FIG = os.path.join(PROJ, "docs", "figures")
os.makedirs(FIG, exist_ok=True)

SEEDS = list(range(20))
K_RANGE = list(range(2, 11))
EPS_GRID = [round(x, 2) for x in np.arange(0.3, 2.01, 0.1)]
MINPTS_GRID = [5, 6, 8, 10]
MAX_NOISE = 0.15

dfA = pd.read_csv(os.path.join(RES, f"model_matrix_A_{TAG}.csv"))
dfB = pd.read_csv(os.path.join(RES, f"model_matrix_B_{TAG}.csv"))
assert list(dfA["shapeID"]) == list(dfB["shapeID"]), "urutan wilayah A dan B berbeda"
zcols = [c for c in dfA.columns if c.startswith("z_")]
XA, XB = dfA[zcols].to_numpy(), dfB[zcols].to_numpy()
print("wilayah:", len(dfA), "| fitur:", zcols)

ari = adjusted_rand_score
report = {"MIN_TOTAL": MIN_TOTAL, "n_wilayah": int(len(dfA)), "seeds": len(SEEDS)}

# 1) stabilitas K-Means
stab = []
for k in K_RANGE:
    LA = [KMeans(k, n_init=10, random_state=s).fit_predict(XA) for s in SEEDS]
    LB = [KMeans(k, n_init=10, random_state=s).fit_predict(XB) for s in SEEDS]
    pa = [ari(LA[i], LA[j]) for i, j in itertools.combinations(range(len(SEEDS)), 2)]
    pb = [ari(LB[i], LB[j]) for i, j in itertools.combinations(range(len(SEEDS)), 2)]
    ab = [ari(LA[s], LB[s]) for s in range(len(SEEDS))]
    stab.append({"k": k,
                 "ARI_antarseed_A_rata2": round(float(np.mean(pa)), 4),
                 "ARI_antarseed_A_min": round(float(np.min(pa)), 4),
                 "ARI_antarseed_B_rata2": round(float(np.mean(pb)), 4),
                 "ARI_antarseed_B_min": round(float(np.min(pb)), 4),
                 "ARI_A_vs_B_seed_sama_rata2": round(float(np.mean(ab)), 4),
                 "ARI_A_vs_B_seed_sama_min": round(float(np.min(ab)), 4)})
pd.DataFrame(stab).to_csv(os.path.join(RES, f"kmeans_stability_{TAG}.csv"), index=False)
report["stabilitas_kmeans"] = stab


# 2) DBSCAN grid
def kdist(X, k):
    d, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    return np.sort(d[:, k])


def eval_db(X, eps, mp):
    lab = DBSCAN(eps=eps, min_samples=mp).fit_predict(X)
    m = lab != -1
    ncl = len(set(lab[m]))
    row = {"eps": eps, "min_samples": mp, "n_klaster": ncl,
           "n_noise": int((~m).sum()), "noise_share": round(float((~m).mean()), 4),
           "silhouette": None, "davies_bouldin": None}
    if ncl >= 2 and m.sum() > ncl:
        row["silhouette"] = round(float(silhouette_score(X[m], lab[m])), 4)
        row["davies_bouldin"] = round(float(davies_bouldin_score(X[m], lab[m])), 4)
    return row, lab


grids = {}
for name, X in (("A", XA), ("B", XB)):
    rows = [eval_db(X, e, mp)[0] for mp in MINPTS_GRID for e in EPS_GRID]
    g = pd.DataFrame(rows)
    g.to_csv(os.path.join(RES, f"dbscan_grid_{name}_{TAG}.csv"), index=False)
    grids[name] = g

cand = grids["A"][(grids["A"]["n_klaster"].between(2, 10)) &
                  (grids["A"]["noise_share"] <= MAX_NOISE) &
                  (grids["A"]["silhouette"].notna())]
report["dbscan_kandidat_A_top5"] = (
    cand.sort_values("silhouette", ascending=False).head(5).to_dict("records"))

if len(cand):
    best = cand.sort_values("silhouette", ascending=False).iloc[0]
    eps_b, mp_b = float(best["eps"]), int(best["min_samples"])
    rA, labA = eval_db(XA, eps_b, mp_b)
    rB, labB = eval_db(XB, eps_b, mp_b)
    out = dfA[["shapeID", "shapeName"]].copy()
    out["dbscan_A"], out["dbscan_B"] = labA, labB
    out.to_csv(os.path.join(RES, f"dbscan_labels_{TAG}.csv"), index=False)
    report["dbscan_terpilih"] = {"eps": eps_b, "min_samples": mp_b,
                                 "A": rA, "B": rB,
                                 "ARI_A_vs_B": round(float(ari(labA, labB)), 4)}
else:
    report["dbscan_terpilih"] = ("tidak ada kombinasi dengan 2-10 klaster dan "
                                 f"noise <= {MAX_NOISE:.0%}; lihat dbscan_grid_*.csv")

# 3) gambar
fig, ax = plt.subplots(figsize=(6, 4))
for mp in (6, 10):
    ax.plot(kdist(XA, mp), label=f"A, minPts={mp}")
ax.set_xlabel("Wilayah (diurutkan)")
ax.set_ylabel("Jarak ke tetangga ke-k")
ax.set_title("Grafik k-distance (skenario A)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG, f"kdistance_A_{TAG}.png"), dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for name in ("A", "B"):
    t = pd.read_csv(os.path.join(RES, f"kmeans_k_selection_{name}_{TAG}.csv"))
    axes[0].plot(t["k"], t["wssse"], marker="o", label=f"Skenario {name}")
    axes[1].plot(t["k"], t["silhouette"], marker="o", label=f"Skenario {name}")
    axes[2].plot(t["k"], t["davies_bouldin"], marker="o", label=f"Skenario {name}")
for a, ttl in zip(axes, ["WSSSE (elbow)", "Silhouette (makin tinggi makin baik)",
                         "Davies-Bouldin (makin rendah makin baik)"]):
    a.set_title(ttl)
    a.set_xlabel("k")
    a.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG, f"kmeans_k_selection_{TAG}.png"), dpi=150)
plt.close(fig)

with open(os.path.join(RES, f"quality_report_stage5_{TAG}.json"), "w") as fh:
    json.dump(report, fh, indent=2, default=str)
print(json.dumps(report, indent=2, default=str))
