"""Run pydynbenchmark on the 9 omicverse trajectory ports.

This script wraps each port with a thin adapter that turns its specific output
into the standard ``{"pseudotime": ..., "branch": ..., "adjacency": ...}``
format expected by ``run_benchmark``.

Usage::

    python examples/benchmark_traj_ports.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
_DEV = _HERE.parent.parent

# Add every port to sys.path
for sub in [
    "py-dynbenchmark",
    "py-SCORPIUS",
    "py-tradeSeq",
    "py-condiments",
    "py-destiny",
    "py-URD",
    "py-Slingshot",
    "py-Monocle3",
    "py-TSCAN",
    "py-CytoTRACE",
]:
    sys.path.insert(0, str(_DEV / sub))

import pydynbenchmark as pyb


# ---- adapters: dataset → method-specific call → standard output ---------- #


def _kmeans_cluster(X, k=5, seed=42):
    from sklearn.cluster import KMeans
    return (KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X).labels_ + 1).astype(str)


def adapter_scorpius(ds):
    import pyscorpius
    space = pyscorpius.reduce_dimensionality(ds.counts, ndim=3)
    traj = pyscorpius.infer_trajectory(space)
    return {"pseudotime": traj["time"]}


def adapter_tscan(ds):
    import pytscan
    expr = pd.DataFrame(ds.counts.T, index=[f"g{i}" for i in range(ds.counts.shape[1])],
                        columns=[f"c{i}" for i in range(ds.counts.shape[0])])
    res = pytscan.exprmclust(expr)
    order = pytscan.TSCANorder(res, orderonly=True)
    cell_to_t = {cid: i / len(order.cell_order) for i, cid in enumerate(order.cell_order)}
    pt = np.array([cell_to_t.get(c, np.nan) for c in expr.columns])
    # Fill NaN with median for fair comparison
    if np.any(np.isnan(pt)):
        pt = np.where(np.isnan(pt), np.nanmedian(pt), pt)
    return {"pseudotime": pt, "branch": np.array([str(c) for c in res.clusterid])}


def adapter_slingshot(ds):
    import pyslingshot
    cl = _kmeans_cluster(ds.embedding, k=min(5, ds.counts.shape[0] // 30))
    sr = pyslingshot.slingshot(ds.embedding, cl, start_cluster=cl[np.argmin(ds.embedding[:, 0])])
    pt = np.nanmean(sr.pseudotime, axis=1)
    # Normalise to [0,1]
    pt = (pt - np.nanmin(pt)) / (np.nanmax(pt) - np.nanmin(pt) + 1e-12)
    return {"pseudotime": pt, "branch": cl}


def adapter_destiny(ds):
    import pydestiny
    expr = np.log2(ds.counts + 1)
    dm = pydestiny.DiffusionMap.fit(expr, sigma="local", n_eigs=5, k=min(20, ds.counts.shape[0] // 5))
    dpt = pydestiny.DPT(dm, root=int(np.argmin(ds.pseudotime)))
    return {"pseudotime": np.asarray(dpt)}


def adapter_urd(ds):
    import pydestiny, pyurd
    expr = np.log2(ds.counts + 1)
    dm = pydestiny.DiffusionMap.fit(expr, sigma="local", n_eigs=5, k=min(20, ds.counts.shape[0] // 5))
    urd = pyurd.URD(cell_names=ds.cell_ids, transitions=dm.transitions,
                    dm_eigenvectors=dm.eigenvectors, dm_eigenvalues=dm.eigenvalues)
    root_idx = np.argsort(ds.pseudotime)[: max(5, len(ds.pseudotime) // 20)].tolist()
    floods = pyurd.floodPseudotime(urd, root_cells=root_idx, n=30,
                                    minimum_cells_flooded=1, seed=42)
    res = pyurd.floodPseudotimeProcess(floods, max_frac_NA=0.9, stability_div=3)
    pt = np.full(len(ds.cell_ids), np.nan)
    for idx, c in enumerate(ds.cell_ids):
        if c in res["pseudotime"].index:
            pt[idx] = res["pseudotime"].loc[c]
    pt = np.where(np.isnan(pt), np.nanmedian(pt), pt)
    return {"pseudotime": pt}


def adapter_cytotrace(ds):
    import pycytotrace
    # counts: dataset has cells x genes; cytotrace wants genes x cells
    res = pycytotrace.cytotrace_run(ds.counts.T)
    # CytoTRACE: high = stem, low = differentiated. Pseudotime convention: low=early.
    pt = 1.0 - res.cytotrace
    return {"pseudotime": pt}


def adapter_monocle3(ds):
    import anndata as ad
    import pymonocle3 as m3
    adata = ad.AnnData(X=ds.counts.astype(np.float32))
    adata.obs_names = ds.cell_ids
    adata.var_names = [f"g{i}" for i in range(ds.counts.shape[1])]
    m3.preprocess_cds(adata, num_dim=min(20, ds.counts.shape[1] - 1))
    m3.reduce_dimension(adata, max_components=2, umap_min_dist=0.1)
    m3.cluster_cells(adata)
    m3.learn_graph(adata)
    # Find root node closest to lowest gold-pt cell
    root_cell_idx = int(np.argmin(ds.pseudotime))
    root_cell_id = ds.cell_ids[root_cell_idx]
    try:
        m3.order_cells(adata, root_cells=[root_cell_id])
        pt = np.asarray(adata.obs["pseudotime"], dtype=float)
        pt = np.where(np.isfinite(pt), pt, np.nanmedian(pt))
    except Exception:
        pt = np.linspace(0, 1, ds.counts.shape[0])
    pt = (pt - pt.min()) / (pt.max() - pt.min() + 1e-12)
    return {"pseudotime": pt}


# ---- main ---------------------------------------------------------------- #

METHODS = {
    "pySCORPIUS":  adapter_scorpius,
    "pyTSCAN":     adapter_tscan,
    "pySlingshot": adapter_slingshot,
    "pydestiny":   adapter_destiny,
    "pyURD":       adapter_urd,
    "pyCytoTRACE": adapter_cytotrace,
    "pyMonocle3":  adapter_monocle3,
    # Note: pytradeseq and pycondiments don't compute pseudotime de novo —
    # they consume someone else's. They're excluded from the benchmark.
}


def main():
    print("=" * 70)
    print("Running pydynbenchmark on omicverse trajectory ports")
    print("=" * 70)
    result = pyb.run_benchmark(METHODS, datasets=None, verbose=True)
    df = result.to_dataframe()
    df.to_csv(_HERE / "benchmark_results.csv", index=False)
    print("\n" + "=" * 70)
    print("Summary (mean across datasets, sorted by overall):")
    print("=" * 70)
    summary = result.summary()
    print(summary.round(3).to_string())
    summary.to_csv(_HERE / "benchmark_summary.csv")
    return result


if __name__ == "__main__":
    main()
