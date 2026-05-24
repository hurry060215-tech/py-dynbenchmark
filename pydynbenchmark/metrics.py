"""Trajectory-inference metrics — Python ports of dyneval's four families.

Each metric returns a scalar in [0, 1] where 1 = perfect prediction.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr
from sklearn.metrics import f1_score


# ----- HIM: Hamming-Ipsen-Mikhailov on milestone-graph adjacencies --------- #


def him_distance(adj_gold: np.ndarray, adj_pred: np.ndarray) -> float:
    """HIM ≈ 1 − ½ (Hamming + Ipsen-Mikhailov).

    Both graphs must have the same node set. Returns score in [0, 1].
    """
    A1 = np.asarray(adj_gold, dtype=np.float64)
    A2 = np.asarray(adj_pred, dtype=np.float64)
    if A1.shape != A2.shape:
        # Pad smaller to match
        n = max(A1.shape[0], A2.shape[0])
        A1p = np.zeros((n, n)); A1p[:A1.shape[0], :A1.shape[1]] = A1
        A2p = np.zeros((n, n)); A2p[:A2.shape[0], :A2.shape[1]] = A2
        A1, A2 = A1p, A2p

    # Hamming
    n = A1.shape[0]
    h = np.abs(A1 - A2).sum() / max(n * (n - 1), 1)

    # Ipsen-Mikhailov from spectra of Laplacians
    def laplacian_eigvals(A):
        D = np.diag(A.sum(axis=1))
        L = D - A
        return np.sort(np.linalg.eigvalsh(L))

    s1 = laplacian_eigvals(A1)
    s2 = laplacian_eigvals(A2)
    # Sort + element-wise distance — coarse but sufficient
    im = np.mean(np.abs(s1 - s2)) / max(s1.max(), s2.max(), 1e-12)

    h_norm = min(1.0, h)
    im_norm = min(1.0, im)
    return float(1.0 - 0.5 * (h_norm + im_norm))


# ----- F1_branches: branch-membership F1 ---------------------------------- #


def _best_label_alignment(gold: np.ndarray, pred: np.ndarray) -> dict:
    """Hungarian assignment of predicted labels to gold via confusion matrix."""
    from scipy.optimize import linear_sum_assignment
    gold_u = np.unique(gold)
    pred_u = np.unique(pred)
    M = np.zeros((len(gold_u), len(pred_u)))
    for i, g in enumerate(gold_u):
        for j, p in enumerate(pred_u):
            M[i, j] = np.sum((gold == g) & (pred == p))
    # Maximise overlap → negate for minimise
    row, col = linear_sum_assignment(-M)
    mapping = {pred_u[c]: gold_u[r] for r, c in zip(row, col)}
    return mapping


def f1_branches(gold_branch, pred_branch) -> float:
    """Mean F1 score across gold branches after Hungarian label-matching."""
    gold = np.asarray(gold_branch)
    pred = np.asarray(pred_branch)
    if len(gold) != len(pred):
        return 0.0
    mapping = _best_label_alignment(gold, pred)
    pred_mapped = np.array([mapping.get(p, "_unmatched_") for p in pred])
    common = np.unique(gold)
    # Treat as multi-class
    return float(f1_score(gold, pred_mapped, labels=common, average="macro", zero_division=0))


# ----- cor_dist: Pearson correlation between geodesic distance matrices --- #


def _geo_dist_from_pseudotime(pt: np.ndarray, branch: np.ndarray) -> np.ndarray:
    """Approx geodesic distance from per-cell pseudotime + branch label."""
    n = len(pt)
    D = np.zeros((n, n))
    branches = np.unique(branch)
    for b in branches:
        mask = branch == b
        idx = np.where(mask)[0]
        # within-branch: |pt_i - pt_j|
        for i in idx:
            for j in idx:
                D[i, j] = abs(pt[i] - pt[j])
    # cross-branch: 0.5 + dt — a crude penalty for jumping branches
    for i in range(n):
        for j in range(n):
            if branch[i] != branch[j]:
                D[i, j] = abs(pt[i] - pt[j]) + 0.5
    return D


def cor_dist(
    gold_pt: np.ndarray,
    gold_branch: np.ndarray,
    pred_pt: np.ndarray,
    pred_branch: np.ndarray,
) -> float:
    """Pearson correlation of vectorised geodesic-distance matrices.

    A score of 1 means cell-cell geodesic distances are perfectly preserved.
    """
    D_g = _geo_dist_from_pseudotime(np.asarray(gold_pt), np.asarray(gold_branch))
    D_p = _geo_dist_from_pseudotime(np.asarray(pred_pt), np.asarray(pred_branch))
    triu = np.triu_indices_from(D_g, k=1)
    r, _ = pearsonr(D_g[triu], D_p[triu])
    return float(max(0.0, r))   # negative correlations clamp to 0


# ----- wcor_features: weighted correlation of top-DE features ------------- #


def wcor_features(
    counts: np.ndarray,
    gold_pt: np.ndarray,
    pred_pt: np.ndarray,
    n_features: int = 20,
) -> float:
    """For top-DE genes (by gold pseudotime correlation), measure how well their
    predicted-pseudotime correlations match. Score = mean |Pearson(g_i_pred, g_i_gold)|.
    """
    gold_pt = np.asarray(gold_pt)
    pred_pt = np.asarray(pred_pt)
    # Gold-side DE: genes most correlated with gold_pt
    gold_r = np.array([pearsonr(counts[:, j], gold_pt)[0]
                       for j in range(counts.shape[1])])
    gold_r = np.nan_to_num(gold_r, nan=0.0)
    top_idx = np.argsort(np.abs(gold_r))[::-1][:n_features]
    out = []
    for j in top_idx:
        rp, _ = pearsonr(counts[:, j], pred_pt)
        if np.isfinite(rp):
            out.append(abs(rp))
    return float(np.mean(out)) if out else 0.0


# ----- aggregate ----------------------------------------------------------- #


def calculate_metrics(
    dataset,
    pred_pt: np.ndarray | None = None,
    pred_branch: np.ndarray | None = None,
    pred_adj: np.ndarray | None = None,
    gold_adj: np.ndarray | None = None,
) -> dict:
    """Compute all available metrics in one go.

    Args:
        dataset: a ``Dataset`` instance.
        pred_pt: predicted per-cell pseudotime; if None, that family is skipped.
        pred_branch: predicted per-cell branch labels.
        pred_adj, gold_adj: optional milestone-graph adjacency matrices for HIM.
    """
    out: dict = {}

    if pred_pt is not None and dataset.pseudotime is not None:
        # cor_dist needs branch too — fall back to all-one branch if absent
        gb = dataset.branch
        pb = pred_branch if pred_branch is not None else np.array(["L1"] * len(pred_pt))
        out["cor_dist"] = cor_dist(dataset.pseudotime, gb, pred_pt, pb)
        out["wcor_features"] = wcor_features(dataset.counts, dataset.pseudotime, pred_pt)

    if pred_branch is not None and dataset.branch is not None:
        out["F1_branches"] = f1_branches(dataset.branch, pred_branch)

    if pred_adj is not None and gold_adj is not None:
        out["HIM"] = him_distance(gold_adj, pred_adj)

    # Aggregate harmonic mean
    finite = [v for v in out.values() if np.isfinite(v) and v >= 0]
    if finite:
        out["overall"] = float(np.power(np.prod(np.array(finite) + 1e-6), 1.0 / len(finite)))
    return out
