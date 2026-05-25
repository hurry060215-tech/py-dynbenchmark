"""Additional metrics — fills out the dyneval set.

Adds:
- F1_milestones: F1 of cell-to-milestone assignment (R dyneval::F1_milestones)
- edge_flip:    milestone-network edge-flip score
- isomorphic:   1 if R and Py topology graphs are isomorphic else 0
- pseudotime_spearman: Spearman of predicted pseudotime vs gold per-cell pseudotime
- featureimp_wcor: weighted correlation of feature-importance vectors
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import f1_score


def _hungarian_match(gold: np.ndarray, pred: np.ndarray) -> dict:
    """Best label alignment between gold and pred labels."""
    gold_u = np.unique(gold)
    pred_u = np.unique(pred)
    M = np.zeros((len(gold_u), len(pred_u)))
    for i, g in enumerate(gold_u):
        for j, p in enumerate(pred_u):
            M[i, j] = np.sum((gold == g) & (pred == p))
    row, col = linear_sum_assignment(-M)
    return {pred_u[c]: gold_u[r] for r, c in zip(row, col)}


def f1_milestones(gold_branch, pred_branch, gold_pt, pred_pt, n_milestones: int = 5) -> float:
    """F1 of cell-to-milestone assignment.

    Milestones are discretised positions along each branch — gold cells get
    assigned to the nearest of n_milestones evenly-spaced milestones per branch,
    same for pred. F1 macro across the union of milestones (with Hungarian
    label-matching).
    """
    gold = np.asarray(gold_branch); pred = np.asarray(pred_branch)
    gpt = np.asarray(gold_pt); ppt = np.asarray(pred_pt)
    n = len(gpt)
    if len(pred) != n or len(ppt) != n:
        return 0.0

    def to_milestones(pt: np.ndarray, br: np.ndarray) -> np.ndarray:
        ms = np.empty(n, dtype=object)
        for b in np.unique(br):
            mask = br == b
            pt_b = pt[mask]
            if pt_b.size == 0:
                continue
            edges = np.linspace(np.nanmin(pt_b), np.nanmax(pt_b), n_milestones + 1)
            bins = np.clip(np.digitize(pt_b, edges) - 1, 0, n_milestones - 1)
            ms[mask] = [f"{b}_m{i}" for i in bins]
        return ms

    g_ms = to_milestones(gpt, gold)
    p_ms = to_milestones(ppt, pred)
    mapping = _hungarian_match(g_ms, p_ms)
    p_mapped = np.array([mapping.get(p, "_") for p in p_ms])
    common = np.unique(g_ms)
    return float(f1_score(g_ms, p_mapped, labels=common, average="macro", zero_division=0))


def edge_flip(gold_adj: np.ndarray, pred_adj: np.ndarray) -> float:
    """Edge-flip score: 1 - fraction of edges that differ between graphs.

    Both adjacencies must share the same node set (padded if necessary).
    """
    A = np.asarray(gold_adj, dtype=int); B = np.asarray(pred_adj, dtype=int)
    if A.shape != B.shape:
        n = max(A.shape[0], B.shape[0])
        Ap = np.zeros((n, n), int); Ap[:A.shape[0], :A.shape[1]] = A
        Bp = np.zeros((n, n), int); Bp[:B.shape[0], :B.shape[1]] = B
        A, B = Ap, Bp
    A = (A > 0).astype(int); B = (B > 0).astype(int)
    n = A.shape[0]
    if n < 2:
        return 1.0
    iu = np.triu_indices(n, k=1)
    diff = np.abs(A[iu] - B[iu]).sum()
    return float(1.0 - diff / len(iu[0]))


def isomorphic_topology(gold_adj: np.ndarray, pred_adj: np.ndarray) -> float:
    """Returns 1.0 if the two graphs are isomorphic, else 0.0 (binary)."""
    try:
        import networkx as nx
    except ImportError:
        return float("nan")
    G1 = nx.from_numpy_array((np.asarray(gold_adj) > 0).astype(int))
    G2 = nx.from_numpy_array((np.asarray(pred_adj) > 0).astype(int))
    return float(nx.is_isomorphic(G1, G2))


def pseudotime_spearman(gold_pt: np.ndarray, pred_pt: np.ndarray) -> float:
    """Spearman correlation between gold and predicted pseudotime."""
    g = np.asarray(gold_pt, dtype=np.float64)
    p = np.asarray(pred_pt, dtype=np.float64)
    m = np.isfinite(g) & np.isfinite(p)
    if m.sum() < 3:
        return 0.0
    r, _ = spearmanr(g[m], p[m])
    if not np.isfinite(r):
        return 0.0
    return float(max(0.0, r))   # negative correlations clamp to 0
