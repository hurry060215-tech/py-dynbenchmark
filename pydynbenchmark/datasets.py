"""Synthetic dyntoy-like dataset generators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    """A reference trajectory dataset.

    Attributes:
        cell_ids: list of cell names.
        counts: (n_cells × n_genes) expression matrix (log-normalized or counts).
        embedding: (n_cells × 2) ground-truth 2D embedding.
        pseudotime: (n_cells,) gold-standard pseudotime per cell, in [0, 1].
        branch: (n_cells,) gold-standard branch assignment (categorical labels).
        topology: dict — name, expected branch labels, etc.
    """
    cell_ids: list[str]
    counts: np.ndarray
    embedding: np.ndarray
    pseudotime: np.ndarray
    branch: np.ndarray
    topology: dict


def _add_gene_noise(n_cells, n_genes, signal_genes, signal, seed=0):
    rng = np.random.RandomState(seed)
    n_signal = signal.shape[1]
    # If signal is bigger than n_genes, expand n_genes
    n_genes_effective = max(n_genes, n_signal)
    counts = rng.poisson(2.0, (n_cells, n_genes_effective)).astype(float)
    counts[:, :n_signal] = signal + rng.poisson(0.5, (n_cells, n_signal))
    return counts


def generate_linear(n_cells: int = 200, n_genes: int = 100, seed: int = 0) -> Dataset:
    rng = np.random.RandomState(seed)
    t = np.sort(rng.uniform(0, 1, n_cells))
    # Embedding: line along x-axis with mild jitter
    embedding = np.column_stack([t * 10 - 5, rng.normal(0, 0.05, n_cells)])
    # Signal: 20 genes monotonically increasing, 20 decreasing
    sig_up = np.outer(t, np.linspace(0.5, 5.0, 20))
    sig_down = np.outer(1 - t, np.linspace(0.5, 5.0, 20))
    signal = np.column_stack([sig_up, sig_down])
    counts = _add_gene_noise(n_cells, n_genes, signal.shape[1], signal, seed=seed)
    branch = np.array(["main"] * n_cells)
    return Dataset(
        cell_ids=[f"cell_{i}" for i in range(n_cells)],
        counts=counts,
        embedding=embedding,
        pseudotime=t,
        branch=branch,
        topology={"name": "linear", "branches": ["main"]},
    )


def generate_bifurcation(n_cells: int = 300, n_genes: int = 100, seed: int = 0) -> Dataset:
    rng = np.random.RandomState(seed)
    n_stem = n_cells // 3
    n_b1 = n_cells // 3
    n_b2 = n_cells - n_stem - n_b1
    # Stem
    t_stem = np.sort(rng.uniform(0, 0.5, n_stem))
    pos_stem = np.column_stack([t_stem * 6 - 3, np.zeros(n_stem)])
    # Branches
    t_b1 = np.sort(rng.uniform(0.5, 1.0, n_b1))
    pos_b1 = np.column_stack([t_b1 * 6 - 3, (t_b1 - 0.5) * 4])
    t_b2 = np.sort(rng.uniform(0.5, 1.0, n_b2))
    pos_b2 = np.column_stack([t_b2 * 6 - 3, -(t_b2 - 0.5) * 4])

    embedding = np.vstack([pos_stem, pos_b1, pos_b2]) + rng.normal(0, 0.1, (n_cells, 2))
    pseudotime = np.concatenate([t_stem, t_b1, t_b2])
    branch = np.concatenate([
        np.array(["A"] * n_stem),
        np.array(["B"] * n_b1),
        np.array(["C"] * n_b2),
    ])

    # Signal genes: 15 up-stem, 15 up-B-only, 15 up-C-only
    up_stem = np.outer(pseudotime * (branch == "A").astype(float), np.linspace(0.5, 5.0, 15))
    up_b   = np.outer(pseudotime * (branch == "B").astype(float), np.linspace(0.5, 5.0, 15))
    up_c   = np.outer(pseudotime * (branch == "C").astype(float), np.linspace(0.5, 5.0, 15))
    signal = np.column_stack([up_stem, up_b, up_c])
    counts = _add_gene_noise(n_cells, n_genes, signal.shape[1], signal, seed=seed)

    return Dataset(
        cell_ids=[f"cell_{i}" for i in range(n_cells)],
        counts=counts,
        embedding=embedding,
        pseudotime=pseudotime,
        branch=branch,
        topology={"name": "bifurcation", "branches": ["A", "B", "C"]},
    )


def generate_tree(n_cells: int = 400, n_genes: int = 120, seed: int = 0) -> Dataset:
    """Two-bifurcation tree: A -> {B, C}, then C -> {D, E}."""
    rng = np.random.RandomState(seed)
    n = n_cells // 5
    # A stem
    t_a = np.sort(rng.uniform(0, 0.3, n))
    pos_a = np.column_stack([t_a * 6 - 3, np.zeros(n)])
    # B branch (up)
    t_b = np.sort(rng.uniform(0.3, 1.0, n))
    pos_b = np.column_stack([t_b * 6 - 3, (t_b - 0.3) * 5])
    # C stem (down)
    t_c = np.sort(rng.uniform(0.3, 0.6, n))
    pos_c = np.column_stack([t_c * 6 - 3, -(t_c - 0.3) * 4])
    # D branch (further down-left)
    t_d = np.sort(rng.uniform(0.6, 1.0, n))
    pos_d = np.column_stack([t_d * 6 - 3 - (t_d - 0.6) * 3, -(t_d - 0.3) * 4])
    # E branch (further down-right)
    n_e = n_cells - 4 * n
    t_e = np.sort(rng.uniform(0.6, 1.0, n_e))
    pos_e = np.column_stack([t_e * 6 - 3 + (t_e - 0.6) * 3, -(t_e - 0.3) * 4])

    embedding = np.vstack([pos_a, pos_b, pos_c, pos_d, pos_e]) + rng.normal(0, 0.1, (n_cells, 2))
    pseudotime = np.concatenate([t_a, t_b, t_c, t_d, t_e])
    branch = np.concatenate([
        np.array(["A"] * n),
        np.array(["B"] * n),
        np.array(["C"] * n),
        np.array(["D"] * n),
        np.array(["E"] * n_e),
    ])

    # Signal: 20 genes per branch
    n_per = 20
    cols = []
    for lab in ["A", "B", "C", "D", "E"]:
        sig = np.outer(pseudotime * (branch == lab).astype(float),
                       np.linspace(0.5, 4.0, n_per))
        cols.append(sig)
    signal = np.column_stack(cols)
    counts = _add_gene_noise(n_cells, n_genes, signal.shape[1], signal, seed=seed)

    return Dataset(
        cell_ids=[f"cell_{i}" for i in range(n_cells)],
        counts=counts,
        embedding=embedding,
        pseudotime=pseudotime,
        branch=branch,
        topology={"name": "tree", "branches": ["A", "B", "C", "D", "E"]},
    )


DATASET_REGISTRY = {
    "linear_200": lambda: generate_linear(n_cells=200, n_genes=100, seed=42),
    "linear_500": lambda: generate_linear(n_cells=500, n_genes=200, seed=43),
    "bifurcation_300": lambda: generate_bifurcation(n_cells=300, n_genes=100, seed=44),
    "tree_400": lambda: generate_tree(n_cells=400, n_genes=120, seed=45),
}
