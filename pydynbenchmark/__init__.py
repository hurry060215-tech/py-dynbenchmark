"""pydynbenchmark — trajectory inference benchmarking framework.

Inspired by dynverse/dynbenchmark (Saelens et al. Nat Biotechnol 2019).
Implements the four canonical dyneval metric families:

- HIM: Hamming-Ipsen-Mikhailov distance on milestone graphs
- F1_branches: F1 score on cell-to-branch assignment
- cor_dist: Pearson r between geodesic distance matrices on gold vs prediction
- wcor_features: weighted Pearson on top differentially-expressed features

plus a synthetic dyntoy-like dataset generator (linear / bifurcation / Y) and
a ``run_benchmark`` harness.

Citation: Saelens, W. et al. *A comparison of single-cell trajectory
inference methods.* Nat Biotechnol 37, 547–554 (2019).
"""

from __future__ import annotations

__version__ = "0.1.0"

from .metrics import (
    him_distance,
    f1_branches,
    cor_dist,
    wcor_features,
    calculate_metrics,
)
from .datasets import (
    Dataset,
    generate_linear,
    generate_bifurcation,
    generate_tree,
    DATASET_REGISTRY,
)
from .harness import run_benchmark, BenchmarkResult

__all__ = [
    "him_distance",
    "f1_branches",
    "cor_dist",
    "wcor_features",
    "calculate_metrics",
    "Dataset",
    "generate_linear",
    "generate_bifurcation",
    "generate_tree",
    "DATASET_REGISTRY",
    "run_benchmark",
    "BenchmarkResult",
    "__version__",
]
