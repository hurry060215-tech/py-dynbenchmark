"""Benchmark harness — run multiple methods on multiple datasets."""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .datasets import Dataset, DATASET_REGISTRY
from .metrics import calculate_metrics


@dataclass
class BenchmarkResult:
    """Tabular result: rows are (method, dataset) pairs, columns are metrics."""
    rows: list[dict] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def summary(self) -> pd.DataFrame:
        """Mean metric per method across datasets."""
        df = self.to_dataframe()
        metric_cols = [c for c in df.columns
                       if c not in ("method", "dataset", "runtime_s", "error")]
        return df.groupby("method")[metric_cols].mean().sort_values(
            by="overall" if "overall" in metric_cols else metric_cols[0],
            ascending=False,
        )


def run_benchmark(
    methods: dict[str, Callable],
    datasets: list[str] | None = None,
    *,
    verbose: bool = True,
) -> BenchmarkResult:
    """Run every method on every dataset and score with dyneval-style metrics.

    Args:
        methods: dict ``method_name → fn(dataset) → {"pseudotime": ..., "branch": ...,
                 "adjacency": ...}``. Each returned key is optional.
        datasets: list of dataset names from ``DATASET_REGISTRY``; default all.
        verbose: print progress.

    Returns:
        ``BenchmarkResult`` whose rows are dicts of ``method``, ``dataset``,
        metric scores, ``runtime_s``.
    """
    if datasets is None:
        datasets = list(DATASET_REGISTRY.keys())

    result = BenchmarkResult()
    for ds_name in datasets:
        ds = DATASET_REGISTRY[ds_name]()
        for m_name, fn in methods.items():
            row = {"method": m_name, "dataset": ds_name}
            t0 = time.time()
            try:
                pred = fn(ds)
                metrics = calculate_metrics(
                    ds,
                    pred_pt=pred.get("pseudotime"),
                    pred_branch=pred.get("branch"),
                    pred_adj=pred.get("adjacency"),
                    gold_adj=pred.get("gold_adjacency"),
                )
                row.update(metrics)
                row["error"] = None
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
                if verbose:
                    print(f"  {m_name} on {ds_name} ERROR: {row['error']}")
                    traceback.print_exc()
            row["runtime_s"] = time.time() - t0
            result.rows.append(row)
            if verbose and row.get("error") is None:
                summary = ", ".join(f"{k}={v:.3f}" for k, v in row.items()
                                    if isinstance(v, (int, float)) and k != "runtime_s")
                print(f"  ✓ {m_name:18s} on {ds_name:18s} [{row['runtime_s']:5.1f}s]: {summary}")
    return result
