import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pydynbenchmark as pyb


def test_import():
    assert pyb.__version__ == "0.1.0"


def test_him():
    A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    assert pyb.him_distance(A, A) == 1.0


def test_f1_branches_perfect():
    gold = np.array(["A", "A", "B", "B"])
    pred = np.array(["1", "1", "2", "2"])  # different label names, perfect alignment
    assert pyb.f1_branches(gold, pred) == 1.0


def test_cor_dist_perfect():
    pt = np.array([0.0, 0.5, 1.0])
    br = np.array(["A", "A", "A"])
    assert pyb.cor_dist(pt, br, pt, br) > 0.95


def test_dataset_linear():
    ds = pyb.generate_linear(n_cells=50, n_genes=100, seed=0)
    assert ds.counts.shape[0] == 50
    assert ds.counts.shape[1] >= 40  # signal expands genes as needed
    assert (ds.pseudotime >= 0).all() and (ds.pseudotime <= 1).all()


def test_run_benchmark_trivial():
    def identity_method(ds):
        return {"pseudotime": ds.pseudotime, "branch": ds.branch}

    res = pyb.run_benchmark(
        {"identity": identity_method},
        datasets=["linear_200"],
        verbose=False,
    )
    df = res.to_dataframe()
    assert len(df) == 1
    assert df.iloc[0]["cor_dist"] > 0.99
