"""
All 15 blog benchmark operations implemented with Dask.

Each function accepts a Dask DataFrame and returns the timed result dict via
BenchmarkTimer.  Calling .compute() on every operation ensures we measure
actual execution time, not just lazy-plan construction.
"""
import numpy as np
import pandas as pd
import dask.dataframe as dd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.benchmark_utils import BenchmarkTimer
from config import VENDOR_LOOKUP


def _vendor_df() -> dd.DataFrame:
    """Small lookup DataFrame used for join operations."""
    return dd.from_pandas(pd.DataFrame(VENDOR_LOOKUP), npartitions=1)


def run_operations(df: dd.DataFrame, label: str = "dask") -> BenchmarkTimer:
    """
    Run all 15 blog operations on *df* and return a populated BenchmarkTimer.

    *label* is used as the timer's library name (e.g. "dask_standard",
    "dask_filtered", "dask_cached").
    """
    t = BenchmarkTimer(label)
    df2 = _vendor_df()

    # 1. complex_arithmetic
    t.time("complex_arithmetic", lambda: (
        np.sin(df["fare_amt"]) + np.cos(df["tip_amt"])
        + np.arctan2(df["fare_amt"], df["tip_amt"])
    ).compute())

    # 2. count
    t.time("count", lambda: len(df))

    # 3. count_index
    t.time("count_index", lambda: len(df.index))

    # 4. groupby_stats  (mean + std of fare_amt by vendor)
    t.time("groupby_stats", lambda: (
        df.groupby("vendor_name")["fare_amt"]
        .agg(["mean", "std"])
        .compute()
    ))

    # 5. join  (inner join with small vendor lookup)
    t.time("join", lambda: df.merge(df2, on="vendor_name").compute())

    # 6. join_count
    t.time("join_count", lambda: len(df.merge(df2, on="vendor_name")))

    # 7. mean
    t.time("mean", lambda: df["fare_amt"].mean().compute())

    # 8. mean_of_complex_arithmetic
    t.time("mean_complex_arithmetic", lambda: (
        np.sin(df["fare_amt"]) + np.cos(df["tip_amt"])
        + np.arctan2(df["fare_amt"], df["tip_amt"])
    ).mean().compute())

    # 9. mean_of_series_addition
    t.time("mean_series_add", lambda: (df["fare_amt"] + df["tip_amt"]).mean().compute())

    # 10. mean_of_series_multiplication
    t.time("mean_series_mul", lambda: (df["fare_amt"] * df["tip_amt"]).mean().compute())

    # 11. read_parquet  — timed separately in the notebook (see read_parquet_timing)

    # 12. series_addition
    t.time("series_add", lambda: (df["fare_amt"] + df["tip_amt"]).compute())

    # 13. series_multiplication
    t.time("series_mul", lambda: (df["fare_amt"] * df["tip_amt"]).compute())

    # 14. std
    t.time("std", lambda: df["fare_amt"].std().compute())

    # 15. value_counts
    t.time("value_counts", lambda: df["payment_type"].value_counts().compute())

    return t


def read_parquet_timing(path: str, label: str = "dask") -> BenchmarkTimer:
    """Time the read_parquet operation (creates lazy DF + forces row count)."""
    import time
    t = BenchmarkTimer(label)

    def _read():
        _df = dd.read_parquet(path)
        _ = len(_df)   # triggers partition scan

    t.time("read_parquet", _read)
    return t
