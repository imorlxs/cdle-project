"""
All 15 blog benchmark operations implemented with Koalas / pyspark.pandas.

Since PySpark 3.2, Koalas is bundled as `pyspark.pandas` (ps).  The import
shim at the top handles both the legacy `databricks.koalas` package and the
current `pyspark.pandas` package so the same operations file works everywhere.

Scalar operations (mean, std, len) trigger PySpark execution automatically.
Series operations use `.to_pandas()` to materialise results, matching the
equivalent `.compute()` call in the Dask version.
"""
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.benchmark_utils import BenchmarkTimer
from config import VENDOR_LOOKUP


# ---------------------------------------------------------------------------
# Koalas / pyspark.pandas import shim
# ---------------------------------------------------------------------------
def _get_ps():
    """Return the pyspark.pandas (or legacy koalas) module."""
    try:
        import pyspark.pandas as ps
        return ps
    except ImportError:
        import databricks.koalas as ks
        return ks


def _vendor_df(ps):
    """Small lookup DataFrame used for join operations."""
    return ps.from_pandas(pd.DataFrame(VENDOR_LOOKUP))


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_operations(df, label: str = "koalas") -> BenchmarkTimer:
    """
    Run all 15 blog operations on a Koalas/pyspark.pandas DataFrame *df*.
    Returns a populated BenchmarkTimer.
    """
    ps = _get_ps()
    t = BenchmarkTimer(label)
    df2 = _vendor_df(ps)

    # 1. complex_arithmetic
    t.time("complex_arithmetic", lambda: (
        np.sin(df["fare_amt"]) + np.cos(df["tip_amt"])
        + np.arctan2(df["fare_amt"], df["tip_amt"])
    ).to_pandas())

    # 2. count
    t.time("count", lambda: len(df))

    # 3. count_index
    t.time("count_index", lambda: len(df.index))

    # 4. groupby_stats
    t.time("groupby_stats", lambda: (
        df.groupby("vendor_name")["fare_amt"]
        .agg(["mean", "std"])
        .to_pandas()
    ))

    # 5. join
    t.time("join", lambda: df.merge(df2, on="vendor_name").to_pandas())

    # 6. join_count
    t.time("join_count", lambda: len(df.merge(df2, on="vendor_name")))

    # 7. mean  (scalar — triggers execution)
    t.time("mean", lambda: df["fare_amt"].mean())

    # 8. mean_of_complex_arithmetic
    t.time("mean_complex_arithmetic", lambda: (
        np.sin(df["fare_amt"]) + np.cos(df["tip_amt"])
        + np.arctan2(df["fare_amt"], df["tip_amt"])
    ).mean())

    # 9. mean_of_series_addition
    t.time("mean_series_add", lambda: (df["fare_amt"] + df["tip_amt"]).mean())

    # 10. mean_of_series_multiplication
    t.time("mean_series_mul", lambda: (df["fare_amt"] * df["tip_amt"]).mean())

    # 11. read_parquet — timed separately via read_parquet_timing()

    # 12. series_addition
    t.time("series_add", lambda: (df["fare_amt"] + df["tip_amt"]).to_pandas())

    # 13. series_multiplication
    t.time("series_mul", lambda: (df["fare_amt"] * df["tip_amt"]).to_pandas())

    # 14. std
    t.time("std", lambda: df["fare_amt"].std())

    # 15. value_counts
    t.time("value_counts", lambda: df["payment_type"].value_counts().to_pandas())

    return t


def read_parquet_timing(path: str, spark, label: str = "koalas") -> BenchmarkTimer:
    """Time read_parquet: create Koalas DataFrame and force row count."""
    ps = _get_ps()
    t = BenchmarkTimer(label)

    def _read():
        _df = ps.read_parquet(path)
        _ = len(_df)

    t.time("read_parquet", _read)
    return t


# ---------------------------------------------------------------------------
# Caching helper
# ---------------------------------------------------------------------------

def cache_dataframe(df):
    """
    Materialise and cache a Koalas DataFrame in Spark's memory.
    Equivalent to Dask's client.persist() + wait().
    """
    df.spark.cache()
    df.spark.apply(lambda sdf: sdf.count())   # force materialisation
    return df
