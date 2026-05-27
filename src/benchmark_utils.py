"""
Shared timing and reporting utilities used by every benchmark notebook.
"""
import time
import pandas as pd
import numpy as np
from typing import Callable, Dict


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

class BenchmarkTimer:
    """Accumulates per-operation timings and formats them as a DataFrame."""

    def __init__(self, library: str):
        self.library = library
        self._results: Dict[str, float] = {}

    def time(self, name: str, func: Callable, warmup: bool = False) -> float:
        """
        Time *func* and store the result.  If *warmup* is True the timing is
        discarded (useful to warm up JIT / caches before the real measurement).
        """
        if warmup:
            try:
                func()
            except Exception:
                pass

        start = time.perf_counter()
        try:
            func()
        except Exception as exc:
            print(f"  [SKIP] {name}: {exc}")
            self._results[name] = float("nan")
            return float("nan")

        elapsed = time.perf_counter() - start
        if not warmup:
            self._results[name] = elapsed
            print(f"  {name:<40s}: {elapsed:.3f}s")
        return elapsed

    def to_series(self) -> pd.Series:
        return pd.Series(self._results, name=self.library)

    def reset(self):
        self._results = {}


# ---------------------------------------------------------------------------
# Results aggregation
# ---------------------------------------------------------------------------

def build_results_table(*timers: BenchmarkTimer) -> pd.DataFrame:
    """
    Combine multiple BenchmarkTimer instances (one per library / scenario)
    into a single comparison table (operations × libraries).
    """
    series = [t.to_series() for t in timers]
    df = pd.concat(series, axis=1)
    df.index.name = "operation"
    return df


def speedup_table(df: pd.DataFrame, baseline_col: str) -> pd.DataFrame:
    """Return a table of speedup ratios relative to *baseline_col*."""
    return df.div(df[baseline_col], axis=0).rename(
        columns=lambda c: f"{c}_speedup" if c != baseline_col else baseline_col
    )


def print_table(df: pd.DataFrame, title: str = "BENCHMARK RESULTS (seconds)"):
    sep = "=" * 70
    print(f"\n{sep}\n{title}\n{sep}")
    print(df.to_string(float_format=lambda x: f"{x:.3f}"))
    print(sep)


def geometric_mean(values) -> float:
    """Geometric mean, ignoring NaN."""
    vals = [v for v in values if not np.isnan(v) and v > 0]
    if not vals:
        return float("nan")
    return float(np.exp(np.mean(np.log(vals))))
