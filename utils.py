"""Utility functions shared across the Drop Score pipeline."""
from contextlib import contextmanager
from typing import Any, Generator, List, Tuple, Union

import time
import numpy as np
import pandas as pd

from config import t_start, log


# ── Exceptions ──────────────────────────────────────────────

class DropScoreError(Exception):
    """Base exception for Drop Score."""
    pass


class DataError(DropScoreError):
    """Data loading or quality issue."""
    pass


class ValidationError(DropScoreError):
    """Benchmark validation failure."""
    pass


class InsufficientDataError(DropScoreError):
    """Not enough data to run analysis."""
    pass


# ── Timing ──────────────────────────────────────────────────

def elapsed() -> str:
    """Return elapsed time since pipeline start as '[Xm Ys]'."""
    m, s = divmod(time.time() - t_start, 60)
    return f"[{int(m)}m{int(s)}s]"


@contextmanager
def timer(label: str) -> Generator[None, None, None]:
    """Context manager that logs elapsed time for a block."""
    start = time.time()
    yield
    secs = time.time() - start
    if secs > 60:
        log.info(f"  [{label}: {secs / 60:.1f} min]")
    else:
        log.info(f"  [{label}: {secs:.0f}s]")


def get_col(df: pd.DataFrame, idx: Any, *names: str) -> float:
    """Look up the first non-null value from *names* columns at *idx*.

    Tries each column name in order; returns np.nan if all miss.
    """
    for c in names:
        if c in df.columns:
            try:
                v = df.loc[idx, c]
                if isinstance(v, pd.Series):
                    v = v.iloc[0]
                if pd.notna(v):
                    return float(v)
            except Exception:
                pass
    return np.nan


def strip_tz(idx: pd.Index) -> pd.Index:
    """Remove timezone info from a DatetimeIndex (no-op for others)."""
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        try:
            return idx.tz_convert(None)
        except Exception:
            return pd.DatetimeIndex([t.replace(tzinfo=None) for t in idx])
    return idx


def clean_X(df: pd.DataFrame, cols: List[str], meds: pd.Series) -> pd.DataFrame:
    """Select *cols*, fill NaN with medians, replace infinities with 0."""
    return df[cols].fillna(meds).replace([np.inf, -np.inf], 0)


def to_scalar(val: Any) -> float:
    """Safely extract a Python float from any pandas/numpy object."""
    if isinstance(val, (pd.DataFrame, pd.Series)):
        val = val.squeeze()
    if isinstance(val, (pd.Series, pd.DataFrame)):
        val = val.iloc[0]
    if hasattr(val, 'item'):
        return float(val.item())
    return float(val)


def ensure_series(s: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    """Squeeze a single-column DataFrame to a Series."""
    if isinstance(s, pd.DataFrame):
        return s.squeeze(axis=1)
    return s
