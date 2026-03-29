"""Utility functions shared across the Drop Score pipeline."""
from contextlib import contextmanager
from typing import Any, Generator, List, Optional, Set, Tuple, Union

import time
import numpy as np
import pandas as pd

from config import t_start, log, KNOWN_FEATURES


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


# ── Input Validation ──────────────────────────────────────

def validate_features(features: List[str], context: str = "") -> None:
    """Validate that locked feature names are in KNOWN_FEATURES.

    Raises ValidationError if unknown features are found.
    """
    unknown = set(features) - KNOWN_FEATURES
    if unknown:
        raise ValidationError(
            f"Unknown features in {context or 'feature list'}: {sorted(unknown)}. "
            f"Known features: {sorted(KNOWN_FEATURES)}"
        )


def validate_data_bundle(bundle: dict, stage: str = "pipeline") -> None:
    """Data quality firewall — validate bundle structure and contents.

    Call after loading a data bundle and before any model/feature work.
    Raises DataError if critical issues are found.
    """
    # Required keys for all stages
    required_keys = {'df_dev', 'df_hold', 'price_dict', 'spy_close'}
    missing = required_keys - set(bundle.keys())
    if missing:
        raise DataError(f"[{stage}] Data bundle missing required keys: {sorted(missing)}")

    df_dev = bundle['df_dev']
    df_hold = bundle['df_hold']

    # DataFrames must not be empty
    if len(df_dev) == 0:
        raise DataError(f"[{stage}] df_dev is empty")
    if len(df_hold) == 0:
        raise DataError(f"[{stage}] df_hold is empty")

    # Required columns in dev/hold
    required_cols = {'ticker', 'report_date', 'price'}
    for name, df in [('df_dev', df_dev), ('df_hold', df_hold)]:
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise DataError(f"[{stage}] {name} missing columns: {sorted(missing_cols)}")

    # Feature columns (if present in bundle)
    if 'fcols_q' in bundle:
        fcols_q = bundle['fcols_q']
        if len(fcols_q) == 0:
            raise DataError(f"[{stage}] fcols_q is empty — no features available")
        missing_fcols = [c for c in fcols_q if c not in df_dev.columns]
        if missing_fcols:
            raise DataError(
                f"[{stage}] {len(missing_fcols)} feature columns in fcols_q "
                f"not found in df_dev: {missing_fcols[:5]}..."
            )

    # Fill medians (if present)
    if 'fill_meds_q' in bundle and 'fcols_q' in bundle:
        meds = bundle['fill_meds_q']
        if len(meds) != len(bundle['fcols_q']):
            raise DataError(
                f"[{stage}] fill_meds_q length ({len(meds)}) != "
                f"fcols_q length ({len(bundle['fcols_q'])})"
            )

    # Price dict sanity
    price_dict = bundle['price_dict']
    if len(price_dict) < 100:
        raise DataError(f"[{stage}] price_dict has only {len(price_dict)} tickers (expected 100+)")

    # Ticker count sanity
    n_tickers = df_dev['ticker'].nunique()
    if n_tickers < 50:
        raise DataError(f"[{stage}] df_dev has only {n_tickers} unique tickers (expected 50+)")

    log.info(f"  Data bundle validated ({stage}): {n_tickers} tickers, "
             f"{len(df_dev):,} dev rows, {len(df_hold):,} hold rows")
