"""Utility functions for Drop Score v17."""
import time
import numpy as np
import pandas as pd

from config import t_start


def elapsed():
    m, s = divmod(time.time() - t_start, 60)
    return f"[{int(m)}m{int(s)}s]"


def get_col(df, idx, *names):
    for c in names:
        if c in df.columns:
            try:
                v = df.loc[idx, c]
                if isinstance(v, pd.Series):
                    v = v.iloc[0]
                if pd.notna(v):
                    return float(v)
            except:
                pass
    return np.nan


def strip_tz(idx):
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        try:
            return idx.tz_convert(None)
        except:
            return pd.DatetimeIndex([t.replace(tzinfo=None) for t in idx])
    return idx


def clean_X(df, cols, meds):
    return df[cols].fillna(meds).replace([np.inf, -np.inf], 0)


def to_scalar(val):
    """Safely extract a Python float from any pandas/numpy object."""
    if isinstance(val, (pd.DataFrame, pd.Series)):
        val = val.squeeze()
    if isinstance(val, (pd.Series, pd.DataFrame)):
        val = val.iloc[0]
    if hasattr(val, 'item'):
        return float(val.item())
    return float(val)


def ensure_series(s):
    """Squeeze a single-column DataFrame to a Series."""
    if isinstance(s, pd.DataFrame):
        return s.squeeze(axis=1)
    return s
