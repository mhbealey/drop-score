"""
Standardized output formatting for Drop Score pipeline.
All print/display functions consolidated here for consistency.
"""
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import log


def print_header(title: str, width: int = 70) -> None:
    """Print a section header with separator lines."""
    log.info("=" * width)
    log.info(title)
    log.info("=" * width)


def print_section(title: str, width: int = 50) -> None:
    """Print a sub-section header."""
    log.info(f"\n  {'=' * width}")
    log.info(f"  {title}")
    log.info(f"  {'=' * width}")


def format_value(val: Any, fmt: str = '.3f') -> str:
    """Format a numeric value, returning 'N/A' if NaN."""
    if val is None or (isinstance(val, float) and (np.isnan(val) or pd.isna(val))):
        return 'N/A'
    return f"{val:{fmt}}"
