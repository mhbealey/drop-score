"""
Typed result objects for Drop Score pipeline outputs.

Replace mystery dicts with dataclasses so typos are caught at write-time,
IDE autocomplete works, and data flow is self-documenting.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class TierResult:
    """Results for a single conviction tier (e.g., top-25%)."""
    name: str
    n_trades: int
    win_rate: float
    raw_pnl: float
    borrow_cost: float
    net_pnl: float
    net_pct: float
    stop_rate: float
    trades_per_quarter: float = 0.0

    @property
    def is_profitable(self) -> bool:
        return self.net_pnl > 0


@dataclass
class WalkForwardResult:
    """Complete results from a walk-forward backtest."""
    target: str
    dev_auc: float
    hold_auc: float
    n_tickers: int
    n_dev_rows: int
    n_hold_rows: int
    features_used: List[str]
    tiers: Dict[str, TierResult] = field(default_factory=dict)
    all_trades: List[dict] = field(default_factory=list, repr=False)

    @property
    def top10(self) -> Optional[TierResult]:
        return self.tiers.get('Top 10%')

    @property
    def top25(self) -> Optional[TierResult]:
        return self.tiers.get('Top 25%')

    @property
    def total_trades(self) -> int:
        full = self.tiers.get('Full')
        return full.n_trades if full else 0


@dataclass
class EquityResult:
    """Results from equity curve simulation."""
    label: str
    start_capital: float
    end_capital: float
    total_return_pct: float
    max_drawdown_pct: float
    calmar_ratio: float
    n_trades: int
    annualized_return: float = 0.0
    avg_positions: float = 0.0
    curve: Optional[pd.DataFrame] = field(default=None, repr=False)
    trade_log: List[dict] = field(default_factory=list, repr=False)
