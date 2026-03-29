"""
DROP SCORE v18 — LOCKED CONFIG + DUAL UNIVERSE + BORROW COSTS
Configuration constants and SimFin column name setup.
"""
import os, time

# ── SimFin column names (define if not already present) ──
_SIMFIN_NAMES = [
    ('CHANGE_FIXED_ASSETS', 'Change in Fixed Assets & Intangibles'),
    ('INTEREST_EXPENSE_NET', 'Interest Expense, Net'),
    ('TOTAL_DEBT', 'Total Debt'),
    ('TOTAL_LIABILITIES', 'Total Liabilities'),
    ('TOTAL_CUR_ASSETS', 'Total Current Assets'),
    ('TOTAL_CUR_LIAB', 'Total Current Liabilities'),
    ('CASH_EQUIV', 'Cash, Cash Equivalents & Short Term Investments'),
    ('NET_CASH_OPS', 'Net Cash from Operating Activities'),
    ('SHARES_DILUTED', 'Shares (Diluted)'),
    ('SHARES_BASIC', 'Shares (Basic)'),
    ('REVENUE', 'Revenue'),
    ('GROSS_PROFIT', 'Gross Profit'),
    ('OPERATING_INCOME', 'Operating Income (Loss)'),
    ('NET_INCOME', 'Net Income'),
    ('TOTAL_ASSETS', 'Total Assets'),
    ('TOTAL_EQUITY', 'Total Equity'),
]

import simfin as sf
from simfin.names import *
for _n, _v in _SIMFIN_NAMES:
    if _n not in dir():
        exec(f"{_n}={_v!r}")

# ── API keys ──
SIMFIN_KEY = os.environ.get('SIMFIN_KEY', 'd77356e9-d47a-4ceb-86b0-224db766fe7a')
FMP_KEY = os.environ.get('FMP_KEY', '')

# ── Model / backtest parameters ──
FWD_WINDOWS = [5, 10, 21, 42, 63]
DROP_THRESH = [0.05, 0.10, 0.15, 0.20, 0.25]
EXCESS_THRESH = [0.05, 0.10, 0.15]
N_FOLDS = 4
N_BOOT = 500
HOLDOUT_MO = 6
ACCT = 25000
MAX_POS = 5
POS_SIZE = 5000
MIN_EVENTS = 300
MAX_BASE_RATE = 0.40
MIN_K = 10
NAN_DROP = 0.50
SLIPPAGE = 0.001
STOP_LOSS = 0.15
PROFIT_TARGET = 0.15
TRAILING_STOP = 0.05
REGIME_SPY_MAX = 0.05
REGIME_VIX_MIN = 12
SECTOR_CAP = 0.30
SKIP_RET5D_DOWN = -0.10
SKIP_VOL_PCT = 0.90
VOL_FLOOR = 500000  # Raised from 200K — removes hard-to-borrow
ENTRY_DELAY = 1  # Enter next trading day, not signal day
SECTOR_ETFS = {
    'XLK': 'Technology', 'XLF': 'Financial', 'XLE': 'Energy',
    'XLV': 'Healthcare', 'XLI': 'Industrials', 'XLY': 'Consumer Cyclical',
    'XLP': 'Consumer Defensive', 'XLB': 'Basic Materials', 'XLU': 'Utilities',
    'XLRE': 'Real Estate', 'XLC': 'Communication',
}
FORCE_RECOMPUTE = False

# ── Locked trading configuration (validated via v17 comparison table) ──
TRADING_TARGET = "exdrop_15_10d"   # Validated winner: 71% win, +$1.95/sh top-25%
TRADING_HOLD = 21                  # 21-day hold matched to 10d excess-drop target
ENTRY_MODE = "confirmed"           # 2% drop in 5 days required before entry
CONFIRMATION_DROP = 0.02           # Minimum decline from signal day's close
CONFIRMATION_WINDOW = 5            # Trading days to wait for confirmation

# ── Borrow costs ──
BORROW_RATE_EASY = 0.03            # 3% annual for avg_vol >= 1M
BORROW_RATE_HARD = 0.06            # 6% annual for avg_vol 500K-1M

# ── Universe mode ──
UNIVERSE_MODE = "both"             # "sp_index", "full", or "both"

# ── Locked feature sets (from v18) ──
UNIVERSE_A_FEATURES = [
    'roa', 'eps', 'vol_60d', 'log_revenue', 'vol_30d',
    'earnings_yield', 'cfo_to_revenue', 'gross_margin',
    'cash_to_assets', 'sales_to_price',
]
UNIVERSE_B_FEATURES = [
    'net_margin', 'rev_per_share', 'vol_60d', 'roe',
    'nm_chg_4q', 'dilution', 'fcf_yield', 'vol_30d',
    'max_dd_63d', 'book_to_price',
]

# ── Timing ──
t_start = time.time()
