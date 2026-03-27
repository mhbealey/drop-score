"""
DROP SCORE v17 — CONVICTION + REGIME + REALISM
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

# ── Timing ──
t_start = time.time()
