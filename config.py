"""
Drop Score Configuration
========================
All model parameters, thresholds, feature definitions, and XBRL mappings.
Change settings here, not in individual modules.
"""
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set


# ── Logging ─────────────────────────────────────────────────
# Dynamic stdout handler: always writes to current sys.stdout,
# which works correctly with the Tee redirect in pipeline.py.
class _DynamicStdoutHandler(logging.StreamHandler):
    """Handler that always references the *current* sys.stdout."""
    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stdout
        super().emit(record)


def setup_dropscore_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the 'dropscore' logger.

    Uses '%(message)s' format so output is identical to print().
    """
    logger = logging.getLogger('dropscore')
    if not logger.handlers:
        handler = _DynamicStdoutHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


log = setup_dropscore_logging()


# ── Version ──────────────────────────────────────────────
VERSION = "18.3"


# ── Trading Configuration ────────────────────────────────
@dataclass(frozen=True)
class TradingConfig:
    """Parameters that define how trades are entered and exited."""
    target: str = "exdrop_15_10d"
    hold_days: int = 21
    entry_mode: str = "confirmed"
    confirm_drop_pct: float = 0.02
    confirm_window_days: int = 5
    slippage: float = 0.001
    stop_loss_pct: float = 0.15
    profit_target_pct: float = 0.15
    trailing_stop_pct: float = 0.05
    min_volume: int = 500_000
    entry_delay: int = 1
    borrow_rate_easy: float = 0.03
    borrow_rate_hard: float = 0.06
    borrow_volume_threshold: int = 1_000_000


# ── Regime Filter Configuration ──────────────────────────
@dataclass(frozen=True)
class RegimeConfig:
    """Market regime filters for walk-forward and equity sim."""
    spy_max_21d: float = 0.05
    vix_min: int = 12
    sector_cap: float = 0.30
    skip_ret5d_down: float = -0.10
    skip_vol_pct: float = 0.90


# ── Feature Configuration ────────────────────────────────
@dataclass(frozen=True)
class FeatureConfig:
    """Feature lists for each universe. Locked -- do not modify without validation."""
    universe_a: tuple = (
        'roa', 'eps', 'vol_60d', 'log_revenue', 'vol_30d',
        'earnings_yield', 'cfo_to_revenue', 'gross_margin',
        'cash_to_assets', 'sales_to_price',
    )
    universe_b: tuple = (
        'net_margin', 'rev_per_share', 'vol_60d', 'roe',
        'nm_chg_4q', 'dilution', 'fcf_yield', 'vol_30d',
        'max_dd_63d', 'book_to_price',
    )


# ── Model Configuration ──────────────────────────────────
@dataclass(frozen=True)
class ModelConfig:
    """XGBoost/LightGBM training parameters."""
    n_folds: int = 4
    n_boot: int = 500
    min_k: int = 10
    min_events: int = 300
    max_base_rate: float = 0.40
    nan_drop: float = 0.50
    holdout_months: int = 6


# ── Target Configuration ─────────────────────────────────
@dataclass(frozen=True)
class TargetConfig:
    """Forward-return windows and drop thresholds for outcome computation."""
    fwd_windows: tuple = (5, 10, 21, 42, 63)
    drop_thresholds: tuple = (0.05, 0.10, 0.15, 0.20, 0.25)
    excess_thresholds: tuple = (0.05, 0.10, 0.15)


# ── Data Configuration ────────────────────────────────────
@dataclass(frozen=True)
class DataConfig:
    """Data sources, caching, and filtering."""
    simfin_key: str = os.environ.get('SIMFIN_KEY', 'd77356e9-d47a-4ceb-86b0-224db766fe7a')
    fmp_key: str = os.environ.get('FMP_KEY', '')
    force_recompute: bool = False
    intermediates_version: str = "v15"


# ── Equity Simulation Configuration ──────────────────────
@dataclass(frozen=True)
class EquityConfig:
    """Account scenarios and equity curve parameters."""
    account_start: int = 25_000
    max_positions: int = 5
    position_size: int = 5_000
    regime_check_pct: float = 0.04


# ── Benchmark Gates (v18 reference) ──────────────────────
@dataclass(frozen=True)
class BenchmarkGates:
    """Minimum thresholds Universe A must meet to pass validation."""
    min_dev_auc: float = 0.741
    min_hold_auc: float = 0.677
    min_top25_trades: int = 15

    # v18 reference values
    v18_tickers: int = 960
    v18_dev_auc: float = 0.791
    v18_hold_auc: float = 0.777
    v18_top25_trades: int = 75
    v18_top25_win: float = 0.71
    v18_top25_pnl: float = 1.98


# ── Sector ETF Mapping ───────────────────────────────────
SECTOR_ETFS: Dict[str, str] = {
    'XLK': 'Technology', 'XLF': 'Financial', 'XLE': 'Energy',
    'XLV': 'Healthcare', 'XLI': 'Industrials', 'XLY': 'Consumer Cyclical',
    'XLP': 'Consumer Defensive', 'XLB': 'Basic Materials', 'XLU': 'Utilities',
    'XLRE': 'Real Estate', 'XLC': 'Communication',
}


# ── SimFin Column Names ──────────────────────────────────
# These map SimFin's Python constants to the string column names used in DataFrames.
# We define them explicitly rather than relying on exec() or simfin.names import.
SIMFIN_COLUMN_NAMES = {
    'CHANGE_FIXED_ASSETS': 'Change in Fixed Assets & Intangibles',
    'INTEREST_EXPENSE_NET': 'Interest Expense, Net',
    'TOTAL_DEBT': 'Total Debt',
    'TOTAL_LIABILITIES': 'Total Liabilities',
    'TOTAL_CUR_ASSETS': 'Total Current Assets',
    'TOTAL_CUR_LIAB': 'Total Current Liabilities',
    'CASH_EQUIV': 'Cash, Cash Equivalents & Short Term Investments',
    'NET_CASH_OPS': 'Net Cash from Operating Activities',
    'SHARES_DILUTED': 'Shares (Diluted)',
    'SHARES_BASIC': 'Shares (Basic)',
    'REVENUE': 'Revenue',
    'GROSS_PROFIT': 'Gross Profit',
    'OPERATING_INCOME': 'Operating Income (Loss)',
    'NET_INCOME': 'Net Income',
    'TOTAL_ASSETS': 'Total Assets',
    'TOTAL_EQUITY': 'Total Equity',
}


# ── XBRL Field Mapping ───────────────────────────────────
# Maps SimFin field names to ordered lists of XBRL tag alternatives.
# Parser tries tags in order; first value found per (ticker, quarter, field) wins.
# All tags for a field are checked across all quarters (no early break).
XBRL_TO_FIELD: Dict[str, List[str]] = {
    'Revenue': [
        'Revenues',
        'RevenueFromContractWithCustomerExcludingAssessedTax',
        'RevenueFromContractWithCustomerIncludingAssessedTax',
        'SalesRevenueNet',
        'SalesRevenueGoodsNet',
        'SalesRevenueServicesNet',
        'RegulatedAndUnregulatedOperatingRevenue',
    ],
    'Cost of Revenue': [
        'CostOfRevenue',
        'CostOfGoodsAndServicesSold',
        'CostOfGoodsSold',
        'CostOfServices',
        'CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization',
    ],
    'Gross Profit': [
        'GrossProfit',
    ],
    'Operating Income (Loss)': [
        'OperatingIncomeLoss',
        'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
        'IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments',
    ],
    'Net Income': [
        'NetIncomeLoss',
        'ProfitLoss',
        'NetIncomeLossAvailableToCommonStockholdersBasic',
        'NetIncomeLossAvailableToCommonStockholdersDiluted',
    ],
    'Interest Expense, Net': [
        'InterestExpense',
        'InterestExpenseDebt',
        'InterestIncomeExpenseNet',
    ],
    'Total Assets': [
        'Assets',
    ],
    'Total Equity': [
        'StockholdersEquity',
        'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
    ],
    'Total Liabilities': [
        'Liabilities',
        'LiabilitiesAndStockholdersEquity',
    ],
    'Total Debt': [
        'LongTermDebt',
        'LongTermDebtNoncurrent',
        'LongTermDebtAndCapitalLeaseObligations',
        'DebtAndCapitalLeaseObligations',
    ],
    'Total Current Assets': [
        'AssetsCurrent',
    ],
    'Total Current Liabilities': [
        'LiabilitiesCurrent',
    ],
    'Cash, Cash Equivalents & Short Term Investments': [
        'CashAndCashEquivalentsAtCarryingValue',
        'CashCashEquivalentsAndShortTermInvestments',
        'Cash',
    ],
    'Net Cash from Operating Activities': [
        'NetCashProvidedByUsedInOperatingActivities',
        'NetCashProvidedByUsedInOperatingActivitiesContinuingOperations',
    ],
    'Change in Fixed Assets & Intangibles': [
        'PaymentsToAcquirePropertyPlantAndEquipment',
        'PaymentsToAcquireProductiveAssets',
        'CapitalExpenditureDiscontinuedOperations',
    ],
    'Shares (Diluted)': [
        'WeightedAverageNumberOfDilutedSharesOutstanding',
        'EntityCommonStockSharesOutstanding',
    ],
    'Shares (Basic)': [
        'WeightedAverageNumberOfShareOutstandingBasicAndDiluted',
        'WeightedAverageNumberOfSharesOutstandingBasic',
        'CommonStockSharesOutstanding',
    ],
}

# Balance sheet fields are point-in-time; income/cash flow need quarterly duration.
BALANCE_SHEET_FIELDS: Set[str] = {
    'Total Assets', 'Total Liabilities', 'Total Equity',
    'Total Current Assets', 'Total Current Liabilities',
    'Cash, Cash Equivalents & Short Term Investments',
    'Total Debt',
    'Shares (Diluted)', 'Shares (Basic)',
}

SIMFIN_INCOME_FIELDS: List[str] = [
    'Revenue', 'Gross Profit', 'Operating Income (Loss)',
    'Net Income', 'Interest Expense, Net',
]
SIMFIN_BALANCE_FIELDS: List[str] = [
    'Total Assets', 'Total Equity', 'Total Liabilities', 'Total Debt',
    'Total Current Assets', 'Total Current Liabilities',
    'Cash, Cash Equivalents & Short Term Investments',
    'Shares (Diluted)', 'Shares (Basic)',
]
SIMFIN_CASHFLOW_FIELDS: List[str] = [
    'Net Cash from Operating Activities',
    'Change in Fixed Assets & Intangibles',
]

ALL_CORE_FIELDS: Set[str] = set(
    SIMFIN_INCOME_FIELDS + SIMFIN_BALANCE_FIELDS + SIMFIN_CASHFLOW_FIELDS
)

# Bump when XBRL_TO_FIELD or parse logic changes to force re-parse from raw JSON.
EDGAR_MAPPING_VERSION: int = 4

# SEC EDGAR rate limiting
SEC_HEADERS: Dict[str, str] = {'User-Agent': 'DropScore michael@dropscore.dev'}
SEC_RATE_DELAY: float = 0.12


# ── Instantiate defaults ─────────────────────────────────
TRADING = TradingConfig()
FEATURES = FeatureConfig()
MODEL = ModelConfig()
TARGETS = TargetConfig()
DATA = DataConfig()
EQUITY = EquityConfig()
BENCHMARKS = BenchmarkGates()
REGIME = RegimeConfig()

# ── Timing ───────────────────────────────────────────────
t_start: float = time.time()


# ── Backward-compatible aliases ──────────────────────────
# These allow existing module imports to work during incremental migration.
# Each maps the old flat constant name to its new dataclass location.
SIMFIN_KEY = DATA.simfin_key
FMP_KEY = DATA.fmp_key
FWD_WINDOWS = list(TARGETS.fwd_windows)
DROP_THRESH = list(TARGETS.drop_thresholds)
EXCESS_THRESH = list(TARGETS.excess_thresholds)
N_FOLDS = MODEL.n_folds
N_BOOT = MODEL.n_boot
HOLDOUT_MO = MODEL.holdout_months
MIN_EVENTS = MODEL.min_events
MAX_BASE_RATE = MODEL.max_base_rate
MIN_K = MODEL.min_k
NAN_DROP = MODEL.nan_drop
SLIPPAGE = TRADING.slippage
STOP_LOSS = TRADING.stop_loss_pct
PROFIT_TARGET = TRADING.profit_target_pct
TRAILING_STOP = TRADING.trailing_stop_pct
VOL_FLOOR = TRADING.min_volume
ENTRY_DELAY = TRADING.entry_delay
REGIME_SPY_MAX = REGIME.spy_max_21d
REGIME_VIX_MIN = REGIME.vix_min
SECTOR_CAP = REGIME.sector_cap
SKIP_RET5D_DOWN = REGIME.skip_ret5d_down
SKIP_VOL_PCT = REGIME.skip_vol_pct
FORCE_RECOMPUTE = DATA.force_recompute
TRADING_TARGET = TRADING.target
TRADING_HOLD = TRADING.hold_days
ENTRY_MODE = TRADING.entry_mode
CONFIRMATION_DROP = TRADING.confirm_drop_pct
CONFIRMATION_WINDOW = TRADING.confirm_window_days
BORROW_RATE_EASY = TRADING.borrow_rate_easy
BORROW_RATE_HARD = TRADING.borrow_rate_hard
UNIVERSE_MODE = "both"
UNIVERSE_A_FEATURES = list(FEATURES.universe_a)
UNIVERSE_B_FEATURES = list(FEATURES.universe_b)
ACCT = EQUITY.account_start
MAX_POS = EQUITY.max_positions
POS_SIZE = EQUITY.position_size


# ── SimFin column name setup ─────────────────────────────
# This block ensures SimFin's Python namespace has the column name constants
# that data.py and features.py rely on (e.g., REVENUE = 'Revenue').
import simfin as sf
from simfin.names import *

for _const_name, _col_value in SIMFIN_COLUMN_NAMES.items():
    if _const_name not in dir():
        exec(f"{_const_name}={_col_value!r}")
