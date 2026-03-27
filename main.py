"""
DROP SCORE v17 — CONVICTION + REGIME + REALISM
1-day entry delay. Top-quartile default. Dynamic mid-Q regime check.
4 paper trade scenarios (incl ramp-up). Vol-adj pure-fundamental test.
Truncated holdout. Staged entry diagnostic. Tech-only WF. Earnings proximity.
"""
import subprocess, sys, os, time, warnings, random, pickle
from datetime import datetime

# ── Install dependencies if needed ──
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg in ["simfin", "yfinance", "xgboost", "lightgbm", "scikit-learn",
            "matplotlib", "tqdm"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        install(pkg)

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)

# ── Tee stdout to results/ ──
os.makedirs('results', exist_ok=True)
_log_path = os.path.join('results', f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

_log_file = open(_log_path, 'w')
sys.stdout = _Tee(sys.__stdout__, _log_file)

# ── Project modules ──
from config import (
    N_BOOT, VOL_FLOOR, t_start,
)
from utils import elapsed
from data import load_all_data
from features import prepare_features
from model import run_vulnerability_model
from walkforward import run_walkforward
from equity import run_equity_scenarios

# ═══════════════════════════════════════════════════════════════
# BANNER
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("DROP SCORE v17 — CONVICTION + REGIME + REALISM")
print("  1-day delay | Top-quartile | Dynamic regime | Ramp-up scenarios")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════
# 1. DATA
# ═══════════════════════════════════════════════════════════════
data = load_all_data()

# ═══════════════════════════════════════════════════════════════
# 2. FEATURES
# ═══════════════════════════════════════════════════════════════
data = prepare_features(data)

# ═══════════════════════════════════════════════════════════════
# 3. VULNERABILITY MODEL
# ═══════════════════════════════════════════════════════════════
data = run_vulnerability_model(data)

# ═══════════════════════════════════════════════════════════════
# 4. WALK-FORWARD
# ═══════════════════════════════════════════════════════════════
data = run_walkforward(data)

# ═══════════════════════════════════════════════════════════════
# 5. EQUITY CURVES + DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════
data = run_equity_scenarios(data)

# ═══════════════════════════════════════════════════════════════
# 6. HOLDOUT + FINAL RESULTS
# ═══════════════════════════════════════════════════════════════
print("HOLDOUT...")
best_v_t = data['best_v_t']
best_v_r = data['best_v_r']
df_dev = data['df_dev']
df_hold = data['df_hold']
df_q = data['df_q']
topf_v = data['topf_v']
v_results = data['v_results']
wf_df = data['wf_df']
wf_top = data['wf_top']
eq_results = data['eq_results']
cache_dir = data['cache_dir']
fcols_q = data['fcols_q']
tgt_rates = data['tgt_rates']
all_wf_trades = data['all_wf_trades']

ho_auc = np.nan
try:
    if (best_v_t in df_hold.columns
            and df_hold[best_v_t].sum() >= 10
            and 'vuln_score' in df_hold.columns):
        yh = df_hold[best_v_t].fillna(0).astype(int)
        hp = df_hold['vuln_score'].values
        valid = ~np.isnan(hp)
        if valid.sum() >= 20 and yh[valid].sum() >= 5:
            ho_auc = roc_auc_score(yh[valid], hp[valid])
            htk = df_hold['ticker'].values[valid]
            utk = np.unique(htk)
            tidx2 = {t: np.where(htk == t)[0] for t in utk}
            hb = []
            for _ in range(N_BOOT):
                bt2 = np.random.choice(utk, len(utk), replace=True)
                idx = np.concatenate([tidx2[t] for t in bt2])
                if (len(idx) > 0
                        and yh.values[valid][idx].sum() > 0
                        and yh.values[valid][idx].sum() < len(idx)):
                    try:
                        hb.append(roc_auc_score(yh.values[valid][idx], hp[valid][idx]))
                    except:
                        pass
            hcl = np.percentile(hb, 2.5) if hb else ho_auc
            hch = np.percentile(hb, 97.5) if hb else ho_auc
            print(f"  Holdout: {ho_auc:.3f} [{hcl:.3f},{hch:.3f}]")
            print(f"  Dev:     {best_v_r['mauc']:.3f} [{best_v_r['clo']:.3f},{best_v_r['chi']:.3f}]")
            gap = best_v_r['mauc'] - ho_auc
            if abs(gap) < 0.05:
                print(f"  \u2705 Consistent ({gap:+.3f})")
            else:
                print(f"  \u26a0\ufe0f  Gap: {gap:+.3f}")
except Exception as e:
    print(f"  Holdout error: {e}")
print()

# FULL RESULTS
ba = best_v_r['mauc']
cl = best_v_r['clo']
print("\u2588" * 70)
print("  DROP SCORE v17 \u2014 FULL RESULTS")
print("\u2588" * 70)
print(f"""
DATA: {df_q['ticker'].nunique()} stocks | {len(df_dev):,} dev + {len(df_hold):,} hold
VULN: {best_v_t} Dev={ba:.3f} [{cl:.3f},{best_v_r['chi']:.3f}] Hold={ho_auc:.3f}
Features: {topf_v}
""")
va_tgts_r = [t for t in v_results if t.startswith('voladj_')]
bva = max(va_tgts_r, key=lambda k: v_results[k]['mauc']) if va_tgts_r else None
va_auc = v_results[bva]['mauc'] if bva else 0
print(f"VOL-ADJUSTED: {bva or 'N/A'} AUC={va_auc:.3f}")
if len(wf_df) > 0:
    print(f"""
WALK-FORWARD: {len(wf_df)} trades (1-day delay, vol floor {VOL_FLOOR:,})
  Full:    win={(wf_df['pnl_per_share']>0).mean():.0%} avg=${wf_df['pnl_per_share'].mean():+.2f}/sh ({wf_df['pnl_pct'].mean()*100:+.1f}%) stops={wf_df['stopped'].mean():.0%}
  Top 25%: win={(wf_top['pnl_per_share']>0).mean():.0%} avg=${wf_top['pnl_per_share'].mean():+.2f}/sh ({wf_top['pnl_pct'].mean()*100:+.1f}%) stops={wf_top['stopped'].mean():.0%}
  Quarters: {(wf_top.groupby('quarter')['pnl_per_share'].sum()>0).sum()}/{wf_top['quarter'].nunique()} profitable
""")
# Print scenarios
for lbl, r in eq_results.items():
    print(f"  {lbl}: ${r['start']:,}\u2192${r['end']:,.0f} ({r['ret']:+.0%}) "
          f"DD={r['max_dd_pct']:.0%} Calmar={r['calmar']:.2f}")

signal_icon = '\u2705' if ba >= 0.70 else '\u26a0\ufe0f'
alpha_icon = '\u2705' if len(wf_top) > 0 and wf_top['pnl_per_share'].mean() > 0 else '\u26a0\ufe0f'
print(f"""
VERDICT:
  Signal: {signal_icon} Dev={ba:.3f} Hold={ho_auc:.3f} VolAdj={va_auc:.3f}
  Alpha: {alpha_icon}
  Conviction: Top 25% outperforms \u2192 trade only highest-scored picks
""")

# Top 20 trades inline
if len(wf_top) > 0:
    print("FIRST 20 TRADES (top-25% conviction):")
    for i, (_, r) in enumerate(wf_top.sort_values('entry_date').head(20).iterrows()):
        ex = 'STOP' if r['stopped'] else 'PT' if r['profit_taken'] else 'EXP'
        print(f"  {r['ticker']:<6} {str(r['entry_date'])[:10]} "
              f"${r['entry_price']:>6.0f}\u2192${r['exit_price']:>6.0f} {ex:<4} "
              f"${r['pnl_per_share']:+.2f}/sh ({r['pnl_pct']*100:+.1f}%)")

# Save to Drive
with open(os.path.join(cache_dir, 'v17_results.pkl'), 'wb') as f:
    pickle.dump({
        'best_v_t': best_v_t, 'best_v_r': best_v_r, 'topf_v': topf_v, 'K': data['K'],
        'ho_auc': ho_auc, 'v_results': v_results, 'wf_trades': all_wf_trades,
        'wf_top_trades': wf_top.to_dict('records') if len(wf_top) > 0 else [],
        'eq_results': {
            k: {kk: vv for kk, vv in v.items() if kk != 'eq_df' and kk != 'trade_log'}
            for k, v in eq_results.items()
        },
        'fcols_q': fcols_q, 'tgt_rates': tgt_rates,
    }, f)

print(f"\nTotal: {(time.time()-t_start)/60:.1f} min")
print(f"Next run (voladj cached, skip list active): ~9 min")
print("Drop Score v17 complete.")

# Close log
print(f"\nLog saved to {_log_path}")
sys.stdout = sys.__stdout__
_log_file.close()
print(f"Log saved to {_log_path}")
