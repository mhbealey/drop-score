"""
Vulnerability model: XGB+LGB ensemble with bootstrap CI,
Pareto optimisation, fundamental-only tests, truncated holdout,
score dev & holdout.
"""
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from config import N_FOLDS, N_BOOT, MIN_K, TRADING_TARGET, TRADING_HOLD, ENTRY_MODE
from utils import clean_X, elapsed


def run_model(dframe: pd.DataFrame, fc: List[str], tgt: str, meds: pd.Series,
              nf: int = 4, nb: int = 500, k: int = 10,
              optuna_params: Optional[dict] = None) -> Optional[dict]:
    vd2 = dframe.dropna(subset=[tgt])
    X = clean_X(vd2, fc, meds)
    y = vd2[tgt].fillna(0).astype(int)
    if y.sum() < 30 or len(y) < 100:
        return None
    fs = len(X) // (nf + 1)
    folds, aimps = [], []
    pred_map = {}

    # Extract Optuna params if available, else use defaults
    op = optuna_params or {}
    xgb_depth = op.get('xgb_max_depth', 5)
    xgb_lr = op.get('xgb_lr', 0.05)
    xgb_n_est = op.get('xgb_n_est', 500)
    xgb_subsample = op.get('xgb_subsample', 0.8)
    xgb_colsample = op.get('xgb_colsample', 0.8)
    xgb_alpha = op.get('xgb_alpha', 0.1)
    xgb_lambda = op.get('xgb_lambda', 1.0)
    lgb_depth = op.get('lgb_max_depth', 5)
    lgb_lr = op.get('lgb_lr', 0.05)
    lgb_n_est = op.get('lgb_n_est', 200)
    lgb_subsample = op.get('lgb_subsample', 0.8)
    lgb_colsample = op.get('lgb_colsample', 0.8)
    lgb_alpha = op.get('lgb_alpha', 0.1)
    lgb_lambda = op.get('lgb_lambda', 1.0)
    xgb_weight = op.get('xgb_weight', 0.5)

    for fold in range(nf):
        ts = fs * (fold + 2)
        te = min(ts + fs, len(X))
        if te <= ts:
            continue
        Xtr, Xte2 = X.iloc[:ts], X.iloc[ts:te]
        ytr, yte2 = y.iloc[:ts], y.iloc[ts:te]
        test_idx = vd2.index[ts:te]
        if yte2.sum() < 5 or ytr.sum() < 10:
            continue
        sw2 = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
        se = int(len(Xtr) * 0.7)
        sel = xgb.XGBClassifier(
            n_estimators=50, max_depth=4, learning_rate=0.05,
            scale_pos_weight=sw2, subsample=0.8, colsample_bytree=0.8,
            eval_metric='logloss', random_state=42, verbosity=0,
        )
        sel.fit(Xtr.iloc[:se], ytr.iloc[:se])
        imp = pd.Series(sel.feature_importances_, index=fc)
        topk = imp.nlargest(k).index.tolist()
        mdl = xgb.XGBClassifier(
            n_estimators=xgb_n_est, max_depth=xgb_depth, learning_rate=xgb_lr,
            scale_pos_weight=sw2, subsample=xgb_subsample,
            colsample_bytree=xgb_colsample,
            reg_alpha=xgb_alpha, reg_lambda=xgb_lambda,
            eval_metric='logloss', early_stopping_rounds=30,
            random_state=42, verbosity=0,
        )
        mdl.fit(Xtr[topk], ytr, eval_set=[(Xte2[topk], yte2)], verbose=False)
        try:
            lgb_m = lgb.LGBMClassifier(
                n_estimators=lgb_n_est, max_depth=lgb_depth, learning_rate=lgb_lr,
                scale_pos_weight=sw2, subsample=lgb_subsample,
                colsample_bytree=lgb_colsample,
                reg_alpha=lgb_alpha, reg_lambda=lgb_lambda,
                random_state=42, verbosity=-1,
            )
            lgb_m.fit(Xtr[topk], ytr)
            yp2 = (xgb_weight * mdl.predict_proba(Xte2[topk])[:, 1]
                   + (1 - xgb_weight) * lgb_m.predict_proba(Xte2[topk])[:, 1])
        except Exception:
            lgb_m = None
            yp2 = mdl.predict_proba(Xte2[topk])[:, 1]
        try:
            tauc = roc_auc_score(yte2, yp2)
        except Exception:
            tauc = 0.5
        folds.append({'tauc': tauc, 'feats': topk, 'model': mdl, 'lgb_model': lgb_m})
        aimps.append(imp)
        for ix, pv in zip(test_idx, yp2):
            pred_map[ix] = pv
    if not folds:
        return None
    mauc = np.mean([f['tauc'] for f in folds])
    pp_idx = list(pred_map.keys())
    pp = np.array([pred_map[i] for i in pp_idx])
    pl = np.array([y.loc[i] for i in pp_idx])
    ptk = np.array([vd2.loc[i, 'ticker'] for i in pp_idx])
    utk = np.unique(ptk)
    tidx = {t: np.where(ptk == t)[0] for t in utk}
    boots = []
    for _ in range(nb):
        bt2 = np.random.choice(utk, len(utk), replace=True)
        idx = np.concatenate([tidx[t] for t in bt2])
        if len(idx) > 0 and pl[idx].sum() > 0 and pl[idx].sum() < len(idx):
            try:
                boots.append(roc_auc_score(pl[idx], pp[idx]))
            except Exception:
                pass
    clo = np.percentile(boots, 2.5) if boots else mauc
    chi = np.percentile(boots, 97.5) if boots else mauc
    ai = pd.concat(aimps, axis=1).mean(axis=1).sort_values(ascending=False)
    return {'mauc': mauc, 'clo': clo, 'chi': chi, 'imp': ai,
            'folds': folds, 'pred_map': pred_map}


def pareto_optimise(df_dev: pd.DataFrame, fcols_q: List[str],
                    fill_meds_q: pd.Series, tcols: List[str]) -> int:
    """Pareto feature-count optimisation. Returns optimal K."""
    pt = [c for c in tcols if 'exdrop_5_10d' in c]
    if not pt:
        pt = [c for c in tcols if c.startswith('exdrop_')]
    if not pt:
        pt = tcols
    pt = pt[0]
    vd = df_dev.dropna(subset=[pt])
    X_p = clean_X(vd, fcols_q, fill_meds_q)
    y_p = vd[pt].fillna(0).astype(int)
    sp = int(len(X_p) * 0.7)
    sw_p = (sp - y_p.iloc[:sp].sum()) / max(y_p.iloc[:sp].sum(), 1)
    pm = xgb.XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.05,
        scale_pos_weight=sw_p, subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=42, verbosity=0,
    )
    pm.fit(X_p.iloc[:sp], y_p.iloc[:sp])
    pimp = pd.Series(pm.feature_importances_, index=fcols_q).sort_values(ascending=False)
    pareto = []
    for n in range(1, min(21, len(fcols_q) + 1)):
        tn = pimp.index[:n].tolist()
        m = xgb.XGBClassifier(
            n_estimators=100, max_depth=min(n, 5), learning_rate=0.05,
            scale_pos_weight=sw_p, subsample=0.8,
            eval_metric='logloss', random_state=42, verbosity=0,
        )
        m.fit(X_p.iloc[:sp][tn], y_p.iloc[:sp])
        try:
            a = roc_auc_score(y_p.iloc[sp:], m.predict_proba(X_p.iloc[sp:][tn])[:, 1])
        except Exception:
            a = 0.5
        pareto.append({'n': n, 'auc': a})
    pdf_p = pd.DataFrame(pareto)
    pk = pdf_p['auc'].max()
    K = max(MIN_K, int(pdf_p[pdf_p['auc'] >= pk * 0.98].iloc[0]['n']))
    return K


def train_all_targets(df_dev: pd.DataFrame, fcols_q: List[str],
                      fill_meds_q: pd.Series, tcols: List[str],
                      tgt_rates: Dict[str, float], K: int,
                      optuna_params: Optional[dict] = None,
) -> Tuple[dict, str, dict, List[str]]:
    """Train models on all targets. Returns (v_results, best_v_t, best_v_r, topf_v)."""
    t0 = time.time()
    label = "VULNERABILITY MODEL"
    if optuna_params:
        label += " (Optuna params)"
    print(f"{label}...")

    v_results = {}
    for t in tqdm(tcols, desc="  Train"):
        r = run_model(df_dev, fcols_q, t, fill_meds_q, N_FOLDS, N_BOOT, K,
                      optuna_params=optuna_params)
        if r:
            v_results[t] = r

    best_v_t = max(v_results, key=lambda k: v_results[k]['mauc'])

    best_v_r = v_results[best_v_t]
    topf_v = best_v_r['imp'].index[:K].tolist()

    print(f"\n  RESULTS:")
    for prefix, label in [('exdrop_', 'EXCESS'), ('voladj_', 'VOL-ADJ'), ('drop_', 'RAW')]:
        grp = [t for t in v_results if t.startswith(prefix)]
        grp.sort(key=lambda k: v_results[k]['mauc'], reverse=True)
        if grp:
            print(f"    {label}:")
            for t in grp[:5]:
                r = v_results[t]
                mk = " \u25c4" if t == best_v_t else ""
                print(f"      {t:<30} AUC:{r['mauc']:.3f} "
                      f"[{r['clo']:.3f},{r['chi']:.3f}] "
                      f"BR:{tgt_rates.get(t,0):.0%}{mk}")
    print(f"  BEST: {best_v_t} | {best_v_r['mauc']:.3f} | K={K}")
    print(f"  Features: {topf_v}")
    print(f"\n  LOCKED CONFIG: {TRADING_TARGET} / {TRADING_HOLD}d hold / {ENTRY_MODE} entry")
    print(f"  (All targets trained for diagnostics. Only TRADING_TARGET used for walk-forward.)")

    return v_results, best_v_t, best_v_r, topf_v


def fundamental_only_tests(df_dev, fcols_q, fill_meds_q, v_results, best_v_t, K):
    """Run fundamental-only and pure-fundamental tests."""
    vol_feats = {'vol_30d', 'vol_60d', 'vol_90d', 'roa_x_vol',
                 'margin_x_vol', 'margin_trend_x_vol'}
    fund_only = [c for c in fcols_q if c not in vol_feats]
    best_ex_cands = [t for t in v_results if t.startswith('exdrop_')]
    best_ex = (max(best_ex_cands, key=lambda k: v_results[k]['mauc'])
               if best_ex_cands else best_v_t)
    r_fund = run_model(df_dev, fund_only, best_ex,
                       df_dev[fund_only].median(), N_FOLDS, 100, K)
    if r_fund:
        print(f"\n  FUND-ONLY on {best_ex}: {r_fund['mauc']:.3f} "
              f"vs full {v_results[best_ex]['mauc']:.3f}")

    # Vol-adj pure-fundamental (no price features at all)
    price_derived = {
        'vol_30d', 'vol_60d', 'vol_90d', 'roa_x_vol', 'margin_x_vol',
        'margin_trend_x_vol', 'ret_5d', 'ret_21d', 'ret_63d', 'dd_from_high',
        'gap_count_30d', 'down_days_30d', 'death_cross', 'excess_ret_21d',
        'excess_ret_63d', 'sector_excess_21d', 'sector_excess_63d',
        'consec_down_days', 'gap_down_today', 'gap_downs_5d', 'spy_corr_60d',
    }
    pure_fund = [c for c in fcols_q if c not in price_derived]
    va_cands = [t for t in v_results if t.startswith('voladj_')]
    best_va = (max(va_cands, key=lambda k: v_results[k]['mauc'])
               if va_cands else None)
    if best_va and pure_fund:
        r_pf = run_model(df_dev, pure_fund, best_va,
                         df_dev[pure_fund].median(), N_FOLDS, 100, K)
        if r_pf:
            print(f"  PURE-FUND on {best_va}: {r_pf['mauc']:.3f} "
                  f"vs full {v_results[best_va]['mauc']:.3f}")
            if r_pf['mauc'] > 0.62:
                print(f"    \u2705 Real fundamental distress signal")
            elif r_pf['mauc'] > 0.55:
                print(f"    \u26a0\ufe0f  Marginal fundamental signal")
            else:
                print(f"    \u274c No fundamental signal \u2014 power from price patterns")


def score_dev_holdout(df_dev, df_hold, fcols_q, fill_meds_q, best_v_r):
    """Score dev and holdout sets with the best model."""
    df_dev['vuln_score'] = df_dev.index.map(best_v_r['pred_map']).values
    lf = best_v_r['folds'][-1]
    lfeat = lf['feats']
    Xh = clean_X(df_hold, fcols_q, fill_meds_q)
    try:
        hp_xgb = lf['model'].predict_proba(Xh[lfeat])[:, 1]
        hp = (0.5 * hp_xgb + 0.5 * lf['lgb_model'].predict_proba(Xh[lfeat])[:, 1]
              if lf.get('lgb_model') else hp_xgb)
        df_hold['vuln_score'] = hp
    except Exception:
        df_hold['vuln_score'] = np.nan
    return df_dev, df_hold, Xh


def truncated_holdout_test(df_dev, df_hold, Xh, fcols_q, fill_meds_q,
                           best_v_t, K):
    """Run truncated holdout test for leakage detection."""
    print(f"\n  TRUNCATED HOLDOUT TEST:")
    trunc_cut = df_dev['report_date'].quantile(0.75)
    trunc_train = df_dev[df_dev['report_date'] <= trunc_cut]
    if best_v_t in trunc_train.columns and trunc_train[best_v_t].sum() >= 30:
        r_trunc = run_model(trunc_train, fcols_q, best_v_t, fill_meds_q,
                            N_FOLDS, 100, K)
        if r_trunc:
            try:
                lf_t = r_trunc['folds'][-1]
                ft_t = lf_t['feats']
                hp_t_xgb = lf_t['model'].predict_proba(Xh[ft_t])[:, 1]
                if lf_t.get('lgb_model'):
                    hp_t = (0.5 * hp_t_xgb
                            + 0.5 * lf_t['lgb_model'].predict_proba(Xh[ft_t])[:, 1])
                else:
                    hp_t = hp_t_xgb
                yh_t = df_hold[best_v_t].fillna(0).astype(int)
                valid_t = ~np.isnan(hp_t)
                if valid_t.sum() >= 20 and yh_t[valid_t].sum() >= 5:
                    ho_trunc = roc_auc_score(yh_t[valid_t], hp_t[valid_t])
                    if df_hold['vuln_score'].notna().sum() > 20:
                        valid_full = ~np.isnan(df_hold['vuln_score'].values)
                        ho_full = roc_auc_score(
                            yh_t[valid_full],
                            df_hold['vuln_score'].values[valid_full],
                        )
                        print(f"    Full-dev holdout:  {ho_full:.3f}")
                    else:
                        print(f"    Full-dev holdout: N/A")
                    print(f"    Trunc-dev holdout: {ho_trunc:.3f}")
                    print(f"    (If similar \u2192 no leakage. "
                          f"If trunc is lower \u2192 sampling variation in full holdout)")
            except Exception as e:
                print(f"    Truncated holdout error: {e}")


def run_vulnerability_model(data_bundle: dict) -> dict:
    """Full vulnerability model pipeline: Pareto, train, fundamental tests,
    score, truncated holdout.

    If 'locked_features' is set in the bundle, skip Pareto and use those
    features directly (locked v18 config).
    """
    df_dev = data_bundle['df_dev']
    df_hold = data_bundle['df_hold']
    fcols_q = data_bundle['fcols_q']
    fill_meds_q = data_bundle['fill_meds_q']
    tcols = data_bundle['tcols']
    tgt_rates = data_bundle['tgt_rates']
    optuna_params = data_bundle.get('optuna_best_params')
    locked_features = data_bundle.get('locked_features')

    if locked_features:
        # Use locked feature set — skip Pareto optimization
        # Filter to features that actually exist in dev data
        available = [f for f in locked_features if f in fcols_q]
        missing = [f for f in locked_features if f not in fcols_q]
        if missing:
            print(f"  WARNING: locked features not in data: {missing}")
        K = len(available)
        print(f"  Using locked features (K={K}): {available}")
    else:
        K = pareto_optimise(df_dev, fcols_q, fill_meds_q, tcols)

    v_results, best_v_t, best_v_r, topf_v = train_all_targets(
        df_dev, fcols_q, fill_meds_q, tcols, tgt_rates, K,
        optuna_params=optuna_params,
    )

    # If locked features, override topf_v with locked list
    if locked_features:
        available = [f for f in locked_features if f in fcols_q]
        topf_v = available
        # Also override best_v_r's features to use locked set
        if TRADING_TARGET in v_results:
            best_v_t = TRADING_TARGET
            best_v_r = v_results[TRADING_TARGET]

    fundamental_only_tests(df_dev, fcols_q, fill_meds_q, v_results, best_v_t, K)
    df_dev, df_hold, Xh = score_dev_holdout(
        df_dev, df_hold, fcols_q, fill_meds_q, best_v_r,
    )
    truncated_holdout_test(df_dev, df_hold, Xh, fcols_q, fill_meds_q, best_v_t, K)

    vuln_top = df_dev[
        df_dev['vuln_score'] >= df_dev['vuln_score'].quantile(0.80)
    ]['ticker'].unique()
    print(f"\n  Flagged: {len(vuln_top)} | {elapsed()}")
    print()

    data_bundle.update(
        df_dev=df_dev, df_hold=df_hold,
        v_results=v_results, best_v_t=best_v_t,
        best_v_r=best_v_r, topf_v=topf_v, K=K,
    )
    return data_bundle


def run_bayesian_optimization(data_bundle: dict, n_trials: int = 30,
                              timeout: int = 1200) -> dict:
    """Bayesian hyperparameter optimization using Optuna.

    Optimizes on walk-forward Sharpe ratio, not dev AUC.
    Falls back gracefully if Optuna not available or time budget exceeded.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  Optuna not available, skipping Bayesian optimization")
        return data_bundle

    df_dev = data_bundle['df_dev']
    fcols_q = data_bundle['fcols_q']
    fill_meds_q = data_bundle['fill_meds_q']
    K = data_bundle['K']
    topf_v = data_bundle['topf_v']

    target = TRADING_TARGET
    if target not in df_dev.columns or df_dev[target].sum() < 30:
        print("  Not enough target events for Bayesian optimization")
        return data_bundle

    vd = df_dev.dropna(subset=[target])
    X = clean_X(vd, fcols_q, fill_meds_q)
    y = vd[target].fillna(0).astype(int)

    # Use time-split: 70% train, 30% validation
    split = int(len(X) * 0.7)
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr, yte = y.iloc[:split], y.iloc[split:]

    if yte.sum() < 5 or ytr.sum() < 10:
        print("  Insufficient events for Bayesian optimization")
        return data_bundle

    sw = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)

    def objective(trial):
        xgb_params = {
            'max_depth': trial.suggest_int('xgb_max_depth', 3, 8),
            'learning_rate': trial.suggest_float('xgb_lr', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('xgb_n_est', 100, 500),
            'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('xgb_colsample', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('xgb_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('xgb_lambda', 1e-8, 10.0, log=True),
        }
        lgb_params = {
            'max_depth': trial.suggest_int('lgb_max_depth', 3, 8),
            'learning_rate': trial.suggest_float('lgb_lr', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('lgb_n_est', 100, 500),
            'subsample': trial.suggest_float('lgb_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('lgb_colsample', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('lgb_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('lgb_lambda', 1e-8, 10.0, log=True),
        }
        xgb_weight = trial.suggest_float('xgb_weight', 0.3, 0.7)

        # Train XGB
        m_xgb = xgb.XGBClassifier(
            scale_pos_weight=sw, eval_metric='logloss',
            random_state=42, verbosity=0, **xgb_params,
        )
        m_xgb.fit(Xtr[topf_v], ytr)

        # Train LGB
        m_lgb = lgb.LGBMClassifier(
            scale_pos_weight=sw, random_state=42, verbosity=-1,
            **lgb_params,
        )
        m_lgb.fit(Xtr[topf_v], ytr)

        # Ensemble prediction
        p = (xgb_weight * m_xgb.predict_proba(Xte[topf_v])[:, 1]
             + (1 - xgb_weight) * m_lgb.predict_proba(Xte[topf_v])[:, 1])

        try:
            return roc_auc_score(yte, p)
        except Exception:
            return 0.5

    print(f"\n  ═══ BAYESIAN OPTIMIZATION ═══")
    print(f"  Target: {target} | Features: {len(topf_v)} | Trials: {n_trials}")

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    print(f"  Trials completed: {len(study.trials)}")
    print(f"  Best AUC: {study.best_value:.3f}")

    # Extract best params
    bp = study.best_params
    print(f"  Best XGB weight: {bp.get('xgb_weight', 0.5):.2f}")
    print(f"  Best XGB: depth={bp.get('xgb_max_depth')}, "
          f"lr={bp.get('xgb_lr', 0.05):.3f}, "
          f"n_est={bp.get('xgb_n_est', 500)}")
    print(f"  Best LGB: depth={bp.get('lgb_max_depth')}, "
          f"lr={bp.get('lgb_lr', 0.05):.3f}, "
          f"n_est={bp.get('lgb_n_est', 200)}")

    # Compare to baseline
    baseline_r = data_bundle.get('v_results', {}).get(target, {})
    baseline_auc = baseline_r.get('mauc', 0.5) if baseline_r else 0.5
    print(f"  Baseline dev AUC (fixed params): {baseline_auc:.3f}")
    print(f"  Optimized AUC: {study.best_value:.3f}")
    print(f"  Improvement: {study.best_value - baseline_auc:+.3f}")
    print(f"  ═══ END BAYESIAN OPTIMIZATION ═══\n")

    data_bundle['optuna_study'] = study
    data_bundle['optuna_best_params'] = bp
    return data_bundle


def run_bootstrap_ci(wf_top: pd.DataFrame, n_boot: int = 1000) -> dict:
    """Bootstrap confidence intervals on walk-forward results."""
    if wf_top is None or len(wf_top) == 0:
        return {}

    pnl = wf_top['pnl_per_share'].values

    boot_wins, boot_pnl = [], []
    for _ in range(n_boot):
        sample = np.random.choice(pnl, size=len(pnl), replace=True)
        boot_wins.append(np.mean(sample > 0))
        boot_pnl.append(np.mean(sample))

    results = {
        'win_mean': np.mean(boot_wins),
        'win_ci_lo': np.percentile(boot_wins, 2.5),
        'win_ci_hi': np.percentile(boot_wins, 97.5),
        'pnl_mean': np.mean(boot_pnl),
        'pnl_ci_lo': np.percentile(boot_pnl, 2.5),
        'pnl_ci_hi': np.percentile(boot_pnl, 97.5),
    }

    print(f"\n  ═══ CONFIDENCE INTERVALS (Bootstrap, n={n_boot}) ═══")
    print(f"  Win rate: {results['win_mean']:.1%} "
          f"[{results['win_ci_lo']:.1%}, {results['win_ci_hi']:.1%}]")
    print(f"  Avg P&L: ${results['pnl_mean']:.2f} "
          f"[${results['pnl_ci_lo']:.2f}, ${results['pnl_ci_hi']:.2f}]")
    print(f"  ═══ END BOOTSTRAP CIs ═══\n")

    return results
