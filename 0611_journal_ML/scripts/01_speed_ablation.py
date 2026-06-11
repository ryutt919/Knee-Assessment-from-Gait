#!/usr/bin/env python3
"""
01_speed_ablation.py — H2: Multi-speed vs Single-speed Ablation

Hypothesis H2: Multi-speed features > single-speed features for ACL classification.

Conditions:
  slow_only   — only slow-speed scalar features per subject
  normal_only — only normal-speed scalar features
  fast_only   — only fast-speed scalar features
  all_speeds  — slow+normal+fast+delta (reference, ≈ 0529_ML AUC=0.9600)

CV: StratifiedKFold(5) outer / StratifiedKFold(3) inner Optuna(50 trials)
Models: RF, XGBoost, LightGBM, SVC-RBF
Metrics: AUC (95% bootstrap CI), paired bootstrap vs all_speeds
"""

import warnings
warnings.filterwarnings("ignore")

import os, sys, json, time, logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import xgboost as xgb
import lightgbm as lgb

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
SANDBOX   = Path(__file__).resolve().parents[1]
DATA_DIR  = ROOT / "data" / "processed"
RESULTS   = SANDBOX / "results"
LOGS      = SANDBOX / "logs"
RESULTS.mkdir(exist_ok=True)
LOGS.mkdir(exist_ok=True)

SCALAR_PATH = DATA_DIR / "features_scalar.csv"
SCALAR_BASELINE_AUC = 0.9600  # 0529_ML RF result

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOGS / "01_speed_ablation.log")]
)
log = logging.getLogger(__name__)

RANDOM_STATE = 42
N_TRIALS     = 50
OUTER_FOLDS  = 5
INNER_FOLDS  = 3
N_BOOTSTRAP  = 1000


# ── Data helpers ───────────────────────────────────────────────────────────────

def load_scalar() -> pd.DataFrame:
    df = pd.read_csv(SCALAR_PATH)
    log.info(f"Loaded features_scalar: {df.shape}")
    return df


def make_subject_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot 237-row (79-subject × 3-speed) scalar CSV → 79-row subject-level dataset.
    Creates slow_X, normal_X, fast_X, delta_fast_slow_X, delta_fast_normal_X, mean_X
    for each numeric feature X.
    """
    META = ['subject_id', 'group', 'speed', 'injured_leg']
    numeric_cols = [c for c in df.columns if c not in META]

    groups = {}
    for spd, sub in df.groupby('speed'):
        sub = sub.set_index('subject_id')[numeric_cols]
        groups[spd] = sub.add_prefix(f"{spd}_")

    pivot = pd.concat(groups.values(), axis=1)

    # delta features
    if 'slow' in groups and 'fast' in groups:
        for col in numeric_cols:
            pivot[f"delta_fast_slow_{col}"] = groups['fast'][f"fast_{col}"] - groups['slow'][f"slow_{col}"]
    if 'normal' in groups and 'fast' in groups:
        for col in numeric_cols:
            pivot[f"delta_fast_normal_{col}"] = groups['fast'][f"fast_{col}"] - groups['normal'][f"normal_{col}"]
    if len(groups) == 3:
        for col in numeric_cols:
            vals = [groups[s][f"{s}_{col}"] for s in groups]
            pivot[f"mean_{col}"] = sum(vals) / 3

    # Attach label
    label_map = df.drop_duplicates('subject_id').set_index('subject_id')['group']
    pivot['group'] = pivot.index.map(label_map)
    pivot['binary_label'] = (pivot['group'] != 'HA').astype(int)
    pivot = pivot.reset_index().rename(columns={'subject_id': 'subject_id'})
    log.info(f"Subject pivot: {pivot.shape}")
    return pivot


def make_speed_condition(pivot: pd.DataFrame, condition: str):
    """Extract feature matrix for a given speed condition."""
    binary_label = pivot['binary_label'].values
    subject_id   = pivot['subject_id'].values if 'subject_id' in pivot.columns else np.arange(len(pivot))

    if condition == 'slow_only':
        feat_cols = [c for c in pivot.columns if c.startswith('slow_')]
    elif condition == 'normal_only':
        feat_cols = [c for c in pivot.columns if c.startswith('normal_')]
    elif condition == 'fast_only':
        feat_cols = [c for c in pivot.columns if c.startswith('fast_')]
    elif condition == 'all_speeds':
        excl = {'subject_id', 'group', 'binary_label'}
        feat_cols = [c for c in pivot.columns if c not in excl]
    else:
        raise ValueError(f"Unknown condition: {condition}")

    X = pivot[feat_cols].fillna(0).values
    return X, binary_label, subject_id, feat_cols


# ── Optuna objectives ──────────────────────────────────────────────────────────

def make_rf_objective(X_tr, y_tr, inner_cv):
    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            max_depth=trial.suggest_int("max_depth", 2, 10),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_float("max_features", 0.1, 1.0),
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model = RandomForestClassifier(**params)
        pipe  = Pipeline([('scaler', StandardScaler()), ('clf', model)])
        return cross_val_score(pipe, X_tr, y_tr, cv=inner_cv, scoring='roc_auc', n_jobs=-1).mean()
    return objective


def make_xgb_objective(X_tr, y_tr, inner_cv):
    scale_pos = (y_tr == 0).sum() / (y_tr == 1).sum()
    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            scale_pos_weight=scale_pos,
            random_state=RANDOM_STATE, eval_metric='auc',
            verbosity=0,
        )
        model = xgb.XGBClassifier(**params)
        pipe  = Pipeline([('scaler', StandardScaler()), ('clf', model)])
        return cross_val_score(pipe, X_tr, y_tr, cv=inner_cv, scoring='roc_auc', n_jobs=1).mean()
    return objective


def make_lgb_objective(X_tr, y_tr, inner_cv):
    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            class_weight='balanced',
            random_state=RANDOM_STATE, verbose=-1,
        )
        model = lgb.LGBMClassifier(**params)
        pipe  = Pipeline([('scaler', StandardScaler()), ('clf', model)])
        return cross_val_score(pipe, X_tr, y_tr, cv=inner_cv, scoring='roc_auc', n_jobs=1).mean()
    return objective


def make_svc_objective(X_tr, y_tr, inner_cv):
    def objective(trial):
        C     = trial.suggest_float("C", 1e-3, 1e3, log=True)
        gamma = trial.suggest_float("gamma", 1e-5, 1.0, log=True)
        model = SVC(C=C, gamma=gamma, kernel='rbf', class_weight='balanced',
                    probability=True, random_state=RANDOM_STATE)
        pipe  = Pipeline([('scaler', StandardScaler()), ('clf', model)])
        return cross_val_score(pipe, X_tr, y_tr, cv=inner_cv, scoring='roc_auc', n_jobs=-1).mean()
    return objective


OBJECTIVE_BUILDERS = {
    'rf':   make_rf_objective,
    'xgb':  make_xgb_objective,
    'lgb':  make_lgb_objective,
    'svc':  make_svc_objective,
}

MODEL_BUILDERS = {
    'rf':  lambda p: Pipeline([('scaler', StandardScaler()),
                                ('clf', RandomForestClassifier(**p, class_weight='balanced',
                                                               random_state=RANDOM_STATE, n_jobs=-1))]),
    'xgb': lambda p: Pipeline([('scaler', StandardScaler()),
                                ('clf', xgb.XGBClassifier(**p, random_state=RANDOM_STATE,
                                                          verbosity=0, eval_metric='auc'))]),
    'lgb': lambda p: Pipeline([('scaler', StandardScaler()),
                                ('clf', lgb.LGBMClassifier(**p, random_state=RANDOM_STATE, verbose=-1,
                                                           class_weight='balanced'))]),
    'svc': lambda p: Pipeline([('scaler', StandardScaler()),
                                ('clf', SVC(**p, kernel='rbf', class_weight='balanced',
                                           probability=True, random_state=RANDOM_STATE))]),
}

XGB_KEYS = {'n_estimators','max_depth','learning_rate','subsample','colsample_bytree','reg_alpha','reg_lambda','scale_pos_weight'}
LGB_KEYS = {'n_estimators','num_leaves','learning_rate','subsample','colsample_bytree','reg_alpha','reg_lambda'}
RF_KEYS  = {'n_estimators','max_depth','min_samples_split','min_samples_leaf','max_features'}
SVC_KEYS = {'C','gamma'}

MODEL_PARAM_KEYS = {'rf': RF_KEYS, 'xgb': XGB_KEYS, 'lgb': LGB_KEYS, 'svc': SVC_KEYS}


# ── Statistical helpers ────────────────────────────────────────────────────────

def bootstrap_ci(y_true, y_scores, n=N_BOOTSTRAP, rs=RANDOM_STATE):
    rng  = np.random.default_rng(rs)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_scores[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def paired_bootstrap_pval(y_true, scores_a, scores_b, n=N_BOOTSTRAP, rs=RANDOM_STATE):
    """Paired bootstrap p-value: H0 = AUC_a == AUC_b"""
    rng   = np.random.default_rng(rs)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        da = roc_auc_score(y_true[idx], scores_a[idx])
        db = roc_auc_score(y_true[idx], scores_b[idx])
        diffs.append(da - db)
    diffs   = np.array(diffs)
    obs     = roc_auc_score(y_true, scores_a) - roc_auc_score(y_true, scores_b)
    centered = diffs - diffs.mean()
    return float(obs), float(np.mean(np.abs(centered) >= np.abs(obs)))


# ── Main nested-CV run ─────────────────────────────────────────────────────────

def run_condition_model(X, y, condition: str, model_name: str) -> dict:
    outer_cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_scores = np.zeros(len(y))
    oof_true   = np.zeros(len(y), dtype=int)
    fold_aucs  = []

    for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        study_name = f"{condition}_{model_name}_f{fold}"
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE + fold),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )
        obj = OBJECTIVE_BUILDERS[model_name](X_tr, y_tr, inner_cv)
        study.optimize(obj, n_trials=N_TRIALS, show_progress_bar=False)

        best_params = {k: v for k, v in study.best_params.items()
                       if k in MODEL_PARAM_KEYS[model_name]}
        if model_name == 'xgb':
            scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
            best_params['scale_pos_weight'] = scale_pos

        pipe = MODEL_BUILDERS[model_name](best_params)
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]

        oof_scores[te_idx] = proba
        oof_true[te_idx]   = y_te
        fold_aucs.append(roc_auc_score(y_te, proba))
        log.info(f"  {condition}/{model_name} fold={fold} auc={fold_aucs[-1]:.4f} "
                 f"best_val={study.best_value:.4f}")

    mean_auc = roc_auc_score(oof_true, oof_scores)
    ci_lo, ci_hi = bootstrap_ci(oof_true, oof_scores)

    return {
        'condition':   condition,
        'model':       model_name,
        'auc':         mean_auc,
        'ci_lo':       ci_lo,
        'ci_hi':       ci_hi,
        'fold_aucs':   fold_aucs,
        'oof_scores':  oof_scores.tolist(),
        'oof_true':    oof_true.tolist(),
    }


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log.info("=" * 60)
    log.info("01_speed_ablation.py — H2 Speed Ablation")
    log.info("=" * 60)

    df    = load_scalar()
    pivot = make_subject_pivot(df)

    conditions = ['slow_only', 'normal_only', 'fast_only', 'all_speeds']
    models     = ['rf', 'xgb', 'lgb', 'svc']

    all_results = []
    condition_oof = {}  # condition → best model oof scores (for comparison)

    for cond in conditions:
        X, y, subj, feat_cols = make_speed_condition(pivot, cond)
        log.info(f"\n{'─'*50}")
        log.info(f"Condition: {cond} | n={len(y)} | features={X.shape[1]}")
        log.info(f"  Class dist: HA={int((y==0).sum())} ACL={int((y==1).sum())}")

        best_auc   = -1
        best_model = None

        for mdl in models:
            log.info(f"\n  Model: {mdl}")
            res = run_condition_model(X, y, cond, mdl)
            all_results.append(res)

            if res['auc'] > best_auc:
                best_auc   = res['auc']
                best_model = res

        condition_oof[cond] = best_model

    # ── Pairwise comparison vs all_speeds ──────────────────────────────────────
    ref = condition_oof.get('all_speeds')
    comparison_rows = []

    if ref is not None:
        y_ref  = np.array(ref['oof_true'])
        s_ref  = np.array(ref['oof_scores'])

        for cond in ['slow_only', 'normal_only', 'fast_only']:
            cmp = condition_oof.get(cond)
            if cmp is None:
                continue
            s_cmp  = np.array(cmp['oof_scores'])
            diff, pval = paired_bootstrap_pval(y_ref, s_ref, s_cmp)
            comparison_rows.append({
                'comparison':  f"all_speeds vs {cond}",
                'auc_diff':    round(diff, 4),
                'pval_bootstrap': round(pval, 4),
                'ref_auc':     round(ref['auc'], 4),
                'cmp_auc':     round(cmp['auc'], 4),
                'significant': pval < 0.05,
            })
            log.info(f"\n  all_speeds({ref['auc']:.4f}) vs {cond}({cmp['auc']:.4f}): "
                     f"diff={diff:.4f} p={pval:.4f}")

    # ── Save results ───────────────────────────────────────────────────────────
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            'condition':    r['condition'],
            'model':        r['model'],
            'auc':          round(r['auc'], 4),
            'ci_lo':        round(r['ci_lo'], 4),
            'ci_hi':        round(r['ci_hi'], 4),
            'fold_auc_mean': round(np.mean(r['fold_aucs']), 4),
            'fold_auc_std':  round(np.std(r['fold_aucs']), 4),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(['condition', 'auc'], ascending=[True, False])
    summary_df.to_csv(RESULTS / "01_speed_ablation_results.csv", index=False)
    log.info(f"\nSaved: results/01_speed_ablation_results.csv")

    if comparison_rows:
        pd.DataFrame(comparison_rows).to_csv(RESULTS / "01_speed_comparison.csv", index=False)
        log.info("Saved: results/01_speed_comparison.csv")

    # Best per condition summary
    best_summary = {}
    for cond in conditions:
        cond_rows = [r for r in all_results if r['condition'] == cond]
        if cond_rows:
            best = max(cond_rows, key=lambda x: x['auc'])
            best_summary[cond] = {
                'model':  best['model'],
                'auc':    round(best['auc'], 4),
                'ci_lo':  round(best['ci_lo'], 4),
                'ci_hi':  round(best['ci_hi'], 4),
            }

    with open(RESULTS / "01_speed_ablation_best.json", 'w') as f:
        json.dump({
            'strategy': 'speed_ablation_nested_cv',
            'conditions': best_summary,
            'scalar_baseline_auc': SCALAR_BASELINE_AUC,
            'n_subjects': len(pivot),
            'outer_folds': OUTER_FOLDS,
            'n_trials': N_TRIALS,
            'elapsed_sec': round(time.time() - t0, 1),
        }, f, indent=2)

    log.info(f"\n{'='*60}")
    log.info("SPEED ABLATION SUMMARY")
    log.info(f"{'='*60}")
    for cond, info in best_summary.items():
        log.info(f"  {cond:15s}: AUC={info['auc']:.4f} [{info['ci_lo']:.4f},{info['ci_hi']:.4f}] ({info['model']})")
    log.info(f"\nTotal elapsed: {time.time()-t0:.1f}s")
    return summary_df


if __name__ == '__main__':
    main()
