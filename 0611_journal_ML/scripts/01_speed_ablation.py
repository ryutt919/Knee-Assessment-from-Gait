#!/usr/bin/env python3
"""
01_speed_ablation.py — H2: Multi-speed vs Single-speed Ablation

Hypothesis H2: Multi-speed features > single-speed features for ACL classification.

Conditions:
  slow_only   — only slow-speed scalar features per subject
  normal_only — only normal-speed scalar features
  fast_only   — only fast-speed scalar features
  all_speeds  — slow+normal+fast+delta (reference, ≈ 0529_ML AUC=0.9600)

CV: StratifiedKFold(5) outer / StratifiedKFold(3) inner Optuna
Models: logreg (baseline) + svm_rbf, rf, gbt, xgboost, lightgbm, catboost, tabpfn
        (shared registry in _models.py; tabpfn auto-skips if unavailable)
Metrics: AUC (95% bootstrap CI), paired bootstrap vs all_speeds
"""

import warnings
warnings.filterwarnings("ignore")

import os, sys, json, time, logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _models as M

SMOKE = os.environ.get("SMOKE") == "1"   # fast end-to-end test (tiny HPO, 2 models)

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
N_TRIALS     = 1 if SMOKE else 25   # per model per fold (7 models × 4 conditions × 5 folds)
OUTER_FOLDS  = 5
INNER_FOLDS  = 3
N_BOOTSTRAP  = 50 if SMOKE else 1000


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

        est, _, best_val = M.tune_and_build(
            model_name, X_tr, y_tr, inner_cv,
            n_trials=N_TRIALS, task="binary", seed=RANDOM_STATE + fold)
        est.fit(X_tr, y_tr)
        proba = est.predict_proba(X_te)[:, 1]

        oof_scores[te_idx] = proba
        oof_true[te_idx]   = y_te
        fold_aucs.append(roc_auc_score(y_te, proba))
        log.info(f"  {condition}/{model_name} fold={fold} auc={fold_aucs[-1]:.4f} "
                 f"best_val={best_val:.4f}")

    mean_auc = roc_auc_score(oof_true, oof_scores)
    ci_lo, ci_hi = bootstrap_ci(oof_true, oof_scores)

    return {
        'condition':   condition,
        'model':       model_name,
        'display':     M.DISPLAY[model_name],
        'role':        M.ROLE[model_name],
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
    models     = (['logreg', 'rf'] if SMOKE else M.available_models())
    skipped    = [m for m in M.ALL_MODELS if m not in models]
    if skipped:
        log.warning(f"Skipped (unavailable): {skipped}")

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
