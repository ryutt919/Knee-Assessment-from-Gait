#!/usr/bin/env python3
"""
02b_optimal_pipeline.py — Binary ACL vs HA, multi-model benchmark (side-fixed)

Pipeline (per outer fold, leakage-free):
  1. Scalar pivot features (slow/normal/fast + delta + mean)
  2. Stride-level gait variability (std + cv per inj/con × speed)  ← side-fixed
  3. Within-fold StandardScaler + RF top-20 feature selection
  4. Pairwise interaction terms (C(20,2) = 190)
  5. Final classifier tuned with Optuna (inner CV) — run for every model

Models: logreg (baseline) + svm_rbf, rf, gbt, xgboost, lightgbm, catboost, tabpfn
        (tabpfn auto-skips if weights/token unavailable)

Side fix: stride_level_peaks.parquet uses side ∈ {injured, contralateral}; the
previous version compared it against injured_leg ∈ {Right, Left} → never matched
→ every stride collapsed to `con`, dropping all injured-side variability. Now
standardized to {inj, con} via _side_utils.
"""

import warnings
warnings.filterwarnings("ignore")

import os, sys, json, time, logging
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _models as M
from _side_utils import build_injured_leg_map, to_inj_con

ROOT    = Path(__file__).resolve().parents[2]
SANDBOX = Path(__file__).resolve().parents[1]
DATA    = ROOT / "data" / "processed"
RESULTS = SANDBOX / "results"
FIGURES = SANDBOX / "figures"
LOGS    = SANDBOX / "logs"
for d in [RESULTS, FIGURES, LOGS]:
    d.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(LOGS / "02b_optimal.log")]
)
log = logging.getLogger(__name__)

SMOKE        = os.environ.get("SMOKE") == "1"   # fast end-to-end test
OUTER_FOLDS  = 5
INNER_FOLDS  = 3
RANDOM_STATE = 42
ENS_SEEDS    = [42] if SMOKE else [42, 88]   # 2-seed ensemble for the headline best model
TOP_K        = 20                # features selected per fold
N_TRIALS     = 1 if SMOKE else 25            # Optuna trials per model per fold
N_BOOTSTRAP  = 50 if SMOKE else 2000

MODELS = (['logreg', 'rf'] if SMOKE else M.ALL_MODELS)   # tabpfn filtered by availability


# ── Feature construction ──────────────────────────────────────────────────────

def build_scalar_pivot(scalar_df):
    META = ['subject_id', 'group', 'speed', 'injured_leg']
    num_cols = [c for c in scalar_df.columns if c not in META]
    sc_piv = {}
    for spd, sub in scalar_df.groupby('speed'):
        sc_piv[spd] = sub.set_index('subject_id')[num_cols]
    pivot = pd.concat([s.add_prefix(f'{k}_') for k, s in sc_piv.items()], axis=1).dropna()
    for c in num_cols:
        pivot[f'dfs_{c}'] = sc_piv['fast'][c] - sc_piv['slow'][c]
        pivot[f'dfn_{c}'] = sc_piv['fast'][c] - sc_piv['normal'][c]
        pivot[f'mean_{c}'] = (sc_piv['fast'][c] + sc_piv['slow'][c] + sc_piv['normal'][c]) / 3
    log.info(f"Scalar pivot: {pivot.shape[0]} subjects × {pivot.shape[1]} features")
    return pivot, sc_piv


def build_stride_variability(stride_df, inj_map):
    """std + cv of each stride feature per (subject, speed, inj/con).

    side-fixed: `side_std` ∈ {inj, con} resolved via _side_utils (handles both
    injured/contralateral and Right/Left encodings)."""
    feat45 = [c for c in stride_df.columns if stride_df[c].dtype in [np.float64, np.float32]]
    stride_df = stride_df.copy()
    stride_df['side_std'] = stride_df.apply(
        lambda r: to_inj_con(r['side'], inj_map.get(r['subject_id'], 'Right')), axis=1)
    n_unresolved = int(stride_df['side_std'].isna().sum())
    if n_unresolved:
        log.warning(f"Stride rows with unresolved side: {n_unresolved}")
    var_rows = []
    for (subj, speed, side_std), grp in stride_df.dropna(subset=['side_std']).groupby(
            ['subject_id', 'speed', 'side_std']):
        pfx = f"{side_std}_{speed}"
        row = {'subject_id': subj}
        for f in feat45:
            v = grp[f].dropna().values
            if len(v) < 3:
                continue
            row[f'{pfx}_std_{f}'] = np.std(v)
            row[f'{pfx}_cv_{f}']  = np.std(v) / abs(np.mean(v)) if abs(np.mean(v)) > 1e-6 else 0
        var_rows.append(row)
    var_df = pd.DataFrame(var_rows).groupby('subject_id').mean()
    n_inj = sum(c.startswith('inj_') for c in var_df.columns)
    n_con = sum(c.startswith('con_') for c in var_df.columns)
    log.info(f"Stride variability: {var_df.shape[0]} subjects × {var_df.shape[1]} features "
             f"(inj={n_inj}, con={n_con})")
    return var_df


# ── Within-fold interaction CV for ONE model ──────────────────────────────────

def run_cv_model(X, y, model_name, seed, topK=TOP_K, n_trials=N_TRIALS):
    """One seed pass: within-fold RF selection + interactions + Optuna-tuned model.
    Returns OOF probabilities for the positive class."""
    skf   = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=False)
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    probs = np.zeros(len(y))
    fold_aucs = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])

        # Feature selection on training fold only
        sel = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                     random_state=seed, n_jobs=-1)
        sel.fit(Xtr, y[tr])
        top = np.argsort(sel.feature_importances_)[-topK:]

        Xtr_top, Xte_top = Xtr[:, top], Xte[:, top]
        inter_tr = [Xtr_top[:, i] * Xtr_top[:, j] for i, j in combinations(range(topK), 2)]
        inter_te = [Xte_top[:, i] * Xte_top[:, j] for i, j in combinations(range(topK), 2)]
        Xtr_a = np.hstack([Xtr, np.column_stack(inter_tr)])
        Xte_a = np.hstack([Xte, np.column_stack(inter_te)])

        est, _, _ = M.tune_and_build(model_name, Xtr_a, y[tr], inner,
                                     n_trials=n_trials, task="binary", seed=seed)
        est.fit(Xtr_a, y[tr])
        probs[te] = est.predict_proba(Xte_a)[:, 1]
        fold_aucs.append(roc_auc_score(y[te], probs[te]))

    oof = roc_auc_score(y, probs)
    log.info(f"  [{model_name}] seed={seed} OOF AUC: {oof:.4f}  folds={[round(f,3) for f in fold_aucs]}")
    return probs, oof, fold_aucs


def benchmark_model(X, y, model_name, ensemble=False):
    """Single-pass benchmark (seed=42). If ensemble=True, also average seeds."""
    seeds = ENS_SEEDS if ensemble else [RANDOM_STATE]
    all_probs = []
    fold_aucs_primary = []
    for seed in seeds:
        p, oof, folds = run_cv_model(X, y, model_name, seed)
        all_probs.append(p)
        if seed == RANDOM_STATE:
            fold_aucs_primary = folds
    probs = np.mean(all_probs, axis=0)
    return probs, roc_auc_score(y, probs), fold_aucs_primary


# ── Stats / figure ─────────────────────────────────────────────────────────────

def bootstrap_ci(y, probs, n=N_BOOTSTRAP, alpha=0.05):
    rng = np.random.default_rng(RANDOM_STATE)
    boot = []
    for _ in range(n):
        idx = rng.choice(len(y), len(y), replace=True)
        if 0 < y[idx].sum() < len(idx):
            boot.append(roc_auc_score(y[idx], probs[idx]))
    lo, hi = np.percentile(boot, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return float(np.mean(boot)), float(np.median(boot)), float(lo), float(hi)


def save_roc_figure(y, probs, auc_val, ci_lo, ci_hi, model_disp, path):
    fpr, tpr, _ = roc_curve(y, probs)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, color='#2563EB',
            label=f'AUC = {auc_val:.4f}\n95% CI [{ci_lo:.3f}, {ci_hi:.3f}]')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'ROC — ACL vs HA ({model_disp})\n(Scalar + Variability + Interactions, side-fixed)',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Saved: {path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log.info("=" * 60)
    log.info("02b_optimal_pipeline.py — Multi-model benchmark (side-fixed)")
    log.info("=" * 60)

    scalar_df = pd.read_csv(DATA / "features_scalar.csv")
    stride_df = pd.read_parquet(DATA / "stride_level_peaks.parquet")

    inj_map = build_injured_leg_map(scalar_df)

    pivot, _ = build_scalar_pivot(scalar_df)
    var_df   = build_stride_variability(stride_df, inj_map)

    sc_meta = scalar_df.drop_duplicates('subject_id').set_index('subject_id')[['group']]
    sc_meta['binary_label'] = (sc_meta['group'] != 'HA').astype(int)

    combined  = pivot.join(var_df, how='inner').join(sc_meta).dropna()
    feat_cols = [c for c in combined.columns if c not in {'group', 'binary_label'}]
    X = combined[feat_cols].fillna(0).values.astype(np.float32)
    y = combined['binary_label'].values
    log.info(f"Combined feature matrix: {X.shape}  ACL={y.sum()}  HA={(y==0).sum()}")

    models = M.available_models(MODELS)
    skipped = [m for m in MODELS if m not in models]
    if skipped:
        log.warning(f"Skipped (unavailable): {skipped}")

    # ── Benchmark every model (single-pass, seed=42) ──────────────────────────
    bench = []
    oof_store = {}
    for name in models:
        log.info(f"\n--- {M.DISPLAY[name]} ({M.ROLE[name]}) ---")
        probs, oof_auc, folds = benchmark_model(X, y, name, ensemble=False)
        b_mean, b_med, ci_lo, ci_hi = bootstrap_ci(y, probs)
        oof_store[name] = probs
        bench.append({
            'model': name, 'display': M.DISPLAY[name], 'role': M.ROLE[name],
            'oof_auc': round(oof_auc, 4),
            'ci_95_lo': round(ci_lo, 4), 'ci_95_hi': round(ci_hi, 4),
            'bootstrap_median': round(b_med, 4),
            'fold_aucs': [round(f, 4) for f in folds],
        })

    bench.sort(key=lambda r: r['oof_auc'], reverse=True)
    best = bench[0]
    log.info(f"\nBest single-pass model: {best['display']} AUC={best['oof_auc']:.4f}")

    # ── 2-seed ensemble for the headline best model (official number) ─────────
    log.info(f"\n--- 2-seed ensemble for headline ({best['display']}) ---")
    ens_probs, ens_auc, ens_folds = benchmark_model(X, y, best['model'], ensemble=True)
    eb_mean, eb_med, eci_lo, eci_hi = bootstrap_ci(y, ens_probs)

    # Subject-level predictions for the headline model
    subj_df = pd.DataFrame({
        'subject_id': np.array(combined.index.tolist()),
        'group': combined['group'].values,
        'binary_label': y,
        'oof_prob': ens_probs,
    })
    subj_df.to_csv(RESULTS / "02b_subject_predictions.csv", index=False)

    save_roc_figure(y, ens_probs, ens_auc, eci_lo, eci_hi, M.DISPLAY[best['model']],
                    FIGURES / "fig_02b_roc.png")

    result = {
        'task': 'binary_acl_vs_ha',
        'side_fixed': True,
        'feature_sets': ['scalar_pivot', 'stride_variability_inj_con', 'within_fold_interactions_190'],
        'n_subjects': int(len(y)), 'n_acl': int(y.sum()), 'n_ha': int((y == 0).sum()),
        'cv_strategy': f'StratifiedKFold({OUTER_FOLDS}, shuffle=False) + inner Optuna({N_TRIALS})',
        'baseline_model': M.BASELINE,
        'benchmark': bench,
        'headline': {
            'model': best['model'], 'display': M.DISPLAY[best['model']],
            'ens_seeds': ENS_SEEDS,
            'ens_oof_auc': round(ens_auc, 4),
            'bootstrap_median': round(eb_med, 4),
            'ci_95_lo': round(eci_lo, 4), 'ci_95_hi': round(eci_hi, 4),
            'fold_aucs': [round(f, 4) for f in ens_folds],
            'target_achieved': bool(ens_auc >= 0.98),
        },
        'skipped_models': skipped,
        'elapsed_s': round(time.time() - t0, 1),
    }
    with open(RESULTS / "02b_optimal_results.json", 'w') as f:
        json.dump(result, f, indent=2)
    pd.DataFrame(bench).to_csv(RESULTS / "02b_benchmark.csv", index=False)
    log.info("Saved: 02b_optimal_results.json + 02b_benchmark.csv")

    log.info(f"\n{'='*60}\nBENCHMARK (ACL vs HA, side-fixed)")
    log.info(f"{'model':<20}{'role':<11}{'AUC':>7}   95% CI")
    for r in bench:
        log.info(f"  {r['display']:<18}{r['role']:<11}{r['oof_auc']:.4f}   "
                 f"[{r['ci_95_lo']:.3f}, {r['ci_95_hi']:.3f}]")
    log.info(f"\nHeadline (2-seed {M.DISPLAY[best['model']]}): AUC={ens_auc:.4f} "
             f"{'✓≥0.98' if ens_auc>=0.98 else '<0.98'}  CI[{eci_lo:.3f},{eci_hi:.3f}]")
    log.info(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
