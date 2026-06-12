#!/usr/bin/env python3
"""
02_raw_waveform_ml.py — H3: Raw Waveform Features vs Scalar Baseline

Core innovation: instead of hand-crafted scalar statistics, use the full
9-channel × 101-timepoint waveform directly as ML features.

Feature Sets (progressive information levels):
  [A] unilateral_injured  — injured-side waveforms per subject-speed (237 rows × 909)
  [B] bilateral_asym ★   — (injured−contralateral) waveforms per subject-speed (237 rows × 909)
  [C] speed_delta        — (fast_asym − slow_asym) per subject (79 rows × 909)
  [D] multi_speed_bilateral — [slow|normal|fast]_asym concat per subject (79 rows × 2727) ← target ≥98%

Processing: StandardScaler + PCA (95% variance, within-fold only) → RF/XGB/LGB + Optuna
CV: GroupKFold(5, group=subject_id) for [A],[B]; StratifiedKFold(5) for [C],[D]
Scalar Baseline: RF AUC=0.9600 (0529_ML)
"""

import warnings
warnings.filterwarnings("ignore")

import os, sys, json, time, logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _models as M
from _side_utils import build_injured_leg_map, to_inj_con


def _np_default(o):
    """JSON encoder fallback for numpy scalar types."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
SANDBOX   = Path(__file__).resolve().parents[1]
DATA_DIR  = ROOT / "data" / "processed"
RESULTS   = SANDBOX / "results"
LOGS      = SANDBOX / "logs"
RESULTS.mkdir(exist_ok=True)
LOGS.mkdir(exist_ok=True)

WAVE_PATH   = DATA_DIR / "waveforms_stride.parquet"
SCALAR_PATH = DATA_DIR / "features_scalar.csv"
SCALAR_BASELINE_AUC = 0.9600

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS / "02_raw_waveform_ml.log")
    ]
)
log = logging.getLogger(__name__)

SMOKE        = os.environ.get("SMOKE") == "1"   # fast end-to-end test
RANDOM_STATE = 42
N_TRIALS     = 1 if SMOKE else 25
OUTER_FOLDS  = 5
INNER_FOLDS  = 3
N_BOOTSTRAP  = 50 if SMOKE else 1000
PCA_VARIANCE = 0.95

CHANNELS = [
    'ankle_adduction', 'ankle_dorsiflexion', 'ankle_int_rotation',
    'hip_adduction',   'hip_flexion',         'hip_int_rotation',
    'knee_adduction',  'knee_flexion',         'knee_int_rotation',
]


# ── Data preparation ───────────────────────────────────────────────────────────

def get_wave_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns
            if any(c.startswith(ch + '_') for ch in CHANNELS)]


def build_bilateral_asymmetry(wave_df: pd.DataFrame, inj_map: dict,
                               wave_cols: list) -> pd.DataFrame:
    """
    Compute bilateral asymmetry waveforms (injured − contralateral) per subject×speed.
    side-fixed: rows split into inj/con via _side_utils.to_inj_con (waveform side
    is Right/Left → compared against the subject's injured leg).
    Returns DataFrame with up to 237 rows (79 subjects × 3 speeds).
    """
    rows = []
    for (subj, speed), grp in wave_df.groupby(['subject_id', 'speed']):
        injured_leg = inj_map.get(subj, 'Right')
        side_std = grp['side'].apply(lambda s: to_inj_con(s, injured_leg))
        inj_rows = grp[side_std == 'inj']
        con_rows = grp[side_std == 'con']

        if len(inj_rows) == 0 or len(con_rows) == 0:
            log.warning(f"  Missing side pair: subj={subj} speed={speed} inj_leg={injured_leg}")
            continue

        inj_wave = inj_rows[wave_cols].values[0]
        con_wave = con_rows[wave_cols].values[0]
        asym     = inj_wave - con_wave

        group  = grp['group'].iloc[0]
        label  = 0 if group == 'HA' else 1

        row = {'subject_id': subj, 'group': group, 'speed': speed, 'binary_label': label}
        row.update(zip(wave_cols, asym))
        rows.append(row)

    df = pd.DataFrame(rows)
    log.info(f"Bilateral asymmetry dataset: {df.shape} "
             f"(HA={int((df.binary_label==0).sum())} ACL={int((df.binary_label==1).sum())})")
    return df


def build_multi_speed_bilateral(asym_df: pd.DataFrame, wave_cols: list) -> pd.DataFrame:
    """
    Pivot bilateral asymmetry to subject-level by concatenating slow/normal/fast.
    Returns 79-row DataFrame with 3×909=2727 waveform features.
    """
    speed_dfs = {}
    for spd, sub in asym_df.groupby('speed'):
        sub = sub.set_index('subject_id')
        speed_dfs[spd] = sub[wave_cols].add_prefix(f"{spd}_")

    speeds = sorted(speed_dfs.keys())
    pivot  = pd.concat([speed_dfs[s] for s in speeds], axis=1)
    pivot  = pivot.dropna()

    # Attach metadata
    meta = asym_df.drop_duplicates('subject_id').set_index('subject_id')[['group', 'binary_label']]
    pivot = pivot.join(meta)
    pivot = pivot.reset_index().rename(columns={'index': 'subject_id'})

    log.info(f"Multi-speed bilateral dataset: {pivot.shape}")
    return pivot


def build_speed_delta(asym_df: pd.DataFrame, wave_cols: list) -> pd.DataFrame:
    """
    Speed delta: (fast_asym − slow_asym) per subject.
    Returns 79-row DataFrame with 909 features.
    """
    slow_df = asym_df[asym_df['speed'] == 'slow'].set_index('subject_id')[wave_cols]
    fast_df = asym_df[asym_df['speed'] == 'fast'].set_index('subject_id')[wave_cols]

    common = slow_df.index.intersection(fast_df.index)
    delta  = fast_df.loc[common] - slow_df.loc[common]
    delta.columns = [f"delta_{c}" for c in wave_cols]

    meta = asym_df.drop_duplicates('subject_id').set_index('subject_id')[['group', 'binary_label']]
    result = delta.join(meta).dropna()
    result = result.reset_index().rename(columns={'index': 'subject_id'})
    log.info(f"Speed delta dataset: {result.shape}")
    return result


# ── PCA within-fold helper ─────────────────────────────────────────────────────

def pca_transform_fold(X_tr, X_te, n_components=PCA_VARIANCE):
    """Fit scaler+PCA on X_tr only, transform both X_tr and X_te."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    max_comp = min(X_tr_s.shape[0] - 1, X_tr_s.shape[1])
    if isinstance(n_components, float):
        pca = PCA(n_components=min(n_components, max_comp), random_state=RANDOM_STATE)
    else:
        pca = PCA(n_components=min(n_components, max_comp), random_state=RANDOM_STATE)

    X_tr_pca = pca.fit_transform(X_tr_s)
    X_te_pca = pca.transform(X_te_s)
    return X_tr_pca, X_te_pca, pca.n_components_


# ── Statistical helpers ────────────────────────────────────────────────────────

def bootstrap_ci(y_true, y_scores, n=N_BOOTSTRAP, rs=RANDOM_STATE):
    rng, aucs = np.random.default_rng(rs), []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_scores[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def paired_bootstrap_pval(y_true, scores_a, scores_b, n=N_BOOTSTRAP, rs=RANDOM_STATE):
    rng, diffs = np.random.default_rng(rs), []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        diffs.append(
            roc_auc_score(y_true[idx], scores_a[idx]) -
            roc_auc_score(y_true[idx], scores_b[idx])
        )
    diffs    = np.array(diffs)
    obs      = roc_auc_score(y_true, scores_a) - roc_auc_score(y_true, scores_b)
    centered = diffs - diffs.mean()
    return float(obs), float(np.mean(np.abs(centered) >= np.abs(obs)))


def wilcoxon_test(fold_aucs_a, fold_aucs_b):
    from scipy.stats import wilcoxon
    try:
        stat, pval = wilcoxon(fold_aucs_a, fold_aucs_b)
        return float(pval)
    except Exception:
        return float('nan')


# ── Core nested CV with within-fold PCA ───────────────────────────────────────

def run_feature_set(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    feature_set: str,
    model_name: str,
    use_group_cv: bool = False,
) -> dict:
    """
    Nested CV with within-fold PCA + Optuna HPO.
    Returns OOF predictions and statistics.
    """
    if use_group_cv:
        outer_cv = StratifiedGroupKFold(n_splits=OUTER_FOLDS)
        inner_cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    else:
        outer_cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        inner_cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_scores = np.zeros(len(y))
    oof_true   = y.copy()
    fold_aucs  = []
    fold_pca_dims = []

    split_kwargs = {'X': X, 'y': y, 'groups': groups} if use_group_cv else {'X': X, 'y': y}

    for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(**split_kwargs)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # Within-fold PCA (prevent leakage)
        X_tr_pca, X_te_pca, n_comps = pca_transform_fold(X_tr, X_te)
        fold_pca_dims.append(n_comps)

        # Optuna HPO on inner CV via shared registry
        est, _, best_val = M.tune_and_build(
            model_name, X_tr_pca, y_tr, inner_cv,
            n_trials=N_TRIALS, task="binary", seed=RANDOM_STATE + fold)
        est.fit(X_tr_pca, y_tr)
        proba = est.predict_proba(X_te_pca)[:, 1]

        oof_scores[te_idx] = proba
        fold_aucs.append(roc_auc_score(y_te, proba))
        log.info(f"  [{feature_set}/{model_name}] fold={fold} "
                 f"auc={fold_aucs[-1]:.4f} pca_dims={n_comps} best_val={best_val:.4f}")

    mean_auc = roc_auc_score(oof_true, oof_scores)
    ci_lo, ci_hi = bootstrap_ci(oof_true, oof_scores)

    return {
        'feature_set':  feature_set,
        'model':        model_name,
        'auc':          mean_auc,
        'ci_lo':        ci_lo,
        'ci_hi':        ci_hi,
        'fold_aucs':    fold_aucs,
        'pca_dims':     fold_pca_dims,
        'oof_scores':   oof_scores.tolist(),
        'oof_true':     oof_true.tolist(),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("02_raw_waveform_ml.py — H3: Raw Waveform Feature Comparison")
    log.info("=" * 70)

    # Load data
    wave_df   = pd.read_parquet(WAVE_PATH)
    scalar_df = pd.read_csv(SCALAR_PATH)
    wave_cols = get_wave_cols(wave_df)
    inj_map   = build_injured_leg_map(scalar_df)

    log.info(f"Waveform data: {wave_df.shape} | wave_cols: {len(wave_cols)}")
    log.info(f"Injured leg map: {len(inj_map)} subjects")

    # Build feature datasets
    log.info("\n[Data Preparation]")
    asym_df   = build_bilateral_asymmetry(wave_df, inj_map, wave_cols)
    multi_df  = build_multi_speed_bilateral(asym_df, wave_cols)
    delta_df  = build_speed_delta(asym_df, wave_cols)

    # Unilateral injured (Feature Set A)
    uni_df    = wave_df.copy()
    uni_df['binary_label'] = (uni_df['group'] != 'HA').astype(int)
    for subj, leg in inj_map.items():
        mask = (uni_df['subject_id'] == subj) & (uni_df['side'] == leg)
    # Filter to injured side only
    uni_injured = []
    for subj, leg in inj_map.items():
        rows = wave_df[(wave_df['subject_id'] == subj) & (wave_df['side'] == leg)]
        if len(rows) > 0:
            rows = rows.copy()
            rows['binary_label'] = (rows['group'] != 'HA').astype(int)
            uni_injured.append(rows)
    uni_df = pd.concat(uni_injured, ignore_index=True) if uni_injured else None

    log.info(f"[A] Unilateral injured: {uni_df.shape if uni_df is not None else 'N/A'}")
    log.info(f"[B] Bilateral asym:      {asym_df.shape}")
    log.info(f"[C] Speed delta:         {delta_df.shape}")
    log.info(f"[D] Multi-speed bilat:   {multi_df.shape}")

    # ── Feature sets configuration ────────────────────────────────────────────
    feat_sets = {
        'A_unilateral': {
            'df': uni_df,
            'feat_prefix': None,  # use wave_cols directly
            'use_group_cv': True,
            'group_col': 'subject_id',
            'wave_cols_flag': True,
        },
        'B_bilateral_asym': {
            'df': asym_df,
            'feat_prefix': None,
            'use_group_cv': True,
            'group_col': 'subject_id',
            'wave_cols_flag': True,
        },
        'C_speed_delta': {
            'df': delta_df,
            'feat_prefix': 'delta_',
            'use_group_cv': False,
            'group_col': None,
            'wave_cols_flag': False,
        },
        'D_multi_speed_bilateral': {
            'df': multi_df,
            'feat_prefix': None,  # all numeric except meta
            'use_group_cv': False,
            'group_col': None,
            'wave_cols_flag': False,
        },
    }

    models    = (['logreg', 'rf'] if SMOKE else M.available_models())
    skipped   = [m for m in M.ALL_MODELS if m not in models]
    if skipped:
        log.warning(f"Skipped (unavailable): {skipped}")
    all_results = []
    best_by_featset = {}

    for fs_name, cfg in feat_sets.items():
        df = cfg['df']
        if df is None or len(df) == 0:
            log.warning(f"Skipping {fs_name}: empty dataset")
            continue

        # Build feature matrix
        META_COLS = {'subject_id', 'group', 'speed', 'side', 'binary_label',
                     'n_strides', 'injured_leg'}
        if cfg['wave_cols_flag']:
            feat_cols = wave_cols
        elif cfg['feat_prefix']:
            feat_cols = [c for c in df.columns if c.startswith(cfg['feat_prefix'])]
        else:
            feat_cols = [c for c in df.columns
                         if c not in META_COLS and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

        X = df[feat_cols].fillna(0).values.astype(np.float32)
        y = df['binary_label'].values.astype(int)
        groups = df[cfg['group_col']].values if cfg['group_col'] else None

        log.info(f"\n{'─'*60}")
        log.info(f"Feature Set [{fs_name}]")
        log.info(f"  n={len(y)} | features={X.shape[1]} | "
                 f"HA={int((y==0).sum())} ACL={int((y==1).sum())}")
        if groups is not None:
            log.info(f"  n_subjects={len(np.unique(groups))} | GroupKFold=True")

        fs_best_auc = -1
        for mdl in models:
            log.info(f"\n  Model: {mdl}")
            res = run_feature_set(X, y, groups, fs_name, mdl, cfg['use_group_cv'])
            all_results.append(res)
            if res['auc'] > fs_best_auc:
                fs_best_auc = res['auc']
                best_by_featset[fs_name] = res

        log.info(f"\n  Best AUC for {fs_name}: {fs_best_auc:.4f}")

    # ── Statistical comparisons vs scalar baseline ─────────────────────────────
    log.info(f"\n{'─'*60}")
    log.info("Statistical Comparisons vs Scalar Baseline")

    comparison_rows = []
    ref_fs = best_by_featset.get('D_multi_speed_bilateral') or \
             best_by_featset.get('B_bilateral_asym')

    if ref_fs is not None:
        for fs_name, res in best_by_featset.items():
            if fs_name == ref_fs['feature_set']:
                continue
            y_t  = np.array(ref_fs['oof_true'])
            s_a  = np.array(ref_fs['oof_scores'])
            s_b  = np.array(res['oof_scores'])
            # Paired bootstrap requires equal sample counts; feature sets differ
            # in N (A/B = subject×speed 237, C/D = subject 79) → skip cross-N pairs.
            if not (len(s_a) == len(s_b) == len(y_t)):
                log.info(f"  skip paired comparison {ref_fs['feature_set']} vs {fs_name} "
                         f"(N {len(s_a)} vs {len(s_b)})")
                continue
            diff, pval = paired_bootstrap_pval(y_t, s_a, s_b)
            comparison_rows.append({
                'comparison': f"{ref_fs['feature_set']} vs {fs_name}",
                'auc_ref':    round(ref_fs['auc'], 4),
                'auc_cmp':    round(res['auc'], 4),
                'auc_diff':   round(diff, 4),
                'pval':       round(pval, 4),
                'significant': pval < 0.05,
            })
            log.info(f"  {ref_fs['feature_set']}({ref_fs['auc']:.4f}) vs "
                     f"{fs_name}({res['auc']:.4f}): diff={diff:.4f} p={pval:.4f}")

    # ── Save results ───────────────────────────────────────────────────────────
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            'feature_set':     r['feature_set'],
            'model':           r['model'],
            'auc':             round(r['auc'], 4),
            'ci_lo':           round(r['ci_lo'], 4),
            'ci_hi':           round(r['ci_hi'], 4),
            'fold_auc_mean':   round(np.mean(r['fold_aucs']), 4),
            'fold_auc_std':    round(np.std(r['fold_aucs']), 4),
            'pca_dims_mean':   round(np.mean(r['pca_dims']), 1) if r['pca_dims'] else 'N/A',
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(['feature_set', 'auc'], ascending=[True, False])
    summary_df.to_csv(RESULTS / "02_waveform_results.csv", index=False)
    log.info(f"\nSaved: results/02_waveform_results.csv")

    if comparison_rows:
        pd.DataFrame(comparison_rows).to_csv(RESULTS / "02_waveform_comparison.csv", index=False)

    # Best summary
    best_summary = {}
    for fs_name, res in best_by_featset.items():
        best_summary[fs_name] = {
            'model':     res['model'],
            'auc':       round(res['auc'], 4),
            'ci_lo':     round(res['ci_lo'], 4),
            'ci_hi':     round(res['ci_hi'], 4),
            'fold_aucs': [round(float(a), 4) for a in res['fold_aucs']],
            'pca_dims':  [int(x) for x in res['pca_dims']],
        }

    # Save OOF scores for Script 03 (SHAP)
    for fs_name, res in best_by_featset.items():
        oof_path = RESULTS / f"02_oof_{fs_name}_{res['model']}.json"
        with open(oof_path, 'w') as f:
            json.dump({'oof_scores': res['oof_scores'], 'oof_true': res['oof_true'],
                       'feature_set': fs_name, 'model': res['model']}, f, default=_np_default)

    with open(RESULTS / "02_waveform_best.json", 'w') as f:
        json.dump({
            'strategy': 'raw_waveform_ml',
            'best_by_feature_set': best_summary,
            'scalar_baseline_auc': SCALAR_BASELINE_AUC,
            'comparisons': comparison_rows,
            'n_trials': N_TRIALS,
            'pca_variance_threshold': PCA_VARIANCE,
            'elapsed_sec': round(time.time() - t0, 1),
        }, f, indent=2, default=_np_default)

    # ── Final summary ──────────────────────────────────────────────────────────
    log.info(f"\n{'='*70}")
    log.info("WAVEFORM ML RESULTS SUMMARY")
    log.info(f"{'='*70}")
    log.info(f"{'Feature Set':<28} {'Model':<6} {'AUC':>6}  95% CI")
    log.info(f"{'─'*60}")
    for fs_name in ['A_unilateral', 'B_bilateral_asym', 'C_speed_delta', 'D_multi_speed_bilateral']:
        if fs_name in best_summary:
            r = best_summary[fs_name]
            log.info(f"  {fs_name:<26} {r['model']:<6} {r['auc']:.4f}  [{r['ci_lo']:.4f},{r['ci_hi']:.4f}]")
    log.info(f"\n  Scalar baseline (0529_ML RF):  {SCALAR_BASELINE_AUC:.4f}")
    log.info(f"\nTotal elapsed: {time.time()-t0:.1f}s")

    return summary_df


if __name__ == '__main__':
    main()
