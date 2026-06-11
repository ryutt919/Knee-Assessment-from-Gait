#!/usr/bin/env python3
"""
04_multiclass.py — 3-Class Classification: ACLD / ACLR / HA

Uses best feature set from script 02 (multi_speed_bilateral).
Clinical relevance: distinguishing reconstruction stage from deficient stage
enables objective monitoring of rehabilitation progress.

Metrics: macro-F1, per-class AUC (OvR), 3×3 confusion matrix
"""

import warnings
warnings.filterwarnings("ignore")

import json, time, logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report,
    confusion_matrix, balanced_accuracy_score
)
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import xgboost as xgb
import lightgbm as lgb

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[2]
SANDBOX  = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RESULTS  = SANDBOX / "results"
FIGURES  = SANDBOX / "figures"
LOGS     = SANDBOX / "logs"
FIGURES.mkdir(exist_ok=True)

WAVE_PATH   = DATA_DIR / "waveforms_stride.parquet"
SCALAR_PATH = DATA_DIR / "features_scalar.csv"

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S",
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(LOGS / "04_multiclass.log")])
log = logging.getLogger(__name__)

RANDOM_STATE = 42
N_TRIALS     = 50
OUTER_FOLDS  = 5
INNER_FOLDS  = 3
PCA_VARIANCE = 0.95
CHANNELS = [
    'ankle_adduction', 'ankle_dorsiflexion', 'ankle_int_rotation',
    'hip_adduction',   'hip_flexion',         'hip_int_rotation',
    'knee_adduction',  'knee_flexion',         'knee_int_rotation',
]
CLASS_NAMES = ['HA', 'ACLR', 'ACLD']


# ── Data helpers (same as script 02) ──────────────────────────────────────────
def get_wave_cols(df):
    return [c for c in df.columns if any(c.startswith(ch + '_') for ch in CHANNELS)]


def get_injured_leg_map(scalar_df):
    mapping = {}
    for _, row in scalar_df.drop_duplicates('subject_id').iterrows():
        leg = row['injured_leg']
        mapping[row['subject_id']] = 'Right' if (pd.isna(leg) or str(leg).lower() == 'nan') else leg
    return mapping


def build_bilateral_asymmetry(wave_df, inj_map, wave_cols):
    rows = []
    for (subj, speed), grp in wave_df.groupby(['subject_id', 'speed']):
        leg = inj_map.get(subj, 'Right')
        inj = grp[grp['side'] == leg]
        con = grp[grp['side'] != leg]
        if len(inj) == 0 or len(con) == 0:
            continue
        asym = inj[wave_cols].values[0] - con[wave_cols].values[0]
        row  = {'subject_id': subj, 'group': grp['group'].iloc[0], 'speed': speed}
        row.update(zip(wave_cols, asym))
        rows.append(row)
    return pd.DataFrame(rows)


def build_multi_speed_bilateral(asym_df, wave_cols):
    speed_dfs = {}
    for spd, sub in asym_df.groupby('speed'):
        speed_dfs[spd] = sub.set_index('subject_id')[wave_cols].add_prefix(f"{spd}_")
    pivot = pd.concat([speed_dfs[s] for s in sorted(speed_dfs)], axis=1).dropna()
    meta  = asym_df.drop_duplicates('subject_id').set_index('subject_id')[['group']]
    return pivot.join(meta).reset_index().rename(columns={'index': 'subject_id'})


def pca_fold(X_tr, X_te):
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)
    max_c = min(X_tr_s.shape[0] - 1, X_tr_s.shape[1])
    pca = PCA(n_components=min(PCA_VARIANCE, max_c), random_state=RANDOM_STATE)
    return pca.fit_transform(X_tr_s), pca.transform(X_te_s), pca.n_components_


# ── Optuna for multiclass ──────────────────────────────────────────────────────

def rf_obj_multi(X_tr, y_tr, inner_cv):
    def obj(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600),
            max_depth=trial.suggest_int("max_depth", 2, 10),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 15),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
            max_features=trial.suggest_float("max_features", 0.1, 1.0),
            class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1,
        )
        clf = RandomForestClassifier(**params)
        return cross_val_score(clf, X_tr, y_tr, cv=inner_cv,
                               scoring='f1_macro', n_jobs=-1).mean()
    return obj


def xgb_obj_multi(X_tr, y_tr, inner_cv):
    def obj(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
            random_state=RANDOM_STATE, verbosity=0, eval_metric='mlogloss',
            objective='multi:softprob', num_class=3,
        )
        clf = xgb.XGBClassifier(**params)
        return cross_val_score(clf, X_tr, y_tr, cv=inner_cv,
                               scoring='f1_macro', n_jobs=1).mean()
    return obj


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log.info("=" * 60)
    log.info("04_multiclass.py — 3-Class ACLD/ACLR/HA Classification")
    log.info("=" * 60)

    wave_df   = pd.read_parquet(WAVE_PATH)
    scalar_df = pd.read_csv(SCALAR_PATH)
    wave_cols = get_wave_cols(wave_df)
    inj_map   = get_injured_leg_map(scalar_df)

    asym_df  = build_bilateral_asymmetry(wave_df, inj_map, wave_cols)
    multi_df = build_multi_speed_bilateral(asym_df, wave_cols)

    META = {'subject_id', 'group', 'speed', 'binary_label', 'side', 'n_strides', 'injured_leg'}
    feat_cols = [c for c in multi_df.columns
                 if c not in META and multi_df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

    X = multi_df[feat_cols].fillna(0).values.astype(np.float32)
    le = LabelEncoder()
    y  = le.fit_transform(multi_df['group'])   # HA→0, ACLD→1, ACLR→2

    class_order = list(le.classes_)
    log.info(f"Classes: {class_order}")
    log.info(f"Distribution: {dict(zip(class_order, [(y==i).sum() for i in range(len(class_order))])  )}")

    outer_cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    models = {
        'rf':  (rf_obj_multi,  {'class_weight': 'balanced', 'random_state': RANDOM_STATE, 'n_jobs': -1}),
        'xgb': (xgb_obj_multi, {'random_state': RANDOM_STATE, 'verbosity': 0,
                                 'eval_metric': 'mlogloss', 'objective': 'multi:softprob',
                                 'num_class': 3}),
    }

    all_results = []
    for model_name, (obj_fn, fixed_params) in models.items():
        log.info(f"\nModel: {model_name}")
        oof_proba = np.zeros((len(y), len(class_order)))
        oof_true  = y.copy()

        for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y)):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            X_tr_pca, X_te_pca, n_comps = pca_fold(X_tr, X_te)

            study = optuna.create_study(
                direction='maximize',
                study_name=f"mc_{model_name}_f{fold}_v1",
                storage=f"sqlite:///{SANDBOX}/optuna/mc_{model_name}_f{fold}_v1.db",
                load_if_exists=True,
                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE + fold),
            )
            existing = len(study.trials)
            study.optimize(obj_fn(X_tr_pca, y_tr, inner_cv),
                           n_trials=max(0, N_TRIALS - existing),
                           show_progress_bar=False)

            bp = {k: v for k, v in study.best_params.items()
                  if k not in fixed_params}

            if model_name == 'rf':
                clf = RandomForestClassifier(**bp, **fixed_params)
            else:
                clf = xgb.XGBClassifier(**bp, **fixed_params)

            clf.fit(X_tr_pca, y_tr)
            oof_proba[te_idx] = clf.predict_proba(X_te_pca)

            fold_pred = oof_proba[te_idx].argmax(axis=1)
            fold_f1   = f1_score(y_te, fold_pred, average='macro')
            log.info(f"  fold={fold} f1_macro={fold_f1:.4f} pca_dims={n_comps}")

        # Metrics
        y_pred  = oof_proba.argmax(axis=1)
        macro_f1 = f1_score(oof_true, y_pred, average='macro')
        bal_acc  = balanced_accuracy_score(oof_true, y_pred)

        # Per-class AUC (OvR)
        per_class_auc = {}
        for i, cls in enumerate(class_order):
            y_bin = (oof_true == i).astype(int)
            if y_bin.sum() > 0:
                per_class_auc[cls] = round(roc_auc_score(y_bin, oof_proba[:, i]), 4)

        # Overall AUC (OvR macro)
        auc_ovr = roc_auc_score(oof_true, oof_proba, multi_class='ovr', average='macro')

        log.info(f"\n  {model_name} | macro_F1={macro_f1:.4f} | bal_acc={bal_acc:.4f} | "
                 f"AUC_ovr={auc_ovr:.4f}")
        log.info(f"  Per-class AUC: {per_class_auc}")
        log.info(f"\n{classification_report(oof_true, y_pred, target_names=class_order)}")

        all_results.append({
            'model': model_name,
            'macro_f1': round(macro_f1, 4),
            'balanced_accuracy': round(bal_acc, 4),
            'auc_ovr': round(auc_ovr, 4),
            'per_class_auc': per_class_auc,
            'oof_proba': oof_proba.tolist(),
            'oof_true': oof_true.tolist(),
            'class_order': class_order,
        })

        # Confusion matrix plot
        cm = confusion_matrix(oof_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_order, yticklabels=class_order, ax=ax)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True', fontsize=11)
        ax.set_title(f'3-Class Confusion Matrix — {model_name.upper()}\n'
                     f'macro-F1={macro_f1:.3f}, AUC(OvR)={auc_ovr:.3f}',
                     fontsize=11, fontweight='bold')
        plt.tight_layout()
        fig.savefig(FIGURES / f"fig_mc_cm_{model_name}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        log.info(f"Saved: fig_mc_cm_{model_name}.png")

    # Save results
    save_rows = [{k: v for k, v in r.items() if k not in {'oof_proba', 'oof_true'}}
                 for r in all_results]
    pd.DataFrame(save_rows).to_csv(RESULTS / "04_multiclass_results.csv", index=False)

    with open(RESULTS / "04_multiclass_best.json", 'w') as f:
        best = max(all_results, key=lambda r: r['macro_f1'])
        json.dump({k: v for k, v in best.items() if k not in {'oof_proba', 'oof_true'}},
                  f, indent=2)

    log.info(f"\n{'='*60}")
    log.info("MULTICLASS SUMMARY")
    for r in all_results:
        log.info(f"  {r['model']}: macro-F1={r['macro_f1']:.4f} "
                 f"AUC(OvR)={r['auc_ovr']:.4f} "
                 f"bal_acc={r['balanced_accuracy']:.4f}")
    log.info(f"  ACLR vs HA: {all_results[-1]['per_class_auc']}")
    log.info(f"\nElapsed: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
