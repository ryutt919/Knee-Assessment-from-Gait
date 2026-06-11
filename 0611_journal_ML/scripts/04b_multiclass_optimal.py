"""
04b_multiclass_optimal.py — 3-Class with Optimal Pipeline
Same feature engineering as 02b_optimal_pipeline.py:
  scalar pivot (864) + stride variability (270) + within-fold RF top-20 interactions (190)
Target: ACLD / ACLR / HA
"""
import warnings; warnings.filterwarnings("ignore")
import json, time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, balanced_accuracy_score,
    roc_auc_score, classification_report, confusion_matrix,
)

ROOT    = Path(__file__).resolve().parents[2]
SANDBOX = Path(__file__).resolve().parents[1]
DATA    = ROOT / "data" / "processed"
RESULTS = SANDBOX / "results"
FIGURES = SANDBOX / "figures"
FIGURES.mkdir(exist_ok=True)

SCALAR_PATH = DATA / "features_scalar.csv"
STRIDE_PATH = DATA / "stride_level_peaks.parquet"

OUTER_FOLDS = 5
TOP_K       = 20
ENS_SEEDS   = [42, 88]


# ── Feature builders (identical to 02b) ──────────────────────────────────────
def build_scalar_pivot(path):
    df = pd.read_csv(path)
    df = df[df["subject_id"].notna()].copy()
    df["subject_id"] = df["subject_id"].str.strip()
    META = {"subject_id", "group", "speed", "binary_label", "side",
            "n_strides", "injured_leg", "Unnamed: 0"}
    num_cols = [c for c in df.columns
                if c not in META and pd.api.types.is_numeric_dtype(df[c])]
    speeds = df["speed"].dropna().unique().tolist()
    speed_dfs = {}
    for spd in speeds:
        sub = df[df["speed"] == spd].copy()
        sub = sub.drop_duplicates("subject_id").set_index("subject_id")
        speed_dfs[spd] = sub[num_cols].add_prefix(f"{spd}_")
    pivot = pd.concat(speed_dfs.values(), axis=1)
    for s1, s2 in combinations(speeds, 2):
        common = [c.replace(f"{s1}_", "") for c in speed_dfs[s1].columns]
        for c in common:
            if f"{s1}_{c}" in pivot.columns and f"{s2}_{c}" in pivot.columns:
                pivot[f"delta_{s1}_{s2}_{c}"] = pivot[f"{s1}_{c}"] - pivot[f"{s2}_{c}"]
    for c in num_cols:
        cols = [f"{s}_{c}" for s in speeds if f"{s}_{c}" in pivot.columns]
        if cols:
            pivot[f"mean_{c}"] = pivot[cols].mean(axis=1)
    meta_src = df.drop_duplicates("subject_id").set_index("subject_id")
    pivot = pivot.join(meta_src[["group"]].rename(columns={"group": "group"}), how="inner")
    return pivot.reset_index().rename(columns={"index": "subject_id"})


def build_stride_variability(path):
    df = pd.read_parquet(path)
    df = df[df["subject_id"].notna()].copy()
    META = {"subject_id", "group", "speed", "binary_label", "side",
            "stride_id", "n_strides", "injured_leg", "trial_id"}
    num_cols = [c for c in df.columns
                if c not in META and pd.api.types.is_numeric_dtype(df[c])]
    if "is_injured" not in df.columns and "side" in df.columns:
        df["is_injured"] = df["side"].apply(
            lambda x: True if str(x).lower() in ("injured", "right") else False)
    grp_cols = ["subject_id", "speed", "is_injured"] if "is_injured" in df.columns \
               else ["subject_id", "speed"]
    agg = df.groupby(grp_cols)[num_cols].agg(["std", lambda x: x.std()/x.mean()
                                               if x.mean() != 0 else 0])
    agg.columns = [f"{c}_{stat}" if stat != "<lambda_0>" else f"{c}_cv"
                   for c, stat in agg.columns]
    agg = agg.reset_index()
    subj_agg = agg.groupby("subject_id").mean(numeric_only=True)
    group_map = df.drop_duplicates("subject_id").set_index("subject_id")["group"]
    subj_agg = subj_agg.join(group_map).reset_index()
    return subj_agg


def run_cv_3class(X, y, class_order, seed, topK=TOP_K, ne_sel=200, ne_final=1000):
    skf   = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=False)
    proba = np.zeros((len(y), len(class_order)))

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        sc  = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])

        # Within-fold feature selection
        sel = RandomForestClassifier(n_estimators=ne_sel, class_weight="balanced",
                                     random_state=seed, n_jobs=-1)
        sel.fit(Xtr, y[tr])
        top = np.argsort(sel.feature_importances_)[-topK:]
        Xtr_top, Xte_top = Xtr[:, top], Xte[:, top]

        # Pairwise interactions
        inter_tr = [Xtr_top[:, i] * Xtr_top[:, j] for i, j in combinations(range(topK), 2)]
        inter_te = [Xte_top[:, i] * Xte_top[:, j] for i, j in combinations(range(topK), 2)]
        Xtr_a = np.hstack([Xtr, np.column_stack(inter_tr)])
        Xte_a = np.hstack([Xte, np.column_stack(inter_te)])

        fm = RandomForestClassifier(n_estimators=ne_final, class_weight="balanced",
                                    random_state=seed, n_jobs=-1)
        fm.fit(Xtr_a, y[tr])
        proba[te] += fm.predict_proba(Xte_a)

        f1 = f1_score(y[te], fm.predict_proba(Xte_a).argmax(axis=1), average="macro")
        print(f"  [seed={seed}] fold={fold} f1_macro={f1:.4f}", flush=True)

    return proba  # raw (not averaged) — caller divides by n_seeds if ensemble


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("04b_multiclass_optimal.py — 3-Class ACLD/ACLR/HA (Optimal Pipeline)", flush=True)
    print("=" * 60, flush=True)

    # ── Build features ────────────────────────────────────────────────────────
    print("Building scalar pivot...", flush=True)
    scalar_df = build_scalar_pivot(SCALAR_PATH)

    print("Building stride variability...", flush=True)
    try:
        var_df = build_stride_variability(STRIDE_PATH)
        has_var = True
    except Exception as e:
        print(f"  Stride variability skipped: {e}", flush=True)
        has_var = False

    # Merge on subject_id
    feat_df = scalar_df.copy()
    if has_var:
        META_VAR = {"subject_id", "group"}
        var_cols = [c for c in var_df.columns if c not in META_VAR]
        feat_df = feat_df.merge(var_df[["subject_id"] + var_cols],
                                on="subject_id", how="left")

    # Keep only 3-class subjects
    feat_df = feat_df[feat_df["group"].isin(["ACLD", "ACLR", "HA"])].copy()

    META = {"subject_id", "group", "binary_label", "speed", "side",
            "n_strides", "injured_leg"}
    feat_cols = [c for c in feat_df.columns
                 if c not in META and pd.api.types.is_numeric_dtype(feat_df[c])]
    feat_df[feat_cols] = feat_df[feat_cols].fillna(0)

    # Encode labels
    label_map  = {"HA": 0, "ACLR": 1, "ACLD": 2}
    class_order = ["HA", "ACLR", "ACLD"]
    X = feat_df[feat_cols].values.astype(np.float32)
    y = feat_df["group"].map(label_map).values

    n_total = len(y)
    dist = {c: int((y == label_map[c]).sum()) for c in class_order}
    print(f"N={n_total}  Distribution: {dist}", flush=True)
    print(f"Feature dim: {X.shape[1]}", flush=True)

    # ── Ensemble over seeds ───────────────────────────────────────────────────
    all_proba = np.zeros((n_total, len(class_order)))
    for seed in ENS_SEEDS:
        print(f"\n--- seed={seed} ---", flush=True)
        proba_seed = run_cv_3class(X, y, class_order, seed)
        all_proba += proba_seed
    all_proba /= len(ENS_SEEDS)

    # ── Metrics ───────────────────────────────────────────────────────────────
    y_pred   = all_proba.argmax(axis=1)
    macro_f1 = f1_score(y, y_pred, average="macro")
    bal_acc  = balanced_accuracy_score(y, y_pred)
    auc_ovr  = roc_auc_score(y, all_proba, multi_class="ovr", average="macro")
    per_cls  = {c: round(roc_auc_score((y == label_map[c]).astype(int),
                                        all_proba[:, label_map[c]]), 4)
                for c in class_order}

    print(f"\n{'='*60}", flush=True)
    print(f"RESULT  macro_F1={macro_f1:.4f}  bal_acc={bal_acc:.4f}  AUC(OvR)={auc_ovr:.4f}", flush=True)
    print(f"Per-class AUC: {per_cls}", flush=True)
    print(f"\n{classification_report(y, y_pred, target_names=class_order)}", flush=True)
    print(f"Elapsed: {time.time()-t0:.1f}s", flush=True)

    # ── Save results ──────────────────────────────────────────────────────────
    result = {
        "model": "RF_interactions_ensemble",
        "pipeline": "scalar_pivot_864 + stride_variability + within_fold_interactions",
        "n_subjects": n_total,
        "distribution": dist,
        "class_order": class_order,
        "ens_seeds": ENS_SEEDS,
        "macro_f1":  round(macro_f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "auc_ovr": round(auc_ovr, 4),
        "per_class_auc": per_cls,
    }
    with open(RESULTS / "04b_multiclass_results.json", "w") as f:
        json.dump(result, f, indent=2)

    # Also update the CSV for report compatibility
    row = {
        "model": "rf_optimal",
        "macro_f1": round(macro_f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "auc_ovr": round(auc_ovr, 4),
        "per_class_auc": str(per_cls),
        "class_order": str(class_order),
    }
    old_df = pd.read_csv(RESULTS / "04_multiclass_results.csv")
    # Append or replace rf_optimal row
    old_df = old_df[old_df["model"] != "rf_optimal"]
    new_df = pd.concat([old_df, pd.DataFrame([row])], ignore_index=True)
    new_df.to_csv(RESULTS / "04_multiclass_results.csv", index=False)
    print(f"Saved: 04b_multiclass_results.json", flush=True)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_order, yticklabels=class_order, ax=ax)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(
        f"3-Class Confusion Matrix — RF Optimal Pipeline\n"
        f"macro-F1={macro_f1:.3f}  AUC(OvR)={auc_ovr:.3f}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(FIGURES / "fig_mc_cm_rf_optimal.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved: fig_mc_cm_rf_optimal.png", flush=True)


if __name__ == "__main__":
    main()
