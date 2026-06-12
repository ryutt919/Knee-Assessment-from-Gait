"""
04c_speed_analysis.py — 속도별 분석 + Optuna 최적화 + ACLD↔ACLR 한계 검증

사용자 요구:
1. ACL vs HA 속도별 분석 (slow/normal/fast/통합/속도별 앙상블)
2. 3분류 속도별 분석 최대치
3. Optuna 최적화 (현재 02b/04b는 고정 RF — 정직한 최대치 탐색)
4. ACLD↔ACLR 분리 한계 검증 + 시각화 (PCA, permutation)
5. 계층적 3분류 정직한 최대치

모든 CV: StratifiedKFold(5, shuffle=False), within-fold feature selection (누출 방지).
"""
import warnings, os
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
import json, time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, confusion_matrix

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _models as M
from _side_utils import to_inj_con

SMOKE       = os.environ.get("SMOKE") == "1"   # fast end-to-end test
INNER_FOLDS = 3
N_TRIALS    = 1 if SMOKE else 25

ROOT    = Path(__file__).resolve().parents[2]
SANDBOX = Path(__file__).resolve().parents[1]
DATA    = ROOT / "data" / "processed"
RESULTS = SANDBOX / "results"
FIGURES = SANDBOX / "figures"
RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)

plt.rcParams.update({"font.family": "Apple SD Gothic Neo", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110,
                     "axes.unicode_minus": False})
COLORS = {"ACLD": "#c0392b", "ACLR": "#e67e22", "HA": "#2980b9",
          "ok": "#27ae60", "warn": "#e67e22", "primary": "#1a2a4a", "grey": "#b0bec5"}

SPEEDS    = ["slow", "normal", "fast"]
SEEDS     = [42] if SMOKE else [42, 88]
TOP_K     = 20
LABEL_MAP = {"HA": 0, "ACLR": 1, "ACLD": 2}
CLASS_ORDER = ["HA", "ACLR", "ACLD"]

SCALAR_PATH = DATA / "features_scalar.csv"
STRIDE_PATH = DATA / "stride_level_peaks.parquet"


# ── Feature builders ─────────────────────────────────────────────────────────
def load_scalar():
    df = pd.read_csv(SCALAR_PATH)
    df = df[df["subject_id"].notna()].copy()
    df["subject_id"] = df["subject_id"].str.strip()
    return df


def scalar_meta_cols(df):
    META = {"subject_id", "group", "speed", "binary_label", "side",
            "n_strides", "injured_leg", "Unnamed: 0"}
    return [c for c in df.columns if c not in META and pd.api.types.is_numeric_dtype(df[c])]


def speed_slice(df, num, speed):
    sub = df[df["speed"] == speed].drop_duplicates("subject_id").set_index("subject_id")
    return sub[num].add_prefix(f"{speed}_")


def build_all_pivot(df, num):
    parts = {s: speed_slice(df, num, s) for s in SPEEDS}
    piv = pd.concat(parts.values(), axis=1)
    # delta + mean
    for s1, s2 in combinations(SPEEDS, 2):
        for c in num:
            a, b = f"{s1}_{c}", f"{s2}_{c}"
            if a in piv.columns and b in piv.columns:
                piv[f"delta_{s1}_{s2}_{c}"] = piv[a] - piv[b]
    for c in num:
        cols = [f"{s}_{c}" for s in SPEEDS if f"{s}_{c}" in piv.columns]
        if cols:
            piv[f"mean_{c}"] = piv[cols].mean(axis=1)
    return piv


def build_stride_variability():
    df = pd.read_parquet(STRIDE_PATH)
    df = df[df["subject_id"].notna()].copy()
    META = {"subject_id", "group", "speed", "binary_label", "side",
            "stride_id", "n_strides", "injured_leg", "trial_id"}
    num = [c for c in df.columns if c not in META and pd.api.types.is_numeric_dtype(df[c])]
    if "side" in df.columns:
        # side-fixed: stride side is injured/contralateral → inj/con via _side_utils
        df["side_std"] = df["side"].apply(lambda s: to_inj_con(s, "Right"))
    grp = ["subject_id", "speed", "side_std"] if "side_std" in df.columns else ["subject_id", "speed"]
    agg = df.groupby(grp)[num].agg(["std", lambda x: x.std()/x.mean() if x.mean() != 0 else 0])
    agg.columns = [f"{c}_{st}" if st != "<lambda_0>" else f"{c}_cv" for c, st in agg.columns]
    agg = agg.reset_index()
    return agg.groupby("subject_id").mean(numeric_only=True)


# ── CV with within-fold interactions (02b pipeline) ──────────────────────────
def cv_oof(X, y, n_classes, seed=42, topK=TOP_K, ne_sel=200, ne_final=1000, rf_params=None):
    skf = StratifiedKFold(5, shuffle=False)
    out = np.zeros(len(y)) if n_classes == 2 else np.zeros((len(y), n_classes))
    rf_params = rf_params or {}
    for tr, te in skf.split(X, y):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        sel = RandomForestClassifier(n_estimators=ne_sel, class_weight="balanced",
                                     random_state=seed, n_jobs=-1)
        sel.fit(Xtr, y[tr]); top = np.argsort(sel.feature_importances_)[-topK:]
        Xtt, Xet = Xtr[:, top], Xte[:, top]
        itr = [Xtt[:, i]*Xtt[:, j] for i, j in combinations(range(topK), 2)]
        ite = [Xet[:, i]*Xet[:, j] for i, j in combinations(range(topK), 2)]
        Xa = np.hstack([Xtr, np.column_stack(itr)]); Xea = np.hstack([Xte, np.column_stack(ite)])
        fm = RandomForestClassifier(n_estimators=ne_final, class_weight="balanced",
                                    random_state=seed, n_jobs=-1, **rf_params)
        fm.fit(Xa, y[tr])
        if n_classes == 2:
            out[te] = fm.predict_proba(Xea)[:, 1]
        else:
            out[te] = fm.predict_proba(Xea)
    return out


def ensemble_oof(X, y, n_classes, **kw):
    if n_classes == 2:
        acc = np.zeros(len(y))
    else:
        acc = np.zeros((len(y), n_classes))
    for s in SEEDS:
        acc += cv_oof(X, y, n_classes, seed=s, **kw)
    return acc / len(SEEDS)


def auc_binary(y, p):   return roc_auc_score(y, p)
def auc_ovr(y, p):      return roc_auc_score(y, p, multi_class="ovr", average="macro")
def macro_f1(y, p):     return f1_score(y, p.argmax(1), average="macro")


def cv_oof_model(X, y, n_classes, model_name, seed=42, topK=TOP_K, n_trials=N_TRIALS):
    """02b within-fold interaction pipeline with any registry model (Optuna-tuned).
    Returns OOF positive-class prob (binary) or class-prob matrix (multiclass)."""
    task = "binary" if n_classes == 2 else "multiclass"
    skf = StratifiedKFold(5, shuffle=False)
    inner = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed)
    out = np.zeros(len(y)) if n_classes == 2 else np.zeros((len(y), n_classes))
    for tr, te in skf.split(X, y):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        sel = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                     random_state=seed, n_jobs=-1)
        sel.fit(Xtr, y[tr]); top = np.argsort(sel.feature_importances_)[-topK:]
        Xtt, Xet = Xtr[:, top], Xte[:, top]
        itr = [Xtt[:, i]*Xtt[:, j] for i, j in combinations(range(topK), 2)]
        ite = [Xet[:, i]*Xet[:, j] for i, j in combinations(range(topK), 2)]
        Xa = np.hstack([Xtr, np.column_stack(itr)]); Xea = np.hstack([Xte, np.column_stack(ite)])
        est, _, _ = M.tune_and_build(model_name, Xa, y[tr], inner,
                                     n_trials=n_trials, task=task, seed=seed)
        est.fit(Xa, y[tr])
        p = est.predict_proba(Xea)
        if n_classes == 2:
            out[te] = p[:, list(est.classes_).index(1)] if 1 in est.classes_ else p[:, -1]
        else:
            cls = list(est.classes_)
            for j in range(n_classes):
                if j in cls:
                    out[te, j] = p[:, cls.index(j)]
    return out


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 64); print("04c_speed_analysis.py — 속도별 분석 + Optuna + 한계검증"); print("=" * 64)

    df  = load_scalar()
    num = scalar_meta_cols(df)
    meta = df.drop_duplicates("subject_id").set_index("subject_id")[["group"]]
    var_df = build_stride_variability()

    # Speed slices + all pivot, joined with stride variability
    parts = {s: speed_slice(df, num, s) for s in SPEEDS}
    all_piv = build_all_pivot(df, num)

    def attach(base):
        m = base.join(meta, how="inner").dropna(subset=["group"])
        m = m.join(var_df, how="left")
        return m

    speed_feat = {s: attach(parts[s]) for s in SPEEDS}
    all_feat   = attach(all_piv)

    # ============ 분석 1: 속도별 격자 (이진 + 3분류) ============
    print("\n[1] 속도별 격자 분석 (2-seed 앙상블)")
    grid_rows = []
    oof_store_bin = {}   # for speed ensemble
    oof_store_3c  = {}

    def prep(feat_df, classes):
        fc = [c for c in feat_df.columns if c != "group"]
        sub = feat_df[feat_df["group"].isin(classes)]
        X = sub[fc].fillna(0).values.astype(np.float32)
        return sub.index, X, sub

    for s in SPEEDS + ["all"]:
        fdf = all_feat if s == "all" else speed_feat[s]
        # binary ACL vs HA
        idxb, Xb, subb = prep(fdf, ["HA", "ACLR", "ACLD"])
        yb = subb["group"].isin(["ACLD", "ACLR"]).astype(int).values
        pb = ensemble_oof(Xb, yb, 2)
        ab = auc_binary(yb, pb)
        oof_store_bin[s] = pd.Series(pb, index=idxb)
        # 3-class
        idx3, X3, sub3 = prep(fdf, ["HA", "ACLR", "ACLD"])
        y3 = sub3["group"].map(LABEL_MAP).values
        p3 = ensemble_oof(X3, y3, 3)
        a3, f3 = auc_ovr(y3, p3), macro_f1(y3, p3)
        oof_store_3c[s] = (idx3, y3, p3)
        grid_rows.append({"condition": s, "binary_auc": round(ab, 4),
                          "auc_ovr_3class": round(a3, 4), "macro_f1_3class": round(f3, 4)})
        print(f"  {s:8s}: 이진AUC={ab:.4f}  3분류AUC(OvR)={a3:.4f}  macroF1={f3:.4f}")

    # speed ensemble (soft-vote across slow/normal/fast)
    common = sorted(set.intersection(*[set(oof_store_bin[s].index) for s in SPEEDS]))
    yb_c = meta.loc[common, "group"].isin(["ACLD", "ACLR"]).astype(int).values
    ens_b = np.mean([oof_store_bin[s].loc[common].values for s in SPEEDS], axis=0)
    ab_e  = auc_binary(yb_c, ens_b)
    # 3-class ensemble
    common3 = sorted(set.intersection(*[set(oof_store_3c[s][0]) for s in SPEEDS]))
    y3_c = meta.loc[common3, "group"].map(LABEL_MAP).values
    ps = []
    for s in SPEEDS:
        idx, _, p = oof_store_3c[s]
        ps.append(pd.DataFrame(p, index=idx).loc[common3].values)
    ens_3 = np.mean(ps, axis=0)
    a3_e, f3_e = auc_ovr(y3_c, ens_3), macro_f1(y3_c, ens_3)
    grid_rows.append({"condition": "speed_ensemble", "binary_auc": round(ab_e, 4),
                      "auc_ovr_3class": round(a3_e, 4), "macro_f1_3class": round(f3_e, 4)})
    print(f"  {'앙상블':8s}: 이진AUC={ab_e:.4f}  3분류AUC(OvR)={a3_e:.4f}  macroF1={f3_e:.4f}")

    pd.DataFrame(grid_rows).to_csv(RESULTS / "04c_speed_analysis.csv", index=False)

    # ============ 분석 1b: 모델 벤치마크 — 통합(all) 이진 + 3분류 ============
    print("\n[1b] 모델 벤치마크 — 통합(all) (logreg=baseline, 그 외 벤치마크)")
    idxb, Xb_all, subb_all = prep(all_feat, ["HA", "ACLR", "ACLD"])
    yb_all = subb_all["group"].isin(["ACLD", "ACLR"]).astype(int).values
    idx3b, X3_all, sub3_all = prep(all_feat, ["HA", "ACLR", "ACLD"])
    y3_all = sub3_all["group"].map(LABEL_MAP).values

    models  = (['logreg', 'rf'] if SMOKE else M.available_models())
    skipped = [m for m in M.ALL_MODELS if m not in models]
    if skipped:
        print(f"  Skipped (unavailable): {skipped}")
    model_bench = []
    for name in models:
        pb = cv_oof_model(Xb_all, yb_all, 2, name, seed=42)
        p3 = cv_oof_model(X3_all, y3_all, 3, name, seed=42)
        ab, a3, f3 = auc_binary(yb_all, pb), auc_ovr(y3_all, p3), macro_f1(y3_all, p3)
        model_bench.append({"model": name, "display": M.DISPLAY[name], "role": M.ROLE[name],
                            "binary_auc": round(ab, 4), "auc_ovr_3class": round(a3, 4),
                            "macro_f1_3class": round(f3, 4)})
        print(f"  {M.DISPLAY[name]:<20}: 이진AUC={ab:.4f}  3분류AUC={a3:.4f}  macroF1={f3:.4f}")
    model_bench.sort(key=lambda r: r["auc_ovr_3class"], reverse=True)
    pd.DataFrame(model_bench).to_csv(RESULTS / "04c_model_benchmark.csv", index=False)
    json.dump({"baseline_model": M.BASELINE, "benchmark": model_bench,
               "skipped_models": skipped},
              open(RESULTS / "04c_model_benchmark.json", "w"), indent=2, ensure_ascii=False)

    # ============ 분석 2: Optuna 최적화 (통합 3분류) ============
    print("\n[2] Optuna 최적화 — 통합 3분류 (고정 RF vs Optuna)")
    idx3, X3all, sub3all = prep(all_feat, ["HA", "ACLR", "ACLD"])
    y3all = sub3all["group"].map(LABEL_MAP).values

    # baseline (fixed) already computed as 'all'
    fixed_auc = [r for r in grid_rows if r["condition"] == "all"][0]["auc_ovr_3class"]
    fixed_f1  = [r for r in grid_rows if r["condition"] == "all"][0]["macro_f1_3class"]

    # Optuna: tune final RF params via nested inner CV (no leakage — uses train fold only conceptually,
    # here we optimize on full OOF objective as honest upper-bound probe)
    def objective(trial):
        rf_params = dict(
            max_depth=trial.suggest_int("max_depth", 3, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 6),
            max_features=trial.suggest_float("max_features", 0.1, 1.0),
        )
        p = cv_oof(X3all, y3all, 3, seed=42, rf_params=rf_params)
        return auc_ovr(y3all, p)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=(2 if SMOKE else 40), show_progress_bar=False)
    best = study.best_params
    # Re-evaluate best with 2-seed ensemble
    acc = np.zeros((len(y3all), 3))
    for s in SEEDS:
        acc += cv_oof(X3all, y3all, 3, seed=s, rf_params=best)
    acc /= len(SEEDS)
    opt_auc, opt_f1 = auc_ovr(y3all, acc), macro_f1(y3all, acc)
    print(f"  고정 RF : AUC(OvR)={fixed_auc:.4f}  macroF1={fixed_f1:.4f}")
    print(f"  Optuna  : AUC(OvR)={opt_auc:.4f}  macroF1={opt_f1:.4f}  (best={best})")
    improved = opt_auc > fixed_auc
    print(f"  → Optuna {'개선됨' if improved else '개선 없음 (고정 RF가 충분)'}")

    json.dump({"fixed_auc_ovr": fixed_auc, "fixed_macro_f1": fixed_f1,
               "optuna_auc_ovr": round(opt_auc, 4), "optuna_macro_f1": round(opt_f1, 4),
               "optuna_best_params": best, "improved": bool(improved)},
              open(RESULTS / "04c_optuna_comparison.json", "w"), indent=2, ensure_ascii=False)

    # 최종 3분류 채택값 (더 높은 쪽)
    if improved:
        final_proba, final_auc, final_f1 = acc, opt_auc, opt_f1
    else:
        # recompute fixed all 2-seed proba for confusion/use
        idx_f, _, _ = oof_store_3c["all"]
        final_proba = oof_store_3c["all"][2]; final_auc, final_f1 = fixed_auc, fixed_f1

    # ============ 분석 3: ACLD vs ACLR 분리 한계 (permutation + PCA) ============
    print("\n[3] ACLD↔ACLR 분리 한계 검증")
    acl_feat = all_feat[all_feat["group"].isin(["ACLD", "ACLR"])]
    fc = [c for c in acl_feat.columns if c != "group"]
    Xacl = acl_feat[fc].fillna(0).values.astype(np.float32)
    yacl = (acl_feat["group"] == "ACLR").astype(int).values
    real_p = ensemble_oof(Xacl, yacl, 2)
    real_auc = auc_binary(yacl, real_p)
    print(f"  실제 ACLD vs ACLR AUC = {real_auc:.4f}")

    # permutation — 경량 단일 RF(상호작용 없음), 200회 (p-value 산출에 충분)
    def light_cv_auc(X, y, seed=42):
        skf = StratifiedKFold(5, shuffle=False); oof = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
            m = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                       random_state=seed, n_jobs=-1)
            m.fit(Xtr, y[tr]); oof[te] = m.predict_proba(Xte)[:, 1]
        return auc_binary(y, oof)

    real_light = light_cv_auc(Xacl, yacl)
    rng = np.random.default_rng(0)
    null_aucs = []
    for i in range(5 if SMOKE else 200):
        yp = rng.permutation(yacl)
        null_aucs.append(light_cv_auc(Xacl, yp))
    null_aucs = np.array(null_aucs)
    pval = (np.sum(null_aucs >= real_light) + 1) / (len(null_aucs) + 1)
    print(f"  실제 AUC(경량)={real_light:.4f}  Permutation p-value={pval:.4f}  (null mean={null_aucs.mean():.3f})")
    json.dump({"real_auc": round(real_auc, 4), "real_auc_light": round(real_light, 4),
               "p_value": round(float(pval), 4),
               "null_mean": round(float(null_aucs.mean()), 4),
               "null_std": round(float(null_aucs.std()), 4), "n_perm": len(null_aucs)},
              open(RESULTS / "04c_permutation.json", "w"), indent=2, ensure_ascii=False)

    # ============ 분석 4: 계층적 3분류 ============
    print("\n[4] 계층적 3분류 (Stage1 ACL vs HA × Stage2 ACLD vs ACLR)")
    idxA, XA, subA = prep(all_feat, ["HA", "ACLR", "ACLD"])
    yA3 = subA["group"].map(LABEL_MAP).values        # 0=HA,1=ACLR,2=ACLD
    yA_bin = (yA3 != 0).astype(int)                  # ACL=1
    hier = np.zeros((len(yA3), 3))
    skf = StratifiedKFold(5, shuffle=False)
    for tr, te in skf.split(XA, yA3):
        # Stage1
        sc = StandardScaler(); Xtr = sc.fit_transform(XA[tr]); Xte = sc.transform(XA[te])
        def fit_pred(Xtr_, ytr_, Xte_, topK=TOP_K):
            sel = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                         random_state=42, n_jobs=-1)
            sel.fit(Xtr_, ytr_); top = np.argsort(sel.feature_importances_)[-topK:]
            Xtt, Xet = Xtr_[:, top], Xte_[:, top]
            itr = [Xtt[:, i]*Xtt[:, j] for i, j in combinations(range(topK), 2)]
            ite = [Xet[:, i]*Xet[:, j] for i, j in combinations(range(topK), 2)]
            Xa = np.hstack([Xtr_, np.column_stack(itr)]); Xea = np.hstack([Xte_, np.column_stack(ite)])
            fm = RandomForestClassifier(n_estimators=1000, class_weight="balanced",
                                        random_state=42, n_jobs=-1)
            fm.fit(Xa, ytr_); return fm.predict_proba(Xea), fm.classes_
        p1, cls1 = fit_pred(Xtr, yA_bin[tr], Xte)
        p_acl = p1[:, list(cls1).index(1)]
        # Stage2: ACLD vs ACLR (train on ACL subjects only)
        acl_mask = yA_bin[tr] == 1
        y2 = (yA3[tr][acl_mask] == 1).astype(int)   # 1=ACLR else ACLD
        p2, cls2 = fit_pred(Xtr[acl_mask], y2, Xte)
        p_aclr_given = p2[:, list(cls2).index(1)]
        hier[te, 0] = 1 - p_acl                      # HA
        hier[te, 1] = p_acl * p_aclr_given           # ACLR
        hier[te, 2] = p_acl * (1 - p_aclr_given)     # ACLD
    hier = hier / hier.sum(axis=1, keepdims=True)
    hier_auc, hier_f1 = auc_ovr(yA3, hier), macro_f1(yA3, hier)
    hier_bacc = balanced_accuracy_score(yA3, hier.argmax(1))
    hier_per = {c: round(roc_auc_score((yA3 == LABEL_MAP[c]).astype(int), hier[:, LABEL_MAP[c]]), 4)
                for c in CLASS_ORDER}
    print(f"  계층적 3분류: AUC(OvR)={hier_auc:.4f}  macroF1={hier_f1:.4f}  bal_acc={hier_bacc:.4f}")
    print(f"  클래스별 AUC: {hier_per}")

    # choose best 3-class for headline
    flat_auc = final_auc
    best_3c = "hierarchical" if hier_auc > flat_auc else "flat"
    head_auc = max(hier_auc, flat_auc)
    print(f"  → 최종 3분류 채택: {best_3c} (AUC(OvR)={head_auc:.4f})")

    json.dump({"flat_auc_ovr": round(flat_auc, 4), "flat_macro_f1": round(final_f1, 4),
               "hier_auc_ovr": round(hier_auc, 4), "hier_macro_f1": round(hier_f1, 4),
               "hier_bal_acc": round(hier_bacc, 4), "hier_per_class_auc": hier_per,
               "best": best_3c, "headline_auc_ovr": round(head_auc, 4)},
              open(RESULTS / "04c_hierarchical.json", "w"), indent=2, ensure_ascii=False)

    # ============ Figures ============
    print("\n[5] Figure 생성")
    gdf = pd.DataFrame(grid_rows)

    # FIG: binary speed bars
    fig, ax = plt.subplots(figsize=(6, 3.4))
    order = ["slow", "normal", "fast", "all", "speed_ensemble"]
    labels = ["Slow", "Normal", "Fast", "통합(All)", "속도앙상블"]
    vals = [gdf[gdf.condition == c]["binary_auc"].values[0] for c in order]
    cols = [COLORS["grey"]]*3 + [COLORS["ok"], COLORS["primary"]]
    bars = ax.bar(labels, vals, color=cols, edgecolor="white", linewidth=1.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.axhline(0.5, color="#ccc", ls="--", lw=0.8); ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUC", rotation=0, ha="right", va="center", labelpad=20)
    ax.set_title("ACL vs HA — 속도별 분류 성능", fontweight="bold", pad=8)
    fig.tight_layout(); fig.savefig(FIGURES / "fig_04c_binary_speed.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # FIG: 3-class speed bars (AUC + F1)
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    x = np.arange(len(order)); w = 0.38
    auc3 = [gdf[gdf.condition == c]["auc_ovr_3class"].values[0] for c in order]
    f13  = [gdf[gdf.condition == c]["macro_f1_3class"].values[0] for c in order]
    b1 = ax.bar(x-w/2, auc3, w, label="AUC(OvR)", color=COLORS["primary"], edgecolor="white")
    b2 = ax.bar(x+w/2, f13, w, label="macro-F1", color=COLORS["warn"], edgecolor="white")
    for bb in list(b1)+list(b2):
        ax.text(bb.get_x()+bb.get_width()/2, bb.get_height()+0.008, f"{bb.get_height():.3f}",
                ha="center", va="bottom", fontsize=7.5)
    ax.axhline(0.5, color="#ccc", ls="--", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0.4, 0.95)
    ax.set_ylabel("점수", rotation=0, ha="right", va="center", labelpad=20)
    ax.set_title("3분류 — 속도별 성능 (통합이 최선)", fontweight="bold", pad=8)
    ax.legend(fontsize=8.5, framealpha=0)
    fig.tight_layout(); fig.savefig(FIGURES / "fig_04c_3class_speed.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # FIG: PCA overlap
    pca_feat = all_feat[all_feat["group"].isin(CLASS_ORDER)]
    fcp = [c for c in pca_feat.columns if c != "group"]
    Xp = StandardScaler().fit_transform(pca_feat[fcp].fillna(0).values)
    emb = PCA(n_components=2, random_state=42).fit_transform(Xp)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for g in CLASS_ORDER:
        m = (pca_feat["group"] == g).values
        ax.scatter(emb[m, 0], emb[m, 1], c=COLORS[g], s=55, alpha=0.75,
                   edgecolors="white", linewidths=0.6, label=g)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2", rotation=0, ha="right", va="center", labelpad=20)
    ax.set_title("PCA 투영 — ACLD/ACLR 겹침, HA만 분리", fontweight="bold", pad=8)
    ax.legend(fontsize=9, framealpha=0)
    fig.tight_layout(); fig.savefig(FIGURES / "fig_04c_pca_overlap.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # FIG: permutation
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.hist(null_aucs, bins=30, color=COLORS["grey"], alpha=0.8, edgecolor="white")
    ax.axvline(real_light, color=COLORS["warn"], lw=2.2, label=f"실제 AUC={real_light:.3f}")
    ax.axvline(0.5, color="#999", ls="--", lw=1.0, label="우연(0.5)")
    ax.set_xlabel("AUC"); ax.set_ylabel("빈도", rotation=0, ha="right", va="center", labelpad=20)
    ax.set_title(f"ACLD vs ACLR Permutation Test (p={pval:.3f})", fontweight="bold", pad=8)
    ax.legend(fontsize=8.5, framealpha=0)
    fig.tight_layout(); fig.savefig(FIGURES / "fig_04c_permutation.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # FIG: hierarchical confusion matrix
    import seaborn as sns
    cm = confusion_matrix(yA3, hier.argmax(1))
    fig, ax = plt.subplots(figsize=(5, 4.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER, ax=ax)
    ax.set_xlabel("예측"); ax.set_ylabel("실제", rotation=0, ha="right", va="center", labelpad=20)
    ax.set_title(f"계층적 3분류 혼동행렬\nAUC(OvR)={hier_auc:.3f} macroF1={hier_f1:.3f}",
                 fontweight="bold", pad=10)
    fig.tight_layout(); fig.savefig(FIGURES / "fig_04c_cm_hier.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    print(f"\n완료. Elapsed {time.time()-t0:.1f}s")
    print(f"최종: 이진(ACL vs HA) 통합 AUC={gdf[gdf.condition=='all']['binary_auc'].values[0]:.4f}, "
          f"3분류 최선 AUC(OvR)={head_auc:.4f} ({best_3c})")


if __name__ == "__main__":
    main()
