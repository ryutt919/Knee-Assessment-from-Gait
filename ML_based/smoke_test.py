"""
스모크 테스트 — 전체 파이프라인 및 모든 모델이 에러 없이 실행되는지 확인.
실제 데이터 사용, --test 모드 설정 (극소량 샘플·최소 epoch).
"""
from __future__ import annotations
import argparse, sys, traceback, time
from pathlib import Path

import numpy as np

ML = Path(__file__).parent
sys.path.insert(0, str(ML))

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--skip-transformer", action="store_true",
                     help="Transformer 섹션 건너뜀 (MPS segfault 디버깅 중 등)")
_args, _ = _parser.parse_known_args()
SKIP_TRANSFORMER = _args.skip_transformer

from omegaconf import OmegaConf

# ── 테스트용 config (test_mode 활성화) ───────────────────────────────────────
BASE_CFG = OmegaConf.load(ML / "configs" / "config.yaml")
SMOKE_OVERRIDE = OmegaConf.create({
    "test_mode": {"enabled": True},
    "cv":        {"n_outer": 2, "n_inner": 2, "seed": 42},
    "optuna":    {"n_trials": {"sklearn": 1, "dl": 1}, "pruner_warmup_steps": 0},
    "features":  {"feature_select": "all"},   # eta2 필터 무시
    "models": {
        "cnn1d": {
            "max_epochs": 2, "batch_size": 8, "patience": 1, "min_delta": 0.0,
            "early_stop": {"monitor": "val_macro_f1", "patience": 1,
                           "min_delta": 0.0, "restore_best": True},
        },
        "transformer": {
            "max_epochs": 2, "batch_size": 4, "patience": 1, "min_delta": 0.0,
            "early_stop": {"monitor": "val_macro_f1", "patience": 1,
                           "min_delta": 0.0, "restore_best": True},
        },
    },
})
CFG = OmegaConf.merge(BASE_CFG, SMOKE_OVERRIDE)

PASS_LIST: list[str] = []
FAIL_LIST: list[str] = []
TEST_LOG:  list[dict] = []   # save_test_result 용 상세 기록


_SECTION_COUNT = 0

def section(title: str, data_info: str = ""):
    global _SECTION_COUNT
    _SECTION_COUNT += 1
    print(f"\n{'='*60}")
    suffix = f"  [{data_info}]" if data_info else ""
    print(f"  {title}{suffix}")
    print("=" * 60)


def check(name: str, fn):
    t0 = time.time()
    try:
        result  = fn()
        elapsed = time.time() - t0
        print(f"  [PASS] {name:<45} ({elapsed:.1f}s)")
        PASS_LIST.append(name)
        TEST_LOG.append({"name": name, "status": "PASS",
                         "elapsed_sec": round(elapsed, 2), "error": None})
        return result
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [FAIL] {name:<45} ({elapsed:.1f}s)")
        traceback.print_exc()
        FAIL_LIST.append(name)
        TEST_LOG.append({"name": name, "status": "FAIL",
                         "elapsed_sec": round(elapsed, 2),
                         "error": traceback.format_exc()})
        return None


# ── 헬퍼: 균형 샘플 인덱스 ───────────────────────────────────────────────────
def _balanced_idx(y: np.ndarray, n_per_class: int = 20) -> np.ndarray:
    rng = np.random.default_rng(42)
    idxs = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        n = min(n_per_class, len(cls_idx))
        idxs.append(rng.choice(cls_idx, n, replace=False))
    return np.sort(np.concatenate(idxs))


# ── 1. 로더 테스트 ────────────────────────────────────────────────────────────
section("1. 로더")

from loaders.scalar_loader   import load_stride_scalar, load_subject_scalar
from loaders.waveform_loader import load_waveform
from loaders.splits          import make_outer_splits, verify_no_leakage

scalar_data   = check("load_stride_scalar",
                       lambda: load_stride_scalar(CFG))
subject_data  = check("load_subject_scalar",
                       lambda: load_subject_scalar(CFG))
waveform_data = check("load_waveform (norm_101)",
                       lambda: load_waveform(CFG))

if scalar_data:
    X_sc, y_sc, g_sc, feat_names_sc = scalar_data
    idx = _balanced_idx(y_sc, n_per_class=20)
    X_sc_b, y_sc_b, g_sc_b = X_sc[idx], y_sc[idx], g_sc[idx]
    print(f"    scalar  X={X_sc_b.shape}, y={y_sc_b.shape}, "
          f"classes={np.unique(y_sc_b).tolist()}, groups={len(set(g_sc_b))}, "
          f"features={len(feat_names_sc)}, top3={feat_names_sc[:3]}")

if waveform_data:
    X_wv, sp_wv, mk_wv, y_wv, g_wv = waveform_data
    idx_wv = _balanced_idx(y_wv, n_per_class=10)
    X_wv_b, sp_wv_b = X_wv[idx_wv], sp_wv[idx_wv] if sp_wv is not None else None
    mk_wv_b, y_wv_b, g_wv_b = (mk_wv[idx_wv] if mk_wv is not None else None,
                                 y_wv[idx_wv], g_wv[idx_wv])
    print(f"    waveform X={X_wv_b.shape}, speed={sp_wv_b.shape if sp_wv_b is not None else None}")

# ── 2. splits 검증 ────────────────────────────────────────────────────────────
section("2. splits 검증")

def _test_splits():
    splits = make_outer_splits(g_sc_b, 2, 42)
    verify_no_leakage(splits, g_sc_b)
    return splits

splits_result = check("make_outer_splits + verify_no_leakage",
                       _test_splits) if scalar_data else None

# ── 3. sklearn 모델 ───────────────────────────────────────────────────────────
_sc_info = f"X={X_sc_b.shape}, y={np.unique(y_sc_b, return_counts=True)[1].tolist()}" if scalar_data else "데이터 없음"
section("3. sklearn 모델 (scalar, 균형 샘플)", _sc_info)

from models.sklearn_models import LogReg, LinearSVCModel, SVMRBF, RandomForest, GBT, XGBoost, LightGBM
from sklearn.preprocessing import StandardScaler


def _fit_predict(model_cls, X, y, extra_kw=None):
    n_classes = len(np.unique(y))
    kw = {"n_classes": n_classes, **(extra_kw or {})}
    model = model_cls(**kw)
    sc = StandardScaler()
    X_s = sc.fit_transform(X)
    model.fit(X_s, y)
    preds  = model.predict(X_s)
    probas = model.predict_proba(X_s)
    assert preds.shape  == (len(y),), f"pred shape {preds.shape}"
    assert probas.shape == (len(y), n_classes), f"proba shape {probas.shape}"


if scalar_data:
    for cls, kw in [
        (LogReg,         None),
        (LinearSVCModel, None),
        (SVMRBF,         None),
        (RandomForest,   None),
        (GBT,            {"n_estimators": 10}),
        (XGBoost,        {"n_estimators": 10}),
        (LightGBM,       {"n_estimators": 10}),
    ]:
        check(f"{cls.__name__}.fit/predict",
              lambda c=cls, k=kw: _fit_predict(c, X_sc_b, y_sc_b, k))

# ── 4. FPCA ───────────────────────────────────────────────────────────────────
_wv_info = f"X={X_wv_b.shape}, y={np.unique(y_wv_b, return_counts=True)[1].tolist()}" if waveform_data else "데이터 없음"
section("4. FPCA 모델", _wv_info)

def _test_fpca():
    from models.fpca import FPCAClassifier
    if waveform_data is None:
        raise RuntimeError("waveform_data 없음")
    n_classes = len(np.unique(y_wv_b))
    model = FPCAClassifier(n_components=5, clf_C=1.0)
    model.fit(X_wv_b, y_wv_b)
    preds  = model.predict(X_wv_b)
    probas = model.predict_proba(X_wv_b)
    assert preds.shape  == (len(y_wv_b),)
    assert probas.shape == (len(y_wv_b), n_classes)

check("FPCAClassifier.fit/predict", _test_fpca)

# ── 5. CNN1D ──────────────────────────────────────────────────────────────────
section("5. CNN1D 모델 (PyTorch)", _wv_info)

def _test_cnn1d():
    from models.cnn1d import CNN1D
    from utils.device import get_device
    if waveform_data is None:
        raise RuntimeError("waveform_data 없음")
    n_classes  = len(np.unique(y_wv_b))
    speed_dim  = sp_wv_b.shape[1] if sp_wv_b is not None else 0
    val_n      = max(2, int(len(y_wv_b) * 0.15))
    val_data   = (X_wv_b[:val_n], sp_wv_b[:val_n] if sp_wv_b is not None else None,
                  mk_wv_b[:val_n] if mk_wv_b is not None else None, y_wv_b[:val_n])
    model = CNN1D(n_classes=n_classes, num_layers=2, channels=16, kernel_size=3,
                  dropout=0.1, lr=1e-3, max_epochs=2, patience=1, min_delta=0.0,
                  speed_dim=speed_dim)
    model.fit(X_wv_b[val_n:], y_wv_b[val_n:],
              speed=sp_wv_b[val_n:] if sp_wv_b is not None else None,
              mask=mk_wv_b[val_n:] if mk_wv_b is not None else None,
              val_data=val_data)
    preds  = model.predict(X_wv_b, speed=sp_wv_b, mask=mk_wv_b)
    probas = model.predict_proba(X_wv_b, speed=sp_wv_b, mask=mk_wv_b)
    assert preds.shape == (len(y_wv_b),)
    assert probas.shape == (len(y_wv_b), n_classes)
    print(f"    device: {model.device}")

check("CNN1D.fit/predict", _test_cnn1d)

# ── 6. Transformer ────────────────────────────────────────────────────────────
_transformer_suffix = "  [SKIPPED: --skip-transformer]" if SKIP_TRANSFORMER else ""
section("6. Transformer 모델 (PyTorch)", _wv_info + _transformer_suffix)

def _test_transformer():
    from models.transformer import TransformerModel
    if waveform_data is None:
        raise RuntimeError("waveform_data 없음")
    n_classes  = len(np.unique(y_wv_b))
    speed_dim  = sp_wv_b.shape[1] if sp_wv_b is not None else 0
    val_n      = max(2, int(len(y_wv_b) * 0.15))
    val_data   = (X_wv_b[:val_n], sp_wv_b[:val_n] if sp_wv_b is not None else None,
                  mk_wv_b[:val_n] if mk_wv_b is not None else None, y_wv_b[:val_n])
    model = TransformerModel(n_classes=n_classes, d_model=32, n_heads=2, num_layers=1,
                             dim_feedforward=64, dropout=0.1, lr=1e-3, max_epochs=2,
                             patience=1, min_delta=0.0, speed_dim=speed_dim)
    model.fit(X_wv_b[val_n:], y_wv_b[val_n:],
              speed=sp_wv_b[val_n:] if sp_wv_b is not None else None,
              mask=mk_wv_b[val_n:] if mk_wv_b is not None else None,
              val_data=val_data)
    preds  = model.predict(X_wv_b, speed=sp_wv_b, mask=mk_wv_b)
    probas = model.predict_proba(X_wv_b, speed=sp_wv_b, mask=mk_wv_b)
    assert preds.shape == (len(y_wv_b),)
    assert probas.shape == (len(y_wv_b), n_classes)
    print(f"    device: {model.device}")

if not SKIP_TRANSFORMER:
    check("TransformerModel.fit/predict", _test_transformer)
else:
    print("  [SKIP] TransformerModel.fit/predict")

# ── 7. eval.metrics ───────────────────────────────────────────────────────────
section("7. eval.metrics")

from eval.metrics import compute_metrics

def _test_metrics_binary():
    rng = np.random.default_rng(0)
    y_t = rng.integers(0, 2, 40)
    y_p = rng.integers(0, 2, 40)
    yp  = rng.random((40, 2)); yp /= yp.sum(axis=1, keepdims=True)
    m   = compute_metrics(y_t, y_p, yp, ["HA", "ACL"])
    assert "macro_f1" in m and "roc_auc" in m

def _test_metrics_multi():
    rng = np.random.default_rng(1)
    y_t = rng.integers(0, 3, 60)
    y_p = rng.integers(0, 3, 60)
    yp  = rng.random((60, 3)); yp /= yp.sum(axis=1, keepdims=True)
    m   = compute_metrics(y_t, y_p, yp, ["ACLD", "ACLR", "HA"])
    assert "f1_ACLD" in m

check("compute_metrics (binary)",     _test_metrics_binary)
check("compute_metrics (multiclass)", _test_metrics_multi)

# ── 8. train.logger ───────────────────────────────────────────────────────────
section("8. train.logger (CSV 로깅)")

from train.logger import log_run

def _test_logger():
    log_run(run_id="smoke_test_run", model_name="logreg", fold=0,
            cfg=CFG, best_params={"C": 1.0, "penalty": "l2"},
            metrics={"macro_f1": 0.75, "balanced_acc": 0.74, "roc_auc": 0.80,
                     "f1_HA": 0.70, "f1_ACL": 0.78, "precision_macro": 0.72,
                     "recall_macro": 0.73, "cm_HA_pred_HA": 10, "cm_HA_pred_ACL": 3},
            train_sec=1.5, n_train=100, n_test=30)
    import glob
    logs = glob.glob(str(ML / f"artifacts-{CFG.version}" / "logs" / "runs_*.csv"))
    assert len(logs) > 0

check("log_run → CSV 저장", _test_logger)

# ── 9. train.optimize (Optuna, 1 trial) ──────────────────────────────────────
section("9. train.optimize (Optuna inner CV, 1 trial)")

def _test_optuna():
    if scalar_data is None:
        raise RuntimeError("scalar_data 없음")
    from train.optimize import run_optuna
    best = run_optuna("logreg", LogReg,
                      X_sc_b, y_sc_b, g_sc_b, CFG, fold=99, is_dl=False)
    assert isinstance(best, dict), f"best_params 타입 오류: {type(best)}"

check("run_optuna (logreg, 1 trial)", _test_optuna)

# ── 10. cross_validate (2 outer folds, logreg) ───────────────────────────────
section("10. cross_validate (logreg, 2 outer folds)")

def _test_cv():
    if scalar_data is None:
        raise RuntimeError("scalar_data 없음")
    from train.cross_validate import run_cv
    fn = feat_names_sc if scalar_data else None
    results = run_cv(model_name="logreg", model_cls=LogReg,
                     X=X_sc_b, y=y_sc_b, groups=g_sc_b, cfg=CFG,
                     speed_data=None, mask_data=None, is_dl=False,
                     feature_names=fn)
    assert len(results) == CFG.cv.n_outer
    assert "macro_f1" in results[0]

check("run_cv (logreg, 2-fold)", _test_cv)

# ── 11. Gait Normality Score ─────────────────────────────────────────────────
section("11. Gait Normality Score")

def _test_recovery():
    import numpy as np
    from recovery_score.components import aggregate_cycle_waveforms, load_cycle_waveforms
    from recovery_score.scorer import GaitNormalityScorer
    cycle_path = ML.parent / "data" / "processed" / "cycle_waveforms_101.parquet"
    dataset = aggregate_cycle_waveforms(load_cycle_waveforms(cycle_path))
    groups = dataset.metadata["group"].to_numpy(dtype=str)
    scorer = GaitNormalityScorer().fit(
        dataset.matrices,
        groups,
        dataset.metadata["subject_id"].to_numpy(dtype=str),
    )
    result = scorer.score_matrices(dataset.matrices[:3])
    required = {"normality_score", "z_deviation", "hip_score", "fast_score", "asymmetry_score"}
    assert required <= set(result.columns)
    assert np.isfinite(result[list(required)].to_numpy()).all()
    print(f"    sample scores: {result['normality_score'].round(1).tolist()}")

check("GaitNormalityScorer.fit/score", _test_recovery)

# ── 12. WaveformScaler ────────────────────────────────────────────────────────
section("12. WaveformScaler")

def _test_waveform_scaler():
    if waveform_data is None:
        raise RuntimeError("waveform_data 없음")
    from loaders.waveform_loader import WaveformScaler
    for mode in ["zscore", "none"]:
        sc = WaveformScaler(mode=mode)
        sc.fit(X_wv_b, y=y_wv_b)
        Xt = sc.transform(X_wv_b)
        assert Xt.shape == X_wv_b.shape
    ha_mask = (y_wv_b == 0)
    if ha_mask.any():
        sc = WaveformScaler(mode="ha_centered")
        sc.fit(X_wv_b, y=y_wv_b, ha_mask=ha_mask)
        assert sc.transform(X_wv_b).shape == X_wv_b.shape

check("WaveformScaler (zscore/ha_centered/none)", _test_waveform_scaler)

# ── 요약 ─────────────────────────────────────────────────────────────────────
section("스모크 테스트 결과")
total = len(PASS_LIST) + len(FAIL_LIST)
print(f"\n  통과: {len(PASS_LIST)}/{total}")
if FAIL_LIST:
    print(f"  실패 항목:")
    for f in FAIL_LIST:
        print(f"    ✗ {f}")
else:
    print("  모든 테스트 통과!")

# ── 결과 파일 저장 ────────────────────────────────────────────────────────────
from train.logger import save_test_result
save_test_result(
    test_type="smoke",
    results=TEST_LOG,
    mode="test",
    extra={"config": {"n_outer": CFG.cv.n_outer, "n_inner": CFG.cv.n_inner,
                      "feature_select": CFG.features.feature_select}},
)

sys.exit(0 if not FAIL_LIST else 1)
