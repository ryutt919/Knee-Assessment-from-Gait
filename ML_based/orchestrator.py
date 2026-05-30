"""
모델 선택 실행 진입점.

사용 예:
    python orchestrator.py --models logreg rf xgboost
    python orchestrator.py --models cnn1d transformer --waveform_type raw_padded
    python orchestrator.py --models all --target multiclass
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

ML   = Path(__file__).parent
ROOT = ML.parent
sys.path.insert(0, str(ML))

from loaders.scalar_loader  import load_stride_scalar
from loaders.waveform_loader import load_waveform
from train.cross_validate   import run_cv

# ── 모델 레지스트리 ────────────────────────────────────────────────────────────
def _build_registry():
    # 모델 임포트를 지연(Lazy)시켜 메인 프로세스에서의 OpenMP 라이브러리 로드를 방지합니다.
    # (loader_type, module_name, class_name, is_dl)
    return {
        "logreg":      ("scalar",   "models.sklearn_models", "LogReg",          False),
        "linearsvc":   ("scalar",   "models.sklearn_models", "LinearSVCModel",  False),
        "svm_rbf":     ("scalar",   "models.sklearn_models", "SVMRBF",          False),
        "rf":          ("scalar",   "models.sklearn_models", "RandomForest",    False),
        "gbt":         ("scalar",   "models.sklearn_models", "GBT",             False),
        "xgboost":     ("scalar",   "models.sklearn_models", "XGBoost",         False),
        "lightgbm":    ("scalar",   "models.sklearn_models", "LightGBM",        False),
        "fpca":        ("waveform", "models.fpca",           "FPCAClassifier",  False),
        "cnn1d":       ("waveform", "models.cnn1d",          "CNN1D",           True),
        "transformer": ("waveform", "models.transformer",    "TransformerModel", True),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="ACL Gait ML Pipeline Orchestrator")
    p.add_argument("--models", nargs="+", default=["all"],
                   help="모델 이름 목록 (all | logreg rf xgboost ...)")
    p.add_argument("--config", default=str(ML / "configs" / "config.yaml"),
                   help="설정 파일 경로")
    p.add_argument("--target", choices=["binary", "multiclass"], default=None,
                   help="분류 타깃 override (config 기본값 사용 시 생략)")
    p.add_argument("--waveform_type", choices=["norm_101", "cycle_norm_101", "raw_padded"], default=None,
                   help="waveform_type override")
    p.add_argument("--waveform_norm", choices=["none", "zscore", "ha_centered"], default=None,
                   help="waveform_norm override")
    p.add_argument("--feature_select", default=None,
                   help="feature_select override (e.g. eta2_top30, all)")
    p.add_argument("--version", default=None,
                   help="실험 버전명 override (config 기본값 사용 시 생략, 예: v2)")
    p.add_argument("--test", action="store_true",
                   help="테스트 모드: 극소량 샘플·최소 epoch으로 파이프라인 실행 확인")
    p.add_argument("--skip-transformer", action="store_true",
                   help="Transformer 모델 건너뜀 (MPS 이슈 등)")
    return p.parse_args()


# ── 데이터 로딩 ───────────────────────────────────────────────────────────────
def _subsample_test(X, y, groups, speed, mask, n_per_group: int = 20) -> tuple:
    """테스트 모드: 그룹(subject)별 n_per_group명을 균등 추출."""
    import numpy as np
    rng = np.random.default_rng(42)
    y_1d      = np.asarray(y).ravel()
    groups_1d = np.asarray(groups).ravel()
    uniq_subjects = np.unique(groups_1d)
    # subject별 레이블 결정 (최빈값)
    subj_label = {s: int(np.bincount(y_1d[groups_1d == s]).argmax()) for s in uniq_subjects}

    # 클래스별 subject 목록
    from collections import defaultdict
    by_class = defaultdict(list)
    for s, lbl in subj_label.items():
        by_class[lbl].append(s)

    selected_subjects = []
    for lbl, subjects in by_class.items():
        n = min(n_per_group, len(subjects))
        selected_subjects.extend(rng.choice(subjects, n, replace=False).tolist())

    mask_idx = np.isin(groups_1d, selected_subjects)
    idx = np.where(mask_idx)[0]
    sp_sub  = speed[idx] if speed is not None else None
    mk_sub  = mask[idx]  if mask  is not None else None
    return X[idx], y[idx], groups_1d[idx], sp_sub, mk_sub


def _load_for_model(loader_type: str, cfg) -> tuple:
    """
    Returns (X, y, groups, speed_data, mask_data, feature_names)
    feature_names: list[str] (scalar) | None (waveform)
    """
    if loader_type == "scalar":
        X, y, groups, feature_names = load_stride_scalar(cfg)
        speed, mask = None, None
    else:
        result = load_waveform(cfg)
        # load_waveform 반환 순서: (X, speed_oh, mask, y, groups)
        X, speed, mask, y, groups = result
        feature_names = None

    if cfg.get("test_mode", {}).get("enabled", False):
        n_per = cfg.test_mode.n_samples // 3  # 3 groups
        X, y, groups, speed, mask = _subsample_test(X, y, groups, speed, mask, n_per)
        print(f"[orchestrator] test 서브샘플: {X.shape[0]}행, {len(set(groups))}피험자")

    return X, y, groups, speed, mask, feature_names


def _run_model_isolated(model_name, module_path, class_name, X, y, groups,
                        cfg, speed, mask, is_dl, feature_names, return_dict):
    import sys, os
    sys.stdout.reconfigure(line_buffering=True)  # 서브프로세스 출력을 실시간으로 flush
    if is_dl:
        os.environ["FORCE_CPU"] = "1"        # MPS segfault 방지
        os.environ["OMP_NUM_THREADS"] = "1"  # PyTorch 멀티스레드 ↔ Optuna 충돌 방지
    try:
        import importlib
        from train.cross_validate import run_cv

        mod = importlib.import_module(module_path)
        model_cls = getattr(mod, class_name)

        fold_results = run_cv(
            model_name=model_name,
            model_cls=model_cls,
            X=X, y=y, groups=groups,
            cfg=cfg,
            speed_data=speed,
            mask_data=mask,
            is_dl=is_dl,
            feature_names=feature_names,
        )
        return_dict["result"] = fold_results
    except Exception as e:
        import traceback
        return_dict["error"] = traceback.format_exc()


# ── 메인 ─────────────────────────────────────────────────────────────────────
def run(args=None):
    if args is None:
        args = parse_args()

    cfg = OmegaConf.load(args.config)

    # --version 플래그: config 기본값 override
    if getattr(args, "version", None):
        OmegaConf.update(cfg, "version", args.version)

    # --test 플래그: 극소량 데이터·최소 epoch으로 파이프라인 실행 확인
    if getattr(args, "test", False):
        tm = cfg.test_mode
        OmegaConf.update(cfg, "test_mode.enabled",      True)
        OmegaConf.update(cfg, "cv.n_outer",             tm.n_outer)
        OmegaConf.update(cfg, "cv.n_inner",             tm.n_inner)
        OmegaConf.update(cfg, "optuna.n_trials.sklearn", tm.n_trials_sklearn)
        OmegaConf.update(cfg, "optuna.n_trials.dl",      tm.n_trials_dl)
        OmegaConf.update(cfg, "features.feature_select", tm.feature_select)
        for m in ("cnn1d", "transformer"):
            OmegaConf.update(cfg, f"models.{m}.max_epochs", tm.max_epochs_dl)
            OmegaConf.update(cfg, f"models.{m}.patience",   1)
        print("[orchestrator] ⚠️  테스트 모드 활성화 — 최소 설정으로 실행")

    # CLI override (test 이후에 적용 → CLI가 우선)
    if args.target:
        OmegaConf.update(cfg, "targets.mode", args.target)
    if args.waveform_type:
        OmegaConf.update(cfg, "features.waveform_type", args.waveform_type)
    if args.waveform_norm:
        OmegaConf.update(cfg, "features.waveform_norm", args.waveform_norm)
    if args.feature_select:
        OmegaConf.update(cfg, "features.feature_select", args.feature_select)

    registry = _build_registry()

    # 모델 선택
    selected = list(registry.keys()) if "all" in args.models else args.models
    if getattr(args, "skip_transformer", False):
        selected = [m for m in selected if m != "transformer"]
        print("[orchestrator] Transformer 건너뜀 (--skip-transformer)")
    unknown  = [m for m in selected if m not in registry]
    if unknown:
        print(f"[orchestrator] 알 수 없는 모델: {unknown}")
        print(f"  사용 가능: {list(registry.keys())}")
        sys.exit(1)

    # config에서 비활성화된 모델 제거
    active = []
    for m in selected:
        model_cfg = cfg.models.get(m, {})
        if model_cfg.get("enabled", True):
            active.append(m)
        else:
            print(f"[orchestrator] {m}: config에서 비활성화됨, 건너뜀")

    print(f"[orchestrator] 실행 모델: {active}")
    print(f"[orchestrator] version={cfg.version}  "
          f"target={cfg.targets.mode}  "
          f"waveform_type={cfg.features.waveform_type}  "
          f"waveform_norm={cfg.features.waveform_norm}")

    all_results: dict[str, list[dict]] = {}
    test_log:    list[dict] = []   # --test 모드 저장용
    import time, traceback as _tb

    # 로더 캐시 (같은 loader_type은 한 번만 로드)
    _cache: dict[str, tuple] = {}

    import multiprocessing as mp

    for model_name in active:
        loader_type, module_path, class_name, is_dl = registry[model_name]

        if loader_type not in _cache:
            print(f"[orchestrator] 데이터 로딩: {loader_type}")
            _cache[loader_type] = _load_for_model(loader_type, cfg)

        X, y, groups, speed, mask, feature_names = _cache[loader_type]

        print(f"\n{'='*60}")
        print(f"[orchestrator] 모델 시작: {model_name}  "
              f"(X={X.shape}, n_subjects={len(np.unique(groups))})")

        t0 = time.time()

        # 모든 모델을 격리된 서브프로세스에서 실행합니다.
        # - sklearn 모델: OpenMP 데드락 방지
        # - DL 모델(CNN1D 등): MPS segfault 격리 (서브프로세스에서 FORCE_CPU=1 적용)
        ctx = mp.get_context("spawn")
        manager = ctx.Manager()
        return_dict = manager.dict()

        p = ctx.Process(target=_run_model_isolated, args=(
            model_name, module_path, class_name, X, y, groups,
            cfg, speed, mask, is_dl, feature_names, return_dict
        ))
        p.start()
        p.join()

        if p.exitcode != 0 and "error" not in return_dict:
            return_dict["error"] = f"Subprocess crashed with exit code {p.exitcode} (Possible Segfault/OOM)"
            
        elapsed = time.time() - t0
            
        if "error" in return_dict:
            err = return_dict["error"]
            print(f"[orchestrator] {model_name} 실패:\n{err}")
            test_log.append({"name": model_name, "status": "FAIL",
                             "elapsed_sec": round(elapsed, 2),
                             "mean_macro_f1": None, "error": err})
        else:
            fold_results = return_dict.get("result", [])
            all_results[model_name] = fold_results
            mean_f1 = float(np.mean([r["macro_f1"] for r in fold_results])) if fold_results else 0.0
            test_log.append({"name": model_name, "status": "PASS",
                             "elapsed_sec": round(elapsed, 2),
                             "mean_macro_f1": round(mean_f1, 4), "error": None})

    # 최종 요약 출력
    print(f"\n{'='*60}")
    print("최종 결과 요약 (macro_f1 mean±std)")
    print(f"{'모델':<14} {'macro_f1':>10}  {'balanced_acc':>12}")
    print("-" * 40)
    for m, folds in all_results.items():
        f1s  = [r["macro_f1"]     for r in folds]
        bacs = [r["balanced_acc"] for r in folds]
        print(f"{m:<14} {np.mean(f1s):.4f}±{np.std(f1s):.4f}  "
              f"{np.mean(bacs):.4f}±{np.std(bacs):.4f}")

    # 결과 파일 저장 (--test 유무 관계없이 항상)
    if test_log:
        from train.logger import save_test_result
        is_test = getattr(args, "test", False)
        save_test_result(
            test_type="orchestrator",
            results=test_log,
            mode="test" if is_test else "full",
            extra={
                "models":       active,
                "target":       cfg.targets.mode,
                "waveform_type": cfg.features.waveform_type,
                "n_outer":      cfg.cv.n_outer,
            },
            version=cfg.version,
        )

    return all_results


if __name__ == "__main__":
    run()
