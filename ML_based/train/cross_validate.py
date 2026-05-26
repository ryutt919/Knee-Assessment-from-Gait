"""
Outer GroupKFold 루프 — 누수 없는 학습·평가
- scaler: train fold에서만 fit
- Optuna inner CV로 HPO
- MLflow + CSV 병행 로깅
"""
from __future__ import annotations
import time
import subprocess
from pathlib import Path
from copy import deepcopy

import mlflow
import numpy as np
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler

from loaders.splits import make_outer_splits, verify_no_leakage
from eval.metrics import compute_metrics
from train.logger import log_run

ROOT = Path(__file__).parent.parent.parent
ML   = Path(__file__).parent.parent


def _get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def run_cv(
    model_name: str,
    model_cls,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: DictConfig,
    speed_data: np.ndarray | None = None,
    mask_data: np.ndarray | None = None,
    is_dl: bool = False,
) -> list[dict]:
    """
    Outer GroupKFold 교차 검증 실행.
    Returns: fold별 metric dict 리스트
    """
    from train.optimize import run_optuna

    label_map = {0: "HA", 1: "ACL"} if cfg.targets.mode == "binary" else {0: "ACLD", 1: "ACLR", 2: "HA"}
    label_names = [label_map[i] for i in sorted(label_map)]

    print(f"[{model_name}] 교차 검증 시작 (outer k={cfg.cv.n_outer})")
    splits = make_outer_splits(groups, cfg.cv.n_outer, cfg.cv.seed, y=y)
    verify_no_leakage(splits, groups)

    mlflow.set_experiment("acl_gait_classification")
    git_hash = _get_git_hash()
    fold_results = []

    for fold, (tr_idx, te_idx) in enumerate(splits):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        gr_tr = groups[tr_idx]

        sp_tr = speed_data[tr_idx] if speed_data is not None else None
        sp_te = speed_data[te_idx] if speed_data is not None else None
        mk_tr = mask_data[tr_idx] if mask_data is not None else None
        mk_te = mask_data[te_idx] if mask_data is not None else None

        # Scaler: train fold에서만 fit
        if X_tr.ndim == 3:
            from loaders.waveform_loader import WaveformScaler
            scaler = WaveformScaler(mode=cfg.features.waveform_norm)
            if cfg.features.waveform_norm == "ha_centered":
                ha_mask = (y_tr == 2) if cfg.targets.mode == "multiclass" else (y_tr == 0)
                scaler.fit(X_tr, ha_mask=ha_mask)
            else:
                scaler.fit(X_tr)
            X_tr = scaler.transform(X_tr)
            X_te = scaler.transform(X_te)
        elif not is_dl:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

        # Optuna inner CV HPO
        best_params = run_optuna(
            model_name=model_name, model_cls=model_cls,
            X_tr=X_tr, y_tr=y_tr, groups_tr=gr_tr, cfg=cfg, fold=fold,
            sp_tr=sp_tr, mk_tr=mk_tr, is_dl=is_dl,
        )

        # 최종 모델 학습
        t0 = time.time()
        with mlflow.start_run(run_name=f"{model_name}_fold{fold}") as run:
            mlflow.log_params({
                "model": model_name, "fold": fold,
                "waveform_type": cfg.features.waveform_type,
                "waveform_norm": cfg.features.waveform_norm,
                "target": cfg.targets.mode,
                "feature_select": cfg.features.feature_select,
                "stride_trim": cfg.features.stride_trim,
                "speed_as_feature": cfg.features.speed_as_feature,
                "n_outer": cfg.cv.n_outer, "n_inner": cfg.cv.n_inner,
                "seed": cfg.cv.seed,
                "git_hash": git_hash,
                **{f"hp_{k}": v for k, v in best_params.items()},
            })

            model = model_cls(**best_params)
            if is_dl:
                val_split_idx = int(len(X_tr) * 0.15)
                val_data = (X_tr[:val_split_idx], sp_tr[:val_split_idx] if sp_tr is not None else None,
                            mk_tr[:val_split_idx] if mk_tr is not None else None, y_tr[:val_split_idx])
                X_fit, y_fit = X_tr[val_split_idx:], y_tr[val_split_idx:]
                sp_fit = sp_tr[val_split_idx:] if sp_tr is not None else None
                mk_fit = mk_tr[val_split_idx:] if mk_tr is not None else None
                model.fit(X_fit, y_fit, speed=sp_fit, mask=mk_fit,
                          val_data=val_data, mlflow_run=run)
                y_pred  = model.predict(X_te, speed=sp_te, mask=mk_te)
                y_proba = model.predict_proba(X_te, speed=sp_te, mask=mk_te)
            else:
                model.fit(X_tr, y_tr)
                y_pred  = model.predict(X_te)
                y_proba = model.predict_proba(X_te)

            train_sec = time.time() - t0
            metrics = compute_metrics(y_te, y_pred, y_proba, label_names)
            mlflow.log_metrics(metrics)
            mlflow.log_param("train_time_sec", f"{train_sec:.1f}")
            mlflow.log_param("n_train_samples", len(X_tr))
            mlflow.log_param("n_test_samples", len(X_te))

        log_run(
            run_id=run.info.run_id, model_name=model_name, fold=fold,
            cfg=cfg, best_params=best_params, metrics=metrics,
            train_sec=train_sec, n_train=len(X_tr), n_test=len(X_te),
            git_hash=git_hash,
        )
        fold_results.append({"fold": fold, **metrics})
        print(f"[cv] fold {fold} {model_name}: macro_f1={metrics['macro_f1']:.4f}")

    mean_f1 = np.mean([r["macro_f1"] for r in fold_results])
    std_f1  = np.std([r["macro_f1"] for r in fold_results])
    print(f"[cv] {model_name} — macro_f1 {mean_f1:.4f}±{std_f1:.4f}")
    return fold_results
