"""
CSV 상세 백업 로거 — MLflow 장애 시에도 재현 가능한 독립 기록.
artifacts/logs/runs_{datetime}.csv 에 실험별 1행 기록.
artifacts/logs/test_{type}_{datetime}.json 에 테스트 결과 저장.
"""
from __future__ import annotations
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

ML = Path(__file__).parent.parent
LOGS_DIR = ML / "artifacts" / "logs"


def _get_git_info() -> tuple[str, str]:
    root = ML.parent
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root), stderr=subprocess.DEVNULL
        ).decode().strip()
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(root), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash, git_branch = "unknown", "unknown"
    return git_hash, git_branch


def log_run(
    run_id: str,
    model_name: str,
    fold: int,
    cfg: DictConfig,
    best_params: dict,
    metrics: dict,
    train_sec: float,
    n_train: int,
    n_test: int,
    git_hash: str = "unknown",
    model_path: str = "",
    shap_path: str = "",
    cm_fig_path: str = "",
) -> None:
    """실험 결과 1행을 CSV에 append."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    _, git_branch = _get_git_info()
    ts = datetime.now().strftime("%Y%m%d")
    csv_path = LOGS_DIR / f"runs_{ts}.csv"

    row: dict = {
        # 실험 식별
        "run_id":          run_id,
        "experiment_name": "acl_gait_classification",
        "datetime":        datetime.now().isoformat(timespec="seconds"),
        "git_hash":        git_hash,
        "git_branch":      git_branch,
        # 데이터 설정
        "target":            cfg.targets.mode,
        "waveform_type":     cfg.features.waveform_type,
        "waveform_norm":     cfg.features.waveform_norm,
        "feature_select":    cfg.features.feature_select,
        "stride_trim":       cfg.features.stride_trim,
        "speed_as_feature":  cfg.features.speed_as_feature,
        # CV 설정
        "n_outer": cfg.cv.n_outer,
        "n_inner": cfg.cv.n_inner,
        "seed":    cfg.cv.seed,
        "fold":    fold,
        # 모델 정보
        "model_name":  model_name,
        "model_class": model_name,
        # 성능 지표
        "macro_f1":        metrics.get("macro_f1",        float("nan")),
        "balanced_acc":    metrics.get("balanced_acc",    float("nan")),
        "f1_HA":           metrics.get("f1_HA",           float("nan")),
        "f1_ACL":          metrics.get("f1_ACL",          float("nan")),
        "f1_ACLD":         metrics.get("f1_ACLD",         float("nan")),
        "f1_ACLR":         metrics.get("f1_ACLR",         float("nan")),
        "precision_macro": metrics.get("precision_macro", float("nan")),
        "recall_macro":    metrics.get("recall_macro",    float("nan")),
        "roc_auc":         metrics.get("roc_auc",         float("nan")),
        # 런타임
        "train_time_sec":  f"{train_sec:.1f}",
        "n_train_samples": n_train,
        "n_test_samples":  n_test,
        # 아티팩트
        "model_path":  model_path,
        "shap_path":   shap_path,
        "cm_fig_path": cm_fig_path,
    }

    # 혼동행렬 (cm_X_pred_Y 컬럼) — metrics에 있는 것 전부 포함
    for k, v in metrics.items():
        if k.startswith("cm_"):
            row[k] = v

    # 하이퍼파라미터 (hp_{key})
    for k, v in best_params.items():
        row[f"hp_{k}"] = v

    # CSV 쓰기 (파일 없으면 header 포함)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_test_result(
    test_type: str,                      # "smoke" | "orchestrator" | "pipeline"
    results: list[dict],                 # [{name, status, elapsed_sec, error?}, ...]
    mode: str = "test",                  # "test" | "full"
    extra: dict | None = None,           # 추가 메타 (모델명, 설정 등)
) -> Path:
    """
    테스트 결과를 artifacts/logs/test_{type}_{datetime}.json 에 저장.

    results 항목 형식:
        {"name": str, "status": "PASS"|"FAIL", "elapsed_sec": float, "error": str|None}
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    git_hash, git_branch = _get_git_info()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")

    payload = {
        "type":       test_type,
        "datetime":   datetime.now().isoformat(timespec="seconds"),
        "git_hash":   git_hash,
        "git_branch": git_branch,
        "mode":       mode,
        "total":      len(results),
        "passed":     passed,
        "failed":     failed,
        "success":    failed == 0,
        "results":    results,
        **(extra or {}),
    }

    out_path = LOGS_DIR / f"test_{test_type}_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[logger] 테스트 결과 저장: {out_path}  ({passed}/{len(results)} 통과)")
    return out_path
