"""
03_optuna_optimization.py — Optuna 기반 파이프라인 하이퍼파라미터 최적화

목적 함수: 통합 Waveform (IMU + 관절 각도) 입력에 대해
           Robust MCD (support_fraction=0.75) 기반 마할라노비스 거리를 사용한
           OOF GroupKFold AUC-ROC 최대화

최적화 대상 하이퍼파라미터:
  - scaling          : zscore | robust | minmax
  - pca_k_method     : kaiser | variance_ratio | fixed
  - pca_variance_ratio (variance_ratio 선택 시): 0.70~0.99
  - pca_fixed_k      (fixed 선택 시)           : 2~100
  - speed_filter     : all | normal | fast | slow
  - distance_metric  : mahalanobis | squared_mahalanobis

결과:
  - results/optuna_mahalanobis.db         SQLite study 영구 보존
  - results/01_optuna_best_params.json    최적 파라미터 및 메트릭
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.trial import TrialState

ROOT    = Path(__file__).resolve().parent.parent.parent
SANDBOX = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_DIR    = SANDBOX / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ─── 02_mahalanobis_pipeline.py 동적 로드 ────────────────────────────────────

def _load_pipeline_func():
    """02_mahalanobis_pipeline의 run_pipeline 함수를 동적 로드."""
    import importlib.util
    script = SANDBOX / "scripts" / "02_mahalanobis_pipeline.py"
    spec = importlib.util.spec_from_file_location("pipe_module", script)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run_pipeline




# ─── Objective ───────────────────────────────────────────────────────────────

class OptimObjective:
    def __init__(self, df: pd.DataFrame, n_splits: int = 5) -> None:
        self.df       = df
        self.n_splits = n_splits
        self.run_pipeline = _load_pipeline_func()

    def __call__(self, trial: optuna.Trial) -> float:
        scaling       = trial.suggest_categorical("scaling", ["zscore", "robust", "minmax"])
        pca_k_method  = trial.suggest_categorical("pca_k_method", ["kaiser", "variance_ratio", "fixed"])
        speed_filter  = trial.suggest_categorical("speed_filter", ["all", "normal", "fast", "slow"])
        dist_metric   = trial.suggest_categorical("distance_metric", ["mahalanobis", "squared_mahalanobis"])

        pca_var_ratio = 0.90
        pca_fixed_k   = 10
        if pca_k_method == "variance_ratio":
            pca_var_ratio = trial.suggest_float("pca_variance_ratio", 0.70, 0.99)
        elif pca_k_method == "fixed":
            pca_fixed_k = trial.suggest_int("pca_fixed_k", 2, 100)

        try:
            result = self.run_pipeline(
                self.df,
                scaling=scaling,
                pca_k_method=pca_k_method,
                pca_variance_ratio=pca_var_ratio,
                pca_fixed_k=pca_fixed_k,
                speed_filter=speed_filter,
                distance_metric=dist_metric,
                n_splits=self.n_splits,
                mcd_support_fraction=0.75,
                verbose=False,
            )
            auc = result["auc_roc"]
            if np.isnan(auc):
                return 0.0
            return auc
        except Exception as e:
            print(f"  [Optuna] Trial {trial.number} 실패: {e}")
            return 0.0


# ─── Early Stopping 콜백 ─────────────────────────────────────────────────────

class EarlyStoppingCallback:
    """
    최근 n_patience trials 동안 best value가 개선되지 않으면 최적화를 중단.

    Args:
        n_patience: 개선 없이 허용할 최대 trial 수
        min_delta  : '개선'으로 인정할 최소 AUC 향상폭
    """

    def __init__(self, n_patience: int = 30, min_delta: float = 1e-4) -> None:
        self.n_patience = n_patience
        self.min_delta  = min_delta
        self._best      = -float("inf")
        self._no_improve = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != TrialState.COMPLETE:
            return
        val = trial.value
        if val is None:
            return
        if val > self._best + self.min_delta:
            self._best = val
            self._no_improve = 0
        else:
            self._no_improve += 1
            if self._no_improve >= self.n_patience:
                print(f"\n[03] ⏹ Early Stop: {self.n_patience} trials 연속 개선 없음 "
                      f"(best={self._best:.4f})")
                study.stop()


# ─── 메인 ────────────────────────────────────────────────────────────────────

def main(test_mode: bool = False, n_trials: int = 200, n_patience: int = 30) -> None:
    src = DATA_PROCESSED / ("mahalanobis_features_test.parquet" if test_mode else "mahalanobis_features.parquet")
    if not src.exists():
        print(f"[03] ❌ 입력 파일 없음: {src}")
        sys.exit(1)

    print(f"[03] 로드: {src}")
    df = pd.read_parquet(src)
    print(f"[03] shape={df.shape}")

    n_subjects = df["subject_id"].nunique()
    n_splits   = min(5, n_subjects // 3) if n_subjects >= 3 else 2
    n_splits   = max(n_splits, 2)

    # 테스트 모드에서는 trials/patience 축소
    if test_mode:
        n_trials  = min(n_trials, 5)
        n_patience = 3

    db_path = RESULTS_DIR / "optuna_mahalanobis.db"
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name="mahalanobis_impairment",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=TPESampler(seed=42),
    )

    # 기존 완료된 trial 수 출력
    existing = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"[03] Optuna study: {existing}개 기존 trial, {n_trials}개 추가 최적화 시작")

    objective = OptimObjective(df, n_splits=n_splits)

    early_stop = EarlyStoppingCallback(n_patience=n_patience)

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1,
        callbacks=[early_stop],
    )

    best = study.best_trial
    print(f"\n[03] ✅ 최적 AUC-ROC: {best.value:.4f}")
    print(f"[03] 최적 파라미터: {best.params}")

    # 결과 저장
    out = RESULTS_DIR / "01_optuna_best_params.json"
    result_dict = {
        "best_auc_roc": best.value,
        "best_params":  best.params,
        "n_trials":     len(study.trials),
        "study_name":   study.study_name,
        "db_path":      str(db_path),
    }
    out.write_text(json.dumps(result_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[03] 저장: {out}")

    # 최적 파라미터 요약 출력
    print("\n[03] === 최적 파라미터 요약 ===")
    for k, v in best.params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    test_mode = "--test" in sys.argv
    n_trials  = 200
    n_patience = 30
    for i, arg in enumerate(sys.argv):
        if arg == "--trials" and i + 1 < len(sys.argv):
            n_trials = int(sys.argv[i + 1])
        if arg == "--patience" and i + 1 < len(sys.argv):
            n_patience = int(sys.argv[i + 1])
    main(test_mode=test_mode, n_trials=n_trials, n_patience=n_patience)
