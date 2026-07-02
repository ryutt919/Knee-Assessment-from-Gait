"""
run_pipeline.py — 전체 파이프라인 통합 실행 스크립트

사용법:
  python run_pipeline.py            # 전체 실행 (대용량 데이터)
  python run_pipeline.py --test     # 극소량 데이터로 파이프라인 검증
  python run_pipeline.py --skip 00  # 0단계(추출) 스킵
  python run_pipeline.py --skip 00 01 --trials 20  # 여러 단계 스킵

단계:
  00: raw_merged.parquet → raw_subset_mahalanobis.parquet (서브셋 추출)
  01: raw_subset → mahalanobis_features.parquet (Stride 분할 + 101pt 보간)
  02: mahalanobis_features → OOF 마할라노비스 거리 + Impairment Score
  03: Optuna 최적화 (기본 50 trials)
  04: SHAP 분석 및 시각화
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

SANDBOX = Path(__file__).resolve().parent
SCRIPTS = SANDBOX / "scripts"


def load_module(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_step(step_name: str, test_mode: bool, **kwargs) -> None:
    print(f"\n{'='*60}")
    print(f"  STEP {step_name}")
    print(f"{'='*60}")
    t0 = time.time()
    mod = load_module(step_name)
    mod.main(test_mode=test_mode, **kwargs)
    elapsed = time.time() - t0
    print(f"\n  ✓ {step_name} 완료 ({elapsed:.1f}초)")


def main() -> None:
    args      = sys.argv[1:]
    test      = "--test" in args
    skip_set  = set()
    n_trials  = 200
    n_patience = 30

    i = 0
    while i < len(args):
        if args[i] == "--skip":
            i += 1
            while i < len(args) and not args[i].startswith("--"):
                skip_set.add(args[i].zfill(2))
                i += 1
        elif args[i] == "--trials" and i + 1 < len(args):
            n_trials = int(args[i + 1])
            i += 2
        elif args[i] == "--patience" and i + 1 < len(args):
            n_patience = int(args[i + 1])
            i += 2
        else:
            i += 1

    mode_label = "테스트 모드 (극소량 데이터)" if test else "전체 데이터 모드"
    print(f"\n{'#'*60}")
    print(f"  Mahalanobis Impairment Score Pipeline")
    print(f"  모드: {mode_label}")
    print(f"  스킵: {skip_set if skip_set else '없음'}")
    print(f"  Optuna: n_trials={n_trials}, early_stop patience={n_patience}")
    print(f"{'#'*60}\n")

    t_total = time.time()

    # 00: 서브셋 추출
    if "00" not in skip_set:
        run_step("00_extract_subset", test_mode=test)

    # 01: 전처리
    if "01" not in skip_set:
        run_step("01_data_preprocessing", test_mode=test)

    # 02: 마할라노비스 파이프라인
    if "02" not in skip_set:
        run_step("02_mahalanobis_pipeline", test_mode=test)

    # 03: Optuna 최적화
    if "03" not in skip_set:
        run_step("03_optuna_optimization", test_mode=test,
                 n_trials=n_trials, n_patience=n_patience)

    # 04: SHAP 분석
    if "04" not in skip_set:
        run_step("04_shap_analysis", test_mode=test)

    total = time.time() - t_total
    print(f"\n{'#'*60}")
    print(f"  ✅ 전체 파이프라인 완료 (총 {total:.1f}초)")
    print(f"  결과 디렉토리: {SANDBOX / 'results'}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
