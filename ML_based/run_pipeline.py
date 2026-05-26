"""
전체 파이프라인 진입점.

단계:
  1. (선택) raw cycle 추출 — stride_raw_waveforms.parquet 생성
  2. 데이터 검증 — verify_data.py
  3. 모델 학습 — orchestrator.py
  4. Recovery Score 계산
  5. 리포트 생성

사용 예:
    python run_pipeline.py                         # 기본 설정 전체 실행
    python run_pipeline.py --skip-extract          # raw cycle 추출 건너뜀
    python run_pipeline.py --models logreg rf      # 지정 모델만 학습
    python run_pipeline.py --skip-train --report   # 리포트만 재생성
"""
from __future__ import annotations
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ML = Path(__file__).parent
sys.path.insert(0, str(ML))


def parse_args():
    p = argparse.ArgumentParser(description="ACL Gait ML Full Pipeline")
    p.add_argument("--config",         default=str(ML / "configs" / "config.yaml"))
    p.add_argument("--models",         nargs="+", default=["all"])
    p.add_argument("--target",         choices=["binary", "multiclass"], default=None)
    p.add_argument("--waveform_type",  choices=["norm_101", "raw_padded"], default=None)
    p.add_argument("--waveform_norm",  choices=["none", "zscore", "ha_centered"], default=None)
    p.add_argument("--feature_select", default=None)
    p.add_argument("--skip-extract",   action="store_true",
                   help="raw cycle 추출 단계 건너뜀 (parquet 이미 존재 시)")
    p.add_argument("--skip-verify",    action="store_true",
                   help="데이터 검증 단계 건너뜀")
    p.add_argument("--skip-train",     action="store_true",
                   help="모델 학습 건너뜀")
    p.add_argument("--skip-recovery",  action="store_true",
                   help="Recovery Score 계산 건너뜀")
    p.add_argument("--report",         action="store_true",
                   help="--skip-train과 함께 사용 시 리포트만 재생성")
    p.add_argument("--test",           action="store_true",
                   help="테스트 모드: 극소량 데이터·최소 epoch으로 전체 파이프라인 실행 확인")
    p.add_argument("--skip-transformer", action="store_true",
                   help="Transformer 모델 건너뜀 (MPS 이슈 등)")
    return p.parse_args()


def step_extract(cfg_path: str, skip: bool) -> None:
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(cfg_path)
    data_root = Path(cfg.data.root)
    out_path  = data_root / "stride_raw_waveforms.parquet"

    if skip:
        if out_path.exists():
            print(f"[pipeline] raw cycle 추출 건너뜀 ({out_path.name} 존재)")
        else:
            print(f"[pipeline] 경고: --skip-extract 지정됐지만 {out_path} 없음")
        return

    if out_path.exists():
        print(f"[pipeline] raw cycle 이미 존재: {out_path}")
        return

    print("[pipeline] Step 1: raw cycle 추출 시작")
    from features.extract_raw_cycles import extract
    extract(cfg)
    print(f"[pipeline] Step 1 완료: {out_path}")


def step_verify(cfg_path: str, skip: bool) -> None:
    if skip:
        print("[pipeline] 데이터 검증 건너뜀")
        return
    print("[pipeline] Step 2: 데이터 검증")
    from features.verify_data import run_verification
    from omegaconf import OmegaConf
    run_verification(OmegaConf.load(cfg_path))


def step_train(args) -> dict:
    print("[pipeline] Step 3: 모델 학습 시작")
    from orchestrator import run, parse_args as orch_parse
    results = run(args)
    print("[pipeline] Step 3 완료")
    return results


def step_recovery(cfg_path: str) -> None:
    print("[pipeline] Step 4: Recovery Score 계산")
    from omegaconf import OmegaConf
    import pandas as pd
    import numpy as np

    cfg = OmegaConf.load(cfg_path)
    data_root = Path(cfg.data.root)

    features_path = data_root / cfg.data.scalar_subject
    if not features_path.exists():
        print(f"[pipeline] Recovery Score: {features_path} 없음, 건너뜀")
        return

    from recovery_score.scorer import RecoveryScorer

    df = pd.read_csv(features_path)
    groups      = df["group"].values
    subject_ids = df["subject_id"].values

    scorer  = RecoveryScorer()
    result  = scorer.compute(df, groups, subject_ids)

    out_path = data_root.parent.parent / "ML_based" / "artifacts" / "recovery_scores.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"[pipeline] Recovery Score 저장: {out_path}")


def step_report() -> None:
    print("[pipeline] Step 5: 리포트 생성")
    from reports.generate_report import generate
    path = generate()
    print(f"[pipeline] 리포트: {path}")


def _run_step(name: str, fn, test_log: list) -> bool:
    """단계 실행 후 결과를 test_log에 기록. 성공 여부 반환."""
    import time, traceback as _tb
    t0 = time.time()
    try:
        fn()
        elapsed = time.time() - t0
        test_log.append({"name": name, "status": "PASS",
                         "elapsed_sec": round(elapsed, 2), "error": None})
        return True
    except Exception:
        elapsed = time.time() - t0
        err = _tb.format_exc()
        print(f"[pipeline] {name} 실패:\n{err}")
        test_log.append({"name": name, "status": "FAIL",
                         "elapsed_sec": round(elapsed, 2), "error": err})
        return False


def main():
    args = parse_args()
    test_log: list[dict] = []
    is_test = getattr(args, "test", False)

    _run_step("extract",  lambda: step_extract(args.config, args.skip_extract),  test_log)
    _run_step("verify",   lambda: step_verify(args.config, args.skip_verify),     test_log)

    if not args.skip_train:
        _run_step("train", lambda: step_train(args), test_log)

    if not args.skip_recovery:
        _run_step("recovery", lambda: step_recovery(args.config), test_log)

    _run_step("report", step_report, test_log)

    # 결과 파일 저장
    from train.logger import save_test_result
    save_test_result(
        test_type="pipeline",
        results=test_log,
        mode="test" if is_test else "full",
        extra={"models": args.models, "config": args.config},
    )

    passed = sum(1 for r in test_log if r["status"] == "PASS")
    print(f"\n[pipeline] 완료 — {passed}/{len(test_log)} 단계 성공.")


if __name__ == "__main__":
    main()
