"""
전체 파이프라인 진입점.

단계:
  1. (선택) cycle waveform 추출 — cycle_waveforms_101.parquet 및 stride_raw_waveforms.parquet 생성
  2. 데이터 검증 — verify_data.py
  3. 모델 학습 — orchestrator.py
  4. Gait Normality Score 계산
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
    p.add_argument("--version",        default=None,
                   help="실험 버전명 (config 기본값 사용 시 생략, 예: v2)")
    p.add_argument("--models",         nargs="+", default=["all"])
    p.add_argument("--target",         choices=["binary", "multiclass"], default=None)
    p.add_argument("--waveform_type",  choices=["norm_101", "cycle_norm_101", "raw_padded"], default=None)
    p.add_argument("--waveform_norm",  choices=["none", "zscore", "ha_centered"], default=None)
    p.add_argument("--feature_select", default=None)
    p.add_argument("--skip-extract",   action="store_true",
                   help="cycle/raw waveform 추출 단계 건너뜀 (parquet 이미 존재 시)")
    p.add_argument("--skip-verify",    action="store_true",
                   help="데이터 검증 단계 건너뜀")
    p.add_argument("--skip-train",     action="store_true",
                   help="모델 학습 건너뜀")
    p.add_argument("--skip-scoring", "--skip-recovery", dest="skip_scoring", action="store_true",
                   help="Gait Normality Score 계산 건너뜀 (--skip-recovery 호환)")
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
    if not data_root.is_absolute():
        data_root = (ML / data_root).resolve()
    cycle_path = data_root / cfg.data.get("cycle_waveforms_101", "cycle_waveforms_101.parquet")
    raw_path = data_root / cfg.data.raw_cycles

    if skip:
        missing = [p.name for p in (cycle_path, raw_path) if not p.exists()]
        if missing:
            print(f"[pipeline] 경고: --skip-extract 지정됐지만 누락 파일 있음: {missing}")
        else:
            print(f"[pipeline] cycle/raw waveform 추출 건너뜀 ({cycle_path.name}, {raw_path.name} 존재)")
        return

    if cycle_path.exists():
        print(f"[pipeline] cycle-level 101pt 이미 존재: {cycle_path}")
    else:
        print("[pipeline] Step 1a: cycle-level 101pt 추출 시작")
        from features.extract_cycle_waveforms_101 import extract as extract_cycle_101
        extract_cycle_101(cfg)
        print(f"[pipeline] Step 1a 완료: {cycle_path}")

    if raw_path.exists():
        print(f"[pipeline] raw padded cycle 이미 존재: {raw_path}")
        return

    print("[pipeline] Step 1b: raw padded cycle 추출 시작")
    from features.extract_raw_cycles import extract as extract_raw
    extract_raw(cfg)
    print(f"[pipeline] Step 1b 완료: {raw_path}")


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


def step_scoring(cfg_path: str, version: str = "v1", test_mode: bool = False) -> None:
    print("[pipeline] Step 4: Gait Normality Score 계산")
    from run_gait_normality_scoring import run_from_config

    paths = run_from_config(
        config_path=cfg_path,
        version=version,
        cv_repeats=1 if test_mode else None,
        bootstrap=20 if test_mode else None,
    )
    print(f"[pipeline] Gait Normality Score 저장: {paths['cohort_scores']}")
    print(f"[pipeline] Gait Normality report: {paths['report']}")


def step_report(version: str = "v1") -> None:
    print("[pipeline] Step 5: 리포트 생성")
    from reports.generate_report import generate
    path = generate(version=version)
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

    # 버전 결정: CLI 플래그 > config 기본값
    from omegaconf import OmegaConf
    _cfg = OmegaConf.load(args.config)
    version = args.version or _cfg.get("version", "v1")
    args.version = version  # orchestrator에 그대로 전달

    print(f"[pipeline] 버전: {version}")

    _run_step("extract",  lambda: step_extract(args.config, args.skip_extract),  test_log)
    _run_step("verify",   lambda: step_verify(args.config, args.skip_verify),     test_log)

    if not args.skip_train:
        _run_step("train", lambda: step_train(args), test_log)

    if not args.skip_scoring:
        _run_step("scoring", lambda: step_scoring(args.config, version, test_mode=is_test), test_log)

    _run_step("report", lambda: step_report(version), test_log)

    # 결과 파일 저장
    from train.logger import save_test_result
    save_test_result(
        test_type="pipeline",
        results=test_log,
        mode="test" if is_test else "full",
        extra={"models": args.models, "config": args.config, "version": version},
        version=version,
    )

    passed = sum(1 for r in test_log if r["status"] == "PASS")
    print(f"\n[pipeline] 완료 — {passed}/{len(test_log)} 단계 성공.")


if __name__ == "__main__":
    main()
