#!/usr/bin/env python3
"""Per-paper pipeline: analyze (01) then review (02) for each paper before moving to the next."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TASK_ROOT.parent
SCRIPTS_DIR = TASK_ROOT / "scripts"
SCRIPT_01 = SCRIPTS_DIR / "01_analyze_reference_papers.py"
SCRIPT_02 = SCRIPTS_DIR / "02_review_paper_summaries.py"
SCRIPT_06 = SCRIPTS_DIR / "06_update_paper_index_table.py"
DEFAULT_INPUT = REPO_ROOT / "docs" / "ref_papers"
EXIT_ALL_UNAVAILABLE = 75
PROVIDERS = ("codex", "claude", "antigravity")


def _discover_pdfs(input_dir: Path) -> list[Path]:
    """Mirrors 01's discover_pdfs() by importing it directly to guarantee identical ordering."""
    mod_name = "_pipeline_analyze01"
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_01)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod  # required for @dataclass to resolve cls.__module__
    spec.loader.exec_module(mod)
    return mod.discover_pdfs(input_dir)


def _run(cmd: list[str]) -> int:
    return subprocess.run(cmd).returncode


def _passthrough(args: argparse.Namespace) -> list[str]:
    """Build the args list that is forwarded to both 01 and 02 subprocesses."""
    flags: list[str] = ["--provider", args.provider]
    if args.model:
        flags += ["--model", args.model]
    if args.codex_model:
        flags += ["--codex-model", args.codex_model]
    if args.claude_model:
        flags += ["--claude-model", args.claude_model]
    if args.antigravity_model:
        flags += ["--antigravity-model", args.antigravity_model]
    flags += ["--input-dir", str(args.input_dir), "--output-dir", str(args.output_dir)]
    if args.overwrite:
        flags.append("--overwrite")
    if args.retry_failed:
        flags.append("--retry-failed")
    return flags


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=PROVIDERS, default="codex")
    parser.add_argument("--model")
    parser.add_argument("--codex-model")
    parser.add_argument("--claude-model")
    parser.add_argument("--antigravity-model")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=TASK_ROOT)
    parser.add_argument("--resume", action="store_true",
                        help="각 스크립트가 이미 성공한 논문을 건너뜁니다 (기본 동작과 동일, 하위 호환성)")
    parser.add_argument("--overwrite", action="store_true",
                        help="이미 성공한 논문도 재처리합니다")
    parser.add_argument("--only",
                        help="파일명에 이 문자열이 포함된 논문만 처리합니다")
    parser.add_argument("--retry-failed", action="store_true",
                        help="이전 실행에서 실패한 논문을 재시도합니다")
    parser.add_argument("--dry-run", action="store_true",
                        help="실제 실행 없이 처리 대상과 명령어를 출력합니다")
    parser.add_argument("--provider-test-all", action="store_true",
                        help="세 provider 응답 테스트만 실행하고 종료합니다")
    parser.add_argument("--table-only", action="store_true",
                        help="분석·검증 없이 논문 인덱스 표만 갱신합니다")
    parser.add_argument("--skip-provider-gate", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    # --provider-test-all: delegate entirely to 01
    if args.provider_test_all:
        return _run([sys.executable, str(SCRIPT_01), "--provider-test-all"]
                    + _passthrough(args))

    # --table-only: 분석·검증 없이 인덱스 표만 갱신
    if args.table_only:
        cmd = [sys.executable, str(SCRIPT_06), "--provider", args.provider,
               "--output-dir", str(args.output_dir)]
        if args.model:
            cmd += ["--model", args.model]
        if args.overwrite:
            cmd.append("--overwrite")
        return _run(cmd)

    pdfs = _discover_pdfs(args.input_dir)
    if args.only:
        pdfs = [p for p in pdfs if args.only.lower() in p.name.lower()]

    if args.dry_run:
        base = _passthrough(args) + ["--skip-provider-gate", "--resume"]
        print(f"대상 논문: {len(pdfs)}편")
        for pdf in pdfs:
            print(f"\n[{pdf.name}]")
            print("  01:", " ".join(
                [sys.executable, str(SCRIPT_01), "--only", pdf.name] + base))
            print("  02:", " ".join(
                [sys.executable, str(SCRIPT_02), "--only", pdf.name] + base))
        return 0

    # Provider gate (1회)
    if not args.skip_provider_gate:
        print("=== Provider 사전 테스트 ===")
        rc = _run([sys.executable, str(SCRIPT_01), "--provider-test-all"]
                  + _passthrough(args))
        if rc != 0:
            print("Provider 테스트 실패 — 파이프라인을 시작하지 않습니다.", file=sys.stderr)
            return 1
        print()

    # 하위 subprocess에 항상 전달되는 공통 flags
    # --resume: 누적 manifest 보존 (논문 간 상태 유지에 필수)
    # --skip-provider-gate: 위에서 이미 1회 실행했으므로 생략
    sub_base = _passthrough(args) + ["--resume", "--skip-provider-gate"]

    total = len(pdfs)
    succeeded = failed_analyze = failed_review = 0

    print(f"=== 파이프라인 시작: {total}편 ===\n")

    for idx, pdf in enumerate(pdfs, 1):
        print(f"[{idx}/{total}] {pdf.name}")

        # Step 1: 분석
        rc1 = _run([sys.executable, str(SCRIPT_01), "--only", pdf.name] + sub_base)
        if rc1 == EXIT_ALL_UNAVAILABLE:
            print("\n모든 provider 소진 — 파이프라인을 중단합니다.")
            return EXIT_ALL_UNAVAILABLE
        if rc1 != 0:
            print(f"  ✗ 분석 실패 (rc={rc1}) — 검증 건너뜀")
            failed_analyze += 1
            continue

        # Step 2: 검증 (02는 --resume 없이는 review_manifest를 덮어씀)
        rc2 = _run([sys.executable, str(SCRIPT_02), "--only", pdf.name] + sub_base)
        if rc2 == EXIT_ALL_UNAVAILABLE:
            print("\n모든 provider 소진 — 파이프라인을 중단합니다.")
            return EXIT_ALL_UNAVAILABLE
        if rc2 != 0:
            print(f"  ✗ 검증 실패 (rc={rc2})")
            failed_review += 1
        else:
            succeeded += 1

    print(f"\n=== 완료 ===")
    print(f"성공: {succeeded}편 | 분석 실패: {failed_analyze}편 | 검증 실패: {failed_review}편")
    return 0 if (failed_analyze + failed_review) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
