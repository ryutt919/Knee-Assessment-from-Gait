#!/usr/bin/env python3
"""Sequential multi-CLI fact-checker comparing existing paper summaries against source PDFs.

Mirrors the CLI-invocation logic of 01_analyze_reference_papers.py (provider
failover across codex/claude/antigravity, identical command templates, manifest
+ resume/retry pattern), but the task is reversed: instead of producing a new
summary, it cross-checks an already-produced summary MD against the original
PDF text and reports an exception list of factual discrepancies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

TASK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_ROOT.parent
DEFAULT_INPUT = REPO_ROOT / "docs" / "ref_papers"
SCHEMA_PATH = TASK_ROOT / "schemas" / "02_summary_factcheck.schema.json"
PROMPT_PATH = TASK_ROOT / "prompts" / "02_summary_factcheck_prompt.md"
ANALYSIS_MANIFEST_PATH = TASK_ROOT / "logs" / "manifest.json"
PROVIDERS = ("codex", "claude", "antigravity")
EXIT_ALL_UNAVAILABLE = 75

SECTION_LABELS = {
    "bibliographic_info": "서지정보",
    "purpose": "연구 목적",
    "design_and_participants": "연구 설계와 대상",
    "methods": "방법",
    "key_results": "핵심 결과",
    "author_conclusion": "저자 결론",
    "limitations": "연구의 한계",
    "reviewer_thoughts": "생각해볼 내용",
    "prior_research_problems": "선행연구의 문제점",
    "study_solution_and_contribution": "연구의 해결 방식과 기여",
    "referenceable_claims": "레퍼런스할 수 있는 내용",
}

UNAVAILABLE_PATTERNS = re.compile(
    r"usage limit|quota (?:exhaust|exceed)|insufficient (?:quota|credit)|"
    r"credit(?:s)? (?:exhaust|deplet)|rate_limit_reached|plan limit|"
    r"authentication (?:failed|required)|not logged in|unauthorized|forbidden|"
    r"invalid api key|token expired|permission denied",
    re.I,
)
TRANSIENT_PATTERNS = re.compile(
    r"\b429\b|too many requests|\b5(?:00|02|03|04|29)\b|overloaded|"
    r"temporar(?:y|ily)|timed? out|timeout|connection reset|service unavailable",
    re.I,
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(path: Path) -> str:
    stem = re.sub(r"[^0-9A-Za-z가-힣]+", "_", path.stem).strip("_")
    return stem[:120] or hashlib.sha256(str(path).encode()).hexdigest()[:16]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as fh:
        json.dump(value, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
        tmp = Path(fh.name)
    tmp.replace(path)


def inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def provider_order(start: str) -> list[str]:
    idx = PROVIDERS.index(start)
    return list(PROVIDERS[idx:] + PROVIDERS[:idx])


def find_binary(provider: str) -> str | None:
    env_key = f"PAPER_ANALYZER_{provider.upper()}_BIN"
    if os.environ.get(env_key):
        return os.environ[env_key]
    names = {"codex": ["codex"], "claude": ["claude"], "antigravity": ["agy"]}[provider]
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    if provider == "antigravity":
        fallback = Path.home() / ".local" / "bin" / "agy"
        if fallback.is_file() and os.access(fallback, os.X_OK):
            return str(fallback)
    return None


def classify_failure(text: str, returncode: int) -> str:
    if UNAVAILABLE_PATTERNS.search(text):
        return "unavailable"
    if TRANSIENT_PATTERNS.search(text) or returncode in (124, 137, 143):
        return "transient"
    return "technical_failure"


def unavailable_status(text: str) -> str:
    if re.search(r"authentication|not logged in|unauthorized|forbidden|invalid api key|token expired|permission denied", text, re.I):
        return "auth_failed"
    return "exhausted"


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    text = re.sub(r'\\x([0-9a-fA-F]{2})', r'\\u00\1', text)
    text = re.sub(r'"\s*/\s*"', ' / ', text)
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
            if isinstance(value, dict) and "title" in value:
                return value
        except json.JSONDecodeError:
            continue
    raise ValueError("응답에서 JSON 객체를 찾지 못했습니다")


def validate_shape_review(data: dict[str, Any], provider: str) -> list[str]:
    required = [
        "provider", "title", "source_pdf", "source_summary_path",
        "sections_checked", "issues_found", "overall_verdict",
        "overall_verdict_reason", "findings", "unverifiable_items",
    ]
    errors = [f"누락 필드: {key}" for key in required if key not in data]
    if data.get("provider") != provider:
        errors.append(f"provider 불일치: {data.get('provider')} != {provider}")
    for key in ("findings", "unverifiable_items"):
        if key in data and not isinstance(data[key], list):
            errors.append(f"{key}는 배열이어야 합니다")
    for i, item in enumerate(data.get("findings", [])):
        required_keys = (
            "section", "quoted_summary_text", "issue_type", "severity",
            "explanation", "source_evidence_quote", "source_locator", "suggested_correction",
        )
        if not isinstance(item, dict) or not all(k in item for k in required_keys):
            errors.append(f"findings {i}는 필수 키를 모두 가져야 합니다")
    for i, item in enumerate(data.get("unverifiable_items", [])):
        if not isinstance(item, dict) or not all(k in item for k in ("quoted_summary_text", "reason")):
            errors.append(f"unverifiable_items {i}는 quoted_summary_text/reason을 가져야 합니다")
    if isinstance(data.get("findings"), list) and isinstance(data.get("issues_found"), int):
        if data["issues_found"] != len(data["findings"]):
            errors.append(f"issues_found({data['issues_found']})가 findings 길이({len(data['findings'])})와 다릅니다")
    return errors


def comparison_normalized(text: str) -> str:
    """Normalize PDF/MD line wrapping without weakening word-sequence checks."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"--- page \d+ ---\s*\d*", "", text)
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    return "".join(ch for ch in text if ch.isalnum())


def validate_review_quotes(data: dict[str, Any], source_text: str, summary_text: str) -> list[str]:
    source_haystack = comparison_normalized(source_text)
    summary_haystack = comparison_normalized(summary_text)
    errors = []
    for i, item in enumerate(data.get("findings", []), 1):
        quoted = comparison_normalized(item.get("quoted_summary_text", ""))
        if not quoted or quoted not in summary_haystack:
            errors.append(f"findings {i} quoted_summary_text가 기존 요약 MD에 없음")
        evidence = comparison_normalized(item.get("source_evidence_quote", ""))
        if not evidence or evidence not in source_haystack:
            errors.append(f"findings {i} source_evidence_quote가 원문 추출 텍스트에 없음")
        if not re.search(r"page\s*\d+|p\.?\s*\d+|페이지\s*\d+", item.get("source_locator", ""), re.I):
            errors.append(f"findings {i} 페이지 위치 누락")
    for i, item in enumerate(data.get("unverifiable_items", []), 1):
        quoted = comparison_normalized(item.get("quoted_summary_text", ""))
        if not quoted or quoted not in summary_haystack:
            errors.append(f"unverifiable_items {i} quoted_summary_text가 기존 요약 MD에 없음")
    return errors


def sanitize_unverified_findings(data: dict[str, Any], source_text: str, summary_text: str) -> list[str]:
    """Drop only the finding units that cannot be proven, keep the rest."""
    dropped: list[str] = []
    kept = []
    for i, item in enumerate(data.get("findings", []), 1):
        probe = {"findings": [item], "unverifiable_items": []}
        if validate_review_quotes(probe, source_text, summary_text):
            dropped.append(f"findings {i}: {item.get('quoted_summary_text', '')[:80]}")
        else:
            kept.append(item)
    data["findings"] = kept
    kept_unverifiable = []
    for i, item in enumerate(data.get("unverifiable_items", []), 1):
        probe = {"findings": [], "unverifiable_items": [item]}
        if validate_review_quotes(probe, source_text, summary_text):
            dropped.append(f"unverifiable_items {i}: {item.get('quoted_summary_text', '')[:80]}")
        else:
            kept_unverifiable.append(item)
    data["unverifiable_items"] = kept_unverifiable
    data["issues_found"] = len(data["findings"])
    return dropped


def discover_pdfs(input_dir: Path) -> list[Path]:
    """Identical discovery/ordering rule as 01_analyze_reference_papers.py so paper
    numbering stays consistent between summaries and reviews."""
    if input_dir.resolve() == DEFAULT_INPUT.resolve():
        result = []
        for child in sorted(input_dir.iterdir()):
            if child.is_dir() and re.match(r"^0[1-6]_", child.name):
                result.extend(sorted(child.glob("*.pdf")))
        return result
    return sorted(input_dir.rglob("*.pdf"))


@dataclass
class ReviewTarget:
    number: int
    pdf: Path
    summary_path: Path


def discover_review_targets(input_dir: Path) -> list[ReviewTarget]:
    if not ANALYSIS_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"01단계 분석 manifest를 찾을 수 없습니다: {ANALYSIS_MANIFEST_PATH}")
    analysis_manifest = json.loads(ANALYSIS_MANIFEST_PATH.read_text(encoding="utf-8"))
    pdfs = discover_pdfs(input_dir)
    global_index = {str(p.resolve()): i for i, p in enumerate(pdfs, 1)}
    targets: list[ReviewTarget] = []
    for key, record in sorted(analysis_manifest.get("papers", {}).items()):
        if record.get("status") != "success":
            continue
        pdf = Path(key)
        summary_path = Path(record["output_path"])
        if not pdf.exists() or not summary_path.exists():
            continue
        number = global_index.get(str(pdf.resolve()))
        if number is None:
            continue
        targets.append(ReviewTarget(number=number, pdf=pdf, summary_path=summary_path))
    return sorted(targets, key=lambda t: t.number)


def extract_pdf(pdf: Path, cache_dir: Path) -> Path:
    """Same cache key convention as 01_analyze_reference_papers.py, so the PDF
    extraction it already produced is reused here without recomputation."""
    cache_dir = cache_dir / f"{safe_name(pdf)}_{sha256(pdf)[:12]}_raw_v2"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / "paper.txt"
    if out.exists() and out.stat().st_size > 0:
        return out
    raw = subprocess.run(["pdftotext", "-raw", str(pdf), "-"], check=True, capture_output=True).stdout
    pages = raw.decode("utf-8", errors="replace").split("\f")
    rendered = "\n".join(f"\n--- PAGE {i} ---\n{page}" for i, page in enumerate(pages, 1))
    out.write_text(rendered, encoding="utf-8")
    return out


def render_prompt(provider: str, pdf: Path, summary_path: Path, source_text: str, summary_text: str) -> str:
    base = PROMPT_PATH.read_text(encoding="utf-8")
    request = base.format(provider=provider, pdf_path=pdf, summary_path=summary_path)
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    return (
        request
        + "\n\n반드시 아래 JSON_SCHEMA의 키 이름과 배열/객체 구조를 정확히 사용하라. 다른 키 이름을 만들지 마라.\n"
        + "<JSON_SCHEMA>\n" + schema + "\n</JSON_SCHEMA>\n"
        + "\n\n아래 SOURCE_TEXT만 원문 근거로 사용하라. 도구를 사용하거나 다른 파일·웹을 조회하지 마라.\n"
        + "<SOURCE_TEXT>\n" + source_text + "\n</SOURCE_TEXT>\n"
        + "\n\n아래 EXISTING_SUMMARY를 검증 대상으로 사용하라.\n"
        + "<EXISTING_SUMMARY>\n" + summary_text + "\n</EXISTING_SUMMARY>\n"
    )


@dataclass
class Invocation:
    ok: bool
    provider: str
    data: dict[str, Any] | None = None
    error: str = ""
    category: str = ""
    returncode: int = 0
    elapsed: float = 0.0
    output: str = ""


def build_command(provider: str, prompt: str, result_path: Path, model: str | None) -> tuple[list[str], str | None]:
    binary = find_binary(provider)
    if not binary:
        raise FileNotFoundError(f"{provider} 실행 파일을 찾을 수 없습니다")
    if provider == "codex":
        cmd = [binary, "exec", "--ephemeral", "--sandbox", "read-only", "--json", "--output-schema", str(SCHEMA_PATH), "-o", str(result_path), "-C", str(TASK_ROOT)]
        if model:
            cmd += ["--model", model]
        cmd += ["-"]
        return cmd, prompt
    if provider == "claude":
        schema = json.dumps(json.loads(SCHEMA_PATH.read_text()), separators=(",", ":"))
        cmd = [binary, "--print", "--no-session-persistence", "--permission-mode", "plan", "--output-format", "stream-json", "--verbose", "--json-schema", schema]
        if model:
            cmd += ["--model", model]
        return cmd, prompt
    cmd = [binary, "--print", "--sandbox", "--print-timeout", "10m"]
    if model:
        cmd += ["--model", model]
    return cmd, prompt


def parse_provider_output(provider: str, stdout: str, result_path: Path) -> dict[str, Any]:
    if provider == "codex" and result_path.exists():
        return extract_json(result_path.read_text(encoding="utf-8"))
    if provider == "claude":
        for line in reversed(stdout.splitlines()):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "result":
                if isinstance(event.get("structured_output"), dict):
                    return event["structured_output"]
                return extract_json(str(event.get("result", "")))
    return extract_json(stdout)


def invoke(provider: str, prompt: str, logs_dir: Path, run_key: str, model: str | None, timeout: int = 900, workspace: Path | None = None) -> Invocation:
    logs_dir.mkdir(parents=True, exist_ok=True)
    result_path = logs_dir / f"{run_key}_{provider}_result.json"
    raw_path = logs_dir / f"{run_key}_{provider}.log"
    try:
        cmd, stdin_text = build_command(provider, prompt, result_path, model)
    except FileNotFoundError as exc:
        return Invocation(False, provider, error=str(exc), category="missing", returncode=127)
    started = time.monotonic()
    try:
        proc = subprocess.run(cmd, input=stdin_text, text=True, capture_output=True, timeout=timeout, cwd=workspace or TASK_ROOT)
        elapsed = time.monotonic() - started
        combined = proc.stdout + ("\nSTDERR:\n" + proc.stderr if proc.stderr else "")
        raw_path.write_text(combined, encoding="utf-8")
        if proc.returncode:
            return Invocation(False, provider, error=combined[-4000:], category=classify_failure(combined, proc.returncode), returncode=proc.returncode, elapsed=elapsed, output=combined)
        try:
            data = parse_provider_output(provider, proc.stdout, result_path)
        except Exception as exc:
            return Invocation(False, provider, error=str(exc), category="invalid_output", elapsed=elapsed, output=combined)
        errors = validate_shape_review(data, provider)
        if errors:
            return Invocation(False, provider, error="; ".join(errors), category="invalid_output", elapsed=elapsed, output=combined)
        return Invocation(True, provider, data=data, elapsed=elapsed, output=combined)
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started
        msg = f"timeout after {timeout}s\n{exc.stdout or ''}\n{exc.stderr or ''}"
        raw_path.write_text(msg, encoding="utf-8")
        return Invocation(False, provider, error=msg, category="transient", returncode=124, elapsed=elapsed)


def minimal_fixture(provider: str) -> dict[str, Any]:
    return {
        "provider": provider,
        "title": "테스트 논문",
        "source_pdf": "fixture.pdf",
        "source_summary_path": "fixture_summary.md",
        "sections_checked": 1,
        "issues_found": 0,
        "overall_verdict": "신뢰 가능",
        "overall_verdict_reason": "테스트 픽스처는 항상 통과",
        "findings": [],
        "unverifiable_items": [],
    }


def provider_test_prompt(provider: str) -> str:
    fixture = minimal_fixture(provider)
    return "도구를 절대 사용하지 마라. 파일을 검색하거나 읽지 마라. 다음 JSON 객체를 변경하거나 설명하지 말고 JSON 하나로만 즉시 그대로 반환하라.\n" + json.dumps(fixture, ensure_ascii=False)


def run_provider_test(provider: str, logs_dir: Path, model: str | None) -> Invocation:
    workspace = logs_dir / "provider_test_workspace" / provider
    workspace.mkdir(parents=True, exist_ok=True)
    return invoke(provider, provider_test_prompt(provider), logs_dir, "review_provider_test", model, timeout=180, workspace=workspace)


def model_for(provider: str, args: argparse.Namespace) -> str | None:
    specific = getattr(args, f"{provider}_model", None)
    if specific:
        return specific
    return args.model if provider == args.provider else None


def render_review(data: dict[str, Any]) -> str:
    parts = [
        f"# 사실검증: {data['title']}", "",
        f"- 원본 PDF: {data['source_pdf']}",
        f"- 검증 대상 요약: {data['source_summary_path']}",
        f"- 검증 provider: {data['provider']}",
        f"- 검토 항목 수: {data['sections_checked']}",
        f"- 발견된 문제 수: {data['issues_found']}",
        f"- 전체 판정: **{data['overall_verdict']}**",
        f"- 판정 근거: {data['overall_verdict_reason']}",
        "",
    ]
    findings = data.get("findings", [])
    parts += ["## 발견된 문제", ""]
    if not findings:
        parts += ["- 문제로 분류된 항목 없음", ""]
    for i, item in enumerate(findings, 1):
        section_label = SECTION_LABELS.get(item["section"], item["section"])
        parts += [
            f"### {i}. [{section_label}] {item['issue_type']} ({item['severity']})", "",
            f"- 요약 문장: “{item['quoted_summary_text']}”",
            f"- 설명: {item['explanation']}",
            f"- 원문 근거: “{item['source_evidence_quote']}” ({item['source_locator']})",
            f"- 수정 제안: {item['suggested_correction']}", "",
        ]
    unverifiable = data.get("unverifiable_items", [])
    if unverifiable:
        parts += ["## 원문에서 확인 불가능한 항목", ""]
        for item in unverifiable:
            parts += [f"- “{item['quoted_summary_text']}” — {item['reason']}"]
        parts += [""]
    return "\n".join(parts).rstrip() + "\n"


def aggregate_reviews(records: dict[str, Any], output_root: Path) -> None:
    successful = [v for _, v in sorted(records.items(), key=lambda kv: kv[1].get("number", 0)) if v.get("status") == "success"]
    lines = ["# 논문 요약 사실검증 종합 리포트", "", f"- 검증 완료: {len(successful)}편", ""]
    verdict_counts: dict[str, int] = {}
    for rec in successful:
        verdict_counts[rec["data"]["overall_verdict"]] = verdict_counts.get(rec["data"]["overall_verdict"], 0) + 1
    lines += ["## 판정 분포", ""]
    for verdict in ("신뢰 가능", "일부 수정 필요", "신뢰 어려움"):
        lines.append(f"- {verdict}: {verdict_counts.get(verdict, 0)}편")
    lines += ["", "## 논문별 요약", ""]
    for rec in successful:
        d = rec["data"]
        lines.append(f"- [{rec['number']:02d}] {d['title']} — **{d['overall_verdict']}** (문제 {d['issues_found']}건) → `{rec['review_path']}`")
    lines += ["", "## 문제가 발견된 논문 상세", ""]
    flagged = [rec for rec in successful if rec["data"]["issues_found"] > 0]
    if not flagged:
        lines += ["- 문제 발견 없음", ""]
    for rec in flagged:
        review_path = Path(rec["review_path"])
        lines += [review_path.read_text(encoding="utf-8").rstrip(), "", "---", ""]
    (output_root / "mds").mkdir(parents=True, exist_ok=True)
    (output_root / "mds" / "03_summary_factcheck_report.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def next_handoff_path(output_root: Path) -> Path:
    folder = output_root / "handoffs"
    folder.mkdir(parents=True, exist_ok=True)
    number = len(list(folder.glob("[0-9][0-9]_review_provider_exhaustion_*.md"))) + 1
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return folder / f"{number:02d}_review_provider_exhaustion_{stamp}.md"


def write_handoff(output_root: Path, manifest: dict[str, Any], current: str, command: str) -> Path:
    path = next_handoff_path(output_root)
    records = manifest.get("papers", {})
    counts = {s: sum(r.get("status") == s for r in records.values()) for s in ("success", "failed")}
    counts["pending"] = max(0, len(manifest.get("inventory", records)) - counts["success"] - counts["failed"])
    states = manifest.get("provider_states", {})
    lines = [
        "# 사실검증 Provider 사용 불가 Handoff", "", f"- 실행 ID: {manifest['run_id']}", f"- 중단 시각: {now_iso()}",
        f"- 현재 논문: {current}", f"- 완료: {counts['success']}", f"- 실패: {counts['failed']}", f"- 대기: {counts['pending']}", "",
        "## Provider 상태", "",
    ]
    for provider in PROVIDERS:
        state = states.get(provider, {})
        lines += [f"### {provider}", f"- 상태: {state.get('status', 'unknown')}", f"- 오류: {state.get('error', '없음')}", ""]
    lines += ["## 재개", "", f"```bash\n{command}\n```", "", "review_manifest와 완료된 MD는 보존되어 있으며 재개 시 성공 항목을 건너뜁니다.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def dashboard(index: int, total: int, provider: str, paper: str, states: dict[str, Any], started: float, message: str) -> None:
    width = 28
    done = int(width * index / max(total, 1))
    bar = "█" * done + "░" * (width - done)
    state_text = " | ".join(f"{p}:{states.get(p, {}).get('status', 'ready')}" for p in PROVIDERS)
    print(f"\r[{bar}] {index}/{total} | {provider:<11} | {time.monotonic()-started:7.1f}s | {paper[:55]:55} | {message[:50]:50}", end="", flush=True)
    if message.startswith(("완료", "실패", "전환", "중단")):
        print(f"\n  {state_text}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=PROVIDERS, default="codex")
    parser.add_argument("--model")
    parser.add_argument("--codex-model")
    parser.add_argument("--claude-model")
    parser.add_argument("--antigravity-model")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=TASK_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--provider-test", action="store_true")
    parser.add_argument("--provider-test-all", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--only")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--skip-provider-gate", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    output_root = args.output_dir.resolve()
    if not inside(output_root, TASK_ROOT):
        parser.error(f"output-dir는 샌드박스 내부여야 합니다: {TASK_ROOT}")

    targets = discover_review_targets(args.input_dir.resolve())
    if args.only:
        targets = [t for t in targets if args.only.lower() in t.pdf.name.lower()]
    binaries = {p: find_binary(p) for p in PROVIDERS}
    print(f"검증 대상: {len(targets)}편 (01단계 manifest 기준 성공한 요약만 포함)")
    for p in PROVIDERS:
        print(f"- {p}: {binaries[p] or 'MISSING'}")
    if args.dry_run:
        print("전환 순서:", " → ".join(provider_order(args.provider)))
        for t in targets:
            print(f"[{t.number:02d}] {t.pdf} <-> {t.summary_path}")
        return 0 if all(binaries.values()) else 2

    logs_dir = output_root / "logs" / "review"
    if args.provider_test or args.provider_test_all:
        test_targets = PROVIDERS if args.provider_test_all else (args.provider,)
        failed = False
        for provider in test_targets:
            print(f"[{provider}] 실제 provider 테스트...", flush=True)
            result = run_provider_test(provider, logs_dir, model_for(provider, args))
            print(f"[{provider}] {'PASS' if result.ok else 'FAIL'} ({result.elapsed:.1f}s) {result.error[:300]}")
            failed |= not result.ok
        return 1 if failed else 0

    if not args.skip_provider_gate:
        gate_results = {p: run_provider_test(p, logs_dir, model_for(p, args)) for p in PROVIDERS}
        if not all(r.ok for r in gate_results.values()):
            print("세 provider 사전 테스트가 모두 통과하지 못해 검증을 시작하지 않습니다.", file=sys.stderr)
            return 1

    manifest_path = logs_dir / "review_manifest.json"
    if args.resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["inventory"] = [str(t.pdf.resolve()) for t in targets]
        manifest["prompt_hash"] = sha256(PROMPT_PATH)
        manifest["schema_hash"] = sha256(SCHEMA_PATH)
    else:
        manifest = {
            "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "created_at": now_iso(),
            "inventory": [str(t.pdf.resolve()) for t in targets],
            "prompt_hash": sha256(PROMPT_PATH), "schema_hash": sha256(SCHEMA_PATH),
            "provider_states": {p: {"status": "available", "error": ""} for p in PROVIDERS},
            "papers": {},
        }
    records = manifest["papers"]
    states = manifest["provider_states"]
    started = time.monotonic()

    for idx, target in enumerate(targets, 1):
        key = str(target.pdf.resolve())
        existing = records.get(key, {})
        if existing.get("status") == "success" and not args.overwrite:
            dashboard(idx, len(targets), existing.get("provider", "skip"), target.pdf.name, states, started, "완료: resume skip")
            continue
        if existing.get("status") == "failed" and not (args.retry_failed or args.overwrite or args.resume):
            continue
        records[key] = {"status": "pending", "number": target.number, "pdf_hash": sha256(target.pdf), "attempts": [], "updated_at": now_iso()}
        atomic_json(manifest_path, manifest)
        text_path = extract_pdf(target.pdf, output_root / "logs" / "cache")
        source_text = text_path.read_text(encoding="utf-8", errors="replace")
        summary_text = target.summary_path.read_text(encoding="utf-8", errors="replace")
        success = False
        errors: list[str] = []
        for provider in provider_order(args.provider):
            if states.get(provider, {}).get("status") != "available":
                continue
            dashboard(idx - 1, len(targets), provider, target.pdf.name, states, started, "검증 중")
            prompt = render_prompt(provider, target.pdf, target.summary_path, source_text, summary_text)
            attempt: Invocation | None = None
            for retry in range(3):
                attempt = invoke(provider, prompt, logs_dir / manifest["run_id"], safe_name(target.pdf), model_for(provider, args), workspace=text_path.parent)
                records[key]["attempts"].append({"provider": provider, "ok": attempt.ok, "category": attempt.category, "error": attempt.error[:2000], "elapsed": attempt.elapsed, "at": now_iso()})
                atomic_json(manifest_path, manifest)
                if attempt.ok:
                    break
                if attempt.category != "transient" or retry == 2:
                    break
                wait = (10, 30)[retry]
                dashboard(idx - 1, len(targets), provider, target.pdf.name, states, started, f"재시도 대기 {wait}s")
                time.sleep(wait)
            assert attempt is not None
            if attempt.ok and attempt.data is not None:
                quote_errors = validate_review_quotes(attempt.data, source_text, summary_text)
                if quote_errors:
                    attempt.ok = False
                    attempt.category = "invalid_output"
                    attempt.error = "; ".join(quote_errors)
            if not attempt.ok and attempt.category == "invalid_output":
                correction = (
                    prompt
                    + "\n\n이전 응답은 다음 자동 검증에 실패했다: "
                    + attempt.error
                    + "\nSOURCE_TEXT와 EXISTING_SUMMARY를 다시 확인하고 인용문을 임의로 고치거나 만들지 말고, 검증 가능한 항목만 포함해 JSON 전체를 다시 반환하라."
                )
                corrected = invoke(
                    provider, correction, logs_dir / manifest["run_id"],
                    safe_name(target.pdf) + "_correction", model_for(provider, args),
                    workspace=text_path.parent,
                )
                records[key]["attempts"].append({
                    "provider": provider, "ok": corrected.ok,
                    "category": corrected.category, "error": corrected.error[:2000],
                    "elapsed": corrected.elapsed, "at": now_iso(), "correction": True,
                })
                if corrected.ok and corrected.data is not None:
                    correction_errors = validate_review_quotes(corrected.data, source_text, summary_text)
                    if correction_errors:
                        dropped = sanitize_unverified_findings(corrected.data, source_text, summary_text)
                        remaining = validate_review_quotes(corrected.data, source_text, summary_text)
                        if remaining:
                            corrected.ok = False
                            corrected.category = "invalid_output"
                            corrected.error = "; ".join(remaining)
                        else:
                            records[key]["dropped_unverified_items"] = dropped
                attempt = corrected
                atomic_json(manifest_path, manifest)
            if attempt.ok and attempt.data is not None:
                attempt.data["provider"] = provider
                attempt.data["source_pdf"] = str(target.pdf)
                attempt.data["source_summary_path"] = str(target.summary_path)
                out_path = output_root / "mds" / "reviews" / f"{target.number:02d}_{safe_name(target.pdf)}_review.md"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(render_review(attempt.data), encoding="utf-8")
                records[key].update({"status": "success", "provider": provider, "review_path": str(out_path), "output_hash": sha256(out_path), "data": attempt.data, "number": target.number, "updated_at": now_iso()})
                success = True
                dashboard(idx, len(targets), provider, target.pdf.name, states, started, "완료")
                break
            errors.append(f"{provider}: {attempt.category}: {attempt.error[:500]}")
            if attempt.category in ("unavailable", "missing") or (attempt.category in ("transient", "technical_failure") and not attempt.ok):
                status = unavailable_status(attempt.error) if attempt.category == "unavailable" else {"missing": "missing"}.get(attempt.category, "technical_failure")
                states[provider] = {"status": status, "error": attempt.error[:2000], "updated_at": now_iso()}
                dashboard(idx - 1, len(targets), provider, target.pdf.name, states, started, "전환")
        if not success:
            records[key].update({"status": "failed", "errors": errors, "number": target.number, "updated_at": now_iso()})
            atomic_json(manifest_path, manifest)
            if not any(states.get(p, {}).get("status") == "available" for p in PROVIDERS):
                resume = shlex.join([sys.executable, str(Path(__file__).relative_to(REPO_ROOT)), "--provider", args.provider, "--resume", "--input-dir", str(args.input_dir), "--output-dir", str(output_root)])
                handoff = write_handoff(output_root, manifest, target.pdf.name, resume)
                dashboard(idx - 1, len(targets), "none", target.pdf.name, states, started, f"중단: {handoff.name}")
                return EXIT_ALL_UNAVAILABLE
            dashboard(idx, len(targets), "none", target.pdf.name, states, started, "실패")
        aggregate_reviews(records, output_root)
        atomic_json(manifest_path, manifest)
    print()
    aggregate_reviews(records, output_root)
    manifest["completed_at"] = now_iso()
    atomic_json(manifest_path, manifest)
    failures = sum(rec.get("status") != "success" for rec in records.values())
    print(f"완료: 성공 {len(records)-failures}, 실패 {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
