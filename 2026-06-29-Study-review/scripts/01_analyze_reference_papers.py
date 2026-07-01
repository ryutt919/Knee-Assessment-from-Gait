#!/usr/bin/env python3
"""Sequential multi-CLI reference-paper analyzer with provider failover."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

TASK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_ROOT.parent
DEFAULT_INPUT = REPO_ROOT / "docs" / "ref_papers"
SCHEMA_PATH = TASK_ROOT / "schemas" / "01_paper_analysis.schema.json"
PROMPT_PATH = TASK_ROOT / "prompts" / "01_reference_paper_analysis_prompt.md"
PROVIDERS = ("codex", "claude", "antigravity")
EXIT_ALL_UNAVAILABLE = 75

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
    # Clean up common invalid python escape sequences outputted by LLMs (like \xad)
    text = re.sub(r'\\x([0-9a-fA-F]{2})', r'\\u00\1', text)
    # Clean up double quotes separated by slash inside JSON values (e.g. "..." / "...")
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


def validate_shape(data: dict[str, Any], provider: str) -> list[str]:
    required = [
        "provider", "title", "paper_bibliography", "bibliographic_info", "purpose",
        "design_and_participants", "methods", "key_results", "author_conclusion",
        "limitations", "reviewer_thoughts", "prior_research_problems",
        "study_solution_and_contribution", "referenceable_claims",
    ]
    errors = [f"누락 필드: {key}" for key in required if key not in data]
    if data.get("provider") != provider:
        errors.append(f"provider 불일치: {data.get('provider')} != {provider}")
    for key in required[4:]:
        if key in data and not isinstance(data[key], list):
            errors.append(f"{key}는 배열이어야 합니다")
    for key in required[4:-1]:
        for i, item in enumerate(data.get(key, [])):
            if not isinstance(item, dict) or not all(k in item for k in ("summary", "evidence_quote", "locator")):
                errors.append(f"{key} {i}는 summary/evidence_quote/locator 객체여야 합니다")
    for i, claim in enumerate(data.get("referenceable_claims", [])):
        if not isinstance(claim, dict):
            errors.append(f"claim {i}는 객체가 아닙니다")
            continue
        for key in ("topic", "original_quote", "korean_translation", "locator", "in_text_citation", "cited_reference", "claim_type", "usage_note"):
            if key not in claim:
                errors.append(f"claim {i} 누락: {key}")
    return errors


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def comparison_normalized(text: str) -> str:
    """Normalize PDF line wrapping without weakening word-sequence checks."""
    text = unicodedata.normalize("NFKC", text).lower()
    # Strip page boundaries and page numbers that follow them
    text = re.sub(r"--- page \d+ ---\s*\d*", "", text)
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    return "".join(ch for ch in text if ch.isalnum())


def validate_quotes(data: dict[str, Any], source_text: str) -> list[str]:
    haystack = comparison_normalized(source_text)
    errors = []
    evidence_sections = (
        "purpose", "design_and_participants", "methods", "key_results",
        "author_conclusion", "limitations", "reviewer_thoughts",
        "prior_research_problems", "study_solution_and_contribution",
    )
    for section in evidence_sections:
        for i, item in enumerate(data.get(section, []), 1):
            quote = item.get("evidence_quote", "")
            parts = [p for p in re.split(r"\s*(?:\.{3,}|…)\s*", quote) if p.strip()]
            for part in parts:
                evidence = comparison_normalized(part)
                if not evidence or evidence not in haystack:
                    errors.append(f"{section} {i} 근거 원문이 추출 텍스트에 없음")
                    break
            if not re.search(r"page\s*\d+|p\.?\s*\d+|페이지\s*\d+", item.get("locator", ""), re.I):
                errors.append(f"{section} {i} 페이지 위치 누락")
    for i, claim in enumerate(data.get("referenceable_claims", []), 1):
        quote = claim.get("original_quote", "")
        parts = [p for p in re.split(r"\s*(?:\.{3,}|…)\s*", quote) if p.strip()]
        for part in parts:
            evidence = comparison_normalized(part)
            if not evidence or evidence not in haystack:
                errors.append(f"claim {i} 원문이 추출 텍스트에 없음")
                break
        if not re.search(r"page\s*\d+|p\.?\s*\d+|페이지\s*\d+", claim.get("locator", ""), re.I):
            errors.append(f"claim {i} 페이지 위치 누락")
        cited_raw = claim.get("cited_reference", "")
        if cited_raw not in ("해당 없음", "없음", "확인 불가"):
            cited_parts = [p for p in re.split(r"(?=\b\d+\.\s+[A-Z])", cited_raw) if p.strip()]
            for part in cited_parts:
                part_clean = re.split(r"\s*(?:\.{3,}|…)\s*", part)[0]
                if comparison_normalized(part_clean) not in haystack:
                    errors.append(f"claim {i} 참고문헌 원문이 추출 텍스트에 없음")
                    break
    return errors


def sanitize_unverified_items(data: dict[str, Any], source_text: str) -> list[str]:
    """Remove only evidence units that cannot be proven against the PDF text."""
    dropped: list[str] = []
    sections = (
        "purpose", "design_and_participants", "methods", "key_results",
        "author_conclusion", "limitations", "reviewer_thoughts",
        "prior_research_problems", "study_solution_and_contribution",
    )
    for section in sections:
        kept = []
        for i, item in enumerate(data.get(section, []), 1):
            probe = {key: [] for key in sections}
            probe[section] = [item]
            probe["referenceable_claims"] = []
            if validate_quotes(probe, source_text):
                dropped.append(f"{section} {i}: {item.get('summary', '')}")
            else:
                kept.append(item)
        data[section] = kept
    kept_claims = []
    for i, claim in enumerate(data.get("referenceable_claims", []), 1):
        probe = {key: [] for key in sections}
        probe["referenceable_claims"] = [claim]
        if validate_quotes(probe, source_text):
            dropped.append(f"referenceable_claims {i}: {claim.get('topic', '')}")
        else:
            kept_claims.append(claim)
    data["referenceable_claims"] = kept_claims
    return dropped


def discover_pdfs(input_dir: Path) -> list[Path]:
    if input_dir.resolve() == DEFAULT_INPUT.resolve():
        result = []
        for child in sorted(input_dir.iterdir()):
            if child.is_dir() and re.match(r"^0[1-6]_", child.name):
                result.extend(sorted(child.glob("*.pdf")))
        return result
    return sorted(input_dir.rglob("*.pdf"))


def extract_pdf(pdf: Path, cache_dir: Path) -> Path:
    cache_dir = cache_dir / f"{safe_name(pdf)}_{sha256(pdf)[:12]}_raw_v2"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / "paper.txt"
    if out.exists() and out.stat().st_size > 0:
        return out
    # -raw preserves reading order for multi-column journal PDFs. -layout often
    # interleaves the left and right columns on each line, which makes exact
    # quotation validation unreliable.
    raw = subprocess.run(["pdftotext", "-raw", str(pdf), "-"], check=True, capture_output=True).stdout
    pages = raw.decode("utf-8", errors="replace").split("\f")
    rendered = "\n".join(f"\n--- PAGE {i} ---\n{page}" for i, page in enumerate(pages, 1))
    out.write_text(rendered, encoding="utf-8")
    return out


def render_prompt(provider: str, pdf: Path, text_path: Path) -> str:
    base = PROMPT_PATH.read_text(encoding="utf-8")
    request = base.format(provider=provider, pdf_path=pdf, text_path=text_path)
    source = text_path.read_text(encoding="utf-8", errors="replace")
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    return (
        request
        + "\n\n반드시 아래 JSON_SCHEMA의 키 이름과 배열/객체 구조를 정확히 사용하라. 다른 키 이름을 만들지 마라.\n"
        + "<JSON_SCHEMA>\n" + schema + "\n</JSON_SCHEMA>\n"
        + "\n\n아래 SOURCE_TEXT만 논문 근거로 사용하라. 도구를 사용하거나 다른 파일·웹을 조회하지 마라.\n"
        + "<SOURCE_TEXT>\n"
        + source
        + "\n</SOURCE_TEXT>\n"
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
    # agy headless mode reads the prompt from stdin. A trailing positional
    # argument is silently ignored by current releases and starts a generic
    # agent task instead.
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
        errors = validate_shape(data, provider)
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
        "paper_bibliography": "Kim, A. (2026). Test paper. Test Journal, 1(1), 1-2.",
        "bibliographic_info": {"authors": "Kim, A.", "year": "2026", "journal": "Test Journal", "doi": "확인 불가", "source_file": "fixture.txt"},
        "purpose": [{"summary": "한국어 테스트", "evidence_quote": "Test evidence.", "locator": "Page 1, Test"}], "design_and_participants": [], "methods": [],
        "key_results": [], "author_conclusion": [], "limitations": [],
        "reviewer_thoughts": [], "prior_research_problems": [],
        "study_solution_and_contribution": [], "referenceable_claims": [],
    }


def provider_test_prompt(provider: str) -> str:
    fixture = minimal_fixture(provider)
    return "도구를 절대 사용하지 마라. 파일을 검색하거나 읽지 마라. 다음 JSON 객체를 변경하거나 설명하지 말고 JSON 하나로만 즉시 그대로 반환하라.\n" + json.dumps(fixture, ensure_ascii=False)


def run_provider_test(provider: str, logs_dir: Path, model: str | None) -> Invocation:
    workspace = logs_dir / "provider_test_workspace" / provider
    workspace.mkdir(parents=True, exist_ok=True)
    return invoke(provider, provider_test_prompt(provider), logs_dir, "provider_test", model, timeout=180, workspace=workspace)


def model_for(provider: str, args: argparse.Namespace) -> str | None:
    specific = getattr(args, f"{provider}_model", None)
    if specific:
        return specific
    return args.model if provider == args.provider else None


def bullet_list(values: Iterable[Any]) -> str:
    values = list(values)
    if not values:
        return "- 확인된 내용 없음"
    lines = []
    for value in values:
        if isinstance(value, dict):
            lines.append(f"- {value['summary']} _(근거: {value['locator']})_")
            lines.append(f"  - 근거 원문: “{value['evidence_quote']}”")
        else:
            lines.append(f"- {value}")
    return "\n".join(lines)


def render_paper(data: dict[str, Any]) -> str:
    b = data["bibliographic_info"]
    parts = [
        f"# {data['title']}", "", data["paper_bibliography"], "",
        "## 서지정보", "", f"- 저자: {b['authors']}", f"- 연도: {b['year']}", f"- 저널: {b['journal']}", f"- DOI: {b['doi']}", f"- 원본 파일: {b['source_file']}", f"- 분석 provider: {data['provider']}", "",
    ]
    sections = [
        ("연구 목적", "purpose"), ("연구 설계와 대상", "design_and_participants"),
        ("방법", "methods"), ("핵심 결과", "key_results"),
        ("저자 결론", "author_conclusion"), ("연구의 한계", "limitations"),
        ("생각해볼 내용", "reviewer_thoughts"),
        ("이 연구가 지적한 선행연구의 문제점", "prior_research_problems"),
        ("이 연구의 해결 방식과 기여", "study_solution_and_contribution"),
    ]
    for title, key in sections:
        parts += [f"## {title}", "", bullet_list(data[key]), ""]
    parts += ["## 레퍼런스할 수 있는 내용", ""]
    claims = data["referenceable_claims"]
    if not claims:
        parts += ["- 검증 가능한 인용 항목을 확인하지 못함", ""]
    for i, claim in enumerate(claims, 1):
        parts += [
            f"### {i}. {claim['topic']}", "",
            f"- 원문 발췌: “{claim['original_quote']}”",
            f"- 한국어 번역: {claim['korean_translation']}",
            f"- 원문 위치: {claim['locator']}",
            f"- 원문 내 인용표기: {claim['in_text_citation']}",
            f"- 해당 선행문헌: {claim['cited_reference']}",
            f"- 주장 유형: {claim['claim_type']}",
            f"- 활용 맥락과 주의: {claim['usage_note']}", "",
        ]
    return "\n".join(parts).rstrip() + "\n"


def render_claims(data: dict[str, Any]) -> str:
    lines = [f"## {data['title']}", "", f"- 분석 대상 논문 인용표기: {data['paper_bibliography']}", f"- 분석 provider: {data['provider']}", ""]
    for i, claim in enumerate(data["referenceable_claims"], 1):
        lines += [
            f"### {i}. {claim['topic']}", "", f"- 원문 발췌: “{claim['original_quote']}”",
            f"- 한국어 번역: {claim['korean_translation']}", f"- 원문 위치: {claim['locator']}",
            f"- 원문 내 인용표기: {claim['in_text_citation']}", f"- 해당 선행문헌: {claim['cited_reference']}",
            f"- 활용 맥락과 주의: {claim['usage_note']}", "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def aggregate(records: dict[str, Any], output_root: Path) -> None:
    successful = [v for _, v in sorted(records.items()) if v.get("status") == "success"]
    full = ["# 전체 참고문헌 논문 분석", ""]
    claims = ["# 레퍼런스 활용 내용 모음", ""]
    for rec in successful:
        paper_path = Path(rec["output_path"])
        full += [paper_path.read_text(encoding="utf-8").rstrip(), "", "---", ""]
        claims += [render_claims(rec["data"]).rstrip(), "", "---", ""]
    (output_root / "mds").mkdir(parents=True, exist_ok=True)
    (output_root / "mds" / "01_all_study_reviews.md").write_text("\n".join(full).rstrip() + "\n", encoding="utf-8")
    (output_root / "mds" / "02_referenceable_claims.md").write_text("\n".join(claims).rstrip() + "\n", encoding="utf-8")


def next_handoff_path(output_root: Path) -> Path:
    folder = output_root / "handoffs"
    folder.mkdir(parents=True, exist_ok=True)
    number = len(list(folder.glob("[0-9][0-9]_provider_exhaustion_*.md"))) + 1
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return folder / f"{number:02d}_provider_exhaustion_{stamp}.md"


def write_handoff(output_root: Path, manifest: dict[str, Any], current: str, command: str) -> Path:
    path = next_handoff_path(output_root)
    records = manifest.get("papers", {})
    counts = {s: sum(r.get("status") == s for r in records.values()) for s in ("success", "failed")}
    counts["pending"] = max(0, len(manifest.get("inventory", records)) - counts["success"] - counts["failed"])
    states = manifest.get("provider_states", {})
    lines = [
        "# Provider 사용 불가 Handoff", "", f"- 실행 ID: {manifest['run_id']}", f"- 중단 시각: {now_iso()}",
        f"- 현재 논문: {current}", f"- 완료: {counts['success']}", f"- 실패: {counts['failed']}", f"- 대기: {counts['pending']}", "",
        f"- 프롬프트 SHA-256: {manifest.get('prompt_hash', '확인 불가')}",
        f"- Schema SHA-256: {manifest.get('schema_hash', '확인 불가')}", "",
        "## Provider 상태", "",
    ]
    for provider in PROVIDERS:
        state = states.get(provider, {})
        lines += [f"### {provider}", f"- 상태: {state.get('status', 'unknown')}", f"- 오류: {state.get('error', '없음')}", f"- reset 시각: {state.get('resets_at', '확인 불가')}", ""]
    lines += ["## 재개", "", f"```bash\n{command}\n```", "", "manifest와 완료된 MD는 보존되어 있으며 재개 시 성공 항목을 건너뜁니다.", ""]
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
    pdfs = discover_pdfs(args.input_dir.resolve())
    global_index = {str(p.resolve()): i for i, p in enumerate(pdfs, 1)}
    if args.only:
        pdfs = [p for p in pdfs if args.only.lower() in p.name.lower()]
    binaries = {p: find_binary(p) for p in PROVIDERS}
    print(f"분석 대상: {len(pdfs)}편")
    for p in PROVIDERS:
        print(f"- {p}: {binaries[p] or 'MISSING'}")
    if args.dry_run:
        print("전환 순서:", " → ".join(provider_order(args.provider)))
        for pdf in pdfs:
            print(pdf)
        return 0 if all(binaries.values()) else 2

    logs_dir = output_root / "logs"
    if args.provider_test or args.provider_test_all:
        targets = PROVIDERS if args.provider_test_all else (args.provider,)
        failed = False
        for provider in targets:
            print(f"[{provider}] 실제 provider 테스트...", flush=True)
            result = run_provider_test(provider, logs_dir, model_for(provider, args))
            print(f"[{provider}] {'PASS' if result.ok else 'FAIL'} ({result.elapsed:.1f}s) {result.error[:300]}")
            failed |= not result.ok
        return 1 if failed else 0

    if not args.skip_provider_gate:
        gate_results = {p: run_provider_test(p, logs_dir, model_for(p, args)) for p in PROVIDERS}
        if not all(r.ok for r in gate_results.values()):
            print("세 provider 사전 테스트가 모두 통과하지 못해 분석을 시작하지 않습니다.", file=sys.stderr)
            return 1

    manifest_path = logs_dir / "manifest.json"
    if args.resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["inventory"] = [str(p.resolve()) for p in pdfs]
        manifest["input_dir"] = str(args.input_dir.resolve())
        manifest["prompt_hash"] = sha256(PROMPT_PATH)
        manifest["schema_hash"] = sha256(SCHEMA_PATH)
    else:
        manifest = {
            "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "created_at": now_iso(), "input_dir": str(args.input_dir.resolve()),
            "inventory": [str(p.resolve()) for p in pdfs],
            "prompt_hash": sha256(PROMPT_PATH), "schema_hash": sha256(SCHEMA_PATH),
            "provider_states": {p: {"status": "available", "error": ""} for p in PROVIDERS},
            "papers": {},
        }
    records = manifest["papers"]
    states = manifest["provider_states"]
    started = time.monotonic()

    for idx, pdf in enumerate(pdfs, 1):
        key = str(pdf.resolve())
        existing = records.get(key, {})
        if existing.get("status") == "success" and not args.overwrite:
            dashboard(idx, len(pdfs), existing.get("provider", "skip"), pdf.name, states, started, "완료: resume skip")
            continue
        if existing.get("status") == "failed" and not (args.retry_failed or args.overwrite or args.resume):
            continue
        records[key] = {"status": "pending", "pdf_hash": sha256(pdf), "attempts": [], "updated_at": now_iso()}
        atomic_json(manifest_path, manifest)
        text_path = extract_pdf(pdf, output_root / "logs" / "cache")
        success = False
        errors: list[str] = []
        for provider in provider_order(args.provider):
            if states.get(provider, {}).get("status") != "available":
                continue
            dashboard(idx - 1, len(pdfs), provider, pdf.name, states, started, "분석 중")
            prompt = render_prompt(provider, pdf, text_path)
            attempt: Invocation | None = None
            for retry in range(3):
                attempt = invoke(provider, prompt, logs_dir / manifest["run_id"], safe_name(pdf), model_for(provider, args), workspace=text_path.parent)
                records[key]["attempts"].append({"provider": provider, "ok": attempt.ok, "category": attempt.category, "error": attempt.error[:2000], "elapsed": attempt.elapsed, "at": now_iso()})
                atomic_json(manifest_path, manifest)
                if attempt.ok:
                    break
                if attempt.category != "transient" or retry == 2:
                    break
                wait = (10, 30)[retry]
                dashboard(idx - 1, len(pdfs), provider, pdf.name, states, started, f"재시도 대기 {wait}s")
                time.sleep(wait)
            assert attempt is not None
            if attempt.ok and attempt.data is not None:
                quote_errors = validate_quotes(attempt.data, text_path.read_text(encoding="utf-8", errors="replace"))
                if quote_errors:
                    attempt.ok = False
                    attempt.category = "invalid_output"
                    attempt.error = "; ".join(quote_errors)
            if not attempt.ok and attempt.category == "invalid_output":
                correction = (
                    prompt
                    + "\n\n이전 응답은 다음 자동 검증에 실패했다: "
                    + attempt.error
                    + "\nPDF 원문을 다시 확인하고 인용문과 참고문헌을 임의로 고치거나 만들지 말고, 검증 가능한 항목만 포함해 JSON 전체를 다시 반환하라."
                )
                corrected = invoke(
                    provider, correction, logs_dir / manifest["run_id"],
                    safe_name(pdf) + "_correction", model_for(provider, args),
                    workspace=text_path.parent,
                )
                records[key]["attempts"].append({
                    "provider": provider, "ok": corrected.ok,
                    "category": corrected.category, "error": corrected.error[:2000],
                    "elapsed": corrected.elapsed, "at": now_iso(), "correction": True,
                })
                if corrected.ok and corrected.data is not None:
                    correction_errors = validate_quotes(
                        corrected.data, text_path.read_text(encoding="utf-8", errors="replace")
                    )
                    if correction_errors:
                        source_text = text_path.read_text(encoding="utf-8", errors="replace")
                        dropped = sanitize_unverified_items(corrected.data, source_text)
                        remaining = validate_quotes(corrected.data, source_text)
                        core_ok = all(corrected.data.get(key) for key in ("purpose", "methods", "key_results"))
                        if remaining or not core_ok:
                            corrected.ok = False
                            corrected.category = "invalid_output"
                            corrected.error = "; ".join(remaining or ["검증 후 핵심 목적/방법/결과가 남지 않음"])
                        else:
                            records[key]["dropped_unverified_items"] = dropped
                attempt = corrected
                atomic_json(manifest_path, manifest)
            if attempt.ok and attempt.data is not None:
                attempt.data["provider"] = provider
                attempt.data["bibliographic_info"]["source_file"] = str(pdf)
                paper_number = global_index[str(pdf.resolve())]
                out_path = output_root / "mds" / "papers" / f"{paper_number:02d}_{safe_name(pdf)}.md"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(render_paper(attempt.data), encoding="utf-8")
                records[key].update({"status": "success", "provider": provider, "output_path": str(out_path), "output_hash": sha256(out_path), "data": attempt.data, "updated_at": now_iso()})
                success = True
                dashboard(idx, len(pdfs), provider, pdf.name, states, started, "완료")
                break
            errors.append(f"{provider}: {attempt.category}: {attempt.error[:500]}")
            if attempt.category in ("unavailable", "missing") or (attempt.category in ("transient", "technical_failure") and not attempt.ok):
                status = unavailable_status(attempt.error) if attempt.category == "unavailable" else {"missing": "missing"}.get(attempt.category, "technical_failure")
                states[provider] = {"status": status, "error": attempt.error[:2000], "updated_at": now_iso()}
                dashboard(idx - 1, len(pdfs), provider, pdf.name, states, started, "전환")
            # Invalid output is per-paper; still try the next provider without globally disabling it.
        if not success:
            records[key].update({"status": "failed", "errors": errors, "updated_at": now_iso()})
            atomic_json(manifest_path, manifest)
            if not any(states.get(p, {}).get("status") == "available" for p in PROVIDERS):
                resume = shlex.join([sys.executable, str(Path(__file__).relative_to(REPO_ROOT)), "--provider", args.provider, "--resume", "--input-dir", str(args.input_dir), "--output-dir", str(output_root)])
                handoff = write_handoff(output_root, manifest, pdf.name, resume)
                dashboard(idx - 1, len(pdfs), "none", pdf.name, states, started, f"중단: {handoff.name}")
                return EXIT_ALL_UNAVAILABLE
            dashboard(idx, len(pdfs), "none", pdf.name, states, started, "실패")
        aggregate(records, output_root)
        atomic_json(manifest_path, manifest)
    print()
    aggregate(records, output_root)
    manifest["completed_at"] = now_iso()
    atomic_json(manifest_path, manifest)
    failures = sum(rec.get("status") != "success" for rec in records.values())
    print(f"완료: 성공 {len(records)-failures}, 실패 {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
