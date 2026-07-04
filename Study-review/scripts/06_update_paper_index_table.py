#!/usr/bin/env python3
"""표 전용 갱신 스크립트: provider 호출로 논문 분류 → mds/06_paper_index_table.md 갱신.

기존 분석(01) 없이 실행 가능. manifest의 PDF 목록을 읽어 각 논문 초록을 provider에게
전달하고, 4가지 boolean 태그와 한국어 제목을 추출해 표를 재생성한다.

이미 paper_tags가 있는 논문(01 스크립트로 재분석된 경우)은 provider를 호출하지 않는다.
결과는 logs/paper_index_cache.json에 캐시된다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

TASK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_ROOT.parent
CLASSIF_SCHEMA_PATH = TASK_ROOT / "schemas" / "06_paper_tags.schema.json"
PROVIDERS = ("codex", "claude", "antigravity")

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

CLASSIF_PROMPT_TEMPLATE = """\
다음 논문 텍스트를 읽고 아래 JSON 하나만 반환하라. 설명이나 코드 블록 없이 JSON만.

{{"title_korean": "논문 제목의 한국어 번역", "acl_research": true, "uses_imu": false, "uses_gait_data": true, "presents_score": false}}

판단 기준:
- title_korean: 논문 제목의 자연스러운 한국어 번역 (전문 용어 영문 병기 허용)
- acl_research: ACL(전방십자인대) 관련 연구이면 true
- uses_imu: IMU, 관성 측정 장치, 또는 웨어러블 센서를 실험에 사용하면 true
- uses_gait_data: 보행(gait/walking) 데이터를 수집하거나 분석하면 true
- presents_score: 새로운 점수(score) 또는 지수(index)를 제안·제시하면 true

<SOURCE_TEXT>
{excerpt}
</SOURCE_TEXT>
"""


def sha256_file(path: Path) -> str:
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


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    text = re.sub(r"\\x([0-9a-fA-F]{2})", r"\\u00\1", text)
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
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    raise ValueError("응답에서 JSON 객체를 찾지 못했습니다")


def extract_pdf(pdf: Path, cache_dir: Path) -> Path:
    """script 01과 동일한 캐시 키를 사용해 기존 캐시를 재활용한다."""
    key_dir = cache_dir / f"{safe_name(pdf)}_{sha256_file(pdf)[:12]}_raw_v2"
    key_dir.mkdir(parents=True, exist_ok=True)
    out = key_dir / "paper.txt"
    if out.exists() and out.stat().st_size > 0:
        return out
    raw = subprocess.run(
        ["pdftotext", "-raw", str(pdf), "-"], check=True, capture_output=True
    ).stdout
    pages = raw.decode("utf-8", errors="replace").split("\f")
    rendered = "\n".join(
        f"\n--- PAGE {i} ---\n{page}" for i, page in enumerate(pages, 1)
    )
    out.write_text(rendered, encoding="utf-8")
    return out


def build_classif_command(
    provider: str, result_path: Path, model: str | None
) -> list[str]:
    binary = find_binary(provider)
    if not binary:
        raise FileNotFoundError(f"{provider} 실행 파일을 찾을 수 없습니다")
    if provider == "codex":
        cmd = [
            binary, "exec", "--ephemeral", "--sandbox", "read-only", "--json",
            "--output-schema", str(CLASSIF_SCHEMA_PATH),
            "-o", str(result_path), "-C", str(TASK_ROOT),
        ]
        if model:
            cmd += ["--model", model]
        cmd += ["-"]
        return cmd
    if provider == "claude":
        schema = json.dumps(
            json.loads(CLASSIF_SCHEMA_PATH.read_text()), separators=(",", ":")
        )
        cmd = [
            binary, "--print", "--no-session-persistence",
            "--permission-mode", "plan", "--output-format", "stream-json",
            "--verbose", "--json-schema", schema,
        ]
        if model:
            cmd += ["--model", model]
        return cmd
    cmd = [binary, "--print", "--sandbox", "--print-timeout", "10m"]
    if model:
        cmd += ["--model", model]
    return cmd


def parse_classif_output(
    provider: str, stdout: str, result_path: Path
) -> dict[str, Any]:
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


def validate_classif(data: dict[str, Any]) -> list[str]:
    errors = []
    if not isinstance(data.get("title_korean"), str) or not data.get("title_korean", "").strip():
        errors.append("title_korean 누락 또는 빈 문자열")
    for k in ("acl_research", "uses_imu", "uses_gait_data", "presents_score"):
        if k not in data:
            errors.append(f"누락: {k}")
        elif not isinstance(data[k], bool):
            errors.append(f"{k}는 boolean이어야 합니다")
    return errors


def invoke_classif(
    provider: str,
    prompt: str,
    logs_dir: Path,
    run_key: str,
    model: str | None,
    timeout: int = 300,
) -> dict[str, Any] | None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    result_path = logs_dir / f"{run_key}_{provider}_classif.json"
    try:
        cmd = build_classif_command(provider, result_path, model)
    except FileNotFoundError as exc:
        print(f"  [{provider}] SKIP: {exc}", file=sys.stderr)
        return None
    for attempt in range(3):
        try:
            proc = subprocess.run(
                cmd, input=prompt, text=True, capture_output=True,
                timeout=timeout, cwd=TASK_ROOT,
            )
            combined = proc.stdout + ("\nSTDERR:\n" + proc.stderr if proc.stderr else "")
            if proc.returncode:
                cat = classify_failure(combined, proc.returncode)
                if cat == "unavailable":
                    print(f"  [{provider}] 사용 불가", file=sys.stderr)
                    return None
                if cat != "transient" or attempt == 2:
                    print(f"  [{provider}] 실패({cat}): {combined[-200:]}", file=sys.stderr)
                    return None
                time.sleep((10, 30)[attempt])
                continue
            data = parse_classif_output(provider, proc.stdout, result_path)
            errs = validate_classif(data)
            if errs:
                if attempt < 2:
                    time.sleep(5)
                    continue
                print(f"  [{provider}] 검증 실패: {'; '.join(errs)}", file=sys.stderr)
                return None
            return data
        except subprocess.TimeoutExpired:
            if attempt == 2:
                return None
            time.sleep(10)
    return None


def provider_order(start: str) -> list[str]:
    idx = PROVIDERS.index(start)
    return list(PROVIDERS[idx:] + PROVIDERS[:idx])


def rebuild_table(
    rows: list[tuple[int, str, str, dict[str, Any]]], output_root: Path
) -> None:
    rows_sorted = sorted(rows, key=lambda r: r[0])
    lines = [
        "# 논문 인덱스 표", "",
        "| # | 논문 제목 | ACL 연구 | IMU 사용 | 보행 데이터 | Score 제시 |",
        "|---|-----------|:--------:|:--------:|:-----------:|:----------:|",
    ]
    for number, title, title_korean, tags in rows_sorted:
        title_cell = f"{title}<br>*{title_korean}*" if title_korean else title
        lines.append(
            f"| {number} | {title_cell} | "
            f"{'✅' if tags.get('acl_research') else '❌'} | "
            f"{'✅' if tags.get('uses_imu') else '❌'} | "
            f"{'✅' if tags.get('uses_gait_data') else '❌'} | "
            f"{'✅' if tags.get('presents_score') else '❌'} |"
        )
    out = output_root / "mds" / "06_paper_index_table.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[표] {out.name} 갱신 완료 ({len(rows_sorted)}편)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=PROVIDERS, default="codex")
    parser.add_argument("--model")
    parser.add_argument("--output-dir", type=Path, default=TASK_ROOT)
    parser.add_argument("--overwrite", action="store_true", help="캐시 무시, 강제 재분류")
    args = parser.parse_args()

    output_root = args.output_dir.resolve()
    manifest_path = output_root / "logs" / "manifest.json"
    if not manifest_path.exists():
        print(f"manifest.json 없음: {manifest_path}", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records: dict[str, Any] = manifest.get("papers", {})

    cache_path = output_root / "logs" / "paper_index_cache.json"
    cache: dict[str, Any] = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    logs_dir = output_root / "logs" / "table_update"
    pdf_cache_dir = output_root / "logs" / "cache"
    rows: list[tuple[int, str, str, dict[str, Any]]] = []

    for pdf_path_str, rec in records.items():
        if rec.get("status") != "success":
            continue
        data = rec.get("data") or {}
        number = rec.get("number", 0)
        title = data.get("title", "")

        # 01 스크립트로 재분석된 논문은 paper_tags가 이미 있음 → 그대로 사용
        existing_tags = data.get("paper_tags")
        existing_korean = data.get("title_korean", "")
        if existing_tags and not args.overwrite:
            rows.append((number, title, existing_korean, existing_tags))
            continue

        # 캐시 확인
        if pdf_path_str in cache and not args.overwrite:
            cached = cache[pdf_path_str]
            rows.append((number, title, cached.get("title_korean", ""), {
                "acl_research": bool(cached.get("acl_research", False)),
                "uses_imu": bool(cached.get("uses_imu", False)),
                "uses_gait_data": bool(cached.get("uses_gait_data", False)),
                "presents_score": bool(cached.get("presents_score", False)),
            }))
            continue

        # PDF 없으면 skip
        pdf = Path(pdf_path_str)
        if not pdf.exists():
            print(f"  PDF 없음 (skip): {pdf.name}", file=sys.stderr)
            rows.append((number, title, existing_korean, {}))
            continue

        print(f"[{number:2d}] {pdf.name[:65]}")
        try:
            text_path = extract_pdf(pdf, pdf_cache_dir)
            excerpt = text_path.read_text(encoding="utf-8", errors="replace")[:8000]
        except Exception as exc:
            print(f"  PDF 추출 실패: {exc}", file=sys.stderr)
            rows.append((number, title, existing_korean, {}))
            continue

        prompt = CLASSIF_PROMPT_TEMPLATE.format(excerpt=excerpt)
        result = None
        for prov in provider_order(args.provider):
            result = invoke_classif(
                prov, prompt, logs_dir, safe_name(pdf), args.model
            )
            if result:
                break

        if result:
            tags: dict[str, Any] = {
                "acl_research": bool(result.get("acl_research")),
                "uses_imu": bool(result.get("uses_imu")),
                "uses_gait_data": bool(result.get("uses_gait_data")),
                "presents_score": bool(result.get("presents_score")),
            }
            title_korean = str(result.get("title_korean", ""))
            cache[pdf_path_str] = {"title_korean": title_korean, **tags}
            atomic_json(cache_path, cache)
            rows.append((number, title, title_korean, tags))
        else:
            print(f"  분류 실패 — 표에 태그 없이 추가", file=sys.stderr)
            rows.append((number, title, existing_korean, {}))

    rebuild_table(rows, output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
