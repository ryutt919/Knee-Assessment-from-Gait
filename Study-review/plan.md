
# 참고문헌 자동 분석 파이프라인

## 요약

- 기본 범위는 `docs/ref_papers/01_*`~`06_*`의 고유 PDF 45편으로 한다.
- 논문마다 독립된 `codex exec`를 순차 호출해 컨텍스트를 분리한다.
- 논문별 MD, 전체 통합 MD, 인용 가능 내용 전용 통합 MD, 재사용 프롬프트를 생성한다.
- 구현 후 1편으로 검증하고 전체 45편 분석을 실행한다.

## 구현 변경

- `2026-06-29-Study-review/` 아래에 번호가 붙은 `scripts/`, `prompts/`, `schemas/`, `mds/`, `logs/` 구조를 만든다.
- Python 표준 라이브러리 기반 실행기에 다음 CLI를 제공한다.
  - 필수/주요 옵션: `--input-dir`, `--output-dir`, `--model`
  - 실행 제어: `--dry-run`, `--resume`, `--overwrite`, `--only`, `--retry-failed`
  - 기본 제외: `99_*`, 날짜 기반 요약 폴더, 프로젝트 노트, 루트·해시 중복 PDF
  - 직접 지정한 폴더는 재사용 가능한 별도 분석 대상으로 허용
- 각 PDF를 `pdftotext -layout`으로 페이지 구분이 보존된 텍스트로 추출한 뒤 `codex exec --ephemeral --sandbox read-only --json --output-schema`를 한 편씩 실행한다.
- Codex 응답은 JSON Schema로 강제하고 실행기가 직접 MD를 렌더링한다. 논문별 결과에는 다음을 포함한다.
  - `# 논문 제목` 바로 아래 APA 7 형식의 해당 논문 참고문헌 표기
  - 연구 목적, 설계·대상, 방법·전처리·통계, 핵심 정량 결과, 저자 결론과 한계
  - 생각해볼 내용, 선행연구의 문제점, 본 연구의 해결 방식과 기여
  - 인용 가능한 핵심 주장 전체
- 각 인용 가능 주장에는 반드시 다음 필드를 둔다.
  - 원문 문장 그대로 발췌
  - 한국어 번역
  - PDF 페이지·절·표/그림 위치
  - 원문에 인쇄된 인용표기
  - 해당 인용표기에 대응하는 선행문헌의 완전한 참고문헌 표기
  - 배경 문헌 인용인지 분석 대상 논문의 자체 결과인지 구분
  - 활용 가능한 주장 범위와 2차 인용 주의사항
- 결과 파일은 다음처럼 생성한다.
  - `mds/papers/`: 논문별 분석 MD
  - `mds/01_all_study_reviews.md`: 전체 분석 통합본
  - `mds/02_referenceable_claims.md`: 분석 대상 논문의 표기와 인용 가능 주장만 모은 통합본
  - `prompts/01_reference_paper_analysis_prompt.md`: 사용자의 전체 요구사항과 출력 규격을 담은 재사용 프롬프트
- 통합본은 성공한 논문 결과에서 매 실행 후 재생성하여 중단돼도 완료분이 보존되게 한다.

## 진행상황과 오류 처리

- 터미널에 전체 진행률 막대, 완료/진행/실패 수, 현재 논문, 경과 시간, Codex 단계, 최근 메시지, 검증 상태를 고정형 대시보드로 표시한다.
- 원본 JSONL, 사람이 읽는 로그, 논문별 상태·SHA-256·시간·오류·산출물 경로를 담은 manifest를 저장한다.
- `Ctrl-C` 시 자식 프로세스를 종료하고 현재 상태를 기록한다.
- `--resume`은 해시와 검증 완료 상태가 같은 논문을 건너뛰고 실패·변경 논문만 다시 처리한다.
- 불완전하거나 텍스트가 부족한 PDF는 임의로 내용을 만들지 않고 `검토 필요`로 기록한다.

## 검증 및 완료 조건

- `--dry-run`으로 45편과 제외 대상이 정확한지 확인한다.
- 예시 Büttner 논문 1편을 먼저 실행해 원문 발췌, 번역, 인용번호와 참고문헌 매핑을 수동·자동 검증한다.
- 모든 원문 발췌가 추출 텍스트에 실제 존재하는지 정규화 비교하며, 페이지와 참고문헌 항목 누락 시 해당 결과를 실패 처리한다.
- 공백·한글·특수문자가 포함된 경로, 강제 실패 후 재개, 중단 후 재개를 검사한다.
- 전체 실행 후 manifest 성공 수, 논문별 파일 수, 두 통합본의 논문 수가 일치해야 한다.
- 기존 사용자 변경사항은 건드리지 않고 `Agent_handoff/03_literature_notes.md`를 현재 상태 형식으로 갱신한다.
- 생성 코드·프롬프트·스키마·분석 결과·handoff만 단일 작업 커밋으로 기록하고, `AGENTS.md`와 `CLAUDE.md`는 변경하지 않는다.

## 기본 가정

- 완전한 참고문헌 표기는 APA 7과 DOI를 우선하며, 인용된 선행문헌 표기는 분석 대상 PDF 참고문헌에 실린 원문도 함께 보존한다.
- 분석 설명은 한국어로 작성하고 직접 발췌문과 참고문헌 원문만 원어를 유지한다.
- 동일 선행문헌이 여러 논문에서 인용돼도 출처 추적을 위해 각 분석 논문 블록에 반복 수록한다.
- 전체 실행은 병렬화하지 않고 논문당 하나의 독립 Codex CLI 세션을 순차 처리한다.


# 다중 CLI 참고문헌 분석 파이프라인

## 요약

- `--provider codex|claude|antigravity` 플래그로 분석 CLI를 선택한다.
- Antigravity는 설치가 확인된 `agy 1.0.13`을 사용한다.
- 세 provider 모두 실제 비대화형 호출 기반 dry test를 통과해야 전체 분석을 허용한다.
- 모든 신규·수정 산출물은 `2026-06-29-Study-review/` 내부에만 둔다.

## 다중 CLI 어댑터

- 공통 실행 인터페이스:
  - `--provider {codex,claude,antigravity}`
  - `--model`, `--input-dir`, `--output-dir`
  - `--dry-run`: 논문 탐색과 명령 구성만 검증
  - `--provider-test`: 선택한 provider의 실제 최소 프롬프트 호출
  - `--provider-test-all`: 세 provider를 모두 실제 호출해 통합 검증
  - `--resume`, `--overwrite`, `--only`, `--retry-failed`
- Provider별 호출:
  - Codex: `codex exec --ephemeral --sandbox read-only --json --output-schema`
  - Claude: `claude --print --no-session-persistence --permission-mode plan --output-format stream-json --json-schema`
  - Antigravity: `agy --print --sandbox`; JSON 전용 응답 프롬프트를 적용하고 실행기가 Schema를 검증
- Antigravity 탐색 순서:
  1. PATH의 `antigravity`
  2. PATH의 `agy`
  3. `~/.local/bin/agy`
- 전역 `.zshrc`, `.zprofile`, `~/.local/bin`은 수정하지 않는다. 실행기 자식 프로세스의 PATH에만 발견된 `agy` 디렉터리를 임시 추가한다.

## 분석 및 출력

- 기본 대상은 `docs/ref_papers/01_*`~`06_*`의 고유 PDF 45편이며 `99_*`, 기존 요약, 프로젝트 노트와 중복본은 제외한다.
- PDF마다 선택한 provider의 독립 세션을 순차 실행한다.
- `pdftotext -layout` 결과와 모든 캐시·로그도 샌드박스 내부에 저장한다.
- 논문별 분석은 다음을 포함한다.
  - 제목 바로 아래 분석 대상 논문의 APA 7 참고문헌 표기
  - 목적, 설계·대상, 방법·전처리·통계, 핵심 정량 결과, 결론, 한계
  - 생각해볼 내용, 선행연구의 문제점, 연구의 해결 방식과 기여
  - 인용 가능한 핵심 주장별 원문, 한국어 번역, 페이지·절, 원문 인용표기, 연결된 선행문헌 전체 표기, 인용 유형과 활용 주의
- 샌드박스 내부 산출물:
  - `scripts/01_analyze_reference_papers.py`
  - `prompts/01_reference_paper_analysis_prompt.md`
  - `schemas/01_paper_analysis.schema.json`
  - `mds/papers/`: 논문별 분석
  - `mds/01_all_study_reviews.md`: 전체 분석 통합본
  - `mds/02_referenceable_claims.md`: 인용 가능 내용 전용 통합본
  - `logs/`: provider별 이벤트·일반 로그·manifest

## 진행상황

- 공통 대시보드에 provider, 모델, 전체 진행률, 현재 논문, 경과 시간, 성공·실패·검토 필요 수, 최근 이벤트와 검증 결과를 표시한다.
- Codex와 Claude의 스트리밍 이벤트는 단계별로 표시한다.
- Antigravity는 구조화 이벤트를 제공하지 않으므로 실행 시간, 프로세스 상태, 최근 출력과 완료 검증 상태를 표시한다.
- 중단 시 현재 상태를 manifest에 보존하고 `--resume`으로 검증 완료 논문을 건너뛴다.

## 테스트 및 완료 조건

- 정적 dry test:
  - 세 provider의 실행 파일·버전·필수 옵션 탐지
  - 45편 대상과 제외 목록 검증
  - 실제 분석 없이 provider별 최종 명령과 경로 검사
- 실제 provider dry test:
  - 동일한 작은 로컬 fixture와 최소 분석 프롬프트를 Codex, Claude, Antigravity 각각 한 번 호출
  - 세 결과 모두 동일 JSON Schema 통과
  - 한국어 필드, 원문 필드, provider 식별자와 종료 코드 확인
  - 어떤 provider든 실패하면 전체 분석을 시작하지 않고 원인과 로그를 표시
- 이후 Büttner 논문 한 편을 선택 provider로 smoke test하고 원문 존재 여부, 페이지, 인용표기와 참고문헌 연결을 검증한다.
- 검증 성공 후 기본 provider로 45편 전체 분석을 실행한다.
- 샌드박스 밖에는 handoff, 지침, 셸 설정, 분석 결과 파일을 생성하거나 수정하지 않는다. 기존 사용자 변경사항도 건드리지 않는다.

## 기본 가정

- `--provider antigravity`는 설치된 `agy` CLI를 의미한다.
- 세 provider dry test는 단순 `--help` 검사가 아니라 실제 모델 응답과 Schema 검증까지 포함한다.
- 동일 선행문헌도 출처 논문별 추적을 위해 각 논문 블록에 반복 수록한다.
- 분석 설명은 한국어로 작성하고 직접 발췌와 참고문헌 원문만 원어를 유지한다.

