# 참고문헌 분석 파이프라인

`docs/ref_papers/01_*`부터 `07_*`까지의 논문을 한 편씩 **분석 → 즉시 검증** 순서로 처리한다.

## 통합 실행 (권장)

```bash
cd /Users/ryutt/Desktop/mini_ryutt/Walking

# 처리 대상 확인 (실제 실행 없음)
python3 Study-review/run_pipeline.py --dry-run

# 전체 실행: 논문 하나씩 분석 → 검증 완료 후 다음 논문
python3 Study-review/run_pipeline.py --provider codex

# 특정 논문만
python3 Study-review/run_pipeline.py --only "Bilateral" --provider codex

# 중단된 실행 재개 (이미 성공한 논문은 자동 skip)
python3 Study-review/run_pipeline.py --provider codex

# 다른 탐색 폴더 지정
python3 Study-review/run_pipeline.py --input-dir path/to/pdfs --provider codex

# batch manifest에 기록된 논문만 증분 분석·검증
python3 Study-review/run_pipeline.py \
  --input-manifest Study-review/mds/2026-07-04/01_batch_manifest.json \
  --provider codex

# provider 응답 테스트만 실행
python3 Study-review/run_pipeline.py --provider-test-all
```

`run_pipeline.py`는 논문별로 `01 → 02`를 순차 실행한다. 분석이 성공한 논문만 즉시 검증 단계로 넘어가며, 모든 provider가 소진되면 종료 코드 75로 중단한다. `--input-manifest` 사용 시 manifest의 `papers[].final_pdf_path`(및 `sha256`)만 정확히 처리하고 선택 배치의 성공 여부만 종료 코드에 반영한다. 기존 분석 번호와 누적 inventory는 보존되며 신규 논문은 기존 최댓값 다음 번호부터 부여된다. 사전검사(`--provider-test-all`)를 통과하면 오래된 `provider_states`를 초기화한 뒤 실행을 시작한다.

날짜 폴더(`Study-review/mds/YYYY-MM-DD/`)에는 해당 배치의 `NN_batch_manifest.json`(원본 파일명·SHA-256·공식 제목·최종 경로·분류 기록)과 `NN_batch_index.md`, 그리고 그 배치가 처리한 분석/검증 MD의 byte-identical 복사본이 저장된다. 정식 산출물은 항상 `mds/papers/{category}`·`mds/reviews/{category}`가 canonical이며, 날짜 폴더는 그 배치 실행분만 모아둔 스냅샷이다.

## 출력 구조

논문별 MD 산출물은 `docs/ref_papers`의 기존 주제 분류를 그대로 따라 저장된다.

```text
Study-review/mds/
├── papers/
│   ├── 01_acl_gait_biomechanics_studies/
│   ├── 02_acl_gait_reviews_meta_analyses/
│   ├── 03_wearable_imu_and_portable_sensing/
│   ├── 04_machine_learning_and_deep_learning/
│   ├── 05_return_to_sport_and_functional_tests/
│   ├── 06_general_gait_and_other_knee_conditions/
│   └── 07_composite_kinematic_kinetic_scoring_indices/
├── reviews/
│   └── ...
└── papers_revised/
    └── ...
```

통합본은 기존처럼 `Study-review/mds/01_all_study_reviews.md`,
`02_referenceable_claims.md`, `03_summary_factcheck_report.md`,
`04_all_papers_revised.md`, `05_referenceable_claims_revised.md`에 생성된다.

기존 flat 산출물을 카테고리 구조로 정리하고 stale manifest 경로를 복구하려면:

```bash
python3 Study-review/scripts/05_categorize_existing_outputs.py --dry-run
python3 Study-review/scripts/05_categorize_existing_outputs.py
```

## 개별 스크립트 직접 실행

```bash
cd /Users/ryutt/Desktop/mini_ryutt/Walking

# 분석만
python3 Study-review/scripts/01_analyze_reference_papers.py --provider codex
python3 Study-review/scripts/01_analyze_reference_papers.py --provider codex --resume

# 검증만 (01 완료 후)
python3 Study-review/scripts/02_review_paper_summaries.py --provider codex

# AS-IS/TO-BE 수정 적용
python3 Study-review/scripts/03_apply_factcheck_revisions.py

# claims 재생성
python3 Study-review/scripts/04_rebuild_referenceable_claims.py

# 기존 산출물 카테고리 정리
python3 Study-review/scripts/05_categorize_existing_outputs.py
```

`--provider claude` 또는 `--provider antigravity`로 시작점을 바꿀 수 있다. 순환 순서는 선택한 provider부터 `codex → claude → antigravity` 순서를 따른다.

분석의 직접 인용문, 인용표기, 선행문헌 표기는 PDF에서 추출된 원문과 자동 대조한다. 일치하지 않는 결과는 성공으로 저장하지 않는다.
