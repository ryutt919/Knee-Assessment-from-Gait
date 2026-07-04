# 마할라노비스 Impairment Score 파이프라인 — ACL Gait Analysis

## Component Status

### Mahalanobis 샌드박스 개요
- **목적**: 관절각도 cycle waveform으로 HA 대비 gait deviation을 계산하고 실제 Mahalanobis quadratic form을 직접 분해
- **위치**: `Walking/Mahalanobis/`
- **상태**: v1 산출물은 legacy로 보존한다. v2 원격 재현은 `mahalanobis/v2-legacy`, 로컬 재현은 기존 `mahalanobis-v2.0.0` tag도 사용할 수 있다. 분리된 과거 tag 계보의 GitHub 대용량 제한 때문에 tag는 로컬 전용이다. v3.1은 `main`으로 승격됐고 `mahalanobis/v3.1-scalar-primary` 개발 브랜치도 동일 커밋을 유지한다. Mean-only session representation, scalar405 고정 primary, scalar+GVS54와 scalar+waveform5454 comparator, pair-aware repeated nested CV, HA LOO calibration과 direct D² contribution을 구현했다. 실제 36-session dry end-to-end 및 manifest 재현 검증을 통과했으며 full repeated run은 별도 실행이 필요하다.

### v3.1 Mean-only Input Profiles
- **Current value/logic**: `cycle mean → trial mean → equal trial mean`으로 한 session당 한 행을 만든다. `scalar_clean_multispeed` 405개가 출판 primary이며 `scalar_plus_gvs54` 459개와 `scalar_plus_waveform5454` 5,859개는 추가 가치 comparator다. Primary는 결과를 본 뒤 교체하지 않는다.
- **Implementation**: `Mahalanobis/scripts/08_v3_1_mean_pipeline.py`; scalar는 full-cycle mean waveform의 peak/min/ROM/IC angle/peak timing과 bounded injured-vs-contralateral symmetry로 생성한다. Fusion block은 training HA에서만 scale하고 feature 수 제곱근으로 정규화한 뒤 PCA+LedoitWolf distance를 계산한다.
- **Related files**: `Mahalanobis/configs/inputs/`, `Mahalanobis/versions/registry.yaml`, `Mahalanobis/tests/02_test_v3_1_pipeline.py`.
- **Rationale**: 해석 가능한 scalar를 사전 지정 primary로 고정하고 GVS 또는 전체 waveform이 scalar보다 제공하는 증분 정보를 동일 split에서 평가한다.

### v3.1 CV, Class Balance, and Score
- **Current value/logic**: Full은 ACL 27/HA 25 biological identities를 stratified outer 5-fold×5 repeats/inner 3-fold로 분할한다. ACLD·ACLR session은 별도 점수지만 같은 fold이며 두 session의 평가 weight 합은 HA identity 한 명과 동일하다.
- **Implementation**: Training HA만 reference/scaling/PCA/covariance에 사용한다. HA LOO robust log-distance를 `overall_z_deviation`으로 보정하고 `normality_score=100-10×z`를 표시한다. AUC, Hedges' g, paired ACLR−ACLD 변화와 paired ΔAUC bootstrap을 별도 estimate type으로 저장한다.
- **Related files**: `Mahalanobis/scripts/09_generate_v3_1_report.py`, `Mahalanobis/artifacts/v3.1/{profile}/{run_id}/`.
- **Rationale**: AUC는 known-group discrimination으로 제한해 해석하고 정상참조 거리, 종단 변화와 수치 안정성을 분리한다.

### v3.1 Side Contract and Reproducibility
- **Current value/logic**: ACL은 `ID.csv`의 실제 injured–contralateral, HA는 Right=pseudo-injured/Left=pseudo-contralateral이다. Pair side 불일치·누락은 실패하며 HA orientation swap sensitivity를 매 실행 저장한다.
- **Implementation**: `run_pipeline.py --pipeline-version 3.1 --input-profile all`; `--resume`은 hash가 같은 완료 run만 재사용하고 `--from-run`은 manifest 설정으로 새 immutable run을 만든다.
- **Related files**: `Mahalanobis/02_plan_v3.md`, `Mahalanobis/versions/registry.yaml`, annotated tag `mahalanobis-v2.0.0`.
- **Rationale**: arbitrary HA pseudo-side 영향을 숨기지 않고 코드·데이터·split 계보와 rollback을 보장한다.

### v2 Balance Modes and CLI
- **Current value/logic**: `mean_aggregate`는 cycle→trial→condition 평균, `inverse_weight`는 모든 cycle을 유지하면서 계층 inverse-count weight를 모델 적합과 score에 적용한다. 기본 `both`, primary는 `inverse_weight`다.
- **Implementation**: `Mahalanobis/scripts/06_v2_nested_pipeline.py`; 실행은 `../.venv/bin/python run_pipeline.py --mode dry|full --balance-mode mean_aggregate|inverse_weight|both`.
- **Related files**: `Mahalanobis/run_pipeline.py`, `Mahalanobis/scripts/07_generate_v2_report.py`, `Mahalanobis/tests/01_test_v2_pipeline.py`.
- **Rationale**: 집계 기반의 보수적 결과와 cycle 변동을 보존한 weighted 결과를 같은 identity splits에서 비교한다.

### v2 Data, CV, and Score
- **Current value/logic**: ACLD/ACLR 27 matched pairs+HA25, 52 identities; outer/inner identity split; slow/normal/fast distance와 robust signed deviation RMS total을 저장한다.
- **Implementation**: QC 실패 시 `qc_audit.json` 기록 후 중단하며 dry/full과 balance mode별 Optuna DB·model·OOF를 `artifacts/{run_id}/`에 격리한다.
- **Related files**: `data/processed/slim_gait.parquet`, `data/processed/cycle_waveforms_101.parquet`, `data/processed/id_pairing_summary.csv`.
- **Rationale**: longitudinal leakage, pilot/full lineage 혼합, clipping floor와 stride-count 과대가중을 제거한다.

---

### 핵심 데이터 발견사항
- **raw_merged.parquet 그룹명**: `ACLD` / `ACLR` / `Healthy adults` / `Healthy adolescents`
  - `Healthy adults` → `HA` 로 매핑 (분석 대상 정상 성인)
  - `Healthy adolescents` → 분석에서 제외
- **추출 컬럼**: 메타(5) + footContacts(4) + jointAngle_42~62(21) + sensorFreeAcceleration(21) + sensorOrientation(28) + sensorMagneticField(21) = **100 컬럼**
- **Waveform 특징 벡터 차원**: 채널 수 × 101 ≈ **7,979 차원**
- **IMU block missingness**: 34 session ID(ACLD 19, ACLR 7, HA 8), 305 trial, 3,427 stride에서 IMU 70채널×101시점 전체가 null; 관절각도는 존재
- **Subject-mean AUC 0.5039**: 92 session형 subject_id 각각의 모든 속도·trial·양발 stride raw distance 단순 평균에 대한 AUC이며 biological identity/pair 집계가 아님
- **Fold class 구성**: GroupKFold는 label stratification을 하지 않아 validation HA session이 fold별 9/5/3/4/4로 불균형

---

### 파이프라인 스크립트

#### `scripts/00_extract_subset.py`
- **기능**: raw_merged.parquet (6.29 GB) → raw_subset_mahalanobis.parquet (~200 MB) 1회 추출
- **테스트 모드**: 그룹별 3명씩 샘플링 (`--test`)
- **그룹 정규화**: `_normalize_groups()` 함수로 Healthy adults→HA, Healthy adolescents 제거
- **출력**: `data/processed/raw_subset_mahalanobis.parquet`

#### `scripts/01_data_preprocessing.py`
- **기능**: Stride 분할 → 101pt 보간 → Waveform 특징 행렬 생성
- **Stride 감지**: `footContacts_0` (좌발), `footContacts_2` (우발) Heel Strike 상승 엣지
- **Trim**: 앞뒤 2 Strides 제거
- **관절 매핑**: 오른발(42~50) / 왼발(54~62) 컬럼 분리 사용
- **출력**: `data/processed/mahalanobis_features.parquet` (행=stride, 열=7987)

#### `scripts/02_mahalanobis_pipeline.py`
- **기능**: GroupKFold(subject_id) CV → PCA + MCD → Impairment Score
- **PCA 상한**: `min(n_ha // 5, 100)` — MCD의 n >> p 조건 충족 강제
- **MCD**: `support_fraction = 0.75` (MinCovDet, sklearn)
- **Impairment Score**: `max(0, (D_M - mu_HA) / sigma_HA)` Z-score 형태
- **출력**: `results/oof_results.parquet`, `results/02_evaluation_report.md`

#### `scripts/03_optuna_optimization.py`
- **기능**: Optuna TPE 기반 파이프라인 하이퍼파라미터 최적화
- **탐색 공간**: scaling(3) × pca_k_method(3) × speed_filter(4) × distance_metric(2)
- **저장**: `results/optuna_mahalanobis.db` (SQLite 영구), `results/01_optuna_best_params.json`
- **호출 방식**: `02_mahalanobis_pipeline.py`를 `importlib`으로 동적 로드

#### `scripts/04_shap_analysis.py`
- **기능**: XGBoost Proxy 모델 학습 → TreeSHAP → 채널별 기여도 시각화
- **Proxy 방식**: 전체 데이터 재적합 score를 XGBoost로 근사. 동일 데이터에서 상관만 출력하며 수치가 artifact에 저장되지 않아 out-of-sample fidelity는 확인되지 않음
- **출력**: `results/03_shap_interpretation/summary_plot.png` + `{subject_id}_waterfall.png`

#### `scripts/05_generate_detailed_report.py`
- **기능**: 코드, parquet schema/OOF, Optuna SQLite, ID pairing, SHAP PNG를 읽기 전용으로 교차검증해 단일 독립 HTML 기술 감사 보고서 생성
- **통계 단위**: stride 결과와 session/biological-identity 집계를 분리하고, AUC CI는 identity-cluster bootstrap으로 계산
- **설명 범위**: Mahalanobis 거리의 raw→stride→HA scaler/PCA/MCD 경로, subject-mean/Optuna/identity leakage 의미, IMU 결측 ID 표, 속도 처리, clipping 정보 손실, fold class 불균형, 개선 항목별 구현·검증 기준
- **출력**: `htmls/01_detailed_experiment_analysis.html`

#### `run_pipeline.py`
- **기능**: v2와 v3.1 파이프라인을 version flag로 통합 실행하고 immutable resume/manifest 재현을 관리
- **옵션**: `--pipeline-version 2.0|3.1`, `--input-profile ...|all`, `--mode dry|full`, `--cv-repeats`, `--from-run`, `--resume`

---

### 알려진 이슈 및 해결 기록

| 이슈 | 원인 | 해결 |
|------|------|------|
| 테스트 서브셋에 HA 없음 | raw_merged 그룹명이 `Healthy adults`라 `HA` 필터 미작동 | `_normalize_groups()` 함수 추가 |
| Kaiser criterion 폭발 (k=128) | HA 학습 수 대비 PC 수 너무 많아 MCD 불안정 | `hard_max = min(n_ha//5, 100)` 상한 추가 |
| 03 Optuna 모듈 import 오류 | 작성 중 잘못된 import 코드 잔류 | `importlib.util` 동적 로드 방식으로 교체 |
| 한글 폰트 경고 | matplotlib 기본 폰트 DejaVu가 한글 미지원 | 경고 발생하나 저장 문제 없음 (향후 fontprops 설정 권장) |

---

### 테스트 검증 결과 (파일럿: 그룹별 3명 = 9명)
- 생성 Stride 수: 1,013개 (HA:347, ACLD:369, ACLR:297)
- Waveform 특징 차원: 7,979
- GroupKFold 3-fold OOF AUC-ROC: **0.82**
- Optuna 5 trials 최적 AUC: 0.6818 (3명이라 fold 다양성 제한)
- SHAP 출력: Summary Plot 1개 + Waterfall Plot 5개 정상 생성

---

### 다음 작업 권장사항
1. **P0 — Optuna 계보 격리**: test/full마다 별도 DB·study를 사용하고 dataset hash를 기록
2. **P0 — Pair-aware nested CV**: ACLD/ACLR biological identity를 같은 outer fold에 고정하고 inner fold에서만 tuning
3. **P0 — ID 계보 수정**: `ID.csv`에 없는 ACLR38과 features에 없는 ACLR36의 원인을 해결한 뒤 side mapping 재검증
4. **P0 — 결측 처리 통일**: OOF와 SHAP의 imputation/scaling 순서를 train-fold 기준으로 일치시키고 group별 센서 결측 원인을 감사
5. **P1 — 점수 명칭/검증**: 외부 임상 anchor 전에는 HA-referenced gait deviation score로 제한하고 identity-safe calibration과 cluster CI 사용
6. **P2 — SHAP 재설계**: run-specific 출력 폴더, grouped holdout proxy fidelity, 한글 폰트, 실제 시점별 설명을 구현

---

## Change History

| Timestamp | Task | Change Summary |
|-----------|------|----------------|
| 2026-07-02 01:16 | 0702_Mahalanobis 파이프라인 구현 | 전체 5개 스크립트 + run_pipeline.py 신규 생성, 테스트 검증 완료 |
| 2026-07-02 11:26 | 0702_Mahalanobis 정밀 기술 감사 | full 기본 OOF AUC 0.4991 확인; 0.7716이 9-session test Optuna trial임을 규명; 21/26 확인 종단쌍 fold 분리, ACLR38/36 계보 불일치, SHAP proxy 한계 문서화; 재현 가능한 단일 HTML 보고서 추가 |
| 2026-07-02 12:43 | Mahalanobis 감사 보고서 설명 확장 | subject-mean·shared Optuna·종단 identity·거리 계산·속도·clipping·fold 불균형 설명 추가; IMU 전체결측 34 ID/305 trial/3,427 stride 표와 P0–P2 개선 구현·검증 기준 상세화 |
| 2026-07-04 00:25 | Mahalanobis v2 고도화 | matched joint-only cohort, identity nested CV, mean/inverse balance flag, 속도별·total score, run-scoped artifacts, direct D² contribution, dry end-to-end 검증 추가 |
| 2026-07-04 17:24 | Mahalanobis v3.1 mean-only scalar primary | scalar405 primary와 GVS54/waveform5454 fusion comparator, identity-balanced repeated nested CV, paired ΔAUC, strict side contract, HA side-swap sensitivity, version registry와 manifest 재현 구현; 실제 12 ACL/12 HA identity dry E2E 및 15 tests 통과 |
| 2026-07-04 17:37 | Mahalanobis v3.1 main 승격 | v3.1을 `main` current로 승격하고 v3 직전 main `eac10aa`를 원격 `mahalanobis/v2-legacy`로 보존; v3.1 개발 브랜치 유지; 과거 분리 이력의 대용량 제한으로 annotated v2 tag는 로컬 전용 유지 |
