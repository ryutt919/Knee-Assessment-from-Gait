# 마할라노비스 Impairment Score 파이프라인 — ACL Gait Analysis

## Component Status

### Mahalanobis 샌드박스 개요
- **목적**: raw IMU 센서 원본 데이터 + 관절각도 Waveform으로 정상인(HA) 대비 ACL 손상군(ACLD/ACLR)의 보행 손상도(Impairment Score) 산출 및 SHAP 해석
- **위치**: `Walking/Mahalanobis/`
- **상태**: 구현 및 전체 데이터 실행 산출물은 존재하나, 2026-07-02 기술 감사에서 Optuna test/full 계보 혼합, biological-identity fold overlap, IMU block missingness, class-unstratified folds, clipped-score floor effect, SHAP proxy 한계가 확인됨. 현재 재현 가능한 full 기본 OOF distance AUC는 0.4991이며 최종 검증 성능으로 사용 불가

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
- **기능**: 전체 파이프라인 통합 실행
- **옵션**: `--test`, `--skip 00 01`, `--trials N`

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
