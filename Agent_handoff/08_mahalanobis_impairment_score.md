# 마할라노비스 Impairment Score 파이프라인 — ACL Gait Analysis

## Component Status

### 0702_Mahalanobis 샌드박스 개요
- **목적**: raw IMU 센서 원본 데이터 + 관절각도 Waveform으로 정상인(HA) 대비 ACL 손상군(ACLD/ACLR)의 보행 손상도(Impairment Score) 산출 및 SHAP 해석
- **위치**: `Walking/0702_Mahalanobis/`
- **상태**: 전체 파이프라인 구현 완료, 테스트 검증 통과 (OOF AUC-ROC 0.82 on pilot 9 subjects)

---

### 핵심 데이터 발견사항
- **raw_merged.parquet 그룹명**: `ACLD` / `ACLR` / `Healthy adults` / `Healthy adolescents`
  - `Healthy adults` → `HA` 로 매핑 (분석 대상 정상 성인)
  - `Healthy adolescents` → 분석에서 제외
- **추출 컬럼**: 메타(5) + footContacts(4) + jointAngle_42~62(21) + sensorFreeAcceleration(21) + sensorOrientation(28) + sensorMagneticField(21) = **100 컬럼**
- **Waveform 특징 벡터 차원**: 채널 수 × 101 ≈ **7,979 차원**

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
- **Proxy 방식**: f(x) = Impairment_Score 를 XGBoost로 근사 (R=1.00 확인)
- **출력**: `results/03_shap_interpretation/summary_plot.png` + `{subject_id}_waterfall.png`

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
1. **전체 데이터 실행**: `python run_pipeline.py` (서브셋 추출 ~10분, 전처리 ~30분 예상)
2. **Optuna 확장**: `--trials 100` 이상으로 본격 최적화
3. **한글 폰트 설정**: matplotlib에 Noto Sans KR 또는 Apple Gothic 적용
4. **속도별 분리 분석**: `speed_filter` 조합별 성능 비교 리포트
5. **Impairment Score 임상 해석**: 특정 ACL 피험자의 Waterfall Plot을 논문 Figure로 활용

---

## Change History

| Timestamp | Task | Change Summary |
|-----------|------|----------------|
| 2026-07-02 01:16 | 0702_Mahalanobis 파이프라인 구현 | 전체 5개 스크립트 + run_pipeline.py 신규 생성, 테스트 검증 완료 |
