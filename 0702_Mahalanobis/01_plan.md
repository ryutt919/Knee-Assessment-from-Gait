# 01_plan.md — IMU 센서 및 관절 각도 통합 마할라노비스 거리 기반 보행 점수화 기획서

이 문서는 `raw_merged.parquet` 데이터를 사용하여 실제 IMU 센서 원본 값(가속도, 오리엔테이션 등)과 관절 각도 데이터를 동기화 및 통합하고, 정상 대조군(HA) 대비 ACL 손상군(ACLD/ACLR)의 보행 손상도(Impairment Score)를 계산하며, SHAP를 통해 그 기여도를 해석하는 상세 실행 계획을 설명합니다.

---

## 1. 목표 (Goal)
`raw_merged.parquet`에서 추출한 실제 IMU 센서 원본 데이터(가속도, 오리엔테이션 등)와 Xsens 추정 관절 각도 데이터를 보행 주기(Stride)별로 분할 및 101 포인트 정규화하여 통합 Waveform 특징 공간을 구축합니다. 정상인(HA) 분포에 기반한 Robust MCD 마할라노비스 거리를 산출하여 **정상 범위에서 벗어난 정도를 나타내는 Impairment Score (보행 손상 점수)**를 정의합니다. 그리고 **SHAP (SHapley Additive exPlanations)** 분석을 적용하여 어떤 센서/관절 채널의 어느 보행 시점(0%~100%)이 환자의 보행 손상 점수 상승에 기여했는지 개별 피험자 수준에서 상세하게 해석합니다.

---

## 2. 이론적 배경 및 아키텍처 (Background & Architecture)

참조 논문에 기반한 핵심 설계 요소:
1. **Lewko (2024)**: *Motion Healthiness Score (MHS)* 제안:
   - PCA를 통한 차원 축소 (Guttman-Kaiser criterion, 고유값 $\ge 1$인 주성분만 유지).
   - 정상 대조군 분포에 대해 **Robust MCD (Minimum Covariance Determinant)** 방법(support fraction = 0.75)을 사용하여 공분산 행렬을 추정, 아웃라이어 영향 배제.
   - 마할라노비스 거리 값을 카이제곱 분포의 95% 분위수(정상 임계값 $t_h$)와 환자군 KDE 분포의 95% 분위수(비정상 임계값 $t_s$)를 경계로 삼아 0~10점 사이의 점수로 Piecewise Linear Interpolation 변환.
2. **Liu et al. (2020)**: *Normalcy Index (NI)* 구현:
   - 대조군의 평균 및 표준편차를 기준으로 전체 변수 표준화.
   - PCA 수행 후, 각 주성분 점수를 해당 고유값의 제곱근(표준편차)으로 나누어 스케일링.
   - 스케일링된 주성분 점수들의 제곱합 $d$ 계산. (수학적으로 원래 공간에서의 마할라노비스 거리 제곱 $D_M^2$과 완전히 동일함).

### Impairment Score & SHAP 해석 개념
- **Impairment Score (손상 점수)**: 정상인(HA)의 Robust MCD 마할라노비스 거리 분포의 평균값을 기준으로 삼아, 정상 영역에서 멀어질수록 점수가 누적 증가하도록 스케일링합니다. ($Z_{imp} = (D_M - \mu_{HA\_dist}) / \sigma_{HA\_dist}$)
- **SHAP 해석**: 마할라노비스 거리 연산 파이프라인(특징 입력 $\rightarrow$ PCA 투영 $\rightarrow$ MCD 마할라노비스 계산)을 하나의 함수 $f(x) = D_M(x)$로 모델링하고, SHAP `KernelExplainer` 또는 대리 회귀 모델(XGBoost Proxy + `TreeExplainer`)을 사용하여 909차원 Waveform의 각 포인트별 기여도를 분석합니다.

### 제안하는 파이프라인 아키텍처

```mermaid
graph TD
    A[raw_merged.parquet] -->|필요 컬럼만 효율적 로드| B[보행 주기 Stride 추출 및 정규화]
    B -->|IMU 가속도/오리엔테이션/관절각도 통합| C[Waveform 특징 결합: 101pt × Channels]
    C --> D[subject_id 기준 GroupKFold 분할]
    D --> E[HA 학습 데이터 fold에 대해 PCA 적합]
    E --> F[HA 학습 데이터 fold에 대해 Robust MCD 공분산 행렬 추정]
    F --> G[검증 fold의 HA 및 ACL 데이터에 대해 마할라노비스 거리 계산]
    G --> H[정규화된 Impairment Score 산출]
    H --> I[HA vs ACL 분류 및 Impairment 스코어 검증: AUC-ROC]
    I --> J[SHAP Explainer 적합: f_x = Impairment_Score]
    J --> K[개별 피험자별 기여도 시각화 및 해석]
    I --> L[Optuna 최적화 루프]
    L -->|센서/관절 채널 조합 및 파라미터 조정| C
```

---

## 3. 구현 단계 (Implementation Steps)

샌드박스 폴더는 상위 `Walking/` 디렉토리 하위에 `0702_Mahalanobis/`로 생성하여 다음과 같이 구조화합니다.

```
Walking/
└── 0702_Mahalanobis/
    ├── 01_plan.md                         # 본 기획안 (한국어)
    ├── scripts/
    │   ├── 00_extract_subset.py           # raw_merged.parquet에서 핵심 변수만 추출하여 subset parquet 생성
    │   ├── 01_data_preprocessing.py       # Stride 분할, Trim, 101pt 보간 및 피처 추출
    │   ├── 02_mahalanobis_pipeline.py     # PCA, MCD 공분산, 마할라노비스 거리 및 Impairment Score 계산
    │   ├── 03_optuna_optimization.py      # Optuna 기반 파이프라인 하이퍼파라미터 최적화
    │   └── 04_shap_analysis.py            # SHAP를 활용한 Impairment Score 기여 피처 해석 및 시각화
    └── results/
        ├── 01_optuna_best_params.json     # 최적화 탐색 결과 요약
        ├── 02_evaluation_report.md        # 교차 검증 결과 요약 보고서
        └── 03_shap_interpretation/        # 피험자별 SHAP Force/Summary Plot 저장 디렉토리
```

### 0단계: 데이터 서브셋 추출 (`00_extract_subset.py`)
- `raw_merged.parquet` (6.29 GB)에서 분석에 사용할 필수 컬럼들만 선별하여 별도의 경량화 파일인 `data/processed/raw_subset_mahalanobis.parquet` (약 200MB 수준)으로 1회성 추출을 수행합니다.
- 추출할 컬럼 목록:
  - 메타: `subject_id`, `group`, `speed`, `file_name`, `time_ms`
  - 발 접촉: `footContacts_0`, `footContacts_1`, `footContacts_2`, `footContacts_3`
  - IMU 원본: `sensorFreeAcceleration_0~20`, `sensorOrientation_0~27`, `sensorMagneticField_0~20`
  - 관절 각도: `jointAngle_42~62` (18개 컬럼)

### 1단계: 데이터 전처리 (`01_data_preprocessing.py`)
- 생성된 경량 `raw_subset_mahalanobis.parquet` 로드.
- 좌/우측 발 접촉 신호(`footContacts_0`, `footContacts_2`)의 heel strike 시점(0에서 1로 상승하는 엣지)을 검출하여 Stride 분할.
- 가속/감속 구간의 영향을 최소화하기 위해 각 Trial별 앞/뒤 2 strides 제거 (Trim).
- 선택된 IMU 센서 원본 채널들과 관절 각도 채널의 데이터를 Stride당 101 포인트로 선형 보간하여 정규화.
- 최종 특징 벡터:
  - 101pt Waveform 특징 벡터: (선택된 IMU 및 관절 채널 수 $\times$ 101) 차원의 1D 벡터 생성.

### 2단계: 공분산 및 마할라노비스 거리 파이프라인 (`02_mahalanobis_pipeline.py`)
- 피험자 기준의 데이터 누수 방지를 위해 `GroupKFold(n_splits=5)`(subject_id 기준) 적용.
- 각 fold에서:
  1. 학습 세트의 정상 대조군(HA) 데이터만을 사용하여 스케일러 및 PCA 적합.
  2. Kaiser criterion 또는 지정 비율을 기준으로 주요 주성분(PC) 개수 $k$ 선정.
  3. 학습 세트 HA의 PC Score 분포에 대해 **Robust MCD 방식(support fraction = 0.75 고정)으로 공분산 행렬 및 평균을 추정**합니다.
  4. 검증 세트의 HA 및 ACL 데이터를 주성분 공간으로 투영 후 마할라노비스 거리 계산.
  5. **Impairment Score** 산출: 정상군의 평균 마할라노비스 거리와 표준편차를 기준으로 Z-score 형태의 양수 점수로 정규화하여 정상에서 멀어질수록 점수가 증가하게 만듭니다.
- 전체 Fold의 OOF(Out-of-Fold) 예측치에 대해 HA vs ACL 분류 성능(AUC-ROC) 및 그룹별 Impairment Score 편차를 평가합니다.

### 3단계: Optuna 최적화 (`03_optuna_optimization.py`)
Optuna 최적화 탐색 공간:
- **입력 데이터 채널 조합 최적화**:
  - `use_acceleration`: 가속도 센서 데이터 포함 여부 `[True, False]`
  - `use_orientation`: 오리엔테이션 센서 데이터 포함 여부 `[True, False]`
  - `use_joint_angles`: 관절 각도 데이터 포함 여부 `[True, False]`
  - `joints_subset`: 관절 각도 사용 시 9개 관절 조합 필터링 (각 관절의 활성화 여부)
- **파이프라인 하이퍼파라미터 최적화**:
  - `scaling`: `["zscore", "robust", "minmax"]`
  - `pca_k_method`: `["kaiser", "variance_ratio", "fixed"]`
  - `pca_variance_ratio` (variance_ratio 사용 시): $0.70 \sim 0.99$
  - `pca_fixed_k` (fixed 사용 시): $2 \sim 100$
  - `speed_filter`: `["all", "normal", "fast", "slow"]`
  - `distance_metric`: `["mahalanobis", "squared_mahalanobis"]`

**최적화 목적 함수**: 통합 Waveform 입력에 대해 Robust MCD (support fraction = 0.75) 기반 마할라노비스 거리를 사용한 Out-of-Fold 분류 AUC-ROC를 최대화.

### 4단계: SHAP 기반 해석 (`04_shap_analysis.py`)
- 최적화된 마할라노비스 파이프라인 $f(x) = \text{Impairment\_Score}(x)$ 모델을 설정합니다.
- SHAP `KernelExplainer`를 학습 데이터의 HA 대표 샘플들을 배경(Background)으로 삼아 초기화합니다.
- 검증 데이터셋의 각 피험자(특히 ACLD/ACLR 손상 환자군)에 대해 SHAP value를 구합니다.
- **분석 결과 시각화**:
  - **Summary Plot**: 전체 피험자군에서 보행 손상 점수(Impairment Score)를 높이는 데 가장 지배적인 기여를 하는 IMU 채널 및 관절 종류를 규명합니다.
  - **Force Plot / Waterfall Plot (개별 피험자)**: 특정 환자의 보행 주기(0%~100%)에서 어떤 센서의 어떤 시점이 비정상성 점수를 증가시켰는지 시각화합니다. (예: "피험자 ACLD10의 knee_flexion 40%~60% stance phase 구간이 손상도 Z-score를 +2.5점 증가시킴"과 같이 정량적 해석 가능)

---

## 5. 기록 및 검증 (Logging & Verification)
- 모든 Optuna trial 결과는 SQLite 파일(`optuna_mahalanobis.db`)에 기록 및 영구 보존.
- 최적의 하이퍼파라미터 및 최고 메트릭 정보를 `results/01_optuna_best_params.json`에 저장.
- MHS와 NI의 ACL 분류 변별력 비교 분석 결과를 `results/02_evaluation_report.md`에 자세히 기록.
- 개별 피험자의 보행 손상도 및 원인 해석 그래프를 `results/03_shap_interpretation/`에 이미지 파일로 저장.
