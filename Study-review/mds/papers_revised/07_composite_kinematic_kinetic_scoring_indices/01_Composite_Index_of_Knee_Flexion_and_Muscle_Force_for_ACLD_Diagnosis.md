# Leveraging Multivariable Linear Regression Analysis to Identify Patients with Anterior Cruciate Ligament Deficiency Using a Composite Index of the Knee Flexion and Muscle Force

> Li H, Huang H, Ren S, Rong Q. (2023). *Bioengineering*, 10(3), 284. DOI: 10.3390/bioengineering10030284. PMID: 36978675. PMCID: PMC10045096.
> 원문: https://pmc.ncbi.nlm.nih.gov/articles/PMC10045096/
> 로컬 PDF: `docs/ref_papers/01_acl_gait_biomechanics_studies/Leveraging Multivariable Linear Regression Analysis to Identify Patients with Anterior Cruciate Ligament Deficiency Using a Composite Index of the Knee Flexion and Muscle Force.pdf`

## 검증 결과 (AS-IS → TO-BE)

| AS-IS | TO-BE |
|---|---|
| 최종 모델과 “복합 지수만” 모델의 81.4%를 같은 결과처럼 기술 | 최적 6-feature 모델(기본 PCA feature 3개 + composite-index PC 3개)은 81.4%; composite-index PC만 쓰면 3개일 때 79.1%, 8개일 때 81.4%로 구분 |
| 근력을 “체중/최대근력 대비”로 정규화 | 먼저 체중(중력)으로 무차원화했고, 이후 각 gait cycle 내 최대값으로 knee flexion과 muscle-force 파형을 0–1 정규화 |
| 34명을 그대로 CV한 것처럼 기술 | 분석 단위는 43개 leg sample(ACLD 25명의 affected leg + 대조군 9명의 양측 18 legs) |
| 단일 임상 점수로 해석 | 논문이 “composite index”라고 명명했지만 실제 산출물은 PCA 성분 최대 8개인 다변량 feature set이며, GDI/GPS 같은 단일 환자 점수는 아님 |
| 성능을 일반화된 진단 정확도로 해석 | 소표본 내부 5-fold CV 결과이며 외부 검증이 없음. feature 선별/PCA의 fold 내부 재학습 여부와 대조군 양측 leg의 동일 fold 배정도 보고되지 않음 |

## 연구 목적

- 조깅 중 무릎 굴곡과 모델 추정 근력을 결합한 PCA 기반 feature set을 만들고, 다중선형회귀로 ACLD와 건강 대조군을 구분하는 것이 목적이다.
- 참가자는 ACLD 25명과 건강 대조군 9명이며 모두 젊은 남성이었다. ACLD의 부상 후 기간은 평균 11.10 ± 6.87개월이었다.

## 방법

- Vicon MX 8-camera motion capture와 AMTI force plate(1000 Hz)를 사용해 10 m 경로의 self-selected jogging을 측정했다. 각 참가자에서 성공 trial 5개를 수집해 평균했다.
- AnyBody 6.0.5 inverse-dynamics model로 무릎 관련 13개 근육의 힘을 추정했다.
- 기본 feature 3개는 stance 평균 muscle force, swing 평균 muscle force, swing knee flexion을 각각 PCA로 요약한 첫 성분이다.
- 별도의 composite-index matrix는 ACLD와 대조군의 knee-flexion/muscle-force 파형에서 pointwise t-test가 유의하고 p값이 최소인 characteristic point들로 구성했다. 이 행렬을 PCA로 줄였을 때 90% 분산 기준 8개 성분이 생성됐다.
- 선형회귀의 target은 ACLD=1, Control=-1이었다. 43개 leg sample에 5-fold CV를 적용했다.

## 결과

- 최적 구성은 기본 feature 3개와 composite-index PC 3개를 합친 6 predictors였다: accuracy 81.4%, precision 87.0%, recall 80.0%, specificity 83.3%, F1 83.3%.
- composite-index PC만 사용하면 1개 72.1%, 3개 79.1%, 8개 81.4%였다.
- 전체 43개 sample에 다시 적합한 회귀모델에서 Composite Index 1 계수는 -2.3055 (p<0.001), Composite Index 2는 -1.5697 (p=0.006)이었다. 이 계수와 R²=0.542는 CV fold별 추정치가 아니라 전체-sample fit 결과다.

## 원문 근거

- “The accuracy rate of the regression model in diagnosing patients with ACLD was 81.4%.” (Abstract)
- “The composite index produced eight features when 90% of the PCA information content was preserved.” (Results)
- “The composite index and characteristic points can help avoid complex subjective diagnosis in clinical practice.” (Conclusions)

## 본 프로젝트와의 관련성

- kinematic waveform과 model-derived kinetics를 함께 축약한다는 문제의식은 Gait Normality Score와 가깝다.
- 그러나 이 연구의 composite index는 건강 기준 거리 점수가 아니라 supervised group-difference point 선별 후 얻은 여러 PCA 성분이다. 따라서 HA-referenced 정상성 점수나 longitudinal recovery score의 직접 선행 예로 부르기에는 구조적 차이가 크다.
- characteristic point 선별과 PCA가 CV 전에 전체 데이터에서 수행됐다면 정보 누출이 생긴다. 논문은 fold별 재학습 절차를 명시하지 않아 81.4%를 leakage-safe OOF 성능으로 간주할 수 없다.

## 저자가 명시한 한계

- 일부 ACLD 환자에 반월판 손상이 동반됐지만 하위군화하지 못했다.
- 부상 후 기간이 균질하지 않아 보상 패턴과 데이터 품질에 영향을 줄 수 있다.
- 계산된 근력을 검증할 EMG가 없었다.
- composite index의 walking 적용은 추가 검증이 필요하다.

## 추가 방법론 한계(본 검증)

- 표본이 작고 대조군이 9명뿐이다.
- 같은 대조군의 양측 leg가 서로 독립인 18 sample처럼 취급됐다.
- 독립 외부 코호트 검증, longitudinal 회복 검증, calibration 분석이 없다.
