# Leveraging explainable machine learning to identify gait biomechanical parameters associated with anterior cruciate ligament injury

> Kokkotis C, Moustakidis S, Tsatalas T, et al. (2022). *Scientific Reports*, 12, 6647. DOI: 10.1038/s41598-022-10666-2. PMID: 35459787. PMCID: PMC9026057.
> 원문: https://pmc.ncbi.nlm.nih.gov/articles/PMC9026057/
> 로컬 PDF: `docs/ref_papers/07_composite_kinematic_kinetic_scoring_indices/Leveraging explainable machine learning to identify gait biomechanical parameters associated with anterior cruciate ligament injury.pdf`

> 이 논문은 명명된 단일 composite score/index를 제시하지 않는다. SHAP은 feature contribution 설명값이며 환자별 정상성·회복 점수가 아니다.

## 검증 결과 (AS-IS → TO-BE)

| AS-IS | TO-BE |
|---|---|
| 94.95%를 일반화 가능한 3-group 성능처럼 기술 | 155/204/298개의 반복 trial을 70/30 무작위 분할한 test accuracy다. subject-grouped split이 보고되지 않아 동일 피험자 trial의 train/test 중복 가능성이 큼 |
| SHAP를 “연속적 기여도 점수”로 보아 composite score의 경계 사례로 포함 | SHAP은 model output에 대한 변수별 기여도다. 단일 gait severity/recovery score 기준에는 불일치 |
| “수술 후 회복이 감지됐다”로 기술 | H4와 GRF3가 ACLD-vs-CON에서는 유의했지만 ACLR-vs-CON에서는 유의하지 않았다는 횡단면 집단 비교다. 같은 환자의 수술 전후 변화가 아님 |
| 데이터셋 비공개 | 공개되어 있지는 않지만 저자는 합리적 요청 시 corresponding author에게 받을 수 있다고 명시 |

## 연구 목적

- ACLD, ACLR, 건강 대조군(CON)의 gait biomechanics를 분류하고, ReliefF와 SHAP으로 분류에 기여한 kinematic/kinetic 변수를 설명하는 것이 목적이다.
- 참가자는 151명: ACLD 44명, ACLR 54명, CON 53명이다. ACLD는 평균 부상 후 약 30일, ACLR은 수술 후 최소 6개월이었다.

## 방법

- 10-camera Vicon(100 Hz)과 Bertec force platform(1000 Hz)으로 self-selected walking을 측정했다.
- 분석 trial 수는 ACLD 155, ACLR 204, CON 298이었다. ACLD/ACLR은 involved limb, CON은 무작위 지정 limb를 사용했다.
- 25개 sagittal-plane/GRF discrete variables를 추출하고 [0,1] 정규화 후 ReliefF로 순위를 정했다.
- SVM, RF, XGBoost, neural network, KNN, logistic regression, decision tree, Naïve Bayes의 8개 모델을 비교했다.
- 평가 방법은 stratified training set을 포함한 stochastic 70/30 random split이었다. 논문은 split을 subject별로 묶었다고 보고하지 않았다.

## 결과

- SVM이 상위 21개 feature에서 test accuracy 94.95%로 최고였고 neural network가 92.89%였다.
- 3-class global SHAP mean magnitude가 0.3보다 큰 변수는 K2, H4, A3, GRF4, GRF7, K1, A4, GRF6이었다.
- CON-vs-ACLD local SHAP의 주요 변수는 H4, K7, GRF3였다.
- H4와 GRF3는 CON-vs-ACLD에서 유의했지만 CON-vs-ACLR에서는 각각 p=0.057, p=0.090이었다. 반면 K7과 GRF4는 ACLR에서도 CON과 유의하게 달랐다.

## 원문 근거

- “Support Vector Machines were proved to be the best performing model (accuracy of 94.95%).” (Abstract)
- “Features, that would have been neglected by the traditional statistical analysis, were identified as contributing parameters.” (Abstract)
- “a stochastic 70–30% random data split was applied” (Methods)

## 본 프로젝트와의 관련성

- ACLD/ACLR/CON 3-group 구성과 motion-capture/force-plate 변수 조합은 본 프로젝트와 가깝다.
- 그러나 이 연구는 분류기와 feature-attribution 연구다. HA-referenced 총점, domain subscore, 개인의 시간 경과에 따른 회복 궤적을 제공하지 않는다.
- 반복 trial random split은 본 프로젝트에서 요구하는 subject-safe validation과 다르다. 94.95%를 비교 기준으로 사용할 때 이 차이를 반드시 명시해야 한다.

## 저자가 명시한 한계

- gait biomechanics 결과는 연구·과제 간 일관성이 낮아 임상적 의미를 신중히 해석해야 한다.
- graft type, 개인별 coping strategy, 재활 protocol, 성별 차이가 해석에 영향을 줄 수 있다.
- SHAP은 개별 feature impact를 설명하지만 feature들이 결합되는 내부 메커니즘 전체를 밝히지는 못한다.

## 추가 방법론 한계(본 검증)

- repeated trial의 subject-grouped holdout/CV가 보고되지 않았다.
- hyperparameter optimization과 feature selection이 test split과 완전히 분리됐는지 상세 절차가 부족하다.
- ACLD와 ACLR은 동일인의 longitudinal pre/post pair가 아니다.
