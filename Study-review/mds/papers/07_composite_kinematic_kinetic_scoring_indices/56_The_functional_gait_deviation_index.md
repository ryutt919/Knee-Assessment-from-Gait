# The functional gait deviation index

Minhas, S. K., Sangeux, M., Polak, J., & Carey, M. (2026). The functional gait deviation index. Journal of Applied Statistics, 53(3), 391–411. https://doi.org/10.1080/02664763.2025.2514150

## 서지정보

- 저자: Sajal Kaur Minhas, Morgan Sangeux, Julia Polak and Michelle Carey
- 연도: 2026
- 저널: Journal of Applied Statistics
- DOI: 10.1080/02664763.2025.2514150
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/07_composite_kinematic_kinetic_scoring_indices/The functional gait deviation index.pdf
- 분석 provider: antigravity

> **한국어 제목**: 기능적 보행 편차 지수

## 분류 태그

- ACL 연구: false
- IMU 사용: false
- 보행 데이터: true
- Score 제시: true

## 연구 목적

- 방대하고 복잡한 보행 분석 데이터에서 환자의 보행 패턴이 건강한 대조군의 평균 프로파일로부터 이탈한 정도를 단일 숫자로 요약하여 정량화하는 지표를 제안하는 것이다. _(근거: PAGE 1, ABSTRACT)_
  - 근거 원문: “Due to the quantity and complexity of the data, it is useful to calculate the amount by which a subject’s gait deviates from an average normal profile and to represent this deviation as a single number.”
- 관절 및 평면별 보행 움직임의 고유한 부드러움과 관절 간의 상호 상관관계를 반영하여 보행 편차를 보다 정확히 평가하는 다변량 함수형 주성분 분석(MFPCA) 기반의 기능적 보행 편차 지수(FGDI)를 개발하는 것이다. _(근거: PAGE 1, ABSTRACT)_
  - 근거 원문: “Utilizing a multivariate functional principal component analysis we propose the functional gait deviation index (FGDI). FGDI accounts for the intrinsic smoothness of the gait movement at each joint/plane and the potential co-variation between the joints.”

## 연구 설계와 대상

- 21명의 우수수형 특발성 파킨슨병(PD) 환자군(여성 5명, 남성 16명)과 42명의 성인 건강한 대조군(여성 18명, 남성 24명)을 대상으로 평가를 수행하였다. _(근거: PAGE 7, Section 3)_
  - 근거 원문: “It includes data from 21 right-handed idiopathic PD individuals (5 females and 16 males), characterized by the following demographics: average age of 65 ± 10 years, average height of 166.5 ± 7.1cm, and average mass of 71.89 ± 12.37kg. Additionally, a healthy control dataset is available from [6], comprising 42 adults (18 females and 24 males).”
- 18명의 일측성 대퇴부 절단 환자군(여성 3명, 남성 15명)을 분석에 포함하여 임상적 기능 수준인 K-Level(K2 대 K3)에 따라 나누어 평가하였다. _(근거: PAGE 10, Section 4)_
  - 근거 원문: “This dataset includes 18 individuals with unilateral above-knee amputations, comprising 3 females and 15 males, with the following demographics: average age of 52 ± 16 years, height of 175.8 ± 9cm, and mass of 88.85 ± 22.61kg.”

## 방법

- FGDI 산출을 위해 양측 다리 통합 평가, 개별 다리 평가, 단일 관절/평면별 평가 등 세 가지 범주로 기학적 변수 하위집합을 설계하여 적용한다. _(근거: PAGE 4, Section 2)_
  - 근거 원문: “For the computation of the FGDI, we concentrate on three distinct subsets of these variables, which are outlined as follows:”
- 통합(Combined) 접근법은 골반을 포함해 좌우 15개의 기학적 변수를 모두 사용하여 전체적인 보행 이상 상태를 하나의 중증도 수치로 제시한다. _(근거: PAGE 4, Section 2)_
  - 근거 원문: “This procedure results in the selection of fifteen kinematic variables, designated as u = 1,...,15. This approach yields a measure of severity by collectively considering both legs, thereby providing an overall assessment of gait abnormality.”
- 개별 다리(Individual Leg) 접근법은 특정 다리의 9개 기학적 변수만을 사용하여 편측성 보행 병리를 개별적으로 정밀 평가한다. _(근거: PAGE 4, Section 2)_
  - 근거 원문: “This method provides a measure of gait pathology for each leg individually, facilitating a detailed assessment of gait abnormality in each leg.”
- 관절/평면별(Joint/Plane Specific) 접근법은 개별 기학적 변수 1개씩을 독립적으로 평가하여 특정 관절이나 움직임 평면 수준의 이상을 파악한다. _(근거: PAGE 5, Section 2)_
  - 근거 원문: “This method offers a detailed evaluation of gait abnormalities at the individual level of each joint or plane.”

## 핵심 결과

- 파킨슨병 환자 중 동결 보행 증상이 있는 집단(freezers)이 없는 집단(non-freezers)에 비해 척도화된 FGDI 점수가 유의미하게 더 높게 나타났다. _(근거: PAGE 9, Section 3.2)_
  - 근거 원문: “Subjects identified as freezers demonstrate higher scaled FGDI values compared to non-freezers, as confirmed by a Wilcoxon rank sum test with a continuity correction, which yielded a p-value of 0.007.”
- 하지 절단 환자군에서 보행 기능 수준이 더 낮은 K2 집단이 K3 집단에 비해 절단측 및 비절단측 다리 모두에서 척도화된 FGDI 값이 통계적으로 유의미하게 높았다. _(근거: PAGE 10, Section 4.1)_
  - 근거 원문: “A Wilcoxon rank sum exact test confirms that the median scaled FGDI values are significantly higher for individuals classified as K2 compared to those classified as K3, resulting in p-values of 0.03 for the amputated side and 0.009 for the non-amputated side.”
- 파킨슨병 환자를 대상으로 한 보행 근사 분석에서 FGDI 기반 근사 오차(평균 RMSE 0.46)가 기존 OA 방법(평균 RMSE 0.83)보다 현저히 낮아 원본 움직임 데이터를 복원하고 설명하는 데 더 정밀함을 입증했다. _(근거: PAGE 13, Section 5.1)_
  - 근거 원문: “In addition, the average RMSE for all joints/planes in PD subjects is 0.46 for FGDI and 0.83 for OA.”

## 저자 결론

- 다변량 함수형 주성분 분석(MFPCA)을 적용한 FGDI는 보행 움직임 고유의 부드러움과 다관절 간 공변이 특성을 파악하여 다중공선성 문제를 제어하면서도 전반적인 보행 기능을 정확하게 정량화한다. _(근거: PAGE 14, Section 7)_
  - 근거 원문: “This index captures the intrinsic smoothness of gait movements across each joint and plane, and it accounts for potential covariation among them. FGDI is correlated with overall gait function, providing a reliable measure of gait abnormality.”
- 척도화된 FGDI는 임상 일상 평가에서 사용되는 범주형 척도(ordinal scales)보다 객관적이고 연속적인 보행 품질 측정이 가능하며, 시간에 따른 미세한 진전 추적 및 환자 맞춤형 치료 계획 수립에 강점이 있다. _(근거: PAGE 15, Section 7)_
  - 근거 원문: “The scaled FGDI provides a quantitative measurement of gait quality, enabling more precise and detailed assessments of gait abnormalities compared to the ordinal scales typically employed by standard measures. Additionally, the scaled FGDI is sensitive enough to detect subtle changes in gait quality over time, making it useful for monitoring progress and evaluating the effectiveness of interventions or treatments.”

## 연구의 한계

- 연구의 한계로서 현재 제안된 MFPCA 기법에는 개인의 성별이나 연령 등의 공변량 정보가 분석 모형에 고려되지 않았으며, 향후 이에 대한 통합 설계가 필요하다. _(근거: PAGE 16, Section 7)_
  - 근거 원문: “Future research will involve incorporating covariates such as the individual’s sex and age into the MFPCA approach.”

## 생각해볼 내용

- 보행 분석에서 시간 순서에 따른 역학 구조와 다중공선성을 MFPCA로 모델링함으로써 평가 시 발생할 수 있는 왜곡(편향)을 줄이고 임상 의사 결정의 신뢰도를 향상시킨 것으로 분석된다. _(근거: PAGE 4, Section 1)_
  - 근거 원문: “Additionally, the proposed approach accounts for the structure of the dependence of human gait leading to a more accurate quantification of gait pathology which can improve clinical decision-making.”
- 임상의들이 복잡한 통계 기법을 몰라도 직관적으로 개별 다리와 관절의 보행 편차를 시각적으로 한눈에 파악할 수 있도록 무료 인터랙티브 R Shiny 웹 애플리케이션 형태의 유틸리티 툴을 제공하여 실용성을 극대화했다. _(근거: PAGE 14, Section 6)_
  - 근거 원문: “A free R Shiny application has been developed to serve as a graphical user interface (GUI) and is accessible at the following URL: https://michelle-carey-ucd.shinyapps.io/FGDI_ShinyApp/.”

## 이 연구가 지적한 선행연구의 문제점

- 기존의 보행 이상 평가지표인 GDI와 GPS는 보행 주기와 관절 간 데이터의 상호 의존성을 통제하지 않아 이상 정도를 평가할 때 편향을 유발할 수 있다. _(근거: PAGE 3, Section 1)_
  - 근거 원문: “This interdependence of kinematic data may result in biased assessments of overall abnormality when using the GDI and GPS indices, as demonstrated in [16].”
- 기존에 제안된 OA(Overall Abnormality) 방법의 경우, 주성분 분석(PCA) 모델링 과정에서 보행 주기 내 시간적 순서 구조를 보존하지 못하고 관절 및 평면 간 결합으로 발생하는 다중공선성 문제를 완전히 대처하지 못한다. _(근거: PAGE 3, Section 1)_
  - 근거 원문: “The temporal sequence of kinematic data throughout the gait cycle is crucial in gait analysis and should be preserved. The interconnectedness across joints and planes often leads to multicollinearity, which can introduce bias in the measurement of overall abnormality.”

## 이 연구의 해결 방식과 기여

- 다변량 함수형 주성분 분석(MFPCA)을 도입하여, 시간 흐름에 따른 보행 데이터의 물리적 연속성(순서)을 보존함과 동시에 기학적 변수 간 다중공선성 문제를 제거하는 이상적인 가중 주성분 특징 공간을 형성했다. _(근거: PAGE 3-4, Section 1)_
  - 근거 원문: “MFPCA effectively addresses potential multicollinearity issues due to interdependencies between joints or planes while preserving the temporal ordering throughout the gait cycle.”
- GDI, GPS, OA의 강점들을 모아 단일화된 다차원 정량 분석 프레임워크를 제공하여 다리 전체의 이상도뿐만 아니라 편측 및 국소 관절별 비대칭을 직관적으로 상세 평가할 수 있게 기여했다. _(근거: PAGE 4, Section 1)_
  - 근거 원문: “FGDI combines the advantages of the existing GDI, GPS and OA approaches. As the FGDI index is easy to interpret and can provide a measure of severity on both legs, as well as on each leg and at each joint separately.”

## 레퍼런스할 수 있는 내용

### 1. 파킨슨병 환자의 임상적 운동 기능 비대칭성

- 원문 발췌: “It is well-established that unilateral motor symptoms are characteristic of PD, as underscored by Miller-Patterson et al. [20], Djaldetti et al. [3].”
- 한국어 번역: 파킨슨병에서 일측성 운동 증상이 전형적으로 나타난다는 것은 잘 정립되어 있다.
- 원문 위치: PAGE 8, Section 3.1
- 원문 내 인용표기: Miller-Patterson et al. [20]
- 해당 선행문헌: [20] C. Miller-Patterson, R. Buesa, N. McLaughlin, R. Jones, U. Akbar, and J.H. Friedman, Motor asymmetry over time in Parkinson’s disease, J. Neurol. Sci. 393 (2018), pp. 14–17.
- 주장 유형: background_citation
- 활용 맥락과 주의: 파킨슨병 환자의 관절 각도 비대칭성 연구나 임상적 비대칭 특징을 인용하여 배경 논거를 설명할 때 활용할 수 있으며, 2차 인용에 유의해야 한다.

### 2. 파킨슨병의 보행 동결이 보행 이탈 심각도에 미치는 영향

- 원문 발췌: “Subjects identified as freezers demonstrate higher scaled FGDI values compared to non-freezers, as confirmed by a Wilcoxon rank sum test with a continuity correction, which yielded a p-value of 0.007.”
- 한국어 번역: 동결 보행이 있는 대상자들은 동결 보행이 없는 대상자들에 비해 더 높은 척도화된 FGDI 값을 보였으며, 이는 연속성 보정이 적용된 Wilcoxon 순위합 검정을 통해 p-value 0.007로 확인되었다.
- 원문 위치: PAGE 9, Section 3.2
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: 파킨슨병에서 보행 동결(freezing of gait) 증상이 정량적인 보행 이상 중증도에 지대하고 유의한 편차를 초래한다는 통계적 근거로 인용할 수 있다.

### 3. 하지 절단 환자의 K-Level 등급별 보행 특징 및 이상 차이

- 원문 발췌: “A Wilcoxon rank sum exact test confirms that the median scaled FGDI values are significantly higher for individuals classified as K2 compared to those classified as K3, resulting in p-values of 0.03 for the amputated side and 0.009 for the non-amputated side.”
- 한국어 번역: Wilcoxon 순위합 정확 검정을 통해 K2 등급으로 분류된 개인이 K3 등급에 비해 척도화된 FGDI 중앙값이 유의미하게 높음을 확인하였으며, 그 결과 p-value는 절단측 0.03, 비절단측 0.009였다.
- 원문 위치: PAGE 10, Section 4.1
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: 절단 장애 환자의 기능적 등급인 K-Level이 낮을수록(K2 vs K3) 절단된 부위뿐만 아니라 보상 패턴이 작용하는 비절단측 정상 다리 전체에서도 비전형적 보행 이탈 편차가 크게 증가한다는 통계 근거로 제시할 수 있다.
