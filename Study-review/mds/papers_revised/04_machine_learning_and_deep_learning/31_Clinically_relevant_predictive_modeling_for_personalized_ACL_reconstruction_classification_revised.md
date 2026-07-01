# Clinically relevant predictive modeling for personalized ACL reconstruction classification

Zhu, X., Henry, R., Jackson, E., Hart, J. M., & Gong, J. (2025). Clinically relevant predictive modeling for personalized ACL reconstruction classification. Smart Health, 36, 100575. https://doi.org/10.1016/j.smhl.2025.100575

## 서지정보

- 저자: Xishi Zhu, Ryan Henry, Emily Jackson, Joe M. Hart, Jiaqi Gong
- 연도: 2025
- 저널: Smart Health
- DOI: 10.1016/j.smhl.2025.100575
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/04_machine_learning_and_deep_learning/Clinically relevant predictive modeling for personalized ACL reconstruction classification.pdf
- 분석 provider: antigravity

## 연구 목적

- 이 연구의 목적은 관성 측정 장치(IMU) 센서와 환자 특성을 결합한 다중 모달 보행 분석을 통합하여 ACL 재건술의 분류 및 회복 진행 상황을 시각화할 수 있는 설명 가능하고 개인화된 예측 모델을 개발하는 것이다. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “We propose an explainable predictive model for ACL reconstruction classification through multi-modal analysis of gait dynamics and patient characteristics.”
- 연구진은 위상 경사 지수(PSI)로 정량화된 신체 부위 간 쌍별 운동이 ACL 재건 분류에 크게 기여하고, 걷기와 조깅 작업 간에 중요 센서 쌍의 조합이 다르며, 환자별 요인(회복 기간)이 모델의 분류 신뢰도와 상관관계가 있다는 세 가지 가설을 검증하고자 하였다. _(근거: PAGE 2, 1. Introduction)_
  - 근거 원문: “We hypothesized that: (1) paired body segment movements quantified by Phase Slope Index (PSI) matrices would significantly contribute to ACL reconstruction classification; (2) the importance of specific sensor pair combinations would differ between walking and jogging tasks; and (3) patient-specific factors would correlate with classification confidence, revealing insights into temporal dynamics of gait recovery.”

## 연구 설계와 대상

- 연구에는 74명의 ACL 재건 수술을 받은 환자(왼쪽 무릎 부상 31명, 오른쪽 무릎 부상 43명)와 5명의 건강한 대조군을 포함하여 총 79명의 참가자가 모집되었다. _(근거: PAGE 2, 3. Methodology)_
  - 근거 원문: “We recruited 79 participants, including 74 ACL patients (31 with left knee injuries, 43 with right knee injuries) and 5 healthy individuals.”

## 방법

- 참가자들의 양쪽 손목, 양쪽 발목 및 천골에 총 5개의 Shimmer IMU 센서를 부착하여 128Hz의 빈도로 가속도계와 자이로스코프 데이터를 수집하였으며, 트레드밀에서 3mph 속도로 5분간 걷고 이어서 6mph 속도로 3분간 조깅하는 작업을 수행하게 하였다. _(근거: PAGE 2, 3. Methodology)_
  - 근거 원문: “Data collection utilized five Shimmer IMU sensors placed on participants’ bodies (bilateral wrists, ankles, and sacrum), recording accelerometer and gyroscope data at 128 Hz. The protocol consisted of two sequential tasks: walking on a treadmill at 3 mph for 5 min, followed by jogging at 6 mph for 3 min.”
- 데이터를 64Hz로 다운샘플링하고 10초의 비중첩 창으로 분할한 후, 센서 판독치 간의 복잡한 상호작용을 포착하기 위해 각 주파수 대역에서 위상 경사 지수(PSI)를 계산하여 쌍별 인과 관계 특성 행렬(총 435개의 고유 특징)을 생성하였다. _(근거: PAGE 2, 3. Methodology)_
  - 근거 원문: “We then downsampled the data from 128 Hz to 64 Hz and segmented it into non-overlapping 10-second windows, providing sufficient temporal resolution to capture multiple gait cycles while maintaining computational efficiency. To capture complex interactions between sensor readings, we calculated the Phase Slope Index (PSI) to form a pairwise causality feature matrix.”
- 다섯 가지 기계학습 모델(SVM, Naive Bayes, Random Forest, KNN, Neural Network)을 사용해 왼쪽/오른쪽 부상 분류 및 건강군/부상군 분류라는 두 가지 이진 분류 작업을 수행했으며, 5-fold 교차 검증을 통해 모델을 평가하였다. _(근거: PAGE 2, 3. Methodology)_
  - 근거 원문: “To investigate our first hypothesis and evaluate PSI-based features, we implemented five machine learning models (SVM, Naive Bayes, Random Forest, KNN, and Neural Network) for two binary classification tasks: Left Injured vs. Right Injured and Injured vs. Healthy. We employed 5-fold cross-validation for walking and jogging phases independently, with randomly shuffled data windows.”

## 핵심 결과

- KNN 분류기가 모든 분류 과제에서 가장 높은 정확도를 나타냈으며, 16Hz 주파수 대역 데이터를 사용했을 때 좌/우 부상 분류의 경우 보행 시 약 93%, 조깅 시 약 98%의 정확도를 보였고, 정상/부상 분류에서는 보행 시 약 98%, 조깅 시 약 99%의 높은 정확도를 기록했다. _(근거: PAGE 3, 4.1. Predictive modeling evaluation)_
  - 근거 원문: “The KNN classifier consistently achieved the highest accuracy scores in all scenarios. For left–right classification, KNN reached a peak accuracy of approximately 93% during walking and 98% during jogging using the 16 Hz frequency range. For healthy-injured classification, KNN’s performance was even more impressive, achieving approximately 98% accuracy for walking and 99% for jogging, both with the 16 Hz filtered data.”
- 순열 중요도 점수 기반의 열지도 분석 결과, 보행 시 좌/우 부상 분류에서는 높은 중요도(0.07 이상)를 가지는 센서 쌍이 5개였던 반면 조깅 시에는 단 2개의 센서 쌍만 0.0175 이상의 중요도를 보여 보행 데이터가 조깅에 비해 부상 측면 분류에 더 다양한 변동성 패턴을 포함하고 있음을 보여준다. _(근거: PAGE 4, 4.2. Task-specific sensor pair importance analysis)_
  - 근거 원문: “For the left vs. right injury classification during walking (Fig. 2a), five sensor pairs showed importance scores above 0.07, indicated by dark red colors. In contrast, the jogging condition (Fig. 2b) displayed only two sensor pairs with importance scores above 0.0175. This difference suggests that walking data exhibits more variability in discriminative patterns between left and right injuries compared to jogging.”
- t-SNE 시각화에서 조깅 단계의 플롯이 보행 단계에 비해 클래스 간의 겹침이 적고 더 명확하게 군집화되었으며, 모델의 분류 신뢰도가 낮은 샘플들은 주로 안쪽 원 영역에 집중되는 경향을 보였다. _(근거: PAGE 4, 4.3. Dimension reduction examination)_
  - 근거 원문: “The t-SNE plots revealed that inner circles predominantly contain instances where the model exhibits low confidence in distinguishing between classes. Notably, jogging phase plots showed more distinctly separated clusters with reduced class overlap compared to walking, suggesting that jogging captures better global data structure.”
- > **[AS-IS]** 회복 기간이 긴 환자들일수록 기계학습 모델이 높은 신뢰도로 분류하기 어려운 경향을 보였는데, 이는 회복 과정이 진행됨에 따라 움직임 패턴이 정상화되어 부상당하지 않은 대조군의 보행 패턴과 유사해지기 때문이다. _(근거: PAGE 5, 4.4. Patient-specific factor analysis)_
>
> **[TO-BE]** 회복 기간이 긴 참가자들은 모델이 높은 신뢰도로 분류하기 어려운 보행 패턴을 보이는 경향이 있었으며, 저자들은 이 결과가 회복이 진행됨에 따라 움직임 패턴이 점차 정상 보행에 가까워진다는 임상적 이해를 뒷받침한다고 해석했다.
>
> _(사실검증 — 인과관계오용/경미: 원문은 회복 기간이 긴 참가자의 보행 패턴이 높은 신뢰도로 분류되기 어려운 경향이 있고, 이것이 회복이 진행되며 보행 패턴이 정상화된다는 임상적 이해를 뒷받침한다고 설명한다. 요약은 이를 '때문이다'로 표현해 원문보다 인과를 더 단정했다.)_
  - 근거 원문: “These findings suggest that participants with longer recovery durations generally exhibit gait patterns that are more difficult for the model to classify with high confidence. This trend supports the clinical understanding that as recovery progresses, movement patterns gradually normalize, becoming more similar to uninjured gait.”

## 저자 결론

- 이 연구에서는 IMU 센서 데이터를 바탕으로 도출한 위상 경사 지수(PSI) 기능이 ACL 재건 상태를 95.37%의 우수한 정확도로 분류할 수 있음을 증명하였으며, 환자의 작업 간 센서 쌍 중요도 차이 및 회복 기간과 모델 신뢰도 간의 상관관계를 통해 보행 패턴이 시간이 지남에 따라 점진적으로 정상화됨을 확인하였다. _(근거: PAGE 6, 5. Discussion & conclusion)_
  - 근거 원문: “Our findings supported our hypotheses: PSI-based features from IMU data effectively classified ACL reconstruction outcomes with 95.37% accuracy using KNN; sensor pair importance differed between walking and jogging tasks, with jogging showing more focused importance patterns; and recovery duration correlated with model confidence, suggesting gait patterns normalize over time post-reconstruction.”
- > **[AS-IS]** 본 설명 가능하고 개인화된 접근 방식은 기계학습 모델의 블랙박스 성격을 완화하여, 임상 의사결정을 돕는 정량적이고 객관적인 도구로서 재활 계획을 개선하고 복귀 기준을 설정하는 데 큰 도움을 줄 수 있다. _(근거: PAGE 6, 5. Discussion & conclusion)_
>
> **[TO-BE]** 본 설명 가능한 접근은 ACL 재건과 관련된 주요 움직임 관계를 식별하고 환자 진행을 추적할 정량 도구를 제공함으로써 임상 의사결정에 도움을 줄 수 있으며, 향후 개인화 재활 프로토콜과 데이터 기반 운동 복귀 기준 연구의 기반을 마련한다.
>
> _(사실검증 — 과장/경미: 원문은 주요 움직임 관계를 식별하고 환자 진행을 추적하는 정량 도구를 제공한다는 임상적 장점을 말하며, 향후 개인화 재활 프로토콜과 데이터 기반 복귀 기준을 가능하게 할 잠재력을 제시한다. 요약의 '복귀 기준을 설정하는 데 큰 도움'은 원문보다 실용적 효과를 더 강하게 단정한다.)_
  - 근거 원문: “The explainable nature of our approach offers significant clinical advantages by identifying key movement relationships affected by ACL reconstruction and providing quantitative tools to track patient progress.”

## 연구의 한계

- 연구의 주요 한계점으로는 비교적 작은 표본 크기, 전적으로 IMU 웨어러블 센서 데이터에만 의존한 분석 설계, 그리고 환자의 보행을 단일 시점에서만 분석한 점 등이 포함된다. _(근거: PAGE 6, 5. Discussion & conclusion)_
  - 근거 원문: “Despite limitations including small sample size, reliance solely on IMU data, and single time point analysis, our novel approach demonstrates the potential of combining multi-modal gait analysis with explainable machine learning for ACL reconstruction assessment.”

## 생각해볼 내용

- > **[AS-IS]** 조깅 데이터 분석이 보행 분석보다 클래스 분리도와 분류 성능 면에서 더 우수한 결과를 보인 것은, 달리기와 같이 부하가 높고 고속인 운동에서 하지 및 상지 간 협응의 미세한 기능적 불균형이나 이상 징후가 더 쉽게 드러나고 모델에 유용한 식별 신호를 제공함을 보여준다. _(근거: PAGE 4, 4.3. Dimension reduction examination)_
>
> **[TO-BE]** 조깅 데이터는 보행 데이터보다 t-SNE 플롯에서 클래스 겹침이 적고 군집 분리가 더 뚜렷했으며, 이는 조깅 조건에서의 더 높은 분류 성능과 일치한다.
>
> _(사실검증 — 근거불충분/경미: 원문은 조깅 플롯에서 군집 분리가 더 뚜렷하고 모델 성능이 더 좋았다고만 설명한다. 고부하·고속 운동에서 하지 및 상지 협응의 미세한 불균형이 더 쉽게 드러난다는 해석은 제시된 원문 근거에서 직접 확인되지 않는다.)_
  - 근거 원문: “Notably, jogging phase plots showed more distinctly separated clusters with reduced class overlap compared to walking, suggesting that jogging captures better global data structure. This improved cluster separation aligns with the superior model performance in jogging conditions relative to walking.”
- 기계학습 모델의 신뢰도가 환자의 회복 기간이 지남에 따라 감소하는 특이한 상관관계는, 재활이 성공하여 보행 패턴이 완전히 정상화될수록 정상군과의 보행 특징 차이가 사라지기 때문에 생기는 현상이며, 이를 역으로 이용해 모델 신뢰도 지표를 보행 패턴 회복의 성숙도 및 정상화의 객관적인 간접 척도로 활용할 수 있음을 나타낸다. _(근거: PAGE 5, 4.4. Patient-specific factor analysis)_
  - 근거 원문: “These findings suggest that participants with longer recovery durations generally exhibit gait patterns that are more difficult for the model to classify with high confidence. This trend supports the clinical understanding that as recovery progresses, movement patterns gradually normalize, becoming more similar to uninjured gait.”

## 이 연구가 지적한 선행연구의 문제점

- 기존의 ACL 재건 결과 평가 및 운동 복귀 판정 방법은 임상 시험, 기능적 테스트, 주관적 평가에 주로 의존하며, 복잡한 신체 분절 간의 상호작용과 운동 패턴을 완벽히 포착하지 못하여 객관성과 정밀도가 떨어진다. _(근거: PAGE 1, 1. Introduction)_
  - 근거 원문: “While valuable, these methods often neglect complex movement patterns and lack precision and objectivity, particularly failing to account for the complex interplay between different body segments during movement.”
- 최근 웨어러블 IMU 센서와 머신러닝을 이용한 보행 분석 연구가 시도되고 있으나, 분석에 사용되는 고도화된 방식들이 임상적 해석이 어려운 '블랙박스' 형태로 이루어져 있어 임상의들이 실제로 신뢰하고 재활 처방에 바로 적용하기 어렵다. _(근거: PAGE 2, 2. Related work)_
  - 근거 원문: “Despite promising performance, these advanced analytical methods face limitations in clinical interpretability, with their ‘‘black box’’ nature hindering translation into actionable insights.”
- 머신러닝 해석력을 높이기 위한 기존 접근법들은 전역적 피처 중요도 분석에 집중되어 환자별 맞춤형 진단에 한계가 있거나, LIME 같은 국소적 해석 기법은 예측마다 결과가 변할 수 있어 설명의 안정성이 떨어진다는 단점이 있다. _(근거: PAGE 2, 2. Related work)_
  - 근거 원문: “Efforts to improve interpretability include feature importance scoring to identify which features are most indicative of ACL outcomes, though this approach focuses primarily on global importance, limiting personalized analysis capabilities. For local interpretation, LIME (Local Interpretable Model-agnostic Explanations) has been adopted (Kim et al., 2022), approximating complex models locally with more interpretable models. However, these explanations are valid only for individual predictions and may not generalize well, potentially leading to unstable explanations.”

## 이 연구의 해결 방식과 기여

- 본 연구는 관성 측정 센서(IMU) 데이터와 환자의 고유한 임상적 특성을 위상 경사 지수(PSI) 및 차원 축소 기법을 통해 통합 분석함으로써, 기계학습 모델의 높은 분류 성능과 임상적 해석력을 동시에 확보하는 방법론을 제안하였다. _(근거: PAGE 2, 1. Introduction)_
  - 근거 원문: “To address these limitations, we propose a novel approach integrating multi-modal gait analysis using IMU sensors with patient-specific characteristics.”
- 시간 경과에 따른 환자 보행 패턴의 정상화 과정을 t-SNE 차원 축소 및 분류 신뢰도로 투명하게 시각화하고 정량적으로 보여줌으로써, 스포츠 의학에서 보다 객관적인 환자별 맞춤 재활 계획과 안전한 운동 복귀 결정을 가능하게 하는 기틀을 마련했다. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “While longer recovery typically leads to more normal gait patterns, our approach provides a quantitative method to visualize this process transparently. This explainable, personalized approach can improve rehabilitation strategies and inform more accurate return-to-sport decisions in sports medicine.”

## 레퍼런스할 수 있는 내용

### 1. ACL 부상이 장기적으로 미치는 임상적 영향

- 원문 발췌: “These injuries not only lead to substantial time away from sport but also increase the risk of early-onset osteoarthritis and other long-term complications (Lohmander et al., 2007).”
- 한국어 번역: 이러한 부상(ACL 부상)은 스포츠 활동 중단 시간의 장기화를 초래할 뿐만 아니라 조기 발병 골관절염 및 기타 장기적인 합병증의 위험을 증가시킨다.
- 원문 위치: PAGE 1, 1. Introduction
- 원문 내 인용표기: (Lohmander et al., 2007)
- 해당 선행문헌: Lohmander, L. S., Englund, P. M., Dahl, L. L., & Roos, E. M. (2007). The long-term consequence of anterior cruciate ligament and meniscus injuries: osteoarthritis. The American Journal of Sports Medicine, 35(10), 1756–1769.
- 주장 유형: background_citation
- 활용 맥락과 주의: ACL 재건 후 지속적인 보행 평가 및 무릎 건강 모니터링이 필요한 이유를 설명하기 위한 도입부 근거로 사용하기에 매우 적합함. 2차 인용 시에는 원문 저자인 Lohmander 등을 참조하여야 함.

### 2. 다중 센서 데이터 융합을 통한 분석 정확도 향상

- 원문 발췌: “Studies implementing this multimodal approach have reported accuracy improvements of 5% to 15% compared to single-source models (Dehzangi et al., 2017), suggesting that multiple measurement perspectives help identify subtle injury risk indicators.”
- 한국어 번역: 이러한 다중 모달 접근법을 구현한 연구들은 단일 소스 모델과 비교하여 5%에서 15%의 정확도 향상을 보고하였으며(Dehzangi et al., 2017), 이는 여러 측정 관점이 미세한 부상 위험 지표를 확인하는 데 기여함을 시사한다.
- 원문 위치: PAGE 2, 2. Related work
- 원문 내 인용표기: (Dehzangi et al., 2017)
- 해당 선행문헌: Dehzangi, O., Taherisadr, M., & ChangalVala, R. (2017). IMU-based gait recognition using convolutional neural networks and multi-sensor fusion. Sensors, 17(12), 2735.
- 주장 유형: background_citation
- 활용 맥락과 주의: 웨어러블 센서를 활용한 보행 분석 모델링 연구에서 단일 센서 대비 다중 센서/다중 모달 융합 방식을 채택해야 하는 정량적 당위성(5~15% 성능 향상)을 서술할 때 인용함.

### 3. 정상군 및 부상군 분류에 대한 KNN의 성능 결과

- 원문 발췌: “For healthy-injured classification, KNN’s performance was even more impressive, achieving approximately 98% accuracy for walking and 99% for jogging, both with the 16 Hz filtered data.”
- 한국어 번역: 정상군과 부상군의 분류에서 KNN의 성능은 더욱 인상적이었으며, 16Hz 주파수로 필터링된 데이터를 사용하여 걷기에서 약 98%, 조깅에서 99%의 정확도를 달성했다.
- 원문 위치: PAGE 3, 4.1. Predictive modeling evaluation
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: 이 논문의 직접적인 정량 결과로, 위상 경사 지수(PSI) 특징과 16Hz 대역의 통계적 필터링을 통해 KNN 모델이 높은 수준으로 환자의 재건 상태를 탐지해 낼 수 있음을 나타낼 때 사용함.

### 4. 회복 기간 진행에 따른 움직임 패턴의 정상화

- 원문 발췌: “This trend supports the clinical understanding that as recovery progresses, movement patterns gradually normalize, becoming more similar to uninjured gait.”
- 한국어 번역: 이러한 경향은 회복이 진행됨에 따라 움직임 패턴이 점차 정상화되고, 부상을 입지 않은 사람의 보행 패턴과 더욱 유사해진다는 임상적 이해를 뒷받침한다.
- 원문 위치: PAGE 5, 4.4. Patient-specific factor analysis
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: uncited_author_claim
- 활용 맥락과 주의: 기계학습 모델의 점수 추이 및 시각화 결과가 임상적으로 정상화 메커니즘을 반영하고 있음을 설명할 때, 저자의 독자적인 주장 및 임상적 해석을 바탕으로 논의 파트 등에서 근거로 활용할 수 있음.
