# Leveraging explainable machine learning to identify gait biomechanical parameters associated with anterior cruciate ligament injury

Kokkotis, C., Moustakidis, S., Tsatalas, T., Ntakolia, C., Chalatsis, G., Konstadakos, S., Hantes, M. E., Giakas, G., & Tsaopoulos, D. (2022). Leveraging explainable machine learning to identify gait biomechanical parameters associated with anterior cruciate ligament injury. Scientific Reports, 12, 6647. https://doi.org/10.1038/s41598-022-10666-2

## 서지정보

- 저자: Christos Kokkotis, Serafeim Moustakidis, Themistoklis Tsatalas, Charis Ntakolia, Georgios Chalatsis, Stylianos Konstadakos, Michael E. Hantes, Giannis Giakas, Dimitrios Tsaopoulos
- 연도: 2022
- 저널: Scientific Reports
- DOI: 10.1038/s41598-022-10666-2
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/04_machine_learning_and_deep_learning/Leveraging explainable machine learning to identify gait biomechanical parameters associated with anterior cruciate ligament injury.pdf
- 분석 provider: antigravity

## 연구 목적

- 설명 가능한 머신러닝 방법론을 개발하여 ACL 부상 진단에서 보행 운동학적 및 역학적 매개변수의 기여도를 식별 및 수량화하고, ACL 결손 환자, 재건술 환자, 건강한 대조군 간의 시상면 보행 생체역학 차이를 조사하고자 한다. _(근거: --- PAGE 1 ---)_
  - 근거 원문: “This paper focuses on the development of an explainable machine learning (ML) empowered methodology to: (i) identify important gait kinematic, kinetic parameters and quantify their contribution in the diagnosis ofACL injury and (ii) investigate the differences in sagittal plane kinematics and kinetics of the gait cycle between ACL deficient,ACL reconstructed and healthy individuals.”
- 설명 가능한 머신러닝과 통계 분석을 결합하여 분류 과정에서 특징 중요도를 추정하고 각 환자 그룹(ACL 결손, 재건, 대조군) 간 시상면 운동학 및 역학 변수의 차이를 탐색한다. _(근거: --- PAGE 2 ---)_
  - 근거 원문: “The aims of this study are: (i) to estimate the feature importance in the classification process and examine how much each of the features contributed to the final ML decisions and (ii) to investigate differences in sagittal plane kinematics and kinetics of the gait cycle between different patient groups based on a novel approach that combines explainable ML and statistical analytics.”

## 연구 설계와 대상

- 총 151명의 피험자가 연구에 참여하였으며, 수술 전 ACL 결손군(ACLD), ACL 재건군(ACLR), 대조군(CON)의 세 그룹으로 구성되었다. _(근거: --- PAGE 7 ---, Participants.)_
  - 근거 원문: “A total of 151 subjects volunteered to participate in this study. Three different groups were defined: (i) ACL-deficient prior to surgery (ACLD), (ii) ACL-reconstructed (ACLR) and (iii) control (CON) group.”
- 대조군 피험자들은 연령, 성별, 신체 활동 수준에 대해 매칭되었으며, 측정 전 12개월 동안 ACL 부상 및 lower extremity 부상 등의 병력이 없었다. _(근거: --- PAGE 7 ---, Participants.)_
  - 근거 원문: “The CON subjects were matched for age, gender, and physical activity status and had no history of ACL injury and neurologic disorder or other lower extremity injuries within 12 months prior to participating in the study.”

## 방법

- 피험자들은 개별 자체 선택 보행 속도(SWS)의 ±5% 범위 내에서 맨발로 10m 실험실 보행로를 걸었다. _(근거: --- PAGE 7 ---, Testingprocedureanddatacollection.)_
  - 근거 원문: “Subsequently, the subjects walked barefoot along the 10 m laboratory walkway within±5% of their individual self-selected walking speed (SWS).”
- 운동학 및 지면 반력(GRF) 데이터는 각각 10Hz 및 40Hz에서 4차 버터워스 필터로 저역 통과 필터링되었다. _(근거: --- PAGE 8 ---, Data analysis.)_
  - 근거 원문: “Kinematic and GRF data were lowpass filtered with a 4th order Butterworth filter at 10 and 40 Hz, respectively.”
- 특징 선택(FS)과 머신러닝 추정기들을 위한 공통의 기반을 마련하고자 데이터를 [0, 1] 범위로 정규화하였다. _(근거: --- PAGE 8 ---, Machinelearningworkflow.)_
  - 근거 원문: “Data were normalised to [0, 1] to build a common basis for the feature selection (FS) and the ML estimators.”
- 최종 머신러닝 출력값에 대한 영향도에 따라 특징들의 순위를 매기고 미니 설명자 모델을 구축하기 위해 SHAP을 도입하였다. _(근거: --- PAGE 9 ---, Machinelearningworkflow.)_
  - 근거 원문: “In this paper, we employed SHAP to rank features in terms of their impact on the final ML outputs and to build a mini explainer model.”

## 핵심 결과

- SVM 모델은 처음 선택된 특징들에 대해 상승 추세를 보였으며, 21개의 특징 그룹에서 가장 높은 분류 정확도인 94.95%를 달성했다. _(근거: --- PAGE 2 ---, Results)_
  - 근거 원문: “Specifically, the SVM model showed an upward trend with respect to the first selected features, with a maximum of 94.95% (which was the overall best performance achieved).”
- 두 번째로 높은 성능을 보인 신경망(NN) 모델은 92.89%의 테스트 정확도를 얻었으며, 15개 초과의 특징을 사용했을 때 변동이 있는 비안정적 상승 추세를 보였다. _(근거: --- PAGE 2 ---, Results)_
  - 근거 원문: “The second-best accuracy (92.89%) was achieved by the NN model, which presented a non-steadily increasing performance with fluctuations for more than 15 selected features.”
- 전체 3개 클래스 분석에서 평균 SHAP 값 크기가 0.3을 초과하여 모델 출력에 가장 큰 영향을 준 변수들은 K2, H4, A3, GRF4, GRF7, K1, A4, GRF6였다. _(근거: --- PAGE 4 ---, Results)_
  - 근거 원문: “In this approach K2, H4, A3, GRF4, GRF7, K1, A4 and GRF6 were the parameters that affected the model output with mean SHAP values higher than 0.3.”
- 대조군과 수술 전 ACL 결손군(ACLD)의 차이를 구별하는 로컬 문제 1에서 H4, K7, GRF3, H1, H2가 예측 출력에 유의미한 영향을 미치는 가장 중요한 변수였다. _(근거: --- PAGE 4 ---, Results)_
  - 근거 원문: “It should be noted that the features H4, K7, GRF3, H1, H2 were the most important variables that significantly affected the prediction output.”
- 설명성 분석에서 중요하게 파악된 변수 중 H4, K7, GRF3, GRF4의 4개 변수에서 대조군과 ACL 결손군 간의 유의미한 통계적 차이가 관찰되었다. _(근거: --- PAGE 4 ---, Results)_
  - 근거 원문: “Significant differences were observed between CON and ACLD for half of the features considered, specifically the first three (H4, K7 and GRF3) along with GRF4;”

## 저자 결론

- 보행 생체역학의 기여도를 이해하는 것은 임상의가 비침습적이고 강력한 예후 도구를 개발하는 데 유용하며, 재건술 후 환자의 비정상 보행 패턴을 식별하여 재활 프로토콜을 수정하고 관절염 발생을 예방하도록 돕는다. _(근거: --- PAGE 7 ---, Summary)_
  - 근거 원문: “Understanding the contribution of gait biomechanics is a valuable tool for creating more powerful and non-invasive prognostic tools in the hands of physicians, that will point out abnormal gait patterns in patients after ACLR to modify the rehabilitation protocol and avoid the development of osteoarthritis.”
- 선택된 매개변수의 성격과 그것이 예측 결과에 미치는 영향(SHAP을 통해 제시됨)은 훈련된 모델의 의사결정 메커니즘 이면의 근거를 밝혀주며, ACL 부상 진단에서 입력 매개변수의 기여도를 정량화하는 대안적이고 보다 총체적인 접근 방식을 제공한다. _(근거: --- PAGE 7 ---, Summary)_
  - 근거 원문: “The nature of the selected parameters along with their impact on the prediction outcome (via SHAP) were discussed to uncover the rationale behind the decision-making mechanism of the trained model and therefore provide an alternative and a more holistic approach of quantifying the contribution of the input parameters in the diagnosis of ACL injury.”

## 연구의 한계

- ACL 재건술 후 보행 생체역학이 변화하지만 연구 및 과제 전반에 걸쳐 일관된 결과를 보여주는 매개변수가 드물기 때문에 임상적 중요성은 주의해서 고려해야 한다. _(근거: --- PAGE 6 ---, Discussion)_
  - 근거 원문: “This can be attributed to the fact that even though gait biomechanics are altered following ACLR, few biomechanical parameters demonstrate consistent results across studies and various tasks10.”
- SHAP은 개별 변수가 모델 출력에 미치는 영향력을 수량화하는 단순한 설명에 국한되어, 변수들의 조합이 최종 결정에 기여하는 복잡한 내부 작동 방식은 여전히 알기 어렵다. _(근거: --- PAGE 7 ---, Discussion)_
  - 근거 원문: “However, SHAP is limited to simple explanations mainly quantifying the impact of individual features to the models’ output40.”

## 생각해볼 내용

- > **[AS-IS]** 설명 가능성 분석 도구(SHAP)의 도입은 블랙박스 머신러닝 모델의 의사결정을 인간이 더 직관적으로 이해하고 임상적으로 해석할 수 있도록 돕는 유용한 가교가 된다. _(근거: --- PAGE 7 ---, Discussion)_
>
> **[TO-BE]** 설명 가능성 분석 도구(SHAP)의 도입은 블랙박스 머신러닝 모델의 의사결정을 인간이 더 잘 이해하도록 돕는 중요한 수단이 된다.
>
> _(사실검증 — 과장/경미: 원문은 SHAP 등이 블랙박스 모델의 결정을 인간이 더 잘 이해하도록 돕는다고 설명하지만, 해당 문장 자체는 '임상적으로 해석'까지 직접 말하지 않는다. 논문 전체가 임상적 맥락을 다루기는 하나, 이 근거문만으로는 임상적 해석을 돕는다고 단정하기에는 표현이 조금 강하다.)_
  - 근거 원문: “Explainability via SHAP or other similar tools is a crucial enabler allowing humans to better comprehend the decisions generated by black box models.”
- 전통적인 통계적 유의성 검정만으로는 놓치기 쉬운 비선형적이거나 복합적인 매개변수들의 상호작용을 머신러닝과 설명성 분석을 결합하여 포착할 수 있다는 점에서 방법론적 의의가 크다. _(근거: --- PAGE 6 ---, Discussion)_
  - 근거 원문: “Features, that would have been neglected by the traditional statistical analysis, are highlighted as contributing parameters that have a significant impact on the ML model’s output when they are combined with other statistically important ones.”

## 이 연구가 지적한 선행연구의 문제점

- 기존의 ACL 분야 머신러닝 연구는 대다수 모델을 블랙박스로 처리하여 투명성이 부족했다. _(근거: --- PAGE 2 ---, Introduction)_
  - 근거 원문: “Despite the relatively large number of ML studies on the field of ACL, the reported trained ML models are treated as black boxes.”
- 모델의 투명성과 설명성 결여로 인해 인공지능 모델이 어떠한 판단 메커니즘을 거쳐 내부적으로 의사결정을 내렸는지 이해하기 어려웠다. _(근거: --- PAGE 2 ---, Introduction)_
  - 근거 원문: “The lack of transparency and explainability of the models result to poor understanding of their inner workings and the rationale behind their decision-making mechanism.”

## 이 연구의 해결 방식과 기여

- 본 연구는 ACL 부상과 관련된 주요 매개변수를 식별하기 위해 설명 가능한 머신러닝 방법론과 통계 분석을 통합한 새로운 접근법을 제안한다. _(근거: --- PAGE 4 ---, Discussion)_
  - 근거 원문: “This paper focuses on the development of a novel approach, which combines an explainable ML-empowered methodology and statistical analysis, for identifying important parameters associated with ACL injury.”
- 본 연구의 주요 기여는 분류 성능뿐만 아니라 각 특징이 의사결정에 얼마나 기여하는지 조사하고, 특징 중요도를 추정하며, 세 환자 그룹 간의 3차원 지면 반력(GRF) 및 시상면 운동학/역학적 보행 패턴의 차이를 조사하는 데 있다. _(근거: --- PAGE 4 ---, Discussion)_
  - 근거 원문: “In addition to the classification part, the main contributions of this paper are: (i) to investigate how much each of the features contributed to the final ML decisions, (ii) to estimate the feature importance in the classification process and (iii) to investigate differences in three dimensional GRFs, sagittal plane kinematics and kinetics of the gait cycle for the CON, ACLD and ACLR groups.”

## 레퍼런스할 수 있는 내용

### 1. SVM 모델의 우수한 분류 성능

- 원문 발췌: “Support Vector Machines were proved to be the best performing model (accuracy of 94.95%) on a group of 21 selected biomechanical parameters.”
- 한국어 번역: 서포트 벡터 머신(SVM)은 21개의 선택된 생체역학 변수 그룹에 대해 가장 성능이 우수한 모델(정확도 94.95%)인 것으로 입증되었다.
- 원문 위치: --- PAGE 1 ---
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: 분석 대상 논문의 자체 실험을 통해 도출된 핵심 분류 성능으로, 다른 문헌에 인용 시 2차 인용이 필요하지 않고 이 논문을 직접 지칭하여 기술할 수 있다.

### 2. 로컬 문제 2에서 K2 변수의 높은 영향도

- 원문 발췌: “Specifically, K2 records a much higher mean absolute value (higher than 0.35) compared to the rest of the features (that exhibit values less than 0.23).”
- 한국어 번역: 구체적으로, K2는 나머지 특징들(0.23 미만의 값을 나타냄)에 비해 훨씬 높은 평균 절대값(0.35 초과)을 기록했다.
- 원문 위치: --- PAGE 4 ---, Results
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: 대조군과 ACL 재건군(ACLR) 간의 차이를 구분할 때 최소 무릎 굴곡 각도(K2)가 SHAP 분석을 통해 매우 중요하게 기여했음을 의미한다.

### 3. 무릎 관절염 초기 진행 판단을 위한 머신러닝 연구 경향

- 원문 발췌: “Recent studies with individual-level datasets of gait analyses from kinetic skeletal tracking and advanced MR imaging (MRI) techniques focused on the determination of early progression of knee osteoarthritis (KOA)30.”
- 한국어 번역: 운동 역학적 골격 추적 및 고급 자기공명영상(MRI) 기술을 통한 개인 수준의 보행 분석 데이터셋을 활용한 최근 연구들은 무릎 관절염(KOA)의 초기 진행을 결정하는 데 초점을 맞추었다.
- 원문 위치: --- PAGE 2 ---, Introduction
- 원문 내 인용표기: 30
- 해당 선행문헌: 30. Brisson, N. M., Gatti, A. A., Damm, P., Duda, G. N. & Maly, M. R. Association of machine learning based predictions of medial knee contact force with cartilage loss over 2.5 years in knee osteoarthritis. Arthr. Rheumatol. 73, 1638–1645. https://doi.org/10.1002/art.41735 (2021).
- 주장 유형: background_citation
- 활용 맥락과 주의: 이 내용은 선행 연구에 대한 언급이므로 이 문장을 다른 논문이나 글에 인용할 때는 원저자인 Brisson et al. (2021)을 2차 인용(또는 1차 출처 직접 인용)해야 안전하다.
