# Machine learning models for predicting return to sports after anterior cruciate ligament reconstruction: Physical performance in early rehabilitation

Hwang, U. J., Kim, J. S., Kim, K. Y., & Chung, K. S. (2024). Machine learning models for predicting return to sports after anterior cruciate ligament reconstruction: Physical performance in early rehabilitation. DIGITAL HEALTH, 10, 1–11. https://doi.org/10.1177/20552076241299065

## 서지정보

- 저자: Ui-jae Hwang, Jin-seong Kim, Keong-yoon Kim, Kyu-sung Chung
- 연도: 2024
- 저널: DIGITAL HEALTH
- DOI: 10.1177/20552076241299065
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/04_machine_learning_and_deep_learning/Machine learning models for predicting return to sports after anterior cruciate ligament reconstruction - Physical performance in early rehabilitation.pdf
- 분석 provider: antigravity

## 연구 목적

- 전방십자인대 재건술(ACLR) 후 3개월 시점의 신체 수행 능력 변수를 바탕으로, 수술 후 12개월 시점의 스포츠 복귀(RTS) 성공을 예측하는 데 가장 우수한 성능을 보이는 머신러닝 모델을 식별하는 것이다. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “In this study, we aimed to identify the best-performing machine learning models for predicting RTS at 12 months post-ACLR, based on physical performance variables at 3 months post-ACLR.”

## 연구 설계와 대상

- 이 연구는 단일 기관에서 단일 의사에 의해 해부학적 단일 다발 전방십자인대 재건술(ACLR)을 받은 18세에서 45세 사이의 환자 102명을 대상으로 한 후향적 환자-대조군 연구이다. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “This case-control study included 102 patients who had undergone ACLR.”
- > **[AS-IS]** 연구 대상자는 2016년 6월부터 2022년 4월 사이에 수술을 받고 수술 후 3개월 및 12개월 시점에 요구되는 모든 테스트를 완료했으며 동반 다발 인대 손상, 골절, 반월상 연골 봉합/절제술, 개정 ACLR 등의 제외 기준에 해당하지 않는 환자들로 구성되었다.
>
> **[TO-BE]** 연구 대상자는 2016년 6월부터 2022년 4월 사이에 수술을 받고 수술 후 3개월 및 12개월 시점의 요구 검사를 완료했으며, 동반 다발 인대 손상, 골절, 반월상연골 뿌리 봉합, 연골 복원, 정렬 교정 절골술, 반월상연골 아전절제 또는 전절제, 재수술 ACLR, 양측 무릎 수술 과거력 등의 제외 기준에 해당하지 않는 환자들로 구성되었다.
>
> _(사실검증 — 사실불일치/중대: 요약은 제외 기준을 '반월상 연골 봉합/절제술'로 넓게 적었지만, 원문은 'meniscal root repair'와 'subtotal or total meniscectomy'를 제외 기준으로 제시한다. 모든 반월상연골 봉합이나 절제술이 제외된 것처럼 읽혀 대상자 기준이 달라진다.)_ _(근거: PAGE 2, Methods - Patients)_
  - 근거 원문: “The medical records of 102 patients who had undergone single-bundle anatomical ACLR using the outside-in technique with a ﬂip-cutter (Arthrex, Naples, FL, USA) between June 2016 and April 2022 were retrospectively reviewed to obtain their demographic and clinical characteristics. A single surgeon performed all the operations. The inclusion criteria for this study were patients who had undergone single-bundle ACLR, aged between 18 and 45 years, and complied with all the required tests at 3 and 12 months post-surgery. The exclusion criteria were as follows: concomitant mul-tiple ligament injury, fracture, meniscal root repair, cartil-age repair, osteotomy to correct mechanical alignment, subtotal or total meniscectomy, revision ACLR, and history of knee surgery on the involved and uninvolved sides.”

## 방법

- 수술 후 3개월 시점에 Biodex 균형 시스템(BBS) 테스트, Y-밸런스 테스트(YBT), 등속성 근력 테스트(concentric strength)를 수행하여 독립 변수를 수집하였고, 수술 후 12개월 시점에 외다리 홉 테스트, 외다리 수직 점프 테스트, Tegner 활동 점수를 타겟 변수로 측정하였다. _(근거: PAGE 2, Methods - Procedure)_
  - 근거 원문: “The physical performance variables (as feature or independent variables) measured at 3 months post-ACLR included the Biodex balance system (BBS) test, Y-balance test (YBT), and isokinetic muscle strength test. The RTS variables (as target or dependent variables) measured at 12 months post-ACLR include the single-leg hop test, single-leg vertical jump test, and Tegner activity score.”
- 12개월 시점의 3가지 RTS 타겟 변수는 외다리 홉 및 수직 점프 테스트의 대칭 지수(LSI)가 10% 미만이고 Tegner 활동 점수가 6점 초과인 경우를 성공 기준으로 하여 이분형 변수로 변환되었다. _(근거: PAGE 4, Methods - Machine learning modeling - Pre-processing and missing data handling)_
  - 근거 원문: “The three RTS targets (single-leg hop test, single-leg vertical jump test, and Tegner activity score) were transformed into dichotomous variables as an LSI of the single-leg hop and vertical jump test<10% and a Tegner activity score>6 points.”
- 102명의 데이터를 80%의 훈련 세트(82명)와 20%의 테스트 세트(20명)로 분할하고, 로지스틱 회귀, 의사결정 나무, 랜덤 포레스트, 그래디언트 부스팅, 서포트 벡터 머신, 인공신경망 등 6가지 알고리즘을 5-fold 교차 검증을 통해 훈련했다. _(근거: PAGE 4, Methods - Machine learning modeling - Machine learning algorithm)_
  - 근거 원문: “We split the complete data (n= 102) into a training set (80%, n=82) for model development and a test set (20%, n=20) for external validation to predict model performance. Six machine learning algorithms were trained via a ﬁve-fold cross-validation, including logistic regression, decision tree, random forest, gradient boosting, support vector machine, and neural network.”

## 핵심 결과

- 외다리 홉 테스트 기반 RTS 성공 예측의 경우 테스트 세트에서 랜덤 포레스트 모델이 가장 높은 성능(AUC 0.952)을 나타냈다. _(근거: PAGE 5, Results - Predictive models of machine learning)_
  - 근거 원문: “Random forest models in the test set best predicted the RTS success based on the single-leg hop test (area under the curve [AUC], 0.952) and Tegner activity score (AUC, 0.949).”
- Tegner 활동 점수 기반 RTS 성공 예측의 경우 테스트 세트에서 랜덤 포레스트 모델이 가장 높은 성능(AUC 0.949)을 보여주었다. _(근거: PAGE 5, Results - Predictive models of machine learning)_
  - 근거 원문: “Regarding RTS success prediction based on Tegner activity score, the random forest algorithm models had the highest AUC in the training (AUC, 0.826 [good]; F1, 0.751) and test (AUC, 0.949 [excellent]; F1, 0.952) sets.”
- 외다리 수직 점프 테스트 기반 RTS 성공 예측의 경우 테스트 세트에서 그래디언트 부스팅 모델이 가장 높은 성능(AUC 0.868)을 나타냈다. _(근거: PAGE 1, Abstract - Results)_
  - 근거 원문: “Gradient boosting models in the test set best predicted the RTS based on the single-leg vertical jump test (AUC, 0.868).”

## 저자 결론

- 전방십자인대 재건술(ACLR) 후 조기 재활 단계(3개월 시점)에서 수정 가능한 요인들을 고려함으로써 성공적인 스포츠 복귀(RTS) 가능성을 향상시킬 수 있다. _(근거: PAGE 1, Abstract - Conclusion)_
  - 근거 원문: “Modiﬁable factors should be considered in the early rehabilitation stage after ACLR to enhance the possibility of a successful RTS.”

## 연구의 한계

- 머신러닝을 적용하기에 표본의 크기(n=102)가 상대적으로 작아 분석 결과의 일반화가 제한될 수 있다. _(근거: PAGE 8, Discussion)_
  - 근거 원문: “First, the sample size was relatively small for machine learning applications, which may limit the ﬁndings’ generalizability.”
- 대퇴사두근 및 햄스트링 근력 강도에 상당한 영향을 줄 수 있는 이식건 종류나 수술 기법에 대한 통제를 분석 모형에 적용하지 않았다. _(근거: PAGE 8, Discussion)_
  - 근거 원문: “Second, we did not incorporate controls for variables such as surgical technique or graft type, despite the well-documented knowledge that these factors can exert distinct inﬂuences on quadriceps and hamstring strength outcomes.”
- 심리적 요인, 수술 관련 수치, 부상 전의 활동 수준 등 비성능 지표 변수들이 예측 모델에서 제외되었다. _(근거: PAGE 8, Discussion)_
  - 근거 원문: “Third, although physical performance outcomes at 3 months post-ACLR were analyzed in the prediction models, other potential predictive variables, such as psychological factors, surgical metrics, and pre-injury activity levels, were omitted.”
- 머신러닝 알고리즘의 다양성으로 인해 본 연구에서 선택한 6가지 모델 이외에 더 우수한 다른 알고리즘 모델이 존재할 가능성이 있다. _(근거: PAGE 8, Discussion)_
  - 근거 원문: “Fourth, the best-performing model may not have been one of the six models we selected, as machine learning algorithms are diverse.”
- 단일 기관에서 획득된 데이터만을 사용하여 진행되었기 때문에 향후 다기관 연구 및 독립 샘플을 통한 검증이 필요하다. _(근거: PAGE 8, Discussion)_
  - 근거 원문: “Lastly, our study was limited to data from a single institution. A multi-centric approach using an independent sample to val-idate the models could enhance the generalizability of the results in future studies.”

## 생각해볼 내용

- 이 연구 결과의 임상적 의의는 재활 초기 단계에서 스포츠 복귀 결과가 좋지 않을 위험이 있는 환자를 선별하고, 보다 목표 지향적이고 개인화된 재활 프로토콜을 적용할 수 있는 기회를 제공한다는 점이다. 그러나 실제 임상 적용을 위해서는 더 큰 규모의 다기관 코호트 연구를 통한 검증이 반드시 수반되어야 한다. _(근거: PAGE 6, Discussion)_
  - 근거 원문: “The clinical implications of our ﬁndings include the potential to identify patients at risk of poor RTS outcomes early in the rehabilitation process. This could allow for more targeted interventions and personalized rehabilitation protocols. However, further validation in larger, multi-center cohorts is needed before clinical implementation.”

## 이 연구가 지적한 선행연구의 문제점

- ACLR 이후 12개월 시점의 RTS 예측을 위해 수술 전이나 수술 후 6개월 시점의 추적 관찰 결과를 활용하는 머신러닝 연구는 주목을 받았으나, 조기 재활 단계(3개월 시점)에서의 예측 인자 개발은 덜 강조되었다. _(근거: PAGE 2, Introduction)_
  - 근거 원문: “Considerable attention has been paid to machine learning models in predicting preoperative26,27 or 6 months follow-up outcomes28,29 for RTS at 12 months post-ACLR, yet less emphasis has been placed on predictors at the early stages of rehabilitation.”

## 이 연구의 해결 방식과 기여

- ACLR 수술 후 3개월 시점에 측정한 등속성 근력 및 균형 능력 등의 조기 재활 단계 변수들을 활용한 18가지 머신러닝 예측 모델을 검증하여 우수한 스포츠 복귀 예측 성능을 가진 모델을 제시했다. _(근거: PAGE 5, Discussion)_
  - 근거 원문: “In this study, we selected eight predictive variables at 3 months post-ACLR and three outcome variables of RTS at 12 months post-ACLR to validate 18 machine learning models, with six machine learning algorithms for each clinical outcome.”
- 본 연구에서 도출된 수정 가능한 변수들은 12개월 시점의 성공적인 스포츠 복귀를 돕기 위해 조기 재활 단계에서 수행할 수 있는 맞춤형 운동 또는 물리치료 치료법의 방향성을 제시한다. _(근거: PAGE 6, Discussion)_
  - 근거 원문: “The modiﬁable variables presented in this study can guide exercise or physical therapy in the early rehabilitation stage for successful RTS at 12 months post-ACLR.”

## 레퍼런스할 수 있는 내용

### 1. 전방십자인대 재건술 후 스포츠 복귀 비율 및 이전 수준 회복 확률

- 원문 발췌: “However, a successful ACLR does not assure a patient’s RTS, as studies indicate that only 63% of patients regain their preinjury activity levels, with reported return rates ranging from 39% to 74%.2–6”
- 한국어 번역: 그러나 성공적인 ACLR이 환자의 RTS를 보장하지는 않는데, 연구들에 따르면 63%의 환자만이 부상 전의 활동 수준을 회복하고 보고된 복귀 비율은 39%에서 74% 사이이기 때문이다.
- 원문 위치: PAGE 1, Introduction
- 원문 내 인용표기: 2–6
- 해당 선행문헌: 2. Ardern CL, Taylor NF, Feller JA, et al. Fifty-ﬁve per cent return to competitive sport following anterior cruciate ligament reconstruction surgery: an updated systematic review and meta-analysis including aspects of physical functioning and contextual factors. Br J Sports Med 2014; 48: 1543–1552.
3. Suzuki M, Ishida T, Matsumoto H, et al. Association of psychological readiness to return to sports with subjective level of return at 12 months after ACL reconstruction. Orthop J Sports Med 2023; 11: 23259671231195030.
4. Joreitz R, Lynch A, Rabuck S, et al. Patient-speciﬁc and surgery-speciﬁc factors that affect return to sport after ACL reconstruction. Int J Sports Phys Ther 2016; 11: 264.
5. LindangerL,StrandT,Mølster AO,et al.Effectof earlyresidual laxity after anterior cruciate ligament reconstruction on long-term laxity, graft failure, return to sports, and subjective outcome at 25 years. Am J Sports Med 2021; 49: 1227–1235.
6. Ortiz E, Zicaro JP, Mansilla IG, et al. Revision anterior cruciate ligament reconstruction: return to sports at a minimum 5-year follow-up. World J Orthop 2022; 13: 12.
- 주장 유형: background_citation
- 활용 맥락과 주의: 전방십자인대 재건술 후 복귀 실패율 및 이전 수준으로 회복하는 비율을 서술할 때 배경 근거로 인용하기 적절함.

### 2. ACLR 수술 직후 무릎 기능 저하 기전

- 원문 발췌: “Following ACLR, patients experience reduced knee muscle strength due to hamstring graft harvesting and quadriceps inhibition,7 along with postural stability compromise attributed to ACL mechanoreceptor injury.8”
- 한국어 번역: ACLR 이후 환자들은 햄스트링 이식건 채취 및 대퇴사두근 억제로 인한 무릎 근력 감소와 함께, ACL 기계적 수용기 손상으로 인한 자세 안정성 저하를 경험한다.
- 원문 위치: PAGE 1, Introduction
- 원문 내 인용표기: 7,8
- 해당 선행문헌: 7. de Jong SN, van Caspel DR, van Haeff MJ, et al. Functional assessment and muscle strength before and after reconstruction of chronic anterior cruciate ligament lesions. Arthroscopy 2007; 23: 21. e21–221. 11.
8. Paterno MV, Schmitt LC, Ford KR, et al. Biomechanical measures during landing and postural stability predict second anterior cruciate ligament injury after anterior cruciate ligament reconstruction and return to sport. Am J Sports Med 2010; 38: 1968–1978.
- 주장 유형: background_citation
- 활용 맥락과 주의: 수술 후 무릎 근력 및 자세 균형 능력이 감소하는 기전에 대한 논리를 구성할 때 배경 인용문으로 유용함.

### 3. 3개월 시점의 조기 재활 치료의 초점

- 원문 발췌: “Within the ﬁrst 3 months after ACLR, the primary focus of rehabilitation is to reduce pain, restore quadriceps and hamstring strength, and incorporate proprioception training.49,50”
- 한국어 번역: ACLR 이후 첫 3개월 이내에 재활의 일차적인 초점은 통증을 줄이고 대퇴사두근 및 햄스트링 근력을 회복시키며, 고유수용성 감각 훈련을 통합하는 것이다.
- 원문 위치: PAGE 7, Discussion
- 원문 내 인용표기: 49,50
- 해당 선행문헌: 49. Erickson LN, Jacobs CA, Johnson DL, et al. Psychosocial factors 3-months after anterior cruciate ligament reconstruction predict 6-month subjective and objective knee outcomes. J Orthop Res 2022; 40: 231–238.
50. Kline PW, Johnson DL, Ireland ML, et al. Clinical predictors of knee mechanics at return to sport following ACL reconstruction. Med Sci Sports Exercise 2016; 48: 90.
- 주장 유형: background_citation
- 활용 맥락과 주의: 수술 직후부터 3개월까지 초기 재활 단계에서 반드시 집중해야 하는 치료적 목표 및 요인들을 설명할 때 적합함.

### 4. 12개월 시점 RTS 예측을 위한 최적 머신러닝 예측 성능 및 알고리즘

- 원문 발췌: “Random forest models in the test set best predicted the RTS success based on the single-leg hop test (area under the curve [AUC], 0.952) and Tegner activity score (AUC, 0.949).”
- 한국어 번역: 테스트 세트의 랜덤 포레스트 모델은 외다리 홉 테스트(AUC, 0.952) 및 Tegner 활동 점수(AUC, 0.949)를 바탕으로 한 RTS 성공을 가장 잘 예측했다.
- > **[AS-IS]** 원문 위치: PAGE 5, Results
>
> **[TO-BE]** 원문 위치: PAGE 1, Abstract - Results
>
> _(사실검증 — 인용표기오류/경미: 해당 항목의 원문 발췌문은 SOURCE_TEXT에서 PAGE 1의 Abstract - Results에 그대로 제시된다. PAGE 5 Results에는 같은 결과가 더 풀어 서술되어 있으나, 요약에 적은 직접 발췌문과 정확히 일치하는 위치는 PAGE 1이다.)_
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: 본 연구의 핵심 발견으로, 조기 신체 기능을 활용하여 높은 예측력으로 12개월 시점의 스포츠 복귀 가능 여부를 분류해내는 랜덤 포레스트 알고리즘의 성과를 직접 제시할 때 인용 가능함.

### 5. 재활 운동 치료를 통한 신체적 성능 변수 개선 가능성

- 원문 발췌: “Physical performance factors, such as strength, balance, and abnormal biomechanical patterns, can be modiﬁed through rehabilitation.”
- 한국어 번역: 근력, 균형, 비정상적인 생체역학적 패턴과 같은 신체 수행 능력 요인들은 재활을 통해 수정될 수 있다.
- 원문 위치: PAGE 2, Introduction
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: uncited_author_claim
- 활용 맥락과 주의: 수술 후 재활 훈련이 타깃으로 삼는 신체 성능 지표들의 가소성과 중재 가능성을 강조할 때 활용할 수 있는 저자의 일반적 주장임.
