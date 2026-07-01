# Estimation of Kinetics Using IMUs to Monitor and Aid in Clinical Decision-Making during ACL Rehabilitation: A Systematic Review

Krishnakumar, S., van Beijnum, B.-J. F., Baten, C. T. M., Veltink, P. H., & Buurke, J. H. (2024). Estimation of Kinetics Using IMUs to Monitor and Aid in Clinical Decision-Making during ACL Rehabilitation: A Systematic Review. Sensors, 24(7), 2163. https://doi.org/10.3390/s24072163

## 서지정보

- 저자: Sanchana Krishnakumar, Bert-Jan F. van Beijnum, Chris T. M. Baten, Peter H. Veltink, Jaap H. Buurke
- 연도: 2024
- 저널: Sensors
- DOI: 10.3390/s24072163
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/03_wearable_imu_and_portable_sensing/Estimation of Kinetics Using IMUs to Monitor and Aid in Clinical Decision-Making during ACL Rehabilitation - A Systematic Review.pdf
- 분석 provider: antigravity

## 연구 목적

- 본 연구는 대안적 알고리즘을 확인하고 종합적인 체계적 문헌고찰을 통해 ACL 재활에서의 적용 가능성을 평가하고자 한다. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “Therefore, this article aims to identify the available algorithms for the estimation of kinetic parameters using kinematics measured only from IMUs and to evaluate their applicability in ACL rehabilitation through a comprehensive systematic review.”
- 이 체계적 문헌고찰의 목적은 추가적인 힘 정보 없이 IMU만을 사용하여 하지 운동역학적 매개변수를 추정하는 알고리즘을 식별하고 논의하는 것이다. _(근거: PAGE 3, 1. Introduction)_
  - 근거 원문: “Therefore, the objective of this systematic review is to identify and discuss algorithms for the estimation of lower limb kinetic parameters only using IMUs without additional force information.”

## 연구 설계와 대상

- 이 체계적 문헌고찰은 PRISMA 성명에 따라 수행되었다. _(근거: PAGE 3, 2.1. Study Design)_
  - 근거 원문: “This systematic review was conducted in accordance with the Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA) statement (Supplementary Table S1) [24].”
- 문헌 선별 및 전판 스크리닝 결과, 정량적 데이터 합성을 위해 최종적으로 71개의 연구가 분석에 포함되었다. _(근거: PAGE 4, 2.4. Study Selection and Quality Assessment)_
  - 근거 원문: “Full-text screening was performed by the first reviewer (S.K), and 71 articles were included for quantitative data synthesis.”

## 방법

- 리뷰를 위해 PubMed, Scopus, SPORTDiscus 데이터베이스가 검색에 활용되었다. _(근거: PAGE 3, 2.2. Search Strategy)_
  - 근거 원문: “The databases used for this review were PubMed, Scopus (Elsevier), and SPORTDiscus (EBSCO host).”
- 포함된 연구들의 품질 평가는 Strom 등이 제시한 체크리스트를 수정하여 구성한 14점의 체크리스트를 통해 진행되었다. _(근거: PAGE 4, 2.4. Study Selection and Quality Assessment)_
  - 근거 원문: “Quality assessment of the included studies was performed (S.K) using a 14-point checklist comprising items listed in Table 1.”

## 핵심 결과

- 분석 대상 문헌들의 다수는 머신러닝 기반 모델을 사용하였으며(약 45%), 그 뒤를 생체역학 모델(약 38%)이 이었다. _(근거: PAGE 13, 4.1. Modelling Techniques and Estimated Kinetic Parameters)_
  - 근거 원문: “The majority of the reviewed articles utilized ML-based models and accounted for around 45% of the reviewed articles, followed by BM (∼38%).”
- 3차원 지면반발력(GRF) 추정에서 가장 우수한 RMSE 값은 수직 드롭 점프 동작에서 전후방 0.018, 내외측 0.008, 수직 0.038(체중 기준 표준화)로 달성되었다. _(근거: PAGE 13, 3.6. Accuracy and Reliability of Tested Approaches)_
  - 근거 원문: “For 3D GRF, the best RMSE values were achieved for vertical drop jump, namely, 0.018, 0.008, and 0.038 (normalized to body weight) for anterior–posterior GRF (AP-GRF), medio-lateral GRF (M-LGRF), and vertical GRF (VGRF) respectively [85].”
- 걷기 과제 중 3차원 순 무릎 관절 모멘트의 추정치에서는 외전-내전 모멘트가 10.58%, 굴곡-신전 모멘트가 9.46%, 외/내회전 모멘트가 17.12%로 가장 낮은 nRMSE(%)를 보여주었다. _(근거: PAGE 13, 3.6. Accuracy and Reliability of Tested Approaches)_
  - 근거 원문: “Among the articles that estimated 3D net knee joint moments, the lowest nRMSE (%) was observed for walking, with values of 10.58, 9.46, and 17.12 for abduction–adduction, flexion–extension, and external/internal rotation moments, respectively [30].”

## 저자 결론

- IMU 센서는 건강한 대상자를 기준으로 시상면상에서 일어나는 움직임에 대해 높은 정확도로 GRF 및 관절 운동역학적 매개변수를 추정할 수 있는 잠재력을 보여주었다. _(근거: PAGE 17, 5. Conclusions)_
  - 근거 원문: “The results of this review indicate that IMUs have good potential to estimate GRF and other joint kinetic parameters with good accuracy for movements primarily in the sagittal plane for healthy cohorts.”
- 하지만 체계적 문헌고찰에 포함된 알고리즘 중 실제 ACL 환자를 대상으로 검증을 거친 알고리즘은 단 하나도 없었다. _(근거: PAGE 17, 5. Conclusions)_
  - 근거 원문: “However, none of these algorithms have been validated on ACL patients.”

## 연구의 한계

- 본 문헌고찰에 포함된 분석 대상 논문이 영어로 작성된 문헌으로만 한정되어 선택 편향의 우려가 존재한다. _(근거: PAGE 16, 4.5. Limitations of the Included Evidence, Review Process, and Future Directions)_
  - 근거 원문: “The articles included in the review were also limited to only English.”
- 질 향상을 목표로 인간을 대상으로 유효성이 검증된 논문으로 분석 대상을 제한한 결정이 고찰 범위의 포괄성을 제한했을 가능성이 있다. _(근거: PAGE 16, 4.5. Limitations of the Included Evidence, Review Process, and Future Directions)_
  - 근거 원문: “The decision to include only articles that validated on human beings while improving the quality of the included results may have limited the inclusiveness of the review.”
- 각 분석 대상 연구에서 각기 다른 센서 부착 위치, 실험 프로토콜, 평가 매트릭을 사용하고 있어, 어떤 모델이 최고 수준의 정확도를 내는 최적 모델인지 직접적으로 비교하고 분석하기 어렵다. _(근거: PAGE 17, 4.5. Limitations of the Included Evidence, Review Process, and Future Directions)_
  - 근거 원문: “The use of varying sensor placement locations, experimental protocols, and reporting metrics used in the included articles made direct comparison and identification of the overall best model with the most accurate outcome challenging.”

## 생각해볼 내용

- 분석 대상 문헌들의 대다수에서 여성 참가자가 과소대표되었으며, 이는 부상 위험이 상대적으로 높은 여성 대상의 연구와 유효성 검증 필요성을 뒷받침한다. _(근거: PAGE 6, 3.2. Participant Characteristics)_
  - 근거 원문: “Female population was underrepresented in 55 articles (∼80%), while 6 articles (∼8%) [30,32,34–37] did not report complete information on the gender distribution of the study population.”
- 임상 및 ACL 재활 단계에서 반드시 고려되어야 할 단일 다리 홉(single-leg hop)이나 트리플 홉(triple hop) 등의 기동 동작에 관한 연구가 부재하다. _(근거: PAGE 16, 4.4. Applicability for ACL Rehabilitation)_
  - 근거 원문: “Important ACL rehabilitation-specific tasks such as single-leg hop and triple hop have not been studied.”

## 이 연구가 지적한 선행연구의 문제점

- 전통적인 보행 운동역학 평가 방식은 고가이면서 복잡하고, 부피가 큰 특성상 실험실 외부의 환경에서는 적용이 매우 제한적이다. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “Conventional methods deployed to estimate kinetics require complex, expensive systems and are limited to laboratory settings.”
- IMU 센서 데이터만을 이용해 보행 변수나 운동역학적 부하를 예측하는 기존 알고리즘들은 실제 환자군을 대상으로 유효성과 일반화 가능성을 검증한 결과가 매우 제한적이다. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “However, the knowledge about their accuracy and generalizability for patient populations is still limited.”

## 이 연구의 해결 방식과 기여

- 이 연구는 외부의 힘 계측 시스템 없이 IMU 센서 데이터만을 입력으로 사용해 하지 관절 모멘트 및 지면반발력을 예측하는 최신 알고리즘들을 비교하고 평가한 정보를 제공한다. _(근거: PAGE 3, 1. Introduction)_
  - 근거 원문: “Thus, a systematic review that compares and evaluates available algorithms for estimation of kinetic parameters (joint kinetics, GRF, and GRM) using only IMU data will provide insights on the state of the art of the accuracy, reliability, and applicability of the available algorithms.”
- 이 연구를 통해 밝힌 지식 공백들은 ACL 재활 모니터링은 물론, 유사 근골격계 질환의 임상적 활용 및 향후 보행 재훈련 프로토콜 개발 연구에 중요한 방향성을 제시한다. _(근거: PAGE 3, 1. Introduction)_
  - 근거 원문: “In addition, it will help to identify the gaps and opportunities for further research and open new avenues for clinical decision-making for ACL rehabilitation and for other conditions.”

## 레퍼런스할 수 있는 내용

### 1. ACL 재활 모니터링에서 정량적 생체역학 평가의 당위성

- 원문 발췌: “Since the treatment is currently based on subjective visual observations during clinical visits, there is huge potential to further optimize the training of patients using quantitative assessment of relevant biomechanical parameters.”
- 한국어 번역: 현재 임상 방문 시 주관적인 시각적 관찰을 토대로 치료 결정이 내려지기 때문에, 관련 생체역학적 매개변수의 정량적 평가를 통해 환자 훈련을 한층 더 최적화할 수 있는 여지가 매우 크다.
- 원문 위치: PAGE 2, 1. Introduction
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: uncited_author_claim
- > **[AS-IS]** 활용 맥락과 주의: 임상에서 수동적이거나 주관적인 외관 검사 대신, 환자의 운동 메커니즘을 보다 명밀하게 추적하기 위해 센서 기반의 정량적 평가가 요구되는 명분을 인용할 때 사용한다.
>
> **[TO-BE]** 활용 맥락과 주의: 임상 방문에서 주관적 시각 관찰에 의존하는 ACL 재활 의사결정을 보완하기 위해, 관련 생체역학 변수의 정량 평가 필요성을 제시할 때 사용할 수 있다.
>
> _(사실검증 — 근거불충분/경미: 원문은 임상 방문 중 주관적 시각 관찰 기반 치료와 정량적 생체역학 평가의 최적화 가능성을 말하지만, '수동적 검사', '외관 검사', '운동 메커니즘을 명밀하게 추적'이라는 표현은 해당 인용문만으로 직접 지지되지 않는다.)_

### 2. 보행 분석에서 계측형 러닝머신 측정 방식이 지닌 한계

- 원문 발췌: “Systems such as instrumented treadmills that measure GRF may also alter the natural pattern of gait [13,14].”
- 한국어 번역: 지면반발력(GRF)을 측정하는 장비가 내장된 러닝머신과 같은 시스템은 보행의 자연스러운 형태를 왜곡시킬 가능성이 있다.
- 원문 위치: PAGE 2, 1. Introduction
- 원문 내 인용표기: [13,14]
- 해당 선행문헌: 13. Lee, S.J.; Hidler, J. Biomechanics of overground vs. treadmill walking in healthy individuals. J. Appl. Physiol. 2008, 104, 747–755.
14. Veras, L.; Diniz-Sousa, F.; Boppre, G.; Devezas, V.; Santos-Sousa, H.; Preto, J.; Vilas-Boas, J.P.; Machado, L.; Oliveira, J.; Fonseca, H. Accelerometer-based prediction of skeletal mechanical loading during walking in normal weight to severely obese subjects. Osteoporos. Int. 2020, 31, 1239–1250.
- 주장 유형: background_citation
- 활용 맥락과 주의: 러닝머신 상에서의 보행이 실생활의 일반적인 평지 보행 패턴과 다를 수 있음을 시사하며, 일상 환경(wearable) 평가의 타당성을 피력하는 문헌적 근거로 활용할 수 있다. 2차 인용에 주의하여 활용해야 한다.

### 3. 여성 인구의 높은 ACL 부상 위험도

- 원문 발췌: “It is also important to note that the female population has an increased risk of ACL injury [100].”
- > **[AS-IS]** 한국어 번역: 여성 집단이 전방십자인대(ACL) 부상 위험에 노출될 확률이 훨씬 높다는 점 또한 중요하게 인지되어야 한다.
>
> **[TO-BE]** 한국어 번역: 여성 집단은 ACL 손상 위험이 증가되어 있다는 점 또한 중요하게 고려해야 한다.
>
> _(사실검증 — 과장/경미: 원문은 여성 집단의 ACL injury risk가 증가되어 있다고만 표현한다. 요약 번역의 '훨씬 높다'는 원문보다 강한 정도 표현이다.)_
- 원문 위치: PAGE 16, 4.4. Applicability for ACL Rehabilitation
- 원문 내 인용표기: [100]
- 해당 선행문헌: 100. The female ACL: Why is it more prone to injury? J. Orthop. 2016, 13, A1–A4.
- 주장 유형: background_citation
- 활용 맥락과 주의: 여성 운동선수 등을 타깃으로 하여 ACL 부상 예방을 하거나, 성별에 적합한 보행 모니터링 알고리즘 설계의 당위성을 제시할 때 논문 근거로 활용할 수 있다.
