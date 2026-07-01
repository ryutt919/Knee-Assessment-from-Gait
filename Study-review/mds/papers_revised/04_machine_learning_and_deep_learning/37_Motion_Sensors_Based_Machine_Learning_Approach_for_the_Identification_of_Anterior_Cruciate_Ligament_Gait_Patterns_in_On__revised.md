# Motion Sensors-Based Machine Learning Approach for the Identification of Anterior Cruciate Ligament Gait Patterns in On-the-Field Activities in Rugby Players

Tedesco, S., Crowe, C., Ryan, A., Sica, M., Scheurer, S., Clifford, A. M., Brown, K. N., & O’Flynn, B. (2020). Motion Sensors-Based Machine Learning Approach for the Identification of Anterior Cruciate Ligament Gait Patterns in On-the-Field Activities in Rugby Players. Sensors, 20(11), 3029. https://doi.org/10.3390/s20113029

## 서지정보

- 저자: Salvatore Tedesco, Colum Crowe, Andrew Ryan, Marco Sica, Sebastian Scheurer, Amanda M. Clifford, Kenneth N. Brown, Brendan O’Flynn
- 연도: 2020
- 저널: Sensors
- DOI: 10.3390/s20113029
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/04_machine_learning_and_deep_learning/Motion Sensors-Based Machine Learning Approach for the Identification of Anterior Cruciate Ligament Gait Patterns in On-the-Field Activities in Rugby Players.pdf
- 분석 provider: antigravity

## 연구 목적

- 낮은 비용과 작은 크기의 웨어러블 센서를 활용하여 방향 전환 활동을 수행하는 럭비 선수들을 대상으로 건강한 그룹과 ACL 재건 수술을 받은 그룹을 머신러닝을 통해 구별할 수 있는지 조사하고자 한다. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “In particular, this study aims to investigate the ability of a set of inertial sensors worn on the
lower-limbs by rugby players involved in a change-of-direction (COD) activity to differentiate
between healthy and post-ACL groups via the use of machine learning.”
- 방향 전환 활동 시 ACL 재건 그룹과 건강한 그룹 간의 유의미한 차이를 감지하여 ACL 손상의 장기적인 영향이 있는지 조사하고, 주체 관련 정보나 표준 보행 시공간 매트릭스 등과 무관한 자동화되고 객관적인 분류 방법을 제공하고자 한다. _(근거: PAGE 3, 1. Introduction)_
  - 근거 원문: “The aim of this study is two-fold:
(i) to investigate whether there is a long after-effect of the ACL damage in rugby players,
detecting significant differences in ACL-reconstructed vs. healthy players, when involved
in a change-of-direction activity;
(ii) to provide an automated and objective method to distinguish between healthy and post-ACL
groups of rugby players which is independent from subject-related information, step detection
and segmentation processes, and standard gait spatiotemporal metrics, through the combination
of a set of inertial sensors worn on the lower-limbs and data-driven machine learning models.”

## 연구 설계와 대상

- Irish 대학생 남성 럭비 선수 중 성공적으로 복귀한 ACL 재건군 6명과 건강한 대조군 6명을 포함하여 총 12명의 피험자가 본 연구에 참가하였다. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “Twelve male participants (six
healthy and six post-ACL athletes who were deemed to have successfully returned to competitive
rugby and tested in the 5–10 year period following the injury) were recruited for the study.”
- 분석 대상은 아일랜드 대학생들 중에서 e-mail, 포스터, 구두 홍보를 통해 모집된 12명의 비엘리트 남성 럭비 선수이다. _(근거: PAGE 3, 2.1. Participants)_
  - 근거 원문: “The analysis in this study is based on a sample of twelve non-elite rugby players (all males,
age: 26 ± 5.2 years; height: 182.6 ± 5.8 cm; mass: 90 ± 12.8 kg). Players were recruited via a general
invitation e-mail, posters, and word of mouth, to students at a University in Ireland.”
- 피험자들은 두 그룹으로 나뉘었다: 왼쪽 다리에 ACL 재건술 역사가 있는 6명의 선수와 하지 부상 이력이 없는 6명의 선수. _(근거: PAGE 3, 2.1. Participants)_
  - 근거 원문: “The subjects were divided in two groups: six players with a history of ACL reconstruction surgery
(age: 29.3 ± 4.5 years; height: 182.3 ± 6.2 cm; mass: 89.2 ± 14.7 kg), and six players with no history
of lower-limbs injuries (age: 22.8 ± 3.7 years; height: 182.8 ± 6.1 cm; mass: 90.8 ± 11.9 kg).”

## 방법

- 각 피험자는 시작 지점에서 출발하여 5m를 달린 후 좌측 또는 우측으로 45도 방향 전환(사이드스텝)을 수행한 뒤 3m를 더 질주하여 정지하는 작업을 10회 반복하였다. _(근거: PAGE 4, 2.2. Data Collection Protocol)_
  - 근거 원문: “Each participant began the data
collection at a pre-defined start point, and was asked to run for 5 m towards a side-step platform.
During the run, the participants were instructed regarding which direction the side-step had to occur
(left or right). They were then required to step at a 45 degree angle from the sidestep board in either
direction, and finally run an additional 3 m to come to a full stop.”
- 다리당 두 개의 관성 측정 장치(IMUs)가 경골 전면(경골 결절 아래 10cm) 및 외측 대퇴부(경골 결절 위 15cm)에 부착되어 3차원 가속도와 자이로스코프 데이터를 수집하였다. _(근거: PAGE 4, 2.2. Data Collection Protocol)_
  - 근거 원문: “Two inertial
measurement units (IMUs) were attached per leg, in particular to the anterior tibia, 10 cm below
the tibial tuberosity, and to the lateral thigh, 15 cm above the tibial tuberosity, using Velcro straps.”
- 테스트 시작 전 모든 착용 센서들의 시작 시점을 시간상 동기화하기 위해 피험자들이 깊은 스쿼트 동작을 수행하도록 하였다. _(근거: PAGE 4, 2.2. Data Collection Protocol)_
  - 근거 원문: “Before each
repetition, subjects were asked to perform a deep squat in order to temporally synchronize the
beginning of the test among all the sensors worn by the participant.”
- 동일한 하부 다리에 착용된 두 IMU의 기준 좌표계를 동일하게 보정하기 위해 Seel 등의 가상 회전 보정 방식을 적용하였다. _(근거: PAGE 5, 2.3. Preliminary Data Processing)_
  - 근거 원문: “Also, in order to have the same reference
system for both IMUs worn on the same leg, the method proposed by Seel et al. [26] has been adopted
to virtually rotate along the horizontal axis the raw inertial data recorded by the sensors worn on the
shank.”
- 수집된 3차원 가속도, 자이로스코프, 저크 신호 등의 통계적 분석 및 주파수 영역 특징 분석을 통해 총 250개의 무브먼트 특징들을 추출하였다. _(근거: PAGE 6, 2.4. Feature Extraction)_
  - 근거 원문: “From the data collected for each repetition, a number of features were extrapolated. The signals
considered for feature extraction were the 3D angular rate, the magnitude of the 3D acceleration, the
3D jerk signal obtained from differentiation of the 3D acceleration, and the 3D acceleration in the
body‐frame and gravity‐frame.”
- 머신러닝 모델의 일반화 능력을 정확하게 추정하기 위해 훈련 데이터에 포함되지 않은 미지의 사용자에 대한 성능을 추정하는 대상자 배제 교차 검증(LOSO-CV)을 사용하여 학습과 평가를 진행하였다. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “Feature selection was implemented in
the learning model, and leave-one-subject-out cross-validation (LOSO-CV) was adopted to estimate
training and test errors.”

## 핵심 결과

- Swing phase, relative stance phase, relative swing phase 등 일부 보행 파라미터는 조건(건강군/수술군)과 하지(좌/우) 간의 상호작용 및 주효과 모두에서 통계적으로 유의미한 차이를 보이지 않았다. _(근거: PAGE 8, 3.1. Gait Analysis Results)_
  - 근거 원문: “Some gait parameters (swing phase, relative stance phase, and relative swing phase) do not
show a statistically significant interaction between condition and limb, and likewise, do not show the
statistical significance of the main effects.”
- 수술 여부를 판별하기 위한 기계학습 모델 중 다층 퍼셉트론(MLP)이 테스트 데이터셋에서 분류 정확도 73.07%로 가장 높은 결과를 나타냈다. _(근거: PAGE 10, 3.2. Machine Learning Model Results)_
  - 근거 원문: “The MLP model shows an accuracy of 73.07% (SE: 8.99%),
sensitivity 78.01%, specificity 68.3%, precision 70.79%, F1-score 74.22%, and Cohen’s Kappa 0.462.”
- 그래디언트 부스팅(XGB) 모델은 분류 정확도 72.32%를 달성했으며, 민감도 81.8%를 나타내어 위음성을 낮추기 위한 측면에서 가장 유리한 성능을 나타냈다. _(근거: PAGE 10, 3.2. Machine Learning Model Results)_
  - 근거 원문: “TheXGB
model shows an accuracy of 72.32% (SE: 10.47%), sensitivity 81.8%, specificity 63.07%, precision 68.56%,
F1-score 74.6%, and Cohen’s Kappa 0.448.”

## 저자 결론

- 본 연구의 결과는 부상 후 5~10년이 지나 이미 성공적으로 스포츠에 복귀하고 정상으로 판단된 선수일지라도, 웨어러블 센서와 머신러닝 기법을 결합하여 필드 활동 시의 잔존하는 ACL 재건 무릎의 특이 보행 패턴을 성공적으로 구별해낼 수 있음을 입증한다. _(근거: PAGE 13, 4. Discussion)_
  - 근거 원문: “The results of this study clearly show that motion sensors can distinguish between players with
ACL-reconstructed knee and healthy players even after 5–10 years following the injury, despite the
previously injured athletes being deemed fully recovered.”

## 연구의 한계

- 참가자군이 모두 남성이고 표본 수가 적기 때문에 모델 결과의 일반화 능력이 제한될 수 있으며, 향후 더 많은 대상군을 통한 검증이 요구된다. _(근거: PAGE 14, 4. Discussion)_
  - 근거 원문: “Moreover, gender and small sample size are other limitations of the study which may limit the
generalizability of the results. Given the novelty of the study, the present investigation was designed
as a pilot proof-of-concept; larger cohort will need to be recruited in the future to confirm those results
as shown by the power analysis.”

## 생각해볼 내용

- 비록 통계적 유의성이 검출된 보행 파라미터들이 있으나 표본 크기가 작고 효과 크기가 매우 작기 때문에 실제 집단 간 차이 효과인지 신뢰하기 어려우며 더 큰 표본 규모의 검증이 필요하다. _(근거: PAGE 12, 4. Discussion)_
  - 근거 원문: “Therefore, even though some statistical significance was detected in the analysis, the small
observed power and effect size do not provide enough confidence that the difference seen between
groups for those variables was a real observed effect and, as a result, further larger studies should
be performed.”

## 이 연구가 지적한 선행연구의 문제점

- 현재 임상에서 흔히 사용하는 주관적 및 객관적 스포츠 복귀(RTS) 평가와 실제 복귀 성공률 간의 관계에 대한 임상적 증거가 부족하며, 기존 복귀 평가의 민감성에 의문이 제기된다. _(근거: PAGE 2, 1. Introduction)_
  - 근거 원문: “At present, there is a dearth of evidence supporting the
relationship between RTS and standard subjective and objective assessments, which questions if
existing RTS assessments and criteria are sensitive or demanding enough to elucidate clinically
relevant indicators.”
- Vicon과 같은 기존 마커 기반 모션 캡처 시스템은 고비용, 전문 분석실 접근성 제한 및 대규모 인원 적용의 한계로 인해 현장 활용에 어려움이 많다. _(근거: PAGE 2, 1. Introduction)_
  - 근거 원문: “While marker-based motion analysis systems (e.g., Vicon) [12] can provide objective assessments
and represent the gold-standard technology adopted in gait analysis for quantitative movement
analysis, their adoption is constrained by cost, access to specialist motion labs, as well as the practicality
of application for larger patient/subject groups and, thus, shows limited use for on-the-field players.”
- 웨어러블 센서를 활용한 선행 ACL 연구가 일부 존재하나, 대부분 실제 환경이 아닌 실내 실험실 트레드밀 환경에서 이루어졌으며 수술 후의 경과 시간 정보 등이 누락되어 있었다. _(근거: PAGE 3, 1. Introduction)_
  - 근거 원문: “However, the tests were carried out in a lab setting and the time since surgery was not provided.”

## 이 연구의 해결 방식과 기여

- > **[AS-IS]** 성공적으로 복귀한 지 5~10년이 경과한 건강한 선수와 ACL 재건 환자를 필드(야외) 환경에서 웨어러블 센서와 데이터 기반 기계학습 모델의 결합을 통해 정밀하게 분류해낸 연구는 본 연구가 최초이다. _(근거: PAGE 3, 1. Introduction)_
>
> **[TO-BE]** 저자들이 알기로, 스포츠에 복귀한 지 5~10년이 지난 ACL 재건 선수와 건강한 선수를 필드 환경에서 관성 센서와 데이터 기반 접근법을 결합해 분류한 연구는 아직 충분히 탐구되지 않았다.
>
> _(사실검증 — 과장/경미: 원문은 저자들이 알기로 해당 조합이 아직 탐구되지 않았다고 표현한다. 요약의 '본 연구가 최초'와 '정밀하게 분류'는 원문보다 단정적이고 강하다.)_
  - 근거 원문: “To the best of the authors’ knowledge, the combination of a data-driven approach and inertial
sensors to classify healthy and ACL-reconstructed subjects on-the-field (with post-ACL athletes
returned to sport and with time from surgery between five and 10 years) is not yet explored.”

## 레퍼런스할 수 있는 내용

### 1. ACL 재건 환자의 실제 스포츠 복귀율 및 경쟁 수준 복귀율 차이

- 원문 발췌: “However, on average, 80% of patients were found to return
to sport, with only 55% returning to competitive levels after ACL reconstruction [4].”
- 한국어 번역: 그러나 평균적으로 80%의 환자가 스포츠로 복귀하지만, ACL 재건술 후 경쟁력 있는 수준으로 복귀하는 환자는 55%에 불과하다.
- 원문 위치: PAGE 2, 1. Introduction
- 원문 내 인용표기: [4]
- 해당 선행문헌: 4. Ardern, C.L.; Webster, K.E.; Taylor, N.F.; Feller, J.A. Return to sport following anterior cruciate ligament
reconstruction surgery: A systematic review and meta-analysis of the state of play. Br. J. Sports Med. 2011,
45, 596–606. [CrossRef] [PubMed]
- 주장 유형: background_citation
- 활용 맥락과 주의: ACL 수술 이후 전체적인 스포츠 복귀와 실제 고수준 경쟁 복귀 간의 간극을 설명할 때 인용 근거로 활용할 수 있다. Ardern 등(2011)의 메타분석 연구에서 도출된 수치로 2차 인용에 주의해야 한다.

### 2. 필드 스포츠 활동 중 관성 센서 기반 ACL 재건 환자와 대조군 분류 타당성

- 원문 발췌: “The results of this study clearly show that motion sensors can distinguish between players with
ACL-reconstructed knee and healthy players even after 5–10 years following the injury, despite the
previously injured athletes being deemed fully recovered.”
- 한국어 번역: 본 연구 결과는 이전에 부상당한 선수들이 임상적으로 완전히 회복된 것으로 간주되었음에도 불구하고, 부상 후 5~10년이 지난 시점의 현장 스포츠 활동에서 동작 센서가 ACL 재건 무릎 환자와 건강한 선수를 판별할 수 있음을 명확히 보여준다.
- 원문 위치: PAGE 13, 4. Discussion
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- > **[AS-IS]** 활용 맥락과 주의: 수술 후 장기 시점에서도 가혹한 필드 방향 전환 테스트 중 웨어러블 센서를 통해 ACL 재건군과 대조군 사이의 잔존 보행 차이를 기계학습으로 성공적으로 식별할 수 있다는 본 논문의 자체 결과를 인용할 때 사용한다.
>
> **[TO-BE]** 활용 맥락과 주의: 수술 후 5~10년이 지난 선수의 필드 방향 전환 과제에서 웨어러블 센서와 머신러닝 접근법이 ACL 재건군의 보행 패턴 식별에 활용될 가능성을 보였다는 본 논문의 자체 결과를 인용할 때 사용한다.
>
> _(사실검증 — 과장/경미: 원문은 필드 스포츠 과제에서 센서와 머신러닝 접근의 feasibility와 구별 가능성을 제시하지만, '가혹한' 테스트라는 표현은 원문 표현보다 강하며, '성공적으로 식별'도 73.07% 정확도와 81.8% 민감도의 제한적 결과를 충분히 반영하지 않는다.)_
