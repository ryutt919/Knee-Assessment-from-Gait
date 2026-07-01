# Learning based lower limb joint kinematic estimation using open source IMU data

Hur, B., Baek, S., Kang, I., & Kim, D. (2025). Learning based lower limb joint kinematic estimation using open source IMU data. Scientific Reports, 15, 5287. https://doi.org/10.1038/s41598-025-89716-4

## 서지정보

- 저자: Benjamin Hur, Sunin Baek, Inseung Kang, Daekyum Kim
- 연도: 2025
- 저널: Scientific Reports
- DOI: https://doi.org/10.1038/s41598-025-89716-4
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/04_machine_learning_and_deep_learning/Learning based lower limb joint kinematic estimation using open source IMU data.pdf
- 분석 provider: antigravity

## 연구 목적

- 본 연구는 관성 측정 장치(IMU)와 딥러닝 프레임워크를 활용하여 하지 관절 운동학(kinematics)을 추정하는 프레임워크를 제안하고 평가하고자 한다. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “This study introduces a deep learning framework for estimating lower-limb joint kinematics using
inertial measurement units (IMUs).”

## 연구 설계와 대상

- 오픈소스 데이터셋에서 건강한 성인 11명(남성 9명, 여성 2명, 데이터 손상으로 4명 제외)의 보행 데이터를 분석에 사용하였다. _(근거: PAGE 3, Methods - Open-source data)_
  - 근거 원문: “For this study, we used an open-source gait data that included IMU(MTw Awinda, Xsens North America
Inc., Culver City, CA, USA) and OMC(Motion Analysis Corporation, Santa Rosa, CA, USA) from 11 healthy
individuals (9 males and 2 females)27.”
- 모델의 적응성 검증을 위해 추가적으로 3명의 피험자로부터 3미터 직선 보행 및 180도 회전을 수행하는 독립적인 데이터를 수집하였다. _(근거: PAGE 3, Methods - Data collection for model adaptability verification)_
  - 근거 원문: “Our dataset includes IMU and OMC data from three
individuals performing a 5-minute session of 3 meters straight walking and 180◦
turns at a self-selected pace.”

## 방법

- 하지 관절 운동학 추정을 위해 CNN과 LSTM 두 가지 딥러닝 네트워크 아키텍처를 구현하여 비교 분석하였다. _(근거: PAGE 4, Methods - Deep learning)_
  - 근거 원문: “In developing our model for estimating lower-limb kinematics during gait, we used Convolutional Neural
Networks (CNNs) and Long Short-Term Memory (LSTMs) networks, which are widely used and are proven to
be effective in deep learning applications for joint kinematic estimation26,33.”
- 훈련 모델은 세 가지 방식(개인화 UI, 일반화 UG, 적응형 UA)으로 학습하고 평가를 수행하였다. _(근거: PAGE 5, Methods - Deep learning)_
  - 근거 원문: “Weevaluatedthreedifferentmethodsofselectingtrainingdatatodevelopdeeplearningmodelsthatestimate
lower-limb kinematics: the ‘user-individualized method’, the ‘user-generalized method’, and the ‘user-adaptive
method’.”
- 각 관절각의 참값(Ground Truth)은 광학 모션 캡처(OMC) 기반의 데이터를 OpenSim 4.4의 역운동학(Inverse Kinematics) 연산을 통해 도출하여 사용하였다. _(근거: PAGE 5, Methods - Inverse kinematics)_
  - 근거 원문: “Forthegroundtruthvaluesofjointangles,weusedOMC-basedinversekinematicscalculatedthroughOpenSim
4.4.”

## 핵심 결과

- 개인화 방식(UI) LSTM 모델은 IMU 기반 역운동학에 비해 평균 RMSE는 49.20%, NRMSE는 50.65% 낮았으며, 상관계수는 20.13% 높았다. _(근거: PAGE 6, Results - Overall model performance comparison)_
  - 근거 원문: “Specifically,theLSTMmodelshowed49.20%loweraverageRMSE,
50.65% lower average NRMSE, and 20.13% higher correlation coefficient compared to IMU-based inverse
kinematics.”
- 일반화 방식(UG)은 IMU 기반 역운동학에 비해 RMSE가 LSTM의 경우 115.87%, CNN의 경우 121.50% 증가하여 높은 오차를 보였다. _(근거: PAGE 6, Results - Overall model performance comparison)_
  - 근거 원문: “Compared to RMSE values of
IMU-based inverse kinematics, the UG showed higher error, with RMSE values 115.87% and 121.50% larger
for the LSTM and CNN models, respectively.”
- 적응형 방식(UA) LSTM 모델은 IMU 기반 역운동학보다 평균 RMSE는 0.4%, NRMSE는 7% 약간 더 낮아 역운동학과 대등한 성능을 보였다. _(근거: PAGE 6, Results - Overall model performance comparison)_
  - 근거 원문: “The average RMSE and NRMSE for the UA LSTM
model were slightly lower than those of IMU-based inverse kinematics (0.4% and 7% respectively).”
- 다양한 IMU 조합 중에서 대퇴골(femur)과 종골(calcaneus)에 장착된 IMU를 결합했을 때 굴곡/신전(sagittal) 엉덩관절 및 무릎관절 각도 추정에서 가장 낮은 RMSE 오차를 나타냈다. _(근거: PAGE 6, Results - IMU combinations)_
  - 근거 원문: “Across
all methods, IMU combinations including femur and calcaneus showed the lowest RMSE values for
sagittal hip and knee joint angle estimation.”

## 저자 결론

- 오픈소스 IMU 데이터셋 기반의 전이학습(UA)을 통해 개인화된 하지 관절 각도 추정 모델의 정확도를 대폭 개선할 수 있으며, 데이터 수집 노력과 비용을 대폭 줄일 수 있다. _(근거: PAGE 8, Discussion)_
  - 근거 원문: “This demonstrated that a personalized joint kinematic
estimation model can be constructed by developing a generalized pre-trained model using open-source datasets
from prior studies and applying transfer learning with a small portion of a novel individual’s data. Because only
aminimalamountofnewdataisnecessary,thisapproachsignificantlyreducedboththetimeandcostassociated
with data collection and model training43,44.”

## 연구의 한계

- UA 모델의 테스트가 소수의 피험자에게만 적용되어, 성능의 보편성을 검증하기 위해 더 광범위하고 다양한 데이터셋에 대한 추가 검증이 필요하다. _(근거: PAGE 9, Discussion)_
  - 근거 원문: “One key limitation is that the UA
was tested on only a few participants. While this provided an initial proof of concept, further testing across a
broader range of datasets is necessary to gain a more comprehensive understanding of its performance.”
- 사전 학습된 기저 모델(UG)이 상대적으로 적은 인원의 데이터로 구축되어 초기 예측 정확도가 낮았으며, 이는 전이학습 모델의 최종 성능 향상에 제한을 준다. _(근거: PAGE 10, Discussion)_
  - 근거 원문: “Another limitation is the training of the base model, which
relied on a limited set of participants. The pre-trained base model used in this study, referred to as the UG, was
developedwithdatafromarelativelysmallgroupofindividuals,resultinginlowerestimationaccuracy.”
- 전이학습(UA)을 수행하기 위해서는 여전히 소량의 마커 기반 광학 모션 캡처(OMC) 데이터가 필요하므로, OMC 시스템에 대한 의존성이 한계로 존재한다. _(근거: PAGE 10, Discussion)_
  - 근거 원문: “Additionally, the UA requires a small amount of marker-based OMC data for transfer learning, indicating
that continued dependence on OMC systems remains a limitation.”

## 생각해볼 내용

- 오픈소스 데이터셋을 딥러닝 기반 하지 운동학 추정에 체계적으로 활용하는 방법론을 제안하여 데이터 부족 문제를 극복하고자 시도하였다. _(근거: PAGE 2, Introduction)_
  - 근거 원문: “Thus, it is evident that developing a more generalizable model
and a systematic approach to handling open-source data for IMU-based kinematic estimation is critical and
beneficial for the field.”

## 이 연구가 지적한 선행연구의 문제점

- 기존의 분석적 방식(역운동학)은 모든 사지에 센서를 부착해야 하여 사전 준비 과정이 복잡하고 정렬/캘리브레이션 부담이 큽니다. _(근거: PAGE 1, Introduction)_
  - 근거 원문: “However, these approaches require IMUs to be located on each individual limb13,14,
introducing exhaustive pre-setup requirements (e.g., aligning and calibrating IMUs for each subject).”
- 기존 분석 방식은 누적 오차로 인해 장기 신호 표류(drift) 현상이 발생합니다. _(근거: PAGE 1, Introduction)_
  - 근거 원문: “Additionally,
IMU-based inverse kinematics suffer from long-term signal drift, primarily due to accumulated errors from
time-varying biases in integrating acceleration data12,13.”
- 개별 피험자의 신체적 차이(관절 길이 등)와 보행 양상의 다양성으로 인해 학습된 모델이 새로운 사용자에게 적용될 때 성능이 저하되는 한계가 있습니다. _(근거: PAGE 2, Introduction)_
  - 근거 원문: “For example,
variations in each individual’s joint lengths and gait patterns can cause the model to perform poorly when
deploying a trained model to new users13.”

## 이 연구의 해결 방식과 기여

- 오픈소스 데이터셋을 효율적으로 활용하기 위해 일반화된 사전 모델을 구축하고 전이학습(transfer learning)을 접목한 딥러닝 프레임워크를 제안하였습니다. _(근거: PAGE 2, Introduction)_
  - 근거 원문: “In this work, we present a deep learning framework that leverages an existing open-source dataset for IMU-
based joint kinematic estimation during walking.”
- 전이학습 기법을 활용함으로써 개별적인 연구에서 방대한 데이터를 새롭게 수집해야 하는 시간적, 비용적 부담을 대폭 감소시켰습니다. _(근거: PAGE 2, Introduction)_
  - 근거 원문: “We also showed that transfer learning enables researchers to efficiently utilize
open-source datasets, minimizing the need for extensive data collection for their own specific motor tasks.”
- 비교 분석을 통해 하지 관절 운동학 추정을 위한 최적의 IMU 개수와 부착 위치를 제안하여 실시간 처리 및 계산량 감축을 유도하였습니다. _(근거: PAGE 2, Introduction)_
  - 근거 원문: “Lastly, through comparative analysis, we determined the optimal number of IMUs and their locations.”

## 레퍼런스할 수 있는 내용

### 1. IMU 기반 역운동학의 신호 드리프트 현상

- 원문 발췌: “Additionally,
IMU-based inverse kinematics suffer from long-term signal drift, primarily due to accumulated errors from
time-varying biases in integrating acceleration data12,13.”
- 한국어 번역: 또한, IMU 기반 역운동학은 가속도 데이터를 적분할 때 발생하는 시간 변동 바이어스의 누적 오류로 인해 주로 발생하는 장기적인 신호 드리프트 문제를 겪는다.
- 원문 위치: PAGE 1, Introduction
- 원문 내 인용표기: 12,13
- 해당 선행문헌: 12. Picerno, P. 25 years of lower limb joint kinematics by using inertial and magnetic sensors: A review of methodological approaches.
Gait & Posture. 51, 239–246 (2017).
13. Hafer,J.F.,etal.Challengesandadvancesintheuseofwearablesensorsforlowerextremitybiomechanics.JournalofBiomechanics,
2023.
- 주장 유형: background_citation
- 활용 맥락과 주의: IMU 센서의 가속도계 데이터 적분 시 생기는 누적 오차로 인한 드리프트 문제를 환기시킬 때 인용할 수 있음. 2차 인용 시 원래 문헌인 Picerno(2017) 및 Hafer et al.(2023)을 검토해야 함.

### 2. 개별 신체적 차이와 보행 패턴의 다양성이 딥러닝 모델의 범용성을 저해함

- 원문 발췌: “For example,
variations in each individual’s joint lengths and gait patterns can cause the model to perform poorly when
deploying a trained model to new users13.”
- 한국어 번역: 예를 들어, 개별 사용자의 관절 길이와 보행 패턴의 다양성은 훈련된 모델을 새로운 사용자에게 배포할 때 성능이 저하되는 원인이 될 수 있다.
- 원문 위치: PAGE 2, Introduction
- 원문 내 인용표기: 13
- 해당 선행문헌: 13. Hafer,J.F.,etal.Challengesandadvancesintheuseofwearablesensorsforlowerextremitybiomechanics.JournalofBiomechanics,
2023.
- 주장 유형: background_citation
- 활용 맥락과 주의: 학습된 딥러닝 모델이 새로운 피험자나 환경에 적용되었을 때 개인 차이로 인해 겪는 성능 저하 한계를 지적할 때 유용함.
