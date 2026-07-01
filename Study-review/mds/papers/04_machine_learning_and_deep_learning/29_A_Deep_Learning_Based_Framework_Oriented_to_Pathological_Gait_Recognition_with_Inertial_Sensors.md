# A Deep Learning-Based Framework Oriented to Pathological Gait Recognition with Inertial Sensors

Palazzo, L., Suglia, V., Grieco, S., Buongiorno, D., Brunetti, A., Carnimeo, L., Amitrano, F., Coccia, A., Pagano, G., D’Addio, G., & Bevilacqua, V. (2025). A Deep Learning-Based Framework Oriented to Pathological Gait Recognition with Inertial Sensors. Sensors, 25(1), 260. https://doi.org/10.3390/s25010260

## 서지정보

- 저자: Lucia Palazzo, Vladimiro Suglia, Sabrina Grieco, Domenico Buongiorno, Antonio Brunetti, Leonarda Carnimeo, Federica Amitrano, Armando Coccia, Gaetano Pagano, Giovanni D’Addio, Vitoantonio Bevilacqua
- 연도: 2025
- 저널: Sensors
- DOI: https://doi.org/10.3390/s25010260
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/04_machine_learning_and_deep_learning/A Deep Learning-Based Framework Oriented to Pathological Gait Recognition with Inertial Sensors.pdf
- 분석 provider: antigravity

## 연구 목적

- 본 연구의 목적은 관성 데이터를 사용하여 건강한 피험자가 모사한 비정상적인 보행 패턴과 정상 보행을 구별하는 CNN 기반 알고리즘의 평가를 제시하는 것이다. _(근거: PAGE 2, Section 1. Introduction)_
  - 근거 원문: “The objective of this work is to present the evaluation of CNN-based algorithms that aim to discriminate normal gait from abnormal human walking patterns, which are emulated by healthy subjects, by means of inertial data.”

## 연구 설계와 대상

- 본 연구는 병리적 보행의 타당한 모사를 보장하기 위해 IRCCS Maugeri의 재활의학과 의사 및 물리치료사 중 19명의 건강한 피험자(남성 9명, 여성 10명)를 모집했다. _(근거: PAGE 4, Section 3.1.1. Participants)_
  - 근거 원문: “Nineteen healthy subjects were recruited among the physiatrists and physiotherapists of IRCCS Maugeri (Bari, Italy) to guarantee a plausible simulation of pathological gaits. Proper balancing among males and females was guaranteed (i.e., 9 males and 10 females) to prevent the model from being biased by sex [31].”

## 방법

- 본 연구에서는 정상 보행과 4가지 병리 보행(실조성, 첨족/발처짐, 편마비, 파킨슨병)을 포함하는 실험 프로토콜을 수행하였다. _(근거: PAGE 4, Section 3.1.2. Walking Actions)_
  - 근거 원문: “In light of this, in addition to normal walking, four pathological gaits were considered and they are ataxic, equine (foot drop), hemiplegic, and Parkinsonian gaits [32].”
- 각 피험자는 5개의 관성 측정 장치(IMU) 센서를 양측 골반, 양측 손목, 그리고 흉골에 착용하고 보행 데이터를 수집하였다. _(근거: PAGE 5, Section 3.1.3. IMU Sensors)_
  - 근거 원문: “Five sensors were selected (see Figure 1) and worn by each participant on both sides of the human pelvis (RP and LP), on the right and left wrists (RW and LW), and on the sternum (S).”
- 분류 파이프라인은 신호를 50% 중첩되는 128개 샘플(1초)의 윈도우로 나누는 윈도잉 절차를 적용하였다. _(근거: PAGE 6, Section 3.2.1. Preprocessing)_
  - 근거 원문: “Subsequently, a windowing procedure was applied to enlarge the dataset dimensionality by dividing the signal into windows of 128 samples (1 s) with 50% overlap (0.5 s); this window width was chosen so as to capture enough motor patterns without excessively increasing the computational cost [39].”

## 핵심 결과

- 모든 보행 패턴은 평균 정확도 100%로 분류되어 기존 연구를 능가하는 결과를 얻었다. _(근거: PAGE 11, Section 4. Results and Discussion)_
  - 근거 원문: “all walking patterns have been classified with an average accuracy of 100%, thus outperforming related works.”
- smCNN-1D 모델을 도입하여 mCNN-1D 모델보다 거의 모든 조합에서 유의하게 낮은 테스트 추론 시간을 달성하였다. _(근거: PAGE 9, Section 4. Results and Discussion)_
  - 근거 원문: “Consequently, the model architecture has been simplified, thus reaching with the smCNN-1D model a test inference time that is significantly lower than the one of the mCNN-1D model in almost all combinations; in addition, the maximum time decreases from approximately 700 ms to about 400 ms for the LP+LW and S+LP+RW sensor pairs.”

## 저자 결론

- 본 연구는 건강한 피험자로부터 수집된 데이터를 통해 정상 및 비정상 보행 패턴을 분류하기 위한 딥러닝 기반 프레임워크의 효과를 입증하였다. _(근거: PAGE 13, Section 5. Conclusions)_
  - 근거 원문: “Given the promising performance of the models used in terms of accuracy and inference time, the authors claim the effectiveness of the proposed workflow in discriminating motor patterns.”

## 연구의 한계

- 본 연구는 실제 환자의 데이터가 아닌 건강한 피험자가 수행한 정상 및 비정상 보행 데이터만을 대상으로 테스트된 예비 타당성 조사라는 한계가 있다. _(근거: PAGE 13, Section 5. Conclusions)_
  - 근거 원문: “Therefore, the proposed workflow should be evaluated by studying data coming from people actually affected by gait disorders to test its usefulness in a clinical scenario.”

## 생각해볼 내용

- 건강한 피험자를 통한 사전 도메인 적응 모델의 훈련 가능성을 시사하여, 환자 데이터 수집의 한계를 극복하기 위한 방법론적 대안을 제시한 점이 우수하다. _(근거: PAGE 2, Section 1. Introduction)_
  - 근거 원문: “In so doing, the effectiveness of a classification pipeline can be evaluated prior to any investigations on actual pathological individuals [2]; this is similar to the concept of cross-subject domain adaptation [26], meaning that the model is pre-trained on abnormal walking patterns simulated by healthy controls before being finally tested on actual pathological data.”

## 이 연구가 지적한 선행연구의 문제점

- 기존의 기계 학습 기반 보행 인식 파이프라인은 실제 분류를 수행하기 전에 복잡하고 시간이 많이 소요되는 특징 공학 단계를 필요로 했다. _(근거: PAGE 3, Section 2. Related Works)_
  - 근거 원문: “Notwithstanding, these pipelines needed a complex and time-demanding feature engineering stage prior to the actual classification [3,8].”

## 이 연구의 해결 방식과 기여

- 본 연구는 수동 특징 추출을 피하기 위해 원시 데이터를 직접 학습할 수 있는 CNN 기반의 딥러닝 아키텍처를 도입하여 복잡한 특징 공학 단계를 배제하였다. _(근거: PAGE 3, Section 2. Related Works)_
  - 근거 원문: “On the other hand, Deep Learning (DL) architectures, such as convolutional neural networks (CNNs) [6,9,31], can be trained directly on raw data, thus avoiding manual feature extraction [3,21].”

## 레퍼런스할 수 있는 내용

### 1. 인간 보행의 생리학적 특성

- 원문 발췌: “Human locomotion is a symmetric motor action [1] that requires the involvement of the central and peripheral nervous systems actuating mechanisms to control limb movements, posture, and muscle tone.”
- 한국어 번역: 인간의 보행은 사지 운동, 자세 및 근육 긴장도를 제어하기 위해 중추 및 말초 신경계의 메커니즘 활성화를 필요로 하는 대칭적인 운동 작용이다.
- 원문 위치: PAGE 1, Section 1. Introduction
- 원문 내 인용표기: [1]
- 해당 선행문헌: 1. Mekruksavanich, S.; Jitpattanakul, A. Deep Residual Network with a CBAM Mechanism for the Recognition of Symmetric and Asymmetric Human Activity Using Wearable Sensors. Symmetry 2024, 16, 554. [CrossRef]
- 주장 유형: background_citation
- 활용 맥락과 주의: 인간 보행의 생리학적 기전과 대칭성에 대한 선행 연구적 근거로 활용 가능하며, 2차 인용에 주의해야 한다.

### 2. 운동 장애 및 신경질환과 비정상 보행 패턴의 관계

- 원문 발췌: “Abnormal locomotor patterns may occur in case of either motor damages or neurological conditions, thus potentially jeopardizing an individual’s safety.”
- 한국어 번역: 운동 장애 또는 신경학적 조건의 경우 비정상적인 보행 패턴이 발생할 수 있으며, 이는 개인의 안전을 위협할 가능성이 있다.
- 원문 위치: PAGE 1, Abstract
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: uncited_author_claim
- 활용 맥락과 주의: 저자의 일반적인 주장으로, 운동 장애나 신경학적 이상이 보행에 미치는 영향과 안전성 문제에 관한 배경 설명에 인용할 수 있다.

### 3. 제안된 딥러닝 프레임워크의 보행 패턴 분류 정확도

- 원문 발췌: “all walking patterns have been classified with an average accuracy of 100%, thus outperforming related works.”
- 한국어 번역: 모든 보행 패턴이 100%의 평균 정확도로 분류되어 관련 연구들을 능가하였다.
- 원문 위치: PAGE 11, Section 4. Results and Discussion
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: 본 논문의 핵심 실험적 발견으로, 제안된 딥러닝 기반 병리 보행 인식 프레임워크의 정확도 성능을 인용할 때 사용된다.
