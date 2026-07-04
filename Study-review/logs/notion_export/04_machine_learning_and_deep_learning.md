# [29] A Deep Learning-Based Framework Oriented to Pathological Gait Recognition with Inertial Sensors

(저자: Lucia Palazzo, Vladimiro Suglia, Sabrina Grieco, Domenico Buongiorno, Antonio Brunetti, Leonarda Carnimeo, Federica Amitrano, Armando Coccia, Gaetano Pagano, Giovanni D’Addio, Vitoantonio Bevilacqua | 연도: 2025 | 저널: Sensors | DOI: https://doi.org/10.3390/s25010260)

Palazzo, L., Suglia, V., Grieco, S., Buongiorno, D., Brunetti, A., Carnimeo, L., Amitrano, F., Coccia, A., Pagano, G., D’Addio, G., & Bevilacqua, V. (2025). A Deep Learning-Based Framework Oriented to Pathological Gait Recognition with Inertial Sensors. Sensors, 25(1), 260. https://doi.org/10.3390/s25010260

## 서지정보

- 저자: Lucia Palazzo, Vladimiro Suglia, Sabrina Grieco, Domenico Buongiorno, Antonio Brunetti, Leonarda Carnimeo, Federica Amitrano, Armando Coccia, Gaetano Pagano, Giovanni D’Addio, Vitoantonio Bevilacqua
- 연도: 2025
- 저널: Sensors
- DOI: https://doi.org/10.3390/s25010260
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/04_machine_learning_and_deep_learning/A Deep Learning-Based Framework Oriented to Pathological Gait Recognition with Inertial Sensors.pdf
- 분석 provider: antigravity

## 연구 목적

- 본 연구의 목적은 관성 데이터를 사용하여 건강한 피험자가 모사한 비정상적인 보행 패턴과 정상 보행을 구별하는 CNN 기반 알고리즘의 평가를 제시하는 것이다. *(근거: PAGE 2, Section 1. Introduction)*
	- 근거 원문: “The objective of this work is to present the evaluation of CNN-based algorithms that aim to discriminate normal gait from abnormal human walking patterns, which are emulated by healthy subjects, by means of inertial data.”

## 연구 설계와 대상

- 본 연구는 병리적 보행의 타당한 모사를 보장하기 위해 IRCCS Maugeri의 재활의학과 의사 및 물리치료사 중 19명의 건강한 피험자(남성 9명, 여성 10명)를 모집했다. *(근거: PAGE 4, Section 3.1.1. Participants)*
	- 근거 원문: “Nineteen healthy subjects were recruited among the physiatrists and physiotherapists of IRCCS Maugeri (Bari, Italy) to guarantee a plausible simulation of pathological gaits. Proper balancing among males and females was guaranteed (i.e., 9 males and 10 females) to prevent the model from being biased by sex \[31\].”

## 방법

- 본 연구에서는 정상 보행과 4가지 병리 보행(실조성, 첨족/발처짐, 편마비, 파킨슨병)을 포함하는 실험 프로토콜을 수행하였다. *(근거: PAGE 4, Section 3.1.2. Walking Actions)*
	- 근거 원문: “In light of this, in addition to normal walking, four pathological gaits were considered and they are ataxic, equine (foot drop), hemiplegic, and Parkinsonian gaits \[32\].”
- 각 피험자는 5개의 관성 측정 장치(IMU) 센서를 양측 골반, 양측 손목, 그리고 흉골에 착용하고 보행 데이터를 수집하였다. *(근거: PAGE 5, Section 3.1.3. IMU Sensors)*
	- 근거 원문: “Five sensors were selected (see Figure 1) and worn by each participant on both sides of the human pelvis (RP and LP), on the right and left wrists (RW and LW), and on the sternum (S).”
- 분류 파이프라인은 신호를 50% 중첩되는 128개 샘플(1초)의 윈도우로 나누는 윈도잉 절차를 적용하였다. *(근거: PAGE 6, Section 3.2.1. Preprocessing)*
	- 근거 원문: “Subsequently, a windowing procedure was applied to enlarge the dataset dimensionality by dividing the signal into windows of 128 samples (1 s) with 50% overlap (0.5 s); this window width was chosen so as to capture enough motor patterns without excessively increasing the computational cost \[39\].”

## 핵심 결과

- 모든 보행 패턴은 평균 정확도 100%로 분류되어 기존 연구를 능가하는 결과를 얻었다. *(근거: PAGE 11, Section 4. Results and Discussion)*
	- 근거 원문: “all walking patterns have been classified with an average accuracy of 100%, thus outperforming related works.”
- smCNN-1D 모델을 도입하여 mCNN-1D 모델보다 거의 모든 조합에서 유의하게 낮은 테스트 추론 시간을 달성하였다. *(근거: PAGE 9, Section 4. Results and Discussion)*
	- 근거 원문: “Consequently, the model architecture has been simplified, thus reaching with the smCNN-1D model a test inference time that is significantly lower than the one of the mCNN-1D model in almost all combinations; in addition, the maximum time decreases from approximately 700 ms to about 400 ms for the LP+LW and S+LP+RW sensor pairs.”

## 저자 결론

- **[AS-IS]** 본 연구는 건강한 피험자로부터 수집된 데이터를 통해 정상 및 비정상 보행 패턴을 분류하기 위한 딥러닝 기반 프레임워크의 효과를 입증하였다. *(근거: PAGE 13, Section 5. Conclusions)*<br>**[TO-BE]** 본 연구는 정확도와 추론 시간 측면에서 유망한 성능을 바탕으로, 건강한 피험자 데이터에서 정상 및 모사된 비정상 보행 패턴을 구별하는 딥러닝 기반 워크플로의 효과 가능성을 제시하였다.<br>*(사실검증 — 과장/경미: 원문은 모델의 정확도와 추론 시간 성능이 유망하므로 저자들이 워크플로의 효과를 주장한다고 표현한다. 요약의 ‘입증하였다’는 예비 타당성 연구라는 원문의 한정된 결론보다 강한 표현이다.)*
	- 근거 원문: “Given the promising performance of the models used in terms of accuracy and inference time, the authors claim the effectiveness of the proposed workflow in discriminating motor patterns.”

## 연구의 한계

- 본 연구는 실제 환자의 데이터가 아닌 건강한 피험자가 수행한 정상 및 비정상 보행 데이터만을 대상으로 테스트된 예비 타당성 조사라는 한계가 있다. *(근거: PAGE 13, Section 5. Conclusions)*
	- 근거 원문: “Therefore, the proposed workflow should be evaluated by studying data coming from people actually affected by gait disorders to test its usefulness in a clinical scenario.”

## 생각해볼 내용

- **[AS-IS]** 건강한 피험자를 통한 사전 도메인 적응 모델의 훈련 가능성을 시사하여, 환자 데이터 수집의 한계를 극복하기 위한 방법론적 대안을 제시한 점이 우수하다. *(근거: PAGE 2, Section 1. Introduction)*<br>**[TO-BE]** 건강한 피험자가 모사한 비정상 보행 데이터를 활용하면 실제 병리 보행 데이터 조사 전에 분류 파이프라인의 효과를 평가하거나, 실제 병리 데이터에 적용하기 전 사전학습 자료로 사용할 수 있음을 시사한다.<br>*(사실검증 — 근거불충분/경미: 원문은 건강한 피험자의 모사 데이터를 실제 병리 데이터 적용 전 평가나 사전학습에 활용할 수 있다고 설명하지만, ‘우수하다’는 평가적 판단 자체는 SOURCE_TEXT에서 직접 확인되는 저자 주장이나 결과가 아니다.)*
	- 근거 원문: “In so doing, the effectiveness of a classification pipeline can be evaluated prior to any investigations on actual pathological individuals \[2\]; this is similar to the concept of cross-subject domain adaptation \[26\], meaning that the model is pre-trained on abnormal walking patterns simulated by healthy controls before being finally tested on actual pathological data.”

## 이 연구가 지적한 선행연구의 문제점

- 기존의 기계 학습 기반 보행 인식 파이프라인은 실제 분류를 수행하기 전에 복잡하고 시간이 많이 소요되는 특징 공학 단계를 필요로 했다. *(근거: PAGE 3, Section 2. Related Works)*
	- 근거 원문: “Notwithstanding, these pipelines needed a complex and time-demanding feature engineering stage prior to the actual classification \[3,8\].”

## 이 연구의 해결 방식과 기여

- **[AS-IS]** 본 연구는 수동 특징 추출을 피하기 위해 원시 데이터를 직접 학습할 수 있는 CNN 기반의 딥러닝 아키텍처를 도입하여 복잡한 특징 공학 단계를 배제하였다. *(근거: PAGE 3, Section 2. Related Works)*<br>**[TO-BE]** 본 연구는 수동 특징 추출을 줄일 수 있는 CNN 기반 딥러닝 접근을 사용했으며, 원문은 이러한 딥러닝 아키텍처가 원시 데이터에서 직접 학습할 수 있다고 설명한다.<br>*(사실검증 — 과장/경미: 인용된 원문은 일반적으로 CNN 등 딥러닝 아키텍처가 원시 데이터에서 직접 학습해 수동 특징 추출을 피할 수 있다고 설명한다. 그러나 요약은 이를 본 연구의 구체적 기여로 단정하고 ‘복잡한 특징 공학 단계를 배제하였다’고 표현해 원문 근거보다 강하다.)*
	- 근거 원문: “On the other hand, Deep Learning (DL) architectures, such as convolutional neural networks (CNNs) \[6,9,31\], can be trained directly on raw data, thus avoiding manual feature extraction \[3,21\].”

## 레퍼런스할 수 있는 내용

### 1. 인간 보행의 생리학적 특성

- 원문 발췌: “Human locomotion is a symmetric motor action \[1\] that requires the involvement of the central and peripheral nervous systems actuating mechanisms to control limb movements, posture, and muscle tone.”
- 한국어 번역: 인간의 보행은 사지 운동, 자세 및 근육 긴장도를 제어하기 위해 중추 및 말초 신경계의 메커니즘 활성화를 필요로 하는 대칭적인 운동 작용이다.
- 원문 위치: PAGE 1, Section 1. Introduction
- 원문 내 인용표기: \[1\]
- 해당 선행문헌: 1. Mekruksavanich, S.; Jitpattanakul, A. Deep Residual Network with a CBAM Mechanism for the Recognition of Symmetric and Asymmetric Human Activity Using Wearable Sensors. Symmetry 2024, 16, 554. \[CrossRef\]
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


---

# [30] Wearable IMU sensor-based motion tracking for post-stroke motor impairment assessment during activities of daily living

(저자: Tanguy Lewko | 연도: 2024 | 저널: Master's Thesis, Harvard University | DOI: https://doi.org/확인 불가)

Lewko, T. (2024). Wearable IMU sensor-based motion tracking for post-stroke motor impairment assessment during activities of daily living (Master's thesis, Harvard University).

## 서지정보

- 저자: Tanguy Lewko
- 연도: 2024
- 저널: Master's Thesis, Harvard University
- DOI: 확인 불가
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/04_machine_learning_and_deep_learning/Calculate_Distance.pdf
- 분석 provider: antigravity

## 연구 목적

- 일상생활 기능 동작(ADLs) 중 뇌졸중 환자의 상지 움직임을 시간적으로 분할하고 평가하기 위한 웨어러블 IMU 센서 플랫폼의 개발 *(근거: Page 2, Abstract)*
	- 근거 원문: “The objective of this research is thus to develop a wearable IMU sensor platform to segment and evaluate stroke survivors’ upper extremity motions during ADLs.”
- 일상생활 활동을 지속적으로 모니터링하여 환측 상지의 사용 빈도 및 운동 품질에 관한 피드백을 환자에게 제공하고, 이를 통해 자가 재활 프로그램의 순응도를 촉진함 *(근거: Page 2, Abstract)*
	- 근거 원문: “Continuous monitoring of ADLs is a viable solution to provide feedback on the quantity and quality of impaired arm functional movements and to promote adherence to home rehabilitation programs.”

## 연구 설계와 대상

- 제안한 활동 분할 알고리즘의 유효성을 검증하기 위해 통제된 환경에서 5명의 건강한 대조군과 5명의 뇌졸중 환자를 대상으로 비디오 그라운드 트루스를 활용한 자연스러운 ADL 데이터셋을 직접 수집함 *(근거: Page 2, Abstract)*
	- 근거 원문: “To demonstrate the feasibility of our activity segmentation algorithm, we collected a dataset of 5 healthy participants and 5 post-stroke participants. We tested our algorithm on two mildly impaired post-stroke participants, highlighting the potential need for tailored models for post-stroke individuals due to the high variability of impairments within the population.”
- 활동 분할 알고리즘의 초기 개발 및 검증을 위해 10명의 건강한 참가자(오른손잡이, 22\~28세)가 6가지 일방 상지 ADL을 수행한 첫 번째 오픈소스 데이터셋(Multi-Sensory dataset)을 활용함 *(근거: Page 2, Abstract)*
	- 근거 원문: “The first dataset contained healthy participants performing six unimanual upper extremity self-care ADLs and was used to develop a deep learning algorithm for precise time segmentation of ADLs.”
- 건강한 대조군과 뇌졸중 환자 간의 운동 분석과 특징 추출을 위해 양치질과 물 마시기 활동 중 상체에 IMU를 장착한 두 번째 오픈소스 데이터셋(StrokeRehab dataset)을 활용함 *(근거: Page 2, Abstract)*
	- 근거 원문: “The second open-source dataset was used to extract samples of healthy and post-stroke participants wearing IMUs on their upper body while drinking and brushing their teeth.”

## 방법

- 합성곱(CNN) 레이어와 양방향 장단기 메모리(BiLSTM) 레이어를 결합하여 IMU 데이터의 시공간 특징을 동시 추출하고, 과거/미래 맥락 정보를 활용하는 33Hz 해상도의 밀집 라벨링(dense labeling) 하이브리드 모델 설계 *(근거: Page 2, Abstract)*
	- 근거 원문: “We developed a hybrid architecture combining convolutional and bidirectional long short-term memory layers to extract spatial and temporal information from the input IMU data. Our algorithm employs dense labeling for time-step resolution segmentation at a frequency of 33 hertz, while integrating past and future knowledge into current predictions through a context-aware framework.”
- 소수 클래스 과소 분류 문제를 예방하기 위해 다수 클래스 샘플 수의 1.5분의 1에 상응하는 크기로 복제하여 가중치를 맞추는 무작위 오버샘플링 적용 *(근거: Page 13, Chapter 3.2)*
	- 근거 원문: “We opt for random oversampling due to the relatively small size of our dataset. We randomly sample the data with replacement until each class is represented by a minimum number of samples equal to the number of samples present in the majority class divided by 1.5, allowing us to balance all the classes we want to segment.”
- 데이터 부족 한계를 해결하고 노이즈 및 장착 편차 강인성을 제고하기 위해 Jittering, Offsetting, Scaling, Zooming, Rotating, Time warping, Down-sampling 등의 데이터 증강 기술 구현 *(근거: Page 20, Chapter 3.3.4)*
	- 근거 원문: “For time series data augmentation, several commonly utilized methods that we implemented include: 1. Jittering Jittering aims to increase the robustness of the model to disturbances by adding noise to the data.”
- 건강한 참가자의 움직임 대조군 대비 불일치성을 다변량 정규 분포 하의 공분산 구조를 반영해 측정하기 위해 주성분 분석(PCA) 차원 축소와 마할라노비스 거리(Mahalanobis distance) 모델을 융합한 MHS 평가지표 개발 *(근거: Page 2, Abstract)*
	- 근거 원문: “The MHS utilizes the Mahalanobis distance to measure the disparity between the movement of a stroke survivor and the same movement executed by healthy subjects.”

## 핵심 결과

- 양 전완에 부착된 2개의 IMU 센서 데이터(가속도계 및 자이로스코프)만을 사용했을 때 성능이 가장 우수했으며, 대상자 간 평가에서 0.83, 대상자 내 평가에서 0.87의 F1 점수를 각각 확보함 *(근거: Page 2, Abstract)*
	- 근거 원문: “We achieved F1 scores of 0.83 and 0.87 in across-subjects and within-subjects evaluations, respectively. Additionally, we determined that segmentation accuracy was highest when using only the accelerometer and gyroscope data from two IMU sensors attached to the forearms.”
- 동작 품질 분류 척도인 MHS의 임계값 분류 정확도는 양치질에서 76%, 물 마시기 transport 단계에서 82%, recovery 단계에서 84%를 나타내 환자와 대조군을 효과적으로 분별함 *(근거: Page 51, Chapter 7.1.2)*
	- 근거 원문: “The optimal thresholds for brushing teeth, drinking transport, and drinking recovery scores were respectively 9.9, 9.3, and 9.5, leading to accuracies of 0.76, 0.82, and 0.84.”
- 뇌졸중 환자는 양치질 수행 시 건강한 대조군에 비해 더 높은 회전 속도(각속도)를 사용하고, 팔 가속도는 더 작게 발생시키며, 어깨 거상(abduction 및 flexion)의 관절 범위가 좁아지는 기구학적 특성을 지님 *(근거: Page 39, Chapter 6.3)*
	- 근거 원문: “Our analyses showed that stroke participants use less accelerations, higher rotational velocities, and smaller shoulder elevation when brushing their teeth.”
- 물 마시기 동작 수행 과정에서 뇌졸중 환자는 대조군 대비 전완 각속도 기반의 SPARC 평탄도 수치가 유의미하게 낮아 동작의 매끄러움이 저하됨을 나타냄 *(근거: Page 41, Chapter 6.4.2)*
	- 근거 원문: “Figures 6.10 and 6.11 show that the healthy group exhibits significantly smoother transport and recovery motion when drinking compared to the mildly and moderately impaired stroke participants”
- **[AS-IS]** 뇌졸중 환자(GG98 및 GG102)를 대상으로 한 ADL 분할 테스트에서 환자 개인 데이터를 반영하지 않은 일반화 모델(F1 0.51\~0.65) 대비 개인 맞춤화 모델(individualized model)은 F1 점수 각각 0.85, 0.87로 월등한 수준을 달성함 *(근거: Page 29, Chapter 5.1)*<br>**[TO-BE]** 뇌졸중 환자(GG98 및 GG102)를 대상으로 한 ADL 분할 테스트에서 개인 맞춤화 모델은 본문 기준 GG98 0.87, GG102 0.88의 ADL segmentation F1 점수를 보였으며, Table 5.1에는 각각 0.85와 0.87로 제시되어 있다.<br>*(사실검증 — 수치오류/중대: 요약은 individualized model의 F1을 GG98=0.85, GG102=0.87로 적었지만, 원문 Table 5.1에서 ADL segmentation의 individualized F1은 GG98=0.85, GG102=0.87이고, 바로 위 본문 설명은 GG98=0.87, GG102=0.88로 서술되어 서로 불일치한다. 또한 요약의 근거 원문으로 제시한 문장은 GG98=0.87, GG102=0.88이라고 되어 있어 요약 수치와 맞지 않는다.)*
	- 근거 원문: “Table 5.1 presents results for each of the settings for the activity segmentation model. The results were much better for the individualized models. For subject GG98, we reached F1 scores of 0.87 for the ADL segmentation and 0.71 for the meal segmentation task. For subject GG102, we obtained F1 scores of 0.88 for the ADL segmentation and 0.82 for the meal segmentation task.”

## 저자 결론

- **[AS-IS]** 전완 양측 2개의 IMU 센서와 CNN-BiLSTM 모델을 접목한 시공간 시점 분할 알고리즘이 건강한 대상자 및 경증 뇌졸중 환자의 일상생활 활동을 세밀한 타임스텝 수준으로 정확하게 분할해낼 수 있음을 입증함 *(근거: Page 54, Chapter 8)*<br>**[TO-BE]** 전완 양측 2개의 IMU 센서와 CNN-BiLSTM 모델을 활용해 건강한 참가자 데이터에서 UE ADL의 연속 시계열 분할 성능을 보였고, 두 명의 경증 뇌졸중 환자에서는 proof-of-concept 수준으로 적용 가능성을 탐색했다.<br>*(사실검증 — 과장/중대: 원문 결론은 원시 IMU 데이터와 CNN-biLSTM 기반 알고리즘으로 UE ADL을 정확히 분할할 수 있음을 보였고, 두 명의 뇌졸중 환자에서 proof-of-concept를 보였다고 한다. 그러나 요약은 건강한 대상자와 경증 뇌졸중 환자 모두에서 정확한 타임스텝 분할이 입증되었다고 강하게 일반화한다. 원문은 뇌졸중 환자에 대해서는 proof-of-concept와 tailored model 필요성을 강조한다.)*
	- 근거 원문: “We have demonstrated the ability to accurately segment UE ADLs from raw IMU data using a deep learning algorithm that utilizes both CNN and biLSTM layers. While most existing work focuses on classifying a pre-segmented window as part of an activity, we focused on precisely segmenting continuous non-pre-segmented data.”
- 뇌졸중 생존자는 환자간 상지 마비 양상의 편차가 매우 심하므로, 단일한 일반화 모델 보다는 환자의 상태와 움직임 유형을 반영하는 개인화된 맞춤형 알고리즘(individualized model)의 보완이 필수적임 *(근거: Page 54, Chapter 8)*
	- 근거 원문: “Our results highlight the potential need for tailored models for post-stroke individuals due to their high motion variability.”

## 연구의 한계

- **[AS-IS]** 학습 및 평가에 투입한 기성 공인 데이터셋(StrokeRehab)에 원본 영상 자료가 배제되어 알고리즘 오감지의 임상적 사유 규명이나 산출된 마할라노비스 거리 척도의 실제 눈 건강성 검증이 불가능하였음 *(근거: Page 54, Chapter 8)*<br>**[TO-BE]** 기성 오픈소스 데이터셋에 원본 영상 등 중요한 정보가 없어 오감지의 이유를 해석하기 어려웠고, MHS에서 건강한 동작에 가깝다고 평가된 움직임이 실제로 건강한 움직임처럼 보이는지도 확인할 수 없었다.<br>*(사실검증 — 번역오류/경미: 요약의 '실제 눈 건강성 검증'은 원문의 'appeared healthy or not'를 부자연스럽고 의미가 틀리게 옮긴 표현이다. 원문은 눈 건강성이 아니라, MHS에서 건강한 동작에 가깝다고 판단된 움직임이 실제로 건강한 움직임처럼 보이는지 확인할 수 없었다는 뜻이다.)*
	- 근거 원문: “The first limitation of this research comes from the use of open-source datasets with important information missing. The datasets we used did not include video data, which made it difficult to interpret the results of our algorithms. For example, we were unable to determine why the algorithm made certain false detections for activity segmentation. Additionally, for the motion healthiness score, we could not verify whether motions that were considered close to healthy appeared healthy or not.”
- 개발된 시점 분할 알고리즘이 오직 경증 환자(mild post-stroke)를 대상으로 파일럿 테스트가 이루어져 중등도나 중증 이상의 운동장애 환자에 대한 타당성이 부족함 *(근거: Page 54, Chapter 8)*
	- 근거 원문: “Activity segmentation should still be evaluated in stroke participants with moderate or severely affected stroke participants.”
- 구현된 모든 검증 과정이 조절 및 정제된 실험실 내 환경에서 진행되어 복잡하고 불규칙한 가정 내(home setting) 실제 일상활동 상에서의 일반화 성과가 불확실함 *(근거: Page 54, Chapter 8)*
	- 근거 원문: “Our algorithms were tested in controlled environments and we need to evaluate their performance in a home setting.”
- 환자가 마비의 대체재로 사용하는 주된 이상 동작인 '보상 패턴(trunk compensation 등)'의 존재 여부와 감지 기능이 MHS 점수 모델 구조에 완전하게 결합되지 못했음 *(근거: Page 54, Chapter 8)*
	- 근거 원문: “Our analysis of motion quality left out some important aspects such as compensation, which is a good indicator of bad use.”

## 생각해볼 내용

- **[AS-IS]** 제시된 환자 개인화 모델(individualized model)의 ADL 분할 성능(F1 0.85-0.87)은 우수하나, 학습 데이터 수집을 위해 가정 내에서 매번 환자 본인의 환측 동작에 대해 3.5시간 이상 비디오 라벨링 등의 고비용 절차가 선행되어야 하므로 자가 학습이나 준지도 전이학습 기법으로의 발전이 실제 상용화의 핵심 열쇠가 될 것이다. *(근거: Page 26, Chapter 4.2)*<br>**[TO-BE]** 한 참가자의 unstructured ADL protocol 데이터를 Table 4.1 기준으로 라벨링하는 데 약 3.5시간이 걸려, 향후 데이터 수집·라벨링 비용을 줄이는 방법이 중요하다.<br>*(사실검증 — 근거불충분/중대: 원문은 한 참가자의 unstructured ADL protocol 라벨링에 약 3.5시간이 걸린다고만 말한다. 요약의 '가정 내에서 매번', '환자 본인의 환측 동작', '자가 학습이나 준지도 전이학습', '상용화의 핵심 열쇠'는 제시된 근거 문장으로 직접 지지되지 않는다.)*
	- 근거 원문: “Labeling the data from one participant doing the unstructured ADL protocol with the labels defined in Table 4.1 takes about 3.5 hours and results in approximately fifty segmented actions.”

## 이 연구가 지적한 선행연구의 문제점

- 기존의 웨어러블 센서 재활 연구들은 대다수가 비사용(nonuse) 방지를 유도하기 위해 마비된 상지의 단순 일일 사용 시간 및 빈도 등의 양적인 측면에만 편중되었으며 동작의 기구학적 품질(질적 수준) 측정은 간과하였다. *(근거: Page 2, Abstract)*
	- 근거 원문: “While the prior art has focused primarily on quantifying the usage of the affected arm, a comprehensive assessment of rehabilitation progress requires evaluating movement quality during functional tasks.”
- 가장 정밀한 광학식 모션 캡처 시스템은 고비용이 요구되고 협소한 사용 공간 제약과 정밀 설정 교육 등의 한계로 인해 통제된 검사실 외에 일반 환자의 홈 케어 환경으로 전용하여 상시 측정하기 곤란했다. *(근거: Page 10, Chapter 2)*
	- 근거 원문: “These systems are reliable in very controlled environments, but they are high-cost and necessitate large space requirements and extended training and setup, so they cannot be used easily outside of the lab.”
- 웨어러블 HAR 기술 연구 중 다수는 미리 일정한 길이로 사전 재단된 윈도우 시계열 데이터 상의 단일 동작을 분류하는 데 머물러 있으며, 끊임없이 일어나는 실시간 시계열 내에서 특정 활동의 개시와 종료 시점을 밝히는 연속적 분할 능력은 미비했다. *(근거: Page 11, Chapter 3.1.2)*
	- 근거 원문: “While most existing work focuses on classifying a pre-segmented window as part of an activity, little work has investigated the precise time segmentation of continuous, non-pre-segmented data.”

## 이 연구의 해결 방식과 기여

- 양치질과 물 마시기(transport, recovery) 각기 다른 복잡성의 운동 과정 속에서 가속도 크기 감쇠, 각속도 역학 변조, 어깨 관절 가동 각도의 손상 등 질적 변형을 포착할 기구학적 특성을 수치로 실증함 *(근거: Page 2, Abstract)*
	- 근거 원문: “We identified robust kinematic metrics that can be computed directly from accelerometer and gyroscope data and that differentiate between healthy and stroke participants.”
- **[AS-IS]** 오픈소스 데이터셋이 결여하고 있던 비디오 및 모션 캡처 그라운드 트루스가 보장된 자가 ADL 수집용 프로토콜을 규정하고, 5명의 대조군 및 5명의 뇌졸중 환자를 통해 전구적인 신규 시계열 데이터셋 구축을 완료함 *(근거: Page 2, Abstract)*<br>**[TO-BE]** 오픈소스 데이터셋의 한계를 보완하기 위해 비디오 기반 라벨과 모션 캡처 데이터를 포함하는 자연istic ADL 수집 프로토콜을 제안하고, 5명의 건강한 참가자와 5명의 뇌졸중 참가자 데이터를 수집했으나, 논문 작성 시점에는 일부만 분석 준비가 완료되었다.<br>*(사실검증 — 과장/중대: 원문 Abstract는 주석이 달린 자연istic IMU 기반 ADL 데이터셋 수집 방법론을 제안했고, 5명 건강한 참가자와 5명 뇌졸중 참가자의 데이터를 수집했다고 한다. 그러나 원문 본문은 논문 작성 시점에 데이터셋 일부만 분석 준비가 되었고 GG98, GG102 및 일부 건강한 참가자 데이터로 proof-of-concept와 초기 테스트를 했다고 명시한다. '구축을 완료함'은 원문보다 강하다.)*
	- 근거 원문: “We also proposed a methodology for collecting an annotated naturalistic IMU-based ADL dataset in a controlled environment, using video ground truth to reliably label ADLs.”

## 레퍼런스할 수 있는 내용

### 1. 뇌졸중 발병 후 6개월 시점의 손 기능 장애 잔존율

- 원문 발췌: “After six months from a stroke, about 65% of patients are unable to use their affected hand for their usual activities \[3\].”
- 한국어 번역: 뇌졸중 발생 6개월 후, 환자의 약 65%는 그들의 일상적인 활동에 환측 손을 사용하지 못한다.
- 원문 위치: Page 8, Chapter 1 Introduction
- 원문 내 인용표기: \[3\]
- 해당 선행문헌: \[3\] Bruce H. Dobkin. Rehabilitation after Stroke. The New England journal of medicine, 352(16):1677–1684, April 2005.
- 주장 유형: background_citation
- 활용 맥락과 주의: 뇌졸중 환자가 장기적으로 겪는 영구적 상지 장애 및 손 기능 마비의 고착 비율(65%)에 대한 근거로 인용 시 유용.

### 2. 뇌졸중의 평생 누적 위험율 증가 추세

- 원문 발췌: “The lifetime risk of stroke has increased over the last 20 years by 50% and is now one in four people \[1\].”
- 한국어 번역: 뇌졸중의 평생 위험은 지난 20년 동안 50% 증가했으며 현재는 4명 중 1명에 달한다.
- 원문 위치: Page 8, Chapter 1 Introduction
- 원문 내 인용표기: \[1\]
- 해당 선행문헌: \[1\] Valery L Feigin, Michael Brainin, Bo Norrving, Sheila Martins, Ralph L Sacco, Werner Hacke, Marc Fisher, Jeyaraj Pandian, and Patrice Lindsay. World Stroke Organization (WSO): Global Stroke Fact Sheet 2022. International Journal of Stroke, 17(1):18–29, January 2022. Publisher: SAGE Publications.
- 주장 유형: background_citation
- 활용 맥락과 주의: 전 세계적으로 뇌졸중 발병률이 증가하고 있으며 평생 발병 위험이 25%에 육박한다는 최근 20년 역학적 보건 통계 자료로 활용 가능.

### 3. 본 연구에서 개발한 MHS 척도의 동일 환자 내 측정 일관성 및 변별 성능

- 원문 발췌: “It showed desirable properties such as high consistency when rating sample motions from the same subject and good separability (\>80% accuracy) between healthy and post-stroke individuals.”
- 한국어 번역: 그것(MHS)은 동일 피험자로부터 추출한 샘플 동작들을 평정할 때 높은 일관성을 보였으며, 건강한 피험자와 뇌졸중 환자군 간에 양호한 분별력(80% 이상의 정확도)을 제공하는 바람직한 성질을 드러냈다.
- 원문 위치: Page 2, Abstract
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: 웨어러블 IMU 데이터의 마할라노비스 거리 분석을 적용해 설계된 MHS 척도가 80% 이상의 정확도로 정상인과 환자를 대조 및 판별해냄을 입증하는 자체 실증 결과.

### 4. 제안된 2개 전완 IMU 탑재 하이브리드 ADL 분할 알고리즘 성능 결과

- 원문 발췌: “We achieved F1 scores of 0.83 and 0.87 in across-subjects and within-subjects evaluations, respectively.”
- 한국어 번역: 우리는 대상자 간 평가와 대상자 내 평가에서 각각 0.83 및 0.87의 F1 점수를 성취했다.
- 원문 위치: Page 2, Abstract
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: 두 개의 전완 가속도 및 자이로 정보를 인풋으로 설정해 맥락 정보와 CNN-BiLSTM 모델을 융합했을 때 획득되는 정량적 F1 분류 지표 근거.

## 자동 위치 매칭 실패 항목 (수동 확인 필요)

- **[AS-IS]** 뇌졸중 환자(GG98 및 GG102)를 대상으로 한 ADL 분할 테스트에서 환자 개인 데이터를 반영하지 않은 일반화 모델(F1 0.51\~0.65) 대비 개인 맞춤화 모델(individualized model)은 F1 점수 각각 0.85, 0.87로 월등한 수준을 달성함 *(근거: Page 29, Chapter 5.1)* **[TO-BE]** ADL segmentation 기준으로 일반화 모델의 F1은 GG98 0.51, GG102 0.65였고, 개인 맞춤화 모델의 F1은 Table 5.1 기준 GG98 0.85, GG102 0.87이었다. *(사실검증 — 누락/경미: 요약은 일반화 모델 F1 0.51~0.65라고만 제시하지만, 원문 Table 5.1에는 meal segmentation generalized F1이 GG98 0.22, GG102 0.54로도 제시된다. ADL segmentation만 말하는 것이라면 이를 명시해야 한다.)*


---

# [31] Clinically relevant predictive modeling for personalized ACL reconstruction classification

(저자: Xishi Zhu, Ryan Henry, Emily Jackson, Joe M. Hart, Jiaqi Gong | 연도: 2025 | 저널: Smart Health | DOI: https://doi.org/10.1016/j.smhl.2025.100575)

Zhu, X., Henry, R., Jackson, E., Hart, J. M., & Gong, J. (2025). Clinically relevant predictive modeling for personalized ACL reconstruction classification. Smart Health, 36, 100575. https://doi.org/10.1016/j.smhl.2025.100575

## 서지정보

- 저자: Xishi Zhu, Ryan Henry, Emily Jackson, Joe M. Hart, Jiaqi Gong
- 연도: 2025
- 저널: Smart Health
- DOI: 10.1016/j.smhl.2025.100575
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/04_machine_learning_and_deep_learning/Clinically relevant predictive modeling for personalized ACL reconstruction classification.pdf
- 분석 provider: antigravity

## 연구 목적

- 이 연구의 목적은 관성 측정 장치(IMU) 센서와 환자 특성을 결합한 다중 모달 보행 분석을 통합하여 ACL 재건술의 분류 및 회복 진행 상황을 시각화할 수 있는 설명 가능하고 개인화된 예측 모델을 개발하는 것이다. *(근거: PAGE 1, Abstract)*
	- 근거 원문: “We propose an explainable predictive model for ACL reconstruction classification through multi-modal analysis of gait dynamics and patient characteristics.”
- 연구진은 위상 경사 지수(PSI)로 정량화된 신체 부위 간 쌍별 운동이 ACL 재건 분류에 크게 기여하고, 걷기와 조깅 작업 간에 중요 센서 쌍의 조합이 다르며, 환자별 요인(회복 기간)이 모델의 분류 신뢰도와 상관관계가 있다는 세 가지 가설을 검증하고자 하였다. *(근거: PAGE 2, 1. Introduction)*
	- 근거 원문: “We hypothesized that: (1) paired body segment movements quantified by Phase Slope Index (PSI) matrices would significantly contribute to ACL reconstruction classification; (2) the importance of specific sensor pair combinations would differ between walking and jogging tasks; and (3) patient-specific factors would correlate with classification confidence, revealing insights into temporal dynamics of gait recovery.”

## 연구 설계와 대상

- 연구에는 74명의 ACL 재건 수술을 받은 환자(왼쪽 무릎 부상 31명, 오른쪽 무릎 부상 43명)와 5명의 건강한 대조군을 포함하여 총 79명의 참가자가 모집되었다. *(근거: PAGE 2, 3. Methodology)*
	- 근거 원문: “We recruited 79 participants, including 74 ACL patients (31 with left knee injuries, 43 with right knee injuries) and 5 healthy individuals.”

## 방법

- 참가자들의 양쪽 손목, 양쪽 발목 및 천골에 총 5개의 Shimmer IMU 센서를 부착하여 128Hz의 빈도로 가속도계와 자이로스코프 데이터를 수집하였으며, 트레드밀에서 3mph 속도로 5분간 걷고 이어서 6mph 속도로 3분간 조깅하는 작업을 수행하게 하였다. *(근거: PAGE 2, 3. Methodology)*
	- 근거 원문: “Data collection utilized five Shimmer IMU sensors placed on participants’ bodies (bilateral wrists, ankles, and sacrum), recording accelerometer and gyroscope data at 128 Hz. The protocol consisted of two sequential tasks: walking on a treadmill at 3 mph for 5 min, followed by jogging at 6 mph for 3 min.”
- 데이터를 64Hz로 다운샘플링하고 10초의 비중첩 창으로 분할한 후, 센서 판독치 간의 복잡한 상호작용을 포착하기 위해 각 주파수 대역에서 위상 경사 지수(PSI)를 계산하여 쌍별 인과 관계 특성 행렬(총 435개의 고유 특징)을 생성하였다. *(근거: PAGE 2, 3. Methodology)*
	- 근거 원문: “We then downsampled the data from 128 Hz to 64 Hz and segmented it into non-overlapping 10-second windows, providing sufficient temporal resolution to capture multiple gait cycles while maintaining computational efficiency. To capture complex interactions between sensor readings, we calculated the Phase Slope Index (PSI) to form a pairwise causality feature matrix.”
- 다섯 가지 기계학습 모델(SVM, Naive Bayes, Random Forest, KNN, Neural Network)을 사용해 왼쪽/오른쪽 부상 분류 및 건강군/부상군 분류라는 두 가지 이진 분류 작업을 수행했으며, 5-fold 교차 검증을 통해 모델을 평가하였다. *(근거: PAGE 2, 3. Methodology)*
	- 근거 원문: “To investigate our first hypothesis and evaluate PSI-based features, we implemented five machine learning models (SVM, Naive Bayes, Random Forest, KNN, and Neural Network) for two binary classification tasks: Left Injured vs. Right Injured and Injured vs. Healthy. We employed 5-fold cross-validation for walking and jogging phases independently, with randomly shuffled data windows.”

## 핵심 결과

- KNN 분류기가 모든 분류 과제에서 가장 높은 정확도를 나타냈으며, 16Hz 주파수 대역 데이터를 사용했을 때 좌/우 부상 분류의 경우 보행 시 약 93%, 조깅 시 약 98%의 정확도를 보였고, 정상/부상 분류에서는 보행 시 약 98%, 조깅 시 약 99%의 높은 정확도를 기록했다. *(근거: PAGE 3, 4.1. Predictive modeling evaluation)*
	- 근거 원문: “The KNN classifier consistently achieved the highest accuracy scores in all scenarios. For left–right classification, KNN reached a peak accuracy of approximately 93% during walking and 98% during jogging using the 16 Hz frequency range. For healthy-injured classification, KNN’s performance was even more impressive, achieving approximately 98% accuracy for walking and 99% for jogging, both with the 16 Hz filtered data.”
- 순열 중요도 점수 기반의 열지도 분석 결과, 보행 시 좌/우 부상 분류에서는 높은 중요도(0.07 이상)를 가지는 센서 쌍이 5개였던 반면 조깅 시에는 단 2개의 센서 쌍만 0.0175 이상의 중요도를 보여 보행 데이터가 조깅에 비해 부상 측면 분류에 더 다양한 변동성 패턴을 포함하고 있음을 보여준다. *(근거: PAGE 4, 4.2. Task-specific sensor pair importance analysis)*
	- 근거 원문: “For the left vs. right injury classification during walking (Fig. 2a), five sensor pairs showed importance scores above 0.07, indicated by dark red colors. In contrast, the jogging condition (Fig. 2b) displayed only two sensor pairs with importance scores above 0.0175. This difference suggests that walking data exhibits more variability in discriminative patterns between left and right injuries compared to jogging.”
- t-SNE 시각화에서 조깅 단계의 플롯이 보행 단계에 비해 클래스 간의 겹침이 적고 더 명확하게 군집화되었으며, 모델의 분류 신뢰도가 낮은 샘플들은 주로 안쪽 원 영역에 집중되는 경향을 보였다. *(근거: PAGE 4, 4.3. Dimension reduction examination)*
	- 근거 원문: “The t-SNE plots revealed that inner circles predominantly contain instances where the model exhibits low confidence in distinguishing between classes. Notably, jogging phase plots showed more distinctly separated clusters with reduced class overlap compared to walking, suggesting that jogging captures better global data structure.”
- **[AS-IS]** 회복 기간이 긴 환자들일수록 기계학습 모델이 높은 신뢰도로 분류하기 어려운 경향을 보였는데, 이는 회복 과정이 진행됨에 따라 움직임 패턴이 정상화되어 부상당하지 않은 대조군의 보행 패턴과 유사해지기 때문이다. *(근거: PAGE 5, 4.4. Patient-specific factor analysis)*<br>**[TO-BE]** 회복 기간이 긴 참가자들은 모델이 높은 신뢰도로 분류하기 어려운 보행 패턴을 보이는 경향이 있었으며, 저자들은 이 결과가 회복이 진행됨에 따라 움직임 패턴이 점차 정상 보행에 가까워진다는 임상적 이해를 뒷받침한다고 해석했다.<br>*(사실검증 — 인과관계오용/경미: 원문은 회복 기간이 긴 참가자의 보행 패턴이 높은 신뢰도로 분류되기 어려운 경향이 있고, 이것이 회복이 진행되며 보행 패턴이 정상화된다는 임상적 이해를 뒷받침한다고 설명한다. 요약은 이를 '때문이다'로 표현해 원문보다 인과를 더 단정했다.)*
	- 근거 원문: “These findings suggest that participants with longer recovery durations generally exhibit gait patterns that are more difficult for the model to classify with high confidence. This trend supports the clinical understanding that as recovery progresses, movement patterns gradually normalize, becoming more similar to uninjured gait.”

## 저자 결론

- 이 연구에서는 IMU 센서 데이터를 바탕으로 도출한 위상 경사 지수(PSI) 기능이 ACL 재건 상태를 95.37%의 우수한 정확도로 분류할 수 있음을 증명하였으며, 환자의 작업 간 센서 쌍 중요도 차이 및 회복 기간과 모델 신뢰도 간의 상관관계를 통해 보행 패턴이 시간이 지남에 따라 점진적으로 정상화됨을 확인하였다. *(근거: PAGE 6, 5. Discussion & conclusion)*
	- 근거 원문: “Our findings supported our hypotheses: PSI-based features from IMU data effectively classified ACL reconstruction outcomes with 95.37% accuracy using KNN; sensor pair importance differed between walking and jogging tasks, with jogging showing more focused importance patterns; and recovery duration correlated with model confidence, suggesting gait patterns normalize over time post-reconstruction.”
- **[AS-IS]** 본 설명 가능하고 개인화된 접근 방식은 기계학습 모델의 블랙박스 성격을 완화하여, 임상 의사결정을 돕는 정량적이고 객관적인 도구로서 재활 계획을 개선하고 복귀 기준을 설정하는 데 큰 도움을 줄 수 있다. *(근거: PAGE 6, 5. Discussion & conclusion)*<br>**[TO-BE]** 본 설명 가능한 접근은 ACL 재건과 관련된 주요 움직임 관계를 식별하고 환자 진행을 추적할 정량 도구를 제공함으로써 임상 의사결정에 도움을 줄 수 있으며, 향후 개인화 재활 프로토콜과 데이터 기반 운동 복귀 기준 연구의 기반을 마련한다.<br>*(사실검증 — 과장/경미: 원문은 주요 움직임 관계를 식별하고 환자 진행을 추적하는 정량 도구를 제공한다는 임상적 장점을 말하며, 향후 개인화 재활 프로토콜과 데이터 기반 복귀 기준을 가능하게 할 잠재력을 제시한다. 요약의 '복귀 기준을 설정하는 데 큰 도움'은 원문보다 실용적 효과를 더 강하게 단정한다.)*
	- 근거 원문: “The explainable nature of our approach offers significant clinical advantages by identifying key movement relationships affected by ACL reconstruction and providing quantitative tools to track patient progress.”

## 연구의 한계

- 연구의 주요 한계점으로는 비교적 작은 표본 크기, 전적으로 IMU 웨어러블 센서 데이터에만 의존한 분석 설계, 그리고 환자의 보행을 단일 시점에서만 분석한 점 등이 포함된다. *(근거: PAGE 6, 5. Discussion & conclusion)*
	- 근거 원문: “Despite limitations including small sample size, reliance solely on IMU data, and single time point analysis, our novel approach demonstrates the potential of combining multi-modal gait analysis with explainable machine learning for ACL reconstruction assessment.”

## 생각해볼 내용

- **[AS-IS]** 조깅 데이터 분석이 보행 분석보다 클래스 분리도와 분류 성능 면에서 더 우수한 결과를 보인 것은, 달리기와 같이 부하가 높고 고속인 운동에서 하지 및 상지 간 협응의 미세한 기능적 불균형이나 이상 징후가 더 쉽게 드러나고 모델에 유용한 식별 신호를 제공함을 보여준다. *(근거: PAGE 4, 4.3. Dimension reduction examination)*<br>**[TO-BE]** 조깅 데이터는 보행 데이터보다 t-SNE 플롯에서 클래스 겹침이 적고 군집 분리가 더 뚜렷했으며, 이는 조깅 조건에서의 더 높은 분류 성능과 일치한다.<br>*(사실검증 — 근거불충분/경미: 원문은 조깅 플롯에서 군집 분리가 더 뚜렷하고 모델 성능이 더 좋았다고만 설명한다. 고부하·고속 운동에서 하지 및 상지 협응의 미세한 불균형이 더 쉽게 드러난다는 해석은 제시된 원문 근거에서 직접 확인되지 않는다.)*
	- 근거 원문: “Notably, jogging phase plots showed more distinctly separated clusters with reduced class overlap compared to walking, suggesting that jogging captures better global data structure. This improved cluster separation aligns with the superior model performance in jogging conditions relative to walking.”
- 기계학습 모델의 신뢰도가 환자의 회복 기간이 지남에 따라 감소하는 특이한 상관관계는, 재활이 성공하여 보행 패턴이 완전히 정상화될수록 정상군과의 보행 특징 차이가 사라지기 때문에 생기는 현상이며, 이를 역으로 이용해 모델 신뢰도 지표를 보행 패턴 회복의 성숙도 및 정상화의 객관적인 간접 척도로 활용할 수 있음을 나타낸다. *(근거: PAGE 5, 4.4. Patient-specific factor analysis)*
	- 근거 원문: “These findings suggest that participants with longer recovery durations generally exhibit gait patterns that are more difficult for the model to classify with high confidence. This trend supports the clinical understanding that as recovery progresses, movement patterns gradually normalize, becoming more similar to uninjured gait.”

## 이 연구가 지적한 선행연구의 문제점

- 기존의 ACL 재건 결과 평가 및 운동 복귀 판정 방법은 임상 시험, 기능적 테스트, 주관적 평가에 주로 의존하며, 복잡한 신체 분절 간의 상호작용과 운동 패턴을 완벽히 포착하지 못하여 객관성과 정밀도가 떨어진다. *(근거: PAGE 1, 1. Introduction)*
	- 근거 원문: “While valuable, these methods often neglect complex movement patterns and lack precision and objectivity, particularly failing to account for the complex interplay between different body segments during movement.”
- 최근 웨어러블 IMU 센서와 머신러닝을 이용한 보행 분석 연구가 시도되고 있으나, 분석에 사용되는 고도화된 방식들이 임상적 해석이 어려운 '블랙박스' 형태로 이루어져 있어 임상의들이 실제로 신뢰하고 재활 처방에 바로 적용하기 어렵다. *(근거: PAGE 2, 2. Related work)*
	- 근거 원문: “Despite promising performance, these advanced analytical methods face limitations in clinical interpretability, with their ‘‘black box’’ nature hindering translation into actionable insights.”
- 머신러닝 해석력을 높이기 위한 기존 접근법들은 전역적 피처 중요도 분석에 집중되어 환자별 맞춤형 진단에 한계가 있거나, LIME 같은 국소적 해석 기법은 예측마다 결과가 변할 수 있어 설명의 안정성이 떨어진다는 단점이 있다. *(근거: PAGE 2, 2. Related work)*
	- 근거 원문: “Efforts to improve interpretability include feature importance scoring to identify which features are most indicative of ACL outcomes, though this approach focuses primarily on global importance, limiting personalized analysis capabilities. For local interpretation, LIME (Local Interpretable Model-agnostic Explanations) has been adopted (Kim et al., 2022), approximating complex models locally with more interpretable models. However, these explanations are valid only for individual predictions and may not generalize well, potentially leading to unstable explanations.”

## 이 연구의 해결 방식과 기여

- 본 연구는 관성 측정 센서(IMU) 데이터와 환자의 고유한 임상적 특성을 위상 경사 지수(PSI) 및 차원 축소 기법을 통해 통합 분석함으로써, 기계학습 모델의 높은 분류 성능과 임상적 해석력을 동시에 확보하는 방법론을 제안하였다. *(근거: PAGE 2, 1. Introduction)*
	- 근거 원문: “To address these limitations, we propose a novel approach integrating multi-modal gait analysis using IMU sensors with patient-specific characteristics.”
- 시간 경과에 따른 환자 보행 패턴의 정상화 과정을 t-SNE 차원 축소 및 분류 신뢰도로 투명하게 시각화하고 정량적으로 보여줌으로써, 스포츠 의학에서 보다 객관적인 환자별 맞춤 재활 계획과 안전한 운동 복귀 결정을 가능하게 하는 기틀을 마련했다. *(근거: PAGE 1, Abstract)*
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
- 활용 맥락과 주의: 웨어러블 센서를 활용한 보행 분석 모델링 연구에서 단일 센서 대비 다중 센서/다중 모달 융합 방식을 채택해야 하는 정량적 당위성(5\~15% 성능 향상)을 서술할 때 인용함.

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


---

# [32] Deep Learning in Gait Parameter Prediction for OA and TKA Patients Wearing IMU Sensors

(저자: Mohsen Sharifi Renani, Casey A. Myers, Rohola Zandie, Mohammad H. Mahoor, Bradley S. Davidson, Chadd W. Clary | 연도: 2020 | 저널: Sensors | DOI: https://doi.org/10.3390/s20195553)

Sharifi Renani, M., Myers, C. A., Zandie, R., Mahoor, M. H., Davidson, B. S., & Clary, C. W. (2020). Deep Learning in Gait Parameter Prediction for OA and TKA Patients Wearing IMU Sensors. Sensors, 20(19), 5553. https://doi.org/10.3390/s20195553

## 서지정보

- 저자: Mohsen Sharifi Renani, Casey A. Myers, Rohola Zandie, Mohammad H. Mahoor, Bradley S. Davidson, Chadd W. Clary
- 연도: 2020
- 저널: Sensors
- DOI: 10.3390/s20195553
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/04_machine_learning_and_deep_learning/Deep Learning in Gait Parameter Prediction for OA and TKA Patients Wearing IMU Sensors.pdf
- 분석 provider: antigravity

## 연구 목적

- 본 연구의 목적은 골관절염(OA) 및 관절 치환술 환자군에서 IMU 데이터로부터 보행 시공간적 변수(STGPs)를 예측하기 위한 여러 최신 심층 신경망 구조의 성능을 평가하고, 예측 정확도를 극대화하기 위한 최적의 센서 조합을 결정하는 것이다. *(근거: PAGE 2, 1. Introduction)*
	- 근거 원문: “Thus, the purpose of this study was two-fold: (1) to access the ability of multiple contemporary deep neural network architectures to predict STGPs from IMU data in the OA and joint-replacement patient populations and (2) to determine the optimal sensor combination to maximize prediction accuracy.”

## 연구 설계와 대상

- 총 29명의 대상자(골관절염 환자 14명, 인공관절 전치환술 환자 15명)가 연구에 참여하였다. *(근거: PAGE 2, 2.1. Gait Measurements of Osteoarthritic and Total Knee-Replacement Subjects)*
	- 근거 원문: “Twenty-nine subjects, including 14 subjects with OA (Age = 67 ± 7, weight = 79 ± 12 kg, height = 168 ± 16 cm, 4 females and 10 males), 15 subjects with total knee arthroplasty (TKA) (Age = 68 ± 4, weight = 76 ± 14 kg, height = 164 ± 9 cm, 11 females and 4 males, 7 uni-lateral and 8 bi-lateral), participated in the study as part of a larger investigation.”
- 대상자들은 일상 보행 속도 범위를 포괄하기 위해 자가 선택 속도, 느린 속도, 빠른 속도의 세 가지 속도로 5m 보행 과제를 15회 수행하였다. *(근거: PAGE 2, 2.1. Gait Measurements of Osteoarthritic and Total Knee-Replacement Subjects)*
	- 근거 원문: “Subjects performed 15 trials of a 5-m walking task at three different speeds: self-selected, slow, and fast to cover the entire range of possible daily walking paces.”

## 방법

- 대상자들의 해부학적 랜드마크에 71개의 반사 마커를 부착하고 여러 사지 세그먼트와 몸통에 17개의 IMU를 장착했으며, 본 연구에서는 발, 종아리, 허벅지, 골반에 위치한 7개의 IMU만 데이터 분석에 사용했다. *(근거: PAGE 2, 2.1. Gait Measurements of Osteoarthritic and Total Knee-Replacement Subjects)*
	- 근거 원문: “Subjects were fitted with 71 reflective markers on anatomical landmarks and 17 IMUs on various limb segments and the trunk. For this study, only the 7 IMUs located on the feet, shanks, thighs \[35,36\], and pelvis \[37\] were used in the subsequent data analysis (Figure 1a,b).”
- 힘 데이터, 모션 캡처(MOCAP), IMU(자유 가속도 및 각속도)의 샘플링 주파수는 각각 1000 Hz, 100 Hz, 40 Hz였다. *(근거: PAGE 3, 2.1. Gait Measurements of Osteoarthritic and Total Knee-Replacement Subjects)*
	- 근거 원문: “The sampling frequency of force data, MOCAP, and IMUs (free acceleration and angular velocity) were 1000 Hz, 100 Hz, and 40 Hz, respectively.”
- IMU 데이터는 100 Hz로 업샘플링되었으며 피크 검출 방법을 사용하여 시상면에서 발 센서의 각속도를 기준으로 양쪽 다리의 전체 보행 주기로 분할되었다. *(근거: PAGE 3, 2.2. Gait Data Processing)*
	- 근거 원문: “IMU data for each trial was up-sampled to 100 Hz and segmented into full strides for each leg based on the angular velocities of the feet sensors in the sagittal plane using the peak detection method (Figure 1c,d) \[41,42\].”
- 사전 신경망 구조 벤치마킹 결과를 바탕으로, 센서 조합에 대한 대규모 실험계획법 연구를 위해 Zrenner 등이 제안한 1D 합성곱 신경망(CNN) 구조가 선택되었다. *(근거: PAGE 4, 2.4. Assessing Optimal Sensor Combinations for Each Gait Characteristic)*
	- 근거 원문: “Based on the result of the preliminary neural network architecture selection, the 1D convolution neural network (CNN) architecture proposed by Zrenner et al. was chosen for a larger design-of-experiment study on sensor combinations \[19\].”
- 발, 골반, 종아리, 허벅지 센서의 15가지 고유 조합을 기반으로 예측 정확도를 분석하기 위해 풀 팩토리얼 실험계획법이 구현되었다. *(근거: PAGE 5, 2.4. Assessing Optimal Sensor Combinations for Each Gait Characteristic)*
	- 근거 원문: “A full factorial design of experiments was implemented to analyze the prediction accuracy based on 15 unique combinations of the feet, pelvis, shank, and thigh sensors (Table 2).”

## 핵심 결과

- 12개 시공간 보행 변수(STGPs)의 백분율 오차는 2.1%(보행 주기 시간)에서 73.7%(발 외향각)까지 분포했으며 전반적으로 공간적 변수보다 시간적 변수에서 더 정확했다. *(근거: PAGE 1, Abstract)*
	- 근거 원문: “Percent error across the 12 STGPs ranged from 2.1% (stride time) to 73.7% (toe-out angle) and overall was more accurate in temporal parameters than spatial parameters.”
- 전체적으로 발-허벅지(F T) 조합이 가장 우수한 평균 순위(5.1)를 나타냈고, 발-종아리(F S, 6.2), 종아리(S, 6.3) 센서 조합이 그 뒤를 이었다. *(근거: PAGE 11, 3.3. Optimal Sensor Combinations for Gait Characteristics)*
	- 근거 원문: “Overall, the feet-thigh (F T) configuration had the best average rank (5.1), followed by the feet-shank (F S, 6.2), and shank (S, 6.3) sensor combinations.”
- 골관절염(OA) 환자군은 인공관절 전치환술(TKA) 환자군에 비해 모든 센서 조합 및 STGPs에 걸쳐 더 큰 평균(19.0%) 및 중앙값(6.6%) NAPE를 보였다(TKA 평균 NAPE = 14.7%, 중앙값 NAPE = 4.6%). *(근거: PAGE 9, 3.3. Optimal Sensor Combinations for Gait Characteristics)*
	- 근거 원문: “The OA cohort had larger mean (19.0%) and median (6.6%) NAPE across all sensor combinations and STGPs compared to TKA (mean NAPE = 14.7%, median NAPE = 4.6%).”

## 저자 결론

- 본 연구는 딥러닝 기반 데이터 구동 방식이 IMU 센서 신호를 바탕으로 OA 및 TKA 환자의 시공간적 보행 특성을 예측할 수 있음을 입증하였다. *(근거: PAGE 15, 5. Conclusions)*
	- 근거 원문: “This study demonstrated that a deep-learning, data-driven approach was able to predict spatial temporal gait characteristics of OA and TKA patients based on signals from IMU sensors.”
- 다양한 센서 조합과 STGPs, 환자군, 보행 속도에 대한 민감도 분석을 통해, 딥러닝이 환자 모니터링 시스템 설계를 방해하고 순응도에 부정적 영향을 미치는 센서 위치 의존성 문제를 극복할 수 있음을 보여주었다. *(근거: PAGE 15, 5. Conclusions)*
	- 근거 원문: “Using a comprehensive analysis of various sensor combinations and their sensitivity to STGPs, patient population, and walking pace, our results showed that deep learning can overcome the dependency on sensor location that hinders the design of patient monitoring systems and negatively impacts patient compliance.”

## 연구의 한계

- 본 연구는 포함된 대상자의 수가 적다는 한계가 있다. *(근거: PAGE 15, 4. Discussion)*
	- 근거 원문: “This study was also limited in the number of subjects that were included.”
- 다른 데이터 구동 방식과 마찬가지로, 본 연구에서 훈련된 신경망은 오직 선택된 모집단에만 적합하다. *(근거: PAGE 15, 4. Discussion)*
	- 근거 원문: “Like other data-driven approaches, the trained network described in this study are only suitable for the selected population.”
- 실험실 외부의 대규모 환자군에게 알고리즘을 적용하는 데 있어 센서 부착 위치의 다양성, 저가형 IMU의 신호 품질 저하, 체질량 지수가 높은 환자의 연부조직 아티팩트, 훈련 데이터셋 범위를 벗어난 보행 변수를 가진 환자 식별 등의 실질적인 한계가 존재한다. *(근거: PAGE 15, 4. Discussion)*
	- 근거 원문: “There are also practical limitations to deploying our algorithm to a large patient population outside of a laboratory environment, including variability in sensor placement, reduced signal quality from low-cost IMUs, soft-tissue artifacts for high body mass index patients, and identification of patients with gait parameters outside the training data set.”

## 생각해볼 내용

- 훈련 데이터 분포를 벗어난 단 하나의 이상치(S21)가 테스트 세트 전체의 오차를 크게 상승시킨 결과는, 기계학습 모델을 실제 임상 환경에 적용할 때 분포 외 데이터(Out-of-distribution)에 대한 취약성이 주요 해결 과제임을 시사한다. *(근거: PAGE 14, 4. Discussion)*
	- 근거 원문: “The impact of subject S21 in the test set is an example of how CNNs result in poor performance when faced with data that are outside the distribution of the training data, which is one of the main challenges in the use of machine-learning models for real world applications.”
- 발-허벅지(F-T) 센서 조합이 통계적으로 유의미하게 우수하지만, 다른 센서 조합과의 예측 정확도 차이는 2\~5% 수준에 불과하므로, 임상 응용 분야에 따라 비용과 환자 편의성을 고려해 센서 조합을 유연하게 설계할 수 있다. *(근거: PAGE 14, 4. Discussion)*
	- 근거 원문: “As noted earlier, while the F-T sensor combination proved to be statistically better than other combinations, a 2–5% improvement in overall STGP prediction accuracy may be impactful during certain clinical applications.”

## 이 연구가 지적한 선행연구의 문제점

- 기존의 보행 특성 측정 방법(MOCAP 및 힘판)은 실험실 환경과 고가의 장비를 요구하며 시간 소모가 크다. *(근거: PAGE 1, 1. Introduction)*
	- 근거 원문: “Conventional methods for measuring gait characteristics that include motion capture (MOCAP) systems and force plates require a laboratory environment and expensive, time-consuming, equipment \[8\].”
- 센서 융합이나 칼만 필터를 사용하는 기존의 이중 적분 기반 방법들은 보행 분할을 위해 stance 구간의 제로 속도 조건(zero-velocity condition)에 의존하지만, 병리적 보행을 가진 환자나 자유 달리기 같은 역동적 활동 중에는 이 조건을 확인하기 어렵다. *(근거: PAGE 2, 1. Introduction)*
	- 근거 원문: “However, clear zero-velocity conditions are difficult to identify for patients with pathological gait or during highly dynamic activities like free running \[19\].”
- 다양한 환자군에 대해 최상의 성능을 내는 최적의 센서 조합을 정량화한 체계적인 연구가 현재 부족한 실정이다. *(근거: PAGE 2, 1. Introduction)*
	- 근거 원문: “Additionally, systematic studies quantifying optimal sensor combinations for the best performance across various patient populations are important to this field, but are lacking.”

## 이 연구의 해결 방식과 기여

- 골관절염(OA) 및 인공관절 전치환술(TKA) 대상자에 대해 연구된 적이 없었던, IMU 데이터 기반의 12가지 STGP 예측 성능 벤치마킹 및 최적의 센서 조합을 식별하는 연구를 수행하였다. *(근거: PAGE 1, Abstract)*
	- 근거 원문: “A study was conducted to benchmark the ability of multiple deep neural network (DNN) architectures to predict 12 STGPs from inertial measurement unit (IMU) data and to identify an optimal sensor combination, which has yet to be studied for OA and TKA subjects.”
- 본 연구 결과는 골관절염 환자 및 관절 치환술을 받을 환자들에게 STGPs의 정확한 실시간 모니터링을 제공하여 치료, 수술 계획 및 재활에 실질적인 도움을 줄 수 있다. *(근거: PAGE 2, 1. Introduction)*
	- 근거 원문: “The results of this study will help patients suffering from OA who may go on to receive a total joint replacement benefit from the accurate real-time patient monitoring of STGPs to inform their treatment, surgical planning, and rehabilitation.”

## 레퍼런스할 수 있는 내용

### 1. 골관절염 환자의 보행 적응 특성

- 원문 발췌: “Patients with progressive OA typically exhibit gait adaptations including decreased joint flexibility, increased stance time on the affected side, cadence, and double support time, and an overall increase in variability of spatial temporal parameters \[31–34\].”
- **[AS-IS]** 한국어 번역: 진행성 골관절염(OA) 환자들은 일반적으로 관절 유연성 감소, 환측 디딤 시간 증가, 보행 속도 및 양하지 지기 시간의 증가, 그리고 시공간 보행 변수의 전반적인 변동성 증가를 포함하는 보행 적응 특성을 보인다.<br>**[TO-BE]** 진행성 골관절염(OA) 환자들은 일반적으로 관절 유연성 감소, 환측 디딤 시간 증가, 보행 빈도(cadence) 및 양하지 지지 시간 증가, 그리고 시공간 보행 변수의 전반적인 변동성 증가를 포함하는 보행 적응 특성을 보인다.<br>*(사실검증 — 번역오류/경미: 원문의 cadence는 보행 속도(speed)가 아니라 보행 빈도 또는 cadence를 의미한다. 요약 번역은 cadence를 보행 속도로 옮겨 변수 의미가 바뀌었다.)*
- 원문 위치: PAGE 2, 1. Introduction
- 원문 내 인용표기: \[31–34\]
- 해당 선행문헌: 31. Bejek, Z.; Paróczai, R.; Illyés, Á.; Kiss, R.M. The influence of walking speed on gait parameters in healthy people and in patients with osteoarthritis. Knee Surg. Sports Traumatol. Arthrosc. 2006. \[CrossRef\] 32. Kiss, R.M.; Bejek, Z.; Szendrői, M. Variability of gait parameters in patients with total knee arthroplasty. Knee Surg. Sports Traumatol. Arthrosc. 2012. \[CrossRef\] 33. Kiss, R.M. Effect of severity of knee osteoarthritis on the variability of gait parameters. J. Electromyogr. Kinesiol. 2011. \[CrossRef\] 34. Hollman, J.H.; McDade, E.M.; Petersen, R.C. Normative spatiotemporal gait parameters in older adults. Gait Posture 2011. \[CrossRef\]
- 주장 유형: background_citation
- 활용 맥락과 주의: 골관절염 환자군의 병리적 보행 변이성과 임상적 보행 적응 양상에 대한 근거로 인용하기 적합함.

### 2. 단일 센서 부착 방식의 보행 분석 한계

- 원문 발췌: “Single body segment mounted IMUs (e.g., wrist or pelvis) are limited in calculation of certain STGPs such as number of steps, step cadence, or step distance which may not be adequate for clinical applications \[26,27\].”
- 한국어 번역: 단일 신체 부위에 부착된 IMU(예: 손목 또는 골반)는 걸음 수, 보행 빈도 또는 걸음 거리와 같은 특정 STGP의 계산에 한계가 있어 임상 적용에 불충분할 수 있다.
- 원문 위치: PAGE 2, 1. Introduction
- 원문 내 인용표기: \[26,27\]
- 해당 선행문헌: 26. Fasel, B.; Duc, C.; Dadashi, F.; Bardyn, F.; Savary, M.; Farine, P.A.; Aminian, K. A wrist sensor and algorithm to determine instantaneous walking cadence and speed in daily life walking. Med. Biol. Eng. Comput. 2017, 55, 1773–1785. \[CrossRef\] 27. Soltani, A.; Dejnabadi, H.; Savary, M.; Aminian, K. Real-world gait speed estimation using wrist sensor: A personalized approach. IEEE J. Biomed. Health Inf. 2019. \[CrossRef\]
- 주장 유형: background_citation
- 활용 맥락과 주의: 단일 부위 센서 착용 시 특정 임상적 보행 지표 획득의 제한점과 다중 센서 사용의 당위성을 설명할 때 인용할 수 있음.

### 3. 고령층 대상 딥러닝 보폭 예측의 우수성

- 원문 발췌: “Using a deep convolutional neural network trained on over 1220 strides from 101 geriatric patients, the algorithm predicted stride length with a mean error of −0.15 cm, which was considerably more accurate than previous integration-based methods \[12\].”
- 한국어 번역: 101명의 고령 환자로부터 얻은 1220개 이상의 보행 주기를 학습한 심층 합성곱 신경망을 사용하여 보폭을 예측한 결과, 평균 오차 -0.15 cm로 기존 적분 기반 방법들보다 훨씬 더 정확한 예측 결과를 보였다.
- 원문 위치: PAGE 2, 1. Introduction
- **[AS-IS]** 원문 내 인용표기: \[12\]
- 해당 선행문헌: 12. Rampp, A.; Barth, J.; Schulein, S.; Gassmann, K.G.; Klucken, J.; Eskofier, B.M. Inertial Sensor-Based Stride Parameter Calculation From Gait Sequences in Geriatric Patients. IEEE Trans. Biomed. Eng. 2015, 62, 1089–1097. \[CrossRef\] \[PubMed\] \> **[TO-BE]** 원문 내 인용표기: \[24\]가 딥러닝 보폭 예측 연구에 해당하며, \[12\]는 비교 대상인 기존 적분 기반 방법으로 구분한다. 해당 선행문헌은 24. Hannink, J.; Kautz, T.; Pasluosta, C.F.; Gassmann, K.G.; Klucken, J.; Eskofier, B.M. Sensor-Based Gait Parameter Extraction With Deep Convolutional Neural Networks. IEEE J. Biomed. Health Inf. 2017, 21, 85–93.<br>*(사실검증 — 인용표기오류/중대: 요약은 고령층 대상 딥러닝 보폭 예측 연구 자체의 인용을 [12]로 매핑했지만, 원문에서 해당 딥러닝 연구는 Hannink et al.의 연구로 서술되며 참고문헌 목록에서는 [24]에 해당한다. 문장 끝의 [12]는 ‘previous integration-based methods’를 가리키는 비교 대상 문헌이다.)*
- 주장 유형: background_citation
- 활용 맥락과 주의: 고령 환자 대상의 딥러닝 보행 매개변수 예측에서 딥러닝 모델이 기존 센서 융합 및 적분 방식보다 정확도가 높음을 입증하기 위한 배경 문헌으로 인용할 수 있음.

### 4. 12가지 보행 시공간 변수의 시간적 vs 공간적 예측 정확도 차이

- 원문 발췌: “Percent error across the 12 STGPs ranged from 2.1% (stride time) to 73.7% (toe-out angle) and overall was more accurate in temporal parameters than spatial parameters.”
- 한국어 번역: 12개 STGP의 백분율 오차는 최소 2.1%(보행 주기 시간)에서 최대 73.7%(발 외향각) 범위였으며, 전반적으로 공간적 변수보다 시간적 변수에서 예측 정확도가 더 높았다.
- 원문 위치: PAGE 1, Abstract
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: IMU 센서 신호 기반 딥러닝 모델로 보행 변수를 예측할 때 시간적 변수(Time-related parameters)가 공간적 변수(Spatial-related parameters)보다 오차가 낮음을 보여주는 본 연구의 직접적 결과로 인용할 수 있음.

### 5. 골관절염 환자군과 인공관절 전치환술 환자군 간의 모델 예측 오차 차이

- 원문 발췌: “The OA cohort had larger mean (19.0%) and median (6.6%) NAPE across all sensor combinations and STGPs compared to TKA (mean NAPE = 14.7%, median NAPE = 4.6%).”
- 한국어 번역: 골관절염(OA) 환자군은 인공관절 전치환술(TKA) 환자군에 비해 모든 센서 구성 및 보행 변수 전반에서 더 높은 평균(19.0%) 및 중앙값(6.6%) NAPE 오차를 보였다(TKA 군은 평균 NAPE 14.7%, 중앙값 NAPE 4.6%).
- 원문 위치: PAGE 9, 3.3. Optimal Sensor Combinations for Gait Characteristics
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: 병리적 상태의 심각성과 보행 변이성의 정도가 딥러닝 모델의 보행 매개변수 예측 능력에 미치는 영향을 비교 인용하는 데 유용함.


---

# [33] Identifying Gait-Related Functional Outcomes in Post-Knee Surgery Patients Using Machine Learning: A Systematic Review

(저자: Christos Kokkotis, Georgios Chalatsis, Serafeim Moustakidis, Athanasios Siouras, Vasileios Mitrousias, Dimitrios Tsaopoulos, Dimitrios Patikas, Nikolaos Aggelousis, Michael Hantes, Giannis Giakas, Dimitrios Katsavelis, Themistoklis Tsatalas | 연도: 2023 | 저널: International Journal of Environmental Research and Public Health | DOI: https://doi.org/10.3390/ijerph20010448)

Kokkotis, C., Chalatsis, G., Moustakidis, S., Siouras, A., Mitrousias, V., Tsaopoulos, D., Patikas, D., Aggelousis, N., Hantes, M., Giakas, G., Katsavelis, D., & Tsatalas, T. (2023). Identifying Gait-Related Functional Outcomes in Post-Knee Surgery Patients Using Machine Learning: A Systematic Review. International Journal of Environmental Research and Public Health, 20(1), 448. https://doi.org/10.3390/ijerph20010448

## 서지정보

- 저자: Christos Kokkotis, Georgios Chalatsis, Serafeim Moustakidis, Athanasios Siouras, Vasileios Mitrousias, Dimitrios Tsaopoulos, Dimitrios Patikas, Nikolaos Aggelousis, Michael Hantes, Giannis Giakas, Dimitrios Katsavelis, Themistoklis Tsatalas
- 연도: 2023
- 저널: International Journal of Environmental Research and Public Health
- DOI: https://doi.org/10.3390/ijerph20010448
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/04_machine_learning_and_deep_learning/Identifying Gait-Related Functional Outcomes in Post-Knee Surgery Patients Using Machine Learning - A Systematic Review.pdf
- 분석 provider: antigravity

## 연구 목적

- 무릎 수술 후 환자의 보행 관련 변화를 감지하고 기계 학습 알고리즘을 사용해 기능적 회복 상태를 결정하는 연구 결과를 종합하고 요약하고자 한다. *(근거: PAGE 1, Abstract)*
	- 근거 원문: “The scope of this study is to summarize the results of a systematic literature review on the identification of gait-related changes and the determination of the functional recovery status of patients after knee surgery using advanced machine learning algorithms.”
- 정형외과 수술 후 보행 분석의 생체역학 데이터를 이용하고 기계 학습 또는 딥러닝 기법을 활용해 무릎 관절의 재활 단계를 평가한 기존 연구들을 확인하여 포괄적인 개요를 제공한다. *(근거: PAGE 2, 1. Introduction)*
	- 근거 원문: “The aim of this review is to identify studies that have utilized machine learning (ML) or deep learning techniques to evaluate the rehabilitation stage of the knee joint following major orthopedic surgery using biomechanical data from gait analysis.”

## 연구 설계와 대상

- 이 연구는 PRISMA 가이드라인에 근거하여 Scopus, PubMed, Semantic Scholar를 포함한 여러 데이터베이스를 검색해 진행된 체계적 문헌 고찰이다. *(근거: PAGE 1, Abstract)*
	- 근거 원문: “The current systematic review was conducted using multiple databases in accordance with the PRISMA guidelines, including Scopus, PubMed, and Semantic Scholar.”
- 데이터베이스 검색을 통해 찾은 총 405개의 논문 중 선정 기준을 충족하고 보행 데이터를 바탕으로 기계 학습을 사용해 수술 후 회복 상태를 정량화한 6개의 논문이 최종 분석에 사용되었다. *(근거: PAGE 1, Abstract)*
	- 근거 원문: “Six out of the 405 articles met our inclusion criteria and were directly related to the quantification of the recovery status using machine learning and gait data.”

## 방법

- **[AS-IS]** MEDLINE(PubMed), Scopus, Semantic Scholar 데이터베이스를 체계적으로 검색했고 수동 참고문헌 검토 검색을 병행하였으며, 영어가 아닌 초록 및 학회 초록 등은 제외되었다. *(근거: PAGE 4, 2.2. Literature Search)*<br>**[TO-BE]** MEDLINE(PubMed), Scopus, Semantic Scholar 데이터베이스를 체계적으로 검색했고 Google Scholar를 통한 수동 검색을 병행했으며, 영어가 아닌 초록, 학회 초록, OpenGrey 데이터베이스는 회색문헌으로 평가되었다.<br>*(사실검증 — 사실불일치/경미: 원문은 영어가 아닌 초록, 학회 초록, OpenGrey 데이터베이스를 회색문헌으로 평가했다고 서술한다. 해당 항목들을 곧바로 제외했다고 요약한 것은 원문 표현과 다르다.)*
	- 근거 원문: “The following databases were searched systematically: (a) MEDLINE (through PubMed), (b) Scopus, and (c) Semantic Scholar. In addition, a manual search was also conducted on Google Scholar to identify articles cited by the collected papers quoting the retrieved papers.”
- 포함된 비무작위 연구들의 품질 평가는 6개 항목(수행 성능 지표, 데이터셋 분포, 정답 라벨 결정, 입력으로 사용된 특징 세트, 정보 공개, 연구 목표)의 체크리스트를 포함하는 수정된 MINORS 지표를 사용하여 수행되었다. *(근거: PAGE 5, 2.6. Quality Assessment)*
	- 근거 원문: “The quality of non-randomized studies was evaluated using a modified methodologic index (MINORS) \[22,23\]. The following information was considered on a six-item checklist: performance metrics, dataset distribution, ground truth label determination, the feature set that was used as inputs, disclosure, and the aim of the study.”

## 핵심 결과

- 문헌 검토 대상인 6개의 논문 중 5개는 무릎 인공관절 치환술(TKA) 환자를, 1개는 전방십자인대(ACL) 수술 환자를 대상으로 분류 과제를 수행한 연구였다. *(근거: PAGE 6, 3. Results)*
	- 근거 원문: “The included studies in this systematic review were classified into the following application domains: (i) TKA surgery (5 studies) and (ii) ACL surgery (1 study).”
- Emmerzaal 등의 TKA 환자 분류 연구는 수술 후 6주의 보행 데이터를 이용해 학습된 로지스틱 회귀 모델이 수술 후 3, 6, 12개월의 회복 상태 변화를 67.3% 정확도로 감지할 수 있음을 보고했다. *(근거: PAGE 7, 3.1. TKA Surgery)*
	- 근거 원문: “A comparison investigation led them to the conclusion that an LR classifier trained on six weeks of post-operative biomechanical data during walking was responsive to changes at 3, 6, and 12 months post-TKA (with a 67.3% accuracy).”
- Martins 등의 TKA 연구에서는 차원 축소법인 KPCA와 다중 클래스 SVM 분류 모델의 결합을 통해 보행 보조 기기 간 차이를 식별하는 데 98%의 분류 정확도를 나타냈다. *(근거: PAGE 7, 3.1. TKA Surgery)*
	- 근거 원문: “They used a multiclass SVM for classification, and this combination of KPCA and MSVM achieved an accuracy of 98%.”
- ACL 수술 영역의 Kokkotis 등의 연구에서는 21개의 생체역학적 매개변수를 피처로 사용하여 SVM 분류기가 최고 94.95%의 분류 정확도를 달성했다. *(근거: PAGE 8, 3.2. ACL Surgery)*
	- 근거 원문: “The best score was achieved (94.95% accuracy) by the SVM classifier, which employed 21 biomechanical parameters.”

## 저자 결론

- 기계 학습 기반의 견고하고 설명 가능한 설명력 모델을 통해 재활 과정에서의 보행 회복 상태 및 핵심 매개변수 기여도를 평가함으로써 임상의에게 비침습적이고 강력한 진단 및 예후 판정 도구를 제공할 수 있다. *(근거: PAGE 10, 4. Discussion and Conclusions)*
	- 근거 원문: “AI is a valuable tool for identifying gait-related changes in post-knee surgery patients. The creation of robust explainable ML models for quantifying the recovery status during the rehabilitation process and the understanding of the contribution of the selected gait biomechanical parameters in the model’s output could lead to the creation of non-invasive and more powerful diagnostic and prognostic tools for clinicians.”
- **[AS-IS]** 정형외과 영역에서의 AI 적용은 개인별 맞춤 재활 중재를 개발하여 비정상 보행 패턴을 조기에 교정하고 무릎 골관절염(OA)의 발생 위험을 차단하는 데 중요한 기여를 할 수 있다. *(근거: PAGE 10, 4. Discussion and Conclusions)*<br>**[TO-BE]** 정형외과 영역에서 AI는 개인 맞춤형 재활 중재를 형성하고 비정상 보행 패턴을 수정하며, 이후 무릎 골관절염 발생을 피하는 데 기여할 가능성이 있다.<br>*(사실검증 — 과장/경미: 원문은 AI가 개인 맞춤 재활 중재 형성에 중요한 역할을 할 수 있으며 비정상 보행 패턴 수정과 이후 무릎 OA 발생을 피하는 데 이어질 수 있다고 조심스럽게 표현한다. 요약의 '조기에 교정'과 '발생 위험을 차단'은 원문보다 강한 표현이다.)*
	- 근거 원문: “Hence, AI in the field of orthopedics may play a key role in forming new personalized rehabilitation interventions for the modification of abnormal gait patterns and subsequently avoid the development of knee OA.”

## 연구의 한계

- 이 연구는 PRISMA 권고사항을 부합하여 수행된 체계적 문헌 고찰이지만, 분석된 문헌들의 이질성으로 인하여 공식적인 정량적 메타 분석은 포함하지 못했다. *(근거: PAGE 10, 4. Discussion and Conclusions)*
	- 근거 원문: “This paper is a systematic review that adheres to the PRISMA recommendations but excludes a more formal quantitative meta-analysis.”
- 본 문헌 검색 과정에서 3개의 주요 온라인 데이터베이스만 활용하고 회색 문헌을 배제한 점이 검토된 최종 포함 연구 수가 적게 식별되는 원인이 되었을 수 있다. *(근거: PAGE 10, 4. Discussion and Conclusions)*
	- 근거 원문: “This results from the observed heterogeneity of the identified studies as limitation can be considered the fact that only three online databases (PubMed, Scopus, and Semantic Scholar) were employed, and the exclusion of the grey literature may have led to the identification of a relatively small number of included studies.”

## 생각해볼 내용

- 해당 무릎 수술 재활 예측 분야는 아직 연구가 완전히 개발되지 않은 초기 단계로 보이며, 2019년 이전의 문헌이 현격히 적은 이유는 컴퓨팅 파워와 보행 관련 빅데이터 구축 수준의 한계 때문일 수 있다. *(근거: PAGE 8, 4. Discussion and Conclusions)*
	- 근거 원문: “From the literature review, it emerged that this area of research is untapped. There is a gap in the existence of literature before 2019, which is possibly resulting from the limited computing power and the non-existence of big data in this field.”
- 체계적 문헌 고찰에 포함된 6개 연구 모두 실제 환자 임상 현장에 적용할 만큼 외부 독립 데이터셋을 사용한 신뢰성 검증 과정이 없었다는 점이 핵심적인 기술적 보완 과제로 판단된다. *(근거: PAGE 10, 4. Discussion and Conclusions)*
	- 근거 원문: “It is noteworthy that none of the employed studies were validated against an external dataset.”

## 이 연구가 지적한 선행연구의 문제점

- TKA 수술과 재활 이후 환자별 만족 수준과 회복 속도는 동일하지 않으며, 약 11\~20%의 환자가 잔존하는 기능적 한계로 인한 수술 후 불편감을 계속 호소한다. *(근거: PAGE 2, 1. Introduction)*
	- 근거 원문: “Previous studies have shown that not all patients have the same rate of satisfaction after surgery and rehabilitation \[6–8\]. Approximately 11–20% of patients experience discomfort following TKA, which is related to persisting functional impairments \[6\].”
- 기존에 많이 쓰이던 설문지 형태의 환자 보고 결과 측정(PROM) 방법은 평가 주체나 환자의 주관적 응답에 따른 회상 편향의 한계가 있다. *(근거: PAGE 2, 1. Introduction)*
	- 근거 원문: “However, these measures rely on subjective statements presented by the patients or the primary caregivers and can be biased due to the recall of events or the raters’ subjective comments on patient performance.”
- 전통적인 연구실 기반의 운동 평가 및 분석은 비용이 비싸며 시간이 오래 소요되고 장비 운영과 평가에 특화된 전문 지식이 필요하다. *(근거: PAGE 2, 1. Introduction)*
	- 근거 원문: “Conventional lab-based methods are time-consuming and require expensive equipment and specialized personnel.”

## 이 연구의 해결 방식과 기여

- **[AS-IS]** 보행 분석에서 추출한 운동 형상학적 변수들과 기계 학습을 융합함으로써, 임상에서 저렴하고 정량화되고 비침습적으로 무릎 환자의 회복 및 재활 단계를 빠르게 평가할 수 있는 대안적 가이드라인을 제공할 수 있다. *(근거: PAGE 1, Abstract)*<br>**[TO-BE]** 무릎 수술 후 일상 활동 복귀 능력을 평가하기 위해 높은 판별력, 비침습성, 낮은 비용을 갖춘 정량적 도구가 필요하며, 기계학습은 이러한 요구를 충족하고 지식 격차를 줄일 수 있는 접근으로 제시된다.<br>*(사실검증 — 누락/경미: 요약은 '운동 형상학적 변수들'로 한정하지만, 원문은 PAGE 1 Abstract에서 구체적으로 운동형상학적 변수만을 말하지 않고 정량적 도구와 기계학습 접근의 필요성을 말한다. 본문에서도 입력 데이터는 kinematic, kinetic, EMG 등으로 제시되어 있어 운동형상학적 변수로 좁히면 범위가 축소된다.)*
	- 근거 원문: “Modern lifestyles require new tools for determining a person’s ability to return to daily activities after knee surgery. These quantitative instruments must feature high discrimination, be non-invasive, and be inexpensive. Machine learning is a revolutionary approach that has the potential to satisfy the aforementioned requirements and bridge the knowledge gap.”
- 수술 후 재활 단계 분석에 첨단 기계 학습을 적용한 연구 경향성을 정량적으로 정리하고 개인별 치료 전략 수립을 위한 인공지능 보조 임상의 의사결정 가능성이 확대되고 있음을 보여주었다. *(근거: PAGE 1, Abstract)*
	- 근거 원문: “The results demonstrated a recent increase in the use of sophisticated machine learning techniques that can provide robust decision-making support during personalized post-treatment interventions for knee-surgery patients.”

## 레퍼런스할 수 있는 내용

### 1. TKA 수술 후 지속적인 기능적 손상 및 잔존 불편 호소 비율

- 원문 발췌: “Approximately 11–20% of patients experience discomfort following TKA, which is related to persisting functional impairments \[6\].”
- 한국어 번역: 인공관절 전치환술(TKA) 후 환자의 약 11\~20%는 지속적인 기능적 한계와 관련된 불편함을 경험한다.
- 원문 위치: PAGE 2, 1. Introduction
- 원문 내 인용표기: \[6\]
- 해당 선행문헌: 6. Gunaratne, R.; Pratt, D.N.; Banda, J.; Fick, D.P.; Khan, R.J.; Robertson, B.W. Patient Dissatisfaction Following Total Knee Arthroplasty: A Systematic Review of the Literature. J. Arthroplast. 2017, 32, 3854–3860. \[CrossRef\] \[PubMed\]
- 주장 유형: background_citation
- 활용 맥락과 주의: TKA 수술 이후에도 기능적 장애가 잔존해 11\~20% 수준의 높은 비율로 불편을 느낄 수 있다는 임상적 근거로 활용할 수 있다. 다만 2차 인용에 주의해야 한다.

### 2. ACLR 수술 환자의 높은 재파열 위험성과 골관절염 조기 발병 우려

- 원문 발췌: “Alongside, the results after anterior cruciate ligament reconstruction (ACLR) can be poor, with an increased risk of ACL re-rupture and earlier onset of OA compared with healthy individuals \[8\].”
- 한국어 번역: 동시에 전방십자인대 재건술(ACLR) 후 결과는 나쁠 수 있으며, 건강한 일반인과 비교할 때 ACL 재파열 위험이 증가하고 골관절염(OA)의 조기 발병 가능성이 높아진다.
- 원문 위치: PAGE 2, 1. Introduction
- 원문 내 인용표기: \[8\]
- 해당 선행문헌: 8. Ajuied, A.; Wong, F.; Smith, C.; Norris, M.; Earnshaw, P.; Back, D.; Davies, A. Anterior Cruciate Ligament Injury and Radiologic Progression of Knee Osteoarthritis: A Systematic Review and Meta-Analysis. Am. J. Sport. Med. 2014, 42, 2242–2252. \[CrossRef\]
- 주장 유형: background_citation
- 활용 맥락과 주의: ACLR 수술 후 발생 가능한 부정적 예후(골관절염 조기 발생, 재파열 위험 증가)를 선행 지표로 서론 등에 인용하기 적절하며 2차 인용에 주의를 요한다.

### 3. KPCA 차원 축소와 다중 클래스 SVM 보행 보조기기 분류의 타당성

- 원문 발췌: “They used a multiclass SVM for classification, and this combination of KPCA and MSVM achieved an accuracy of 98%.”
- 한국어 번역: 그들은 분류를 위해 다중 클래스 SVM을 사용했으며, KPCA와 MSVM의 결합 방식은 98%의 분류 정확도를 달성했다.
- 원문 위치: PAGE 7, 3.1. TKA Surgery
- 원문 내 인용표기: \[27\]
- 해당 선행문헌: 27. Martins, M.; Santos, C.; Costa, L.; Frizera, A. Feature Reduction with PCA/KPCA for Gait Classification with Different Assistive Devices. Int. J. Intell. Comput. Cybern. 2015, 8, 363–382. \[CrossRef\]
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: TKA 환자의 보행 유형 및 보조 기기 사용 상태 식별에서 KPCA를 활용한 차원 축소와 다중 클래스 SVM(MSVM) 분류 모델의 우수한 예측 성능(98% 정확도)을 입증하는 결과 데이터로 사용 가능하다. 2차 인용에 유의한다.

### 4. 21개 보행 특징과 SVM 기반의 ACL 부상 및 수술 후 보행 유형 최적 분류 정확도

- 원문 발췌: “The best score was achieved (94.95% accuracy) by the SVM classifier, which employed 21 biomechanical parameters.”
- 한국어 번역: 21개의 생체역학적 매개변수를 활용한 SVM 분류기가 가장 뛰어난 점수인 94.95% 정확도를 달성했다.
- 원문 위치: PAGE 8, 3.2. ACL Surgery
- 원문 내 인용표기: \[29\]
- 해당 선행문헌: 29. Kokkotis, C.; Moustakidis, S.; Tsatalas, T.; Ntakolia, C.; Chalatsis, G.; Konstadakos, S.; Hantes, M.E.; Giakas, G.; Tsaopoulos, D. Leveraging Explainable Machine Learning to Identify Gait Biomechanical Parameters Associated with Anterior Cruciate Ligament Injury. Sci. Rep. 2022, 12, 6647. \[CrossRef\]
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: 전방십자인대 부상 환자군(ACLD, ACLR, Control)의 분류 예측 모델 구축 시 21개의 차원 축소된 보행 변수와 SVM을 통해 94.95%의 높은 성능을 도출했다는 실증 결과로 인용할 수 있다. 2차 인용에 유의한다.


---

# [34] Learning based lower limb joint kinematic estimation using open source IMU data

(저자: Benjamin Hur, Sunin Baek, Inseung Kang, Daekyum Kim | 연도: 2025 | 저널: Scientific Reports | DOI: https://doi.org/10.1038/s41598-025-89716-4)

Hur, B., Baek, S., Kang, I., & Kim, D. (2025). Learning based lower limb joint kinematic estimation using open source IMU data. Scientific Reports, 15, 5287. https://doi.org/10.1038/s41598-025-89716-4

## 서지정보

- 저자: Benjamin Hur, Sunin Baek, Inseung Kang, Daekyum Kim
- 연도: 2025
- 저널: Scientific Reports
- DOI: https://doi.org/10.1038/s41598-025-89716-4
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/04_machine_learning_and_deep_learning/Learning based lower limb joint kinematic estimation using open source IMU data.pdf
- 분석 provider: antigravity

## 연구 목적

- 본 연구는 관성 측정 장치(IMU)와 딥러닝 프레임워크를 활용하여 하지 관절 운동학(kinematics)을 추정하는 프레임워크를 제안하고 평가하고자 한다. *(근거: PAGE 1, Abstract)*
	- 근거 원문: “This study introduces a deep learning framework for estimating lower-limb joint kinematics using inertial measurement units (IMUs).”

## 연구 설계와 대상

- 오픈소스 데이터셋에서 건강한 성인 11명(남성 9명, 여성 2명, 데이터 손상으로 4명 제외)의 보행 데이터를 분석에 사용하였다. *(근거: PAGE 3, Methods - Open-source data)*
	- 근거 원문: “For this study, we used an open-source gait data that included IMU(MTw Awinda, Xsens North America Inc., Culver City, CA, USA) and OMC(Motion Analysis Corporation, Santa Rosa, CA, USA) from 11 healthy individuals (9 males and 2 females)27.”
- 모델의 적응성 검증을 위해 추가적으로 3명의 피험자로부터 3미터 직선 보행 및 180도 회전을 수행하는 독립적인 데이터를 수집하였다. *(근거: PAGE 3, Methods - Data collection for model adaptability verification)*
	- 근거 원문: “Our dataset includes IMU and OMC data from three individuals performing a 5-minute session of 3 meters straight walking and 180◦ turns at a self-selected pace.”

## 방법

- 하지 관절 운동학 추정을 위해 CNN과 LSTM 두 가지 딥러닝 네트워크 아키텍처를 구현하여 비교 분석하였다. *(근거: PAGE 4, Methods - Deep learning)*
	- 근거 원문: “In developing our model for estimating lower-limb kinematics during gait, we used Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTMs) networks, which are widely used and are proven to be effective in deep learning applications for joint kinematic estimation26,33.”
- 훈련 모델은 세 가지 방식(개인화 UI, 일반화 UG, 적응형 UA)으로 학습하고 평가를 수행하였다. *(근거: PAGE 5, Methods - Deep learning)*
	- 근거 원문: “Weevaluatedthreedifferentmethodsofselectingtrainingdatatodevelopdeeplearningmodelsthatestimate lower-limb kinematics: the ‘user-individualized method’, the ‘user-generalized method’, and the ‘user-adaptive method’.”
- 각 관절각의 참값(Ground Truth)은 광학 모션 캡처(OMC) 기반의 데이터를 OpenSim 4.4의 역운동학(Inverse Kinematics) 연산을 통해 도출하여 사용하였다. *(근거: PAGE 5, Methods - Inverse kinematics)*
	- 근거 원문: “Forthegroundtruthvaluesofjointangles,weusedOMC-basedinversekinematicscalculatedthroughOpenSim 4.4.”

## 핵심 결과

- 개인화 방식(UI) LSTM 모델은 IMU 기반 역운동학에 비해 평균 RMSE는 49.20%, NRMSE는 50.65% 낮았으며, 상관계수는 20.13% 높았다. *(근거: PAGE 6, Results - Overall model performance comparison)*
	- 근거 원문: “Specifically,theLSTMmodelshowed49.20%loweraverageRMSE, 50.65% lower average NRMSE, and 20.13% higher correlation coefficient compared to IMU-based inverse kinematics.”
- 일반화 방식(UG)은 IMU 기반 역운동학에 비해 RMSE가 LSTM의 경우 115.87%, CNN의 경우 121.50% 증가하여 높은 오차를 보였다. *(근거: PAGE 6, Results - Overall model performance comparison)*
	- 근거 원문: “Compared to RMSE values of IMU-based inverse kinematics, the UG showed higher error, with RMSE values 115.87% and 121.50% larger for the LSTM and CNN models, respectively.”
- 적응형 방식(UA) LSTM 모델은 IMU 기반 역운동학보다 평균 RMSE는 0.4%, NRMSE는 7% 약간 더 낮아 역운동학과 대등한 성능을 보였다. *(근거: PAGE 6, Results - Overall model performance comparison)*
	- 근거 원문: “The average RMSE and NRMSE for the UA LSTM model were slightly lower than those of IMU-based inverse kinematics (0.4% and 7% respectively).”
- 다양한 IMU 조합 중에서 대퇴골(femur)과 종골(calcaneus)에 장착된 IMU를 결합했을 때 굴곡/신전(sagittal) 엉덩관절 및 무릎관절 각도 추정에서 가장 낮은 RMSE 오차를 나타냈다. *(근거: PAGE 6, Results - IMU combinations)*
	- 근거 원문: “Across all methods, IMU combinations including femur and calcaneus showed the lowest RMSE values for sagittal hip and knee joint angle estimation.”

## 저자 결론

- 오픈소스 IMU 데이터셋 기반의 전이학습(UA)을 통해 개인화된 하지 관절 각도 추정 모델의 정확도를 대폭 개선할 수 있으며, 데이터 수집 노력과 비용을 대폭 줄일 수 있다. *(근거: PAGE 8, Discussion)*
	- 근거 원문: “This demonstrated that a personalized joint kinematic estimation model can be constructed by developing a generalized pre-trained model using open-source datasets from prior studies and applying transfer learning with a small portion of a novel individual’s data. Because only aminimalamountofnewdataisnecessary,thisapproachsignificantlyreducedboththetimeandcostassociated with data collection and model training43,44.”

## 연구의 한계

- UA 모델의 테스트가 소수의 피험자에게만 적용되어, 성능의 보편성을 검증하기 위해 더 광범위하고 다양한 데이터셋에 대한 추가 검증이 필요하다. *(근거: PAGE 9, Discussion)*
	- 근거 원문: “One key limitation is that the UA was tested on only a few participants. While this provided an initial proof of concept, further testing across a broader range of datasets is necessary to gain a more comprehensive understanding of its performance.”
- 사전 학습된 기저 모델(UG)이 상대적으로 적은 인원의 데이터로 구축되어 초기 예측 정확도가 낮았으며, 이는 전이학습 모델의 최종 성능 향상에 제한을 준다. *(근거: PAGE 10, Discussion)*
	- 근거 원문: “Another limitation is the training of the base model, which relied on a limited set of participants. The pre-trained base model used in this study, referred to as the UG, was developedwithdatafromarelativelysmallgroupofindividuals,resultinginlowerestimationaccuracy.”
- 전이학습(UA)을 수행하기 위해서는 여전히 소량의 마커 기반 광학 모션 캡처(OMC) 데이터가 필요하므로, OMC 시스템에 대한 의존성이 한계로 존재한다. *(근거: PAGE 10, Discussion)*
	- 근거 원문: “Additionally, the UA requires a small amount of marker-based OMC data for transfer learning, indicating that continued dependence on OMC systems remains a limitation.”

## 생각해볼 내용

- 오픈소스 데이터셋을 딥러닝 기반 하지 운동학 추정에 체계적으로 활용하는 방법론을 제안하여 데이터 부족 문제를 극복하고자 시도하였다. *(근거: PAGE 2, Introduction)*
	- 근거 원문: “Thus, it is evident that developing a more generalizable model and a systematic approach to handling open-source data for IMU-based kinematic estimation is critical and beneficial for the field.”

## 이 연구가 지적한 선행연구의 문제점

- 기존의 분석적 방식(역운동학)은 모든 사지에 센서를 부착해야 하여 사전 준비 과정이 복잡하고 정렬/캘리브레이션 부담이 큽니다. *(근거: PAGE 1, Introduction)*
	- 근거 원문: “However, these approaches require IMUs to be located on each individual limb13,14, introducing exhaustive pre-setup requirements (e.g., aligning and calibrating IMUs for each subject).”
- 기존 분석 방식은 누적 오차로 인해 장기 신호 표류(drift) 현상이 발생합니다. *(근거: PAGE 1, Introduction)*
	- 근거 원문: “Additionally, IMU-based inverse kinematics suffer from long-term signal drift, primarily due to accumulated errors from time-varying biases in integrating acceleration data12,13.”
- 개별 피험자의 신체적 차이(관절 길이 등)와 보행 양상의 다양성으로 인해 학습된 모델이 새로운 사용자에게 적용될 때 성능이 저하되는 한계가 있습니다. *(근거: PAGE 2, Introduction)*
	- 근거 원문: “For example, variations in each individual’s joint lengths and gait patterns can cause the model to perform poorly when deploying a trained model to new users13.”

## 이 연구의 해결 방식과 기여

- 오픈소스 데이터셋을 효율적으로 활용하기 위해 일반화된 사전 모델을 구축하고 전이학습(transfer learning)을 접목한 딥러닝 프레임워크를 제안하였습니다. *(근거: PAGE 2, Introduction)*
	- 근거 원문: “In this work, we present a deep learning framework that leverages an existing open-source dataset for IMU- based joint kinematic estimation during walking.”
- 전이학습 기법을 활용함으로써 개별적인 연구에서 방대한 데이터를 새롭게 수집해야 하는 시간적, 비용적 부담을 대폭 감소시켰습니다. *(근거: PAGE 2, Introduction)*
	- 근거 원문: “We also showed that transfer learning enables researchers to efficiently utilize open-source datasets, minimizing the need for extensive data collection for their own specific motor tasks.”
- **[AS-IS]** 비교 분석을 통해 하지 관절 운동학 추정을 위한 최적의 IMU 개수와 부착 위치를 제안하여 실시간 처리 및 계산량 감축을 유도하였습니다. *(근거: PAGE 2, Introduction)*<br>**[TO-BE]** 비교 분석을 통해 하지 관절 운동학 추정을 위한 최적의 IMU 개수와 부착 위치를 제시하였다.<br>*(사실검증 — 근거불충분/경미: 요약의 앞부분인 최적 IMU 개수와 위치를 결정했다는 내용은 제시된 PAGE 2 인용문으로 지지된다. 그러나 같은 bullet의 ‘실시간 처리 및 계산량 감축’은 제시된 인용문에는 없고, PAGE 9 Discussion의 별도 문장에 근거해야 한다.)*
	- 근거 원문: “Lastly, through comparative analysis, we determined the optimal number of IMUs and their locations.”

## 레퍼런스할 수 있는 내용

### 1. IMU 기반 역운동학의 신호 드리프트 현상

- 원문 발췌: “Additionally, IMU-based inverse kinematics suffer from long-term signal drift, primarily due to accumulated errors from time-varying biases in integrating acceleration data12,13.”
- 한국어 번역: 또한, IMU 기반 역운동학은 가속도 데이터를 적분할 때 발생하는 시간 변동 바이어스의 누적 오류로 인해 주로 발생하는 장기적인 신호 드리프트 문제를 겪는다.
- 원문 위치: PAGE 1, Introduction
- 원문 내 인용표기: 12,13
- 해당 선행문헌: 12. Picerno, P. 25 years of lower limb joint kinematics by using inertial and magnetic sensors: A review of methodological approaches. Gait & Posture. 51, 239–246 (2017). 13. Hafer,J.F.,etal.Challengesandadvancesintheuseofwearablesensorsforlowerextremitybiomechanics.JournalofBiomechanics, 2023.
- 주장 유형: background_citation
- 활용 맥락과 주의: IMU 센서의 가속도계 데이터 적분 시 생기는 누적 오차로 인한 드리프트 문제를 환기시킬 때 인용할 수 있음. 2차 인용 시 원래 문헌인 Picerno(2017) 및 Hafer et al.(2023)을 검토해야 함.

### 2. 개별 신체적 차이와 보행 패턴의 다양성이 딥러닝 모델의 범용성을 저해함

- 원문 발췌: “For example, variations in each individual’s joint lengths and gait patterns can cause the model to perform poorly when deploying a trained model to new users13.”
- 한국어 번역: 예를 들어, 개별 사용자의 관절 길이와 보행 패턴의 다양성은 훈련된 모델을 새로운 사용자에게 배포할 때 성능이 저하되는 원인이 될 수 있다.
- 원문 위치: PAGE 2, Introduction
- 원문 내 인용표기: 13
- 해당 선행문헌: 13. Hafer,J.F.,etal.Challengesandadvancesintheuseofwearablesensorsforlowerextremitybiomechanics.JournalofBiomechanics, 2023.
- 주장 유형: background_citation
- 활용 맥락과 주의: 학습된 딥러닝 모델이 새로운 피험자나 환경에 적용되었을 때 개인 차이로 인해 겪는 성능 저하 한계를 지적할 때 유용함.


---

# [35] Leveraging explainable machine learning to identify gait biomechanical parameters associated with anterior cruciate ligament injury

(저자: Christos Kokkotis, Serafeim Moustakidis, Themistoklis Tsatalas, Charis Ntakolia, Georgios Chalatsis, Stylianos Konstadakos, Michael E. Hantes, Giannis Giakas, Dimitrios Tsaopoulos | 연도: 2022 | 저널: Scientific Reports | DOI: https://doi.org/10.1038/s41598-022-10666-2)

Kokkotis, C., Moustakidis, S., Tsatalas, T., Ntakolia, C., Chalatsis, G., Konstadakos, S., Hantes, M. E., Giakas, G., & Tsaopoulos, D. (2022). Leveraging explainable machine learning to identify gait biomechanical parameters associated with anterior cruciate ligament injury. Scientific Reports, 12, 6647. https://doi.org/10.1038/s41598-022-10666-2

## 서지정보

- 저자: Christos Kokkotis, Serafeim Moustakidis, Themistoklis Tsatalas, Charis Ntakolia, Georgios Chalatsis, Stylianos Konstadakos, Michael E. Hantes, Giannis Giakas, Dimitrios Tsaopoulos
- 연도: 2022
- 저널: Scientific Reports
- DOI: 10.1038/s41598-022-10666-2
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/04_machine_learning_and_deep_learning/Leveraging explainable machine learning to identify gait biomechanical parameters associated with anterior cruciate ligament injury.pdf
- 분석 provider: antigravity

## 연구 목적

- 설명 가능한 머신러닝 방법론을 개발하여 ACL 부상 진단에서 보행 운동학적 및 역학적 매개변수의 기여도를 식별 및 수량화하고, ACL 결손 환자, 재건술 환자, 건강한 대조군 간의 시상면 보행 생체역학 차이를 조사하고자 한다. *(근거: --- PAGE 1 ---)*
	- 근거 원문: “This paper focuses on the development of an explainable machine learning (ML) empowered methodology to: (i) identify important gait kinematic, kinetic parameters and quantify their contribution in the diagnosis ofACL injury and (ii) investigate the differences in sagittal plane kinematics and kinetics of the gait cycle between ACL deficient,ACL reconstructed and healthy individuals.”
- 설명 가능한 머신러닝과 통계 분석을 결합하여 분류 과정에서 특징 중요도를 추정하고 각 환자 그룹(ACL 결손, 재건, 대조군) 간 시상면 운동학 및 역학 변수의 차이를 탐색한다. *(근거: --- PAGE 2 ---)*
	- 근거 원문: “The aims of this study are: (i) to estimate the feature importance in the classification process and examine how much each of the features contributed to the final ML decisions and (ii) to investigate differences in sagittal plane kinematics and kinetics of the gait cycle between different patient groups based on a novel approach that combines explainable ML and statistical analytics.”

## 연구 설계와 대상

- 총 151명의 피험자가 연구에 참여하였으며, 수술 전 ACL 결손군(ACLD), ACL 재건군(ACLR), 대조군(CON)의 세 그룹으로 구성되었다. *(근거: --- PAGE 7 ---, Participants.)*
	- 근거 원문: “A total of 151 subjects volunteered to participate in this study. Three different groups were defined: (i) ACL-deficient prior to surgery (ACLD), (ii) ACL-reconstructed (ACLR) and (iii) control (CON) group.”
- 대조군 피험자들은 연령, 성별, 신체 활동 수준에 대해 매칭되었으며, 측정 전 12개월 동안 ACL 부상 및 lower extremity 부상 등의 병력이 없었다. *(근거: --- PAGE 7 ---, Participants.)*
	- 근거 원문: “The CON subjects were matched for age, gender, and physical activity status and had no history of ACL injury and neurologic disorder or other lower extremity injuries within 12 months prior to participating in the study.”

## 방법

- 피험자들은 개별 자체 선택 보행 속도(SWS)의 ±5% 범위 내에서 맨발로 10m 실험실 보행로를 걸었다. *(근거: --- PAGE 7 ---, Testingprocedureanddatacollection.)*
	- 근거 원문: “Subsequently, the subjects walked barefoot along the 10 m laboratory walkway within±5% of their individual self-selected walking speed (SWS).”
- 운동학 및 지면 반력(GRF) 데이터는 각각 10Hz 및 40Hz에서 4차 버터워스 필터로 저역 통과 필터링되었다. *(근거: --- PAGE 8 ---, Data analysis.)*
	- 근거 원문: “Kinematic and GRF data were lowpass filtered with a 4th order Butterworth filter at 10 and 40 Hz, respectively.”
- 특징 선택(FS)과 머신러닝 추정기들을 위한 공통의 기반을 마련하고자 데이터를 \[0, 1\] 범위로 정규화하였다. *(근거: --- PAGE 8 ---, Machinelearningworkflow.)*
	- 근거 원문: “Data were normalised to \[0, 1\] to build a common basis for the feature selection (FS) and the ML estimators.”
- 최종 머신러닝 출력값에 대한 영향도에 따라 특징들의 순위를 매기고 미니 설명자 모델을 구축하기 위해 SHAP을 도입하였다. *(근거: --- PAGE 9 ---, Machinelearningworkflow.)*
	- 근거 원문: “In this paper, we employed SHAP to rank features in terms of their impact on the final ML outputs and to build a mini explainer model.”

## 핵심 결과

- SVM 모델은 처음 선택된 특징들에 대해 상승 추세를 보였으며, 21개의 특징 그룹에서 가장 높은 분류 정확도인 94.95%를 달성했다. *(근거: --- PAGE 2 ---, Results)*
	- 근거 원문: “Specifically, the SVM model showed an upward trend with respect to the first selected features, with a maximum of 94.95% (which was the overall best performance achieved).”
- 두 번째로 높은 성능을 보인 신경망(NN) 모델은 92.89%의 테스트 정확도를 얻었으며, 15개 초과의 특징을 사용했을 때 변동이 있는 비안정적 상승 추세를 보였다. *(근거: --- PAGE 2 ---, Results)*
	- 근거 원문: “The second-best accuracy (92.89%) was achieved by the NN model, which presented a non-steadily increasing performance with fluctuations for more than 15 selected features.”
- 전체 3개 클래스 분석에서 평균 SHAP 값 크기가 0.3을 초과하여 모델 출력에 가장 큰 영향을 준 변수들은 K2, H4, A3, GRF4, GRF7, K1, A4, GRF6였다. *(근거: --- PAGE 4 ---, Results)*
	- 근거 원문: “In this approach K2, H4, A3, GRF4, GRF7, K1, A4 and GRF6 were the parameters that affected the model output with mean SHAP values higher than 0.3.”
- 대조군과 수술 전 ACL 결손군(ACLD)의 차이를 구별하는 로컬 문제 1에서 H4, K7, GRF3, H1, H2가 예측 출력에 유의미한 영향을 미치는 가장 중요한 변수였다. *(근거: --- PAGE 4 ---, Results)*
	- 근거 원문: “It should be noted that the features H4, K7, GRF3, H1, H2 were the most important variables that significantly affected the prediction output.”
- 설명성 분석에서 중요하게 파악된 변수 중 H4, K7, GRF3, GRF4의 4개 변수에서 대조군과 ACL 결손군 간의 유의미한 통계적 차이가 관찰되었다. *(근거: --- PAGE 4 ---, Results)*
	- 근거 원문: “Significant differences were observed between CON and ACLD for half of the features considered, specifically the first three (H4, K7 and GRF3) along with GRF4;”

## 저자 결론

- 보행 생체역학의 기여도를 이해하는 것은 임상의가 비침습적이고 강력한 예후 도구를 개발하는 데 유용하며, 재건술 후 환자의 비정상 보행 패턴을 식별하여 재활 프로토콜을 수정하고 관절염 발생을 예방하도록 돕는다. *(근거: --- PAGE 7 ---, Summary)*
	- 근거 원문: “Understanding the contribution of gait biomechanics is a valuable tool for creating more powerful and non-invasive prognostic tools in the hands of physicians, that will point out abnormal gait patterns in patients after ACLR to modify the rehabilitation protocol and avoid the development of osteoarthritis.”
- 선택된 매개변수의 성격과 그것이 예측 결과에 미치는 영향(SHAP을 통해 제시됨)은 훈련된 모델의 의사결정 메커니즘 이면의 근거를 밝혀주며, ACL 부상 진단에서 입력 매개변수의 기여도를 정량화하는 대안적이고 보다 총체적인 접근 방식을 제공한다. *(근거: --- PAGE 7 ---, Summary)*
	- 근거 원문: “The nature of the selected parameters along with their impact on the prediction outcome (via SHAP) were discussed to uncover the rationale behind the decision-making mechanism of the trained model and therefore provide an alternative and a more holistic approach of quantifying the contribution of the input parameters in the diagnosis of ACL injury.”

## 연구의 한계

- ACL 재건술 후 보행 생체역학이 변화하지만 연구 및 과제 전반에 걸쳐 일관된 결과를 보여주는 매개변수가 드물기 때문에 임상적 중요성은 주의해서 고려해야 한다. *(근거: --- PAGE 6 ---, Discussion)*
	- 근거 원문: “This can be attributed to the fact that even though gait biomechanics are altered following ACLR, few biomechanical parameters demonstrate consistent results across studies and various tasks10.”
- SHAP은 개별 변수가 모델 출력에 미치는 영향력을 수량화하는 단순한 설명에 국한되어, 변수들의 조합이 최종 결정에 기여하는 복잡한 내부 작동 방식은 여전히 알기 어렵다. *(근거: --- PAGE 7 ---, Discussion)*
	- 근거 원문: “However, SHAP is limited to simple explanations mainly quantifying the impact of individual features to the models’ output40.”

## 생각해볼 내용

- **[AS-IS]** 설명 가능성 분석 도구(SHAP)의 도입은 블랙박스 머신러닝 모델의 의사결정을 인간이 더 직관적으로 이해하고 임상적으로 해석할 수 있도록 돕는 유용한 가교가 된다. *(근거: --- PAGE 7 ---, Discussion)*<br>**[TO-BE]** 설명 가능성 분석 도구(SHAP)의 도입은 블랙박스 머신러닝 모델의 의사결정을 인간이 더 잘 이해하도록 돕는 중요한 수단이 된다.<br>*(사실검증 — 과장/경미: 원문은 SHAP 등이 블랙박스 모델의 결정을 인간이 더 잘 이해하도록 돕는다고 설명하지만, 해당 문장 자체는 '임상적으로 해석'까지 직접 말하지 않는다. 논문 전체가 임상적 맥락을 다루기는 하나, 이 근거문만으로는 임상적 해석을 돕는다고 단정하기에는 표현이 조금 강하다.)*
	- 근거 원문: “Explainability via SHAP or other similar tools is a crucial enabler allowing humans to better comprehend the decisions generated by black box models.”
- 전통적인 통계적 유의성 검정만으로는 놓치기 쉬운 비선형적이거나 복합적인 매개변수들의 상호작용을 머신러닝과 설명성 분석을 결합하여 포착할 수 있다는 점에서 방법론적 의의가 크다. *(근거: --- PAGE 6 ---, Discussion)*
	- 근거 원문: “Features, that would have been neglected by the traditional statistical analysis, are highlighted as contributing parameters that have a significant impact on the ML model’s output when they are combined with other statistically important ones.”

## 이 연구가 지적한 선행연구의 문제점

- 기존의 ACL 분야 머신러닝 연구는 대다수 모델을 블랙박스로 처리하여 투명성이 부족했다. *(근거: --- PAGE 2 ---, Introduction)*
	- 근거 원문: “Despite the relatively large number of ML studies on the field of ACL, the reported trained ML models are treated as black boxes.”
- 모델의 투명성과 설명성 결여로 인해 인공지능 모델이 어떠한 판단 메커니즘을 거쳐 내부적으로 의사결정을 내렸는지 이해하기 어려웠다. *(근거: --- PAGE 2 ---, Introduction)*
	- 근거 원문: “The lack of transparency and explainability of the models result to poor understanding of their inner workings and the rationale behind their decision-making mechanism.”

## 이 연구의 해결 방식과 기여

- 본 연구는 ACL 부상과 관련된 주요 매개변수를 식별하기 위해 설명 가능한 머신러닝 방법론과 통계 분석을 통합한 새로운 접근법을 제안한다. *(근거: --- PAGE 4 ---, Discussion)*
	- 근거 원문: “This paper focuses on the development of a novel approach, which combines an explainable ML-empowered methodology and statistical analysis, for identifying important parameters associated with ACL injury.”
- 본 연구의 주요 기여는 분류 성능뿐만 아니라 각 특징이 의사결정에 얼마나 기여하는지 조사하고, 특징 중요도를 추정하며, 세 환자 그룹 간의 3차원 지면 반력(GRF) 및 시상면 운동학/역학적 보행 패턴의 차이를 조사하는 데 있다. *(근거: --- PAGE 4 ---, Discussion)*
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


---

# [36] Machine learning models for predicting return to sports after anterior cruciate ligament reconstruction: Physical performance in early rehabilitation

(저자: Ui-jae Hwang, Jin-seong Kim, Keong-yoon Kim, Kyu-sung Chung | 연도: 2024 | 저널: DIGITAL HEALTH | DOI: https://doi.org/10.1177/20552076241299065)

Hwang, U. J., Kim, J. S., Kim, K. Y., & Chung, K. S. (2024). Machine learning models for predicting return to sports after anterior cruciate ligament reconstruction: Physical performance in early rehabilitation. DIGITAL HEALTH, 10, 1–11. https://doi.org/10.1177/20552076241299065

## 서지정보

- 저자: Ui-jae Hwang, Jin-seong Kim, Keong-yoon Kim, Kyu-sung Chung
- 연도: 2024
- 저널: DIGITAL HEALTH
- DOI: 10.1177/20552076241299065
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/04_machine_learning_and_deep_learning/Machine learning models for predicting return to sports after anterior cruciate ligament reconstruction - Physical performance in early rehabilitation.pdf
- 분석 provider: antigravity

## 연구 목적

- 전방십자인대 재건술(ACLR) 후 3개월 시점의 신체 수행 능력 변수를 바탕으로, 수술 후 12개월 시점의 스포츠 복귀(RTS) 성공을 예측하는 데 가장 우수한 성능을 보이는 머신러닝 모델을 식별하는 것이다. *(근거: PAGE 1, Abstract)*
	- 근거 원문: “In this study, we aimed to identify the best-performing machine learning models for predicting RTS at 12 months post-ACLR, based on physical performance variables at 3 months post-ACLR.”

## 연구 설계와 대상

- 이 연구는 단일 기관에서 단일 의사에 의해 해부학적 단일 다발 전방십자인대 재건술(ACLR)을 받은 18세에서 45세 사이의 환자 102명을 대상으로 한 후향적 환자-대조군 연구이다. *(근거: PAGE 1, Abstract)*
	- 근거 원문: “This case-control study included 102 patients who had undergone ACLR.”
- **[AS-IS]** 연구 대상자는 2016년 6월부터 2022년 4월 사이에 수술을 받고 수술 후 3개월 및 12개월 시점에 요구되는 모든 테스트를 완료했으며 동반 다발 인대 손상, 골절, 반월상 연골 봉합/절제술, 개정 ACLR 등의 제외 기준에 해당하지 않는 환자들로 구성되었다.<br>**[TO-BE]** 연구 대상자는 2016년 6월부터 2022년 4월 사이에 수술을 받고 수술 후 3개월 및 12개월 시점의 요구 검사를 완료했으며, 동반 다발 인대 손상, 골절, 반월상연골 뿌리 봉합, 연골 복원, 정렬 교정 절골술, 반월상연골 아전절제 또는 전절제, 재수술 ACLR, 양측 무릎 수술 과거력 등의 제외 기준에 해당하지 않는 환자들로 구성되었다.<br>*(사실검증 — 사실불일치/중대: 요약은 제외 기준을 '반월상 연골 봉합/절제술'로 넓게 적었지만, 원문은 'meniscal root repair'와 'subtotal or total meniscectomy'를 제외 기준으로 제시한다. 모든 반월상연골 봉합이나 절제술이 제외된 것처럼 읽혀 대상자 기준이 달라진다.)* *(근거: PAGE 2, Methods - Patients)*
	- 근거 원문: “The medical records of 102 patients who had undergone single-bundle anatomical ACLR using the outside-in technique with a ﬂip-cutter (Arthrex, Naples, FL, USA) between June 2016 and April 2022 were retrospectively reviewed to obtain their demographic and clinical characteristics. A single surgeon performed all the operations. The inclusion criteria for this study were patients who had undergone single-bundle ACLR, aged between 18 and 45 years, and complied with all the required tests at 3 and 12 months post-surgery. The exclusion criteria were as follows: concomitant mul-tiple ligament injury, fracture, meniscal root repair, cartil-age repair, osteotomy to correct mechanical alignment, subtotal or total meniscectomy, revision ACLR, and history of knee surgery on the involved and uninvolved sides.”

## 방법

- 수술 후 3개월 시점에 Biodex 균형 시스템(BBS) 테스트, Y-밸런스 테스트(YBT), 등속성 근력 테스트(concentric strength)를 수행하여 독립 변수를 수집하였고, 수술 후 12개월 시점에 외다리 홉 테스트, 외다리 수직 점프 테스트, Tegner 활동 점수를 타겟 변수로 측정하였다. *(근거: PAGE 2, Methods - Procedure)*
	- 근거 원문: “The physical performance variables (as feature or independent variables) measured at 3 months post-ACLR included the Biodex balance system (BBS) test, Y-balance test (YBT), and isokinetic muscle strength test. The RTS variables (as target or dependent variables) measured at 12 months post-ACLR include the single-leg hop test, single-leg vertical jump test, and Tegner activity score.”
- 12개월 시점의 3가지 RTS 타겟 변수는 외다리 홉 및 수직 점프 테스트의 대칭 지수(LSI)가 10% 미만이고 Tegner 활동 점수가 6점 초과인 경우를 성공 기준으로 하여 이분형 변수로 변환되었다. *(근거: PAGE 4, Methods - Machine learning modeling - Pre-processing and missing data handling)*
	- 근거 원문: “The three RTS targets (single-leg hop test, single-leg vertical jump test, and Tegner activity score) were transformed into dichotomous variables as an LSI of the single-leg hop and vertical jump test\<10% and a Tegner activity score\>6 points.”
- 102명의 데이터를 80%의 훈련 세트(82명)와 20%의 테스트 세트(20명)로 분할하고, 로지스틱 회귀, 의사결정 나무, 랜덤 포레스트, 그래디언트 부스팅, 서포트 벡터 머신, 인공신경망 등 6가지 알고리즘을 5-fold 교차 검증을 통해 훈련했다. *(근거: PAGE 4, Methods - Machine learning modeling - Machine learning algorithm)*
	- 근거 원문: “We split the complete data (n= 102) into a training set (80%, n=82) for model development and a test set (20%, n=20) for external validation to predict model performance. Six machine learning algorithms were trained via a ﬁve-fold cross-validation, including logistic regression, decision tree, random forest, gradient boosting, support vector machine, and neural network.”

## 핵심 결과

- 외다리 홉 테스트 기반 RTS 성공 예측의 경우 테스트 세트에서 랜덤 포레스트 모델이 가장 높은 성능(AUC 0.952)을 나타냈다. *(근거: PAGE 5, Results - Predictive models of machine learning)*
	- 근거 원문: “Random forest models in the test set best predicted the RTS success based on the single-leg hop test (area under the curve \[AUC\], 0.952) and Tegner activity score (AUC, 0.949).”
- Tegner 활동 점수 기반 RTS 성공 예측의 경우 테스트 세트에서 랜덤 포레스트 모델이 가장 높은 성능(AUC 0.949)을 보여주었다. *(근거: PAGE 5, Results - Predictive models of machine learning)*
	- 근거 원문: “Regarding RTS success prediction based on Tegner activity score, the random forest algorithm models had the highest AUC in the training (AUC, 0.826 \[good\]; F1, 0.751) and test (AUC, 0.949 \[excellent\]; F1, 0.952) sets.”
- 외다리 수직 점프 테스트 기반 RTS 성공 예측의 경우 테스트 세트에서 그래디언트 부스팅 모델이 가장 높은 성능(AUC 0.868)을 나타냈다. *(근거: PAGE 1, Abstract - Results)*
	- 근거 원문: “Gradient boosting models in the test set best predicted the RTS based on the single-leg vertical jump test (AUC, 0.868).”

## 저자 결론

- 전방십자인대 재건술(ACLR) 후 조기 재활 단계(3개월 시점)에서 수정 가능한 요인들을 고려함으로써 성공적인 스포츠 복귀(RTS) 가능성을 향상시킬 수 있다. *(근거: PAGE 1, Abstract - Conclusion)*
	- 근거 원문: “Modiﬁable factors should be considered in the early rehabilitation stage after ACLR to enhance the possibility of a successful RTS.”

## 연구의 한계

- 머신러닝을 적용하기에 표본의 크기(n=102)가 상대적으로 작아 분석 결과의 일반화가 제한될 수 있다. *(근거: PAGE 8, Discussion)*
	- 근거 원문: “First, the sample size was relatively small for machine learning applications, which may limit the ﬁndings’ generalizability.”
- 대퇴사두근 및 햄스트링 근력 강도에 상당한 영향을 줄 수 있는 이식건 종류나 수술 기법에 대한 통제를 분석 모형에 적용하지 않았다. *(근거: PAGE 8, Discussion)*
	- 근거 원문: “Second, we did not incorporate controls for variables such as surgical technique or graft type, despite the well-documented knowledge that these factors can exert distinct inﬂuences on quadriceps and hamstring strength outcomes.”
- 심리적 요인, 수술 관련 수치, 부상 전의 활동 수준 등 비성능 지표 변수들이 예측 모델에서 제외되었다. *(근거: PAGE 8, Discussion)*
	- 근거 원문: “Third, although physical performance outcomes at 3 months post-ACLR were analyzed in the prediction models, other potential predictive variables, such as psychological factors, surgical metrics, and pre-injury activity levels, were omitted.”
- 머신러닝 알고리즘의 다양성으로 인해 본 연구에서 선택한 6가지 모델 이외에 더 우수한 다른 알고리즘 모델이 존재할 가능성이 있다. *(근거: PAGE 8, Discussion)*
	- 근거 원문: “Fourth, the best-performing model may not have been one of the six models we selected, as machine learning algorithms are diverse.”
- 단일 기관에서 획득된 데이터만을 사용하여 진행되었기 때문에 향후 다기관 연구 및 독립 샘플을 통한 검증이 필요하다. *(근거: PAGE 8, Discussion)*
	- 근거 원문: “Lastly, our study was limited to data from a single institution. A multi-centric approach using an independent sample to val-idate the models could enhance the generalizability of the results in future studies.”

## 생각해볼 내용

- 이 연구 결과의 임상적 의의는 재활 초기 단계에서 스포츠 복귀 결과가 좋지 않을 위험이 있는 환자를 선별하고, 보다 목표 지향적이고 개인화된 재활 프로토콜을 적용할 수 있는 기회를 제공한다는 점이다. 그러나 실제 임상 적용을 위해서는 더 큰 규모의 다기관 코호트 연구를 통한 검증이 반드시 수반되어야 한다. *(근거: PAGE 6, Discussion)*
	- 근거 원문: “The clinical implications of our ﬁndings include the potential to identify patients at risk of poor RTS outcomes early in the rehabilitation process. This could allow for more targeted interventions and personalized rehabilitation protocols. However, further validation in larger, multi-center cohorts is needed before clinical implementation.”

## 이 연구가 지적한 선행연구의 문제점

- ACLR 이후 12개월 시점의 RTS 예측을 위해 수술 전이나 수술 후 6개월 시점의 추적 관찰 결과를 활용하는 머신러닝 연구는 주목을 받았으나, 조기 재활 단계(3개월 시점)에서의 예측 인자 개발은 덜 강조되었다. *(근거: PAGE 2, Introduction)*
	- 근거 원문: “Considerable attention has been paid to machine learning models in predicting preoperative26,27 or 6 months follow-up outcomes28,29 for RTS at 12 months post-ACLR, yet less emphasis has been placed on predictors at the early stages of rehabilitation.”

## 이 연구의 해결 방식과 기여

- ACLR 수술 후 3개월 시점에 측정한 등속성 근력 및 균형 능력 등의 조기 재활 단계 변수들을 활용한 18가지 머신러닝 예측 모델을 검증하여 우수한 스포츠 복귀 예측 성능을 가진 모델을 제시했다. *(근거: PAGE 5, Discussion)*
	- 근거 원문: “In this study, we selected eight predictive variables at 3 months post-ACLR and three outcome variables of RTS at 12 months post-ACLR to validate 18 machine learning models, with six machine learning algorithms for each clinical outcome.”
- 본 연구에서 도출된 수정 가능한 변수들은 12개월 시점의 성공적인 스포츠 복귀를 돕기 위해 조기 재활 단계에서 수행할 수 있는 맞춤형 운동 또는 물리치료 치료법의 방향성을 제시한다. *(근거: PAGE 6, Discussion)*
	- 근거 원문: “The modiﬁable variables presented in this study can guide exercise or physical therapy in the early rehabilitation stage for successful RTS at 12 months post-ACLR.”

## 레퍼런스할 수 있는 내용

### 1. 전방십자인대 재건술 후 스포츠 복귀 비율 및 이전 수준 회복 확률

- 원문 발췌: “However, a successful ACLR does not assure a patient’s RTS, as studies indicate that only 63% of patients regain their preinjury activity levels, with reported return rates ranging from 39% to 74%.2–6”
- 한국어 번역: 그러나 성공적인 ACLR이 환자의 RTS를 보장하지는 않는데, 연구들에 따르면 63%의 환자만이 부상 전의 활동 수준을 회복하고 보고된 복귀 비율은 39%에서 74% 사이이기 때문이다.
- 원문 위치: PAGE 1, Introduction
- 원문 내 인용표기: 2–6
- 해당 선행문헌: 2. Ardern CL, Taylor NF, Feller JA, et al. Fifty-ﬁve per cent return to competitive sport following anterior cruciate ligament reconstruction surgery: an updated systematic review and meta-analysis including aspects of physical functioning and contextual factors. Br J Sports Med 2014; 48: 1543–1552. 3. Suzuki M, Ishida T, Matsumoto H, et al. Association of psychological readiness to return to sports with subjective level of return at 12 months after ACL reconstruction. Orthop J Sports Med 2023; 11: 23259671231195030. 4. Joreitz R, Lynch A, Rabuck S, et al. Patient-speciﬁc and surgery-speciﬁc factors that affect return to sport after ACL reconstruction. Int J Sports Phys Ther 2016; 11: 264. 5. LindangerL,StrandT,Mølster AO,et al.Effectof earlyresidual laxity after anterior cruciate ligament reconstruction on long-term laxity, graft failure, return to sports, and subjective outcome at 25 years. Am J Sports Med 2021; 49: 1227–1235. 6. Ortiz E, Zicaro JP, Mansilla IG, et al. Revision anterior cruciate ligament reconstruction: return to sports at a minimum 5-year follow-up. World J Orthop 2022; 13: 12.
- 주장 유형: background_citation
- 활용 맥락과 주의: 전방십자인대 재건술 후 복귀 실패율 및 이전 수준으로 회복하는 비율을 서술할 때 배경 근거로 인용하기 적절함.

### 2. ACLR 수술 직후 무릎 기능 저하 기전

- 원문 발췌: “Following ACLR, patients experience reduced knee muscle strength due to hamstring graft harvesting and quadriceps inhibition,7 along with postural stability compromise attributed to ACL mechanoreceptor injury.8”
- 한국어 번역: ACLR 이후 환자들은 햄스트링 이식건 채취 및 대퇴사두근 억제로 인한 무릎 근력 감소와 함께, ACL 기계적 수용기 손상으로 인한 자세 안정성 저하를 경험한다.
- 원문 위치: PAGE 1, Introduction
- 원문 내 인용표기: 7,8
- 해당 선행문헌: 7. de Jong SN, van Caspel DR, van Haeff MJ, et al. Functional assessment and muscle strength before and after reconstruction of chronic anterior cruciate ligament lesions. Arthroscopy 2007; 23: 21. e21–221. 11. 8. Paterno MV, Schmitt LC, Ford KR, et al. Biomechanical measures during landing and postural stability predict second anterior cruciate ligament injury after anterior cruciate ligament reconstruction and return to sport. Am J Sports Med 2010; 38: 1968–1978.
- 주장 유형: background_citation
- 활용 맥락과 주의: 수술 후 무릎 근력 및 자세 균형 능력이 감소하는 기전에 대한 논리를 구성할 때 배경 인용문으로 유용함.

### 3. 3개월 시점의 조기 재활 치료의 초점

- 원문 발췌: “Within the ﬁrst 3 months after ACLR, the primary focus of rehabilitation is to reduce pain, restore quadriceps and hamstring strength, and incorporate proprioception training.49,50”
- 한국어 번역: ACLR 이후 첫 3개월 이내에 재활의 일차적인 초점은 통증을 줄이고 대퇴사두근 및 햄스트링 근력을 회복시키며, 고유수용성 감각 훈련을 통합하는 것이다.
- 원문 위치: PAGE 7, Discussion
- 원문 내 인용표기: 49,50
- 해당 선행문헌: 49. Erickson LN, Jacobs CA, Johnson DL, et al. Psychosocial factors 3-months after anterior cruciate ligament reconstruction predict 6-month subjective and objective knee outcomes. J Orthop Res 2022; 40: 231–238. 50. Kline PW, Johnson DL, Ireland ML, et al. Clinical predictors of knee mechanics at return to sport following ACL reconstruction. Med Sci Sports Exercise 2016; 48: 90.
- 주장 유형: background_citation
- 활용 맥락과 주의: 수술 직후부터 3개월까지 초기 재활 단계에서 반드시 집중해야 하는 치료적 목표 및 요인들을 설명할 때 적합함.

### 4. 12개월 시점 RTS 예측을 위한 최적 머신러닝 예측 성능 및 알고리즘

- 원문 발췌: “Random forest models in the test set best predicted the RTS success based on the single-leg hop test (area under the curve \[AUC\], 0.952) and Tegner activity score (AUC, 0.949).”
- 한국어 번역: 테스트 세트의 랜덤 포레스트 모델은 외다리 홉 테스트(AUC, 0.952) 및 Tegner 활동 점수(AUC, 0.949)를 바탕으로 한 RTS 성공을 가장 잘 예측했다.
- **[AS-IS]** 원문 위치: PAGE 5, Results<br>**[TO-BE]** 원문 위치: PAGE 1, Abstract - Results<br>*(사실검증 — 인용표기오류/경미: 해당 항목의 원문 발췌문은 SOURCE_TEXT에서 PAGE 1의 Abstract - Results에 그대로 제시된다. PAGE 5 Results에는 같은 결과가 더 풀어 서술되어 있으나, 요약에 적은 직접 발췌문과 정확히 일치하는 위치는 PAGE 1이다.)*
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


---

# [37] Motion Sensors-Based Machine Learning Approach for the Identification of Anterior Cruciate Ligament Gait Patterns in On-the-Field Activities in Rugby Players

(저자: Salvatore Tedesco, Colum Crowe, Andrew Ryan, Marco Sica, Sebastian Scheurer, Amanda M. Clifford, Kenneth N. Brown, Brendan O’Flynn | 연도: 2020 | 저널: Sensors | DOI: https://doi.org/10.3390/s20113029)

Tedesco, S., Crowe, C., Ryan, A., Sica, M., Scheurer, S., Clifford, A. M., Brown, K. N., & O’Flynn, B. (2020). Motion Sensors-Based Machine Learning Approach for the Identification of Anterior Cruciate Ligament Gait Patterns in On-the-Field Activities in Rugby Players. Sensors, 20(11), 3029. https://doi.org/10.3390/s20113029

## 서지정보

- 저자: Salvatore Tedesco, Colum Crowe, Andrew Ryan, Marco Sica, Sebastian Scheurer, Amanda M. Clifford, Kenneth N. Brown, Brendan O’Flynn
- 연도: 2020
- 저널: Sensors
- DOI: 10.3390/s20113029
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/04_machine_learning_and_deep_learning/Motion Sensors-Based Machine Learning Approach for the Identification of Anterior Cruciate Ligament Gait Patterns in On-the-Field Activities in Rugby Players.pdf
- 분석 provider: antigravity

## 연구 목적

- 낮은 비용과 작은 크기의 웨어러블 센서를 활용하여 방향 전환 활동을 수행하는 럭비 선수들을 대상으로 건강한 그룹과 ACL 재건 수술을 받은 그룹을 머신러닝을 통해 구별할 수 있는지 조사하고자 한다. *(근거: PAGE 1, Abstract)*
	- 근거 원문: “In particular, this study aims to investigate the ability of a set of inertial sensors worn on the lower-limbs by rugby players involved in a change-of-direction (COD) activity to differentiate between healthy and post-ACL groups via the use of machine learning.”
- 방향 전환 활동 시 ACL 재건 그룹과 건강한 그룹 간의 유의미한 차이를 감지하여 ACL 손상의 장기적인 영향이 있는지 조사하고, 주체 관련 정보나 표준 보행 시공간 매트릭스 등과 무관한 자동화되고 객관적인 분류 방법을 제공하고자 한다. *(근거: PAGE 3, 1. Introduction)*
	- 근거 원문: “The aim of this study is two-fold: (i) to investigate whether there is a long after-effect of the ACL damage in rugby players, detecting significant differences in ACL-reconstructed vs. healthy players, when involved in a change-of-direction activity; (ii) to provide an automated and objective method to distinguish between healthy and post-ACL groups of rugby players which is independent from subject-related information, step detection and segmentation processes, and standard gait spatiotemporal metrics, through the combination of a set of inertial sensors worn on the lower-limbs and data-driven machine learning models.”

## 연구 설계와 대상

- Irish 대학생 남성 럭비 선수 중 성공적으로 복귀한 ACL 재건군 6명과 건강한 대조군 6명을 포함하여 총 12명의 피험자가 본 연구에 참가하였다. *(근거: PAGE 1, Abstract)*
	- 근거 원문: “Twelve male participants (six healthy and six post-ACL athletes who were deemed to have successfully returned to competitive rugby and tested in the 5–10 year period following the injury) were recruited for the study.”
- 분석 대상은 아일랜드 대학생들 중에서 e-mail, 포스터, 구두 홍보를 통해 모집된 12명의 비엘리트 남성 럭비 선수이다. *(근거: PAGE 3, 2.1. Participants)*
	- 근거 원문: “The analysis in this study is based on a sample of twelve non-elite rugby players (all males, age: 26 ± 5.2 years; height: 182.6 ± 5.8 cm; mass: 90 ± 12.8 kg). Players were recruited via a general invitation e-mail, posters, and word of mouth, to students at a University in Ireland.”
- 피험자들은 두 그룹으로 나뉘었다: 왼쪽 다리에 ACL 재건술 역사가 있는 6명의 선수와 하지 부상 이력이 없는 6명의 선수. *(근거: PAGE 3, 2.1. Participants)*
	- 근거 원문: “The subjects were divided in two groups: six players with a history of ACL reconstruction surgery (age: 29.3 ± 4.5 years; height: 182.3 ± 6.2 cm; mass: 89.2 ± 14.7 kg), and six players with no history of lower-limbs injuries (age: 22.8 ± 3.7 years; height: 182.8 ± 6.1 cm; mass: 90.8 ± 11.9 kg).”

## 방법

- 각 피험자는 시작 지점에서 출발하여 5m를 달린 후 좌측 또는 우측으로 45도 방향 전환(사이드스텝)을 수행한 뒤 3m를 더 질주하여 정지하는 작업을 10회 반복하였다. *(근거: PAGE 4, 2.2. Data Collection Protocol)*
	- 근거 원문: “Each participant began the data collection at a pre-defined start point, and was asked to run for 5 m towards a side-step platform. During the run, the participants were instructed regarding which direction the side-step had to occur (left or right). They were then required to step at a 45 degree angle from the sidestep board in either direction, and finally run an additional 3 m to come to a full stop.”
- 다리당 두 개의 관성 측정 장치(IMUs)가 경골 전면(경골 결절 아래 10cm) 및 외측 대퇴부(경골 결절 위 15cm)에 부착되어 3차원 가속도와 자이로스코프 데이터를 수집하였다. *(근거: PAGE 4, 2.2. Data Collection Protocol)*
	- 근거 원문: “Two inertial measurement units (IMUs) were attached per leg, in particular to the anterior tibia, 10 cm below the tibial tuberosity, and to the lateral thigh, 15 cm above the tibial tuberosity, using Velcro straps.”
- 테스트 시작 전 모든 착용 센서들의 시작 시점을 시간상 동기화하기 위해 피험자들이 깊은 스쿼트 동작을 수행하도록 하였다. *(근거: PAGE 4, 2.2. Data Collection Protocol)*
	- 근거 원문: “Before each repetition, subjects were asked to perform a deep squat in order to temporally synchronize the beginning of the test among all the sensors worn by the participant.”
- 동일한 하부 다리에 착용된 두 IMU의 기준 좌표계를 동일하게 보정하기 위해 Seel 등의 가상 회전 보정 방식을 적용하였다. *(근거: PAGE 5, 2.3. Preliminary Data Processing)*
	- 근거 원문: “Also, in order to have the same reference system for both IMUs worn on the same leg, the method proposed by Seel et al. \[26\] has been adopted to virtually rotate along the horizontal axis the raw inertial data recorded by the sensors worn on the shank.”
- 수집된 3차원 가속도, 자이로스코프, 저크 신호 등의 통계적 분석 및 주파수 영역 특징 분석을 통해 총 250개의 무브먼트 특징들을 추출하였다. *(근거: PAGE 6, 2.4. Feature Extraction)*
	- 근거 원문: “From the data collected for each repetition, a number of features were extrapolated. The signals considered for feature extraction were the 3D angular rate, the magnitude of the 3D acceleration, the 3D jerk signal obtained from differentiation of the 3D acceleration, and the 3D acceleration in the body‐frame and gravity‐frame.”
- 머신러닝 모델의 일반화 능력을 정확하게 추정하기 위해 훈련 데이터에 포함되지 않은 미지의 사용자에 대한 성능을 추정하는 대상자 배제 교차 검증(LOSO-CV)을 사용하여 학습과 평가를 진행하였다. *(근거: PAGE 1, Abstract)*
	- 근거 원문: “Feature selection was implemented in the learning model, and leave-one-subject-out cross-validation (LOSO-CV) was adopted to estimate training and test errors.”

## 핵심 결과

- Swing phase, relative stance phase, relative swing phase 등 일부 보행 파라미터는 조건(건강군/수술군)과 하지(좌/우) 간의 상호작용 및 주효과 모두에서 통계적으로 유의미한 차이를 보이지 않았다. *(근거: PAGE 8, 3.1. Gait Analysis Results)*
	- 근거 원문: “Some gait parameters (swing phase, relative stance phase, and relative swing phase) do not show a statistically significant interaction between condition and limb, and likewise, do not show the statistical significance of the main effects.”
- 수술 여부를 판별하기 위한 기계학습 모델 중 다층 퍼셉트론(MLP)이 테스트 데이터셋에서 분류 정확도 73.07%로 가장 높은 결과를 나타냈다. *(근거: PAGE 10, 3.2. Machine Learning Model Results)*
	- 근거 원문: “The MLP model shows an accuracy of 73.07% (SE: 8.99%), sensitivity 78.01%, specificity 68.3%, precision 70.79%, F1-score 74.22%, and Cohen’s Kappa 0.462.”
- 그래디언트 부스팅(XGB) 모델은 분류 정확도 72.32%를 달성했으며, 민감도 81.8%를 나타내어 위음성을 낮추기 위한 측면에서 가장 유리한 성능을 나타냈다. *(근거: PAGE 10, 3.2. Machine Learning Model Results)*
	- 근거 원문: “TheXGB model shows an accuracy of 72.32% (SE: 10.47%), sensitivity 81.8%, specificity 63.07%, precision 68.56%, F1-score 74.6%, and Cohen’s Kappa 0.448.”

## 저자 결론

- 본 연구의 결과는 부상 후 5\~10년이 지나 이미 성공적으로 스포츠에 복귀하고 정상으로 판단된 선수일지라도, 웨어러블 센서와 머신러닝 기법을 결합하여 필드 활동 시의 잔존하는 ACL 재건 무릎의 특이 보행 패턴을 성공적으로 구별해낼 수 있음을 입증한다. *(근거: PAGE 13, 4. Discussion)*
	- 근거 원문: “The results of this study clearly show that motion sensors can distinguish between players with ACL-reconstructed knee and healthy players even after 5–10 years following the injury, despite the previously injured athletes being deemed fully recovered.”

## 연구의 한계

- 참가자군이 모두 남성이고 표본 수가 적기 때문에 모델 결과의 일반화 능력이 제한될 수 있으며, 향후 더 많은 대상군을 통한 검증이 요구된다. *(근거: PAGE 14, 4. Discussion)*
	- 근거 원문: “Moreover, gender and small sample size are other limitations of the study which may limit the generalizability of the results. Given the novelty of the study, the present investigation was designed as a pilot proof-of-concept; larger cohort will need to be recruited in the future to confirm those results as shown by the power analysis.”

## 생각해볼 내용

- 비록 통계적 유의성이 검출된 보행 파라미터들이 있으나 표본 크기가 작고 효과 크기가 매우 작기 때문에 실제 집단 간 차이 효과인지 신뢰하기 어려우며 더 큰 표본 규모의 검증이 필요하다. *(근거: PAGE 12, 4. Discussion)*
	- 근거 원문: “Therefore, even though some statistical significance was detected in the analysis, the small observed power and effect size do not provide enough confidence that the difference seen between groups for those variables was a real observed effect and, as a result, further larger studies should be performed.”

## 이 연구가 지적한 선행연구의 문제점

- 현재 임상에서 흔히 사용하는 주관적 및 객관적 스포츠 복귀(RTS) 평가와 실제 복귀 성공률 간의 관계에 대한 임상적 증거가 부족하며, 기존 복귀 평가의 민감성에 의문이 제기된다. *(근거: PAGE 2, 1. Introduction)*
	- 근거 원문: “At present, there is a dearth of evidence supporting the relationship between RTS and standard subjective and objective assessments, which questions if existing RTS assessments and criteria are sensitive or demanding enough to elucidate clinically relevant indicators.”
- Vicon과 같은 기존 마커 기반 모션 캡처 시스템은 고비용, 전문 분석실 접근성 제한 및 대규모 인원 적용의 한계로 인해 현장 활용에 어려움이 많다. *(근거: PAGE 2, 1. Introduction)*
	- 근거 원문: “While marker-based motion analysis systems (e.g., Vicon) \[12\] can provide objective assessments and represent the gold-standard technology adopted in gait analysis for quantitative movement analysis, their adoption is constrained by cost, access to specialist motion labs, as well as the practicality of application for larger patient/subject groups and, thus, shows limited use for on-the-field players.”
- 웨어러블 센서를 활용한 선행 ACL 연구가 일부 존재하나, 대부분 실제 환경이 아닌 실내 실험실 트레드밀 환경에서 이루어졌으며 수술 후의 경과 시간 정보 등이 누락되어 있었다. *(근거: PAGE 3, 1. Introduction)*
	- 근거 원문: “However, the tests were carried out in a lab setting and the time since surgery was not provided.”

## 이 연구의 해결 방식과 기여

- **[AS-IS]** 성공적으로 복귀한 지 5\~10년이 경과한 건강한 선수와 ACL 재건 환자를 필드(야외) 환경에서 웨어러블 센서와 데이터 기반 기계학습 모델의 결합을 통해 정밀하게 분류해낸 연구는 본 연구가 최초이다. *(근거: PAGE 3, 1. Introduction)*<br>**[TO-BE]** 저자들이 알기로, 스포츠에 복귀한 지 5\~10년이 지난 ACL 재건 선수와 건강한 선수를 필드 환경에서 관성 센서와 데이터 기반 접근법을 결합해 분류한 연구는 아직 충분히 탐구되지 않았다.<br>*(사실검증 — 과장/경미: 원문은 저자들이 알기로 해당 조합이 아직 탐구되지 않았다고 표현한다. 요약의 '본 연구가 최초'와 '정밀하게 분류'는 원문보다 단정적이고 강하다.)*
	- 근거 원문: “To the best of the authors’ knowledge, the combination of a data-driven approach and inertial sensors to classify healthy and ACL-reconstructed subjects on-the-field (with post-ACL athletes returned to sport and with time from surgery between five and 10 years) is not yet explored.”

## 레퍼런스할 수 있는 내용

### 1. ACL 재건 환자의 실제 스포츠 복귀율 및 경쟁 수준 복귀율 차이

- 원문 발췌: “However, on average, 80% of patients were found to return to sport, with only 55% returning to competitive levels after ACL reconstruction \[4\].”
- 한국어 번역: 그러나 평균적으로 80%의 환자가 스포츠로 복귀하지만, ACL 재건술 후 경쟁력 있는 수준으로 복귀하는 환자는 55%에 불과하다.
- 원문 위치: PAGE 2, 1. Introduction
- 원문 내 인용표기: \[4\]
- 해당 선행문헌: 4. Ardern, C.L.; Webster, K.E.; Taylor, N.F.; Feller, J.A. Return to sport following anterior cruciate ligament reconstruction surgery: A systematic review and meta-analysis of the state of play. Br. J. Sports Med. 2011, 45, 596–606. \[CrossRef\] \[PubMed\]
- 주장 유형: background_citation
- 활용 맥락과 주의: ACL 수술 이후 전체적인 스포츠 복귀와 실제 고수준 경쟁 복귀 간의 간극을 설명할 때 인용 근거로 활용할 수 있다. Ardern 등(2011)의 메타분석 연구에서 도출된 수치로 2차 인용에 주의해야 한다.

### 2. 필드 스포츠 활동 중 관성 센서 기반 ACL 재건 환자와 대조군 분류 타당성

- 원문 발췌: “The results of this study clearly show that motion sensors can distinguish between players with ACL-reconstructed knee and healthy players even after 5–10 years following the injury, despite the previously injured athletes being deemed fully recovered.”
- 한국어 번역: 본 연구 결과는 이전에 부상당한 선수들이 임상적으로 완전히 회복된 것으로 간주되었음에도 불구하고, 부상 후 5\~10년이 지난 시점의 현장 스포츠 활동에서 동작 센서가 ACL 재건 무릎 환자와 건강한 선수를 판별할 수 있음을 명확히 보여준다.
- 원문 위치: PAGE 13, 4. Discussion
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- **[AS-IS]** 활용 맥락과 주의: 수술 후 장기 시점에서도 가혹한 필드 방향 전환 테스트 중 웨어러블 센서를 통해 ACL 재건군과 대조군 사이의 잔존 보행 차이를 기계학습으로 성공적으로 식별할 수 있다는 본 논문의 자체 결과를 인용할 때 사용한다.<br>**[TO-BE]** 활용 맥락과 주의: 수술 후 5\~10년이 지난 선수의 필드 방향 전환 과제에서 웨어러블 센서와 머신러닝 접근법이 ACL 재건군의 보행 패턴 식별에 활용될 가능성을 보였다는 본 논문의 자체 결과를 인용할 때 사용한다.<br>*(사실검증 — 과장/경미: 원문은 필드 스포츠 과제에서 센서와 머신러닝 접근의 feasibility와 구별 가능성을 제시하지만, '가혹한' 테스트라는 표현은 원문 표현보다 강하며, '성공적으로 식별'도 73.07% 정확도와 81.8% 민감도의 제한적 결과를 충분히 반영하지 않는다.)*


---

# [38] Optimizing wearable IMU configurations for running gait analysis: a machine learning-based sensor fusion approach

(저자: Ye Yuan, Yaohui Yu, Shanshan Cai, Weidong Cheng | 연도: 2026 | 저널: Frontiers in Bioengineering and Biotechnology | DOI: https://doi.org/10.3389/fbioe.2026.1762919)

Yuan, Y., Yu, Y., Cai, S., & Cheng, W. (2026). Optimizing wearable IMU configurations for running gait analysis: a machine learning-based sensor fusion approach. Frontiers in Bioengineering and Biotechnology, 14, 1762919. https://doi.org/10.3389/fbioe.2026.1762919

## 서지정보

- 저자: Ye Yuan, Yaohui Yu, Shanshan Cai, Weidong Cheng
- 연도: 2026
- 저널: Frontiers in Bioengineering and Biotechnology
- DOI: 10.3389/fbioe.2026.1762919
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/04_machine_learning_and_deep_learning/Optimizing wearable IMU configurations for running gait analysis - a machine learning-based sensor fusion approach.pdf
- 분석 provider: antigravity

## 연구 목적

- 본 연구는 고차원 17개 센서 네트워크를 측정 정확도 저하 없이 최소화-최적화된 하위 집합으로 줄일 수 있는지 타당성을 결정하기 위해 머신러닝 기법을 적용한다. *(근거: PAGE 1, Objective)*
	- 근거 원문: “This study applies machine learning (ML) techniques to address this hardware limitation by determining the feasibility of reducing a high-dimensional 17-sensor network to a “minimal-optimal” subset without compromising measurement accuracy.”
- 본 연구의 주된 목적은 골드 스탠다드인 17개 센서 네트워크에 대해 축소된 IMU 구성(1-3개 센서)의 파라미터 추정 성능을 체계적으로 평가하는 것이다. *(근거: PAGE 2, 1 Introduction)*
	- 근거 원문: “Therefore,theprimaryobjectiveofthispaperistosystematically evaluate the parameter estimation performance of reduced IMU configurations (1–3 sensors) against a gold-standard 17-sensor network.”

## 연구 설계와 대상

- 지역 달리기 클럽과 소셜 미디어 광고를 통해 25명의 여가용 러너(남성 15명, 여성 10명)를 모집했다. *(근거: PAGE 2, 2.1 Participants)*
	- 근거 원문: “We recruited twenty-five recreational runners (15 male, 10 female) through local running clubs and social media advertisements.”
- 참가자 선정 기준은 18\~45세 연령, 지난 1년간 주당 최소 15km 달리기, 12km/h 속도로 5분 이상 지속 달리기 가능, 심혈관 및 신경계 질환이 없는 것이었다. *(근거: PAGE 2, 2.1 Participants)*
	- 근거 원문: “Inclusion criteria for participation were: (1) age between 18 and 45 years; (2) an average weekly running volume of at least 15 km over the past year; (3) the ability to run continuously at 12 km/h for at least 5 min; and (4) no history of cardiovascular or neurological diseases.”
- 참가자들은 느린 페이스, 중간 페이스, 템포 페이스를 대표하는 8, 10, 12 km/h의 고정된 속도에서 각각 3분씩 세 차례 러닝 트라이얼을 수행했다. *(근거: PAGE 3, 2.3 Experimental protocol)*
	- 근거 원문: “The mainprotocolconsistedofthree3-minrunningtrialsatfixed speeds of 8 km/h, 10 km/h, and 12 km/h, representing slow, medium, and tempo paces.”

## 방법

- 골드 스탠다드로 Xsens MVN Awinda 관성 모션 캡처 시스템을 사용했으며, 100 Hz로 샘플링하는 17개의 무선 IMU로 구성되었다. *(근거: PAGE 3, 2.2 Experimental equipment and setup)*
	- 근거 원문: “We used the Xsens MVN Awinda inertial motion capture system (Xsens Technologies B.V., Netherlands) as the gold standard. This system comprises 17 wireless IMUs (MTw2) sampling at 100 Hz.”
- 원시 IMU 신호는 고주파 시계열 데이터를 회귀에 적합한 피처 공간으로 변환하기 위해 50% 중첩되는 250 ms 윈도우 크기의 슬라이딩 윈도우 방식을 사용하여 처리되었다. *(근거: PAGE 3, 2.5.1 Feature engineering)*
	- 근거 원문: “Raw IMU signals (3-axis acceleration and 3-axis angular velocity) were processed using a sliding-window approach to transform high- frequency time-series data into a feature space suitable for regression (Figure 2). Signals were segmented into 250 ms windows with a 50% overlap.”
- 기준 모델인 선형 회귀(LR) 모델 및 딥러닝(LSTM) 신경망 모델과 비교하여 랜덤 포레스트(RF) 모델의 성능을 벤치마킹했다. *(근거: PAGE 4, 2.5.2 Model selection and training)*
	- 근거 원문: “We benchmarked the RF model against a baseline Linear Regression (LR) model and a Long Short-Term Memory (LSTM) neural network.”
- 데이터 누수를 방지하고 모델 일반화를 확보하기 위해, 훈련 세트(20명, 80%)와 보류 테스트 세트(5명, 20%)로 데이터셋을 분할하는 대상자 독립적 검증 방식을 채택했다. *(근거: PAGE 5, 2.5.3 Validation strategy and statistical analysis)*
	- 근거 원문: “To ensure model generalization and prevent data leakage, we employed a strict subject-independent validation. The dataset was randomly split into a training set (20 participants, 80%) and a hold- out test set (5 participants, 20%).”

## 핵심 결과

- 단일 요추센서(Lumbosacral IMU) 구성은 Cadence, Vertical Oscillation, Ground Contact Time 등의 전역 파라미터를 고정밀도로 재구성할 수 있었으나 보행 비대칭성 검지에는 실패했다. *(근거: PAGE 1, Results)*
	- 근거 원문: “Analysis revealed that a single lumbosacral IMU could successfully reconstruct global parameters (Cadence, Vertical Oscillation, Ground Contact Time) with high precision (R2 \>0.95,MAPE\<5%), outperforming standard commercial benchmarks. However, this single-node setup failed to detect gait asymmetry (R2 0.52).”
- 요추와 양측 발목 센서를 결합한 분산형 3-센서 융합 구성은 단일 노드 구성의 대칭성 감지 한계를 해결하여 모든 파라미터에서 전신 시스템과 필적하는 성능을 보여주었다. *(근거: PAGE 1, Results)*
	- 근거 원문: “A distributed three-sensor fusion configuration (Lumbosacral + Bilateral Ankles) resolved this limitation, achieving results comparable to the full-body system for all parameters (R2 \>0.91,MAPE  7.12%).”
- 전체 피처 세트에서 랜덤 포레스트(RF) 모델은 GCT에 대해 R2 성능을 선형 회귀(LR) 대비 0.15 이상 향상시켜 게이트 역학의 비선형적 특성을 확인했다. *(근거: PAGE 5, 3.2 Predictive performance of IMU configurations)*
	- 근거 원문: “On the full feature set, the RF model significantly outperformed a baseline Linear Regression model (R2 improvement of \>0.15 for GCT), confirming the non-linear nature of gait dynamics.”

## 저자 결론

- 본 연구는 센서 어레이 설계를 최적화하기 위한 머신러닝 프레임워크를 검증하고 있으며, 제안된 3-센서 융합은 차세대 웨어러블 디바이스를 위한 견고하고 저비용의 아키텍처 청사진을 제공한다. *(근거: PAGE 1, Conclusion)*
	- 근거 원문: “This study validates a machine learning framework for optimizing sensor array design. The proposed three-sensor fusion offers a robust, low-cost architectural blueprint for next-generation wearable devices, proving that complex deep learning is not always required when sensor placement is biomechanically optimized.”
- 단일 요추 장착형 IMU를 머신러닝 모델과 결합했을 때 복잡한 다중 센서 설정 없이도 주요 러닝 게이트 파라미터를 정확하게 예측할 수 있는 뛰어난 효용성이 입증되었다. *(근거: PAGE 5, 4.1 Principal findings and interpretation)*
	- 근거 원문: “The central finding of this study is the remarkable efficacy of a single, lumbosacral-mounted IMU when combined with a machine learning model. Our results compellingly demonstrate that it is possible to accurately predict key running gait parameters without resorting to a complex “Christmas tree” sensor setup.”

## 연구의 한계

- 본 연구는 표면이 균일하고 평평한 트레드밀에서 진행되었으며, 이는 지면 변동성과 공기 저항이 존재하는 야외 실외 달리기와 비교할 때 보행 역학이 약간 다를 수 있다는 한계가 있다. *(근거: PAGE 9, 4.5 Limitations and future directions)*
	- 근거 원문: “First, this study was conducted on a treadmill, which provides a homogenous, flat surface. We acknowledge that treadmill running lacks the surface variability and air resistance of overground running, and gait mechanics may differ slightly (Van Hooren et al., 2020).”
- 검증된 속도 범위(8-12 km/h)는 일정한 정상 상태의 지구력 달리기를 대표하므로, 보행 역학이 근본적으로 변하는 단거리 전력 질주나 고강도 인터벌 러닝(\>15 km/h)에 직접 외삽하여 적용할 수 없다. *(근거: PAGE 10, 4.5 Limitations and future directions)*
	- 근거 원문: “Second, the validated speed range (8–12 km/h) represents steady-state endurance running. It is crucial to note that the proposed “minimal-optimal” configurations cannot be directly extrapolated to sprinting or high-intensity interval running (\>15 km/h).”
- 최고 시험 속도(12 km/h)에서 오차가 약간 증가하는 것을 관찰했는데, 이는 시간 분해능 부족보다는 고충격 착지 시 피부 변형으로 인해 센서가 뼈에 대해 상대적으로 움직이는 연조직 아티팩트(STA)에 기인한다. *(근거: PAGE 10, 4.5 Limitations and future directions)*
	- 근거 원문: “However, at the highest tested speed (12 km/h), we observed a marginal increase in error. This is likely attributable to Soft Tissue Artifacts (STA)—the secondary motion of the sensor relative to the bone caused by skin deformation during high-impact landing—rather than the temporal resolution itself.”

## 생각해볼 내용

- 저자는 3-센서 구성이 가격과 편의성 면에서 최적의 대안이 될 수 있음을 시사한다. *(근거: PAGE 1, Conclusion)*
	- 근거 원문: “The proposed three-sensor fusion offers a robust, low-cost architectural blueprint for next-generation wearable devices, proving that complex deep learning is not always required when sensor placement is biomechanically optimized.” \> **[AS-IS]** - 단일 센서 구성은 보행 비대칭성 감지에 치명적인 한계가 있어 임상적으로 유용한 비대칭성 평가를 위해서는 3-센서 구성이 필수적이다. *(근거: PAGE 1, Results)* \> \> **[TO-BE]** 단일 센서 구성은 보행 비대칭성 감지에 한계가 있었고, 요추와 양측 발목을 결합한 3-센서 구성이 이 한계를 해결했다. \> \> *(사실검증 — 과장/경미: 원문은 단일 센서가 gait asymmetry를 감지하지 못했고 3센서 구성이 이를 해결했다고 설명하지만, '치명적인 한계'나 '임상적으로 유용한 평가를 위해 필수적'이라는 표현은 원문보다 강한 단정이다.)*
	- 근거 원문: “However, this single-node setup failed to detect gait asymmetry (R2 0.52). A distributed three-sensor fusion configuration (Lumbosacral + Bilateral Ankles) resolved this limitation, achieving results comparable to the full-body system for all parameters”
- 임베디드 기기 구현 시 랜덤 포레스트는 LSTM 등 딥러닝 모델 대비 낮은 연산 복잡도와 메모리 요구량 덕분에 유리하다. *(근거: PAGE 9, 4.4 System feasibility and embedded implementation)*
	- 근거 원문: “Unlike Deep Neural Networks (DNNs) or LSTMs, which require computationally expensive matrix multiplications and substantial RAM for activation maps, the RF inference process consists of a series of simple conditional checks (if-else statements).”

## 이 연구가 지적한 선행연구의 문제점

- 연구실 기반의 3D 광학 모션 캡처 시스템은 정밀하지만 가격이 대단히 비싸고 통제된 실험실 환경에 국한되며 특수 전문 지식이 필요하다. *(근거: PAGE 2, 1 Introduction)*
	- 근거 원문: “They are prohibitively expensive, confined to controlled laboratory environments, and require highly specialized expertise for data collection and processing.”
- 전신 운동학을 복원하기 위해 다수의 센서 어레이(예: 17개)를 부착하는 방식은 비용 및 복잡성을 높이고 긴 셋업 시간을 요구하며 사용자의 자연스러운 보행을 방해할 수 있다. *(근거: PAGE 2, 1 Introduction)*
	- 근거 원문: “This “Christmas tree” effect, while feasible for research, presents significant practical barriers: it is still costly and complex, places a heavy time burden on the user for setup (often 15–30 min), and negatively impacts user comfort, which may even alter the natural gait being measured (Caldas et al., 2017).”

## 이 연구의 해결 방식과 기여
 \> **[AS-IS]** - 머신러닝을 활용하여 신체의 중요 노드(질량 중심 및 말단 효과기)에서 수집한 데이터의 정보 중복성을 디코딩함으로써 부족한 하드웨어를 가상화하는 방법을 제안한다. *(근거: PAGE 2, 1 Introduction)*<br>**[TO-BE]** 머신러닝을 활용해 질량 중심 및 말단 효과기 등 중요 노드에서 얻은 최소 입력과 전신 시스템 출력 간의 비선형 매핑을 학습함으로써 누락된 센서를 효과적으로 '가상화'하는 접근을 제안한다.<br>*(사실검증 — 근거불충분/경미: 제시된 근거 원문은 정보 중복성과 중요 노드의 잠재 특징이 핵심 시공간 보행 스칼라를 추정하기에 충분하다는 가설을 말하지만, '디코딩' 및 '부족한 하드웨어를 가상화'한다는 부분은 이어지는 문장에서 확인된다. 현재 evidence_quote만으로는 요약 문장 전체를 충분히 지지하지 못한다.)*
	- 근거 원문: “This study proposes that machine learning (ML) is the ideal tool to address this hardware-accuracy dilemma. From a signal processing perspective, human locomotion involves highly coordinated kinematic chains, implying significant information redundancy across different body segments. We hypothesize that data acquired from critical nodes—specifically the Center of Mass (CoM) and end-effectors—contain sufficient latent features to estimate the key spatio-temporal gait scalars of the system.”
- 제안된 3-센서 구성(요추 + 양측 발목)은 요추 센서의 전역적 파라미터 검출 성능과 발목 센서의 시간적/비대칭성 검출 성능을 효과적으로 결합하여 골드 스탠다드 수준의 신뢰성을 보여주었다. *(근거: PAGE 7, 4.1 Principal findings and interpretation)*
	- 근거 원문: “It successfully combines the global-parameter strength of the lumbar sensor with the temporal and asymmetry-detecting strengths of the ankle sensors, achieving high performance (R2 \>0.91) across all measured parameters.”

## 레퍼런스할 수 있는 내용

### 1. 러닝 활동과 심혈관 건강 및 정신 웰빙의 긍정적인 관계

- 원문 발췌: “Running is one of the most popular and accessible forms of physical activity worldwide, offering significant benefits for cardiovascular health, mental wellbeing, and overall longevity (Lee et al., 2014).”
- 한국어 번역: 달리기는 전 세계적으로 가장 인기 있고 접근하기 쉬운 신체 활동 중 하나로, 심혈관 건강, 정신적 웰빙 및 전반적인 수명 연장에 상당한 이점을 제공한다 (Lee et al., 2014).
- 원문 위치: PAGE 2, 1 Introduction
- 원문 내 인용표기: (Lee et al., 2014)
- 해당 선행문헌: Lee,D.C.,Pate,R.R.,Lavie,C.J.,Sui,X.,Church,T.S.,andBlair,S.N.(2014).Leisure- timerunningreducesall-causeandcardiovascularmortalityrisk.J.Am.Coll.Cardiol.64 (5), 472–481. doi:10.1016/j.jacc.2014.04.058
- 주장 유형: background_citation
- 활용 맥락과 주의: 달리기가 건강 및 수명에 미치는 긍정적 기여를 설명할 때 선행 문헌의 근거로 인용할 수 있음. 2차 인용에 주의해야 함.

### 2. 러닝 부상 위험과 연관된 이상 보행 분석 파라미터

- 원문 발췌: “For instance, excessive vertical impact forces, high vertical oscillation (VO), prolonged ground contact time (GCT), and excessive pronation are considered key risk factors (Hreljac, 2004; Davis and Powers, 2010).”
- 한국어 번역: 예를 들어, 과도한 수직 충격력, 높은 수직 진동(VO), 연장된 지면 접촉 시간(GCT), 과도한 회내는 주요 위험 요소로 간주된다 (Hreljac, 2004; Davis and Powers, 2010).
- 원문 위치: PAGE 2, 1 Introduction
- 원문 내 인용표기: (Hreljac, 2004; Davis and Powers, 2010) \> **[AS-IS]** - 해당 선행문헌: Hreljac, A. (2004). Impact and overuse injuries in runners. Med. and Sci. Sports and Exerc. 36 (5), 845–849. doi:10.1249/01.mss.0000126803.66636.dd<br>**[TO-BE]** 해당 선행문헌은 Hreljac (2004)와 Davis and Powers (2010)를 모두 포함해야 한다.<br>*(사실검증 — 인용표기오류/중대: 요약의 원문 내 인용표기는 Hreljac 2004와 Davis and Powers 2010 두 문헌인데, 해당 선행문헌에는 Hreljac 2004만 제시되어 Davis and Powers 2010이 누락되었다.)*
- 주장 유형: background_citation
- 활용 맥락과 주의: 러닝 관련 부상(RRIs)의 위험 요인이 되는 생체역학적 보행 지표들을 뒷받침하기 위해 인용 가능함. 2차 인용에 주의해야 함.

### 3. 발목 위치 IMU 센서의 보행 시간적 매개변수 측정의 탁월성

- 원문 발췌: “The ankle sensors (Config 2) were superior for temporal metrics because their signals provide unambiguous, high-amplitude spikes and reversals corresponding to the discrete events of initial contact (IC) and toe-off (TO) (Aminian et al., 2002).”
- 한국어 번역: 발목 센서(Config 2)는 신호가 초기 접촉(IC) 및 발가락 떼기(TO)의 개별 이벤트에 해당하는 명확하고 높은 진폭의 스파이크 및 반전을 제공하기 때문에 시간적 측정 기준에 우수했다 (Aminian et al., 2002).
- 원문 위치: PAGE 8, 4.2 Biomechanical interpretation and model trust
- 원문 내 인용표기: (Aminian et al., 2002)
- 해당 선행문헌: Aminian, K., Najafi, B., Büla, C., Leyvraz, P. F., and Robert, P. (2002). Spatio- temporal parameters of gait measured by an ambulatory system using miniature gyroscopes. J. Biomechanics 35 (5), 689–699. doi:10.1016/s0021-9290(02) 00008-8
- 주장 유형: background_citation
- 활용 맥락과 주의: 달리기 보행 분석에서 발목 위치 센서가 왜 접촉 시간과 같은 시간적 지표 검출에 유리한지 기하학적/생체역학적 배경 설명 시 인용 가능함. 2차 인용에 주의해야 함.


---

# [39] Transfer Learning of Human Activities Based on IMU Sensors: A Review

(저자: Sara Ashry , Supratim Das, Mahdie Rafiei , Jan Baumbach , and Linda Baumbach | 연도: 2025 | 저널: IEEE SENSORS JOURNAL | DOI: https://doi.org/10.1109/JSEN.2024.3510097)

Ashry, S., Das, S., Rafiei, M., Baumbach, J., & Baumbach, L. (2025). Transfer Learning of Human Activities Based on IMU Sensors: A Review. IEEE Sensors Journal, 25(3), 4115-4126. https://doi.org/10.1109/JSEN.2024.3510097

## 서지정보

- 저자: Sara Ashry , Supratim Das, Mahdie Rafiei , Jan Baumbach , and Linda Baumbach
- 연도: 2025
- 저널: IEEE SENSORS JOURNAL
- DOI: 10.1109/JSEN.2024.3510097
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/04_machine_learning_and_deep_learning/Transfer Learning of Human Activities Based on IMU Sensors - A Review.pdf
- 분석 provider: antigravity

## 연구 목적

- IMU 센서 데이터를 이용한 인간 행동 인식(HAR)에 적용된 전이 학습(TL) 방법론들을 종합적으로 검토하고 분석하여 연구자와 개발자에게 종합적인 자원을 제공하는 것이다. *(근거: Page 1, Abstract)*
	- 근거 원문: “Our objective is to provide a comprehensive resource for researchers and developers by summarizing the existing activities, feature extractions, and TL techniques in the related studies.”

## 연구 설계와 대상

- PubMed, ACM, Scopus 데이터베이스에서 447개의 연구를 검색하고, 그 중 최종적으로 포함 기준을 충족하는 33개의 핵심 연구를 선정하여 분석하였다. *(근거: Page 1, Abstract)*
	- 근거 원문: “We analyzed 447 studies from PubMed, ACM, and Scopus datasets, of which we ultimately selected 33 pivotal studies that met our inclusion criteria.”

## 방법

- 본 체계적 문헌고찰은 PRISMA 가이드라인을 따르며, 연구 질문 공식화, 검색 쿼리 설정, 문헌 선택을 위한 선정 및 제외 기준 구체화를 포함한다. *(근거: Page 1, Abstract)*
	- 근거 원문: “Our methodology follows the structure of the preferred reporting items for systematic reviews and meta-analysis (PRISMA) statement by formulating precise research questions, establishing search queries, and specifying inclusion and exclusion criteria for study selection.”
- 문헌 검색은 2023년 9월 14일에 첫 번째 저자에 의해 수행되었다. *(근거: Page 4, Section II-D)*
	- 근거 원문: “The search was performed by the first author of this publication on September 14, 2023.”

## 핵심 결과

- 전이 학습은 사전 학습된 모델을 재사용함으로써 인간 행동 인식(HAR)의 성능을 향상시켰다. *(근거: Page 1, Abstract)*
	- 근거 원문: “Overall, we found that TL has enhanced HAR performance by reusing pretrained models.”

## 저자 결론

- 전이 학습을 적용할 때 부작용을 피하기 위해서는 관련된 전이 정보를 신중하게 선택하는 것이 중요하다. *(근거: Page 1, Abstract)*
	- 근거 원문: “However, it is important to carefully select relevant transfer information to avoid any potential adverse effects.”
- 활동의 특성에 알고리즘을 맞추는 것이 필수적인데, 걷기 같은 일상적 활동에는 AlexNet과 같은 단순한 모델이 적합한 반면, 청소 같은 정밀한 작업에는 DenseNet과 같은 더 복잡한 모델이 적합하다. *(근거: Page 1, Abstract)*
	- 근거 원문: “Aligning algorithms with activity nature is essential—simpler models like AlexNet are suitable for routine activities such as walking, while more complex models like DenseNet are better for intricate tasks like cleaning.”

## 연구의 한계

- 분석 대상 문헌을 영어로 출판된 연구로만 제한하여 분석 결과에 편향이 발생했을 가능성이 있다. *(근거: Page 9, Section V)*
	- 근거 원문: “First, we only included studies published in English, which could have a biased impact on the analysis.”
- 데이터 추출 과정이 첫 번째 저자(Sara Ashry) 1인에 의해서만 단독으로 수행되어 미세한 오류가 발생했을 가능성이 존재한다. *(근거: Page 10, Section V)*
	- 근거 원문: “the data extraction from the included articles was solely performed by one author (Sara Ashry), potentially led to minor errors.”

## 생각해볼 내용

- 실시간 모바일 애플리케이션의 효율성을 고려할 때, 연산 효율성이 높은 MobileNet과 같이 모바일 플랫폼 및 간단한 활동에 잘 적합한 경량화 모델을 사용하는 것을 권장한다. *(근거: Page 9, Section IV)*
	- 근거 원문: “For efficiency considerations in real-time applications, we suggest using lightweight models like MobileNet, which are well-suited for mobile platforms and simple activities due to their computational efficiency.”

## 이 연구가 지적한 선행연구의 문제점

- IMU 센서와 전이 학습을 사용하여 개인 위생 및 노인 돌봄과 같은 구체적인 행동을 평가하는 연구가 현재 부족한 실정이다. *(근거: Page 1, Abstract)*
	- 근거 원문: “We conclude that there is a lack of studies assessing specific activities, such as personal hygiene and elder care, using IMU sensors and TL.”
- **[AS-IS]** 딥러닝은 검증 데이터와 동일한 분포를 가진 대량의 레이블링된 학습 데이터가 필요하지만, 실세계 시나리오에서는 충분한 학습 데이터를 수집하는 것이 비용이 많이 들고 시간이 오래 걸리며 때로는 불가능하다. *(근거: Page 2, Section I)*<br>**[TO-BE]** 딥러닝은 검증 데이터와 동일한 분포를 가진 대량의 레이블링된 학습 데이터가 있을 때 특히 잘 작동하지만, 실세계 시나리오에서는 충분한 학습 데이터를 수집하는 것이 비용이 많이 들고 시간이 오래 걸리며 때로는 불가능하다.<br>*(사실검증 — 과장/경미: 원문은 딥러닝이 동일 분포의 대량 레이블 학습 데이터가 있을 때 이상적으로 잘 작동한다고 설명한다. 요약의 '필요하지만'은 원문보다 더 강한 필수 조건처럼 읽힐 수 있다.)*
	- 근거 원문: “Ideally, deep learning thrives when there is an abundance of labeled training data sharing the same distribution as the test data. However, in numerous scenarios, gathering sufficient real-world training data proves to be costly, time-consuming, or even unfeasible.”

## 이 연구의 해결 방식과 기여

- 33개의 연구를 바탕으로 IMU 센서 기반 인간 행동 인식(HAR)을 위한 전이 학습 접근법을 체계적으로 요약하여, 연구자가 실무에서 적절한 전이 학습 방안을 선택하는 데 기여한다. *(근거: Page 3, Section I)*
	- 근거 원문: “We systematically summarize TL approaches for HAR based on IMU sensors for 33 studies, which may support researchers in selecting appropriate TL approaches in practice.”

## 레퍼런스할 수 있는 내용

### 1. 헬스케어 및 낙상 감지 분야에서의 HAR의 역할

- 원문 발췌: “The importance of HAR plays a pivotal role in monitoring and understanding human behavior; among various applications, such as in healthcare, it is used to monitor patient activity, aiding in remote patient care and fall detection \[2\].”
- 한국어 번역: HAR의 중요성은 인간의 행동을 모니터링하고 이해하는 데 있어 중추적인 역할을 하며, 헬스케어와 같은 다양한 애플리케이션 중에서 환자 활동을 모니터링하여 원격 환자 케어 및 낙상 감지를 돕는 데 사용된다.
- 원문 위치: Page 2, Section I
- 원문 내 인용표기: \[2\]
- 해당 선행문헌: \[2\] F. Serpush, M. B. Menhaj, B. Masoumi, and B. Karasfi, “Wearable sensor-based human activity recognition in the smart healthcare system,” Comput. Intell. Neurosci., vol. 2022, pp. 1–31, Feb. 2022.
- 주장 유형: background_citation
- 활용 맥락과 주의: 헬스케어 분야에서 HAR이 낙상 감지 및 환자 상태 모니터링에 쓰이는 근거로 사용 가능하며, 2차 인용 시 \[2\] 논문을 참조해야 함.


---

