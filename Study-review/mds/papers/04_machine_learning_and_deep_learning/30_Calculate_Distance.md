# Wearable IMU sensor-based motion tracking for post-stroke motor impairment assessment during activities of daily living

Lewko, T. (2024). Wearable IMU sensor-based motion tracking for post-stroke motor impairment assessment during activities of daily living (Master's thesis, Harvard University).

## 서지정보

- 저자: Tanguy Lewko
- 연도: 2024
- 저널: Master's Thesis, Harvard University
- DOI: 확인 불가
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/04_machine_learning_and_deep_learning/Calculate_Distance.pdf
- 분석 provider: antigravity

## 연구 목적

- 일상생활 기능 동작(ADLs) 중 뇌졸중 환자의 상지 움직임을 시간적으로 분할하고 평가하기 위한 웨어러블 IMU 센서 플랫폼의 개발 _(근거: Page 2, Abstract)_
  - 근거 원문: “The objective of this research is thus to develop a wearable IMU sensor platform to segment and evaluate stroke survivors’ upper extremity motions during ADLs.”
- 일상생활 활동을 지속적으로 모니터링하여 환측 상지의 사용 빈도 및 운동 품질에 관한 피드백을 환자에게 제공하고, 이를 통해 자가 재활 프로그램의 순응도를 촉진함 _(근거: Page 2, Abstract)_
  - 근거 원문: “Continuous monitoring of ADLs is a viable solution to provide feedback on the quantity and quality of impaired arm functional movements and to promote adherence to home rehabilitation programs.”

## 연구 설계와 대상

- 제안한 활동 분할 알고리즘의 유효성을 검증하기 위해 통제된 환경에서 5명의 건강한 대조군과 5명의 뇌졸중 환자를 대상으로 비디오 그라운드 트루스를 활용한 자연스러운 ADL 데이터셋을 직접 수집함 _(근거: Page 2, Abstract)_
  - 근거 원문: “To demonstrate the feasibility of our activity segmentation algorithm, we collected a dataset of 5 healthy participants and 5 post-stroke participants. We tested our algorithm on two mildly impaired post-stroke participants, highlighting the potential need for tailored models for post-stroke individuals due to the high variability of impairments within the population.”
- 활동 분할 알고리즘의 초기 개발 및 검증을 위해 10명의 건강한 참가자(오른손잡이, 22~28세)가 6가지 일방 상지 ADL을 수행한 첫 번째 오픈소스 데이터셋(Multi-Sensory dataset)을 활용함 _(근거: Page 2, Abstract)_
  - 근거 원문: “The first dataset contained healthy participants performing six unimanual upper extremity self-care ADLs and was used to develop a deep learning algorithm for precise time segmentation of ADLs.”
- 건강한 대조군과 뇌졸중 환자 간의 운동 분석과 특징 추출을 위해 양치질과 물 마시기 활동 중 상체에 IMU를 장착한 두 번째 오픈소스 데이터셋(StrokeRehab dataset)을 활용함 _(근거: Page 2, Abstract)_
  - 근거 원문: “The second open-source dataset was used to extract samples of healthy and post-stroke participants wearing IMUs on their upper body while drinking and brushing their teeth.”

## 방법

- 합성곱(CNN) 레이어와 양방향 장단기 메모리(BiLSTM) 레이어를 결합하여 IMU 데이터의 시공간 특징을 동시 추출하고, 과거/미래 맥락 정보를 활용하는 33Hz 해상도의 밀집 라벨링(dense labeling) 하이브리드 모델 설계 _(근거: Page 2, Abstract)_
  - 근거 원문: “We developed a hybrid architecture combining convolutional and bidirectional long short-term memory layers to extract spatial and temporal information from the input IMU data. Our algorithm employs dense labeling for time-step resolution segmentation at a frequency of 33 hertz, while integrating past and future knowledge into current predictions through a context-aware framework.”
- 소수 클래스 과소 분류 문제를 예방하기 위해 다수 클래스 샘플 수의 1.5분의 1에 상응하는 크기로 복제하여 가중치를 맞추는 무작위 오버샘플링 적용 _(근거: Page 13, Chapter 3.2)_
  - 근거 원문: “We opt for random oversampling due to the relatively small size of our dataset. We randomly sample the data with replacement until each class is represented by a minimum number of samples equal to the number of samples present in the majority class divided by 1.5, allowing us to balance all the classes we want to segment.”
- 데이터 부족 한계를 해결하고 노이즈 및 장착 편차 강인성을 제고하기 위해 Jittering, Offsetting, Scaling, Zooming, Rotating, Time warping, Down-sampling 등의 데이터 증강 기술 구현 _(근거: Page 20, Chapter 3.3.4)_
  - 근거 원문: “For time series data augmentation, several commonly utilized methods that we implemented include: 1. Jittering
Jittering aims to increase the robustness of the model to disturbances by adding noise to the data.”
- 건강한 참가자의 움직임 대조군 대비 불일치성을 다변량 정규 분포 하의 공분산 구조를 반영해 측정하기 위해 주성분 분석(PCA) 차원 축소와 마할라노비스 거리(Mahalanobis distance) 모델을 융합한 MHS 평가지표 개발 _(근거: Page 2, Abstract)_
  - 근거 원문: “The MHS utilizes the Mahalanobis distance to measure the disparity between the movement of a stroke survivor and the same movement executed by healthy subjects.”

## 핵심 결과

- 양 전완에 부착된 2개의 IMU 센서 데이터(가속도계 및 자이로스코프)만을 사용했을 때 성능이 가장 우수했으며, 대상자 간 평가에서 0.83, 대상자 내 평가에서 0.87의 F1 점수를 각각 확보함 _(근거: Page 2, Abstract)_
  - 근거 원문: “We achieved F1 scores of 0.83 and 0.87 in across-subjects and within-subjects evaluations, respectively. Additionally, we determined that segmentation accuracy was highest when using only the accelerometer and gyroscope data from two IMU sensors attached to the forearms.”
- 동작 품질 분류 척도인 MHS의 임계값 분류 정확도는 양치질에서 76%, 물 마시기 transport 단계에서 82%, recovery 단계에서 84%를 나타내 환자와 대조군을 효과적으로 분별함 _(근거: Page 51, Chapter 7.1.2)_
  - 근거 원문: “The optimal thresholds for brushing teeth, drinking transport, and drinking recovery scores were respectively 9.9, 9.3, and 9.5, leading to accuracies of 0.76, 0.82, and 0.84.”
- 뇌졸중 환자는 양치질 수행 시 건강한 대조군에 비해 더 높은 회전 속도(각속도)를 사용하고, 팔 가속도는 더 작게 발생시키며, 어깨 거상(abduction 및 flexion)의 관절 범위가 좁아지는 기구학적 특성을 지님 _(근거: Page 39, Chapter 6.3)_
  - 근거 원문: “Our analyses showed that stroke participants use less accelerations, higher rotational velocities, and smaller shoulder elevation when brushing their teeth.”
- 물 마시기 동작 수행 과정에서 뇌졸중 환자는 대조군 대비 전완 각속도 기반의 SPARC 평탄도 수치가 유의미하게 낮아 동작의 매끄러움이 저하됨을 나타냄 _(근거: Page 41, Chapter 6.4.2)_
  - 근거 원문: “Figures 6.10 and 6.11 show that the healthy group exhibits significantly smoother transport and recovery motion when drinking compared to the mildly and moderately impaired stroke participants”
- 뇌졸중 환자(GG98 및 GG102)를 대상으로 한 ADL 분할 테스트에서 환자 개인 데이터를 반영하지 않은 일반화 모델(F1 0.51~0.65) 대비 개인 맞춤화 모델(individualized model)은 F1 점수 각각 0.85, 0.87로 월등한 수준을 달성함 _(근거: Page 29, Chapter 5.1)_
  - 근거 원문: “Table 5.1 presents results for each of the settings for the activity segmentation model. The results were much better for the individualized models. For subject GG98, we reached F1 scores of 0.87 for the ADL segmentation and 0.71 for the meal segmentation task. For subject GG102, we obtained F1 scores of 0.88 for the ADL segmentation and 0.82 for the meal segmentation task.”

## 저자 결론

- 전완 양측 2개의 IMU 센서와 CNN-BiLSTM 모델을 접목한 시공간 시점 분할 알고리즘이 건강한 대상자 및 경증 뇌졸중 환자의 일상생활 활동을 세밀한 타임스텝 수준으로 정확하게 분할해낼 수 있음을 입증함 _(근거: Page 54, Chapter 8)_
  - 근거 원문: “We have demonstrated the ability to accurately segment UE ADLs from raw IMU data using a deep learning algorithm that utilizes both CNN and biLSTM layers. While most existing work focuses on classifying a pre-segmented window as part of an activity, we focused on precisely segmenting continuous non-pre-segmented data.”
- 뇌졸중 생존자는 환자간 상지 마비 양상의 편차가 매우 심하므로, 단일한 일반화 모델 보다는 환자의 상태와 움직임 유형을 반영하는 개인화된 맞춤형 알고리즘(individualized model)의 보완이 필수적임 _(근거: Page 54, Chapter 8)_
  - 근거 원문: “Our results highlight the potential need for tailored models for post-stroke individuals due to their high motion variability.”

## 연구의 한계

- 학습 및 평가에 투입한 기성 공인 데이터셋(StrokeRehab)에 원본 영상 자료가 배제되어 알고리즘 오감지의 임상적 사유 규명이나 산출된 마할라노비스 거리 척도의 실제 눈 건강성 검증이 불가능하였음 _(근거: Page 54, Chapter 8)_
  - 근거 원문: “The first limitation of this research comes from the use of open-source datasets with important information missing. The datasets we used did not include video data, which made it difficult to interpret the results of our algorithms. For example, we were unable to determine why the algorithm made certain false detections for activity segmentation. Additionally, for the motion healthiness score, we could not verify whether motions that were considered close to healthy appeared healthy or not.”
- 개발된 시점 분할 알고리즘이 오직 경증 환자(mild post-stroke)를 대상으로 파일럿 테스트가 이루어져 중등도나 중증 이상의 운동장애 환자에 대한 타당성이 부족함 _(근거: Page 54, Chapter 8)_
  - 근거 원문: “Activity segmentation should still be evaluated in stroke participants with moderate or severely affected stroke participants.”
- 구현된 모든 검증 과정이 조절 및 정제된 실험실 내 환경에서 진행되어 복잡하고 불규칙한 가정 내(home setting) 실제 일상활동 상에서의 일반화 성과가 불확실함 _(근거: Page 54, Chapter 8)_
  - 근거 원문: “Our algorithms were tested in controlled environments and we need to evaluate their performance in a home setting.”
- 환자가 마비의 대체재로 사용하는 주된 이상 동작인 '보상 패턴(trunk compensation 등)'의 존재 여부와 감지 기능이 MHS 점수 모델 구조에 완전하게 결합되지 못했음 _(근거: Page 54, Chapter 8)_
  - 근거 원문: “Our analysis of motion quality left out some important aspects such as compensation, which is a good indicator of bad use.”

## 생각해볼 내용

- 제시된 환자 개인화 모델(individualized model)의 ADL 분할 성능(F1 0.85-0.87)은 우수하나, 학습 데이터 수집을 위해 가정 내에서 매번 환자 본인의 환측 동작에 대해 3.5시간 이상 비디오 라벨링 등의 고비용 절차가 선행되어야 하므로 자가 학습이나 준지도 전이학습 기법으로의 발전이 실제 상용화의 핵심 열쇠가 될 것이다. _(근거: Page 26, Chapter 4.2)_
  - 근거 원문: “Labeling the data from one participant doing the unstructured ADL protocol with the labels defined in Table 4.1 takes about 3.5 hours and results in approximately fifty segmented actions.”

## 이 연구가 지적한 선행연구의 문제점

- 기존의 웨어러블 센서 재활 연구들은 대다수가 비사용(nonuse) 방지를 유도하기 위해 마비된 상지의 단순 일일 사용 시간 및 빈도 등의 양적인 측면에만 편중되었으며 동작의 기구학적 품질(질적 수준) 측정은 간과하였다. _(근거: Page 2, Abstract)_
  - 근거 원문: “While the prior art has focused primarily on quantifying the usage of the affected arm, a comprehensive assessment of rehabilitation progress requires evaluating movement quality during functional tasks.”
- 가장 정밀한 광학식 모션 캡처 시스템은 고비용이 요구되고 협소한 사용 공간 제약과 정밀 설정 교육 등의 한계로 인해 통제된 검사실 외에 일반 환자의 홈 케어 환경으로 전용하여 상시 측정하기 곤란했다. _(근거: Page 10, Chapter 2)_
  - 근거 원문: “These systems are reliable in very controlled environments, but they are high-cost and necessitate large space requirements and extended training and setup, so they cannot be used easily outside of the lab.”
- 웨어러블 HAR 기술 연구 중 다수는 미리 일정한 길이로 사전 재단된 윈도우 시계열 데이터 상의 단일 동작을 분류하는 데 머물러 있으며, 끊임없이 일어나는 실시간 시계열 내에서 특정 활동의 개시와 종료 시점을 밝히는 연속적 분할 능력은 미비했다. _(근거: Page 11, Chapter 3.1.2)_
  - 근거 원문: “While most existing work focuses on classifying a pre-segmented window as part of an activity, little work has investigated the precise time segmentation of continuous, non-pre-segmented data.”

## 이 연구의 해결 방식과 기여

- 양치질과 물 마시기(transport, recovery) 각기 다른 복잡성의 운동 과정 속에서 가속도 크기 감쇠, 각속도 역학 변조, 어깨 관절 가동 각도의 손상 등 질적 변형을 포착할 기구학적 특성을 수치로 실증함 _(근거: Page 2, Abstract)_
  - 근거 원문: “We identified robust kinematic metrics that can be computed directly from accelerometer and gyroscope data and that differentiate between healthy and stroke participants.”
- 오픈소스 데이터셋이 결여하고 있던 비디오 및 모션 캡처 그라운드 트루스가 보장된 자가 ADL 수집용 프로토콜을 규정하고, 5명의 대조군 및 5명의 뇌졸중 환자를 통해 전구적인 신규 시계열 데이터셋 구축을 완료함 _(근거: Page 2, Abstract)_
  - 근거 원문: “We also proposed a methodology for collecting an annotated naturalistic IMU-based ADL dataset in a controlled environment, using video ground truth to reliably label ADLs.”

## 레퍼런스할 수 있는 내용

### 1. 뇌졸중 발병 후 6개월 시점의 손 기능 장애 잔존율

- 원문 발췌: “After six months from a stroke, about 65% of patients are unable to use their affected hand for their usual activities [3].”
- 한국어 번역: 뇌졸중 발생 6개월 후, 환자의 약 65%는 그들의 일상적인 활동에 환측 손을 사용하지 못한다.
- 원문 위치: Page 8, Chapter 1 Introduction
- 원문 내 인용표기: [3]
- 해당 선행문헌: [3] Bruce H. Dobkin. Rehabilitation after Stroke. The New England journal of medicine, 352(16):1677–1684, April 2005.
- 주장 유형: background_citation
- 활용 맥락과 주의: 뇌졸중 환자가 장기적으로 겪는 영구적 상지 장애 및 손 기능 마비의 고착 비율(65%)에 대한 근거로 인용 시 유용.

### 2. 뇌졸중의 평생 누적 위험율 증가 추세

- 원문 발췌: “The lifetime risk of stroke has increased over the last 20 years by 50% and is now one in four people [1].”
- 한국어 번역: 뇌졸중의 평생 위험은 지난 20년 동안 50% 증가했으며 현재는 4명 중 1명에 달한다.
- 원문 위치: Page 8, Chapter 1 Introduction
- 원문 내 인용표기: [1]
- 해당 선행문헌: [1] Valery L Feigin, Michael Brainin, Bo Norrving, Sheila Martins, Ralph L Sacco, Werner Hacke, Marc Fisher, Jeyaraj Pandian, and Patrice Lindsay. World Stroke Organization (WSO): Global Stroke Fact Sheet 2022. International Journal of Stroke, 17(1):18–29, January 2022. Publisher: SAGE Publications.
- 주장 유형: background_citation
- 활용 맥락과 주의: 전 세계적으로 뇌졸중 발병률이 증가하고 있으며 평생 발병 위험이 25%에 육박한다는 최근 20년 역학적 보건 통계 자료로 활용 가능.

### 3. 본 연구에서 개발한 MHS 척도의 동일 환자 내 측정 일관성 및 변별 성능

- 원문 발췌: “It showed desirable properties such as high consistency when rating sample motions from the same subject and good separability (>80% accuracy) between healthy and post-stroke individuals.”
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
