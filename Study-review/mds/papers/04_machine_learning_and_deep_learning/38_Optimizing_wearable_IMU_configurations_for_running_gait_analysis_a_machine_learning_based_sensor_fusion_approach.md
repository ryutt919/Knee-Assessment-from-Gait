# Optimizing wearable IMU configurations for running gait analysis: a machine learning-based sensor fusion approach

Yuan, Y., Yu, Y., Cai, S., & Cheng, W. (2026). Optimizing wearable IMU configurations for running gait analysis: a machine learning-based sensor fusion approach. Frontiers in Bioengineering and Biotechnology, 14, 1762919. https://doi.org/10.3389/fbioe.2026.1762919

## 서지정보

- 저자: Ye Yuan, Yaohui Yu, Shanshan Cai, Weidong Cheng
- 연도: 2026
- 저널: Frontiers in Bioengineering and Biotechnology
- DOI: 10.3389/fbioe.2026.1762919
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/04_machine_learning_and_deep_learning/Optimizing wearable IMU configurations for running gait analysis - a machine learning-based sensor fusion approach.pdf
- 분석 provider: antigravity

## 연구 목적

- 본 연구는 고차원 17개 센서 네트워크를 측정 정확도 저하 없이 최소화-최적화된 하위 집합으로 줄일 수 있는지 타당성을 결정하기 위해 머신러닝 기법을 적용한다. _(근거: PAGE 1, Objective)_
  - 근거 원문: “This study applies machine learning (ML) techniques to address this
hardware limitation by determining the feasibility of reducing a high-dimensional
17-sensor network to a “minimal-optimal” subset without
compromising measurement accuracy.”
- 본 연구의 주된 목적은 골드 스탠다드인 17개 센서 네트워크에 대해 축소된 IMU 구성(1-3개 센서)의 파라미터 추정 성능을 체계적으로 평가하는 것이다. _(근거: PAGE 2, 1 Introduction)_
  - 근거 원문: “Therefore,theprimaryobjectiveofthispaperistosystematically
evaluate the parameter estimation performance of reduced IMU
configurations (1–3 sensors) against a gold-standard 17-sensor
network.”

## 연구 설계와 대상

- 지역 달리기 클럽과 소셜 미디어 광고를 통해 25명의 여가용 러너(남성 15명, 여성 10명)를 모집했다. _(근거: PAGE 2, 2.1 Participants)_
  - 근거 원문: “We recruited twenty-five recreational runners (15 male,
10 female) through local running clubs and social media
advertisements.”
- 참가자 선정 기준은 18~45세 연령, 지난 1년간 주당 최소 15km 달리기, 12km/h 속도로 5분 이상 지속 달리기 가능, 심혈관 및 신경계 질환이 없는 것이었다. _(근거: PAGE 2, 2.1 Participants)_
  - 근거 원문: “Inclusion criteria for participation were: (1) age between 18 and
45 years; (2) an average weekly running volume of at least 15 km
over the past year; (3) the ability to run continuously at 12 km/h for
at least 5 min; and (4) no history of cardiovascular or neurological
diseases.”
- 참가자들은 느린 페이스, 중간 페이스, 템포 페이스를 대표하는 8, 10, 12 km/h의 고정된 속도에서 각각 3분씩 세 차례 러닝 트라이얼을 수행했다. _(근거: PAGE 3, 2.3 Experimental protocol)_
  - 근거 원문: “The
mainprotocolconsistedofthree3-minrunningtrialsatfixed speeds
of 8 km/h, 10 km/h, and 12 km/h, representing slow, medium, and
tempo paces.”

## 방법

- 골드 스탠다드로 Xsens MVN Awinda 관성 모션 캡처 시스템을 사용했으며, 100 Hz로 샘플링하는 17개의 무선 IMU로 구성되었다. _(근거: PAGE 3, 2.2 Experimental equipment and setup)_
  - 근거 원문: “We used the Xsens MVN Awinda inertial motion capture
system (Xsens Technologies B.V., Netherlands) as the gold
standard. This system comprises 17 wireless IMUs (MTw2)
sampling at 100 Hz.”
- 원시 IMU 신호는 고주파 시계열 데이터를 회귀에 적합한 피처 공간으로 변환하기 위해 50% 중첩되는 250 ms 윈도우 크기의 슬라이딩 윈도우 방식을 사용하여 처리되었다. _(근거: PAGE 3, 2.5.1 Feature engineering)_
  - 근거 원문: “Raw IMU signals (3-axis acceleration and 3-axis angular velocity)
were processed using a sliding-window approach to transform high-
frequency time-series data into a feature space suitable for regression
(Figure 2). Signals were segmented into 250 ms windows with a 50%
overlap.”
- 기준 모델인 선형 회귀(LR) 모델 및 딥러닝(LSTM) 신경망 모델과 비교하여 랜덤 포레스트(RF) 모델의 성능을 벤치마킹했다. _(근거: PAGE 4, 2.5.2 Model selection and training)_
  - 근거 원문: “We benchmarked the
RF model against a baseline Linear Regression (LR) model and a Long
Short-Term Memory (LSTM) neural network.”
- 데이터 누수를 방지하고 모델 일반화를 확보하기 위해, 훈련 세트(20명, 80%)와 보류 테스트 세트(5명, 20%)로 데이터셋을 분할하는 대상자 독립적 검증 방식을 채택했다. _(근거: PAGE 5, 2.5.3 Validation strategy and statistical analysis)_
  - 근거 원문: “To ensure model generalization and prevent data leakage, we
employed a strict subject-independent validation. The dataset was
randomly split into a training set (20 participants, 80%) and a hold-
out test set (5 participants, 20%).”

## 핵심 결과

- 단일 요추센서(Lumbosacral IMU) 구성은 Cadence, Vertical Oscillation, Ground Contact Time 등의 전역 파라미터를 고정밀도로 재구성할 수 있었으나 보행 비대칭성 검지에는 실패했다. _(근거: PAGE 1, Results)_
  - 근거 원문: “Analysis revealed that a single lumbosacral IMU could successfully
reconstruct global parameters (Cadence, Vertical Oscillation, Ground Contact
Time) with high precision (R2
>0.95,MAPE<5%), outperforming standard
commercial benchmarks. However, this single-node setup failed to detect gait
asymmetry (R2
 0.52).”
- 요추와 양측 발목 센서를 결합한 분산형 3-센서 융합 구성은 단일 노드 구성의 대칭성 감지 한계를 해결하여 모든 파라미터에서 전신 시스템과 필적하는 성능을 보여주었다. _(근거: PAGE 1, Results)_
  - 근거 원문: “A distributed three-sensor fusion configuration
(Lumbosacral + Bilateral Ankles) resolved this limitation, achieving results
comparable to the full-body system for all parameters
(R2
>0.91,MAPE  7.12%).”
- 전체 피처 세트에서 랜덤 포레스트(RF) 모델은 GCT에 대해 R2 성능을 선형 회귀(LR) 대비 0.15 이상 향상시켜 게이트 역학의 비선형적 특성을 확인했다. _(근거: PAGE 5, 3.2 Predictive performance of IMU configurations)_
  - 근거 원문: “On the full feature set, the RF model
significantly outperformed a baseline Linear Regression model
(R2
improvement of >0.15 for GCT), confirming the non-linear
nature of gait dynamics.”

## 저자 결론

- 본 연구는 센서 어레이 설계를 최적화하기 위한 머신러닝 프레임워크를 검증하고 있으며, 제안된 3-센서 융합은 차세대 웨어러블 디바이스를 위한 견고하고 저비용의 아키텍처 청사진을 제공한다. _(근거: PAGE 1, Conclusion)_
  - 근거 원문: “This study validates a machine learning framework for optimizing
sensor array design. The proposed three-sensor fusion offers a robust, low-cost
architectural blueprint for next-generation wearable devices, proving that
complex deep learning is not always required when sensor placement is
biomechanically optimized.”
- 단일 요추 장착형 IMU를 머신러닝 모델과 결합했을 때 복잡한 다중 센서 설정 없이도 주요 러닝 게이트 파라미터를 정확하게 예측할 수 있는 뛰어난 효용성이 입증되었다. _(근거: PAGE 5, 4.1 Principal findings and interpretation)_
  - 근거 원문: “The central finding of this study is the remarkable efficacy of a
single, lumbosacral-mounted IMU when combined with a
machine learning model. Our results compellingly
demonstrate that it is possible to accurately predict key
running gait parameters without resorting to a complex
“Christmas tree” sensor setup.”

## 연구의 한계

- 본 연구는 표면이 균일하고 평평한 트레드밀에서 진행되었으며, 이는 지면 변동성과 공기 저항이 존재하는 야외 실외 달리기와 비교할 때 보행 역학이 약간 다를 수 있다는 한계가 있다. _(근거: PAGE 9, 4.5 Limitations and future directions)_
  - 근거 원문: “First, this study was conducted on a treadmill,
which provides a homogenous, flat surface. We acknowledge that
treadmill running lacks the surface variability and air resistance of
overground running, and gait mechanics may differ slightly (Van
Hooren et al., 2020).”
- 검증된 속도 범위(8-12 km/h)는 일정한 정상 상태의 지구력 달리기를 대표하므로, 보행 역학이 근본적으로 변하는 단거리 전력 질주나 고강도 인터벌 러닝(>15 km/h)에 직접 외삽하여 적용할 수 없다. _(근거: PAGE 10, 4.5 Limitations and future directions)_
  - 근거 원문: “Second, the validated speed range (8–12 km/h) represents
steady-state endurance running. It is crucial to note that the
proposed “minimal-optimal” configurations cannot be directly
extrapolated to sprinting or high-intensity interval running
(>15 km/h).”
- 최고 시험 속도(12 km/h)에서 오차가 약간 증가하는 것을 관찰했는데, 이는 시간 분해능 부족보다는 고충격 착지 시 피부 변형으로 인해 센서가 뼈에 대해 상대적으로 움직이는 연조직 아티팩트(STA)에 기인한다. _(근거: PAGE 10, 4.5 Limitations and future directions)_
  - 근거 원문: “However, at the highest tested speed (12 km/h), we observed a
marginal increase in error. This is likely attributable to Soft Tissue
Artifacts (STA)—the secondary motion of the sensor relative to the
bone caused by skin deformation during high-impact
landing—rather than the temporal resolution itself.”

## 생각해볼 내용

- 저자는 3-센서 구성이 가격과 편의성 면에서 최적의 대안이 될 수 있음을 시사한다. _(근거: PAGE 1, Conclusion)_
  - 근거 원문: “The proposed three-sensor fusion offers a robust, low-cost
architectural blueprint for next-generation wearable devices, proving that
complex deep learning is not always required when sensor placement is
biomechanically optimized.”
- 단일 센서 구성은 보행 비대칭성 감지에 치명적인 한계가 있어 임상적으로 유용한 비대칭성 평가를 위해서는 3-센서 구성이 필수적이다. _(근거: PAGE 1, Results)_
  - 근거 원문: “However, this single-node setup failed to detect gait
asymmetry (R2
 0.52). A distributed three-sensor fusion configuration
(Lumbosacral + Bilateral Ankles) resolved this limitation, achieving results
comparable to the full-body system for all parameters”
- 임베디드 기기 구현 시 랜덤 포레스트는 LSTM 등 딥러닝 모델 대비 낮은 연산 복잡도와 메모리 요구량 덕분에 유리하다. _(근거: PAGE 9, 4.4 System feasibility and embedded implementation)_
  - 근거 원문: “Unlike Deep Neural
Networks (DNNs) or LSTMs, which require computationally
expensive matrix multiplications and substantial RAM for
activation maps, the RF inference process consists of a series of
simple conditional checks (if-else statements).”

## 이 연구가 지적한 선행연구의 문제점

- 연구실 기반의 3D 광학 모션 캡처 시스템은 정밀하지만 가격이 대단히 비싸고 통제된 실험실 환경에 국한되며 특수 전문 지식이 필요하다. _(근거: PAGE 2, 1 Introduction)_
  - 근거 원문: “They are prohibitively expensive, confined to controlled
laboratory environments, and require highly specialized expertise
for data collection and processing.”
- 전신 운동학을 복원하기 위해 다수의 센서 어레이(예: 17개)를 부착하는 방식은 비용 및 복잡성을 높이고 긴 셋업 시간을 요구하며 사용자의 자연스러운 보행을 방해할 수 있다. _(근거: PAGE 2, 1 Introduction)_
  - 근거 원문: “This “Christmas
tree” effect, while feasible for research, presents significant practical
barriers: it is still costly and complex, places a heavy time burden on
the user for setup (often 15–30 min), and negatively impacts user
comfort, which may even alter the natural gait being measured
(Caldas et al., 2017).”

## 이 연구의 해결 방식과 기여

- 머신러닝을 활용하여 신체의 중요 노드(질량 중심 및 말단 효과기)에서 수집한 데이터의 정보 중복성을 디코딩함으로써 부족한 하드웨어를 가상화하는 방법을 제안한다. _(근거: PAGE 2, 1 Introduction)_
  - 근거 원문: “This study proposes that machine learning (ML) is the ideal
tool to address this hardware-accuracy dilemma. From a
signal processing perspective, human locomotion involves
highly coordinated kinematic chains, implying significant
information redundancy across different body segments. We
hypothesize that data acquired from critical nodes—specifically
the Center of Mass (CoM) and end-effectors—contain sufficient
latent features to estimate the key spatio-temporal gait scalars of
the system.”
- 제안된 3-센서 구성(요추 + 양측 발목)은 요추 센서의 전역적 파라미터 검출 성능과 발목 센서의 시간적/비대칭성 검출 성능을 효과적으로 결합하여 골드 스탠다드 수준의 신뢰성을 보여주었다. _(근거: PAGE 7, 4.1 Principal findings and interpretation)_
  - 근거 원문: “It
successfully combines the global-parameter strength of the
lumbar sensor with the temporal and asymmetry-detecting
strengths of the ankle sensors, achieving high performance
(R2
>0.91) across all measured parameters.”

## 레퍼런스할 수 있는 내용

### 1. 러닝 활동과 심혈관 건강 및 정신 웰빙의 긍정적인 관계

- 원문 발췌: “Running is one of the most popular and accessible forms of
physical activity worldwide, offering significant benefits for
cardiovascular health, mental wellbeing, and overall longevity
(Lee et al., 2014).”
- 한국어 번역: 달리기는 전 세계적으로 가장 인기 있고 접근하기 쉬운 신체 활동 중 하나로, 심혈관 건강, 정신적 웰빙 및 전반적인 수명 연장에 상당한 이점을 제공한다 (Lee et al., 2014).
- 원문 위치: PAGE 2, 1 Introduction
- 원문 내 인용표기: (Lee et al., 2014)
- 해당 선행문헌: Lee,D.C.,Pate,R.R.,Lavie,C.J.,Sui,X.,Church,T.S.,andBlair,S.N.(2014).Leisure-
timerunningreducesall-causeandcardiovascularmortalityrisk.J.Am.Coll.Cardiol.64
(5), 472–481. doi:10.1016/j.jacc.2014.04.058
- 주장 유형: background_citation
- 활용 맥락과 주의: 달리기가 건강 및 수명에 미치는 긍정적 기여를 설명할 때 선행 문헌의 근거로 인용할 수 있음. 2차 인용에 주의해야 함.

### 2. 러닝 부상 위험과 연관된 이상 보행 분석 파라미터

- 원문 발췌: “For instance, excessive vertical impact forces, high vertical
oscillation (VO), prolonged ground contact time (GCT), and
excessive pronation are considered key risk factors (Hreljac,
2004; Davis and Powers, 2010).”
- 한국어 번역: 예를 들어, 과도한 수직 충격력, 높은 수직 진동(VO), 연장된 지면 접촉 시간(GCT), 과도한 회내는 주요 위험 요소로 간주된다 (Hreljac, 2004; Davis and Powers, 2010).
- 원문 위치: PAGE 2, 1 Introduction
- 원문 내 인용표기: (Hreljac, 2004; Davis and Powers, 2010)
- 해당 선행문헌: Hreljac, A. (2004). Impact and overuse injuries in runners. Med. and Sci. Sports and
Exerc. 36 (5), 845–849. doi:10.1249/01.mss.0000126803.66636.dd
- 주장 유형: background_citation
- 활용 맥락과 주의: 러닝 관련 부상(RRIs)의 위험 요인이 되는 생체역학적 보행 지표들을 뒷받침하기 위해 인용 가능함. 2차 인용에 주의해야 함.

### 3. 발목 위치 IMU 센서의 보행 시간적 매개변수 측정의 탁월성

- 원문 발췌: “The
ankle sensors (Config 2) were superior for temporal metrics
because their signals provide unambiguous, high-amplitude
spikes and reversals corresponding to the discrete events of
initial contact (IC) and toe-off (TO) (Aminian et al., 2002).”
- 한국어 번역: 발목 센서(Config 2)는 신호가 초기 접촉(IC) 및 발가락 떼기(TO)의 개별 이벤트에 해당하는 명확하고 높은 진폭의 스파이크 및 반전을 제공하기 때문에 시간적 측정 기준에 우수했다 (Aminian et al., 2002).
- 원문 위치: PAGE 8, 4.2 Biomechanical interpretation and model trust
- 원문 내 인용표기: (Aminian et al., 2002)
- 해당 선행문헌: Aminian, K., Najafi, B., Büla, C., Leyvraz, P. F., and Robert, P. (2002). Spatio-
temporal parameters of gait measured by an ambulatory system using miniature
gyroscopes. J. Biomechanics 35 (5), 689–699. doi:10.1016/s0021-9290(02)
00008-8
- 주장 유형: background_citation
- 활용 맥락과 주의: 달리기 보행 분석에서 발목 위치 센서가 왜 접촉 시간과 같은 시간적 지표 검출에 유리한지 기하학적/생체역학적 배경 설명 시 인용 가능함. 2차 인용에 주의해야 함.
