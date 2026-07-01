# Deep Learning in Gait Parameter Prediction for OA and TKA Patients Wearing IMU Sensors

Sharifi Renani, M., Myers, C. A., Zandie, R., Mahoor, M. H., Davidson, B. S., & Clary, C. W. (2020). Deep Learning in Gait Parameter Prediction for OA and TKA Patients Wearing IMU Sensors. Sensors, 20(19), 5553. https://doi.org/10.3390/s20195553

## 서지정보

- 저자: Mohsen Sharifi Renani, Casey A. Myers, Rohola Zandie, Mohammad H. Mahoor, Bradley S. Davidson, Chadd W. Clary
- 연도: 2020
- 저널: Sensors
- DOI: 10.3390/s20195553
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/04_machine_learning_and_deep_learning/Deep Learning in Gait Parameter Prediction for OA and TKA Patients Wearing IMU Sensors.pdf
- 분석 provider: antigravity

## 연구 목적

- 본 연구의 목적은 골관절염(OA) 및 관절 치환술 환자군에서 IMU 데이터로부터 보행 시공간적 변수(STGPs)를 예측하기 위한 여러 최신 심층 신경망 구조의 성능을 평가하고, 예측 정확도를 극대화하기 위한 최적의 센서 조합을 결정하는 것이다. _(근거: PAGE 2, 1. Introduction)_
  - 근거 원문: “Thus, the purpose of this study was two-fold: (1) to access the ability of multiple contemporary deep neural network architectures to predict STGPs from IMU data in the OA and joint-replacement patient populations and (2) to determine the optimal sensor combination to maximize prediction accuracy.”

## 연구 설계와 대상

- 총 29명의 대상자(골관절염 환자 14명, 인공관절 전치환술 환자 15명)가 연구에 참여하였다. _(근거: PAGE 2, 2.1. Gait Measurements of Osteoarthritic and Total Knee-Replacement Subjects)_
  - 근거 원문: “Twenty-nine subjects, including 14 subjects with OA (Age = 67 ± 7, weight = 79 ± 12 kg, height = 168 ± 16 cm, 4 females and 10 males), 15 subjects with total knee arthroplasty (TKA) (Age = 68 ± 4, weight = 76 ± 14 kg, height = 164 ± 9 cm, 11 females and 4 males, 7 uni-lateral and 8 bi-lateral), participated in the study as part of a larger investigation.”
- 대상자들은 일상 보행 속도 범위를 포괄하기 위해 자가 선택 속도, 느린 속도, 빠른 속도의 세 가지 속도로 5m 보행 과제를 15회 수행하였다. _(근거: PAGE 2, 2.1. Gait Measurements of Osteoarthritic and Total Knee-Replacement Subjects)_
  - 근거 원문: “Subjects performed 15 trials of a 5-m walking task at three different speeds: self-selected, slow, and fast to cover the entire range of possible daily walking paces.”

## 방법

- 대상자들의 해부학적 랜드마크에 71개의 반사 마커를 부착하고 여러 사지 세그먼트와 몸통에 17개의 IMU를 장착했으며, 본 연구에서는 발, 종아리, 허벅지, 골반에 위치한 7개의 IMU만 데이터 분석에 사용했다. _(근거: PAGE 2, 2.1. Gait Measurements of Osteoarthritic and Total Knee-Replacement Subjects)_
  - 근거 원문: “Subjects were fitted with 71 reflective markers on anatomical landmarks and 17 IMUs on various limb segments and the trunk. For this study, only the 7 IMUs located on the feet, shanks, thighs [35,36], and pelvis [37] were used in the subsequent data analysis (Figure 1a,b).”
- 힘 데이터, 모션 캡처(MOCAP), IMU(자유 가속도 및 각속도)의 샘플링 주파수는 각각 1000 Hz, 100 Hz, 40 Hz였다. _(근거: PAGE 3, 2.1. Gait Measurements of Osteoarthritic and Total Knee-Replacement Subjects)_
  - 근거 원문: “The sampling frequency of force data, MOCAP, and IMUs (free acceleration and angular velocity) were 1000 Hz, 100 Hz, and 40 Hz, respectively.”
- IMU 데이터는 100 Hz로 업샘플링되었으며 피크 검출 방법을 사용하여 시상면에서 발 센서의 각속도를 기준으로 양쪽 다리의 전체 보행 주기로 분할되었다. _(근거: PAGE 3, 2.2. Gait Data Processing)_
  - 근거 원문: “IMU data for each trial was up-sampled to 100 Hz and segmented into full strides for each leg based on the angular velocities of the feet sensors in the sagittal plane using the peak detection method (Figure 1c,d) [41,42].”
- 사전 신경망 구조 벤치마킹 결과를 바탕으로, 센서 조합에 대한 대규모 실험계획법 연구를 위해 Zrenner 등이 제안한 1D 합성곱 신경망(CNN) 구조가 선택되었다. _(근거: PAGE 4, 2.4. Assessing Optimal Sensor Combinations for Each Gait Characteristic)_
  - 근거 원문: “Based on the result of the preliminary neural network architecture selection, the 1D convolution neural network (CNN) architecture proposed by Zrenner et al. was chosen for a larger design-of-experiment study on sensor combinations [19].”
- 발, 골반, 종아리, 허벅지 센서의 15가지 고유 조합을 기반으로 예측 정확도를 분석하기 위해 풀 팩토리얼 실험계획법이 구현되었다. _(근거: PAGE 5, 2.4. Assessing Optimal Sensor Combinations for Each Gait Characteristic)_
  - 근거 원문: “A full factorial design of experiments was implemented to analyze the prediction accuracy based on 15 unique combinations of the feet, pelvis, shank, and thigh sensors (Table 2).”

## 핵심 결과

- 12개 시공간 보행 변수(STGPs)의 백분율 오차는 2.1%(보행 주기 시간)에서 73.7%(발 외향각)까지 분포했으며 전반적으로 공간적 변수보다 시간적 변수에서 더 정확했다. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “Percent error across the 12 STGPs ranged from 2.1% (stride time) to 73.7% (toe-out angle) and overall was more accurate in temporal parameters than spatial parameters.”
- 전체적으로 발-허벅지(F T) 조합이 가장 우수한 평균 순위(5.1)를 나타냈고, 발-종아리(F S, 6.2), 종아리(S, 6.3) 센서 조합이 그 뒤를 이었다. _(근거: PAGE 11, 3.3. Optimal Sensor Combinations for Gait Characteristics)_
  - 근거 원문: “Overall, the feet-thigh (F T) configuration had the best average rank (5.1), followed by the feet-shank (F S, 6.2), and shank (S, 6.3) sensor combinations.”
- 골관절염(OA) 환자군은 인공관절 전치환술(TKA) 환자군에 비해 모든 센서 조합 및 STGPs에 걸쳐 더 큰 평균(19.0%) 및 중앙값(6.6%) NAPE를 보였다(TKA 평균 NAPE = 14.7%, 중앙값 NAPE = 4.6%). _(근거: PAGE 9, 3.3. Optimal Sensor Combinations for Gait Characteristics)_
  - 근거 원문: “The OA cohort had larger mean (19.0%) and median (6.6%) NAPE across all sensor combinations and STGPs compared to TKA (mean NAPE = 14.7%, median NAPE = 4.6%).”

## 저자 결론

- 본 연구는 딥러닝 기반 데이터 구동 방식이 IMU 센서 신호를 바탕으로 OA 및 TKA 환자의 시공간적 보행 특성을 예측할 수 있음을 입증하였다. _(근거: PAGE 15, 5. Conclusions)_
  - 근거 원문: “This study demonstrated that a deep-learning, data-driven approach was able to predict spatial temporal gait characteristics of OA and TKA patients based on signals from IMU sensors.”
- 다양한 센서 조합과 STGPs, 환자군, 보행 속도에 대한 민감도 분석을 통해, 딥러닝이 환자 모니터링 시스템 설계를 방해하고 순응도에 부정적 영향을 미치는 센서 위치 의존성 문제를 극복할 수 있음을 보여주었다. _(근거: PAGE 15, 5. Conclusions)_
  - 근거 원문: “Using a comprehensive analysis of various sensor combinations and their sensitivity to STGPs, patient population, and walking pace, our results showed that deep learning can overcome the dependency on sensor location that hinders the design of patient monitoring systems and negatively impacts patient compliance.”

## 연구의 한계

- 본 연구는 포함된 대상자의 수가 적다는 한계가 있다. _(근거: PAGE 15, 4. Discussion)_
  - 근거 원문: “This study was also limited in the number of subjects that were included.”
- 다른 데이터 구동 방식과 마찬가지로, 본 연구에서 훈련된 신경망은 오직 선택된 모집단에만 적합하다. _(근거: PAGE 15, 4. Discussion)_
  - 근거 원문: “Like other data-driven approaches, the trained network described in this study are only suitable for the selected population.”
- 실험실 외부의 대규모 환자군에게 알고리즘을 적용하는 데 있어 센서 부착 위치의 다양성, 저가형 IMU의 신호 품질 저하, 체질량 지수가 높은 환자의 연부조직 아티팩트, 훈련 데이터셋 범위를 벗어난 보행 변수를 가진 환자 식별 등의 실질적인 한계가 존재한다. _(근거: PAGE 15, 4. Discussion)_
  - 근거 원문: “There are also practical limitations to deploying our algorithm to a large patient population outside of a laboratory environment, including variability in sensor placement, reduced signal quality from low-cost IMUs, soft-tissue artifacts for high body mass index patients, and identification of patients with gait parameters outside the training data set.”

## 생각해볼 내용

- 훈련 데이터 분포를 벗어난 단 하나의 이상치(S21)가 테스트 세트 전체의 오차를 크게 상승시킨 결과는, 기계학습 모델을 실제 임상 환경에 적용할 때 분포 외 데이터(Out-of-distribution)에 대한 취약성이 주요 해결 과제임을 시사한다. _(근거: PAGE 14, 4. Discussion)_
  - 근거 원문: “The impact of subject S21 in the test set is an example of how CNNs result in poor performance when faced with data that are outside the distribution of the training data, which is one of the main challenges in the use of machine-learning models for real world applications.”
- 발-허벅지(F-T) 센서 조합이 통계적으로 유의미하게 우수하지만, 다른 센서 조합과의 예측 정확도 차이는 2~5% 수준에 불과하므로, 임상 응용 분야에 따라 비용과 환자 편의성을 고려해 센서 조합을 유연하게 설계할 수 있다. _(근거: PAGE 14, 4. Discussion)_
  - 근거 원문: “As noted earlier, while the F-T sensor combination proved to be statistically better than other combinations, a 2–5% improvement in overall STGP prediction accuracy may be impactful during certain clinical applications.”

## 이 연구가 지적한 선행연구의 문제점

- 기존의 보행 특성 측정 방법(MOCAP 및 힘판)은 실험실 환경과 고가의 장비를 요구하며 시간 소모가 크다. _(근거: PAGE 1, 1. Introduction)_
  - 근거 원문: “Conventional methods for measuring gait characteristics that include motion capture (MOCAP) systems and force plates require a laboratory environment and expensive, time-consuming, equipment [8].”
- 센서 융합이나 칼만 필터를 사용하는 기존의 이중 적분 기반 방법들은 보행 분할을 위해 stance 구간의 제로 속도 조건(zero-velocity condition)에 의존하지만, 병리적 보행을 가진 환자나 자유 달리기 같은 역동적 활동 중에는 이 조건을 확인하기 어렵다. _(근거: PAGE 2, 1. Introduction)_
  - 근거 원문: “However, clear zero-velocity conditions are difficult to identify for patients with pathological gait or during highly dynamic activities like free running [19].”
- 다양한 환자군에 대해 최상의 성능을 내는 최적의 센서 조합을 정량화한 체계적인 연구가 현재 부족한 실정이다. _(근거: PAGE 2, 1. Introduction)_
  - 근거 원문: “Additionally, systematic studies quantifying optimal sensor combinations for the best performance across various patient populations are important to this field, but are lacking.”

## 이 연구의 해결 방식과 기여

- 골관절염(OA) 및 인공관절 전치환술(TKA) 대상자에 대해 연구된 적이 없었던, IMU 데이터 기반의 12가지 STGP 예측 성능 벤치마킹 및 최적의 센서 조합을 식별하는 연구를 수행하였다. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “A study was conducted to benchmark the ability of multiple deep neural network (DNN) architectures to predict 12 STGPs from inertial measurement unit (IMU) data and to identify an optimal sensor combination, which has yet to be studied for OA and TKA subjects.”
- 본 연구 결과는 골관절염 환자 및 관절 치환술을 받을 환자들에게 STGPs의 정확한 실시간 모니터링을 제공하여 치료, 수술 계획 및 재활에 실질적인 도움을 줄 수 있다. _(근거: PAGE 2, 1. Introduction)_
  - 근거 원문: “The results of this study will help patients suffering from OA who may go on to receive a total joint replacement benefit from the accurate real-time patient monitoring of STGPs to inform their treatment, surgical planning, and rehabilitation.”

## 레퍼런스할 수 있는 내용

### 1. 골관절염 환자의 보행 적응 특성

- 원문 발췌: “Patients with progressive OA typically exhibit gait adaptations including decreased joint flexibility, increased stance time on the affected side, cadence, and double support time, and an overall increase in variability of spatial temporal parameters [31–34].”
- 한국어 번역: 진행성 골관절염(OA) 환자들은 일반적으로 관절 유연성 감소, 환측 디딤 시간 증가, 보행 속도 및 양하지 지기 시간의 증가, 그리고 시공간 보행 변수의 전반적인 변동성 증가를 포함하는 보행 적응 특성을 보인다.
- 원문 위치: PAGE 2, 1. Introduction
- 원문 내 인용표기: [31–34]
- 해당 선행문헌: 31. Bejek, Z.; Paróczai, R.; Illyés, Á.; Kiss, R.M. The influence of walking speed on gait parameters in healthy people and in patients with osteoarthritis. Knee Surg. Sports Traumatol. Arthrosc. 2006. [CrossRef]
32. Kiss, R.M.; Bejek, Z.; Szendrői, M. Variability of gait parameters in patients with total knee arthroplasty. Knee Surg. Sports Traumatol. Arthrosc. 2012. [CrossRef]
33. Kiss, R.M. Effect of severity of knee osteoarthritis on the variability of gait parameters. J. Electromyogr. Kinesiol. 2011. [CrossRef]
34. Hollman, J.H.; McDade, E.M.; Petersen, R.C. Normative spatiotemporal gait parameters in older adults. Gait Posture 2011. [CrossRef]
- 주장 유형: background_citation
- 활용 맥락과 주의: 골관절염 환자군의 병리적 보행 변이성과 임상적 보행 적응 양상에 대한 근거로 인용하기 적합함.

### 2. 단일 센서 부착 방식의 보행 분석 한계

- 원문 발췌: “Single body segment mounted IMUs (e.g., wrist or pelvis) are limited in calculation of certain STGPs such as number of steps, step cadence, or step distance which may not be adequate for clinical applications [26,27].”
- 한국어 번역: 단일 신체 부위에 부착된 IMU(예: 손목 또는 골반)는 걸음 수, 보행 빈도 또는 걸음 거리와 같은 특정 STGP의 계산에 한계가 있어 임상 적용에 불충분할 수 있다.
- 원문 위치: PAGE 2, 1. Introduction
- 원문 내 인용표기: [26,27]
- 해당 선행문헌: 26. Fasel, B.; Duc, C.; Dadashi, F.; Bardyn, F.; Savary, M.; Farine, P.A.; Aminian, K. A wrist sensor and algorithm to determine instantaneous walking cadence and speed in daily life walking. Med. Biol. Eng. Comput. 2017, 55, 1773–1785. [CrossRef]
27. Soltani, A.; Dejnabadi, H.; Savary, M.; Aminian, K. Real-world gait speed estimation using wrist sensor: A personalized approach. IEEE J. Biomed. Health Inf. 2019. [CrossRef]
- 주장 유형: background_citation
- 활용 맥락과 주의: 단일 부위 센서 착용 시 특정 임상적 보행 지표 획득의 제한점과 다중 센서 사용의 당위성을 설명할 때 인용할 수 있음.

### 3. 고령층 대상 딥러닝 보폭 예측의 우수성

- 원문 발췌: “Using a deep convolutional neural network trained on over 1220 strides from 101 geriatric patients, the algorithm predicted stride length with a mean error of −0.15 cm, which was considerably more accurate than previous integration-based methods [12].”
- 한국어 번역: 101명의 고령 환자로부터 얻은 1220개 이상의 보행 주기를 학습한 심층 합성곱 신경망을 사용하여 보폭을 예측한 결과, 평균 오차 -0.15 cm로 기존 적분 기반 방법들보다 훨씬 더 정확한 예측 결과를 보였다.
- 원문 위치: PAGE 2, 1. Introduction
- 원문 내 인용표기: [12]
- 해당 선행문헌: 12. Rampp, A.; Barth, J.; Schulein, S.; Gassmann, K.G.; Klucken, J.; Eskofier, B.M. Inertial Sensor-Based Stride Parameter Calculation From Gait Sequences in Geriatric Patients. IEEE Trans. Biomed. Eng. 2015, 62, 1089–1097. [CrossRef] [PubMed]
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
