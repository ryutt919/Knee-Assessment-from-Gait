# GAIT ANALYSIS USING IMU SENSOR

Gujarathi, T., & Bhole, K. (2019). Gait Analysis Using IMU Sensor. In 10th ICCCNT 2019. IEEE. https://doi.org/확인 불가

## 서지정보

- 저자: Trupti Gujarathi, Kalyani Bhole
- 연도: 2019
- 저널: 10th ICCCNT 2019
- DOI: 확인 불가
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/06_general_gait_and_other_knee_conditions/Gait Analysis Using IMU Sensor.pdf
- 분석 provider: antigravity

## 연구 목적

- 본 연구는 가속도계와 자이로스코프 센서(MPU6050)에서 획득한 각도를 활용하여 보행 중 각 보행 주기의 단계를 식별하기 위한 관성 측정 장치(IMU) 기반의 보행 분석 방법을 설명하는 것을 목적으로 한다. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “We described the IMU-based gait analysis method that uses angles obtained from an accelerometer and gyroscope sensor within MPU6050 to identify phases of each gait cycle during walking.”

## 연구 설계와 대상

- 실험을 위해 6명의 사람들로 구성된 건강한 피험자 그룹을 선정하였으며, 이들에게 직선 복도의 40미터 거리를 정상 속도로 보행하도록 설계하였다. _(근거: PAGE 3, III. SYSTEM DESIGN, C. Experimental setup)_
  - 근거 원문: “Data was collected from a group of six people was selected for gait analysis. They were asked to walk for 40 meters on a straight line at a normal speed.”

## 방법

- 양다리의 정강이(shank)에 부착한 두 개의 MPU6050 센서로부터 신호를 획득하였고, Arduino uno 마이크로컨트롤러와 HC-05 블루투스 모듈을 사용하여 데이터를 안드로이드 앱 'Blueterm'으로 무선 전송하여 저장하였으며, MATLAB 환경에서 보행 매개변수(stride time, stance time, step time, cadence 등)를 추출하기 위한 알고리즘을 개발해 가동하였다. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “We have used two MPU6050 sensors placed on a shank of both legs to collect gait signals. For experimental purpose, each participant was asked to walk for a distance of 40 meters for the straight corridor at a normal speed. These gait signal data measured by using Arduino uno micro-controller is then transmitted to an android app ‘Blueterm’ wirelessly via the HC-05 Bluetooth module. Collected data by the app is stored as a text file in a device containing app. This database is further processed to an algorithm that has been developed using MATLAB to extract a period of events that happened during walking such as stride time, stance time, step time, cadence, etc.”

## 핵심 결과

- 6명의 건강한 피험자로부터 보행 데이터를 획득하였고, 알고리즘을 통해 측정된 보행 매개변수 값들을 정상 성인의 표준 기준값들과 대조 검증하여 정상 성인의 표준 범위 보행 매개변수 값들과 부합함을 확인하였다. _(근거: PAGE 4, IV. RESULTS AND DISCUSSIONS)_
  - 근거 원문: “Measured gait parameters values validated against the standard values of the same for healthy adults. Our result value has matched with the standard gait parameters values as mentioned in II-B.”

## 저자 결론

- 확인된 내용 없음

## 연구의 한계

- 향후에는 무선 기반의 실시간 보행 특징 추출 자동화가 필요하며, 비정형(비정상) 보행 환자들을 대상으로 감지 민감도 및 유효성을 입증하기 위한 검증 단계를 추가로 수행해야 한다. _(근거: PAGE 5, VI. FUTURE SCOPE)_
  - 근거 원문: “Our future work will aim to make wireless, fully automatic real-time gait feature extraction system which can be used for a long duration to continuous monitoring and to identify the abnormal gait patterns for the assessment of elderly fall risk, rehabilitation, athlete’s application. The next step would be to validate this method on subjects with atypical gait and see if there is enough sensitivity to detect gait abnormalities.”

## 생각해볼 내용

- 제안한 보행 분석 시스템의 검증 과정에서 정밀 모션 캡처 시스템이나 힘판(force plate) 등 공인된 3D 표준 장비와의 동시 측정을 통한 정량적 오차나 오차 범위 분석을 거치지 않고, 문헌상 표준 성인의 기존 보행 범위 수치와 매칭되는지만 확인했기 때문에 시스템 고유의 기계적·생체역학적 유효성 및 정밀도 검증의 엄밀성이 다소 부족하다. _(근거: PAGE 4, IV. RESULTS AND DISCUSSIONS)_
  - 근거 원문: “Measured gait parameters values validated against the standard values of the same for healthy adults. Our result value has matched with the standard gait parameters values as mentioned in II-B.”

## 이 연구가 지적한 선행연구의 문제점

- 확인된 내용 없음

## 이 연구의 해결 방식과 기여

- > **[AS-IS]** 하체의 다수 분절에 센서를 부착하는 방식과 달리 양다리의 정강이(shank)에 단 두 개의 MPU6050 센서만을 위치시켜 보행 신호를 수집하고, 웨어러블 IMU 센서와 MATLAB 보행 단계 검출 알고리즘을 통해 수술 후 환자의 생체역학적 안정성 및 정형외과 및 재활 단계에서 환자의 예후를 저비용으로 간단하고 편리하게 평가할 수 있는 시스템을 구축하였다. _(근거: PAGE 1, Abstract)_
>
> **[TO-BE]** 하체의 다수 분절에 센서를 부착하는 기존 방식과 달리, 본 연구는 양다리 정강이에 두 개의 MPU6050 센서를 부착하고 MATLAB 기반 보행 단계 검출 알고리즘을 사용해 보행 매개변수를 정량화하는 웨어러블 IMU 시스템을 제시하였다.
>
> _(사실검증 — 근거불충분/중대: 요약 문장은 단 두 개의 센서 사용, 웨어러블 IMU 시스템, MATLAB 알고리즘, 정형외과·재활 모니터링 가능성은 원문과 부합한다. 그러나 제시된 근거 원문은 '저비용', '간단하고 편리하게', '수술 후 환자의 예후를 평가'하는 시스템을 구축했다는 표현 전체를 직접 지지하지 않는다. 저비용·단순 기술 개발은 결론부에 별도로 언급되지만, 환자 대상 예후 평가까지 실험적으로 구축·검증했다는 의미로 읽히면 원문보다 강하다.)_
  - 근거 원문: “Basically, this paper presents a wearable IMU sensor-based system and its associated gait analysis algorithm to obtain quantitative measurements of the individual’s gait parameters to monitor patient progress in orthopedics and rehabilitation.”

## 레퍼런스할 수 있는 내용

### 1. 보행 분석의 개념 정의

- 원문 발췌: “Gait analysis is the systemic study of locomotion of human being during walking [1].”
- 한국어 번역: 보행 분석은 보행 중인 인간의 이동 운동(locomotion)에 대한 체계적인 연구이다.
- 원문 위치: PAGE 1, I. INTRODUCTION
- 원문 내 인용표기: [1]
- 해당 선행문헌: [1] Whittle, M.W., 2007. Chap. 5 Application of gait analysis. In An introduction to gait analysis (p. 177). Butterworth-Heinemann.
- 주장 유형: background_citation
- 활용 맥락과 주의: 보행 분석(Gait analysis)의 학술적 개념 정의를 도입할 때 사용될 수 있으며, 1차 문헌(Whittle, 2007)을 확인하는 것이 권장된다.

### 2. 보행 양상 변화의 생체역학적 의미

- 원문 발췌: “Changes in gait styles implies important information of a person’s fitness that would be used for assessing or analysis individuals with pathological situations that have an effect on their ability to walk and the complete biomechanic system [3].”
- 한국어 번역: 보행 스타일의 변화는 보행 능력과 전반적인 생체역학적 계에 영향을 미치는 병리적 상태를 지닌 피험자들을 분석하거나 평가하는 데 적용 가능한 건강 상태 정보를 내포한다.
- 원문 위치: PAGE 1, I. INTRODUCTION
- 원문 내 인용표기: [3]
- 해당 선행문헌: [3] Ahmadi, A., Destelle, F., Unzueta, L., Monaghan, D.S., Linaza, M.T., Moran, K. and O’Connor, N.E., 2016. 3D human gait reconstruction and monitoring using body-worn inertial sensors and kinematic modeling. IEEE Sensors Journal, 16(24), pp.8823-8831.
- 주장 유형: background_citation
- 활용 맥락과 주의: 보행 양상의 변화가 임상학적 피트니스 평가 및 생체역학 전반을 분석하는 데 중요한 임상적 지표가 됨을 논리적으로 뒷받침할 때 인용될 수 있다.

### 3. 정강이 부착 자이로센서 측정 유효성

- 원문 발췌: “The results of the experiment demonstrate that the chosen sensor is effective for the data acquisition & it improves the accuracy for gait analysis.”
- > **[AS-IS]** 한국어 번역: 실험 결과를 통해 선택된 센서가 데이터 획득에 효과적이며 보행 분석의 정확도를 개선함을 검증하였다.
>
> **[TO-BE]** 실험 결과는 선택된 센서가 데이터 획득에 효과적이며 보행 분석 정확도 향상에 도움이 된다고 저자들이 결론내렸음을 보여준다.
>
> _(사실검증 — 과장/중대: 원문은 선택한 센서가 데이터 획득에 효과적이고 보행 분석 정확도를 개선한다고 결론적으로 서술하지만, 요약의 '검증하였다'는 표현은 정밀 기준 장비와의 비교나 정량적 정확도 검증이 수행된 것처럼 강하게 해석될 수 있다. 원문 결과는 정상 성인 표준값과의 부합을 보고한 수준이다.)_
- 원문 위치: PAGE 4, V. CONCLUSION
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- > **[AS-IS]** 활용 맥락과 주의: 정강이에 MPU6050 센서를 부착해 분석하는 방식의 데이터 수집 유효성 및 보행 매개변수 정확도 입증의 기초 근거로 사용할 수 있다.
>
> **[TO-BE]** 정강이에 MPU6050 센서를 부착해 보행 매개변수를 산출하고, 6명의 건강한 피험자 결과가 문헌상 정상 성인 표준값과 부합했다는 제한적 근거로 사용할 수 있다.
>
> _(사실검증 — 과장/중대: 원문은 6명의 건강한 피험자에서 산출된 보행 매개변수가 표준값과 맞았다고 보고하지만, '정확도 입증'이라고 하면 독립 기준 장비 대비 정확도 검증을 완료한 근거처럼 확장된다. 원문에는 모션캡처·힘판 등 기준 장비와의 오차 분석이 제시되지 않는다.)_
