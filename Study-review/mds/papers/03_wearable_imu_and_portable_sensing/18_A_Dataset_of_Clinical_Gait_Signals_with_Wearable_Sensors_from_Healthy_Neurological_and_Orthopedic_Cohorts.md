# A Dataset of Clinical Gait Signals with Wearable Sensors from Healthy, Neurological, and Orthopedic Cohorts

Voisard, C., Barrois, R., del’Escalopier, N., Vayatis, N., Vidal, P.-P., Yelnik, A., Ricard, D., & Oudre, L. (2025). A Dataset of Clinical Gait Signals with Wearable Sensors from Healthy, Neurological, and Orthopedic Cohorts. Scientific Data, 12, 1674. https://doi.org/10.1038/s41597-025-05959-w

## 서지정보

- 저자: Cyril Voisard, Rémi Barrois, Nicolas del’Escalopier, Nicolas Vayatis, Pierre-Paul Vidal, Alain Yelnik, Damien Ricard & Laurent Oudre
- 연도: 2025
- 저널: Scientific Data
- DOI: https://doi.org/10.1038/s41597-025-05959-w
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/03_wearable_imu_and_portable_sensing/A Dataset of Clinical Gait Signals with Wearable Sensors from Healthy, Neurological, and Orthopedic Cohorts.pdf
- 분석 provider: antigravity

## 연구 목적

- 본 연구는 대규모, 다중 병리 설계, 표준화된 임상 주석 및 보행 장애의 다양한 표현을 특징으로 하는 대규모 관성 보행 데이터베이스를 소개하는 것을 목적으로 한다. _(근거: --- PAGE 2 ---, Background & Summary)_
  - 근거 원문: “This paper introduces a large inertial gait database that aligns with these initiatives, which stands out due
to its large-scale, multi-pathology design, standardized clinical annotations, and diverse representation of
gait impairments, enabling robust cross-pathology comparisons and machine learning applications.”
- 오픈 액세스의 정제되고 주석이 달린 데이터베이스를 제공하여 관성 센서를 이용한 미래의 보행 정량화 분야 발전에 기여하고자 한다. _(근거: --- PAGE 1 ---, Abstract)_
  - 근거 원문: “Thisdatasetcan
beusedtostudykinematicparameters,gaitcyclestimeseries,andvariousindicatorsforquantifying
gaitinroutineclinicalpractice.”

## 연구 설계와 대상

- 연구에는 건강한 대조군 73명, 정형외과 환자 44명(고관절 골관절염, 슬관절 골관절염, 전방십자인대 손상), 신경과 환자 143명(뇌졸중, 파킨슨병, 화학요법 유발 말초신경병증, 방사선 유발 백질뇌증)을 포함하여 총 260명의 참가자가 모집되었다. _(근거: --- PAGE 2 ---, Methods, Participants and pathologies)_
  - 근거 원문: “The 73 healthy subjects (HS, 41 males and 32 females
aged18to87)reportednomedicalimpairmentandwereconsideredhealthyafteraclinicalexaminationbymed-
icaldoctors.Forpatients,twogroupswereformedamongthosehospitalizedinneurology,physicalmedicineand
rehabilitation, or orthopedic surgery departments. The orthopedic group included 44 subjects (21 males and 23
females) and was divided into three cohorts: hip osteoarthritis (HOA, aged 36 to 89), knee osteoarthritis (KOA,
aged 43 to 90) and anterior cruciate ligament injury (ACL, aged 22 to 64). The neurological group included 143
subjects (92 males and 51 females) and was divided into four cohorts: cerebrovascular accident (CVA, aged 41
to 83), Parkinson’s disease (PD, aged 55 to 90), chemotherapy-induced peripheral neuropathy (CIPN, aged 42
to 81) and radiation-induced leukoencephalopathy (84).”
- 각 코호트 간의 연령 차이는 모집 편향 때문이 아니라 포함된 질환군의 실제 역학적 프로파일을 반영한 결과다. _(근거: --- PAGE 2 ---, Methods, Participants and pathologies)_
  - 근거 원문: “Age differences between cohorts reflect the
coherent epidemiological profiles of the included pathological groups (e.g., degenerative neurological disorders
vs. acute orthopedic conditions) and were not due to recruitment bias.”

## 방법

- 참가자의 머리, 허리 아래(L4/L5), 그리고 양쪽 발등에 총 4개의 IMU 센서(XSens™ 또는 Technoconcept®)를 부착하여 데이터를 기록했다. _(근거: --- PAGE 2 ---, Methods, Recording devices)_
  - 근거 원문: “Four IMU devices were attached to the head (HE), lower back L4/L5 (LB), and on the
dorsalfaceofeachfoot(LFfortheleftfootandRFfortherightfoot)oftheparticipants.”
- 보행 평가 프로토콜은 6m에서 10m 사이의 직선 보행 후 180도 회전하여 출발점으로 다시 돌아오는 단거리 왕복 보행 테스트로 구성되었다. _(근거: --- PAGE 3 ---, Methods, Protocol)_
  - 근거 원문: “The gait quantification test consisted in a short straight walk test between 6m and 10m with a 180°
turnatthehalfwaypointandareturn.”
- 중력의 영향을 제거하고 보행 가속도 성분만을 분리하기 위해 보행 전 최소 2초간 정지해 있는 정적 단계 동안의 가속도 평균 벡터를 빼는 방식으로 전처리를 수행했고, 자이로스코프 오프셋 역시 동일한 정적 단계의 평균값을 사용해 보정했다. _(근거: --- PAGE 3 ---, Methods, Data processing)_
  - 근거 원문: “• The subject is assumed to be motionless for at least the first 2 seconds. To correct for gravitational effects and
isolate gait acceleration, the acceleration signals were processed by subtracting the static acceleration vector
estimated during this pre-walking static phase from the entire signal. Gyroscope offsets were similarly cor-
rected using the mean static phase value.”
- 처리된 신호의 품질을 더욱 높이기 위해 차단 주파수 14 Hz의 8차 저역통과 버터워스 필터를 데이터에 적용했다. _(근거: --- PAGE 4 ---, Methods, Data processing)_
  - 근거 원문: “• Toimprovethequalityoftheprocessedsignal,alow-passButterworthfilteroforder8withacutofffrequency
of 14Hz is applied. This filter setting is consistent with the trends reported in the literature48
.”
- 센서 연결이 1.5초 미만으로 짧게 일시 단절되어 발생한 결측 데이터 구간은 선형 보간법을 통해 보정하여 결측치를 채웠다. _(근거: --- PAGE 3 ---, Methods, Data processing)_
  - 근거 원문: “• Missing data correspond to brief interruptions (less than 1.5 seconds) in the connection between the sensors
and the control unit. In the processed signal, they were completed by linear interpolation.”

## 핵심 결과

- 건강한 대조군 코호트의 시공간적 보행 변수 분석 결과는 기존 문헌들의 표준적인 정상치 값들과 통계적으로 유의미한 차이를 보이지 않아 외적 타당성이 검증되었다. _(근거: --- PAGE 11 ---, Technical Validation, Gait parameters validation)_
  - 근거 원문: “Statistically, the healthy cohort’s results align closely
with normative literature values, showing no significant differences compared to published norms (p > 0.05,
independenttwo-samplet-tests)3,55
(seeTable11).Itenhancestheexternalvalidityofourdataset.”
- 초기 정적인 자세에서 측정된 가속도 센서 평균값이 중력 가속도인 9.95 ± 0.35 m/s²에 가깝고, 정지 시의 자이로스코프 변동이 운동 중에 관찰된 평균값의 4% 미만으로 측정되어 센서가 잘 캘리브레이션되었음을 보여준다. _(근거: --- PAGE 9 ---, Technical Validation, Protocol and sensors validation)_
  - 근거 원문: “Thus,the sensors used were well-calibrated, with still acceleration at the beginning of each trial correspond-
ing to gravity (mean 9.95 ± 0.35 m/s2
) and gyration below 4% of the mean gyration observed during movement.”

## 저자 결론

- 수집된 여러 대상자 집단은 다차원적인 분석 관점을 제공하여 근골격계와 중추 및 말초 신경계 사이의 복잡한 상호 작용을 탐색하게 해주며, 보행 장애 연구 및 개인 맞춤형 후속 치료 개발의 임상적 및 과학적 가치를 극대화한다. _(근거: --- PAGE 2 ---, Background & Summary)_
  - 근거 원문: “Together, these populations offer a multidimensional perspective, allowing the exploration of not only the
specificities of each pathology but also the complex interactions between the musculoskeletal, central, and
peripheral nervous systems. This complementarity enhances the clinical and scientific relevance of the dataset
for studying gait disorders and developing personalized follow-up.”

## 연구의 한계

- 건강한 대조군이 주로 병원 방문객 중 자원봉사자로 모집되어 선택 편향이 발생할 수 있고, 이에 따라 연구 결과를 일반 인구 집단으로 즉시 일반화하기에 한계가 존재할 수 있다. _(근거: --- PAGE 12 ---, Technical Validation, Limitations)_
  - 근거 원문: “One limitation of the dataset51
concerns the recruitment of healthy control participants.
Specifically, healthy individuals were primarily recruited among hospital visitors, which may introduce a selec-
tion bias and limit the generalizability of the findings to the broader population.”
- 임상 환경 조건의 제약으로 인하여 보행 경로의 길이(6~10m)에 다소의 변동이 있으며, 이는 시공간 보행 매개변수의 계산 방식에 영향을 미치지는 않으나, 보행 개시 후 감속하기 전의 정상 상태 보행(steady-state walking) 구간 길이를 축소시켜 매개변수 결과값에 가변성을 가중시킬 가능성이 있다. _(근거: --- PAGE 12 ---, Technical Validation, Limitations)_
  - 근거 원문: “Another key consideration when using this dataset is the variability in the straight walking path length (6
to 10 meters), constrained by the clinical environment. While this variability does not significantly affect the
calculationmethodofspatiotemporalparameters,itmayreducethedurationofthesteady-statewalkingphase,
after initiation and before deceleration. This could introduce additional variability in parameters value.”

## 생각해볼 내용

- 본 연구에서 구축한 대규모 데이터셋은 임상적 유용성을 극대화하기 위하여 각 병리별로 환자의 심각도를 평가할 수 있는 가장 적절한 임상적 또는 방사선학적 평가지표(WOMAC, FMA-LE, UPDRS III 등)를 의사가 수집해 함께 기록했다는 측면에서 설계의 완성도가 매우 뛰어나다. _(근거: --- PAGE 1 ---, Abstract)_
  - 근거 원문: “Foreachpathology,themostrelevantclinicalor
radioclinicalscorehasbeencalculatedtoprovideinsightintothegravityofthedisease.”

## 이 연구가 지적한 선행연구의 문제점

- 기존에 공개된 보행 데이터베이스들은 디지털 헬스 분야의 중개 연구나 완벽한 보행 분석을 하기 위해 요구되는 데이터 규모, 다양성 또는 깊이 있는 임상적 정보(메타데이터)가 부족한 한계가 있었다. _(근거: --- PAGE 2 ---, Background & Summary)_
  - 근거 원문: “However, these databases often lack either the scale, diversity, or clinical depth necessary for translational
research in digital health and complete gait analysis.”
- 다양한 보행 평가 프로토콜과 센서 유형, 수량, 장착 위치의 다양성은 대규모의 엄격하게 큐레이션된 메타 분석용 통합 데이터베이스를 구축하는 데 있어 커다란 기술적 장벽으로 작용해왔다. _(근거: --- PAGE 2 ---, Background & Summary)_
  - 근거 원문: “Theseprotocols,whetherconductedin
real-worldconditions32,33
orathome34,35
,alongwiththediversityinthetype,number,andplacementofsensors36
,
set challenges in establishing large, rigorously curated meta-analysis databases.”

## 이 연구의 해결 방식과 기여

- 본 연구에서 구축한 데이터베이스는 다중 질환 집단을 포함하는 대규모 설계와 표준화된 임상 주석, 다양한 보행 장애 수준을 아우르는 표현형을 확보하여 신뢰도 높은 교차 병리 비교 연구와 기계 학습 알고리즘 응용 연구의 초석을 닦았다. _(근거: --- PAGE 2 ---, Background & Summary)_
  - 근거 원문: “This paper introduces a large inertial gait database that aligns with these initiatives, which stands out due
to its large-scale, multi-pathology design, standardized clinical annotations, and diverse representation of
gait impairments, enabling robust cross-pathology comparisons and machine learning applications.”
- 다양한 질환을 앓고 있는 전체 환자 코호트를 포함함으로써 보행 장애가 매우 심하게 변형된 보행 데이터를 풍부하게 확보하였고, 이를 통해 보행 분할 및 탐지 알고리즘들의 성능을 한계까지 실증하고 고도화할 수 있는 기회를 제공한다. _(근거: --- PAGE 2 ---, Background & Summary)_
  - 근거 원문: “The patient
cohorts as a whole provide a huge range of gaits, some of which are severely altered, enabling the segmentation
and detection algorithms to be put to the test.”

## 레퍼런스할 수 있는 내용

### 1. 보행 분석의 학술적/의학적 관심 증가

- 원문 발췌: “Thestudyofgaitanalysishasincreasedexponentiallyinmedicalandpreventiveinterestduetoitscriticalrolein
understanding various physiological functions and pathologies1
.”
- 한국어 번역: 보행 분석 연구는 다양한 생리적 기능과 병리를 이해하는 데 중요한 역할로 인해 의학 및 예방 분야에서 기하급수적으로 관심이 증가해 왔다.
- 원문 위치: --- PAGE 1 ---, Background & Summary
- 원문 내 인용표기: 1
- 해당 선행문헌: 1. Eskofier, B. M. et al. An Overview of Smart Shoes in the Internet of Health Things: Gait and Mobility Assessment in Health
Promotion and Disease Monitoring. Applied Sciences https://doi.org/10.3390/app7100986 (2017).
- 주장 유형: background_citation
- 활용 맥락과 주의: 보행 분석이 다양한 생리적 기능과 병리 연구에서 의학적으로 급격히 중요해지고 있음을 서론에서 선행 연구 근거로 언급할 때 인용할 수 있다.

### 2. IMU 센서의 보행 분석 타당성 검증

- 원문 발췌: “Numerous validation studies
have shown IMUs can provide equivalent accuracy in detecting gait kinematics traditional motion capture sys-
tems10,11
.”
- 한국어 번역: 수많은 타당성 검증 연구에 따르면 IMU는 전통적인 모션 캡처 시스템과 대등한 정확도로 보행 운동학을 감지할 수 있다.
- 원문 위치: --- PAGE 1 ---, Background & Summary
- 원문 내 인용표기: 10,11
- 해당 선행문헌: 10. Kanzler, C. M. et al. Inertial sensor based and shoe size independent gait analysis including heel and toe clearance estimation. 2015
37th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC) https://doi.org/10.1109/
EMBC.2015.7319618 (2015).
11. Wagstaff, B., Peretroukhin, V. & Kelly, J. Robust Data-Driven Zero-Velocity Detection for Foot-Mounted Inertial Navigation. IEEE
Sensors Journal https://doi.org/10.1109/JSEN.2019.2944412 (2020).
- 주장 유형: background_citation
- 활용 맥락과 주의: IMU 센서가 기존 3차원 광학 모션 캡처 시스템과 비교하여 신뢰할 수 있는 정확도를 제공함을 서론에서 입증할 때 인용할 수 있다.

### 3. 건강한 대조군 보행 매개변수의 타당성

- 원문 발췌: “Statistically, the healthy cohort’s results align closely
with normative literature values, showing no significant differences compared to published norms (p > 0.05,
independenttwo-samplet-tests)3,55
(seeTable11).Itenhancestheexternalvalidityofourdataset.”
- 한국어 번역: 통계적으로 건강한 코호트의 결과는 기존 문헌의 규범 값들과 밀접하게 일치하며, 발표된 규범들과 비교하여 유의미한 차이를 보이지 않아(p > 0.05, 독립 이표본 t-검정) 데이터셋의 외적 타당성을 향상시킨다.
- 원문 위치: --- PAGE 11 ---, Technical Validation, Gait parameters validation
- 원문 내 인용표기: 3,55
- 해당 선행문헌: 3. Voisard,C.etal.InnovativemultidimensionalgaitevaluationusingIMUinmultiplesclerosis:introducingthesemiogram.Frontiers
in Neurology 14, 1237162, https://doi.org/10.3389/fneur.2023.1237162 (2023).
55. Latorre, J., Colomer, C., Alcañiz Raya, M. & Llorens, R. Gait analysis with the Kinect v2: Normative study with healthy individuals
and comprehensive study of its sensitivity, validity, and reliability in individuals with stroke. Journal of NeuroEngineering and
Rehabilitation https://doi.org/10.1186/s12984-019-0568-y (2019).
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: 본 연구에서 측정된 건강한 성인의 보행 매개변수 결과가 기존 문헌들의 표준값과 비교했을 때 통계적으로 유의미한 차이가 없는 신뢰할 수 있는 수준임을 주장할 때 사용된다.

### 4. 병원 환경 내 직선 보행 거리 가변성이 보행 정상상태 단계에 미치는 영향

- 원문 발췌: “While this variability does not significantly affect the
calculationmethodofspatiotemporalparameters,itmayreducethedurationofthesteady-statewalkingphase,
after initiation and before deceleration. This could introduce additional variability in parameters value.”
- 한국어 번역: 이러한 가변성이 시공간 매개변수의 계산 방법에는 유의미한 영향을 미치지 않지만, 보행 개시 후 및 감속 전의 정상 상태 보행 단계의 기간을 단축시킬 수 있다. 이는 매개변수 값에 추가적인 가변성을 초래할 수 있다.
- 원문 위치: --- PAGE 12 ---, Technical Validation, Limitations
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: uncited_author_claim
- 활용 맥락과 주의: 임상 환경의 제약으로 인한 직선 보행 거리(6~10m)의 가변성이 보행 개시 및 감속기 사이의 정상 보행 상태 구간을 줄이고, 이로 인해 보행 지표 변동성이 증가할 가능성이 있음을 논의할 때 사용될 수 있다.
