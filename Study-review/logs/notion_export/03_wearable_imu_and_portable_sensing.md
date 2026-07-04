# [18] A Dataset of Clinical Gait Signals with Wearable Sensors from Healthy, Neurological, and Orthopedic Cohorts

(저자: Cyril Voisard, Rémi Barrois, Nicolas del’Escalopier, Nicolas Vayatis, Pierre-Paul Vidal, Alain Yelnik, Damien Ricard & Laurent Oudre | 연도: 2025 | 저널: Scientific Data | DOI: https://doi.org/10.1038/s41597-025-05959-w)

Voisard, C., Barrois, R., del’Escalopier, N., Vayatis, N., Vidal, P.-P., Yelnik, A., Ricard, D., & Oudre, L. (2025). A Dataset of Clinical Gait Signals with Wearable Sensors from Healthy, Neurological, and Orthopedic Cohorts. Scientific Data, 12, 1674. https://doi.org/10.1038/s41597-025-05959-w

## 서지정보

- 저자: Cyril Voisard, Rémi Barrois, Nicolas del’Escalopier, Nicolas Vayatis, Pierre-Paul Vidal, Alain Yelnik, Damien Ricard & Laurent Oudre
- 연도: 2025
- 저널: Scientific Data
- DOI: https://doi.org/10.1038/s41597-025-05959-w
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/03_wearable_imu_and_portable_sensing/A Dataset of Clinical Gait Signals with Wearable Sensors from Healthy, Neurological, and Orthopedic Cohorts.pdf
- 분석 provider: antigravity

## 연구 목적

- 본 연구는 대규모, 다중 병리 설계, 표준화된 임상 주석 및 보행 장애의 다양한 표현을 특징으로 하는 대규모 관성 보행 데이터베이스를 소개하는 것을 목적으로 한다. *(근거: --- PAGE 2 ---, Background & Summary)*
	- 근거 원문: “This paper introduces a large inertial gait database that aligns with these initiatives, which stands out due to its large-scale, multi-pathology design, standardized clinical annotations, and diverse representation of gait impairments, enabling robust cross-pathology comparisons and machine learning applications.”
- 오픈 액세스의 정제되고 주석이 달린 데이터베이스를 제공하여 관성 센서를 이용한 미래의 보행 정량화 분야 발전에 기여하고자 한다. *(근거: --- PAGE 1 ---, Abstract)*
	- 근거 원문: “Thisdatasetcan beusedtostudykinematicparameters,gaitcyclestimeseries,andvariousindicatorsforquantifying gaitinroutineclinicalpractice.”

## 연구 설계와 대상

- 연구에는 건강한 대조군 73명, 정형외과 환자 44명(고관절 골관절염, 슬관절 골관절염, 전방십자인대 손상), 신경과 환자 143명(뇌졸중, 파킨슨병, 화학요법 유발 말초신경병증, 방사선 유발 백질뇌증)을 포함하여 총 260명의 참가자가 모집되었다. *(근거: --- PAGE 2 ---, Methods, Participants and pathologies)*
	- 근거 원문: “The 73 healthy subjects (HS, 41 males and 32 females aged18to87)reportednomedicalimpairmentandwereconsideredhealthyafteraclinicalexaminationbymed- icaldoctors.Forpatients,twogroupswereformedamongthosehospitalizedinneurology,physicalmedicineand rehabilitation, or orthopedic surgery departments. The orthopedic group included 44 subjects (21 males and 23 females) and was divided into three cohorts: hip osteoarthritis (HOA, aged 36 to 89), knee osteoarthritis (KOA, aged 43 to 90) and anterior cruciate ligament injury (ACL, aged 22 to 64). The neurological group included 143 subjects (92 males and 51 females) and was divided into four cohorts: cerebrovascular accident (CVA, aged 41 to 83), Parkinson’s disease (PD, aged 55 to 90), chemotherapy-induced peripheral neuropathy (CIPN, aged 42 to 81) and radiation-induced leukoencephalopathy (84).”
- 각 코호트 간의 연령 차이는 모집 편향 때문이 아니라 포함된 질환군의 실제 역학적 프로파일을 반영한 결과다. *(근거: --- PAGE 2 ---, Methods, Participants and pathologies)*
	- 근거 원문: “Age differences between cohorts reflect the coherent epidemiological profiles of the included pathological groups (e.g., degenerative neurological disorders vs. acute orthopedic conditions) and were not due to recruitment bias.”

## 방법

- 참가자의 머리, 허리 아래(L4/L5), 그리고 양쪽 발등에 총 4개의 IMU 센서(XSens™ 또는 Technoconcept®)를 부착하여 데이터를 기록했다. *(근거: --- PAGE 2 ---, Methods, Recording devices)*
	- 근거 원문: “Four IMU devices were attached to the head (HE), lower back L4/L5 (LB), and on the dorsalfaceofeachfoot(LFfortheleftfootandRFfortherightfoot)oftheparticipants.”
- 보행 평가 프로토콜은 6m에서 10m 사이의 직선 보행 후 180도 회전하여 출발점으로 다시 돌아오는 단거리 왕복 보행 테스트로 구성되었다. *(근거: --- PAGE 3 ---, Methods, Protocol)*
	- 근거 원문: “The gait quantification test consisted in a short straight walk test between 6m and 10m with a 180° turnatthehalfwaypointandareturn.”
- 중력의 영향을 제거하고 보행 가속도 성분만을 분리하기 위해 보행 전 최소 2초간 정지해 있는 정적 단계 동안의 가속도 평균 벡터를 빼는 방식으로 전처리를 수행했고, 자이로스코프 오프셋 역시 동일한 정적 단계의 평균값을 사용해 보정했다. *(근거: --- PAGE 3 ---, Methods, Data processing)*
	- 근거 원문: “• The subject is assumed to be motionless for at least the first 2 seconds. To correct for gravitational effects and isolate gait acceleration, the acceleration signals were processed by subtracting the static acceleration vector estimated during this pre-walking static phase from the entire signal. Gyroscope offsets were similarly cor- rected using the mean static phase value.”
- 처리된 신호의 품질을 더욱 높이기 위해 차단 주파수 14 Hz의 8차 저역통과 버터워스 필터를 데이터에 적용했다. *(근거: --- PAGE 4 ---, Methods, Data processing)*
	- 근거 원문: “• Toimprovethequalityoftheprocessedsignal,alow-passButterworthfilteroforder8withacutofffrequency of 14Hz is applied. This filter setting is consistent with the trends reported in the literature48 .”
- 센서 연결이 1.5초 미만으로 짧게 일시 단절되어 발생한 결측 데이터 구간은 선형 보간법을 통해 보정하여 결측치를 채웠다. *(근거: --- PAGE 3 ---, Methods, Data processing)*
	- 근거 원문: “• Missing data correspond to brief interruptions (less than 1.5 seconds) in the connection between the sensors and the control unit. In the processed signal, they were completed by linear interpolation.”

## 핵심 결과

- 건강한 대조군 코호트의 시공간적 보행 변수 분석 결과는 기존 문헌들의 표준적인 정상치 값들과 통계적으로 유의미한 차이를 보이지 않아 외적 타당성이 검증되었다. *(근거: --- PAGE 11 ---, Technical Validation, Gait parameters validation)*
	- 근거 원문: “Statistically, the healthy cohort’s results align closely with normative literature values, showing no significant differences compared to published norms (p \> 0.05, independenttwo-samplet-tests)3,55 (seeTable11).Itenhancestheexternalvalidityofourdataset.”
- 초기 정적인 자세에서 측정된 가속도 센서 평균값이 중력 가속도인 9.95 ± 0.35 m/s²에 가깝고, 정지 시의 자이로스코프 변동이 운동 중에 관찰된 평균값의 4% 미만으로 측정되어 센서가 잘 캘리브레이션되었음을 보여준다. *(근거: --- PAGE 9 ---, Technical Validation, Protocol and sensors validation)*
	- 근거 원문: “Thus,the sensors used were well-calibrated, with still acceleration at the beginning of each trial correspond- ing to gravity (mean 9.95 ± 0.35 m/s2 ) and gyration below 4% of the mean gyration observed during movement.”

## 저자 결론
 \> **[AS-IS]** - 수집된 여러 대상자 집단은 다차원적인 분석 관점을 제공하여 근골격계와 중추 및 말초 신경계 사이의 복잡한 상호 작용을 탐색하게 해주며, 보행 장애 연구 및 개인 맞춤형 후속 치료 개발의 임상적 및 과학적 가치를 극대화한다. *(근거: --- PAGE 2 ---, Background & Summary)*<br>**[TO-BE]** 수집된 여러 대상자 집단은 다차원적인 분석 관점을 제공하여 근골격계와 중추 및 말초 신경계 사이의 복잡한 상호 작용을 탐색하게 해주며, 보행 장애 연구와 개인 맞춤형 추적 관찰 개발을 위한 데이터셋의 임상적·과학적 관련성을 높인다.<br>*(사실검증 — 과장/경미: 원문은 데이터셋의 임상적·과학적 관련성을 'enhances'한다고 표현하지만, 요약은 '가치를 극대화한다'고 하여 원문보다 강한 효과를 단정한다.)*
	- 근거 원문: “Together, these populations offer a multidimensional perspective, allowing the exploration of not only the specificities of each pathology but also the complex interactions between the musculoskeletal, central, and peripheral nervous systems. This complementarity enhances the clinical and scientific relevance of the dataset for studying gait disorders and developing personalized follow-up.”

## 연구의 한계

- 건강한 대조군이 주로 병원 방문객 중 자원봉사자로 모집되어 선택 편향이 발생할 수 있고, 이에 따라 연구 결과를 일반 인구 집단으로 즉시 일반화하기에 한계가 존재할 수 있다. *(근거: --- PAGE 12 ---, Technical Validation, Limitations)*
	- 근거 원문: “One limitation of the dataset51 concerns the recruitment of healthy control participants. Specifically, healthy individuals were primarily recruited among hospital visitors, which may introduce a selec- tion bias and limit the generalizability of the findings to the broader population.”
- 임상 환경 조건의 제약으로 인하여 보행 경로의 길이(6\~10m)에 다소의 변동이 있으며, 이는 시공간 보행 매개변수의 계산 방식에 영향을 미치지는 않으나, 보행 개시 후 감속하기 전의 정상 상태 보행(steady-state walking) 구간 길이를 축소시켜 매개변수 결과값에 가변성을 가중시킬 가능성이 있다. *(근거: --- PAGE 12 ---, Technical Validation, Limitations)*
	- 근거 원문: “Another key consideration when using this dataset is the variability in the straight walking path length (6 to 10 meters), constrained by the clinical environment. While this variability does not significantly affect the calculationmethodofspatiotemporalparameters,itmayreducethedurationofthesteady-statewalkingphase, after initiation and before deceleration. This could introduce additional variability in parameters value.”

## 생각해볼 내용
 \> **[AS-IS]** - 본 연구에서 구축한 대규모 데이터셋은 임상적 유용성을 극대화하기 위하여 각 병리별로 환자의 심각도를 평가할 수 있는 가장 적절한 임상적 또는 방사선학적 평가지표(WOMAC, FMA-LE, UPDRS III 등)를 의사가 수집해 함께 기록했다는 측면에서 설계의 완성도가 매우 뛰어나다. *(근거: --- PAGE 1 ---, Abstract)*<br>**[TO-BE]** 본 연구 데이터셋은 각 병리별 질환 중증도에 대한 정보를 제공하기 위해 관련 임상 또는 방사선임상 점수(WOMAC, FMA-LE, UPDRS III 등)를 함께 포함한다.<br>*(사실검증 — 과장/경미: 원문은 각 병리에 관련 임상 또는 방사선임상 점수가 계산되었다고만 하며, '임상적 유용성 극대화'나 '설계의 완성도가 매우 뛰어나다'는 평가는 원문보다 강한 분석자의 판단이다.)*
	- 근거 원문: “Foreachpathology,themostrelevantclinicalor radioclinicalscorehasbeencalculatedtoprovideinsightintothegravityofthedisease.”

## 이 연구가 지적한 선행연구의 문제점

- 기존에 공개된 보행 데이터베이스들은 디지털 헬스 분야의 중개 연구나 완벽한 보행 분석을 하기 위해 요구되는 데이터 규모, 다양성 또는 깊이 있는 임상적 정보(메타데이터)가 부족한 한계가 있었다. *(근거: --- PAGE 2 ---, Background & Summary)*
	- 근거 원문: “However, these databases often lack either the scale, diversity, or clinical depth necessary for translational research in digital health and complete gait analysis.”
- 다양한 보행 평가 프로토콜과 센서 유형, 수량, 장착 위치의 다양성은 대규모의 엄격하게 큐레이션된 메타 분석용 통합 데이터베이스를 구축하는 데 있어 커다란 기술적 장벽으로 작용해왔다. *(근거: --- PAGE 2 ---, Background & Summary)*
	- 근거 원문: “Theseprotocols,whetherconductedin real-worldconditions32,33 orathome34,35 ,alongwiththediversityinthetype,number,andplacementofsensors36 , set challenges in establishing large, rigorously curated meta-analysis databases.”

## 이 연구의 해결 방식과 기여
 \> **[AS-IS]** - 본 연구에서 구축한 데이터베이스는 다중 질환 집단을 포함하는 대규모 설계와 표준화된 임상 주석, 다양한 보행 장애 수준을 아우르는 표현형을 확보하여 신뢰도 높은 교차 병리 비교 연구와 기계 학습 알고리즘 응용 연구의 초석을 닦았다. *(근거: --- PAGE 2 ---, Background & Summary)*<br>**[TO-BE]** 본 연구의 데이터베이스는 대규모 다중 병리 설계, 표준화된 임상 주석, 다양한 보행 장애 표현을 갖추어 교차 병리 비교와 기계학습 응용을 가능하게 한다.<br>*(사실검증 — 과장/경미: 원문은 이러한 설계가 견고한 교차 병리 비교와 기계학습 적용을 가능하게 한다고 했으나, 요약의 '초석을 닦았다'와 '신뢰도 높은'은 원문보다 평가적으로 강화된 표현이다.)*
	- 근거 원문: “This paper introduces a large inertial gait database that aligns with these initiatives, which stands out due to its large-scale, multi-pathology design, standardized clinical annotations, and diverse representation of gait impairments, enabling robust cross-pathology comparisons and machine learning applications.”
- 다양한 질환을 앓고 있는 전체 환자 코호트를 포함함으로써 보행 장애가 매우 심하게 변형된 보행 데이터를 풍부하게 확보하였고, 이를 통해 보행 분할 및 탐지 알고리즘들의 성능을 한계까지 실증하고 고도화할 수 있는 기회를 제공한다. *(근거: --- PAGE 2 ---, Background & Summary)*
	- 근거 원문: “The patient cohorts as a whole provide a huge range of gaits, some of which are severely altered, enabling the segmentation and detection algorithms to be put to the test.”

## 레퍼런스할 수 있는 내용

### 1. 보행 분석의 학술적/의학적 관심 증가

- 원문 발췌: “Thestudyofgaitanalysishasincreasedexponentiallyinmedicalandpreventiveinterestduetoitscriticalrolein understanding various physiological functions and pathologies1 .”
- 한국어 번역: 보행 분석 연구는 다양한 생리적 기능과 병리를 이해하는 데 중요한 역할로 인해 의학 및 예방 분야에서 기하급수적으로 관심이 증가해 왔다.
- 원문 위치: --- PAGE 1 ---, Background & Summary
- 원문 내 인용표기: 1
- 해당 선행문헌: 1. Eskofier, B. M. et al. An Overview of Smart Shoes in the Internet of Health Things: Gait and Mobility Assessment in Health Promotion and Disease Monitoring. Applied Sciences https://doi.org/10.3390/app7100986 (2017).
- 주장 유형: background_citation
- 활용 맥락과 주의: 보행 분석이 다양한 생리적 기능과 병리 연구에서 의학적으로 급격히 중요해지고 있음을 서론에서 선행 연구 근거로 언급할 때 인용할 수 있다.

### 2. IMU 센서의 보행 분석 타당성 검증

- 원문 발췌: “Numerous validation studies have shown IMUs can provide equivalent accuracy in detecting gait kinematics traditional motion capture sys- tems10,11 .” \> **[AS-IS]** - 한국어 번역: 수많은 타당성 검증 연구에 따르면 IMU는 전통적인 모션 캡처 시스템과 대등한 정확도로 보행 운동학을 감지할 수 있다. \> \> **[TO-BE]** 수많은 타당성 검증 연구는 IMU가 보행 운동학 감지에서 전통적 모션 캡처 시스템과 관련해 동등한 수준의 정확도를 제공할 수 있음을 보여주었다고 원문은 서술한다. \> \> *(사실검증 — 번역오류/경미: SOURCE_TEXT의 원문은 'detecting gait kinematics traditional motion capture systems'처럼 문법적으로 불완전하게 추출되어 있으며, '전통적인 모션 캡처 시스템과 대등한 정확도'라는 비교 의미는 문맥상 가능하지만 SOURCE_TEXT 문장 자체에 'with' 또는 'as'가 명시되어 있지 않다. 요약은 원문 추출 텍스트보다 더 명확한 비교문으로 보정했다.)*
- 원문 위치: --- PAGE 1 ---, Background & Summary
- 원문 내 인용표기: 10,11
- 해당 선행문헌: 10. Kanzler, C. M. et al. Inertial sensor based and shoe size independent gait analysis including heel and toe clearance estimation. 2015 37th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC) https://doi.org/10.1109/ EMBC.2015.7319618 (2015). 11. Wagstaff, B., Peretroukhin, V. & Kelly, J. Robust Data-Driven Zero-Velocity Detection for Foot-Mounted Inertial Navigation. IEEE Sensors Journal https://doi.org/10.1109/JSEN.2019.2944412 (2020).
- 주장 유형: background_citation
- 활용 맥락과 주의: IMU 센서가 기존 3차원 광학 모션 캡처 시스템과 비교하여 신뢰할 수 있는 정확도를 제공함을 서론에서 입증할 때 인용할 수 있다.

### 3. 건강한 대조군 보행 매개변수의 타당성

- 원문 발췌: “Statistically, the healthy cohort’s results align closely with normative literature values, showing no significant differences compared to published norms (p \> 0.05, independenttwo-samplet-tests)3,55 (seeTable11).Itenhancestheexternalvalidityofourdataset.”
- 한국어 번역: 통계적으로 건강한 코호트의 결과는 기존 문헌의 규범 값들과 밀접하게 일치하며, 발표된 규범들과 비교하여 유의미한 차이를 보이지 않아(p \> 0.05, 독립 이표본 t-검정) 데이터셋의 외적 타당성을 향상시킨다.
- 원문 위치: --- PAGE 11 ---, Technical Validation, Gait parameters validation
- 원문 내 인용표기: 3,55
- 해당 선행문헌: 3. Voisard,C.etal.InnovativemultidimensionalgaitevaluationusingIMUinmultiplesclerosis:introducingthesemiogram.Frontiers in Neurology 14, 1237162, https://doi.org/10.3389/fneur.2023.1237162 (2023). 55. Latorre, J., Colomer, C., Alcañiz Raya, M. & Llorens, R. Gait analysis with the Kinect v2: Normative study with healthy individuals and comprehensive study of its sensitivity, validity, and reliability in individuals with stroke. Journal of NeuroEngineering and Rehabilitation https://doi.org/10.1186/s12984-019-0568-y (2019).
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: 본 연구에서 측정된 건강한 성인의 보행 매개변수 결과가 기존 문헌들의 표준값과 비교했을 때 통계적으로 유의미한 차이가 없는 신뢰할 수 있는 수준임을 주장할 때 사용된다.

### 4. 병원 환경 내 직선 보행 거리 가변성이 보행 정상상태 단계에 미치는 영향

- 원문 발췌: “While this variability does not significantly affect the calculationmethodofspatiotemporalparameters,itmayreducethedurationofthesteady-statewalkingphase, after initiation and before deceleration. This could introduce additional variability in parameters value.”
- 한국어 번역: 이러한 가변성이 시공간 매개변수의 계산 방법에는 유의미한 영향을 미치지 않지만, 보행 개시 후 및 감속 전의 정상 상태 보행 단계의 기간을 단축시킬 수 있다. 이는 매개변수 값에 추가적인 가변성을 초래할 수 있다.
- 원문 위치: --- PAGE 12 ---, Technical Validation, Limitations
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: uncited_author_claim
- 활용 맥락과 주의: 임상 환경의 제약으로 인한 직선 보행 거리(6\~10m)의 가변성이 보행 개시 및 감속기 사이의 정상 보행 상태 구간을 줄이고, 이로 인해 보행 지표 변동성이 증가할 가능성이 있음을 논의할 때 사용될 수 있다.


---

# [19] A scoping review of portable sensing for out-of-lab anterior cruciate ligament injury prevention and rehabilitation

(저자: Tian Tan, Anthony A. Gatti, Bingfei Fan, Kevin G. Shea, Seth L. Sherman, Scott D. Uhlrich, Jennifer L. Hicks, Scott L. Delp, Peter B. Shull, Akshay S. Chaudhari | 연도: 2023 | 저널: npj Digital Medicine | DOI: https://doi.org/10.1038/s41746-023-00782-2)

Tan, T., Gatti, A. A., Fan, B., Shea, K. G., Sherman, S. L., Uhlrich, S. D., Hicks, J. L., Delp, S. L., Shull, P. B., & Chaudhari, A. S. (2023). A scoping review of portable sensing for out-of-lab anterior cruciate ligament injury prevention and rehabilitation. npj Digital Medicine, 6(1), 46. https://doi.org/10.1038/s41746-023-00782-2

## 서지정보

- 저자: Tian Tan, Anthony A. Gatti, Bingfei Fan, Kevin G. Shea, Seth L. Sherman, Scott D. Uhlrich, Jennifer L. Hicks, Scott L. Delp, Peter B. Shull, Akshay S. Chaudhari
- 연도: 2023
- 저널: npj Digital Medicine
- DOI: 10.1038/s41746-023-00782-2
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/03_wearable_imu_and_portable_sensing/A scoping review of portable sensing for out-of-lab anterior cruciate ligament injury prevention and rehabilitation.pdf
- 분석 provider: antigravity

## 연구 목적

- 본 종설의 목적은 실험실 외 환경에서 전방십자인대(ACL) 및 전방십자인대 재건술(ACLR)에 적용되는 휴대용 센싱 기술에 대한 연구를 요약하고, 향후 연구 및 개발을 위한 새로운 기회에 대한 저자들의 관점을 제시하는 것이다. *(근거: PAGE 1, Abstract)*
	- 근거 원문: “The purpose of this review is to summarize research on out-of-lab portable sensing applied to ACL and ACLR and offer our perspectives on new opportunities for future research and development.”

## 연구 설계와 대상

- 포함된 연구에서 모집된 피험자 수는 최소 9명에서 최대 169명이었으며, 중앙값은 24명이었다. *(근거: PAGE 5, RESULTS - Experimental design)*
	- 근거 원문: “The number of subjects recruited in the included studies ranged from 9 to 169, with the median being 24 (Fig. 4a).”
- 포함된 연구 중 21편(43%)은 ACLR 수술을 받은 환자를 피험자로 모집했다. *(근거: PAGE 5, RESULTS - Experimental design)*
	- 근거 원문: “Twenty-one studies (43%) recruited patients following ACLR.”

## 방법

- 검색 결과 1344개의 문헌이 검색되었고, 그 중 1990년부터 2022년 사이에 출판된 49개의 연구가 분석에 포함되었다. *(근거: PAGE 2, RESULTS)*
	- 근거 원문: “Our search yielded 1344 articles, of which 49 articles were included (Fig. 1), dating from 1990 to 2022.”
- 수집된 데이터의 생체역학적 매개변수 추정을 위한 방법론을 물리학 기반 모델링, 기계 학습 모델, 직접적인 특징 추출의 세 가지 범주로 분류하여 분석했다. *(근거: PAGE 5, RESULTS - Methodology of biomechanical parameter estimation)*
	- 근거 원문: “We categorized the methodologies for the analysis of the acquired data into three separate categories: (1) physics-based modeling that includes studies with kinematics reconstructed from raw sensor measurements and kinetics estimated via inverse dynamics or musculoskeletal models, (2) machine learning models to estimate subjects’ status or estimate parameters, and (3) direct feature extraction using investigator-deﬁned parameters from the raw sensor data.”
- 포함된 단면 연구들의 질적 수준을 평가하기 위해 AXIS 평가 도구를 적용했다. *(근거: PAGE 10, METHODS - Quality assessment)*
	- 근거 원문: “The Appraisal tool for Cross-Sectional Studies (AXIS) was used to assess the quality of the included studies130.”

## 핵심 결과

- 단일 센서 유형으로는 IMU가 가장 흔하게 사용되었으며(22%), 깊이 카메라(16%), RGB 카메라(8%), EMG(4%) 순이었다. *(근거: PAGE 2, RESULTS)*
	- 근거 원문: “IMUs were the most common sensor used in isolation (22%), followed by depth cameras (16%), RGB cameras (8%), and EMG (4%) (Fig. 2b).”
- 추정된 생체역학적 매개변수는 운동학적 매개변수가 71%로 가장 많았고, 시공간 매개변수(18%), 운동역학(12%), 근육 활성화(10%)가 그 뒤를 이었다. *(근거: PAGE 2, RESULTS)*
	- 근거 원문: “Kinematic parameters were the dominant target (71%), followed by spatiotemporal parameters (18%), kinetics (12%), and muscle activation (10%) (Fig. 2c).”
- 가장 흔한 분석 방식은 직접적인 특징 추출(37%)이었으며, 물리학 기반 모델링(24%), 기계 학습(22%) 순으로 뒤를 이었다. *(근거: PAGE 2, RESULTS)*
	- 근거 원문: “Direct feature extraction (37%) was the most common analysis approach, followed by physics-based modeling (24%), and machine learning (22%) (Fig. 2d).”
- 분석 대상이 된 포함 문헌의 72%가 분석적 검증(I단계) 수준이었고, 24%는 예비 임상 검증(II단계) 수준이었다. *(근거: PAGE 5, RESULTS - Readiness for deployment)*
	- 근거 원문: “72% of the included studies are in stage I, analytical validation, as they proposed a novel method in-laboratory and associated its outcome with ACL injury risk or rehabilitation status. 24% of the included studies are in stage II, preliminary clinical validation, as they demonstrated their clinical utility.”

## 저자 결론

- 휴대용 센싱은 재활 과정을 통해 환자의 진행 상황을 모니터링하고 운동선수가 부상 위험 요인을 줄이도록 훈련하는 데 잠재적으로 사용될 수 있다. *(근거: PAGE 9, DISCUSSION)*
	- 근거 원문: “Through these studies, we showed that portable sensing can potentially be used to monitor patient progress through the rehabilitation process and train athletes to reduce injury risk factors.”
- 그러나 유망한 결과에도 불구하고, 이러한 휴대용 센싱 기술들의 타당성과 신뢰성은 아직 명확하게 확립되지 않았다. *(근거: PAGE 9, DISCUSSION)*
	- 근거 원문: “However, despite their promising results, the validity and reliability of these portable sensing methods are not well-established.”

## 연구의 한계

- 포함된 연구들이 보고한 정확도 지표가 다르고, 대상 동작과 생체역학 매개변수도 다양하여 성능을 통계적으로 비교하거나 합성할 수 없다. *(근거: PAGE 9, DISCUSSION)*
	- 근거 원문: “We are unable to aggregate or statistically compare the performance of portable sensing approaches because the included studies reported different accuracy metrics and investigated different motions and biomechanical parameters.”
- 체계적 문헌고찰의 표준적인 PICO 프레임워크를 기반으로 임상적 연구 질문을 사전에 정의하지 않아 종설의 임상적 영향력이 일부 제한될 수 있다. *(근거: PAGE 9, DISCUSSION)*
	- 근거 원문: “Another limitation is that we did not formulate clinical research questions following the patient, intervention, comparison, outcome (PICO) framework128, which may limit the clinical impact of our review.”

## 생각해볼 내용

- 시간에 따른 생체역학적 변화를 추적하는 능력은 임상의의 재활 방식 수립에 정보를 제공하고 환자의 장기 참여를 독려하는 데 기여할 수 있다. *(근거: PAGE 8, DISCUSSION)*
	- 근거 원문: “Additionally, the ability to track biomechanical changes over time will both inform the rehabilitation approaches of clinicians and promote long-term patient engagement.”
- 부상 위험 요인을 조기에 스크리닝하고 차별화된 훈련 방식을 제공하기 위해 휴대용 센싱으로 전통적 변수를 대체하거나 새로운 위험 지표를 찾는 것은 연구 기회로서 가치가 크다. *(근거: PAGE 7, DISCUSSION)*
	- 근거 원문: “Using portable sensing to estimate traditional parameters or identify new parameters associated with ACL injury risk represents a signiﬁcant research opportunity.”

## 이 연구가 지적한 선행연구의 문제점

- 실험실 기반의 기존 생체역학적 평가는 부상 위험과 재활 진행률을 정밀하게 평가할 수 있으나, 측정 장비가 비싸고 대다수의 환자들이 쉽게 접근할 수 없다. *(근거: PAGE 1, Abstract)*
	- 근거 원문: “Laboratory-based biomechanical assessment can evaluate ACL injury risk and rehabilitation progress after ACLR; however, lab-based measurements are expensive and inaccessible to most people.”
- 기존의 무릎 건강 관련 종설들은 모든 휴대용 센싱 장치를 총망라하지 않고 웨어러블 센싱에만 집중했으며, 깊이 카메라 및 RGB 카메라에 대한 평가가 제외되었다. *(근거: PAGE 2, INTRODUCTION)*
	- 근거 원문: “Previous reviews report on wearable sensing (IMU, EMG, pedometer, goniometer, and pressure insole) for knee health, as opposed to all portable sensing studied in this review. In our manuscript, we also characterize depth cameras and RGB cameras.”
- 또한, 기존의 리뷰들은 전방십자인대 부상에 명확히 초점을 맞추지 않았거나 센서 비대칭성 평가와 같은 좁은 범위의 성능 평가에만 편중되어 있었다. *(근거: PAGE 2, INTRODUCTION)*
	- 근거 원문: “Also, prior reviews either did not speciﬁcally focus on ACL injury30,36–38, or were focused on a speciﬁc aspect of sensor performance such as asymmetry identiﬁcation39.”

## 이 연구의 해결 방식과 기여

- 본 연구는 측정 대상 운동, 센싱 하드웨어 구성, 데이터 분석 모델링, 임상 적용에 걸쳐 전방십자인대 평가 분야의 기존 휴대용 센서 문헌을 포괄적으로 종합하였다. *(근거: PAGE 2, INTRODUCTION)*
	- 근거 원문: “To this end, we undertook this review to summarize the existing portable-sensor-based ACL assessment literature, including current target motions, sensing approaches, modeling techniques, and clinical applications.”
- 휴대용 센싱 방법들의 유효성, 재현성 및 일반화 가능성을 향상시키기 위한 해결과제와 기술의 임상적 실효성을 넓히기 위한 장기적 기회들을 고찰로 제언하였다. *(근거: PAGE 2, INTRODUCTION)*
	- 근거 원문: “We also offer our perspectives on (1) future work that is necessary to achieve greater clinical impact and (2) new opportunities that may enhance the validity, reproducibility, and generalizability of the assessment methods.”

## 레퍼런스할 수 있는 내용

### 1. Annual ACL injury and ACLR surgery volume in the US

- 원문 발췌: “Anterior cruciate ligament (ACL) injury is common in sports, with an estimated 400,000 people injuring their ACL in the United States each year1, leading to over 129,000 ACL reconstruction (ACLR) surgeries2.”
- 한국어 번역: 전방십자인대(ACL) 부상은 스포츠에서 흔히 발생하며, 미국에서만 매년 약 400,000명이 전방십자인대 부상을 입고1, 이로 인해 129,000건 이상의 전방십자인대 재건술(ACLR) 수술로 이어진다2.
- 원문 위치: PAGE 1, INTRODUCTION
- 원문 내 인용표기: 1, 2
- 해당 선행문헌: 1. Murray, M. M. The ACL Handbook: Knee Biology, Mechanics, and Treatment (eds Murray, M. M., Vavken, P. & Fleming, B.) p. 19–28 (Springer New York, 2013). 2. Mall, N. A. et al. Incidence and trends of anterior cruciate ligament reconstruction in the united states. Am. J. Sports Med. 42, 2363–2370 (2014).
- 주장 유형: background_citation
- 활용 맥락과 주의: 미국 내 전방십자인대 부상 및 재건술의 연간 발생 규모를 증명하는 통계로 사용 가능하며, 2차 인용에 주의해야 한다.

### 2. Age and complications of ACLR patients

- 원문 발췌: “Concerningly, nearly half of these patients are under 20 years of age, and they suffer from not only over 20% reinjury rates3,4 but also 50–80% knee osteoarthritis rates within a decade of injury5,6.”
- 한국어 번역: 우려스럽게도, 이 환자들의 거의 절반은 20세 미만이며, 이들은 20% 이상의 재부상율3,4뿐만 아니라 부상 후 10년 이내에 50-80%의 무릎 골관절염 발생률5,6을 겪는다.
- 원문 위치: PAGE 1, INTRODUCTION
- 원문 내 인용표기: 3,4, 5,6
- 해당 선행문헌: 3. Webster, K. E. & Feller, J. A. Exploring the high reinjury rate in younger patients undergoing anterior cruciate ligament reconstruction. Am. J. Sports Med. 44, 2827–2832 (2016). 4. Barber-Westin, S. & Noyes, F. R. One in 5 athletes sustain reinjury upon return to high-risk sports after acl reconstruction: a systematic review in 1239 athletes younger than 20 years. Sports Health 12, 587–597 (2020). 5. Nishimori, M. et al. Articular cartilage injury of the posterior lateral tibial plateau associated with acute anterior cruciate ligament injury. Knee Surg. Sports Traumatol. Arthrosc. 16, 270–274 (2008). 6. Muthuri, S., McWilliams, D., Doherty, M. & Zhang, W. History of knee injuries and knee osteoarthritis: a meta-analysis of observational studies. Osteoarthritis Cartilage 19, 1286–1293 (2011).
- 주장 유형: background_citation
- 활용 맥락과 주의: 젊은 연령대 ACLR 환자의 높은 재부상 위험과 무릎 관절염 발병률 등 부상 장기화 시의 부작용을 서술하는 근거로 활용할 수 있으며, 2차 인용 시 주의가 필요하다.

### 3. Dominant biomechanical parameters in portable sensing studies

- 원문 발췌: “Kinematic parameters were the dominant target (71%), followed by spatiotemporal parameters (18%), kinetics (12%), and muscle activation (10%) (Fig. 2c).”
- 한국어 번역: 운동학적 매개변수가 지배적인 측정 대상이었으며(71%), 시공간 매개변수(18%), 운동역학(12%), 근육 활성화(10%)가 그 뒤를 이었다(그림 2c).
- 원문 위치: PAGE 2, RESULTS
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: 십자인대 평가 분야의 휴대용 센서 기반 연구들에서 어떤 매개변수들이 주로 타깃이 되었는지를 통계적으로 제시할 때 유용하다.


---

# [20] Angular Velocities and Linear Accelerations Derived from Inertial Measurement Units Can Be Used as Proxy Measures of Knee Variables Associated with ACL Injury

(저자: Holly S. R. Jones, Victoria H. Stiles, Jasper Verheul and Isabel S. Moore | 연도: 2022 | 저널: Sensors | DOI: https://doi.org/10.3390/s22239286)

Jones, H. S. R., Stiles, V. H., Verheul, J., & Moore, I. S. (2022). Angular Velocities and Linear Accelerations Derived from Inertial Measurement Units Can Be Used as Proxy Measures of Knee Variables Associated with ACL Injury. Sensors, 22(23), 9286. https://doi.org/10.3390/s22239286

## 서지정보

- 저자: Holly S. R. Jones, Victoria H. Stiles, Jasper Verheul and Isabel S. Moore
- 연도: 2022
- 저널: Sensors
- DOI: 10.3390/s22239286
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/03_wearable_imu_and_portable_sensing/Angular Velocities and Linear Accelerations Derived from Inertial Measurement Units Can Be Used as Proxy Measures of Knee Variables Associated with ACL Injury.pdf
- 분석 provider: antigravity

## 연구 목적

- 이 연구의 목적은 ACL 부상 위험을 모니터링하기 위해 수행되는 다양한 동작들에서 실험실 측정 무릎 변수(무릎 관절 가동 범위, 무릎 모멘트 변화량, 무릎 강성)와 경골 및 대퇴부에 부착된 IMU 측정 지표(각속도 및 가속도) 간의 상관관계를 파악하는 것이다. *(근거: PAGE 3, Section 1. Introduction)*
	- 근거 원문: “Therefore, the aim of this study was to identify the strength of the correlations between laboratory-derived knee variables associated with ACL injury risk (knee RoM, change in knee moment, and knee stiffness) and metrics derived from IMUs (angular velocities and accelerations) placed on the tibia and thigh during movements performed in standard assessments to monitor ACL injury risk (bilateral and unilateral drop jumps, and a cutting manoeuvre).”
- 3회 측정 평균값(기존 ACL 위험 모니터링 프로토콜)이 안정적인 데이터를 제공하는지 확인하기 위해, 3회 및 5회 측정의 평균값과 무릎 변수 간의 상관관계를 비교한다. *(근거: PAGE 3, Section 1. Introduction)*
	- 근거 원문: “To confirm whether mean IMU-derived metrics calculated from three trials (existing ACL risk-monitoring protocols) provided stable data, correlations between the knee variables and mean IMU-derived metrics from three and five trials were compared for all movements.”
- 무릎 변수와 IMU 측정 지표 간의 관계가 있는 경우, 경골 또는 대퇴부 중 어떤 IMU 위치가 가장 강한 상관관계를 나타내는지 식별한다. *(근거: PAGE 3, Section 1. Introduction)*
	- 근거 원문: “Finally, if a relationship was found between the knee variables and IMU-derived metrics, this study identified the location of the IMU (tibia or thigh) which demonstrated the strongest correlations.”

## 연구 설계와 대상

- 18\~35세 사이의 건강한 다방향 필드 스포츠 남성 운동선수 19명이 본 연구에 참여했다. *(근거: PAGE 3, Section 2.1. Participants)*
	- 근거 원문: “Nineteen male multidirectional field sport athletes (i.e., football, rugby union, and Americanfootball)agedbetween18and35yearsparticipatedinthisstudy(age: 24 ± 4 years; height: 1.82 ± 0.07 m; mass: 85.7 ± 9.4 kg).”
- 참가자들은 실험 전 6개월 동안 하지 부상이 없어야 했다. *(근거: PAGE 3, Section 2.1. Participants)*
	- 근거 원문: “Participants were required to be free from lower-limb injury in the 6 months prior to testing.”
- 각 참가자는 데이터 수집 전에 고지된 동의를 제공하였으며, 연구는 윤리위원회의 승인을 받았다. *(근거: PAGE 3, Section 2.1. Participants)*
	- 근거 원문: “Each participant provided informed consent priortodatacollection. Ethicalapprovalwas obtainedfromCardiffMetropolitanUniversity ethics committee, with reference number PGR-3539.”
- 연구 설계상 참가자들은 양측 및 단측 드롭 점프와 90도 커팅 동작을 수행하였으며, 이때 지면반력, 3차원 운동학 및 3축 IMU 데이터를 기록하였다. *(근거: PAGE 1, Abstract)*
	- 근거 원문: “Ground reaction forces, three-dimensional kinematics, and triaxial IMU data were recorded from nineteen healthy male participants performing bilateral and unilateral drop jumps, and a 90◦ cutting task.”

## 방법

- 참가자들은 가벼운 러닝과 스트레칭으로 구성된 짧은 웜업을 완료한 후, 양측 드롭 점프(30cm), 단측 드롭 점프(20cm), 90도 커팅 동작을 차례로 수행했다. *(근거: PAGE 3, Section 2.2. Experimental Procedure)*
	- 근거 원문: “Participants completed a short warm-up consisting of slow running and stretching, and then performed the following three movements (in order): a bilateral drop jump from 30 cm, a unilateral drop jump from 20 cm, and a 90◦ pre-planned cut, following previously described protocols \[21,25\].”
- 운동학 데이터 수집을 위해 12대의 카메라 3차원 동작 분석 시스템(250 Hz)을 사용하였다. *(근거: PAGE 3, Section 2.3. Biomechanical Data Collection)*
	- 근거 원문: “A 12-camera three-dimensional motion capture system (250 Hz; Vicon Motion Systems Ltd., Oxford, UK) was used to collect kinematic data.”
- 분석을 위한 운동학 및 지면반력 데이터 필터링에는 15 Hz 차단 주파수의 4차 저역 통과 버터워스 필터가 사용되었다. *(근거: PAGE 4, Section 2.3. Biomechanical Data Collection)*
	- 근거 원문: “Raw marker trajectories and GRF data used for inverse dynamic analysis calculations were filtered using a fourth-order low-pass Butterworth filter at 15 Hz \[28\].”
- 제동 단계(Braking phase)는 힘판 및 동작 분석 데이터를 기준으로 지면 접촉 순간(GRF \> 20 N)부터 최대 무릎 굴곡이 일어나는 시점까지로 정의했다. *(근거: PAGE 4, Section 2.4. Biomechanical Data Processing and Analysis)*
	- 근거 원문: “The braking phase was defined as the time between initial contact (determined as GRF \> 20 N) to maximum knee flexion.”
- IMU 각속도 및 가속도 데이터의 샘플링 속도를 동작 분석 시스템(250 Hz)과 일치시키기 위해 다운샘플링을 적용하였다. *(근거: PAGE 5, Section 2.5. IMU Data Processing and Analysis)*
	- 근거 원문: “Gyroscope and accelerometer data were then down-sampled to match the collection frequency of the motion capture system (250 Hz).”
- 비정규 분포를 보이는 데이터 특성을 고려하여, 무릎 변수와 IMU 측정 지표 간의 관계를 분석하기 위해 스피어만 상관분석(Spearman's correlations)을 실시했다. *(근거: PAGE 5, Section 2.6. Statistical Analysis)*
	- 근거 원문: “Due to the non-normality of data, multiple Spearman’s correlations were run to determine the relationship between the gold-standard motion analysis and force-plate-derived knee variables (knee RoM, change in knee moment, and knee stiffness) and the IMU-derived metrics of peak angular velocity, area under the angular velocity curve, angular velocity rate, peak acceleration, area under the acceleration curve, and acceleration rate in each movement (bilateral and unilateral drop jump and the cut).”

## 핵심 결과

- 모든 동작에서 무릎 관절 가동 범위(RoM)와 경골 각속도 곡선 아래 면적 사이에 유의한 강한 양의 상관관계가 관찰되었다. *(근거: PAGE 8, Section 3.1. IMU-Derived Angular Velocities vs. Knee Variables)*
	- 근거 원문: “There was a significant strong positive relationship between knee RoM and the area under the tibia angular velocity curve during all movements (Figure 2).”
- 단측 드롭 점프 동작에서 모든 무릎 변수(무릎 RoM, 무릎 모멘트 변화량, 무릎 강성)와 경골 가속도 곡선 아래 면적 사이에 유의한 강한 상관관계가 관찰되었다. *(근거: PAGE 9, Section 3.2. IMU-Derived Accelerations vs. Knee Variables)*
	- 근거 원문: “Significant strong correlations were observed in the unilateral drop jump between all knee variables and the area under the tibia acceleration curve (Figure 3).”
- 양측 드롭 점프에서 무릎 강성과 대퇴 각속도 곡선 아래 면적 사이에 유의한 중간 정도의 음의 상관관계가 관찰되었으며, 무릎 RoM과 대퇴 각속도 곡선 아래 면적 사이에는 유의한 중간 정도의 양의 상관관계가 관찰되었다. *(근거: PAGE 8, Section 3.1. IMU-Derived Angular Velocities vs. Knee Variables)*
	- 근거 원문: “In the bilateral drop jump, a significant moderate negative correlation was observed between knee stiffness and the area under the thigh angular velocity curve, as well as a significant moderate positive correlation between knee RoM and the area under the thigh angular velocity curve.”
- 3회 시험 평균값과 5회 시험 평균값을 사용했을 때 무릎 변수와 IMU 유도 지표 간에 유사한 수준의 일치성이 관찰되었다. *(근거: PAGE 5, Section 3. Results)*
	- 근거 원문: “Similar levels of correspondence were observed between taking the mean of three trials and taking the mean of five trials (Tables 1 and 2).”

## 저자 결론

- 본 연구의 결과는 ACL 부상 위험을 모니터링하기 위해 사용되는 평가 동작들에서 IMU 유도 각속도 및 가속도를 무릎 변수의 대리 지표로 사용하는 것이 가능함을 시사한다. *(근거: PAGE 11, Section 5. Conclusions)*
	- 근거 원문: “The findings from this study suggest that it may be feasible to use IMU-derived angular velocities and accelerations as proxy measures of knee variables in movements included in practitioner assessments used to monitor ACL injury risk.”
- 대퇴부에 부착된 IMU에 비해 경골에 부착된 IMU에서 얻은 결과 각속도 및 가속도가 ACL 부상 관련 무릎 변수와 가장 강한 상관관계를 보였다. 따라서 각 하지에 단 하나의 센서만 부착하고자 한다면 경골 부착을 권장한다. *(근거: PAGE 11, Section 5. Conclusions)*
	- 근거 원문: “Finally, the resultant angular velocities and accelerations derived from a tibia-mounted IMU were most strongly correlated with knee variables associated with ACL injury, as opposed to those derived from a thigh-located IMU. Therefore, if practitioners were looking to apply only a single sensor on each lower limb, IMUs located on each tibia would be recommended.”
- 구체적으로, 경골 각속도 곡선 아래 면적은 양측 및 단측 드롭 점프와 커팅 동작에서 무릎 RoM의 대리 지표로 사용될 수 있다. *(근거: PAGE 12, Section 5. Conclusions)*
	- 근거 원문: “Specifically, the area under the tibia angular velocity curve may be used as a proxy measure for knee RoM in the bilateral and unilateral drop jumps, and the cut.”
- 경골 가속도 곡선 아래 면적은 실무자가 단측 드롭 점프 상황에 한해 무릎 관절 강성, 무릎 모멘트 변화량, 무릎 RoM의 차이를 감지하는 유용한 대리 지표가 될 수 있다. *(근거: PAGE 12, Section 5. Conclusions)*
	- 근거 원문: “The area under the tibia acceleration curve may be a useful proxy measure for practitioners wanting to detect differences in knee joint stiffness, change in knee moment, and knee RoM in an applied setting, but only in a unilateral drop jump.”

## 연구의 한계

- 첫째, 본 연구에서는 건강한 참가자들만을 평가했기 때문에 ACL 재건 수술을 받은 개인이 보이는 비정상적인 움직임 변동성이 존재할 때도 이 상관관계가 유지되는지 확인하기 어렵다. *(근거: PAGE 11, Section 4. Discussion)*
	- 근거 원문: “Firstly, since the level of variation when performing movements likely displayed in ACL-reconstructed individuals could have made it difficult to detect any associations that may have existed, only healthy participants were assessed in this study.”
- ACL 부상 또는 재부상 위험이 있는 참가자 그룹을 비교군으로 포함하지 않았기 때문에, IMU 유도 무릎 변수 대리 지표가 2차 ACL 부상 위험이 높은 개별 선수를 식별하는 데 사용될 수 있는지 여부는 확인할 수 없다. *(근거: PAGE 11, Section 4. Discussion)*
	- 근거 원문: “As a group of participants at risk of ACL injury or re-injury were not included for comparison, it is not possible to confirm whether IMU-derived proxy measures of knee variables could be used to identify individuals at higher risk of second ACL injury.”
- 둘째, 대퇴부에 부착된 IMU의 위치가 3차원 동작 분석용 기술 클러스터(technical cluster)의 경직된 플레이트 위치와 일치했다. *(근거: PAGE 11, Section 4. Discussion)*
	- 근거 원문: “Secondly, the position of the thigh-located IMUs coincided with the position of the rigid plates of the technical clusters.”
- 셋째, 본 연구에서는 제동 단계를 식별하기 위해 모션 및 힘 데이터를 사용했는데, 이는 실제 현장 환경에서는 불가능하므로 현장 적용을 위해서는 IMU 단독으로 제동 단계를 정의하는 연구가 추가로 필요하다. *(근거: PAGE 11, Section 4. Discussion)*
	- 근거 원문: “Using motion and force data to detect initial contact and maximum knee flexion to determine the beginning and end of the braking phase, respectively, would not be possible in the field.”

## 생각해볼 내용

- 대체로 단측 드롭 점프에서만 가속도 곡선 면적이 모든 무릎 변수와 강하게 부합한 이유를 Derrick의 유효 질량 이론(effective mass theory)으로 설명한 부분은 흥미롭다. 무릎이 더 많이 굽혀질수록 몸체로부터 하퇴부의 유효 질량이 분리되어 가속도가 빨라진다는 분석은 타당해 보인다. *(근거: PAGE 10, Section 4. Discussion)*
	- 근거 원문: “Briefly, the effective mass theory proposes that the effective mass of the shank-foot complex is reduced by its uncoupling from the rest of the body through increased sagittal plane knee RoM. Subsequently, the lower effective mass can be accelerated more quickly throughout the brakingphase, resultinginthelargerareaunderthetibiaaccelerationcurvevaluesobserved at increased knee RoM in this study.”
- **[AS-IS]** 본 연구에서 건강한 성인만을 대상으로 제한한 것은 타당한 통제 연구이나, 실제 임상 현장이나 스포츠 현장에서 재활 상태를 모니터링하기 위해서는 ACL 재건 환자를 대상으로 한 후속 타당성 연구가 필수적이다. *(근거: PAGE 11, Section 4. Discussion)*<br>**[TO-BE]** 본 연구는 건강한 참가자만을 대상으로 했으므로, 실제 재활·스포츠 현장 적용 가능성을 평가하려면 ACL 재건 환자와 비손상 대조군을 비교하는 후속 연구가 필요하다.<br>*(사실검증 — 과장/경미: 원문은 ACL 재건 환자와 비손상 대조군 비교 연구가 가능하다고 제안하지만, '필수적'이라고 단정하지는 않는다.)*
	- 근거 원문: “Since correlations were observed between knee variables (knee RoM, change in knee moment, and knee stiffness) and angular velocities and accelerations derived from IMUs in healthy participants, future research could compare patients who have had an ACL reconstruction to non-injured controls to investigate the feasibility of using IMUs on ACL reconstructed individuals.”
- IMU의 3축 데이터를 통합하여 결과 각속도 및 가속도를 계산한 것은 센서의 부착 방향 정렬 오차 문제를 우회하여 필드 측정의 반복성을 크게 향상시킬 수 있는 현명한 접근법이다. *(근거: PAGE 11, Section 4. Discussion)*
	- 근거 원문: “As a result, combining all axes from a triaxial IMU to calculate resultant angular velocities and accelerations, as used in this study, would be beneficial since the orientation of the IMU does not have to be aligned to a specific axis, thus improving the repeatability of using an IMU in the field \[45\].”
- 이 연구는 제동 단계를 나누기 위해 동작 분석용 힘판 데이터를 썼으나, 현장에서는 힘판을 쓸 수 없으므로 IMU 가속도 파형 자체에서 지면 접촉 시점(initial contact)과 무릎 최대 굴곡 시점을 자동으로 검출하는 알고리즘 개발이 실질적인 활용을 위한 병목이 될 것이다. *(근거: PAGE 11, Section 4. Discussion)*
	- 근거 원문: “Future research should focus on determining initial contact and maximum knee flexion events using IMU-derived metrics from a tibia-located IMU to define the braking phase, without the need for gold-standard equipment, in movements that practitioners would use to assess ACL injury risk.”

## 이 연구가 지적한 선행연구의 문제점

- 기존의 ACL 부상 위험 모니터링은 실제 필드 환경(훈련 이나 경기 중)에서 선수의 움직임과 부하를 정확히 측정하기 어렵다는 한계가 있었다. *(근거: PAGE 2, Section 1. Introduction)*
	- 근거 원문: “However, a major limitation for practitioners when monitoring ACL injury risk is the inability to assess an athlete in applied field-based settings.”
- 골드 스탠다드 실험실 시스템(3D 모션 캡처 및 힘판)은 정확하지만, 시간이 오래 걸리고 장비가 비싸며 숙련된 인력이 필요하여 대다수 실무자가 이용하기 어렵다. *(근거: PAGE 2, Section 1. Introduction)*
	- 근거 원문: “Although gold-standard laboratory-based systems (i.e., marker-based three-dimensional motion capture and force plates) can be used to accurately measure these variables, these systems are time-consuming, and require costly equipment and skilled personnel, thus, they are not accessible to most practitioners.”
- ACL 부상 위험 모니터링에 자주 사용되는 동작(드롭 점프, 커팅 등)을 수행할 때 무릎 부상 관련 변수들과 IMU의 각속도/가속도 지표 사이에 어떤 상관관계가 있는지 규명한 기존 연구가 부족했다. *(근거: PAGE 2, Section 1. Introduction)*
	- 근거 원문: “Further research is therefore needed to determine if relationships exist between knee variables associated with ACL injury and IMU angular velocity and acceleration metrics during movements performed in practitioner assessments used to monitor ACL injury risk.”

## 이 연구의 해결 방식과 기여

- **[AS-IS]** 이 연구는 ACL 부상 위험 모니터링 평가 동작 중 무릎 관절 변수를 추정하기 위해 IMU에서 유도된 각속도 및 가속도를 대리 지표로 사용할 수 있는 가능성을 증명했다. *(근거: PAGE 11, Section 5. Conclusions)*<br>**[TO-BE]** 이 연구는 ACL 부상 위험 모니터링 평가 동작 중 무릎 관절 변수를 추정하기 위해 IMU에서 유도된 각속도 및 가속도를 대리 지표로 사용할 수 있을 가능성을 시사했다.<br>*(사실검증 — 과장/중대: 원문은 IMU 유도 지표를 대리 지표로 사용하는 것이 '가능할 수 있음'을 시사한다고 표현한다. 요약의 '증명했다'는 탐색적 상관연구의 결론을 확정적 검증처럼 강화한다.)*
	- 근거 원문: “The findings from this study suggest that it may be feasible to use IMU-derived angular velocities and accelerations as proxy measures of knee variables in movements included in practitioner assessments used to monitor ACL injury risk.”
- 3회 측정의 평균값만으로도 여러 무릎 변수의 대리 지표로서 충분한 안정성을 가짐을 입증하여, 현장 평가 시 수행자의 부담을 덜고 테스트 효율성을 향상시킬 수 있는 근거를 제공하였다. *(근거: PAGE 11, Section 4. Discussion)*
	- 근거 원문: “Similar levels of correspondence with knee variables were reported between taking the mean of three trials and taking the mean of five trials (Tables 1 and 2). This has implications for field-based assessments that seek to monitor ACL injury risk, as it demonstrates that a mean value based on three trials provides sufficient stability to evaluate relevant IMU-derived metrics as proxies for several knee variables.”
- 대퇴부에 부착된 IMU보다 경골에 부착된 IMU에서 얻은 결과 각속도 및 가속도가 무릎 변수들과 가장 강하게 상관되어 있음을 밝혀내어, 단일 센서만 사용할 때의 최적의 위치(경골)를 제시했다. *(근거: PAGE 11, Section 4. Discussion)*
	- 근거 원문: “Subsequently, if practitioners only have the time and finances available to apply a single sensor to each lower limb, this study identified that, compared with those of the thigh-mounted IMU, the resultant angular velocities and accelerations derived from a tibial-mounted IMU are most strongly correlated with knee variables used to assess ACL injury risk.”
- **[AS-IS]** 3축 IMU의 모든 축을 결합하여 결과 각속도 및 가속도를 계산하는 방식을 적용해, 센서의 부착 정렬에 구애받지 않고 실무에서 편리하게 활용할 수 있어 필드 평가의 반복재현성을 높였다. *(근거: PAGE 11, Section 4. Discussion)*<br>**[TO-BE]** 3축 IMU의 모든 축을 결합하여 결과 각속도 및 가속도를 계산하는 방식은 센서 방향을 특정 축에 맞출 필요를 줄여, 현장 IMU 사용의 반복성을 개선하는 데 유익할 수 있다.<br>*(사실검증 — 과장/경미: 원문은 방향을 특정 축에 맞출 필요가 없어 현장 반복성을 개선하는 데 유익할 것이라고 설명한다. 실제로 반복재현성이 향상되었음을 실험적으로 입증했다는 표현은 원문보다 강하다.)*
	- 근거 원문: “As a result, combining all axes from a triaxial IMU to calculate resultant angular velocities and accelerations, as used in this study, would be beneficial since the orientation of the IMU does not have to be aligned to a specific axis, thus improving the repeatability of using an IMU in the field \[45\].”

## 레퍼런스할 수 있는 내용

### 1. 비접촉성 ACL 파열의 빈도와 심각성

- 원문 발췌: “Non-contact anterior cruciate ligament (ACL) ruptures are one of the most common and severe injuries in multidirectional field sports \[1,2\].”
- 한국어 번역: 비접촉성 전방 십자 인대(ACL) 파열은 다방향 필드 스포츠에서 가장 흔하고 심각한 부상 중 하나이다.
- 원문 위치: PAGE 1, Section 1. Introduction
- 원문 내 인용표기: \[1,2\]
- 해당 선행문헌: 1. Alentorn-Geli, E.; Myer, G.; Silvers, H.; Samitier, G.; Romero, D.; Lazaro-Haro, C.; Cugat, R. Prevention of non-contact anterior cruciate ligament injuries in soccer players. Part 1: Mechanisms of injury and underlying risk factors. Knee Surg. Sport. Traumatol. Arthrosc. 2009, 17, 705–729. \[CrossRef\] \[PubMed\] 2. Moses, B.; Orchard, J.; Orchard, J. Systematic Review: Annual Incidence of ACL Injury and Surgery in Various Populations. Res. Sport. Med. 2012, 20, 157–179. \[CrossRef\]
- 주장 유형: background_citation
- 활용 맥락과 주의: 다방향 필드 스포츠(예: 축구, 럭비 등)에서 비접촉성 ACL 부상의 흔함과 심각성을 설명할 때 인용할 수 있다. 2차 인용 시 실제 Incidence 수치나 세부 메커니즘은 해당 원저작물 \[1\]과 \[2\]를 확인해야 한다.

### 2. ACL 재건 후 스포츠 복귀 시 재부상 위험도

- 원문 발췌: “Specifically, when returning to sports that include frequent cutting and pivoting following ACL reconstruction, an athlete has a 3.9-fold increased risk of sustaining an ipsilateral ACL injury and a 5-fold increased risk of sustaining a contralateral ACL injury \[8\].”
- 한국어 번역: 구체적으로, ACL 재건 후 빈번한 커팅과 피보팅을 포함하는 스포츠로 복귀할 때, 운동선수는 동측 ACL 부상을 입을 위험이 3.9배 증가하고 대측 ACL 부상을 입을 위험이 5배 증가한다.
- 원문 위치: PAGE 1, Section 1. Introduction
- 원문 내 인용표기: \[8\]
- 해당 선행문헌: 8. Webster, K.E.; Feller, J.A.; Leigh, W.B.; Richmond, A.K. Younger Patients Are at Increased Risk for Graft Rupture and Contralateral Injury After Anterior Cruciate Ligament Reconstruction. Am. J. Sports Med. 2014, 42, 641–647. \[CrossRef\] \[PubMed\]
- 주장 유형: background_citation
- 활용 맥락과 주의: ACL 재건 수술 후 회전 및 방향 전환이 잦은 스포츠로 복귀할 때 동측 및 반대측 무릎의 재부상 상대 위험 비율(각각 3.9배, 5배)을 뒷받침하는 근거로 인용 가능하다. 대상 환자군 및 연령대(예: 젊은 환자군 대상 여부 등)의 세부 사항은 Webster 등(2014)의 원문을 확인할 필요가 있다.

### 3. 무릎 RoM과 경골 각속도 곡선 아래 면적 간의 상관관계

- 원문 발췌: “A significant strong positive correlation was observed between knee RoM and the area under the tibia angular velocity curve in all movements.”
- 한국어 번역: 모든 동작에서 무릎 관절 가동 범위(RoM)와 경골 각속도 곡선 아래 면적 사이에 유의한 강한 양의 상관관계가 관찰되었다.
- 원문 위치: PAGE 1, Abstract
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- **[AS-IS]** 활용 맥락과 주의: 이 연구의 자체 발견 사실로, 양측 및 단측 드롭 점프와 90도 커팅 동작 전체에서 경골 부착 IMU의 결과 각속도 적분값(곡선 아래 면적)이 무릎 관절 가동 범위(RoM)의 대리 지표로 유효함을 보여준다.<br>**[TO-BE]** 활용 맥락과 주의: 이 연구의 자체 발견 사실로, 양측 및 단측 드롭 점프와 90도 커팅 동작에서 경골 부착 IMU의 결과 각속도 곡선 아래 면적이 무릎 RoM의 대리 지표로 사용될 수 있음을 시사한다.<br>*(사실검증 — 과장/중대: 원문은 강한 양의 상관관계와 대리 지표로 사용될 수 있음을 제시하지만, '유효함을 보여준다'는 표현은 타당도 검증이 완료된 것처럼 확정적이다.)*

### 4. 단측 드롭 점프에서 무릎 변수들과 경골 가속도 곡선 아래 면적 간의 상관관계

- 원문 발췌: “Significant strong correlations were also observed in the unilateral drop jump between knee RoM, change in knee moment, and knee stiffness, and the area under the tibia acceleration curve (rs = 0.776, rs = −0.712, and rs = −0.765, respectively).”
- 한국어 번역: 또한 단측 드롭 점프에서 무릎 RoM, 무릎 모멘트 변화량, 무릎 강성과 경골 가속도 곡선 아래 면적 사이에 유의한 강한 상관관계가 관찰되었다(각각 rs = 0.776, rs = -0.712, rs = -0.765).
- 원문 위치: PAGE 1, Abstract
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- **[AS-IS]** 활용 맥락과 주의: 이 연구의 자체 발견 사실로, 단측 드롭 점프 동작 시 경골 가속도 적분값이 무릎 RoM, 모멘트 변화량, 관절 강성을 모두 강력하게 대변할 수 있음을 나타낸다. 단, 단측 동작에만 국한하여 강한 상관관계를 보였음에 유의해야 한다.<br>**[TO-BE]** 활용 맥락과 주의: 이 연구의 자체 발견 사실로, 단측 드롭 점프 동작에서 경골 가속도 곡선 아래 면적은 무릎 RoM, 모멘트 변화량, 관절 강성의 차이를 감지하는 데 유용한 대리 지표일 수 있다. 이 해석은 단측 드롭 점프에 한정된다.<br>*(사실검증 — 과장/중대: 원문은 단측 드롭 점프에서 강한 상관관계를 보였고 유용한 대리 지표일 수 있다고 말한다. '강력하게 대변할 수 있음'은 상관관계를 대리 측정의 확정적 성능으로 과도하게 해석한다.)*


---

# [21] Estimation of Kinetics Using IMUs to Monitor and Aid in Clinical Decision-Making during ACL Rehabilitation: A Systematic Review

(저자: Sanchana Krishnakumar, Bert-Jan F. van Beijnum, Chris T. M. Baten, Peter H. Veltink, Jaap H. Buurke | 연도: 2024 | 저널: Sensors | DOI: https://doi.org/10.3390/s24072163)

Krishnakumar, S., van Beijnum, B.-J. F., Baten, C. T. M., Veltink, P. H., & Buurke, J. H. (2024). Estimation of Kinetics Using IMUs to Monitor and Aid in Clinical Decision-Making during ACL Rehabilitation: A Systematic Review. Sensors, 24(7), 2163. https://doi.org/10.3390/s24072163

## 서지정보

- 저자: Sanchana Krishnakumar, Bert-Jan F. van Beijnum, Chris T. M. Baten, Peter H. Veltink, Jaap H. Buurke
- 연도: 2024
- 저널: Sensors
- DOI: 10.3390/s24072163
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/03_wearable_imu_and_portable_sensing/Estimation of Kinetics Using IMUs to Monitor and Aid in Clinical Decision-Making during ACL Rehabilitation - A Systematic Review.pdf
- 분석 provider: antigravity

## 연구 목적

- 본 연구는 대안적 알고리즘을 확인하고 종합적인 체계적 문헌고찰을 통해 ACL 재활에서의 적용 가능성을 평가하고자 한다. *(근거: PAGE 1, Abstract)*
	- 근거 원문: “Therefore, this article aims to identify the available algorithms for the estimation of kinetic parameters using kinematics measured only from IMUs and to evaluate their applicability in ACL rehabilitation through a comprehensive systematic review.”
- 이 체계적 문헌고찰의 목적은 추가적인 힘 정보 없이 IMU만을 사용하여 하지 운동역학적 매개변수를 추정하는 알고리즘을 식별하고 논의하는 것이다. *(근거: PAGE 3, 1. Introduction)*
	- 근거 원문: “Therefore, the objective of this systematic review is to identify and discuss algorithms for the estimation of lower limb kinetic parameters only using IMUs without additional force information.”

## 연구 설계와 대상

- 이 체계적 문헌고찰은 PRISMA 성명에 따라 수행되었다. *(근거: PAGE 3, 2.1. Study Design)*
	- 근거 원문: “This systematic review was conducted in accordance with the Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA) statement (Supplementary Table S1) \[24\].”
- 문헌 선별 및 전판 스크리닝 결과, 정량적 데이터 합성을 위해 최종적으로 71개의 연구가 분석에 포함되었다. *(근거: PAGE 4, 2.4. Study Selection and Quality Assessment)*
	- 근거 원문: “Full-text screening was performed by the first reviewer (S.K), and 71 articles were included for quantitative data synthesis.”

## 방법

- 리뷰를 위해 PubMed, Scopus, SPORTDiscus 데이터베이스가 검색에 활용되었다. *(근거: PAGE 3, 2.2. Search Strategy)*
	- 근거 원문: “The databases used for this review were PubMed, Scopus (Elsevier), and SPORTDiscus (EBSCO host).”
- 포함된 연구들의 품질 평가는 Strom 등이 제시한 체크리스트를 수정하여 구성한 14점의 체크리스트를 통해 진행되었다. *(근거: PAGE 4, 2.4. Study Selection and Quality Assessment)*
	- 근거 원문: “Quality assessment of the included studies was performed (S.K) using a 14-point checklist comprising items listed in Table 1.”

## 핵심 결과

- 분석 대상 문헌들의 다수는 머신러닝 기반 모델을 사용하였으며(약 45%), 그 뒤를 생체역학 모델(약 38%)이 이었다. *(근거: PAGE 13, 4.1. Modelling Techniques and Estimated Kinetic Parameters)*
	- 근거 원문: “The majority of the reviewed articles utilized ML-based models and accounted for around 45% of the reviewed articles, followed by BM (∼38%).”
- 3차원 지면반발력(GRF) 추정에서 가장 우수한 RMSE 값은 수직 드롭 점프 동작에서 전후방 0.018, 내외측 0.008, 수직 0.038(체중 기준 표준화)로 달성되었다. *(근거: PAGE 13, 3.6. Accuracy and Reliability of Tested Approaches)*
	- 근거 원문: “For 3D GRF, the best RMSE values were achieved for vertical drop jump, namely, 0.018, 0.008, and 0.038 (normalized to body weight) for anterior–posterior GRF (AP-GRF), medio-lateral GRF (M-LGRF), and vertical GRF (VGRF) respectively \[85\].”
- 걷기 과제 중 3차원 순 무릎 관절 모멘트의 추정치에서는 외전-내전 모멘트가 10.58%, 굴곡-신전 모멘트가 9.46%, 외/내회전 모멘트가 17.12%로 가장 낮은 nRMSE(%)를 보여주었다. *(근거: PAGE 13, 3.6. Accuracy and Reliability of Tested Approaches)*
	- 근거 원문: “Among the articles that estimated 3D net knee joint moments, the lowest nRMSE (%) was observed for walking, with values of 10.58, 9.46, and 17.12 for abduction–adduction, flexion–extension, and external/internal rotation moments, respectively \[30\].”

## 저자 결론

- IMU 센서는 건강한 대상자를 기준으로 시상면상에서 일어나는 움직임에 대해 높은 정확도로 GRF 및 관절 운동역학적 매개변수를 추정할 수 있는 잠재력을 보여주었다. *(근거: PAGE 17, 5. Conclusions)*
	- 근거 원문: “The results of this review indicate that IMUs have good potential to estimate GRF and other joint kinetic parameters with good accuracy for movements primarily in the sagittal plane for healthy cohorts.”
- 하지만 체계적 문헌고찰에 포함된 알고리즘 중 실제 ACL 환자를 대상으로 검증을 거친 알고리즘은 단 하나도 없었다. *(근거: PAGE 17, 5. Conclusions)*
	- 근거 원문: “However, none of these algorithms have been validated on ACL patients.”

## 연구의 한계

- 본 문헌고찰에 포함된 분석 대상 논문이 영어로 작성된 문헌으로만 한정되어 선택 편향의 우려가 존재한다. *(근거: PAGE 16, 4.5. Limitations of the Included Evidence, Review Process, and Future Directions)*
	- 근거 원문: “The articles included in the review were also limited to only English.”
- 질 향상을 목표로 인간을 대상으로 유효성이 검증된 논문으로 분석 대상을 제한한 결정이 고찰 범위의 포괄성을 제한했을 가능성이 있다. *(근거: PAGE 16, 4.5. Limitations of the Included Evidence, Review Process, and Future Directions)*
	- 근거 원문: “The decision to include only articles that validated on human beings while improving the quality of the included results may have limited the inclusiveness of the review.”
- 각 분석 대상 연구에서 각기 다른 센서 부착 위치, 실험 프로토콜, 평가 매트릭을 사용하고 있어, 어떤 모델이 최고 수준의 정확도를 내는 최적 모델인지 직접적으로 비교하고 분석하기 어렵다. *(근거: PAGE 17, 4.5. Limitations of the Included Evidence, Review Process, and Future Directions)*
	- 근거 원문: “The use of varying sensor placement locations, experimental protocols, and reporting metrics used in the included articles made direct comparison and identification of the overall best model with the most accurate outcome challenging.”

## 생각해볼 내용

- 분석 대상 문헌들의 대다수에서 여성 참가자가 과소대표되었으며, 이는 부상 위험이 상대적으로 높은 여성 대상의 연구와 유효성 검증 필요성을 뒷받침한다. *(근거: PAGE 6, 3.2. Participant Characteristics)*
	- 근거 원문: “Female population was underrepresented in 55 articles (∼80%), while 6 articles (∼8%) \[30,32,34–37\] did not report complete information on the gender distribution of the study population.”
- 임상 및 ACL 재활 단계에서 반드시 고려되어야 할 단일 다리 홉(single-leg hop)이나 트리플 홉(triple hop) 등의 기동 동작에 관한 연구가 부재하다. *(근거: PAGE 16, 4.4. Applicability for ACL Rehabilitation)*
	- 근거 원문: “Important ACL rehabilitation-specific tasks such as single-leg hop and triple hop have not been studied.”

## 이 연구가 지적한 선행연구의 문제점

- 전통적인 보행 운동역학 평가 방식은 고가이면서 복잡하고, 부피가 큰 특성상 실험실 외부의 환경에서는 적용이 매우 제한적이다. *(근거: PAGE 1, Abstract)*
	- 근거 원문: “Conventional methods deployed to estimate kinetics require complex, expensive systems and are limited to laboratory settings.”
- IMU 센서 데이터만을 이용해 보행 변수나 운동역학적 부하를 예측하는 기존 알고리즘들은 실제 환자군을 대상으로 유효성과 일반화 가능성을 검증한 결과가 매우 제한적이다. *(근거: PAGE 1, Abstract)*
	- 근거 원문: “However, the knowledge about their accuracy and generalizability for patient populations is still limited.”

## 이 연구의 해결 방식과 기여

- 이 연구는 외부의 힘 계측 시스템 없이 IMU 센서 데이터만을 입력으로 사용해 하지 관절 모멘트 및 지면반발력을 예측하는 최신 알고리즘들을 비교하고 평가한 정보를 제공한다. *(근거: PAGE 3, 1. Introduction)*
	- 근거 원문: “Thus, a systematic review that compares and evaluates available algorithms for estimation of kinetic parameters (joint kinetics, GRF, and GRM) using only IMU data will provide insights on the state of the art of the accuracy, reliability, and applicability of the available algorithms.”
- 이 연구를 통해 밝힌 지식 공백들은 ACL 재활 모니터링은 물론, 유사 근골격계 질환의 임상적 활용 및 향후 보행 재훈련 프로토콜 개발 연구에 중요한 방향성을 제시한다. *(근거: PAGE 3, 1. Introduction)*
	- 근거 원문: “In addition, it will help to identify the gaps and opportunities for further research and open new avenues for clinical decision-making for ACL rehabilitation and for other conditions.”

## 레퍼런스할 수 있는 내용

### 1. ACL 재활 모니터링에서 정량적 생체역학 평가의 당위성

- 원문 발췌: “Since the treatment is currently based on subjective visual observations during clinical visits, there is huge potential to further optimize the training of patients using quantitative assessment of relevant biomechanical parameters.”
- 한국어 번역: 현재 임상 방문 시 주관적인 시각적 관찰을 토대로 치료 결정이 내려지기 때문에, 관련 생체역학적 매개변수의 정량적 평가를 통해 환자 훈련을 한층 더 최적화할 수 있는 여지가 매우 크다.
- 원문 위치: PAGE 2, 1. Introduction
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: uncited_author_claim
- **[AS-IS]** 활용 맥락과 주의: 임상에서 수동적이거나 주관적인 외관 검사 대신, 환자의 운동 메커니즘을 보다 명밀하게 추적하기 위해 센서 기반의 정량적 평가가 요구되는 명분을 인용할 때 사용한다.<br>**[TO-BE]** 활용 맥락과 주의: 임상 방문에서 주관적 시각 관찰에 의존하는 ACL 재활 의사결정을 보완하기 위해, 관련 생체역학 변수의 정량 평가 필요성을 제시할 때 사용할 수 있다.<br>*(사실검증 — 근거불충분/경미: 원문은 임상 방문 중 주관적 시각 관찰 기반 치료와 정량적 생체역학 평가의 최적화 가능성을 말하지만, '수동적 검사', '외관 검사', '운동 메커니즘을 명밀하게 추적'이라는 표현은 해당 인용문만으로 직접 지지되지 않는다.)*

### 2. 보행 분석에서 계측형 러닝머신 측정 방식이 지닌 한계

- 원문 발췌: “Systems such as instrumented treadmills that measure GRF may also alter the natural pattern of gait \[13,14\].”
- 한국어 번역: 지면반발력(GRF)을 측정하는 장비가 내장된 러닝머신과 같은 시스템은 보행의 자연스러운 형태를 왜곡시킬 가능성이 있다.
- 원문 위치: PAGE 2, 1. Introduction
- 원문 내 인용표기: \[13,14\]
- 해당 선행문헌: 13. Lee, S.J.; Hidler, J. Biomechanics of overground vs. treadmill walking in healthy individuals. J. Appl. Physiol. 2008, 104, 747–755. 14. Veras, L.; Diniz-Sousa, F.; Boppre, G.; Devezas, V.; Santos-Sousa, H.; Preto, J.; Vilas-Boas, J.P.; Machado, L.; Oliveira, J.; Fonseca, H. Accelerometer-based prediction of skeletal mechanical loading during walking in normal weight to severely obese subjects. Osteoporos. Int. 2020, 31, 1239–1250.
- 주장 유형: background_citation
- 활용 맥락과 주의: 러닝머신 상에서의 보행이 실생활의 일반적인 평지 보행 패턴과 다를 수 있음을 시사하며, 일상 환경(wearable) 평가의 타당성을 피력하는 문헌적 근거로 활용할 수 있다. 2차 인용에 주의하여 활용해야 한다.

### 3. 여성 인구의 높은 ACL 부상 위험도

- 원문 발췌: “It is also important to note that the female population has an increased risk of ACL injury \[100\].”
- **[AS-IS]** 한국어 번역: 여성 집단이 전방십자인대(ACL) 부상 위험에 노출될 확률이 훨씬 높다는 점 또한 중요하게 인지되어야 한다.<br>**[TO-BE]** 한국어 번역: 여성 집단은 ACL 손상 위험이 증가되어 있다는 점 또한 중요하게 고려해야 한다.<br>*(사실검증 — 과장/경미: 원문은 여성 집단의 ACL injury risk가 증가되어 있다고만 표현한다. 요약 번역의 '훨씬 높다'는 원문보다 강한 정도 표현이다.)*
- 원문 위치: PAGE 16, 4.4. Applicability for ACL Rehabilitation
- 원문 내 인용표기: \[100\]
- 해당 선행문헌: 100. The female ACL: Why is it more prone to injury? J. Orthop. 2016, 13, A1–A4.
- 주장 유형: background_citation
- 활용 맥락과 주의: 여성 운동선수 등을 타깃으로 하여 ACL 부상 예방을 하거나, 성별에 적합한 보행 모니터링 알고리즘 설계의 당위성을 제시할 때 논문 근거로 활용할 수 있다.


---

# [22] Evaluating the Accuracy of IMU Devices in Detecting Clinically Significant Changes in Knee Flexion and Extension Angles

(저자: Alexandra Grace Ligeti | 연도: 확인 불가 | 저널: University of Strathclyde | DOI: https://doi.org/확인 불가)

Ligeti, A. G. (확인 불가). Evaluating the Accuracy of IMU Devices in Detecting Clinically Significant Changes in Knee Flexion and Extension Angles (Doctoral dissertation, University of Strathclyde).

## 서지정보

- 저자: Alexandra Grace Ligeti
- 연도: 확인 불가
- 저널: University of Strathclyde
- DOI: 확인 불가
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/03_wearable_imu_and_portable_sensing/Evaluating the Accuracy of IMU Devices in Detecting Clinically Significant Changes in Knee Flexion and Extension Angles.pdf
- 분석 provider: antigravity

## 연구 목적

- 임상적으로 유의미한 임계값 범주 내에서 무릎 굴곡 각도를 측정할 때 상용화된 Stryker의 MotionSense™ 기술과 Seel 알고리즘을 적용한 유선 연구용 IMU 기기 두 가지의 정확도를 비교 평가하는 것이다. *(근거: PAGE 9, Abstract)*
	- 근거 원문: “This study aimed to evaluate the accuracy of two different wearable IMU devices (a Stryker (USA) commercially available technology, MotionSense™ and a wired IMU research device implementing the Seel Algorithm (Seel, Raisch and Schauer, 2014), in measuring knee flexion angles within clinically significant thresholds.”

## 연구 설계와 대상

- 다양한 연령층으로 구성된 건강한 성인 34명(20-36세의 젊은층 20명, 60-84세의 고령층 14명)과 수술 전 및 수술 후(수술 후 1주 및 6주)의 무릎 인공관절 치환술(TKA) 환자군 10명(53-71세)을 대상으로 다양한 일상생활 활동(ADLs) 전반에 걸쳐 측정을 수행하였다. *(근거: PAGE 9, Abstract)*
	- 근거 원문: “Measurements were evaluated across a diverse healthy adult population of varying ages (20 healthy younger participants, ages ranging between 20 - 36 years old and 14 healthy older participants, ages ranging between 60 - 84 years old) and within a TKA clinical population (10 TKA participants, ages ranging between 53 - 71 years old) both preoperatively and postoperatively (1 week postoperatively and at 6 weeks postoperatively), across a broad range of activities of daily living (ADL’s).”

## 방법

- 상용화된 MotionSense™ 기술은 Madgwick 필터를 구현한 독점 소프트웨어 모바일 앱을 이용해 시상면 무릎 각도를 측정하며, 유선 연구용 IMU 기기는 Seel 알고리즘을 사용하여 계산한다. 두 기기의 측정값은 하체에 16개의 재귀반사 마커를 부착한 Plug-In Gait(PIG) 모델 규격의 광학식 모션 캡처 시스템인 Vicon의 측정값과 대조되었다. *(근거: PAGE 9, Abstract)*
	- 근거 원문: “The commercially available MotionSense™ technology determines sagittal plane knee angle using a mobile-based app with proprietary software that implements a Madgwick filter (Madgwick, 2010), while the wired research IMU device calculates sagittal plane knee angle using the Seel algorithm (Seel, Raisch and Schauer, 2014). Both technologies’ measurements were compared against the gold standard opto-electronic motion capture system, Vicon, which tracked 16 retro-reflective markers that were attached to the lower body as per the PlugInGaitTM (PIG) model.”
- 두 IMU 기술 모두 무릎 굴곡의 영점이 마커 부착 위치에 영향을 받기 때문에, 각 데이터 세트에서 평균 무릎 굴곡 값을 먼저 차감한 후에 각 운동 주기 창 내에서 두 기술 간의 평균 제곱근 오차(RMSE)를 도출하였다. *(근거: PAGE 10, Abstract)*
	- 근거 원문: “For both IMU technologies the zero point for knee flexion depends on marker placement, therefore, the mean knee flexion was subtracted from each data set before calculating a root mean square error (RMSE) between the technologies, determined in each movement cycle window.”
- **[AS-IS]** MotionSense™ 데이터의 분석은 MATLAB의 interp1 함수를 통해 100Hz로 업샘플링을 거쳐 교차상관 함수 xcorr을 적용함으로써 최대 굴곡 지점을 기준으로 각 주기 데이터를 동기화했다. *(근거: PAGE 10, Abstract)*<br>**[TO-BE]** MotionSense™ 데이터 분석에서는 MATLAB interp1 함수로 100Hz 업샘플링을 수행한 뒤, peak flexion에서 peak flexion까지 식별한 movement cycle window를 xcorr 기반 교차상관으로 시간 동기화하였다.<br>*(사실검증 — 과장/경미: 원문은 MotionSense™ 분석에서 peak flexion-to-peak flexion으로 식별된 movement cycle windows를 cross-correlation으로 시간 동기화했다고 설명한다. 요약의 ‘각 주기 데이터’는 모든 주기 데이터 전반을 포괄하는 표현처럼 읽혀 원문의 한정된 분석 창 표현보다 약간 넓다.)*
	- 근거 원문: “Following up-sampling to 100Hz using the MATLAB (MathWorks, 2024) interp1 function, cross-correlation was used to time synchronise the movement cycle windows identified from peak flexion to peak flexion using the xcorr MATLAB (MathWorks, 2024) function for each technology.”

## 핵심 결과

- 건강한 집단과 임상 집단 모두에서 그리고 더 큰 가동범위(ROM)와 빠른 관절 속도를 포함한 모든 동작 시 두 기기 모두 5° 미만의 RMSE를 보여주었으며, 기기별 오차는 MotionSense™가 0.86° - 4.70°, 유선 IMU 기기가 2.92° - 4.78°의 범위를 보였다. *(근거: PAGE 10, Abstract)*
	- 근거 원문: “Results presented RMSE of less than 5° across both devices, across both healthy and clinical populations and across all activities, including those involving larger ROM and higher joint velocities. RMSE values ranged between 0.86° - 4.70° for the MotionSense™ device, while RMSE values ranged between 2.92° - 4.78° for the wired IMU device.”
- 평가된 각 센서 기술에 있어 건강한 피험자 및 환자군 등 대상 그룹 간 정확도의 통계적으로 유의미한 차이는 인정되지 않았다 (p \> 0.05). *(근거: PAGE 10, Abstract)*
	- 근거 원문: “No statistically significant differences between the population groups for each technology was evidenced (p \> 0.05).”
- 더 큰 각도의 무릎 굴곡이 수반되는 활동에서 측정 시스템 간에 더 큰 편차가 관찰되었으며, 젊고 건강한 피험자의 굴곡/신전 활동(ROM 116.5°) 시 RMSE는 3.65°였으나 무릎 치환 수술 후 1주 차 환자의 보행(ROM 31.6°) 시 RMSE는 1.48°에 머물렀다. *(근거: PAGE 10-11, Abstract)*
	- 근거 원문: “Notably, greater discrepancies between the measurement systems were observed during activities involving larger degrees of flexion, for example during the flexion/extension activity performed by the younger healthy population a ROM of 116.5° and RMSE of 3.65° was reported between MotionSense™ and Vicon opto-electronic motion capture system, whereas a RMSE of 1.48° and a ROM of 31.6° was reported for the 1 week postoperative session for the walking activity.”
- 보행 활동에서 디딤기(stance phase)보다 빠른 움직임이 일어나는 흔듦기(swing phase) 구간 동안 두 측정 시스템 간에 상대적으로 더 큰 불일치가 확인되었다. *(근거: PAGE 11, Abstract)*
	- 근거 원문: “Furthermore larger differences were also evidenced during periods associated with faster motion (swing phase displayed larger differences compared to the stance phase for the walking activity).”
- 분석 대상이 된 웨어러블 IMU 기기들은 준수한 상관 계수를 보이며 모든 피험자 그룹의 무릎 굴곡 운동 양상을 정밀하게 추적할 수 있음을 검증하였다. *(근거: PAGE 11, Abstract)*
	- 근거 원문: “The wearable IMU technologies revealed strong coefficients of correlation and were able to accurately track knee flexion patterns across all population groups.”

## 저자 결론

- 웨어러블 IMU 장치는 시상면 무릎 관절 각도를 정밀하게 측정할 수 있어 임상 환경으로의 실효성 있는 도입을 뒷받침하며, 원내 정기 대면 평가를 대체하여 환자의 기능 회복 추이를 원격으로 연속 모니터링할 수 있는 대안을 제시한다. *(근거: PAGE 11, Abstract)*
	- 근거 원문: “This study concludes that wearable IMU devices can accurately measure sagittal knee angle supporting their integration into clinical settings. Their ability to provide accurate, objective data validates their use as a practical alternative to traditional in-clinic assessments, particularly in enabling remote and continuous tracking of patient progress. As such, IMUs may represent a valuable asset in modern rehabilitation strategies, facilitating more efficient, patient-centred care.”
- TKA 환자 코호트의 결과는 환자 개개인의 회복 양상과 수술 예후가 고도로 개별화되어 있음을 시사하며, 따라서 개인 맞춤형 재활 프로그램의 시행과 이를 뒷받침할 수 있는 혁신적인 정밀 측정 기술들의 통합이 필요함을 역설한다. *(근거: PAGE 11, Abstract)*
	- 근거 원문: “The findings from the TKA cohort underscore the highly patient-specific nature of recovery and postoperative outcomes, further emphasising the need for personalised rehabilitation approaches and the requirement for innovative technologies to deliver this level of personalised care.”

## 연구의 한계

- 스폰서인 Stryker사와의 계약 조항에 따라 Stryker의 MotionSense™ 장치와 연구에 쓰인 다른 유선 IMU 기기 간의 데이터를 활용한 직접적인 맞비교 분석이 금지되었으며, 이에 따라 두 시스템에 대한 데이터 가공 및 분석 처리를 독자적인 상이한 방법론을 이용해 별개로 리포트해야 했다. *(근거: PAGE 7, Disclosures and Collaboration)*
	- 근거 원문: “In addition to the collaboration with Philippe Martin, the terms of the contractual agreement with Stryker prohibited direct comparisons between Stryker's technology and other IMU-based systems. As a result, separate analyses were conducted for each IMU technology (MotionSense™ and the wired research IMU device). Differences in analysis methodologies and reporting are therefore intentional and reflect adherence to the contractual requirement to avoid direct comparisons between these technologies.”
- 본 논문의 임상 평가 대상인 무릎 인공관절 치환수술 환자군(TKA cohort)의 표본 수(10명)가 상대적으로 작다는 점이다. *(근거: PAGE 42, 1.1 Introduction)*
	- 근거 원문: “Though this study has a smaller clinical population, evaluations within this clinical group have been carried out across three separate data collection sessions providing a clearer indication of the performance of such devices both preoperatively and postoperatively.”

## 생각해볼 내용

- 임상 및 일반 건강관리 환경에서 웨어러블 IMU 기기를 도입할 시 회복 기간 동안 유효한 환자의 원격 비대면 모니터링 체계를 구축할 수 있고 홈 재활 프로토콜에 대한 순응도를 효과적으로 유도할 수 있는 상당한 임상적 효용을 가진다. *(근거: PAGE 11, Abstract)*
	- 근거 원문: “The use of wearable IMUs within clinical and healthcare settings offers substantial benefits within the recovery period, including remote monitoring capabilities and enhanced compliance with rehabilitation protocols.”
- 수술 전, 수술 중 및 수술 후 등 다차원적이고 장기적인 시점에서 객관적인 기능적 운동 데이터와 주관적인 환자 설문(PROMs) 데이터를 동시에 수집 및 연계하여 분석함으로써, 수술 후 최적의 기능 회복을 보장하는 인자들에 관한 더 종합적인 분석과 이해를 이룰 수 있다. *(근거: PAGE 45, 1.2 Clinical Problem)*
	- 근거 원문: “Moreover, by collecting both objective and subjective data at various timepoints: preoperative, intraoperative, and postoperative, a broader and more detailed understanding can be gained from the different factors that contribute to more favourable postoperative outcomes.”

## 이 연구가 지적한 선행연구의 문제점

- TKA 수술 직후 회복을 측정하기 위해 웨어러블 센서의 측정 유효성을 검증한 문헌은 제한적이며, 특히 관절 속도, 가해지는 충격 및 관절 가동범위(ROM)의 스펙트럼이 폭넓은 다채로운 기능성 동작들을 다루거나 고령 임상 집단과 연령대를 매칭시킬 수 있는 고령의 건강 대조군을 포함해 분석한 연구가 없었다. *(근거: PAGE 39-40, 1.1 Introduction)*
	- 근거 원문: “There is limited literature establishing the validity of wearable sensors to assess knee function shortly following TKA. Particularly literature that focusses on evaluating the accuracy of such devices over many different types of functional activities, that vary in speed, impact and across a broad ROM, that incorporate a relatively large healthy control group of both younger and older participants which presents an opportunity to age-match to a TKA clinical population.”
- 기존 선행 유효성 검사 문헌들은 표본 크기가 극히 협소하거나 관찰 데이터를 단일 시점에서만 취득하였고, 혹은 단순 보행이나 평이한 무릎 굴곡/신전 활동 등 제한된 몇 가지 움직임에 대해서만 유효성 평가를 수행하는 데 그쳤다. *(근거: PAGE 40, 1.1 Introduction)*
	- 근거 원문: “Of the available literature, only a handful (Antunes et al., 2021; Chen et al., 2022; Cornish et al., 2024; Fain et al., 2024; Hafer et al., 2020; Parrington et al., 2021; Wang et al., 2025; Versteyhe et al., 2020) evaluate the accuracy of such devices within a clinical population. However, these studies generally include a restricted population pool, record data at a single time point or only include a simple flexion/extension movement or walking.”
- 보통 이전의 실험들은 3-12명의 매우 한정된 수의 젊고 건강한 피험자들을 모집하여 각자 다른 광학 분석 장치와 마커 모델 표준하에서 기기를 검증함으로써 다양한 특이 보행을 보이는 임상 환자군에 적용하기에 일반화 가능성이 현저히 떨어졌다. *(근거: PAGE 40, 1.1 Introduction)*
	- 근거 원문: “Typically, investigations have recruited younger healthy cohorts with a maximum 3 - 12 individuals, all assessing different IMU technologies and algorithms against different 3D motion capture systems and models (Poitras et al., 2019).”
- 대부분의 선행 문헌들이 건강한 젊은 성인군이나 고정된 특정 단일 회복 시점의 환자만을 고집하였고 고령층이나 조기 수술 후 급성기의 실제 회복 상태를 제대로 대표하지 못하였다. *(근거: PAGE 41, 1.1 Introduction)*
	- 근거 원문: “Many studies focus exclusively on either healthy younger adults or patients at a single stage of recovery (Antunes et al., 2021; Cornish et al., 2024; Fain et al., 2024; Parrington et al., 2021; Versteyhe et al., 2020), often omitting older adults or those in the early postoperative period.”

## 이 연구의 해결 방식과 기여

- 본 연구는 임상 환자군의 나이대와 부합하는 20세부터 84세에 이르는 넓은 나이 범위의 건강 성인 대조군 34명을 확보하고 연령 증가에 따르는 보행 운동학적 변동성을 평가에 반영하였다. *(근거: PAGE 42, 1.1 Introduction)*
	- 근거 원문: “By including a larger healthy cohort of 34 individuals across a wide age range (20– 84 years old), which enables similar age group comparisons to the TKA population, enhancing the clinical relevance of the findings, however, also taking into consideration the natural variations within gait kinematics of healthy individuals as they age.”
- 수술 전 상태의 환자 및 수술 후 서로 다른 단계에 위치한 무릎 치환 수술 환자군을 종단적으로 모두 포괄함으로써 시간에 따른 회복 국면별 IMU 정확도 성능을 조망할 수 있는 구조를 취했다. *(근거: PAGE 42, 1.1 Introduction)*
	- 근거 원문: “Furthermore, the inclusion of both preoperative and postoperative TKA patients enables the assessment of IMU performance across different stages of the recovery process.”
- MATLAB을 이용하여 독자적인 무릎 관절 굴곡 연산 알고리즘을 검증함으로써 특정 하드웨어 제조사에 결속되지 않는 유연하고 비용 효율이 높으며 실용적인 대안 모니터링 경로를 개척하였다. *(근거: PAGE 42, 1.1 Introduction)*
	- 근거 원문: “By validating this bespoke IMU knee flexion algorithm in MATLAB (MathWorks, 2024), it becomes possible to use any IMU device to measure a patient's knee ROM throughout recovery, offering a cost-effective, adaptable and practical alternative to conventional methods such as motion capture systems.”

## 레퍼런스할 수 있는 내용

### 1. IMU 기기의 정확도 (RMSE 5° 미만)

- 원문 발췌: “Results presented RMSE of less than 5° across both devices, across both healthy and clinical populations and across all activities, including those involving larger ROM and higher joint velocities.”
- 한국어 번역: 두 기기 모두 건강한 대상자군과 임상군 모두에서, 그리고 더 큰 가동범위(ROM)와 더 빠른 관절 속도를 포함하는 모든 활동에서 5° 미만의 RMSE를 나타냈다.
- 원문 위치: PAGE 10, Abstract
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: 이 문장은 MotionSense™ 및 Seel 알고리즘을 적용한 IMU 기기가 Vicon 모션 캡처 시스템 대비 시상면 상의 무릎 굴곡 각도 측정에서 임상적으로 허용 가능한 수준(5° 미만)의 오차를 보여줌을 뒷받침함. 3차원 보행 분석과의 비교 분석 결과이며 시상면(Sagittal plane) 무릎 각도에 국한됨.

### 2. MotionSense™ 및 유선 IMU 기기의 구체적 RMSE 오차 범위

- 원문 발췌: “RMSE values ranged between 0.86° - 4.70° for the MotionSense™ device, while RMSE values ranged between 2.92° - 4.78° for the wired IMU device.”
- 한국어 번역: MotionSense™ 기기의 경우 RMSE 값이 0.86° - 4.70° 범위였으며, 유선 IMU 기기의 경우 RMSE 값이 2.92° - 4.78° 범위였다.
- 원문 위치: PAGE 10, Abstract
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: 본 논문에서 검증한 두 기기(MotionSense™와 유선 연구용 IMU 기기)의 구체적인 무릎 각도 오차 범위를 지시함. 다른 환경이나 부착 조건에서는 오차 범위가 달라질 수 있음에 유의.

### 3. 대상자 그룹 간 무릎 각도 측정 오차의 통계적 차이 부재

- 원문 발췌: “No statistically significant differences between the population groups for each technology was evidenced (p \> 0.05).”
- 한국어 번역: 각 기술에 대해 인구 집단 그룹 간에 통계적으로 유의미한 차이는 입증되지 않았다 (p \> 0.05).
- 원문 위치: PAGE 10, Abstract
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: 본 연구에 참여한 젊은 건강 대조군, 고령 건강 대조군, TKA 임상 환자군 간에 IMU 측정 정확도(오차)가 통계적으로 유의미한 차이를 보이지 않았음을 지지함.

### 4. 무릎 굴곡 각도 크기에 따른 측정 오차의 영향

- 원문 발췌: “Notably, greater discrepancies between the measurement systems were observed during activities involving larger degrees of flexion, for example during the flexion/extension activity performed by the younger healthy population a ROM of 116.5° and RMSE of 3.65° was reported between MotionSense™ and Vicon opto-electronic motion capture system, whereas a RMSE of 1.48° and a ROM of 31.6° was reported for the 1 week postoperative session for the walking activity.”
- 한국어 번역: 특히 굴곡 각도가 큰 활동에서 측정 시스템 간에 더 큰 불일치가 관찰되었는데, 예를 들어 젊고 건강한 인구 집단이 수행한 굴곡/신전 활동(가동범위 116.5°)에서는 MotionSense™와 Vicon 광학식 모션 캡처 시스템 간에 3.65°의 RMSE가 보고된 반면, 수술 후 1주 차 보행 활동(가동범위 31.6°)에서는 1.48°의 RMSE가 보고되었다.
- 원문 위치: PAGE 10-11, Abstract
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: 무릎 관절 가동범위(ROM)의 크기가 증가할수록 IMU와 모션 캡처 시스템 간의 측정 오차(RMSE)가 증가하는 경향이 있음을 지지함. 임상적으로 대각도 굴곡 동작 시 센서의 움직임이나 연산 방식의 한계로 오차가 커질 수 있음을 유념해야 함.


---

# [23] IMU-Based Joint Angle Measurement for Gait Analysis

(저자: Thomas Seel, Jörg Raisch and Thomas Schauer | 연도: 2014 | 저널: Sensors | DOI: https://doi.org/10.3390/s140406891)

Seel, T., Raisch, J., & Schauer, T. (2014). IMU-Based Joint Angle Measurement for Gait Analysis. Sensors, 14(4), 6891-6909. https://doi.org/10.3390/s140406891

## 서지정보

- 저자: Thomas Seel, Jörg Raisch and Thomas Schauer
- 연도: 2014
- 저널: Sensors
- DOI: 10.3390/s140406891
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/03_wearable_imu_and_portable_sensing/IMU-Based Joint Angle Measurement for Gait Analysis.pdf
- 분석 provider: antigravity

## 연구 목적

- 센서가 인체 세그먼트에 대해 부착된 특정 방향을 가정하지 않고 관성 측정 데이터를 기초로 관절 각도를 계산하는 방법을 제안한다. *(근거: PAGE 1, Abstract)*
	- 근거 원문: “This contribution is concerned with joint angle calculation based on inertial measurement data in the context of human motion analysis. Unlike most robotic devices, the human body lacks even surfaces and right angles. Therefore, we focus on methods that avoid assuming certain orientations in which the sensors are mounted with respect to the body segments.”

## 연구 설계와 대상

- 대퇴부 절단 환자의 보행 시험 데이터를 활용하여 광학식 3차원 동작 포착 시스템과 관성 측정 장치(IMU) 기반 방법을 비교하였다. *(근거: PAGE 1, Abstract)*
	- 근거 원문: “We provide results from gait trials of a transfemoral amputee in which we compare the inertial measurement unit (IMU)-based methods to an optical 3D motion capture system.”
- 대퇴부 절단 환자의 양쪽 다리(보철물 측 및 대조측 다리)의 대퇴부, 하퇴부, 발에 탄성 몸체 스트랩을 사용하여 각 세그먼트 당 1개의 IMU(Xsens MTw)를 위치나 방향 제한 없이 부착하였다. *(근거: PAGE 14, Section 4. Experimental Results and Discussion)*
	- 근거 원문: “Furthermore, we use elastic body straps to equip the upper and lower leg, as well as the foot, of both the prosthesis and the contralateral leg with one inertial measurement unit (Xsens MTw \[1\]) each, as depicted in Figure 7.”

## 방법

- 비선형 최소제곱 오차 함수를 최소화하는 가우스-뉴턴 알고리즘 또는 기타 표준 최적화 방식을 사용하여 관절 축의 방향 및 위치 좌표를 식별한다. *(근거: PAGE 8, Section 3.1.1)*
	- 근거 원문: “This optimization might be implemented using a Gauss-Newton algorithm, as further described in \[12\], or any other standard optimization method \[31\].”
- 관절 위치 벡터를 찾기 위해 비선형 최소제곱 기법인 가우스-뉴턴 알고리즘을 사용해 오차를 최소화한다. *(근거: PAGE 10, Section 3.1.3)*
	- 근거 원문: “We minimize Ψ̃(o1,o2) over its arguments via a Gauss-Newton algorithm, the implementation of which is described in \[12\].”
- 자이로스코프 기반 각도와 가속도 기반 각도를 결합하기 위해 상보 필터나 칼만 필터와 같은 센서 융합 도구를 사용한다. *(근거: PAGE 12, Section 3.2.2)*
	- 근거 원문: “Therefore, it is advantageous to combine both angles using a standard tool of sensor fusion, e.g., a complementary filter \[32\] or a Kalman filter.”

## 핵심 결과

- 모든 시험에서 두 가지 IMU 기반 방법은 관성 데이터를 완전히 다른 방식으로 사용함에도 불구하고 유사한 값을 도출했다. *(근거: PAGE 15, Section 4. Experimental Results and Discussion)*
	- 근거 원문: “In all trials, both IMU-based approaches yield similar values, although they use the inertial data in completely different ways.”
- 인체 다리에서의 무릎 각도 측정 오차는 보철물 측 오차보다 약 4배 더 컸다. *(근거: PAGE 16, Section 4. Experimental Results and Discussion)*
	- 근거 원문: “It is important to note that the errors on the human leg are about four times larger than on the prosthesis.”
- **[AS-IS]** 보철물 측과 대조측 모두 발목관절 저측/배측 굴곡 각도 측정에서 편차가 약 1도 내외였다. *(근거: PAGE 15, Figure 9 Caption)*<br>**[TO-BE]** 발목 저측/배측 굴곡 각도 측정에서 보철물 측 평균 RMSE는 0.81도, 대조측 평균 RMSE는 1.62도였으며, 본문과 그림 설명에서는 전체적으로 약 1도 수준의 편차로 설명되었다.<br>*(사실검증 — 수치오류/경미: Figure 9 캡션은 양쪽 모두 약 1도라고 표현하지만, Table 1의 6회 시험 평균은 보철물 0.81도, 대조측 1.62도로 제시된다. '대조측도 약 1도 내외'라고 단정하면 표의 평균값을 충분히 반영하지 못한다.)*
	- 근거 원문: “Both on the prosthesis side and on the contralateral side, the deviation is about 1◦ .”

## 저자 결론

- 관절의 운동학적 구속조건을 활용하여 임의의 동작 데이터를 통해 관절 축의 방향 및 위치 좌표를 추정하는 방식은 이전에 제안된 보정 자세나 동작이 필요한 방식보다 실용적이고 강건하다. *(근거: PAGE 16, Section 5. Conclusions)*
	- 근거 원문: “We proposed a set of methods that allow us to determine the local joint axis and position coordinates from arbitrary motions by exploitation of the kinematic constraints of the joint.”
- 가속도계와 자이로스코프만 사용하는 방법은 지자기 센서를 배제하므로 실내 환경이나 자기 왜곡이 있는 곳에서도 사용 가능하다. *(근거: PAGE 16, Section 5. Conclusions)*
	- 근거 원문: “The second and novel method employs only accelerometer and gyroscope readings. Since the use of magnetometers is avoided, it can be used indoors and in the proximity of magnetic disturbances.”

## 연구의 한계

- 인체 다리에서는 근육과 피부의 운동으로 인해 관성 센서와 마커가 서로에 대해 상대적으로 움직이며, 이는 측정 오차의 원인이 된다. *(근거: PAGE 16, Section 4. Experimental Results and Discussion)*
	- 근거 원문: “However, on the human leg, the inertial sensors and the markers move relative to each other as a result of muscle and skin motions.”
- 향후 연구는 연부 조직 움직임으로 인한 오차 효과를 어떻게 보상하거나 최소화할 것인가에 집중될 것이다. *(근거: PAGE 17, Section 5. Conclusions)*
	- 근거 원문: “Future research will be dedicated to the question of how these effects can be compensated for or minimized.”

## 생각해볼 내용

- 선행 연구들이 광학 마커를 IMU에 직접 부착하여 연부조직 아티팩트를 우회적으로 피했던 것과 달리, 본 논문은 실제 해부학적 랜드마크에 마커를 배치함으로써 더 현실적이고 가혹한 환경에서 정밀도를 검증하였다. *(근거: PAGE 1, Abstract)*
	- 근거 원문: “Unlike most authors, we place the optical markers on anatomical landmarks instead of attaching them to the IMUs.”
- **[AS-IS]** 마그네토미터를 제외하고 자이로스코프와 가속도계만 사용해 구속조건 식별 및 각도 측정을 성공적으로 완수한 설계 방식은 실생활이나 자성 왜곡이 흔한 병원 실내 임상 시험 환경에서 대단히 유용할 것으로 평가된다. *(근거: PAGE 1, Abstract)*<br>**[TO-BE]** 마그네토미터를 제외하고 자이로스코프와 가속도계만 사용하도록 설계되어, 균질한 자기장에 의존하지 않는다는 점이 장점으로 제시된다.<br>*(사실검증 — 과장/경미: 원문은 자이로스코프와 가속도계만 사용하여 균질한 자기장에 의존하지 않는 방법을 제안한다고 설명한다. '실생활이나 자성 왜곡이 흔한 병원 실내 임상 시험 환경에서 대단히 유용'하다는 평가는 원문보다 적용 맥락과 효용을 강하게 확장한 해석이다.)*
	- 근거 원문: “In particular, we propose methods that use only gyroscopes and accelerometers and, therefore, do not rely on a homogeneous magnetic field.”

## 이 연구가 지적한 선행연구의 문제점

- 일부 문헌에서는 관성 센서가 미리 정의된 방향으로 정밀하게 정렬되어 부착될 수 있다고 현실과 동떨어진 가정을 하거나 정렬 오차 문제를 아예 무시한다. *(근거: PAGE 3, Section 1.3)*
	- 근거 원문: “First, we shall note that in some publications, this problem is ignored completely by assuming that the IMUs can be mounted precisely in a predefined orientation towards the joint; see, e.g., \[9,10\].”
- 센서 부착 방향 및 위치 정보를 수동으로 계측하는 방식은 3차원 공간에서 매우 번거로우며 오차가 발생하기 쉽다. *(근거: PAGE 3, Section 1.3)*
	- 근거 원문: “Both quantities might be measured manually, but in three-dimensional space, this is a cumbersome task that yields low accuracy results, as demonstrated, e.g., in \[9,12\].”
- 자세 보정이나 특정 교정 움직임을 활용하는 방식은 피험자가 지시 동작을 수행하는 정밀도에 의해서 그 보정의 정확도가 제한을 받는다. *(근거: PAGE 4, Section 1.3)*
	- 근거 원문: “However, it is important to note that, both in calibration postures and calibration motions, the accuracy is limited by the precision with which the subject can perform the postures or motions.”
- 지자기 센서 데이터는 강자성 물질 등에 의한 자기 왜곡에 의해 방위각(heading) 추정 시 정확도 저하가 발생할 수 있다. *(근거: PAGE 2, Section 1.1)*
	- 근거 원문: “Therefore, the presence of magnetic disturbances (as induced, e.g., by ferromagnetic material) may limit the accuracy of the orientation estimates, as demonstrated in \[5,6\].”

## 이 연구의 해결 방식과 기여

- 관절의 기하학적/운동학적 제약조건을 비용 함수 최소화에 활용하여 특정 정렬이나 보정 동작 없이도 임의의 움직임 데이터만으로 관절 축 및 위치 좌표를 자동 식별하는 기법을 개발했다. *(근거: PAGE 7, Section 3.1)*
	- 근거 원문: “However, these coordinates can be identified from the measurement data of arbitrary motions by exploiting kinematic constraints, as explained in \[12\].”
- 제안된 모든 방법이 마그네토미터 데이터를 일체 사용하지 않도록 하여 왜곡이 빈번한 환경에서도 신뢰성을 유지할 수 있도록 하였다. *(근거: PAGE 7, Section 3)*
	- 근거 원문: “All of the methods that we will introduce use only angular rates and accelerations, while the use of magnetometer readings is completely avoided.”
- **[AS-IS]** 복잡한 수동 측정이나 캘리브레이션 자세/동작 없이 센서를 부착하고 간단히 다리를 움직이기만 하면 실시간 각도를 출력하는 플러그 앤 플레이 방식의 혁신적 보행 분석 환경을 제공한다. *(근거: PAGE 17, Section 5. Conclusions)*<br>**[TO-BE]** 복잡한 수동 측정이나 정밀한 캘리브레이션 자세/동작을 대체할 수 있어, 센서를 부착하고 몇 초간 다리를 움직인 뒤 실시간 관절각을 얻는 플러그 앤 플레이 보행 분석으로 이어질 가능성을 제시한다.<br>*(사실검증 — 과장/경미: 원문은 이러한 방법이 플러그 앤 플레이 보행 분석의 가능성을 열며, 온라인 사용 구현과 실시간 측정은 향후 연구 주제라고 서술한다. 요약의 '제공한다'와 '혁신적'은 현재 완성된 환경을 이미 제공한 것처럼 표현해 원문보다 강하다.)*
	- 근거 원문: “these new methods open the door to a plug-and-play gait analysis, in which one simply attaches the IMUs, moves the legs for a few seconds and then receives joint angle measurements in real time.”

## 레퍼런스할 수 있는 내용

### 1. IMU 기반 힌지 관절 각도 측정의 가능성

- 원문 발췌: “It has been demonstrated in many publications, e.g., \[7\] and the references therein, that inertial measurement data can be used to calculate hinge joint angles when at least one IMU is attached to each side of the joint.”
- 한국어 번역: 관절의 각 측면에 적어도 하나의 IMU가 부착되었을 때 관성 측정 데이터를 사용하여 힌지 관절 각도를 계산할 수 있음이 \[7\] 및 그 안의 참고문헌 등 많은 문헌에서 입증되었다.
- 원문 위치: PAGE 2, Section 1.2
- 원문 내 인용표기: \[7\]
- 해당 선행문헌: 7. Cheng, P.; Oelmann, B. Joint-Angle Measurement Using Accelerometers and Gyroscopes—A Survey. IEEE Trans. Instrum. Meas. 2010, 59, 404–414.
- 주장 유형: background_citation
- 활용 맥락과 주의: 관절 양측에 IMU를 부착하여 힌지 관절 각도를 계산하는 선행 연구들의 일반적인 배경으로 인용 가능함. 원인용 문헌인 Cheng & Oelmann (2010)을 직접 검토하여 2차 인용에 주의할 것.

### 2. 지자기 센서 오차 요인으로서의 자기 교란

- 원문 발췌: “Therefore, the presence of magnetic disturbances (as induced, e.g., by ferromagnetic material) may limit the accuracy of the orientation estimates, as demonstrated in \[5,6\].”
- 한국어 번역: 따라서 강자성 물질 등에 의해 유발되는 자기 교란의 존재는 \[5,6\]에서 입증된 바와 같이 방향 추정의 정확도를 제한할 수 있다.
- 원문 위치: PAGE 2, Section 1.1
- 원문 내 인용표기: \[5,6\]
- 해당 선행문헌: 5. Bachmann, E.; Yun, X.; Brumfield, A. Limitations of Attitude Estimation Algorithms for Inertial/Magnetic Sensor Modules. IEEE Robot. Autom. Mag. 2007, 14, 76–87. 6. De Vries, W.H.; Veeger, H.E.; Baten, C.T.; van der Helm, F.C. Magnetic distortion in motion labs, implications for validating inertial magnetic sensors. Gait Posture 2009, 29, 535–541.
- 주장 유형: background_citation
- 활용 맥락과 주의: 자기 교란이 지자기 센서 기반 방향 추정(특히 azimuth/heading)의 오차를 초래할 수 있음을 서술하는 선행 연구 배경으로 인용 가능함. 원인용 문헌인 Bachmann (2007) 또는 De Vries (2009)를 참고할 것.

### 3. 제안된 무릎 굴곡/신전 각도 오차 결과

- 원문 발췌: “Root mean square errors of the knee flexion/extension angles are found to be less than 1◦ on the prosthesis and about 3◦ on the human leg.”
- 한국어 번역: 무릎 굴곡/신전 각도의 제곱평균제곱근 오차는 보철물에서 1도 미만, 실제 사람 다리에서 약 3도이다.
- 원문 위치: PAGE 1, Abstract
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: 본 논문의 핵심 실험 결과로, 제안한 알고리즘의 적용 시 보철물 다리와 생체 다리에서의 무릎 각도 오차 범위를 지칭할 때 인용할 수 있음.


---

# [24] Laboratory Environments Mask Gait Differences Between Healthy Participants and Patients With Anterior Cruciate Ligament Reconstruction

(저자: Lauren R. Parola, Vu Phan, Eni Halilaj | 연도: 2026 | 저널: IEEE Transactions on Neural Systems and Rehabilitation Engineering | DOI: https://doi.org/10.1109/TNSRE.2025.3645799)

Parola, L. R., Phan, V., & Halilaj, E. (2026). Laboratory Environments Mask Gait Differences Between Healthy Participants and Patients With Anterior Cruciate Ligament Reconstruction. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 34, 767-775. https://doi.org/10.1109/TNSRE.2025.3645799

## 서지정보

- 저자: Lauren R. Parola, Vu Phan, Eni Halilaj
- 연도: 2026
- 저널: IEEE Transactions on Neural Systems and Rehabilitation Engineering
- DOI: 10.1109/TNSRE.2025.3645799
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/03_wearable_imu_and_portable_sensing/Laboratory Environments Mask Gait Differences Between Healthy Participants and Patients With Anterior Cruciate Ligament Reconstruction.pdf
- 분석 provider: antigravity

## 연구 목적

- 본 연구는 실험실 환경이 수술 후 보행에 대해 내리는 결론을 흐리게 하는지 조사하고, 자연 환경 바이오메카닉스로의 이동을 지지하는 초기 근거를 제공하고자 했다. *(근거: PAGE 2, I. INTRODUCTION)*
	- 근거 원문: “Accordingly, the aim of this study was to investigate whether laboratory environments cloud the conclusions we draw about post-surgical gait, providing initial support for the move toward natural-environment biomechanics.”
- 저자들은 ACLR 환자가 실험실에 비해 일상생활에서 더 긴 보행 주기 시간, 더 큰 이중 지지 의존도, 그리고 더 뚜렷한 절뚝거림을 보일 것이라고 가설을 세웠다. *(근거: PAGE 2, I. INTRODUCTION)*
	- 근거 원문: “We hypothesized that patients after ACLR walk with a longer gait-cycle time, greater reliance on double support, and a more pronounced limp in daily life compared to the laboratory.”
- 저자들은 또한 ACLR 환자와 건강한 참가자 사이의 차이가 실험실보다 일상생활에서 더 클 것이라고 가설을 세웠다. *(근거: PAGE 2, I. INTRODUCTION)*
	- 근거 원문: “Additionally, we hypothesized that differences between patients after ACLR and healthy participants would be higher in daily life than in the laboratory.”

## 연구 설계와 대상

- 본 연구를 위해 최종적으로 6명의 ACLR 후 환자(남성 3명, 여성 3명)와 6명의 건강한 대조군(남성 3명, 여성 3명)이 분석에 포함되었다. *(근거: PAGE 2, II. METHODS)*
	- 근거 원문: “Six patients post-ACLR (3M, 3F; mean ± std age: 16.83 ± 2.48 years; height: 1.74 ± 0.05 m; weight: 68.175 ± 7.82 kg) and six healthy participants (3M, 3F; mean ± std age: 22.83 ± 2.78 years; 1.76 ± 0.078 m; 66.0 ± 9.2 kg) were recruited for this study, after receiving approval from Carnegie Mellon University’s Institutional Review Board and obtaining informed consent.”
- 초기 대상자 중 2명의 환자가 추적 관찰에 실패하고, 건강한 참가자 1명이 3일 동안만 원격 모니터링 데이터를 완료하여 최종 분석에서 제외되어 최종 샘플 수는 12명이 되었다. *(근거: PAGE 2, II. METHODS)*
	- 근거 원문: “Initially, eight patients and seven healthy participants were assessed for eligibility. Two patients failed to follow up and one healthy participant only completed three days of remote monitoring data. Therefore, these three participants were removed from the study before data analysis, yielding a final sample size of 12 participants.”
- 6명의 ACLR 환자는 동일한 외과의에게 단일 사지에 자가 대퇴사두근 건을 이용한 일차 단일 다발 재건술을 받았다. *(근거: PAGE 2, II. METHODS)*
	- 근거 원문: “Six ACLR patients underwent a primary single-bundle reconstruction with an autologous quadriceps tendon on a single limb by the same surgeon.”
- 건강한 참가자는 무릎 부상 또는 다른 주요 근골격계 부상 이력이 없어야 했으며, ACLR 환자는 수술 당시 부상 후 10주 이내여야 했고 보행 기록은 수술 3개월 후에 측정되었다. *(근거: PAGE 2, II. METHODS)*
	- 근거 원문: “To be eligible to participate, healthy participants had to have no prior history of knee injury or any other major musculoskeletal injury. ACLR patients had to be within 10 weeks of injury at the time of surgery. We recorded their gait three months after the surgery.”

## 방법

- 보행 평가는 실험실 러닝머신, 실험실 지상 보행, 일상생활의 세 가지 조건으로 수행되었다. *(근거: PAGE 2, II. METHODS)*
	- 근거 원문: “We evaluated walking in three conditions: laboratory treadmill, laboratory overground, and daily life (Figure 1).”
- 실험실에서 참가자들은 러닝머신 보행 2회(회당 2분) 및 지상 보행 2회(회당 30초) 동안 마커 기반 모션 캡처 및 IMU로 추적되었다. *(근거: PAGE 2, II. METHODS)*
	- 근거 원문: “In the laboratory, participants were tracked with marker-based motion capture and IMUs for two trials of treadmill walking (2 minutes per trial) and two trials of overground walking (30 seconds per trial).”
- 일상생활 모니터링을 위해 참가자들은 4개의 IMU 센서를 집으로 가져가 대퇴부와 정강이 외측에 착용한 채 5일 연속 착용하도록 안내받았다. *(근거: PAGE 2, II. METHODS)*
	- 근거 원문: “For the daily life monitoring component, participants were sent home with four IMU sensors and instructed to wear them for five consecutive days, with the sensors placed laterally on the thighs and shanks.”
- **[AS-IS]** 자이로스코프의 각속도는 차단 주파수 0.25 Hz의 고통과 필터와 35 Hz의 저통과 필터로 필터링되었다. *(근거: PAGE 3, II. METHODS)*<br>**[TO-BE]** 자이로스코프의 각속도는 차단 주파수 0.25 Hz의 고역통과 필터와 35 Hz의 저역통과 필터로 필터링되었다.<br>*(사실검증 — 번역오류/경미: '고통과 필터'는 원문의 high-pass filter를 잘못 옮긴 오탈자성 번역이다. 의미상 '고역통과 필터'가 맞다.)*
	- 근거 원문: “Angular velocities from the gyroscopes were filtered with a high-pass filter with a cutoff frequency of 0.25 Hz, followed by a low-pass filter with a cut-off frequency of 35 Hz.”
- 가설 검정을 위해 환경을 반복 측정 요소로 하는 반복측정 분산분석(RM ANOVA)을 사용하였으며, 유의수준은 본페로니 교정을 적용하여 0.017로 설정되었다. *(근거: PAGE 3, II. METHODS)*
	- 근거 원문: “Repeated measures (RM) analyses of variance (ANOVA) were used to test our two leading hypotheses, with the environment as our repeated measure.”

## 핵심 결과

- ACLR 후 환자들은 실험실에 비해 일상생활에서 더 긴 보행 주기와 이중 지지 시간을 나타냈다. *(근거: PAGE 4, III. RESULTS)*
	- 근거 원문: “Patients post-ACLR walked with greater gait-cycle and double-support times in daily life compared to the laboratory (p \< 0.017; Figure 2; Table III).”
- ACLR 후 환자들은 실험실 환경에서는 건강한 참가자들과 유사하게 걸었으나, 일상생활에서는 더 긴 보행 주기 시간, 더 긴 이중 지지 시간, 그리고 더 짧은 단일 지지 시간을 나타냈다. *(근거: PAGE 4, III. RESULTS)*
	- 근거 원문: “Patients post-ACLR walked similarly to healthy participants in the laboratory, but not in daily life, where they walked with longer gait-cycle and double-support time and shorter single-support time (Figure 2).”
- 일상생활 중 ACLR 환자들의 단일 지지 불균형은 건강한 참가자들보다 높았다. *(근거: PAGE 4, III. RESULTS)*
	- 근거 원문: “Single-support asymmetry was also higher in ACLR patients than in healthy participants during daily life.”
- 건강한 참가자는 ACLR 환자보다 하루 평균 935회 더 많은 보행 주기를 완료했다. *(근거: PAGE 5, III. RESULTS)*
	- 근거 원문: “Healthy participants completed 935 more gait cycles per day than ACLR patients (p = 0.016)(Figure 3A).”

## 저자 결론

- 본 연구 결과는 실험실 환경이 건강한 참가자와 ACLR 환자 간의 보행 차이를 숨길 수 있으며, 자연 환경 생체역학으로의 이동이 수술 후 보행에 대한 이해를 확장하는 데 새로운 기대를 제시함을 보여준다. *(근거: PAGE 6, IV. DISCUSSION)*
	- 근거 원문: “While we did not study the factors that make the gait laboratory not an ideal environment to capture free-living behavior, on their own, these findings provide initial evidence that the laboratory environment masks gait differences between healthy participants and ACLR patients and suggests that the move to natural-environment biomechanics holds new promise in expanding our understanding of post-surgical gait.”
- 실험실과 일상생활 간에 관찰된 보행 특징의 차이는 기존의 보행 분석 연구들이 실제 자연 환경에서의 행동을 현실적으로 포착하지 못했을 가능성을 시사한다. *(근거: PAGE 1, Abstract)*
	- 근거 원문: “While the reasons behind the observed differences were not studied, these results suggest that gait-analysis studies to date may have not realistically captured natural-environment behavior.”
- 실험실에서 관찰되는 것과 일상생활에서의 보행 차이는 실험실 관찰 내용이 일상생활의 운동 패턴을 대변하지 못할 수 있음을 나타낸다. *(근거: PAGE 7, IV. DISCUSSION)*
	- 근거 원문: “The differences we observed in temporal parameters between groups in daily life, however, suggest that what is observed in the laboratory may not be representative of movement patterns in daily life.”

## 연구의 한계

- 건강한 대조군과 ACLR 그룹의 나이가 매칭되지 않았다. *(근거: PAGE 6, IV. DISCUSSION)*
	- 근거 원문: “First, healthy and ACLR groups were not age-matched.”
- 원격 모니터링 동안 참가자가 매일 아침 직접 IMU를 장착했으며, 센서-신체 보정이 수행되지 않았다. *(근거: PAGE 6, IV. DISCUSSION)*
	- 근거 원문: “Second, participants were tasked with applying the IMUs themselves each morning during the remote monitoring component of the study and no sensor-to-body calibration was implemented before extracting temporal parameters from the sagittal plane angular velocity data.”
- 실험실 환경에 비해 일상생활에서 다양한 지형 조건이 존재했으나 이를 통제하지 못했다. *(근거: PAGE 6, IV. DISCUSSION)*
	- 근거 원문: “A third limitation is the variable terrain in daily life compared to the laboratory.”
- 사후 검정력 분석에 따르면 일부 비유의미한 보행 변수는 통계적 유의성을 확인하기 위해 더 큰 표본 크기가 필요하다. *(근거: PAGE 7, IV. DISCUSSION)*
	- 근거 원문: “Fourth, a post-hoc power analysis for the second hypothesis revealed that limp and step time asymmetry required a larger sample size to help in rejecting the null.”
- 대퇴사두근 근력 등 회복 특성이 수술 후 보행에 미칠 수 있는 영향을 고려하지 않고, 모든 ACLR 회복 표현형을 단일 집단으로 묶어 분석하였다. *(근거: PAGE 7, IV. DISCUSSION)*
	- 근거 원문: “Clustering all ACLR recovery phenotypes into a single population is therefore not ideal, and modern data science tools such as cluster analysis and larger datasets will play a critical role in refining future analyses and insights.”

## 생각해볼 내용

- 실험실 보행 분석 연구가 보행 대칭성의 자연스러운 회복 시점을 실제보다 이르게 평가했을 수 있음을 설명한다. *(근거: PAGE 7, IV. DISCUSSION)*
	- 근거 원문: “Our results, together with the growing body of evidence in other clinical populations, therefore, call into question our existing understanding of gait recovery timelines, which have largely been informed by laboratory-based assessment and highlight the need for remote assessments to address the gait lab effect.”
- **[AS-IS]** 일상생활에서의 다양한 지형 조건이나 주의 분산 요소가 오히려 수술 후 환자의 보상적 보행 전략을 관찰하는 데 필수적인 자극이 될 수 있음을 의미한다. *(근거: PAGE 7, IV. DISCUSSION)*<br>**[TO-BE]** 주의 분산이 거의 없는 실험실 보행은 참가자가 보상적 보행 전략을 가리게 만들 수 있으므로, 일상생활 보행 관찰이 실험실에서 드러나지 않는 보행 양상을 포착하는 데 도움이 될 수 있다.<br>*(사실검증 — 과장/경미: 원문은 실험실의 주의 분산이 없는 환경이 보상적 보행 전략을 가릴 수 있고, 일상생활의 지형 차이는 통제하지 못한 한계라고 설명한다. 그러나 요약은 다양한 지형 조건이나 주의 분산 요소가 '필수적인 자극'이라고 더 강하게 해석한다.)*
	- 근거 원문: “Distraction-free laboratory gait may inadvertently cause a participant to mask compensatory gait strategies in the laboratory.”
- 동작 분석이 고가의 실험실 환경에서 간편한 모바일 웨어러블 시스템으로 전환되면 임상 연구와 임상 진료 현장 간의 격차를 해소할 수 있다. *(근거: PAGE 8, IV. DISCUSSION)*
	- 근거 원문: “Even if traditional laboratory studies identified the gait patterns leading to poor recovery and eventual osteoarthritis, the gap between research and clinical practice would remain wide because traditional gait analysis is too expensive and time-consuming for implementation in clinics.”

## 이 연구가 지적한 선행연구의 문제점

- 마커 기반 모션 캡처는 실험실 공간, 고가의 장비 및 전문 연구원이 필요해 임상 및 대규모 생체역학 연구로의 확장이 제한된다. *(근거: PAGE 1, I. INTRODUCTION)*
	- 근거 원문: “Marker-based motion capture, which has been historically used to study gait, requires designated laboratory space, expensive equipment, and trained biomechanists, preventing scalability to clinics and large biomechanics research studies.”
- 또한, 연구자의 존재가 환자의 보행 움직임에 영향을 미칠 수 있다. *(근거: PAGE 1, I. INTRODUCTION)*
	- 근거 원문: “Additionally, the presence of researchers may influence how patients move.”
- 수술 후 보행에 관한 기존의 지식은 대부분 실험실 내 평가를 기반으로 하고 있어, 일상 모니터링이 이러한 통찰력을 어떻게 바꿀 수 있는지 보여주는 연구가 아직 없었다. *(근거: PAGE 1, I. INTRODUCTION)*
	- 근거 원문: “Existing knowledge of post-ACLR gait is largely based on laboratory assessments, with no study to date providing evidence of how daily-life monitoring may reshape these insights.”

## 이 연구의 해결 방식과 기여

- 본 연구는 ACLR 수술 후 원격 환자 모니터링에서 다중 센서 프로토콜 사용에 대한 초기 지지 근거를 제공한다. *(근거: PAGE 8, IV. DISCUSSION)*
	- 근거 원문: “Altogether, this work provides initial support for the use of multi-sensor protocols in remote patient monitoring following ACLR surgery.”
- 웨어러블 센서는 수술 후 보행에 대한 깊은 이해와 외상 후 골관절염 위험에 처한 환자의 특정 패턴을 파악할 수 있는 경로를 제공한다. *(근거: PAGE 1, Abstract)*
	- 근거 원문: “Wearable sensors now offer a path toward deeper understanding of post-surgical gait and the specific patterns that may place certain patients at risk for post-traumatic osteoarthritis.”

## 레퍼런스할 수 있는 내용

### 1. 실험실 참여 의식이 보행 속도 등에 미치는 영향 (호손 효과)

- 원문 발췌: “Recent work has identified a similar effect in gait analysis, where study participants increase their gait speed, cadence, and stride length when researchers are present in the laboratory \[3\], \[4\], \[5\].”
- 한국어 번역: 최근 연구에 따르면 보행 분석에서도 유사한 효과가 확인되었는데, 연구원이 실험실에 있을 때 피험자들은 보행 속도, 분당 걸음 수, 보폭을 증가시킨다.
- 원문 위치: PAGE 1, I. INTRODUCTION
- 원문 내 인용표기: \[3\], \[4\], \[5\]
- 해당 선행문헌: \[3\] K. B. Friesen, Z. Zhang, P. G. Monaghan, G. D. Oliver, and J. A. Roper, “All eyes on you: How researcher presence changes the way you walk,” Sci. Rep., vol. 10, no. 1, Oct. 2020, Art. no. 1, doi: 10.1038/s41598-020-73734-5. \[4\] J. Jeon et al., “Influence of the Hawthorne effect on spatiotemporal parameters, kinematics, ground reaction force, and the symmetry of the dominant and nondominant lower limbs during gait,” J. Biomechanics, vol. 152, May 2023, Art. no. 111555, doi: 10.1016/j.jbiomech.2023.111555. \[5\] L. A. Hutchinson, M. J. Brown, K. J. Deluzio, and A. R. De Asha, “Self-selected walking speed increases when individuals are aware of being recorded,” Gait Posture, vol. 68, pp. 78–80, Feb. 2019, doi: 10.1016/j.gaitpost.2018.11.016.
- 주장 유형: background_citation
- 활용 맥락과 주의: 실험실 환경에서의 보행 평가가 호손 효과로 인해 실제 일상 보행 능력을 왜곡할 수 있음을 지적할 때 유용하다. 2차 인용에 주의해야 한다.

### 2. ACLR 환자의 일상생활 및 실험실 간 보행 주기와 이중 지지 차이

- 원문 발췌: “We found that patients following ACLR walk with longer gait-cycle times and rely more on double support in daily life compared to the laboratory.”
- 한국어 번역: 우리는 ACLR 후 환자들이 실험실에 비해 일상생활에서 보행 주기가 길어지고 이중 지지에 더 많이 의존하며 걷는다는 것을 발견했다.
- 원문 위치: PAGE 6, IV. DISCUSSION
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: ACLR 수술 후 환자들이 일상생활에서 실험실 검사 시보다 더 조심스럽거나 보수적인 보행 패턴(긴 보행 주기, 긴 이중 지지 시간)을 채택함을 나타내는 핵심 정량 결과로 직접 인용할 수 있다.

### 3. ACLR 환자와 건강 대조군의 실험실 및 일상생활 보행 차이 비교

- 원문 발췌: “Patients walked similarly to healthy participants in the laboratory, but in daily life they walked with a longer gait-cycle time (1.22 ± 0.03 s vs. 1.08 ± 0.06 s), longer double-support phase (21.8 ± 1.3 % vs. 17.9 ± 2.7 % Gait Cycle Time), and greater single-support asymmetry (8.4 ± 0.9 % vs 4.1 ± 0.7 %).”
- 한국어 번역: 환자들은 실험실에서 건강한 참가자들과 유사하게 걸었으나, 일상생활에서는 더 긴 보행 주기 시간(1.22 ± 0.03초 대 1.08 ± 0.06초), 더 긴 이중 지지 단계(보행 주기 시간의 21.8 ± 1.3% 대 17.9 ± 2.7%), 그리고 더 큰 단일 지지 대칭성 결여(8.4 ± 0.9% 대 4.1 ± 0.7%)를 보였다.
- 원문 위치: PAGE 1, Abstract
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: 실험실 환경이 환자와 건강한 대조군 간의 보행 차이를 차단(마스킹)하여 수술 후 회복 평가를 왜곡할 수 있다는 주장을 뒷받침하는 핵심 정량 지표다.


---

# [25] The COMPWALK-ACL: A Dataset of Multi-pace IMU Gait Kinematics in Adolescents, Adults, and ACL Injured Patients

(저자: Tomer Yona, Bezalel Peskin, Arielle Fischer | 연도: 2026 | 저널: Scientific Data | DOI: https://doi.org/10.1038/s41597-025-06307-8)

Yona, T., Peskin, B., & Fischer, A. (2026). The COMPWALK-ACL: A Dataset of Multi-pace IMU Gait Kinematics in Adolescents, Adults, and ACL Injured Patients. Scientific Data, 13(4). https://doi.org/10.1038/s41597-025-06307-8

## 서지정보

- 저자: Tomer Yona, Bezalel Peskin, Arielle Fischer
- 연도: 2026
- 저널: Scientific Data
- DOI: https://doi.org/10.1038/s41597-025-06307-8
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/03_wearable_imu_and_portable_sensing/The COMPWALK-ACL - A Dataset of Multi-pace IMU Gait Kinematics in Adolescents, Adults, and ACL Injured Patients.pdf
- 분석 provider: antigravity

## 연구 목적

- 본 논문은 Xsens Awinda IMU 시스템을 사용하여 획득한 하지 운동역학 데이터셋을 제공하는 것을 목적으로 한다. *(근거: Page 1, Abstract)*
	- 근거 원문: “This paper presentsalowerlimbkinematicdatasetacquiredwiththeXsensAwindaIMUsystem.”
- 연령에 따른 가변성과 수술 후 변화를 표준화된 IMU 프로토콜을 사용하여 포착하는 공개 데이터셋을 개발하고자 했다. *(근거: Page 2, Background & Summary)*
	- 근거 원문: “Toaddress thisgap,wedevelopedtheCOMPWALK-ACLdatasetasanopenlyavailableresourcecapturingbothage-related variability and post-surgical changes using a standardized IMU protocol.”

## 연구 설계와 대상

- 본 연구는 건강한 성인 25명, 건강한 청소년 27명, ACL 부상자 40명(이 중 27명은 ACL 재건술 3개월 후 추적 조사 완료)을 포함한 총 92명의 참가자로 구성된 IMU 기반 보행 데이터셋이다. *(근거: Page 2, Background & Summary)*
	- 근거 원문: “Itis an IMU-based gait dataset comprising data from 92 participants: 25 healthy adults, 27 healthy adolescents, and 40 individuals with ACL injury, of whom 27 completed a follow-up assessment three months after ACL recon- struction.”
- 건강한 성인 코호트는 18\~45세의 건강한 남성과 여성으로 구성되었다. *(근거: Page 2, Methods)*
	- 근거 원문: “The healthy adults’ cohort included healthy males and females aged 18–45 years.”
- 참가자 정보 보호를 위해 모든 데이터는 수집 시점에 비식별화되었다. *(근거: Page 2, Methods)*
	- 근거 원문: “To protect participant information, all data were de-identified at the point of collection.”
- 각 코호트의 인구통계학적 특성 요약은 표 1에 제시되어 있다. *(근거: Page 2, Methods)*
	- 근거 원문: “A summary of demographic characteristics for each cohort is presented in Table 1.”

## 방법

- 보행 테스트 중 참가자들에게 어떠한 외부 피드백도 제공되지 않았다. *(근거: Page 3, Methods)*
	- 근거 원문: “No external feedback was provided.”
- 직선 오버그라운드 보행 구간의 데이터만 분석에 활용되었다. *(근거: Page 3, Methods)*
	- 근거 원문: “Only the straight overground walking segments were used for analysis.”
- 각 속도 조건에 대해 참가자들은 3회의 연속된 트라이얼을 수행했다. *(근거: Page 3, Methods)*
	- 근거 원문: “For each speed condition, participants completed three consecutive trials.”

## 핵심 결과

- 느린 보행 속도에서 건강한 성인은 ACLD 그룹보다 유의하게 더 빠르게 걸었다(0.87±0.02m/s 대 0.79±0.01m/s, p=0.0037). *(근거: Page 5, Technical Validation)*
	- 근거 원문: “At slow walking speed, healthy adults walked significantly faster than the ACLD group (0.87±0.02m/s vs. 0.79±0.01m/s,p=0.0037),withnoothersignificantpairwisedifferences.”
- 모든 속도 조건에서 보행 속도, 케이던스, 보폭에 대해 그룹 간 유의미한 차이가 관찰되었다. *(근거: Page 5, Technical Validation)*
	- 근거 원문: “At all speed conditions, significant group differences were observed in gait speed, cadence, and stride length.”
- 보통 보행 속도에서 ACLD 및 ACLR 그룹은 건강한 성인보다 유의하게 더 느린 속도를 보였다(ACLD/ACLR 모두 1.19m/s 대 건강한 성인 1.33m/s, p\<0.001). *(근거: Page 5, Technical Validation)*
	- 근거 원문: “At normal walking speed, both ACLD and ACLR groups demonstrated significantly slower gait speed than healthy adults (ACLD: 1.19±0.01m/s; ACLR: 1.19±0.02m/s; healthy adults: 1.33±0.02m/s; both p\<0.001).”

## 저자 결론

- 이 데이터셋은 청소년-성인의 전 생애에 걸친 규준 관절각 참조 데이터 구축, 스트라이드 세분화 및 보행 속도 분류 알고리즘의 벤치마킹, 그리고 ACLR 수술 후 회복 경과 평가를 위한 종단적 변화 추적 등 다양한 목적으로 활용될 수 있다. *(근거: Page 7, Usage Notes)*
	- 근거 원문: “The dataset supports a range of uses, including establishing normative joint-angle references across the ado- lescent–adult lifespan, benchmarking stride-segmentation or gait-speed classification algorithms, and tracking longitudinal changes after ACLR to evaluate recovery.”
- 본 데이터셋은 하지 관절 운동역학 및 시공간 파라미터를 포함하여, 다양한 보행 속도와 임상 조건에서 IMU 기반 보행 분석법 개발 및 연령별 규준 보행 연구 등을 가능하게 한다. *(근거: Page 1, Abstract)*
	- 근거 원문: “Thedatasetcontainsspatiotemporalparameters,aswellaslowerlimbjoint kinematics.Itenablesresearchonnormativegaitacrossagegroups,theeffectsofACLinjuryandearly recoveryonmovementpatterns,andthedevelopmentofIMU-basedgaitanalysismethodsunder differentwalkingspeedsandclinicalconditions.”

## 연구의 한계

- 지면반력기 데이터와 카메라 기반 모션 캡처가 누락되어 골드 스탠다드 데이터가 없으므로 관절 각도 및 시공간 지표의 유효성을 검증할 수 없다. *(근거: Page 7, Usage Notes)*
	- 근거 원문: “Second,force-platedataandcamera-based motioncapturearenotincluded.Therefore,withoutgold-standardmeasurements,thedatasetcannotbeusedto validateanyofthereportedmetrics,includingjointkinematicsandspatiotemporalparameters.”
- 각 오버그라운드 테스트의 거리가 약 20m로 짧아, 확보된 안정된 보행 주기(stride)가 8\~10개에 불과하다. *(근거: Page 7, Usage Notes)*
	- 근거 원문: “First, eachovergroundtrialspansonly\~20m,yielding8–10steadystrides.”
- 각 참가자의 세션 내에서는 신발과 보행 표면이 일정했으나 참가자 간에 표준화되지 않아 피험자 간 차이가 유발되었을 수 있다. *(근거: Page 7, Usage Notes)*
	- 근거 원문: “Lastly,footwear and walkway surface were kept consistent within a session but not standardised across participants, possibly introducing between-subject differences.”

## 생각해볼 내용

- **[AS-IS]** 제시된 데이터셋의 보행 지표 측정값 크기가 제조사 백서의 값과 유사하므로 데이터의 생리학적 신뢰도가 높음을 알 수 있다. *(근거: Page 6, Technical Validation)*<br>**[TO-BE]** 제시된 건강한 성인 코호트의 시공간 보행 지표는 Xsens 백서에 보고된 기준값과 크기 면에서 유사했다.<br>*(사실검증 — 과장/경미: 원문은 건강한 성인 코호트의 시공간 보행 파라미터가 Xsens 백서 값과 크기 면에서 comparable하다고만 설명한다. 이를 근거로 데이터의 '생리학적 신뢰도가 높음'을 단정하는 것은 원문보다 강한 해석이다.)*
	- 근거 원문: “Observed values were comparable in magnitude to those reported in the whitepaper.”
- **[AS-IS]** 코드 저장소에 예제 스크립트가 포함되어 연구자들의 데이터셋 재사용성과 활용성이 증대될 것이다. *(근거: Page 7, Code availability)*<br>**[TO-BE]** 코드 저장소에는 데이터셋의 추가적인 잠재 활용 예를 보여주는 몇 가지 예제 스크립트가 포함되어 있다.<br>*(사실검증 — 근거불충분/경미: 원문은 GitHub 저장소에 데이터셋의 잠재적 추가 활용을 보여주는 예제 스크립트가 포함되어 있다고만 말한다. '재사용성과 활용성이 증대될 것'이라는 효과는 원문에서 직접 확인되지 않는다.)*
	- 근거 원문: “The repository also includes a few example scripts illustrating additional potential uses of the dataset.”

## 이 연구가 지적한 선행연구의 문제점

- 보행 속도가 관절 운동역학 및 보상 작용을 크게 조절함에도 불구하고, 건강한 대조군과 ACL 환자군 모두에서 속도 변화를 체계적으로 적용한 보행 데이터셋이 매우 부족하다. *(근거: Page 1, Background & Summary)*
	- 근거 원문: “Yet,fewpubliclyavailabledatasetssystematicallyincorporatevariationinwalkingspeedacrossbothhealthy andACL-injuredpopulations,despitethefactthatspeedstronglymodulatesjointkinematicsandcompensatory strategies10,17 .”
- 기존 데이터셋들은 대부분 동일한 피험자 내에서의 수술 전 및 수술 후 종단적 측정을 포함하지 못하고 있다. *(근거: Page 2, Background & Summary)*
	- 근거 원문: “Additionally, most do not include pre- and post-operative measures within the same individuals18–21 .”

## 이 연구의 해결 방식과 기여

- 기존의 데이터셋들과 대조적으로 COMPWALK-ACL 데이터셋은 고유한 가치와 기여를 가진다. *(근거: Page 2, Background & Summary)*
	- 근거 원문: “Comparedtoexistingdatasets,theCOMPWALK-ACLdataset(COMParingmulti-paceWALKingkinemat- icsviaIMUinhealthyadolescents,adults,andindividualswithACLinjury)providesauniquecontribution.”
- 생태학적으로 타당한 조건 하에 피험자 내 종단적 분석 및 그룹 간 비교를 지원하여 ACL 재건술 후의 생체역학적 적응 연구와 연령대별 규준 마련에 기여한다. *(근거: Page 2, Background & Summary)*
	- 근거 원문: “This dataset supports within-subject andbetween-groupanalysesunderecologicallyvalidconditions,facilitatingthestudyofpost-ACLbiomechan- ical adaptations as well as the development of normative gait references for both youth and adults.”

## 레퍼런스할 수 있는 내용

### 1. 인간 보행 정보의 임상적 가치

- 원문 발췌: “Human gait provides objective information about an individual’s movement capabilities, neurological func- tion, and musculoskeletal health1,2 .”
- 한국어 번역: 인간의 보행은 개인의 움직임 능력, 신경학적 기능 및 근골격계 건강에 대한 객관적인 정보를 제공한다.
- 원문 위치: Page 1, Background & Summary
- 원문 내 인용표기: 1,2
- 해당 선행문헌: 1. Das, R., Paul, S., Mourya, G. K., Kumar, N. & Hussain, M. Recent Trends and Practices Toward Assessment and Rehabilitation of Neurodegenerative Disorders: Insights From Human Gait. Front Neurosci. 16, 859298, https://doi.org/10.3389/fnins.2022.859298 (2022). 2. Winner, T. S. et al. Discovering individual-specific gait signatures from data-driven models of neuromechanical dynamics. PLoS Comput Biol. 19(10), e1011556, https://doi.org/10.1371/journal.pcbi.1011556 (2023).
- 주장 유형: background_citation
- 활용 맥락과 주의: 인간 보행 분석이 신경학적 및 근골격계 건강 상태를 진단하고 평가하는 데 유용한 객관적 지표임을 지지할 때 인용할 수 있다.

### 2. 전통적인 보행 분석 시스템의 한계

- 원문 발췌: “While accurate, these systems are limited to controlled environments and are not suitable for use outside the laboratory7 .”
- 한국어 번역: 이러한 시스템은 정확하지만 통제된 환경으로 제한되며 실험실 외부에서 사용하기에 적합하지 않다.
- 원문 위치: Page 1, Background & Summary
- 원문 내 인용표기: 7
- 해당 선행문헌: 7. Prisco, G. et al. Validity of Wearable Inertial Sensors for Gait Analysis: A Systematic Review. Diagnostics (Basel). 15, https://doi. org/10.3390/diagnostics15010036 (2024).
- 주장 유형: background_citation
- 활용 맥락과 주의: 전통적인 카메라 기반 보행 분석 시스템이 가진 공간적 제약과 실험실 외부 적용의 어려움을 지지할 때 인용할 수 있다.

### 3. IMU 센서의 보행 분석 적용 장점

- 원문 발췌: “Wearable sensors, such as Inertial Measurement Units (IMUs), offer a portable and cost-effective alternative that enables gait assessment in real-world and clinical settings8 .”
- 한국어 번역: 관성 측정 장치(IMU)와 같은 웨어러블 센서는 실제 환경 및 임상 환경에서 보행 평가를 가능하게 하는 휴대 가능하고 비용 효율적인 대안을 제공한다.
- 원문 위치: Page 1, Background & Summary
- 원문 내 인용표기: 8
- 해당 선행문헌: 8. Jung, S. et al. The Use of Inertial Measurement Units for the Study of Free Living Environment Activity Assessment: A Literature Review. Sensors (Basel). 20(19), https://doi.org/10.3390/s20195625 (2020).
- 주장 유형: background_citation
- 활용 맥락과 주의: IMU 센서가 전통적인 랩 기반 장비에 비해 높은 휴대성과 경제성을 가지며 실생활 및 임상 연구에 적합하다는 주장을 뒷받침할 때 사용된다.

### 4. 빠른 속도 조건에서 ACL 그룹과 건강한 대조군의 보행 속도 차이

- 원문 발췌: “At fast speed, both ACL groups exhibited significantly slower gait speed than healthy adults (ACLD: 1.73±0.03m/s; ACLR: 1.66±0.03m/s; healthy adults: 1.94±0.03m/s; both p\<0.001).”
- 한국어 번역: 빠른 속도에서 두 ACL 그룹 모두 건강한 성인에 비해 유의하게 느린 보행 속도를 보였다(ACLD: 1.73±0.03m/s, ACLR: 1.66±0.03m/s, 건강한 성인: 1.94±0.03m/s, 모두 p\<0.001).
- 원문 위치: Page 5, Technical Validation
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: ACL 부상 환자(ACLD) 및 재건술 후 3개월 환자(ACLR)가 빠른 보행 시 건강한 대조군에 비해 보행 속도가 유의하게 감소한다는 자체 분석 결과다.


---

# [26] THE ROLE OF INERTIAL SENSORS IN THE FUNCTIONAL ASSESSMENT OF THE KNEE AFTER ANTERIOR CRUCIATE LIGAMENT RECONSTRUCTION SURGERY. A SYSTEMATIC REVIEW.

(저자: Vassilios Baliotis | 연도: 2022 | 저널:  | DOI: https://doi.org/확인 불가)

Baliotis, V. (2022). The role of inertial sensors in the functional assessment of the knee after anterior cruciate ligament reconstruction surgery: A systematic review (Postgraduate thesis, National and Kapodistrian University of Athens).

## 서지정보

- 저자: Vassilios Baliotis
- 연도: 2022 \> **[AS-IS]** - 저널: National and Kapodistrian University of Athens (Postgraduate Thesis)<br>**[TO-BE]** - 문서 유형/기관: Postgraduate Thesis, School of Medicine, National and Kapodistrian University of Athens<br>*(사실검증 — 사실불일치/경미: 원문은 이 문서를 저널 논문이 아니라 National and Kapodistrian University of Athens 의과대학 석사 과정 요구사항으로 작성된 postgraduate thesis/dissertation로 제시한다. 기관명 자체는 맞지만 '저널'이라는 항목명은 문서 유형을 잘못 암시한다.)*
- DOI: 확인 불가
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/03_wearable_imu_and_portable_sensing/The Role of Inertial Sensors in the Functional Assessment of the Knee After Anterior Cruciate Ligament Reconstruction Surgery - A Systematic Review.pdf
- 분석 provider: antigravity

## 연구 목적

- 본 연구의 주된 목적은 휴대 가능하고 저렴한 관성 센서가 전방십자인대 재건술(ACLR) 환자에게서 신뢰할 수 있는 사지 간 운동학적 비대칭을 밝혀낼 수 있는지에 대한 문헌을 체계적으로 고찰하는 것이다. *(근거: Page 6, ABSTRACT)*
	- 근거 원문: “This article's primary purpose is to systematically review the literature on whether portableandaffordable inertial sensors can reveal reliableinter-limb kinematic asymmetries in ACL reconstructed (ACLR) patients.”
- 관성 측정 장치(IMU)가 재건된 무릎의 기능에서 비대칭을 구별할 수 있는지와 IMU로부터 도출된 측정값이 표준 동작 분석 시스템의 측정값과 유의미하게 상관관계가 있는지를 조사하는 것이다. *(근거: Page 14, 1.6 Objectives)*
	- 근거 원문: “To investigate whether IMUs can distinguish asymmetries in the function of the reconstructed knee and whether metrics derived from IMUs can significantly correlate with metrics from the gold standard motion analysis systems.”

## 연구 설계와 대상

- 본 연구는 PRISMA-DTA 지침을 따르는 체계적 고찰 설계이다. *(근거: Page 6, ABSTRACT)*
	- 근거 원문: “Design: Systematic review following PRISMA-DTA guidelines”
- 최종 선정된 5개 연구에는 총 109명의 17\~40세 피험자가 포함되었으며, 그 중 92명은 전방십자인대 재건술(ACLR)을 받았고 수술 후 시간은 3개월에서 3.5년 사이였다. *(근거: Page 19, 3.1 Demographic data)*
	- 근거 원문: “A total of 109subjects between 17 and40years oldwere examined. Only one study included a healthy control group consisting of 17 people. Ninety-two had undergone ACL reconstruction, with time since surgery ranging from three months to three and a half years.”

## 방법

- PUBMED, CINAHL, SCOPUS, SPORTDISCUS 데이터베이스를 검색하여 동적 테스트나 보행 중 ACLR 피험자를 평가할 때 관성 센서의 사용을 동작 분석 시스템과 비교한 연구를 선정하였다. *(근거: Page 6, ABSTRACT)*
	- 근거 원문: “We conducted a database search of PUBMED, CINAHL, SCOPUS, and SPORTDISCUS, checking for publications that compared the use of inertial sensors to motion analysis systems, assessing subjects with ACLR during a dynamic test or walking.”
- 선정된 연구들의 방법론적 질은 QUADAS-2 체크리스트에 따라 평가되었다. *(근거: Page 6, ABSTRACT)*
	- 근거 원문: “The reviewers appraised every eligible study for its methodological quality according to the QUADAS-2 checklist.”

## 핵심 결과

- 5개 중 4개 연구에서 관성 센서로부터 추출된 특징들이 ACLR 환자들의 사지 간 유의미한 차이를 나타냈다. *(근거: Page 6, ABSTRACT)*
	- 근거 원문: “In four studies, the extracted features from inertial sensors revealed significant inter- limb differences in ACLR patients.”
- 관성 센서에서 추출한 특정 특징들은 동작 분석 시스템의 측정값과 강한 상관관계를 보였다. *(근거: Page 6, ABSTRACT)*
	- 근거 원문: “Some of these features were strongly correlated with metrics acquired from the motion analysis systems.” \> **[AS-IS]** - Sigward 등의 연구에서는 보행의 부하 반응 단계 동안 무릎 신전 모멘트(KMom)와 정경(shank) 각속도 비율 사이에 양의 상관관계가 나타났다. *(근거: Page 23, 3.8 IMUs correlation with the Motion analysis system)* \> \> **[TO-BE]** Sigward 등의 연구에서는 보행의 부하 반응 단계 동안 무릎 신전 모멘트(KMom)와 정강이 각속도(SAV) 사이에 양의 상관관계가 나타났다. \> \> *(사실검증 — 근거불충분/경미: 요약은 '정경 각속도 비율'이라고 표현하지만, 제시된 본문 근거 문장은 loading response phase 동안 knee extensor moment와 shank angular velocity의 양의 상관을 말한다. 표 3.7에는 between limb ratio of SAV와 between limb ratio of KMom가 제시되어 있어 비율 표현이 완전히 배제되지는 않지만, 인용한 문장만으로는 '각속도 비율'이라고 특정하기에는 근거가 부족하다.)*
	- 근거 원문: “The second article testing walking by Sigward et al. (2016) found a positive correlation between knee extensor moment (KMom) and shank angular velocity during the loading response phase.”

## 저자 결론

- 관성 센서는 ACLR 수술 후 동적 무릎 평가에서 동작 분석 시스템과 비교할 만한 정보를 제공할 수 있어 임상적 가치가 크다. *(근거: Page 6, ABSTRACT)*
	- 근거 원문: “Conclusion: Inertial sensors can provide comparable information to motion analysis systems during dynamic knee assessment after ACLR surgery. This has significant potential for clinical use, enhancing patients' functional assessment and the monitoring of the rehabilitation process.”
- 고가의 동작 분석 시스템과 힘판이 없는 환경에서 관성 센서는 무릎 기능을 분석하는 실용적인 대안이 될 수 있다. *(근거: Page 28, 4.3 Conclusion)*
	- 근거 원문: “Inertial sensor measurement units can serve as a valuable instrument in the clinical quantification of dynamickneeloadingasymmetries. Intheabsenceof high-endmotion analysis systems and force platforms, inertial sensors may be a practical alternative for analyzing knee function.”

## 연구의 한계

- 충분한 문헌 검색에도 불구하고 선정 기준을 만족한 연구는 5개에 불과했다. *(근거: Page 27, 4.2 Limitations)*
	- 근거 원문: “Firstof all, althoughweretrievedenough articles referring to the dynamic assessment of the knee after an ACLR surgery with the use of IMUs, only five met the eligible criteria.”
- 포함된 연구들이 수행한 작업과 결과 측정 항목 측면에서 균일하지 않았다. *(근거: Page 27, 4.2 Limitations)*
	- 근거 원문: “Secondly, the included studies were not uniform regarding the executed tasks and the outcome measures.”
- 보고된 진단 정확도가 언급된 단계 이외의 다른 재활 단계에도 적용되는지 또는 복수의 동적 테스트에서도 유지되는지 여부는 알 수 없다. *(근거: Page 27-28, 4.2 Limitations)*
	- 근거 원문: “Additionally, it is unknown if the reported diagnostic accuracy applies to other phases of rehabilitation than those mentioned or is maintained with multiple dynamic tests.”

## 생각해볼 내용

- 부하 반응 중 정경 각속도의 사지 간 차이를 IMU를 사용하여 감지하는 것은 임상의가 일상 진료에서 사지 간 결손을 평가하고 대처하는 데 도움이 될 수 있다. *(근거: Page 25, 4.1 Clinical implications and considerations)*
	- 근거 원문: “Detecting between limp differences in shank angular velocity during loading response with the help of IMUs may be helpful for the clinician to assess between limp deficits and address them in their everyday practice.”
- 힘판과 마커 기반 모션 캡처 시스템이 없는 상황에서, 대퇴부에 장착된 센서가 ACLR 수술 후 단일 하지 부하 작업 중 변화된 무릎 부하를 정량화하는 데 도움이 될 수 있다. *(근거: Page 26, 4.1 Clinical implications and considerations)*
	- 근거 원문: “These results imply that in the lack of force platforms and a marker-based motion capture system, inertial sensors and, more specifically, a sensor based on the thigh may be helpful in the clinic to quantify modified knee loading during a single limb loading task after an ACLR surgery.”

## 이 연구가 지적한 선행연구의 문제점

- 골드 스탠다드 광학 모션 캡처 시스템은 고가의 구매 및 유지 보수 비용이 들고, 전담 인력이 필요하며, 시간이 오래 걸리고 공간이 제한되는 등 상당한 단점이 있다. *(근거: Page 11, 1.1 Optical motion capture systems)*
	- 근거 원문: “However, although these are still the gold-standard systems, they also have considerable disadvantages; establishing a gait laboratory requires equipment purchases that average \$300,000 maintenance contracts for hardware and software (\$30–\$50,000) and full-time laboratory personnel. Theyarealso time-consuming and confinedto a small collectionspace (Simon et al., 2004).”
- 현재 ACLR 수술 후 선수의 스포츠 복귀(RTS) 여부를 평가할 수 있는 명확한 골드 스탠다드 기준이 존재하지 않는다. *(근거: Page 26, 4.1 Clinical implications and considerations)*
	- 근거 원문: “Presently, there is still no gold standard to clear athletes for return to sport (RTS) after ACLR surgery. The clinical decision is multifactorial and based on a combination of physical and psychological factors (Cheney et al., 2020).”

## 이 연구의 해결 방식과 기여

- 이 연구는 골드 스탠다드 동작 분석 시스템과 비교하여 ACLR 수술 후 무릎 기능 평가에 IMU를 사용하는 것에 초점을 맞춘 최초의 체계적 고찰이다. *(근거: Page 24, 4. DISCUSSION)*
	- 근거 원문: “To our knowledge, thisis thefirst review focusingon the use of IMUs in assessing knee function after ACLR surgery compared with gold standard motion analysis systems.” \> **[AS-IS]** - 관성 센서 측정 장치는 동적 무릎 부하 비대칭을 임상적으로 정량화하기 위한 실용적인 대안을 제시한다. *(근거: Page 28, 4.3 Conclusion)* \> \> **[TO-BE]** 관성 센서 측정 장치는 동적 무릎 부하 비대칭을 임상적으로 정량화하는 데 가치 있는 도구가 될 수 있으며, 고급 동작 분석 시스템과 힘판이 없는 환경에서는 무릎 기능 분석의 실용적 대안이 될 수 있다. \> \> *(사실검증 — 과장/경미: 원문은 관성 센서가 임상적 정량화에 가치 있는 도구가 될 수 있고, 고급 동작 분석 시스템과 힘판이 없을 때 실용적인 대안일 수 있다고 가능성 표현을 사용한다. 요약은 '대안을 제시한다'로 단정해 원문의 신중한 표현보다 강하다.)*
	- 근거 원문: “Inertial sensor measurement units can serve as a valuable instrument in the clinical quantification of dynamickneeloadingasymmetries. Intheabsenceof high-endmotion analysis systems and force platforms, inertial sensors may be a practical alternative for analyzing knee function.”

## 레퍼런스할 수 있는 내용

### 1. 골드 스탠다드 동작 캡처 시스템의 한계

- 원문 발췌: “However, although these are still the gold-standard systems, they also have considerable disadvantages; establishing a gait laboratory requires equipment purchases that average \$300,000 maintenance contracts for hardware and software (\$30–\$50,000) and full-time laboratory personnel. Theyarealso time-consuming and confinedto a small collectionspace (Simon et al., 2004).”
- 한국어 번역: 그러나 이들이 여전히 골드 스탠다드 시스템임에도 불구하고 상당한 단점이 있다. 보행 연구소를 설립하려면 평균 300,000달러에 달하는 장비 구입비, 하드웨어 및 소프트웨어 유지보수 계약비(30,000\~50,000달러), 그리고 전임 연구소 직원이 필요하다. 또한 시간 소모가 많고 좁은 측정 공간에 한정된다.
- 원문 위치: Page 11, 1.1 Optical motion capture systems
- 원문 내 인용표기: (Simon et al., 2004)
- 해당 선행문헌: 48. Simon, S. (2004). 'Quantification of human motion: gait analysis—benefits and limitations to its application to clinical problems'. Journal of Biomechanics, 37(12), pp.1869-1880.
- 주장 유형: background_citation
- 활용 맥락과 주의: 카메라 기반의 광학적 동작 분석 시스템의 높은 설치 비용, 유지보수 비용, 인력 요구사항 및 공간적 제약을 설명할 때 2차 인용으로 활용할 수 있다. Simon (2004)의 원문을 직접 확인하는 것이 권장된다.

### 2. 관성 센서를 활용한 ACLR 환자의 사지 간 비대칭 감지

- 원문 발췌: “In four studies, the extracted features from inertial sensors revealed significant inter- limb differences in ACLR patients.”
- 한국어 번역: 4개 연구에서 관성 센서로부터 추출된 특징들이 ACLR 환자들의 사지 간 유의미한 차이를 밝혀냈다.
- 원문 위치: Page 6, ABSTRACT
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: 이 논문의 자체 분석 결과(체계적 고찰 결과)로, 관성 센서가 ACLR 환자의 비대칭성을 감지할 수 있음을 지지하는 근거로 인용할 수 있다.


---

# [27] Towards Out-of-Lab Anterior Cruciate Ligament Injury Prevention and Rehabilitation Assessment: A Review of Portable Sensing Approaches

(저자: Tian Tan, Anthony A. Gatti, Bingfei Fan, Kevin G. Shea, Seth L. Sherman, Scott D. Uhlrich, Jennifer L. Hicks, Scott L. Delp, Peter B. Shull, Akshay S. Chaudhari | 연도: 2022 | 저널: medRxiv | DOI: https://doi.org/10.1101/2022.10.19.22281252)

Tan, T., Gatti, A. A., Fan, B., Shea, K. G., Sherman, S. L., Uhlrich, S. D., Hicks, J. L., Delp, S. L., Shull, P. B., & Chaudhari, A. S. (2022). Towards Out-of-Lab Anterior Cruciate Ligament Injury Prevention and Rehabilitation Assessment: A Review of Portable Sensing Approaches. medRxiv. https://doi.org/10.1101/2022.10.19.22281252

## 서지정보

- 저자: Tian Tan, Anthony A. Gatti, Bingfei Fan, Kevin G. Shea, Seth L. Sherman, Scott D. Uhlrich, Jennifer L. Hicks, Scott L. Delp, Peter B. Shull, Akshay S. Chaudhari
- 연도: 2022
- 저널: medRxiv
- DOI: 10.1101/2022.10.19.22281252
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/03_wearable_imu_and_portable_sensing/Towards Out-of-Lab Anterior Cruciate Ligament Injury Prevention and Rehabilitation Assessment - A Review of Portable Sensing Approaches.pdf
- 분석 provider: antigravity

## 연구 목적

- 본 리뷰 논문은 실험실 외 공간에서 전방십자인대(ACL) 부상 및 전방십자인대 재건술(ACLR)에 적용된 휴대용 센싱 연구를 요약하고, 향후 개발을 위한 새로운 연구 기회에 대한 관점을 제공하는 것을 목적으로 한다. *(근거: PAGE 2, Abstract)*
	- 근거 원문: “The purpose of this review is to summarize research on out-of-lab portable sensing applied to ACL and ACLR and offer our perspectives on new opportunities for future research and development.”

## 연구 설계와 대상

- 본 연구는 문헌 분석을 통해 전방십자인대 관련 실험실 외 평가를 수행한 49개의 독창적인 연구 논문을 선정하여 분석하였다. *(근거: PAGE 5, 2. Results)*
	- 근거 원문: “Our search yielded 1344 articles, of which 49 articles were included (Fig. 1), dating from 1990 to 2022.”
- 선정된 연구들에 참여한 피험자의 수는 최소 9명에서 최대 169명이었으며, 피험자 수의 중앙값은 24명이었다. *(근거: PAGE 11, 2.5. Experimental Design)*
	- 근거 원문: “The number of subjects recruited in the included studies ranged from 9 to 169, with the median being 24 (Fig. 4a).”

## 방법

- 2022년 3월 6일까지 발표된 논문을 Medline 및 Web of Science 데이터베이스에서 관련 키워드를 조합하여 검색하였다. *(근거: PAGE 17, 5.1. Literature SearchApproach)*
	- 근거 원문: “We searched articles published up to March 6, 2022 from the following databases: Medline (1950-) and Web of Science Core Collection (1950-).”
- 두 명의 저자가 독립적으로 검색된 논문들의 제목, 초록, 키워드를 검토하여 1차 선별 작업을 수행하였다. *(근거: PAGE 18, 5.2. Inclusion and Exclusion Criteria)*
	- 근거 원문: “Two authors (T.T. and A.A.G.) independently reviewed titles, abstracts, and keywords of all the retrieved articles.”

## 핵심 결과

- 단독으로 가장 널리 사용된 센서는 관성측정장치(IMU, 22%)였으며, 그 뒤를 깊이 카메라(16%), RGB 카메라(8%), 근전도(4%)가 이었다. *(근거: PAGE 5, 2. Results)*
	- 근거 원문: “IMUs were the most common sensor used in isolation (22%), followed by depth cameras (16%), red-green-blue (RGB) cameras (8%), and electromyography (EMG) (4%) (Fig. 2b).”
- 추정된 생체역학 파라미터는 운동학(kinematics)이 71%로 가장 높은 비중을 차지했고, 시공간적 변수(18%), 운동역학(kinetics, 12%), 근육 활성도(10%) 순이었다. *(근거: PAGE 5, 2. Results)*
	- 근거 원문: “Kinematic parameters were the dominant target (71%), followed by spatiotemporal parameters (18%), kinetics (12%), and muscle activation (10%) (Fig. 2c).”
- 데이터 분석 모델링 기법 중에는 직접적인 특징 추출(direct feature extraction) 방식이 37%로 가장 많았고, 물리 기반 모델링(24%)과 머신러닝 기법(22%)이 그 뒤를 이었다. *(근거: PAGE 5, 2. Results)*
	- 근거 원문: “Direct feature extraction (37%) was the most common analysis approach, followed by physics-based modeling (24%), and machine learning (22%) (Fig. 2d).”

## 저자 결론

- 본 연구의 분석 대상 연구들을 종합한 결과, 휴대용 센싱 기술은 전방십자인대 수술 후 환자의 재활 경과를 모니터링하고 부상 위험 요인을 경감하기 위한 훈련에 유용하게 사용될 수 있다. *(근거: PAGE 17, 4. Conclusion)*
	- 근거 원문: “Through these studies, we showed that portable sensing can be used to monitor patient progress through the rehabilitation process and train athletes to reduce injury risk factors.”
- **[AS-IS]** 휴대용 센싱 접근법의 기술적 발전은 전방십자인대 부상 위험 요인 식별, 고위험 동작 완화, 맞춤형 재활 프로그램 수립, 스포츠 복귀 판정 정량화에 널리 기여할 것이다.<br>**[TO-BE]** 이러한 발전이 성공적으로 이루어진다면, 휴대용 센싱 접근법은 전방십자인대 부상 위험 요인 추정, 고위험 동작 완화, 맞춤형 재활 패러다임 수립, 스포츠 복귀 준비도 정량화에 널리 활용될 수 있다.<br>*(사실검증 — 과장/경미: 원문은 'If successful'이라는 조건을 붙여 향후 가능성을 제시하지만, 요약은 조건부 표현을 생략해 기술 발전이 반드시 기여할 것처럼 단정적으로 표현했다.)* *(근거: PAGE 17, 4. Conclusion)*
	- 근거 원문: “If successful, these advances will enable widespread use of portable-sensing approaches to estimate ACL injury risk factors, mitigate high-risk movements, customize rehabilitation paradigms for improved long-term health outcomes, and quantify return-to-sport readiness.”

## 연구의 한계

- 전방십자인대 부상 위험 요인을 추정한 센싱 연구 중, 추정 결과를 피험자의 미래 실제 부상 발생 여부와 대조하여 전향적으로 임상 타당성을 검증한 연구는 없었다. *(근거: PAGE 11, 2.5. Experimental Design)*
	- 근거 원문: “None of the studies that estimated ACL injury risk factors prospectively evaluated their estimation results against subjects’ future injury occurrence.”
- 유사한 목적으로 수행된 연구들 간에 피험자 코호트 구성, 기준값(ground truth)의 출처, 타당도 평가지표가 상이하여 기술 간의 타당성이나 검사-재검사 신뢰도를 객관적으로 비교하기가 어렵다. *(근거: PAGE 16, 3.4. Dataset and Benchmarking)*
	- 근거 원문: “Although many included studies estimated the same biomechanical parameter during the same type of motion (Table 2), significant differences exist in their recruited cohorts of subjects, sources of ground truth, and metrics of validity. It is therefore difficult to impossible to truly compare the validity, sensitivity to parameter changes, or test-retest repeatability between these methods.”

## 생각해볼 내용

- 카메라 기반의 비착용형 센서는 신체에 부착하는 번거로움 없이 수동적인 데이터 수집이 가능하여, 다기관 대규모 데이터 수집 시 실용성과 효율성을 극대화할 수 있다. *(근거: PAGE 13, 3.1.1. Injury Risk Screening)*
	- 근거 원문: “Deployment of such a multi-center effort is particularly feasible for non-wearable sensors (e.g., RGB and depth cameras), which can passively collect data without the burden of donning and doffing the sensors.”
- **[AS-IS]** 휴대용 센서의 자동 추적 기능은 모바일 앱이나 가상 코칭 플랫폼에서 게이미피케이션(놀이화) 요소를 결합해 운동선수들의 프로그램 참여율과 흥미를 지속적으로 높이는 데 유용하다.<br>**[TO-BE]** 휴대용 센싱 기술의 훈련 및 진행 상황 자동 추적 기능은 게이미피케이션 같은 새로운 전략을 통해 운동선수들이 부상 예방 프로그램을 완료하도록 동기를 부여하고 참여를 유도할 가능성이 있다.<br>*(사실검증 — 근거불충분/경미: 원문은 자동 추적 기능이 게이미피케이션 같은 새로운 전략의 가능성을 열 수 있다고만 말한다. '모바일 앱이나 가상 코칭 플랫폼' 및 '지속적으로 높이는 데 유용하다'는 구체적 적용 매체와 효과 지속성은 제시된 원문 근거에서 확인되지 않는다.)* *(근거: PAGE 14, 3.1.2. Injury Prevention Training)*
	- 근거 원문: “The capabilities for automatic tracking of training and progress provided by portable sensing technologies could also open the door to new strategies, such as gamification, to motivate and engage athletes in completing injury prevention programs \[121, 122\].”

## 이 연구가 지적한 선행연구의 문제점

- 전통적인 광학식 동작 분석 시스템과 힘판 기반의 측정 기술은 정확도가 높아 표준(gold standard)으로 여겨지지만, 특수 실험실에 공간이 한정되어 일반인들의 접근성이 매우 떨어진다. *(근거: PAGE 3, 1. Introduction)*
	- 근거 원문: “Although these devices are considered the gold standard for measurement, they confine the assessment to specialized motion laboratories, making evaluation inaccessible to a majority of people.”
- 기존에 임상에서 널리 활용된 홉 테스트(hop test)는 관절의 기능과 부하 상태를 간접적으로만 반영하므로 하지가 실제로 가지는 생체역학적 비대칭성을 완전히 포착하지 못하고 숨길 우려가 있으며, 테스트 절차에 민감하게 영향을 받는다. *(근거: PAGE 14-15, 3.1.3. Return-to-Sport Decision Making)*
	- 근거 원문: “Furthermore, it was reported that since hop tests only indirectly assess knee function and loading, they may mask asymmetry in lower limb biomechanics \[90, 92-94\]. Hop tests are also sensitive to small alterations in the test procedures \[95\].”

## 이 연구의 해결 방식과 기여

- 본 논문은 정교한 분석 모델링 기술의 활용, 데이터 획득의 표준화, 대규모 공개 벤치마크 데이터셋 구축 등 향후 임상 현장 적용에 필수적인 최신 연구 개발 방향을 체계적으로 도출하였다. *(근거: PAGE 2, Abstract)*
	- 근거 원문: “By synthesizing these results, we describe important opportunities that exist for using sophisticated modeling techniques to enable more accurate assessment along with standardization of data collection and creation of large benchmark datasets.”
- 기존 연구들을 실제 임상 배포 가능성 수준에 따라 개발 및 프로토타입 단계(Stage I), 예비 임상 검증 단계(Stage II), 실생활 외부 임상 검증 단계(Stage III)의 3단계로 분류하여 향후 보완해야 할 격차를 명시하였다. *(근거: PAGE 11, 2.6. Readiness for Deployment)*
	- 근거 원문: “We categorized the included studies into three stages based on their readiness for deployment (Fig. 5).”

## 레퍼런스할 수 있는 내용

### 1. ACL 부상 후 골관절염 발생률

- 원문 발췌: “Concerningly, nearly half of these patients are under 20 years of age, and they suffer from not only over 20% reinjury rates \[115, 116\] but also 50-80% knee osteoarthritis rates within a decade of injury \[3, 4\].”
- 한국어 번역: 우려스럽게도, 이 환자들의 거의 절반은 20세 미만이며, 이들은 20% 이상의 재부상률\[115, 116\]을 겪을 뿐만 아니라 부상 후 10년 이내에 50-80%의 무릎 골관절염 발생률\[3, 4\]을 겪는다.
- 원문 위치: PAGE 2, 1. Introduction
- 원문 내 인용표기: \[3, 4\]
- 해당 선행문헌: \[3\] Nishimori, M. et al. Articular cartilage injury of the posterior lateral tibial plateau associated with acute anterior cruciate ligament injury. Knee Surgery, Sports Traumatology, Arthroscopy 16 (3), 270-274 (2008). \[4\] Muthuri, S., McWilliams, D., Doherty, M. & Zhang, W. History of knee injuries and knee osteoarthritis: a meta-analysis of observational studies. Osteoarthritis and cartilage 19 (11), 1286-1293 (2011).
- 주장 유형: background_citation
- 활용 맥락과 주의: 전방십자인대 부상 환자(특히 20세 미만 젊은 환자)의 장기적인 예후(높은 재부상률 및 골관절염 발생 위험)를 언급할 때 근거로 사용 가능하다.

### 2. 미국 내 연간 전방십자인대 부상 및 재건술 건수

- 원문 발췌: “Anterior cruciate ligament (ACL) injury is common in sports, with an estimated 400,000 people injuring their ACL in the United States each year \[1\], leading to over 129,000 ACL reconstruction (ACLR) surgeries \[2\].”
- 한국어 번역: 전방십자인대(ACL) 부상은 스포츠에서 흔하게 발생하며, 미국에서만 매년 약 400,000명이 ACL 부상을 입는 것으로 추정되고\[1\], 이로 인해 129,000건 이상의 ACL 재건(ACLR) 수술이 시행된다\[2\].
- 원문 위치: PAGE 2, 1. Introduction
- 원문 내 인용표기: \[1\], \[2\]
- 해당 선행문헌: \[1\] Murray, M. M. in History of acl treatment and current gold standard of care (eds Murray, M. M., Vavken, P. & Fleming, B.) The ACL Handbook: Knee Biology, Mechanics, and Treatment 19-28 (Springer New York, New York, NY, 2013). \[2\] Mall, N. A. et al. Incidence and trends of anterior cruciate ligament reconstruction in the united states. The American journal of sports medicine 42 (10), 2363-2370 (2014).
- 주장 유형: background_citation
- 활용 맥락과 주의: 미국 내 전방십자인대 부상 빈도와 수술 유병률을 설명하는 서론 작성 시 근거 자료로 인용할 수 있다.

### 3. 센싱 연구 피험자의 성별 편향성 분포

- 원문 발췌: “Twenty-five studies (51%) did not exhibit biases across the sex of included subjects, in that the percentages of females were between 34% and 66% (Fig. 4b).”
- 한국어 번역: 분석된 연구 중 25개(51%)는 여성 비율이 34%에서 66% 사이로 포함된 피험자의 성별에 따른 편향을 보이지 않았다(그림 4b).
- 원문 위치: PAGE 11, 2.5. Experimental Design
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: 전방십자인대 관련 휴대용 센싱 연구에서 피험자 성별 편향이 존재하지 않았던 비율을 제시하는 본 연구 자체 결과 근거로 사용 가능하다.


---

# [28] Wearable Devices for the Quantitative Assessment of Knee Joint Function After Anterior Cruciate Ligament Injury or Reconstruction: A Scoping Review

(저자: Oliwia Ptaszyk, Tarek Boutefnouchet, Gerard Cummins, Jin Min Kim, Ziyun Ding | 연도: 2025 | 저널: Sensors | DOI: https://doi.org/10.3390/s25185837)

Ptaszyk, O., Boutefnouchet, T., Cummins, G., Kim, J. M., & Ding, Z. (2025). Wearable Devices for the Quantitative Assessment of Knee Joint Function After Anterior Cruciate Ligament Injury or Reconstruction: A Scoping Review. Sensors, 25(18), 5837. https://doi.org/10.3390/s25185837

## 서지정보

- 저자: Oliwia Ptaszyk, Tarek Boutefnouchet, Gerard Cummins, Jin Min Kim, Ziyun Ding
- 연도: 2025
- 저널: Sensors
- DOI: https://doi.org/10.3390/s25185837
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/03_wearable_imu_and_portable_sensing/Wearable Devices for the Quantitative Assessment of Knee Joint Function After Anterior Cruciate Ligament Injury or Reconstruction - A Scoping Review.pdf
- 분석 provider: antigravity

## 연구 목적

- 본 범위 검토(scoping review)의 목적은 ACL 손상 또는 재건술 후 무릎 관절 결과를 정량화하는 데 있어 웨어러블 기기의 사용 현황을 매핑하고, 이들의 임상적 준비도와 방법론적 품질을 평가하는 것이다. *(근거: PAGE 1, Abstract)*
	- 근거 원문: “This scoping review aimed to map the use of wearable devices in quantifying knee outcomes following ACL injury or reconstruc- tion, and to evaluate their clinical readiness and methodological quality.”

## 연구 설계와 대상

- 선정 기준에 부합하는 연구는 웨어러블 기기를 사용하여 ACL 관련 무릎 결과를 평가하는 ACL/ACLR 환자군 또는 건강한 대조군 대상의 영어로 작성된 인간 대상 연구였다. *(근거: PAGE 1, Abstract)*
	- 근거 원문: “Eligible studies were human, English-language studies in ACL/ACLR populations or healthy cohorts assessing ACL-relevant knee outcomes with wearable devices.”
- 총 32개의 연구가 최종 선정 기준을 충족하여 분석에 포함되었다. *(근거: PAGE 6, Section 3)*
	- 근거 원문: “In total, 32 studies were included in the review (Figure 3).”

## 방법

- 본 검토는 Arksey와 O'Malley 프레임워크 및 PRISMA-ScR(범위 검토를 위한 체계적 문헌고찰 및 메타분석 보고 지침 확장판) 가이드라인을 따랐다. *(근거: PAGE 4, Section 2)*
	- 근거 원문: “This review followed the Arksey and O’Malley framework \[30\] and the PRISMA- ScR (Preferred Reporting Items for Systematic Reviews and Meta-Analyses extension for Scoping Reviews) guidelines \[31\].”
- 2025년 8월 27일까지 MEDLINE(Ovid), Embase(Ovid), APA PsycInfo(Ovid), PubMed, Scopus 데이터베이스를 대상으로 문헌 검색을 수행하였다. *(근거: PAGE 4, Section 2)*
	- 근거 원문: “MEDLINE (Ovid), Em- base (Ovid), APA PsycInfo (Ovid), PubMed, and Scopus were searched up to 27 August 2025.”

## 핵심 결과

- 관절 운동학 측정에는 관성측정장치(IMU)가 가장 흔히 사용되었고, 단독 가속도계는 피벗 시프트(pivot-shift) 특징을 정량화했으며, 힘 감지 인솔은 양측 부하를 측정했다. *(근거: PAGE 1, Abstract)*
	- 근거 원문: “Inertial measurement units (IMUs) were used most often for kinematics. Standalone accelerometers quantified pivot-shift features, while force-sensing insoles captured bilateral loading.”
- 전자기 트래커와 전기각도계는 더 높은 정밀도의 비교 기준으로 기능했으나 임상 워크플로우 상의 한계가 있었다. *(근거: PAGE 1, Abstract)*
	- 근거 원문: “Electromagnetic trackers and electrogoniometers served as higher-precision comparators but were workflow-limited.”
- 분석 대상 연구들의 기술성숙도(TRL) 대역은 주로 3\~6에 집중되어 있었으며, 실제 임상에 완전히 통합된 사례는 없었다. *(근거: PAGE 1, Abstract)*
	- 근거 원문: “TRL bands clustered at 3–6, and none reached clinical integration.”

## 저자 결론

- 저자들은 웨어러블 기기를 일상적 치료에 도입하기 위해 과제 맞춤형 샘플링, 투명한 보정 절차, 기준 타당도 검증, 환자 보고 결과 지표(PROMs)와의 병용, 다기관 워크플로우 임상시험을 제안한다. *(근거: PAGE 1, Abstract)*
	- 근거 원문: “We propose task-matched sampling, transparent calibration, criterion validation, pairing with patient-reported outcome measures (PROMs), and multi-site workflow trials to progress towards routine care.”
- 웨어러블 기기의 잠재력에도 불구하고 임상 적용을 제한하는 요인은 기준 시스템 대비 불일치한 검증, 부실하게 보고된 보정 및 테스트 프로토콜, 표준화된 결과 측정 지표의 부재 등이다. *(근거: PAGE 27, Section 5)*
	- 근거 원문: “Key barriers include inconsistent validation against gold-standard systems, poorly reported calibration and testing protocols, and a lack of standardised outcome measures.”

## 연구의 한계

- 본 검토 연구는 저자 팀 내 정형외과 의사의 임상적 전문 지식의 혜택을 받았으나, 환자, 물리치료사, 기술 개발자 등 더 넓은 그룹과의 공식적인 이해관계자 자문 과정을 거치지 않았다. *(근거: PAGE 27, Section 4.7)*
	- 근거 원문: “While this review benefited from clinical expertise within the author team, including an orthopaedic surgeon, it did not involve a formal stakeholder consultation process with broader groups, such as patients, physiotherapists, or technology developers.”
- 본 연구의 기술성숙도(TRL) 매핑은 장치/시스템 수준에서 개작된 대역과 정성적 판단을 사용하였으므로 결정적인 순위라기보다는 지표 수준의 성숙도로 해석되어야 한다. *(근거: PAGE 27, Section 4.7)*
	- 근거 원문: “Finally, our TRL mapping used adapted bands and qualitative judgements at the device/system level. These assignments depend on what each study reported and should be interpreted as indicative maturity rather than definitive rankings.”

## 생각해볼 내용

- 분석 대상 연구들의 기술성숙도(TRL)가 대부분 3-6 단계에 머물러 있어 실제 임상에 통합된 사례가 없으며, 이는 웨어러블 기기의 실질적인 임상 적용을 위해 극복해야 할 주요 과제이다. *(근거: PAGE 1, Abstract)*
	- 근거 원문: “TRL bands clustered at 3–6, and none reached clinical integration.”
- **[AS-IS]** 성공적인 임상 전환을 위해서는 기술 연구(TRL 3-6) 수준을 넘어 임상 워크플로우에 직접 통합된 다기관 시험(TRL 7 이상)으로 연구가 확장되어야 한다. *(근거: PAGE 27, Section 5)*<br>**[TO-BE]** 성공적인 임상 전환을 위해서는 현재의 중간 수준 TRL(3-6)을 넘어 더 큰 규모의 임상 연구, 임상 워크플로우 통합, 규제 경로 마련이 필요하다.<br>*(사실검증 — 과장/경미: 제시된 근거 문장은 더 큰 임상 연구, 워크플로우 통합, 규제 경로의 필요성을 말하지만, 요약처럼 반드시 TRL 7 이상 또는 임상 워크플로우에 직접 통합된 다기관 시험으로 확장되어야 한다고 구체적으로 단정하지는 않는다. TRL 7의 정의는 표 1에서 workflow trials/multisite pilots로 제시되지만, 해당 요약의 근거 인용문만으로는 표현이 다소 강하다.)*
	- 근거 원문: “Most applications sit at intermediate TRLs (3–6), reflecting pre-integration maturity and the need for larger clinical studies, workflow integration, and regulatory pathways before clinical adoption.”

## 이 연구가 지적한 선행연구의 문제점

- 웨어러블 기기를 탐색한 여러 연구가 존재하지만, 많은 연구가 기기 사양, 연구 프로토콜, 검증 지표에 대한 상세한 보고가 부족했으며, 구현을 위한 기술 준비도에 공식적인 프레임워크를 적용한 연구가 거의 없었다. *(근거: PAGE 3, Section 1)*
	- 근거 원문: “Although several studies have explored wearable devices in this context, many lacked detailed reporting of device specifications, study protocols, and validation metrics, and few applied formal frameworks to technology readiness for implementation.”
- 최근의 리뷰들에 따르면 이질적인 센서 설정과 맞춤형 분석 파이프라인으로 인해 연구 간 비교가 어렵고, 대부분의 ACL 웨어러블 연구는 제한된 임상 도입 수준의 개념 검증 단계에 머물러 있다. *(근거: PAGE 4, Section 1)*
	- 근거 원문: “Recent reviews describe heterogeneous sensor setups and custom analysis pipelines that hinder cross-study comparison. Most ACL wearable studies remain proof-of-concept with limited clinical adoption \[27,28\].”

## 이 연구의 해결 방식과 기여

- 본 범위 검토는 IMU, 가속도계, 전자기 추적, 인솔, 전기각도계 전반의 32개 연구를 종합하고 직접 측정된 결과와 모델 추정 대리 지표를 구분함으로써 기존 연구의 공백을 해결하고자 했다. *(근거: PAGE 4, Section 1)*
	- 근거 원문: “This scoping review addresses those gaps by synthesising 32 studies across IMUs, accelerometers, electromagnetic tracking, insoles, and electrogoniometers, and by distinguishing directly measured outcomes from model-estimated surrogates.”
- 개작된 기술성숙도(TRL) 매핑을 적용하고 연구별 타당성 보고 수준(자체 검증, 선행문헌 인용, 미보고)을 평가하여 임상 전환적 성숙도를 체계적으로 묘사했다. *(근거: PAGE 4, Section 1)*
	- 근거 원문: “It applies an adapted technology readiness level (TRL) mapping and evaluates validity reporting (in-study criterion, prior-only, or not reported) to describe translational maturity.”
- 저자들의 지식 범위 내에서 본 연구는 ACL 분야에서 TRL 매핑, 연구 수준의 타당도 보고 평가, 그리고 이기종 기기 간 프로토콜 패턴 비교를 결합한 최초의 검토 논문이다. *(근거: PAGE 4, Section 1)*
	- 근거 원문: “To the authors’ knowledge, this is the first ACL-focused review to combine TRL mapping, study-level validation reporting, and cross-device comparison of protocol patterns.”

## 레퍼런스할 수 있는 내용

### 1. ACL의 해부학적 및 역학적 안정성 역할

- 원문 발췌: “The anterior cruciate ligament (ACL) is essential for maintaining knee stability, pri- marily by restricting anterior tibial translation and secondarily by rotational control \[1\].”
- 한국어 번역: 전방십자인대(ACL)는 일차적으로 경골의 전방 전위를 제한하고 이차적으로 회전을 제어함으로써 무릎 안정성을 유지하는 데 필수적이다.
- 원문 위치: PAGE 1, Section 1
- 원문 내 인용표기: \[1\]
- 해당 선행문헌: 1. Giuliani, J.R.; Kilcoyne, K.G.; Rue, J.P.H. Anterior Cruciate Ligament Anatomy: A Review of the Anteromedial and Posterolateral Bundles. J. Knee Surg. 2009, 22, 148–154. \[CrossRef\]
- 주장 유형: background_citation
- 활용 맥락과 주의: 전방십자인대의 일차적 및 이차적 역학 기능에 대한 기초 설명으로 인용하기 적합하다.

### 2. Lachman 검사의 ACL 파열 진단 정확도

- 원문 발췌: “According to a meta-analysis by Benjaminse et al. \[5\], when performed without anaesthesia, the Lachman test demonstrated the highest diagnostic accuracy for detecting ACL ruptures, with a pooled sensitivity of 85% and specificity of 94%.”
- 한국어 번역: Benjaminse 등의 메타분석\[5\]에 따르면, 마취 없이 수행할 때 Lachman 검사는 ACL 파열 감지에 있어 가장 높은 진단 정확도를 나타냈으며, 이때 취합된 민감도는 85%, 특이도는 94%였다.
- 원문 위치: PAGE 2, Section 1
- 원문 내 인용표기: \[5\]
- 해당 선행문헌: 5. Benjaminse, A.; Gokeler, A.; Van Der Schans, C.P. Clinical Diagnosis of an Anterior Cruciate Ligament Rupture: A Meta-Analysis. J. Orthop. Sports Phys. Ther. 2006, 36, 267–288. \[CrossRef\]
- 주장 유형: background_citation
- 활용 맥락과 주의: 비마취 하에 진행되는 Lachman 검사의 민감도와 특이도에 대한 정량적 진단 근거로 활용할 수 있다.

### 3. ACLR 이후 운동 복귀 비율

- 원문 발췌: “Only 65% of patients who undergo ACLR return to pre-injury levels of sport participation \[22\].”
- 한국어 번역: ACLR을 받은 환자 중 65%만이 부상 전 수준의 스포츠 참여로 복귀한다.
- 원문 위치: PAGE 3, Section 1
- 원문 내 인용표기: \[22\]
- 해당 선행문헌: 22. Ardern, C.L.; Taylor, N.F.; Feller, J.A.; Webster, K.E. Fifty-Five per Cent Return to Competitive Sport Following Anterior Cruciate Ligament Reconstruction Surgery: An Updated Systematic Review and Meta-Analysis Including Aspects of Physical Functioning and Contextual Factors. Br. J. Sports Med. 2014, 48, 1543–1552. \[CrossRef\] \[PubMed\]
- 주장 유형: background_citation
- 활용 맥락과 주의: ACLR 수술 후 스포츠 활동 복귀율이 높지 않음을 설명할 때 인용하기 적절하다.

### 4. ACL 연구용 웨어러블 기기의 임상 전환 미흡

- 원문 발췌: “TRL bands clustered at 3–6, and none reached clinical integration.”
- 한국어 번역: 기술성숙도(TRL) 대역이 3-6에 밀집되어 있으며, 임상에 완전히 통합된 웨어러블 기기는 없었다.
- 원문 위치: PAGE 1, Abstract
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: 현재 무릎 관절 기능 평가용 웨어러블 기술이 대부분 상용화나 일상 임상 치료에 활용되지 않고 프로토타입이나 연구 단계에 있음을 보여주는 자체 결과로 인용할 수 있다.

### 5. 웨어러블 중 IMU의 실용성과 장점

- 원문 발췌: “IMUs are the most commonly used wearable devices in ACL research and are currently the most practical option for multi-planar kinematics in both laboratory and field settings, with sagittal angles typically being the most accurate.”
- 한국어 번역: IMU는 ACL 연구에서 가장 흔하게 사용되는 웨어러블 장치이며, 현재 실험실과 현장 환경 모두에서 다평면 운동학을 위한 가장 실용적인 옵션이고, 대개 시상면 각도가 가장 정확하다.
- 원문 위치: PAGE 27, Section 5
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: 임상 또는 현장 환경에서 관절 운동학 평가용 웨어러블 기기를 선택할 때 IMU의 가장 큰 장점과 특징에 대해 인용할 수 있다.

### 6. ACLR 후 2차 슬관절 부상 위험

- 원문 발췌: “The risk of sustaining a secondary knee injury, in the contralateral knee or the reconstructed ACL, has been estimated at 29% following primary ACLR \[24\].”
- 한국어 번역: 1차 ACLR 이후 반대측 무릎이나 재건된 ACL에서 2차 무릎 부상을 입을 위험은 29%로 추정되었다.
- 원문 위치: PAGE 3, Section 1
- 원문 내 인용표기: \[24\]
- 해당 선행문헌: 24. Paterno, M.V.; Rauh, M.J.; Schmitt, L.C.; Ford, K.R.; Hewett, T.E. Incidence of Second ACL Injuries 2 Years after Primary ACL Reconstruction and Return to Sport. Am. J. Sports Med. 2014, 42, 1567–1573. \[CrossRef\]
- 주장 유형: background_citation
- 활용 맥락과 주의: 1차 전방십자인대 재건술 후 반대측 혹은 재건 부위의 2차 부상 재발 확률이 높음을 입증하기 위해 인용하기에 적절하다.


---

