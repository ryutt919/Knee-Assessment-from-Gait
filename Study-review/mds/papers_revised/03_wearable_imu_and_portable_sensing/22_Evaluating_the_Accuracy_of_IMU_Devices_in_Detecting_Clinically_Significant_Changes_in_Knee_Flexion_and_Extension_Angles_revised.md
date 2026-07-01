# Evaluating the Accuracy of IMU Devices in Detecting Clinically Significant Changes in Knee Flexion and Extension Angles

Ligeti, A. G. (확인 불가). Evaluating the Accuracy of IMU Devices in Detecting Clinically Significant Changes in Knee Flexion and Extension Angles (Doctoral dissertation, University of Strathclyde).

## 서지정보

- 저자: Alexandra Grace Ligeti
- 연도: 확인 불가
- 저널: University of Strathclyde
- DOI: 확인 불가
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/03_wearable_imu_and_portable_sensing/Evaluating the Accuracy of IMU Devices in Detecting Clinically Significant Changes in Knee Flexion and Extension Angles.pdf
- 분석 provider: antigravity

## 연구 목적

- 임상적으로 유의미한 임계값 범주 내에서 무릎 굴곡 각도를 측정할 때 상용화된 Stryker의 MotionSense™ 기술과 Seel 알고리즘을 적용한 유선 연구용 IMU 기기 두 가지의 정확도를 비교 평가하는 것이다. _(근거: PAGE 9, Abstract)_
  - 근거 원문: “This study aimed to evaluate the accuracy of two different wearable IMU devices (a Stryker (USA) commercially available technology, MotionSense™ and a wired IMU research device implementing the Seel Algorithm (Seel, Raisch and Schauer, 2014), in measuring knee flexion angles within clinically significant thresholds.”

## 연구 설계와 대상

- 다양한 연령층으로 구성된 건강한 성인 34명(20-36세의 젊은층 20명, 60-84세의 고령층 14명)과 수술 전 및 수술 후(수술 후 1주 및 6주)의 무릎 인공관절 치환술(TKA) 환자군 10명(53-71세)을 대상으로 다양한 일상생활 활동(ADLs) 전반에 걸쳐 측정을 수행하였다. _(근거: PAGE 9, Abstract)_
  - 근거 원문: “Measurements were evaluated across a diverse healthy adult population of varying ages (20 healthy younger participants, ages ranging between 20 - 36 years old and 14 healthy older participants, ages ranging between 60 - 84 years old) and within a TKA clinical population (10 TKA participants, ages ranging between 53 - 71 years old) both preoperatively and postoperatively (1 week postoperatively and at 6 weeks postoperatively), across a broad range of activities of daily living (ADL’s).”

## 방법

- 상용화된 MotionSense™ 기술은 Madgwick 필터를 구현한 독점 소프트웨어 모바일 앱을 이용해 시상면 무릎 각도를 측정하며, 유선 연구용 IMU 기기는 Seel 알고리즘을 사용하여 계산한다. 두 기기의 측정값은 하체에 16개의 재귀반사 마커를 부착한 Plug-In Gait(PIG) 모델 규격의 광학식 모션 캡처 시스템인 Vicon의 측정값과 대조되었다. _(근거: PAGE 9, Abstract)_
  - 근거 원문: “The commercially available MotionSense™ technology determines sagittal plane knee angle using a mobile-based app with proprietary software that implements a Madgwick filter (Madgwick, 2010), while the wired research IMU device calculates sagittal plane knee angle using the Seel algorithm (Seel, Raisch and Schauer, 2014). Both technologies’ measurements were compared against the gold standard opto-electronic motion capture system, Vicon, which tracked 16 retro-reflective markers that were attached to the lower body as per the PlugInGaitTM (PIG) model.”
- 두 IMU 기술 모두 무릎 굴곡의 영점이 마커 부착 위치에 영향을 받기 때문에, 각 데이터 세트에서 평균 무릎 굴곡 값을 먼저 차감한 후에 각 운동 주기 창 내에서 두 기술 간의 평균 제곱근 오차(RMSE)를 도출하였다. _(근거: PAGE 10, Abstract)_
  - 근거 원문: “For both IMU technologies the zero point for knee flexion depends on marker placement, therefore, the mean knee flexion was subtracted from each data set before calculating a root mean square error (RMSE) between the technologies, determined in each movement cycle window.”
- > **[AS-IS]** MotionSense™ 데이터의 분석은 MATLAB의 interp1 함수를 통해 100Hz로 업샘플링을 거쳐 교차상관 함수 xcorr을 적용함으로써 최대 굴곡 지점을 기준으로 각 주기 데이터를 동기화했다. _(근거: PAGE 10, Abstract)_
>
> **[TO-BE]** MotionSense™ 데이터 분석에서는 MATLAB interp1 함수로 100Hz 업샘플링을 수행한 뒤, peak flexion에서 peak flexion까지 식별한 movement cycle window를 xcorr 기반 교차상관으로 시간 동기화하였다.
>
> _(사실검증 — 과장/경미: 원문은 MotionSense™ 분석에서 peak flexion-to-peak flexion으로 식별된 movement cycle windows를 cross-correlation으로 시간 동기화했다고 설명한다. 요약의 ‘각 주기 데이터’는 모든 주기 데이터 전반을 포괄하는 표현처럼 읽혀 원문의 한정된 분석 창 표현보다 약간 넓다.)_
  - 근거 원문: “Following up-sampling to 100Hz using the MATLAB (MathWorks, 2024) interp1 function, cross-correlation was used to time synchronise the movement cycle windows identified from peak flexion to peak flexion using the xcorr MATLAB (MathWorks, 2024) function for each technology.”

## 핵심 결과

- 건강한 집단과 임상 집단 모두에서 그리고 더 큰 가동범위(ROM)와 빠른 관절 속도를 포함한 모든 동작 시 두 기기 모두 5° 미만의 RMSE를 보여주었으며, 기기별 오차는 MotionSense™가 0.86° - 4.70°, 유선 IMU 기기가 2.92° - 4.78°의 범위를 보였다. _(근거: PAGE 10, Abstract)_
  - 근거 원문: “Results presented RMSE of less than 5° across both devices, across both healthy and clinical populations and across all activities, including those involving larger ROM and higher joint velocities. RMSE values ranged between 0.86° - 4.70° for the MotionSense™ device, while RMSE values ranged between 2.92° - 4.78° for the wired IMU device.”
- 평가된 각 센서 기술에 있어 건강한 피험자 및 환자군 등 대상 그룹 간 정확도의 통계적으로 유의미한 차이는 인정되지 않았다 (p > 0.05). _(근거: PAGE 10, Abstract)_
  - 근거 원문: “No statistically significant differences between the population groups for each technology was evidenced (p > 0.05).”
- 더 큰 각도의 무릎 굴곡이 수반되는 활동에서 측정 시스템 간에 더 큰 편차가 관찰되었으며, 젊고 건강한 피험자의 굴곡/신전 활동(ROM 116.5°) 시 RMSE는 3.65°였으나 무릎 치환 수술 후 1주 차 환자의 보행(ROM 31.6°) 시 RMSE는 1.48°에 머물렀다. _(근거: PAGE 10-11, Abstract)_
  - 근거 원문: “Notably, greater discrepancies between the measurement systems were observed during activities involving larger degrees of flexion, for example during the flexion/extension activity performed by the younger healthy population a ROM of 116.5° and RMSE of 3.65° was reported between MotionSense™ and Vicon opto-electronic motion capture system, whereas a RMSE of 1.48° and a ROM of 31.6° was reported for the 1 week postoperative session for the walking activity.”
- 보행 활동에서 디딤기(stance phase)보다 빠른 움직임이 일어나는 흔듦기(swing phase) 구간 동안 두 측정 시스템 간에 상대적으로 더 큰 불일치가 확인되었다. _(근거: PAGE 11, Abstract)_
  - 근거 원문: “Furthermore larger differences were also evidenced during periods associated with faster motion (swing phase displayed larger differences compared to the stance phase for the walking activity).”
- 분석 대상이 된 웨어러블 IMU 기기들은 준수한 상관 계수를 보이며 모든 피험자 그룹의 무릎 굴곡 운동 양상을 정밀하게 추적할 수 있음을 검증하였다. _(근거: PAGE 11, Abstract)_
  - 근거 원문: “The wearable IMU technologies revealed strong coefficients of correlation and were able to accurately track knee flexion patterns across all population groups.”

## 저자 결론

- 웨어러블 IMU 장치는 시상면 무릎 관절 각도를 정밀하게 측정할 수 있어 임상 환경으로의 실효성 있는 도입을 뒷받침하며, 원내 정기 대면 평가를 대체하여 환자의 기능 회복 추이를 원격으로 연속 모니터링할 수 있는 대안을 제시한다. _(근거: PAGE 11, Abstract)_
  - 근거 원문: “This study concludes that wearable IMU devices can accurately measure sagittal knee angle supporting their integration into clinical settings. Their ability to provide accurate, objective data validates their use as a practical alternative to traditional in-clinic assessments, particularly in enabling remote and continuous tracking of patient progress. As such, IMUs may represent a valuable asset in modern rehabilitation strategies, facilitating more efficient, patient-centred care.”
- TKA 환자 코호트의 결과는 환자 개개인의 회복 양상과 수술 예후가 고도로 개별화되어 있음을 시사하며, 따라서 개인 맞춤형 재활 프로그램의 시행과 이를 뒷받침할 수 있는 혁신적인 정밀 측정 기술들의 통합이 필요함을 역설한다. _(근거: PAGE 11, Abstract)_
  - 근거 원문: “The findings from the TKA cohort underscore the highly patient-specific nature of recovery and postoperative outcomes, further emphasising the need for personalised rehabilitation approaches and the requirement for innovative technologies to deliver this level of personalised care.”

## 연구의 한계

- 스폰서인 Stryker사와의 계약 조항에 따라 Stryker의 MotionSense™ 장치와 연구에 쓰인 다른 유선 IMU 기기 간의 데이터를 활용한 직접적인 맞비교 분석이 금지되었으며, 이에 따라 두 시스템에 대한 데이터 가공 및 분석 처리를 독자적인 상이한 방법론을 이용해 별개로 리포트해야 했다. _(근거: PAGE 7, Disclosures and Collaboration)_
  - 근거 원문: “In addition to the collaboration with Philippe Martin, the terms of the contractual agreement with Stryker prohibited direct comparisons between Stryker's technology and other IMU-based systems. As a result, separate analyses were conducted for each IMU technology (MotionSense™ and the wired research IMU device). Differences in analysis methodologies and reporting are therefore intentional and reflect adherence to the contractual requirement to avoid direct comparisons between these technologies.”
- 본 논문의 임상 평가 대상인 무릎 인공관절 치환수술 환자군(TKA cohort)의 표본 수(10명)가 상대적으로 작다는 점이다. _(근거: PAGE 42, 1.1 Introduction)_
  - 근거 원문: “Though this study has a smaller clinical population, evaluations within this clinical group have been carried out across three separate data collection sessions providing a clearer indication of the performance of such devices both preoperatively and postoperatively.”

## 생각해볼 내용

- 임상 및 일반 건강관리 환경에서 웨어러블 IMU 기기를 도입할 시 회복 기간 동안 유효한 환자의 원격 비대면 모니터링 체계를 구축할 수 있고 홈 재활 프로토콜에 대한 순응도를 효과적으로 유도할 수 있는 상당한 임상적 효용을 가진다. _(근거: PAGE 11, Abstract)_
  - 근거 원문: “The use of wearable IMUs within clinical and healthcare settings offers substantial benefits within the recovery period, including remote monitoring capabilities and enhanced compliance with rehabilitation protocols.”
- 수술 전, 수술 중 및 수술 후 등 다차원적이고 장기적인 시점에서 객관적인 기능적 운동 데이터와 주관적인 환자 설문(PROMs) 데이터를 동시에 수집 및 연계하여 분석함으로써, 수술 후 최적의 기능 회복을 보장하는 인자들에 관한 더 종합적인 분석과 이해를 이룰 수 있다. _(근거: PAGE 45, 1.2 Clinical Problem)_
  - 근거 원문: “Moreover, by collecting both objective and subjective data at various timepoints: preoperative, intraoperative, and postoperative, a broader and more detailed understanding can be gained from the different factors that contribute to more favourable postoperative outcomes.”

## 이 연구가 지적한 선행연구의 문제점

- TKA 수술 직후 회복을 측정하기 위해 웨어러블 센서의 측정 유효성을 검증한 문헌은 제한적이며, 특히 관절 속도, 가해지는 충격 및 관절 가동범위(ROM)의 스펙트럼이 폭넓은 다채로운 기능성 동작들을 다루거나 고령 임상 집단과 연령대를 매칭시킬 수 있는 고령의 건강 대조군을 포함해 분석한 연구가 없었다. _(근거: PAGE 39-40, 1.1 Introduction)_
  - 근거 원문: “There is limited literature establishing the validity of wearable sensors to assess knee function shortly following TKA. Particularly literature that focusses on evaluating the accuracy of such devices over many different types of functional activities, that vary in speed, impact and across a broad ROM, that incorporate a relatively large healthy control group of both younger and older participants which presents an opportunity to age-match to a TKA clinical population.”
- 기존 선행 유효성 검사 문헌들은 표본 크기가 극히 협소하거나 관찰 데이터를 단일 시점에서만 취득하였고, 혹은 단순 보행이나 평이한 무릎 굴곡/신전 활동 등 제한된 몇 가지 움직임에 대해서만 유효성 평가를 수행하는 데 그쳤다. _(근거: PAGE 40, 1.1 Introduction)_
  - 근거 원문: “Of the available literature, only a handful (Antunes et al., 2021; Chen et al., 2022; Cornish et al., 2024; Fain et al., 2024; Hafer et al., 2020; Parrington et al., 2021; Wang et al., 2025; Versteyhe et al., 2020) evaluate the accuracy of such devices within a clinical population. However, these studies generally include a restricted population pool, record data at a single time point or only include a simple flexion/extension movement or walking.”
- 보통 이전의 실험들은 3-12명의 매우 한정된 수의 젊고 건강한 피험자들을 모집하여 각자 다른 광학 분석 장치와 마커 모델 표준하에서 기기를 검증함으로써 다양한 특이 보행을 보이는 임상 환자군에 적용하기에 일반화 가능성이 현저히 떨어졌다. _(근거: PAGE 40, 1.1 Introduction)_
  - 근거 원문: “Typically, investigations have recruited younger healthy cohorts with a maximum 3 - 12 individuals, all assessing different IMU technologies and algorithms against different 3D motion capture systems and models (Poitras et al., 2019).”
- 대부분의 선행 문헌들이 건강한 젊은 성인군이나 고정된 특정 단일 회복 시점의 환자만을 고집하였고 고령층이나 조기 수술 후 급성기의 실제 회복 상태를 제대로 대표하지 못하였다. _(근거: PAGE 41, 1.1 Introduction)_
  - 근거 원문: “Many studies focus exclusively on either healthy younger adults or patients at a single stage of recovery (Antunes et al., 2021; Cornish et al., 2024; Fain et al., 2024; Parrington et al., 2021; Versteyhe et al., 2020), often omitting older adults or those in the early postoperative period.”

## 이 연구의 해결 방식과 기여

- 본 연구는 임상 환자군의 나이대와 부합하는 20세부터 84세에 이르는 넓은 나이 범위의 건강 성인 대조군 34명을 확보하고 연령 증가에 따르는 보행 운동학적 변동성을 평가에 반영하였다. _(근거: PAGE 42, 1.1 Introduction)_
  - 근거 원문: “By including a larger healthy cohort of 34 individuals across a wide age range (20– 84 years old), which enables similar age group comparisons to the TKA population, enhancing the clinical relevance of the findings, however, also taking into consideration the natural variations within gait kinematics of healthy individuals as they age.”
- 수술 전 상태의 환자 및 수술 후 서로 다른 단계에 위치한 무릎 치환 수술 환자군을 종단적으로 모두 포괄함으로써 시간에 따른 회복 국면별 IMU 정확도 성능을 조망할 수 있는 구조를 취했다. _(근거: PAGE 42, 1.1 Introduction)_
  - 근거 원문: “Furthermore, the inclusion of both preoperative and postoperative TKA patients enables the assessment of IMU performance across different stages of the recovery process.”
- MATLAB을 이용하여 독자적인 무릎 관절 굴곡 연산 알고리즘을 검증함으로써 특정 하드웨어 제조사에 결속되지 않는 유연하고 비용 효율이 높으며 실용적인 대안 모니터링 경로를 개척하였다. _(근거: PAGE 42, 1.1 Introduction)_
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

- 원문 발췌: “No statistically significant differences between the population groups for each technology was evidenced (p > 0.05).”
- 한국어 번역: 각 기술에 대해 인구 집단 그룹 간에 통계적으로 유의미한 차이는 입증되지 않았다 (p > 0.05).
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
