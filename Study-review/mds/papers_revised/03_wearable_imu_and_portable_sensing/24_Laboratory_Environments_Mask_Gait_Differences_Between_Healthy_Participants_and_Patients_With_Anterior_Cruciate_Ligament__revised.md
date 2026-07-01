# Laboratory Environments Mask Gait Differences Between Healthy Participants and Patients With Anterior Cruciate Ligament Reconstruction

Parola, L. R., Phan, V., & Halilaj, E. (2026). Laboratory Environments Mask Gait Differences Between Healthy Participants and Patients With Anterior Cruciate Ligament Reconstruction. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 34, 767-775. https://doi.org/10.1109/TNSRE.2025.3645799

## 서지정보

- 저자: Lauren R. Parola, Vu Phan, Eni Halilaj
- 연도: 2026
- 저널: IEEE Transactions on Neural Systems and Rehabilitation Engineering
- DOI: 10.1109/TNSRE.2025.3645799
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/03_wearable_imu_and_portable_sensing/Laboratory Environments Mask Gait Differences Between Healthy Participants and Patients With Anterior Cruciate Ligament Reconstruction.pdf
- 분석 provider: antigravity

## 연구 목적

- 본 연구는 실험실 환경이 수술 후 보행에 대해 내리는 결론을 흐리게 하는지 조사하고, 자연 환경 바이오메카닉스로의 이동을 지지하는 초기 근거를 제공하고자 했다. _(근거: PAGE 2, I. INTRODUCTION)_
  - 근거 원문: “Accordingly, the aim of this study was to investigate whether laboratory environments cloud the conclusions we draw about post-surgical gait, providing initial support for the move toward natural-environment biomechanics.”
- 저자들은 ACLR 환자가 실험실에 비해 일상생활에서 더 긴 보행 주기 시간, 더 큰 이중 지지 의존도, 그리고 더 뚜렷한 절뚝거림을 보일 것이라고 가설을 세웠다. _(근거: PAGE 2, I. INTRODUCTION)_
  - 근거 원문: “We hypothesized that patients after ACLR walk with a longer gait-cycle time, greater reliance on double support, and a more pronounced limp in daily life compared to the laboratory.”
- 저자들은 또한 ACLR 환자와 건강한 참가자 사이의 차이가 실험실보다 일상생활에서 더 클 것이라고 가설을 세웠다. _(근거: PAGE 2, I. INTRODUCTION)_
  - 근거 원문: “Additionally, we hypothesized that differences between patients after ACLR and healthy participants would be higher in daily life than in the laboratory.”

## 연구 설계와 대상

- 본 연구를 위해 최종적으로 6명의 ACLR 후 환자(남성 3명, 여성 3명)와 6명의 건강한 대조군(남성 3명, 여성 3명)이 분석에 포함되었다. _(근거: PAGE 2, II. METHODS)_
  - 근거 원문: “Six patients post-ACLR (3M, 3F; mean ± std age: 16.83 ± 2.48 years; height: 1.74 ± 0.05 m; weight: 68.175 ± 7.82 kg) and six healthy participants (3M, 3F; mean ± std age: 22.83 ± 2.78 years; 1.76 ± 0.078 m; 66.0 ± 9.2 kg) were recruited for this study, after receiving approval from Carnegie Mellon University’s Institutional Review Board and obtaining informed consent.”
- 초기 대상자 중 2명의 환자가 추적 관찰에 실패하고, 건강한 참가자 1명이 3일 동안만 원격 모니터링 데이터를 완료하여 최종 분석에서 제외되어 최종 샘플 수는 12명이 되었다. _(근거: PAGE 2, II. METHODS)_
  - 근거 원문: “Initially, eight patients and seven healthy participants were assessed for eligibility. Two patients failed to follow up and one healthy participant only completed three days of remote monitoring data. Therefore, these three participants were removed from the study before data analysis, yielding a final sample size of 12 participants.”
- 6명의 ACLR 환자는 동일한 외과의에게 단일 사지에 자가 대퇴사두근 건을 이용한 일차 단일 다발 재건술을 받았다. _(근거: PAGE 2, II. METHODS)_
  - 근거 원문: “Six ACLR patients underwent a primary single-bundle reconstruction with an autologous quadriceps tendon on a single limb by the same surgeon.”
- 건강한 참가자는 무릎 부상 또는 다른 주요 근골격계 부상 이력이 없어야 했으며, ACLR 환자는 수술 당시 부상 후 10주 이내여야 했고 보행 기록은 수술 3개월 후에 측정되었다. _(근거: PAGE 2, II. METHODS)_
  - 근거 원문: “To be eligible to participate, healthy participants had to have no prior history of knee injury or any other major musculoskeletal injury. ACLR patients had to be within 10 weeks of injury at the time of surgery. We recorded their gait three months after the surgery.”

## 방법

- 보행 평가는 실험실 러닝머신, 실험실 지상 보행, 일상생활의 세 가지 조건으로 수행되었다. _(근거: PAGE 2, II. METHODS)_
  - 근거 원문: “We evaluated walking in three conditions: laboratory treadmill, laboratory overground, and daily life (Figure 1).”
- 실험실에서 참가자들은 러닝머신 보행 2회(회당 2분) 및 지상 보행 2회(회당 30초) 동안 마커 기반 모션 캡처 및 IMU로 추적되었다. _(근거: PAGE 2, II. METHODS)_
  - 근거 원문: “In the laboratory, participants were tracked with marker-based motion capture and IMUs for two trials of treadmill walking (2 minutes per trial) and two trials of overground walking (30 seconds per trial).”
- 일상생활 모니터링을 위해 참가자들은 4개의 IMU 센서를 집으로 가져가 대퇴부와 정강이 외측에 착용한 채 5일 연속 착용하도록 안내받았다. _(근거: PAGE 2, II. METHODS)_
  - 근거 원문: “For the daily life monitoring component, participants were sent home with four IMU sensors and instructed to wear them for five consecutive days, with the sensors placed laterally on the thighs and shanks.”
- > **[AS-IS]** 자이로스코프의 각속도는 차단 주파수 0.25 Hz의 고통과 필터와 35 Hz의 저통과 필터로 필터링되었다. _(근거: PAGE 3, II. METHODS)_
>
> **[TO-BE]** 자이로스코프의 각속도는 차단 주파수 0.25 Hz의 고역통과 필터와 35 Hz의 저역통과 필터로 필터링되었다.
>
> _(사실검증 — 번역오류/경미: '고통과 필터'는 원문의 high-pass filter를 잘못 옮긴 오탈자성 번역이다. 의미상 '고역통과 필터'가 맞다.)_
  - 근거 원문: “Angular velocities from the gyroscopes were filtered with a high-pass filter with a cutoff frequency of 0.25 Hz, followed by a low-pass filter with a cut-off frequency of 35 Hz.”
- 가설 검정을 위해 환경을 반복 측정 요소로 하는 반복측정 분산분석(RM ANOVA)을 사용하였으며, 유의수준은 본페로니 교정을 적용하여 0.017로 설정되었다. _(근거: PAGE 3, II. METHODS)_
  - 근거 원문: “Repeated measures (RM) analyses of variance (ANOVA) were used to test our two leading hypotheses, with the environment as our repeated measure.”

## 핵심 결과

- ACLR 후 환자들은 실험실에 비해 일상생활에서 더 긴 보행 주기와 이중 지지 시간을 나타냈다. _(근거: PAGE 4, III. RESULTS)_
  - 근거 원문: “Patients post-ACLR walked with greater gait-cycle and double-support times in daily life compared to the laboratory (p < 0.017; Figure 2; Table III).”
- ACLR 후 환자들은 실험실 환경에서는 건강한 참가자들과 유사하게 걸었으나, 일상생활에서는 더 긴 보행 주기 시간, 더 긴 이중 지지 시간, 그리고 더 짧은 단일 지지 시간을 나타냈다. _(근거: PAGE 4, III. RESULTS)_
  - 근거 원문: “Patients post-ACLR walked similarly to healthy participants in the laboratory, but not in daily life, where they walked with longer gait-cycle and double-support time and shorter single-support time (Figure 2).”
- 일상생활 중 ACLR 환자들의 단일 지지 불균형은 건강한 참가자들보다 높았다. _(근거: PAGE 4, III. RESULTS)_
  - 근거 원문: “Single-support asymmetry was also higher in ACLR patients than in healthy participants during daily life.”
- 건강한 참가자는 ACLR 환자보다 하루 평균 935회 더 많은 보행 주기를 완료했다. _(근거: PAGE 5, III. RESULTS)_
  - 근거 원문: “Healthy participants completed 935 more gait cycles per day than ACLR patients (p = 0.016)(Figure 3A).”

## 저자 결론

- 본 연구 결과는 실험실 환경이 건강한 참가자와 ACLR 환자 간의 보행 차이를 숨길 수 있으며, 자연 환경 생체역학으로의 이동이 수술 후 보행에 대한 이해를 확장하는 데 새로운 기대를 제시함을 보여준다. _(근거: PAGE 6, IV. DISCUSSION)_
  - 근거 원문: “While we did not study the factors that make the gait laboratory not an ideal environment to capture free-living behavior, on their own, these findings provide initial evidence that the laboratory environment masks gait differences between healthy participants and ACLR patients and suggests that the move to natural-environment biomechanics holds new promise in expanding our understanding of post-surgical gait.”
- 실험실과 일상생활 간에 관찰된 보행 특징의 차이는 기존의 보행 분석 연구들이 실제 자연 환경에서의 행동을 현실적으로 포착하지 못했을 가능성을 시사한다. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “While the reasons behind the observed differences were not studied, these results suggest that gait-analysis studies to date may have not realistically captured natural-environment behavior.”
- 실험실에서 관찰되는 것과 일상생활에서의 보행 차이는 실험실 관찰 내용이 일상생활의 운동 패턴을 대변하지 못할 수 있음을 나타낸다. _(근거: PAGE 7, IV. DISCUSSION)_
  - 근거 원문: “The differences we observed in temporal parameters between groups in daily life, however, suggest that what is observed in the laboratory may not be representative of movement patterns in daily life.”

## 연구의 한계

- 건강한 대조군과 ACLR 그룹의 나이가 매칭되지 않았다. _(근거: PAGE 6, IV. DISCUSSION)_
  - 근거 원문: “First, healthy and ACLR groups were not age-matched.”
- 원격 모니터링 동안 참가자가 매일 아침 직접 IMU를 장착했으며, 센서-신체 보정이 수행되지 않았다. _(근거: PAGE 6, IV. DISCUSSION)_
  - 근거 원문: “Second, participants were tasked with applying the IMUs themselves each morning during the remote monitoring component of the study and no sensor-to-body calibration was implemented before extracting temporal parameters from the sagittal plane angular velocity data.”
- 실험실 환경에 비해 일상생활에서 다양한 지형 조건이 존재했으나 이를 통제하지 못했다. _(근거: PAGE 6, IV. DISCUSSION)_
  - 근거 원문: “A third limitation is the variable terrain in daily life compared to the laboratory.”
- 사후 검정력 분석에 따르면 일부 비유의미한 보행 변수는 통계적 유의성을 확인하기 위해 더 큰 표본 크기가 필요하다. _(근거: PAGE 7, IV. DISCUSSION)_
  - 근거 원문: “Fourth, a post-hoc power analysis for the second hypothesis revealed that limp and step time asymmetry required a larger sample size to help in rejecting the null.”
- 대퇴사두근 근력 등 회복 특성이 수술 후 보행에 미칠 수 있는 영향을 고려하지 않고, 모든 ACLR 회복 표현형을 단일 집단으로 묶어 분석하였다. _(근거: PAGE 7, IV. DISCUSSION)_
  - 근거 원문: “Clustering all ACLR recovery phenotypes into a single population is therefore not ideal, and modern data science tools such as cluster analysis and larger datasets will play a critical role in refining future analyses and insights.”

## 생각해볼 내용

- 실험실 보행 분석 연구가 보행 대칭성의 자연스러운 회복 시점을 실제보다 이르게 평가했을 수 있음을 설명한다. _(근거: PAGE 7, IV. DISCUSSION)_
  - 근거 원문: “Our results, together with the growing body of evidence in other clinical populations, therefore, call into question our existing understanding of gait recovery timelines, which have largely been informed by laboratory-based assessment and highlight the need for remote assessments to address the gait lab effect.”
- > **[AS-IS]** 일상생활에서의 다양한 지형 조건이나 주의 분산 요소가 오히려 수술 후 환자의 보상적 보행 전략을 관찰하는 데 필수적인 자극이 될 수 있음을 의미한다. _(근거: PAGE 7, IV. DISCUSSION)_
>
> **[TO-BE]** 주의 분산이 거의 없는 실험실 보행은 참가자가 보상적 보행 전략을 가리게 만들 수 있으므로, 일상생활 보행 관찰이 실험실에서 드러나지 않는 보행 양상을 포착하는 데 도움이 될 수 있다.
>
> _(사실검증 — 과장/경미: 원문은 실험실의 주의 분산이 없는 환경이 보상적 보행 전략을 가릴 수 있고, 일상생활의 지형 차이는 통제하지 못한 한계라고 설명한다. 그러나 요약은 다양한 지형 조건이나 주의 분산 요소가 '필수적인 자극'이라고 더 강하게 해석한다.)_
  - 근거 원문: “Distraction-free laboratory gait may inadvertently cause a participant to mask compensatory gait strategies in the laboratory.”
- 동작 분석이 고가의 실험실 환경에서 간편한 모바일 웨어러블 시스템으로 전환되면 임상 연구와 임상 진료 현장 간의 격차를 해소할 수 있다. _(근거: PAGE 8, IV. DISCUSSION)_
  - 근거 원문: “Even if traditional laboratory studies identified the gait patterns leading to poor recovery and eventual osteoarthritis, the gap between research and clinical practice would remain wide because traditional gait analysis is too expensive and time-consuming for implementation in clinics.”

## 이 연구가 지적한 선행연구의 문제점

- 마커 기반 모션 캡처는 실험실 공간, 고가의 장비 및 전문 연구원이 필요해 임상 및 대규모 생체역학 연구로의 확장이 제한된다. _(근거: PAGE 1, I. INTRODUCTION)_
  - 근거 원문: “Marker-based motion capture, which has been historically used to study gait, requires designated laboratory space, expensive equipment, and trained biomechanists, preventing scalability to clinics and large biomechanics research studies.”
- 또한, 연구자의 존재가 환자의 보행 움직임에 영향을 미칠 수 있다. _(근거: PAGE 1, I. INTRODUCTION)_
  - 근거 원문: “Additionally, the presence of researchers may influence how patients move.”
- 수술 후 보행에 관한 기존의 지식은 대부분 실험실 내 평가를 기반으로 하고 있어, 일상 모니터링이 이러한 통찰력을 어떻게 바꿀 수 있는지 보여주는 연구가 아직 없었다. _(근거: PAGE 1, I. INTRODUCTION)_
  - 근거 원문: “Existing knowledge of post-ACLR gait is largely based on laboratory assessments, with no study to date providing evidence of how daily-life monitoring may reshape these insights.”

## 이 연구의 해결 방식과 기여

- 본 연구는 ACLR 수술 후 원격 환자 모니터링에서 다중 센서 프로토콜 사용에 대한 초기 지지 근거를 제공한다. _(근거: PAGE 8, IV. DISCUSSION)_
  - 근거 원문: “Altogether, this work provides initial support for the use of multi-sensor protocols in remote patient monitoring following ACLR surgery.”
- 웨어러블 센서는 수술 후 보행에 대한 깊은 이해와 외상 후 골관절염 위험에 처한 환자의 특정 패턴을 파악할 수 있는 경로를 제공한다. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “Wearable sensors now offer a path toward deeper understanding of post-surgical gait and the specific patterns that may place certain patients at risk for post-traumatic osteoarthritis.”

## 레퍼런스할 수 있는 내용

### 1. 실험실 참여 의식이 보행 속도 등에 미치는 영향 (호손 효과)

- 원문 발췌: “Recent work has identified a similar effect in gait analysis, where study participants increase their gait speed, cadence, and stride length when researchers are present in the laboratory [3], [4], [5].”
- 한국어 번역: 최근 연구에 따르면 보행 분석에서도 유사한 효과가 확인되었는데, 연구원이 실험실에 있을 때 피험자들은 보행 속도, 분당 걸음 수, 보폭을 증가시킨다.
- 원문 위치: PAGE 1, I. INTRODUCTION
- 원문 내 인용표기: [3], [4], [5]
- 해당 선행문헌: [3] K. B. Friesen, Z. Zhang, P. G. Monaghan, G. D. Oliver, and J. A. Roper, “All eyes on you: How researcher presence changes the way you walk,” Sci. Rep., vol. 10, no. 1, Oct. 2020, Art. no. 1, doi: 10.1038/s41598-020-73734-5. [4] J. Jeon et al., “Influence of the Hawthorne effect on spatiotemporal parameters, kinematics, ground reaction force, and the symmetry of the dominant and nondominant lower limbs during gait,” J. Biomechanics, vol. 152, May 2023, Art. no. 111555, doi: 10.1016/j.jbiomech.2023.111555. [5] L. A. Hutchinson, M. J. Brown, K. J. Deluzio, and A. R. De Asha, “Self-selected walking speed increases when individuals are aware of being recorded,” Gait Posture, vol. 68, pp. 78–80, Feb. 2019, doi: 10.1016/j.gaitpost.2018.11.016.
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
