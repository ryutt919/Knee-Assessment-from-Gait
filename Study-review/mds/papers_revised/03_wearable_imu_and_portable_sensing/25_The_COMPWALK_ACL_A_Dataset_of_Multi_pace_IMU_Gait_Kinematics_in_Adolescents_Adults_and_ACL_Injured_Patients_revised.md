# The COMPWALK-ACL: A Dataset of Multi-pace IMU Gait Kinematics in Adolescents, Adults, and ACL Injured Patients

Yona, T., Peskin, B., & Fischer, A. (2026). The COMPWALK-ACL: A Dataset of Multi-pace IMU Gait Kinematics in Adolescents, Adults, and ACL Injured Patients. Scientific Data, 13(4). https://doi.org/10.1038/s41597-025-06307-8

## 서지정보

- 저자: Tomer Yona, Bezalel Peskin, Arielle Fischer
- 연도: 2026
- 저널: Scientific Data
- DOI: https://doi.org/10.1038/s41597-025-06307-8
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/03_wearable_imu_and_portable_sensing/The COMPWALK-ACL - A Dataset of Multi-pace IMU Gait Kinematics in Adolescents, Adults, and ACL Injured Patients.pdf
- 분석 provider: antigravity

## 연구 목적

- 본 논문은 Xsens Awinda IMU 시스템을 사용하여 획득한 하지 운동역학 데이터셋을 제공하는 것을 목적으로 한다. _(근거: Page 1, Abstract)_
  - 근거 원문: “This paper presentsalowerlimbkinematicdatasetacquiredwiththeXsensAwindaIMUsystem.”
- 연령에 따른 가변성과 수술 후 변화를 표준화된 IMU 프로토콜을 사용하여 포착하는 공개 데이터셋을 개발하고자 했다. _(근거: Page 2, Background & Summary)_
  - 근거 원문: “Toaddress thisgap,wedevelopedtheCOMPWALK-ACLdatasetasanopenlyavailableresourcecapturingbothage-related variability and post-surgical changes using a standardized IMU protocol.”

## 연구 설계와 대상

- 본 연구는 건강한 성인 25명, 건강한 청소년 27명, ACL 부상자 40명(이 중 27명은 ACL 재건술 3개월 후 추적 조사 완료)을 포함한 총 92명의 참가자로 구성된 IMU 기반 보행 데이터셋이다. _(근거: Page 2, Background & Summary)_
  - 근거 원문: “Itis
an IMU-based gait dataset comprising data from 92 participants: 25 healthy adults, 27 healthy adolescents, and
40 individuals with ACL injury, of whom 27 completed a follow-up assessment three months after ACL recon-
struction.”
- 건강한 성인 코호트는 18~45세의 건강한 남성과 여성으로 구성되었다. _(근거: Page 2, Methods)_
  - 근거 원문: “The healthy adults’ cohort included healthy males and females aged 18–45 years.”
- 참가자 정보 보호를 위해 모든 데이터는 수집 시점에 비식별화되었다. _(근거: Page 2, Methods)_
  - 근거 원문: “To protect participant information, all data were de-identified at the point of collection.”
- 각 코호트의 인구통계학적 특성 요약은 표 1에 제시되어 있다. _(근거: Page 2, Methods)_
  - 근거 원문: “A summary of demographic characteristics for each cohort is presented in Table 1.”

## 방법

- 보행 테스트 중 참가자들에게 어떠한 외부 피드백도 제공되지 않았다. _(근거: Page 3, Methods)_
  - 근거 원문: “No external feedback was provided.”
- 직선 오버그라운드 보행 구간의 데이터만 분석에 활용되었다. _(근거: Page 3, Methods)_
  - 근거 원문: “Only the straight overground walking segments were used for analysis.”
- 각 속도 조건에 대해 참가자들은 3회의 연속된 트라이얼을 수행했다. _(근거: Page 3, Methods)_
  - 근거 원문: “For each speed condition, participants completed three
consecutive trials.”

## 핵심 결과

- 느린 보행 속도에서 건강한 성인은 ACLD 그룹보다 유의하게 더 빠르게 걸었다(0.87±0.02m/s 대 0.79±0.01m/s, p=0.0037). _(근거: Page 5, Technical Validation)_
  - 근거 원문: “At slow walking speed, healthy adults walked significantly faster than the ACLD group (0.87±0.02m/s vs.
0.79±0.01m/s,p=0.0037),withnoothersignificantpairwisedifferences.”
- 모든 속도 조건에서 보행 속도, 케이던스, 보폭에 대해 그룹 간 유의미한 차이가 관찰되었다. _(근거: Page 5, Technical Validation)_
  - 근거 원문: “At all speed conditions, significant group differences were observed in gait speed, cadence, and stride length.”
- 보통 보행 속도에서 ACLD 및 ACLR 그룹은 건강한 성인보다 유의하게 더 느린 속도를 보였다(ACLD/ACLR 모두 1.19m/s 대 건강한 성인 1.33m/s, p<0.001). _(근거: Page 5, Technical Validation)_
  - 근거 원문: “At normal walking speed, both ACLD and ACLR groups demonstrated significantly slower gait speed than
healthy adults (ACLD: 1.19±0.01m/s; ACLR: 1.19±0.02m/s; healthy adults: 1.33±0.02m/s; both p<0.001).”

## 저자 결론

- 이 데이터셋은 청소년-성인의 전 생애에 걸친 규준 관절각 참조 데이터 구축, 스트라이드 세분화 및 보행 속도 분류 알고리즘의 벤치마킹, 그리고 ACLR 수술 후 회복 경과 평가를 위한 종단적 변화 추적 등 다양한 목적으로 활용될 수 있다. _(근거: Page 7, Usage Notes)_
  - 근거 원문: “The dataset supports a range of uses, including establishing normative joint-angle references across the ado-
lescent–adult lifespan, benchmarking stride-segmentation or gait-speed classification algorithms, and tracking
longitudinal changes after ACLR to evaluate recovery.”
- 본 데이터셋은 하지 관절 운동역학 및 시공간 파라미터를 포함하여, 다양한 보행 속도와 임상 조건에서 IMU 기반 보행 분석법 개발 및 연령별 규준 보행 연구 등을 가능하게 한다. _(근거: Page 1, Abstract)_
  - 근거 원문: “Thedatasetcontainsspatiotemporalparameters,aswellaslowerlimbjoint
kinematics.Itenablesresearchonnormativegaitacrossagegroups,theeffectsofACLinjuryandearly
recoveryonmovementpatterns,andthedevelopmentofIMU-basedgaitanalysismethodsunder
differentwalkingspeedsandclinicalconditions.”

## 연구의 한계

- 지면반력기 데이터와 카메라 기반 모션 캡처가 누락되어 골드 스탠다드 데이터가 없으므로 관절 각도 및 시공간 지표의 유효성을 검증할 수 없다. _(근거: Page 7, Usage Notes)_
  - 근거 원문: “Second,force-platedataandcamera-based
motioncapturearenotincluded.Therefore,withoutgold-standardmeasurements,thedatasetcannotbeusedto
validateanyofthereportedmetrics,includingjointkinematicsandspatiotemporalparameters.”
- 각 오버그라운드 테스트의 거리가 약 20m로 짧아, 확보된 안정된 보행 주기(stride)가 8~10개에 불과하다. _(근거: Page 7, Usage Notes)_
  - 근거 원문: “First,
eachovergroundtrialspansonly~20m,yielding8–10steadystrides.”
- 각 참가자의 세션 내에서는 신발과 보행 표면이 일정했으나 참가자 간에 표준화되지 않아 피험자 간 차이가 유발되었을 수 있다. _(근거: Page 7, Usage Notes)_
  - 근거 원문: “Lastly,footwear
and walkway surface were kept consistent within a session but not standardised across participants, possibly
introducing between-subject differences.”

## 생각해볼 내용

- > **[AS-IS]** 제시된 데이터셋의 보행 지표 측정값 크기가 제조사 백서의 값과 유사하므로 데이터의 생리학적 신뢰도가 높음을 알 수 있다. _(근거: Page 6, Technical Validation)_
>
> **[TO-BE]** 제시된 건강한 성인 코호트의 시공간 보행 지표는 Xsens 백서에 보고된 기준값과 크기 면에서 유사했다.
>
> _(사실검증 — 과장/경미: 원문은 건강한 성인 코호트의 시공간 보행 파라미터가 Xsens 백서 값과 크기 면에서 comparable하다고만 설명한다. 이를 근거로 데이터의 '생리학적 신뢰도가 높음'을 단정하는 것은 원문보다 강한 해석이다.)_
  - 근거 원문: “Observed values were comparable in magnitude to those reported in the whitepaper.”
- > **[AS-IS]** 코드 저장소에 예제 스크립트가 포함되어 연구자들의 데이터셋 재사용성과 활용성이 증대될 것이다. _(근거: Page 7, Code availability)_
>
> **[TO-BE]** 코드 저장소에는 데이터셋의 추가적인 잠재 활용 예를 보여주는 몇 가지 예제 스크립트가 포함되어 있다.
>
> _(사실검증 — 근거불충분/경미: 원문은 GitHub 저장소에 데이터셋의 잠재적 추가 활용을 보여주는 예제 스크립트가 포함되어 있다고만 말한다. '재사용성과 활용성이 증대될 것'이라는 효과는 원문에서 직접 확인되지 않는다.)_
  - 근거 원문: “The repository also includes a few example scripts illustrating additional potential uses of the dataset.”

## 이 연구가 지적한 선행연구의 문제점

- 보행 속도가 관절 운동역학 및 보상 작용을 크게 조절함에도 불구하고, 건강한 대조군과 ACL 환자군 모두에서 속도 변화를 체계적으로 적용한 보행 데이터셋이 매우 부족하다. _(근거: Page 1, Background & Summary)_
  - 근거 원문: “Yet,fewpubliclyavailabledatasetssystematicallyincorporatevariationinwalkingspeedacrossbothhealthy
andACL-injuredpopulations,despitethefactthatspeedstronglymodulatesjointkinematicsandcompensatory
strategies10,17 .”
- 기존 데이터셋들은 대부분 동일한 피험자 내에서의 수술 전 및 수술 후 종단적 측정을 포함하지 못하고 있다. _(근거: Page 2, Background & Summary)_
  - 근거 원문: “Additionally, most do not include pre-
and post-operative measures within the same individuals18–21 .”

## 이 연구의 해결 방식과 기여

- 기존의 데이터셋들과 대조적으로 COMPWALK-ACL 데이터셋은 고유한 가치와 기여를 가진다. _(근거: Page 2, Background & Summary)_
  - 근거 원문: “Comparedtoexistingdatasets,theCOMPWALK-ACLdataset(COMParingmulti-paceWALKingkinemat-
icsviaIMUinhealthyadolescents,adults,andindividualswithACLinjury)providesauniquecontribution.”
- 생태학적으로 타당한 조건 하에 피험자 내 종단적 분석 및 그룹 간 비교를 지원하여 ACL 재건술 후의 생체역학적 적응 연구와 연령대별 규준 마련에 기여한다. _(근거: Page 2, Background & Summary)_
  - 근거 원문: “This dataset supports within-subject
andbetween-groupanalysesunderecologicallyvalidconditions,facilitatingthestudyofpost-ACLbiomechan-
ical adaptations as well as the development of normative gait references for both youth and adults.”

## 레퍼런스할 수 있는 내용

### 1. 인간 보행 정보의 임상적 가치

- 원문 발췌: “Human gait provides objective information about an individual’s movement capabilities, neurological func-
tion, and musculoskeletal health1,2 .”
- 한국어 번역: 인간의 보행은 개인의 움직임 능력, 신경학적 기능 및 근골격계 건강에 대한 객관적인 정보를 제공한다.
- 원문 위치: Page 1, Background & Summary
- 원문 내 인용표기: 1,2
- 해당 선행문헌: 1. Das, R., Paul, S., Mourya, G. K., Kumar, N. & Hussain, M. Recent Trends and Practices Toward Assessment and Rehabilitation of
Neurodegenerative Disorders: Insights From Human Gait. Front Neurosci. 16, 859298, https://doi.org/10.3389/fnins.2022.859298
(2022).
2. Winner, T. S. et al. Discovering individual-specific gait signatures from data-driven models of neuromechanical dynamics. PLoS
Comput Biol. 19(10), e1011556, https://doi.org/10.1371/journal.pcbi.1011556 (2023).
- 주장 유형: background_citation
- 활용 맥락과 주의: 인간 보행 분석이 신경학적 및 근골격계 건강 상태를 진단하고 평가하는 데 유용한 객관적 지표임을 지지할 때 인용할 수 있다.

### 2. 전통적인 보행 분석 시스템의 한계

- 원문 발췌: “While accurate, these systems are limited to controlled environments and are not suitable for
use outside the laboratory7 .”
- 한국어 번역: 이러한 시스템은 정확하지만 통제된 환경으로 제한되며 실험실 외부에서 사용하기에 적합하지 않다.
- 원문 위치: Page 1, Background & Summary
- 원문 내 인용표기: 7
- 해당 선행문헌: 7. Prisco, G. et al. Validity of Wearable Inertial Sensors for Gait Analysis: A Systematic Review. Diagnostics (Basel). 15, https://doi.
org/10.3390/diagnostics15010036 (2024).
- 주장 유형: background_citation
- 활용 맥락과 주의: 전통적인 카메라 기반 보행 분석 시스템이 가진 공간적 제약과 실험실 외부 적용의 어려움을 지지할 때 인용할 수 있다.

### 3. IMU 센서의 보행 분석 적용 장점

- 원문 발췌: “Wearable sensors, such as Inertial Measurement Units (IMUs), offer a portable and cost-effective alternative
that enables gait assessment in real-world and clinical settings8 .”
- 한국어 번역: 관성 측정 장치(IMU)와 같은 웨어러블 센서는 실제 환경 및 임상 환경에서 보행 평가를 가능하게 하는 휴대 가능하고 비용 효율적인 대안을 제공한다.
- 원문 위치: Page 1, Background & Summary
- 원문 내 인용표기: 8
- 해당 선행문헌: 8. Jung, S. et al. The Use of Inertial Measurement Units for the Study of Free Living Environment Activity Assessment: A Literature
Review. Sensors (Basel). 20(19), https://doi.org/10.3390/s20195625 (2020).
- 주장 유형: background_citation
- 활용 맥락과 주의: IMU 센서가 전통적인 랩 기반 장비에 비해 높은 휴대성과 경제성을 가지며 실생활 및 임상 연구에 적합하다는 주장을 뒷받침할 때 사용된다.

### 4. 빠른 속도 조건에서 ACL 그룹과 건강한 대조군의 보행 속도 차이

- 원문 발췌: “At fast speed, both ACL groups exhibited significantly slower gait speed than healthy adults (ACLD:
1.73±0.03m/s; ACLR: 1.66±0.03m/s; healthy adults: 1.94±0.03m/s; both p<0.001).”
- 한국어 번역: 빠른 속도에서 두 ACL 그룹 모두 건강한 성인에 비해 유의하게 느린 보행 속도를 보였다(ACLD: 1.73±0.03m/s, ACLR: 1.66±0.03m/s, 건강한 성인: 1.94±0.03m/s, 모두 p<0.001).
- 원문 위치: Page 5, Technical Validation
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: ACL 부상 환자(ACLD) 및 재건술 후 3개월 환자(ACLR)가 빠른 보행 시 건강한 대조군에 비해 보행 속도가 유의하게 감소한다는 자체 분석 결과다.
