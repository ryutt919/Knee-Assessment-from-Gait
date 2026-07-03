# Derivation of the Gait Deviation Index for Spinal Cord Injury

Herrera-Valenzuela, D., Sinovas-Alonso, I., Moreno, J. C., Gil-Agudo, Á., & del-Ama, A. J. (2022). Derivation of the Gait Deviation Index for Spinal Cord Injury. Frontiers in Bioengineering and Biotechnology, 10, Article 874074. https://doi.org/10.3389/fbioe.2022.874074

## 서지정보

- 저자: Diana Herrera-Valenzuela, Isabel Sinovas-Alonso, Juan C. Moreno, Ángel Gil-Agudo, Antonio J. del-Ama
- 연도: 2022
- 저널: Frontiers in Bioengineering and Biotechnology
- DOI: 10.3389/fbioe.2022.874074
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/07_composite_kinematic_kinetic_scoring_indices/Derivation of the Gait Deviation Index for Spinal Cord Injury.pdf
- 분석 provider: antigravity

> **한국어 제목**: 척수 손상을 위한 보행 편차 지수의 유도

## 분류 태그

- ACL 연구: false
- IMU 사용: false
- 보행 데이터: true
- Score 제시: true

## 연구 목적

- 본 연구의 주된 목적은 GDI 이면의 수학적 방법론을 성인 척수 손상(SCI) 환자의 데이터셋에 적용하여 새로운 SCI-GDI를 도출하고, 이를 기존 GDI 및 WISCI II와 비교하여 평가하는 것이다. _(근거: PAGE 3, Section 1)_
  - 근거 원문: “The main objective of this article is to investigate the application of the mathematical methodology behind the GDI (Schwartz and Rozumalski, 2008) to a dataset of adults with SCI, resulting in the new SCI-GDI.”

## 연구 설계와 대상

- C1~L5의 손상 레벨과 ASIA 손상 척도(AIS) C~D를 가진 16~70세 사이의 척수 손상 환자로부터 수집된 302개 보행(strides) 데이터를 분석 대상으로 삼았다. _(근거: PAGE 3, Section 2.1)_
  - 근거 원문: “A total of 302 strides from patients aged between 16 and 70 years old (33.91 ± 17.86), with injury levels between C1 and L5 and the ASIA impairment scale (AIS) C to D were gathered.”
- 대조군으로는 보행 병리가 없는 성인의 446개 보행 데이터가 수집되었다. _(근거: PAGE 3, Section 2.1)_
  - 근거 원문: “In addition, a control group with the 3D kinematic gait data of 446 strides from adults without gait pathologies was collected.”

## 방법

- 하지에 22개 활성 마커를 부착하는 표준 프로토콜 하에 세 대의 스캐너와 두 대의 Kistler 힘판을 구비한 10m 보행로에서 Codamotion 모션 캡처 시스템을 사용하여 3차원 보행 운동학 데이터를 수집하였다. _(근거: PAGE 3, Section 2.2)_
  - 근거 원문: “A Codamotion® motion capture system (Charnwood Dynamics Ltd, United Kingdom) was used to capture 3D kinematic gait data. The standard protocol with 22 active markers placed on the lower limbs (Charnwood Dynamics Limited, 2004), three scanners, and two force platforms Kistler 9286A (Kistler Group, Switzerland) in the center of a 10-meter walkway were used.”
- 3D 보행 분석을 기반으로 수집된 9개 관절 각도 데이터를 이용하여 척수 손상 보행 데이터를 가장 높은 피델리티로 재건하고 대부분의 분산을 설명할 수 있는 최적의 축소 정규직교 기저(SCI-GDI 기저)를 계산하였다. _(근거: PAGE 4, Section 2.4)_
  - 근거 원문: “Using the dataset described previously, the ﬁrst step of the data analysis was the computation of what we call the SCI-GDI basis, that is, the optimal reduced order orthonormal basis to reconstruct gait data of SCI with high ﬁdelity and to account for most of the variance of the SCI dataset (data analysis details in Section 2.4.1).”

## 핵심 결과

- 척수 손상 환자군의 보행 패턴에서 대부분의 분산을 설명하고 보행 곡선의 고품질 재건을 제공하기 위해 21개 특징으로 구성된 기저가 필요함을 보여주었다. _(근거: PAGE 1, ABSTRACT)_
  - 근거 원문: “Our ﬁndings show that a 21-feature basis is necessary to account for most of the variance in gait patterns in the SCI population and to provide high-quality reconstructions of the gait curves included in the dataset and in foreign data.”

## 저자 결론

- 척수 손상 환자에게 뇌성마비 기반의 오리지널 GDI를 적용하면 보행 기능이 과대평가될 수 있으므로, 보행 장애의 민감한 평가를 위해 새로운 SCI-GDI의 도입을 권장한다. _(근거: PAGE 1, ABSTRACT)_
  - 근거 원문: “In conclusion, the implementation of the original GDI in SCI may lead to overestimation of gait function, and our new SCI-GDI is moresensitivetolargergaitimpairmentthantheGDI.”

## 연구의 한계

- SCI-GDI 기저 연산이 안정적인 결과를 나타냈으나, 검증 결과가 훈련 데이터보다 낮아 일반성을 확보하고 stride 수에 무관하게 결과가 독립적임을 확인하기 위해서는 더 큰 데이터셋의 구축이 요구된다. _(근거: PAGE 10, Section 4)_
  - 근거 원문: “First, as mentioned before, even though the computation of the SCI-GDI basis showed stable results, using a larger dataset would allow us to verify that our results (number of features m, VAF and reconstruction percentages) indeed remain independently of the number of strides in the database.”

## 생각해볼 내용

- Codamotion 등 대부분의 모션 캡처 시스템에서 골반 마커 부착 기준으로 사용되는 전상장골극과 후상장골극은 연부 조직 인공물(soft tissue artefacts)의 영향을 받기 쉬워 골반의 3차원 운동학 측정값의 오차 전파를 야기하는 내재적 한계가 있다. _(근거: PAGE 10, Section 4)_
  - 근거 원문: “The anatomical landmarks used to place or align the pelvic markers on most motion capture systems, including the Codamotion, are the anterior and posterior superior iliac spines. These are bony protuberances in the pelviscovered with adipose tissue; therefore, the markers cannot be placed accurately on the subjects (C-Motion Wiki Documentation, 2019) and are prone to soft tissue artefacts (Langley et al., 2019).”
- 연구진은 SCI-GDI가 척수 손상 환자의 임상적 및 기능적 다양성을 충실히 반영할 수 있도록 중증도, 손상 수준, 발병 기간, 성별, 연령 측면에서 매우 다양하고 넓은 범위의 보행 데이터를 의도적으로 수집하여 적용 범위를 보장하고자 했다. _(근거: PAGE 11, Section 4)_
  - 근거 원문: “We intentionally captured a wide variety of gait data of SCI with different severity, neurological level of injury, time since injury onset, sex, and age, to capture the largest variety in gait patterns we had access to, and guarantee that the SCI-GDI could properly represent any of these patterns.”

## 이 연구가 지적한 선행연구의 문제점

- 확인된 내용 없음

## 이 연구의 해결 방식과 기여

- 소아 뇌성마비 환아에 기초해 개발된 기존 GDI의 한계를 극복하기 위해 성인 척수 손상 환자 집단의 3DGA 데이터셋에 동일한 수학적 방법론을 성공적으로 적용하여 척수 손상 전용의 SCI-GDI를 개발해 냈다. _(근거: PAGE 3, Section 1)_
  - 근거 원문: “The main objective of this article is to investigate the application of the mathematical methodology behind the GDI (Schwartz and Rozumalski, 2008) to a dataset of adults with SCI, resulting in the new SCI-GDI.”
- 오리지널 GDI의 15개 특징 기저 대신 성인 척수 손상 데이터로부터 도출된 21개 특징 벡터 기저를 이용함으로써 척수 손상 환자의 보행 곡선을 훨씬 정밀하게 복원할 수 있으며 임상적 과대평가 위험을 해결한다. _(근거: PAGE 11, Section 5)_
  - 근거 원문: “The SCI-GDI is calculated using a 21-feature vectorial basis derived from gait data of adult population with SCI, instead of the 15-feature basis used for the original GDI.”

## 레퍼런스할 수 있는 내용

### 1. 오리지널 GDI 적용 시 척수 손상 환자의 보행 기능 과대평가 및 SCI-GDI의 민감도

- 원문 발췌: “In conclusion, the implementation of the original GDI in SCI may lead to overestimation of gait function, and our new SCI-GDI is moresensitivetolargergaitimpairmentthantheGDI.”
- 한국어 번역: 결론적으로, 척수 손상 환자에게 오리지널 GDI를 적용하면 보행 기능이 과대평가될 수 있으며, 우리의 새로운 SCI-GDI가 기존 GDI보다 큰 보행 장애에 더 민감하다.
- 원문 위치: PAGE 1, ABSTRACT
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: 척수 손상 환자의 보행 능력 분석 시 뇌성마비용 GDI를 사용하면 임상적 보행 기능이 실제보다 과대평가될 위험성이 크고, 척수 손상 특이적 지표의 도입 필요성과 민감도를 지지하는 데 활용할 수 있음.
