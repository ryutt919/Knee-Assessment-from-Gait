# Gait Classification Using Mahalanobis–Taguchi System for Health Monitoring Systems Following Anterior Cruciate Ligament Reconstruction

Sakeran, H., Abu Osman, N. A., & Abdul Majid, M. S. (2019). Gait Classification Using Mahalanobis–Taguchi System for Health Monitoring Systems Following Anterior Cruciate Ligament Reconstruction. Applied Sciences, 9(16), 3306. https://doi.org/10.3390/app9163306

## 서지정보

- 저자: Hamzah Sakeran, Noor Azuan Abu Osman, Mohd Shukry Abdul Majid
- 연도: 2019
- 저널: Applied Sciences
- DOI: 10.3390/app9163306
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/07_composite_kinematic_kinetic_scoring_indices/Gait Classification Using Mahalanobis–Taguchi System for Health Monitoring Systems Following Anterior Cruciate Ligament Reconstruction.pdf
- 분석 provider: antigravity

> **한국어 제목**: 전방 십자 인대 재건술 후 건강 모니터링 시스템을 위한 마할라노비스-다구치 시스템 기반 보행 분류

## 분류 태그

- ACL 연구: true
- IMU 사용: false
- 보행 데이터: true
- Score 제시: false

## 연구 목적

- 마할라노비스 거리(MD)를 사용하여 건강한 대조군과 전방십자인대 재건술(ACLR) 환자군의 보행 패턴을 분류하고, 다구치 방법(TM)을 적용해 유용한 시공간 보행 변수들을 선택하여 환자의 회복 과정 및 업무 복귀(RTW) 가능성을 객관적으로 평가하는 시스템을 제안하는 것이다. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “The objective was to use Mahalanobis distance (MD) to classify between the gait patterns of the control and ACLR groups, while the Taguchi Method (TM) was employed to choose the useful features. Moreover, MD was also utilised to ascertain whether the ACLR group approaching RTW. The combination of these two methods is called as Mahalanobis-Taguchi System (MTS).”

## 연구 설계와 대상

- 본 연구는 실험실 환경에서 15명의 건강한 남성 대조군(HG)과 10명의 일측성 일차성 전방십자인대 재건술(ACLR)을 받은 남성 환자군(PG)의 보행 패턴 데이터를 수집하여 비교 및 분석을 진행하였다. 두 집단 간의 나이, 체중, 키 분포는 통계적으로 유의미한 차이가 없었다(p > 0.05). _(근거: PAGE 1, Abstract / PAGE 6, Section 4.1. Participants)_
  - 근거 원문: “This study compared the gait of 15 control subjects to a group of 10 subjects with laboratory. ... Ten patients with a unilateral primary ACLR were chosen for the ACLR group (PG). This group was composed of 10 males. ... Meanwhile, 15 healthy subjects (HG) were selected for the healthy group with the condition that they had no previous history of lower extremity injuries, surgeries, or neuropathy. The HG consisted of 15 males. The height, age, and weight distribution of the two groups were not significantly (p > 0.05) (Table 1).”

## 방법

- 대상자들은 8m 보행로를 7회 걸었으며, 장비 및 환경에 대한 익숙해짐을 위해 처음 2회 왕복을 제외한 나머지 5회 왕복 중에서 대조군은 우측 다리, 환자군은 환측 다리를 대상으로 4회의 보행 주기를 분석하였다. _(근거: PAGE 6, Section 4.2. Procedures)_
  - 근거 원문: “Subjects were instructed to walk seven times on an 8 m walkway. The first two laps were not measured to allow for familiarization with the task and instrumentation. The last five laps were assessed to catch four gait cycles, employing the right limb from the HG and the injured limb from the PG.”
- 모션 캡처 카메라 5대와 지면반력 측정을 위한 힘판 2대를 사용하여 보행 데이터를 획득하였으며, Visual 3D Pro v6 프로그램을 활용하여 보행 패턴을 나타내는 11개의 시공간 변수들을 도출하였다. _(근거: PAGE 6, Section 4.2. Procedures / PAGE 7)_
  - 근거 원문: “Motion data were collected using five Oqus cameras of a motion capture system (Oqus, Qualisys AB, Gothenburg, Sweden). 36 reflective markers were placed on the joint landmarks and segments of the lower limb as denoted in Figure 2. ... Two force plates were reset for the next trial. The vertical ground reaction force was detected by two force plates with dimensions of 400 × 600 mm (Bertec, Worthington, Ohio, OH, USA) while the subject performed several walks at his own pace. ... The 11 spatiotemporal parameters (Table 2) of the trial were generated in Visual 3D Pro v6.”
- 마할라노비스-다구치 시스템(MTS) 방법론에 따라, 대조군 데이터를 기준으로 마할라노비스 공간(MS)을 구축(Step I)하고, 환자군 데이터를 이에 정규화하여 대조군 영역 바깥으로 이탈하는지를 검증(Step II)한 후, 직교배열표와 S/N 비의 Gain 분석을 통해 가장 유용한 특징 변수를 추출(Step III)하고 최종 회복도 진단에 활용(Step IV)하는 단계를 수행하였다. _(근거: PAGE 3, Section 2.2.1. Four Steps in MTS / PAGE 4)_
  - 근거 원문: “There are four steps in a MTS [40]: Step 0: Identification of assessment criteria and collection of patients’ spatiotemporal data Step I: Mahalanobis space (MS) creation ... Step II: Validation of MS ... Step III: Identification of useful features ... Step IV: Future diagnosis”

## 핵심 결과

- MTS 분석 결과 대조군(HG)의 마할라노비스 거리(MD)는 0.560에서 1.180 범위에 분포한 반면, ACLR 환자군(PG)의 MD는 2.308에서 1509.811 범위에 걸쳐 비정상 거리가 대조군에 비해 크게 벗어나 있음을 성공적으로 식별하였다. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “The results showed that gait deviations can be identified successfully, while the ACLR can be classified with higher precision by MTS. The MDs of the healthy group ranged from 0.560 to 1.180, while the MDs of the ACLR group ranged from 2.308 to 1509.811.”
- 민감도 및 특이도 분석을 수행한 결과, 건강한 대조군과 환자군을 100%의 민감도와 특이도로 구분할 수 있는 마할라노비스 거리(MD)의 최적 임계값(Threshold)은 1.5로 나타났다. _(근거: PAGE 10, Section 4.3. Results (Step II: Validation of MS))_
  - 근거 원문: “After considering sensitivity and specificity, we discovered the best threshold value was 1.5, which provided 100% sensitivity and specificity.”
- 11개의 변수 중 L12(2^11) 직교배열표 및 S/N 비의 Gain 분석 결과 음수의 영향력을 가진 Step Length(X1), Stride Length(X2), Stance Time (%GC)(X8)의 3가지 변수가 비정상 식별에 유의미하지 않은 요소로 판명되어 최종 8개의 유용한 변수들로 특징이 압축되었다. _(근거: PAGE 1, Abstract / PAGE 10, Section 4.3. Results (Step III: Identification of useful features))_
  - 근거 원문: “Out of the 11 spatiotemporal parameters analysed, only eight parameters were considered as useful features. ... As revealed in Tables 10 and 11, X1, X2, and X8 did not have a significant effect on the MD. Hence, the number of attributes was decreased from 11 to 8.”
- 최종 선정된 8가지 핵심 보행 변수는 Step Width, Swing Time (s), Stance Time (s), Double Support Time (s), Single Support Time (s), Double Support Time (%GC), Gait Speed (m/s), Stride Speed이다. _(근거: PAGE 12, Section 6.1. ACLR Diagnosis)_
  - 근거 원문: “In this study, eight important features, including step width, swing time (s), stance time (s), double support time (s), single support time (s), double support time (%GC), gait speed (m/s), and stride speed, were selected by utilising OAs and S/N ratios.”

## 저자 결론

- MTS 기법은 줄어든 개수의 유용한 보행 특징 변수들을 가지고도 ACLR 환자의 회복 진행 상황을 효과적으로 감지할 수 있으며, 진단 및 임상 평가 과정에서 주관성을 배제하고 객관적인 지표를 제공할 수 있다. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “These results indicate that MTS can effectively detect the ACLR recovery progress with reduced number of useful features. MTS enabled doctors or physiotherapists to provide a clinical assessment of their patients with more objective way.”
- 계산된 환자의 마할라노비스 거리(MD)의 크기는 비정상 보행 패턴의 심각도(deviation)를 대변하므로, 치료 경과에 따라 환자의 MD 값이 대조군 수준(임계값 1.5 이하)으로 작아질 때 업무에 완전히 복귀(RTW)할 준비가 되었음을 나타내는 강력한 예측 및 평가 도구로 사용될 수 있다. _(근거: PAGE 13, Section 6.2. Return to Work / PAGE 14, Conclusions)_
  - 근거 원문: “It is obvious that for lower MD values, the deviation from healthy participants is likely be less and the patients will have an excellent likelihood of RTW [31]. ... For individuals with greater MD values, constant rehabilitation should continue until the MD [23]. ... In addition, the size of the MD reflected the degree of abnormality. Hence, MDs can indicate the seriousness of an abnormality.”

## 연구의 한계

- 전방십자인대 재건술 후 환자들의 복귀를 가장 잘 준비시키기 위해서는 추후 신경근 조절 결손(deficits in neuromuscular control), 신경 운동 상태(neuromotor status) 및 심리적 준비 상태(psychological readiness) 등 개별적인 결손 요인을 명확히 표적으로 삼는 추가적인 연구가 수행되어야 한다. _(근거: PAGE 14, Conclusions)_
  - 근거 원문: “Further research is essential to optimally target deficits in neuromuscular control, neuromotor status, and psychological readiness in order to best prepare individuals for RTW following ACLR.”

## 생각해볼 내용

- 비교 대상이 대조군 15명과 환자군 10명으로 전체 표본의 수가 매우 적어, MTS 모델 자체의 정확도나 임계값(Threshold) 1.5의 신뢰성 및 보편성을 일반화하기에는 한계가 있다. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “This study compared the gait of 15 control subjects to a group of 10 subjects with laboratory.”
- 우측 하지 지배성(right leg dominance)을 이유로 환자군의 환측 다리를 오직 대조군의 우측 다리와만 비교하였는데, 이는 보행 시 대칭성이나 개인 간 생체역학적 차이를 온전히 보정하지 못할 우려가 있다. _(근거: PAGE 6, Section 4.2. Procedures)_
  - 근거 원문: “We compared the ACLR to the right limbs only because most individuals are right leg dominant and we wanted to compare those limbs to a right control limb that is more likely to exhibit ideal and stable knee biomechanics [44,45].”

## 이 연구가 지적한 선행연구의 문제점

- 기존의 스포츠나 신체 활동 복귀 판단 조건들은 주관적이며, 동반 질환 또는 재부상 등의 잠재적 위험 식별 능력이 결여되어 있어 잔류 결손을 정량화하여 평가할 수 있는 객관적인 방법이 필요하다. _(근거: PAGE 2, Section 2.1. ACLR)_
  - 근거 원문: “Nevertheless, these conditions are subjective and do not include the identification of possible dangers such as potential comorbidities or re-injuries; therefore, methods that quantify residual deficits are likely to be preferable [34].”
- 전방십자인대 재건술 환자의 재활 성과와 밀접한 관련이 있는 업무 복귀(RTW) 과정을 정량적이고 체계적인 분석적 기법을 통해 평가하고 예측하고자 한 연구는 기존 문헌에서 찾을 수 없었다. _(근거: PAGE 2, Section 1. Introduction)_
  - 근거 원문: “In our review, we found no significant exiting research that focussed on patients’ return to work (RTW) after ACLR using an analytical strategy.”

## 이 연구의 해결 방식과 기여

- 본 연구는 시공간 보행 매개변수를 활용하여 전방십자인대 재건술 후 환자들의 회복 양상과 복귀(RTW) 가능성을 분석적이고 연속적인 척도로 수치화할 수 있는 마할라노비스-다구치 시스템(MTS)을 도입하여 보행 편차를 정량 진단하는 객관적인 의사결정 보조 시스템의 유효성을 실증하였다. _(근거: PAGE 2, Section 1. Introduction)_
  - 근거 원문: “This study aims to method of ACLR analysis that can predict RTW according to spatiotemporal data. Our proposed method uses the MD to differentiate the routine of ACLR; based on signal-to-noise ratios (S/N ratios) and orthogonal arrays (OAs), useful features can be identified [19,21,22,24].”

## 레퍼런스할 수 있는 내용

### 1. ACLR 환자의 보행 중 무릎 각도 편차 지속 현상

- 원문 발췌: “It has been proven that ACLR subjects have an inclination to pose altered knee angles during gait, even 12 months after the operation [36].”
- 한국어 번역: ACLR 환자들은 수술 후 12개월이 지난 시점에도 보행 중에 비정상적인 무릎 각도를 취하는 경향이 있음이 입증되었다.
- 원문 위치: PAGE 2, Section 2.1. ACLR
- 원문 내 인용표기: [36]
- 해당 선행문헌: Culvenor, A.G.; Perraton, L.; Guermazi, A.; Bryant, A.L.; Whitehead, T.S.; Morris, H.G.; Crossley, K.M. Knee kinematics and kinetics are associated with early patellofemoral osteoarthritis following anterior cruciate ligament reconstruction. Osteoarthr. Cartil. 2016, 24, 1548–1553. [CrossRef] [PubMed]
- 주장 유형: background_citation
- 활용 맥락과 주의: 전방십자인대 재건술을 거치더라도 약 1년 경과 시까지 기구학적 무릎 정렬 또는 각도의 이상 보행 패턴이 완전히 정상화되지 않고 비정상적으로 유지될 수 있음을 지지하는 근거로 사용 가능하다.

### 2. 마할라노비스 거리(MD)를 이용한 대조군과 ACLR 환자의 보행 패턴 범위 구분

- 원문 발췌: “The MDs of the healthy group ranged from 0.560 to 1.180, while the MDs of the ACLR group ranged from 2.308 to 1509.811.”
- 한국어 번역: 건강한 대조군의 마할라노비스 거리(MD)는 0.560에서 1.180 범위에 분포했던 반면, ACLR 환자군의 마할라노비스 거리는 2.308에서 1509.811 범위였다.
- 원문 위치: PAGE 1, Abstract
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: 본 연구의 핵심 정량적 결과로, 대조군의 기준 데이터 분포와 환자군의 보행 비정상성 편차가 마할라노비스 거리 척도를 통해 통계적으로 뚜렷하고 큰 폭의 범위 차이로 분류될 수 있음을 증명하는 직접적 수치이다.

### 3. 재활 후 장기적인 하지 운동 루틴 적응 현상

- 원문 발췌: “Biomechanical studies have demonstrated that after two years of rehabilitation injured people tend to adapt their lower limb motion routine for many tasks [26,27].”
- 한국어 번역: 생체역학 연구들은 부상을 입은 사람들이 2년의 재활을 거친 후에도 여러 동작에서 자신들의 하지 운동 루틴을 적응시켜 나가는(대체 동작을 형성하는) 경향이 있음을 보여주었다.
- 원문 위치: PAGE 2, Section 2.1. ACLR
- 원문 내 인용표기: [26,27]
- 해당 선행문헌: 26. White, K.; Logerstedt, D.; Snyder-Mackler, L. Gait Asymmetries Persist 1 Year After Anterior Cruciate Ligament Reconstruction. Orthop. J. Sport. Med. 2013, 1, 1–6. [CrossRef] [PubMed]
27. Sigward, S.M.; Lin, P.; Pratt, K. Knee loading asymmetries during gait and running in early rehabilitation following anterior cruciate ligament reconstruction: A longitudinal study. Clin. Biomech. 2016, 32, 249–254. [CrossRef] [PubMed]
- 주장 유형: background_citation
- 활용 맥락과 주의: 장기 재활(2년) 후에도 보행 비대칭이나 정상과 다른 보상적 하지 운동 패턴이 잔존하여 고착화될 수 있음을 설명할 때 2차 인용으로 제시할 수 있다.
