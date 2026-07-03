# Unsupervised Detection of Pathological Gait Patterns via Instantaneous Center of Rotation Analysis

Molina Arias, L., & Smoleń, M. (2026). Unsupervised Detection of Pathological Gait Patterns via Instantaneous Center of Rotation Analysis. Applied Sciences, 16(8), 3976. https://doi.org/10.3390/app16083976

## 서지정보

- 저자: Ludwin Molina Arias, Magdalena Smoleń
- 연도: 2026
- 저널: Applied Sciences
- DOI: 10.3390/app16083976
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/04_machine_learning_and_deep_learning/Unsupervised Detection of Pathological Gait Patterns via Instantaneous Center of Rotation Analysis.pdf
- 분석 provider: antigravity

> **한국어 제목**: 순간 회전 중심 분석을 통한 병리적 보행 패턴의 비지도 감지

## 분류 태그

- ACL 연구: false
- IMU 사용: false
- 보행 데이터: true
- Score 제시: false

## 연구 목적

- 순간 회전 중심(ICR) 궤적을 이용하여 병리적 보행 패턴을 감지하는 비지도 학습 프레임워크를 제안하는 것. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “This study introduces a novel unsupervised framework, ICR-LLS, for detecting pathological gait patterns using instantaneous center of rotation (ICR) trajectories of the shank in the sagittal plane.”
- 비지도 학습 방식을 사용하여 레이블 데이터에 의존하지 않고 파킨슨병(PD) 환자의 미세한 운동 조절 장애의 감지 정확도를 개선하는 대안을 제시함. _(근거: PAGE 3, Section 1)_
  - 근거 원문: “By leveraging this underutilized biomechanical descriptor, the proposed ICR-LLS methodology offers a novel, data-driven framework to improve the detection of subtle motor control impairments in PD through an unsupervised learning approach without relying on labeled data.”

## 연구 설계와 대상

- 비교 분석을 위해 건강한 대조군 30명과 약물 복용을 중단(OFF)한 상태의 파킨슨병(PD) 환자 10명의 데이터를 기존 공개 데이터셋에서 추출함. _(근거: PAGE 8, Section 3.2)_
  - 근거 원문: “To allow a comparative analysis between groups, data from 30 neurologically intact healthy participants and 10 individuals diagnosed with PD in the OFF medication state were selected from the original datasets.”
- 두 그룹 간의 교란 요인을 최소화하기 위해 특히 연령 변수의 인구통계학적 분포가 유사하도록 선정 기준을 설정함. _(근거: PAGE 8, Section 3.2)_
  - 근거 원문: “The primary inclusion criterion was to ensure comparable demographic distributions between groups, particularly in terms of age, to minimize possible confounding effects.”

## 방법

- 양측 가쪽 넙다리 관절융기와 가쪽 복사 뼈 부위에 부착된 마커의 2D 키네마틱 데이터를 이용해 시상면상에서 섕크(종아리)의 ICR 궤적을 산출함. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “ICR trajectories were computed from two-dimensional kinematic data captured at the lateral femoral epicondyle and lateral malleolus for both shanks, producing four-dimensional multivariate time series for each gait trial.”
- 보행 주기 간의 시간적 정렬 어긋남을 최소화하기 위해 교차 상관 분석을 적용하여 시간 정렬을 우선 수행함. _(근거: PAGE 5, Section 2.2)_
  - 근거 원문: “To account for temporal misalignments arising from stride-to-stride variability, each pair of trajectories is first circularly aligned using cross-correlation.”
- 시상면에서의 기하학적 형태적 구조를 보존하면서 비선형 시간 변형을 다루기 위해 동적 시간 워핑(DTW)을 사용하여 비유사도를 계산함. _(근거: PAGE 5, Section 2.2)_
  - 근거 원문: “Following circular time alignment, trajectory dissimilarity is quantified using Dynamic Time Warping (DTW), which accommodates non-linear temporal deformations while preserving the spatial structure of the trajectories.”

## 핵심 결과

- 제안한 ICR-LLS 방법론을 데이터셋에 평가한 결과, NMI 0.449 및 군집 분리-조밀도 비율(SCR) 6.754를 달성하여 의미 있는 군집 구조가 관찰됨. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “The framework is evaluated on a public dataset comprising individuals with Parkinson’s disease (PD) and healthy controls, achieving a normalized mutual information (NMI) of 0.449 and a Separation-to-Compactness Ratio (SCR) of 6.754, indicating a meaningful cluster structure.”
- 비지도 학습 군집화의 성능을 분류 관점에서 매핑하여 분석하였을 때 정확도 90%, 민감도 70%, 특이도 96.7%의 우수한 결과를 보임. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “In addition, classification-oriented metrics yield an accuracy of 90%, sensitivity of 70%, and specificity of 96.7%, supporting the method’s effectiveness in distinguishing pathological gait.”
- 기존 보행 피처 기반 군집화 모델(NMI 0.101, SCR 3.014, 민감도 40%)에 비해 제안된 ICR-LLS가 더 강인한 탐지 성능과 뚜렷한 군집 분리력을 보여줌. _(근거: PAGE 18, Section 3.4.2)_
  - 근거 원문: “The baseline based on conventional gait features consistently underperforms the proposed ICR-LLS approach in all configurations.”

## 저자 결론

- 단순한 2D 측정 마커 정보와 비지도 학습의 결합을 통해, 임상 라벨 없이도 보행의 변동성을 해석적으로 탐색할 수 있는 비 biomechanical 지표로서 ICR 궤적의 유효성을 확인함. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “By combining minimal 2D kinematic inputs with unsupervised learning, ICR-LLS provides an interpretable framework for the exploratory analysis of gait variability, and although further validation is required, the findings suggest that ICR trajectories may serve as a meaningful biomechanical descriptor for characterizing pathological locomotion.”
- 비지도 학습 모델에 의해 분할된 정상 집단과 이상치(outliers)의 구분이 실제 파킨슨병 환자의 기하급수적 운동 임상 점수(UPDRS 및 H&Y 스테이지) 차이와 통계적으로 부합함. _(근거: PAGE 20, Section 4)_
  - 근거 원문: “These findings demonstrate that, although derived from an unsupervised anomaly detection framework, the partition induced by ICR-based clustering aligns with clinically significant differences in the severity of motor symptoms.”

## 연구의 한계

- 섕크의 2D 시상면상 키네마틱 정보만을 다루기 때문에 다른 평면에서의 변형이나 보상 작용에 따른 보행 특성을 온전히 반영하기 어려움. _(근거: PAGE 21, Section 4)_
  - 근거 원문: “The exclusive focus on 2D shank kinematics may not fully capture multi-planar or compensatory movements, potentially limiting sensitivity for certain gait phenotypes.”
- 현재 검증 방식은 통제된 대조군과 파킨슨병 그룹에 한정되어 있으며 뇌졸중이나 다발성 경화증과 같이 이질성이 큰 다양한 질환군으로 일반화하는 것에는 한계가 존재함. _(근거: PAGE 21, Section 4)_
  - 근거 원문: “In addition, current validation is restricted to a controlled scenario involving healthy controls and individuals with PD. Extending the framework to other neurological conditions, such as stroke or multiple sclerosis, introduces additional complexity, as these populations exhibit high intra-class variability and may not conform to a single-cluster-plus-outliers structure.”

## 생각해볼 내용

- 군집 내 응집 및 분리도(SCR)를 높이기 위해 극단적으로 파라미터를 튜닝할 시 오히려 실제 파킨슨병 진단 분류 정확성(NMI 및 민감도)이 낮아지는 trade-off가 발생함을 고려해야 함. _(근거: PAGE 12, Section 3.3.4)_
  - 근거 원문: “Each configuration was evaluated using NMI and SCR, excluding trivial solutions that consist of a single cluster with no separation. Figure 7 shows the resulting performance grid and reveals a clear inverse relationship between NMI and SCR: parameter settings that maximize agreement with the ground truth do not necessarily yield the most compact and well-separated clusters.”

## 이 연구가 지적한 선행연구의 문제점

- 기존에 제안된 지도학습 기반 기계학습 모델은 수집된 특정 훈련 데이터셋에 크게 의존하여 새로운 환경이나 경계선 상의 상태 변화를 일반화하기 어려움. _(근거: PAGE 3, Section 1)_
  - 근거 원문: “While supervised models are powerful, they rely on labeled data and may not generalize well to new or ambiguous cases.”
- 순간 회전 중심(ICR) 개념은 척추나 슬관절 등의 부분 운동성 연구에는 활용되었으나, 동적 보행 전반에 걸친 하지 분절 회전 분석 도구로서는 활용이 부족했음. _(근거: PAGE 3, Section 1)_
  - 근거 원문: “However, its application to gait analysis, especially for capturing continuous segmental rotations during locomotion, has been largely neglected. Prior studies have primarily focused on isolated joint mechanics or specific motor tasks without addressing the dynamic evolution of the ICR throughout the gait cycle.”

## 이 연구의 해결 방식과 기여

- 보행 분석에서 하부 운동성의 섕크 ICR 데이터가 정상과 병리적 상태를 효과적으로 대변한다는 바이오메카닉스 해석 기반의 비지도 탐지 기술을 제안함. _(근거: PAGE 3, Section 1)_
  - 근거 원문: “By leveraging this underutilized biomechanical descriptor, the proposed ICR-LLS methodology offers a novel, data-driven framework to improve the detection of subtle motor control impairments in PD through an unsupervised learning approach without relying on labeled data.”
- 바닥에 매립된 고가의 지면반력기(GRF)나 대형 3D 장비 없이 단순 카메라 형태의 2D 입력 데이터를 통해서도 강인하게 질환 특징을 파악할 수 있는 인프라 최소화에 기여함. _(근거: PAGE 21, Section 4)_
  - 근거 원문: “Unlike GRF-based methods that require force plates or 3D motion capture systems, the proposed framework relies on a limited number of sagittal-plane landmarks, allowing compatibility with simplified acquisition systems such as monocular video setups. This facilitates deployment in clinical and outpatient settings and supports longitudinal monitoring.”

## 레퍼런스할 수 있는 내용

### 1. 파킨슨병 환자의 낙상 비율과 사망률

- 원문 발췌: “It is estimated that approximately 70% of PD patients experience at least one fall annually, and 39% suffer recurrent falls. Importantly, recurrent falls are associated with a median survival of just six years [5].”
- 한국어 번역: PD 환자의 약 70%가 매년 최소 한 번의 낙상을 경험하고, 39%는 재발성 낙상을 겪는 것으로 추정된다. 중요한 것은, 재발성 낙상이 단 6년의 생존 기간 중앙값과 관련이 있다는 점이다.
- 원문 위치: PAGE 2, Section 1
- 원문 내 인용표기: [5]
- 해당 선행문헌: 5. Allen, N.E.; Schwarzel, A.K.; Canning, C.G. Recurrent Falls in Parkinson’s Disease: A Systematic Review. Park. Dis. 2013, 2013, 906274. [CrossRef]
- 주장 유형: background_citation
- 활용 맥락과 주의: 파킨슨병 환자의 운동장애가 야기하는 심각한 낙상 문제와 재발 시의 사망 위험을 임상적 근거로 제시할 때 인용할 수 있음.

### 2. ICR-LLS 방법론과 기존 보행 특징 기반 방식의 성능 비교

- 원문 발췌: “The baseline based on conventional gait features consistently underperforms the proposed ICR-LLS approach in all configurations. The Max NMI solution achieves moderate agreement with clinical labels (NMI = 0.367) and high specificity (100.0%), but limited sensitivity (40.0%), indicating a bias toward correctly identifying healthy controls while not detecting a substantial portion of pathological cases.”
- 한국어 번역: 전형적인 보행 특징을 기반으로 한 기준 모델은 모든 설정에서 제안된 ICR-LLS 접근법보다 일관되게 낮은 성능을 보였다. 최대 NMI 솔루션은 임상 레이블과 보통 수준의 일치도(NMI = 0.367) 및 높은 특이도(100.0%)를 달성했으나 제한된 민감도(40.0%)를 보여, 건강한 대조군은 정확히 식별하는 반면 병리적 증례의 상당 부분을 감지하지 못하는 편향을 나타냈다.
- 원문 위치: PAGE 18, Section 3.4.2
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: 비지도 학습 환경에서 Joint flexion/extension 등의 일반 각도 특징보다 섕크의 2D 순간 회전 중심 궤적이 정상군과 이상군 분류 성능을 개선하는 핵심적인 증거로 사용될 수 있음.

### 3. 파킨슨병 환자의 보행 시 순간 회전 중심(ICR) 궤적 변화 특징

- 원문 발췌: “Although the overall trajectory patterns are broadly similar between the groups, PD subjects exhibit more pronounced asymptotic patterns and greater variability, especially along the vertical axis. These observations suggest a reduced rotational excursion of the shank in PD, reflecting a more rigid lower-limb movement compared to healthy controls.”
- 한국어 번역: 비록 전반적인 궤적 패턴은 그룹 간에 대체로 유사하지만, PD 피험자들은 특히 수직 축을 따라 더 뚜렷한 점근적(asymptotic) 패턴과 더 큰 변동성을 보인다. 이러한 관찰 결과는 PD에서 섕크의 회전 운동 범위가 감소했음을 시사하며, 이는 건강한 대조군에 비해 더 경직된 하지 움직임을 반영한다.
- 원문 위치: PAGE 11, Section 3.3.1
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: uncited_author_claim
- 활용 맥락과 주의: 파킨슨병 환자들의 하지 경직(rigidity) 특성이 ICR 분석 하에 시상면의 종아리 회전 제한 및 점근선 형태로 발현됨을 설명하는 저자들의 독자적 주장으로 사용 가능함.
