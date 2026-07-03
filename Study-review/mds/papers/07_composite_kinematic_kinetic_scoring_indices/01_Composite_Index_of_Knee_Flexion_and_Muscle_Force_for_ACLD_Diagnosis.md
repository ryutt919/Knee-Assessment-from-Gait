# Leveraging Multivariable Linear Regression Analysis to Identify Patients with Anterior Cruciate Ligament Deficiency Using a Composite Index of the Knee Flexion and Muscle Force

> Li H, Huang H, Ren S, Rong Q. (2023). *Bioengineering (Basel)*, 10(3), 284. DOI: 10.3390/bioengineering10030284. PMID: 36978675.
> 검증 URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC10045096/

#### 연구 목적

- 조깅 시 무릎 굴곡 kinematics와 하지 근력(muscle force, kinetics 기반)을 결합한 단일 복합 지수(composite index)로 ACL 결손(ACLD) 환자를 진단하고자 함. 임상의의 주관적 판단에 의존하는 기존 진단 방식을 재현 가능한 정량적 도구로 대체하는 것이 목표.
- 대상: ACLD 환자 25명 vs 건강 대조군 9명.

#### 결과

- 최종 모델(원본 feature 3개 + 복합 지수 feature 3개)이 진단 정확도 81.4%, precision 87.0%, recall 80.0%, specificity 83.3%, F1-score 83.3%, R²=0.542를 달성.
- Composite Index 1의 회귀계수 = -2.3055 (p<0.001)로 가장 유의한 예측 변수였으며, Composite Index 2 = -1.5697 (p=0.006).

#### 방법

- Vicon 8-카메라 모션캡처(kinematics) + AMTI force plate(1000Hz, kinetics) 동시 계측, 조깅 조건.
- AnyBody 근골격계 모델링(inverse dynamics)으로 무릎 관련 근육 13개의 근력을 체중/최대근력 대비로 정규화.
- Gait cycle 전 구간에서 t-검정(p<0.05)으로 특징점(characteristic point)을 선별한 뒤, (피험자 × 특징점) 행렬을 PCA로 축소(분산 90% 기준, 8개 성분).
- PCA로 축소된 성분을 다중선형회귀 분류기(Y = β0 + β1X1 + ... + βqXq, ACLD=1, Control=-1)에 투입, 5-fold cross-validation으로 검증.

#### 레퍼런스할 수 있는 내용 (원문 발췌, 해당 부분 원문에서 레퍼런스한 논문 참고논문 표기 해야함 )

- "This study was performed to identify patients with ACLD using multivariable linear regression through a composite index that combined kinematics and muscle forces." (Introduction)
- "However, few studies have combined kinematics and muscle forces to extract features." (Introduction)
- "The composite index and characteristic points can help avoid complex subjective diagnosis in clinical practice." (Conclusions)
- "The feature choice is the most important variable, regardless of the statistical method." (Introduction, Reinbolt et al. 인용)

#### 생각해볼만한 내용

- kinematics와 kinetics(근력)를 함께 PCA로 묶어 "복합 지수"로 표현했다는 점에서 우리 프로젝트의 Gait Normality Score(HA-referenced, kinematics/kinetics 도메인 subscore)와 문제의식이 유사함. 다만 이 연구는 ACLD vs 건강군의 이분류 "진단"이 목적이고, 회복도(recovery)나 종적(longitudinal) 추적을 다루지 않음 — 우리 프로젝트가 채우는 공백(회복 궤적 추적)이 명확히 드러나는 지점.
- 표본 25명(ACLD) vs 9명(대조군)으로 소규모이며 독립 재현 검증이 없음. 정확도 81.4% 자체는 GDI/GPS 계열 지수의 성숙도에 비해 아직 초기 단계임을 시사.

#### 이 연구에서 지적하는 선행연구들의 문제점

- "Only a few studies have used kinematics and dynamics data to diagnose patients with ACLD. There are even fewer studies that can be directly reproduced and can rapidly diagnose patients."
- "Clinical diagnosis of ACLD is complicated and expensive, and the diagnosis process requires the subjective judgment of clinicians."

#### 이 연구에서 선행연구를 해결하는 방식

- kinematics(무릎 굴곡)와 kinetics(근력)를 하나의 PCA 기반 복합 지수로 통합하고, 이를 단순하고 재현 가능한 선형회귀 분류기에 투입함으로써 기존에 두 도메인을 따로 다루던 연구들과 차별화.

#### Limitations (저자 명시)

- 일부 ACLD 환자가 반월판 손상을 동반하여 하위군 분석이 추가로 필요.
- 부상 후 경과 기간이 6개월~4년으로 분산되어 데이터 동질성에 영향.
- EMG 검증 없음.
- 조깅 조건만 검증되었고 보행(walking)에 대한 추가 검증 필요.
- 표본이 작고 대조군이 남성으로 편중됨.
