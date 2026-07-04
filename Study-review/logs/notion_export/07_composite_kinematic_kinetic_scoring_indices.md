# [46] Leveraging Multivariable Linear Regression Analysis to Identify Patients with Anterior Cruciate Ligament Deficiency Using a Composite Index of the Knee Flexion and Muscle Force

(저자: Li H, Huang H, Ren S, Rong Q | 연도: 2023 | 저널: Bioengineering (Basel) | DOI: https://doi.org/10.3390/bioengineering10030284)

> Li H, Huang H, Ren S, Rong Q. (2023). *Bioengineering (Basel)*, 10(3), 284. DOI: 10.3390/bioengineering10030284. PMID: 36978675. 검증 URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC10045096/

#### 연구 목적

- 조깅 시 무릎 굴곡 kinematics와 하지 근력(muscle force, kinetics 기반)을 결합한 단일 복합 지수(composite index)로 ACL 결손(ACLD) 환자를 진단하고자 함. 임상의의 주관적 판단에 의존하는 기존 진단 방식을 재현 가능한 정량적 도구로 대체하는 것이 목표.
- 대상: ACLD 환자 25명 vs 건강 대조군 9명.

#### 결과

- 최종 모델(원본 feature 3개 + 복합 지수 feature 3개)이 진단 정확도 81.4%, precision 87.0%, recall 80.0%, specificity 83.3%, F1-score 83.3%, R²=0.542를 달성.
- Composite Index 1의 회귀계수 = -2.3055 (p\<0.001)로 가장 유의한 예측 변수였으며, Composite Index 2 = -1.5697 (p=0.006).

#### 방법

- Vicon 8-카메라 모션캡처(kinematics) + AMTI force plate(1000Hz, kinetics) 동시 계측, 조깅 조건.
- AnyBody 근골격계 모델링(inverse dynamics)으로 무릎 관련 근육 13개의 근력을 체중/최대근력 대비로 정규화.
- Gait cycle 전 구간에서 t-검정(p\<0.05)으로 특징점(characteristic point)을 선별한 뒤, (피험자 × 특징점) 행렬을 PCA로 축소(분산 90% 기준, 8개 성분).
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
- 부상 후 경과 기간이 6개월\~4년으로 분산되어 데이터 동질성에 영향.
- EMG 검증 없음.
- 조깅 조건만 검증되었고 보행(walking)에 대한 추가 검증 필요.
- 표본이 작고 대조군이 남성으로 편중됨.


---

# [47] Leveraging explainable machine learning to identify gait biomechanical parameters associated with anterior cruciate ligament injury

(저자: Kokkotis C, Moustakidis S, Tsatalas T, Ntakolia C, Chalatsis G, Konstadakos S, Hantes ME, Giakas G, Tsaopoulos D | 연도: 2022 | 저널: Scientific Reports | DOI: https://doi.org/10.1038/s41598-022-10666-2)

> Kokkotis C, Moustakidis S, Tsatalas T, Ntakolia C, Chalatsis G, Konstadakos S, Hantes ME, Giakas G, Tsaopoulos D. (2022). *Scientific Reports*, 12, 6647. DOI: 10.1038/s41598-022-10666-2. PMID: 35459787. 검증 URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC9026057/ (주의: 명명된 단일 "index"를 제시하는 논문은 아니며, SHAP 기반 feature importance를 사실상 연속적 기여도 점수로 사용하는 borderline 사례로 포함함)

#### 연구 목적

- ACLD(ACL 결손), ACLR(ACL 재건, 수술 후 6개월 이상), 건강 대조군(CON) 세 그룹을 구분하는 설명 가능한(explainable) ML 방법론을 구축하고, SHAP 값으로 각 gait kinematic/kinetic 변수의 기여도를 정량화. 본 프로젝트의 ACLD/ACLR/HA 3-way 분류 파이프라인과 구조적으로 직접적인 유사성을 가짐.

#### 결과

- 151명(ACLD 44명, ACLR 54명, CON 53명) 대상. SVM이 최고 성능: 21개 선별 feature로 정확도 94.95%, Neural Network가 92.89%로 2위.
- SHAP importance \> 0.3인 8개 변수(K2, H4, A3, GRF4, GRF7, K1, A4, GRF6) 확인.
- 상위 3개 ACLD 판별 변수 중 2개(H4, GRF3)는 ACLR 수술 후 CON과 유의한 차이가 사라져, 모델을 통해 부분적 생체역학적 회복이 감지 가능함을 시사.

#### 방법

- Vicon 10-카메라(100Hz, kinematics) + Bertec force platform(1000Hz, kinetics) 계측.
- Sagittal plane gait 변수 25개(GRF, 고관절/무릎/발목 각도 및 모멘트)를 그룹별 155/204/298 trial에서 추출.
- 8개 ML 분류기(SVM, RF, XGBoost, NN, KNN, Logistic Regression, Decision Trees, Naïve Bayes) 비교, SHAP 값으로 feature 기여도 순위화.

#### 레퍼런스할 수 있는 내용 (원문 발췌, 해당 부분 원문에서 레퍼런스한 논문 참고논문 표기 해야함 )

- "Support Vector Machines were proved to be the best performing model (accuracy of 94.95%) on a group of 21 selected biomechanical parameters." (Abstract)
- "Features, that would have been neglected by the traditional statistical analysis, were identified as contributing parameters." (Abstract)
- "H1, H2, GRF6 and GRF5...were identified as important by SHAP whereas their distributions had no significant differences between CON and ACLD." (Discussion)

#### 생각해볼만한 내용

- 본 연구는 모집단 구성(ACLD/ACLR/CON 3-way)과 계측 방식(3D motion capture + force plate)이 우리 프로젝트와 매우 유사한 가장 근접한 선행연구. 다만 "점수"가 아니라 분류 정확도 + SHAP feature importance로 결과를 제시하며, 임상적으로 해석 가능한 단일 지표(HA 대비 총점/도메인 subscore)로 패키징하지는 않음.
- ACLR 수술 후 특정 변수가 CON 수준으로 정상화되는 현상을 포착한 점은, 우리 GNS(Gait Normality Score)가 회복도를 추적하는 것과 같은 방향성을 갖지만, 이 연구는 종적(longitudinal) 단일 코호트 추적이 아니라 그룹 간 횡단(cross-sectional) 비교라는 차이가 있음.

#### 이 연구에서 지적하는 선행연구들의 문제점

- "\[Prior ML classifiers\] are treated as black boxes. The lack of transparency and explainability of the models result to poor understanding of their inner workings."

#### 이 연구에서 선행연구를 해결하는 방식

- Explainable AI(SHAP)를 전통적 통계 분석과 결합하여, ANOVA 등 전통적 통계로는 놓쳤던 변수(H1, H2, GRF5/6)까지 포착 — 정확도와 임상적 해석 가능성을 동시에 확보.

#### Limitations (저자 명시)

- "The clinical significance...should be considered with caution. This can be attributed to the fact that even though gait biomechanics are altered following ACLR, few biomechanical parameters demonstrate consistent results across studies."
- 수술 기법, 재활 프로토콜, 개인별 보상 전략, 성별 차이가 통제되지 않음.
- 데이터셋 비공개.


---

# [48] Use of the normalcy index for the assessment of abnormal gait in the anterior cruciate ligament deficiency combined with meniscus injury

(저자: Liu X, Huang H, Ren S, Rong Q, Ao Y | 연도: 2020 | 저널: Computational Methods in Biomechanics and Biomedical Engineering | DOI: https://doi.org/10.1080/10255842.2020.1789119)

> Liu X, Huang H, Ren S, Rong Q, Ao Y. (2020). *Computational Methods in Biomechanics and Biomedical Engineering*, 23(14), 1102-1108. DOI: 10.1080/10255842.2020.1789119. PMID: 32648770. 검증 URL: https://pubmed.ncbi.nlm.nih.gov/32648770/ (주의: 출판연도 2020년으로 "최근 5년(2021-2026)" 기준에서는 벗어나지만, 위 01번 문헌과 동일 연구그룹의 직접 선행연구이자 GDI/GPS 계열 단일 지수를 ACL 손상군에 적용한 사례라서 배경 문헌으로 포함)

#### 연구 목적

- Normalcy Index(NI, GDI/GPS 계열의 gait waveform 편차 기반 단일 복합 점수)가 ACL 파열 + 반월판 손상 동반 환자의 보행 이상 정도를 정량화할 수 있는지, 그리고 손상 중증도와 함께 점수가 변화하는지 검증.
- 대상: 반월판 손상 정도가 다양한 ACL+반월판 손상 환자 25명 vs 건강 대조군 12명.

#### 결과

- 환자군 NI가 대조군 대비 유의하게 높음(악화, P\<0.05).
- 반월판 손상 중증도가 커질수록 NI가 단조 증가(Jonkheere-Terpstra test, P\<0.001) — 즉 복합 점수가 손상 중증도 gradient를 그대로 반영.

#### 방법

- NI는 gait waveform이 정상 참조 범위(reference range)로부터 벗어난 정도를 종합해 산출하는 단일 지수(GDI/GPS와 동일 계열의 접근).
- 반월판 손상 중증도별 하위군으로 나누어 NI 추세를 비모수 순서 검정(Jonkheere-Terpstra test)으로 분석.

#### 레퍼런스할 수 있는 내용 (원문 발췌, 해당 부분 원문에서 레퍼런스한 논문 참고논문 표기 해야함 )

- "a concise yet effective tool" (Abstract/Conclusion, 병합 손상 평가 도구로서의 NI를 지칭)

#### 생각해볼만한 내용

- 우리 GNS와 가장 개념적으로 가까운 선행연구 — GDI/GPS 스타일 waveform 편차 단일 점수를 ACL 관련 손상군에 실제 적용한 드문 사례. 다만 (1) 발표 연도가 5년 기준을 벗어나고, (2) HA-referenced가 아니라 자체 정상 참조 범위 기반이며, (3) ACLD/ACLR을 구분하지 않고 반월판 동반손상 중증도만 다루어, 우리 프로젝트의 ACLD/ACLR/HA 구분 및 회복 추적과는 목적이 다름.
- 이 연구가 2020년에 이미 "단조 증가하는 단일 손상중증도 점수"를 검증했다는 사실은, 우리 GNS 설계가 이 방향의 후속 연구로서 타당성이 있음을 뒷받침하는 근거로 인용 가능.

#### 이 연구에서 지적하는 선행연구들의 문제점

- ACL 손상과 반월판 손상을 각각 별도로 평가하며, 통합된 기능적/생체역학적 중증도 지표가 부재했다는 문제의식(Abstract/Conclusion 취지 기반).

#### 이 연구에서 선행연구를 해결하는 방식

- 단일 Normalcy Index로 병합 손상(ACL+반월판)의 중증도를 하나의 점수로 통합해, 수술 전후 모니터링에 활용 가능한 "concise yet effective tool"을 제시.

#### Limitations (저자 명시)

- Abstract 상세 미기재. 가장 작은 중증도 하위군은 5명에 불과해 통계적 검정력이 제한적일 수 있음.


---

