# Leveraging explainable machine learning to identify gait biomechanical parameters associated with anterior cruciate ligament injury

> Kokkotis C, Moustakidis S, Tsatalas T, Ntakolia C, Chalatsis G, Konstadakos S, Hantes ME, Giakas G, Tsaopoulos D. (2022). *Scientific Reports*, 12, 6647. DOI: 10.1038/s41598-022-10666-2. PMID: 35459787.
> 검증 URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC9026057/
> (주의: 명명된 단일 "index"를 제시하는 논문은 아니며, SHAP 기반 feature importance를 사실상 연속적 기여도 점수로 사용하는 borderline 사례로 포함함)

#### 연구 목적

- ACLD(ACL 결손), ACLR(ACL 재건, 수술 후 6개월 이상), 건강 대조군(CON) 세 그룹을 구분하는 설명 가능한(explainable) ML 방법론을 구축하고, SHAP 값으로 각 gait kinematic/kinetic 변수의 기여도를 정량화. 본 프로젝트의 ACLD/ACLR/HA 3-way 분류 파이프라인과 구조적으로 직접적인 유사성을 가짐.

#### 결과

- 151명(ACLD 44명, ACLR 54명, CON 53명) 대상. SVM이 최고 성능: 21개 선별 feature로 정확도 94.95%, Neural Network가 92.89%로 2위.
- SHAP importance > 0.3인 8개 변수(K2, H4, A3, GRF4, GRF7, K1, A4, GRF6) 확인.
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

- "[Prior ML classifiers] are treated as black boxes. The lack of transparency and explainability of the models result to poor understanding of their inner workings."

#### 이 연구에서 선행연구를 해결하는 방식

- Explainable AI(SHAP)를 전통적 통계 분석과 결합하여, ANOVA 등 전통적 통계로는 놓쳤던 변수(H1, H2, GRF5/6)까지 포착 — 정확도와 임상적 해석 가능성을 동시에 확보.

#### Limitations (저자 명시)

- "The clinical significance...should be considered with caution. This can be attributed to the fact that even though gait biomechanics are altered following ACLR, few biomechanical parameters demonstrate consistent results across studies."
- 수술 기법, 재활 프로토콜, 개인별 보상 전략, 성별 차이가 통제되지 않음.
- 데이터셋 비공개.
