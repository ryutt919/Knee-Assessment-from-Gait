# Time to normalization of gait following ACL reconstruction compared with healthy controls: A systematic review and meta-analysis

Chen, S., Gong, H., Lyu, C., Li, K., & Shaharudin, S. (2026). Time to normalization of gait following ACL reconstruction compared with healthy controls: A systematic review and meta-analysis. Gait & Posture, 123, 109972. https://doi.org/10.1016/j.gaitpost.2025.109972

## 서지정보

- 저자: Shiwei Chen, Han Gong, Chennan Lyu, Kehan Li, Shazlin Shaharudin
- 연도: 2026
- 저널: Gait & Posture
- DOI: https://doi.org/10.1016/j.gaitpost.2025.109972
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/02_acl_gait_reviews_meta_analyses/Time to normalization of gait following ACL reconstruction compared with healthy controls - A systematic review and meta-analysis.pdf
- 분석 provider: antigravity

## 연구 목적

- 전방 십자 인대 재건술(ACLR) 환자와 건강한 대조군 간의 핵심 보행 변수 차이를 평가하고, 수술 후 이러한 변수가 정상화되는 시기를 추정하는 것을 목적으로 한다. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “This review aimed to assess differences in key gait parameters between ACLR patients and healthy controls, and to estimate when these parameters normalize postoperatively.”

## 연구 설계와 대상

- 수술 후 최소 3개월이 경과한 일차성 ACL 파열 재건술 환자를 대상자로 선정하였다. _(근거: PAGE 2, 2.1. Eligibility criteria)_
  - 근거 원문: “Participants included in this study were patients with primary ACL rupture who underwent surgical reconstruction and had a postoperative recovery period of at least three months.”
- 5개 데이터베이스 검색을 통해 총 976명의 참가자가 포함된 20개 연구를 선정하여 분석하였다. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “A systematic search across five databases yielded 5251 records, with 20 studies (n= 976) retained.”

## 방법

- 대조군과 실험군 간 보행 변수의 평균 및 표준편차 데이터를 사용하여 효과크기인 표준화된 평균 차이(SMD, Cohen's d)와 그 분산을 계산하였다. _(근거: PAGE 3, 2.6. Statistical analysis)_
  - 근거 원문: “First, pooled standard deviations (pooled SD) were calculated from the means and standard deviations reported by experimental and control groups. From these, standardized mean differences (SMD, Cohen’s d) and their variances were computed to quantify effect sizes and statistical uncertainty [25].”
- 동일 연구 내 여러 시점 간의 상관관계를 통제하기 위해 상관계수를 r=0.85로 가정하고 공분산 행렬을 구축하였다. _(근거: PAGE 3, 2.6. Statistical analysis)_
  - 근거 원문: “To account for correlation between multiple time points within the same study, we assumed a correlation coefficient of r=0.85, consistent with previous longitudinal meta-analyses in ACLR patients [26].”
- R의 metafor 패키지를 사용하여 제한된 최대우도(REML)법에 의한 종단 혼합효과 메타분석을 수행하였다. _(근거: PAGE 3, 2.6. Statistical analysis)_
  - 근거 원문: “A longitudinal mixed-effects meta-analysis was performed using the restricted maximum likelihood (REML) method, primarily employing the metafor package in R, following established analytical frameworks [28,29].”

## 핵심 결과

- 건강한 대조군에 비해 ACLR 환자는 보행 시 최대 무릎 굴곡 각도와 최대 무릎 굴곡 모멘트가 유의하게 낮았으나, 보행 속도는 유의한 차이를 보이지 않았다. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “Compared to healthy controls, ACLR patients assessed at 3–107 months postoperatively exhibited significantly lower peak knee flexion angle (d = − 0.48, 95% CI: − 0.87 to − 0.10) and peak knee flexion moment (d = − 1.06, 95 % CI: − 2.06 to − 0.07), while walking speed was non-significant (d = − 0.17, 95 % CI: − 0.47–0.13).”
- 모델 예측 결과 집단 간 차이가 사라지는 시점은 무릎 굴곡 각도의 경우 약 16.2개월, 무릎 굴곡 모멘트의 경우 약 10.1개월로 나타났다. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “Logarithmic modeling indicated that group differences became non-significant at 16.2 months for flexion angle and 10.1 months for flexion moment. Walking speed was statistically non-significant at any timepoint.”

## 저자 결론

- ACLR 환자는 수술 후 3개월 이상 시점에서도 대조군에 비해 최대 무릎 굴곡 각도와 모멘트가 유의하게 감소해 있으며, 이는 수술 후 각각 약 16.2개월과 10.1개월에 정상화된다. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “At ≥ 3 months post-ACLR, patients exhibited significantly reduced peak knee flexion angle and flexion moment compared to controls. These deficits normalized at approximately 16.2 and 10.1 months, respectively.”
- 수술 후 첫 10-16개월 이내의 초기 재활 과정에서 대퇴사두근 강화와 무릎 굴곡 각도 회복에 집중하여 보행 생체역학적 결손을 해결해야 한다. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “These findings suggest that early-phase rehabilitation should emphasize quadriceps strengthening and knee flexion restoration within the first 10–16 months to address persistent biomechanical deficits.”

## 연구의 한계

- 수술 후 3개월 미만의 급성기 보행 데이터를 배제하여 초기 보행 변화를 파악하지 못했을 가능성이 있다. _(근거: PAGE 7, 4.3. Limitations and future directions)_
  - 근거 원문: “Given the limited and inconsistent reporting of acute postoperative gait data, this study included only ACLR patients assessed at ≥ 3 months post-surgery, potentially missing early-phase gait changes.”
- 다수준 종단 메타분석 모델을 적용했음에도 불구하고, 특히 최대 무릎 굴곡 모멘트(τ² = 1.15) 등에서 상당한 이질성이 잔존하였다. _(근거: PAGE 7, 4.3. Limitations and future directions)_
  - 근거 원문: “Despite applying a multilevel modeling framework with continuous autoregressive covariance and robust variance estimation to account for within-study correlations, notable heterogeneity remained—particularly for peak knee flexion moment (τ² = 1.15).”

## 생각해볼 내용

- 다수준 분석으로 보행 변수의 시점별 정상화 시기를 정밀하게 도출하였으나, 이식건 종류, 재활 프로토콜 등의 임상적 요인 누락에 따른 높은 이질성이 결과 일반화 시 한계로 작용할 수 있다. _(근거: PAGE 7, 4.3. Limitations and future directions)_
  - 근거 원문: “This variability may reflect differences in graft types, rehabilitation protocols, or follow-up durations. However, insufficient reporting of these variables precluded stratified analyses, limiting our ability to identify sources of heterogeneity.”

## 이 연구가 지적한 선행연구의 문제점

- 기존 메타분석들은 각 연구를 독립된 단위로 보고 단일 시점에서만 효과크기를 추출하는 전통적 방식을 사용하여 여러 시점의 추적 관찰 데이터를 충분히 활용하지 못했다. _(근거: PAGE 2, 1. Introduction)_
  - 근거 원문: “However, most existing meta-analyses have utilized traditional frameworks, treating each study as an independent unit and extracting effect sizes only at single time points [13], thus failing to fully leverage data available from multiple follow-up assessments [14,15].”

## 이 연구의 해결 방식과 기여

- 본 연구는 여러 시점의 데이터를 통합하는 다수준 종단 메타분석 방식을 적용하여 수술 후 보행 회복의 구체적인 시간적 패턴과 궤적을 확인하고자 했다. _(근거: PAGE 2, 1. Introduction)_
  - 근거 원문: “To address these limitations, the present study aims to employ a multilevel longitudinal meta-analytic approach. This method moves beyond the traditional meta-analytic assumption of single effect-size extraction by integrating data across multiple follow-up time points, thus providing detailed insight into the temporal patterns of gait recovery.”

## 레퍼런스할 수 있는 내용

### 1. 전방 십자 인대 부상의 빈도와 심각성

- 원문 발췌: “Injuries to the anterior cruciate ligament (ACL) represent one of the most frequent and serious forms of knee trauma in athletic populations [1].”
- 한국어 번역: 전방 십자 인대(ACL) 부상은 운동선수 집단에서 가장 빈번하고 심각한 형태의 무릎 외상 중 하나를 나타낸다.
- 원문 위치: PAGE 1, 1. Introduction
- 원문 내 인용표기: [1]
- 해당 선행문헌: [1] F. Mancino, B. Kayani, A. Gabr, A. Fontalis, R. Plastow, F.S. Haddad, Anterior cruciate ligament injuries in female athletes: risk factors and strategies for prevention, Bone Jt Open 5 (2) (2024) 94–100, https://doi.org/10.1302/2633-1462.52.Bjo-2023-0166.
- 주장 유형: background_citation
- 활용 맥락과 주의: 전방 십자 인대 부상이 운동선수에게 흔하고 심각한 부상임을 뒷받침할 때 사용하며, 2차 인용에 유의해야 한다.

### 2. ACLR 이후 보행 이상의 지속과 이로 인한 재부상 및 관절 퇴행 위험

- 원문 발췌: “Gait abnormalities often persist after anterior cruciate ligament reconstruction (ACLR) and may increase the risk of reinjury and joint degeneration.”
- 한국어 번역: 보행 이상은 전방 십자 인대 재건술(ACLR) 이후에도 흔히 지속되며, 이는 재부상 및 관절 퇴행의 위험을 증가시킬 수 있다.
- 원문 위치: PAGE 1, Abstract
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: 수술 후 보행 이상 지속으로 인한 재부상 및 관절 퇴행 위험성 증가의 분석 배경을 인용할 때 사용한다.

### 3. ACL 부상이 통증, 불안정성 및 골관절염 등 관절 기능과 삶의 질에 미치는 장기적 영향

- 원문 발췌: “ACL injuries often lead to joint pain, instability, and early-onset osteoarthritis, significantly impairing physical performance and quality of life [3].”
- 한국어 번역: 전방 십자 인대(ACL) 부상은 흔히 관절 통증, 불안정성 및 조기 발병 골관절염으로 이어져 신체 기능과 삶의 질을 크게 저하시킨다.
- 원문 위치: PAGE 1, 1. Introduction
- 원문 내 인용표기: [3]
- 해당 선행문헌: [3] N.A. Friel, C.R. Chu, The role of ACL injury in the development of posttraumatic knee osteoarthritis, Clin. Sports Med. 32 (1) (2013) 1–12, https://doi.org/10.1016/j.csm.2012.08.017.
- 주장 유형: background_citation
- 활용 맥락과 주의: 전방 십자 인대 부상이 관절 기능 결손 및 조기 골관절염을 초래하여 물리적 기능과 삶의 질을 저하시킨다는 내용을 뒷받침할 때 사용하며, 2차 인용에 주의해야 한다.
