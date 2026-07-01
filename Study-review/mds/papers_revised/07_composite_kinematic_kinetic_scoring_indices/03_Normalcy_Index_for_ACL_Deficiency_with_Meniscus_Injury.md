# Use of the normalcy index for the assessment of abnormal gait in the anterior cruciate ligament deficiency combined with meniscus injury

> Liu X, Huang H, Ren S, Rong Q, Ao Y. (2020). *Computer Methods in Biomechanics and Biomedical Engineering*, 23(14), 1102–1108. DOI: 10.1080/10255842.2020.1789119. PMID: 32648770.
> PubMed: https://pubmed.ncbi.nlm.nih.gov/32648770/
> 저자 공개 원문: https://www.researchgate.net/publication/342855790_Use_of_the_normalcy_index_for_the_assessment_of_abnormal_gait_in_the_anterior_cruciate_ligament_deficiency_combined_with_meniscus_injury
> 로컬 PDF: `docs/ref_papers/07_composite_kinematic_kinetic_scoring_indices/Use of the normalcy index for the assessment of abnormal gait in the anterior cruciate ligament deficiency combined with meniscus injury.pdf`

## 검증 결과 (AS-IS → TO-BE)

| AS-IS | TO-BE |
|---|---|
| NI를 “gait waveform 편차 기반” 지수로 설명 | 3개 kinematic + 17개 kinetic **이산 변수**를 표준화하고 PCA 공간에서 계산한 squared distance |
| HA-referenced가 아니라 자체 정상 참조 범위라고 구분 | 12명 건강 대조군의 평균·표준편차·PCA를 직접 참조하므로 명백한 healthy-control-referenced 지수 |
| 환자 25명을 모두 ACL+반월판 손상군으로 기술 | 25명 중 isolated ACLD 8명, lateral meniscus 동반 5명, medial 동반 5명, bilateral meniscus 동반 7명 |
| NI가 반월판 손상 중증도를 “그대로 반영”한다고 강하게 해석 | group mean의 순서 경향은 유의했지만 환자군 간 pairwise 차이는 유의하지 않았고 범위가 크게 겹쳐 정확한 손상 유형 판별은 불가능 |
| 저자 한계를 “초록에 상세 미기재”로 대체 | 본문에 hinge-knee/sagittal-only model, coper 상태 미통제, 소표본과 부상 후 기간 층화 필요를 명시 |

## 연구 목적과 표본

- 건강 보행으로부터의 다변량 거리를 하나의 Normalcy Index(NI)로 축약해 ACLD 및 반월판 동반손상의 gait abnormality를 평가했다.
- 환자 25명은 ACLD 8, ACLD+lateral meniscus 5, ACLD+medial meniscus 5, ACLD+medial/lateral meniscus 7명으로 구성됐다. 건강 대조군은 12명이었다.

## 방법

- Vicon MX 8-camera(100 Hz), AMTI force plates(1000 Hz), 10 m self-selected level walking을 사용했다. 각 참가자에서 5개 성공 trial을 수집해 평균했다.
- AnyBody 6.0.5와 Twente Lower Extremity model로 muscle forces와 joint moments를 계산했다.
- kinematic data는 gait cycle 101 points, kinetic data는 stance 61 points로 보간했지만 NI 입력은 전체 waveform이 아니라 peak, phase, impulse 등 20개 이산 변수였다.
- 건강 대조군에서 각 변수의 평균과 표준편차로 z-standardization하고 PCA eigenvector/eigenvalue로 whitening한 뒤, 각 대상자의 squared Euclidean length를 NI로 정의했다. 값이 클수록 정상에서 멀다.
- 대조군은 양측 24 legs로 정상 분포를 구성했다. 환자군은 손상측을 평가했다.

## 결과

| 그룹 | n | Mean NI (range) |
|---|---:|---:|
| Control | 12 | 18.33 (3.54–48.01) |
| ACLD | 8 | 32.83 (15.86–66.96) |
| ACLD + lateral meniscus | 5 | 38.69 (21.17–59.70) |
| ACLD + medial meniscus | 5 | 61.07 (42.05–88.06) |
| ACLD + medial/lateral meniscus | 7 | 68.45 (30.19–149.23) |

- 대조군과 각 환자군의 차이는 p<0.05였다.
- Control < ACLD < ACLDL < ACLDM < ACLDML 순서 경향은 Jonckheere–Terpstra test p<0.001이었다.
- 그러나 ACLD, ACLDL, ACLDM, ACLDML 환자군 사이의 개별 비교는 유의하지 않았다. 넓은 범위 중첩 때문에 하나의 NI로 특정 반월판 손상 유형을 식별할 수 없다.

## 원문 근거

- “The NI method can be simplified as the square of the distance between the patient’s parameters and the average value of normal subjects.” (Methods)
- “no clear conclusion could be made in terms of the relationship between each diagnosis category ... and NI scores.” (Discussion)
- “The normalcy index is a simple yet effective tool to evaluate movement disorders.” (Conclusion)

## 본 프로젝트와의 관련성

- 세 문헌 중 본 프로젝트의 “건강인으로부터 얼마나 멀리 떨어졌는가”라는 정의에 가장 직접적으로 대응한다.
- 다만 20개 discrete features 기반이며 full-waveform score가 아니다. 또한 ACLR, longitudinal recovery, 외부 정상 참조군 검증을 수행하지 않았다.
- 건강 대조군 12명의 양측 leg를 정상 참조와 평가에 함께 사용하므로 leave-one-subject-out 정상성 검증이 아니다.

## 저자가 명시한 한계

- 무릎을 sagittal-plane hinge joint로 단순화했다.
- coper/non-coper 같은 post-injury activity level을 통제하지 않아 NI 차이를 가릴 수 있다.
- 환자 하위군 표본이 작다.
- 더 많은 gait data와 부상 후 기간별 층화가 필요하다.
- musculoskeletal-model simplification과 skin-motion artifact의 오차가 불가피하다.
