# 검색 종합: ACL 부상 심각도·회복도·비대칭도의 Kinematics/Kinetics 기반 복합 점수 문헌조사 (2021-2026)

## 검색 개요

- 목적: 최근 5년(2021-01-01 ~ 2026-07-01) 이내에 ACL 손상 환자(ACLD/ACLR)의 부상 심각도, 회복 정도, 좌우 비대칭 정도를 kinematics 및/또는 kinetics 기반 **단일 복합 점수(composite score/index)** 로 제시하려 시도한 연구를 조사.
- 판정 기준: (1) 개별 변수 비교표가 아니라 실제로 하나의 복합 점수/지수로 통합했을 것, (2) ACL 손상/재건/결손 환자군 대상일 것(비손상 스크리닝 대상 제외), (3) 주관적 PRO(IKDC/KOOS/Lysholm/ACL-RSI 등)만이 아니라 kinematics/kinetics 기반일 것, (4) 동료심사 문헌일 것.
- WebSearch 쿼리 17건(GDI/GPS/MDP, composite/asymmetry index, ML 기반 점수, RTS 생체역학 도구, IMU 기반 점수, PCA/waveform 접근 등) 실행, 후보 논문 약 12편을 WebFetch로 원문/초록 대조 검증.
- 엄격한 기준 적용 결과 **완전히 부합하는 문헌은 2편, 기준을 살짝 벗어나거나 경계선상인 문헌 1편**으로, 목표했던 5-10편보다 수가 적음. 기준을 느슨히 하지 않고 부합하지 않는 논문을 억지로 포함하지 않았음.

## 채택된 문헌 (상세는 개별 파일 참고)

| 파일 | 논문 | 부합도 |
|---|---|---|
| [01](01_Composite_Index_of_Knee_Flexion_and_Muscle_Force_for_ACLD_Diagnosis.md) | Li et al. 2023, *Bioengineering* — PCA 기반 무릎굴곡+근력 복합 지수로 ACLD 진단 | 완전 부합 (가장 강한 매치) |
| [02](02_Explainable_ML_SHAP_Gait_Biomechanical_Parameters_ACL_Injury.md) | Kokkotis et al. 2022, *Scientific Reports* — SHAP 기반 explainable ML, ACLD/ACLR/CON 3-way 분류 | Borderline (명명된 단일 지수는 아니고 feature importance 기반) |
| [03](03_Normalcy_Index_for_ACL_Deficiency_with_Meniscus_Injury.md) | Liu et al. 2020, *Comp Methods Biomech Biomed Eng* — Normalcy Index로 ACL+반월판 손상 중증도 gradient 검증 | 연도상 5년 기준 밖(2020), 배경 문헌으로 포함 |

## 검증 후 제외된 후보 (투명성을 위해 기록)

| 논문 | 제외 사유 |
|---|---|
| Seagers et al. 2026, "AIR Score" (Composite z-score index), *Orthop J Sports Med*, DOI 10.1177/23259671261433009 | Kinematics/kinetics 기반 진짜 z-score 복합 지수로 방법론은 우수하나, 대상이 "Female adolescent recreational athletes without a history of knee injury" — 비손상 청소년 스크리닝 대상이라 ACL 손상 환자군 기준 불충족 |
| Alhefzi et al. 2026, *Frontiers Bioeng Biotechnol*, DOI 10.3389/fbioe.2026.1762965 | 개별 변수별 비대칭 지수(AI%, LSI%)만 산출, 통합 단일 점수 아님. 원문 내 수치가 정형화된 패턴으로 보여 미검증(unverified) 표시 |
| Zhou et al. 2022, *Frontiers Bioeng Biotechnol*, DOI 10.3389/fbioe.2022.974724 (다면 kinematics RTS test battery) | LSI 임계값을 4개 도메인에서 개별 기준으로 유지, 통합 지수화하지 않음 |
| Büttner et al. 2024/2025, *J Orthop Res*, DOI 10.1002/jor.26001 (bilateral waveform analysis) | Composite/peak 점수화를 명시적으로 배제하고 전체 waveform functional mixed-effects modeling을 사용 — 단일 점수 미산출 |
| Queen et al. 2019/2020, Normalized Symmetry Index, *J Biomechanics*, DOI 10.1016/j.jbiomech.2019.109531 | 5년 기준 밖. NSI도 변수별 산출이며 통합 점수 아님 |
| Skvortsov et al. 2023, *J Clin Med*, DOI 10.3390/jcm12144803 (급성 ACL 파열 중증도 보행 분석) | 개별 gait/EMG 변수를 각각 분석, 복합 지수 미산출 |
| ACLISS (Seil et al. 2023, *KSSTA*) | 구조적/수술적 조직손상 중증도 척도이며 kinematics/kinetics 기반 아님 |

## 종합 (우리 프로젝트 Gait Normality Score와의 위치 비교)

- ACL 영역의 "복합 생체역학 점수화" 연구는 뇌성마비/신경정형외과 영역의 GDI/GPS/MDP처럼 성숙·표준화되어 있지 않고 파편화되어 있음.
- ACL 연구는 여전히 개별 변수 비교(peak 각도/모멘트 비대칭 등)나, 최근에는 오히려 복합 점수화를 명시적으로 지양하는 전체 waveform 통계기법(SPM, functional mixed-effects model)이 주류.
- 실제로 복합 점수를 제시한 사례는 두 갈래: (1) 소규모 PCA+회귀 진단 지수(01, 03 — 동일 연구그룹, n=25-34, 정확도 ~81%, 독립 재현 검증 없음), (2) SHAP 기반 ML 분류기(02 — 정확도 ~95%이나 "점수"가 아니라 feature importance 순위로 제시).
- GDI/GPS와 가장 유사한 방법론(z-score 기반 복합 지수)인 AIR Score(Seagers et al. 2026)는 손상 환자가 아닌 비손상 선수의 사전 위험 스크리닝용으로 개발되어 대상군 기준에서 제외됨 — 이는 "손상 후 회복 추적용 검증된 지수"의 공백을 방증.
- 이번 조사에서 waveform 수준의 kinematics+kinetics를 결합해 (a) HA(건강대조군) 기준값에 정렬(referenced)되고, (b) ACLD/ACLR을 구분해 검증되고, (c) 시간에 따른 회복 궤적(longitudinal recovery)을 정량화하는, GDI/GPS에 대응하는 지수를 제시한 문헌은 발견되지 않음.
- 즉 본 프로젝트의 Gait Normality Score(HA-referenced, kinematics/kinetics 도메인 subscore 포함)는 (i) 해석 불가능한 블랙박스 분류나 (ii) 종적 추적을 염두에 두지 않은 소표본 진단 지수 사이의 실질적 공백을 메우는 방향으로 위치할 수 있음.

## 참고: 조사의 한계

- 엄격한 판정 기준(단일 복합 점수 + ACL 손상 환자군 + kinematics/kinetics 기반 + 동료심사 + 2021-2026) 적용 시 완전 부합 문헌은 2편뿐이며, 목표했던 5-10편에 못 미침.
- 기준을 완화하면(예: 비손상 선수 위험 스크리닝 도구 포함, 개별 다중지표 test battery 포함) 후보 pool을 넓힐 수 있으나, 이번 조사에서는 "복합 점수" 정의를 엄격히 지켜 억지 포함을 하지 않음.
