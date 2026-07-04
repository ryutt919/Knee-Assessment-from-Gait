# 검색 종합: ACL kinematics/kinetics 기반 복합 점수 문헌 검증 (2021–2026)

## 판정 기준

1. 하나의 환자별 composite score/index를 실제 산출할 것.
2. ACLD 또는 ACLR 대상자를 직접 포함할 것.
3. kinematics 및/또는 kinetics에 기반할 것.
4. 2021-01-01부터 2026-07-01까지의 동료심사 문헌일 것.

기존 문서에 적힌 “WebSearch 17건, 후보 약 12편”은 재현 가능한 검색 로그가 없어 수량 자체는 검증하지 않았다. 이번 검증에서는 DOI, 원문 PDF/HTML, 방법·결과 표를 다시 대조했다.

## 검증 결과 (AS-IS → TO-BE)

<table fit-page-width="true" header-row="true">
<tr>
<td>AS-IS</td>
<td>TO-BE</td>
</tr>
<tr>
<td>“완전히 부합 2편, borderline 1편”</td>
<td>엄격한 네 기준을 모두 충족하는 최근 문헌은 **0편**. 최근 근접 문헌 1편(Li 2023), 비점수 ML 문헌 1편(Kokkotis 2022), 연도 밖 직접 선행문헌 1편(Liu 2020)</td>
</tr>
<tr>
<td>Li 2023을 단일 복합 점수로 간주</td>
<td>논문 용어는 composite index지만 실제로는 최대 8개 PCA component feature와 회귀분류기를 사용해 GDI/GPS식 단일 score 기준에는 부분 부합</td>
</tr>
<tr>
<td>Kokkotis 2022의 SHAP를 연속적 composite score처럼 취급</td>
<td>SHAP은 변수별 model-attribution이며 환자별 단일 severity/recovery score가 아님</td>
</tr>
<tr>
<td>Liu 2020을 waveform 지수이자 비-HA 참조로 기술</td>
<td>20개 discrete kinematic/kinetic 변수의 healthy-control-referenced PCA distance</td>
</tr>
<tr>
<td>“01, 03은 n=25–34, 정확도 약 81%”로 묶음</td>
<td>81.4% accuracy는 Li 2023에만 해당. Liu 2020은 25 patients + 12 controls의 NI group comparison이며 분류 정확도를 보고하지 않음</td>
</tr>
</table>

## 핵심 문헌

<table fit-page-width="true" header-row="true">
<tr>
<td>파일</td>
<td>실제 부합도</td>
<td>핵심 이유</td>
</tr>
<tr>
<td>[01](01_Composite_Index_of_Knee_Flexion_and_Muscle_Force_for_ACLD_Diagnosis.md)</td>
<td>최근 근접 문헌</td>
<td>ACLD, kinematics+model-derived muscle force, PCA+regression. 다만 단일 환자 점수가 아니라 여러 PCA features</td>
</tr>
<tr>
<td>[02](02_Explainable_ML_SHAP_Gait_Biomechanical_Parameters_ACL_Injury.md)</td>
<td>기준 불충족, 관련 ML 문헌</td>
<td>ACLD/ACLR/CON 3-class이나 결과는 classifier와 feature-level SHAP attribution</td>
</tr>
<tr>
<td>[03](03_Normalcy_Index_for_ACL_Deficiency_with_Meniscus_Injury.md)</td>
<td>연도 밖 직접 선행문헌</td>
<td>실제 단일 healthy-reference NI이지만 2020년 출판</td>
</tr>
</table>

## 검증 후 제외된 후보

<table fit-page-width="true" header-row="true">
<tr>
<td>논문</td>
<td>검증된 제외 사유</td>
</tr>
<tr>
<td>Seagers et al. 2026, AIR Score, DOI 10.1177/23259671261433009</td>
<td>27명의 knee-injury history가 없는 14–18세 여성 recreational athletes 대상 위험회복탄력성 screening. 실제 composite z-score와 trunk/hip/knee/foot subscores를 제시하지만 ACL 환자 회복 점수는 아님</td>
</tr>
<tr>
<td>Alhefzi et al. 2026, DOI 10.3389/fbioe.2026.1762965</td>
<td>ACLR 86명의 8개 gait asymmetry 변수를 각각 AI%로 계산. 통합 단일 composite score 없음</td>
</tr>
<tr>
<td>Zhou et al. 2022, DOI 10.3389/fbioe.2022.974724</td>
<td>isokinetic, hop, laxity, 6DOF kinematics 등 test battery를 함께 사용하지만 하나의 composite score로 통합하지 않음</td>
</tr>
<tr>
<td>Büttner et al. 2024 online/2025 issue, DOI 10.1002/jor.26001</td>
<td>bilateral whole-waveform functional mixed-effects analysis이며 단일 score를 산출하지 않음</td>
</tr>
<tr>
<td>Queen et al. 2020, DOI 10.1016/j.jbiomech.2019.109531</td>
<td>normalized symmetry index를 개별 변수에 적용하며 단일 다변량 score가 아님. 출판연도도 범위 밖</td>
</tr>
<tr>
<td>Skvortsov et al. 2023, DOI 10.3390/jcm12144803</td>
<td>acute ACL tear의 개별 gait/EMG 변수를 분석하며 composite score 없음</td>
</tr>
<tr>
<td>Seil et al. 2023, ACLISS, DOI 10.1007/s00167-023-07311-4</td>
<td>meniscus, cartilage, subchondral bone, collateral ligament의 구조적 손상 척도이며 gait kinematics/kinetics 기반이 아님</td>
</tr>
</table>

## 본 프로젝트의 위치

- 2021–2026 ACL 문헌에서 “healthy reference로부터의 환자별 단일 거리 + total/domain subscores + ACLD/ACLR 검증 + longitudinal recovery”를 모두 충족하는 지수는 이번 검증 범위에서 확인되지 않았다.
- 가장 가까운 계산 구조는 연도 밖의 Liu 2020 NI지만 discrete features, 소표본, ACLR/longitudinal validation 부재라는 차이가 있다.
- Li 2023은 supervised characteristic-point selection과 회귀분류, Kokkotis 2022는 classifier/SHAP이므로 본 프로젝트의 normative deviation score와 직접 동등하게 비교하면 안 된다.
- AIR Score는 점수 구조상 유용한 방법론 참고문헌이지만, uninjured screening cohort에서만 평가되어 post-injury recovery validity의 근거는 아니다.

## 결론의 강도

- “문헌이 전혀 없다”가 아니라, **이번 검색·검증 범위에서 엄격한 조건을 모두 만족하는 문헌을 확인하지 못했다**고 표현해야 한다.
- novelty claim은 추후 데이터베이스별 재현 가능한 systematic search와 독립 검토를 거쳐야 하며, 현재 결과는 scoping verification 수준이다.
