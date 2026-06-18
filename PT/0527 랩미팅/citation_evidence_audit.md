# Citation Evidence Audit: Capstone PPT and 0527 HTML

작성일: 2026-06-17 10:40

## Scope

검토 대상은 다음 세 가지입니다.

- 원본 PPT: `PT/스포츠과학과 캡스톤디자인 발표 19101207 김태현.pptx`
- 인용 추가 PPT: `PT/스포츠과학과 캡스톤디자인 발표 19101207 김태현_cited.pptx`
- 비교 HTML: `PT/0527 랩미팅/comprehensive_presentation.html`

근거는 `docs/ref_papers/`의 로컬 PDF를 우선 확인했고, 로컬에 없거나 서지가 의심되는 항목은 외부 검색으로 보완했습니다. 외부 확인 대상은 HTML `[10]`, `[13]`, `[14]`, `[16]`, `[17]`, `[19]`, `[21]`, `[22]` 및 PPT 전용 Tan/Kokkotis 항목입니다.

## Bottom Line

현재 인용은 "대체로 주제 방향은 맞지만, 그대로 제출하기에는 서지와 claim 매칭 오류가 남아 있음"으로 판단합니다.

가장 큰 문제는 세 가지입니다.

1. PPT와 HTML의 각주 번호 체계가 서로 다릅니다. PPT `[1]`은 HTML `[1]`이 아닙니다. 이 자체는 가능하지만, 같은 발표 묶음에서는 혼동을 만들기 때문에 최종본은 하나의 번호 체계로 통일하는 편이 안전합니다.
2. HTML 참고문헌 중 `[10]`, `[13]`, `[14]`, `[16]`, `[17]`, `[22]`는 제목 또는 서지가 실제 확인된 자료와 다릅니다.
3. 일부 문장은 근거 논문이 "연관성"만 보여주는데 슬라이드 문장은 인과처럼 쓰였거나, ACLR 환자 자료를 "건강한 성인" 일반 통계처럼 쓴 곳이 있습니다.

## Immediate Corrections

| Priority | Location | Current issue | Required correction |
|---|---|---|---|
| High | PPT cited References `[1]` | `Gao & Zheng (2014)`로 표기되어 있으나 로컬 PDF는 Slater, Hart, Kelly & Kuenze (2017) 논문입니다. | PPT 스크립트와 References 슬라이드의 `[1]` 서지를 Slater et al. (2017), `Progressive Changes in Walking Kinematics and Kinetics...`, Journal of Athletic Training으로 수정합니다. |
| High | HTML `[13]` | Tedesco et al.의 제목이 다른 Sensors 논문 제목처럼 잘못 들어가 있습니다. | `Motion Sensors-Based Machine Learning Approach for the Identification of Anterior Cruciate Ligament Gait Patterns in On-the-Field Activities in Rugby Players`로 수정합니다. DOI: `10.3390/s20113029`. |
| High | HTML `[16]` | `Mpaliotes, Vasileios (2022)` 정확 일치 자료를 찾지 못했습니다. | 이 항목은 삭제하거나, 확인 가능한 Alshehri (2019) KU dissertation 또는 다른 검증 자료로 교체합니다. |
| High | HTML `[22]` | `Pelvis and Trunk Motion Influence...` 제목은 Sigward 2016 ACL gait 논문과 맞지 않습니다. | ACLR 보행/달리기 knee-loading asymmetry 근거가 필요하면 Sigward, Lin & Pratt (2016), Clinical Biomechanics, DOI `10.1016/j.clinbiomech.2015.11.003`로 교체합니다. |
| Medium | HTML line 174 | Lisee `[6]`는 post-ACLR daily-step tertile 자료이지 "건강한 성인 하루 6,000-12,000보" 일반 근거가 아닙니다. | 문장을 "post-ACLR 대상자의 관찰 범위"로 바꾸거나 일반 성인 보행량 근거를 별도로 추가합니다. |
| Medium | HTML line 201 | Wellsandt `[19]`는 early unloading과 5년 OA의 연관을 보여주지만, "과도하게 줄인 환자들이 OA 발생률 증가"는 인과처럼 읽힙니다. | "초기 낮은 knee loading이 5년 후 early OA와 연관"으로 수정합니다. |
| Medium | HTML lines 142, 147 | 재파열률/OA 발생률에 `[5,7]`만 붙은 것은 약합니다. | 재파열/OA burden은 Tan et al. 또는 Hart et al.을 명시적으로 추가합니다. |
| Medium | HTML line 217 | `[16,22]`로 tibial angular velocity와 KEM 직접 연관을 주장하기 어렵습니다. | `[11]` Jones et al. 중심으로 수정하거나, 정확히 shank angular velocity/KEM을 다룬 검증 논문으로 교체합니다. |

## HTML Claim-Level Audit

| HTML line | Claim | Current citation | Verdict | Recommendation |
|---:|---|---|---|---|
| 137 | 미국 내 연간 재건술 25만 건 | `[10]` | Partial. Zhu Smart Health 2025는 ACLR classification 및 burden 수치를 뒷받침할 수 있으나 HTML 제목이 부정확합니다. 다른 로컬 근거는 250,000 ACL injuries 및 130,000 ACLR procedures로 구분합니다. | `[10]` 제목을 실제 제목으로 수정하거나, "ACL injuries 250,000 / ACLR procedures 130,000"로 문장을 바꾸고 Slater et al. 또는 Kokkotis를 사용합니다. |
| 142 | 수술 후 2년 이내 재파열률 | `[5,7]` | Weak. Yu review는 재손상 위험을 논하지만, Kaur는 주로 gait movement pattern meta-analysis입니다. | Tan et al. 2023 또는 Webster/Feller 재손상 논문을 추가합니다. |
| 147 | 10-15년 내 외상 후 골관절염 발생 | `[5,7]` | Partial. Yu/Kaur는 장기 OA risk 문맥은 맞지만 정확 수치 근거로는 Hart et al. 또는 Tan et al.이 더 직접적입니다. | PPT의 Hart/Tan 근거를 HTML에도 추가하거나 문장을 "장기 PTOA 위험 증가"로 완화합니다. |
| 156 | RTS 결정이 대부분 주관적 판단에 의존 | `[8,20]` | Partial. Di Stasi는 RTS pass/fail gait 차이를 잘 뒷받침하지만, "대부분 주관적" 표현은 별도 RTS guideline 근거가 필요합니다. | "RTS 기능 기준 통과 여부에 따라 gait pattern이 다름"으로 수정하거나 RTS decision-making review를 추가합니다. |
| 174 | 건강한 성인의 하루 6,000-12,000보 | `[6]` | Incorrect wording. Lisee는 ACLR 대상자 daily-step tertile 자료입니다. | "ACLR 대상자의 관찰된 daily step tertile은 약 3,326-12,680보"로 수정합니다. |
| 179 | 무릎 경직 전략 | `[5,8]` | OK. Yu review와 Lewek는 quadriceps weakness, knee flexion/KEM 감소, stiff-knee strategy 문맥을 뒷받침합니다. | 유지 가능합니다. |
| 185 | 전신적·양측성 영향 | `[1]` | OK. Büttner waveform 분석은 ACLR limb, uninvolved limb, controls 간 stance-phase 차이를 다룹니다. | 유지 가능합니다. |
| 199 | daily steps가 적은 군에서 vGRF/KEM 감소 | `[6]` | OK. Lisee가 daily-step tertile과 vGRF/KEM 차이를 직접 다룹니다. | 유지 가능합니다. |
| 200 | KAM/KEM 지속 감소가 연골 접촉 응력/세포 대사 저해 | `[15]` | Partial. Hall은 ACLR gait와 knee OA perspective를 다루지만 세포 대사까지는 직접 근거가 약합니다. | "OA 관련 부하 환경 변화" 정도로 완화하거나 cartilage mechanobiology 근거를 추가합니다. |
| 201 | 초기 부하 감소 환자에서 5년 OA 증가 | `[19]` | Good but causal wording. Wellsandt는 early lower knee loading과 5-year early OA의 association 근거입니다. | "연관됨"으로 바꿉니다. |
| 217 | 낮은 경골 각속도와 KEM 직접 연관 | `[16,22]` | Weak/incorrect. `[16]`은 서지 미확인, `[22]`는 제목 불일치입니다. | Jones `[11]` 또는 정확한 shank angular velocity 논문으로 대체합니다. |
| 221 | IMU 각속도/가속도와 무릎 모멘트/ROM 상관 `rs=0.71-0.77` | `[11]` | OK. Jones et al.에서 tibia acceleration/angular velocity와 knee biomechanics의 strong correlation을 보고합니다. | 유지 가능합니다. |
| 225 | 단일 IMU로 3D 하지 운동역학 추정 가능 | `[12,21]` | OK with caveat. Krishnakumar review와 Lee & Park single-sacrum-IMU kinetics estimation이 뒷받침합니다. Lee & Park는 ACL-specific은 아닙니다. | "보행 kinetics 추정 가능성"으로 유지하되 ACL 임상 적용은 별도 검증 필요라고 표현합니다. |
| 236 | 자기선택 속도 평가는 비대칭을 위장 | `[3]` | Partial. Krishnan은 KEM 기계적 요인 분석 근거로 좋지만, speed masking claim은 Garcia/Lai가 더 직접적입니다. | `[2,4]`로 변경하는 것이 안전합니다. |
| 242 | 편안한 속도에서 GRF 변조 작고 대칭성 높음 | `[2]` | OK. Garcia는 walking speed별 GRF asymmetry를 직접 비교합니다. | 유지 가능합니다. |
| 247 | 빠른 속도에서 비대칭이 명확히 노출 | `[4,5]` | OK. Lai는 fast walking에서 knee kinematic asymmetry, Yu는 review 차원의 보강 근거입니다. | 유지 가능합니다. |
| 256 | IMU 보행 연구는 활발 | `[9,10]` | OK with bibliography correction. COMPWALK-ACL 및 Zhu Smart Health가 이 문맥을 뒷받침합니다. | `[10]` 제목만 실제 제목으로 수정합니다. |
| 265 | 대부분 단일 속도 프로토콜, 속도 의존 비대칭 포착 실패 | `[2,3]` | Partial. Garcia는 직접 근거, Krishnan은 보조 근거입니다. | `[2,4]` 또는 `[2,4,5]`로 바꾸는 편이 더 직접적입니다. |
| 269 | 피크값 특징 공학 한정, 시계열 정보 누락 | `[1,13,14]` | Partial. Büttner는 waveform/stance-phase 분석 필요성을 직접 지지합니다. `[13]`, `[14]`는 ML/IMU feasibility 자료이지 이 claim의 직접 근거는 약합니다. | `[1]` 중심으로 두고, 기존 IMU ML의 feature-engineering 한계 근거를 별도로 추가합니다. |

## HTML 참고문헌별 검토

| 번호 | HTML 표기 | 확인된 자료 | 로컬/웹 | 판정 |
|---|---|---|---|---|
| `[1]` | Büttner et al. (2024), Bilateral Waveform Analysis... | Büttner et al., `Bilateral waveform analysis of gait biomechanics presurgery to 12 months following ACL reconstruction compared to controls`, DOI `10.1002/jor.26001`. | `docs/ref_papers/01_acl_gait_biomechanics_studies/Bilateral waveform analysis...pdf` | 적절합니다. 다만 최종 출판 연도 확인은 필요할 수 있습니다. 현재 확인된 정보상 2024년에 accepted 상태입니다. |
| `[2]` | Garcia et al. (2021), Gait Asymmetries... | Garcia et al., `Gait asymmetries are exacerbated at faster walking speeds in individuals with acute ACL reconstruction`. | `docs/ref_papers/01_acl_gait_biomechanics_studies/Gait asymmetries are exacerbated...pdf` | 적절합니다. 속도 의존적 비대칭을 강하게 뒷받침합니다. |
| `[3]` | Krishnan et al. (2022), Mechanical Factors... | Krishnan et al., `Mechanical Factors Contributing to Altered Knee Extension Moment During Gait After ACL Reconstruction`. | `docs/ref_papers/01_acl_gait_biomechanics_studies/Mechanical Factors Contributing...pdf` | KEM 기전 설명에는 적절합니다. 다만 보행 속도 프로토콜 관련 주장에는 부분적 근거에 그칩니다. |
| `[4]` | Lai et al. (2024), fast speed ACLR walking | Lai et al., `Whether Patients with ACL Reconstruction Walking at a Fast Speed Have More Knee Kinematic Asymmetries...`. | `docs/ref_papers/01_acl_gait_biomechanics_studies/Whether Patients with ACL Reconstruction Walking at a Fast Speed...pdf` | 적절합니다. 빠른 보행 조건의 비대칭을 직접 뒷받침합니다. |
| `[5]` | Yu et al. (2026), Advances in Gait Alterations... | Yu et al., ACLR 후 보행 변화와 재활을 다룬 review 논문. | `docs/ref_papers/02_acl_gait_reviews_meta_analyses/Advances in Gait Alterations...pdf` | 넓은 배경 근거로는 적절합니다. 다만 재파열률/OA 발생률 같은 정확한 수치 근거로는 약합니다. |
| `[6]` | Lisee et al. (2022), Linking Gait Biomechanics and Daily Steps... | Lisee et al., daily-step tertile과 ACLR limb stance biomechanics를 다룬 논문. | `docs/ref_papers/01_acl_gait_biomechanics_studies/Linking Gait Biomechanics and Daily Steps...pdf` | ACLR 환자의 daily step과 loading 관계 근거로는 적절합니다. 건강한 성인의 일반 보행량 기준으로 쓰기에는 부적절합니다. |
| `[7]` | Kaur et al. (2016), Movement Patterns... | Kaur et al., ACLR 후 무릎 움직임에 대한 systematic review/meta-analysis. | `docs/ref_papers/02_acl_gait_reviews_meta_analyses/Movement Patterns...pdf` | review 근거로는 적절합니다. 정확한 재파열률/OA 발생률 수치 근거로는 약합니다. |
| `[8]` | Lewek et al. (2002), quadriceps strength and gait | Lewek et al., ACLR 후 insufficient quadriceps strength가 보행에 미치는 영향을 다룬 논문. | `docs/ref_papers/01_acl_gait_biomechanics_studies/The effect of insufficient quadriceps strength...pdf` | quadriceps와 gait mechanics 설명에 적절합니다. |
| `[9]` | Yona, Peskin & Fischer (2026), COMPWALK-ACL | Yona et al., `The COMPWALK-ACL: A Dataset of Multi-pace IMU Gait Kinematics...`. | `docs/ref_papers/03_wearable_imu_and_portable_sensing/The COMPWALK-ACL...pdf` | 적절합니다. multi-pace IMU gait dataset 근거로 직접 사용 가능합니다. |
| `[10]` | Zhu et al. (2025), Multi-modal Gait Analysis... | Zhu et al., `Clinically relevant predictive modeling for personalized ACL reconstruction classification`, Smart Health 36, 100575, DOI `10.1016/j.smhl.2025.100575`. | 웹: ScienceDirect | 서지 불일치가 있습니다. 실제 제목으로 수정해야 합니다. |
| `[11]` | Jones et al. (2022), Angular Velocities... | Jones et al., ACLR 후 knee biomechanics의 proxy measure로 IMU-derived angular velocity/linear acceleration을 다룬 논문. | `docs/ref_papers/03_wearable_imu_and_portable_sensing/Angular Velocities...pdf` | 적절합니다. IMU correlation claim을 직접 뒷받침합니다. |
| `[12]` | Krishnakumar et al. (2024), Estimation of Kinetics Using IMUs... | Krishnakumar et al., ACL rehabilitation decision support에서 IMU 기반 kinetics estimation을 다룬 systematic review. | `docs/ref_papers/03_wearable_imu_and_portable_sensing/Estimation of Kinetics Using IMUs...pdf` | 적절합니다. review 근거로 직접 사용 가능합니다. |
| `[13]` | Tedesco et al. (2020), A Wearable System for Gait Training... | Tedesco et al., `Motion Sensors-Based Machine Learning Approach for the Identification of Anterior Cruciate Ligament Gait Patterns in On-the-Field Activities in Rugby Players`, DOI `10.3390/s20113029`. | 웹: MDPI/PubMed | 제목 불일치가 있습니다. 현재 제목은 다른 Sensors 논문 제목으로 보입니다. |
| `[14]` | Yuan et al. (2026), IMU-Based Running Gait Analysis... | Yuan et al., `Optimizing wearable IMU configurations for running gait analysis: a machine learning-based sensor fusion approach`, DOI `10.3389/fbioe.2026.1762919`. | 웹: Frontiers/PubMed/PMC | 제목이 느슨하게 요약되어 있습니다. ACL-specific 논문은 아니며, 현재 HTML 본문에서는 참고문헌 목록에만 등장합니다. |
| `[15]` | Hall, Stevermer & Gillette (2012), Gait Analysis Post ACL Reconstruction... | Hall et al., `Gait analysis post anterior cruciate ligament reconstruction: Knee osteoarthritis perspective`. | `docs/ref_papers/02_acl_gait_reviews_meta_analyses/Gait analysis post anterior cruciate ligament reconstruction - Knee osteoarthritis perspective.pdf` | OA perspective 근거로는 적절합니다. 다만 세포 대사까지 포함한 상세 claim에는 부분적 근거입니다. |
| `[16]` | Mpaliotes, Vasileios (2022), IMU ACL thesis | 권위 있는 정확 일치 자료를 찾지 못했습니다. 가능한 대체 후보는 Yasir S. Alshehri (2019)의 post-ACLR inertial sensors 관련 KU dissertation입니다. | 웹: KU ScholarWorks 후보만 확인 | 미확인 항목입니다. 정확한 파일 또는 출처가 확인되기 전까지 사용하지 않는 편이 안전합니다. |
| `[17]` | Hur et al. (2025), transfer-learning IMU kinematics | Hur et al., `Learning based lower limb joint kinematic estimation using open source IMU data`, DOI `10.1038/s41598-025-89716-4`. | 웹: Nature/PubMed/PMC | 제목 불일치가 있습니다. IMU 기반 kinematic estimation은 뒷받침하지만 ACL-specific 근거는 아닙니다. 참고문헌 목록에만 있는 항목입니다. |
| `[18]` | Davis-Wilson et al. (2020), Bilateral Gait... | Davis-Wilson et al., `Bilateral Gait Six and Twelve Months Post-Anterior Cruciate Ligament Reconstruction Compared With Controls`. | `docs/ref_papers/01_acl_gait_biomechanics_studies/Bilateral Gait Six and Twelve Months...pdf` | 좋은 근거 자료이지만 현재 HTML 본문에는 직접 인용되지 않았습니다. bilateral/stiff-gait section을 강화하는 데 사용할 수 있습니다. |
| `[19]` | Wellsandt et al. (2016), Decreased Knee Joint Loading... | Wellsandt et al., `Decreased Knee Joint Loading Associated With Early Knee Osteoarthritis After ACL Injury`, DOI `10.1177/0363546515608475`. | 웹: PubMed/PMC/SAGE | 적절한 자료입니다. 다만 문장 표현은 인과가 아니라 연관성으로 써야 합니다. |
| `[20]` | Di Stasi et al. (2013), RTS criteria | Di Stasi et al., `Gait Patterns Differ Between ACL-Reconstructed Athletes Who Pass Return-to-Sport Criteria and Those Who Fail`. | `docs/ref_papers/05_return_to_sport_and_functional_tests/Gait Patterns Differ...pdf` | RTS와 gait pattern 관계 근거로 적절합니다. |
| `[21]` | Lee & Park (2020), single-IMU lower-limb kinetics | Lee & Park, `Estimation of Three-Dimensional Lower Limb Kinetics Data during Walking Using Machine Learning from a Single IMU Attached to the Sacrum`, DOI `10.3390/s20216277`. | 웹: MDPI/PubMed | single-IMU kinetics estimation 근거로 적절합니다. 다만 ACL-specific 논문은 아닙니다. |
| `[22]` | Sigward et al. (2016), pelvis/trunk valgus loading | 가장 가까운 ACL 관련 자료는 Sigward, Lin & Pratt, `Knee loading asymmetries during gait and running in early rehabilitation following ACL reconstruction`, DOI `10.1016/j.clinbiomech.2015.11.003`입니다. | 웹: Clinical Biomechanics/PubMed | 서지 불일치가 있습니다. 교체하거나 삭제해야 합니다. |

## PPT 참고문헌 검토

인용이 추가된 PPT는 현재 자체 참고문헌 번호 `[1]-[8]`을 사용합니다. 이 번호는 HTML의 `[1]-[22]` 번호와 일치하지 않습니다.

| PPT 번호 | 현재 PPT에서 의도한 출처 | 확인된 자료 | 판정 |
|---|---|---|---|
| `[1]` | Gao & Zheng (2014), progressive changes | 로컬 PDF는 Slater, Hart, Kelly & Kuenze (2017), `Progressive Changes in Walking Kinematics and Kinetics After ACL Injury and Reconstruction`, Journal of Athletic Training, DOI `10.4085/1062-6050-52.6.06`입니다. | PPT 스크립트와 reference slide의 서지가 잘못되어 있습니다. 저자와 연도를 수정해야 합니다. |
| `[2]` | Tan et al. (2022), out-of-lab ACL portable sensing | 로컬 PDF는 2022 medRxiv preprint입니다. peer-reviewed version은 Tan et al. (2023), `A scoping review of portable sensing for out-of-lab anterior cruciate ligament injury prevention and rehabilitation`, npj Digital Medicine, DOI `10.1038/s41746-023-00782-2`입니다. | 가능하면 peer-reviewed 2023 버전을 사용하는 것이 좋습니다. |
| `[3]` | Hart et al. (2016), knee kinematics/joint moments after ACLR | 로컬 PDF는 Hart et al.의 BJSM systematic review/meta-analysis와 일치합니다. | 적절합니다. ACLR 후 10-20년 내 50% 이상 OA 및 gait biomechanics 근거로 강합니다. |
| `[4]` | COMPWALK-ACL | HTML `[9]`와 같은 자료입니다. | 적절합니다. |
| `[5]` | Lisee daily steps | HTML `[6]`와 같은 자료입니다. | ACLR daily steps/loading 근거로는 적절하지만, healthy-adult steps 근거로는 부적절합니다. |
| `[6]` | Büttner waveform | HTML `[1]`과 같은 자료입니다. | 적절합니다. |
| `[7]` | Krishnakumar IMU kinetics review | HTML `[12]`와 같은 자료입니다. | 적절합니다. |
| `[8]` | Kokkotis explainable ML | 로컬 PDF는 Kokkotis et al. (2022), Scientific Reports, DOI `10.1038/s41598-022-10666-2`와 일치합니다. | 적절합니다. SHAP/explainable ML 및 94.95% SVM claim을 직접 뒷받침합니다. |

## 권장 통합 참고문헌 전략

최종 deck의 일관성을 위해 PPT와 HTML의 번호 체계를 따로 두기보다 하나의 통합 참고문헌 목록을 사용하는 것이 좋습니다.

최소한 안전하게 유지할 수 있는 통합 목록:

1. Slater, Hart, Kelly & Kuenze (2017): ACL injury/ACLR burden 및 progressive walking biomechanics 근거.
2. Tan et al. (2023): portable sensing, reinjury, OA burden, out-of-lab assessment 근거.
3. Hart et al. (2016): ACLR 후 knee kinematics/joint moments 및 OA 근거.
4. Yona et al. (2026): COMPWALK-ACL multi-pace IMU dataset 근거.
5. Lisee et al. (2022): daily steps와 ACLR gait biomechanics 근거.
6. Büttner et al. (2024/2025): bilateral waveform/stance-phase analysis 근거.
7. Garcia et al. (2021) 및 Lai et al. (2024): speed-dependent asymmetry 근거.
8. Krishnakumar et al. (2024), Jones et al. (2022), Lee & Park (2020): IMU kinetics/biomechanics estimation 근거.
9. Di Stasi et al. (2013): RTS criteria와 gait pattern differences 근거.
10. Wellsandt et al. (2016): ACL injury 후 lower knee loading과 early OA의 연관성 근거.
11. Kokkotis et al. (2022): explainable ML/SHAP ACL gait classification 근거.
12. Tedesco et al. (2020): wearable IMU + ML 기반 ACL gait pattern identification 근거. 단, 제목은 수정해야 합니다.

검증 전까지 삭제하거나 별도로 격리할 항목:

- HTML `[16]` Mpaliotes/Vasileios 2022.
- HTML `[22]` 현재 Sigward pelvis/trunk title.
- HTML `[13]` 현재 제목 문자열.

## 출처 확인 메모

로컬 근거 파일은 `docs/ref_papers/` 아래 PDF를 `/tmp/citation_audit_text/`로 텍스트 변환해 검색했습니다. 로컬 PDF가 없거나 서지가 의심되는 항목은 ScienceDirect, MDPI, Frontiers, Nature Scientific Reports, PubMed/PMC, SAGE, KU ScholarWorks, Clinical Biomechanics/PubMed 같은 공식 또는 색인 출처로 확인했습니다.

외부 확인 링크:

- Zhu et al. (2025), ScienceDirect: `https://www.sciencedirect.com/science/article/pii/S2352648325000364`
- Tedesco et al. (2020), MDPI: `https://www.mdpi.com/1424-8220/20/11/3029`
- Yuan et al. (2026), Frontiers: `https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2026.1762919/full`
- Hur et al. (2025), Nature Scientific Reports: `https://www.nature.com/articles/s41598-025-89716-4`
- Wellsandt et al. (2016), PubMed: `https://pubmed.ncbi.nlm.nih.gov/26493337/`
- Lee & Park (2020), MDPI: `https://www.mdpi.com/1424-8220/20/21/6277`
- Sigward, Lin & Pratt (2016), PubMed: `https://pubmed.ncbi.nlm.nih.gov/26640045/`
- Tan et al. (2023), npj Digital Medicine: `https://www.nature.com/articles/s41746-023-00782-2`
- Kokkotis et al. (2022), Scientific Reports: `https://www.nature.com/articles/s41598-022-10666-2`
