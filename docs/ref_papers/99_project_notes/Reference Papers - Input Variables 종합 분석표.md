# Reference Papers — Input Variables 종합 분석표

> 생성일: 2026-03-28
> 대상 폴더: `docs/ref_papers/`
> 논문 수: 11편

---

## 목차

1. [파일명 변경 내역](#파일명-변경-내역)
2. [논문별 상세 분석](#논문별-상세-분석)
   - [P1 · Goetschius 2018](#p1--goetschius-2018)
   - [P2 · Ursei 2020](#p2--ursei-2020)
   - [P3 · Erhart-Hledik 2018](#p3--erhart-hledik-2018)
   - [P4 · Krishnan 2022](#p4--krishnan-2022)
   - [P5 · Garcia 2022](#p5--garcia-2022)
   - [P6 · Hwang 2024](#p6--hwang-2024)
   - [P7 · Yuan 2026](#p7--yuan-2026)
   - [P8 · Voisard 2025](#p8--voisard-2025)
   - [P9 · Yona 2026](#p9--yona-2026)
   - [P10 · Hur 2025](#p10--hur-2025)
   - [P11 · Palazzo 2025](#p11--palazzo-2025)
3. [전체 Input Variables 통합 비교표](#전체-input-variables-통합-비교표)
4. [중요 Feature 요약 (결론/Discussion 기반)](#중요-feature-요약)

---

## 논문별 상세 분석

### P1 · Goetschius 2018

**제목:** Gait Biomechanics in Anterior Cruciate Ligament–reconstructed Knees at Different Time Frames Postsurgery
**저자:** John Goetschius, Jay Hertel, Susan A. Saliba, Stephen F. Brockmeier, Joseph M. Hart
**저널:** Medicine & Science in Sports & Exercise, Vol. 50, No. 11, pp. 2209–2216
**연도:** 2018 · DOI: 10.1249/MSS.0000000000001693
**파일:** `Goetschius_2018_gait_biomechanics_ACLR_time_frames.pdf`

#### Input Variables

| 변수                           | 카테고리              | 단위         | 설명                        |
| ------------------------------ | --------------------- | ------------ | --------------------------- |
| External knee flexion moment   | Kinetics (Sagittal)   | N·m·kg⁻¹·m⁻¹ | 슬관절 굴곡 외부 모멘트     |
| External knee extension moment | Kinetics (Sagittal)   | N·m·kg⁻¹·m⁻¹ | 슬관절 신전 외부 모멘트     |
| External knee adduction moment | Kinetics (Frontal)    | N·m·kg⁻¹·m⁻¹ | 슬관절 내전 외부 모멘트     |
| External knee abduction moment | Kinetics (Frontal)    | N·m·kg⁻¹·m⁻¹ | 슬관절 외전 외부 모멘트     |
| External hip adduction moment  | Kinetics (Frontal)    | N·m·kg⁻¹·m⁻¹ | 고관절 내전 외부 모멘트     |
| External hip abduction moment  | Kinetics (Frontal)    | N·m·kg⁻¹·m⁻¹ | 고관절 외전 외부 모멘트     |
| Knee flexion motion            | Kinematics (Sagittal) | degrees      | 슬관절 굴곡 각도            |
| Knee abduction motion          | Kinematics (Frontal)  | degrees      | 슬관절 외전 각도            |
| Hip abduction motion           | Kinematics (Frontal)  | degrees      | 고관절 외전 각도            |
| vGRF                           | Ground Reaction Force | N·kg⁻¹       | 수직 지면반력 (체중 정규화) |

**원문 발췌 — 변수 정의 (Methods, "Data collection", p. 2210):**

> "Primary gait variables included knee and hip kinetics and kinematics in the sagittal and frontal planes and vertical ground reaction forces (vGRF)."

**원문 발췌 — 단위 설명 (Methods, "Data processing", p. 2210):**

> "Vertical ground reaction forces were reported in newtons normalized by body mass (N·kg⁻¹), kinematic variables were reported in rotational degrees, and kinetic variables were reported in external moments, newton-meters (N·m) normalized to body mass and height (N·m·kg⁻¹·m⁻¹)."

**원문 발췌 — 측정 조건 (p. 2210):**

> "Walking and jogging trials were performed at standardized speeds of 1.34 m·s⁻¹ (3.0 mph) and 2.68 m·s⁻¹ (6.0 mph), respectively."

**측정 장비:** 12× Bonita-10 cameras (Vicon), split-belt instrumented treadmill (Bertec), Motion Monitor + Nexus software

#### 결론 및 주요 Feature

**원문 발췌 — 결과 요약 (Abstract - Results):**

> "Early ACLR group demonstrated reduced knee flexion, knee extension, knee adduction, and hip adduction moments on the ACLR limb. Mid ACLR group demonstrated no gait differences between limbs or other groups. Late ACLR group demonstrated reduced knee flexion moments, and greater knee and hip adduction moments in their ACLR limb."

**원문 발췌 — "Quadriceps Avoidance" 패턴 (Discussion, p. 2214):**

> "The presence of the 'quadriceps avoidance' pattern in the late ACLR group may be related to an inability of the quadriceps muscles to eccentrically absorb joint loads. Theoretically, an inability to adequately attenuate forces on the knee could lead to progressive degradation of articular surfaces."

**원문 발췌 — 결론 (Conclusions, p. 2215):**

> "The early ACLR group demonstrated lower sagittal and frontal plane joint loading on the ACLR limb compared with contralateral and control limbs. The late ACLR group demonstrated lower sagittal plane joint loading compared with control limbs and greater frontal plane joint loading compared with contralateral and control limbs."

**핵심 중요 변수:** vGRF, external knee extension moment, knee flexion motion, external knee adduction moment, hip abduction motion (Early ACLR Walking에서 유의한 집단 내 비대칭 차이 확인)

---

### P2 · Ursei 2020

**제목:** Foot and ankle compensation for anterior cruciate ligament deficiency during gait in children
**저자:** Monica E. Ursei, Franck Accadbled, Marino Scandella, Gorka Knorr, Caroline Munzer, Pascal Swider, Jérome Briot, Jérome Sales de Gauzy
**저널:** Orthopaedics & Traumatology: Surgery & Research, Vol. 106, pp. 179–183
**연도:** 2020 · DOI: 10.1016/j.otsr.2019.07.009
**파일:** `Ursei_2020_foot_ankle_compensation_ACL_deficiency_children.pdf`

#### Input Variables

| 변수                                 | 카테고리           | 측정 시점 (Gait Cycle %)  | 단위      |
| ------------------------------------ | ------------------ | ------------------------- | --------- |
| Ankle dorsiflexion / plantar flexion | Kinematics (Ankle) | 0%, 25%, 60%, 83% GC      | degrees   |
| External foot progression angle      | Kinematics (Foot)  | 25% (stance), 70% (swing) | degrees   |
| Walking speed                        | Spatiotemporal     | —                         | m/s       |
| Cadence                              | Spatiotemporal     | —                         | steps/min |
| Stride length                        | Spatiotemporal     | —                         | m         |
| Step length                          | Spatiotemporal     | —                         | m         |
| Single-leg stance (%)                | Spatiotemporal     | —                         | % of GC   |
| Double-leg stance (%)                | Spatiotemporal     | —                         | % of GC   |

**원문 발췌 — 측정 변수 정의 (Methods, p. 180):**

> "The following spatiotemporal parameters were recorded: speed, cadence, stride and step length, and percentages of time spent in single-leg and double-leg stance. Kinematic data on the ankle and foot in the sagittal plane were compared to those obtained in a control group of 37 healthy children."

**원문 발췌 — 발목 각도 측정 시점 (p. 180):**

> "Ankle flexion and extension were computed at the following points of the GC: initial contact (0%), mid-stance (25%), terminal stance (60%), and swing phase (83%). The foot progression angle was determined at mid-stance (25%) and during the swing phase (70%)."

**측정 장비:** Vicon 360 (5× 60-Hz IR cameras), 15 reflective markers (Helen Hayes model)

#### 결론 및 주요 Feature

**주요 결과 (Abstract — Results):**

> "Compared to the reference values, the ankle was in plantar flexion at initial contact in 41 patients, and ankle dorsiflexion during the stance phase was diminished in 39 patients."

**수치 결과 (Table 1, p. 181):**

| 변수                             | 환자군 (n=47)  | 대조군 (n=37)  | p값        |
| -------------------------------- | -------------- | -------------- | ---------- |
| Ankle — 0% GC (Initial contact)  | −3.44 ± 3.54°  | 0.74 ± 3.67°   | 2.55×10⁻⁹  |
| Ankle — 25% GC (Mid-stance)      | 5.91 ± 2.15°   | 7.96 ± 2.52°   | 1.28×10⁻¹¹ |
| Ankle — 60% GC (Terminal stance) | −14.18 ± 6.40° | −12.99 ± 6.60° | >0.05      |
| Ankle — 83% GC (Swing)           | 0.78 ± 2.72°   | 1.09 ± 3.43°   | >0.05      |

**원문 발췌 — 결론 (p. 182):**

> "Our findings confirm the hypothesis that children alter their gait to compensate for ACL deficiency. In particular, plantar flexion of the ankle at initial contact was noted. Compensatory behaviours during gait may be difficult to detect during the physical examination. Gait analysis is a reliable tool for elucidating compensatory mechanisms and optimising rehabilitation therapy."

**핵심 중요 변수:** **Ankle dorsiflexion at initial contact (0% GC)**, **ankle dorsiflexion at mid-stance (25% GC)** — 통계적으로 가장 유의한 ACL 결핍 지표

---

### P3 · Erhart-Hledik 2018

**제목:** Longitudinal changes in knee gait mechanics between 2 and 8 years after anterior cruciate ligament reconstruction
**저자:** Jennifer C. Erhart-Hledik, Constance R. Chu, Jessica L. Asay, Thomas P. Andriacchi
**저널:** Journal of Orthopaedic Research, Vol. 36, No. 5, pp. 1478–1486
**연도:** 2018 · DOI: 10.1002/jor.23770
**파일:** `Erhart-Hledik_2018_longitudinal_knee_gait_mechanics_ACLR.pdf`

#### Input Variables

##### 관절 Kinetics (역학적 변수)

| 변수                             | 약어        | 설명                        |
| -------------------------------- | ----------- | --------------------------- |
| Peak knee flexion moment         | KFM         | 최대 슬관절 굴곡 모멘트     |
| Knee adduction moment (1st peak) | KAM1        | 슬관절 내전 모멘트 1차 피크 |
| Knee adduction moment (2nd peak) | KAM2        | 슬관절 내전 모멘트 2차 피크 |
| Knee adduction moment impulse    | KAM Impulse | 슬관절 내전 모멘트 충격량   |
| Knee extension moment (1st peak) | KEM1        | 슬관절 신전 모멘트 1차 피크 |
| Knee extension moment (2nd peak) | KEM2        | 슬관절 신전 모멘트 2차 피크 |
| Peak internal rotation moment    | KIRM        | 슬관절 내회전 모멘트        |
| Peak external rotation moment    | KERM        | 슬관절 외회전 모멘트        |

**원문 발췌 — Kinetics 변수 정의 (Gait Analysis, p. 4):**

> "For kinetic analyses, our outcome variables included the peak knee flexion moment (KFM), knee adduction moment first peak (KAM1), second peak (KAM2), and impulse (KAM Impulse), first and second peak extension moments (KEM1 and KEM2), and peak internal (KIRM) and external (KERM) rotation moments."

##### 관절 Kinematics (운동학적 변수)

| 변수                                  | 측정 구간                            | 설명                      |
| ------------------------------------- | ------------------------------------ | ------------------------- |
| Peak knee flexion angle               | Stance 전반부                        | 스탠스 전반부 최대 굴곡각 |
| Minimum knee flexion angle            | Stance 후반부                        | 스탠스 후반부 최소 굴곡각 |
| Peak varus angle                      | Stance 전반부                        | 최대 내반 각도            |
| Peak valgus angle                     | Stance 후반부                        | 최대 외반 각도            |
| Average external rotation angle       | Whole stance (heel-strike → toe-off) | 평균 외회전각             |
| Average anterior femoral displacement | Whole stance                         | 평균 대퇴골 전방 변위     |

**원문 발췌 — Kinematics 변수 정의 (Gait Analysis, p. 4–5):**

> "For kinematic analyses, our outcome variables included the peak knee flexion angle during the first half of stance phase, minimum knee flexion angle in the second half of stance phase, peak varus angle in the first half of stance phase, peak valgus angle in the second half of stance phase, average external rotation angle during stance (heel-strike to toe-off), and average anterior femoral displacement during stance (heel-strike to toe-off)."

##### Spatiotemporal Variables

| 변수          | 설명         |
| ------------- | ------------ |
| Walking speed | 보행 속도    |
| Step length   | 한 걸음 길이 |

**측정 장비:** Qualisys Medical optoelectronic system (120 Hz), Bertec force plate, BioMove software (Stanford), Cardan/joint coordinate system 기반 3D kinematics

#### 결론 및 주요 Feature

**원문 발췌 — Kinetics 결과 (Results - Kinetics, p. 5):**

> "Over the follow-up period, there was a reduction in the magnitude of KEM1 (p=0.048) and KEM2 (p<0.001), while an increase in peak KFM (p=0.002) was seen. KAM1 (p=0.009), KAM2 (p=0.009), KAM Impulse (p=0.004), and KIRM (p<0.001) decreased over follow-up."

**원문 발췌 — Kinematics 결과 (Results - Kinematics, p. 6):**

> "Over the follow-up period, there was a significant increase in peak knee flexion angle in the first half of stance phase (p=0.026) and in the minimum knee flexion angle in the second half of stance phase (p<0.001). The average external rotation angle increased from 2 to 8 years post-ACLR (p=0.007), while a reduction in average anterior femoral displacement was observed (p=0.006)."

**원문 발췌 — Discussion (p. 6):**

> "The results of this study demonstrate that there are longitudinal changes from 2 to 8 years post-ACLR in knee joint kinetic and kinematic features that have been related to clinical patient-reported outcomes after ACLR and the progression of knee OA."

**종단 변화 방향 요약:**

| 변수                                 | 2→8년 변화       | p값    |
| ------------------------------------ | ---------------- | ------ |
| KFM                                  | 증가 (호전)      | 0.002  |
| KEM1                                 | 감소             | 0.048  |
| KEM2                                 | 감소             | <0.001 |
| KAM1                                 | 감소             | 0.009  |
| KIRM                                 | 감소             | <0.001 |
| Peak flexion angle (1st half stance) | 증가 (호전)      | 0.026  |
| External rotation angle              | 증가 (악화 가능) | 0.007  |

---

### P4 · Krishnan 2022

**제목:** Mechanical Factors Contributing to Altered Knee Extension Moment during Gait after ACL Reconstruction: A Longitudinal Analysis
**저자:** Chandramouli Krishnan, Alexa K. Johnson, Riann M. Palmieri-Smith
**저널:** Medicine & Science in Sports & Exercise, Vol. 54, No. 12, pp. 2208–2215
**연도:** 2022 · DOI: 10.1249/MSS.0000000000003014
**파일:** `Krishnan_2022_mechanical_factors_knee_extension_moment_ACLR.pdf`

#### Input Variables (Predictors)

| 변수                                  | 카테고리   | 측정 구간                | 단위          |
| ------------------------------------- | ---------- | ------------------------ | ------------- |
| Knee flexion angle at initial contact | Kinematics | Initial contact          | degrees       |
| Peak knee flexion angle               | Kinematics | Early stance (first 50%) | degrees       |
| Peak vGRF                             | Kinetics   | Early stance             | % body weight |

**원문 발췌 — Predictor 정의 (Abstract / Methods, p. 1):**

> "knee flexion angle at initial contact, peak knee flexion angle, and vertical ground reaction force (vGRF) contribute to knee extension moments during gait in individuals with anterior cruciate ligament (ACL) reconstruction."

**원문 발췌 — 변수 측정 (Gait Biomechanics Testing, p. 4):**

> "Knee flexion angle at initial contact, peak knee flexion angle, peak vGRF, and peak knee extension moment were calculated for each limb during the early stance phase of gait for all three timepoints."

**원문 발췌 — vGRF 정규화 (p. 4):**

> "vGRF was normalized to the participant's body weight. All biomechanical data were time normalized to the stance phase of the gait cycle."

**원문 발췌 — 역학적 중요성 근거 (Introduction, p. 2):**

> "mechanical factors such as knee flexion angle and vertical ground reaction forces (vGRFs) could directly influence knee joint loading, as they serve as primary inputs to the calculation of knee joint moments and contact forces."

#### Output Variable (Target)

| 변수                       | 설명                                   |
| -------------------------- | -------------------------------------- |
| Peak knee extension moment | 최대 슬관절 신전 모멘트 (Early stance) |

**원문 발췌 — 회귀 모델 정의 (Statistical Analyses, p. 4):**

> "Stepwise multiple linear regression analysis was used to determine the contribution of knee flexion angle at initial contact, peak knee flexion angle during early stance, and peak vGRF during early stance to knee joint loading (i.e., peak knee moment) during the early stance phase of gait."

**측정 장비:** 15-camera MX-13 Vicon (240 Hz), AMTI OR6-7 force plates (1200 Hz), 34개 14mm 반사 마커, Visual 3D software (v3.90)

#### 결론 및 주요 Feature

**원문 발췌 — 설명력 (Discussion, p. 6):**

> "these variables explained more than 75% of the variance in the peak knee extension moment data in both the ACL-reconstructed and the non-ACL-reconstructed limbs... peak knee flexion angle and peak vGRF symmetry values still explained more than 67% of the variance in peak knee extension moment symmetry."

**원문 발췌 — 결론 (Abstract Conclusions, p. 2):**

> "Standardized beta coefficients indicated that changes in knee flexion angle had a greater impact (>2x) on knee extension moments than vGRF at both timepoints in both limbs (βvGRF = 0.204–0.309; βkneeflexion = 0.703–0.831)."

**회귀 모델 성능 요약:**

| 분석 조건                        | R²    | BF₁₀  |
| -------------------------------- | ----- | ----- |
| Reconstructed limb (raw)         | 0.767 | >1000 |
| Non-reconstructed limb (raw)     | 0.815 | >1000 |
| Symmetry values                  | 0.673 | >1000 |
| Change scores (TP2−TP1, ACL)     | 0.775 | >1000 |
| Change scores (TP3−TP1, non-ACL) | 0.883 | >1000 |

**핵심 중요 변수:** **Peak knee flexion angle** (β = 0.703–0.831, vGRF의 2배 이상 영향력)

---

### P5 · Garcia 2022

**제목:** Gait Asymmetries are Exacerbated at Faster Walking Speeds in Individuals with Acute Anterior Cruciate Ligament Reconstruction
**저자:** Steven A. Garcia, Scott R. Brown, Mary Koje, Chandramouli Krishnan, Riann M. Palmieri-Smith
**저널:** Journal of Orthopaedic Research, Vol. 40, No. 1, pp. 219–230
**연도:** 2022 · DOI: 10.1002/jor.25117
**파일:** `Garcia_2022_gait_asymmetries_walking_speed_ACLR.pdf`

#### Input Variables

| 변수                            | 카테고리              | 설명                                  | 단위          |
| ------------------------------- | --------------------- | ------------------------------------- | ------------- |
| Vertical GRF (vGRF)             | Ground Reaction Force | 수직 지면반력 파형                    | % body weight |
| Posterior-Anterior GRF (PA-GRF) | Ground Reaction Force | 전후방향 지면반력 파형                | % body weight |
| GRF Asymmetry (vGRF)            | Derived               | Injured limb GRF − Uninjured limb GRF | % body weight |
| GRF Asymmetry (PA-GRF)          | Derived               | Injured limb GRF − Uninjured limb GRF | % body weight |
| Walking speed condition         | Experimental factor   | 80%, 100%, 120% self-selected speed   | m/s           |

**원문 발췌 — 측정 변수 (Abstract, p. 1):**

> "Bilateral vertical and posterior-anterior GRFs were recorded at each speed."

**원문 발췌 — GRF 정규화 (Data Reduction, p. 5):**

> "Ensemble GRF curves were normalized to percent body weight (% BW) by dividing each element of the GRF waveform by the participant's body weight in Newtons and multiplying by 100."

**원문 발췌 — 비대칭 산출 방식 (Data Reduction, p. 5):**

> "For ACL-reconstructed participants, the uninjured limb GRF was subtracted from the injured limb GRF across each percent of the stance phase."

**원문 발췌 — PA-GRF 연구 동기 (Introduction, p. 3):**

> "evaluation of other components of the GRF during walking, such as the posterior-anterior GRF, have received far less attention in the literature despite evidence suggesting this parameter may provide additional insight into the extent of biomechanical asymmetries present after surgery."

**측정 장비:** Bertec split-belt treadmill (ITC-11-20L/R), LabVIEW 2011, Butterworth filter (8th order, 500 Hz lowpass)
**통계:** Statistical Parametric Mapping (SPM1D), 2×3 repeated-measures ANOVA (group × speed)

#### 결론 및 주요 Feature

**원문 발췌 — 주요 결과 (Discussion, p. 7):**

> "We found that the asymmetry in both vertical and posterior-anterior GRFs increased at faster walking speeds but decreased at slower speeds in individuals with ACL reconstruction. Conversely, speed did not affect GRF asymmetries in uninjured controls."

**원문 발췌 — 결론 (Conclusion, p. 10):**

> "Increased walking speed magnifies biomechanical asymmetries in the vertical and posterior-anterior GRF in individuals with ACL reconstruction. At slower speeds, individuals with ACL reconstruction can walk with more symmetrical GRFs (i.e., decreased interlimb differences) compared with self-selected and fast speeds."

**수치 결과:**

| 조건                          | vGRF 비대칭       | 통계                   |
| ----------------------------- | ----------------- | ---------------------- |
| 80% 선호 속도                 | ~10% BW           | 유의                   |
| 120% 선호 속도                | ~18–20% BW        | 유의                   |
| ACL군 Group×Speed interaction | —                 | F\*[2,86]=7.07, p<0.05 |
| 정상 대조군                   | 속도 간 차이 없음 | p>0.05                 |

**핵심 중요 변수:** **vGRF asymmetry**, **PA-GRF asymmetry** — 속도가 증가할수록 비대칭 악화

---

### P6 · Hwang 2024

**제목:** Machine learning models for predicting return to sports after anterior cruciate ligament reconstruction: Physical performance in early rehabilitation
**저자:** Ui-jae Hwang, Jin-seong Kim, Keong-yoon Kim, Kyu-sung Chung
**저널:** DIGITAL HEALTH (Sage)
**연도:** 2024 · DOI: 10.1177/20552076241299065
**파일:** `Hwang_2024_ML_predicting_RTS_ACLR.pdf`

#### Input Variables (수술 후 3개월 측정, 8개 변수)

| 변수                                    | 측정 도구             | 카테고리               |
| --------------------------------------- | --------------------- | ---------------------- |
| Age                                     | 인구통계              | Demographic            |
| BMI                                     | 인구통계              | Demographic            |
| 60°/s knee extensor peak torque (PT)    | HUMAC-NORM isokinetic | Muscle strength        |
| 60°/s knee flexor peak torque (PT)      | HUMAC-NORM isokinetic | Muscle strength        |
| 180°/s knee extensor average power (AP) | HUMAC-NORM isokinetic | Muscle power           |
| 180°/s knee flexor average power (AP)   | HUMAC-NORM isokinetic | Muscle power           |
| YBT total score                         | Y-Balance Test        | Dynamic balance        |
| BBS overall index                       | Biodex Balance System | Static/dynamic balance |

**원문 발췌 — 예측 변수 목록 (Pre-processing section, p. 4):**

> "This study included eight numerical predictors (age, body mass index [BMI], 60°/s knee extensor PT, 60°/s knee flexor PT, 180°/s knee extensor AP, 180°/s knee flexor AP, YBT total score, and BBS overall index)."

**원문 발췌 — Biodex Balance System 설명 (Methods, "Biodex balance system test", p. 2):**

> "Postural stability was measured using the BBS (Biodex Medical Systems Inc., Shirley, NY, USA). The BBS electronically generates the anteroposterior (sagittal plane), mediolateral (frontal plane), and overall indexes. This study's overall index included the anteroposterior (sagittal plane) and mediolateral (frontal plane) indexes in detail."

**원문 발췌 — Y-Balance Test 설명 (Methods, "Y-balance test", p. 3):**

> "Dynamic balance was measured using the YBT (Move2Perform; Evansville, IN, USA). The patients were instructed to stand on their weight-bearing leg in a box and gently push the side of the box using the unsupported leg, reaching as far as possible in the anterior, posteromedial, and posterolateral directions. The average distances achieved in the three directions were analyzed to assess dynamic balance."

**원문 발췌 — 등속성 근력 설명 (Methods, "Isokinetic muscle strength test", p. 3):**

> "Isokinetic muscle strength was measured using the HUMAC-NORM isokinetic extremity system... The peak torque (PT) and the average power (AP) of the knee flexors and extensors were assessed at the 60°/s and 180°/s angular velocities, respectively."

#### Target Variables (수술 후 12개월 측정, RTS 결과)

| 변수                          | 설명                  |
| ----------------------------- | --------------------- |
| Single-leg hop test           | 편측 홉 거리          |
| Single-leg vertical jump test | 편측 수직 점프 높이   |
| Tegner activity score         | 스포츠 활동 수준 점수 |

#### 결론 및 주요 Feature

**모델 성능 요약:**

| Target                   | 최고 모델         | AUC (Test) | Accuracy | F1    |
| ------------------------ | ----------------- | ---------- | -------- | ----- |
| Single-leg hop test      | Random Forest     | **0.952**  | 0.850    | 0.880 |
| Tegner activity score    | Random Forest     | **0.949**  | 0.950    | 0.952 |
| Single-leg vertical jump | Gradient Boosting | **0.868**  | —        | —     |

**원문 발췌 — 주요 예측 변수 (Results, p. 5):**

> "RTS-single leg hop test was best predicted by random forest, with YBT total score, BMI, 60°/s knee extensor PT and 180°/s knee extensor AP in feature importance, and high 60°/s knee extensor PT, high YBT total score, high 180°/s knee extensor AP, low BMI in Shapley additive explanation being key factors."

**원문 발췌 — 결론 (Conclusion, p. 9):**

> "The findings highlighted modifiable factors such as 60°/s knee extensor PT, YBT total score, 180°/s knee extensor and flexor AP, BBS overall index, and BMI in predicting successful RTS."

**핵심 중요 변수 (SHAP 기반):** **YBT total score**, **60°/s knee extensor PT**, **180°/s knee extensor AP**, **BMI** (낮을수록 RTS 성공 예측)

---

### P7 · Yuan 2026

**제목:** Optimizing wearable IMU configurations for running gait analysis: a machine learning-based sensor fusion approach
**저자:** Ye Yuan, Yaohui Yu, Shanshan Cai, Weidong Cheng
**저널:** Frontiers in Bioengineering and Biotechnology
**연도:** 2026 · DOI: 10.3389/fbioe.2026.1762919
**파일:** `Yuan_2026_IMU_running_gait_ML_sensor_fusion.pdf`

#### Input Variables

##### 센서 배치 조건 (독립변수)

| 구성          | 위치                      | IMU 수 |
| ------------- | ------------------------- | ------ |
| Config 1 (C1) | Lumbar (L5/S1) only       | 1      |
| Config 2 (C2) | Bilateral Ankles only     | 2      |
| Config 3 (C3) | Lumbar + Bilateral Ankles | 3      |

**원문 발췌 — 구성 정의 (Methods, "Minimal configuration subset construction", p. 3):**

> "We tested three primary minimal configurations: Config 1 (C1): Lumbar-Only, using only the L5/S1 sensor (1 IMU); Config 2 (C2): Ankles-Only, using both ankle sensors (2 IMUs); and Config 3 (C3): Lumbar + Ankles, combining the L5/S1 and both ankle sensors (3 IMUs)."

##### 원시 신호 입력

**원문 발췌 — Raw signal (Methods, "Feature engineering", p. 3):**

> "Raw IMU signals (3-axis acceleration and 3-axis angular velocity) were processed using a sliding-window approach to transform high-frequency time-series data into a feature space suitable for regression."

##### 추출된 Feature 목록 (Time-domain + Frequency-domain)

**원문 발췌 — Feature 추출 (Methods, "Feature engineering", p. 4):**

> "Time-domain statistics, including the mean, standard deviation (STD), root mean square (RMS), minimum, maximum, peak-to-peak amplitude, skewness, kurtosis, and zero-crossing rate, were calculated to capture the signal intensity and morphological characteristics of the impact and swing phases. Complementing these, frequency-domain features—specifically dominant frequency, spectral energy, and spectral entropy—were extracted after applying a Fast Fourier Transform (FFT)."

| 도메인    | Feature                | 설명              |
| --------- | ---------------------- | ----------------- |
| Time      | Mean                   | 평균              |
| Time      | STD                    | 표준편차          |
| Time      | RMS                    | 제곱평균제곱근    |
| Time      | Min / Max              | 최솟값 / 최댓값   |
| Time      | Peak-to-peak amplitude | 피크-피크 진폭    |
| Time      | Skewness               | 왜도              |
| Time      | Kurtosis               | 첨도              |
| Time      | Zero-crossing rate     | 영점 교차율       |
| Frequency | Dominant frequency     | 지배 주파수       |
| Frequency | Spectral energy        | 스펙트럼 에너지   |
| Frequency | Spectral entropy       | 스펙트럼 엔트로피 |

**적용 축:** 가속도(Acc) X/Y/Z, 각속도(Gyro) X/Y/Z → sensor 위치별 조합

##### Target Variables (예측 대상)

**원문 발췌 — Target 정의 (Table 2, "Definitions of target running gait parameters", p. 4):**

| 파라미터                  | 단위      | 정의                                                            |
| ------------------------- | --------- | --------------------------------------------------------------- |
| Cadence                   | steps/min | Steps per minute                                                |
| Ground Contact Time (GCT) | ms        | Duration from initial foot-strike (IC) to toe-off (TO)          |
| Flight Time (FT)          | ms        | Duration from one foot's TO to the other foot's IC              |
| Vertical Oscillation (VO) | cm        | Peak-to-trough vertical displacement of L5/S1 during gait cycle |
| Gait Symmetry Index (SI)  | %         | Percentage difference between left and right GCT                |

#### 결론 및 주요 Feature

**Top 10 Features (GCT 예측, Config 1 기준, RFE 적용, Table 5, p. 8):**

| Rank | Feature             | Axis | Type  | Gini Importance |
| ---- | ------------------- | ---- | ----- | --------------- |
| 1    | Accel_Z_variance    | Z    | Accel | 0.187           |
| 2    | Accel_X_variance    | X    | Accel | 0.112           |
| 3    | Accel_Z_rms         | Z    | Accel | 0.091           |
| 4    | Gyro_Y_energy_0-5Hz | Y    | Gyro  | 0.075           |
| 5    | Accel_Z_mean        | Z    | Accel | 0.066           |
| 6    | Accel_X_rms         | X    | Accel | 0.051           |
| 7    | Gyro_Y_variance     | Y    | Gyro  | 0.040           |
| 8    | Accel_Z_kurtosis    | Z    | Accel | 0.032           |
| 9    | Accel_X_mean        | X    | Accel | 0.029           |
| 10   | Accel_Y_variance    | Y    | Accel | 0.025           |

**모델 성능 (Table 3, p. 6):**

| Target               | Config      | R²       | RMSE     | MAPE   |
| -------------------- | ----------- | -------- | -------- | ------ |
| Cadence              | C1 (Lumbar) | **0.99** | 1.15 spm | 0.85%  |
| Vertical Oscillation | C1          | **0.96** | 0.41 cm  | 4.12%  |
| Ground Contact Time  | C1          | **0.95** | 7.98 ms  | 4.88%  |
| Flight Time          | C1          | 0.91     | 10.12 ms | 6.15%  |
| Gait Symmetry Index  | C1          | 0.52     | 4.55     | 21.45% |
| Gait Symmetry Index  | C3          | **0.91** | 1.90     | 7.12%  |

**원문 발췌 — 결론 (Conclusion, p. 10):**

> "We confirmed that while a single lumbosacral IMU is surprisingly powerful, it is blind to asymmetry. We identified a three-sensor configuration (lumbosacral + bilateral ankles) as the minimal-optimal solution, capable of accurately predicting a comprehensive suite of global, temporal, and symmetry-based running gait parameters (R² > 0.91; MAPE < 8% for all)."

**원문 발췌 — Feature 중요성 해석 (Discussion, p. 8):**

> "The dynamics of the CoM, particularly its vertical and anteroposterior acceleration (which our feature importance analysis confirmed as critical), are a direct reflection of the forces produced by and acting upon the lower limbs."

**핵심 중요 변수:** **Accel_Z_variance** (Lumbar, Gini 0.187), **Accel_X_variance** — 수직·전후 가속도 분산이 GCT 예측의 핵심

---

### P8 · Voisard 2025

**제목:** A Dataset of Clinical Gait Signals with Wearable Sensors from Healthy, Neurological, and Orthopedic Cohorts
**저자:** Cyril Voisard, Rémi Barrois, Nicolas de l'Escalopier, Nicolas Vayatis, Pierre-Paul Vidal, Alain Yelnik, Damien Ricard, Laurent Oudre
**저널:** Scientific Data (Nature Portfolio), Vol. 12, Article 1674
**연도:** 2025 · DOI: 10.1038/s41597-025-05959-w
**파일:** `Voisard_2025_clinical_gait_signals_wearable_IMU_dataset.pdf`

#### Input Variables

##### Raw IMU 신호 변수 (Table 3, p. 7)

| 변수명 | 설명                     | 단위  |
| ------ | ------------------------ | ----- |
| Acc_X  | Acceleration, x-axis     | m/s²  |
| Acc_Y  | Acceleration, y-axis     | m/s²  |
| Acc_Z  | Acceleration, z-axis     | m/s²  |
| Gyr_X  | Angular velocity, x-axis | rad/s |
| Gyr_Y  | Angular velocity, y-axis | rad/s |
| Gyr_Z  | Angular velocity, z-axis | rad/s |

**원문 발췌 — Raw signal 설명 (Table 3, p. 6–7):**

> "The main data labels for both sensors are summarized in Table 3, with corresponding units."

##### 전처리 신호 변수 (Table 4, p. 7) — 4개 센서 × 9채널

**IMU 부착 위치:** HE (머리), LB (요추 L4/L5), LF (왼발 등쪽), RF (오른발 등쪽)

**원문 발췌 — 센서 위치 (p. 2–3):**

> "Four IMU devices were attached to the head (HE), lower back L4/L5 (LB), and on the dorsal face of each foot (LF for the left foot and RF for the right foot) of the participants."

| 변수 (센서별)                | 설명                      |
| ---------------------------- | ------------------------- |
| {HE/LB/LF/RF}\_Acc_X/Y/Z     | 원시 가속도 3축           |
| {HE/LB/LF/RF}\_FreeAcc_X/Y/Z | 중력 성분 제거 가속도 3축 |
| {HE/LB/LF/RF}\_Gyr_X/Y/Z     | 각속도 3축                |

**원문 발췌 — 전처리 변수 설명 (Table 4, p. 7):**

> "Main time-series labels for preprocessed data. The axes are given in the sensors reference frame."

##### 메타데이터 (임상/인구통계) 변수 (Table 5, p. 8–9)

**원문 발췌 — Metadata 설명 (p. 8):**

> "The first metadata are specific to the patient (epidemiological and demographic data) at the time of registration. They include the subject label, trial numbering, age, gender, height, weight, body mass index, laterality, group, pathology and clinical or radioclinical score."

| Key                              | 설명                           |
| -------------------------------- | ------------------------------ |
| age, gender, height, weight, BMI | 인구통계                       |
| pathology, group                 | 질환군 분류                    |
| evaluationScoreName / Value      | 코호트별 임상 점수             |
| TUG                              | Time-Up and Go test (초)       |
| visualGaitEvaluation             | 의사 보행 시각 평가 (0–4점)    |
| leftGaitEvents / rightGaitEvents | Toe-off, Heel-strike 샘플 위치 |
| uturnBoundaries                  | U턴 구간 경계                  |

##### 코호트별 임상 점수 (Table 2)

| 코호트 | 질환                   | 임상 점수        |
| ------ | ---------------------- | ---------------- |
| HS     | 정상                   | —                |
| HOA    | 고관절 골관절염        | WOMAC (/100)     |
| KOA    | 슬관절 골관절염        | WOMAC (/100)     |
| ACL    | ACL 재건               | IKDC (/100)      |
| CVA    | 뇌졸중                 | FMA-LE (/34)     |
| PD     | 파킨슨병               | UPDRS III (/108) |
| CIPN   | 화학요법 유발 신경병증 | TNSc (/28)       |
| RIL    | 하지 방사선            | OCS (/30)        |

##### 파생 시공간 보행 파라미터 (Table 10)

| 변수                          | 약어   | 단위 |
| ----------------------------- | ------ | ---- |
| Gait velocity                 | V      | m/s  |
| Step length                   | SteL   | m    |
| U-turn time                   | UtrT   | s    |
| Stride time                   | StrT   | s    |
| CV of stride time             | CVStrT | %    |
| Double stance time proportion | dstT   | %    |
| CV of double stance time      | CVdstT | %    |

**측정 장비:** XSens™ (±2000 deg/s, 100 Hz), Technoconcept® I4 Motion (100 Hz); Butterworth LPF (8th order, 14 Hz cutoff)

#### 결론 및 주요 Feature

**원문 발췌 — 데이터 품질 검증 (p. 11):**

> "Statistically, the healthy cohort's results align closely with normative literature values, showing no significant differences compared to published norms (p > 0.05, independent two-sample t-tests)."

**데이터 규모:** 260명, 1,356개 보행 시험, 총 11시간 이상, 9,492개 파일
**Gait event 탐지 정확도:** Technoconcept 99.2%, XSens 99.1%

**코호트별 보행 속도:**

| 코호트       | Gait speed (m/s) | Step length (m) |
| ------------ | ---------------- | --------------- |
| HS (Healthy) | 1.17 (0.22)      | 0.67 (0.10)     |
| ACL          | 0.98 (0.24)      | 0.62 (0.10)     |
| KOA          | 0.76 (0.21)      | 0.51 (0.12)     |

---

### P9 · Yona 2026

**제목:** The COMPWALK-ACL: A Dataset of Multi-pace IMU Gait Kinematics in Adolescents, Adults, and ACL Injured Patients
**저자:** Tomer Yona, Bezalel Peskin, Arielle Fischer
**저널:** Scientific Data (Nature Portfolio)
**연도:** 2026 · DOI: 10.1038/s41597-025-06307-8
**파일:** `Yona_2026_COMPWALK-ACL_multi-pace_IMU_gait_dataset.pdf`

#### Input Variables

##### 시공간 보행 파라미터

**원문 발췌 — 신뢰도 평가 변수 (Technical Validation, p. 4):**

> "Within-participant reliability was assessed by calculating the coefficient of variation (CV) for five spatiotemporal gait parameters: gait speed, cadence, stride length, step width, and stride/step time."

| 변수             | 단위      |
| ---------------- | --------- |
| Gait speed       | m/s       |
| Cadence          | steps/min |
| Stride length    | m         |
| Step width       | cm        |
| Stride/Step time | s         |

##### 하지 관절 키네마틱스 (Lower Limb Joint Kinematics)

**원문 발췌 — 데이터 구조 (Data Records, p. 3):**

> "The .xlsx file contains a flattened, tabular version of key time-series variables... 'Segment Orientation', provided in both Quaternion and Euler angle formats for rotational data; 'Segment Position', detailing the 3D coordinates of each segment; and sheets for 'Segment Velocity', 'Segment Acceleration', 'Segment Angular Velocity', and 'Segment Angular Acceleration'."

**원문 발췌 — 관절각 변수 (Data Records, p. 3):**

> "There are sheets for 'Joint Angles' and 'Ergonomic Joint Angles'... clear anatomical labels for joint angle variables, such as 'Left Ankle Dorsiflexion/Plantarflexion'... Furthermore, the data includes a sheet for the 'Center of Mass', which provides the position, velocity, and acceleration of the body's overall center of mass."

**포함 관절각 변수:**

| 관절  | 면       | 변수명 예시                       |
| ----- | -------- | --------------------------------- |
| Hip   | Sagittal | Hip Flexion/Extension             |
| Knee  | Sagittal | Knee Flexion/Extension            |
| Ankle | Sagittal | Ankle Dorsiflexion/Plantarflexion |

##### Raw IMU 신호 변수

**원문 발췌 — 센서 사양 (Equipment and calibration, p. 3):**

> "we recorded lower limb kinematics utilizing a model that consists of seven lightweight (16 grams) sensors, each containing a 3D accelerometer, a 3D gyroscope, and a 3D magnetometer. The recordings were captured at a frequency of 100 Hz."

**센서 부착 위치 (7개):** 양쪽 발, 양쪽 경골, 양쪽 대퇴, 골반

| 신호 종류                     | 설명                |
| ----------------------------- | ------------------- |
| 3D 가속도 (free acceleration) | 중력 보정 가속도    |
| 3D 각속도 (angular velocity)  | 회전 속도           |
| 3D 자기장 (magnetic field)    | 방향 추정 보조      |
| Segment 위치 / 속도 / 가속도  | 세그먼트별 3D 좌표  |
| Quaternion / Euler angle      | 방향 (orientation)  |
| Heel/Toe contact              | 이진 발 접지 이벤트 |

##### 메타데이터 (ID.csv)

**원문 발췌 — 메타데이터 키 (Data Records, p. 4):**

> "a metadata file titled ID.csv provides participant information, including: ID, Group, Sex: Male/Female, Age: in years, Mass: Body mass in kilograms, Height: Stature in centimeters, Injured leg: Side of injury (for ACL participants only)"

#### 결론 및 주요 Feature

**원문 발췌 — 집단 간 차이 (Cohort Comparisons by Speed, p. 5):**

> "At normal walking speed, both ACLD and ACLR groups demonstrated significantly slower gait speed than healthy adults (ACLD: 1.19 ± 0.01 m/s; ACLR: 1.19 ± 0.02 m/s; healthy adults: 1.33 ± 0.02 m/s; both p < 0.001). Cadence was significantly lower in both ACL groups relative to healthy adults."

**코호트별 보행 파라미터 (Table 4):**

| Cohort              | Gait Speed (m/s) | Cadence (steps/min) | Stride Length (m) | Stride Time (s) |
| ------------------- | ---------------- | ------------------- | ----------------- | --------------- |
| Healthy Adults      | 1.37 ± 0.47      | 112 ± 21.60         | 1.44 ± 0.28       | 1.12 ± 0.23     |
| Healthy Adolescents | 1.45 ± 0.40      | 113 ± 17.70         | 1.52 ± 0.25       | 1.09 ± 0.17     |
| ACLD                | 1.24 ± 0.40      | 104 ± 18.30         | 1.39 ± 0.27       | 1.19 ± 0.21     |
| ACLR                | 1.23 ± 0.38      | 102 ± 16.70         | 1.41 ± 0.24       | 1.21 ± 0.20     |

**신뢰도 (CV):** Gait speed 2.29–5.85%, Cadence 2.00–5.02%, Stride length 1.70–5.35%

---

### P10 · Hur 2025

**제목:** Learning based lower limb joint kinematic estimation using open source IMU data
**저자:** Benjamin Hur, Sunin Baek, Inseung Kang, Daekyum Kim
**저널:** Scientific Reports (Nature Portfolio)
**연도:** 2025 · DOI: 10.1038/s41598-025-89716-4
**파일:** `Hur_2025_learning_lower_limb_kinematics_IMU.pdf`

#### Input Variables

##### IMU 원시 신호 (8개 센서 × 9채널 = 72 variables)

**원문 발췌 — 입력 구조 (Deep learning, p. 4):**

> "we processed the data from 8 IMUs to have 72 variables for each time step as a deep learning input, encompassing x, y, z measurements of acceleration, gyroscope, and Euler angles."

| 채널         | 측정값             | 축      |
| ------------ | ------------------ | ------- |
| Acceleration | 선형 가속도        | x, y, z |
| Gyroscope    | 각속도             | x, y, z |
| Euler angles | 방향 (orientation) | x, y, z |

- **총 입력:** 9 channels × 8 IMUs = **72 variables**
- **슬라이딩 윈도우:** 50 samples (0.5초)
- **입력 텐서:** [72 × 50]

**원문 발췌 — 전처리 (Data preprocessing, p. 3):**

> "We normalized the accelerations, the angular velocities, and the Euler angles, to have zero mean and unit variance."

##### IMU 부착 위치 (8개)

**원문 발췌 — 센서 위치 (Fig.2 캡션, p. 4):**

> "Total of eight IMUs, positioned on the torso, pelvis, bilateral femurs, tibias, and calcanei, are used as model inputs."

| 위치                             | 수  |
| -------------------------------- | --- |
| Torso (흉부)                     | 1   |
| Pelvis (골반)                    | 1   |
| Bilateral femurs (양측 대퇴)     | 2   |
| Bilateral tibias (양측 경골)     | 2   |
| Bilateral calcanei (양측 발꿈치) | 2   |

#### Target Variables (예측 대상 관절각, 10개)

**원문 발췌 — 출력 변수 (Fig.2 캡션, p. 4):**

> "The model outputs are the bilateral sagittal, frontal, and transverse hip, sagittal knee, and sagittal ankle joint angles."

| 관절  | 면                     | 양측  |
| ----- | ---------------------- | ----- |
| Hip   | Sagittal (굴곡/신전)   | 좌/우 |
| Hip   | Frontal (내/외전)      | 좌/우 |
| Hip   | Transverse (내/외회전) | 좌/우 |
| Knee  | Sagittal (굴곡/신전)   | 좌/우 |
| Ankle | Sagittal (배굴/저굴)   | 좌/우 |

#### 결론 및 주요 Feature

**원문 발췌 — LSTM vs IMU-IK 비교 (Results, p. 5):**

> "The UI outperformed IMU-based inverse kinematics in RMSE values, NRMSE values, and Pearson correlation coefficient for both LSTM and CNN models. Specifically, the LSTM model showed 49.20% lower average RMSE, 50.65% lower average NRMSE, and 20.13% higher correlation coefficient compared to IMU-based inverse kinematics."

**원문 발췌 — 핵심 IMU 조합 (Discussion, p. 8):**

> "femur and calcaneus are the most critical IMU placements, whereas the pelvis is the least important."

**원문 발췌 — Transfer Learning 효과 (p. 6):**

> "applying transfer learning to a generalized pre-trained model improves estimation performance by 53.86%"

**모델 성능 (Femur+Calcaneus 조합):**

| 방법                   | 평균 RMSE                            | 특징                |
| ---------------------- | ------------------------------------ | ------------------- |
| UI (개인별 학습)       | Hip 3.83°, Knee 5.82°, Ankle 3.18°   | 최고 성능           |
| UA (Transfer Learning) | Hip 14.03°, Knee 16.98°, Ankle 7.64° | UG 대비 53.86% 개선 |
| IMU-IK                 | —                                    | 비교 기준           |

**핵심 중요 변수:** **Femur + Calcaneus IMU 조합** (sagittal hip & knee 추정 핵심), **Tibia** (ankle 추정 보조); Pelvis는 중요도 낮음

---

### P11 · Palazzo 2025

**제목:** A Deep Learning-Based Framework Oriented to Pathological Gait Recognition with Inertial Sensors
**저자:** Lucia Palazzo, Vladimiro Suglia, Sabrina Grieco, Domenico Buongiorno, Antonio Brunetti, Leonarda Carnimeo, Federica Amitrano, Armando Coccia, Gaetano Pagano, Giovanni D'Addio, Vitoantonio Bevilacqua
**저널:** Sensors (MDPI), Vol. 25, No. 260
**연도:** 2025 · DOI: 10.3390/s25010260
**파일:** `Palazzo_2025_deep_learning_pathological_gait_recognition_IMU.pdf`

#### Input Variables

##### IMU 원시 신호 (5개 센서 × 9채널 = 45 channels)

**원문 발췌 — 입력 구조 (Preprocessing, p. 6):**

> "Data from the inertial sensors were initially acquired for each subject and repetition as a matrix of size Ns × (5 · 9), where Ns is the number of samples in a single repetition, 5 is the total number of IMU sensors, and 9 is the total number of IMU components (3 for the accelerometer, 3 for the gyroscope, 3 for the magnetometer)."

| 채널          | 측정값      | 축      |
| ------------- | ----------- | ------- |
| Accelerometer | 선형 가속도 | x, y, z |
| Gyroscope     | 각속도      | x, y, z |
| Magnetometer  | 자기장 세기 | x, y, z |

- **총 채널:** 9 × 5 = **45 channels**
- **샘플링:** 128 Hz
- **윈도우:** 128 samples (1초), 50% overlap

##### IMU 부착 위치 (5개)

**원문 발췌 — 센서 위치 (Section 3.1.3, p. 5):**

> "Five sensors were selected (see Figure 1) and worn by each participant on both sides of the human pelvis (RP and LP), on the right and left wrists (RW and LW), and on the sternum (S)."

| 센서 코드 | 위치                     |
| --------- | ------------------------ |
| S         | Sternum (흉골)           |
| LP        | Left Pelvis (좌측 골반)  |
| RP        | Right Pelvis (우측 골반) |
| LW        | Left Wrist (좌측 손목)   |
| RW        | Right Wrist (우측 손목)  |

**측정 장비 사양 (Section 3.1.3, p. 5):**

> "The experimental data were recorded with a sampling rate of 128 Hz from a 3-axis 14-bit accelerometer to measure linear acceleration, a 3-axis 16-bit gyroscope to acquire angular velocity, and a 3-axis 16-bit magnetometer for magnetic field intensity."

#### Target Variables (분류 클래스)

**원문 발췌 — 분류 대상 (Section 3.1.2, p. 4):**

> "in addition to normal walking, four pathological gaits were considered and they are ataxic, equine (foot drop), hemiplegic, and Parkinsonian gaits."

1. Normal gait (정상)
2. Hemiplegic gait (편마비)
3. Equine gait / Foot Drop (족하수)
4. Ataxic-cerebellar gait (소뇌성 실조)
5. Parkinsonian gait (파킨슨)

#### 결론 및 주요 Feature

**분류 성능 (제안 프레임워크):**

| 보행 클래스  | Accuracy |
| ------------ | -------- |
| Normal       | **100%** |
| Hemiplegia   | **100%** |
| Ataxia       | **100%** |
| Foot Drop    | **100%** |
| Parkinsonian | **100%** |

**원문 발췌 — 성능 결과 (Section 4, p. 9):**

> "The median of accuracy and recall for the two models is 100% in almost all sensor configurations, despite the dataset imbalance."

**원문 발췌 — 센서 성분별 비교 (Section 4, p. 10–11):**

> "The outcomes show better performance in both accuracy and recall for the accelerometer and gyroscope, whereas the outcome worsens when passing the magnetometer entry as input to the model, since the recall is about 80% only in three combinations."

**원문 발췌 — Wrist 센서 중요성 (Section 4, p. 11):**

> "the motor behavior of human wrists is different for each type of walking action: it is stationary on the affected side in the hemiplegic gait close to the sternum, and it follows hand tremors in the Parkinsonian gait and arm sway in normal walking."

**핵심 중요 변수:** **Accelerometer + Gyroscope** 조합 (최고 성능); Magnetometer 단독 시 recall 약 80%로 저하; **Wrist 센서** — 병리 보행 유형 변별에 핵심 역할

---

## 전체 Input Variables 통합 비교표

| 변수 카테고리                      | 구체 변수                                                                        | 사용 논문            |
| ---------------------------------- | -------------------------------------------------------------------------------- | -------------------- |
| **vGRF**                           | Peak vertical GRF, GRF asymmetry                                                 | P1, P4, P5           |
| **PA-GRF**                         | Posterior-anterior GRF, asymmetry                                                | P5                   |
| **Knee kinetics (sagittal)**       | Knee flexion moment (KFM), knee extension moment (KEM1/2)                        | P1, P3, P4           |
| **Knee kinetics (frontal)**        | Knee adduction moment (KAM1/2, impulse)                                          | P1, P3               |
| **Knee kinetics (transverse)**     | Internal/external rotation moment (KIRM/KERM)                                    | P3                   |
| **Hip kinetics**                   | Hip adduction/abduction moment                                                   | P1                   |
| **Knee kinematics**                | Knee flexion angle (IC, peak, stance), knee abduction/varus/valgus angle         | P1, P3, P4           |
| **Hip kinematics**                 | Hip abduction angle, external rotation                                           | P1, P3               |
| **Ankle/foot kinematics**          | Ankle dorsiflexion/plantarflexion (0%, 25%, 60%, 83% GC), foot progression angle | P2                   |
| **Spatiotemporal — gait speed**    | Walking speed, self-selected speed conditions                                    | P1, P3, P5, P8, P9   |
| **Spatiotemporal — cadence**       | Steps/min                                                                        | P7, P8, P9           |
| **Spatiotemporal — stride/step**   | Stride length, step length, step width, stride time                              | P3, P8, P9           |
| **Spatiotemporal — stance**        | Single/double leg stance %, double stance time                                   | P2, P8               |
| **IMU: Acceleration (raw)**        | Acc_X/Y/Z (3축), free acceleration                                               | P7, P8, P9, P10, P11 |
| **IMU: Angular velocity (raw)**    | Gyr_X/Y/Z (3축)                                                                  | P7, P8, P9, P10, P11 |
| **IMU: Magnetometer**              | Magnetic field 3축                                                               | P11                  |
| **IMU: Euler/Quaternion**          | Segment orientation                                                              | P9, P10              |
| **IMU: Time-domain features**      | Mean, STD, RMS, skewness, kurtosis, zero-crossing rate                           | P7                   |
| **IMU: Frequency-domain features** | Dominant frequency, spectral energy, spectral entropy                            | P7                   |
| **Clinical — muscle strength**     | 60°/s knee extensor/flexor PT, 180°/s knee extensor/flexor AP (isokinetic)       | P6                   |
| **Clinical — balance (static)**    | BBS overall index (anteroposterior + mediolateral)                               | P6                   |
| **Clinical — balance (dynamic)**   | YBT total score (anterior, posteromedial, posterolateral)                        | P6                   |
| **Demographics**                   | Age, BMI, sex, height, weight                                                    | P6, P8, P9           |
| **Clinical scores**                | IKDC, WOMAC, UPDRS III, FMA-LE, TUG, visualGaitEvaluation                        | P8                   |
| **Limb symmetry**                  | GRF asymmetry (injured − uninjured), limb symmetry values                        | P4, P5               |
| **Anterior femoral displacement**  | Average during stance phase                                                      | P3                   |

---

## 중요 Feature 요약

결론/Discussion/Feature Importance 기반으로 각 논문이 강조한 핵심 변수를 정리합니다.

| 논문                        | 연구 목적                       | 핵심 중요 변수                                                       | 근거                                                       |
| --------------------------- | ------------------------------- | -------------------------------------------------------------------- | ---------------------------------------------------------- |
| **P1 · Goetschius 2018**    | ACLR 시기별 보행 차이           | vGRF, knee extension moment, knee adduction moment                   | Early ACLR: sagittal loading↓; Late ACLR: frontal loading↑ |
| **P2 · Ursei 2020**         | ACL 결핍 아동 발목 보상         | **Ankle dorsiflexion at 0% GC (initial contact)**                    | p=2.55×10⁻⁹, 가장 유의한 ACL 결핍 지표                     |
| **P3 · Erhart-Hledik 2018** | ACLR 후 2→8년 종단 변화         | **KFM↑, KEM↓, External rotation↑**                                   | p<0.001, OA 진행과 연관                                    |
| **P4 · Krishnan 2022**      | knee extension moment 기여 역학 | **Peak knee flexion angle** (β=0.703–0.831)                          | vGRF 대비 2배 이상 영향력; R²=0.77–0.88                    |
| **P5 · Garcia 2022**        | 보행 속도별 GRF 비대칭          | **vGRF asymmetry**, PA-GRF asymmetry                                 | 속도 증가 → 비대칭 악화 (ACL군만)                          |
| **P6 · Hwang 2024**         | ML로 RTS 예측                   | **YBT total score, 60°/s knee extensor PT, 180°/s knee extensor AP** | SHAP 분석 기반; RF AUC=0.952                               |
| **P7 · Yuan 2026**          | IMU 최적 구성 + ML              | **Accel_Z_variance (Lumbar)** (Gini=0.187), Accel_X_variance         | R²=0.99 (Cadence); C3 (3 IMU) 비대칭 예측 필수             |
| **P8 · Voisard 2025**       | 다병리 임상 IMU 데이터셋        | Gait velocity, stride time, double stance time                       | 정상 vs 병리 집단 변별                                     |
| **P9 · Yona 2026**          | ACL 다속도 IMU 데이터셋         | Gait speed, cadence, stride length                                   | ACLD/ACLR: gait speed 유의하게 낮음 (p<0.001)              |
| **P10 · Hur 2025**          | IMU→관절각 추정 (DL)            | **Femur + Calcaneus IMU** (sagittal hip & knee); Tibia (ankle)       | Pelvis 중요도 최저; Transfer Learning 53.86% 개선          |
| **P11 · Palazzo 2025**      | 병리 보행 분류 (DL)             | **Accelerometer + Gyroscope**, Wrist 센서                            | Magnetometer 단독 recall 80%로 저하; 모든 병리 100% 분류   |

---

_끝_
