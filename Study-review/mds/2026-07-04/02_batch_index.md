# 2026-07-04 신규 논문 배치

## 요약

- 입력 PDF: 18개
- 고유 논문: 15편
- SHA-256 완전 중복: 3개
- `00_upstream` 잔여 PDF: 0개
- 파이프라인 입력: `01_batch_manifest.json`의 `papers` 15개만 사용

## 분류 현황

| 분류 | 신규 고유 논문 수 |
|---|---:|
| `01_acl_gait_biomechanics_studies` | 1 |
| `03_wearable_imu_and_portable_sensing` | 2 |
| `04_machine_learning_and_deep_learning` | 1 |
| `05_return_to_sport_and_functional_tests` | 3 |
| `06_general_gait_and_other_knee_conditions` | 2 |
| `07_composite_kinematic_kinetic_scoring_indices` | 6 |
| 합계 | 15 |

## 고유 논문 15편

| 원래 파일명 | 공식 제목 | 분류 |
|---|---|---|
| `전방 십자 인대 손상 환자의 병적 보행의 특징에 관한 연구_ 운동 형상학 및 운동 역학적 특징.pdf` | 전방 십자 인대 손상 환자의 병적 보행의 특징에 관한 연구 - 운동 형상학 및 운동 역학적 특징 | `01_acl_gait_biomechanics_studies` |
| `Anterior Cruciate Ligament Out-of-Laboratory Longitudinal Gait Assessment of Participants Before and After Anterior Cruciate Ligament Reconstruction Surgery_ An Observational Longitudinal Study.pdf` | Out-of-Laboratory Longitudinal Gait Assessment of Participants Before and After Anterior Cruciate Ligament Reconstruction Surgery: An Observational Longitudinal Study | `03_wearable_imu_and_portable_sensing` |
| `Gait Analysis Post Anterior Cruciate Ligament Reconstruction Using Inertial Sensors_ A Longitudinal Study.pdf` | Gait Analysis Post Anterior Cruciate Ligament Reconstruction Using Inertial Sensors: A Longitudinal Study | `03_wearable_imu_and_portable_sensing` |
| `applsci-09-03306.pdf` | Gait Classification Using Mahalanobis–Taguchi System for Health Monitoring Systems Following Anterior Cruciate Ligament Reconstruction | `07_composite_kinematic_kinetic_scoring_indices` |
| `Individual hop analysis and reactive strength ratios provide better discrimination of ACL reconstructed limb deficits than triple hop for distance scores in athletes returning to sport.pdf` | Individual hop analysis and reactive strength ratios provide better discrimination of ACL reconstructed limb deficits than triple hop for distance scores in athletes returning to sport | `05_return_to_sport_and_functional_tests` |
| `kjsm-39-1-34.pdf` | 하지 검사 프로토콜을 이용한 전방십자인대 재건술 후 무릎관절의 객관적 및 주관적 분석 | `05_return_to_sport_and_functional_tests` |
| `전방십자인대 재건술 후 16 주간의 기능적인 재활운동 프로그램이 여성 축구선수의 슬관절 기.pdf` | 전방십자인대 재건술 후 16 주간의 기능적인 재활운동 프로그램이 여성 축구선수의 슬관절 기능 회복에 미치는 영향 | `05_return_to_sport_and_functional_tests` |
| `75-Submission File-154-1-10-20170816.pdf` | The Development of a Normative Gait Database | `06_general_gait_and_other_knee_conditions` |
| `shin2020.pdf` | Does kinematic gait quality improve with functional gait recovery? A longitudinal pilot study on early post-stroke individuals | `07_composite_kinematic_kinetic_scoring_indices` |
| `Journal Orthopaedic Research - 2021 - Biggs - Gait function improvements  using Cardiff Classifier  are related to.pdf` | Gait function improvements, using Cardiff Classifier, are related to patient-reported function and pain following hip arthroplasty | `07_composite_kinematic_kinetic_scoring_indices` |
| `The functional gait deviation index.pdf` | The functional gait deviation index | `07_composite_kinematic_kinetic_scoring_indices` |
| `applsci-09-04680-v2.pdf` | Gait Analysis and Mathematical Index-Based Health Management Following Anterior Cruciate Ligament Reconstruction | `07_composite_kinematic_kinetic_scoring_indices` |
| `applsci-16-03976.pdf` | Unsupervised Detection of Pathological Gait Patterns via Instantaneous Center of Rotation Analysis | `04_machine_learning_and_deep_learning` |
| `fbioe-10-874074.pdf` | Derivation of the Gait Deviation Index for Spinal Cord Injury | `07_composite_kinematic_kinetic_scoring_indices` |
| `journal.pone.0351930.pdf` | Investigating compensatory adjustments induced by rhythmic auditory stimulation for changes in temporal gait symmetry in lower-limb prosthetic users | `06_general_gait_and_other_knee_conditions` |

## SHA-256 완전 중복 3개

| 격리한 원래 파일명 | SHA-256 | 대표 논문 분류 |
|---|---|---|
| `applsci-09-03306 (1).pdf` | `aec7340797f09fd945fbf9dcb5a73c676dc248933ae6cb0848cc20b607662b87` | `07_composite_kinematic_kinetic_scoring_indices` |
| `applsci-09-04680-v2 (1).pdf` | `8ca1278c55dff523285792473053dc0de05ac7c6a0160939860e2e901f34b490` | `07_composite_kinematic_kinetic_scoring_indices` |
| `attr-article-p823.pdf` | `07d438dc952ccf83a6dde8076beafc080ff71aca3b32ca654d8ece8cb7402ada` | `03_wearable_imu_and_portable_sensing` |

중복 파일은 삭제하지 않고 `docs/ref_papers/99_duplicates_or_alternate_versions/`에 공식 제목으로 보관했다. 각 파일의 절대 목적지와 대표 파일 연결은 `01_batch_manifest.json`의 `duplicates`에 기록되어 있다.
