# Feature Lineup — ACL Gait Biomechanics 분석

> **데이터 출처**: `data/processed/raw_merged.parquet` (운동학적 피처), `data/gait_analysis_global.csv` (시공간 피처)  
> **그룹 구성**: Healthy adults (HA, n=25) | ACLR (n=27) | ACLD (n=27, ACLR과 동일 피험자)  
> **평가 기준**: 환자군(ACLR/ACLD) → `ID.csv`의 Injured leg 기반 환측/건측 분리. Healthy → 우측=환측 대체, 좌측=건측 대체.

---

## 📋 피처 매핑 테이블

| Index | Category | Feature | 컬럼명 (raw_merged.parquet) | 추출 방식 | 선정 근거 |
|---|---|---|---|---|---|
| 1 | Joint kinematic | Knee sagittal plane kinematics | `jointAngle_45` (R_Knee_x) / `jointAngle_57` (L_Knee_x) | 환측/건측에 맞춰 해당 컬럼의 **Max (Peak Flexion)** 추출 | 11개 연구에서 보고된 핵심 지표. ACLR 그룹의 굴곡 감소 패턴(모더레이트 수준 근거) |
| 2 | Joint kinematic | Knee frontal plane kinematics | `jointAngle_46` (R_Knee_y) / `jointAngle_58` (L_Knee_y) | 환측/건측에 맞춰 해당 컬럼의 **Min (Peak Adduction)** 추출 | ACLR 환측의 내전 모멘트 감소는 무릎 불안정성의 전형적 징후 |
| 3 | Joint kinematic | Knee transverse plane kinematics | `jointAngle_47` (R_Knee_z) / `jointAngle_59` (L_Knee_z) | 환측/건측에 맞춰 해당 컬럼의 **Max (Peak Internal Rotation)** 추출 | 부하 단계(Loading phase)에서 내측 경골 회전 감소가 강한 근거로 보고됨 |
| 4 | Joint kinematic | Hip & Ankle kinematics | Hip: `jointAngle_42`/`43` (R_Hip_x/y) / `jointAngle_54`/`55` (L_Hip_x/y) | Hip: Peak Flexion (Max), Peak Adduction (Min) | 고관절 및 발목은 무릎 불안정 보상 전략(Hip/Ankle strategy)의 핵심 |
| | | | Ankle: `jointAngle_48` (R_Ankle_x) / `jointAngle_60` (L_Ankle_x) | Ankle: Peak Dorsiflexion (Max) | |
| 5 | Derived | Bilateral kinematic asymmetry (LSI) | 위 1~4번 피처에서 환측(I)값과 건측(C)값을 사용 | **LSI = 100 × (I / C)** — 무릎/고관절/발목 굴곡 각도 모두에 개별 적용. Healthy는 우측=I, 좌측=C | 좌우 운동학적 비대칭은 복귀 수행 능력(RTP)의 핵심 예측 지표 |
| 6 | Derived | Phase-specific gait features | — | **우선 제외(보류)** — 보행 주기 정규화 및 입각기 구간 분할 알고리즘이 별도 필요 | 차후 단계에서 구현 예정 |
| 7 | Spatiotemporal | Spatiotemporal features | `gait_speed_mps`, `cadence_spm`, `stride_length_mean_m`, `step_width_mean_m_orth`, `double_support_pct`, `single_support_L_pct`, `single_support_R_pct` | `gait_analysis_global.csv`에서 직접 추출 (group × pace_condition 별) | 속도/보폭/케이던스: 보행 효율. 지지기 비율: 환측 체중 부하 대칭성. 기저면 폭: 불안정 보상 여부 |
| 8 | Speed-related | Robustness to speed changes | pace_condition: `fast`, `normal`, `slow` (gait_analysis_global.csv) | 1~7번 피처를 속도 조건(fast/normal/slow) 별로 분리하여 **Two-way Mixed ANOVA** (Group × Speed) 상호작용 검증 | 보행 속도는 무릎 내측 모멘트와 고관절 굴곡 로딩에 직접 상관. 느린 속도→관절 부하 저감 보상 전략 확인 |

---

## 🗺️ 컬럼 인덱스 대응표 (data_description.csv 기반)

### Joint Angle 컬럼 매핑
| 컬럼명 | 관절 | 축 (의미) | 분석 피처 |
|---|---|---|---|
| `jointAngle_42` | jRightHip | x (굴곡/신전) | Right Hip Flexion |
| `jointAngle_43` | jRightHip | y (내전/외전) | Right Hip Adduction |
| `jointAngle_45` | jRightKnee | x (굴곡/신전) | Right Knee Flexion |
| `jointAngle_46` | jRightKnee | y (내전/외전) | Right Knee Adduction |
| `jointAngle_47` | jRightKnee | z (내/외회전) | Right Knee Internal Rotation |
| `jointAngle_48` | jRightAnkle | x (족배굴곡/족저굴곡) | Right Ankle Dorsiflexion |
| `jointAngle_54` | jLeftHip | x (굴곡/신전) | Left Hip Flexion |
| `jointAngle_55` | jLeftHip | y (내전/외전) | Left Hip Adduction |
| `jointAngle_57` | jLeftKnee | x (굴곡/신전) | Left Knee Flexion |
| `jointAngle_58` | jLeftKnee | y (내전/외전) | Left Knee Adduction |
| `jointAngle_59` | jLeftKnee | z (내/외회전) | Left Knee Internal Rotation |
| `jointAngle_60` | jLeftAnkle | x (족배굴곡/족저굴곡) | Left Ankle Dorsiflexion |

### Spatiotemporal 컬럼 매핑 (gait_analysis_global.csv)
| 컬럼명 | 의미 | 선정/제외 |
|---|---|---|
| `gait_speed_mps` | 보행 속도 (m/s) | ✅ 선정 |
| `cadence_spm` | 분당 보수 (steps/min) | ✅ 선정 |
| `stride_length_mean_m` | 평균 보폭 (m) | ✅ 선정 |
| `step_width_mean_m_orth` | 보행 기저면 폭 (m) | ✅ 선정 (불안정 보상 확인) |
| `double_support_pct` | 양발 지지기 비율 (%) | ✅ 선정 (체중 분산 보상) |
| `single_support_L_pct` | 좌측 단일 지지기 비율 (%) | ✅ 선정 (비대칭 확인) |
| `single_support_R_pct` | 우측 단일 지지기 비율 (%) | ✅ 선정 (비대칭 확인) |
| `step_time_sd_s` | 스텝 시간 표준편차 | ❌ 제외 (변동성 지표, 거시 목적과 상이) |
| `stride_length_sd_m` | 보폭 표준편차 | ❌ 제외 (변동성 지표, 거시 목적과 상이) |
| `n_steps_total` | 총 스텝 수 | ❌ 제외 (단위 지표, 비교 목적과 상이) |

---

## 🔬 통계 분석 파이프라인

### Step 1: 사전 검정
- **Shapiro-Wilk Test** (정규성): 각 그룹 × 속도 조건별 잔차 정규성
- **Levene's Test** (등분산성): 집단 간 분산 동질성

### Step 2: 메인 통계 모델
| 조건 | 적용 모델 |
|---|---|
| 정규성/등분산 만족 | **Two-way Mixed ANOVA** (Group × Speed, Subject_ID = Random Effect) |
| 정규성 또는 등분산 기각 | **Kruskal-Wallis Test** (비모수, 속도 조건별 개별 수행) |

- **고정 효과(Fixed Effect)**: `Group` (HA/ACLR/ACLD), `Speed` (fast/normal/slow)  
- **임의 효과(Random Effect)**: `Subject_ID` (개인 편차 통제)  
- **두 요인 간 상호작용(Interaction)**: `Group × Speed`

### Step 3: 사후 검정 및 효과 크기
| 모델 | 사후 검정 | 효과 크기 |
|---|---|---|
| ANOVA | Tukey HSD | Partial η² |
| Kruskal-Wallis | Dunn's Test + Bonferroni 보정 | Epsilon-squared (ε²) |

### Step 4: 변수-그룹 간 상관관계
1. **Z-score Standardization**: 전체 피처를 StandardScaler로 정규화 (단위 차이 제거)
2. **Point-Biserial Correlation**: 그룹을 0/1 더미 변수 (`is_HA`, `is_ACLR`, `is_ACLD`)로 변환 후 피처별 상관계수 산출
3. **Heatmap**: 시각화 (색상 방향: 양의 상관=빨강, 음의 상관=파랑)
