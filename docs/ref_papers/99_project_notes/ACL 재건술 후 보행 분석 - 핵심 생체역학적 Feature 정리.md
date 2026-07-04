# ACL 재건술 후 보행 분석 - 핵심 생체역학적 Feature 정리

> **목적**: ACL 재건술(ACLR) 환자의 회복 단계별 보행 데이터 모델링에 사용할 생체역학적 변수를 선행연구에서 정리

---

## 논문 목록

| # | 논문 제목 | 저자 | 출판연도 | 연구 유형 |
|---|---------|------|----------|----------|
| 1 | Bilateral Gait Six and Twelve Months Post-ACL Reconstruction Compared to Controls | Hooper et al. | 2002 | 종단 비교 연구 |
| 2 | Gait analysis post anterior cruciate ligament reconstruction: Knee osteoarthritis perspective | Al-Amin et al. | 2015 | 文献 리뷰 |
| 3 | Gait Patterns Differ Between ACL-Reconstructed Athletes Who Pass Return-to-Sport Criteria and Those Who Fail | Di Stasi et al. | 2013 | 비교 코호트 연구 |
| 4 | Progressive Changes in Walking Kinematics and Kinetics After ACL Injury and Reconstruction: A Review and Meta-Analysis | Gao & Zheng | 2014 | 메타 분석 |
| 5 | Knee Kinematics and Joint Moments During Gait Following ACLR: A Systematic Review and Meta-Analysis | Hart et al. | 2016 | 체계적 리뷰 & 메타분석 |
| 6 | Movement Patterns of the Knee During Gait Following ACL Reconstruction: A Systematic Review and Meta-Analysis | Kaur et al. | 2016 | 체계적 리뷰 & 메타분석 |
| 7 | Estimation of Kinetics Using IMUs to Monitor and Aid in Clinical Decision-Making during ACL Rehabilitation | Krishnakumar et al. | 2024 | 체계적 리뷰 |

---

# 1. Bilateral Gait Six and Twelve Months Post-ACL Reconstruction Compared to Controls

**저자**: Hooper et al. (2002) | **저널**: *British Journal of Sports Medicine*

## 논문 개요
ACLR 수술 후 6개월 및 12개월 시점의 보행 운동학/운동역학 변수를 건강한 대조군과 비교하여, ACLR 환자가 어떤 보상 전략(compensatory strategy)을 채택하는지 분석.

## 주요 발췌 내용

> "At 6 months post-ACLR, peak knee flexion angle during walking was significantly lower in the ACLR limb compared to the contralateral limb. This 'stiffened-knee' gait pattern persisted at 12 months."

> "Knee extension moment was significantly reduced in the ACLR limb at both 6 and 12 months compared to the contralateral side, suggesting persistent quadriceps inhibition."

> "Ground reaction forces showed a reduced first peak (loading response) in the ACLR limb, indicating weight-bearing avoidance strategy."

## 핵심 Feature

| Feature | 측정 시점 | 주요 발견 |
|---------|----------|---------|
| **Peak Knee Flexion Angle (KFA)** | Loading Response | ACLR측: 대측 대비 유의하게 감소 (stiffened-knee) |
| **Peak Knee Extension Moment (KEM)** | Mid-Stance | ACLR측: 6M, 12M 모두 대측 대비 감소 |
| **Vertical GRF - 1st Peak** | Loading Response | ACLR측: 하중 회피 전략으로 감소 |
| **Vertical GRF - 2nd Peak** | Terminal Stance | 6M → 12M 회복 추세 |
| **Knee Flexion Moment (KFM)** | Mid-Stance | ACLR측 대측 대비 낮은 값 |
| **Knee Adduction Moment (KAM)** | Stance Phase | 대측–ACLR 간 차이 없음 |
| **Concentric Power** | Push-off Phase | ACLR측 감소 |
| **Eccentric Power (Stair Descent)** | Stair descent | ACLR측 감소 |

### 임상적 의의
- 6개월에서 12개월로 갈수록 vGRF는 회복 추세 but KFA, KEM은 여전히 비대칭
- **Stiffened-knee 전략**: ACLR 환자는 무릎 굴곡을 줄이고 고관절/발목 전략으로 보상

---

# 2. Gait analysis post anterior cruciate ligament reconstruction: Knee osteoarthritis perspective

**저자**: Al-Amin et al. (2015) | **저널**: *World Journal of Orthopedics*

## 논문 개요
ACLR 후 보행 역학의 변화가 장기적으로 무릎 골성관절염(OA) 발병에 미치는 영향을 문헌 기반으로 분석. ACLR 후 비정상적 보행 패턴이 관절 연골 부하를 변화시킨다는 기계적 메커니즘 규명.

## 주요 발췌 내용

> "Reduced knee flexion angle and extensor moment during walking after ACLR is a well-established finding, and this altered loading pattern may initiate degenerative changes in the cartilage."

> "The knee adduction moment (KAM) is a strong predictor of medial tibio-femoral compartment loading and OA progression. Changes in KAM post-ACLR have been inconsistently reported."

> "Knee varus moment (external knee adduction moment) reflects the medial compartment load distribution during stance phase of gait."

## 핵심 Feature

| Feature | 관련 결과변수 | OA 연관성 |
|---------|------------|----------|
| **Knee Flexion Angle (KFA)** | 감소 → 연골 면압 변화 | ✅ 직접 연관 |
| **Knee Extension Moment (KEM)** | 감소 → Quadriceps avoidance | ✅ 직접 연관 |
| **Knee Adduction Moment (KAM)** | 내측 구획 부하 지표 | ✅ 강력한 OA 예측인자 |
| **Knee Varus Moment** | 내측–외측 부하 분배 | ✅ OA 진행과 연관 |
| **Hip Extension Moment** | 보상 전략의 대리 지표 | ⚠️ 간접 연관 |
| **Tibio-femoral Contact Force** | 관절 내부 부하 | ✅ OA 메커니즘 |
| **Patellofemoral Joint Force** | 슬개골–대퇴골 접촉 부하 | ✅ 초기 OA 변화와 연관 |

### 시간대별 변화 요약

```
ACLR 직후 (< 6M):  KFA ↑, KEM ↓ (급성 통증/부종)
6~12M:             KFA ↓ → stiffened-knee 발현
1~3년:             지속적 KFA 감소, KEM 감소
≥ 3년:             일부 회복 but 정상화 미완성
```

---

# 3. Gait Patterns Differ Between ACL-Reconstructed Athletes Who Pass Return-to-Sport Criteria and Those Who Fail

**저자**: Di Stasi et al. (2013) | **저널**: *Journal of Orthopaedic & Sports Physical Therapy*

## 논문 개요
ACLR 수술 6개월 후 RTS(Return-to-Sport) 기준 통과/실패 그룹 간의 보행 운동학·운동역학 패턴 차이를 분석. RTS 기준 통과 여부를 구분하는 생체역학적 지표 탐색.

## 주요 발췌 내용

> "The RTS-Fail group demonstrated significantly lower peak knee flexion angle (22.6° vs 25.1°) and lower knee extensor moment compared to the RTS-Pass group during walking."

> "Hip extensor moment was significantly greater in the RTS-Fail group, suggesting a compensatory hip strategy to offload the knee."

> "Knee power absorption during loading response was significantly reduced in both groups compared to contralateral limbs, but more pronounced in RTS-Fail."

> "Knee extension moment in the ACLR limb was 0.37±0.03 Nm/kg·m for RTS-Fail vs 0.42±0.03 Nm/kg·m for RTS-Pass (p<0.05)."

## 핵심 Feature

| Feature | RTS-Fail | RTS-Pass | 통계적 유의성 |
|---------|----------|----------|------------|
| **Peak Knee Flexion Angle (KFA)** | 22.6±1.4° | 25.1±1.5° | p < 0.05 |
| **Knee Extension Moment (KEM)** | 0.37±0.03 Nm/kg·m | 0.42±0.03 Nm/kg·m | p < 0.05 |
| **Knee Power Absorption** | 현저히 감소 | 중간 감소 | p < 0.05 |
| **Hip Extension Moment** | 증가 (보상 전략) | 정상 범위 | p < 0.05 |
| **Knee Flexion Excursion** | 감소 | 정상 근접 | p < 0.05 |
| **Limb Symmetry Index (LSI)** | 낮음 | 높음 | 핵심 분류 지표 |

### 임상적 의의
- **RTS 기준 통과/실패의 생체역학적 기준** 제시
- Quadriceps avoidance가 RTS 실패 그룹에서 더 심각
- 고관절 전략 증가 → 무릎 부하 회피의 정도가 회복 단계 분류에 활용 가능
- **LSI(Limb Symmetry Index)**: 양측 비대칭성 지표로서 회복 평가의 핵심

---

# 4. Progressive Changes in Walking Kinematics and Kinetics After Anterior Cruciate Ligament Injury and Reconstruction: A Review and Meta-Analysis

**저자**: Gao & Zheng (2014) | **저널**: *American Journal of Sports Medicine*

## 논문 개요
ACL 손상(ACLD) 및 ACLR 후 장기적 보행 변화를 메타분석. 수술 전/후 시간 경과에 따른 운동학·운동역학 지표의 변화 궤적(trajectory) 분석.

## 주요 발췌 내용

> "Across multiple studies, peak knee flexion angle during walking was consistently reduced after ACLR compared to controls, with effect sizes ranging from -0.76 to -2.21 standard deviation."

> "Peak knee extensor moment showed a progressive reduction from the early post-operative period and remained lower than controls even at 1-3 years post-ACLR."

> "Knee adduction moment was significantly lower in ACLD patients compared to controls but returned toward normal levels after reconstruction."

> "The 'quadriceps avoidance' gait pattern—characterized by reduced knee flexion and extensor moment—was most prominent at 6-12 months post-ACLR."

> "Gait speed significantly influences kinematic and kinetic measurements; studies should control for walking speed as a covariate."

## 핵심 Feature (시간대별 변화)

### 운동학적 Feature
| Feature | ACLD | < 6M post-ACLR | 6~12M | 1~3Y | ≥ 3Y |
|---------|------|----------------|-------|------|------|
| **Peak Knee Flexion Angle** | ↓ | ↑ (통증/부종) | ↓↓ | ↓ | 부분 회복 |
| **Knee Flexion Excursion** | ↓ | ↓ | ↓↓ | ↓ | ↓ |
| **Tibial Internal Rotation** | 변화 | 변화 | 변화 | 불명확 | 불명확 |

### 운동역학적 Feature
| Feature | ACLD | < 6M post-ACLR | 6~12M | 1~3Y | ≥ 3Y |
|---------|------|----------------|-------|------|------|
| **Knee Extension Moment (KEM)** | ↓↓ | ↓↓↓ | ↓↓ | ↓ | 부분 회복 |
| **Knee Flexion Moment (KFM)** | ↓ | ↑ | ↓ | 정상화 경향 | 정상 |
| **Knee Adduction Moment (KAM)** | ↓ | 불명확 | 무차이 | 무차이 | 무차이 |

### 보행 속도 관련 주의사항
> ⚠️ **보행 속도 Confounding Effect**: 보행 속도가 빠를수록 KEM, vGRF 모두 증가. 그룹 간 비교 시 속도 보정 필수

---

# 5. Knee Kinematics and Joint Moments During Gait Following Anterior Cruciate Ligament Reconstruction: A Systematic Review and Meta-Analysis

**저자**: Hart et al. (2016) | **저널**: *British Journal of Sports Medicine*

## 논문 개요
34개 연구를 포함한 체계적 리뷰. 수술 후 시간대 (<6M, 6~12M, 1~3Y, ≥3Y)별로 ACLR 무릎의 관절 운동학·모멘트 변화를 메타분석.

## 주요 발췌 내용

> "Meta-analysis revealed greater knee flexion angles (<6M, SMD: 1.06) but lower peak knee flexion angles at 1–3 years (SMD: -2.21) and ≥3 years (SMD: -1.38) post-ACLR."

> "Knee flexion moment was lower at 6–12 months post-ACLR (SMD: -0.76) compared to healthy controls—strong evidence of quadriceps avoidance."

> "Strong evidence of no difference in peak knee adduction moment >3 years after ACLR (SMD: 0.09)—sagittal plane biomechanics are more relevant than frontal plane post-ACLR."

> "Graft type affected outcomes: patellar-tendon graft showed lower knee flexion moments, hamstring-tendon graft showed smaller knee adduction angles at 6–12 months."

## 핵심 Feature (메타분석 결과 요약)

### Sagittal Plane (시상면) - 가장 중요한 평면
| Feature | 시기 | SMD | 의미 |
|---------|------|-----|------|
| **Peak Knee Flexion Angle** | < 6M vs Control | +1.06 | 초기 굴곡 증가 |
| **Peak Knee Flexion Angle** | 1~3Y vs Control | -2.21 | 장기 굴곡 감소 |
| **Peak Knee Flexion Angle** | 6~12M vs Contralateral | -1.74 | 양측 비대칭 |
| **Peak Knee Flexion Moment** | 6~12M vs Control | -0.76 | Quadriceps avoidance |
| **Peak Knee Extension Moment** | < 6M vs Control | -3.55 | 초기 급격한 감소 |
| **Peak Knee Flexion Moment** | 6~12M vs Contralateral | -1.29 | 환측 감소 |

### Frontal Plane (관상면)
| Feature | 시기 | 증거 수준 | 결론 |
|---------|------|---------|------|
| **Peak Knee Adduction Moment** | ≥ 3Y vs Control | Strong | 유의미한 차이 없음 |
| **Peak Knee Adduction Moment** | 6~12M vs Control | Moderate | 유의미한 차이 없음 |
| **Peak Knee Adduction Angle** | 6~12M (Hamstring graft) | Moderate | 감소 (less varus) |

### Transverse Plane (횡단면)
| Feature | 시기 | 증거 수준 | 결론 |
|---------|------|---------|------|
| **Knee Internal Rotation Angle** | 6~12M | Limited | 상충되는 증거 |
| **Knee External Rotation Moment** | 1~3Y | Limited | 감소 |

### 결론적 우선순위
```
핵심 Feature 우선순위 (Hart et al. 기준):
1위: Sagittal plane KFA, KFM, KEM  ← 가장 일관된 증거
2위: Frontal plane KAM              ← 장기적으로 정상화됨
3위: Transverse plane rotation      ← 증거 부족, 추가 연구 필요
```

---

# 6. Movement Patterns of the Knee During Gait Following ACL Reconstruction: A Systematic Review and Meta-Analysis

**저자**: Kaur et al. (2016) | **저널**: *Sports Medicine*

## 논문 개요
40개 연구 포함. 보행(walking), 계단 오르내리기(stair), 달리기(running) 시 ACLR 무릎의 운동학·운동역학 패턴을 분석. 시간 경과에 따른 회복 궤적도 분석.

## 주요 발췌 내용

> "Strong evidence for no significant difference in peak flexion angles between ACLR and control groups during walking. However, moderate evidence for less flexion in ACLR compared to contralateral limb (ES: -0.06; MD: 4.3°)."

> "Strong evidence was found for lower peak flexion moments in ACLR participants compared to control groups and contralateral limb during walking and stair activities."

> "Strong to moderate evidence for lower peak adduction moment in ACLR participants for the injured compared with the contralateral limbs during walking and stair descent."

> "Joint kinematics are restored, on average, 6 years following reconstruction, while knee external flexion moments remain lower than controls."

> "Knee adduction moments are lower within the first year following surgery and higher than controls during later phases (5 years)—potential risk for OA."

## 핵심 Feature

### 보행(Walking) 시 주요 Feature
| Feature | ACLR vs Control | ACLR vs Contralateral | 증거 수준 |
|---------|----------------|----------------------|---------|
| **Peak Knee Flexion Angle** | 차이 없음 | 감소 (4.3°↓) | Strong / Moderate |
| **Peak Knee Flexion Moment** | 감소 | 감소 | Strong |
| **Peak Knee Adduction Moment** | 차이 없음 | 감소 | Strong |

### 계단(Stair) 시 주요 Feature
| Feature | ACLR vs Control | ACLR vs Contralateral | 증거 수준 |
|---------|----------------|----------------------|---------|
| **Peak Knee Flexion Angle (Ascent)** | 차이 없음 | 감소 | Moderate |
| **Peak Knee Extension Moment** | 감소 | 감소 | Strong |
| **Peak Knee Adduction Moment (Descent)** | 감소 | 감소 | Strong |

### 달리기(Running) 시 주요 Feature
| Feature | 발견 | 증거 수준 |
|---------|------|---------|
| **Peak Knee Flexion Moment** | ACLR < Control | Limited |
| **Peak Knee Flexion Angle** | 유사 | Limited |

### 시간 경과에 따른 회복 패턴
```
Peak Knee Flexion Angle:   회복 완료 시점 ≈ 6년
Peak Knee Flexion Moment:  5년 후에도 여전히 낮음 ← 중요!
Knee Adduction Moment:     1년 이내 낮음 → 4~5년 후 오히려 높아짐
```

> ⚠️ **중요**: 보행 속도(gait speed)가 KAM 및 KFM에 영향. 연구 간 비교 시 속도 보정을 confounding factor로 명시적 처리해야 함 (이 논문은 bias 평가 항목에 속도 통제 여부를 포함).

---

# 7. Estimation of Kinetics Using IMUs to Monitor and Aid in Clinical Decision-Making during ACL Rehabilitation: A Systematic Review

**저자**: Krishnakumar et al. (2024) | **저널**: *Sensors*

## 논문 개요
IMU(관성 측정 장치)를 이용해 보행 운동역학(GRF, 관절 모멘트)을 추정하는 알고리즘을 체계적으로 리뷰. ACL 재활 임상 의사결정 지원을 위한 IMU 기반 킨에틱스 추정의 현 수준과 한계 분석.

## 주요 발췌 내용

> "Knee moments and patellofemoral joint forces are important biomechanical parameters to be assessed during ACL rehabilitation. Hip and ankle moments may also be useful."

> "IMUs have exhibited potential in estimating kinetic parameters with good accuracy, particularly for sagittal movements in healthy cohorts. However, algorithms have not been validated on ACL patients."

> "The most common sensor placement locations: lower back (pelvis/sacrum) — thigh — shank — feet. Bilateral placement is generally preferred."

> "Clinical difference for VGRF = 0.24 BW; knee flexion-extension moment = 0.035 BW·m. A model achieving RMSE lower than these values is clinically relevant for ACL patients."

> "ACL patients are known to have increased knee flexion for at least a year after reconstruction, which must be accounted for in algorithm validation."

> "Knee adduction-abduction, flexion-extension external rotation moments had lowest nRMSE of 10.58, 9.46, and 17.12% respectively for walking using 3D IMU-based approaches."

## 핵심 Feature

### IMU 기반으로 추정 가능한 Kinetic Feature
| Feature | 추정 가능 여부 | 최적 정확도(walking) | 방법론 |
|---------|-------------|-------------------|-------|
| **Vertical GRF (VGRF)** | ✅ 가능 | RMSE: 0.076 BW | BM/ML |
| **Anterior-Posterior GRF (APGRF)** | ✅ 가능 | rRMSE: 9.40% | BM/ML |
| **Medio-Lateral GRF (MLGRF)** | ✅ 가능 | rRMSE: 13.10% | BM/ML |
| **Peak VGRF (pVGRF)** | ✅ 가능 | RMSE: 0.12~0.14 BW (running) | SM/ML |
| **Knee Flexion-Extension Moment** | ✅ 가능 | nRMSE: 9.46% | ML(3D) |
| **Knee Adduction-Abduction Moment** | ✅ 가능 | nRMSE: 10.58% | ML(3D) |
| **Knee Internal-External Rotation Moment** | ⚠️ 어려움 | nRMSE: 17.12% | ML(3D) |
| **Patellofemoral Joint Force** | ⚠️ 제한적 | - | MS모델 |
| **Joint Reaction Force** | ⚠️ 제한적 | - | MS모델 |

### ACL 재활 관련 핵심 kinetic Feature (임상 의의)
| Feature | ACL 재활 관련 임상 중요도 |
|---------|-------------------------|
| **Knee Flexion-Extension Moment** | ⭐⭐⭐ 최고 중요도 (Quadriceps 기능 반영) |
| **VGRF** | ⭐⭐⭐ 하중 수용 능력 (체중 부하 회피 감지) |
| **Hip & Ankle Moments** | ⭐⭐ 보상 전략 감지에 중요 |
| **Knee Adduction Moment** | ⭐⭐ 내측 구획 부하 (OA 위험도) |
| **Patellofemoral Joint Force** | ⭐⭐ 슬개-대퇴 OA 초기 예측 |

### IMU 배치 권장 위치 (ACL 보행 분석 기준)
```
필수(optimal):
  - Pelvis/Sacrum (하중 중심 근접)
  - Thigh (bilateral, 환측+건측)
  - Shank (bilateral)

추가 권장:
  - Foot/Shoe (GRF 추정 정확도 향상)
  - Upper back (전신 운동 파악)

샘플링 레이트:
  - 보행 분석: 100~200 Hz 충분
  - 달리기/점프: 400+ Hz 권장
```

---

# 종합 요약: 핵심 생체역학적 Feature Set

## 1차 Feature (모든 논문에서 일관되게 중요하게 언급)

| Feature | 약어 | 측정 구간 | ACL 이상 방향 | 중요도 |
|---------|------|----------|-------------|-------|
| Peak Knee Flexion Angle | KFA | Loading Response | ↓ 감소 | ⭐⭐⭐ |
| Peak Knee Extension Moment | KEM | Mid-Stance | ↓ 감소 | ⭐⭐⭐ |
| Peak Knee Flexion Moment | KFM | Mid-Stance | ↓ 감소 | ⭐⭐⭐ |
| Vertical Ground Reaction Force | vGRF | Loading Response (1st peak) | ↓ 감소 | ⭐⭐⭐ |
| Knee Flexion Excursion (ROM) | ROM_knee | Stance Phase | ↓ 감소 | ⭐⭐⭐ |

## 2차 Feature (다수 논문에서 중요하게 다뤄짐)

| Feature | 약어 | 측정 구간 | ACL 이상 방향 | 중요도 |
|---------|------|----------|-------------|-------|
| Peak Knee Adduction Moment | KAM | Mid-Stance | 불일치 (시기별) | ⭐⭐ |
| Hip Extension Moment | HEM | Mid-Stance | ↑ 증가 (보상) | ⭐⭐ |
| Knee Power Absorption | KPA | Loading Response | ↓ 감소 | ⭐⭐ |
| Limb Symmetry Index | LSI | Stance Phase | ↓ 감소 | ⭐⭐ |
| Gait Speed | GS | Full Cycle | ↓ 감소 | ⭐⭐ |

## 3차 Feature (특정 관점에서 활용)

| Feature | 약어 | 관련 관점 |
|---------|------|---------|
| Patellofemoral Joint Force | PFJ | OA 위험도 평가 |
| Tibial Internal Rotation Angle | TIR | 횡단면 이상 감지 |
| AP-GRF (Braking / Propulsion) | APGRF | 추진력 평가 |
| Knee Varus Moment | KVM | 내측 구획 OA |
| Ankle Dorsiflexion Moment | ADM | 보상 전략 평가 |

## 시간대별 회복 단계 구분 기준

| 회복 단계 | 기간 | 주요 생체역학적 변화 |
|----------|------|-------------------|
| **급성기** | < 6M | KFA 높음(부종), KEM 급감, vGRF 감소 |
| **재활 초기** | 6~12M | KFA 최저점, Quadriceps avoidance 최심 |
| **재활 중기** | 1~3Y | 점진 회복, 양측 비대칭 지속 |
| **장기 추적** | ≥ 3Y | KFA 부분 회복, KFM 낮은 상태 유지 |
| **완전 회복 예상** | ≈ 6Y | Kinematic(각도) 정상화, moment는 여전히 낮을 수 있음 |

---

## 방법론적 주의사항

1. **보행 속도 보정 (Confounding)**: 속도 차이가 KEM, vGRF에 직접 영향 → 그룹 간 비교 시 속도 통제 또는 공변량으로 처리 필수
2. **양측 분석**: ACLR측 단독 분석이 아닌 건측(contralateral)의 보상 작용 동시 분석 필요
3. **이식편 유형**: Patellar tendon vs Hamstring tendon graft에 따라 KFM, KAM 차이 있음
4. **성별 고려**: 여성이 ACL 부상 위험 높고 신경근육 제어 패턴이 다름 → 성별 층화 분석 권장
5. **IMU 측정 시**: Pelvis+Thigh+Shank bilateral 배치가 최적. 100~200Hz로 보행 분석 가능

---

*작성일: 2026-04-01 | 분석 논문 수: 7편 | 참고: agent_temp/ 내 추출된 텍스트 기반*

---

# 📋 논문 전체 요약 표

| 번호 | 논문 제목 | 설명 | 결과 | 핵심 Feature |
|:---:|---------|------|------|------------|
| 1 | **Bilateral Gait Six and Twelve Months Post-ACL Reconstruction Compared to Controls** (Hooper et al., 2002) | ACLR 수술 후 6개월, 12개월 시점의 보행 운동학·운동역학 변수를 건강한 대조군 및 건측 다리와 비교. 환측의 보상 전략(stiffened-knee) 패턴을 정량화한 종단 연구. | · 6M, 12M 모두 ACLR측의 Peak KFA와 KEM이 건측 대비 유의하게 낮음<br>· vGRF 1st peak(하중 수용) 감소 → 체중 부하 회피 확인<br>· 12M 시점에 vGRF는 회복 추세, but 운동역학적 비대칭 지속 | Peak KFA, Peak KEM, vGRF (1st / 2nd peak), Knee Flexion Moment, Concentric / Eccentric Power |
| 2 | **Gait Analysis post ACL Reconstruction: Knee Osteoarthritis Perspective** (Al-Amin et al., 2015) | ACLR 후 보행 역학 변화가 장기적으로 무릎 골성관절염(OA) 발병에 미치는 영향을 문헌 기반으로 분석. 기계적 부하 변화와 연골 퇴행 메커니즘 규명. | · KFA 감소 → 연골 면압 분포 변화<br>· KAM은 내측 구획 부하의 강력한 OA 예측인자<br>· ACLR 후 Patellofemoral OA 위험이 초기(1년 이내)부터 증가<br>· KEM 감소는 장기적 관절 퇴행과 연관 | KFA, KEM, KAM (Knee Adduction Moment), Knee Varus Moment, Patellofemoral Joint Force, Tibio-femoral Contact Force |
| 3 | **Gait Patterns Differ Between ACL-Reconstructed Athletes Who Pass Return-to-Sport Criteria and Those Who Fail** (Di Stasi et al., 2013) | ACLR 6개월 후 RTS(Return-to-Sport) 기준 통과/실패 그룹 간 보행 패턴 차이 비교. 보행 생체역학 변수로 RTS 성공 여부를 구분하는 지표 탐색. | · RTS-Fail 그룹: KFA 22.6° vs RTS-Pass 25.1° (p<0.05)<br>· RTS-Fail 그룹: KEM 0.37 vs RTS-Pass 0.42 Nm/kg·m (p<0.05)<br>· RTS-Fail 그룹에서 Hip Extension Moment 보상 증가<br>· Knee Power Absorption이 실패 그룹에서 더 크게 감소 | Peak KFA, Knee Extension Moment, Hip Extension Moment, Knee Power Absorption, Knee Flexion Excursion, Limb Symmetry Index (LSI) |
| 4 | **Progressive Changes in Walking Kinematics and Kinetics After ACL Injury and Reconstruction: A Review and Meta-Analysis** (Gao & Zheng, 2014) | ACL 손상(ACLD) 및 ACLR 후 시간 경과에 따른 보행 운동학·운동역학 변화 궤적을 메타분석. 수술 전후 각 시기별 변화 패턴과 보행 속도의 혼동 효과 정량화. | · ACLD→ACLR: 초기(<6M) KFA 일시적 증가 후 1~3Y에 최저점<br>· KEM은 ACLR 후 전 기간에 걸쳐 지속적으로 낮은 수준<br>· KAM: ACLD 시 낮았다가 ACLR 후 일부 회복<br>· **보행 속도(Gait Speed)가 KEM·vGRF에 유의한 혼동 효과** | Peak KFA (시기별 변화), Peak KEM, Peak KFM, KAM, Gait Speed (covariable), Tibial Internal Rotation |
| 5 | **Knee Kinematics and Joint Moments During Gait Following ACLR: A Systematic Review and Meta-Analysis** (Hart et al., 2016) | 34개 연구를 포함한 대규모 체계적 리뷰. 수술 후 4개 시기(<6M / 6~12M / 1~3Y / ≥3Y)별로 시상면·관상면·횡단면 운동학 및 모멘트를 메타분석. | · 시상면 변화가 가장 일관되고 임상적으로 중요<br>· KFA: 초기 증가 → 1~3Y 최대 감소(SMD -2.21)<br>· KFM: 6~12M에 대조군 대비 유의하게 낮음(SMD -0.76)<br>· KAM: ≥3Y에서 정상화(강한 증거) → OA에서의 역할 불명확<br>· 이식편 유형(Patellar vs Hamstring)이 KFM·KAM에 영향 | Peak KFA (SMD별), Peak KFM, Peak KEM, Peak KAM, Knee Internal Rotation Angle, Knee External Rotation Moment |
| 6 | **Movement Patterns of the Knee During Gait Following ACL Reconstruction: A Systematic Review and Meta-Analysis** (Kaur et al., 2016) | 40개 연구 포함. 보행·계단·달리기 3가지 과제에서 ACLR 무릎의 운동학·운동역학 패턴 및 시간 경과에 따른 회복 궤적 분석. | · 보행: KFA 대조군과 차이 없음, but 건측 대비 4.3° 적음<br>· KFM: 보행·계단 모두 대조군·건측 대비 감소(강한 증거)<br>· KAM: 수술 1년 내 낮음 → 4~5년 후 오히려 대조군보다 높아짐<br>· KFA 회복 ≈ 6년, KFM은 5년 후에도 여전히 낮음<br>· 보행 속도를 bias 평가 항목으로 명시 처리 | Peak KFA (walking / stair / running), Peak KFM, Peak KEM (stair), Peak KAM, Gait Speed (confounding), Knee Adduction Moment (stair descent) |
| 7 | **Estimation of Kinetics Using IMUs to Monitor and Aid in Clinical Decision-Making during ACL Rehabilitation** (Krishnakumar et al., 2024) | IMU(관성 측정 장치)로 보행 운동역학(GRF, 관절 모멘트)을 추정하는 71개 알고리즘을 체계적으로 리뷰. ACL 재활 임상 적용 가능성과 한계 분석. | · IMU는 건강한 피험자 대상 sagittal 평면에서 GRF·관절 모멘트 추정에 좋은 정확도(walking KFM: nRMSE 9.46%)<br>· ACL 환자 대상 알고리즘 검증은 아직 없음<br>· 최적 센서 위치: Pelvis > Thigh > Shank 순<br>· 임상 차이(clinical difference): vGRF 0.24BW, KFM 0.035BW·m → RMSE 기준으로 이보다 낮아야 임상적 활용 가능 | VGRF, APGRF, MLGRF, Peak VGRF, Knee Flexion-Extension Moment, Knee Adduction-Abduction Moment, Patellofemoral Joint Force, Hip & Ankle Moments |
