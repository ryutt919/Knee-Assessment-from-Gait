# 📊 gait_analysis_global.csv 컬럼 설명

`COMP-WALK.ipynb` 노트북에서 생성하는 CSV 파일은 **보행 분석(Gait Analysis)** 데이터를 담고 있으며, 각 컬럼의 의미는 다음과 같습니다.

---

## 🔖 메타데이터 (Metadata)

| 컬럼명 | 의미 |
|--------|------|
| **group** | 참가자 그룹 (예: ACLD, ACLR, Healthy adolescents) |
| **participant** | 참가자 ID (예: ACLD1, HK1) |
| **pace_condition** | 보행 속도 조건 (fast, normal, slow) |
| **source_file** | 원본 MVNX 파일명 |

---

## 📏 보행 이벤트 수 (Gait Event Counts)

| 컬럼명 | 의미 |
|--------|------|
| **n_steps_total** | 전체 스텝(발걸음) 수 |
| **n_strides_total** | 전체 스트라이드(보폭) 수 (한 발이 땅에 닿은 후 같은 발이 다시 땅에 닿을 때까지) |
| **n_steps_trimmed** | 안정성을 위해 앞뒤 2개씩 제거한 후의 스텝 수 |
| **n_strides_trimmed** | 안정성을 위해 앞뒤 2개씩 제거한 후의 스트라이드 수 |

---

## ⚡ 보행 속도 및 케이던스 (Speed & Cadence)

| 컬럼명 | 의미 | 단위 |
|--------|------|------|
| **gait_speed_mps** | 보행 속도 | m/s (미터/초) |
| **cadence_spm** | 케이던스 (분당 스텝 수) | steps/min |

**계산 방식:**
- `gait_speed_mps = stride_length_mean_m / stride_time_mean_s`
- `cadence_spm = n_steps_trimmed / (trial_time / 60)`

---

## ⏱️ 시간 파라미터 (Temporal Parameters)

| 컬럼명 | 의미 | 단위 |
|--------|------|------|
| **step_time_mean_s** | 스텝 시간 평균 (한 발에서 다른 발까지 걸리는 시간) | 초 (s) |
| **step_time_sd_s** | 스텝 시간 표준편차 | 초 (s) |
| **stride_time_mean_s** | 스트라이드 시간 평균 (같은 발이 두 번 땅에 닿는 시간 간격) | 초 (s) |
| **stride_time_sd_s** | 스트라이드 시간 표준편차 | 초 (s) |

**의미:**
- 시간 파라미터는 보행의 **리듬과 일관성**을 나타냄
- 표준편차가 클수록 보행이 불규칙함을 의미

---

## 📐 공간 파라미터 (Spatial Parameters)

| 컬럼명 | 의미 | 단위 |
|--------|------|------|
| **step_length_mean_m** | 스텝 길이 평균 (한 발에서 다른 발까지의 거리) | 미터 (m) |
| **step_length_sd_m** | 스텝 길이 표준편차 | 미터 (m) |
| **stride_length_mean_m** | 스트라이드 길이 평균 (같은 발의 두 연속 접촉 사이 거리) | 미터 (m) |
| **stride_length_sd_m** | 스트라이드 길이 표준편차 | 미터 (m) |
| **step_width_mean_m_orth** | 보폭 너비 평균 (좌우 발 사이의 수직 거리) | 미터 (m) |
| **step_width_sd_m_orth** | 보폭 너비 표준편차 | 미터 (m) |

**의미:**
- **Step Length**: 보행의 효율성을 나타냄
- **Stride Length**: 일반적으로 step length의 약 2배
- **Step Width**: 균형과 안정성을 나타냄 (너무 넓거나 좁으면 불안정)

---

## 🦶 지지 단계 비율 (Support Phase Percentages)

| 컬럼명 | 의미 | 단위 |
|--------|------|------|
| **double_support_pct** | 양발 지지 단계 비율 (두 발이 모두 땅에 닿아있는 시간) | % |
| **single_support_L_pct** | 왼발 단일 지지 단계 비율 (왼발만 땅에 닿아있는 시간) | % |
| **single_support_R_pct** | 오른발 단일 지지 단계 비율 (오른발만 땅에 닿아있는 시간) | % |

**의미:**
- **Double Support**: 보행 속도가 느릴수록 증가
- **Single Support**: 균형 능력을 나타냄
- 좌우 single support의 차이가 크면 **비대칭 보행**을 의미

---

## 💡 주요 개념 설명

### Step vs Stride

```
Step (스텝):
  왼발 → 오른발 (1 step)
  
Stride (스트라이드):
  왼발 → 오른발 → 왼발 (1 stride = 2 steps)
```

**관계:**
- 1 Stride = 2 Steps
- Stride Length ≈ 2 × Step Length

---

### 데이터 처리 과정

1. **MVNX 파일 로드**: Xsens 모션 캡처 데이터
2. **Heel Strike 감지**: 발의 뒤꿈치가 땅에 닿는 순간 감지
3. **Trimming**: 안정성을 위해 처음과 마지막 2개 이벤트 제거
   ```python
   trim = 2
   events_trim = events[trim:-trim]
   ```
4. **파라미터 계산**: 제거 후 남은 데이터로 평균과 표준편차 계산

---

## 📊 활용 사례

### 1. 재활 평가
- ACL 재건술 후 보행 회복 정도 평가
- 좌우 비대칭성 모니터링

### 2. 그룹 간 비교
- 환자군 vs 건강한 대조군
- 연령대별 보행 패턴 차이

### 3. 속도 조건별 분석
- Fast, Normal, Slow 조건에서의 보행 전략 변화
- 속도에 따른 안정성 변화

### 4. 임상 지표
- Gait Speed: 전반적인 기능 수준
- Cadence: 보행 효율성
- Step Width: 균형 능력
- Support Phase: 안정성 및 자신감

---

## 🔍 데이터 품질 확인

### 결측치 처리
- Heel strike가 감지되지 않은 경우 → 해당 파일 스킵
- 계산 불가능한 경우 → `None` 값 할당

### 이상치 확인
- 표준편차가 평균 대비 매우 큰 경우 → 보행이 불안정
- Step width가 비정상적으로 큰 경우 → 균형 문제 가능성

---

## 📝 참고사항

- **단위 일관성**: 모든 길이는 미터(m), 시간은 초(s)
- **소수점 자리**: 대부분 3자리 또는 4자리로 반올림
- **파일 형식**: CSV (쉼표로 구분)
- **인코딩**: UTF-8

---

## 📚 관련 문서

- `COMP-WALK.ipynb`: 데이터 생성 노트북
- `gait_analysis_exploration_executed.ipynb`: 데이터 분석 노트북
- `gait_analysis_global.csv`: 생성된 데이터 파일

---

**작성일**: 2026-01-28  
**버전**: 1.0  
**작성자**: Gait Analysis Pipeline
