# THE DEVELOPMENT OF A NORMATIVE GAIT DATABASE

Chester, V. L., Tingley, M., & Biden, E. N. (확인 불가). The Development of a Normative Gait Database. Institute of Biomedical Engineering, University of New Brunswick.

## 서지정보

- 저자: Chester, V. L., Tingley, M., and Biden, E. N.
- 연도: 확인 불가
- 저널: 확인 불가
- DOI: 확인 불가
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/06_general_gait_and_other_knee_conditions/The Development of a Normative Gait Database.pdf
- 분석 provider: antigravity

> **한국어 제목**: 표준 보행 데이터베이스의 개발 (The Development of a Normative Gait Database)

## 분류 태그

- ACL 연구: false
- IMU 사용: false
- 보행 데이터: true
- Score 제시: true

## 연구 목적

- 뉴브런즈윅 대학교 생체의학공학 연구소(IBME)를 위한 자체 표준 데이터베이스를 개발하여, 동일한 임상의, 장비, 마커 시스템, 데이터 처리 기술을 바탕으로 표준 데이터와 환자 데이터를 신뢰성 있게 비교하고자 함. _(근거: PAGE 1, INTRODUCTION)_
  - 근거 원문: “the objective of this
study was to develop a normative database for the
Institute of Biomedical Engineering (IBME) at the
University of New Brunswick. In doing so, patient
data can be compared to normative data obtained by
the same clinician, equipment, marker systems, and
data processing techniques. As a result, more reliable
comparisons of data sets can be achieved.”

## 연구 설계와 대상

- 연구 부서 안내문과 캠퍼스 및 보육원 게시판을 통해 1~13세 아동 58명을 모집하고 학부모 동의를 취득함. _(근거: PAGE 1, METHOD - Subjects)_
  - 근거 원문: “Fifty-eight
children aged 1-13 years old were
recruited from the Fredericton area by distributing
research bulletins around the University of New
Brunswick campus and local daycare centres.
Parental consent was obtained prior to each child’s
participation in the study.”

## 방법

- 6대의 카메라 Vicon 512 모션 캡처 시스템을 사용하여 60Hz 속도로 반사 마커의 3차원 궤적을 추적하고, 보행 통로에 내장된 두 개의 힘판을 통해 3차원 지면 반력과 모멘트를 수집함. _(근거: PAGE 1, METHOD - Instrumentation/Apparatus)_
  - 근거 원문: “A Vicon 512 motion capture system (Oxford Metrics
Ltd.) was employed to track the three-dimensional
trajectories of reflective markers placed on the
subjects’ skin at a sampling frequency of 60 Hz. In
addition, two force plates (Kistler 9281B21 and
AMTI BP5918), collected the three-dimensional
ground reaction forces and moments during each
gait cycle.”
- 정상 보행 주기와 성공적인 힘판 접지 기준에 따라 보행 평가 시행을 선정하고, 관절 각도는 오일러 각과 투영 각의 두 가지 다른 방법으로 처리함. _(근거: PAGE 2, METHOD - Data Analysis)_
  - 근거 원문: “From the selected
trials, cadence, velocity, and percent of cycle spent in
single stance were calculated for each gait cycle. The
single gait cycle, which most closely approximated
the mean of all gait cycles on these three measures,
was selected as the final trial for analysis. Joint
angles were then processed using two different
methods: 1) Euler angles, and 2) projected angles.”
- 내장 좌표계에서 계산된 선속도, 각속도, 가속도는 5점 도함수를 이용해 구했으며, 6Hz 저역 통과 버터워스 필터로 필터링함. _(근거: PAGE 2, METHOD - Euler Method)_
  - 근거 원문: “The required absolute linear and angular
velocities and accelerations were calculated from the
embedded coordinate systems using a five-point
derivative. These data were filtered using a 6 Hz
low-pass Butterworth filter.”

## 핵심 결과

- 3~13세 아동의 71개 보행 주기에 대해 샌디에이고 평균 표준치 기준으로 분류한 결과 49%가 이상 또는 비정상으로 분류됨. _(근거: PAGE 3, RESULTS)_
  - 근거 원문: “The
classification of IBME’s kinematic data for
children aged 3-13 years, based on San Diego mean
normative values, resulted in 49% of cycles being
classified as unusual or abnormal.”
- 분류기를 IBME 표준 데이터로 재보정하였을 때, 새로운 지수는 94%의 보행 주기를 정상으로 분류하였으며 3세 미만 아동의 미성숙 보행 패턴을 80% 감지해 냄. _(근거: PAGE 3, RESULTS)_
  - 근거 원문: “However, when
the statistical classifier was recalibrated using the
IBME normative data, the new index gave results
similar to those of Tingley et al: the score behaved
like an F11,61 statistic for the training data, classifying
94% of cycles as normal. Further testing using the
gait patterns of younger children showed that the
classifier was also capable of detecting 80% of
immature (i.e. abnormal) gait patterns.”
- 각 실험실 간의 시상면 관절 각도 변위 곡선의 공변량 구조 및 평균 각도 패턴을 비교하는 두 가지 통계 테스트 모두 유의수준이 매우 높게 나타남 (p=0.000). _(근거: PAGE 3, RESULTS)_
  - 근거 원문: “Both tests
yielded highly significant P-values (p=0.000).”

## 저자 결론

- 3차원 오일러 각을 사용하는 보행 분석은 동일한 알고리즘을 사용하여 산출된 표준 데이터를 기준으로 삼아야 하며, 보행 정상성을 평가하는 통계 분류기도 최신 데이터베이스로 재보정되어야 함. _(근거: PAGE 4, CONCLUSION)_
  - 근거 원문: “Gait analyses using three-dimensional Euler angles
should refer to normative data developed using the
same algorithms. In addition, statistical classifiers of
gait normality should be recalibrated on more recent
databases. Efforts to develop new and large
normative databases with modern equipment and
processing techniques are warranted.”

## 연구의 한계

- 확인된 내용 없음

## 생각해볼 내용

- 서로 다른 기술적 및 계산적 방식(Euler 대 Projected 각도 등)으로 생성된 데이터베이스는 동일 지표에서도 편차를 보이기 때문에, 외부 표준 보행 데이터베이스를 오용할 경우 오진이 발생할 수 있음을 나타냄. _(근거: PAGE 3, DISCUSSION)_
  - 근거 원문: “It is
possible that a patient’s gait data could be incorrectly
diagnosed as abnormal using normative data from
other labs. The results of this study suggest that
databases developed using different technological
and computational methods will show different
normative values.”

## 이 연구가 지적한 선행연구의 문제점

- 마커 셋팅, 데이터 처리 방식, 임상의의 숙련도 차이 및 모션 캡처 장비 기술 수준 차이로 인해 각 실험실 간에 수집된 표준 보행 데이터를 일대일로 비교하기가 어려움. _(근거: PAGE 1, INTRODUCTION)_
  - 근거 원문: “However, it is often difficult to
compare data across labs due to differences in marker
sets, data processing techniques, and reliability of the
clinician. As a result, caution must be exercised
when comparing patient data to normative results
obtained from other labs. Further difficulty in
comparing patient data to normative data sets is due
to advances in computer technology which have
dramatically improved motion analysis systems and
data processing capability over the last decade.”

## 이 연구의 해결 방식과 기여

- 뉴브런즈윅 대학교 생체의학공학 연구소(IBME)의 마커 시스템, 계측기, 동일한 데이터 처리 기법을 이용해 표준 보행 데이터베이스를 수집 및 제공하여 신뢰할 수 있는 환자 데이터 비교 기준을 마련함. _(근거: PAGE 1, INTRODUCTION)_
  - 근거 원문: “the objective of this
study was to develop a normative database for the
Institute of Biomedical Engineering (IBME) at the
University of New Brunswick. In doing so, patient
data can be compared to normative data obtained by
the same clinician, equipment, marker systems, and
data processing techniques. As a result, more reliable
comparisons of data sets can be achieved.”

## 레퍼런스할 수 있는 내용

### 1. 보행 분석의 주요 목적

- 원문 발췌: “One of the main objectives of gait
analysis is to identify deviations in a patient’s gait
from ‘normal’ movement patterns. The underlying
causes of these abnormal movement patterns are then
identified and treatment recommendations are
formulated (Davis, 1997).”
- 한국어 번역: 보행 분석의 주요 목적 중 하나는 환자의 보행이 '정상' 운동 패턴에서 벗어나는 편차를 식별하는 것이다. 그런 다음 이러한 비정상적인 운동 패턴의 근본 원인을 찾아내고 치료 권장 사항을 수립한다.
- 원문 위치: PAGE 1, INTRODUCTION
- 원문 내 인용표기: (Davis, 1997)
- 해당 선행문헌: Davis, R. B. (1997). Reflections on clinical gait analysis. Journal
of Electromyography and Kinesiology, 7 (4), p. 251-257.
- 주장 유형: background_citation
- 활용 맥락과 주의: 임상 보행 분석을 수행하는 근본적인 목표와 치료 계획 도출의 논리적 배경을 서술할 때 인용할 수 있음.

### 2. 관절 시상면 각도의 일관성

- 원문 발췌: “sagittal hip, knee, and
ankle joint angles tend
to demonstrate greater consistency across labs than
smaller rotations in other planes (Biden et al., 1987).”
- 한국어 번역: 시상면에서의 엉덩관절, 무릎관절, 발목관절 각도는 다른 면에서의 더 작은 회전들에 비해 실험실 간에 더 큰 일관성을 보이는 경향이 있다.
- 원문 위치: PAGE 3, METHOD - Statistical Analysis
- 원문 내 인용표기: (Biden et al., 1987)
- 해당 선행문헌: Biden, E.N., Olshen, R.A., Sutherland, D.H., Gage, J., & Kadaba,
M. (1987). Comparison of gait data from multiple labs.
Transactions of the Orthopaedic Research Society, January.
- 주장 유형: background_citation
- 활용 맥락과 주의: 다른 보행 평면에 비해 시상면 관절 각도 측정값이 여러 실험실 간에도 신뢰성 있게 비교하기 용이함을 뒷받침하는 근거로 사용 가능.

### 3. 각도 계산 방식에 따른 데이터베이스 유사도

- 원문 발췌: “IBME’s normative
data showed more similarity to the San Diego values
when joint angles were calculated as projected angles
instead of Euler angles.”
- 한국어 번역: 관절 각도를 오일러 각 대신 투영 각으로 계산했을 때, IBME의 표준 데이터가 샌디에이고 값들과 더 유사한 것으로 나타났다.
- 원문 위치: PAGE 4, DISCUSSION
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: 보행 데이터베이스 구축 및 각도 변환 시 오일러 각과 평면 투영 각 중 어떤 알고리즘을 사용하느냐에 따라 정량적 결과가 달라진다는 점의 근거로 사용 가능.
