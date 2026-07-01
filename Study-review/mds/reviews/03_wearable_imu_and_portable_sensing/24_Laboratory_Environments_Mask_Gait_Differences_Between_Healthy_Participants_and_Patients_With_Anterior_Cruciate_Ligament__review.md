# 사실검증: Laboratory Environments Mask Gait Differences Between Healthy Participants and Patients With Anterior Cruciate Ligament Reconstruction

- 원본 PDF: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/03_wearable_imu_and_portable_sensing/Laboratory Environments Mask Gait Differences Between Healthy Participants and Patients With Anterior Cruciate Ligament Reconstruction.pdf
- 검증 대상 요약: /Users/ryutt/Desktop/mini_ryutt/Walking/2026-06-29-Study-review/mds/papers/24_Laboratory_Environments_Mask_Gait_Differences_Between_Healthy_Participants_and_Patients_With_Anterior_Cruciate_Ligament_.md
- 검증 provider: codex
- 검토 항목 수: 37
- 발견된 문제 수: 2
- 전체 판정: **일부 수정 필요**
- 판정 근거: 대부분의 요약 항목은 SOURCE_TEXT와 일치하지만, 일부 표현에서 원문보다 강한 해석이 들어가거나 번역 오탈자가 있어 경미한 수정이 필요하다.

## 발견된 문제

### 1. [방법] 번역오류 (경미)

- 요약 문장: “자이로스코프의 각속도는 차단 주파수 0.25 Hz의 고통과 필터와 35 Hz의 저통과 필터로 필터링되었다. _(근거: PAGE 3, II. METHODS)_”
- 설명: '고통과 필터'는 원문의 high-pass filter를 잘못 옮긴 오탈자성 번역이다. 의미상 '고역통과 필터'가 맞다.
- 원문 근거: “Angular velocities from the gyroscopes were filtered with a high-pass filter with a cutoff frequency of 0.25 Hz, followed by a low-pass filter with a cut-off frequency of 35 Hz.” (PAGE 3, II. METHODS)
- 수정 제안: 자이로스코프의 각속도는 차단 주파수 0.25 Hz의 고역통과 필터와 35 Hz의 저역통과 필터로 필터링되었다.

### 2. [생각해볼 내용] 과장 (경미)

- 요약 문장: “일상생활에서의 다양한 지형 조건이나 주의 분산 요소가 오히려 수술 후 환자의 보상적 보행 전략을 관찰하는 데 필수적인 자극이 될 수 있음을 의미한다. _(근거: PAGE 7, IV. DISCUSSION)_”
- 설명: 원문은 실험실의 주의 분산이 없는 환경이 보상적 보행 전략을 가릴 수 있고, 일상생활의 지형 차이는 통제하지 못한 한계라고 설명한다. 그러나 요약은 다양한 지형 조건이나 주의 분산 요소가 '필수적인 자극'이라고 더 강하게 해석한다.
- 원문 근거: “Distraction-free laboratory gait may inadvertently cause a participant to mask compensatory gait strategies in the laboratory.” (PAGE 7, IV. DISCUSSION)
- 수정 제안: 주의 분산이 거의 없는 실험실 보행은 참가자가 보상적 보행 전략을 가리게 만들 수 있으므로, 일상생활 보행 관찰이 실험실에서 드러나지 않는 보행 양상을 포착하는 데 도움이 될 수 있다.

## 원문에서 확인 불가능한 항목

- “분석 provider: antigravity” — SOURCE_TEXT에는 요약 생성 또는 분석 provider에 관한 정보가 없어 원문 근거로 확인할 수 없다.
