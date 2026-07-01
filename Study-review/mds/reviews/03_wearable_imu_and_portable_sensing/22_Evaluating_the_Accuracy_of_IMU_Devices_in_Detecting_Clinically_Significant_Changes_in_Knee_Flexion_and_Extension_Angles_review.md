# 사실검증: Evaluating the Accuracy of IMU Devices in Detecting Clinically Significant Changes in Knee Flexion and Extension Angles

- 원본 PDF: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/03_wearable_imu_and_portable_sensing/Evaluating the Accuracy of IMU Devices in Detecting Clinically Significant Changes in Knee Flexion and Extension Angles.pdf
- 검증 대상 요약: /Users/ryutt/Desktop/mini_ryutt/Walking/2026-06-29-Study-review/mds/papers/22_Evaluating_the_Accuracy_of_IMU_Devices_in_Detecting_Clinically_Significant_Changes_in_Knee_Flexion_and_Extension_Angles.md
- 검증 provider: codex
- 검토 항목 수: 58
- 발견된 문제 수: 1
- 전체 판정: **일부 수정 필요**
- 판정 근거: 대부분의 요약 항목은 SOURCE_TEXT와 일치한다. 다만 MotionSense™ 분석 동기화 방법을 설명한 한 문장에서 원문의 ‘peak flexion to peak flexion movement cycle windows’라는 한정이 ‘각 주기 데이터’ 전체로 다소 일반화되어 경미한 수정이 필요하다.

## 발견된 문제

### 1. [방법] 과장 (경미)

- 요약 문장: “MotionSense™ 데이터의 분석은 MATLAB의 interp1 함수를 통해 100Hz로 업샘플링을 거쳐 교차상관 함수 xcorr을 적용함으로써 최대 굴곡 지점을 기준으로 각 주기 데이터를 동기화했다. _(근거: PAGE 10, Abstract)_”
- 설명: 원문은 MotionSense™ 분석에서 peak flexion-to-peak flexion으로 식별된 movement cycle windows를 cross-correlation으로 시간 동기화했다고 설명한다. 요약의 ‘각 주기 데이터’는 모든 주기 데이터 전반을 포괄하는 표현처럼 읽혀 원문의 한정된 분석 창 표현보다 약간 넓다.
- 원문 근거: “Following up-sampling to 100Hz using the MATLAB (MathWorks, 2024) interp1 function, cross-correlation was used to time synchronise the movement cycle windows identified from peak flexion to peak flexion using the xcorr MATLAB (MathWorks, 2024) function for each technology.” (PAGE 10, Abstract)
- 수정 제안: MotionSense™ 데이터 분석에서는 MATLAB interp1 함수로 100Hz 업샘플링을 수행한 뒤, peak flexion에서 peak flexion까지 식별한 movement cycle window를 xcorr 기반 교차상관으로 시간 동기화하였다.

## 원문에서 확인 불가능한 항목

- “분석 provider: antigravity” — SOURCE_TEXT에는 요약 생성 또는 분석 provider에 관한 정보가 없어 원문 대조로 확인할 수 없다.
