# 사실검증: Motion Sensors-Based Machine Learning Approach for the Identification of Anterior Cruciate Ligament Gait Patterns in On-the-Field Activities in Rugby Players

- 원본 PDF: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/04_machine_learning_and_deep_learning/Motion Sensors-Based Machine Learning Approach for the Identification of Anterior Cruciate Ligament Gait Patterns in On-the-Field Activities in Rugby Players.pdf
- 검증 대상 요약: /Users/ryutt/Desktop/mini_ryutt/Walking/2026-06-29-Study-review/mds/papers/37_Motion_Sensors_Based_Machine_Learning_Approach_for_the_Identification_of_Anterior_Cruciate_Ligament_Gait_Patterns_in_On_.md
- 검증 provider: codex
- 검토 항목 수: 29
- 발견된 문제 수: 2
- 전체 판정: **일부 수정 필요**
- 판정 근거: 대부분의 요약은 SOURCE_TEXT와 일치하지만, 최초성/분류 성능을 원문보다 강하게 표현한 과장 2건이 확인된다. 중대 오류는 없다.

## 발견된 문제

### 1. [연구의 해결 방식과 기여] 과장 (경미)

- 요약 문장: “성공적으로 복귀한 지 5~10년이 경과한 건강한 선수와 ACL 재건 환자를 필드(야외) 환경에서 웨어러블 센서와 데이터 기반 기계학습 모델의 결합을 통해 정밀하게 분류해낸 연구는 본 연구가 최초이다. _(근거: PAGE 3, 1. Introduction)_”
- 설명: 원문은 저자들이 알기로 해당 조합이 아직 탐구되지 않았다고 표현한다. 요약의 '본 연구가 최초'와 '정밀하게 분류'는 원문보다 단정적이고 강하다.
- 원문 근거: “To the best of the authors’ knowledge, the combination of a data-driven approach and inertial sensors to classify healthy and ACL-reconstructed subjects on-the-field (with post-ACL athletes returned to sport and with time from surgery between five and 10 years) is not yet explored.” (PAGE 3, 1. Introduction)
- 수정 제안: 저자들이 알기로, 스포츠에 복귀한 지 5~10년이 지난 ACL 재건 선수와 건강한 선수를 필드 환경에서 관성 센서와 데이터 기반 접근법을 결합해 분류한 연구는 아직 충분히 탐구되지 않았다.

### 2. [레퍼런스할 수 있는 내용] 과장 (경미)

- 요약 문장: “활용 맥락과 주의: 수술 후 장기 시점에서도 가혹한 필드 방향 전환 테스트 중 웨어러블 센서를 통해 ACL 재건군과 대조군 사이의 잔존 보행 차이를 기계학습으로 성공적으로 식별할 수 있다는 본 논문의 자체 결과를 인용할 때 사용한다.”
- 설명: 원문은 필드 스포츠 과제에서 센서와 머신러닝 접근의 feasibility와 구별 가능성을 제시하지만, '가혹한' 테스트라는 표현은 원문 표현보다 강하며, '성공적으로 식별'도 73.07% 정확도와 81.8% 민감도의 제한적 결과를 충분히 반영하지 않는다.
- 원문 근거: “The results of this study suggest the feasibility to use body-worn motion sensors and machine learning approaches for the identification of post-ACL gait patterns in athletes performing sport tasks on-the-field even a number of years after the injury occurred.” (PAGE 14, 5. Conclusions)
- 수정 제안: 활용 맥락과 주의: 수술 후 5~10년이 지난 선수의 필드 방향 전환 과제에서 웨어러블 센서와 머신러닝 접근법이 ACL 재건군의 보행 패턴 식별에 활용될 가능성을 보였다는 본 논문의 자체 결과를 인용할 때 사용한다.

## 원문에서 확인 불가능한 항목

- “분석 provider: antigravity” — SOURCE_TEXT에는 요약 생성 또는 분석 provider에 관한 정보가 없으므로 원문으로 확인할 수 없다.
