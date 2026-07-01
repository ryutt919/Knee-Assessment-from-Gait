# 사실검증: Clinically relevant predictive modeling for personalized ACL reconstruction classification

- 원본 PDF: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/04_machine_learning_and_deep_learning/Clinically relevant predictive modeling for personalized ACL reconstruction classification.pdf
- 검증 대상 요약: /Users/ryutt/Desktop/mini_ryutt/Walking/2026-06-29-Study-review/mds/papers/31_Clinically_relevant_predictive_modeling_for_personalized_ACL_reconstruction_classification.md
- 검증 provider: codex
- 검토 항목 수: 26
- 발견된 문제 수: 3
- 전체 판정: **일부 수정 필요**
- 판정 근거: 대부분의 요약은 원문과 일치하지만, 일부 항목에서 원문보다 강한 인과 표현이나 실현 가능성을 단정한 표현이 있어 경미한 수정이 필요하다.

## 발견된 문제

### 1. [핵심 결과] 인과관계오용 (경미)

- 요약 문장: “회복 기간이 긴 환자들일수록 기계학습 모델이 높은 신뢰도로 분류하기 어려운 경향을 보였는데, 이는 회복 과정이 진행됨에 따라 움직임 패턴이 정상화되어 부상당하지 않은 대조군의 보행 패턴과 유사해지기 때문이다. _(근거: PAGE 5, 4.4. Patient-specific factor analysis)_”
- 설명: 원문은 회복 기간이 긴 참가자의 보행 패턴이 높은 신뢰도로 분류되기 어려운 경향이 있고, 이것이 회복이 진행되며 보행 패턴이 정상화된다는 임상적 이해를 뒷받침한다고 설명한다. 요약은 이를 '때문이다'로 표현해 원문보다 인과를 더 단정했다.
- 원문 근거: “These findings suggest that participants with longer recovery durations generally exhibit gait patterns that are more difficult for the model to classify with high confidence. This trend supports the clinical understanding that as recovery progresses, movement patterns gradually normalize, becoming more similar to uninjured gait.” (PAGE 5, 4.4. Patient-specific factor analysis)
- 수정 제안: 회복 기간이 긴 참가자들은 모델이 높은 신뢰도로 분류하기 어려운 보행 패턴을 보이는 경향이 있었으며, 저자들은 이 결과가 회복이 진행됨에 따라 움직임 패턴이 점차 정상 보행에 가까워진다는 임상적 이해를 뒷받침한다고 해석했다.

### 2. [저자 결론] 과장 (경미)

- 요약 문장: “본 설명 가능하고 개인화된 접근 방식은 기계학습 모델의 블랙박스 성격을 완화하여, 임상 의사결정을 돕는 정량적이고 객관적인 도구로서 재활 계획을 개선하고 복귀 기준을 설정하는 데 큰 도움을 줄 수 있다. _(근거: PAGE 6, 5. Discussion & conclusion)_”
- 설명: 원문은 주요 움직임 관계를 식별하고 환자 진행을 추적하는 정량 도구를 제공한다는 임상적 장점을 말하며, 향후 개인화 재활 프로토콜과 데이터 기반 복귀 기준을 가능하게 할 잠재력을 제시한다. 요약의 '복귀 기준을 설정하는 데 큰 도움'은 원문보다 실용적 효과를 더 강하게 단정한다.
- 원문 근거: “By providing interpretable results that enhance clinical decision-making, this work establishes a foundation for future research integrating quantitative gait analysis into clinical practice, potentially enabling more personalized rehabilitation protocols and data-driven return-to-sport criteria.” (PAGE 6, 5. Discussion & conclusion)
- 수정 제안: 본 설명 가능한 접근은 ACL 재건과 관련된 주요 움직임 관계를 식별하고 환자 진행을 추적할 정량 도구를 제공함으로써 임상 의사결정에 도움을 줄 수 있으며, 향후 개인화 재활 프로토콜과 데이터 기반 운동 복귀 기준 연구의 기반을 마련한다.

### 3. [생각해볼 내용] 근거불충분 (경미)

- 요약 문장: “조깅 데이터 분석이 보행 분석보다 클래스 분리도와 분류 성능 면에서 더 우수한 결과를 보인 것은, 달리기와 같이 부하가 높고 고속인 운동에서 하지 및 상지 간 협응의 미세한 기능적 불균형이나 이상 징후가 더 쉽게 드러나고 모델에 유용한 식별 신호를 제공함을 보여준다. _(근거: PAGE 4, 4.3. Dimension reduction examination)_”
- 설명: 원문은 조깅 플롯에서 군집 분리가 더 뚜렷하고 모델 성능이 더 좋았다고만 설명한다. 고부하·고속 운동에서 하지 및 상지 협응의 미세한 불균형이 더 쉽게 드러난다는 해석은 제시된 원문 근거에서 직접 확인되지 않는다.
- 원문 근거: “Notably, jogging phase plots showed more distinctly separated clusters with reduced class overlap compared to walking, suggesting that jogging captures better global data structure. This improved cluster separation aligns with the superior model performance in jogging conditions relative to walking.” (PAGE 4, 4.3. Dimension reduction examination)
- 수정 제안: 조깅 데이터는 보행 데이터보다 t-SNE 플롯에서 클래스 겹침이 적고 군집 분리가 더 뚜렷했으며, 이는 조깅 조건에서의 더 높은 분류 성능과 일치한다.

## 원문에서 확인 불가능한 항목

- “분석 provider: antigravity” — SOURCE_TEXT에는 요약 생성 또는 분석 provider 정보가 포함되어 있지 않아 원문으로 확인할 수 없다.
