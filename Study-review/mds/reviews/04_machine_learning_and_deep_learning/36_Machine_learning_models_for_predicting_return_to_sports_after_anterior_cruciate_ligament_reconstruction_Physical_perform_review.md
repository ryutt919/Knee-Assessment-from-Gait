# 사실검증: Machine learning models for predicting return to sports after anterior cruciate ligament reconstruction: Physical performance in early rehabilitation

- 원본 PDF: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/04_machine_learning_and_deep_learning/Machine learning models for predicting return to sports after anterior cruciate ligament reconstruction - Physical performance in early rehabilitation.pdf
- 검증 대상 요약: /Users/ryutt/Desktop/mini_ryutt/Walking/2026-06-29-Study-review/mds/papers/36_Machine_learning_models_for_predicting_return_to_sports_after_anterior_cruciate_ligament_reconstruction_Physical_perform.md
- 검증 provider: codex
- 검토 항목 수: 30
- 발견된 문제 수: 2
- 전체 판정: **신뢰 어려움**
- 판정 근거: 대부분의 핵심 요약은 원문과 일치하지만, 제외 기준 중 반월상연골 관련 항목을 원문보다 넓게 서술한 중대 오류가 있으며, 레퍼런스 항목의 원문 위치 표기 오류도 확인된다.

## 발견된 문제

### 1. [연구 설계와 대상] 사실불일치 (중대)

- 요약 문장: “연구 대상자는 2016년 6월부터 2022년 4월 사이에 수술을 받고 수술 후 3개월 및 12개월 시점에 요구되는 모든 테스트를 완료했으며 동반 다발 인대 손상, 골절, 반월상 연골 봉합/절제술, 개정 ACLR 등의 제외 기준에 해당하지 않는 환자들로 구성되었다.”
- 설명: 요약은 제외 기준을 '반월상 연골 봉합/절제술'로 넓게 적었지만, 원문은 'meniscal root repair'와 'subtotal or total meniscectomy'를 제외 기준으로 제시한다. 모든 반월상연골 봉합이나 절제술이 제외된 것처럼 읽혀 대상자 기준이 달라진다.
- 원문 근거: “The exclusion criteria were as follows: concomitant multiple ligament injury, fracture, meniscal root repair, cartilage repair, osteotomy to correct mechanical alignment, subtotal or total meniscectomy, revision ACLR, and history of knee surgery on the involved and uninvolved sides.” (PAGE 2, Methods - Patients)
- 수정 제안: 연구 대상자는 2016년 6월부터 2022년 4월 사이에 수술을 받고 수술 후 3개월 및 12개월 시점의 요구 검사를 완료했으며, 동반 다발 인대 손상, 골절, 반월상연골 뿌리 봉합, 연골 복원, 정렬 교정 절골술, 반월상연골 아전절제 또는 전절제, 재수술 ACLR, 양측 무릎 수술 과거력 등의 제외 기준에 해당하지 않는 환자들로 구성되었다.

### 2. [레퍼런스할 수 있는 내용] 인용표기오류 (경미)

- 요약 문장: “원문 위치: PAGE 5, Results”
- 설명: 해당 항목의 원문 발췌문은 SOURCE_TEXT에서 PAGE 1의 Abstract - Results에 그대로 제시된다. PAGE 5 Results에는 같은 결과가 더 풀어 서술되어 있으나, 요약에 적은 직접 발췌문과 정확히 일치하는 위치는 PAGE 1이다.
- 원문 근거: “Random forest models in the test set best predicted the RTS success based on the single-leg hop test (area under the curve [AUC], 0.952) and Tegner activity score (AUC, 0.949).” (PAGE 1, Abstract - Results)
- 수정 제안: 원문 위치: PAGE 1, Abstract - Results

## 원문에서 확인 불가능한 항목

- “분석 provider: antigravity” — SOURCE_TEXT에는 요약 생성 또는 분석 provider 정보가 없으므로 원문 근거로 확인할 수 없다.
