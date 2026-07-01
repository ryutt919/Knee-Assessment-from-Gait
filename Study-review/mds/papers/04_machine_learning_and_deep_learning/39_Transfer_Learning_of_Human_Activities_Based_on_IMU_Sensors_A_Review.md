# Transfer Learning of Human Activities Based on IMU Sensors: A Review

Ashry, S., Das, S., Rafiei, M., Baumbach, J., & Baumbach, L. (2025). Transfer Learning of Human Activities Based on IMU Sensors: A Review. IEEE Sensors Journal, 25(3), 4115-4126. https://doi.org/10.1109/JSEN.2024.3510097

## 서지정보

- 저자: Sara Ashry , Supratim Das, Mahdie Rafiei , Jan Baumbach , and Linda Baumbach
- 연도: 2025
- 저널: IEEE SENSORS JOURNAL
- DOI: 10.1109/JSEN.2024.3510097
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/04_machine_learning_and_deep_learning/Transfer Learning of Human Activities Based on IMU Sensors - A Review.pdf
- 분석 provider: antigravity

## 연구 목적

- IMU 센서 데이터를 이용한 인간 행동 인식(HAR)에 적용된 전이 학습(TL) 방법론들을 종합적으로 검토하고 분석하여 연구자와 개발자에게 종합적인 자원을 제공하는 것이다. _(근거: Page 1, Abstract)_
  - 근거 원문: “Our objective is to provide a comprehensive resource for researchers and developers by summarizing the existing activities, feature extractions, and TL techniques in the related studies.”

## 연구 설계와 대상

- PubMed, ACM, Scopus 데이터베이스에서 447개의 연구를 검색하고, 그 중 최종적으로 포함 기준을 충족하는 33개의 핵심 연구를 선정하여 분석하였다. _(근거: Page 1, Abstract)_
  - 근거 원문: “We analyzed 447 studies from PubMed, ACM, and Scopus datasets, of which we ultimately selected 33 pivotal studies that met our inclusion criteria.”

## 방법

- 본 체계적 문헌고찰은 PRISMA 가이드라인을 따르며, 연구 질문 공식화, 검색 쿼리 설정, 문헌 선택을 위한 선정 및 제외 기준 구체화를 포함한다. _(근거: Page 1, Abstract)_
  - 근거 원문: “Our methodology follows the structure of the preferred reporting items for systematic reviews and meta-analysis (PRISMA) statement by formulating precise research questions, establishing search queries, and specifying inclusion and exclusion criteria for study selection.”
- 문헌 검색은 2023년 9월 14일에 첫 번째 저자에 의해 수행되었다. _(근거: Page 4, Section II-D)_
  - 근거 원문: “The search was performed by the first author of this publication on September 14, 2023.”

## 핵심 결과

- 전이 학습은 사전 학습된 모델을 재사용함으로써 인간 행동 인식(HAR)의 성능을 향상시켰다. _(근거: Page 1, Abstract)_
  - 근거 원문: “Overall, we found that TL has enhanced HAR performance by reusing pretrained models.”

## 저자 결론

- 전이 학습을 적용할 때 부작용을 피하기 위해서는 관련된 전이 정보를 신중하게 선택하는 것이 중요하다. _(근거: Page 1, Abstract)_
  - 근거 원문: “However, it is important to carefully select relevant transfer information to avoid any potential adverse effects.”
- 활동의 특성에 알고리즘을 맞추는 것이 필수적인데, 걷기 같은 일상적 활동에는 AlexNet과 같은 단순한 모델이 적합한 반면, 청소 같은 정밀한 작업에는 DenseNet과 같은 더 복잡한 모델이 적합하다. _(근거: Page 1, Abstract)_
  - 근거 원문: “Aligning algorithms with activity nature is essential—simpler models like AlexNet are suitable for routine activities such as walking, while more complex models like DenseNet are better for intricate tasks like cleaning.”

## 연구의 한계

- 분석 대상 문헌을 영어로 출판된 연구로만 제한하여 분석 결과에 편향이 발생했을 가능성이 있다. _(근거: Page 9, Section V)_
  - 근거 원문: “First, we only included studies published in English, which could have a biased impact on the analysis.”
- 데이터 추출 과정이 첫 번째 저자(Sara Ashry) 1인에 의해서만 단독으로 수행되어 미세한 오류가 발생했을 가능성이 존재한다. _(근거: Page 10, Section V)_
  - 근거 원문: “the data extraction from the included articles was solely performed by one author (Sara Ashry), potentially led to minor errors.”

## 생각해볼 내용

- 실시간 모바일 애플리케이션의 효율성을 고려할 때, 연산 효율성이 높은 MobileNet과 같이 모바일 플랫폼 및 간단한 활동에 잘 적합한 경량화 모델을 사용하는 것을 권장한다. _(근거: Page 9, Section IV)_
  - 근거 원문: “For efficiency considerations in real-time applications, we suggest using lightweight models like MobileNet, which are well-suited for mobile platforms and simple activities due to their computational efficiency.”

## 이 연구가 지적한 선행연구의 문제점

- IMU 센서와 전이 학습을 사용하여 개인 위생 및 노인 돌봄과 같은 구체적인 행동을 평가하는 연구가 현재 부족한 실정이다. _(근거: Page 1, Abstract)_
  - 근거 원문: “We conclude that there is a lack of studies assessing specific activities, such as personal hygiene and elder care, using IMU sensors and TL.”
- 딥러닝은 검증 데이터와 동일한 분포를 가진 대량의 레이블링된 학습 데이터가 필요하지만, 실세계 시나리오에서는 충분한 학습 데이터를 수집하는 것이 비용이 많이 들고 시간이 오래 걸리며 때로는 불가능하다. _(근거: Page 2, Section I)_
  - 근거 원문: “Ideally, deep learning thrives when there is an abundance of labeled training data sharing the same distribution as the test data. However, in numerous scenarios, gathering sufficient real-world training data proves to be costly, time-consuming, or even unfeasible.”

## 이 연구의 해결 방식과 기여

- 33개의 연구를 바탕으로 IMU 센서 기반 인간 행동 인식(HAR)을 위한 전이 학습 접근법을 체계적으로 요약하여, 연구자가 실무에서 적절한 전이 학습 방안을 선택하는 데 기여한다. _(근거: Page 3, Section I)_
  - 근거 원문: “We systematically summarize TL approaches for HAR based on IMU sensors for 33 studies, which may support researchers in selecting appropriate TL approaches in practice.”

## 레퍼런스할 수 있는 내용

### 1. 헬스케어 및 낙상 감지 분야에서의 HAR의 역할

- 원문 발췌: “The importance of HAR plays a pivotal role in monitoring and understanding human behavior; among various applications, such as in healthcare, it is used to monitor patient activity, aiding in remote patient care and fall detection [2].”
- 한국어 번역: HAR의 중요성은 인간의 행동을 모니터링하고 이해하는 데 있어 중추적인 역할을 하며, 헬스케어와 같은 다양한 애플리케이션 중에서 환자 활동을 모니터링하여 원격 환자 케어 및 낙상 감지를 돕는 데 사용된다.
- 원문 위치: Page 2, Section I
- 원문 내 인용표기: [2]
- 해당 선행문헌: [2] F. Serpush, M. B. Menhaj, B. Masoumi, and B. Karasfi, “Wearable sensor-based human activity recognition in the smart healthcare system,” Comput. Intell. Neurosci., vol. 2022, pp. 1–31, Feb. 2022.
- 주장 유형: background_citation
- 활용 맥락과 주의: 헬스케어 분야에서 HAR이 낙상 감지 및 환자 상태 모니터링에 쓰이는 근거로 사용 가능하며, 2차 인용 시 [2] 논문을 참조해야 함.
