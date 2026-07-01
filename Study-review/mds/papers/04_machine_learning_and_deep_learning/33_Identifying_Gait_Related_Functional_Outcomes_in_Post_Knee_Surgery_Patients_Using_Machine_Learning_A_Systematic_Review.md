# Identifying Gait-Related Functional Outcomes in Post-Knee Surgery Patients Using Machine Learning: A Systematic Review

Kokkotis, C., Chalatsis, G., Moustakidis, S., Siouras, A., Mitrousias, V., Tsaopoulos, D., Patikas, D., Aggelousis, N., Hantes, M., Giakas, G., Katsavelis, D., & Tsatalas, T. (2023). Identifying Gait-Related Functional Outcomes in Post-Knee Surgery Patients Using Machine Learning: A Systematic Review. International Journal of Environmental Research and Public Health, 20(1), 448. https://doi.org/10.3390/ijerph20010448

## 서지정보

- 저자: Christos Kokkotis, Georgios Chalatsis, Serafeim Moustakidis, Athanasios Siouras, Vasileios Mitrousias, Dimitrios Tsaopoulos, Dimitrios Patikas, Nikolaos Aggelousis, Michael Hantes, Giannis Giakas, Dimitrios Katsavelis, Themistoklis Tsatalas
- 연도: 2023
- 저널: International Journal of Environmental Research and Public Health
- DOI: https://doi.org/10.3390/ijerph20010448
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/04_machine_learning_and_deep_learning/Identifying Gait-Related Functional Outcomes in Post-Knee Surgery Patients Using Machine Learning - A Systematic Review.pdf
- 분석 provider: antigravity

## 연구 목적

- 무릎 수술 후 환자의 보행 관련 변화를 감지하고 기계 학습 알고리즘을 사용해 기능적 회복 상태를 결정하는 연구 결과를 종합하고 요약하고자 한다. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “The scope of this study is to summarize the results of a systematic literature review on the identification of gait-related changes and the determination of the functional recovery status of patients after knee surgery using advanced machine learning algorithms.”
- 정형외과 수술 후 보행 분석의 생체역학 데이터를 이용하고 기계 학습 또는 딥러닝 기법을 활용해 무릎 관절의 재활 단계를 평가한 기존 연구들을 확인하여 포괄적인 개요를 제공한다. _(근거: PAGE 2, 1. Introduction)_
  - 근거 원문: “The aim of this review is to identify studies that have utilized machine learning (ML) or deep learning techniques to evaluate the rehabilitation stage of the knee joint following major orthopedic surgery using biomechanical data from gait analysis.”

## 연구 설계와 대상

- 이 연구는 PRISMA 가이드라인에 근거하여 Scopus, PubMed, Semantic Scholar를 포함한 여러 데이터베이스를 검색해 진행된 체계적 문헌 고찰이다. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “The current systematic review was conducted using multiple databases in accordance with the PRISMA guidelines, including Scopus, PubMed, and Semantic Scholar.”
- 데이터베이스 검색을 통해 찾은 총 405개의 논문 중 선정 기준을 충족하고 보행 데이터를 바탕으로 기계 학습을 사용해 수술 후 회복 상태를 정량화한 6개의 논문이 최종 분석에 사용되었다. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “Six out of the 405 articles met our inclusion criteria and were directly related to the quantification of the recovery status using machine learning and gait data.”

## 방법

- MEDLINE(PubMed), Scopus, Semantic Scholar 데이터베이스를 체계적으로 검색했고 수동 참고문헌 검토 검색을 병행하였으며, 영어가 아닌 초록 및 학회 초록 등은 제외되었다. _(근거: PAGE 4, 2.2. Literature Search)_
  - 근거 원문: “The following databases were searched systematically: (a) MEDLINE (through PubMed), (b) Scopus, and (c) Semantic Scholar. In addition, a manual search was also conducted on Google Scholar to identify articles cited by the collected papers quoting the retrieved papers.”
- 포함된 비무작위 연구들의 품질 평가는 6개 항목(수행 성능 지표, 데이터셋 분포, 정답 라벨 결정, 입력으로 사용된 특징 세트, 정보 공개, 연구 목표)의 체크리스트를 포함하는 수정된 MINORS 지표를 사용하여 수행되었다. _(근거: PAGE 5, 2.6. Quality Assessment)_
  - 근거 원문: “The quality of non-randomized studies was evaluated using a modified methodologic index (MINORS) [22,23]. The following information was considered on a six-item checklist: performance metrics, dataset distribution, ground truth label determination, the feature set that was used as inputs, disclosure, and the aim of the study.”

## 핵심 결과

- 문헌 검토 대상인 6개의 논문 중 5개는 무릎 인공관절 치환술(TKA) 환자를, 1개는 전방십자인대(ACL) 수술 환자를 대상으로 분류 과제를 수행한 연구였다. _(근거: PAGE 6, 3. Results)_
  - 근거 원문: “The included studies in this systematic review were classified into the following application domains: (i) TKA surgery (5 studies) and (ii) ACL surgery (1 study).”
- Emmerzaal 등의 TKA 환자 분류 연구는 수술 후 6주의 보행 데이터를 이용해 학습된 로지스틱 회귀 모델이 수술 후 3, 6, 12개월의 회복 상태 변화를 67.3% 정확도로 감지할 수 있음을 보고했다. _(근거: PAGE 7, 3.1. TKA Surgery)_
  - 근거 원문: “A comparison investigation led them to the conclusion that an LR classifier trained on six weeks of post-operative biomechanical data during walking was responsive to changes at 3, 6, and 12 months post-TKA (with a 67.3% accuracy).”
- Martins 등의 TKA 연구에서는 차원 축소법인 KPCA와 다중 클래스 SVM 분류 모델의 결합을 통해 보행 보조 기기 간 차이를 식별하는 데 98%의 분류 정확도를 나타냈다. _(근거: PAGE 7, 3.1. TKA Surgery)_
  - 근거 원문: “They used a multiclass SVM for classification, and this combination of KPCA and MSVM achieved an accuracy of 98%.”
- ACL 수술 영역의 Kokkotis 등의 연구에서는 21개의 생체역학적 매개변수를 피처로 사용하여 SVM 분류기가 최고 94.95%의 분류 정확도를 달성했다. _(근거: PAGE 8, 3.2. ACL Surgery)_
  - 근거 원문: “The best score was achieved (94.95% accuracy) by the SVM classifier, which employed 21 biomechanical parameters.”

## 저자 결론

- 기계 학습 기반의 견고하고 설명 가능한 설명력 모델을 통해 재활 과정에서의 보행 회복 상태 및 핵심 매개변수 기여도를 평가함으로써 임상의에게 비침습적이고 강력한 진단 및 예후 판정 도구를 제공할 수 있다. _(근거: PAGE 10, 4. Discussion and Conclusions)_
  - 근거 원문: “AI is a valuable tool for identifying gait-related changes in post-knee surgery patients. The creation of robust explainable ML models for quantifying the recovery status during the rehabilitation process and the understanding of the contribution of the selected gait biomechanical parameters in the model’s output could lead to the creation of non-invasive and more powerful diagnostic and prognostic tools for clinicians.”
- 정형외과 영역에서의 AI 적용은 개인별 맞춤 재활 중재를 개발하여 비정상 보행 패턴을 조기에 교정하고 무릎 골관절염(OA)의 발생 위험을 차단하는 데 중요한 기여를 할 수 있다. _(근거: PAGE 10, 4. Discussion and Conclusions)_
  - 근거 원문: “Hence, AI in the field of orthopedics may play a key role in forming new personalized rehabilitation interventions for the modification of abnormal gait patterns and subsequently avoid the development of knee OA.”

## 연구의 한계

- 이 연구는 PRISMA 권고사항을 부합하여 수행된 체계적 문헌 고찰이지만, 분석된 문헌들의 이질성으로 인하여 공식적인 정량적 메타 분석은 포함하지 못했다. _(근거: PAGE 10, 4. Discussion and Conclusions)_
  - 근거 원문: “This paper is a systematic review that adheres to the PRISMA recommendations but excludes a more formal quantitative meta-analysis.”
- 본 문헌 검색 과정에서 3개의 주요 온라인 데이터베이스만 활용하고 회색 문헌을 배제한 점이 검토된 최종 포함 연구 수가 적게 식별되는 원인이 되었을 수 있다. _(근거: PAGE 10, 4. Discussion and Conclusions)_
  - 근거 원문: “This results from the observed heterogeneity of the identified studies as limitation can be considered the fact that only three online databases (PubMed, Scopus, and Semantic Scholar) were employed, and the exclusion of the grey literature may have led to the identification of a relatively small number of included studies.”

## 생각해볼 내용

- 해당 무릎 수술 재활 예측 분야는 아직 연구가 완전히 개발되지 않은 초기 단계로 보이며, 2019년 이전의 문헌이 현격히 적은 이유는 컴퓨팅 파워와 보행 관련 빅데이터 구축 수준의 한계 때문일 수 있다. _(근거: PAGE 8, 4. Discussion and Conclusions)_
  - 근거 원문: “From the literature review, it emerged that this area of research is untapped. There is a gap in the existence of literature before 2019, which is possibly resulting from the limited computing power and the non-existence of big data in this field.”
- 체계적 문헌 고찰에 포함된 6개 연구 모두 실제 환자 임상 현장에 적용할 만큼 외부 독립 데이터셋을 사용한 신뢰성 검증 과정이 없었다는 점이 핵심적인 기술적 보완 과제로 판단된다. _(근거: PAGE 10, 4. Discussion and Conclusions)_
  - 근거 원문: “It is noteworthy that none of the employed studies were validated against an external dataset.”

## 이 연구가 지적한 선행연구의 문제점

- TKA 수술과 재활 이후 환자별 만족 수준과 회복 속도는 동일하지 않으며, 약 11~20%의 환자가 잔존하는 기능적 한계로 인한 수술 후 불편감을 계속 호소한다. _(근거: PAGE 2, 1. Introduction)_
  - 근거 원문: “Previous studies have shown that not all patients have the same rate of satisfaction after surgery and rehabilitation [6–8]. Approximately 11–20% of patients experience discomfort following TKA, which is related to persisting functional impairments [6].”
- 기존에 많이 쓰이던 설문지 형태의 환자 보고 결과 측정(PROM) 방법은 평가 주체나 환자의 주관적 응답에 따른 회상 편향의 한계가 있다. _(근거: PAGE 2, 1. Introduction)_
  - 근거 원문: “However, these measures rely on subjective statements presented by the patients or the primary caregivers and can be biased due to the recall of events or the raters’ subjective comments on patient performance.”
- 전통적인 연구실 기반의 운동 평가 및 분석은 비용이 비싸며 시간이 오래 소요되고 장비 운영과 평가에 특화된 전문 지식이 필요하다. _(근거: PAGE 2, 1. Introduction)_
  - 근거 원문: “Conventional lab-based methods are time-consuming and require expensive equipment and specialized personnel.”

## 이 연구의 해결 방식과 기여

- 보행 분석에서 추출한 운동 형상학적 변수들과 기계 학습을 융합함으로써, 임상에서 저렴하고 정량화되고 비침습적으로 무릎 환자의 회복 및 재활 단계를 빠르게 평가할 수 있는 대안적 가이드라인을 제공할 수 있다. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “Modern lifestyles require new tools for determining a person’s ability to return to daily activities after knee surgery. These quantitative instruments must feature high discrimination, be non-invasive, and be inexpensive. Machine learning is a revolutionary approach that has the potential to satisfy the aforementioned requirements and bridge the knowledge gap.”
- 수술 후 재활 단계 분석에 첨단 기계 학습을 적용한 연구 경향성을 정량적으로 정리하고 개인별 치료 전략 수립을 위한 인공지능 보조 임상의 의사결정 가능성이 확대되고 있음을 보여주었다. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “The results demonstrated a recent increase in the use of sophisticated machine learning techniques that can provide robust decision-making support during personalized post-treatment interventions for knee-surgery patients.”

## 레퍼런스할 수 있는 내용

### 1. TKA 수술 후 지속적인 기능적 손상 및 잔존 불편 호소 비율

- 원문 발췌: “Approximately 11–20% of patients experience discomfort following TKA, which is related to persisting functional impairments [6].”
- 한국어 번역: 인공관절 전치환술(TKA) 후 환자의 약 11~20%는 지속적인 기능적 한계와 관련된 불편함을 경험한다.
- 원문 위치: PAGE 2, 1. Introduction
- 원문 내 인용표기: [6]
- 해당 선행문헌: 6. Gunaratne, R.; Pratt, D.N.; Banda, J.; Fick, D.P.; Khan, R.J.; Robertson, B.W. Patient Dissatisfaction Following Total Knee Arthroplasty: A Systematic Review of the Literature. J. Arthroplast. 2017, 32, 3854–3860. [CrossRef] [PubMed]
- 주장 유형: background_citation
- 활용 맥락과 주의: TKA 수술 이후에도 기능적 장애가 잔존해 11~20% 수준의 높은 비율로 불편을 느낄 수 있다는 임상적 근거로 활용할 수 있다. 다만 2차 인용에 주의해야 한다.

### 2. ACLR 수술 환자의 높은 재파열 위험성과 골관절염 조기 발병 우려

- 원문 발췌: “Alongside, the results after anterior cruciate ligament reconstruction (ACLR) can be poor, with an increased risk of ACL re-rupture and earlier onset of OA compared with healthy individuals [8].”
- 한국어 번역: 동시에 전방십자인대 재건술(ACLR) 후 결과는 나쁠 수 있으며, 건강한 일반인과 비교할 때 ACL 재파열 위험이 증가하고 골관절염(OA)의 조기 발병 가능성이 높아진다.
- 원문 위치: PAGE 2, 1. Introduction
- 원문 내 인용표기: [8]
- 해당 선행문헌: 8. Ajuied, A.; Wong, F.; Smith, C.; Norris, M.; Earnshaw, P.; Back, D.; Davies, A. Anterior Cruciate Ligament Injury and Radiologic Progression of Knee Osteoarthritis: A Systematic Review and Meta-Analysis. Am. J. Sport. Med. 2014, 42, 2242–2252. [CrossRef]
- 주장 유형: background_citation
- 활용 맥락과 주의: ACLR 수술 후 발생 가능한 부정적 예후(골관절염 조기 발생, 재파열 위험 증가)를 선행 지표로 서론 등에 인용하기 적절하며 2차 인용에 주의를 요한다.

### 3. KPCA 차원 축소와 다중 클래스 SVM 보행 보조기기 분류의 타당성

- 원문 발췌: “They used a multiclass SVM for classification, and this combination of KPCA and MSVM achieved an accuracy of 98%.”
- 한국어 번역: 그들은 분류를 위해 다중 클래스 SVM을 사용했으며, KPCA와 MSVM의 결합 방식은 98%의 분류 정확도를 달성했다.
- 원문 위치: PAGE 7, 3.1. TKA Surgery
- 원문 내 인용표기: [27]
- 해당 선행문헌: 27. Martins, M.; Santos, C.; Costa, L.; Frizera, A. Feature Reduction with PCA/KPCA for Gait Classification with Different Assistive Devices. Int. J. Intell. Comput. Cybern. 2015, 8, 363–382. [CrossRef]
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: TKA 환자의 보행 유형 및 보조 기기 사용 상태 식별에서 KPCA를 활용한 차원 축소와 다중 클래스 SVM(MSVM) 분류 모델의 우수한 예측 성능(98% 정확도)을 입증하는 결과 데이터로 사용 가능하다. 2차 인용에 유의한다.

### 4. 21개 보행 특징과 SVM 기반의 ACL 부상 및 수술 후 보행 유형 최적 분류 정확도

- 원문 발췌: “The best score was achieved (94.95% accuracy) by the SVM classifier, which employed 21 biomechanical parameters.”
- 한국어 번역: 21개의 생체역학적 매개변수를 활용한 SVM 분류기가 가장 뛰어난 점수인 94.95% 정확도를 달성했다.
- 원문 위치: PAGE 8, 3.2. ACL Surgery
- 원문 내 인용표기: [29]
- 해당 선행문헌: 29. Kokkotis, C.; Moustakidis, S.; Tsatalas, T.; Ntakolia, C.; Chalatsis, G.; Konstadakos, S.; Hantes, M.E.; Giakas, G.; Tsaopoulos, D. Leveraging Explainable Machine Learning to Identify Gait Biomechanical Parameters Associated with Anterior Cruciate Ligament Injury. Sci. Rep. 2022, 12, 6647. [CrossRef]
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: 전방십자인대 부상 환자군(ACLD, ACLR, Control)의 분류 예측 모델 구축 시 21개의 차원 축소된 보행 변수와 SVM을 통해 94.95%의 높은 성능을 도출했다는 실증 결과로 인용할 수 있다. 2차 인용에 유의한다.
