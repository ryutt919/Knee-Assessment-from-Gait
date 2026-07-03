# Investigating compensatory adjustments induced by rhythmic auditory stimulation for changes in temporal gait symmetry in lower-limb prosthetic users

Gouda, A., Arshad, M. Z., & Andrysek, J. (2026). Investigating compensatory adjustments induced by rhythmic auditory stimulation for changes in temporal gait symmetry in lower-limb prosthetic users. PLoS One, 21(6), e0351930. https://doi.org/10.1371/journal.pone.0351930

## 서지정보

- 저자: Aliaa Gouda, Muhammad Zeeshan Arshad, Jan Andrysek
- 연도: 2026
- 저널: PLoS One
- DOI: https://doi.org/10.1371/journal.pone.0351930
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/06_general_gait_and_other_knee_conditions/Investigating compensatory adjustments induced by rhythmic auditory stimulation for changes in temporal gait symmetry in lower-limb prosthetic users.pdf
- 분석 provider: antigravity

> **한국어 제목**: 하지 의지 사용자의 시간적 보행 대칭성 변화에 대하여 리듬 청각 자극에 의해 유발되는 보상적 조절 조사

## 분류 태그

- ACL 연구: false
- IMU 사용: true
- 보행 데이터: true
- Score 제시: false

## 연구 목적

- 하지 의지 사용자(LLPU)의 보행 대칭성과 전반적인 움직임 패턴에 리듬 청각 자극(RAS)이 미치는 영향을 분석하고자 함. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “This study investigated the effects of rhythmic auditory stimulation (RAS) on gait symmetry and overall movement patterns in lower-limb prosthesis users (LLPUs).”
- 시간적 보행 대칭성 향상을 위해 설계된 RAS가 전반적인 하지 운동학 패턴에 미치는 영향을 규명하고 이것이 보행에 유익한지 혹은 해로운지 판단하고자 함. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “We aimed to determine how RAS, designed to enhance temporal gait symmetry, influenced overall lower-limb kinematic patterns and whether such changes were beneficial or detrimental to gait.”

## 연구 설계와 대상

- RAS 유무에 따른 하지 의지 사용자(LLPU)와 건장한 대조군(AB)의 보행 패턴 변화를 비교 분석하는 전향적 단면 연구 설계를 적용함. _(근거: PAGE 3, Methods)_
  - 근거 원문: “This prospective cross-sectional study compared gait patterns of LLPU and AB with and without RAS.”
- 보행 평가 및 대조 기준을 확립하기 위해 10명의 건장한 성인 대조군(여성 7명, 남성 3명, 평균 연령 25.3±8.8세)을 분석에 포함시킴. _(근거: PAGE 4, Participants)_
  - 근거 원문: “The study also included ten AB participants (7 females, 3 males; 25.3±8.8 years; height 170.5±7.91 cm; weight 66.7±11.7 kg).”
- 신경학적 요인에 의한 보행 왜곡을 배제하기 위해 이전에 신경 질환을 진단받은 이력이 없는 대상자들로 모집단을 구성함. _(근거: PAGE 4, Participants)_
  - 근거 원문: “All participants had no previously known neurological disorders.”

## 방법

- 동작 변화 유도를 위한 시간적 자극(RAS) 제공 바이오피드백 시스템과 보행 운동학 데이터를 수집하는 웨어러블 모션 캡처 시스템을 결합하여 분석을 수행함. _(근거: PAGE 4, Dataset – experimental protocol)_
  - 근거 원문: “They were equipped with two systems: (1) a biofeedback system designed to provide RAS to elicit changes in temporal symmetry and (2) a wearable motion capture system to capture lower limb kinematics.”
- 웨어러블 동작 분석을 위해 7개의 3축 관성 센서가 내장된 Xsens MVN Awinda 시스템을 장착하여 관절 운동 데이터를 추출함. _(근거: PAGE 4, Dataset – experimental protocol)_
  - 근거 원문: “The wearable sensor-based motion capture system used was the Xsens MVN Awinda (Movella North America Inc., Henderson, NV, USA) that included seven triaxial inertial sensors.”
- 개별 보행 주기 동안 나타나는 표준 보행 패턴과의 총체적인 편차(보행 이상도)를 정량화하기 위해 보행 프로파일 점수(GPS)를 산출하여 지표로 활용함. _(근거: PAGE 6, Overall gait changes – gait profile score)_
  - 근거 원문: “The GPS is calculated as the root mean square (RMS) of all the mean GVSs, providing a single value that reflects overall gait abnormality with lower GPS values being indicative of better gait.”
- 기계학습 분류기의 성능 최적화를 위한 하이퍼파라미터 탐색 기법으로 5-fold 교차 검증 그리드 서치를 활용함. _(근거: PAGE 7, Feature analysis)_
  - 근거 원문: “To determine the best hyperparameters, a 5-fold cross-validation grid search was implemented.”

## 핵심 결과

- RAS 제공 시 건장한 대조군(AB)은 보행 이상도가 증가하여 유의미하게 변화한 반면(1.46±1.42°), 대퇴 절단(TFA, -0.03±0.82°)과 하퇴 절단(TTA, 0.24±0.44°) 그룹은 전반적인 보행 편차에서 유의한 변화를 나타내지 않음. _(근거: PAGE 7, Results - Gait profile score)_
  - 근거 원문: “AB GPS score increased by 1.46 ± 1.42°, while TFA and TTA changed by only –0.03 ± 0.82° and 0.24 ± 0.44°, respectively.”
- 의지 사용자의 절단 수술 후 경과 기간이 길수록 RAS 개입 하에서 전반적인 보행 패턴(GPS)이 유의하게 개선되는(감소하는) 경향을 보임(r²=0.37, p=0.02). _(근거: PAGE 7, Results - Gait profile score)_
  - 근거 원문: “For years since amputation, both preRAS and RAS show weak and non-significant relationships, but the GPS difference demonstrates a significant decrease with more years since amputation (r²=0.37, p=0.02).”
- 의지 사용자 집단의 전반적 수치 변화는 미미했으나, 개별 수준에서 분석 시 14명 중 5명은 특정 관절의 변동성(GVS) 지표에서 1.7° 이상의 임상적으로 유의미한 변화를 보임. _(근거: PAGE 8, Results - Gait profile score)_
  - 근거 원문: “Specifically, 5 of the 14 LLPU participants had a clinically significant change (greater than 1.7° in magnitude) for at least one joint-level GVS score.”
- 기계학습 분석 결과, RF 및 SVM 분류 모델은 preRAS 보행 주기와 RAS 보행 주기를 90% 이상의 고성능 정확도로 훌륭하게 분별하였으며, RF 모델의 성능이 가장 우수했음. _(근거: PAGE 7, Results - Machine learning – feature analysis)_
  - 근거 원문: “Both RF and SVM models performed well (>90%) when classifying the different conditions (preRAS and RAS), with RF performing the best (Table 7).”

## 저자 결론

- RAS는 하지 의지 사용자의 보행 대칭성을 효과적으로 개선할 수 있으며, 이 과정에서 운동학적 관절 움직임에 원치 않는 대규모 보상 조절 작용을 추가로 발생시키지 않으므로 임상 재활 훈련에 긍정적인 도구임. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “This study highlights that RAS can improve gait symmetry in LLPUs without inducing other significant, compensatory changes in their movement patterns, a positive finding for clinical rehabilitation.”
- RAS에 대응하는 보행 조정 기전은 환자 개인의 절단 위치(의지 형태) 및 의지 사용 숙련도에 따라 개별적으로 달라질 수 있으므로 임상 적용 및 처방 시 이러한 요인을 신중히 고려해야 함. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “However, our results show that individual responses can vary depending on factors such as prosthesis type and user experience and these should be considered in clinical care.”

## 연구의 한계

- 보행 분석에 있어서 상체의 움직임 지표나 동작에 가해지는 역학적 힘(kinetic measures)과 같은 추가적인 보행 지표 데이터들을 획득하여 분석하지 못함. _(근거: PAGE 14, Discussion)_
  - 근거 원문: “One limitation of this study is the lack of additional gait parameters data, such as upper-body movements or kinetic measures.”
- 실험실 환경에서의 단기적인 보상 반응 및 적응 전략만 측정했기 때문에, 장기적으로 RAS 시스템을 활용할 경우 보행 운동학에 어떤 영향이 미칠지는 밝혀지지 않음. _(근거: PAGE 14, Discussion)_
  - 근거 원문: “This study assessed only the short-term compensatory changes, and it remains unclear whether kinematics would change further with longer term use of the RAS system.”

## 생각해볼 내용

- 건장한 사람(AB)의 경우 인위적으로 주어지는 비대칭 자극(섭동)에 맞추기 위해 보폭을 늘리려 발목 저측굴곡을 줄이고 고관절 굴곡을 유의하게 증가시키는 적극적인 보상 기전을 관찰할 수 있음. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “To further explore lower-limb kinematic changes associated with temporal symmetry, a perturbation-based task involving able-bodied (AB) individuals demonstrated the presence of unique kinematic changes, including reduced ankle plantarflexion and increased hip flexion, reflecting adjustments to synchronize with the asymmetric beat.”
- 하지 의지 사용자 그룹의 경우 발목 관절 제어 범위가 제한되는 수동 고정형 발목 의지(passive fixed-ankle prostheses)를 사용했기 때문에, 건장한 대조군처럼 특정 발목 및 고관절 각도를 정밀하게 조절하지 못하고 미세하게 분산된 형태로 보행 조절을 달성함. _(근거: PAGE 13, Discussion)_
  - 근거 원문: “Similar adjustments were not observed in the LLPU group due possibly to the restricted ankle range of motion and movement, as all LLPU participants used passive fixed-ankle prostheses, limiting control over joint response and adjustments.”

## 이 연구가 지적한 선행연구의 문제점

- 재활 보행 훈련 시 하나의 특정 보행 변수(예: stance time)를 표적으로 지정하여 조정하는 것이 의도하지 않은 다른 보행 변수들이나 전체 보행 시너지 패턴에 어떠한 연쇄적 부작용을 일으키는지에 대해 충분히 다루지 못함. _(근거: PAGE 2, Introduction)_
  - 근거 원문: “However, it remains unclear and has not been thoroughly explored in research how targeting a specific gait parameter (or multiple parameters) will impact other parameters and, consequently, overall gait patterns.”
- 선행 연구(Roerdink 등)에 의하면 보폭 대칭성(step-length asymmetry)이라는 단일 지표에 의존하여 비대칭을 평가하는 방식은 상반되게 나타나는 몸통 진행이나 발 위치의 비대칭을 간과하게 만들어 불충분함. _(근거: PAGE 2, Introduction)_
  - 근거 원문: “The authors concluded that analyzing changes in a single gait parameter (for example step-length in their study) was insufficient and did not adequately capture important changes in overall gait symmetry.”

## 이 연구의 해결 방식과 기여

- 하지 의지 사용자의 보행 대칭성을 최적화하는 과제와 건장한 대조군의 보행 대칭성을 교란하는 섭동 과제를 대치 설계함으로써, 신체 기계적 제약 유무에 따라 시스템이 운동을 재조직하는 방식을 체계적으로 대비할 수 있는 프레임워크를 정립함. _(근거: PAGE 3, Introduction)_
  - 근거 원문: “By examining these two forms of adaptation under matched magnitudes of temporal symmetry change (despite opposite directions of change), we aim to characterize how constrained and unconstrained systems reorganize movement to meet similar temporal demands.”
- 전통적인 일변량 분석법을 넘어, 독립적으로 다룰 때는 포착하기 힘든 변수 간 상호작용 및 운동 양상을 다변량 기반의 시스템 수준에서 분석할 수 있도록 기계학습 모델링 기법을 보행 분석에 통합함. _(근거: PAGE 13, Discussion)_
  - 근거 원문: “The machine learning analysis further extends these findings by providing a multivariate, system-level perspective on gait adaptation, complementing traditional statistical approaches by capturing interactions that may not be evident when parameters are examined independently.”

## 레퍼런스할 수 있는 내용

### 1. 편측 하지 의지 사용자의 건측 다리 의존도 증가 경향

- 원문 발췌: “One common deviation in unilateral LLPU gait is increased reliance on the intact limb, which can generate greater propulsion and support during gait to compensate for the limitations, or perceived limitations, of the prosthetic limb [2].”
- 한국어 번역: 편측성 하지 의지 사용자(LLPU) 보행에서 흔히 나타나는 편차 중 하나는 건측 다리에 대한 의존도 증가이며, 이는 의지측 다리의 한계 또는 인지된 한계를 보상하기 위해 보행 시 더 큰 추진력과 지지력을 생성할 수 있습니다.
- 원문 위치: PAGE 2, Introduction
- 원문 내 인용표기: [2]
- 해당 선행문헌: 2. Rutkowska-Kucharska A, Kowal M, Winiarski S. Relationship between Asymmetry of Gait and Muscle Torque in Patients after Unilateral Transfemoral Amputation. Appl Bionics Biomech. 2018;2018:5190816. https://doi.org/10.1155/2018/5190816 PMID: 29755583
- 주장 유형: background_citation
- 활용 맥락과 주의: 편측 하지 절단 환자가 의지 측의 손실된 역량을 메우기 위해 지면 반발력, 지지 시간 등을 건측에 가중시키는 보행 불균형 패턴을 인용하는 기본 학술적 근거로 활용할 수 있다. 2차 인용에 주의가 필요하다.

### 2. 하지 절단 환자의 만성적 보상 보행 비대칭으로 인한 근골격계 문제 유발

- 원문 발췌: “These compensatory mechanisms, while functional, can lead to increased metabolic energy expenditure and joint stress, resulting in long-term musculoskeletal issues such as lower back pain or joint degeneration [2–4].”
- 한국어 번역: 이러한 보상 메커니즘은 기능적이기는 하지만, 대사 에너지 소비와 관절 스트레스를 증가시켜 요통이나 관절 퇴행과 같은 장기적인 근골격계 문제를 야기할 수 있습니다.
- 원문 위치: PAGE 2, Introduction
- 원문 내 인용표기: [2–4]
- 해당 선행문헌: 2. Rutkowska-Kucharska A, Kowal M, Winiarski S. Relationship between Asymmetry of Gait and Muscle Torque in Patients after Unilateral Transfemoral Amputation. Appl Bionics Biomech. 2018;2018:5190816. https://doi.org/10.1155/2018/5190816 PMID: 29755583
3. Adamczyk PG, Kuo AD. Mechanisms of Gait Asymmetry Due to Push-Off Deficiency in Unilateral Amputees. IEEE Trans Neural Syst Rehabil Eng. 2015;23(5):776–85. https://doi.org/10.1109/TNSRE.2014.2356722 PMID: 25222950
4. Richards R, van den Noort JC, Dekker J, Harlaar J. Gait Retraining With Real-Time Biofeedback to Reduce Knee Adduction Moment: Systematic Review of Effects and Methods Used. Arch Phys Med Rehabil. 2017;98(1):137–50. https://doi.org/10.1016/j.apmr.2016.07.006 PMID: 27485366
- 주장 유형: background_citation
- 활용 맥락과 주의: 보행 비대칭의 장기화가 인체 역학적 부하(관절 피로, 요통) 및 대사 효율 저하를 초래한다는 의학적 타당성을 학술지에 인용 설명할 때 직접적인 지지 근거로 사용한다.

### 3. 정상 성인 및 의지 사용자의 표준 보행 이상도(GPS) 범주 차이

- 원문 발췌: “The GPS values for typical AB gait lie in the range of 5–6° [24], whereas for LLPUs it is higher at 9.2–10.7° [25,26].”
- 한국어 번역: 전형적인 건장한 성인(AB) 보행의 GPS 값은 5–6° 범위에 있는 반면, 하지 의지 사용자(LLPUs)의 경우 9.2–10.7°로 더 높습니다.
- 원문 위치: PAGE 6, Overall gait changes – gait profile score
- 원문 내 인용표기: [24], [25,26]
- 해당 선행문헌: 24. Fukuchi CA, Duarte M. Gait Profile Score in able-bodied and post-stroke individuals adjusted for the effect of gait speed. Gait Posture. 2019;69:40–5. https://doi.org/10.1016/j.gaitpost.2019.01.018 PMID: 30660950
25. Ferreira AEK, Neves EB, Melanda AG, Pauleto AC, Iucksch DD, Knaut LAM, et al. Transtibial Amputee Gait: Kinematics and Temporal-Spatial Analysis. In: XIII Mediterranean Conference on Medical and Biological Engineering and Computing 2013: MEDICON 2013, 25-28 September 2013, Seville, Spain. 2014. p. 61–4.
26. Kark L, Vickers D, McIntosh A, Simmons A. Use of gait summary measures with lower limb amputees. Gait Posture. 2012;35(2):238–43. https://doi.org/10.1016/j.gaitpost.2011.09.013 PMID: 22000790
- 주장 유형: background_citation
- 활용 맥락과 주의: 동작분석 시 GPS(보행 프로파일 점수)의 해석 준거로서, 건강 대조군의 점수 기준과 절단 환자들의 평균적인 점수 기준을 대조 제시할 때 신뢰성 있는 선행 데이터로 활용할 수 있다.

### 4. RAS를 통한 보상 행동 없는 의지 보행 대칭성 개선

- 원문 발췌: “This study highlights that RAS can improve gait symmetry in LLPUs without inducing other significant, compensatory changes in their movement patterns, a positive finding for clinical rehabilitation.”
- 한국어 번역: 이 연구는 리듬 청각 자극(RAS)이 하지 의지 사용자(LLPU)의 전반적인 움직임 패턴에서 다른 유의미한 보상적 변화를 유발하지 않으면서 보행 대칭성을 개선할 수 있음을 보여주며, 이는 임상 재활에 긍정적인 결과입니다.
- 원문 위치: PAGE 1, Abstract
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: 본 연구의 자체 주요 성과를 인용할 때 활용한다. 청각을 활용한 재활 훈련이 전신 움직임 조절(운동학 패턴)의 다른 2차 왜곡을 수반하지 않는 안전한 훈련 전략임을 지지하는 직접 문항이다.
