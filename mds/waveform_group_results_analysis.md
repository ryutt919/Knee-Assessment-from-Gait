---
noteId: "0809a01034cb11f19aa10dab1cd744f4"
tags: []
---
# Waveform-Based Group Analysis Results

## 문서 목적

이 문서는 [`waveform_group_analysis.py`](/Users/ryutt/Desktop/mini_ryutt/Walking/scripts/analysis/waveform_group_analysis.py)로 생성한 결과 파일들을 바탕으로,
성인 `HA`, `ACLD`, `ACLR` 3그룹의 lower-limb 3-plane 각도 차이가 어떤 방식으로 분석되었고, 실제로 어떤 결과가 나왔는지를 해석하기 위해 작성되었다.

결과 탐색과 그림 생성은 [`waveform_group_results_review.ipynb`](/Users/ryutt/Desktop/mini_ryutt/Walking/notebooks/waveform_group_results_review.ipynb)에서 수행한다.
이 문서는 그 노트북의 해석판이라고 보면 된다.

분석에 사용한 결과 파일들은 모두 [`data/processed/waveform_based`](/Users/ryutt/Desktop/mini_ryutt/Walking/data/processed/waveform_based) 아래에 저장되어 있다.

## 왜 이런 분석 구조를 썼는가

이번 분석은 `peak 하나`로 끝내는 방식보다, gait cycle 전체 파형이 그룹 차이를 어떻게 보여주는지를 먼저 확인하는 것이 목적이었다.
ACL 관련 보행 차이는 peak value 하나보다도, 보행 주기의 어느 구간에서 패턴이 달라지는지가 더 중요할 가능성이 높기 때문이다.

그래서 분석 구조를 아래 순서로 잡았다.

1. `SPM1D waveform analysis`
   gait cycle 0-100% 전체 파형을 speed별로 비교한다.
   이 단계는 “어느 관절-평면이 어느 구간에서 그룹 차이를 보이는가”를 찾기 위한 것이다.
2. `Feature summarization + LMM`
   SPM에서 차이가 난 구간과 임상적으로 이해 가능한 요약량을 feature로 만든다.
   이 단계는 “그 차이를 어떤 feature가 설명하는가”를 해석 가능한 회귀 형태로 정리하기 위한 것이다.
3. `Elastic-net ranking`
   SPM 또는 LMM에서 살아남은 feature를 대상으로 그룹 구분 기여도를 순위화한다.
   이 단계는 “어떤 요소가 상대적으로 더 중요한가”를 우선순위로 보는 보조 분석이다.

즉, 이번 파이프라인은 `탐지 -> 해석 -> 순위화` 구조다.

## 데이터가 실제로 어떻게 처리되었는가

### 1. 대상군

- 사용 그룹: `HA`, `ACLD`, `ACLR`
- 제외 그룹: `Healthy adolescents`

### 2. side basis 정렬

ACL 참가자는 `ID.csv`의 `Injured leg`를 기준으로 `injured`, `contralateral`을 정렬했다.
Healthy adult는 실제 injured side가 없기 때문에, ACLD의 injured-leg 비율에 맞춰 pseudo-injured side를 부여했다.

이 정렬을 한 이유는 단순 left/right 평균보다,
“손상측 기준으로 보았을 때 어떤 패턴이 생기는가”를 그룹 간에 직접 비교하기 위해서다.

### 3. stride 정의

한 stride는 한쪽 발의 heel strike에서 다음 같은 쪽 heel strike까지로 정의했다.

- Left stride: `L heel strike -> next L heel strike`
- Right stride: `R heel strike -> next R heel strike`

그리고 각 stride를 101포인트로 보간해 `0-100% gait cycle`로 정규화했다.

### 4. 왜 speed별로 따로 분석했는가

속도는 관절각 패턴에 매우 큰 영향을 주기 때문에,
`slow`, `normal`, `fast`를 한 모델에서 섞어버리면 그룹 효과가 속도 효과에 묻히거나 해석이 애매해질 수 있다.

그래서 1차 waveform 비교는 speed별로 따로 했다.

## 결과 파일별 의미

### [`subject_speed_waveforms.csv`](/Users/ryutt/Desktop/mini_ryutt/Walking/data/processed/waveform_based/subject_speed_waveforms.csv)

- 총 9,828행
- `subject × speed × side_basis × feature` 단위의 평균 파형
- 각 행에 `point_000`부터 `point_100`까지 101포인트가 들어 있음
  - subjec마다 3 speed 가 있고 각 speed마다 3번의 보행 집합이 있음, 보행 집합간의 데이터 간격이 있음. 현재 데이터를 어떻게 처리한 것인지 확인 필요

이 파일은 “분석의 원재료”에 해당한다.
실제 파형 그림을 그릴 때 가장 먼저 보는 파일이다.

### [`spm_results.csv`](/Users/ryutt/Desktop/mini_ryutt/Walking/data/processed/waveform_based/spm_results.csv)

- 총 346행
- primary omnibus, primary post-hoc, paired ACL 결과가 모두 포함됨
- 각 행은 유의 cluster 하나 또는 비유의 test 하나를 나타냄

핵심 컬럼은 아래와 같다.

- `test_type`: `omnibus`, `posthoc`, `paired_posthoc`
- `speed`, `side_basis`, `feature`
- `start_pct`, `end_pct`: 유의 구간
- `test_p_fdr`: FDR 보정 p-value
- `effect_size`: cluster mean 기준 효과크기

### [`feature_table.csv`](/Users/ryutt/Desktop/mini_ryutt/Walking/data/processed/waveform_based/feature_table.csv)

- 총 273행
- `subject × speed` 단위의 wide feature table
- peak max/min, ROM, peak timing, phase mean, SPM region mean, injured-contralateral difference, LSI가 들어 있음

이 파일은 LMM과 ranking의 입력 테이블이다.

### [`lmm_results.csv`](/Users/ryutt/Desktop/mini_ryutt/Walking/data/processed/waveform_based/lmm_results.csv)

- 총 504행
- primary 3그룹 분석과 paired ACL subset 분석이 함께 있음
- group main effect와 group × speed interaction에 대한 FDR 결과가 들어 있음

### [`feature_ranking.csv`](/Users/ryutt/Desktop/mini_ryutt/Walking/data/processed/waveform_based/feature_ranking.csv)

- 총 44행
- elastic-net 기준 상위 설명 feature 순위표

이 파일은 “가장 설명력이 큰 feature가 무엇인가”를 한눈에 보기 위한 테이블이다.

### [`validation_summary.csv`](/Users/ryutt/Desktop/mini_ryutt/Walking/data/processed/waveform_based/validation_summary.csv)

- 총 30행
- joint mapping, sign convention, speed-ROM sanity check, HA vs ACL sagittal ROM sanity check를 포함

### [`sensitivity_comparison.csv`](/Users/ryutt/Desktop/mini_ryutt/Walking/data/processed/waveform_based/sensitivity_comparison.csv)

- 총 54행
- full-trial과 mid-trial sensitivity 결과의 FDR 결정이 같은지 비교

## 1차 결과: SPM waveform 결과 해석

### 전체 요약

full-trial 기준 primary FDR 유의 결과는 총 63개 cluster였다.

- omnibus: 27개
- post-hoc: 36개

이 숫자는 “그룹 간 waveform 차이가 생각보다 여러 곳에서 관찰되었다”는 뜻이지만,
동시에 모든 관절이 골고루 중요했다는 뜻은 아니다.
유의 결과는 몇몇 feature에 집중되었다.

### 가장 자주 반복된 feature

유의 cluster 빈도 상위 feature는 아래와 같았다.

- `knee_flexion`
- `knee_adduction`
- `knee_int_rotation`
- `hip_adduction`
- `ankle_dorsiflexion`

여기서 가장 일관된 중심축은 `knee_flexion`과 `knee_adduction`이다.
즉, 이번 데이터셋에서는 그룹 차이가 무릎 sagittal과 frontal plane에서 가장 뚜렷하게 드러났다.

### omnibus에서 보인 패턴

가장 강한 omnibus 결과는 아래 패턴으로 요약할 수 있다.

1. `injured knee_flexion`
   특히 `fast`에서 3개 구간, `normal`과 `slow`에서도 유의 구간이 나타났다.
   이는 무릎 굴곡 파형이 속도 증가와 함께 그룹 차이를 강하게 드러낸다는 의미다.
2. `injured knee_adduction`
   `slow`, `normal`, `fast` 모두에서 유의 구간이 나타났다.
   특히 late stance나 후반부 구간에서 HA와 ACL 그룹의 차이가 컸다.
3. `contralateral hip_adduction`
   `slow`, `normal`, `fast` 모두 contralateral 측에서 반복적으로 등장했다.
   이건 손상측 자체뿐 아니라 반대측 보상 전략이 존재할 가능성을 시사한다.
4. `fast contralateral knee_int_rotation`
   contralateral 측 무릎 회전 패턴도 fast에서 유의했다.
5. `injured ankle_dorsiflexion`
   fast에서만 보조적으로 유의했으며, 무릎만큼 지배적이지는 않았다.

### group mean 방향성

omnibus 표에서 실제 mean을 보면, 예를 들어:

- `fast / injured / knee_flexion / 32-45%`
  `HA 6.28`, `ACLD 8.04`, `ACLR 10.70`
- `slow / injured / knee_adduction / 15-40%`
  `HA -1.19`, `ACLD -0.58`, `ACLR -0.32`

즉 단순히 “ACL 그룹이 항상 작다”는 형태가 아니라,
속도와 구간에 따라 HA보다 커지는 구간도 있고 작아지는 구간도 있다.
이 점이 waveform analysis가 필요한 이유다.
peak 하나만 보면 이런 방향성 반전을 놓치기 쉽다.

### post-hoc에서 보인 패턴

post-hoc 결과를 보면 가장 자주 반복적으로 유의한 비교는 `HA vs ACLR`였다.

예시:

- `slow / injured / knee_flexion / HA vs ACLR`
- `normal / injured / knee_adduction / HA vs ACLR`
- `fast / contralateral / knee_flexion / HA vs ACLR`
- `fast / injured / knee_int_rotation / HA vs ACLR`

반면 `HA vs ACLD`는 유의한 경우가 있었지만 상대적으로 제한적이었다.
이 결과는 적어도 이번 데이터셋에서는 ACLR 그룹이 HA와 더 멀리 떨어진 waveform 특성을 보일 가능성을 시사한다.

이 해석은 직관과 다를 수 있다.
보통 재건 후 ACLR가 healthy에 가까워질 것이라고 기대할 수 있지만,
실제 데이터에서는 재건 후에도 distinct한 movement strategy가 남아 있을 수 있다.

## paired ACLD vs ACLR 결과 해석

같은 사람에서 ACLD와 ACLR가 모두 있는 subset에서 paired SPM을 수행한 결과,
유의 결과는 총 15개 cluster였다.

반복적으로 나온 feature는 아래였다.

- `contralateral hip_flexion`
- `contralateral knee_adduction`
- `injured knee_flexion`
- `contralateral ankle_dorsiflexion`

이 paired 결과는 cross-sectional primary 결과와 중요한 연결점을 준다.

특히:

- `normal / contralateral / knee_adduction`
- `slow, normal, fast / contralateral or injured / knee_flexion`

에서 ACLD와 ACLR 사이 차이가 나타났다는 것은,
단순한 between-group 차이뿐 아니라 상태 변화에 따른 longitudinal 방향성도 일부 존재할 수 있음을 시사한다.

여기서 `mean_diff`는 `ACLR - ACLD` 방향으로 계산되어 있다.
따라서 양수는 ACLR가 ACLD보다 해당 cluster mean이 크다는 뜻이다.

## 2차 결과: LMM 해석

primary subset에서 group main effect 또는 interaction FDR 기준으로 살아남은 feature는 44개였다.

### 가장 강한 LMM 신호

가장 강한 feature들은 거의 모두 무릎에서 나왔다.

상위 예시는 아래와 같다.

- `knee_adduction_injured_spm_region_mean`
- `knee_flexion_injured_peak_min`
- `knee_flexion_injured_rom`
- `knee_flexion_injured_terminal_stance_mean`
- `knee_flexion_contralateral_rom`
- `knee_flexion_diff_rom`
- `knee_flexion_lsi_rom`

이 결과는 waveform 단계에서 본 결론과 일치한다.
즉, “가장 중요한 차이는 무릎에서 나오고, 그중에서도 굴곡과 내외반 관련 feature가 중심”이라는 것이다.

### interaction이 중요한 이유

일부 feature는 group main effect뿐 아니라 group × speed interaction도 강했다.
대표적으로:

- `knee_flexion_injured_rom`
- `knee_flexion_contralateral_rom`
- `ankle_dorsiflexion_injured_loading_response_mean`
- `ankle_dorsiflexion_diff_loading_response_mean`

이건 “그룹 차이가 모든 속도에서 일정한 크기로 유지되는 것이 아니라, 속도에 따라 더 커지거나 줄어든다”는 뜻이다.
즉, speed를 pooling하지 않고 분리해서 본 분석 설계가 타당했다는 근거가 된다.

### asymmetry가 절대값보다 중요한가

이번 결과에서는 absolute feature만 중요한 것이 아니었다.

상위 LMM feature 중에는:

- `knee_flexion_diff_rom`
- `knee_flexion_lsi_rom`
- `ankle_dorsiflexion_diff_terminal_stance_mean`
- `ankle_dorsiflexion_diff_loading_response_mean`

처럼 injured-contralateral difference나 LSI 기반 feature도 포함되었다.

즉, 절대적인 각도 크기뿐 아니라 좌우 비대칭도 그룹 차이를 설명하는 중요한 축이었다.

## 3차 결과: elastic-net ranking 해석

ranking 테이블에서 상위 feature는 아래와 같았다.

1. `knee_adduction_injured_spm_region_mean`
2. `knee_flexion_diff_terminal_stance_mean`
3. `hip_adduction_contralateral_spm_region_mean`
4. `knee_flexion_diff_rom`
5. `ankle_dorsiflexion_diff_loading_response_mean`
6. `knee_flexion_injured_peak_min`

이 순위는 매우 중요하다.

왜냐하면 이것이 단순히 “유의했다”가 아니라,
여러 후보 중 실제 그룹 구분에 가장 기여하는 feature가 무엇인지 보여주기 때문이다.

### ranking에서 읽어야 할 핵심

1. `knee-derived feature dominance`
   상위권이 무릎 feature에 집중되어 있다.
   이는 전체 파이프라인에서 일관되게 반복된 결론이다.
2. `difference / spm_region feature의 강세`
   단순 peak max보다, 특정 유의 구간 평균이나 side difference가 더 높은 순위를 차지했다.
   이것도 waveform-first 분석의 장점을 지지한다.
3. `moderate classification utility`
   best model의 성능은 대략:

   - `cv_macro_f1 = 0.598`
   - `cv_accuracy = 0.605`

이 값은 아주 높은 분류 성능은 아니다.
따라서 ranking 결과는 “진단 분류기”로 쓰기보다는
“설명력이 상대적으로 큰 biomechanical candidate”를 우선순위화한 결과로 해석하는 것이 적절하다.

## validation 결과 해석

### 통과한 항목

1. `joint mapping`
   18/18 통과
   raw `jointAngle_*`와 xlsx anatomical label이 일치했다.
   즉, 축 매핑은 올바르게 구현되었다.
2. `sign convention`
   hip flexion, knee flexion, ankle dorsiflexion의 primary peak가 모두 양수였다.
   따라서 sagittal sign direction도 큰 문제는 없다고 볼 수 있다.
3. `speed ROM monotonic`
   hip flexion과 ankle dorsiflexion은 `slow < normal < fast`를 만족했다.

### 실패한 항목

1. `knee_flexion`의 speed-ROM monotonic

   - observed: `slow 58.12 | normal 62.53 | fast 60.41`
   - 즉 fast가 normal보다 약간 작았다.
2. `HA >= ACLD/ACLR` 가정 일부 실패

   - `knee_flexion:fast`
   - `ankle_dorsiflexion:fast`

이 실패는 파이프라인 오류라기보다,
데이터가 단순한 “healthy가 항상 더 크다” 가정을 따르지 않는다는 뜻으로 보는 편이 맞다.
오히려 이것이 이번 분석에서 waveform 기반 접근이 필요한 이유를 다시 보여준다.

## sensitivity 결과 해석

full-trial과 mid-trial sensitivity를 비교했을 때,
54개 비교 중 FDR 결정이 달랐던 경우는 1개뿐이었다.

불일치한 항목은 아래였다.

- `normal / contralateral / knee_flexion`
  - full: 비유의
  - midtrial: 유의

이 결과는 두 가지를 의미한다.

1. 전체적으로는 trimming 유무에 따라 결론이 크게 흔들리지 않았다.
2. 그러나 일부 borderline feature는 stride 선택 방식에 민감할 수 있다.

따라서 최종 결론은 full-trial 기준으로 두되,
`normal contralateral knee_flexion` 같은 경계 사례는 과도하게 강한 결론을 피하는 것이 좋다.

## 최종 해석

이번 결과를 한 문장으로 요약하면 다음과 같다.

`성인 HA, ACLD, ACLR 비교에서 그룹 차이를 가장 일관되게 설명한 요소는 무릎 굴곡과 무릎 내외반 관련 waveform 및 그 파생 feature였고, 일부 contralateral hip adduction과 ankle dorsiflexion feature가 이를 보조했다.`

조금 더 구체적으로 정리하면:

1. `무릎이 핵심`
   knee flexion과 knee adduction이 waveform, LMM, ranking 모두에서 중심 신호였다.
2. `특히 ACLR가 HA와 많이 달랐다`
   post-hoc에서 `HA vs ACLR` 유의 결과가 가장 자주 나타났다.
3. `보상 전략은 손상측만의 문제가 아니다`
   contralateral hip adduction과 contralateral knee-related feature도 중요했다.
   이는 양측적 적응이나 보상 전략을 시사한다.
4. `속도는 반드시 고려해야 한다`
   interaction 결과와 speed별 SPM 패턴을 보면 speed를 무시하면 해석이 흐려질 가능성이 높다.
5. `비대칭 정보가 중요하다`
   difference와 LSI feature가 상위권에 포함되어, 좌우 차이가 중요한 설명 변수임을 보여준다.

## 해석 시 주의점

### 1. SPM의 유의는 “구간의 차이”다

이 결과는 “peak 하나의 차이”가 아니라,
특정 gait-cycle 구간에서의 차이를 의미한다.
따라서 결과를 말할 때는 가능하면:

- 관절
- 평면
- 속도
- side basis
- gait cycle 구간

을 함께 언급하는 것이 좋다.

### 2. ranking은 예측기가 아니라 우선순위표다

elastic-net 성능이 아주 높지 않기 때문에,
ranking은 진단 분류기로 쓰기보다 biomechanical priority list로 보는 것이 맞다.

### 3. LSI 해석은 조심해야 한다

분모가 작은 경우 LSI가 크게 튈 수 있다.
따라서 LSI는 단독으로 해석하기보다 absolute feature나 diff feature와 함께 보는 것이 안전하다.

## 추천 보고 방식

논문형 결과 정리는 아래 순서를 권장한다.

1. `Primary waveform findings`
   speed별 SPM 결과에서 유의한 joint-plane과 구간 제시
2. `Post-hoc interpretation`
   어느 그룹쌍에서 차이가 주로 발생했는지 설명
3. `Feature-level interpretation`
   LMM과 ranking을 이용해 어떤 요약 feature가 핵심인지 정리
4. `Validation and sensitivity`
   mapping이 맞고, 결과가 대체로 안정적이지만 일부 fast 또는 borderline feature는 조심해서 해석한다고 명시

## 함께 보면 좋은 파일

- 결과 검토 노트북: [`waveform_group_results_review.ipynb`](/Users/ryutt/Desktop/mini_ryutt/Walking/notebooks/waveform_group_results_review.ipynb)
- 분석 스크립트: [`waveform_group_analysis.py`](/Users/ryutt/Desktop/mini_ryutt/Walking/scripts/analysis/waveform_group_analysis.py)
- 결과 폴더: [`waveform_based`](/Users/ryutt/Desktop/mini_ryutt/Walking/data/processed/waveform_based)
