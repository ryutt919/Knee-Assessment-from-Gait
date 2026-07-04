# Mahalanobis 파이프라인 — 의문점 정리 (Q&A)

각 항목은 `Mahalanobis/htmls/01_detailed_experiment_analysis.html`(감사 리포트)에서 발췌한 원문 의문점을 그대로 적고,
실제 코드(`Mahalanobis/scripts/01_data_preprocessing.py`, `02_mahalanobis_pipeline.py`)를 근거로 설명을 붙였습니다.

---

## Q1. P1-4. Identity-cluster 불확실성과 paired longitudinal 분석

> **현재 문제:** stride를 독립 표본처럼 계산한 AUC/검정은 표준오차를 과소평가합니다. 현재 cluster bootstrap도 estimand가 stride-weighted입니다. **변경:** primary estimand를 identity-balanced session score로 사전 지정하고 identity bootstrap으로 CI를 계산합니다. ACLD→ACLR 변화는 같은 환자 내 paired difference와 paired bootstrap/혼합모형으로 분석하며 HA 비교와 분리합니다. HA12 같은 극단값 제외 규칙, complete-IMU subset, 속도별 분석은 결과를 본 뒤 정하지 않고 sensitivity protocol에 미리 씁니다. **완료 기준:** n은 stride가 아니라 독립 identity로 보고되고 paired/unpaired 분석이 혼합되지 않아야 합니다.
>
> → 용어의 의미를 잘 모르겠음, 이해가 잘 안됨

### 설명

이 문단은 현재 코드가 **하지 않고 있는 일**을 나열한 개선 제안(action item)입니다. 하나씩 풀면:

- **stride를 독립 표본처럼 계산한 AUC**: `02_mahalanobis_pipeline.py:225`의 `roc_auc_score(y_val, dm_val)`는 stride 한 줄 한 줄을 서로 독립인 관측치처럼 취급합니다. 그런데 실제로는 한 사람의 stride 9,540개 중 다수가 같은 사람에서 나온 반복측정이라, "표본 수"가 부풀려져서 AUC의 표준오차(불확실성 폭)가 실제보다 작게 계산됩니다.
- **cluster bootstrap이 stride-weighted estimand**: 지금 방식으로 신뢰구간을 재표본추출(bootstrap)해도, 여전히 "stride 1개 = 1표"로 세기 때문에 stride를 많이 남긴 사람(보행이 길거나 trial이 많은 사람)이 결과를 더 많이 좌우합니다. **estimand**는 통계학 용어로 "우리가 실제로 추정하려는 대상(모수)"를 뜻합니다 — 여기서는 "stride 단위 평균"을 추정하고 있다는 말입니다.
- **identity-balanced session score**: 각 사람(정확히는 아래 Q7의 biological identity)을 1표로 세도록, 사람마다 session 단위 요약 점수(예: 그 사람 stride 거리의 평균) 하나만 만들어서 비교하자는 제안입니다.
- **identity bootstrap**: stride를 뽑는 대신 **사람(identity) 단위로** 복원추출해서 신뢰구간(CI)을 계산하는 방법입니다. 이렇게 하면 한 사람의 반복측정이 여러 번 중복으로 뽑히더라도 "몇 명이 뽑혔는가"는 원래 인원수와 같게 유지되어 불확실성이 과소평가되지 않습니다.
- **ACLD→ACLR paired difference / paired bootstrap / 혼합모형**: 같은 환자가 수술 전(ACLD)과 수술 후(ACLR)에 각각 측정된 경우, 이 둘은 "같은 사람의 전/후 비교"이므로 서로 다른 두 사람을 비교하듯 취급하면 안 됩니다. 전/후 차이만 뽑아 그 차이에 대해 bootstrap을 하거나(paired bootstrap), 혼합모형으로 사람을 random effect로 넣어 분석해야 합니다. 이는 "ACL 그룹 vs HA 그룹" 비교와는 완전히 다른 질문이므로 분리해서 보고해야 합니다.
- **HA12 제외/complete-IMU subset/속도별 분석을 사전에 정한다(sensitivity protocol)**: 실제로 리포트 Q1(HA12가 HA 평균 거리를 지배)에서 나온 것처럼, 특정 이상치를 빼거나 부분집합만 쓰는 판단을 결과를 보고 나서 정하면 "원하는 결론이 나올 때까지 조건을 바꾸는" 사후 편향이 생깁니다. 그래서 "이런 민감도 분석을 하겠다"는 규칙을 결과를 보기 **전에** 문서화해 두라는 것입니다.
- **완료 기준 (n은 identity 기준)**: 논문/리포트에 "n=9,540 stride"가 아니라 "n=66 independent identity"로 표본 크기를 적어야 하고, paired 분석(전/후 비교)과 unpaired 분석(그룹 간 비교)의 결과를 한 표에 섞지 말아야 한다는 뜻입니다.

**현재 코드 상태**: `02_mahalanobis_pipeline.py`에는 identity bootstrap, paired 분석, 사전 sensitivity protocol이 전혀 구현되어 있지 않습니다. 이 항목은 "구현된 로직 설명"이 아니라 "앞으로 고쳐야 할 통계 설계 제안"입니다.

---

## Q2. mixed-effects model

> stride는 유지하되 identity와 trial을 random effect로 포함합니다. stride·trial·사람 변이를 구분하려는 연구에 적합합니다.
>
> → 자세한 설명

### 설명

Q1의 "identity-balanced session score"는 사람 단위로 **먼저 평균 내서** 정보를 줄이는 접근입니다. 반면 **mixed-effects model(혼합효과모형)**은 stride 데이터를 하나도 버리지 않고 그대로 다 쓰되, 통계 모형 안에서 "이 변동이 어디서 왔는지"를 구조적으로 나눕니다.

- **fixed effect**: 그룹(HA/ACLD/ACLR), 속도(slow/normal/fast) 등 "우리가 관심 있는, 모두에게 같은 방식으로 적용되는" 효과.
- **random effect**: `(1 | identity)`, `(1 | identity:trial)` 처럼 "같은 사람 안에서, 같은 trial 안에서 값들이 서로 닮아있다(상관되어 있다)"는 사실을 모형에 반영하는 항. 사람마다 고유한 baseline(기저 수준)과 trial마다의 미세한 편차를 각각 별도의 분산 성분으로 추정합니다.

이렇게 하면 전체 변동을 `Var(사람 간) + Var(같은 사람의 trial 간) + Var(같은 trial의 stride 간)`으로 분해할 수 있어서, "그룹 차이가 사람 수준에서 나는 차이인지, 아니면 개별 stride의 잡음인지"를 구분할 수 있습니다. Q1의 identity bootstrap은 계산이 더 간단하지만 정보 손실이 있고, mixed-effects model은 stride 단위 정보를 다 활용하면서도 올바른 표준오차를 준다는 장점이 있습니다. 다만 구현이 더 복잡하고(예: `statsmodels.MixedLM`, `lme4`), 마할라노비스 거리처럼 이미 여러 변환(PCA/MCD)을 거친 값에 대해 정규성/등분산 가정이 얼마나 맞는지 별도로 검토해야 합니다.

**현재 코드 상태**: 이 프로젝트에는 mixed-effects model이 구현되어 있지 않습니다. `02_mahalanobis_pipeline.py`는 GroupKFold + 단순 그룹별 평균/표준편차(`group_stats`, `save_report`)만 계산합니다. Q1과 마찬가지로 향후 통계 검증 단계에서 도입을 검토해야 할 항목입니다.

---

## Q3. fold별 HA 기준화

> **fold별 HA 기준화:** validation subject를 제외한 training fold 중 HA stride만으로 각 특징의 scaler를 적합합니다. 변환 뒤 남은 NaN은 0으로 바뀌며, z-score 설정에서는 해당 특징의 HA 평균값으로 대체한 것과 같습니다.
>
> → 각 fold에서 HA 통계값으로 z-score 한다는 뜻?

### 설명

네, 맞습니다. 정확히는 `02_mahalanobis_pipeline.py:147-169`에서:

```python
for fold_i, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups_arr)):
    X_train_all, X_val = X[train_idx], X[val_idx]
    ...
    ha_mask = y_train_all == 0          # 이 fold의 training 쪽에서 HA(정상군)만
    X_ha = X_train_all[ha_mask]

    scaler = ScalerCls()                # 기본값은 StandardScaler (zscore)
    scaler.fit(X_ha)                    # ★ HA train stride로만 평균/표준편차 계산
    X_ha_scaled  = scaler.transform(X_ha)
    X_val_scaled = scaler.transform(X_val)   # ACLD/ACLR/HA validation 전부 같은 scaler 적용
```

즉 5-fold GroupKFold(피험자 단위 분할)에서 **매 fold마다** validation으로 빠진 사람을 제외한 나머지 training fold 안의 **HA(정상군) stride만** 골라서 `StandardScaler`를 적합합니다. `zscore` 설정에서 StandardScaler는 `(x - mean) / std`를 계산하므로, 이 mean/std는 **ACL 그룹이 전혀 섞이지 않은, 그 fold의 HA 데이터만의 평균과 표준편차**입니다. 이렇게 만든 scaler를 validation set(HA+ACLD+ACLR 전부)에 그대로 적용하므로, 모든 stride는 "그 fold의 HA 기준으로 몇 표준편차만큼 떨어져 있는가"로 변환됩니다.

**NaN 처리 부분**도 같은 맥락입니다 (`169`행 `np.nan_to_num(X_val_scaled, nan=0.0)`). z-score 변환 후 값이 0이라는 것은 `(x - mean)/std = 0`, 즉 `x = mean`이라는 뜻이므로, 원래 값이 결측이라 변환값이 NaN이었던 지점을 0으로 채우는 것은 결과적으로 **"그 특징 값을 HA 평균값으로 대체한 것"과 수학적으로 동일**합니다. HA 기준 대비 편차가 전혀 없다고 가정하고 채우는 셈입니다.

---

## Q4. stride 생성 — heel-strike edge 구간

> **stride 생성:** `subject_id × group × speed × file_name`을 trial로 묶고 좌우 heel-contact의 0→1 상승 edge 사이를 stride로 자릅니다. 40 sample 미만 구간은 버리고 각 trial·leg의 처음/마지막 2 stride를 제거합니다.
>
> → 0→1 순간과 1→0순간까지?

### 설명

아닙니다. **0→1 상승 edge와 1→0 하강 edge 사이가 아니라, 연속된 두 개의 0→1 상승 edge 사이**를 하나의 stride로 자릅니다. 실제 코드(`01_data_preprocessing.py:85-94`):

```python
def detect_heel_strikes(signal, min_samples=40):
    binary = ...를 0/1 정수로 변환
    edges = np.where(np.diff(binary) == 1)[0] + 1   # 0→1로 바뀌는 시점만 찾음
    segments = []
    for i in range(len(edges) - 1):
        s, e = edges[i], edges[i+1]                  # 이번 heel-strike ~ 다음 heel-strike
        if e - s >= min_samples:                      # 40 샘플 미만이면 버림
            segments.append((s, e))
```

`footContacts_0`(Left) / `footContacts_2`(Right)는 발이 땅에 닿아있는 동안 1, 떠 있는 동안 0인 이진 신호입니다. `np.diff(binary) == 1`은 0→1로 바뀌는 지점(=발이 땅에 닿기 시작하는 순간, heel strike)만 골라냅니다. 즉 1→0(발이 땅에서 떨어지는 순간, toe-off)은 stride 경계로 쓰이지 않습니다.

한 stride는 **"이번 heel strike부터 (같은 발의) 다음 heel strike 직전까지"** — 즉 gait cycle 정의상 표준적인 heel-strike to heel-strike 한 주기입니다(입각기 stance + 유각기 swing을 모두 포함). 이 구간 길이가 40 샘플보다 짧으면 노이즈로 보고 버리고, 한 trial·한쪽 다리에서 뽑힌 stride 중 처음/마지막 2개는 가속·감속 구간이라 제거합니다(`STRIDE_TRIM = 2`).

---

## Q5. PCA와 정상 분포 (normative distribution)

> **PCA와 정상 분포:** 같은 HA training stride로 PCA를 적합하고 선택된 PC 공간에서 MCD가 robust center `μ`와 covariance `Σ`를 추정합니다. ACL label은 이 정상 기준 적합에 사용되지 않습니다.
>
> → 용어가 어색한데 영어 용어는 영어로 하고 더 자세히 설명 필요

### 설명

`02_mahalanobis_pipeline.py:171-203`을 단계별로 풀면:

1. **PCA (Principal Component Analysis, 주성분분석)** — 앞서 Q3에서 fold별 HA 기준으로 z-score 정규화된 `X_ha_scaled`(HA train stride, 7,979차원 waveform)에 대해서만 `PCA()`를 적합합니다(`171-173`행). PCA는 7,979개 원본 feature(채널×101 포인트로 서로 강하게 상관된 waveform 값들)를 서로 독립인(orthogonal) 축인 **PC(주성분)** 몇 개로 압축합니다.
2. **PC 개수 선택 (`select_pca_k`, `56-75`행)** — 기본 방식은 **Kaiser criterion**: eigenvalue(고유값, 그 PC가 설명하는 분산의 크기)가 1 이상인 PC만 채택합니다. 다만 MCD가 안정적으로 동작하려면 표본 수(HA stride 수)가 차원 수보다 충분히 많아야 하므로, `hard_max = min(n_ha // 5, 100)`로 PC 개수 상한을 걸어둡니다(예: HA stride가 500개면 최대 100개 PC).
3. **PC score로 투영** — 선택된 k개 PC로 다시 `PCA(n_components=k)`를 적합하고, HA train과 validation(HA+ACL) stride 모두를 이 k차원 PC 공간으로 투영합니다(`X_ha_pc`, `X_val_pc`).
4. **MCD (Minimum Covariance Determinant, 최소공분산행렬식)** — `sklearn.covariance.MinCovDet`을 HA train의 PC score(`X_ha_pc`)에만 적합합니다(`190-192`행). 일반 공분산(covariance) 추정은 이상치(outlier) 하나에도 크게 흔들리는데, MCD는 전체 표본 중 `support_fraction`(기본 0.75, 즉 75%) 비율의 "가장 덜 흩어져 있는" 부분집합만 골라 그 부분집합으로 평균과 공분산을 계산합니다. 그 결과가 바로 **robust(이상치에 강건한) center `μ`(`mcd.location_`)와 covariance `Σ`(`mcd.covariance_`)** 입니다.
5. **"ACL label은 이 적합에 쓰이지 않는다"** — 1~4번 전 과정이 오직 `y_train_all == 0`(HA)인 stride로만 이루어집니다(`153-154`행 `ha_mask`). 즉 "정상 보행이 어떤 분포를 이루는가"라는 기준(정상 분포, normative distribution)을 ACLD/ACLR 데이터를 전혀 보지 않고 HA만으로 만듭니다. 이렇게 만든 기준(μ, Σ)에 대해서만 이후 HA/ACLD/ACLR validation stride의 마할라노비스 거리를 재기 때문에, "ACL 환자가 정상 분포에서 얼마나 벗어나 있는가"를 측정하는 구조가 됩니다. 그래서 이 방식은 지도학습 분류기가 아니라 **HA-referenced anomaly/deviation score**입니다.

---

## Q6. 마할라노비스 거리 전제조건 검사 여부

> - 마할라노비스 전제조건을 만족하는지 검사하는가?
>   1. 공분산 행렬이 안정적으로 추정되어야 한다.
>   2. feature 간 관계가 대체로 선형이라고 가정한다.
>   3. 정규분포에 가까울수록 해석이 안정적이다.
>   4. 공분산 행렬이 invertible해야 한다.

### 설명 (실제 코드 대비 검증)

결론부터 말하면, **현재 코드는 이 4가지 전제조건을 명시적으로 검사하지 않습니다.** 대신 일부는 우회(mitigate)하는 장치가 있고, 일부는 전혀 다루지 않습니다.

1. **공분산 안정성** — 직접 검사(예: 조건수 condition number 출력, eigenvalue 분포 확인)는 없습니다. 대신 `select_pca_k`의 `hard_max = min(n_ha // 5, 100)`(`63-65`행)이 "표본 수가 차원 수의 5배 이상이 되도록" PC 개수를 강제로 제한하는 **간접적** 안정화 장치입니다. 정말 표본이 부족하면 MCD가 예외를 던질 수 있고, 이 경우 `EmpiricalCovariance`(비강건 일반 공분산)로 조용히 폴백합니다(`193-202`행) — 이 폴백이 발생했는지 여부가 로그에만 남고 리포트 지표로 집계되지는 않습니다.
2. **선형 관계 가정** — 검사하지 않습니다. PCA 자체가 선형 변환이고 마할라노비스 거리도 이차형식(quadratic form)이므로, waveform 채널 간 관계가 비선형이면 이 방식이 그 구조를 놓칠 수 있다는 점은 리포트에서도 지적된 한계이지 코드가 검증하는 대상이 아닙니다.
3. **정규분포 가정** — Shapiro-Wilk 검정이나 Q-Q plot 같은 정규성 검사는 코드에 없습니다. MCD를 쓰는 이유가 바로 "표본이 완전히 다변량정규(multivariate normal)가 아니어도, 이상치 영향을 줄여 중심/공분산 추정을 어느 정도 강건하게 만들기 위함"이지만, 이것이 정규성을 확인하거나 강제하는 것은 아닙니다. 즉 MCD는 정규성 가정의 **위반을 완화**할 뿐, 가정이 성립하는지 **검증**하지는 않습니다.
4. **invertibility(역행렬 존재)** — `compute_mahalanobis`(`78-83`행)에서 `np.linalg.inv` 대신 `np.linalg.pinv`(pseudo-inverse, 유사역행렬)를 사용합니다. `pinv`는 행렬이 특이(singular)하거나 역행렬이 존재하지 않아도 항상 결과를 반환하므로 코드가 죽지는 않지만, 이는 "역행렬이 존재하는지 확인 후 처리"가 아니라 **애초에 존재 여부와 무관하게 근사값을 계산하는 회피책**입니다. 공분산이 거의 특이(near-singular)한 상태에서 pinv를 쓰면 거리값이 왜곡될 수 있는데 이를 경고하거나 로그로 남기는 코드는 없습니다.

요약하면, 이 파이프라인은 "전제조건을 통계적으로 검증"하기보다 **"위반 가능성을 낮추는 공학적 장치(PC 차원 제한, MCD, pinv)"**만 갖추고 있습니다. Q1/Q6이 지적하는 것처럼, 실제 감사 리포트(`htmls/01_detailed_experiment_analysis.html`)도 이 지점을 한계로 명시하고 있습니다.

---

## Q7. stride-leg rows가 뭐지?

### 설명

**stride-leg row**란 데이터프레임의 한 행(row)이 의미하는 관측 단위를 말합니다: **"한 trial(파일) 안에서, 한쪽 다리(Left 또는 Right)의, 하나의 stride(연속된 두 heel-strike 사이 구간)"**가 정확히 한 행입니다.

`01_data_preprocessing.py:150-206`의 반복문 구조를 보면:

```python
for (subject_id, group, speed, file_name), trial_df in df.groupby(group_cols):   # trial 단위
    for actual_leg, heel_col in HEEL_COLS.items():                               # Left / Right 각각
        segments = detect_heel_strikes(...)                                      # 이 다리의 stride들
        for local_idx, (start, end) in enumerate(trimmed):                       # stride 하나마다
            row = {..., "actual_leg": actual_leg, "stride_idx": local_idx, ...}
            records.append(row)                                                   # ← 이게 1개의 stride-leg row
```

즉 한 행의 식별자는 `(subject_id, group, speed, trial_id, actual_leg, stride_idx)` 조합입니다. 같은 trial이라도 왼발/오른발이 서로 다른 행으로 나뉘고(그래서 "-leg"), 같은 다리 안에서도 stride마다 별도 행이 됩니다(그래서 "stride-"). 결과물 `mahalanobis_features.parquet`의 9,540개 행이 바로 이 단위입니다.

이 용어가 강조하는 핵심은 Q1/Q7이 서로 연결된다는 점입니다: **9,540 stride-leg row ≠ 9,540명의 서로 다른 사람.** 실제로는 92개 `subject_id`("session형 ID", 예: `ACLD8`, `ACLR8`처럼 그룹-환자번호 조합)에서 반복측정된 stride들이고, `subject_id` 문자열의 숫자 접미사를 기준으로 같은 환자의 수술 전/후 세션을 하나로 묶으면 66개의 **biological identity**(생물학적 개체 수)만 남습니다(26쌍의 ACLD/ACLR 종단쌍이 92→66으로 줄어드는 이유). GroupKFold는 `subject_id`(=92개 session) 단위로만 fold를 나누기 때문에, 같은 환자의 ACLD8과 ACLR8이 서로 다른 fold(train/validation)에 배정될 수 있고, 이것이 Q1에서 지적한 "biological identity 독립성이 깨진다"는 문제의 근거입니다.

---

## 참고 코드 위치

| 항목 | 파일:라인 |
|---|---|
| stride 분할 (heel-strike edge) | `scripts/01_data_preprocessing.py:85-94, 150-214` |
| injured/contralateral side 매핑 | `scripts/01_data_preprocessing.py:58-82` |
| fold별 HA-only scaler/PCA/MCD | `scripts/02_mahalanobis_pipeline.py:147-203` |
| PC 개수 선택 (Kaiser + hard_max) | `scripts/02_mahalanobis_pipeline.py:56-75` |
| 마할라노비스 거리 계산 (pinv) | `scripts/02_mahalanobis_pipeline.py:78-83` |
| 감사 리포트 원문 (P1-4, stride-leg row 등) | `htmls/01_detailed_experiment_analysis.html` |
