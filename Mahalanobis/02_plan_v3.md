# Mahalanobis v3.1 — Mean-only Scalar Primary 정상참조 점수

## Revision History

### v3 → v3.1 Feedback

- **기존 v3 요약**: `inverse_weight`와 `mean_aggregate`를 함께 실행하고, cycle909·GVS54·waveform5454·legacy scalar·clean scalar·fusion 중 inner CV가 입력 profile을 선택하도록 계획했다. Primary endpoint는 세 속도의 GVS를 합친 GPS 계열 거리였다.
- **사용자 피드백**: 입력을 subject/session-level mean으로 고정하고, 출판용 `scalar_clean_multispeed`를 primary로 사전 지정한다. GVS54와 waveform5454는 scalar에 더했을 때의 추가 가치만 비교한다.
- **v3.1 변경**: `mean_aggregate`만 허용하고 primary 사후 교체를 금지한다. ACL은 실제 injured–contralateral로 정렬하고, HA는 Right=pseudo-injured convention과 Left/Right swap sensitivity를 함께 보고한다.
- **통계 보완**: AUC는 identity-balanced known-group validity로 제한해 해석한다. Hedges' g, ACLD→ACLR paired change, HA calibration, pseudo-side sensitivity를 분리된 estimate type으로 저장한다.

## 1. v2 Full 실행 해석과 v3.1 필요성

기준 실행은 [20260704-003445_full_s42 보고서](/Users/ryutt/Desktop/mini_ryutt/Walking/Mahalanobis/artifacts/20260704-003445_full_s42/report.html)이며 79 sessions, 52 biological identities, outer 5/inner 3-fold, profile당 20 Optuna trials를 사용했다.

- `inverse_weight total_distance` AUC는 0.501(95% CI 0.354–0.661), `mean_aggregate`는 0.509(0.357–0.664)였다.
- 최고 comparator `all_speed_concat`도 AUC 0.550(0.401–0.707), MCD sensitivity는 0.506으로 기존 waveform Mahalanobis 표현은 known-group validity를 지지하지 않았다.
- Inner AUC는 평균 0.55–0.59였지만 outer에서 chance로 하락했고, fold별 scaling·PCA·covariance 선택도 불안정했다.
- ACLR distance는 ACLD보다 평균 0.295(`inverse_weight`) 또는 0.506(`mean_aggregate`) 높았다. HA 쪽으로 이동한 pair가 각각 9/27, 4/27뿐이므로 recovery를 주장할 수 없다.
- HA12 제외 AUC 변화는 −0.0004, 모든 HA leave-one-out 최대 변화도 약 ±0.02로 과거 HA12 단일 이상치 문제는 해소됐다.
- 두 balance mode 순위 상관은 Spearman 0.888이므로 weighting 변경만으로는 성능이 개선되지 않았다.
- v2 total은 `RMS(slow_z, normal_z, fast_z)`라 음수 방향도 양의 크기로 바뀐다. 이 값은 v3.1에서 `legacy_speed_rms` sensitivity로만 유지한다.
- 기존 scalar pair-aware supervised 결과 RF 0.957, NCA 0.887, PLS-DA 0.861, PCA 0.844, LDA 0.805는 데이터의 분리 가능성을 보여주지만 HA-reference 정상성 점수의 타당성은 아니다.
- `0529_ML` AUC 0.960은 ACLD·ACLR biological pair를 잠그지 않은 과거 결과이므로 최종 근거로 사용하지 않는다.

## 2. 연구 질문과 사전 지정 Primary

### 연구 질문

> 해석 가능한 multi-speed bilateral scalar 정상참조 점수에 GVS54 또는 전체 관절각 waveform을 추가하면 ACL known-group validity와 종단 반응성이 개선되는가?

### 개발 방식

`HA-referenced, label-guided development score`로 개발한다.

1. Outer validation biological identity를 완전히 잠근다.
2. Outer-training HA만 사용해 reference, scaling, PCA, covariance와 calibration을 적합한다.
3. Inner pair-aware CV의 identity-balanced ACL-vs-HA AUC는 각 profile 내부 파라미터 선택에만 사용한다.
4. `scalar_clean_multispeed`를 출판 primary로 고정하며 comparator 성능을 본 뒤 교체하지 않는다.
5. 현재 cohort는 development evidence로 명시하고 최종 공식을 동결한 뒤 독립 cohort에서 외부 검증한다.

점수는 HA까지의 거리지만 개발 과정에 ACL label 기반 AUC를 사용하므로 외부 검증 전에는 “완전 비지도 바이오마커”나 “임상 회복 점수”로 표현하지 않는다.

## 3. 분석 단위와 Mean 집계

모든 v3.1 profile의 원자료와 집계 계보를 동일하게 고정한다.

```text
cycle waveform 101 points
→ cycle을 trial 안에서 평균
→ trial을 session × speed × side 안에서 동일가중 평균
→ slow·normal·fast와 injured·contralateral을 결합
→ 한 session당 한 feature row
```

- Full cohort는 ACLD 27, ACLR 27, HA 25의 79 session과 52 biological identities다.
- ACLD·ACLR은 종단 변화를 보존하기 위해 서로 다른 session row로 남기되 모든 inner/outer split에서 같은 identity fold에 둔다.
- Full identity class는 ACL 27, HA 25로 거의 균형이다.
- 평가에서 ACL identity 한 명의 ACLD·ACLR session weight 합은 1, HA 한 명의 weight도 1이다.
- `inverse_weight`는 v3.1에서 실행하지 않으며 v2 tag/worktree에서만 재현한다.

## 4. 입력 Profile Registry

| Profile | 구성 | 차원 | 역할 |
|---|---|---:|---|
| `scalar_clean_multispeed` | 3속도 bilateral scalar + stable symmetry | 405 | 고정 primary |
| `scalar_plus_gvs54` | scalar405 + GVS54 | 459 | 저차원 waveform comparator |
| `scalar_plus_waveform5454` | scalar405 + mean waveform5454 | 5,859 | 전체 waveform comparator |

### 4.1 `scalar_clean_multispeed`

Trial-balanced mean full-cycle waveform에서 다음 5개를 계산한다.

```text
peak          first prominent peak; 없으면 방향별 global extremum
min           full-cycle minimum
ROM           maximum - minimum
IC_angle      cycle 0% angle
peak_timing   peak 위치, 0–100% cycle
```

속도당 feature 구성:

```text
9 channels × 5 metrics × 2 sides = 90 bilateral
9 channels × 5 metrics = 45 symmetry
3 speeds × 135 = 405
```

모든 symmetry는 다음 bounded signed difference를 사용한다.

```text
(injured - contralateral) /
(abs(injured) + abs(contralateral) + ε)
```

- 값은 `[-1, 1]`을 벗어나면 실패한다.
- Signed angle에 단순 LSI를 적용해 분모 근처에서 값이 폭발하는 문제를 제거한다.
- 기존 stance 기반 `features_scalar.csv`를 직접 읽지 않고 `cycle_waveforms_101.parquet`에서 동일 mean 계보로 다시 계산한다.

### 4.2 `scalar_plus_gvs54`

각 inner/outer training HA mean waveform과 대상 waveform의 RMS 차이를 계산한다.

```text
GVS = sqrt(mean((subject waveform - training HA mean waveform)^2))
3 speeds × 2 sides × 9 channels = 54 GVS features
```

- Validation session은 HA reference 계산에 접근할 수 없다.
- 구현값은 `ML_based/recovery_score/components.py::gvs_matrix`와 수치 parity를 가져야 한다.
- GVS raw GPS는 comparator 설명용 subscore로 별도 저장한다.

### 4.3 `scalar_plus_waveform5454`

```text
3 speeds × 2 sides × 9 channels × 101 points = 5,454 waveform features
405 scalar + 5,454 waveform = 5,859
```

- 전체 waveform의 국소 시점 정보를 보존하지만 HA 표본 수보다 매우 고차원이므로 PCA와 shrinkage covariance를 fold 내부에서 적용한다.
- Scalar와 waveform block 각각을 training HA로 standard/robust scaling하고 `sqrt(active feature count)`로 나눠 block 총분산을 동등하게 만든다.

## 5. Side Contract

- ACLD·ACLR은 `data/ID.csv`의 실제 injured leg를 `injured`, 반대측을 `contralateral`로 매핑한다.
- ACLD·ACLR pair의 injured side가 다르거나 metadata가 누락되면 QC 기록 후 즉시 실패한다.
- HA는 사전 선택한 `Right=pseudo-injured`, `Left=pseudo-contralateral` convention을 사용한다.
- HA convention은 schema, manifest와 HTML에 명시한다.
- 모든 HA의 좌우 orientation만 뒤집어 primary pipeline을 다시 적합한 sensitivity를 저장한다.

## 6. 정상참조 모델과 Endpoint

각 block은 training HA에서만 scaling하고 zero-variance feature를 제거한다. Block normalization 후 PCA와 Ledoit–Wolf covariance를 적합한다.

```text
D = sqrt((x - μ_HA)' Σ⁻¹ (x - μ_HA))
```

Training HA leave-one-out distance를 robust log calibration한다.

```text
overall_z_deviation =
  (log(D + ε) - median(log(D_HA_LOO))) /
  (1.4826 × MAD(log(D_HA_LOO)))

normality_score = 100 - 10 × overall_z_deviation
```

- `overall_z < 0`: 전형적인 HA보다 HA 중심에 가까움.
- `overall_z = 0`: training HA LOO 중앙 수준.
- `overall_z > 0`: 전형적인 HA보다 멂.
- 모든 profile은 동일 endpoint schema를 사용한다.
- Feature contribution은 정확히 `D²`와 합이 일치해야 한다.
- Slow/normal/fast, joint, GVS raw GPS와 top feature contribution은 explanatory output이며 primary total에 사후 혼합하지 않는다.

## 7. Nested CV, 클래스 균형과 평가

### Split과 튜닝

- Full: outer 5-fold × 5 repeats, inner 3-fold.
- Split unit: 52 biological identities.
- 각 fold의 ACL:HA identity 비율을 stratification으로 유지한다.
- ACLD·ACLR pair fold 일치율은 100%, train-validation identity overlap은 0이어야 한다.
- Optuna objective는 identity-balanced ROC-AUC다.
- AUC는 known-group discrimination으로만 해석하며 임상 중증도 타당성으로 확장하지 않는다.

### Primary와 comparator

- Primary: scalar identity-balanced OOF AUC, 95% stratified identity-bootstrap CI, Hedges' g.
- Comparator: `ΔAUC = AUC(fusion) − AUC(scalar)`.
- ΔAUC CI는 같은 bootstrap identity draw를 두 profile에 적용하는 paired bootstrap으로 계산한다.
- ACLD→ACLR은 `ACLR overall_z − ACLD overall_z` paired change와 paired identity-bootstrap CI로 별도 보고한다.
- `summary_metrics.csv`는 `estimate_type`, `estimate`, `ci_low`, `ci_high`로 AUC·effect size·paired change를 구분한다.

### Development gate

- Full repeated nested OOF point AUC ≥0.80.
- Repeat별 AUC 변동, bootstrap CI, covariance rank·condition number와 HA influence를 함께 통과해야 한다.
- Dry run AUC는 기능 검증값일 뿐 성능 근거나 gate 판정에 사용하지 않는다.
- 반복적인 v3.1 수정에 사용된 outer 결과는 development feedback이며 최종 논문 검증은 독립 cohort가 필요하다.

## 8. CLI, Artifact와 Rollback

### CLI

```bash
python Mahalanobis/run_pipeline.py --list-profiles

python Mahalanobis/run_pipeline.py \
  --pipeline-version 3.1 \
  --input-profile scalar_clean_multispeed \
  --mode full

python Mahalanobis/run_pipeline.py \
  --pipeline-version 3.1 \
  --input-profile all \
  --mode full
```

- v3.1은 `mean_aggregate`만 허용하고 inverse/both 요청은 즉시 거부한다.
- `--input-profile all`은 세 profile이 동일 split hash를 공유하도록 실행한다.
- `--from-run <manifest.json>`은 저장된 실행 설정을 새 immutable run으로 재현한다.
- `--resume`은 code/data/config hash가 같은 완료 run만 읽기 전용으로 재사용한다.

### Artifact

```text
Mahalanobis/artifacts/v3.1/{input_profile}/{run_id}/
├── manifest.json
├── qc_audit.json
├── outer_splits.csv
├── scalar_feature_schema.csv
├── oof_session_scores.parquet
├── oof_session_scores_averaged.parquet
├── summary_metrics.csv
├── ha_side_swap_summary.csv
├── ha_side_swap_session_sensitivity.csv
├── profiles/{profile}/models/
├── profiles/{profile}/fold_diagnostics.csv
├── profiles/{profile}/top_feature_contributions.parquet
└── report.html
```

### Version과 rollback

- v2 baseline commit `7fbecb9`에 annotated tag `mahalanobis-v2.0.0`을 둔다.
- v3.1 branch는 `mahalanobis/v3.1-scalar-primary`다.
- `versions/registry.yaml`이 pipeline version, tag/branch, entrypoint, profile과 artifact schema를 관리한다.
- 과거 코드 실행은 현재 작업 폴더를 뒤집지 않고 별도 worktree를 사용한다.

```bash
git worktree add ../Walking-worktrees/mahalanobis-v2.0.0 mahalanobis-v2.0.0
```

## 9. Test and Acceptance Plan

- Profile shape가 scalar `405`, scalar+GVS `459`, scalar+waveform `5,859`인지 검증한다.
- Full cohort 79 sessions, 52 identities와 각 session의 세 속도·양측을 검증한다.
- 전체 cycle set 복제와 trial 순서 변경 후 mean representation이 동일해야 한다.
- Missing ID/side/speed/trial, pair side mismatch와 silent fallback을 금지한다.
- Stable symmetry는 finite이고 `[-1,1]` 범위여야 한다.
- GVS54는 기존 `ML_based` GVS와 수치적으로 일치해야 한다.
- Outer validation 변경이 training HA reference를 바꾸지 않는 sentinel test를 통과해야 한다.
- 모든 raw distance는 0 이상이고 `overall_z`는 음수를 보존해야 한다.
- 모든 contribution 합은 수치 허용오차 내에서 `D²`와 같아야 한다.
- Profile별 split hash가 동일하고 class identity 분포가 stratified되어야 한다.
- HA Right/Left swap sensitivity를 실제 산출해야 한다.
- `--resume`은 완료 artifact를 수정하지 않아야 한다.
- `--from-run` 재실행의 split, summary, averaged OOF와 side sensitivity hash가 일치해야 한다.
- v2 unit test와 CLI가 그대로 통과해야 한다.

## 10. 실제 극소량 End-to-End 검증

실제 `cycle_waveforms_101.parquet`에서 ACL 12, HA 12 identities를 사용했다. ACL은 ACLD·ACLR 두 session을 유지해 총 36 session이다.

```bash
.venv/bin/python Mahalanobis/run_pipeline.py \
  --pipeline-version 3.1 \
  --input-profile all \
  --mode dry \
  --outer-folds 2 \
  --inner-folds 2 \
  --cv-repeats 1 \
  --trials 1 \
  --bootstrap 50 \
  --dry-identities-per-class 12 \
  --seed 42 \
  --run-id v3.1-tiny-e2e-final-s42
```

검증 artifact:

```text
Mahalanobis/artifacts/v3.1/all/v3.1-tiny-e2e-final-s42/
```

- 세 profile 모두 OOF, model, diagnostics, contribution, summary, side-swap sensitivity와 HTML 생성 완료.
- `--resume` immutable reuse 완료.
- `--from-run` 재실행과 summary, split, averaged OOF, side-swap summary SHA-256 일치.
- 이 실행은 파이프라인 기능 검증이며 AUC 성능 결론에는 사용하지 않는다.

## 11. 외부 검증과 저널 전략

최종 공식을 잠근 뒤 독립 시기·기관 또는 신규 cohort에서 다음을 평가한다.

- 외부 ACL-vs-HA identity-balanced AUC와 95% CI.
- Test-retest ICC, SEM/MDC.
- IKDC/KOOS, hop/strength test, return-to-sport, 수술 후 기간과의 construct validity.
- ACLD→ACLR responsiveness와 clinical anchor 변화의 상관.
- 성별·연령·체격·속도·장비/site sensitivity.

저널 우선순위:

1. **Gait & Posture**: 현 데이터와 다속도 ACL gait representation 비교에 가장 직접적이다.
2. **Journal of NeuroEngineering and Rehabilitation**: 임상 endpoint와 rehabilitation measurement를 보강할 경우 적합하다.
3. **IEEE Transactions on Biomedical Engineering**: block-normalized normative fusion의 방법론적 신규성과 외부 검증이 필요하다.
4. **npj Digital Medicine**: 다기관 외부 검증, clinical anchor, test-retest와 실제 digital biomarker context of use가 확보된 이후 stretch target으로 둔다.

Label-guided 개발은 TRIPOD+AI에 맞춰 보고하고 risk of bias는 PROBAST+AI로 감사한다. AUC가 0.8 gate에 도달하지 않더라도 입력표현별 실패와 supervised upper bound를 투명하게 보고하며 성능이 나온 profile만 사후 선택한 것처럼 서술하지 않는다.
