
# Mahalanobis v3 — 다중 입력 정상참조 점수 및 재현 가능한 버전 관리

## 1. 최신 full 실행 해석

기준 실행은 [20260704-003445_full_s42 보고서](/Users/ryutt/Desktop/mini_ryutt/Walking/Mahalanobis/artifacts/20260704-003445_full_s42/report.html)이며 79 sessions, 52 biological identities, outer 5/inner 3-fold, profile당 20 trials를 사용했다.

- Primary `inverse_weight total_distance` AUC는 0.501(95% CI 0.354–0.661), `mean_aggregate`는 0.509(0.357–0.664)다.
- 최고 comparator인 `all_speed_concat`도 AUC 0.550(0.401–0.707)이고 MCD sensitivity는 0.506이다. 현재 waveform Mahalanobis 표현은 known-group validity를 지지하지 않는다.
- Inner AUC는 평균 0.55–0.59였지만 outer에서 chance로 하락했다. Fold별 scaling·PCA·shrinkage 선택도 달라 현재 표본에서 튜닝 안정성이 낮다.
- ACLR total distance가 ACLD보다 평균 0.295(`inverse_weight`) 또는 0.506(`mean_aggregate`) 높았다. ACLR가 HA 쪽으로 이동한 pair는 각각 9/27, 4/27뿐이므로 recovery 방향을 주장할 수 없다.
- HA12 제외 영향은 AUC −0.0004로 사실상 사라졌다. 현재 HA 한 명 제외에 따른 최대 AUC 변화도 약 ±0.02다.
- 두 balance mode 순위 상관은 Spearman 0.888이므로 weighting 방식만 바꾸는 것은 해결책이 아니다.
- 현재 total은 세 signed deviation을 제곱한 RMS라 음수 방향도 양의 크기로 변환한다. v3에서는 전체 54-block raw distance를 먼저 계산해 한 번만 보정한 `overall_z`를 primary로 사용하고 기존 RMS는 legacy sensitivity로 남긴다.
- 직접 contribution에서 knee flexion과 ankle dorsiflexion이 반복적으로 컸지만, 분류 성능이 chance이므로 ACL 병태 기여도가 아니라 해당 거리모델 내부의 deviation 분해로만 해석한다.
- 기존 scalar pair-aware supervised 결과는 RF 0.957, NCA 0.887, PLS-DA 0.861, PCA 0.844, LDA 0.805였다. 데이터에 분리 신호는 있으나 이것은 정상참조 점수의 성능이 아니라 supervised benchmark다.
- `0529_ML`의 AUC 0.960은 session-ID `StratifiedKFold` 결과라 longitudinal pair leakage 기준을 충족하지 않으며 최종 근거로 사용하지 않는다.

## 2. v3 연구 설계

### Primary 개발 방식

`HA-referenced, label-guided development score`로 개발한다.

1. Outer validation identity는 완전히 잠근다.
2. Outer-training 내부에서 각 후보 profile의 HA reference를 만든다.
3. Inner pair-aware CV의 ACL-vs-HA AUC로 입력 profile과 파라미터를 선택한다.
4. 선택된 설정을 outer-training HA에 재적합하고 outer-validation을 한 번만 평가한다.
5. 현재 cohort 결과는 development evidence로 명시하며, 최종 공식을 동결한 후 독립 코호트에서 외부 검증한다.

점수는 HA까지의 거리지만 개발에 ACL label을 사용하므로 외부 검증 전에는 “완전히 비지도인 정상성 바이오마커”로 표현하지 않는다. Supervised RF/NCA는 별도 benchmark로 유지하며 정상성·회복 점수로 해석하지 않는다.

### Endpoint

- Primary: 전체 `3 speeds × 2 sides × 9 joints` GVS를 하나의 raw GPS distance로 통합한 뒤 HA LOO log calibration을 적용한 `overall_z_deviation`.
- 표시 점수: `normality_score = 100 - 10 × overall_z_deviation`.
- Secondary: slow·normal·fast, hip·knee·ankle, bilateral, asymmetry subscores.
- Legacy sensitivity: 기존 `RMS(speed_signed_z)`, MCD distance, all-speed PCA distance.
- Paired ACLD→ACLR은 별도 longitudinal endpoint로 보고하고 수술 후 개선 방향을 사전 가정하지 않는다.
- `summary_metrics.csv`는 AUC와 paired mean을 같은 `identity_auc` 열에 넣지 않고 `estimate_type`, `estimate`, `ci_low`, `ci_high`로 분리한다.

## 3. 입력 Profile Registry

모든 profile은 동일한 biological-identity split에서 비교한다.

| Profile                     | 단위 및 차원                              | 역할                       |
| --------------------------- | ----------------------------------------- | -------------------------- |
| `cycle909_inverse`        | cycle별 9×101, inverse-count weight      | 현재 v2 primary 재현       |
| `condition909_mean`       | session×speed×side 평균 9×101          | 현재 mean comparator       |
| `gvs54_mlbased`           | 3×2×9 HA waveform RMS deviation         | v3 primary 후보            |
| `waveform5454`            | 3×2×9×101 session waveform             | 고차원 sensitivity         |
| `scalar144_legacy`        | 속도별 양측 90 + LSI 45 + asymmetry 9     | 기존 scalar 실험 정확 재현 |
| `scalar864_legacy`        | scalar144의 3속도+2 delta+평균            | 과거 high-AUC 표현 재검증  |
| `scalar_clean_multispeed` | strict bilateral scalar와 안정적 symmetry | 출판용 scalar primary 후보 |
| `fusion_gvs_scalar_clean` | GVS54 + cleaned scalar                    | label-guided fusion 후보   |

Scalar의 5개 metric은 `peak`, `min`, `ROM`, `IC_angle`, `peak_timing`으로 고정한다.

- Legacy profile은 기존 LSI 정의를 그대로 보존하되 legacy임을 명시한다.
- Clean profile에서는 ROM·peak timing처럼 비음수인 metric만 denominator threshold를 통과할 때 ratio를 사용한다.
- Signed peak/min/IC angle은 LSI 대신 `(injured−contralateral)/(abs(injured)+abs(contralateral)+ε)` 형태의 bounded symmetry를 사용한다.
- Injured-side 기본 `Right`, 작은 분모 ratio, 누락 ID·속도·trial의 silent fallback을 모두 금지한다.
- Feature 생성, imputation, scaling, selection과 fusion은 inner-training fold 안에서만 fit한다.
- `slim_gait.parquet`과 derived inputs의 hash·schema·generator version을 manifest에 저장한다.

Public CLI는 다음 형태로 통합한다.

```bash
python run_pipeline.py --list-profiles

python run_pipeline.py \
  --pipeline-version v3 \
  --input-profile gvs54_mlbased \
  --score-profile label_guided_normative \
  --mode full

python run_pipeline.py \
  --pipeline-version v3 \
  --input-profile all \
  --score-profile label_guided_normative \
  --select-profile-in-inner-cv
```

## 4. 버전 전환과 Rollback

### Git 운용

- v3 개발 branch: `mahalanobis/v3-input-registry`
- 현재 확정본 `7fbecb9`에는 annotated tag `mahalanobis-v2.0.0`을 부여한다.
- 입력 profile 변경은 branch를 새로 만들지 않고 config로 전환한다.
- scoring 알고리즘·artifact schema처럼 코드 계약이 바뀔 때만 새 minor/major branch와 tag를 만든다.

### Registry 구조

```text
Mahalanobis/
├── configs/
│   ├── inputs/
│   ├── scores/
│   └── cv/
├── versions/registry.yaml
├── pipelines/
│   └── v3/
└── artifacts/{pipeline_version}/{input_profile}/{run_id}/
```

`registry.yaml`에는 version, git tag/commit, entrypoint, 호환 input/score profile, artifact schema version과 deprecated 여부를 기록한다.

### Rollback 방식

현재 작업 폴더를 checkout으로 뒤집지 않고 별도 worktree를 만든다.

```bash
git worktree add \
  ../Walking-worktrees/mahalanobis-v2.0.0 \
  mahalanobis-v2.0.0
```

- 동일 코드 버전 내 재현: `run_pipeline.py --from-run <artifact>/manifest.json`
- 과거 코드까지 정확히 재현: version manager가 해당 tag의 worktree를 생성하고 그 폴더에서 manifest를 실행
- 기존 artifact는 immutable로 유지하고 새 실행은 항상 새 run ID를 사용
- Resume 시 code/data/config/schema hash가 하나라도 다르면 거부
- Destructive `git reset`이나 현재 작업 폴더의 강제 checkout은 사용하지 않는다.

## 5. 검증 및 AUC 0.8 Gate

### 내부 개발

- 52 biological identities 기준 repeated nested CV: outer 5-fold×5 repeats, inner 3-fold.
- ACLD/ACLR pair는 모든 split에서 동일 fold에 둔다.
- 후보 profile과 hyperparameter 선택은 inner AUC에서만 수행한다.
- 각 profile의 고정 OOF 결과와 inner-selected winner OOF를 모두 저장한다.
- Primary development gate:
  - identity-balanced nested OOF point AUC ≥0.80
  - seed/repeat별 AUC 변동과 95% identity-bootstrap CI 보고
  - covariance rank·condition, HA influence, bootstrap reliability가 사전 기준을 통과
- 여러 v3 결과를 본 뒤 다시 수정하면 outer 결과도 development feedback으로 간주한다. 최종 논문 검증은 독립 데이터가 필요하다.

### 외부 검증

최종 공식을 잠근 뒤 다른 시기·기관 또는 새 피험자 코호트에서 다음을 평가한다.

- 외부 ACL-vs-HA AUC ≥0.80
- test-retest ICC, SEM/MDC
- IKDC/KOOS, hop/strength test, return-to-sport, 수술 후 기간과의 construct validity
- 성별·연령·체격·속도·장비/site sensitivity
- ACLD→ACLR responsiveness와 임상 anchor 기반 방향성

### 테스트

- 모든 profile의 예상 shape·feature명·identity 수 검증
- 누락 side/ID/분모 오류가 audit을 남기고 실패하는지 검증
- Profile별 동일 outer/inner split hash 검증
- Outer-validation label이 feature 생성·selection·calibration에 접근하지 못하는 sentinel test
- GVS54를 `ML_based` scorer와 비교해 수치 parity 검증
- Legacy scalar144/864가 기존 artifact와 일치하는지 snapshot test
- Clean symmetry가 finite·bounded이고 극단 LSI를 생성하지 않는지 테스트
- `--from-run` 재현 결과와 원 artifact metrics/hash 일치 확인
- v2 tag worktree에서 최신 v2 full artifact를 재현하는 rollback smoke test

## 6. 저널 전략

현재 결과만으로 validated digital biomarker를 주장하기에는 부족하다. `npj Digital Medicine`은 validated digital biomarkers와 임상 적용을 다루지만 small-scale preliminary·purely observational 연구는 일반적으로 고려하지 않는다고 명시한다. 따라서 외부 cohort, 임상 anchor, test-retest 없이 바로 투고하는 것은 적합도가 낮다. [npj Digital Medicine aims and scope](https://www.nature.com/npjdigitalmed/aims)

- **Stretch target — npj Digital Medicine**: 외부 다기관 검증, clinical anchor, 재현 가능한 versioned software와 실제 디지털 바이오마커 활용 시나리오까지 확보한 경우.
- **Engineering stretch — IEEE Transactions on Biomedical Engineering**: GVS/scalar fusion, cluster-balanced covariance, multi-speed normative modeling 자체에 충분한 방법론적 신규성과 외부 검증이 있을 경우. [IEEE T-BME author information](https://ieeexplore.ieee.org/document/10535349)
- **강한 분야 저널 — Journal of NeuroEngineering and Rehabilitation**: rehabilitation measurement, wearable/engineering technology, precision rehabilitation과 연결하고 임상 endpoint를 추가한 경우 적합하다. [JNER field direction](https://link.springer.com/article/10.1186/s12984-025-01580-5)
- **현 데이터에 가장 현실적인 분야 저널 — Gait & Posture**: 다속도 ACL gait, 정상참조 방법 비교, waveform/scalar representation과 paired biomechanics를 중심으로 구성할 경우 scope가 직접 맞는다. [Gait &amp; Posture aims and scope](https://www.sciencedirect.com/journal/gait-and-posture)

Label-guided 개발과 classifier comparator는 [TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378)에 맞춰 보고하고, risk-of-bias는 [PROBAST+AI](https://www.bmj.com/content/388/bmj-2024-082505)로 사전 감사한다. Digital biomarker라는 용어는 측정값·임상적 의미·evidence chain을 분리해야 한다는 평가 원칙을 따른다. [Digital biomarker evidence framework](https://www.nature.com/articles/s41746-022-00583-z)

### 논문 중심 문장

> 다속도 관절각 waveform과 임상적으로 해석 가능한 scalar feature 표현을 비교하여, ACL injury/reconstruction에서 HA-referenced gait deviation score의 known-group validity, 안정성, 종단 반응성과 재현성을 평가했다.

AUC가 목표에 도달하지 않더라도 입력표현별 실패 원인과 supervised upper bound를 포함한 검증 연구로 남기며, 성능이 나온 profile만 사후적으로 선택한 것처럼 보고하지 않는다.
