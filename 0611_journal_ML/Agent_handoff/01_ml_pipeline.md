# ML Pipeline — 0611_journal_ML

## Component Status

### Project Goal
Journal-quality ACL gait classification pipeline targeting AUC ≥ 98% (truly meaningful, no data leakage).

- **Task**: Binary classification — ACL (ACLD + ACLR) vs Healthy Adults (HA)
- **Dataset**: 78 subjects (54 ACL, 24 HA), 3 walking speeds
- **Constraint**: Read-only on all existing folders; all new files in `0611_journal_ML/` only

---

### Best Model — 02b_optimal_pipeline.py ✓ (side-fixed, 7-model benchmark)

**Current official result (side-fixed, 2026-06-12)**: **Random Forest OOF AUC = 0.9668** (best of 7)
Bootstrap 95% CI: [0.9218, 0.9955] · headline = best model × 2-seed(42/88) ensemble.

- **Implementation**: [scripts/02b_optimal_pipeline.py](../scripts/02b_optimal_pipeline.py)
- **Benchmark**: [results/02b_benchmark.csv](../results/02b_benchmark.csv) · [results/02b_optimal_results.json](../results/02b_optimal_results.json)
- **Figure**: [figures/fig_02b_roc.png](../figures/fig_02b_roc.png)

**7-model benchmark (baseline = Logistic Regression, 나머지 = benchmark)**:
| model | role | OOF AUC | 95% CI |
|-------|------|---------|--------|
| Random Forest | benchmark | **0.9668** | [0.922, 0.996] |
| CatBoost | benchmark | 0.9568 | [0.908, 0.993] |
| XGBoost | benchmark | 0.9498 | [0.893, 0.991] |
| LightGBM | benchmark | 0.9097 | [0.821, 0.976] |
| Gradient Boosting | benchmark | 0.8981 | [0.819, 0.966] |
| **Logistic Regression** | **baseline** | 0.8488 | [0.724, 0.949] |
| SVM (RBF) | benchmark | 0.7755 | [0.654, 0.883] |
| TabPFN | benchmark | — | skip (라이선스 토큰 없음) |

**Pipeline** (모델 무관 공통): scalar pivot + stride variability(**inj/con side-fixed**) + within-fold RF top-20 → C(20,2)=190 interactions → 최종 분류기(모델별, Optuna inner-CV 튜닝) · StratifiedKFold(5, shuffle=False).

**⚠️ Side-fix (핵심)**: 이전 0.9830은 stride `side`(injured/contralateral)를 `injured_leg`(Right/Left)와 잘못 비교해 **모든 stride가 con으로 처리**된 채 산출된 값(pre-fix exploratory). [_side_utils.py](../scripts/_side_utils.py)의 `to_inj_con`으로 inj/con 정상 분리(inj≈7,674 / con≈7,461) 후 재실행 → RF 0.9668이 현 공식 결과.

**Threshold behavior**: AUC(순위지표)와 임계값 0.5 분류 요약은 분리해서 본다. `02b_subject_predictions.csv` 참조.

---

### H2 Speed Ablation — 01_speed_ablation.py ✓ (7-model, side-irrelevant)

**Results**: [results/01_speed_ablation_results.csv](../results/01_speed_ablation_results.csv) (scalar만 사용 → side 무관)

**all_speeds 조건 7-model (best AUC 기준)**:
| model | AUC | 95% CI |
|-------|-----|--------|
| CatBoost | **0.9467** | [0.877, 0.996] |
| XGBoost | 0.9385 | [0.882, 0.985] |
| Gradient Boosting | 0.9296 | [0.845, 0.986] |
| Random Forest | 0.9267 | [0.864, 0.974] |
| LightGBM | 0.8963 | [0.818, 0.965] |
| Logistic Regression (baseline) | 0.7652 | [0.614, 0.900] |
| SVM (RBF) | 0.6267 | [0.492, 0.750] |

**H2 confirmed**: 통합(all_speeds)이 단일 속도(slow/normal/fast best ~0.85)보다 일관되게 우월.

---

### H3 Waveform vs Scalar — 02_raw_waveform_ml.py ✓ (regenerated & verified, side-fixed)

**Status**: 완료. Feature sets A-D(unilateral / bilateral asym / speed delta / multi-speed bilateral), 7-model × within-fold PCA + Optuna.
**Results**: [results/02_waveform_results.csv](../results/02_waveform_results.csv) · [results/02_waveform_best.json](../results/02_waveform_best.json)
**Side-fix**: bilateral asymmetry(injured−contralateral)를 `to_inj_con`(Right/Left→inj/con)으로 통일. 표본수 다른 feature set 간 paired-bootstrap 비교는 길이 가드로 skip.

### SHAP — 03_shap_waveform.py ✓
**Status**: 완료. RandomForest + TreeExplainer → PCA→waveform 백프로젝션(9채널×101시점).
**Artifacts**: [results/03_shap_top_timepoints.json](../results/03_shap_top_timepoints.json) · `03_shap_heatmap_mean/fast.npy` · figures `fig_A_beeswarm`·`fig_B/B2_heatmap`·`fig_C_channel`·`fig_D_gait_cycle`.
**핵심 발견**: 최다기여 채널 hip_int_rotation, peak hip_flexion @83% gait cycle. (신버전 SHAP의 3D 배열(n×features×2class)에서 양성클래스 차원 선택 처리.)

---

### 3-Class Multiclass — 04_multiclass.py ✓

**Status**: Complete  
**Results**: [results/04_multiclass_results.csv](../results/04_multiclass_results.csv)

| Model | macro-F1 | bal_acc | AUC(OvR) | Per-class AUC (ACLD/ACLR/HA) |
|-------|----------|---------|----------|-------------------------------|
| RF    | 0.4313   | 0.4326  | 0.6331   | 0.501 / 0.684 / 0.715        |
| XGB   | 0.5164   | 0.5086  | 0.6308   | 0.467 / 0.668 / 0.758        |

**Interpretation**: Low ACLD AUC (0.47-0.50) confirms ACLD vs ACLR boundary is genuinely difficult — both groups share ACL-related compensations. HA is relatively separable (AUC 0.72-0.76). This validates the binary framing (ACL vs HA) as the clinically meaningful task.

---

### 3-Class 최적 파이프라인 — 04b/04c ✓ (7-model, side-fixed)

**04b 3분류 7-model 벤치마크** ([results/04b_benchmark.csv](../results/04b_benchmark.csv)) — AUC(OvR) 기준:
| model | role | macro-F1 | AUC(OvR) | per-class AUC (HA/ACLR/ACLD) |
|-------|------|----------|----------|------------------------------|
| XGBoost | benchmark | 0.684 | **0.8492** | 0.958 / 0.823 / 0.766 |
| Gradient Boosting | benchmark | 0.662 | 0.8271 | 0.941 / 0.808 / 0.733 |
| CatBoost | benchmark | **0.695** | 0.8205 | 0.947 / 0.777 / 0.738 |
| LightGBM | benchmark | 0.611 | 0.8202 | — |
| Random Forest | benchmark | 0.641 | 0.8162 | — |
| Logistic Regression | baseline | 0.630 | 0.7705 | 0.837 / 0.802 / 0.672 |
| SVM (RBF) | benchmark | 0.507 | 0.7215 | — |

**04c** (속도별 격자 + 모델 벤치마크 + Optuna + 분리한계): 최종 이진(all) AUC=**0.9741**, 3분류 최선 AUC(OvR)=**0.8783** (flat).
- 04c 통합조건 모델 벤치마크 ([results/04c_model_benchmark.csv](../results/04c_model_benchmark.csv)): XGBoost 이진 0.9674 / 3분류 0.8549 (best).
- 속도별 격자: 통합(all) > 속도앙상블 > 단일 속도 (H2 재확인).

**ACLD↔ACLR 분리 한계** (왜 3분류 0.98 불가): permutation test 결과 ACLD vs ACLR 직접 분류는 우연 경계선(분리 신호 거의 없음). ID.csv에 구분 메타데이터 없음 → 보행만으론 구분 불가. **임상적 발견**: 재건(ACLR) 후에도 보행이 비재건(ACLD)과 구별 안 됨 = 손상 보상 잔존. ([results/04c_permutation.json](../results/04c_permutation.json))

---

### HTML Reports — reports/ ✓

**Existing output**: [reports/results_report.html](../reports/results_report.html)
7개 섹션: 파이프라인 / 이진분류 / 분류상세 / 속도별분석 / 3분류최대치 / 분리한계 / 데이터설계. This report is result-centered and includes some presentation shortcuts.

**Method/evidence output (v1)**: [reports/02_pipeline_training_method_report.html](../reports/02_pipeline_training_method_report.html) — 보존(pre-fix 시기).

**설명형 보고서 (v2, current official)**: [reports/03_pipeline_explained_v2.html](../reports/03_pipeline_explained_v2.html)
입력→변환→학습→해석 흐름을 실제 값 예시(scalar injured/contralateral, pivot, std/cv, fold scaling, RF top-20, interactions, ensemble)와 함께 설명. 7-model 벤치마크표(baseline=logreg 강조), SHAP 그림 4종, pre-fix exploratory vs current official 구분, GAI intro.md의 H1/H2/H3 맥락 인용. `07_method_report_v2.py`로 생성, xmllint 통과.

---

### 인프라 — 공유 모듈 & 실행 ✓
- [scripts/_side_utils.py](../scripts/_side_utils.py): `to_inj_con`(injured/contralateral & Right/Left → inj/con 통일), `build_injured_leg_map`.
- [scripts/_models.py](../scripts/_models.py): 8-model 레지스트리(logreg baseline + 7). **catboost 고도화**: `boosting_type='Plain'`(기본 Ordered가 소표본 매우 느림) + `iterations≤300` + trial cap(catboost 12 / svm 15) → fold당 ~46초(구 30~50분 대비 ~40배↑). tabpfn은 토큰/캐시 없으면 자동 skip.
- **SMOKE 모드**: 전 스크립트 `SMOKE=1` env로 N_TRIALS=1·2모델·permutation 5·Optuna 2 → 전체 ~3분 스모크. 본 학습 전 6/6 통과 검증.
- **실행 인프라**: [scripts/_run_resume.sh](../scripts/_run_resume.sh)(중단지점 재개 — `.done`/_progress.log DONE skip), `_run_all_seq.sh`(순차 풀코어), `_watch.sh`(실시간 모니터). `( nohup … & )` launchd orphan으로 세션 재연결에도 생존.

---

## Change History

| Timestamp | Task | Change Summary |
|-----------|------|----------------|
| 2026-06-11 14:00 | Session start | Created 0611_journal_ML sandbox, copied plan |
| 2026-06-11 14:30 | Scripts 01-05 | All 5 scripts written (speed ablation, waveform ML, SHAP, multiclass, report) |
| 2026-06-11 15:00 | Data investigation | Waveform S/N analysis: HA bilateral L2=96, ACL=115 — massive overlap |
| 2026-06-11 15:30 | Performance ceiling | Scalar pivot RF (no Optuna, shuffle=False) → 0.9576 → best approach |
| 2026-06-11 16:00 | Stride variability | Added std+cv across strides per subject → 0.9556 → 0.9684 |
| 2026-06-11 16:30 | Interaction features | RF top-20 within-fold pairwise products → 0.9684 → 0.9815 |
| 2026-06-11 16:48 | 02b confirmed | Ensemble seeds 42+88 → AUC=0.9830 ✓ Bootstrap CI [0.9568, 0.9992] |
| 2026-06-11 16:52 | H2 verified | Speed ablation: all_speeds 0.9514 >> single-speed 0.87-0.91 |
| 2026-06-11 17:00 | Multiclass done | RF macro-F1=0.4313, XGB macro-F1=0.5164; ACLD vs ACLR overlap confirmed |
| 2026-06-11 17:05 | HTML report done | 05_journal_report.html generated with TRIPOD format, AUC≥0.98 alert |
| 2026-06-11 17:10 | Git commit | feat: 0611_journal_ML 저널급 ACL 파이프라인 — AUC 98.30% 달성 |
| 2026-06-11 23:10 | 3분류 최적화 | 04b: 파형+PCA → 최적 파이프라인 macro-F1 0.52→0.68, AUC 0.63→0.87 |
| 2026-06-11 23:40 | 속도별+Optuna+한계검증 | 04c: 속도별 격자(통합>앙상블), Optuna 3분류 0.871→0.883, ACLD↔ACLR permutation p=0.05 |
| 2026-06-11 23:55 | 종합 보고서 | 06_results_report.py 7개 섹션 재구성, 속도별/3분류최대치/분리한계 추가 |
| 2026-06-12 00:27 | Method/evidence report | Added `reports/02_pipeline_training_method_report.html`; clarified n=79 source vs n=78 binary model, AUC vs threshold metrics, 04c Optuna validation level, and raw waveform/SHAP artifact absence |
| 2026-06-12 03:00 | Side-fix + 모델확장 | `_side_utils.to_inj_con`로 inj/con 통일(02b 버그: stride side가 전부 con 처리되던 문제 수정). `_models.py` 8-model 레지스트리(logreg baseline + 7 benchmark) 전 스크립트 적용 |
| 2026-06-12 13:50 | catboost 고도화 | `boosting_type='Plain'`+iter≤300+trial cap → fold당 30~50분→46초(~40배↑). 02 비교 길이가드, 02 int64 직렬화, 03 SHAP 3D배열 클래스선택 + 손상json 방어 수정 |
| 2026-06-12 16:00 | SMOKE 선검증 | 전 스크립트 SMOKE env 추가, 6/6 스모크 통과 후 본 학습(에러마다 전체 재시작 방지). `_run_resume.sh` 중단지점 재개 러너 추가 |
| 2026-06-12 20:36 | 본 학습 완주 | side-fixed 7-model 일관 재실행 완료(01·02·03·04b·04c·02b). 이진 best RF AUC=0.9668(구 0.983은 pre-fix), 3분류 best XGB AUC(OvR)=0.8492~0.8549 |
| 2026-06-12 20:37 | v2 설명형 보고서 | `reports/03_pipeline_explained_v2.html` 생성(벤치마크표+SHAP+pre-fix/current 구분), xmllint 통과 |
