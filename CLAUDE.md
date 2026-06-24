# ACL Gait Analysis — ML Pipeline

## Project Overview

IMU-based gait classification for ACL injury groups using waveform and scalar features.

- **Task**: Classify ACLD / ACLR / HA from stride-level biomechanical data
- **Target**: Binary `{ACLD:1, ACLR:1, HA:0}` (default) or multiclass
- **Gait Normality Score**: GDI/GPS-style HA-referenced waveform deviation with total and domain subscores
- **Platform**: macOS Apple Silicon, Python 3.11, PyTorch, Optuna, MLflow

---

## Instruction File Sync

`CLAUDE.md` and `AGENTS.md` must remain identical. When updating either file, update the other file in the same change with the exact same content.

---

> General operating rules, file placement & naming, Mandatory Rules, and Agent Handoff protocol are defined in the `<work_documentation>` block of the global `~/.claude/CLAUDE.md`.

---

## Entry Points (run from `ML_based/`)

```bash
# Always activate venv first
source .venv/bin/activate
cd ML_based

# Full 5-step pipeline
python run_pipeline.py
python run_pipeline.py --skip-extract --skip-verify   # skip preprocessing
python run_pipeline.py --models logreg rf xgboost     # select models
python run_pipeline.py --test                          # fast smoke run (tiny data)

# Gait Normality Score only
python run_gait_normality_scoring.py
python run_gait_normality_scoring.py --cv-repeats 2 --bootstrap 20  # fast validation

# Model selection only
python orchestrator.py --models all
python orchestrator.py --models cnn1d transformer --waveform_type cycle_norm_101
python orchestrator.py --models cnn1d transformer --waveform_type raw_padded

# Integration check (12 sections)
python smoke_test.py
python smoke_test.py --skip-transformer               # skip if MPS issue
```

---

## Model Registry

| Type | Models |
|------|--------|
| Scalar (7) | `logreg` `linearsvc` `svm_rbf` `rf` `gbt` `xgboost` `lightgbm` |
| Waveform (3) | `fpca` `cnn1d` `transformer` |

- **Scalar input**: `stride_level_peaks.parquet` → η²-guided top-K feature selection
- **Waveform input**: `waveforms_stride.parquet` (subject-speed-side mean), `cycle_waveforms_101.parquet` (cycle-level norm_101), or `stride_raw_waveforms.parquet` (raw_padded)

---

## Data Flow

```
data/processed/  (harness outputs)
  features_scalar.csv          237×148
  stride_level_peaks.parquet   15,135×50
  waveforms_stride.parquet     474×914
  cycle_waveforms_101.parquet  8,079×921  (trimmed cycle-level, 9×101)
  stride_raw_waveforms.parquet 8,002×3,163 (trimmed full-cycle raw padded)
  slim_gait.parquet            995,144×27
  feature_ranking.csv          432×8  (η² ranked)
        ↓
[Step 1a] features/extract_cycle_waveforms_101.py  →  cycle_waveforms_101.parquet
[Step 1b] features/extract_raw_cycles.py  →  stride_raw_waveforms.parquet
[Step 2] features/verify_data.py
[Step 3] orchestrator.py  (Loader → GroupKFold → Optuna → MLflow)
[Step 4] recovery_score/scorer.py + validation.py  →  artifacts-vN/gait_normality/
[Step 5] reports/generate_report.py  →  HTML summary
```

---

## Key Design Decisions

| Decision | Value | Reason |
|----------|-------|--------|
| CV strategy | Outer 5-fold / Inner 3-fold GroupKFold (`subject_id`) | prevent subject leakage |
| Feature selection | η² top-30 from `feature_ranking.csv` | small-N overfitting control |
| Stride trim | 2 strides front/back per trial | exclude accel/decel |
| Cycle waveform unit | `subject_id × speed × trial_id × actual_leg × cycle_idx` | preserve per-cycle samples without trial-boundary leakage |
| Injured side mapping | `data/ID.csv` `Injured leg`; HA uses Right as pseudo-injured | avoid fixed-Right side errors |
| Waveform norm (classify) | `zscore` (train-fold fit) | scale invariance |
| Waveform norm (score) | trial-balanced GVS vs training HA mean; HA-LOO log-distance calibration | 100=HA mean, −10 points=1 HA SD farther |
| Scoring CV unit | 52 biological identities (25 HA + 27 ACLD/ACLR pairs) | keep longitudinal ACLD/ACLR sessions in one fold |
| Optuna metric | `macro_f1` (maximize) | class imbalance |
| DL execution | `spawn` subprocess isolated | segfault prevention on Apple Silicon |

---

## Known Bugs & Pitfalls ⚠️

### CNN1D dilation padding — `models/cnn1d.py`
```python
# WRONG
padding = kernel_size // 2
# CORRECT
padding = (kernel_size - 1) * dilation // 2
```
Without dilation correction spatial dim shrinks per layer → `RuntimeError: Kernel size > input size`.

### CNN1D subprocess isolation — `orchestrator.py`
DL models must run in `spawn` subprocess with env:
```python
env["FORCE_CPU"] = "1"
env["OMP_NUM_THREADS"] = "1"
```
PyTorch 2.x + Optuna trial loop causes SIGSEGV (exit -11) on Apple Silicon without isolation.

### scalar_loader feature mapping — `loaders/scalar_loader.py`
Strip suffixes before mapping to parquet columns:
```python
STRIP = ["_LSI", "_injured", "_uninjured", "_asym", "_contralateral"]
```
After stripping, use **prefix match** (not exact match). Missing `_contralateral` + exact match → only 10/30 features selected (20+ is correct).
Verify: `[scalar_loader] eta2 feature select: N개`

### Optuna stale study — `train/optimize.py`
Study name must include feature count:
```python
study_key = f"{model}_fold{fold}_nf{X_tr.shape[1]}"
```
Without `nf`, changing feature set reuses old study DB → metrics plateau.
Verify: `[Optuna] {model} fold=N: nf={n}, 기존 0 trials`

### Raw cycle injured-side mapping — `features/extract_raw_cycles.py`
Do not hard-code `injured=Right` and `contralateral=Left`. Use `data/ID.csv` `Injured leg` for ACLD/ACLR, keep `actual_leg` separately, and derive `side_basis`. HA convention: Right=`injured` pseudo-side, Left=`contralateral`.

### Cycle-level waveform extraction — `features/extract_cycle_waveforms_101.py`
`cycle_waveforms_101.parquet` is full gait cycle data, not subject-mean waveform data. Process by `(subject_id, group, speed, file_name)` and actual leg, detect heel-strike→heel-strike cycles, drop first/last 2 cycles per trial/leg, then interpolate each cycle to 101 points. Use `--waveform_type cycle_norm_101` to train directly on this file.

---

## Experiment Tracking (Versioned Artifacts)

Artifacts are versioned per experiment run:
```
ML_based/artifacts/        v1 — initial experiments
ML_based/artifacts-v2/     v2 — current
ML_based/artifacts-vN/     vN — future versions
```

Each version contains:
```
artifacts-vN/
├── mlruns/    MLflow run records
├── optuna/    Optuna study .db files
├── logs/      CSV backup logs (runs_YYYYMMDD.csv)
└── figures/   visualization PNGs
```

```bash
# MLflow UI — specify current version
mlflow ui --backend-store-uri ML_based/artifacts-v2/mlruns
```

---

## 0529_ML Sandbox — AUC 95% Achieved (2026-05-30)

Separate sandbox at `0529_ML/` targeting AUC ≥ 95% with traditional ML only (no DL).

**Key result**: Random Forest **AUC = 0.9600** · XGBoost 0.9541 · LightGBM 0.9467

**Strategy** (`0529_ML/scripts/03_boost_auc.py`):
- Pivot `features_scalar.csv` (237 rows, 79 subjects × 3 speeds) → **79-row subject-level dataset**
- Feature engineering: `slow_X`, `normal_X`, `fast_X`, `delta_fast_slow_X`, `delta_fast_normal_X`, `mean_X` per numeric feature → **864 total features**
- CV: outer `StratifiedKFold(5)` at subject level (1 row = 1 subject, no GroupKFold needed)
- SVC uses nested `GridSearchCV(inner=3)` for C/gamma tuning
- All preprocessing inside each fold · OOF evaluation · meta/label columns excluded

**Reports**: `0529_ML/htmls/03_boost_result_report.html` · `0529_ML/results/13_boost_best_summary.json`
