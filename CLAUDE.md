# ACL Gait Analysis — ML Pipeline

## Project Overview

IMU-based gait classification for ACL injury groups using waveform and scalar features.

- **Task**: Classify ACLD / ACLR / HA from stride-level biomechanical data
- **Target**: Binary `{ACLD:1, ACLR:1, HA:0}` (default) or multiclass
- **Recovery Score**: SHAP-weighted composite of 5 biomechanical components
- **Platform**: macOS Apple Silicon, Python 3.11, PyTorch, Optuna, MLflow

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

# Model selection only
python orchestrator.py --models all
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
- **Waveform input**: `waveforms_stride.parquet` (norm_101) or `stride_raw_waveforms.parquet` (raw_padded)

---

## Data Flow

```
data/processed/  (harness outputs)
  features_scalar.csv          237×148
  stride_level_peaks.parquet   15,135×50
  waveforms_stride.parquet     474×914
  slim_gait.parquet            995,144×27
  feature_ranking.csv          432×8  (η² ranked)
        ↓
[Step 1] features/extract_raw_cycles.py  →  stride_raw_waveforms.parquet
[Step 2] features/verify_data.py
[Step 3] orchestrator.py  (Loader → GroupKFold → Optuna → MLflow)
[Step 4] recovery_score/scorer.py  →  recovery_scores.csv
[Step 5] reports/generate_report.py  →  HTML summary
```

---

## Key Design Decisions

| Decision | Value | Reason |
|----------|-------|--------|
| CV strategy | Outer 5-fold / Inner 3-fold GroupKFold (`subject_id`) | prevent subject leakage |
| Feature selection | η² top-30 from `feature_ranking.csv` | small-N overfitting control |
| Stride trim | 2 strides front/back per trial | exclude accel/decel |
| Waveform norm (classify) | `zscore` (train-fold fit) | scale invariance |
| Waveform norm (score) | `ha_centered` (HA mean) | deviation from healthy baseline |
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

## Mandatory Rules After Every Code Change

After implementing or modifying code, always do these three things **in order**:

1. **Korean git commit**
   ```
   feat: 기능 추가 설명
   fix: 버그 수정 설명
   refactor: 리팩토링 설명
   ```

2. **Update this CLAUDE.md** — reflect any new bugs, design changes, or command changes in the relevant section immediately.

3. **Update HTML status report** — regenerate `ML_based/reports/status_report.html` with:
   - Always use **light theme** (white background, dark text)
   - Date and summary of changes
   - Latest model performance (macro_f1 per model)
   - Known issues and next tasks
   - Artifact version status
