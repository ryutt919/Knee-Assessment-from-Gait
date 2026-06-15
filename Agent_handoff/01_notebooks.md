# Notebooks — ACL Gait Analysis

## Component Status
Describe the current state of each component. Update in-place during work.
Do not duplicate content.

### Peak Segmentation Notebook
- **Current value/logic**: `notebooks/peak_segmentation.ipynb` computes knee flexion segmentation directly with the `harness` stance segmentation and first-peak feature definition instead of reading precomputed `peak_records.csv` for plotting.
- **Implementation**: The notebook imports `JOINT_COLS`, `PEAK_DIRECTION`, `build_stance_contact_signal`, and `get_stance_segments` from `harness/scripts/analysis/preprocess.py`, plus `get_first_peak` and `get_stance_scalar_features` from `harness/utils/peak_utils.py`. It loads `data/processed/raw_merged.parquet`, resolves injured/contralateral side from `data/ID.csv`, computes stance contact with `heel_toe_or`, detects the first prominence peak within each stance segment for knee flexion, applies the same IQR threshold convention used by `iqr_filter_values`, then plots a configurable 10-second window with stance shading and IQR pass/rejected first-peak markers.
- **Related files**: `notebooks/peak_segmentation.ipynb`; `harness/scripts/analysis/preprocess.py`; `data/ID.csv`; `data/processed/raw_merged.parquet`.
- **Rationale**: The notebook's prior peak visualization logic diverged from the harness implementation; importing the harness functions keeps segmentation and peak markers aligned with the pipeline source of truth.

---

## Change History
Record only deltas. Do not repeat content already in Component Status.

| Timestamp | Task | Change Summary |
|-----------|------|----------------|
| 2026-06-15 23:47 | Fix peak segmentation notebook paths | fixed path constants -> filename-based resolver; `data_descrpition2.csv` only -> `data_descrpition2.csv` or `data_descrpition.csv` candidates |
| 2026-06-16 00:03 | Align peak segmentation notebook with harness | precomputed peak-record plotting -> direct harness stance/peak computation for knee flexion 10-second segmented graph |
| 2026-06-16 00:10 | Correct knee flexion peak semantics | stance argmax markers -> first prominence peak markers from `get_first_peak()` because knee flexion peak should be the early-stance loading-response peak |
