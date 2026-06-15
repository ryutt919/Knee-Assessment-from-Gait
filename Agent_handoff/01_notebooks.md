# Notebooks — ACL Gait Analysis

## Component Status
Describe the current state of each component. Update in-place during work.
Do not duplicate content.

### Peak Segmentation Notebook
- **Current value/logic**: `notebooks/peak_segmentation.ipynb` now computes knee flexion segmentation directly with the `harness` peak-detection implementation instead of reading precomputed `peak_records.csv` for plotting.
- **Implementation**: The notebook imports `JOINT_COLS`, `PEAK_DIRECTION`, `build_stance_contact_signal`, `get_stance_segments`, and `detect_peaks_with_iqr` from `harness/scripts/analysis/preprocess.py`. It loads `data/processed/raw_merged.parquet`, resolves injured/contralateral side from `data/ID.csv`, computes stance contact with `heel_toe_or`, detects knee flexion peaks with `peak_method="argextrema"` and the harness IQR limits, then plots a configurable 10-second window with stance shading and IQR pass/rejected peak markers.
- **Related files**: `notebooks/peak_segmentation.ipynb`; `harness/scripts/analysis/preprocess.py`; `data/ID.csv`; `data/processed/raw_merged.parquet`.
- **Rationale**: The notebook's prior peak visualization logic diverged from the harness implementation; importing the harness functions keeps segmentation and peak markers aligned with the pipeline source of truth.

---

## Change History
Record only deltas. Do not repeat content already in Component Status.

| Timestamp | Task | Change Summary |
|-----------|------|----------------|
| 2026-06-15 23:47 | Fix peak segmentation notebook paths | fixed path constants -> filename-based resolver; `data_descrpition2.csv` only -> `data_descrpition2.csv` or `data_descrpition.csv` candidates |
| 2026-06-16 00:03 | Align peak segmentation notebook with harness | precomputed peak-record plotting -> direct harness stance/peak computation for knee flexion 10-second segmented graph |
