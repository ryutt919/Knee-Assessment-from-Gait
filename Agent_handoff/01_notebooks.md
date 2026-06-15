# Notebooks — ACL Gait Analysis

## Component Status
Describe the current state of each component. Update in-place during work.
Do not duplicate content.

### Peak Segmentation Notebook
- **Current value/logic**: `notebooks/peak_segmentation.ipynb` resolves required input files by filename from the detected project root instead of assuming fixed relative paths.
- **Implementation**: The first setup cell finds `AGENTS.md` by walking up from `Path.cwd()`, then uses `find_project_file()` with direct `data/` and `data/processed/` candidates plus filtered recursive search. Search skips `.git`, virtualenv/cache/runtime folders, and artifact folders. Column metadata accepts either `data_descrpition2.csv` or the current `data_descrpition.csv`.
- **Related files**: `notebooks/peak_segmentation.ipynb`; `data/ID.csv`; `data/processed/raw_merged.parquet`; `data/processed/analysis_data.csv`; `data/processed/peak_records.csv`; `data/data_descrpition.csv`.
- **Rationale**: The dependency file layout/name changed, so filename-based resolution keeps the notebook runnable from the project root or `notebooks/` without hard-coding the old `data_descrpition2.csv` path.

---

## Change History
Record only deltas. Do not repeat content already in Component Status.

| Timestamp | Task | Change Summary |
|-----------|------|----------------|
| 2026-06-15 23:47 | Fix peak segmentation notebook paths | fixed path constants -> filename-based resolver; `data_descrpition2.csv` only -> `data_descrpition2.csv` or `data_descrpition.csv` candidates |
