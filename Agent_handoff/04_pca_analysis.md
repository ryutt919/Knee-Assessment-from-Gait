# PCA Analysis — ACL Gait Analysis

## Component Status
Describe the current state of each component. Update in-place during work.
Do not duplicate content.

### 2026-06-24 Subject-Level PCA Sandbox
- **Current value/logic**: `2026-06-24_PCA/` contains a read-only-source PCA analysis for `slim_gait.parquet`, `raw_merged.parquet`, and `cycle_waveforms_101.parquet`.
- **Implementation**: `scripts/01_run_pca_analysis.py` builds subject-level speed-aware summary features with `pyarrow` batch reads, standardizes features, runs PCA, computes group centroid distances, assigns KMeans clusters, builds hierarchical clustering linkage, adds biomechanical feature descriptions for `jointAngle_*` loadings, and renders PNG figures plus HTML reports.
- **Related files**: `2026-06-24_PCA/scripts/01_run_pca_analysis.py`; `2026-06-24_PCA/htmls/01_execution_log.html`; `2026-06-24_PCA/htmls/02_pca_group_distance_report.html`; `2026-06-24_PCA/results/00_source_file_integrity.json`.
- **Rationale**: The source parquet files include repeated timepoint/cycle rows and `raw_merged.parquet` is 5.9GB, so subject-level aggregation and batch processing are used to make PCA/group-distance/clustering interpretable without loading the full raw file into memory.

---

## Change History
Record only deltas. Do not repeat content already in Component Status.

| Timestamp | Task | Change Summary |
|-----------|------|----------------|
| 2026-06-24 15:22 | Add PCA sandbox analysis | no PCA sandbox -> `2026-06-24_PCA/` subject-level PCA pipeline with results, figures, execution log, final HTML report, and source integrity verification |
| 2026-06-24 15:35 | Clarify PCA loading feature names | raw `jointAngle_*`-only loading labels -> loading CSV/JSON/HTML include `feature_description`, actual Right/Left side, statistic, speed, and source column |
