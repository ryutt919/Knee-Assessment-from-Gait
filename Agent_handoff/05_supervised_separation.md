# Supervised Separation — Walking

## Component Status

### 2026-06-24 Supervised Separation Sandbox
- **Current value/logic**: `2026-06-24_supervised_separation/` contains a subject-level supervised embedding experiment for `ACLD` / `ACLR` / `HA` and binary `ACL` / `HA` separation.
- **Implementation**: `scripts/01_supervised_separation.py` builds scalar pivot, PCA-subject feature sets, scalar+waveform/raw/fusion feature sets, then evaluates PCA, LDA, PLS-DA, NCA, and RandomForest probability embeddings with 5-fold `StratifiedKFold`.
- **Related files**: `2026-06-24_supervised_separation/scripts/01_supervised_separation.py`, `results/embedding_separation_summary.csv`, `results/best_embedding_summary.json`, `htmls/02_supervised_separation_report.html`, `htmls/01_execution_log.html`.
- **Rationale**: The target is not just predictive AUC but fold-wise embedding separability. All imputation, variance filtering, feature selection, scaling, and embedding fitting happen inside each training fold before test-fold transformation.

### Feature Sets
- **Current value/logic**: Seven subject-level feature sets are evaluated: `scalar_pivot`, `pca_slim_subject`, `pca_raw_subject`, `pca_cycle_subject`, `fusion_scalar_cycle`, `fusion_scalar_raw`, and `fusion_all_common`.
- **Implementation**: `features_scalar.csv` is pivoted by speed with mean and fast-slow/fast-normal deltas. Existing PCA sandbox subject feature CSVs are read as fixed inputs. High-dimensional folds use train-fold-only `SelectKBest(f_classif)` capped at 160 selected features.
- **Related files**: `results/feature_set_inventory.csv`, `data/processed/features_scalar.csv`, `2026-06-24_PCA/results/*_subject_features.csv`.
- **Rationale**: This compares scalar-only, PCA-derived gait summaries, waveform-derived summaries, raw-derived summaries, and fusion candidates without modifying source data.

### Best Current Results
- **Current value/logic**: Binary `ACL` vs `HA` best embedding is `fusion_scalar_raw` + `rf_proba`: silhouette `0.6379`, centroid/within ratio `5.1927`, balanced accuracy `0.9200`, binary AUC `0.9589`.
- **Implementation**: Best selection sorts by highest fold-wise `silhouette_mean`, then `centroid_ratio_mean`, then lower `davies_bouldin_mean`.
- **Related files**: `results/best_embedding_summary.json`, `results/embedding_separation_summary.csv`.
- **Rationale**: Binary separation benefits from fused scalar and raw-derived subject features when represented through RandomForest probability embedding.

### Three-Group Bottleneck
- **Current value/logic**: Three-group best embedding is `scalar_pivot` + `rf_proba`: silhouette `0.1685`, centroid/within ratio `2.5816`, balanced accuracy `0.6400`, 3-class OvR AUC `0.8431`.
- **Implementation**: Pairwise group distances are written for each task, feature set, method, and fold; the report includes an ACLD vs ACLR pairwise separation figure.
- **Related files**: `results/pairwise_group_distances.csv`, `figures/multiclass_acld_aclr_pairwise_separation.png`.
- **Rationale**: Three-group separation is much weaker than binary separation, so ACLD vs ACLR overlap should be treated as the primary bottleneck rather than relying on a visually attractive 2D plot.

### Verification
- **Current value/logic**: Latest run produced 350 metric rows with 0 failed rows. Source integrity check covers original parquet files, `features_scalar.csv`, and PCA sandbox subject features; `unchanged=True`.
- **Implementation**: HTML report has 7 figure links; Playwright/browser verification loaded `02_supervised_separation_report.html`, all 7 PNGs with HTTP 200, and `01_execution_log.html`.
- **Related files**: `results/00_source_file_integrity.json`, `htmls/01_execution_log.html`, `htmls/02_supervised_separation_report.html`.
- **Rationale**: This keeps the sandbox reproducible and verifies that source files and existing PCA outputs were read-only.

---

## Change History

| Timestamp | Task | Change Summary |
|-----------|------|----------------|
| 2026-06-24 15:55 | 학습 기반 그룹 분리도 최대화 실험 | new sandbox → `2026-06-24_supervised_separation/` added with fold-wise supervised embedding evaluation, reports, figures, integrity logs, and source read-only verification |
