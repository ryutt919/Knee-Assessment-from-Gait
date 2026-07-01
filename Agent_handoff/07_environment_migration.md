# Environment Migration — ACL Gait Analysis

## Component Status

### Environment dependency records
- **Current value/logic**: Four independent virtual environments are recorded as exact macOS snapshots and curated WSL2 requirements.
- **Implementation**: `environments/01_environment_manifest.yaml` maps each environment to its Python version, freeze file, portable requirements, PyTorch version, and platform exclusions.
- **Related files**: `environments/*.txt`, `environments/01_environment_manifest.yaml`, `.gitignore`
- **Rationale**: Preserve the current Mac package state while allowing each environment to be rebuilt independently on WSL2.

### Platform-specific packages
- **Current value/logic**: macOS-only packages remain in freeze snapshots but are excluded from WSL requirements. PyTorch is installed separately according to the target CPU or CUDA configuration.
- **Implementation**: Root excludes `appnope`; the journal environment excludes `mlx` and `mlx-metal`; PyTorch versions remain recorded in the manifest.
- **Related files**: `environments/01_environment_manifest.yaml`, `environments/03_root_wsl_requirements.txt`, `environments/09_0611_wsl_requirements.txt`
- **Rationale**: Avoid platform-incompatible installs without losing source-environment provenance.

### Code-required dependency gaps
- **Current value/logic**: Portable requirements include code imports that are missing from the current source freezes.
- **Implementation**: Root WSL adds `shap==0.51.0` for SHAP evaluation; journal WSL adds `umap-learn==0.5.11` for the embedding script. The manifest distinguishes these additions from exact snapshots.
- **Related files**: `ML_based/eval/shap_analysis.py`, `0611_journal_ML/scripts/08_slim_gait_embedding.py`, `environments/01_environment_manifest.yaml`
- **Rationale**: A portable environment must cover actual code imports, while the snapshot remains an unmodified record of the current Mac environment.

---

## Change History

| Timestamp | Task | Change Summary |
|-----------|------|----------------|
| 2026-07-01 21:28 | Record environments for Windows migration | No dependency records → four macOS snapshots plus four WSL requirements and a migration manifest |
