# Gait Normality Scoring — Walking

## Component Status

### GDI/GPS Normality Scorer
- **Current value/logic**: `GaitNormalityScorer` measures trial-balanced 101-point joint-angle waveforms against the training HA mean. `normality_score = 100 - 10 × z(log(GPS distance))`; higher values are closer to the HA reference and scientific outputs are not clipped.
- **Implementation**: Cycles are averaged within trial, then trials within `subject_id × speed × side_basis`. HA leave-one-out raw distances calibrate the total, hip/knee/ankle, slow/normal/fast, bilateral, and asymmetry scores. Point metrics are explanation-only and are not added to the total.
- **Related files**: `ML_based/recovery_score/components.py`, `ML_based/recovery_score/scorer.py`, `ML_based/run_gait_normality_scoring.py`.
- **Rationale**: This implements the requested normal-deviation construct without treating a classifier probability or ACLD/ACLR timepoint label as clinical severity.

### Pair-Aware Validation and Uncertainty
- **Current value/logic**: Validation uses 52 biological identities: 25 HA identities and 27 longitudinal ACL identities. ACLD/ACLR sessions from one patient remain in the same fold; maximum train/test identity overlap is zero.
- **Implementation**: Repeated 5-fold OOF runs 10 repeats. Every fold fits the HA reference, calibration, shrinkage-Mahalanobis comparator, and ridge-logit comparator on training identities only. Trial→cycle hierarchical bootstrap runs 200 iterations per session for a 95% score interval.
- **Related files**: `ML_based/recovery_score/validation.py`, `ML_based/tests/01_test_gait_normality_score.py`.
- **Rationale**: Session-level random folds leak the paired patient identity and inflate validation; nested resampling quantifies measurement uncertainty without claiming test-retest MDC.

### Current Validation Result
- **Current value/logic**: Cross-fitted HA calibration is `100.23 ± 10.78`. Overall normative GPS distance is not supported as a known-group impairment discriminator: AUC `0.485`, Cohen's d `0.073`. ACLR moved closer to HA than ACLD in `12/27` pairs; mean ACLR−ACLD score change is `−0.41`, Wilcoxon `p=0.714`.
- **Implementation**: The report explicitly labels primary known-group validation as unsupported. The asymmetry subscore and ridge-logit comparator each show secondary AUC `0.722`; they remain separate from the canonical total score.
- **Related files**: `ML_based/artifacts-v1/gait_normality/validation_summary.json`, `gait_normality_report.html`, `oof_scores.csv`.
- **Rationale**: Successful implementation does not establish clinical validity. The current data reject the assumption that unweighted whole-waveform RMS distance alone represents ACL impairment severity.

### Outputs and Interpretation
- **Current value/logic**: The output contains 79 finite session scores, total/domain subscores, raw distance, z-deviation, bootstrap CI, top waveform regions, and top HA-standardized point-metric deviations.
- **Implementation**: The CLI writes cohort scores, repeated and aggregated OOF tables, group summary, validation JSON, serialized model JSON, two figures, and an HTML report under `ML_based/artifacts-vN/gait_normality/`.
- **Related files**: `ML_based/run_pipeline.py`, `ML_based/configs/config.yaml`.
- **Rationale**: The score is named Gait Normality/Deviation Score, not Recovery Score, and must not be used as MDC, MCID, or return-to-sport evidence without external clinical anchors and test-retest data.

---

## Change History

| Timestamp | Task | Change Summary |
|-----------|------|----------------|
| 2026-06-24 19:36 | Implement HA-referenced gait scoring | broken scalar component/sigmoid score → trial-balanced GDI/GPS waveform scorer, pair-aware repeated OOF, comparator analysis, hierarchical bootstrap CI, explanations, model artifact, and HTML report |
