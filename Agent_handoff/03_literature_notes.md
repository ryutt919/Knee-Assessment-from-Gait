# Literature Notes — ACL Gait Analysis

## Component Status
Describe the current state of each component. Update in-place during work.
Do not duplicate content.

### ACL Kinematics Problem Summary
- **Current value/logic**: `docs/ref_papers/2026-06-18_acl_kinematics_summary/01_acl_patient_kinematics_problem_summary.md` summarizes kinematics-related problems reported across the ACL gait biomechanics and review/meta-analysis PDF folders.
- **Implementation**: The note covers 17 PDF papers from `docs/ref_papers/01_acl_gait_biomechanics_studies` and `docs/ref_papers/02_acl_gait_reviews_meta_analyses`. Each paper section uses the full paper title as a subsection and includes a short original excerpt, Korean interpretation, a structured summary table, and a reference entry. Cross-paper synthesis highlights sagittal-plane stiffened-knee patterns, tibial rotation/translation asymmetries, bilateral adaptation, speed/task dependence, and limits of self-selected walking speed.
- **Related files**: `docs/ref_papers/2026-06-18_acl_kinematics_summary/01_acl_patient_kinematics_problem_summary.md`; `docs/ref_papers/01_acl_gait_biomechanics_studies/`; `docs/ref_papers/02_acl_gait_reviews_meta_analyses/`.
- **Rationale**: The user needed a date-stamped literature summary grounded in the local PDF collection, with mandatory original excerpts, interpretation, tables, and references for ACL patient kinematics problems.

### Time to Normalization ACLR Gait Detail Note
- **Current value/logic**: `docs/ref_papers/2026-06-18_acl_kinematics_summary/02_time_to_normalization_gait_aclr_detailed.md` provides a focused Korean explanation of Chen et al.'s gait normalization meta-analysis.
- **Implementation**: The note centers on study background, methods, main results, time-based recovery estimates, clinical significance, limitations, and report-ready interpretation sentences. It emphasizes that peak knee flexion angle and peak knee flexion moment remain reduced after ACLR, with model-estimated statistical normalization at 16.2 and 10.1 months, while walking speed is not significantly different and may mask knee-level deficits.
- **Related files**: `docs/ref_papers/2026-06-18_acl_kinematics_summary/02_time_to_normalization_gait_aclr_detailed.md`; `docs/ref_papers/02_acl_gait_reviews_meta_analyses/Time to normalization of gait following ACL reconstruction compared with healthy controls - A systematic review and meta-analysis.pdf`.
- **Rationale**: The user requested a deeper single-paper MD explanation focused on results, significance, and background.

### Whole Waveform Analysis Evidence Note
- **Current value/logic**: `docs/ref_papers/2026-06-18_acl_kinematics_summary/03_whole_waveform_analysis_evidence_summary.md` summarizes whether the local `docs/ref_papers` collection contains papers arguing for whole waveform, continuous gait-cycle, or whole-stance analysis.
- **Implementation**: The note is based on text extraction from 53 local reference PDFs/notes and separates direct evidence from supporting review/meta-analysis evidence. It identifies Büttner et al., Garcia et al., Davis-Wilson et al., Lisee et al., the 2024 fast-walking SPM1D kinematics paper, and Capin et al. as direct or methodologically direct support. It also records that Kaur et al. and Chen et al. support the rationale indirectly by highlighting the limits of peak/discrete variables and missing phase-specific metrics.
- **Related files**: `docs/ref_papers/2026-06-18_acl_kinematics_summary/03_whole_waveform_analysis_evidence_summary.md`; `docs/ref_papers/01_acl_gait_biomechanics_studies/`; `docs/ref_papers/02_acl_gait_reviews_meta_analyses/`; `docs/ref_papers/03_wearable_imu_and_portable_sensing/`; `docs/ref_papers/04_machine_learning_and_deep_learning/`.
- **Rationale**: The user asked whether any prior papers in the full reference collection claim that whole waveform analysis is needed, and wanted a new MD summary if evidence existed.

---

## Change History
Record only deltas. Do not repeat content already in Component Status.

| Timestamp | Task | Change Summary |
|-----------|------|----------------|
| 2026-06-18 13:01 | Summarize ACL kinematics literature | no date-stamped kinematics summary -> new 17-paper Korean MD summary with short excerpts, interpretations, per-paper tables, and reference entries |
| 2026-06-18 13:12 | Detail gait normalization meta-analysis | single-paper section in broad summary -> standalone detailed Korean MD focused on background, results, significance, limitations, and report-ready wording |
| 2026-06-18 13:36 | Summarize whole-waveform evidence | no whole-waveform evidence note -> new Korean MD separating direct waveform/SPM evidence from indirect peak/discrete-variable limitation evidence |
