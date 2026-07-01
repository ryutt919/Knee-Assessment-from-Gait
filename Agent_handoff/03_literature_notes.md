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

### Buttner Waveform Analysis Detail Note
- **Current value/logic**: `docs/ref_papers/2026-06-18_acl_kinematics_summary/04_buttner_waveform_analysis_detailed.md` provides a standalone Korean explanation of Büttner et al.'s bilateral waveform ACLR gait study.
- **Implementation**: The note clarifies the 2025 Journal of Orthopaedic Research citation, confirms the study is a habitual walking gait study, and explains the background, cohort design, stance-phase waveform methods, vGRF/KFA/KEM/KAM results, bilateral interpretation, peak/discrete-variable limitations, clinical implications, limitations, and report-ready wording.
- **Related files**: `docs/ref_papers/2026-06-18_acl_kinematics_summary/04_buttner_waveform_analysis_detailed.md`; `docs/ref_papers/01_acl_gait_biomechanics_studies/Bilateral waveform analysis of gait biomechanics presurgery to 12 months following ACL reconstruction compared to controls.pdf`.
- **Rationale**: The user asked for a new MD file explaining the Büttner et al. study in detail after clarifying that it is a walking gait study.

### Reference Paper Automatic Analysis Pipeline
- **Current value/logic**: Completed automatic extraction and validation of 45 reference papers using the multi-CLI sequential adapter script `Study-review/scripts/01_analyze_reference_papers.py`.
- **Implementation**: The pipeline processes 45 unique PDFs from `docs/ref_papers/01_*` to `06_*` sequentially. Every paper review has verified bibliography information, purpose, design, methods, key results, author conclusions, limitations, reviewer thoughts, and referenceable claims (with page locator, translation, and cited references). Original quotes are dynamically verified against the raw PDF text. Per-paper outputs now save under `Study-review/mds/papers/{docs/ref_papers category}/`.
- **Encountered Issues & Resolutions**:
  1. *JSON Syntax Failures*:
     - **Issue**: LLMs outputted invalid Python-style hex escape sequences (like `\xad`) inside JSON string values, and string concatenations separated by operators outside quotes (like `"..." / "..."`).
     - **Resolution**: Updated `extract_json()` in the script to clean raw strings before parsing (replacing `\\xXX` with `\\u00XX` and merging `" / "` separated quotes).
  2. *Quote Verification Failures*:
     - **Issue (Page Boundaries)**: Hyphenated words split across page boundaries (e.g., `theirreal-\n--- PAGE 324 ---\n323\nworldfeasibility`) failed exact sequence matching due to page headers/numbers inserted mid-word.
     - **Resolution**: Updated `comparison_normalized()` to strip out page boundary markers and page numbers before normalizing.
     - **Issue (Ellipses)**: LLMs extracted quotes containing ellipses (`...`) to denote omissions, which failed continuous substring verification.
     - **Resolution**: Updated `validate_quotes()` to split quotes by ellipsis patterns and verify each sub-segment independently.
- **Related files**:
  - Script: `Study-review/scripts/01_analyze_reference_papers.py`
  - Shared path utility: `Study-review/scripts/path_utils.py`
  - Schema: `Study-review/schemas/01_paper_analysis.schema.json`
  - Prompt: `Study-review/prompts/01_reference_paper_analysis_prompt.md`
  - Output reviews directory: `Study-review/mds/papers/` (45 Markdown files, categorized below the directory)
  - Aggregated reviews: `Study-review/mds/01_all_study_reviews.md`
  - Aggregated claims: `Study-review/mds/02_referenceable_claims.md`
  - Manifest: `Study-review/logs/manifest.json`
- **Rationale**: Provides structured, evidence-grounded literature notes with 100% verified original quotes to avoid hallucinated claims in research papers.

### Reference Paper Summary Fact-check Pipeline
- **Current value/logic**: Completed a second-pass fact-check of all 45 paper summaries against their source PDFs using `Study-review/scripts/02_review_paper_summaries.py`, which mirrors the CLI-invocation logic (provider failover, retry, manifest/resume) of `01_analyze_reference_papers.py` but cross-checks an existing summary MD against the PDF instead of generating a new one.
- **Implementation**: Discovers the 45 (PDF, summary MD) pairs from `01`'s `logs/manifest.json` success records, reuses `01`'s cached `pdftotext -raw` extraction, and prompts the model to flag only factual issues (exception report) — `사실불일치`/`번역오류`/`과장`/`누락`/`인과관계오용`/`수치오류`/`인용표기오류`/`근거불충분` — each with a verbatim quote from the summary, a verbatim counter-quote from the source PDF, and a page locator. Both quote sides are automatically validated against the actual text before being accepted (same normalized-substring technique as `01`). Result: 45/45 success, verdicts 신뢰 가능 2 / 일부 수정 필요 29 / 신뢰 어려움 14, 122 issues flagged total. Per-paper fact-check outputs now save under `Study-review/mds/reviews/{docs/ref_papers category}/`.
- **Encountered Issues & Resolutions**:
  1. *`claude` provider 1M-context billing error*: In this environment `claude --print` defaults to the `[1m]` context model, which returned `API Error: Usage credits required for 1M context`. Resolved by passing `--claude-model claude-sonnet-4-6` (standard-context) when invoking the script.
  2. *Verdict/findings inconsistency (1/45)*: paper 08 (`Longitudinal changes in knee gait mechanics...`) got `overall_verdict: 일부 수정 필요` with `issues_found: 0` — the model described a real gap in `overall_verdict_reason` prose but failed to add a structured `findings` entry for it. Schema validation didn't catch this since it only enforces `issues_found == len(findings)`, not verdict/findings semantic consistency. Worth a manual look if re-running with stricter prompt wording.
- **Related files**:
  - Script: `Study-review/scripts/02_review_paper_summaries.py`
  - Schema: `Study-review/schemas/02_summary_factcheck.schema.json`
  - Prompt: `Study-review/prompts/02_summary_factcheck_prompt.md`
  - Output reviews directory: `Study-review/mds/reviews/` (45 Markdown files, categorized below the directory)
  - Aggregate report: `Study-review/mds/03_summary_factcheck_report.md`
  - Manifest: `Study-review/logs/review/review_manifest.json`
- **Rationale**: The user wanted the existing 45-paper summaries independently checked against the original PDFs for factual accuracy, using the same CLI-calling approach as the original analysis pipeline, with only problem items reported (exception report) rather than a full re-statement of every item.

### Study-review Categorized Output Migration
- **Current value/logic**: `Study-review/mds/papers`, `Study-review/mds/reviews`, and `Study-review/mds/papers_revised` keep their top-level roles, and each now stores per-paper MD files below the matching `docs/ref_papers` category folder.
- **Implementation**: `Study-review/scripts/path_utils.py` derives the category from the PDF parent folder and falls back to `00_uncategorized` for custom inputs without a `01_`-`06_` category. `Study-review/scripts/05_categorize_existing_outputs.py` migrated the existing 45 analysis MDs, 45 fact-check MDs, and 45 revised MDs; it also repaired stale manifest paths that pointed at `2026-06-29-Study-review`.
- **Related files**: `Study-review/scripts/path_utils.py`; `Study-review/scripts/05_categorize_existing_outputs.py`; `Study-review/mds/papers/`; `Study-review/mds/reviews/`; `Study-review/mds/papers_revised/`; `Study-review/logs/manifest.json`; `Study-review/logs/review/review_manifest.json`.
- **Rationale**: The user wanted existing one-paper-per-MD review outputs organized by the existing reference-paper taxonomy and wanted future runs to save directly into that taxonomy.

---

## Change History
Record only deltas. Do not repeat content already in Component Status.

| Timestamp | Task | Change Summary |
|-----------|------|----------------|
| 2026-06-18 13:01 | Summarize ACL kinematics literature | no date-stamped kinematics summary -> new 17-paper Korean MD summary with short excerpts, interpretations, per-paper tables, and reference entries |
| 2026-06-18 13:12 | Detail gait normalization meta-analysis | single-paper section in broad summary -> standalone detailed Korean MD focused on background, results, significance, limitations, and report-ready wording |
| 2026-06-18 13:36 | Summarize whole-waveform evidence | no whole-waveform evidence note -> new Korean MD separating direct waveform/SPM evidence from indirect peak/discrete-variable limitation evidence |
| 2026-06-18 13:43 | Detail Buttner waveform gait study | brief cross-paper summary -> standalone Korean MD explaining citation, walking gait design, stance waveform methods, results, significance, and limitations |
| 2026-06-30 03:35 | Automate reference paper analysis | Initial 1-paper baseline -> Multi-CLI adapter completed and executed for 45 papers, correcting JSON escapes and hyphen splits to achieve 45/45 success |
| 2026-06-30 04:25 | Fact-check 45 paper summaries vs source PDFs | No second-pass verification -> new 02 script/prompt/schema reusing 01's CLI pattern, executed for 45/45 papers as an exception report (122 issues flagged, 14 papers rated 신뢰 어려움) |
| 2026-07-01 19:44 | Categorize Study-review MD outputs | Flat `papers`/`reviews`/`papers_revised` per-paper files -> category subfolders matching `docs/ref_papers/01_*` to `06_*`; future pipeline writes and manifests now use categorized paths |
