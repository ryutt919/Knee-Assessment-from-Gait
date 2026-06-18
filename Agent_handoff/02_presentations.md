# Presentations — ACL Gait Analysis

## Component Status
Describe the current state of each component. Update in-place during work.
Do not duplicate content.

### Capstone Citation Deck
- **Current value/logic**: The capstone presentation has a cited copy that preserves the source deck and adds inline citation markers, slide-bottom footnotes, and two final References slides. A separate citation audit now compares the cited PPT references against `PT/0527 랩미팅/comprehensive_presentation.html` and all 22 HTML references.
- **Implementation**: `PT/scripts/01_add_citations_to_capstone_ppt.py` edits the PPTX package XML directly using `lxml`, because `python-pptx` is not installed in the available Python environments. It reads `PT/스포츠과학과 캡스톤디자인 발표 19101207 김태현.pptx`, writes `PT/스포츠과학과 캡스톤디자인 발표 19101207 김태현_cited.pptx`, corrects the unsupported `ACL surgeries` wording on slide 2 to `ACL injuries`, and cites local `docs/ref_papers` sources for ACL burden, reinjury/OA rates, IMU gait assessment, multi-speed gait, waveform analysis, daily steps, lab-system limitations, and SHAP-based explainable ML. `PT/0527 랩미팅/citation_evidence_audit.md` records claim-level verdicts and flags bibliography mismatches in PPT `[1]` and HTML `[10]`, `[13]`, `[14]`, `[16]`, `[17]`, `[22]`; explanatory text from `HTML 참고문헌별 검토` onward is localized in Korean while preserving paper titles, DOI values, and links.
- **Related files**: `PT/scripts/01_add_citations_to_capstone_ppt.py`; `PT/스포츠과학과 캡스톤디자인 발표 19101207 김태현_cited.pptx`; `PT/0527 랩미팅/citation_evidence_audit.md`; `PT/0527 랩미팅/comprehensive_presentation.html`; `docs/ref_papers/`.
- **Rationale**: The deck text came from prior studies but lacked traceable references; the cited copy makes the source support visible without overwriting the original presentation, and the audit identifies which citations are safe, partial, or must be corrected before final submission.

---

## Change History
Record only deltas. Do not repeat content already in Component Status.

| Timestamp | Task | Change Summary |
|-----------|------|----------------|
| 2026-06-17 10:14 | Add citations to capstone deck | uncited presentation -> cited PPTX copy with inline markers, slide footnotes, References slides, and reproducible PPTX XML edit script |
| 2026-06-17 10:40 | Audit PPT and HTML citations | PPT-only references plus HTML `[1]`-`[22]` -> claim-level evidence audit with corrected-source recommendations and mismatch list |
| 2026-06-18 09:46 | Localize citation audit explanations | English explanations below HTML reference audit -> Korean explanations while preserving source titles, DOI values, and URLs |
