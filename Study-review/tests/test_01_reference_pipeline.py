#!/usr/bin/env python3
import importlib.util
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SCRIPT = ROOT / "scripts" / "01_analyze_reference_papers.py"
SPEC = importlib.util.spec_from_file_location("paper_pipeline", SCRIPT)
pipeline = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = pipeline
SPEC.loader.exec_module(pipeline)

REVIEW_SPEC = importlib.util.spec_from_file_location(
    "review_pipeline", ROOT / "scripts" / "02_review_paper_summaries.py"
)
review_pipeline = importlib.util.module_from_spec(REVIEW_SPEC)
assert REVIEW_SPEC.loader
sys.modules[REVIEW_SPEC.name] = review_pipeline
REVIEW_SPEC.loader.exec_module(review_pipeline)

RUNNER_SPEC = importlib.util.spec_from_file_location("study_review_runner", ROOT / "run_pipeline.py")
runner = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC.loader
sys.modules[RUNNER_SPEC.name] = runner
RUNNER_SPEC.loader.exec_module(runner)

PATH_UTILS_SPEC = importlib.util.spec_from_file_location("path_utils_test", ROOT / "scripts" / "path_utils.py")
path_utils = importlib.util.module_from_spec(PATH_UTILS_SPEC)
assert PATH_UTILS_SPEC.loader
PATH_UTILS_SPEC.loader.exec_module(path_utils)


class PipelineTests(unittest.TestCase):
    def test_default_inventory_is_62_unique_primary_pdfs_including_category_07(self):
        pdfs = pipeline.discover_pdfs(pipeline.DEFAULT_INPUT)
        self.assertEqual(len(pdfs), 62)
        self.assertEqual(len({pipeline.sha256(p) for p in pdfs}), 62)
        self.assertTrue(all(p.parent.name[:2] in {"01", "02", "03", "04", "05", "06", "07"} for p in pdfs))
        self.assertTrue(any(p.parent.name.startswith("07_") for p in pdfs))

    def test_category_for_default_pdf_uses_parent_folder(self):
        pdf = pipeline.discover_pdfs(pipeline.DEFAULT_INPUT)[0]
        self.assertEqual(path_utils.category_for_pdf(pdf), pdf.parent.name)

    def test_categorized_output_path_includes_kind_and_category(self):
        pdf = pipeline.discover_pdfs(pipeline.DEFAULT_INPUT)[0]
        out = path_utils.categorized_output_path(ROOT, "papers", pdf, "01_example.md")
        self.assertEqual(out, ROOT / "mds" / "papers" / pdf.parent.name / "01_example.md")

    def test_uncategorized_fallback_for_custom_input(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            pdf = Path(td) / "custom.pdf"
            pdf.write_bytes(b"%PDF-1.4\n")
            self.assertEqual(path_utils.category_for_pdf(pdf), "00_uncategorized")

    def test_category_07_is_preserved(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            category = Path(td) / "07_scoring_indices"
            category.mkdir()
            pdf = category / "score.pdf"
            pdf.write_bytes(b"%PDF-1.4\n")
            self.assertEqual(path_utils.category_for_pdf(pdf), "07_scoring_indices")

    def test_input_manifest_selects_exact_batch_and_validates_hash(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            td = Path(td)
            selected = td / "selected.pdf"
            ignored = td / "ignored.pdf"
            selected.write_bytes(b"%PDF selected")
            ignored.write_bytes(b"%PDF ignored")
            manifest = td / "batch.json"
            manifest.write_text(json.dumps({"papers": [{
                "canonical_pdf_path": str(selected), "sha256": pipeline.sha256(selected)
            }]}), encoding="utf-8")
            self.assertEqual(pipeline.select_pdfs(td, manifest), [selected.resolve()])
            payload = json.loads(manifest.read_text())
            payload["papers"][0]["sha256"] = "0" * 64
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                pipeline.select_pdfs(td, manifest)

    def test_incremental_inventory_and_numbers_are_stable(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            td = Path(td)
            old = td / "old.pdf"
            new_a = td / "a.pdf"
            new_b = td / "b.pdf"
            for pdf in (old, new_a, new_b):
                pdf.write_bytes(b"%PDF " + pdf.name.encode())
            records = {str(old.resolve()): {"number": 45, "status": "failed"}}
            numbers = pipeline.stable_numbers(records, [old, new_a, new_b])
            self.assertEqual(numbers[str(old.resolve())], 45)
            self.assertEqual(numbers[str(new_a.resolve())], 46)
            self.assertEqual(numbers[str(new_b.resolve())], 47)
            inventory = pipeline.merge_inventory([str(old.resolve())], [new_a, new_b])
            self.assertEqual(inventory, [str(old.resolve()), str(new_a.resolve()), str(new_b.resolve())])

    def test_selection_counts_ignore_failures_outside_current_batch(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            selected = Path(td) / "selected.pdf"
            stale = Path(td) / "stale.pdf"
            records = {
                str(selected.resolve()): {"status": "success"},
                str(stale.resolve()): {"status": "failed"},
            }
            self.assertEqual(pipeline.selection_counts(records, [selected]), (1, 0))

    def test_review_uses_analysis_manifest_number_and_exact_selection(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            td = Path(td)
            selected = td / "selected.pdf"
            ignored = td / "ignored.pdf"
            summary = td / "summary.md"
            ignored_summary = td / "ignored.md"
            for pdf in (selected, ignored):
                pdf.write_bytes(b"%PDF " + pdf.name.encode())
            summary.write_text("summary", encoding="utf-8")
            ignored_summary.write_text("ignored", encoding="utf-8")
            analysis_manifest = td / "analysis.json"
            analysis_manifest.write_text(json.dumps({"papers": {
                str(selected.resolve()): {"status": "success", "number": 61, "output_path": str(summary)},
                str(ignored.resolve()): {"status": "success", "number": 2, "output_path": str(ignored_summary)},
            }}), encoding="utf-8")
            batch = td / "batch.json"
            batch.write_text(json.dumps({"papers": [{"canonical_pdf_path": str(selected)}]}), encoding="utf-8")
            targets = review_pipeline.discover_review_targets(td, batch, analysis_manifest)
            self.assertEqual([(target.number, target.pdf) for target in targets], [(61, selected.resolve())])

    def test_parent_gate_reset_clears_stale_provider_states(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            out = Path(td)
            for manifest in (out / "logs" / "manifest.json", out / "logs" / "review" / "review_manifest.json"):
                manifest.parent.mkdir(parents=True, exist_ok=True)
                manifest.write_text(json.dumps({
                    "provider_states": {provider: {"status": "exhausted", "error": "quota"}
                                        for provider in runner.PROVIDERS}
                }), encoding="utf-8")
            runner._reset_provider_states(out)
            for manifest in (out / "logs" / "manifest.json", out / "logs" / "review" / "review_manifest.json"):
                states = json.loads(manifest.read_text())["provider_states"]
                self.assertTrue(all(value == {"status": "available", "error": ""}
                                    for value in states.values()))

    def test_provider_rotation_is_cyclic(self):
        self.assertEqual(pipeline.provider_order("codex"), ["codex", "claude", "antigravity"])
        self.assertEqual(pipeline.provider_order("claude"), ["claude", "antigravity", "codex"])
        self.assertEqual(pipeline.provider_order("antigravity"), ["antigravity", "codex", "claude"])

    def test_failure_classification(self):
        self.assertEqual(pipeline.classify_failure("usage limit reached", 1), "unavailable")
        self.assertEqual(pipeline.classify_failure("HTTP 429 too many requests", 1), "transient")
        self.assertEqual(pipeline.classify_failure("unexpected crash", 1), "technical_failure")

    def test_quote_reference_and_locator_validation(self):
        source = "--- PAGE 1 ---\nTest evidence.\n--- PAGE 2 ---\nACL injury is serious [4].\nReferences\n[4] Smith J. ACL study."
        data = pipeline.minimal_fixture("codex")
        data["referenceable_claims"] = [{
            "topic": "심각성", "original_quote": "ACL injury is serious [4].",
            "korean_translation": "ACL 손상은 심각하다.", "locator": "Page 2, Introduction",
            "in_text_citation": "[4]", "cited_reference": "[4] Smith J. ACL study.",
            "claim_type": "background_citation", "usage_note": "2차 인용 주의"
        }]
        self.assertEqual(pipeline.validate_quotes(data, source), [])
        data["referenceable_claims"][0]["original_quote"] = "Invented sentence."
        self.assertIn("원문이 추출 텍스트에 없음", pipeline.validate_quotes(data, source)[0])

    def test_sanitizer_drops_only_unverified_units(self):
        source = "--- PAGE 1 ---\nVerified purpose.\nVerified method.\nVerified result."
        data = pipeline.minimal_fixture("codex")
        data["purpose"] = [
            {"summary": "검증됨", "evidence_quote": "Verified purpose.", "locator": "Page 1"},
            {"summary": "거짓", "evidence_quote": "Invented purpose.", "locator": "Page 1"},
        ]
        data["methods"] = [{"summary": "방법", "evidence_quote": "Verified method.", "locator": "Page 1"}]
        data["key_results"] = [{"summary": "결과", "evidence_quote": "Verified result.", "locator": "Page 1"}]
        dropped = pipeline.sanitize_unverified_items(data, source)
        self.assertEqual(len(data["purpose"]), 1)
        self.assertIn("거짓", dropped[0])
        self.assertEqual(pipeline.validate_quotes(data, source), [])

    def test_output_boundary(self):
        self.assertTrue(pipeline.inside(ROOT / "mds", ROOT))
        self.assertFalse(pipeline.inside(ROOT.parent / "outside", ROOT))

    def test_handoff_contains_resume_and_provider_states(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            out = Path(td)
            manifest = {
                "run_id": "test-run", "papers": {},
                "provider_states": {p: {"status": "exhausted", "error": "quota"} for p in pipeline.PROVIDERS},
            }
            handoff = pipeline.write_handoff(out, manifest, "paper.pdf", "python runner.py --resume")
            text = handoff.read_text()
            self.assertIn("paper.pdf", text)
            self.assertIn("--resume", text)
            self.assertEqual(text.count("상태: exhausted"), 3)

    def test_all_provider_adapters_parse_same_schema(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            td = Path(td)
            fake = td / "fake_provider.py"
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "prompt=sys.stdin.read()\n"
                "start=prompt.find('{'); data=json.loads(prompt[start:])\n"
                "provider=data['provider']\n"
                "if provider=='codex':\n"
                "  out=sys.argv[sys.argv.index('-o')+1]; open(out,'w').write(json.dumps(data,ensure_ascii=False))\n"
                "elif provider=='claude':\n"
                "  print(json.dumps({'type':'result','structured_output':data},ensure_ascii=False))\n"
                "else: print(json.dumps(data,ensure_ascii=False))\n",
                encoding="utf-8",
            )
            fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
            env = {f"PAPER_ANALYZER_{p.upper()}_BIN": str(fake) for p in pipeline.PROVIDERS}
            with patch.dict(os.environ, env, clear=False):
                for provider in pipeline.PROVIDERS:
                    result = pipeline.run_provider_test(provider, td / "logs", None)
                    self.assertTrue(result.ok, (provider, result.error))


if __name__ == "__main__":
    unittest.main(verbosity=2)
