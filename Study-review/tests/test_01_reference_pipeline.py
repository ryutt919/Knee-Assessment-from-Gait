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

PATH_UTILS_SPEC = importlib.util.spec_from_file_location("path_utils_test", ROOT / "scripts" / "path_utils.py")
path_utils = importlib.util.module_from_spec(PATH_UTILS_SPEC)
assert PATH_UTILS_SPEC.loader
PATH_UTILS_SPEC.loader.exec_module(path_utils)


class PipelineTests(unittest.TestCase):
    def test_default_inventory_is_45_unique_primary_pdfs(self):
        pdfs = pipeline.discover_pdfs(pipeline.DEFAULT_INPUT)
        self.assertEqual(len(pdfs), 45)
        self.assertEqual(len({pipeline.sha256(p) for p in pdfs}), 45)
        self.assertTrue(all(p.parent.name[:2] in {"01", "02", "03", "04", "05", "06"} for p in pdfs))

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
