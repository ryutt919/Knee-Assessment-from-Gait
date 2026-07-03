"""Run the leakage-safe Mahalanobis v2 pipeline.

Examples (run from Mahalanobis/):
  python run_pipeline.py --mode dry --trials 2 --outer-folds 3 --inner-folds 2
  python run_pipeline.py --mode full --balance-mode both --trials 20
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SANDBOX = Path(__file__).resolve().parent
SCRIPTS = SANDBOX / "scripts"
ARTIFACTS = SANDBOX / "artifacts"


def load_module(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    if spec is None or spec.loader is None:
        raise ImportError(filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=SANDBOX, text=True,
        capture_output=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pair-aware Mahalanobis gait-deviation pipeline v2")
    parser.add_argument("--mode", choices=["dry", "full"], default="full")
    parser.add_argument("--balance-mode", choices=["mean_aggregate", "inverse_weight", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true", help="Refuse unsafe resume; completed immutable runs are never overwritten")
    return parser.parse_args()


def main() -> Path:
    args = parse_args()
    if args.outer_folds < 2 or args.inner_folds < 2 or args.trials < 1:
        raise SystemExit("folds must be >=2 and trials must be >=1")
    run_id = args.run_id or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{args.mode}_s{args.seed}"
    artifact_dir = ARTIFACTS / run_id
    engine = load_module("06_v2_nested_pipeline.py", "mahalanobis_v2_engine")
    report_module = load_module("07_generate_v2_report.py", "mahalanobis_v2_report")
    if artifact_dir.exists():
        manifest_path = artifact_dir / "manifest.json"
        if not args.resume or not manifest_path.exists() or not (artifact_dir / "report.html").exists():
            raise SystemExit(f"run directory already exists but is not safely resumable: {artifact_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "mode": args.mode, "balance_mode": args.balance_mode, "seed": args.seed,
            "outer_folds": args.outer_folds, "inner_folds": args.inner_folds,
            "n_trials_per_study": args.trials,
            "slim_sha256": engine.sha256(engine.SLIM),
            "cycles_sha256": engine.sha256(engine.CYCLES),
            "pairing_sha256": engine.sha256(engine.PAIRING),
            "engine_source_sha256": engine.sha256(SCRIPTS / "06_v2_nested_pipeline.py"),
            "report_source_sha256": engine.sha256(SCRIPTS / "07_generate_v2_report.py"),
            "runner_source_sha256": engine.sha256(Path(__file__)),
        }
        mismatches = {key: (manifest.get(key), value) for key, value in expected.items() if manifest.get(key) != value}
        if mismatches:
            raise SystemExit(f"resume manifest mismatch: {mismatches}")
        print(f"verified completed run reused without mutation: {artifact_dir}")
        return artifact_dir
    result = engine.run(
        artifact_dir=artifact_dir,
        balance_mode=args.balance_mode,
        mode=args.mode,
        seed=args.seed,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        n_trials=args.trials,
    )
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_id"] = run_id
    manifest["git_revision"] = git_revision()
    manifest["cli"] = vars(args)
    manifest["runner_source_sha256"] = engine.sha256(Path(__file__))
    manifest["report_source_sha256"] = engine.sha256(SCRIPTS / "07_generate_v2_report.py")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    report = report_module.generate(artifact_dir)
    print(result["summary"].to_string(index=False))
    print(f"report: {report}")
    return artifact_dir


if __name__ == "__main__":
    main()
