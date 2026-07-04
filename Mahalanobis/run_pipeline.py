"""Run versioned leakage-safe Mahalanobis pipelines.

Examples (run from Mahalanobis/):
  python run_pipeline.py --mode dry --trials 2 --outer-folds 3 --inner-folds 2
  python run_pipeline.py --mode full --balance-mode both --trials 20
  python run_pipeline.py --pipeline-version 3.1 --input-profile all --mode dry \
    --outer-folds 2 --inner-folds 2 --cv-repeats 1 --trials 1 --bootstrap 50
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Versioned pair-aware Mahalanobis gait-deviation pipeline")
    parser.add_argument("--pipeline-version", choices=["2.0", "3.1"], default="2.0")
    parser.add_argument("--mode", choices=["dry", "full"], default="full")
    parser.add_argument("--balance-mode", choices=["mean_aggregate", "inverse_weight", "both"], default=None)
    parser.add_argument(
        "--input-profile",
        choices=["scalar_clean_multispeed", "scalar_plus_gvs54", "scalar_plus_waveform5454", "all"],
        default="scalar_clean_multispeed",
    )
    parser.add_argument("--list-profiles", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--cv-repeats", type=int, default=5)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--dry-identities-per-class", type=int, default=12)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true", help="Refuse unsafe resume; completed immutable runs are never overwritten")
    parser.add_argument("--from-run", type=Path, default=None, help="Re-run the exact configuration in a saved manifest")
    return parser.parse_args(argv)


def _apply_manifest(args: argparse.Namespace) -> argparse.Namespace:
    if args.from_run is None:
        return args
    manifest_path = args.from_run
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"from-run manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = str(manifest.get("pipeline_version"))
    if version not in {"2.0", "3.1"}:
        raise SystemExit(f"unsupported manifest pipeline version: {version}")
    args.pipeline_version = version
    for source, target in (
        ("mode", "mode"), ("seed", "seed"), ("outer_folds", "outer_folds"),
        ("inner_folds", "inner_folds"), ("cv_repeats", "cv_repeats"),
        ("n_trials_per_study", "trials"), ("n_bootstrap", "bootstrap"),
        ("dry_identities_per_class", "dry_identities_per_class"),
    ):
        if manifest.get(source) is not None:
            setattr(args, target, manifest[source])
    if version == "2.0":
        args.balance_mode = manifest["balance_mode"]
    else:
        profiles = manifest.get("input_profiles", [])
        args.input_profile = profiles[0] if len(profiles) == 1 else "all"
        args.balance_mode = "mean_aggregate"
    return args


def _update_manifest(path: Path, run_id: str, args: argparse.Namespace, engine, report_filename: str) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["run_id"] = run_id
    manifest["git_revision"] = git_revision()
    manifest["cli"] = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    manifest["runner_source_sha256"] = engine.sha256(Path(__file__))
    manifest["report_source_sha256"] = engine.sha256(SCRIPTS / report_filename)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _run_v2(args: argparse.Namespace, run_id: str) -> Path:
    balance_mode = args.balance_mode or "both"
    artifact_dir = ARTIFACTS / run_id
    engine = load_module("06_v2_nested_pipeline.py", "mahalanobis_v2_engine")
    report_module = load_module("07_generate_v2_report.py", "mahalanobis_v2_report")
    if artifact_dir.exists():
        manifest_path = artifact_dir / "manifest.json"
        if not args.resume or not manifest_path.exists() or not (artifact_dir / "report.html").exists():
            raise SystemExit(f"run directory already exists but is not safely resumable: {artifact_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "mode": args.mode, "balance_mode": balance_mode, "seed": args.seed,
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
        artifact_dir=artifact_dir, balance_mode=balance_mode, mode=args.mode,
        seed=args.seed, outer_folds=args.outer_folds, inner_folds=args.inner_folds,
        n_trials=args.trials,
    )
    _update_manifest(artifact_dir / "manifest.json", run_id, args, engine, "07_generate_v2_report.py")
    report = report_module.generate(artifact_dir)
    print(result["summary"].to_string(index=False))
    print(f"report: {report}")
    return artifact_dir


def _run_v3_1(args: argparse.Namespace, run_id: str) -> Path:
    if args.balance_mode not in (None, "mean_aggregate"):
        raise SystemExit("pipeline v3.1 only permits --balance-mode mean_aggregate")
    if args.cv_repeats < 1 or args.bootstrap < 1:
        raise SystemExit("cv-repeats and bootstrap must be >=1")
    engine = load_module("08_v3_1_mean_pipeline.py", "mahalanobis_v3_1_engine")
    report_module = load_module("09_generate_v3_1_report.py", "mahalanobis_v3_1_report")
    profiles = engine.PROFILES if args.input_profile == "all" else (args.input_profile,)
    artifact_dir = ARTIFACTS / "v3.1" / args.input_profile / run_id
    if artifact_dir.exists():
        manifest_path = artifact_dir / "manifest.json"
        if not args.resume or not manifest_path.exists() or not (artifact_dir / "report.html").exists():
            raise SystemExit(f"run directory already exists but is not safely resumable: {artifact_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "pipeline_version": "3.1", "mode": args.mode,
            "balance_mode": "mean_aggregate", "input_profiles": list(profiles),
            "seed": args.seed, "outer_folds": args.outer_folds,
            "inner_folds": args.inner_folds, "cv_repeats": args.cv_repeats,
            "n_trials_per_study": args.trials, "n_bootstrap": args.bootstrap,
            "dry_identities_per_class": args.dry_identities_per_class if args.mode == "dry" else None,
            "slim_sha256": engine.sha256(engine.V2.SLIM),
            "cycles_sha256": engine.sha256(engine.V2.CYCLES),
            "pairing_sha256": engine.sha256(engine.V2.PAIRING),
            "engine_source_sha256": engine.sha256(SCRIPTS / "08_v3_1_mean_pipeline.py"),
            "report_source_sha256": engine.sha256(SCRIPTS / "09_generate_v3_1_report.py"),
            "runner_source_sha256": engine.sha256(Path(__file__)),
            "configuration_sha256": engine.configuration_hashes(profiles),
        }
        mismatches = {key: (manifest.get(key), value) for key, value in expected.items() if manifest.get(key) != value}
        if mismatches:
            raise SystemExit(f"resume manifest mismatch: {mismatches}")
        print(f"verified completed run reused without mutation: {artifact_dir}")
        return artifact_dir
    result = engine.run(
        artifact_dir=artifact_dir, input_profiles=profiles, mode=args.mode,
        seed=args.seed, outer_folds=args.outer_folds, inner_folds=args.inner_folds,
        cv_repeats=args.cv_repeats, n_trials=args.trials,
        n_bootstrap=args.bootstrap, dry_identities_per_class=args.dry_identities_per_class,
    )
    _update_manifest(artifact_dir / "manifest.json", run_id, args, engine, "09_generate_v3_1_report.py")
    report = report_module.generate(artifact_dir)
    print(result["summary"].to_string(index=False))
    print(f"report: {report}")
    return artifact_dir


def main() -> Path:
    args = _apply_manifest(parse_args())
    if args.list_profiles:
        print("scalar_clean_multispeed (primary, 405 features)")
        print("scalar_plus_gvs54 (comparator, 459 features)")
        print("scalar_plus_waveform5454 (comparator, 5859 features)")
        return SANDBOX
    if args.outer_folds < 2 or args.inner_folds < 2 or args.trials < 1:
        raise SystemExit("folds must be >=2 and trials must be >=1")
    run_id = args.run_id or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{args.mode}_s{args.seed}"
    return _run_v3_1(args, run_id) if args.pipeline_version == "3.1" else _run_v2(args, run_id)


if __name__ == "__main__":
    main()
