#!/usr/bin/env python3
"""Run HA-referenced GDI/GPS gait normality scoring and validation."""
from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from recovery_score.validation import run_scoring_pipeline

ML_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ML_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACL Gait Normality Score")
    parser.add_argument("--config", default=str(ML_ROOT / "configs" / "config.yaml"))
    parser.add_argument("--version", default=None, help="artifact version, e.g. v2")
    parser.add_argument("--cv-repeats", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=None, help="hierarchical bootstrap iterations")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def resolve_data_path(config_path: str | Path, configured_root: str, filename: str) -> Path:
    root = Path(configured_root)
    if not root.is_absolute():
        root = (ML_ROOT / root).resolve()
    return root / filename


def run_from_config(
    config_path: str | Path,
    version: str | None = None,
    cv_repeats: int | None = None,
    bootstrap: int | None = None,
    seed: int | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    cfg = OmegaConf.load(config_path)
    scoring_cfg = cfg.get("scoring", {})
    resolved_version = version or cfg.get("version", "v1")
    cycle_path = resolve_data_path(
        config_path,
        cfg.data.root,
        cfg.data.get("cycle_waveforms_101", "cycle_waveforms_101.parquet"),
    )
    features_path = resolve_data_path(config_path, cfg.data.root, cfg.data.scalar_subject)
    resolved_output = Path(output_dir) if output_dir else ML_ROOT / f"artifacts-{resolved_version}" / "gait_normality"
    return run_scoring_pipeline(
        cycle_path=cycle_path,
        features_path=features_path,
        id_path=PROJECT_ROOT / "data" / "ID.csv",
        output_dir=resolved_output,
        n_splits=int(scoring_cfg.get("n_splits", 5)),
        n_repeats=int(cv_repeats or scoring_cfg.get("cv_repeats", 10)),
        bootstrap_iterations=int(bootstrap or scoring_cfg.get("bootstrap_iterations", 200)),
        seed=int(seed if seed is not None else scoring_cfg.get("seed", 42)),
    )


def main() -> int:
    args = parse_args()
    paths = run_from_config(
        config_path=args.config,
        version=args.version,
        cv_repeats=args.cv_repeats,
        bootstrap=args.bootstrap,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print("[gait-normality] completed")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
