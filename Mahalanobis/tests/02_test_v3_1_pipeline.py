from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

MAHALANOBIS = Path(__file__).resolve().parents[1]


def load_engine():
    path = MAHALANOBIS / "scripts" / "08_v3_1_mean_pipeline.py"
    spec = importlib.util.spec_from_file_location("test_v3_1_engine", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


engine = load_engine()


def synthetic_features(n_per_class: int = 12) -> engine.SessionFeatures:
    rng = np.random.default_rng(7)
    time = np.linspace(0, 2 * np.pi, 101)
    base = np.empty((3, 2, 9, 101))
    for speed in range(3):
        for side in range(2):
            for joint in range(9):
                base[speed, side, joint] = (
                    10 * np.sin(time + joint / 5) + speed * 1.5 + side * .4 + joint
                )
    metadata = []
    waveforms = []
    for idx in range(n_per_class):
        metadata.append({"subject_id": f"HA{idx}", "biological_identity": f"HA::{idx}", "group": "HA"})
        waveforms.append(base + rng.normal(0, .5, base.shape) + idx * .03)
    for idx in range(n_per_class):
        identity = f"ACL::{idx:02d}"
        for group, offset in (("ACLD", 1.2), ("ACLR", .8)):
            metadata.append({"subject_id": f"{group}{idx}", "biological_identity": identity, "group": group})
            altered = base + rng.normal(0, .6, base.shape)
            altered[:, 0, 5] += offset * np.sin(time) + offset
            waveforms.append(altered)
    waveform_array = np.asarray(waveforms)
    scalar, names = engine.extract_scalar_features(waveform_array)
    return engine.SessionFeatures(pd.DataFrame(metadata), waveform_array, scalar, names)


def test_scalar_schema_and_bounded_symmetry():
    features = synthetic_features()
    assert features.scalar.shape == (36, 405)
    symmetry = [idx for idx, name in enumerate(features.scalar_names) if name.endswith("__symmetry")]
    assert len(symmetry) == 135
    assert np.isfinite(features.scalar).all()
    assert np.max(np.abs(features.scalar[:, symmetry])) <= 1.0


def test_versioned_profile_registry_and_config_hashes():
    registry = yaml.safe_load((MAHALANOBIS / "versions" / "registry.yaml").read_text())
    v31 = next(item for item in registry["versions"] if item["pipeline_version"] == "3.1")
    assert tuple(v31["profiles"]) == engine.PROFILES
    hashes = engine.configuration_hashes(engine.PROFILES)
    assert len(hashes) == 6
    assert all(len(value) == 64 for value in hashes.values())


def test_profile_shapes_and_gvs_reference_is_training_only():
    features = synthetic_features()
    ha = np.flatnonzero(features.metadata["group"].eq("HA").to_numpy())[:4]
    for profile, expected in engine.profile_feature_counts().items():
        blocks, _, reference = engine.profile_blocks(features, profile, ha)
        assert sum(block.shape[1] for block in blocks.values()) == expected
        if profile == "scalar_plus_gvs54":
            assert np.allclose(reference, features.waveforms[ha].mean(axis=0))
            original = reference.copy()
            features.waveforms[-1] += 1000
            _, _, unchanged = engine.profile_blocks(features, profile, ha)
            assert np.allclose(original, unchanged)


def test_gvs54_matches_ml_based_component_formula():
    path = MAHALANOBIS.parent / "ML_based" / "recovery_score" / "components.py"
    spec = importlib.util.spec_from_file_location("test_ml_gvs_components", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    features = synthetic_features()
    reference = features.waveforms[:6].mean(axis=0)
    expected = module.gvs_matrix(features.waveforms, reference).reshape(len(features.waveforms), -1)
    assert np.allclose(engine._gvs(features.waveforms, reference), expected)


def test_ha_side_swap_changes_only_ha_orientation():
    features = synthetic_features()
    swapped = engine.swap_ha_side_orientation(features)
    ha = features.metadata["group"].eq("HA").to_numpy()
    assert np.allclose(swapped.waveforms[ha], features.waveforms[ha, :, ::-1, :, :])
    assert np.allclose(swapped.waveforms[~ha], features.waveforms[~ha])
    assert swapped.scalar.shape == features.scalar.shape


def test_block_model_contributions_sum_to_distance_squared():
    features = synthetic_features()
    ha = np.flatnonzero(features.metadata["group"].eq("HA").to_numpy())[:6]
    blocks, names, _ = engine.profile_blocks(features, "scalar_plus_gvs54", ha)
    model = engine.BlockNormativeModel(3, "standard").fit(blocks, names, ha)
    query = {key: value[-3:] for key, value in blocks.items()}
    distance = model.distance(query)
    contribution = model.contributions(query)
    assert np.all(distance >= 0)
    assert np.allclose(contribution.sum(axis=1), np.square(distance), atol=1e-8)


def test_pair_aware_class_balanced_splits():
    features = synthetic_features()
    splits, _ = engine.V2.make_identity_splits(features.metadata, 2, 42)
    for train, validation in splits:
        assert not train & validation
        validation_meta = features.metadata[features.metadata["biological_identity"].isin(validation)]
        counts = {
            "ACL": validation_meta.loc[validation_meta["group"].ne("HA"), "biological_identity"].nunique(),
            "HA": validation_meta.loc[validation_meta["group"].eq("HA"), "biological_identity"].nunique(),
        }
        assert counts == {"ACL": 6, "HA": 6}
        assert validation_meta.groupby("biological_identity")["group"].apply(
            lambda x: set(x) in ({"HA"}, {"ACLD", "ACLR"})
        ).all()


def test_tiny_core_pipeline_writes_complete_artifacts(tmp_path: Path):
    features = synthetic_features()
    split_pairs, table = engine.V2.make_identity_splits(features.metadata, 2, 42)
    splits = [(0, fold, train, validation) for fold, (train, validation) in enumerate(split_pairs)]
    artifact = tmp_path / "tiny"
    artifact.mkdir()
    scores = {}
    for profile in engine.PROFILES:
        result, diagnostics = engine.run_profile(features, profile, splits, artifact, 2, 1, 42)
        scores[profile] = result
        assert len(result) == len(features.metadata)
        assert np.isfinite(result[[
            "raw_distance", "overall_z_deviation", "normality_score",
            "gvs_raw_gps", "slow_gvs_distance", "normal_gvs_distance", "fast_gvs_distance",
            "legacy_speed_rms",
        ]]).all().all()
        assert (result["legacy_speed_rms"] >= 0).all()
        assert any((result[f"{speed}_signed_deviation"] < 0).any() for speed in engine.SPEEDS)
        assert len(diagnostics) == 2
    summary = engine.summarize(scores, artifact, 20, 42)
    assert set(summary["profile"]) == set(engine.PROFILES)
    assert (artifact / "summary_metrics.csv").exists()
    assert (artifact / "oof_session_scores_averaged.parquet").exists()
    assert table["biological_identity"].is_unique


def test_v3_cli_contract():
    listed = subprocess.run(
        [sys.executable, str(MAHALANOBIS / "run_pipeline.py"), "--list-profiles"],
        capture_output=True, text=True, check=False,
    )
    assert listed.returncode == 0
    assert "scalar_clean_multispeed" in listed.stdout
    invalid = subprocess.run(
        [
            sys.executable, str(MAHALANOBIS / "run_pipeline.py"),
            "--pipeline-version", "3.1", "--balance-mode", "inverse_weight",
            "--mode", "dry", "--outer-folds", "2", "--inner-folds", "2",
        ],
        capture_output=True, text=True, check=False,
    )
    assert invalid.returncode != 0
    assert "only permits" in (invalid.stdout + invalid.stderr)


def test_dry_cohort_rejects_too_few_identities():
    frame = pd.DataFrame({
        "biological_identity": [f"HA::{i}" for i in range(12)] + [f"ACL::{i}" for i in range(12)],
        "group": ["HA"] * 12 + ["ACLD"] * 12,
    })
    with pytest.raises(ValueError, match="at least 12"):
        engine._select_dry(frame, 42, 11)
