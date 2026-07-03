from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MAHALANOBIS = Path(__file__).resolve().parents[1]


def load_engine():
    path = MAHALANOBIS / "scripts" / "06_v2_nested_pipeline.py"
    spec = importlib.util.spec_from_file_location("test_v2_engine", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


engine = load_engine()


def toy_cycles() -> tuple[pd.DataFrame, list[str]]:
    records = []
    for subject, identity, group in [
        ("HA1", "HA::HA1", "HA"), ("HA2", "HA::HA2", "HA"),
        ("ACLD1", "ACL::01", "ACLD"), ("ACLR1", "ACL::01", "ACLR"),
    ]:
        for speed in engine.SPEEDS:
            for side in engine.SIDES:
                for trial in ("t1", "t2"):
                    for cycle in range(2):
                        base = float(len(subject) + cycle)
                        records.append({
                            "subject_id": subject, "biological_identity": identity, "group": group,
                            "speed": speed, "side_basis": side, "trial_id": trial,
                            "cycle_idx": cycle, "f0": base, "f1": base + 1,
                        })
    return pd.DataFrame(records), ["f0", "f1"]


def test_pair_aware_splits_have_no_identity_overlap():
    meta = pd.DataFrame([
        *({"biological_identity": f"HA::{i}", "group": "HA"} for i in range(6)),
        *({"biological_identity": f"ACL::{i}", "group": group} for i in range(6) for group in ("ACLD", "ACLR")),
    ])
    splits, table = engine.make_identity_splits(meta, 3, 42)
    assert len(splits) == 3
    assert table["biological_identity"].is_unique
    for train, validation in splits:
        assert not train & validation


def test_mean_and_inverse_weight_are_duplicate_invariant():
    frame, features = toy_cycles()
    weighted = engine.assign_cycle_weights(frame)
    duplicate = pd.concat([frame, frame], ignore_index=True)
    weighted_duplicate = engine.assign_cycle_weights(duplicate)
    aggregate = engine.mean_aggregate(frame, features)
    aggregate_duplicate = engine.mean_aggregate(duplicate, features)
    key = ["subject_id", "speed", "side_basis"]
    pd.testing.assert_frame_equal(
        aggregate.sort_values(key).reset_index(drop=True),
        aggregate_duplicate.sort_values(key).reset_index(drop=True),
    )
    original_mean = np.average(weighted["f0"], weights=weighted["cycle_weight"])
    duplicate_mean = np.average(weighted_duplicate["f0"], weights=weighted_duplicate["cycle_weight"])
    assert duplicate_mean == pytest.approx(original_mean)
    assert np.allclose(weighted.groupby("subject_id")["cycle_weight"].sum(), 1.0)
    original_model = engine.WeightedMahalanobis("standard", 1, 0.2).fit(
        weighted[features].to_numpy(), weighted["cycle_weight"].to_numpy()
    )
    duplicate_model = engine.WeightedMahalanobis("standard", 1, 0.2).fit(
        weighted_duplicate[features].to_numpy(), weighted_duplicate["cycle_weight"].to_numpy()
    )
    assert np.allclose(original_model.mean_, duplicate_model.mean_)
    assert np.allclose(original_model.covariance_, duplicate_model.covariance_)


def test_raw_distance_and_exact_contribution_identity():
    x = np.array([[0.0, 0.0], [1.0, 0.5], [-1.0, -0.5], [0.5, -1.0], [-0.5, 1.0]])
    weights = np.ones(len(x))
    model = engine.WeightedMahalanobis("standard", 2, 0.2).fit(x, weights)
    query = np.array([[0.2, -0.1], [2.0, 1.0]])
    distance = model.distance(query)
    contributions = model.contributions(query)
    assert np.all(distance >= 0)
    assert np.allclose(contributions.sum(axis=1), np.square(distance), atol=1e-10)


def test_signed_calibration_preserves_negative_values():
    training = pd.DataFrame({"group": ["HA"] * 5, "raw_distance": [1.0, 1.5, 2.0, 2.5, 4.0]})
    center, scale = engine._robust_calibration(training)
    signed = (np.log(np.array([1.0, 4.0])) - center) / scale
    assert signed[0] < 0 < signed[1]


def test_all_speed_concatenation_and_cli_balance_choices():
    frame, features = toy_cycles()
    condition = engine.mean_aggregate(frame, features)
    joined, concat_cols = engine.concatenate_mean_speeds(condition, features)
    assert len(joined) == frame["subject_id"].nunique()
    assert len(concat_cols) == len(engine.SPEEDS) * len(engine.SIDES) * len(features)
    invalid = subprocess.run(
        [sys.executable, str(MAHALANOBIS / "run_pipeline.py"), "--balance-mode", "invalid"],
        capture_output=True, text=True, check=False,
    )
    assert invalid.returncode == 2
    assert "invalid choice" in invalid.stderr
