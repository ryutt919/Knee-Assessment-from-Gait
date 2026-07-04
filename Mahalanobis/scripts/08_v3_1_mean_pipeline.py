"""Mahalanobis v3.1: mean-only, subject/session-level normative scoring.

The publication primary is a strict multi-speed scalar representation. Two
pre-specified comparators add either 54 Gait Variable Scores or the complete
5,454-point mean waveform. ACLD/ACLR sessions remain separate rows while their
shared biological identity is locked to the same inner and outer CV folds.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import optuna
import pandas as pd
from scipy.signal import find_peaks
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
SANDBOX = Path(__file__).resolve().parents[1]
V2_PATH = SANDBOX / "scripts" / "06_v2_nested_pipeline.py"
CONFIG_ROOT = SANDBOX / "configs"
REGISTRY_PATH = SANDBOX / "versions" / "registry.yaml"

SPEEDS = ("slow", "normal", "fast")
SIDES = ("injured", "contralateral")
JOINTS = (
    "hip_adduction", "hip_int_rotation", "hip_flexion",
    "knee_adduction", "knee_int_rotation", "knee_flexion",
    "ankle_adduction", "ankle_int_rotation", "ankle_dorsiflexion",
)
METRICS = ("peak", "min", "ROM", "IC_angle", "peak_timing")
PEAK_DIRECTION = {
    "hip_adduction": "min", "hip_int_rotation": "max", "hip_flexion": "max",
    "knee_adduction": "min", "knee_int_rotation": "max", "knee_flexion": "max",
    "ankle_adduction": "min", "ankle_int_rotation": "max", "ankle_dorsiflexion": "max",
}
PROFILES = (
    "scalar_clean_multispeed",
    "scalar_plus_gvs54",
    "scalar_plus_waveform5454",
)
PRIMARY_PROFILE = "scalar_clean_multispeed"
EPSILON = 1e-12


def _load_v2():
    spec = importlib.util.spec_from_file_location("mahalanobis_v2_shared", V2_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(V2_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V2 = _load_v2()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def profile_feature_counts() -> dict[str, int]:
    return {
        "scalar_clean_multispeed": 405,
        "scalar_plus_gvs54": 459,
        "scalar_plus_waveform5454": 5859,
    }


def configuration_hashes(profiles: Iterable[str]) -> dict[str, str]:
    paths = {
        f"input::{profile}": CONFIG_ROOT / "inputs" / f"{profile}.yaml"
        for profile in profiles
    }
    paths.update({
        "score": CONFIG_ROOT / "scores" / "label_guided_normative.yaml",
        "cv": CONFIG_ROOT / "cv" / "repeated_nested_5x5.yaml",
        "registry": REGISTRY_PATH,
    })
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"versioned configuration files missing: {missing}")
    return {name: sha256(path) for name, path in paths.items()}


@dataclass(frozen=True)
class SessionFeatures:
    metadata: pd.DataFrame
    waveforms: np.ndarray  # session x speed x side x joint x 101
    scalar: np.ndarray  # session x 405
    scalar_names: tuple[str, ...]


def _first_peak(curve: np.ndarray, direction: str, prominence: float = 1.0) -> int:
    target = curve if direction == "max" else -curve
    indices, _ = find_peaks(target, prominence=prominence)
    if len(indices):
        return int(indices[0])
    return int(np.argmax(curve) if direction == "max" else np.argmin(curve))


def _curve_metrics(curve: np.ndarray, direction: str) -> dict[str, float]:
    curve = np.asarray(curve, dtype=np.float64)
    if curve.shape != (101,) or not np.isfinite(curve).all():
        raise ValueError("scalar extraction requires one finite 101-point mean cycle")
    peak_idx = _first_peak(curve, direction)
    return {
        "peak": float(curve[peak_idx]),
        "min": float(curve.min()),
        "ROM": float(curve.max() - curve.min()),
        "IC_angle": float(curve[0]),
        "peak_timing": float(peak_idx),
    }


def extract_scalar_features(waveforms: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    """Create 405 strict bilateral and bounded-symmetry features."""
    waveforms = np.asarray(waveforms, dtype=np.float64)
    expected = (len(SPEEDS), len(SIDES), len(JOINTS), 101)
    if waveforms.ndim != 5 or waveforms.shape[1:] != expected:
        raise ValueError(f"unexpected waveform tensor for scalar extraction: {waveforms.shape}")
    rows: list[list[float]] = []
    names: list[str] | None = None
    for session in waveforms:
        values: list[float] = []
        row_names: list[str] = []
        for speed_idx, speed in enumerate(SPEEDS):
            by_side: dict[tuple[str, str], dict[str, float]] = {}
            for joint_idx, joint in enumerate(JOINTS):
                for side_idx, side in enumerate(SIDES):
                    metrics = _curve_metrics(session[speed_idx, side_idx, joint_idx], PEAK_DIRECTION[joint])
                    by_side[(joint, side)] = metrics
                    for metric in METRICS:
                        row_names.append(f"{speed}__{joint}__{metric}__{side}")
                        values.append(metrics[metric])
            for joint in JOINTS:
                for metric in METRICS:
                    injured = by_side[(joint, "injured")][metric]
                    contra = by_side[(joint, "contralateral")][metric]
                    symmetry = (injured - contra) / (abs(injured) + abs(contra) + EPSILON)
                    row_names.append(f"{speed}__{joint}__{metric}__symmetry")
                    values.append(float(symmetry))
        if names is None:
            names = row_names
        elif names != row_names:
            raise RuntimeError("non-deterministic scalar feature schema")
        rows.append(values)
    assert names is not None
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape[1] != 405 or not np.isfinite(matrix).all():
        raise ValueError(f"invalid clean scalar matrix: shape={matrix.shape}")
    symmetry_idx = [idx for idx, name in enumerate(names) if name.endswith("__symmetry")]
    if np.max(np.abs(matrix[:, symmetry_idx])) > 1.0 + 1e-12:
        raise ValueError("bounded symmetry escaped [-1, 1]")
    return matrix, tuple(names)


def build_session_features(cycles: pd.DataFrame, wave_cols: list[str]) -> SessionFeatures:
    """Trial-balanced mean aggregation followed by one row per session."""
    condition = V2.mean_aggregate(cycles, wave_cols)
    joined, _ = V2.concatenate_mean_speeds(condition, wave_cols)
    metadata = joined[["subject_id", "biological_identity", "group"]].copy().reset_index(drop=True)
    waveforms = np.empty((len(joined), 3, 2, 9, 101), dtype=np.float64)
    for speed_idx, speed in enumerate(SPEEDS):
        for side_idx, side in enumerate(SIDES):
            columns = [f"{speed}__{side}__{column}" for column in wave_cols]
            waveforms[:, speed_idx, side_idx] = joined[columns].to_numpy(dtype=np.float64).reshape(-1, 9, 101)
    scalar, scalar_names = extract_scalar_features(waveforms)
    if metadata["subject_id"].duplicated().any() or not np.isfinite(waveforms).all():
        raise ValueError("session representation must contain one finite row per session")
    return SessionFeatures(metadata, waveforms, scalar, scalar_names)


def swap_ha_side_orientation(features: SessionFeatures) -> SessionFeatures:
    """Swap only HA Right/Left pseudo-orientation for the pre-specified sensitivity."""
    waveforms = features.waveforms.copy()
    ha = features.metadata["group"].eq("HA").to_numpy()
    waveforms[ha] = waveforms[ha, :, ::-1, :, :]
    scalar, names = extract_scalar_features(waveforms)
    if names != features.scalar_names:
        raise RuntimeError("HA side swap changed scalar schema")
    return SessionFeatures(features.metadata.copy(), waveforms, scalar, names)


def validate_side_contract(cycles: pd.DataFrame) -> dict:
    required = {"subject_id", "group", "actual_leg", "side_basis", "injured_leg"}
    if not required.issubset(cycles.columns):
        raise ValueError(f"side audit missing columns: {sorted(required-set(cycles.columns))}")
    ha = cycles[cycles["group"].eq("HA")]
    ha_ok = (
        ((ha["actual_leg"].eq("Right")) & ha["side_basis"].eq("injured"))
        | ((ha["actual_leg"].eq("Left")) & ha["side_basis"].eq("contralateral"))
    ).all()
    if not ha_ok:
        raise ValueError("HA side convention must be Right=pseudo-injured, Left=pseudo-contralateral")
    acl = cycles[cycles["group"].isin(["ACLD", "ACLR"])]
    acl_ok = (
        (acl["actual_leg"].eq(acl["injured_leg"]) & acl["side_basis"].eq("injured"))
        | (~acl["actual_leg"].eq(acl["injured_leg"]) & acl["side_basis"].eq("contralateral"))
    ).all()
    if not acl_ok:
        raise ValueError("ACL side_basis disagrees with actual injured leg")
    pair_legs = acl[["biological_identity", "subject_id", "injured_leg"]].drop_duplicates()
    mismatches = pair_legs.groupby("biological_identity")["injured_leg"].nunique()
    if (mismatches != 1).any():
        raise ValueError(f"ACLD/ACLR injured-side mismatch: {mismatches[mismatches.ne(1)].to_dict()}")
    return {
        "acl_side_basis": "actual injured vs contralateral",
        "ha_side_basis": "Right=pseudo-injured, Left=pseudo-contralateral",
        "pair_side_consistency": True,
    }


def _gvs(waveforms: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if reference.shape != (3, 2, 9, 101):
        raise ValueError(f"invalid HA waveform reference shape: {reference.shape}")
    return np.sqrt(np.mean(np.square(waveforms - reference[None, ...]), axis=-1)).reshape(len(waveforms), -1)


def waveform_explanatory_distances(waveforms: np.ndarray, reference: np.ndarray) -> dict[str, np.ndarray]:
    """Raw GPS/GVS subdomains; these never enter the scalar primary total."""
    tensor = np.sqrt(np.mean(np.square(waveforms - reference[None, ...]), axis=-1))
    output = {"gvs_raw_gps": np.sqrt(np.mean(np.square(tensor), axis=(1, 2, 3)))}
    for speed_idx, speed in enumerate(SPEEDS):
        output[f"{speed}_gvs_distance"] = np.sqrt(np.mean(np.square(tensor[:, speed_idx]), axis=(1, 2)))
    for domain, indices in {
        "hip": np.arange(0, 3), "knee": np.arange(3, 6), "ankle": np.arange(6, 9),
    }.items():
        output[f"{domain}_gvs_distance"] = np.sqrt(np.mean(np.square(tensor[:, :, :, indices]), axis=(1, 2, 3)))
    bilateral = waveforms.mean(axis=2) - reference.mean(axis=1)[None, ...]
    output["bilateral_gvs_distance"] = np.sqrt(np.mean(np.square(bilateral), axis=(1, 2, 3)))
    asymmetry = (waveforms[:, :, 0] - waveforms[:, :, 1]) - (reference[:, 0] - reference[:, 1])[None, ...]
    output["asymmetry_gvs_distance"] = np.sqrt(np.mean(np.square(asymmetry), axis=(1, 2, 3)))
    return output


def _loo_speed_calibration(waveforms: np.ndarray, ha_indices: np.ndarray) -> dict[str, tuple[float, float]]:
    values = {speed: [] for speed in SPEEDS}
    for held_out in ha_indices:
        keep = ha_indices[ha_indices != held_out]
        reference = waveforms[keep].mean(axis=0)
        distances = waveform_explanatory_distances(waveforms[[held_out]], reference)
        for speed in SPEEDS:
            values[speed].append(float(distances[f"{speed}_gvs_distance"][0]))
    calibration = {}
    for speed, raw in values.items():
        logged = np.log(np.maximum(np.asarray(raw), EPSILON))
        center = float(np.median(logged))
        scale = float(1.4826 * np.median(np.abs(logged - center)))
        if not np.isfinite(scale) or scale < 1e-8:
            raise ValueError(f"HA LOO speed calibration is degenerate: {speed}")
        calibration[speed] = (center, scale)
    return calibration


def profile_blocks(
    features: SessionFeatures,
    profile: str,
    reference_indices: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, ...]], np.ndarray | None]:
    if profile not in PROFILES:
        raise ValueError(f"unknown input profile: {profile}")
    if len(reference_indices) < 2:
        raise ValueError("at least two training HA identities are required")
    blocks = {"scalar": features.scalar}
    names: dict[str, tuple[str, ...]] = {"scalar": features.scalar_names}
    reference: np.ndarray | None = None
    if profile == "scalar_plus_gvs54":
        reference = features.waveforms[reference_indices].mean(axis=0)
        blocks["gvs54"] = _gvs(features.waveforms, reference)
        names["gvs54"] = tuple(
            f"{speed}__{side}__{joint}__GVS"
            for speed in SPEEDS for side in SIDES for joint in JOINTS
        )
    elif profile == "scalar_plus_waveform5454":
        blocks["waveform5454"] = features.waveforms.reshape(len(features.waveforms), -1)
        names["waveform5454"] = tuple(
            f"{speed}__{side}__{joint}__{point:03d}"
            for speed in SPEEDS for side in SIDES for joint in JOINTS for point in range(101)
        )
    actual = sum(block.shape[1] for block in blocks.values())
    if actual != profile_feature_counts()[profile]:
        raise RuntimeError(f"profile dimension mismatch for {profile}: {actual}")
    return blocks, names, reference


@dataclass
class BlockNormativeModel:
    n_components: int
    scale_method: str
    block_state_: dict[str, dict] | None = None
    center_: np.ndarray | None = None
    components_: np.ndarray | None = None
    covariance_: np.ndarray | None = None
    precision_: np.ndarray | None = None
    feature_precision_: np.ndarray | None = None
    active_names_: tuple[str, ...] = ()
    rank_: int = 0
    condition_: float = math.nan

    def _fit_block(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        if self.scale_method == "robust":
            location = np.median(x, axis=0)
            scale = 1.4826 * np.median(np.abs(x - location), axis=0)
        elif self.scale_method == "standard":
            location = x.mean(axis=0)
            scale = x.std(axis=0, ddof=1)
        else:
            raise ValueError(f"unknown scale method: {self.scale_method}")
        active = np.isfinite(scale) & (scale > 1e-8)
        if not active.any():
            raise ValueError("training HA block has no variable features")
        factor = math.sqrt(int(active.sum()))
        transformed = (x[:, active] - location[active]) / scale[active] / factor
        return transformed, {"location": location, "scale": scale, "active": active, "factor": factor}

    def fit(self, blocks: dict[str, np.ndarray], names: dict[str, tuple[str, ...]], reference_indices: np.ndarray) -> "BlockNormativeModel":
        if len(reference_indices) < 2:
            raise ValueError("normative model requires at least two HA reference rows")
        self.block_state_ = {}
        transformed = []
        active_names: list[str] = []
        for block_name, matrix in blocks.items():
            z, state = self._fit_block(np.asarray(matrix[reference_indices], dtype=np.float64))
            self.block_state_[block_name] = state
            transformed.append(z)
            active_names.extend(np.asarray(names[block_name], dtype=object)[state["active"]].tolist())
        reference = np.concatenate(transformed, axis=1)
        self.center_ = reference.mean(axis=0)
        centered = reference - self.center_
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        rank = int(np.sum(singular > 1e-10))
        k = min(max(1, self.n_components), max(1, rank), len(reference_indices) - 1)
        self.components_ = vt[:k]
        pc = centered @ self.components_.T
        covariance = LedoitWolf(assume_centered=True).fit(pc)
        self.covariance_ = covariance.covariance_
        self.precision_ = covariance.precision_
        self.feature_precision_ = self.components_.T @ self.precision_ @ self.components_
        self.active_names_ = tuple(active_names)
        self.rank_ = int(np.linalg.matrix_rank(self.covariance_))
        self.condition_ = float(np.linalg.cond(self.covariance_))
        if not np.isfinite(self.condition_):
            raise ValueError("non-finite covariance condition number")
        return self

    def transform(self, blocks: dict[str, np.ndarray]) -> np.ndarray:
        if self.block_state_ is None or self.center_ is None:
            raise RuntimeError("model has not been fitted")
        output = []
        for block_name, state in self.block_state_.items():
            matrix = np.asarray(blocks[block_name], dtype=np.float64)
            active = state["active"]
            output.append(
                (matrix[:, active] - state["location"][active])
                / state["scale"][active]
                / state["factor"]
            )
        return np.concatenate(output, axis=1) - self.center_

    def distance(self, blocks: dict[str, np.ndarray]) -> np.ndarray:
        delta = self.transform(blocks)
        squared = np.einsum("ij,jk,ik->i", delta, self.feature_precision_, delta)
        return np.sqrt(np.maximum(squared, 0.0))

    def contributions(self, blocks: dict[str, np.ndarray]) -> np.ndarray:
        delta = self.transform(blocks)
        return delta * (delta @ self.feature_precision_.T)


def _identity_auc(metadata: pd.DataFrame, scores: np.ndarray) -> float:
    y = metadata["group"].ne("HA").astype(int).to_numpy()
    count = metadata.groupby("biological_identity")["subject_id"].transform("nunique").to_numpy()
    weights = 1.0 / count
    if len(np.unique(y)) != 2:
        raise ValueError("AUC requires both ACL and HA identities")
    return float(roc_auc_score(y, scores, sample_weight=weights))


def _index_for_identities(metadata: pd.DataFrame, identities: Iterable[str]) -> np.ndarray:
    return np.flatnonzero(metadata["biological_identity"].isin(set(identities)).to_numpy())


def tune_inner(
    features: SessionFeatures,
    profile: str,
    outer_train_ids: set[str],
    n_splits: int,
    seed: int,
    n_trials: int,
) -> tuple[dict, float]:
    train_meta = features.metadata[features.metadata["biological_identity"].isin(outer_train_ids)]
    inner_splits, _ = V2.make_identity_splits(train_meta, n_splits, seed)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_components": trial.suggest_int("n_components", 1, 12),
            "scale_method": trial.suggest_categorical("scale_method", ["standard", "robust"]),
        }
        rows = []
        for inner_train_ids, inner_val_ids in inner_splits:
            train_idx = _index_for_identities(features.metadata, inner_train_ids)
            val_idx = _index_for_identities(features.metadata, inner_val_ids)
            ha_idx = train_idx[features.metadata.iloc[train_idx]["group"].eq("HA").to_numpy()]
            blocks, names, _ = profile_blocks(features, profile, ha_idx)
            model = BlockNormativeModel(**params).fit(blocks, names, ha_idx)
            scores = model.distance({key: value[val_idx] for key, value in blocks.items()})
            fold = features.metadata.iloc[val_idx].copy()
            fold["score"] = scores
            rows.append(fold)
        oof = pd.concat(rows, ignore_index=True)
        return _identity_auc(oof, oof["score"].to_numpy())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return dict(study.best_params), float(study.best_value)


def _loo_calibration(
    features: SessionFeatures,
    profile: str,
    ha_indices: np.ndarray,
    params: dict,
) -> tuple[float, float, np.ndarray]:
    distances = []
    for held_out in ha_indices:
        keep = ha_indices[ha_indices != held_out]
        blocks, names, _ = profile_blocks(features, profile, keep)
        model = BlockNormativeModel(**params).fit(blocks, names, keep)
        one = {key: value[[held_out]] for key, value in blocks.items()}
        distances.append(float(model.distance(one)[0]))
    distances_array = np.asarray(distances, dtype=np.float64)
    logged = np.log(np.maximum(distances_array, EPSILON))
    center = float(np.median(logged))
    scale = float(1.4826 * np.median(np.abs(logged - center)))
    if not np.isfinite(scale) or scale < 1e-8:
        raise ValueError("HA LOO robust log-distance calibration is degenerate")
    return center, scale, distances_array


def _top_contributions(
    metadata: pd.DataFrame,
    model: BlockNormativeModel,
    blocks: dict[str, np.ndarray],
    indices: np.ndarray,
    repeat: int,
    fold: int,
    profile: str,
    limit: int = 10,
) -> pd.DataFrame:
    subset = {key: value[indices] for key, value in blocks.items()}
    contributions = model.contributions(subset)
    distances = model.distance(subset)
    if not np.allclose(contributions.sum(axis=1), np.square(distances), atol=1e-8, rtol=1e-7):
        raise RuntimeError("feature contributions do not sum to D^2")
    rows = []
    for local, source_idx in enumerate(indices):
        ranked = np.argsort(np.abs(contributions[local]))[::-1][:limit]
        for rank, feature_idx in enumerate(ranked, start=1):
            rows.append({
                "profile": profile, "repeat": repeat, "outer_fold": fold,
                "subject_id": metadata.iloc[source_idx]["subject_id"],
                "rank": rank, "feature": model.active_names_[feature_idx],
                "contribution_to_d2": float(contributions[local, feature_idx]),
                "raw_distance_squared": float(distances[local] ** 2),
            })
    return pd.DataFrame(rows)


def run_profile(
    features: SessionFeatures,
    profile: str,
    splits: list[tuple[int, int, set[str], set[str]]],
    artifact_dir: Path,
    inner_folds: int,
    n_trials: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile_dir = artifact_dir / "profiles" / profile
    (profile_dir / "models").mkdir(parents=True, exist_ok=False)
    score_rows = []
    diagnostics = []
    contribution_rows = []
    for repeat, fold, train_ids, val_ids in splits:
        params, inner_auc = tune_inner(
            features, profile, train_ids, inner_folds,
            seed + repeat * 1000 + fold * 17 + PROFILES.index(profile), n_trials,
        )
        train_idx = _index_for_identities(features.metadata, train_ids)
        val_idx = _index_for_identities(features.metadata, val_ids)
        ha_idx = train_idx[features.metadata.iloc[train_idx]["group"].eq("HA").to_numpy()]
        blocks, names, reference = profile_blocks(features, profile, ha_idx)
        model = BlockNormativeModel(**params).fit(blocks, names, ha_idx)
        center, scale, loo = _loo_calibration(features, profile, ha_idx, params)
        raw = model.distance({key: value[val_idx] for key, value in blocks.items()})
        z = (np.log(np.maximum(raw, EPSILON)) - center) / scale
        fold_scores = features.metadata.iloc[val_idx].copy()
        fold_scores.insert(0, "profile", profile)
        fold_scores["repeat"] = repeat
        fold_scores["outer_fold"] = fold
        fold_scores["raw_distance"] = raw
        fold_scores["overall_z_deviation"] = z
        fold_scores["normality_score"] = 100.0 - 10.0 * z
        explanation_reference = features.waveforms[ha_idx].mean(axis=0)
        explanation = waveform_explanatory_distances(features.waveforms[val_idx], explanation_reference)
        for name, values in explanation.items():
            fold_scores[name] = values
        speed_calibration = _loo_speed_calibration(features.waveforms, ha_idx)
        speed_z_columns = []
        for speed in SPEEDS:
            column = f"{speed}_signed_deviation"
            raw_speed = fold_scores[f"{speed}_gvs_distance"].to_numpy()
            speed_center, speed_scale = speed_calibration[speed]
            fold_scores[column] = (np.log(np.maximum(raw_speed, EPSILON)) - speed_center) / speed_scale
            speed_z_columns.append(column)
        fold_scores["legacy_speed_rms"] = np.sqrt(
            np.mean(np.square(fold_scores[speed_z_columns].to_numpy()), axis=1)
        )
        score_rows.append(fold_scores)
        diagnostics.append({
            "profile": profile, "repeat": repeat, "outer_fold": fold,
            "inner_best_auc": inner_auc, **params,
            "n_train_identities": len(train_ids), "n_validation_identities": len(val_ids),
            "n_training_ha": len(ha_idx), "active_features": len(model.active_names_),
            "covariance_rank": model.rank_, "condition_number": model.condition_,
            "ha_loo_log_center": center, "ha_loo_log_scale": scale,
            "ha_loo_distance_min": float(loo.min()), "ha_loo_distance_max": float(loo.max()),
            "explanation_reference": "outer-training HA mean waveform",
        })
        contribution_rows.append(_top_contributions(
            features.metadata, model, blocks, val_idx, repeat, fold, profile,
        ))
        with (profile_dir / "models" / f"repeat{repeat}_fold{fold}.pkl").open("wb") as handle:
            pickle.dump({
                "model": model, "params": params, "profile": profile,
                "ha_reference_waveform": explanation_reference,
                "ha_log_center": center, "ha_log_scale": scale,
                "scalar_schema": features.scalar_names,
            }, handle)
    scores = pd.concat(score_rows, ignore_index=True)
    diagnostics_frame = pd.DataFrame(diagnostics)
    scores.to_parquet(profile_dir / "oof_session_scores.parquet", index=False)
    diagnostics_frame.to_csv(profile_dir / "fold_diagnostics.csv", index=False)
    pd.concat(contribution_rows, ignore_index=True).to_parquet(
        profile_dir / "top_feature_contributions.parquet", index=False,
    )
    return scores, diagnostics_frame


def _average_repeats(scores: pd.DataFrame) -> pd.DataFrame:
    keys = ["profile", "subject_id", "biological_identity", "group"]
    columns = [
        "raw_distance", "overall_z_deviation", "normality_score", "gvs_raw_gps",
        *(f"{speed}_gvs_distance" for speed in SPEEDS),
        *(f"{domain}_gvs_distance" for domain in ("hip", "knee", "ankle", "bilateral", "asymmetry")),
        *(f"{speed}_signed_deviation" for speed in SPEEDS),
        "legacy_speed_rms",
    ]
    return scores.groupby(keys, as_index=False, observed=True)[columns].mean()


def _stratified_identity_bootstrap_auc(frame: pd.DataFrame, value: str, n_bootstrap: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    acl = frame.loc[frame["group"].ne("HA"), "biological_identity"].unique()
    ha = frame.loc[frame["group"].eq("HA"), "biological_identity"].unique()
    estimates = []
    for _ in range(n_bootstrap):
        pieces = []
        for draw, identity in enumerate(np.r_[rng.choice(acl, len(acl), replace=True), rng.choice(ha, len(ha), replace=True)]):
            piece = frame[frame["biological_identity"].eq(identity)].copy()
            piece["bootstrap_identity"] = f"{draw}::{identity}"
            pieces.append(piece)
        sampled = pd.concat(pieces, ignore_index=True)
        y = sampled["group"].ne("HA").astype(int).to_numpy()
        count = sampled.groupby("bootstrap_identity")["subject_id"].transform("nunique").to_numpy()
        estimates.append(float(roc_auc_score(y, sampled[value], sample_weight=1.0 / count)))
    return np.asarray(estimates)


def _hedges_g(frame: pd.DataFrame, value: str) -> float:
    working = frame.assign(identity_class=np.where(frame["group"].eq("HA"), "HA", "ACL"))
    identity = working.groupby(["biological_identity", "identity_class"], as_index=False)[value].mean()
    acl = identity.loc[identity["identity_class"].eq("ACL"), value].to_numpy()
    ha = identity.loc[identity["identity_class"].eq("HA"), value].to_numpy()
    pooled = math.sqrt(((len(acl)-1)*acl.var(ddof=1) + (len(ha)-1)*ha.var(ddof=1)) / (len(acl)+len(ha)-2))
    if pooled <= 0:
        return math.nan
    d = (acl.mean() - ha.mean()) / pooled
    correction = 1.0 - 3.0 / (4.0 * (len(acl) + len(ha)) - 9.0)
    return float(correction * d)


def _paired_bootstrap_auc_delta(
    primary: pd.DataFrame,
    comparator: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    keys = ["subject_id", "biological_identity", "group"]
    merged = comparator[keys + ["overall_z_deviation"]].merge(
        primary[keys + ["overall_z_deviation"]], on=keys, validate="one_to_one",
        suffixes=("_comparator", "_primary"),
    )
    rng = np.random.default_rng(seed)
    acl = merged.loc[merged["group"].ne("HA"), "biological_identity"].unique()
    ha = merged.loc[merged["group"].eq("HA"), "biological_identity"].unique()
    estimates = []
    for _ in range(n_bootstrap):
        pieces = []
        draws = np.r_[rng.choice(acl, len(acl), replace=True), rng.choice(ha, len(ha), replace=True)]
        for draw, identity in enumerate(draws):
            piece = merged[merged["biological_identity"].eq(identity)].copy()
            piece["bootstrap_identity"] = f"{draw}::{identity}"
            pieces.append(piece)
        sampled = pd.concat(pieces, ignore_index=True)
        y = sampled["group"].ne("HA").astype(int).to_numpy()
        count = sampled.groupby("bootstrap_identity")["subject_id"].transform("nunique").to_numpy()
        weight = 1.0 / count
        comparator_auc = roc_auc_score(y, sampled["overall_z_deviation_comparator"], sample_weight=weight)
        primary_auc = roc_auc_score(y, sampled["overall_z_deviation_primary"], sample_weight=weight)
        estimates.append(float(comparator_auc - primary_auc))
    return np.asarray(estimates)


def summarize(scores_by_profile: dict[str, pd.DataFrame], artifact_dir: Path, n_bootstrap: int, seed: int) -> pd.DataFrame:
    averaged = {profile: _average_repeats(scores) for profile, scores in scores_by_profile.items()}
    rows = []
    bootstraps: dict[str, np.ndarray] = {}
    for profile, frame in averaged.items():
        auc = _identity_auc(frame, frame["overall_z_deviation"].to_numpy())
        boot = _stratified_identity_bootstrap_auc(frame, "overall_z_deviation", n_bootstrap, seed + PROFILES.index(profile))
        bootstraps[profile] = boot
        rows.extend([
            {"profile": profile, "estimate_type": "identity_auc", "estimate": auc,
             "ci_low": float(np.quantile(boot, .025)), "ci_high": float(np.quantile(boot, .975))},
            {"profile": profile, "estimate_type": "hedges_g_acl_vs_ha", "estimate": _hedges_g(frame, "overall_z_deviation"),
             "ci_low": math.nan, "ci_high": math.nan},
        ])
        paired = frame[frame["group"].isin(["ACLD", "ACLR"])].pivot(
            index="biological_identity", columns="group", values="overall_z_deviation"
        ).dropna()
        diff = paired["ACLR"] - paired["ACLD"]
        rng = np.random.default_rng(seed + 100 + PROFILES.index(profile))
        paired_boot = np.asarray([rng.choice(diff.to_numpy(), len(diff), replace=True).mean() for _ in range(n_bootstrap)])
        rows.append({
            "profile": profile, "estimate_type": "paired_aclr_minus_acld_z", "estimate": float(diff.mean()),
            "ci_low": float(np.quantile(paired_boot, .025)), "ci_high": float(np.quantile(paired_boot, .975)),
        })
    if PRIMARY_PROFILE in averaged:
        primary = averaged[PRIMARY_PROFILE][["subject_id", "overall_z_deviation"]].rename(
            columns={"overall_z_deviation": "primary_z"}
        )
        for profile in averaged:
            if profile == PRIMARY_PROFILE:
                continue
            comparator = averaged[profile].merge(primary, on="subject_id", validate="one_to_one")
            auc_delta = _identity_auc(comparator, comparator["overall_z_deviation"].to_numpy()) - _identity_auc(
                comparator, comparator["primary_z"].to_numpy()
            )
            delta_boot = _paired_bootstrap_auc_delta(
                averaged[PRIMARY_PROFILE], averaged[profile], n_bootstrap,
                seed + 500 + PROFILES.index(profile),
            )
            rows.append({
                "profile": profile, "estimate_type": "auc_delta_vs_scalar_primary", "estimate": auc_delta,
                "ci_low": float(np.quantile(delta_boot, .025)), "ci_high": float(np.quantile(delta_boot, .975)),
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(artifact_dir / "summary_metrics.csv", index=False)
    pd.concat(averaged.values(), ignore_index=True).to_parquet(
        artifact_dir / "oof_session_scores_averaged.parquet", index=False,
    )
    return summary


def _select_dry(frame: pd.DataFrame, seed: int, per_class: int) -> pd.DataFrame:
    if per_class < 12:
        raise ValueError("nested dry run requires at least 12 identities per class")
    identities = frame[["biological_identity", "group"]].drop_duplicates()
    acl = sorted(identities.loc[identities["group"].ne("HA"), "biological_identity"].unique())
    ha = sorted(identities.loc[identities["group"].eq("HA"), "biological_identity"].unique())
    if per_class > min(len(acl), len(ha)):
        raise ValueError("dry identity count exceeds available cohort")
    rng = np.random.default_rng(seed)
    keep = set(rng.choice(acl, per_class, replace=False)) | set(rng.choice(ha, per_class, replace=False))
    return frame[frame["biological_identity"].isin(keep)].copy()


def run(
    artifact_dir: Path,
    input_profiles: tuple[str, ...],
    mode: str = "full",
    seed: int = 42,
    outer_folds: int = 5,
    inner_folds: int = 3,
    cv_repeats: int = 5,
    n_trials: int = 20,
    n_bootstrap: int = 2000,
    dry_identities_per_class: int = 12,
) -> dict:
    unknown = sorted(set(input_profiles) - set(PROFILES))
    if unknown:
        raise ValueError(f"unknown profiles: {unknown}")
    artifact_dir.mkdir(parents=True, exist_ok=False)
    cycles, wave_cols, provenance = V2.validate_and_load_cycles(artifact_dir / "qc_audit.json")
    side_contract = validate_side_contract(cycles)
    full_counts = cycles.groupby("group")["subject_id"].nunique().to_dict()
    if mode == "dry":
        cycles = _select_dry(cycles, seed, dry_identities_per_class)
    features = build_session_features(cycles, wave_cols)
    schema = pd.DataFrame({"feature": features.scalar_names, "index": range(len(features.scalar_names))})
    schema.to_csv(artifact_dir / "scalar_feature_schema.csv", index=False)

    all_splits: list[tuple[int, int, set[str], set[str]]] = []
    split_tables = []
    for repeat in range(cv_repeats):
        repeat_splits, table = V2.make_identity_splits(features.metadata, outer_folds, seed + repeat)
        table.insert(0, "repeat", repeat)
        split_tables.append(table)
        all_splits.extend((repeat, fold, train, validation) for fold, (train, validation) in enumerate(repeat_splits))
    split_table = pd.concat(split_tables, ignore_index=True)
    split_table.to_csv(artifact_dir / "outer_splits.csv", index=False)
    split_hash = hashlib.sha256(split_table.to_csv(index=False).encode()).hexdigest()

    scores_by_profile = {}
    diagnostics_by_profile = {}
    for profile in input_profiles:
        scores, diagnostics = run_profile(
            features, profile, all_splits, artifact_dir, inner_folds, n_trials, seed,
        )
        scores_by_profile[profile] = scores
        diagnostics_by_profile[profile] = diagnostics

    sensitivity_profile = PRIMARY_PROFILE if PRIMARY_PROFILE in input_profiles else input_profiles[0]
    swapped = swap_ha_side_orientation(features)
    sensitivity_root = artifact_dir / "sensitivity_ha_side_swap"
    sensitivity_root.mkdir()
    swapped_scores, swapped_diagnostics = run_profile(
        swapped, sensitivity_profile, all_splits, sensitivity_root,
        inner_folds, n_trials, seed + 10000,
    )
    primary_average = _average_repeats(scores_by_profile[sensitivity_profile])
    swapped_average = _average_repeats(swapped_scores)
    side_comparison = primary_average[[
        "subject_id", "biological_identity", "group", "overall_z_deviation",
    ]].merge(
        swapped_average[["subject_id", "overall_z_deviation"]],
        on="subject_id", validate="one_to_one", suffixes=("_right_pseudo", "_left_pseudo"),
    )
    side_comparison["z_change_left_minus_right"] = (
        side_comparison["overall_z_deviation_left_pseudo"]
        - side_comparison["overall_z_deviation_right_pseudo"]
    )
    side_comparison.to_csv(artifact_dir / "ha_side_swap_session_sensitivity.csv", index=False)
    side_summary = pd.DataFrame([{
        "right_pseudo_identity_auc": _identity_auc(primary_average, primary_average["overall_z_deviation"].to_numpy()),
        "left_pseudo_identity_auc": _identity_auc(swapped_average, swapped_average["overall_z_deviation"].to_numpy()),
        "auc_change_left_minus_right": (
            _identity_auc(swapped_average, swapped_average["overall_z_deviation"].to_numpy())
            - _identity_auc(primary_average, primary_average["overall_z_deviation"].to_numpy())
        ),
        "mean_absolute_session_z_change": float(side_comparison["z_change_left_minus_right"].abs().mean()),
        "max_absolute_session_z_change": float(side_comparison["z_change_left_minus_right"].abs().max()),
    }])
    side_summary.to_csv(artifact_dir / "ha_side_swap_summary.csv", index=False)
    combined = pd.concat(scores_by_profile.values(), ignore_index=True)
    combined.to_parquet(artifact_dir / "oof_session_scores.parquet", index=False)
    summary = summarize(scores_by_profile, artifact_dir, n_bootstrap, seed)

    manifest = {
        "pipeline_version": "3.1",
        "artifact_schema_version": "3.1.0",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": mode,
        "balance_mode": "mean_aggregate",
        "primary_profile": PRIMARY_PROFILE,
        "input_profiles": list(input_profiles),
        "seed": seed,
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "cv_repeats": cv_repeats,
        "n_trials_per_study": n_trials,
        "n_bootstrap": n_bootstrap,
        "dry_identities_per_class": dry_identities_per_class if mode == "dry" else None,
        "full_cohort_counts": full_counts,
        "analysis_sessions": int(features.metadata["subject_id"].nunique()),
        "analysis_identities": int(features.metadata["biological_identity"].nunique()),
        "analysis_identity_class_counts": {
            "ACL": int(features.metadata.loc[features.metadata["group"].ne("HA"), "biological_identity"].nunique()),
            "HA": int(features.metadata.loc[features.metadata["group"].eq("HA"), "biological_identity"].nunique()),
        },
        "feature_counts": profile_feature_counts(),
        "configuration_sha256": configuration_hashes(input_profiles),
        "waveform_schema_sha256": hashlib.sha256("\n".join(wave_cols).encode()).hexdigest(),
        "scalar_schema_sha256": hashlib.sha256("\n".join(features.scalar_names).encode()).hexdigest(),
        "split_sha256": split_hash,
        "side_contract": side_contract,
        "ha_side_swap_sensitivity": True,
        "ha_side_swap_profile": sensitivity_profile,
        "engine_source_sha256": sha256(Path(__file__)),
        **provenance,
    }
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return {"manifest": manifest, "summary": summary, "scores": combined, "diagnostics": diagnostics_by_profile}
