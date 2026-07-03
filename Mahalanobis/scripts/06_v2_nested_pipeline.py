"""Pair-safe nested-CV Mahalanobis gait-deviation pipeline.

The legacy 02-04 scripts are intentionally not imported.  This module owns the
complete lineage from a validated cycle table to locked OOF scores and exact
quadratic-form contributions.
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.covariance import MinCovDet

ROOT = Path(__file__).resolve().parents[2]
SANDBOX = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
PAIRING = DATA / "id_pairing_summary.csv"
ID_CSV = ROOT / "data" / "ID.csv"
SLIM = DATA / "slim_gait.parquet"
CYCLES = DATA / "cycle_waveforms_101.parquet"

SPEEDS = ("slow", "normal", "fast")
SIDES = ("injured", "contralateral")
JOINTS = (
    "hip_adduction", "hip_int_rotation", "hip_flexion",
    "knee_adduction", "knee_int_rotation", "knee_flexion",
    "ankle_adduction", "ankle_int_rotation", "ankle_dorsiflexion",
)
META = {
    "subject_id", "biological_identity", "group", "speed", "trial_id",
    "actual_leg", "side_basis", "injured_leg", "cycle_idx",
    "cycle_idx_original", "cycle_len", "n_cycles_raw", "n_cycles_trimmed",
    "cycle_weight",
}


def waveform_columns(frame: pd.DataFrame) -> list[str]:
    cols = [c for c in frame.columns if any(c.startswith(f"{j}_") for j in JOINTS)]
    expected = {f"{j}_{i:03d}" for j in JOINTS for i in range(101)}
    missing = sorted(expected - set(cols))
    extra = sorted(set(cols) - expected)
    if missing or extra:
        raise ValueError(f"waveform schema mismatch: missing={missing[:5]}, extra={extra[:5]}")
    return sorted(cols, key=lambda c: (JOINTS.index(c.rsplit("_", 1)[0]), int(c.rsplit("_", 1)[1])))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_qc(path: Path, checks: list[dict], status: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": status, "checks": checks}, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_and_load_cycles(qc_path: Path) -> tuple[pd.DataFrame, list[str], dict]:
    """Fail-fast cohort and metadata gate; always writes a machine-readable audit."""
    checks: list[dict] = []
    try:
        for name, path in (("slim", SLIM), ("cycles", CYCLES), ("pairing", PAIRING), ("id_csv", ID_CSV)):
            ok = path.exists()
            checks.append({"check": f"{name}_exists", "ok": ok, "value": str(path)})
            if not ok:
                raise FileNotFoundError(path)

        pairing = pd.read_csv(PAIRING)
        paired = pairing.loc[pairing["pair_status"].eq("paired")].copy()
        if len(paired) != 27 or paired["ID_ACLD"].nunique() != 27 or paired["ID_ACLR"].nunique() != 27:
            raise ValueError("pairing table must contain exactly 27 unique confirmed pairs")
        pair_map: dict[str, str] = {}
        for row in paired.itertuples(index=False):
            identity = f"ACL::{int(row.num):02d}"
            pair_map[str(row.ID_ACLD)] = identity
            pair_map[str(row.ID_ACLR)] = identity

        ids = pd.read_csv(ID_CSV)
        if ids["ID"].astype(str).duplicated().any():
            raise ValueError("duplicate IDs in data/ID.csv")
        leg_map = ids.set_index(ids["ID"].astype(str))["Injured leg"].to_dict()
        bad_legs = {s: leg_map.get(s) for s in pair_map if leg_map.get(s) not in ("Left", "Right")}
        if bad_legs:
            raise ValueError(f"missing/invalid injured leg metadata: {bad_legs}")

        slim_meta = pd.read_parquet(SLIM, columns=["subject_id", "group", "speed", "file_name"])
        slim_sessions = slim_meta[["subject_id", "group"]].drop_duplicates()
        expected_acl = set(pair_map)
        actual_acl = set(slim_sessions.loc[slim_sessions["group"].isin(["ACLD", "ACLR"]), "subject_id"].astype(str))
        if actual_acl != expected_acl:
            raise ValueError(f"slim paired cohort mismatch: missing={sorted(expected_acl-actual_acl)}, extra={sorted(actual_acl-expected_acl)}")

        frame = pd.read_parquet(CYCLES)
        frame["subject_id"] = frame["subject_id"].astype(str).replace({"ACLR38": "ACLR36"})
        frame["group"] = frame["group"].astype(str)
        frame["speed"] = frame["speed"].astype(str)
        frame["trial_id"] = frame["trial_id"].astype(str)
        frame["side_basis"] = frame["side_basis"].astype(str)
        frame["biological_identity"] = frame["subject_id"].map(pair_map)
        is_ha = frame["group"].eq("HA")
        frame.loc[is_ha, "biological_identity"] = "HA::" + frame.loc[is_ha, "subject_id"]
        if frame["biological_identity"].isna().any():
            bad = sorted(frame.loc[frame["biological_identity"].isna(), "subject_id"].unique())
            raise ValueError(f"unmatched cycle subjects: {bad}")
        slim_keys = set(map(tuple, slim_meta[["subject_id", "speed", "file_name"]].astype(str).drop_duplicates().to_numpy()))
        cycle_keys = set(map(tuple, frame[["subject_id", "speed", "trial_id"]].astype(str).drop_duplicates().to_numpy()))
        if cycle_keys - slim_keys:
            raise ValueError(f"cycle table contains trials absent from slim_gait: {sorted(cycle_keys-slim_keys)[:5]}")
        skipped_short_trials = sorted(slim_keys - cycle_keys)
        acl_rows = frame[frame["group"].isin(["ACLD", "ACLR"])]
        expected_legs = acl_rows["subject_id"].map(leg_map)
        if "injured_leg" not in frame or not acl_rows["injured_leg"].astype(str).eq(expected_legs.astype(str)).all():
            raise ValueError("cycle injured-leg metadata disagrees with data/ID.csv")

        wave_cols = waveform_columns(frame)
        required = {"subject_id", "group", "speed", "trial_id", "side_basis", "cycle_idx", *wave_cols}
        missing_cols = sorted(required - set(frame.columns))
        if missing_cols:
            raise ValueError(f"required cycle columns missing: {missing_cols[:10]}")
        if not np.isfinite(frame[wave_cols].to_numpy(dtype=np.float64)).all():
            raise ValueError("non-finite joint-angle waveform values")
        if set(frame["speed"].unique()) != set(SPEEDS) or set(frame["side_basis"].unique()) != set(SIDES):
            raise ValueError("unexpected speed or side values")

        sessions = frame[["subject_id", "group", "biological_identity"]].drop_duplicates()
        counts = sessions.groupby("group")["subject_id"].nunique().to_dict()
        identities = sessions["biological_identity"].nunique()
        speed_cells = frame.groupby("subject_id")["speed"].nunique()
        if counts != {"ACLD": 27, "ACLR": 27, "HA": 25} or identities != 52:
            raise ValueError(f"cohort invariant failed: counts={counts}, identities={identities}")
        if not speed_cells.eq(3).all():
            raise ValueError(f"sessions without all three speeds: {speed_cells[speed_cells.ne(3)].to_dict()}")
        side_cells = frame.groupby(["subject_id", "speed"])["side_basis"].nunique()
        if not side_cells.eq(2).all():
            raise ValueError(f"session-speed cells without both sides: {side_cells[side_cells.ne(2)].to_dict()}")

        trial_cells = frame.groupby(["subject_id", "speed"])["trial_id"].nunique()
        if (trial_cells < 1).any():
            raise ValueError("a session-speed cell has no valid trial")
        checks.extend([
            {"check": "paired_alias", "ok": True, "value": "ACLR38 -> ACLR36"},
            {"check": "cohort", "ok": True, "value": counts},
            {"check": "identities", "ok": True, "value": identities},
            {"check": "session_speed_cells", "ok": True, "value": int(len(speed_cells))},
            {"check": "trial_snapshot", "ok": True, "value": int(frame[["subject_id", "speed", "trial_id"]].drop_duplicates().shape[0])},
            {"check": "slim_cycle_trial_lineage", "ok": True, "value": {"valid_trials": len(cycle_keys), "skipped_no_valid_cycles": len(skipped_short_trials)}},
            {"check": "skipped_trial_keys", "ok": True, "value": skipped_short_trials},
            {"check": "cycle_rows", "ok": True, "value": int(len(frame))},
            {"check": "joint_features", "ok": True, "value": len(wave_cols)},
        ])
        _write_qc(qc_path, checks, "pass")
        provenance = {"slim_sha256": sha256(SLIM), "cycles_sha256": sha256(CYCLES), "pairing_sha256": sha256(PAIRING)}
        return frame, wave_cols, provenance
    except Exception as exc:
        checks.append({"check": "fatal", "ok": False, "value": f"{type(exc).__name__}: {exc}"})
        _write_qc(qc_path, checks, "fail")
        raise


def assign_cycle_weights(frame: pd.DataFrame) -> pd.DataFrame:
    """Equal cycle->trial->side->speed->session hierarchy."""
    out = frame.copy()
    cycle_n = out.groupby(["subject_id", "speed", "side_basis", "trial_id"])["cycle_idx"].transform("count")
    trial_n = out.groupby(["subject_id", "speed", "side_basis"])["trial_id"].transform("nunique")
    side_n = out.groupby(["subject_id", "speed"])["side_basis"].transform("nunique")
    speed_n = out.groupby("subject_id")["speed"].transform("nunique")
    out["cycle_weight"] = 1.0 / (cycle_n * trial_n * side_n * speed_n)
    totals = out.groupby("subject_id")["cycle_weight"].sum()
    if not np.allclose(totals.to_numpy(), 1.0, atol=1e-12):
        raise RuntimeError(f"hierarchical cycle weights do not sum to one: {totals.to_dict()}")
    return out


def mean_aggregate(frame: pd.DataFrame, wave_cols: list[str]) -> pd.DataFrame:
    """Cycle mean within trial, then equal trial mean for each session-speed-side."""
    trial_keys = ["subject_id", "biological_identity", "group", "speed", "side_basis", "trial_id"]
    condition_keys = trial_keys[:-1]
    trial_means = frame.groupby(trial_keys, observed=True, sort=False)[wave_cols].mean()
    condition = trial_means.groupby(condition_keys, observed=True, sort=False).mean().reset_index()
    condition["cycle_weight"] = 0.5
    return condition


def concatenate_mean_speeds(condition: pd.DataFrame, wave_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """One trial-balanced slow|normal|fast, injured|contralateral block per session."""
    index = ["subject_id", "biological_identity", "group"]
    blocks = []
    concat_cols = []
    for speed in SPEEDS:
        for side in SIDES:
            block = condition[condition["speed"].eq(speed) & condition["side_basis"].eq(side)][index + wave_cols].copy()
            renamed = {column: f"{speed}__{side}__{column}" for column in wave_cols}
            block = block.rename(columns=renamed).set_index(index)
            blocks.append(block)
            concat_cols.extend(renamed.values())
    joined = pd.concat(blocks, axis=1, join="inner").reset_index()
    expected = condition["subject_id"].nunique()
    if len(joined) != expected or joined[concat_cols].isna().any().any():
        raise ValueError(f"incomplete all-speed concatenation: expected={expected}, actual={len(joined)}")
    joined["speed"] = "all_concat"
    joined["cycle_weight"] = 1.0
    return joined, concat_cols


def make_identity_splits(session_meta: pd.DataFrame, n_splits: int, seed: int) -> tuple[list[tuple[set[str], set[str]]], pd.DataFrame]:
    identities = session_meta[["biological_identity", "group"]].drop_duplicates()
    labels = identities.groupby("biological_identity")["group"].apply(lambda x: "HA" if set(x) == {"HA"} else "ACL")
    identity_ids = labels.index.to_numpy()
    y = labels.to_numpy()
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits: list[tuple[set[str], set[str]]] = []
    rows = []
    for fold, (tr, va) in enumerate(splitter.split(identity_ids, y)):
        train_ids, val_ids = set(identity_ids[tr]), set(identity_ids[va])
        if train_ids & val_ids:
            raise RuntimeError("biological identity leakage")
        splits.append((train_ids, val_ids))
        rows.extend({"outer_fold": fold, "biological_identity": identity, "partition": "validation"} for identity in sorted(val_ids))
    table = pd.DataFrame(rows)
    if table["biological_identity"].nunique() != len(identity_ids) or table.duplicated("biological_identity").any():
        raise RuntimeError("each identity must appear in exactly one outer validation fold")
    return splits, table


@dataclass
class WeightedMahalanobis:
    scale: str
    n_components: int
    shrinkage: float
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    pca_center_: np.ndarray | None = None
    components_: np.ndarray | None = None
    covariance_: np.ndarray | None = None
    precision_: np.ndarray | None = None
    feature_precision_: np.ndarray | None = None
    rank_: int = 0
    condition_: float = math.nan

    def fit(self, x: np.ndarray, weights: np.ndarray) -> "WeightedMahalanobis":
        x = np.asarray(x, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        if len(x) != len(w) or len(x) < 3 or np.any(w <= 0):
            raise ValueError("invalid weighted training data")
        w = w / w.sum()
        self.mean_ = np.sum(x * w[:, None], axis=0)
        variance = np.sum(np.square(x - self.mean_) * w[:, None], axis=0)
        self.scale_ = np.sqrt(np.maximum(variance, 1e-12)) if self.scale == "standard" else np.ones(x.shape[1])
        z = (x - self.mean_) / self.scale_
        self.pca_center_ = np.sum(z * w[:, None], axis=0)
        centered = z - self.pca_center_
        _, _, vt = np.linalg.svd(centered * np.sqrt(w[:, None]), full_matrices=False)
        k = max(1, min(self.n_components, len(vt), x.shape[1]))
        self.components_ = vt[:k]
        pc = centered @ self.components_.T
        covariance = (pc * w[:, None]).T @ pc
        diagonal = np.diag(np.diag(covariance))
        self.covariance_ = (1.0 - self.shrinkage) * covariance + self.shrinkage * diagonal
        eig = np.linalg.eigvalsh(self.covariance_)
        if eig.min() <= 1e-12:
            raise ValueError(f"non-positive covariance eigenvalue: {eig.min()}")
        self.precision_ = np.linalg.inv(self.covariance_)
        self.feature_precision_ = self.components_.T @ self.precision_ @ self.components_
        self.rank_ = int(np.linalg.matrix_rank(self.covariance_))
        self.condition_ = float(np.linalg.cond(self.covariance_))
        return self

    def _delta(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=np.float64) - self.mean_) / self.scale_
        return z - self.pca_center_

    def distance(self, x: np.ndarray) -> np.ndarray:
        delta = self._delta(x)
        quadratic = np.einsum("ij,jk,ik->i", delta, self.feature_precision_, delta)
        return np.sqrt(np.maximum(quadratic, 0.0))

    def contributions(self, x: np.ndarray) -> np.ndarray:
        """Exact additive decomposition of squared Mahalanobis distance."""
        delta = self._delta(x)
        return delta * (delta @ self.feature_precision_)


def _session_distance(rows: pd.DataFrame, distances: np.ndarray) -> pd.DataFrame:
    tmp = rows[["subject_id", "biological_identity", "group", "speed", "cycle_weight"]].copy()
    tmp["distance_sq"] = np.square(distances)
    def aggregate(group: pd.DataFrame) -> float:
        return float(np.sqrt(np.average(group["distance_sq"], weights=group["cycle_weight"])))
    result = tmp.groupby(["subject_id", "biological_identity", "group", "speed"], observed=True).apply(aggregate, include_groups=False)
    return result.rename("raw_distance").reset_index()


def _identity_auc(scores: pd.DataFrame, value: str = "raw_distance") -> float:
    y = scores["group"].ne("HA").astype(int).to_numpy()
    counts = scores.groupby("biological_identity")["subject_id"].transform("nunique").to_numpy()
    weights = 1.0 / counts
    if len(np.unique(y)) < 2:
        return math.nan
    return float(roc_auc_score(y, scores[value].to_numpy(), sample_weight=weights))


def _robust_calibration(train_scores: pd.DataFrame) -> tuple[float, float]:
    values = np.log(np.maximum(train_scores.loc[train_scores["group"].eq("HA"), "raw_distance"].to_numpy(), 1e-12))
    center = float(np.median(values))
    mad = float(1.4826 * np.median(np.abs(values - center)))
    if not np.isfinite(mad) or mad < 1e-8:
        raise ValueError("HA robust log-distance MAD is zero")
    return center, mad


def _fit_score(
    rows_train: pd.DataFrame,
    rows_val: pd.DataFrame,
    wave_cols: list[str],
    params: dict,
    require_calibration: bool = True,
) -> tuple[WeightedMahalanobis, pd.DataFrame, tuple[float, float]]:
    ha = rows_train[rows_train["group"].eq("HA")]
    model = WeightedMahalanobis(**params).fit(ha[wave_cols].to_numpy(), ha["cycle_weight"].to_numpy())
    train_scores = _session_distance(rows_train, model.distance(rows_train[wave_cols].to_numpy()))
    scores = _session_distance(rows_val, model.distance(rows_val[wave_cols].to_numpy()))
    if require_calibration:
        calibration = _robust_calibration(train_scores)
        scores["signed_deviation"] = (np.log(np.maximum(scores["raw_distance"], 1e-12)) - calibration[0]) / calibration[1]
    else:
        calibration = (math.nan, math.nan)
        scores["signed_deviation"] = math.nan
    return model, scores, calibration


def _fit_mcd_sensitivity(
    rows_train: pd.DataFrame,
    rows_val: pd.DataFrame,
    wave_cols: list[str],
    params: dict,
) -> tuple[WeightedMahalanobis, pd.DataFrame]:
    """Trial-balanced MCD comparator; never used to select the primary model."""
    ha = rows_train[rows_train["group"].eq("HA")]
    model = WeightedMahalanobis(**params).fit(ha[wave_cols].to_numpy(), ha["cycle_weight"].to_numpy())
    delta = model._delta(ha[wave_cols].to_numpy())
    pc = delta @ model.components_.T
    mcd = MinCovDet(support_fraction=0.75, random_state=42).fit(pc)
    eig = np.linalg.eigvalsh(mcd.covariance_)
    if eig.min() <= 1e-12:
        raise ValueError("MCD sensitivity covariance is singular")
    model.pca_center_ = model.pca_center_ + mcd.location_ @ model.components_
    model.covariance_ = mcd.covariance_
    model.precision_ = np.linalg.inv(model.covariance_)
    model.feature_precision_ = model.components_.T @ model.precision_ @ model.components_
    model.rank_ = int(np.linalg.matrix_rank(model.covariance_))
    model.condition_ = float(np.linalg.cond(model.covariance_))
    train_scores = _session_distance(rows_train, model.distance(rows_train[wave_cols].to_numpy()))
    center, scale = _robust_calibration(train_scores)
    scores = _session_distance(rows_val, model.distance(rows_val[wave_cols].to_numpy()))
    scores["signed_deviation"] = (np.log(np.maximum(scores["raw_distance"], 1e-12)) - center) / scale
    return model, scores


def tune_inner(
    rows: pd.DataFrame,
    wave_cols: list[str],
    train_ids: set[str],
    n_inner: int,
    seed: int,
    n_trials: int,
    storage: str,
    study_name: str,
) -> tuple[dict, optuna.Study]:
    meta = rows.loc[rows["biological_identity"].isin(train_ids), ["biological_identity", "group"]].drop_duplicates()
    inner_splits, _ = make_identity_splits(meta, n_inner, seed)
    ha_ids = set(meta.loc[meta["group"].eq("HA"), "biological_identity"])
    minimum_inner_ha = min(len(inner_train & ha_ids) for inner_train, _ in inner_splits)
    max_k = max(2, min(12, minimum_inner_ha - 1))
    k_choices = sorted(set([2, min(4, max_k), min(6, max_k), max_k]))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "scale": trial.suggest_categorical("scale", ["standard", "none"]),
            "n_components": trial.suggest_categorical("n_components", k_choices),
            "shrinkage": trial.suggest_categorical("shrinkage", [0.05, 0.2, 0.5, 1.0]),
        }
        aucs = []
        for inner_train, inner_val in inner_splits:
            tr = rows[rows["biological_identity"].isin(inner_train)]
            va = rows[rows["biological_identity"].isin(inner_val)]
            try:
                _, scores, _ = _fit_score(tr, va, wave_cols, params, require_calibration=False)
                aucs.append(_identity_auc(scores))
            except (ValueError, np.linalg.LinAlgError) as exc:
                reason = f"{type(exc).__name__}: {exc}"
                trial.set_user_attr("failure", reason)
                raise optuna.TrialPruned(reason) from exc
        return float(np.nanmean(aucs))

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=False,
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return dict(study.best_params), study


def _top_contributions(
    rows: pd.DataFrame,
    model: WeightedMahalanobis,
    wave_cols: list[str],
    fold: int,
    mode: str,
) -> pd.DataFrame:
    contrib = model.contributions(rows[wave_cols].to_numpy())
    tmp = rows[["subject_id", "biological_identity", "group", "speed", "cycle_weight"]].copy()
    weighted = contrib * tmp["cycle_weight"].to_numpy()[:, None]
    group_keys = ["subject_id", "biological_identity", "group", "speed"]
    out = []
    for key, idx in tmp.groupby(group_keys, observed=True).indices.items():
        indices = np.asarray(idx)
        total_weight = tmp.iloc[indices]["cycle_weight"].sum()
        values = weighted[indices].sum(axis=0) / total_weight
        order = np.argsort(np.abs(values))[::-1][:20]
        for rank, feature_idx in enumerate(order, 1):
            out.append({
                **dict(zip(group_keys, key)), "outer_fold": fold, "balance_mode": mode,
                "rank": rank, "feature": wave_cols[feature_idx], "contribution_d2": float(values[feature_idx]),
            })
    return pd.DataFrame(out)


def run_balance_mode(
    cycles: pd.DataFrame,
    wave_cols: list[str],
    mode: str,
    outer_splits: list[tuple[set[str], set[str]]],
    artifact_dir: Path,
    n_inner: int,
    n_trials: int,
    seed: int,
) -> pd.DataFrame:
    rows = mean_aggregate(cycles, wave_cols) if mode == "mean_aggregate" else cycles.copy()
    mode_dir = artifact_dir / mode
    (mode_dir / "models").mkdir(parents=True, exist_ok=True)
    storage_path = mode_dir / "optuna.db"
    storage = f"sqlite:///{storage_path}"
    all_scores, diagnostics, contributions, mcd_scores = [], [], [], []

    for speed in SPEEDS:
        speed_rows = rows[rows["speed"].eq(speed)].copy()
        for fold, (train_ids, val_ids) in enumerate(outer_splits):
            params, study = tune_inner(
                speed_rows, wave_cols, train_ids, n_inner, seed + fold,
                n_trials, storage, f"{mode}_{speed}_outer{fold}",
            )
            train = speed_rows[speed_rows["biological_identity"].isin(train_ids)]
            val = speed_rows[speed_rows["biological_identity"].isin(val_ids)]
            model, scores, calibration = _fit_score(train, val, wave_cols, params)
            scores["outer_fold"] = fold
            scores["balance_mode"] = mode
            all_scores.append(scores)
            diagnostics.append({
                "balance_mode": mode, "speed": speed, "outer_fold": fold,
                "inner_best_auc": float(study.best_value), **params,
                "covariance_rank": model.rank_, "condition_number": model.condition_,
                "ha_log_center": calibration[0], "ha_log_scale": calibration[1],
                "train_identities": len(train_ids), "validation_identities": len(val_ids),
            })
            contributions.append(_top_contributions(val, model, wave_cols, fold, mode))
            with (mode_dir / "models" / f"{speed}_outer{fold}.pkl").open("wb") as handle:
                pickle.dump({"model": model, "params": params, "calibration": calibration, "wave_cols": wave_cols}, handle)
            if mode == "mean_aggregate":
                try:
                    mcd_model, mcd_score = _fit_mcd_sensitivity(train, val, wave_cols, params)
                    mcd_score["outer_fold"] = fold
                    mcd_scores.append(mcd_score)
                    diagnostics.append({
                        "balance_mode": "mean_aggregate_mcd_sensitivity", "speed": speed, "outer_fold": fold,
                        "inner_best_auc": float(study.best_value), **params,
                        "covariance_rank": mcd_model.rank_, "condition_number": mcd_model.condition_,
                        "ha_log_center": math.nan, "ha_log_scale": math.nan,
                        "train_identities": len(train_ids), "validation_identities": len(val_ids),
                    })
                except (ValueError, np.linalg.LinAlgError) as exc:
                    diagnostics.append({
                        "balance_mode": "mean_aggregate_mcd_sensitivity", "speed": speed, "outer_fold": fold,
                        "inner_best_auc": float(study.best_value), **params,
                        "covariance_rank": 0, "condition_number": math.inf,
                        "ha_log_center": math.nan, "ha_log_scale": math.nan,
                        "train_identities": len(train_ids), "validation_identities": len(val_ids),
                        "status": f"failed: {type(exc).__name__}: {exc}",
                    })

    concat_scores = []
    if mode == "mean_aggregate":
        concat_rows, concat_cols = concatenate_mean_speeds(rows, wave_cols)
        for fold, (train_ids, val_ids) in enumerate(outer_splits):
            params, study = tune_inner(
                concat_rows, concat_cols, train_ids, n_inner, seed + 100 + fold,
                n_trials, storage, f"{mode}_all_concat_outer{fold}",
            )
            train = concat_rows[concat_rows["biological_identity"].isin(train_ids)]
            val = concat_rows[concat_rows["biological_identity"].isin(val_ids)]
            model, scores, calibration = _fit_score(train, val, concat_cols, params)
            scores["outer_fold"] = fold
            concat_scores.append(scores[["subject_id", "biological_identity", "group", "outer_fold", "raw_distance", "signed_deviation"]])
            diagnostics.append({
                "balance_mode": mode, "speed": "all_concat", "outer_fold": fold,
                "inner_best_auc": float(study.best_value), **params,
                "covariance_rank": model.rank_, "condition_number": model.condition_,
                "ha_log_center": calibration[0], "ha_log_scale": calibration[1],
                "train_identities": len(train_ids), "validation_identities": len(val_ids),
            })
            contributions.append(_top_contributions(val, model, concat_cols, fold, mode))
            with (mode_dir / "models" / f"all_concat_outer{fold}.pkl").open("wb") as handle:
                pickle.dump({"model": model, "params": params, "calibration": calibration, "wave_cols": concat_cols}, handle)

    long = pd.concat(all_scores, ignore_index=True)
    wide_raw = long.pivot(index=["subject_id", "biological_identity", "group", "outer_fold", "balance_mode"], columns="speed", values="raw_distance")
    wide_z = long.pivot(index=["subject_id", "biological_identity", "group", "outer_fold", "balance_mode"], columns="speed", values="signed_deviation")
    wide = wide_raw.add_suffix("_distance").join(wide_z.add_suffix("_signed_deviation")).reset_index()
    wide["total_distance"] = np.sqrt(np.mean(np.square(wide[[f"{s}_signed_deviation" for s in SPEEDS]].to_numpy()), axis=1))
    if concat_scores:
        concat = pd.concat(concat_scores, ignore_index=True).rename(columns={
            "raw_distance": "all_speed_concat_distance",
            "signed_deviation": "all_speed_concat_signed_deviation",
        })
        wide = wide.merge(concat, on=["subject_id", "biological_identity", "group", "outer_fold"], how="left", validate="one_to_one")
    if mcd_scores:
        mcd_long = pd.concat(mcd_scores, ignore_index=True)
        mcd_z = mcd_long.pivot(index=["subject_id", "biological_identity", "group", "outer_fold"], columns="speed", values="signed_deviation")
        mcd_raw = mcd_long.pivot(index=["subject_id", "biological_identity", "group", "outer_fold"], columns="speed", values="raw_distance")
        mcd_wide = mcd_raw.add_prefix("mcd_").add_suffix("_distance").join(mcd_z.add_prefix("mcd_").add_suffix("_signed_deviation")).reset_index()
        mcd_wide["mcd_total_distance"] = np.sqrt(np.mean(np.square(mcd_wide[[f"mcd_{s}_signed_deviation" for s in SPEEDS]].to_numpy()), axis=1))
        wide = wide.merge(mcd_wide, on=["subject_id", "biological_identity", "group", "outer_fold"], how="left", validate="one_to_one")
        mcd_long.to_parquet(mode_dir / "mcd_sensitivity_speed_scores.parquet", index=False)
    long.to_parquet(mode_dir / "oof_speed_scores.parquet", index=False)
    wide.to_parquet(mode_dir / "oof_session_scores.parquet", index=False)
    pd.DataFrame(diagnostics).to_csv(mode_dir / "fold_diagnostics.csv", index=False)
    pd.concat(contributions, ignore_index=True).to_parquet(mode_dir / "top_contributions.parquet", index=False)
    return wide


def identity_bootstrap_auc(scores: pd.DataFrame, value: str, seed: int, repeats: int = 500) -> tuple[float, float, float]:
    point = _identity_auc(scores, value)
    identities = np.asarray(scores["biological_identity"].unique())
    identity_to_index = {identity: index for index, identity in enumerate(identities)}
    row_identity = scores["biological_identity"].map(identity_to_index).to_numpy()
    rows_per_identity = scores.groupby("biological_identity")["subject_id"].transform("nunique").to_numpy()
    y = scores["group"].ne("HA").astype(int).to_numpy()
    values = scores[value].to_numpy()
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        multiplicity = np.bincount(rng.integers(0, len(identities), len(identities)), minlength=len(identities))
        sample_weight = multiplicity[row_identity] / rows_per_identity
        included = sample_weight > 0
        if np.unique(y[included]).size < 2:
            continue
        estimates.append(float(roc_auc_score(y, values, sample_weight=sample_weight)))
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return point, float(lo), float(hi)


def summarize_results(mode_scores: dict[str, pd.DataFrame], artifact_dir: Path, seed: int) -> pd.DataFrame:
    rows = []
    for mode, scores in mode_scores.items():
        endpoints = [f"{s}_distance" for s in SPEEDS] + ["total_distance"]
        if "all_speed_concat_distance" in scores:
            endpoints.append("all_speed_concat_distance")
        if "mcd_total_distance" in scores:
            endpoints.append("mcd_total_distance")
        for value in endpoints:
            auc, lo, hi = identity_bootstrap_auc(scores, value, seed)
            rows.append({"balance_mode": mode, "endpoint": value, "identity_auc": auc, "ci_low": lo, "ci_high": hi})
        paired = scores[scores["group"].isin(["ACLD", "ACLR"])].pivot(index="biological_identity", columns="group", values="total_distance").dropna()
        diff = paired["ACLR"] - paired["ACLD"]
        rng = np.random.default_rng(seed)
        boot = [float(rng.choice(diff.to_numpy(), size=len(diff), replace=True).mean()) for _ in range(1000)]
        rows.append({
            "balance_mode": mode, "endpoint": "paired_ACLR_minus_ACLD_total",
            "identity_auc": float(diff.mean()), "ci_low": float(np.quantile(boot, .025)), "ci_high": float(np.quantile(boot, .975)),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(artifact_dir / "summary_metrics.csv", index=False)
    influence = []
    for mode, scores in mode_scores.items():
        baseline = _identity_auc(scores, "total_distance")
        for subject in sorted(scores.loc[scores["group"].eq("HA"), "subject_id"].unique()):
            reduced = scores[~scores["subject_id"].eq(subject)]
            influence.append({
                "balance_mode": mode, "excluded_ha": subject,
                "baseline_auc": baseline, "leave_one_ha_auc": _identity_auc(reduced, "total_distance"),
                "auc_change": _identity_auc(reduced, "total_distance") - baseline,
                "excluded_total_distance": float(scores.loc[scores["subject_id"].eq(subject), "total_distance"].iloc[0]),
            })
    pd.DataFrame(influence).to_csv(artifact_dir / "ha_leave_one_out_influence.csv", index=False)
    return summary


def select_dry_cohort(frame: pd.DataFrame, seed: int, n_acl: int = 12, n_ha: int = 12) -> pd.DataFrame:
    """Small but pair-complete cohort selected after the full preflight gate."""
    identities = frame[["biological_identity", "group"]].drop_duplicates()
    acl = sorted(identities.loc[identities["group"].ne("HA"), "biological_identity"].unique())
    ha = sorted(identities.loc[identities["group"].eq("HA"), "biological_identity"].unique())
    rng = np.random.default_rng(seed)
    keep = set(rng.choice(acl, n_acl, replace=False)) | set(rng.choice(ha, n_ha, replace=False))
    return frame[frame["biological_identity"].isin(keep)].copy()


def run(
    artifact_dir: Path,
    balance_mode: str = "both",
    mode: str = "full",
    seed: int = 42,
    outer_folds: int = 5,
    inner_folds: int = 3,
    n_trials: int = 20,
) -> dict:
    artifact_dir.mkdir(parents=True, exist_ok=False)
    cycles, wave_cols, provenance = validate_and_load_cycles(artifact_dir / "qc_audit.json")
    cycles = assign_cycle_weights(cycles)
    full_counts = cycles.groupby("group")["subject_id"].nunique().to_dict()
    if mode == "dry":
        cycles = select_dry_cohort(cycles, seed)
    session_meta = cycles[["subject_id", "biological_identity", "group"]].drop_duplicates()
    outer_splits, split_table = make_identity_splits(session_meta, outer_folds, seed)
    split_table.to_csv(artifact_dir / "outer_splits.csv", index=False)
    modes = ["mean_aggregate", "inverse_weight"] if balance_mode == "both" else [balance_mode]
    mode_scores = {
        name: run_balance_mode(cycles, wave_cols, name, outer_splits, artifact_dir, inner_folds, n_trials, seed)
        for name in modes
    }
    combined = pd.concat(mode_scores.values(), ignore_index=True)
    combined.to_parquet(artifact_dir / "oof_session_scores.parquet", index=False)
    summary = summarize_results(mode_scores, artifact_dir, seed)
    manifest = {
        "pipeline_version": "2.0",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": mode,
        "balance_mode": balance_mode,
        "primary_balance_mode": "inverse_weight",
        "seed": seed,
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "n_trials_per_study": n_trials,
        "full_cohort_counts": full_counts,
        "analysis_sessions": int(session_meta["subject_id"].nunique()),
        "analysis_identities": int(session_meta["biological_identity"].nunique()),
        "waveform_features": len(wave_cols),
        "waveform_schema_sha256": hashlib.sha256("\n".join(wave_cols).encode()).hexdigest(),
        "engine_source_sha256": sha256(Path(__file__)),
        **provenance,
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"manifest": manifest, "summary": summary, "scores": combined}
