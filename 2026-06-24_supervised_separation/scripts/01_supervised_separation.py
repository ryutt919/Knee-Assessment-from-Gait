#!/usr/bin/env python3
"""Supervised embedding experiments for gait group separation."""

from __future__ import annotations

import html
import json
import re
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SANDBOX = Path(__file__).resolve().parents[1]
RESULTS_DIR = SANDBOX / "results"
FIGURES_DIR = SANDBOX / "figures"
HTML_DIR = SANDBOX / "htmls"
LOGS_DIR = SANDBOX / "logs"

RANDOM_STATE = 42
OUTER_FOLDS = 5
MAX_SELECTED_FEATURES = 160
NCA_MAX_ITER = 120

SOURCE_FILES = {
    "source_slim_gait_parquet": PROJECT_ROOT / "data/processed/slim_gait.parquet",
    "source_raw_merged_parquet": PROJECT_ROOT / "data/processed/raw_merged.parquet",
    "source_cycle_waveforms_101_parquet": PROJECT_ROOT / "data/processed/cycle_waveforms_101.parquet",
    "features_scalar": PROJECT_ROOT / "data/processed/features_scalar.csv",
    "pca_slim_subject_features": PROJECT_ROOT / "2026-06-24_PCA/results/slim_gait_subject_features.csv",
    "pca_raw_subject_features": PROJECT_ROOT / "2026-06-24_PCA/results/raw_merged_subject_features.csv",
    "pca_cycle_subject_features": PROJECT_ROOT / "2026-06-24_PCA/results/cycle_waveforms_101_subject_features.csv",
    "pca_summary": PROJECT_ROOT / "2026-06-24_PCA/results/00_pca_summaries.json",
}

TASKS = {
    "binary_acl_vs_ha": {
        "label_order": ["HA", "ACL"],
        "positive_label": "ACL",
        "title": "ACL vs HA",
    },
    "multiclass_3group": {
        "label_order": ["HA", "ACLD", "ACLR"],
        "positive_label": None,
        "title": "ACLD / ACLR / HA",
    },
}

METHODS = ["pca", "lda", "plsda", "nca", "rf_proba"]
METHOD_LABELS = {
    "pca": "PCA baseline",
    "lda": "LDA",
    "plsda": "PLS-DA",
    "nca": "NCA",
    "rf_proba": "RandomForest probability",
}
GROUP_COLORS = {
    "HA": "#2E7D32",
    "ACLD": "#C62828",
    "ACLR": "#1565C0",
    "ACL": "#6A1B9A",
}


class RunLog:
    def __init__(self) -> None:
        self.lines: list[dict[str, str]] = []

    def add(self, message: str) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.lines.append({"time": stamp, "message": message})
        print(f"[{stamp}] {message}", flush=True)

    def write(self, path: Path) -> None:
        path.write_text(
            "\n".join(f"[{row['time']}] {row['message']}" for row in self.lines) + "\n",
            encoding="utf-8",
        )


@dataclass
class PreparedFold:
    x_train: np.ndarray
    x_test: np.ndarray
    selected_features: int


def ensure_dirs() -> None:
    for directory in [RESULTS_DIR, FIGURES_DIR, HTML_DIR, LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def source_stats() -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for name, path in SOURCE_FILES.items():
        st = path.stat()
        stats[name] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "size_bytes": st.st_size,
            "mtime_ns": st.st_mtime_ns,
            "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        }
    return stats


def load_scalar_subject_pivot(log: RunLog) -> pd.DataFrame:
    df = pd.read_csv(SOURCE_FILES["features_scalar"])
    meta = ["subject_id", "group", "speed", "injured_leg"]
    numeric_cols = [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]
    rows: list[dict[str, Any]] = []
    for subject_id, sub in df.groupby("subject_id", sort=True):
        row: dict[str, Any] = {
            "subject_id": str(subject_id),
            "group": str(sub["group"].iloc[0]),
        }
        by_speed = {str(r["speed"]): r for _, r in sub.iterrows()}
        for speed, speed_row in by_speed.items():
            for col in numeric_cols:
                row[f"{speed}__{col}"] = speed_row[col]
        for col in numeric_cols:
            vals = {speed: by_speed[speed][col] for speed in by_speed}
            row[f"mean__{col}"] = np.nanmean(list(vals.values()))
            if "fast" in vals and "slow" in vals:
                row[f"delta_fast_slow__{col}"] = vals["fast"] - vals["slow"]
            if "fast" in vals and "normal" in vals:
                row[f"delta_fast_normal__{col}"] = vals["fast"] - vals["normal"]
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["group", "subject_id"]).reset_index(drop=True)
    log.add(f"Built scalar_subject_pivot: {out.shape[0]} subjects x {out.shape[1] - 2} features")
    return out


def load_pca_subject_features(name: str, path: Path, log: RunLog) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["subject_id"] = df["subject_id"].astype(str)
    df["group"] = df["group"].astype(str)
    feature_cols = [c for c in df.columns if c not in {"subject_id", "group"}]
    renamed = df[["subject_id", "group", *feature_cols]].rename(
        columns={col: f"{name}__{col}" for col in feature_cols}
    )
    log.add(f"Loaded {name}: {renamed.shape[0]} subjects x {renamed.shape[1] - 2} features")
    return renamed


def merge_feature_frames(name: str, frames: list[pd.DataFrame], log: RunLog) -> pd.DataFrame:
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["subject_id", "group"], how="inner")
    merged = merged.sort_values(["group", "subject_id"]).reset_index(drop=True)
    log.add(f"Built {name}: {merged.shape[0]} subjects x {merged.shape[1] - 2} features")
    return merged


def build_feature_sets(log: RunLog) -> dict[str, pd.DataFrame]:
    scalar = load_scalar_subject_pivot(log)
    slim = load_pca_subject_features("slim_gait", SOURCE_FILES["pca_slim_subject_features"], log)
    raw = load_pca_subject_features("raw_merged", SOURCE_FILES["pca_raw_subject_features"], log)
    cycle = load_pca_subject_features("cycle_waveforms_101", SOURCE_FILES["pca_cycle_subject_features"], log)
    feature_sets = {
        "scalar_pivot": scalar,
        "pca_slim_subject": slim,
        "pca_raw_subject": raw,
        "pca_cycle_subject": cycle,
        "fusion_scalar_cycle": merge_feature_frames("fusion_scalar_cycle", [scalar, cycle], log),
        "fusion_scalar_raw": merge_feature_frames("fusion_scalar_raw", [scalar, raw], log),
        "fusion_all_common": merge_feature_frames("fusion_all_common", [scalar, slim, raw, cycle], log),
    }
    inventory = []
    for name, df in feature_sets.items():
        inventory.append(
            {
                "feature_set": name,
                "subjects": len(df),
                "features": df.shape[1] - 2,
                "HA": int((df["group"] == "HA").sum()),
                "ACLD": int((df["group"] == "ACLD").sum()),
                "ACLR": int((df["group"] == "ACLR").sum()),
            }
        )
    pd.DataFrame(inventory).to_csv(RESULTS_DIR / "feature_set_inventory.csv", index=False)
    return feature_sets


def labels_for_task(groups: pd.Series, task: str) -> np.ndarray:
    if task == "binary_acl_vs_ha":
        return np.where(groups.eq("HA"), "HA", "ACL")
    return groups.astype(str).to_numpy()


def biological_identity(subject_id: str) -> str:
    """Map longitudinal ACLD/ACLR sessions to one biological identity."""
    match = re.fullmatch(r"(ACLD|ACLR|HA)(.+)", str(subject_id))
    if match is None:
        return f"UNKNOWN::{subject_id}"
    cohort, suffix = match.groups()
    return f"ACL::{suffix}" if cohort in {"ACLD", "ACLR"} else f"HA::{suffix}"


def pair_aware_splits(subjects: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratify 52 identities while keeping each ACLD/ACLR pair in one fold."""
    identities = np.array([biological_identity(subject) for subject in subjects])
    unique_identities = np.array(sorted(set(identities)))
    identity_labels = np.array(
        ["ACL" if identity.startswith("ACL::") else "HA" for identity in unique_identities]
    )
    splitter = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for identity_train, identity_test in splitter.split(unique_identities, identity_labels):
        train_ids = set(unique_identities[identity_train])
        test_ids = set(unique_identities[identity_test])
        if train_ids & test_ids:
            raise RuntimeError("Pair-aware split leaked a biological identity")
        train_idx = np.flatnonzero(np.isin(identities, list(train_ids)))
        test_idx = np.flatnonzero(np.isin(identities, list(test_ids)))
        splits.append((train_idx, test_idx))
    return splits


def feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    feature_cols = [c for c in df.columns if c not in {"subject_id", "group"}]
    return df[feature_cols].to_numpy(dtype=np.float64, copy=False), feature_cols


def prepare_fold(
    x_train_raw: np.ndarray,
    x_test_raw: np.ndarray,
    y_train: np.ndarray,
    feature_count_limit: int = MAX_SELECTED_FEATURES,
) -> PreparedFold:
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(x_train_raw)
    x_test = imputer.transform(x_test_raw)

    variance = VarianceThreshold(threshold=0.0)
    x_train = variance.fit_transform(x_train)
    x_test = variance.transform(x_test)

    if x_train.shape[1] > feature_count_limit:
        k = min(feature_count_limit, x_train.shape[1])
        selector = SelectKBest(f_classif, k=k)
        # Constant features can reappear inside a fold after imputation and variance filtering.
        # SelectKBest assigns them non-informative scores; suppress the expected sklearn warnings.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                module=r"sklearn\.feature_selection\._univariate_selection",
            )
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                module=r"sklearn\.feature_selection\._univariate_selection",
            )
            x_train = selector.fit_transform(x_train, y_train)
            x_test = selector.transform(x_test)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return PreparedFold(x_train=x_train, x_test=x_test, selected_features=x_train.shape[1])


def fit_transform_embedding(
    method: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    label_encoder = LabelEncoder().fit(y_train)
    y_train_int = label_encoder.transform(y_train)
    labels = label_encoder.classes_.tolist()
    n_classes = len(labels)
    n_components = min(2, max(1, n_classes - 1), x_train.shape[1], x_train.shape[0] - 1)

    if method == "pca":
        model = PCA(n_components=min(2, x_train.shape[1], x_train.shape[0] - 1), random_state=RANDOM_STATE)
        return model.fit_transform(x_train), model.transform(x_test), labels

    if method == "lda":
        model = LinearDiscriminantAnalysis(n_components=n_components)
        train_emb = model.fit_transform(x_train, y_train)
        test_emb = model.transform(x_test)
        return pad_embedding(train_emb), pad_embedding(test_emb), labels

    if method == "plsda":
        y_onehot = label_binarize(y_train, classes=labels)
        if y_onehot.shape[1] == 1:
            y_onehot = np.column_stack([1 - y_onehot[:, 0], y_onehot[:, 0]])
        model = PLSRegression(n_components=min(2, x_train.shape[1], x_train.shape[0] - 1))
        model.fit(x_train, y_onehot)
        return model.transform(x_train), model.transform(x_test), labels

    if method == "nca":
        model = NeighborhoodComponentsAnalysis(
            n_components=min(2, x_train.shape[1]),
            init="auto",
            random_state=RANDOM_STATE,
            max_iter=NCA_MAX_ITER,
            tol=1e-4,
        )
        train_emb = model.fit_transform(x_train, y_train_int)
        test_emb = model.transform(x_test)
        return train_emb, test_emb, labels

    if method == "rf_proba":
        model = RandomForestClassifier(
            n_estimators=250,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        labels = list(model.classes_)
        return model.predict_proba(x_train), model.predict_proba(x_test), labels

    raise ValueError(f"Unknown method: {method}")


def pad_embedding(embedding: np.ndarray) -> np.ndarray:
    if embedding.ndim == 1:
        embedding = embedding.reshape(-1, 1)
    if embedding.shape[1] == 1:
        return np.column_stack([embedding[:, 0], np.zeros(embedding.shape[0])])
    return embedding


def centroid_map(embedding: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    return {label: embedding[labels == label].mean(axis=0) for label in sorted(set(labels))}


def distance_scores(test_emb: np.ndarray, centroids: dict[str, np.ndarray], label_order: list[str]) -> np.ndarray:
    score_cols = []
    for label in label_order:
        if label not in centroids:
            score_cols.append(np.full(test_emb.shape[0], np.nan))
        else:
            score_cols.append(-np.linalg.norm(test_emb - centroids[label], axis=1))
    return np.column_stack(score_cols)


def softmax_scores(scores: np.ndarray) -> np.ndarray:
    finite_scores = np.nan_to_num(scores, nan=-1e12, neginf=-1e12, posinf=1e12)
    shifted = finite_scores - np.max(finite_scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    denom = exp_scores.sum(axis=1, keepdims=True)
    return exp_scores / np.where(denom == 0, 1.0, denom)


def nearest_centroid_predict(scores: np.ndarray, label_order: list[str]) -> np.ndarray:
    valid_scores = np.nan_to_num(scores, nan=-np.inf)
    return np.array([label_order[idx] for idx in np.argmax(valid_scores, axis=1)])


def embedding_separation_metrics(embedding: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    unique = sorted(set(labels))
    if len(unique) < 2 or len(unique) >= len(labels):
        return {
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
            "centroid_ratio": np.nan,
            "mean_centroid_distance": np.nan,
            "mean_within_scatter": np.nan,
        }

    centroids = centroid_map(embedding, labels)
    centroid_values = np.vstack([centroids[label] for label in unique])
    centroid_dist = pdist(centroid_values, metric="euclidean")
    within = []
    for label in unique:
        group_emb = embedding[labels == label]
        within.extend(np.linalg.norm(group_emb - centroids[label], axis=1).tolist())
    mean_within = float(np.mean(within)) if within else np.nan
    mean_centroid = float(np.mean(centroid_dist)) if len(centroid_dist) else np.nan
    return {
        "silhouette": safe_metric(lambda: silhouette_score(embedding, labels)),
        "davies_bouldin": safe_metric(lambda: davies_bouldin_score(embedding, labels)),
        "calinski_harabasz": safe_metric(lambda: calinski_harabasz_score(embedding, labels)),
        "centroid_ratio": mean_centroid / mean_within if mean_within and mean_within > 0 else np.nan,
        "mean_centroid_distance": mean_centroid,
        "mean_within_scatter": mean_within,
    }


def safe_metric(fn) -> float:
    try:
        value = fn()
        return float(value)
    except Exception:
        return np.nan


def pairwise_rows(
    task: str,
    feature_set: str,
    method: str,
    fold: int,
    embedding: np.ndarray,
    labels: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for a, b in combinations(sorted(set(labels)), 2):
        mask = np.isin(labels, [a, b])
        pair_emb = embedding[mask]
        pair_labels = labels[mask]
        centroids = centroid_map(pair_emb, pair_labels)
        dist = float(np.linalg.norm(centroids[a] - centroids[b]))
        scatter = []
        for label in [a, b]:
            group_emb = pair_emb[pair_labels == label]
            scatter.extend(np.linalg.norm(group_emb - centroids[label], axis=1).tolist())
        within = float(np.mean(scatter)) if scatter else np.nan
        rows.append(
            {
                "task": task,
                "feature_set": feature_set,
                "method": method,
                "fold": fold,
                "group_a": a,
                "group_b": b,
                "centroid_distance": dist,
                "within_scatter": within,
                "distance_within_ratio": dist / within if within and within > 0 else np.nan,
            }
        )
    return rows


def evaluate_feature_set(
    task: str,
    feature_set: str,
    df: pd.DataFrame,
    log: RunLog,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    x, _feature_names = feature_matrix(df)
    y = labels_for_task(df["group"], task)
    subjects = df["subject_id"].astype(str).to_numpy()
    identities = np.array([biological_identity(subject) for subject in subjects])
    label_order = TASKS[task]["label_order"]
    folds = pair_aware_splits(subjects)

    metric_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []

    for method in METHODS:
        method_start = time.time()
        for fold, (train_idx, test_idx) in enumerate(folds, start=1):
            try:
                identity_overlap = set(identities[train_idx]) & set(identities[test_idx])
                if identity_overlap:
                    raise RuntimeError(f"Biological identity leakage: {sorted(identity_overlap)}")
                prepared = prepare_fold(x[train_idx], x[test_idx], y[train_idx])
                train_emb, test_emb, _model_labels = fit_transform_embedding(
                    method, prepared.x_train, prepared.x_test, y[train_idx]
                )
                train_emb = np.asarray(train_emb, dtype=float)
                test_emb = np.asarray(test_emb, dtype=float)
                centroids = centroid_map(train_emb, y[train_idx])
                scores = distance_scores(test_emb, centroids, label_order)
                prob_scores = softmax_scores(scores)
                pred = nearest_centroid_predict(scores, label_order)
                sep = embedding_separation_metrics(test_emb, y[test_idx])

                if task == "binary_acl_vs_ha":
                    acl_idx = label_order.index("ACL")
                    auc = safe_metric(lambda: roc_auc_score((y[test_idx] == "ACL").astype(int), prob_scores[:, acl_idx]))
                    auc_ovr = np.nan
                else:
                    auc = np.nan
                    auc_label_order = sorted(label_order)
                    auc_col_idx = [label_order.index(label) for label in auc_label_order]
                    auc_probs = prob_scores[:, auc_col_idx]
                    auc_ovr = safe_metric(
                        lambda: roc_auc_score(
                            y[test_idx],
                            auc_probs,
                            labels=auc_label_order,
                            multi_class="ovr",
                            average="macro",
                        )
                    )

                metric_rows.append(
                    {
                        "task": task,
                        "feature_set": feature_set,
                        "method": method,
                        "fold": fold,
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        "n_train_identities": len(set(identities[train_idx])),
                        "n_test_identities": len(set(identities[test_idx])),
                        "identity_overlap": len(identity_overlap),
                        "input_features": x.shape[1],
                        "selected_features": prepared.selected_features,
                        **sep,
                        "nearest_centroid_balanced_accuracy": safe_metric(
                            lambda: balanced_accuracy_score(y[test_idx], pred)
                        ),
                        "nearest_centroid_macro_f1": safe_metric(lambda: f1_score(y[test_idx], pred, average="macro")),
                        "binary_auc": auc,
                        "auc_ovr_3class": auc_ovr,
                        "runtime_sec": round(time.time() - method_start, 3),
                        "status": "ok",
                    }
                )
                pair_rows.extend(pairwise_rows(task, feature_set, method, fold, test_emb, y[test_idx]))
                for row_idx, emb, scores_row, probs_row, pred_label in zip(test_idx, test_emb, scores, prob_scores, pred):
                    oof_rows.append(
                        {
                            "task": task,
                            "feature_set": feature_set,
                            "method": method,
                            "fold": fold,
                            "subject_id": subjects[row_idx],
                            "identity_id": identities[row_idx],
                            "true_group": df["group"].iloc[row_idx],
                            "task_label": y[row_idx],
                            "pred_label": pred_label,
                            "emb_1": float(emb[0]) if len(emb) > 0 else np.nan,
                            "emb_2": float(emb[1]) if len(emb) > 1 else 0.0,
                            **{f"score_{label}": float(scores_row[i]) for i, label in enumerate(label_order)},
                            **{f"prob_{label}": float(probs_row[i]) for i, label in enumerate(label_order)},
                        }
                    )
            except Exception as exc:
                metric_rows.append(
                    {
                        "task": task,
                        "feature_set": feature_set,
                        "method": method,
                        "fold": fold,
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        "n_train_identities": len(set(identities[train_idx])),
                        "n_test_identities": len(set(identities[test_idx])),
                        "identity_overlap": len(set(identities[train_idx]) & set(identities[test_idx])),
                        "input_features": x.shape[1],
                        "selected_features": np.nan,
                        "silhouette": np.nan,
                        "davies_bouldin": np.nan,
                        "calinski_harabasz": np.nan,
                        "centroid_ratio": np.nan,
                        "mean_centroid_distance": np.nan,
                        "mean_within_scatter": np.nan,
                        "nearest_centroid_balanced_accuracy": np.nan,
                        "nearest_centroid_macro_f1": np.nan,
                        "binary_auc": np.nan,
                        "auc_ovr_3class": np.nan,
                        "runtime_sec": round(time.time() - method_start, 3),
                        "status": f"failed: {type(exc).__name__}: {exc}",
                    }
                )
        log.add(f"Evaluated {task}/{feature_set}/{method}")
    return metric_rows, pair_rows, oof_rows


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "centroid_ratio",
        "mean_centroid_distance",
        "mean_within_scatter",
        "nearest_centroid_balanced_accuracy",
        "nearest_centroid_macro_f1",
        "binary_auc",
        "auc_ovr_3class",
        "selected_features",
    ]
    grouped = (
        metrics[metrics["status"] == "ok"]
        .groupby(["task", "feature_set", "method"], observed=True)[numeric_cols]
        .agg(["mean", "std"])
    )
    grouped.columns = [f"{col}_{agg}" for col, agg in grouped.columns]
    out = grouped.reset_index()
    return out.sort_values(["task", "silhouette_mean", "centroid_ratio_mean"], ascending=[True, False, False])


def choose_best(agg: pd.DataFrame) -> pd.DataFrame:
    best_rows = []
    for task, sub in agg.groupby("task", observed=True):
        ranked = sub.sort_values(
            ["silhouette_mean", "centroid_ratio_mean", "davies_bouldin_mean"],
            ascending=[False, False, True],
        )
        best_rows.append(ranked.iloc[0].to_dict())
    return pd.DataFrame(best_rows)


def fit_full_embedding(task: str, feature_set: str, method: str, df: pd.DataFrame) -> pd.DataFrame:
    x, _ = feature_matrix(df)
    y = labels_for_task(df["group"], task)
    prepared = prepare_fold(x, x, y)
    train_emb, full_emb, _ = fit_transform_embedding(method, prepared.x_train, prepared.x_test, y)
    full_emb = np.asarray(full_emb, dtype=float)
    if full_emb.shape[1] == 1:
        full_emb = pad_embedding(full_emb)
    out = pd.DataFrame(
        {
            "task": task,
            "feature_set": feature_set,
            "method": method,
            "subject_id": df["subject_id"].astype(str),
            "group": df["group"].astype(str),
            "task_label": y,
            "emb_1": full_emb[:, 0],
            "emb_2": full_emb[:, 1] if full_emb.shape[1] > 1 else 0.0,
            "descriptive_only": True,
        }
    )
    return out


def build_final_embeddings(
    feature_sets: dict[str, pd.DataFrame],
    agg: pd.DataFrame,
    best: pd.DataFrame,
    log: RunLog,
) -> pd.DataFrame:
    rows = []
    wanted = set()
    for _, row in best.iterrows():
        wanted.add((row["task"], row["feature_set"], row["method"]))
        wanted.add((row["task"], row["feature_set"], "pca"))
    for task, feature_set, method in sorted(wanted):
        rows.append(fit_full_embedding(task, feature_set, method, feature_sets[feature_set]))
        log.add(f"Generated descriptive full embedding: {task}/{feature_set}/{method}")
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(RESULTS_DIR / "final_embedding_coordinates.csv", index=False)
    return out


def make_scatter(
    coords: pd.DataFrame,
    task: str,
    feature_set: str,
    method: str,
    path: Path,
    title: str,
) -> None:
    sub = coords[(coords["task"] == task) & (coords["feature_set"] == feature_set) & (coords["method"] == method)]
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    for label in TASKS[task]["label_order"]:
        g = sub[sub["task_label"] == label]
        if g.empty:
            continue
        ax.scatter(
            g["emb_1"],
            g["emb_2"],
            s=70,
            alpha=0.84,
            label=f"{label} (n={len(g)})",
            color=GROUP_COLORS.get(label, "#455A64"),
            edgecolor="white",
            linewidth=0.8,
        )
    ax.axhline(0, color="#B0B0B0", linewidth=0.8)
    ax.axvline(0, color="#B0B0B0", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Embedding 1")
    ax.set_ylabel("Embedding 2")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_heatmap(agg: pd.DataFrame, task: str, path: Path) -> None:
    sub = agg[agg["task"] == task]
    pivot = sub.pivot(index="feature_set", columns="method", values="silhouette_mean").reindex(columns=METHODS)
    fig_h = max(5.0, 0.55 * len(pivot) + 2.0)
    fig, ax = plt.subplots(figsize=(9.5, fig_h))
    values = pivot.to_numpy(dtype=float)
    image = ax.imshow(values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([METHOD_LABELS.get(c, c) for c in pivot.columns], rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"{TASKS[task]['title']} - fold-wise silhouette mean")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_pairwise_plot(pair_df: pd.DataFrame, path: Path) -> None:
    sub = pair_df[
        (pair_df["task"] == "multiclass_3group")
        & (
            ((pair_df["group_a"] == "ACLD") & (pair_df["group_b"] == "ACLR"))
            | ((pair_df["group_a"] == "ACLR") & (pair_df["group_b"] == "ACLD"))
        )
    ]
    summary = (
        sub.groupby(["feature_set", "method"], observed=True)["distance_within_ratio"]
        .mean()
        .reset_index()
        .sort_values("distance_within_ratio", ascending=False)
        .head(18)
    )
    labels = summary["feature_set"] + "\n" + summary["method"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(summary)), summary["distance_within_ratio"], color="#546E7A")
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ACLD vs ACLR distance / within scatter")
    ax.set_title("ACLD vs ACLR pairwise separation")
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_figures(coords: pd.DataFrame, agg: pd.DataFrame, pair_df: pd.DataFrame, best: pd.DataFrame) -> dict[str, str]:
    figure_paths: dict[str, str] = {}
    for _, row in best.iterrows():
        task = row["task"]
        feature_set = row["feature_set"]
        method = row["method"]
        best_path = FIGURES_DIR / f"{task}_best_embedding.png"
        make_scatter(
            coords,
            task,
            feature_set,
            method,
            best_path,
            f"{TASKS[task]['title']} best supervised embedding: {feature_set} / {METHOD_LABELS[method]}",
        )
        figure_paths[f"{task}_best"] = str(best_path.relative_to(SANDBOX))

        compare_path = FIGURES_DIR / f"{task}_pca_vs_best.png"
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
        for ax, compare_method, label in [
            (axes[0], "pca", "PCA baseline"),
            (axes[1], method, f"Best: {METHOD_LABELS[method]}"),
        ]:
            sub = coords[
                (coords["task"] == task)
                & (coords["feature_set"] == feature_set)
                & (coords["method"] == compare_method)
            ]
            for group_label in TASKS[task]["label_order"]:
                g = sub[sub["task_label"] == group_label]
                ax.scatter(
                    g["emb_1"],
                    g["emb_2"],
                    s=60,
                    alpha=0.84,
                    label=group_label,
                    color=GROUP_COLORS.get(group_label, "#455A64"),
                    edgecolor="white",
                    linewidth=0.7,
                )
            ax.axhline(0, color="#B0B0B0", linewidth=0.8)
            ax.axvline(0, color="#B0B0B0", linewidth=0.8)
            ax.set_title(label)
            ax.grid(alpha=0.18)
        axes[0].legend(frameon=False)
        fig.suptitle(f"{TASKS[task]['title']} - {feature_set}: PCA vs supervised")
        fig.tight_layout()
        fig.savefig(compare_path, dpi=180)
        plt.close(fig)
        figure_paths[f"{task}_compare"] = str(compare_path.relative_to(SANDBOX))

        heatmap_path = FIGURES_DIR / f"{task}_method_feature_heatmap.png"
        make_heatmap(agg, task, heatmap_path)
        figure_paths[f"{task}_heatmap"] = str(heatmap_path.relative_to(SANDBOX))

    pair_path = FIGURES_DIR / "multiclass_acld_aclr_pairwise_separation.png"
    make_pairwise_plot(pair_df, pair_path)
    figure_paths["acld_aclr_pairwise"] = str(pair_path.relative_to(SANDBOX))
    return figure_paths


def df_to_html_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    shown = df.head(max_rows).copy()
    return shown.to_html(index=False, classes="data-table", border=0, float_format=lambda x: f"{x:.4f}")


def html_img(path: str, caption: str) -> str:
    return f"<figure><img src='../{html.escape(path)}' alt='{html.escape(caption)}'><figcaption>{html.escape(caption)}</figcaption></figure>"


def write_html(path: Path, body: str) -> None:
    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 40px; color: #1f2933; line-height: 1.55; }
    h1 { font-size: 32px; margin-bottom: 8px; }
    h2 { font-size: 24px; margin-top: 34px; border-bottom: 2px solid #e5e7eb; padding-bottom: 6px; }
    h3 { font-size: 18px; margin-top: 22px; }
    .note { background: #f4f7fb; border-left: 4px solid #456a8a; padding: 12px 14px; margin: 14px 0; }
    .warn { background: #fff8e1; border-left-color: #f9a825; }
    .data-table { border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }
    .data-table th, .data-table td { border: 1px solid #d9dee7; padding: 7px 8px; text-align: left; vertical-align: top; }
    .data-table th { background: #f3f6fa; }
    .figure-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(430px, 1fr)); gap: 24px; align-items: start; }
    figure { margin: 0; border: 1px solid #d9dee7; padding: 12px; border-radius: 6px; background: #fff; }
    figure img { width: 100%; height: auto; display: block; }
    figcaption { margin-top: 8px; font-size: 13px; color: #52606d; }
    code { background: #eef2f7; padding: 1px 4px; border-radius: 4px; }
    """
    path.write_text(
        f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>{html.escape(path.stem)}</title><style>{css}</style></head><body>{body}</body></html>",
        encoding="utf-8",
    )


def render_reports(
    log: RunLog,
    source_before: dict[str, Any],
    source_after: dict[str, Any],
    inventory: pd.DataFrame,
    metrics: pd.DataFrame,
    agg: pd.DataFrame,
    best: pd.DataFrame,
    pair_df: pd.DataFrame,
    figures: dict[str, str],
) -> None:
    integrity_ok = source_before == source_after
    log_body = [
        "<h1>Supervised Separation 실행 로그</h1>",
        f"<div class='note'>원본 및 기존 PCA 입력 무결성: <strong>{'변경 없음' if integrity_ok else '변경 감지'}</strong></div>",
        "<h2>입력 파일 상태</h2>",
        df_to_html_table(pd.DataFrame.from_dict(source_after, orient='index').reset_index(names='source'), 20),
        "<h2>Feature set inventory</h2>",
        df_to_html_table(inventory, 20),
        "<h2>Leakage 방지 설계</h2>",
        "<ul><li>25 HA + 27 ACL longitudinal pair를 52 biological identities로 분할하고 ACLD/ACLR 쌍을 같은 fold에 배정했다.</li><li>outer fold마다 imputer, variance filter, SelectKBest, scaler, embedding을 train fold에만 fit했다.</li><li>전체 데이터 fit embedding은 descriptive 시각화 전용이며 검증 지표에는 사용하지 않았다.</li></ul>",
        "<h2>실행 메시지</h2>",
        "<table class='data-table'><thead><tr><th>Time</th><th>Message</th></tr></thead><tbody>",
    ]
    for row in log.lines:
        log_body.append(f"<tr><td>{html.escape(row['time'])}</td><td>{html.escape(row['message'])}</td></tr>")
    log_body.append("</tbody></table>")
    write_html(HTML_DIR / "01_execution_log.html", "\n".join(log_body))

    top_cols = [
        "task",
        "feature_set",
        "method",
        "silhouette_mean",
        "centroid_ratio_mean",
        "davies_bouldin_mean",
        "nearest_centroid_balanced_accuracy_mean",
        "binary_auc_mean",
        "auc_ovr_3class_mean",
    ]
    pair_summary = (
        pair_df.groupby(["task", "group_a", "group_b"], observed=True)["distance_within_ratio"]
        .max()
        .reset_index()
        .sort_values(["task", "distance_within_ratio"], ascending=[True, False])
    )
    body = [
        "<h1>학습 기반 그룹 분리도 최대화 실험</h1>",
        "<div class='note'>주 지표는 fold-wise supervised embedding의 silhouette와 centroid/within ratio이다. AUC와 macro-F1는 보조 지표로만 해석한다.</div>",
        "<h2>Best embedding summary</h2>",
        df_to_html_table(best[top_cols], 10),
        "<h2>전체 비교표</h2>",
        df_to_html_table(agg[top_cols], 50),
        "<h2>Pairwise 병목 분석</h2>",
        df_to_html_table(pair_summary, 30),
        "<div class='note warn'>3그룹에서는 ACLD↔ACLR pairwise separation을 별도로 확인한다. 이 값이 HA 관련 pair보다 낮으면 3분류 병목은 ACLD와 ACLR의 중첩이다.</div>",
        "<h2>시각화</h2>",
        "<div class='figure-grid'>",
    ]
    for key, fig_path in figures.items():
        body.append(html_img(fig_path, key))
    body.extend(
        [
            "</div>",
            "<h2>방법</h2>",
            "<ul><li>52 biological identities 기준 pair-aware 5-fold이며 ACLD/ACLR longitudinal pair는 같은 fold에 배정.</li><li>각 fold 내부에서만 preprocessing과 embedding fit.</li><li>고차원 feature set은 train fold 내부 SelectKBest로 최대 160개 feature만 사용.</li><li>PCA는 비지도 baseline, LDA/PLS-DA/NCA/RF probability는 supervised embedding 후보.</li></ul>",
        ]
    )
    write_html(HTML_DIR / "02_supervised_separation_report.html", "\n".join(body))


def main() -> int:
    ensure_dirs()
    log = RunLog()
    log.add("Start supervised separation experiment")
    source_before = source_stats()
    write_json(RESULTS_DIR / "00_source_file_stats_before.json", source_before)

    feature_sets = build_feature_sets(log)
    inventory = pd.read_csv(RESULTS_DIR / "feature_set_inventory.csv")

    metric_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    for task in TASKS:
        for feature_set_name, df in feature_sets.items():
            log.add(f"Evaluate {task}/{feature_set_name}")
            m, p, o = evaluate_feature_set(task, feature_set_name, df, log)
            metric_rows.extend(m)
            pair_rows.extend(p)
            oof_rows.extend(o)

    metrics = pd.DataFrame(metric_rows)
    pair_df = pd.DataFrame(pair_rows)
    oof_df = pd.DataFrame(oof_rows)
    agg = aggregate_metrics(metrics)
    best = choose_best(agg)
    coords = build_final_embeddings(feature_sets, agg, best, log)
    figures = make_figures(coords, agg, pair_df, best)

    metrics.to_csv(RESULTS_DIR / "embedding_separation_metrics.csv", index=False)
    pair_df.to_csv(RESULTS_DIR / "pairwise_group_distances.csv", index=False)
    oof_df.to_csv(RESULTS_DIR / "oof_embedding_scores.csv", index=False)
    agg.to_csv(RESULTS_DIR / "embedding_separation_summary.csv", index=False)
    write_json(
        RESULTS_DIR / "best_embedding_summary.json",
        {
            "best_by_task": best.to_dict(orient="records"),
            "selection_rule": "highest fold-wise silhouette_mean, then centroid_ratio_mean, then lower davies_bouldin_mean",
            "primary_metric": "embedding separation, not prediction AUC",
            "validation_unit": "52 biological identities: 25 HA + 27 ACL longitudinal pairs",
            "pair_aware": True,
            "descriptive_full_embeddings_only": True,
            "figures": figures,
        },
    )

    source_after = source_stats()
    integrity = {
        "unchanged": source_before == source_after,
        "before": source_before,
        "after": source_after,
    }
    write_json(RESULTS_DIR / "00_source_file_integrity.json", integrity)
    log.add(f"Source file integrity unchanged={integrity['unchanged']}")
    render_reports(log, source_before, source_after, inventory, metrics, agg, best, pair_df, figures)
    log.add("Completed supervised separation experiment")
    log.write(LOGS_DIR / "01_supervised_separation.log")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
