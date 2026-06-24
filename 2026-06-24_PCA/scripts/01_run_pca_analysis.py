#!/usr/bin/env python3
"""Subject-level PCA, distance, and clustering analysis for gait parquet files."""

from __future__ import annotations

import html
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SANDBOX = Path(__file__).resolve().parents[1]
RESULTS_DIR = SANDBOX / "results"
FIGURES_DIR = SANDBOX / "figures"
HTML_DIR = SANDBOX / "htmls"
LOGS_DIR = SANDBOX / "logs"

SOURCE_FILES = {
    "slim_gait": PROJECT_ROOT / "data/processed/slim_gait.parquet",
    "raw_merged": PROJECT_ROOT / "data/processed/raw_merged.parquet",
    "cycle_waveforms_101": PROJECT_ROOT / "data/processed/cycle_waveforms_101.parquet",
}

DATASET_TITLES = {
    "slim_gait": "slim_gait.parquet",
    "raw_merged": "raw_merged.parquet",
    "cycle_waveforms_101": "cycle_waveforms_101.parquet",
}

META_COLUMNS = {
    "group",
    "subject_id",
    "speed",
    "file_name",
    "trial_id",
    "actual_leg",
    "side_basis",
    "injured_leg",
    "cycle_idx",
    "cycle_idx_original",
    "time_ms",
}

SPEED_ORDER = ["slow", "normal", "fast"]
GROUP_ORDER = ["HA", "ACLD", "ACLR"]
GROUP_COLORS = {"HA": "#2E7D32", "ACLD": "#C62828", "ACLR": "#1565C0"}
GROUP_ALIASES = {"Healthy adults": "HA"}
BATCH_SIZE = 20_000
RANDOM_STATE = 42


class RunLog:
    def __init__(self) -> None:
        self.lines: list[dict[str, str]] = []

    def add(self, message: str) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.lines.append({"time": stamp, "message": message})
        print(f"[{stamp}] {message}", flush=True)

    def write_text(self, path: Path) -> None:
        path.write_text(
            "\n".join(f"[{row['time']}] {row['message']}" for row in self.lines) + "\n",
            encoding="utf-8",
        )


@dataclass
class RunningStats:
    count: np.ndarray
    total: np.ndarray
    total_sq: np.ndarray
    min_val: np.ndarray
    max_val: np.ndarray


def ensure_dirs() -> None:
    for directory in [RESULTS_DIR, FIGURES_DIR, HTML_DIR, LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


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


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parquet_summary(path: Path) -> dict[str, Any]:
    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    return {
        "rows": pf.metadata.num_rows,
        "columns": len(schema.names),
        "row_groups": pf.metadata.num_row_groups,
        "column_names": schema.names,
    }


def is_numeric_type(dtype: pa.DataType) -> bool:
    return pa.types.is_integer(dtype) or pa.types.is_floating(dtype) or pa.types.is_boolean(dtype)


def numeric_columns_for_dataset(name: str, path: Path) -> list[str]:
    schema = pq.ParquetFile(path).schema_arrow
    numeric: list[str] = []
    for field in schema:
        col = field.name
        if col in META_COLUMNS:
            continue
        if name == "cycle_waveforms_101" and not col.rsplit("_", 1)[-1].isdigit():
            continue
        if is_numeric_type(field.type):
            numeric.append(col)
    return numeric


def table_unique_values(path: Path, columns: list[str]) -> pd.DataFrame:
    table = pq.read_table(path, columns=columns)
    return table.to_pandas()


def build_reference_cohort(log: RunLog) -> dict[str, set[str]]:
    df = table_unique_values(SOURCE_FILES["slim_gait"], ["group", "subject_id"])
    df["group"] = df["group"].replace(GROUP_ALIASES)
    df["subject_id"] = df["subject_id"].astype(str)
    cohort: dict[str, set[str]] = {}
    for group in GROUP_ORDER:
        cohort[group] = set(df.loc[df["group"] == group, "subject_id"].unique())
    log.add(
        "Reference cohort from slim_gait: "
        + ", ".join(f"{group}={len(cohort[group])}" for group in GROUP_ORDER)
    )
    return cohort


def cohort_report(reference: dict[str, set[str]], log: RunLog) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for name, path in SOURCE_FILES.items():
        df = table_unique_values(path, ["group", "subject_id"])
        df["mapped_group"] = df["group"].replace(GROUP_ALIASES)
        df["subject_id"] = df["subject_id"].astype(str)
        for original_group in sorted(df["group"].astype(str).unique()):
            mapped = GROUP_ALIASES.get(original_group, original_group)
            subjs = set(df.loc[df["group"].astype(str) == original_group, "subject_id"].unique())
            reference_subjects = reference.get(mapped, set())
            records.append(
                {
                    "dataset": name,
                    "original_group": original_group,
                    "mapped_group": mapped,
                    "subjects_in_file": len(subjs),
                    "rows_in_file": int((df["group"].astype(str) == original_group).sum()),
                    "subjects_used_by_reference_rule": len(subjs & reference_subjects),
                    "subjects_extra_vs_reference": len(subjs - reference_subjects)
                    if mapped in reference
                    else len(subjs),
                    "subjects_missing_from_reference": len(reference_subjects - subjs)
                    if mapped in reference
                    else None,
                }
            )
    out = pd.DataFrame.from_records(records)
    out.to_csv(RESULTS_DIR / "00_cohort_report.csv", index=False)
    log.add("Saved cohort report: results/00_cohort_report.csv")
    return out


def update_stats(stats: RunningStats, arr: np.ndarray) -> None:
    finite = np.isfinite(arr)
    counts = finite.sum(axis=0).astype(np.float64)
    if not np.any(counts):
        return
    safe = np.where(finite, arr, 0.0)
    stats.count += counts
    stats.total += safe.sum(axis=0)
    stats.total_sq += np.square(safe).sum(axis=0)
    valid_min = counts > 0
    batch_min = np.where(finite, arr, np.inf).min(axis=0)
    batch_max = np.where(finite, arr, -np.inf).max(axis=0)
    stats.min_val[valid_min] = np.minimum(stats.min_val[valid_min], batch_min[valid_min])
    stats.max_val[valid_min] = np.maximum(stats.max_val[valid_min], batch_max[valid_min])


def new_stats(n_features: int) -> RunningStats:
    return RunningStats(
        count=np.zeros(n_features, dtype=np.float64),
        total=np.zeros(n_features, dtype=np.float64),
        total_sq=np.zeros(n_features, dtype=np.float64),
        min_val=np.full(n_features, np.inf, dtype=np.float64),
        max_val=np.full(n_features, -np.inf, dtype=np.float64),
    )


def key_sort_value(key: tuple[str, str, str]) -> tuple[str, int, str]:
    subject, group, speed = key
    speed_idx = SPEED_ORDER.index(speed) if speed in SPEED_ORDER else len(SPEED_ORDER)
    return subject, speed_idx, group


def aggregate_subject_features(
    name: str,
    path: Path,
    reference: dict[str, set[str]],
    log: RunLog,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    numeric_cols = numeric_columns_for_dataset(name, path)
    read_cols = ["group", "subject_id", "speed", *numeric_cols]
    pf = pq.ParquetFile(path)
    key_stats: dict[tuple[str, str, str], RunningStats] = {}
    n_batches = 0
    n_rows_seen = 0
    n_rows_used = 0
    t0 = time.time()

    log.add(
        f"{name}: start batch aggregation with {len(numeric_cols)} numeric columns, "
        f"{pf.metadata.num_rows} rows, batch_size={BATCH_SIZE}"
    )

    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=read_cols):
        n_batches += 1
        frame = batch.to_pandas()
        n_rows_seen += len(frame)
        frame["group"] = frame["group"].replace(GROUP_ALIASES)
        frame["subject_id"] = frame["subject_id"].astype(str)
        frame["speed"] = frame["speed"].astype(str)
        group_mask = frame["group"].isin(GROUP_ORDER)
        subject_mask = np.zeros(len(frame), dtype=bool)
        for group in GROUP_ORDER:
            mask = (frame["group"] == group) & frame["subject_id"].isin(reference[group])
            subject_mask |= mask.to_numpy()
        frame = frame.loc[group_mask.to_numpy() & subject_mask]
        if frame.empty:
            continue
        n_rows_used += len(frame)
        grouped = frame.groupby(["subject_id", "group", "speed"], sort=False, observed=True)
        for key, group_frame in grouped:
            arr = group_frame[numeric_cols].to_numpy(dtype=np.float64, copy=False)
            if key not in key_stats:
                key_stats[key] = new_stats(len(numeric_cols))
            update_stats(key_stats[key], arr)
        if n_batches % 10 == 0:
            log.add(f"{name}: processed {n_batches} batches, rows_seen={n_rows_seen:,}, rows_used={n_rows_used:,}")

    rows_by_subject: dict[str, dict[str, Any]] = {}
    for key in sorted(key_stats, key=key_sort_value):
        subject, group, speed = key
        stats = key_stats[key]
        row = rows_by_subject.setdefault(subject, {"subject_id": subject, "group": group})
        valid = stats.count > 0
        means = np.full(len(numeric_cols), np.nan)
        stds = np.full(len(numeric_cols), np.nan)
        mins = np.full(len(numeric_cols), np.nan)
        maxs = np.full(len(numeric_cols), np.nan)
        means[valid] = stats.total[valid] / stats.count[valid]
        variance = np.zeros(len(numeric_cols), dtype=np.float64)
        variance[valid] = stats.total_sq[valid] / stats.count[valid] - np.square(means[valid])
        stds[valid] = np.sqrt(np.maximum(variance[valid], 0.0))
        mins[valid] = stats.min_val[valid]
        maxs[valid] = stats.max_val[valid]
        for col, mean, std, min_val, max_val in zip(numeric_cols, means, stds, mins, maxs):
            prefix = f"{speed}__{col}"
            row[f"{prefix}__mean"] = mean
            row[f"{prefix}__std"] = std
            row[f"{prefix}__min"] = min_val
            row[f"{prefix}__max"] = max_val

    feature_df = pd.DataFrame.from_records(list(rows_by_subject.values()))
    group_rank = {group: idx for idx, group in enumerate(GROUP_ORDER)}
    feature_df = feature_df.sort_values(
        by=["group", "subject_id"],
        key=lambda s: s.map(group_rank).fillna(99) if s.name == "group" else s,
    ).reset_index(drop=True)
    feature_path = RESULTS_DIR / f"{name}_subject_features.csv"
    feature_df.to_csv(feature_path, index=False)

    summary = {
        "dataset": name,
        "source_path": str(path.relative_to(PROJECT_ROOT)),
        "source_rows": pf.metadata.num_rows,
        "source_columns": len(pf.schema_arrow.names),
        "numeric_columns_used": len(numeric_cols),
        "batch_size": BATCH_SIZE,
        "batches_processed": n_batches,
        "rows_seen": n_rows_seen,
        "rows_used_after_reference_filter": n_rows_used,
        "subject_rows": int(len(feature_df)),
        "subject_counts": feature_df["group"].value_counts().reindex(GROUP_ORDER).fillna(0).astype(int).to_dict(),
        "subject_feature_columns": int(feature_df.shape[1] - 2),
        "aggregation_seconds": round(time.time() - t0, 3),
        "memory_strategy": "pyarrow ParquetFile.iter_batches with projected columns; no full raw_merged pandas load",
    }
    write_json(RESULTS_DIR / f"{name}_aggregation_summary.json", summary)
    log.add(
        f"{name}: saved subject features ({feature_df.shape[0]} subjects x "
        f"{feature_df.shape[1] - 2} features) to {feature_path.relative_to(SANDBOX)}"
    )
    return feature_df, summary


def clean_for_pca(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str], dict[str, Any]]:
    feature_cols = [c for c in feature_df.columns if c not in {"subject_id", "group"}]
    x_raw = feature_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    all_nan_cols = x_raw.columns[x_raw.isna().all()].tolist()
    x_raw = x_raw.drop(columns=all_nan_cols)
    imputer = SimpleImputer(strategy="median")
    x_imputed = imputer.fit_transform(x_raw)
    selector = VarianceThreshold(threshold=0.0)
    x_nonconst = selector.fit_transform(x_imputed)
    kept_features = x_raw.columns[selector.get_support()].tolist()
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_nonconst)
    nan_count = int(feature_df[feature_cols].isna().sum().sum())
    info = {
        "input_feature_columns": len(feature_cols),
        "all_nan_columns_dropped": len(all_nan_cols),
        "zero_variance_columns_dropped": int(len(x_raw.columns) - len(kept_features)),
        "features_kept_for_pca": len(kept_features),
        "missing_values_imputed": nan_count,
        "non_finite_after_scaling": int((~np.isfinite(x_scaled)).sum()),
    }
    return feature_df[["subject_id", "group"]].copy(), x_scaled, kept_features, info


def run_pca_analysis(name: str, feature_df: pd.DataFrame, log: RunLog) -> dict[str, Any]:
    meta, x_scaled, feature_names, clean_info = clean_for_pca(feature_df)
    n_samples, n_features = x_scaled.shape
    if n_samples < 3 or n_features < 2:
        raise ValueError(f"{name}: need at least 3 samples and 2 non-constant features for PCA")
    n_components = min(10, n_samples - 1, n_features)
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    pcs = pca.fit_transform(x_scaled)

    pc_cols = [f"PC{i}" for i in range(1, n_components + 1)]
    scores = pd.concat([meta, pd.DataFrame(pcs, columns=pc_cols)], axis=1)
    kmeans = KMeans(n_clusters=3, n_init=20, random_state=RANDOM_STATE)
    scores["kmeans_cluster"] = kmeans.fit_predict(x_scaled)
    scores.to_csv(RESULTS_DIR / f"{name}_pca_scores.csv", index=False)

    explained = pd.DataFrame(
        {
            "component": pc_cols,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    explained.to_csv(RESULTS_DIR / f"{name}_pca_explained_variance.csv", index=False)

    loadings = pd.DataFrame(pca.components_.T, columns=pc_cols)
    loadings.insert(0, "feature", feature_names)
    loadings.to_csv(RESULTS_DIR / f"{name}_pca_loadings.csv", index=False)

    dist_components = [f"PC{i}" for i in range(1, min(5, n_components) + 1)]
    centroids = scores.groupby("group", observed=True)[dist_components].mean().reindex(GROUP_ORDER).dropna(how="all")
    distance_matrix = pd.DataFrame(
        squareform(pdist(centroids.to_numpy(), metric="euclidean")),
        index=centroids.index,
        columns=centroids.index,
    )
    distance_matrix.to_csv(RESULTS_DIR / f"{name}_group_centroid_distances.csv")

    subject_distances: list[dict[str, Any]] = []
    centroid_lookup = centroids.to_dict(orient="index")
    for _, row in scores.iterrows():
        group = row["group"]
        if group not in centroid_lookup:
            continue
        target = np.array([centroid_lookup[group][pc] for pc in dist_components], dtype=float)
        observed = row[dist_components].to_numpy(dtype=float)
        subject_distances.append(
            {
                "subject_id": row["subject_id"],
                "group": group,
                "distance_to_own_group_centroid": float(np.linalg.norm(observed - target)),
            }
        )
    pd.DataFrame(subject_distances).to_csv(RESULTS_DIR / f"{name}_subject_to_centroid_distances.csv", index=False)

    link = linkage(x_scaled, method="ward")
    linkage_df = pd.DataFrame(link, columns=["cluster_a", "cluster_b", "distance", "sample_count"])
    linkage_df.to_csv(RESULTS_DIR / f"{name}_hierarchical_linkage.csv", index=False)

    cluster_comp = pd.crosstab(scores["kmeans_cluster"], scores["group"]).reindex(columns=GROUP_ORDER).fillna(0).astype(int)
    cluster_comp.to_csv(RESULTS_DIR / f"{name}_cluster_composition.csv")

    top_loadings = {}
    for pc in pc_cols[:3]:
        top = loadings[["feature", pc]].assign(abs_loading=lambda d: d[pc].abs())
        top_loadings[pc] = top.sort_values("abs_loading", ascending=False).head(12).to_dict(orient="records")

    summary = {
        "dataset": name,
        "n_subjects": int(n_samples),
        "group_counts": scores["group"].value_counts().reindex(GROUP_ORDER).fillna(0).astype(int).to_dict(),
        "features_for_pca": int(n_features),
        "n_components": int(n_components),
        "pc1_variance": float(pca.explained_variance_ratio_[0]),
        "pc2_variance": float(pca.explained_variance_ratio_[1]) if n_components > 1 else None,
        "cumulative_variance_5pc": float(np.cumsum(pca.explained_variance_ratio_)[min(4, n_components - 1)]),
        "cleaning": clean_info,
        "distance_components": dist_components,
        "top_loadings": top_loadings,
    }
    write_json(RESULTS_DIR / f"{name}_pca_summary.json", summary)
    make_figures(name, scores, explained, distance_matrix, cluster_comp, link, log)
    log.add(f"{name}: PCA, distances, clustering, and figures completed")
    return summary


def make_figures(
    name: str,
    scores: pd.DataFrame,
    explained: pd.DataFrame,
    distance_matrix: pd.DataFrame,
    cluster_comp: pd.DataFrame,
    link: np.ndarray,
    log: RunLog,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for group in GROUP_ORDER:
        subset = scores[scores["group"] == group]
        if subset.empty:
            continue
        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            label=f"{group} (n={len(subset)})",
            s=65,
            alpha=0.84,
            color=GROUP_COLORS[group],
            edgecolor="white",
            linewidth=0.8,
        )
    ax.axhline(0, color="#B0B0B0", linewidth=0.8)
    ax.axvline(0, color="#B0B0B0", linewidth=0.8)
    ax.set_title(f"{DATASET_TITLES[name]} - PCA subject scores")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name}_01_pca_scatter.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    x = np.arange(1, len(explained) + 1)
    ax.bar(x, explained["explained_variance_ratio"], color="#455A64", alpha=0.75, label="Individual")
    ax.plot(x, explained["cumulative_explained_variance_ratio"], marker="o", color="#D84315", label="Cumulative")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title(f"{DATASET_TITLES[name]} - explained variance")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name}_02_explained_variance.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    values = distance_matrix.to_numpy()
    image = ax.imshow(values, cmap="YlGnBu")
    ax.set_xticks(range(len(distance_matrix.columns)))
    ax.set_xticklabels(distance_matrix.columns)
    ax.set_yticks(range(len(distance_matrix.index)))
    ax.set_yticklabels(distance_matrix.index)
    ax.set_title(f"{DATASET_TITLES[name]} - group centroid distance")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", color="#1A1A1A")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name}_03_group_distance_heatmap.png", dpi=180)
    plt.close(fig)

    labels = (scores["subject_id"].astype(str) + " " + scores["group"].astype(str)).tolist()
    fig, ax = plt.subplots(figsize=(12, 7))
    dendrogram(link, labels=labels, leaf_rotation=90, leaf_font_size=7, ax=ax, color_threshold=None)
    ax.set_title(f"{DATASET_TITLES[name]} - hierarchical clustering")
    ax.set_ylabel("Ward distance")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name}_04_dendrogram.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    bottom = np.zeros(len(cluster_comp))
    x_pos = np.arange(len(cluster_comp))
    for group in GROUP_ORDER:
        vals = cluster_comp[group].to_numpy() if group in cluster_comp else np.zeros(len(cluster_comp))
        ax.bar(x_pos, vals, bottom=bottom, color=GROUP_COLORS[group], alpha=0.82, label=group)
        bottom += vals
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"C{idx}" for idx in cluster_comp.index])
    ax.set_ylabel("Subject count")
    ax.set_title(f"{DATASET_TITLES[name]} - KMeans cluster composition")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name}_05_cluster_composition.png", dpi=180)
    plt.close(fig)
    log.add(f"{name}: saved 5 figures")


def df_to_html_table(df: pd.DataFrame, max_rows: int = 20, index: bool = False) -> str:
    return df.head(max_rows).to_html(index=index, classes="data-table", border=0, float_format=lambda x: f"{x:.4f}")


def rel(path: Path) -> str:
    return html.escape(os.path.relpath(path, start=HTML_DIR))


def render_execution_log(
    log: RunLog,
    source_before: dict[str, Any],
    source_after: dict[str, Any],
    parquet_summaries: dict[str, Any],
    cohort_df: pd.DataFrame,
    aggregation_summaries: dict[str, Any],
) -> None:
    integrity_ok = source_before == source_after
    body = [
        "<h1>2026-06-24 PCA 실행 로그</h1>",
        f"<p class='status {'ok' if integrity_ok else 'bad'}'>원본 parquet 무결성 확인: {'변경 없음' if integrity_ok else '변경 감지'}</p>",
        "<h2>원본 파일 상태</h2>",
        df_to_html_table(pd.DataFrame.from_dict(source_after, orient="index").reset_index(names="dataset")),
        "<h2>Parquet 구조</h2>",
        df_to_html_table(
            pd.DataFrame(
                [
                    {
                        "dataset": name,
                        "rows": item["rows"],
                        "columns": item["columns"],
                        "row_groups": item["row_groups"],
                    }
                    for name, item in parquet_summaries.items()
                ]
            )
        ),
        "<h2>Cohort 점검</h2>",
        df_to_html_table(cohort_df, max_rows=50),
        "<h2>집계 요약</h2>",
        df_to_html_table(pd.DataFrame(list(aggregation_summaries.values()))),
        "<h2>실행 메시지</h2>",
        "<table class='data-table'><thead><tr><th>Time</th><th>Message</th></tr></thead><tbody>",
    ]
    for row in log.lines:
        body.append(f"<tr><td>{html.escape(row['time'])}</td><td>{html.escape(row['message'])}</td></tr>")
    body.append("</tbody></table>")
    write_html(HTML_DIR / "01_execution_log.html", "\n".join(body))


def render_final_report(
    pca_summaries: dict[str, Any],
    cohort_df: pd.DataFrame,
) -> None:
    body = [
        "<h1>2026-06-24 PCA 그룹 거리 및 Clustering 분석 보고서</h1>",
        "<p>세 parquet 파일을 원본 수정 없이 읽고, subject-level 요약 feature를 만든 뒤 PCA, 그룹 중심 거리, KMeans, hierarchical clustering을 수행했다.</p>",
        "<h2>분석 기준</h2>",
        "<ul>",
        "<li>주 분석 단위: subject-level</li>",
        "<li>그룹 기준: slim_gait의 HA 25명, ACLD 27명, ACLR 27명 reference cohort</li>",
        "<li>raw_merged의 Healthy adults는 HA로 매핑하고, Healthy adolescents 및 reference 밖 대상자는 주 분석에서 제외</li>",
        "<li>대용량 raw_merged는 pyarrow batch read로 처리하며 전체 pandas load를 사용하지 않음</li>",
        "</ul>",
        "<h2>Cohort 요약</h2>",
        df_to_html_table(cohort_df, max_rows=50),
        "<h2>핵심 PCA 요약</h2>",
        df_to_html_table(
            pd.DataFrame(
                [
                    {
                        "dataset": name,
                        "subjects": s["n_subjects"],
                        "HA": s["group_counts"].get("HA", 0),
                        "ACLD": s["group_counts"].get("ACLD", 0),
                        "ACLR": s["group_counts"].get("ACLR", 0),
                        "features": s["features_for_pca"],
                        "PC1_var": s["pc1_variance"],
                        "PC2_var": s["pc2_variance"],
                        "cum_var_5PC": s["cumulative_variance_5pc"],
                    }
                    for name, s in pca_summaries.items()
                ]
            )
        ),
    ]
    for name, summary in pca_summaries.items():
        body.extend(
            [
                f"<h2>{html.escape(DATASET_TITLES[name])}</h2>",
                "<div class='figure-grid'>",
                figure_block(FIGURES_DIR / f"{name}_01_pca_scatter.png", "PCA subject scatter"),
                figure_block(FIGURES_DIR / f"{name}_02_explained_variance.png", "Explained variance"),
                figure_block(FIGURES_DIR / f"{name}_03_group_distance_heatmap.png", "Group centroid distance"),
                figure_block(FIGURES_DIR / f"{name}_04_dendrogram.png", "Hierarchical clustering"),
                figure_block(FIGURES_DIR / f"{name}_05_cluster_composition.png", "KMeans cluster composition"),
                "</div>",
                "<h3>Top loading features</h3>",
            ]
        )
        for pc, rows in summary["top_loadings"].items():
            top_df = pd.DataFrame(rows)[["feature", pc, "abs_loading"]]
            body.append(f"<h4>{pc}</h4>")
            body.append(df_to_html_table(top_df, max_rows=12))
    write_html(HTML_DIR / "02_pca_group_distance_report.html", "\n".join(body))


def figure_block(path: Path, caption: str) -> str:
    return (
        "<figure>"
        f"<img src='{rel(path)}' alt='{html.escape(caption)}'>"
        f"<figcaption>{html.escape(caption)}</figcaption>"
        "</figure>"
    )


def write_html(path: Path, body: str) -> None:
    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 40px; color: #1f2933; line-height: 1.55; }
    h1 { font-size: 32px; margin-bottom: 8px; }
    h2 { font-size: 24px; margin-top: 36px; border-bottom: 2px solid #e5e7eb; padding-bottom: 6px; }
    h3 { font-size: 19px; margin-top: 24px; }
    h4 { font-size: 16px; margin-bottom: 6px; }
    .status { display: inline-block; padding: 8px 12px; border-radius: 6px; font-weight: 700; }
    .ok { background: #e8f5e9; color: #1b5e20; }
    .bad { background: #ffebee; color: #b71c1c; }
    .data-table { border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }
    .data-table th, .data-table td { border: 1px solid #d9dee7; padding: 7px 8px; text-align: left; vertical-align: top; }
    .data-table th { background: #f3f6fa; }
    .figure-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 24px; align-items: start; }
    figure { margin: 0; border: 1px solid #d9dee7; padding: 12px; border-radius: 6px; background: #fff; }
    figure img { width: 100%; height: auto; display: block; }
    figcaption { margin-top: 8px; font-size: 13px; color: #52606d; }
    """
    html_text = f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>{html.escape(path.stem)}</title><style>{css}</style></head><body>{body}</body></html>"
    path.write_text(html_text, encoding="utf-8")


def main() -> int:
    ensure_dirs()
    log = RunLog()
    log.add("Start PCA sandbox analysis")
    source_before = source_stats()
    write_json(RESULTS_DIR / "00_source_file_stats_before.json", source_before)

    for name, path in SOURCE_FILES.items():
        if not path.exists():
            raise FileNotFoundError(path)

    parquet_summaries = {name: parquet_summary(path) for name, path in SOURCE_FILES.items()}
    write_json(
        RESULTS_DIR / "00_parquet_summaries.json",
        {
            name: {k: v for k, v in summary.items() if k != "column_names"}
            for name, summary in parquet_summaries.items()
        },
    )
    reference = build_reference_cohort(log)
    cohort_df = cohort_report(reference, log)

    aggregation_summaries: dict[str, Any] = {}
    pca_summaries: dict[str, Any] = {}
    for name, path in SOURCE_FILES.items():
        features, agg_summary = aggregate_subject_features(name, path, reference, log)
        aggregation_summaries[name] = agg_summary
        pca_summaries[name] = run_pca_analysis(name, features, log)

    write_json(RESULTS_DIR / "00_aggregation_summaries.json", aggregation_summaries)
    write_json(RESULTS_DIR / "00_pca_summaries.json", pca_summaries)

    source_after = source_stats()
    integrity = {
        "unchanged": source_before == source_after,
        "before": source_before,
        "after": source_after,
    }
    write_json(RESULTS_DIR / "00_source_file_integrity.json", integrity)
    log.add(f"Source file integrity unchanged={integrity['unchanged']}")

    render_execution_log(log, source_before, source_after, parquet_summaries, cohort_df, aggregation_summaries)
    render_final_report(pca_summaries, cohort_df)
    log.write_text(LOGS_DIR / "01_run_pca_analysis.log")
    log.add("Completed PCA sandbox analysis")
    log.write_text(LOGS_DIR / "01_run_pca_analysis.log")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
