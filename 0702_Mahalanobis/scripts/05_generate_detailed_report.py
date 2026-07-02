#!/usr/bin/env python3
"""Generate the evidence-backed 0702 Mahalanobis experiment audit report.

This script is intentionally read-only with respect to the experiment artifacts.
It reads selected parquet columns, source files, the Optuna SQLite database, and
existing SHAP PNGs, then writes one self-contained HTML report.
"""
from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import platform
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import sklearn
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[2]
SANDBOX = ROOT / "0702_Mahalanobis"
RESULTS = SANDBOX / "results"
FEATURES = ROOT / "data" / "processed" / "mahalanobis_features.parquet"
OOF = RESULTS / "oof_results.parquet"
OOF_TEST = RESULTS / "oof_results_test.parquet"
ID_CSV = ROOT / "data" / "ID.csv"
PAIRING = ROOT / "data" / "processed" / "id_pairing_summary.csv"
OPTUNA_DB = RESULTS / "optuna_mahalanobis.db"
OPTUNA_JSON = RESULTS / "01_optuna_best_params.json"
EVAL_MD = RESULTS / "02_evaluation_report.md"
SHAP_DIR = RESULTS / "03_shap_interpretation"
OUTPUT = SANDBOX / "htmls" / "01_detailed_experiment_analysis.html"

META_COLS = [
    "subject_id", "group", "speed", "trial_id", "actual_leg", "side_basis",
    "stride_idx", "cycle_len", "label", "mahal_dist", "impairment_score",
]
GROUP_ORDER = ["HA", "ACLD", "ACLR"]
COLORS = {"HA": "#2878b5", "ACLD": "#cf4b3f", "ACLR": "#e49b35"}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def fnum(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:,.{digits}f}"


def biological_identity(subject_id: str, confirmed_pair_map: dict[str, str]) -> str:
    if subject_id in confirmed_pair_map:
        return confirmed_pair_map[subject_id]
    match = re.fullmatch(r"(ACLD|ACLR|HA)(.+)", str(subject_id))
    if match is None:
        return f"UNKNOWN::{subject_id}"
    cohort, suffix = match.groups()
    return f"HA::{suffix}" if cohort == "HA" else f"UNPAIRED::{subject_id}"


def sha256_short(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()[:16]


def mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def table(frame: pd.DataFrame, classes: str = "") -> str:
    return frame.to_html(
        index=False,
        border=0,
        classes=f"data-table {classes}".strip(),
        escape=False,
        na_rep="NA",
    )


def fig_svg(fig: plt.Figure) -> str:
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    svg = buffer.getvalue()
    svg = re.sub(r"<\?xml.*?\?>|<!DOCTYPE.*?>", "", svg, flags=re.S).strip()
    return "\n".join(line.rstrip() for line in svg.splitlines())


def png_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def cluster_bootstrap_auc(
    frame: pd.DataFrame,
    score_col: str,
    cluster_col: str = "identity_id",
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile CI by resampling biological-identity clusters."""
    rng = np.random.default_rng(seed)
    clusters = frame[cluster_col].drop_duplicates().to_numpy()
    payload = {
        key: (
            group["label"].to_numpy(dtype=int),
            group[score_col].to_numpy(dtype=float),
        )
        for key, group in frame.groupby(cluster_col, sort=False)
    }
    values: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        labels = np.concatenate([payload[key][0] for key in sampled])
        scores = np.concatenate([payload[key][1] for key in sampled])
        if np.unique(labels).size == 2:
            values.append(float(roc_auc_score(labels, scores)))
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def load_optuna() -> tuple[pd.DataFrame, dict[int, dict[str, object]]]:
    con = sqlite3.connect(OPTUNA_DB)
    trials = pd.read_sql_query(
        """
        SELECT t.trial_id, t.number, t.datetime_start, t.datetime_complete,
               v.value
        FROM trials t JOIN trial_values v USING(trial_id)
        ORDER BY t.number
        """,
        con,
    )
    params = pd.read_sql_query(
        "SELECT trial_id, param_name, param_value, distribution_json FROM trial_params",
        con,
    )
    con.close()
    decoded: dict[int, dict[str, object]] = {}
    trial_number = trials.set_index("trial_id")["number"].to_dict()
    for row in params.itertuples(index=False):
        dist = json.loads(row.distribution_json)
        attrs = dist["attributes"]
        value: object = row.param_value
        if dist["name"] == "CategoricalDistribution":
            value = attrs["choices"][int(value)]
        elif dist["name"] == "IntDistribution":
            value = int(value)
        decoded.setdefault(int(trial_number[row.trial_id]), {})[row.param_name] = value
    trials["datetime_start"] = pd.to_datetime(trials["datetime_start"])
    full_created = pd.Timestamp(datetime.fromtimestamp(FEATURES.stat().st_mtime))
    trials["phase"] = np.where(trials["datetime_start"] < full_created, "TEST 3-fold", "FULL 5-fold")
    if not {"TEST 3-fold", "FULL 5-fold"}.issubset(set(trials["phase"])):
        raise RuntimeError("Cannot separate test/full Optuna phases from artifact timestamps")
    return trials, decoded


def feature_inventory() -> tuple[dict[str, object], pd.DataFrame]:
    parquet = pq.ParquetFile(FEATURES)
    names = parquet.schema_arrow.names
    feature_names = names[8:]
    channel_names = sorted({name.rsplit("_", 1)[0] for name in feature_names})
    type_rows = []
    total_null = 0
    total_cells = 0
    for label, prefix in [
        ("Joint angles", ("hip_", "knee_", "ankle_")),
        ("Free acceleration", ("sensorFreeAcceleration",)),
        ("Orientation components", ("sensorOrientation",)),
        ("Magnetic field", ("sensorMagneticField",)),
    ]:
        cols = [c for c in feature_names if c.startswith(prefix)]
        null_count = 0
        for col in cols:
            idx = names.index(col)
            for row_group in range(parquet.metadata.num_row_groups):
                stats = parquet.metadata.row_group(row_group).column(idx).statistics
                null_count += int(stats.null_count or 0) if stats and stats.has_null_count else 0
        cells = parquet.metadata.num_rows * len(cols)
        total_null += null_count
        total_cells += cells
        type_rows.append(
            {
                "종류": label,
                "채널": len({c.rsplit("_", 1)[0] for c in cols}),
                "특징 열": len(cols),
                "결측률": f"{100 * null_count / cells:.1f}%" if cells else "NA",
            }
        )
    inventory = {
        "rows": parquet.metadata.num_rows,
        "columns": len(names),
        "features": len(feature_names),
        "channels": len(channel_names),
        "null_count": total_null,
        "total_cells": total_cells,
        "null_rate": total_null / total_cells,
    }
    return inventory, pd.DataFrame(type_rows)


def make_figures(df: pd.DataFrame, folds: pd.DataFrame, trials: pd.DataFrame) -> dict[str, str]:
    figures: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    for score, color, label in [
        ("mahal_dist", "#5b4b8a", "Raw distance"),
        ("impairment_score", "#d97706", "Clipped fold score"),
    ]:
        fpr, tpr, _ = roc_curve(df["label"], df[score])
        auc = roc_auc_score(df["label"], df[score])
        ax.plot(fpr, tpr, lw=2.4, color=color, label=f"{label} AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#7f8c8d", lw=1)
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Default full OOF ROC (stride-weighted)")
    ax.legend(frameon=False)
    ax.grid(alpha=.18)
    figures["roc"] = fig_svg(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    bars = ax.bar(folds["fold"], folds["auc"], color=["#cf4b3f" if x < .5 else "#2878b5" for x in folds["auc"]])
    ax.axhline(.5, color="#555", ls="--", lw=1)
    for bar, value in zip(bars, folds["auc"]):
        ax.text(bar.get_x() + bar.get_width()/2, value + .015, f"{value:.3f}", ha="center", fontsize=9)
    ax.set(ylim=(0, .78), xlabel="Fold", ylabel="ROC-AUC", title="Fold instability")
    ax.grid(axis="y", alpha=.18)
    figures["folds"] = fig_svg(fig)

    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    for ax, metric, title in zip(axes, ["mahal_dist", "impairment_score"], ["Mahalanobis distance", "Clipped score"]):
        arrays = [df.loc[df.group == group, metric].to_numpy() for group in GROUP_ORDER]
        violin = ax.violinplot(arrays, positions=np.arange(3), showmedians=True, showextrema=False)
        for body, group in zip(violin["bodies"], GROUP_ORDER):
            body.set_facecolor(COLORS[group]); body.set_alpha(.35)
        for pos, (group, values) in enumerate(zip(GROUP_ORDER, arrays)):
            sample = rng.choice(values, min(350, len(values)), replace=False)
            ax.scatter(rng.normal(pos, .055, len(sample)), sample, s=5, alpha=.16, color=COLORS[group])
        ax.set_xticks(range(3), GROUP_ORDER)
        ax.set_yscale("symlog", linthresh=.2)
        ax.set_title(title + " (stride rows, symlog)")
        ax.grid(axis="y", alpha=.15)
    figures["distributions"] = fig_svg(fig)

    subject = df.groupby(["subject_id", "group"], as_index=False).agg(
        distance_mean=("mahal_dist", "mean"), score_mean=("impairment_score", "mean")
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7))
    for ax, metric, title in zip(axes, ["distance_mean", "score_mean"], ["Subject mean distance", "Subject mean score"]):
        arrays = [subject.loc[subject.group == group, metric].to_numpy() for group in GROUP_ORDER]
        ax.boxplot(arrays, positions=np.arange(3), widths=.5, showfliers=False)
        for pos, (group, values) in enumerate(zip(GROUP_ORDER, arrays)):
            ax.scatter(rng.normal(pos, .055, len(values)), values, s=22, alpha=.72, color=COLORS[group])
        ax.set_xticks(range(3), GROUP_ORDER)
        ax.set_yscale("symlog", linthresh=.2)
        ax.set_title(title + " (n=92 sessions)")
        ax.grid(axis="y", alpha=.15)
    figures["subjects"] = fig_svg(fig)

    fig, ax = plt.subplots(figsize=(8.3, 4.8))
    for phase, marker, color in [("TEST 3-fold", "o", "#d97706"), ("FULL 5-fold", "o", "#2878b5")]:
        part = trials[trials.phase == phase]
        ax.scatter(part.number, part.value, label=phase, marker=marker, color=color, alpha=.78, s=28)
    ax.axhline(.5, color="#555", ls="--", lw=1)
    ax.set(xlabel="Trial number", ylabel="OOF objective AUC", title="One Optuna study mixes test and full-data trials")
    ax.legend(frameon=False)
    ax.grid(alpha=.15)
    figures["optuna"] = fig_svg(fig)
    return figures


def main() -> None:
    required = [FEATURES, OOF, OOF_TEST, ID_CSV, OPTUNA_DB, OPTUNA_JSON, EVAL_MD]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required audit inputs missing: {missing}")

    inventory, feature_types = feature_inventory()
    paired = pd.read_csv(PAIRING)
    paired = paired[paired["pair_status"] == "paired"].copy()
    confirmed_pair_map: dict[str, str] = {}
    for row in paired.itertuples(index=False):
        identity_name = f"ACL::{int(row.num)}"
        confirmed_pair_map[str(row.ID_ACLD)] = identity_name
        confirmed_pair_map[str(row.ID_ACLR)] = identity_name

    df = pd.read_parquet(OOF, columns=META_COLS)
    df["identity_id"] = df["subject_id"].map(
        lambda subject: biological_identity(str(subject), confirmed_pair_map)
    )
    test_df = pd.read_parquet(OOF_TEST, columns=["subject_id", "group", "label", "mahal_dist", "impairment_score"])
    trials, trial_params = load_optuna()
    json_best = json.loads(OPTUNA_JSON.read_text(encoding="utf-8"))

    # Reconstruct the deterministic GroupKFold assignment without refitting the pipeline.
    splitter = GroupKFold(n_splits=5)
    fold_rows: list[dict[str, object]] = []
    subject_fold: dict[str, int] = {}
    zeros = np.zeros((len(df), 1))
    for fold, (train_idx, val_idx) in enumerate(splitter.split(zeros, df.label, df.subject_id), 1):
        for subject in df.iloc[val_idx].subject_id.unique():
            subject_fold[str(subject)] = fold
        train_ids = set(df.iloc[train_idx].identity_id)
        val_ids = set(df.iloc[val_idx].identity_id)
        positive = df.iloc[val_idx].impairment_score.to_numpy() > 1e-10
        slope, intercept = np.polyfit(
            df.iloc[val_idx].loc[positive, "impairment_score"],
            df.iloc[val_idx].loc[positive, "mahal_dist"],
            1,
        )
        fold_rows.append({
            "fold": fold,
            "rows": len(val_idx),
            "sessions": df.iloc[val_idx].subject_id.nunique(),
            "identities": df.iloc[val_idx].identity_id.nunique(),
            "identity_overlap": len(train_ids & val_ids),
            "auc": roc_auc_score(df.iloc[val_idx].label, df.iloc[val_idx].mahal_dist),
            "mu_inferred": intercept,
            "sigma_inferred": slope,
            "zero_floor": 1 - positive.mean(),
        })
    folds = pd.DataFrame(fold_rows)

    paired_present = paired[
        paired.ID_ACLD.isin(subject_fold) & paired.ID_ACLR.isin(subject_fold)
    ].copy()
    paired_present["same_fold"] = paired_present.apply(
        lambda row: subject_fold[row.ID_ACLD] == subject_fold[row.ID_ACLR], axis=1
    )
    confirmed_split = int((~paired_present.same_fold).sum())
    confirmed_same = int(paired_present.same_fold.sum())
    data_subjects = set(df.subject_id.unique())
    id_subjects = set(pd.read_csv(ID_CSV).ID.astype(str))
    unexpected = sorted(data_subjects - id_subjects)
    absent_expected = sorted(set(paired.ID_ACLR.dropna().astype(str)) - data_subjects)

    # Performance at each defensible aggregation unit.
    session = df.groupby(["subject_id", "group", "identity_id"], as_index=False).agg(
        label=("label", "first"),
        distance_mean=("mahal_dist", "mean"), distance_median=("mahal_dist", "median"),
        score_mean=("impairment_score", "mean"), score_median=("impairment_score", "median"),
    )
    identity = df.groupby("identity_id", as_index=False).agg(
        label=("label", "first"), distance_mean=("mahal_dist", "mean"),
        distance_median=("mahal_dist", "median"), score_mean=("impairment_score", "mean"),
        score_median=("impairment_score", "median"),
    )
    stride_auc = roc_auc_score(df.label, df.mahal_dist)
    score_auc = roc_auc_score(df.label, df.impairment_score)
    stride_ci = cluster_bootstrap_auc(df, "mahal_dist")
    score_ci = cluster_bootstrap_auc(df, "impairment_score")

    # The 9-session pilot uses 3-fold GroupKFold; audit its class composition.
    test_splitter = GroupKFold(n_splits=3)
    test_fold_rows = []
    test_zeros = np.zeros((len(test_df), 1))
    for fold, (_, val_idx) in enumerate(
        test_splitter.split(test_zeros, test_df.label, test_df.subject_id), 1
    ):
        validation = test_df.iloc[val_idx]
        test_fold_rows.append({
            "Fold": fold,
            "검증 sessions": validation.subject_id.nunique(),
            "검증 groups": ", ".join(sorted(validation.group.unique())),
            "HA rows": int((validation.group == "HA").sum()),
            "ACL rows": int((validation.group != "HA").sum()),
            "fold AUC 가능": "예" if validation.label.nunique() == 2 else "아니오 (single class)",
        })
    test_fold_table = pd.DataFrame(test_fold_rows)

    auc_rows = [
        ["Stride (9,540 rows)", "raw distance", stride_auc, *stride_ci],
        ["Stride (9,540 rows)", "clipped score", score_auc, *score_ci],
        ["Session mean (92)", "raw distance", roc_auc_score(session.label, session.distance_mean), np.nan, np.nan],
        ["Session median (92)", "raw distance", roc_auc_score(session.label, session.distance_median), np.nan, np.nan],
        [f"Biological identity mean ({len(identity)})", "raw distance", roc_auc_score(identity.label, identity.distance_mean), np.nan, np.nan],
        [f"Biological identity median ({len(identity)})", "raw distance", roc_auc_score(identity.label, identity.distance_median), np.nan, np.nan],
    ]
    auc_table = pd.DataFrame(auc_rows, columns=["분석 단위", "지표", "AUC", "95% CI low", "95% CI high"])
    for col in ["AUC", "95% CI low", "95% CI high"]:
        auc_table[col] = auc_table[col].map(lambda x: fnum(x, 3))

    pair_auc_rows = []
    for group in ["ACLD", "ACLR"]:
        rows = df[df.group.isin(["HA", group])]
        sessions = session[session.group.isin(["HA", group])]
        pair_auc_rows.append({
            "비교": f"{group} vs HA",
            "stride distance AUC": fnum(roc_auc_score((rows.group != "HA").astype(int), rows.mahal_dist), 3),
            "session-mean distance AUC": fnum(roc_auc_score((sessions.group != "HA").astype(int), sessions.distance_mean), 3),
            "session-median distance AUC": fnum(roc_auc_score((sessions.group != "HA").astype(int), sessions.distance_median), 3),
        })

    group_stats = df.groupby("group", sort=False).agg(
        sessions=("subject_id", "nunique"), identities=("identity_id", "nunique"), strides=("mahal_dist", "size"),
        d_mean=("mahal_dist", "mean"), d_sd=("mahal_dist", "std"), d_median=("mahal_dist", "median"),
        d_q1=("mahal_dist", lambda x: x.quantile(.25)), d_q3=("mahal_dist", lambda x: x.quantile(.75)),
        score_mean=("impairment_score", "mean"), score_sd=("impairment_score", "std"),
        score_median=("impairment_score", "median"), score_zero=("impairment_score", lambda x: (x == 0).mean()),
    ).reset_index().rename(columns={"group": "그룹", "sessions": "세션", "identities": "identity", "strides": "stride"})
    for col in ["d_mean", "d_sd", "d_median", "d_q1", "d_q3", "score_mean", "score_sd", "score_median"]:
        group_stats[col] = group_stats[col].map(lambda x: fnum(x, 3))
    group_stats["score_zero"] = group_stats["score_zero"].map(lambda x: f"{100*x:.1f}%")

    condition = df.groupby(["group", "speed"], as_index=False).agg(
        strides=("mahal_dist", "size"), sessions=("subject_id", "nunique"),
        distance_mean=("mahal_dist", "mean"), distance_median=("mahal_dist", "median"),
        score_mean=("impairment_score", "mean"), score_median=("impairment_score", "median"),
    )
    for col in ["distance_mean", "distance_median", "score_mean", "score_median"]:
        condition[col] = condition[col].map(lambda x: fnum(x, 3))
    side_condition = df.groupby(["group", "side_basis", "actual_leg"], as_index=False).agg(
        strides=("mahal_dist", "size"), sessions=("subject_id", "nunique"),
        distance_median=("mahal_dist", "median"), score_median=("impairment_score", "median"),
    )
    for col in ["distance_median", "score_median"]:
        side_condition[col] = side_condition[col].map(lambda x: fnum(x, 3))

    outliers = df.groupby(["subject_id", "group"], as_index=False).agg(
        n=("mahal_dist", "size"), distance_mean=("mahal_dist", "mean"),
        distance_median=("mahal_dist", "median"), distance_max=("mahal_dist", "max"),
        score_mean=("impairment_score", "mean"), score_max=("impairment_score", "max"),
    ).sort_values("distance_mean", ascending=False).head(12)
    for col in ["distance_mean", "distance_median", "distance_max", "score_mean", "score_max"]:
        outliers[col] = outliers[col].map(lambda x: fnum(x, 3))
    ha12_removed = df[df.subject_id != "HA12"]
    ha12_auc = roc_auc_score(ha12_removed.label, ha12_removed.mahal_dist)
    ha12_mean = df.loc[df.subject_id == "HA12", "mahal_dist"].mean()
    ha_mean_full = df.loc[df.group == "HA", "mahal_dist"].mean()
    ha_mean_without = ha12_removed.loc[ha12_removed.group == "HA", "mahal_dist"].mean()

    test_auc = roc_auc_score(test_df.label, test_df.mahal_dist)
    test_score_auc = roc_auc_score(test_df.label, test_df.impairment_score)
    pilot = trials[trials.phase == "TEST 3-fold"]
    full = trials[trials.phase == "FULL 5-fold"]
    pilot_best = pilot.loc[pilot.value.idxmax()]
    full_best = full.loc[full.value.idxmax()]
    pilot_params = trial_params[int(pilot_best.number)]
    full_params = trial_params[int(full_best.number)]

    fold_display = folds.copy()
    fold_display.columns = ["Fold", "검증 stride", "검증 session", "검증 identity", "train/val identity overlap", "AUC", "μ_HA 추정", "σ_HA 추정", "0 floor"]
    for col in ["AUC", "μ_HA 추정", "σ_HA 추정"]:
        fold_display[col] = fold_display[col].map(lambda x: fnum(x, 3))
    fold_display["0 floor"] = fold_display["0 floor"].map(lambda x: f"{100*x:.1f}%")

    lineage = pd.DataFrame([
        [mtime(OOF_TEST), "Test default OOF", "oof_results_test.parquet", f"1,013 rows; distance AUC {test_auc:.4f}; score AUC {test_score_auc:.4f}"],
        [str(pilot.datetime_start.min()), "Test Optuna", "trials #0–12 in shared DB", f"3-fold / 9 sessions; best #{int(pilot_best.number)} = {pilot_best.value:.6f}"],
        [mtime(FEATURES), "Full preprocessing", "mahalanobis_features.parquet", "9,540 stride rows; 7,979 waveform features"],
        [mtime(OOF), "Full default OOF", "oof_results.parquet + 02_evaluation_report.md", f"zscore + kaiser + all; AUC {stride_auc:.6f}"],
        [str(full.datetime_start.min()), "Full Optuna", "trials #13–83 in same shared DB", f"best full-era #{int(full_best.number)} = {full_best.value:.6f}; no OOF saved"],
        [mtime(OPTUNA_JSON), "JSON export", "01_optuna_best_params.json", "Exports global study best: test trial #12, not full-era best"],
        [mtime(SHAP_DIR / "summary_plot.png"), "Full SHAP proxy run", "summary + 10 current waterfall PNGs", "Loads contaminated JSON; 5 stale test PNGs remain in directory"],
    ], columns=["시각", "실행", "산출물", "감사 판정"])

    figures = make_figures(df, folds, trials)
    representative_images = []
    for name, label in [
        ("summary_plot.png", "Global channel summary"),
        ("ACLR24_waterfall.png", "ACLR24 channel aggregation"),
        ("ACLD40_waterfall.png", "ACLD40 channel aggregation"),
    ]:
        path = SHAP_DIR / name
        if path.exists():
            representative_images.append(
                f'<figure><img src="{png_data_uri(path)}" alt="{esc(label)}"><figcaption>{esc(label)} · {esc(mtime(path))}</figcaption></figure>'
            )

    stale_shap = sorted(
        p.name for p in SHAP_DIR.glob("*.png") if p.stat().st_mtime < FEATURES.stat().st_mtime
    )
    artifact_rows = []
    artifact_paths = [
        SANDBOX / "01_plan.md",
        SANDBOX / "run_pipeline.py",
        *(SANDBOX / "scripts" / name for name in [
            "00_extract_subset.py", "01_data_preprocessing.py", "02_mahalanobis_pipeline.py",
            "03_optuna_optimization.py", "04_shap_analysis.py",
        ]),
        FEATURES, OOF, OOF_TEST, OPTUNA_DB, OPTUNA_JSON, EVAL_MD, ID_CSV, PAIRING,
        SHAP_DIR / "summary_plot.png", SHAP_DIR / "ACLR24_waterfall.png", SHAP_DIR / "ACLD40_waterfall.png",
    ]
    for path in artifact_paths:
        artifact_rows.append([str(path.relative_to(ROOT)), f"{path.stat().st_size/1024/1024:.2f} MB", mtime(path), sha256_short(path)])
    artifact_table = pd.DataFrame(artifact_rows, columns=["파일", "크기", "수정 시각", "SHA-256 앞 16자리"])
    environment_table = pd.DataFrame([
        ["OS", platform.platform()],
        ["Python", sys.version.split()[0]],
        ["NumPy", np.__version__],
        ["pandas", pd.__version__],
        ["scikit-learn", sklearn.__version__],
        ["PyArrow", pyarrow.__version__],
        ["Matplotlib", matplotlib.__version__],
    ], columns=["구성요소", "버전"])

    css = """
    :root{--ink:#17212b;--muted:#5e6b78;--paper:#f4f7f8;--card:#fff;--line:#dce4e8;--critical:#9f1239;--high:#b45309;--medium:#8a6d1d;--low:#2563eb}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Noto Sans KR","Segoe UI",sans-serif;line-height:1.68;font-size:17px}
    header{background:linear-gradient(135deg,#122635,#204a5f);color:#fff;padding:64px max(5vw,28px) 52px}header .eyebrow{letter-spacing:.13em;text-transform:uppercase;color:#a9d7e9;font-weight:700;font-size:.82rem}h1{font-size:clamp(2.3rem,5vw,4.6rem);line-height:1.06;margin:.3em 0 .2em}header p{max-width:980px;font-size:1.18rem;color:#d8e8ef}
    main{max-width:1320px;margin:0 auto;padding:34px 28px 90px}nav{position:sticky;top:0;z-index:5;background:rgba(255,255,255,.96);border-bottom:1px solid var(--line);overflow-x:auto;white-space:nowrap;padding:12px max(3vw,20px)}nav a{color:#294b5c;text-decoration:none;margin-right:20px;font-size:.9rem;font-weight:650}
    section{background:var(--card);border:1px solid var(--line);border-radius:18px;margin:26px 0;padding:clamp(24px,4vw,48px);box-shadow:0 10px 26px rgba(20,40,50,.045)}h2{font-size:2rem;line-height:1.2;margin:0 0 22px}h3{font-size:1.28rem;margin-top:32px}p,li{max-width:1050px}.lead{font-size:1.18rem;color:#314651}.muted{color:var(--muted)}code{background:#edf2f4;border-radius:5px;padding:.1em .35em;font-size:.88em}.formula{font-family:Georgia,serif;background:#f5f9fa;border-left:4px solid #39758f;padding:18px 22px;margin:18px 0;font-size:1.08rem;overflow:auto}
    .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px}.metric{border:1px solid var(--line);border-radius:14px;padding:18px;background:#fbfdfd}.metric .value{font-size:2rem;font-weight:800;letter-spacing:-.03em}.metric .label{color:var(--muted);font-size:.86rem}.callout{border-radius:12px;padding:18px 20px;margin:18px 0;border-left:5px solid}.critical{background:#fff1f2;border-color:var(--critical)}.high{background:#fff7ed;border-color:var(--high)}.medium{background:#fffbeb;border-color:var(--medium)}.info{background:#eff6ff;border-color:var(--low)}.badge{display:inline-block;color:#fff;border-radius:999px;padding:3px 10px;font-size:.75rem;font-weight:800;letter-spacing:.04em;margin-right:8px}.badge.critical{background:var(--critical)}.badge.high{background:var(--high)}.badge.medium{background:var(--medium);color:#fff}.badge.low{background:var(--low)}
    .table-wrap{overflow:auto;max-height:620px;border:1px solid var(--line);border-radius:12px;margin:15px 0}.data-table{border-collapse:collapse;width:100%;font-size:.86rem}.data-table th,.data-table td{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}.data-table th{position:sticky;top:0;background:#edf4f6;z-index:1}.data-table tr:nth-child(even){background:#fafcfc}.plot{overflow:auto;margin:20px 0}.plot svg{display:block;max-width:100%;height:auto;margin:auto}.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:20px}.flow{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:24px 0}.flow span{background:#e8f1f4;border:1px solid #cbdde4;border-radius:10px;padding:12px 15px;font-weight:700}.flow b{color:#7293a1}.status-list li{margin:.7em 0}.image-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}.image-grid figure{margin:0;border:1px solid var(--line);border-radius:12px;padding:10px;background:#fafcfc}.image-grid img{width:100%;height:auto;display:block}.image-grid figcaption{font-size:.8rem;color:var(--muted);padding:8px}.source{font-size:.79rem;color:#536874;border-top:1px dashed var(--line);padding-top:10px;margin-top:18px}.verdict{font-weight:800}.allow{color:#166534}.ban{color:#9f1239}
    @media(max-width:650px){body{font-size:15px}main{padding:18px 12px 60px}section{border-radius:12px;padding:22px 16px}.grid2{grid-template-columns:1fr}}
    @media print{nav{display:none}body{background:#fff;font-size:11pt}header{padding:28px;color:#000;background:#fff;border-bottom:2px solid #000}header p,header .eyebrow{color:#222}main{max-width:none;padding:0}section{break-inside:avoid;box-shadow:none;border:0;border-bottom:1px solid #aaa;border-radius:0;margin:0;padding:22px 0}.table-wrap{max-height:none;overflow:visible}.data-table th{position:static}.plot svg{max-height:500px}.image-grid img{max-height:420px;object-fit:contain}}
    """

    html_doc = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light"><link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg'/%3E">
<title>0702 Mahalanobis 실험 기술 감사</title><style>{css}</style></head><body>
<header><div class="eyebrow">Evidence-backed technical audit · 2026-07-02</div><h1>0702 Mahalanobis<br>실험 정밀 감사</h1>
<p>코드, 데이터 계보, 기본 OOF, Optuna SQLite, SHAP PNG를 실제 파일에서 교차검증한 보고서입니다. 반복 stride를 사람 수로 오인하지 않으며, 확인되지 않은 성능과 임상적 의미를 분리합니다.</p></header>
<nav>{''.join(f'<a href="#{i}">{i}. {label}</a>' for i,label in [(1,'요약'),(2,'설계'),(3,'데이터'),(4,'수학'),(5,'계보'),(6,'성능'),(7,'타당성'),(8,'SHAP'),(9,'결론'),(10,'개선'),(11,'재현성')])}</nav>
<main>
<section id="1"><h2>1. Executive Summary</h2>
<div class="cards">
<div class="metric"><div class="value">{stride_auc:.3f}</div><div class="label">재현 가능한 full default OOF distance AUC</div></div>
<div class="metric"><div class="value">{pilot_best.value:.3f}</div><div class="label">test 9-session Optuna 최고값 · 최종 성능 아님</div></div>
<div class="metric"><div class="value">{full_best.value:.3f}</div><div class="label">full-era 탐색 최고 목적값 · 비-nested</div></div>
<div class="metric"><div class="value">{confirmed_split}/{len(paired_present)}</div><div class="label">ID-confirmed ACLD/ACLR pairs split across folds</div></div>
<div class="metric"><div class="value">{100*inventory['null_rate']:.1f}%</div><div class="label">waveform feature cell missingness</div></div>
</div>
<ol class="status-list">
<li><strong>신뢰 가능한 사실:</strong> full 기본 설정 OOF raw-distance AUC는 <strong>{stride_auc:.4f}</strong>이며 subject-mean AUC는 <strong>{roc_auc_score(session.label, session.distance_mean):.4f}</strong>입니다. 둘 다 사실상 chance 수준입니다.</li>
<li><strong>Critical:</strong> JSON의 <strong>0.7716</strong>은 full 5-fold가 아니라 9-session test subset의 3-fold trial #12입니다. 같은 DB/study에 test와 full trial이 섞였습니다.</li>
<li><strong>Critical:</strong> 확인된 종단쌍 {len(paired_present)}개 중 {confirmed_split}개가 session-ID GroupKFold의 서로 다른 fold에 배치되었습니다. session stride overlap은 없지만 biological identity 독립성이 깨집니다.</li>
<li><strong>High:</strong> HA12의 mean distance는 {ha12_mean:.1f}로 HA 평균을 지배합니다. HA12를 제외하는 민감도 계산에서 distance AUC는 {stride_auc:.3f}→{ha12_auc:.3f}, HA mean은 {ha_mean_full:.2f}→{ha_mean_without:.2f}로 변합니다. 이는 제외 근거가 아니라 불안정성 증거입니다.</li>
<li><strong>High:</strong> SHAP은 원 Mahalanobis 모델이 아닌 in-sample XGBoost proxy 설명이며, 결측 처리도 OOF 파이프라인과 다릅니다. 인과적·임상적 기여도로 사용할 수 없습니다.</li>
</ol>
<div class="grid2"><div class="callout info"><span class="badge low">사용 가능</span><strong>현재 산출물이 보여주는 것</strong><br>기본 설정에서 계산된 HA-referenced multivariate deviation의 OOF 분포와 그 불안정성.</div>
<div class="callout critical"><span class="badge critical">사용 금지</span><strong>현재 산출물이 보여주지 못하는 것</strong><br>0.7716의 full-data 성능, 독립 검증된 최적 모델, 임상적 impairment/severity, causal sensor contribution.</div></div>
</section>

<section id="2"><h2>2. 실험 질문과 구현 설계</h2>
<p class="lead">질문은 “각 stride의 79-channel waveform이 training-fold HA 분포에서 얼마나 멀리 있는가”입니다. 구현은 분류 확률이 아니라 정상 참조 거리입니다.</p>
<div class="flow"><span>raw trial</span><b>→</b><span>heel-strike stride</span><b>→</b><span>79 × 101 waveform</span><b>→</b><span>HA-only scaler</span><b>→</b><span>HA-only PCA</span><b>→</b><span>HA-only MCD</span><b>→</b><span>D<sub>M</sub></span><b>→</b><span>clipped fold z-score</span></div>
<p>한 행은 한 trial의 한쪽 발에서 연속 heel strike 사이를 자른 <strong>stride-leg row</strong>입니다. 앞뒤 2 stride를 제거하고 각 채널을 101점으로 선형 보간합니다. 따라서 9,540행은 9,540명이 아니라 92 session에서 반복 측정된 stride입니다.</p>
<div class="callout medium"><span class="badge medium">설계 주의</span>orientation 28개 성분도 일반 선형 보간됩니다. 이 값이 quaternion 구성요소라면 sign continuity, SLERP, unit-norm 재정규화가 없어 물리적 interpolation의 타당성을 별도 확인해야 합니다.</div>
<p class="source">근거: <code>01_data_preprocessing.py:7–17, 85–103, 146–204</code> · <code>02_mahalanobis_pipeline.py:95–137</code></p>
</section>

<section id="3"><h2>3. 데이터 구성과 분석 단위</h2>
<div class="cards"><div class="metric"><div class="value">{len(df):,}</div><div class="label">stride-leg rows</div></div><div class="metric"><div class="value">{df.subject_id.nunique()}</div><div class="label">session형 subject IDs</div></div><div class="metric"><div class="value">{df.identity_id.nunique()}</div><div class="label">suffix 기반 biological identities</div></div><div class="metric"><div class="value">{inventory['features']:,}</div><div class="label">waveform features</div></div><div class="metric"><div class="value">79 × 101</div><div class="label">channels × normalized points</div></div></div>
<h3>그룹별 규모와 분포</h3><div class="table-wrap">{table(group_stats)}</div>
<p><strong>identity 수 해석:</strong> ID 메타데이터로 확인된 ACLD/ACLR pair만 하나로 합쳤습니다. 25 HA + 26 confirmed ACL pairs + 14 ACLD-only sessions + 1 unresolved ACLR38 = <strong>66 analysis identities</strong>입니다. <code>ACLR38</code>을 suffix만으로 <code>ACLD38</code>과 임의 결합하지 않았습니다.</p>
<h3>특징 공간과 결측</h3><div class="table-wrap">{table(feature_types)}</div>
<div class="callout high"><span class="badge high">High</span>총 waveform cell {inventory['total_cells']:,}개 중 {inventory['null_count']:,}개({100*inventory['null_rate']:.2f}%)가 null입니다. 관절각도는 0%지만 IMU 계열은 각각 약 35.9%입니다. OOF와 SHAP의 결측 처리 순서가 달라 설명 대상이 같은 모델이라고 볼 수 없습니다.</div>
<h3>속도별 행·세션과 결과</h3><div class="table-wrap">{table(condition)}</div>
<h3>side basis × actual leg</h3><div class="table-wrap">{table(side_condition)}</div>
<div class="callout critical"><span class="badge critical">Critical metadata defect</span>데이터의 <code>{esc(', '.join(unexpected))}</code>는 <code>data/ID.csv</code>에 없고, 기대되는 <code>{esc(', '.join(absent_expected))}</code>는 features에 없습니다. 누락 ID는 코드에서 injured leg를 조용히 Right로 기본 지정하므로 <code>ACLR38</code> 108 stride의 side_basis가 검증되지 않았습니다.</div>
<p class="source">근거: parquet schema/count 직접 계산 · <code>01_data_preprocessing.py:58–82</code> · <code>data/ID.csv</code> · <code>data/processed/id_pairing_summary.csv</code></p>
</section>

<section id="4"><h2>4. 구현된 수학</h2>
<p>Fold <em>f</em>에서 training HA stride만 사용해 scaler, PCA, MCD를 순서대로 적합합니다. 검증 stride <strong>x</strong>의 구현식은 다음과 같습니다.</p>
<div class="formula">x<sub>s</sub> = scaler<sub>HA,train,f</sub>(x)<br>z = PCA<sub>HA,train,f</sub>(x<sub>s</sub>)<br>D<sub>M</sub>(x) = √[(z − μ<sub>MCD,f</sub>)ᵀ pinv(Σ<sub>MCD,f</sub>)(z − μ<sub>MCD,f</sub>)]<br>score(x) = max(0, [D<sub>M</sub>(x) − mean(D<sub>HA,train,f</sub>)] / sd(D<sub>HA,train,f</sub>))</div>
<p><strong>구체적 예:</strong> 어떤 fold의 HA training distance 평균이 35, SD가 40이면 distance 55의 stride는 score 0.50입니다. distance 25는 z=-0.25이지만 clipping 후 0이 되어 “평균보다 정상에 가까운 정도” 정보가 사라집니다.</p>
<ul>
<li>PCA 상한 <code>min(n_HA_strides // 5, 100)</code>은 사람 수가 아니라 반복 stride 수를 독립 표본처럼 사용합니다. 실제 독립 HA는 약 20명/fold이므로 공분산 안정성을 과대평가합니다.</li>
<li>MCD support는 고정 0.75가 아니라 <code>max(floor(0.75n), k+1)/n</code>으로 상향 조정됩니다. 실패하면 empirical covariance로 바뀌지만 산출물에는 fallback 여부가 기록되지 않습니다.</li>
<li><code>pinv</code>는 singular/ill-conditioned covariance에도 수치를 반환하지만, 조건수나 rank가 저장되지 않아 안정성을 확인할 수 없습니다.</li>
<li><code>mahalanobis</code>와 <code>squared_mahalanobis</code>는 비음수 거리의 단조변환이므로 같은 split의 ROC-AUC 탐색 축으로 중복입니다.</li>
</ul>
<p class="source">근거: <code>02_mahalanobis_pipeline.py:56–90, 161–218</code></p>
</section>

<section id="5"><h2>5. 결과 계보: 무엇이 언제 생성됐는가</h2>
<div class="table-wrap">{table(lineage)}</div><div class="plot">{figures['optuna']}</div>
<div class="callout critical"><span class="badge critical">Root cause</span><code>test_mode</code>와 full mode가 동일한 SQLite 경로, study name, <code>load_if_exists=True</code>를 공유합니다. 이후 JSON은 phase가 아니라 전역 <code>study.best_trial</code>을 저장합니다. 따라서 test trial #12의 0.771564가 full run 후에도 “best”로 남았습니다.</div>
<h3>두 Optuna 최고값의 정확한 의미</h3>
<div class="grid2"><div class="callout high"><strong>0.771564 · trial #{int(pilot_best.number)}</strong><br>1,013 rows / 9 sessions / 3-fold <strong>test subset</strong>의 튜닝 목적값.<br><code>{esc(json.dumps(pilot_params, ensure_ascii=False))}</code></div>
<div class="callout medium"><strong>{full_best.value:.6f} · trial #{int(full_best.number)}</strong><br>full-era trials #13–83 중 최고지만 pilot history가 섞인 TPE와 동일 5-fold 반복 선택값. 독립 검증 아님.<br><code>{esc(json.dumps(full_params, ensure_ascii=False))}</code></div></div>
<h3>9-session test 3-fold의 class 구성</h3><div class="table-wrap">{table(test_fold_table)}</div>
<div class="callout high"><span class="badge high">High</span>test fold 중 하나는 HA가 없는 single-class validation fold입니다. 해당 fold AUC는 정의되지 않지만 전체 pooled OOF objective는 계산됩니다. 따라서 0.7716은 작은 pilot일 뿐 아니라 fold별 검증 가능성도 균일하지 않습니다.</div>
<p><strong>최적 설정 OOF는 없습니다.</strong> step 02만 기본 설정 OOF/Markdown을 저장하고, step 03은 objective scalar와 JSON/DB만 저장합니다. 따라서 현재 파일에서 직접 재계산 가능한 full 기본 산출물은 OOF <strong>{stride_auc:.4f}</strong>입니다. Optuna trial은 code/data hash와 OOF를 저장하지 않아 exact replay가 확립되지 않았습니다.</p>
<p class="source">근거: <code>02_mahalanobis_pipeline.py:302–337</code> · <code>03_optuna_optimization.py:132–191</code> · SQLite trial timestamps 직접 조회</p>
</section>

<section id="6"><h2>6. 성능 결과</h2>
<div class="grid2"><div class="plot">{figures['roc']}</div><div class="plot">{figures['folds']}</div></div>
<p>95% CI는 stride 단순 bootstrap이 아니라 <strong>metadata-confirmed biological identity cluster bootstrap 1,000회</strong>입니다. resampling 단위는 identity지만 AUC estimand 자체는 각 cluster의 모든 stride를 연결하므로 stride 수가 많은 identity가 더 큰 가중을 갖습니다. 아래 session/identity mean도 trial-balanced가 아니라 해당 단위의 모든 stride 평균입니다.</p>
<div class="table-wrap">{table(auc_table)}</div>
<h3>ACLD와 ACLR를 분리해 HA와 비교</h3><div class="table-wrap">{table(pd.DataFrame(pair_auc_rows))}</div>
<p>ACLD vs HA는 raw distance 방향이 오히려 반대이고, ACLR vs HA만 약한 분리 신호가 있습니다. 하나의 “ACL impairment” 축으로 합치는 해석은 지지되지 않습니다.</p>
<div class="plot">{figures['distributions']}</div><div class="plot">{figures['subjects']}</div>
<h3>Fold calibration과 합산 문제</h3><div class="table-wrap">{table(fold_display)}</div>
<p>OOF에는 fold, PCA k, MCD rank/condition, μ/σ가 저장되지 않습니다. 위 μ/σ는 score&gt;0 행에서 <code>distance = μ + σ×score</code>를 역산한 값입니다. μ는 {folds.mu_inferred.min():.1f}–{folds.mu_inferred.max():.1f}, σ는 {folds.sigma_inferred.min():.1f}–{folds.sigma_inferred.max():.1f}로 크게 달라 raw distance와 clipped score 모두 fold 간 직접 비교가 약합니다. 실제로 test OOF에서는 raw distance AUC {test_auc:.3f}가 score AUC {test_score_auc:.3f}로 뒤집힙니다.</p>
<h3>극단값과 민감도</h3><div class="table-wrap">{table(outliers)}</div>
<div class="callout high"><span class="badge high">High</span>HA12 제외는 사후적이며 정당한 분석이 아닙니다. 다만 한 identity가 전체 HA mean과 AUC 방향을 바꿀 수 있다는 사실은 robust reference가 최종 HA calibration에서 충분히 유지되지 않았음을 보여줍니다. MCD 이후 μ/σ는 모든 HA-train distance의 일반 mean/SD입니다.</div>
</section>

<section id="7"><h2>7. 방법론적 타당성 감사</h2>
<h3>누수와 독립성</h3>
<div class="callout critical"><span class="badge critical">Critical</span>session <code>subject_id</code> overlap은 fold마다 0이지만, ID-confirmed longitudinal pair {len(paired_present)}개 중 {confirmed_split}개가 서로 다른 fold에 있습니다({confirmed_same}개만 같은 fold). Fold별 biological identity train/validation overlap은 {', '.join(map(str, folds.identity_overlap))}개입니다.</div>
<p>모델 적합은 HA train만 사용하므로 ACL train session이 scaler/PCA/MCD에 직접 들어가지는 않습니다. 그러나 동일 환자의 ACLD/ACLR 시점이 tuning objective와 불확실성 계산에 반복 기여하고, identity 독립성 가정을 위반합니다. 최종 주장을 위해서는 identity 단위 outer split 안에서만 Optuna를 수행하는 nested CV 또는 잠근 independent holdout이 필요합니다.</p>
<h3>반복 stride와 가중</h3><p>stride 수가 많은 session이 PCA/MCD, AUC, SHAP에 더 큰 가중을 갖습니다. 통계적 불확실성은 subject/identity가 단위여야 하며, trial→subject 균형 집계 또는 hierarchical model이 필요합니다.</p>
<h3>점수의 의미와 바닥효과</h3><p>score=0 비율은 HA {100*(df.loc[df.group=='HA'].impairment_score==0).mean():.1f}%, ACLD {100*(df.loc[df.group=='ACLD'].impairment_score==0).mean():.1f}%, ACLR {100*(df.loc[df.group=='ACLR'].impairment_score==0).mean():.1f}%입니다. clipping은 정상 쪽 차이를 모두 버리고, fold별 다른 기준을 하나의 수치처럼 보이게 합니다.</p>
<div class="callout info"><span class="badge low">명칭 권고</span>외부 임상 anchor, test-retest, construct validity, responsiveness가 없으므로 <strong>HA-referenced gait deviation score</strong> 또는 <strong>gait normality/deviation score</strong>가 정직합니다. “impairment”, “severity”, “recovery”는 현재 근거 범위를 넘습니다.</div>
</section>

<section id="8"><h2>8. SHAP 감사</h2>
<div class="callout critical"><span class="badge critical">탐색적 XGBoost proxy 설명</span>원 Mahalanobis/MCD 모델의 직접 설명이 아니며, OOF·인과·임상적 기여도가 아닙니다.</div>
<ul>
<li>전체 HA로 새 scaler/PCA/MCD를 적합해 target score를 다시 만들고, 전체 9,540 stride에 XGBoost proxy를 fit합니다. 동일 행에서 Pearson R 하나만 계산하며 R², MAE, calibration, grouped holdout이 없습니다. R 값도 파일에 저장되지 않았습니다.</li>
<li><code>oof_df</code>는 로드·전달되지만 함수 내부에서 사용되지 않습니다. SHAP target은 OOF score가 아닙니다.</li>
<li>OOF는 scaling 후 NaN→0, SHAP은 raw NaN→0 후 scaling입니다. waveform cell 결측 {100*inventory['null_rate']:.2f}%이므로 차이가 작지 않으며 group-dependent acquisition artifact가 feature rank에 섞일 수 있습니다.</li>
<li>global summary는 101개 시점의 |SHAP|을 채널별로 합쳐 방향과 timing을 제거합니다. individual “waterfall”은 subject stride 평균 후 101개 signed SHAP 합을 그린 막대그래프로, base value와 additive total이 있는 실제 waterfall이 아닙니다.</li>
<li>현재 PNG는 test waterfall 5개, full summary 1개, full waterfall 10개가 같은 폴더에 혼재합니다. 기존 파일을 정리하지 않는 구현 때문입니다.</li>
</ul>
<div class="image-grid">{''.join(representative_images)}</div>
<p><strong>시각 품질:</strong> PNG는 1500×1200 또는 1500×1050 해상도지만, 한글 폰트가 사각형 glyph로 깨져 있습니다. 대표 그림은 02:17 full run 파일만 사용했고, 01:17 stale test 파일(<code>{esc(', '.join(stale_shap))}</code>)은 제외했습니다.</p>
<p class="source">근거: <code>04_shap_analysis.py:67–120, 140–238, 271–320, 328–363</code> · PNG 직접 시각 검사</p>
</section>

<section id="9"><h2>9. 결론과 표현 가이드</h2>
<h3 class="allow">논문·발표에서 사용할 수 있는 문장</h3>
<blockquote>“HA training strides를 참조한 PCA–MCD Mahalanobis deviation을 탐색적으로 계산했다. 기본 full-data GroupKFold OOF raw-distance AUC는 0.499였고, biological-identity cluster bootstrap과 subject-level 집계에서도 불확실성이 컸다. 따라서 현재 점수는 임상적 손상도가 아니라 정상 참조 편차의 탐색적 지표로 해석한다.”</blockquote>
<h3 class="ban">사용하면 안 되는 문장</h3>
<ul><li>“최적화된 모델의 검증 AUC는 0.772였다.” — test 9-session trial이고 최적 OOF가 없음.</li><li>“Mahalanobis impairment score가 ACL 손상도를 정량화한다.” — 임상 anchor/construct validation 없음.</li><li>“SHAP이 특정 센서 또는 gait-cycle 시점의 임상적 원인을 규명했다.” — proxy/in-sample/channel aggregation이며 timing과 causality를 제공하지 않음.</li><li>“GroupKFold로 환자 누수를 완전히 막았다.” — session ID만 격리했고 종단 biological identity는 분리됨.</li></ul>
<p class="verdict">현재 실험은 “이 설계와 데이터에서 정상 참조 거리 모델이 불안정하며 알려진 그룹을 일관되게 구분하지 못했다”는 결과를 보여줍니다. 실패를 숨기지 않는 것이 다음 설계를 위한 유효한 결론입니다.</p>
</section>

<section id="10"><h2>10. 개선 우선순위</h2>
<div class="table-wrap">{table(pd.DataFrame([
['P0','Test/full Optuna 격리','mode·dataset hash가 포함된 study/DB로 분리하고 full study를 새로 시작','0.7716 계보 오염 제거'],
['P0','Biological-identity outer CV','ACLD/ACLR pair를 동일 outer fold에 고정; ACLR38/36 메타데이터 해결','identity leakage와 side 오류 제거'],
['P0','Nested optimization + locked OOF','outer fold 내부에서만 tuning; 최적 OOF와 fold metadata 저장','선택 편향 없는 성능 추정'],
['P0','결측 원인/처리 통일','센서별 missingness audit; train-fold imputer; OOF/SHAP 동일 순서','acquisition artifact 감소'],
['P1','표본 균형 reference','cycle→trial→subject 균형 평균 또는 hierarchical weighting','stride 수가 많은 session의 과대가중 방지'],
['P1','안정적 covariance','shrinkage covariance/regularized Mahalanobis 비교, rank·condition 저장','pinv/MCD 불안정성 진단'],
['P1','점수 calibration 재설계','identity-safe HA LOO/outer-fold calibration, clipping 없는 raw z도 보존','fold 비교성과 바닥효과 개선'],
['P1','Cluster uncertainty','identity bootstrap, paired ACLD→ACLR analysis, sensitivity protocol 사전 정의','독립 단위에 맞는 CI'],
['P2','Proxy fidelity 검증','subject-grouped holdout R²/MAE/rank agreement; OOF target만 설명','SHAP 신뢰 범위 명시'],
['P2','시각/산출물 분리','run-specific SHAP directory, 폰트 지정, 진짜 waterfall/시점 heatmap','혼합 계보와 읽기 오류 제거'],
], columns=['우선순위','조치','필요 변경','기대 효과']))}</div>
</section>

<section id="11"><h2>11. 재현성 부록</h2>
<h3>보고서 생성 환경</h3><div class="table-wrap">{table(environment_table)}</div>
<h3>감사 입력 파일 fingerprint</h3><div class="table-wrap">{table(artifact_table)}</div>
<h3>실행 명령</h3><div class="formula"><code>cd {esc(ROOT)}</code><br><code>.venv/bin/python 0702_Mahalanobis/scripts/05_generate_detailed_report.py</code></div>
<h3>검증 가능한 산출물 계보</h3><ul><li><code>mahalanobis_features.parquet</code> → 기본 hardcoded step 02 → <code>oof_results.parquet</code> + Markdown.</li><li>동일 features → step 03 shared study → DB scalar objectives + global-best JSON. 최적 OOF 없음.</li><li>features + contaminated JSON → full-data refit score → in-sample XGBoost proxy → SHAP PNG.</li></ul>
<h3>확정할 수 없는 내용</h3><ul><li>최적 full configuration의 저장된 OOF prediction과 독립 성능.</li><li>DB Optuna scalar의 exact replay. trial별 code hash, dataset hash, OOF, fold metadata가 저장되지 않음.</li><li>각 fold의 실제 PCA k, MCD fallback 여부, covariance rank/condition. 현재 산출물에 저장되지 않음.</li><li>SHAP proxy의 out-of-sample fidelity와 저장된 Pearson R.</li><li><code>ACLR38</code>의 실제 injured leg 및 <code>ACLR36</code> 누락 원인.</li><li>HA12가 생물학적 극단값인지 acquisition/preprocessing 오류인지.</li></ul>
<p class="muted">보고서 생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} KST · 외부 CDN/네트워크 의존 없음 · 새 차트는 inline SVG, 기존 SHAP PNG는 base64 내장.</p>
</section>
</main></body></html>"""

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(html_doc, encoding="utf-8")
    print(f"[05] report written: {OUTPUT}")
    print(f"[05] bytes: {OUTPUT.stat().st_size:,}")
    print(f"[05] default full OOF AUC: {stride_auc:.6f}")
    print(f"[05] cluster bootstrap 95% CI: [{stride_ci[0]:.3f}, {stride_ci[1]:.3f}]")
    print(f"[05] confirmed pair split: {confirmed_split}/{len(paired_present)}")


if __name__ == "__main__":
    main()
