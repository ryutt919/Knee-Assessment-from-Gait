"""Pair-aware validation, uncertainty, comparators, and reporting."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .components import (
    EPSILON,
    WaveformDataset,
    aggregate_cycle_waveforms,
    hierarchical_bootstrap_matrix,
    load_cycle_waveforms,
    raw_distances,
)
from .scorer import GaitNormalityScorer


def pair_aware_repeated_splits(
    metadata: pd.DataFrame,
    n_splits: int = 5,
    n_repeats: int = 10,
    seed: int = 42,
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    identity_table = metadata[["identity_id"]].drop_duplicates().sort_values("identity_id")
    identities = identity_table["identity_id"].to_numpy(dtype=str)
    labels = np.where(np.char.startswith(identities, "ACL::"), "ACL", "HA")
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=seed,
    )
    row_identities = metadata["identity_id"].to_numpy(dtype=str)
    for split_idx, (identity_train, identity_test) in enumerate(splitter.split(identities, labels)):
        repeat = split_idx // n_splits + 1
        fold = split_idx % n_splits + 1
        train_ids = set(identities[identity_train])
        test_ids = set(identities[identity_test])
        if train_ids & test_ids:
            raise RuntimeError("pair-aware split leaked a biological identity")
        train_rows = np.flatnonzero(np.isin(row_identities, list(train_ids)))
        test_rows = np.flatnonzero(np.isin(row_identities, list(test_ids)))
        yield repeat, fold, train_rows, test_rows


def _comparator_training_features(
    scorer: GaitNormalityScorer,
    matrices: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    features = scorer.gvs_features(matrices)
    ha_indices = np.flatnonzero(groups == "HA")
    ha_matrices = matrices[ha_indices]
    for local_idx, row_idx in enumerate(ha_indices):
        keep = np.arange(len(ha_matrices)) != local_idx
        loo_mean = ha_matrices[keep].mean(axis=0)
        _, gvs = raw_distances(matrices[row_idx], loo_mean)
        features[row_idx] = np.log(np.maximum(gvs[0].ravel(), EPSILON))
    return features


def run_oof_validation(
    dataset: WaveformDataset,
    n_splits: int = 5,
    n_repeats: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    metadata = dataset.metadata.reset_index(drop=True)
    groups = metadata["group"].to_numpy(dtype=str)
    rows: list[pd.DataFrame] = []
    for repeat, fold, train_idx, test_idx in pair_aware_repeated_splits(
        metadata, n_splits=n_splits, n_repeats=n_repeats, seed=seed
    ):
        overlap = set(metadata.loc[train_idx, "identity_id"]) & set(metadata.loc[test_idx, "identity_id"])
        if overlap:
            raise RuntimeError(f"biological identity leakage: {sorted(overlap)}")
        scorer = GaitNormalityScorer().fit(
            dataset.matrices[train_idx],
            groups[train_idx],
            metadata.loc[train_idx, "subject_id"].to_numpy(dtype=str),
        )
        fold_scores = scorer.score_matrices(dataset.matrices[test_idx])

        train_features = _comparator_training_features(
            scorer, dataset.matrices[train_idx], groups[train_idx]
        )
        test_features = scorer.gvs_features(dataset.matrices[test_idx])
        train_labels = (groups[train_idx] != "HA").astype(int)
        comparator = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=5000,
                random_state=seed,
            ),
        )
        comparator.fit(train_features, train_labels)
        fold_scores["ridge_logit_margin"] = comparator.decision_function(test_features)
        fold_scores.insert(0, "identity_overlap", 0)
        fold_scores.insert(0, "n_test_identities", metadata.loc[test_idx, "identity_id"].nunique())
        fold_scores.insert(0, "n_train_identities", metadata.loc[train_idx, "identity_id"].nunique())
        fold_scores.insert(0, "fold", fold)
        fold_scores.insert(0, "repeat", repeat)
        fold_meta = metadata.loc[test_idx, ["identity_id", "subject_id", "group"]].reset_index(drop=True)
        fold_meta = fold_meta.rename(columns={"group": "session_group"})
        rows.append(pd.concat([fold_meta, fold_scores], axis=1))
    return pd.concat(rows, ignore_index=True)


def aggregate_oof_scores(oof: pd.DataFrame) -> pd.DataFrame:
    keys = ["identity_id", "subject_id", "session_group"]
    numeric = [
        col
        for col in oof.select_dtypes(include="number").columns
        if col not in {"repeat", "fold", "n_train_identities", "n_test_identities", "identity_overlap"}
    ]
    means = oof.groupby(keys, observed=True)[numeric].mean().reset_index()
    stability = (
        oof.groupby(keys, observed=True)["normality_score"]
        .std(ddof=1)
        .reset_index(name="normality_score_cv_sd")
    )
    return means.merge(stability, on=keys, how="left")


def score_reference_cohort(
    dataset: WaveformDataset,
) -> tuple[GaitNormalityScorer, pd.DataFrame, dict[str, GaitNormalityScorer]]:
    metadata = dataset.metadata.reset_index(drop=True)
    groups = metadata["group"].to_numpy(dtype=str)
    full_scorer = GaitNormalityScorer().fit(
        dataset.matrices,
        groups,
        metadata["subject_id"].to_numpy(dtype=str),
    )
    ha_scorers: dict[str, GaitNormalityScorer] = {}
    score_rows: list[pd.DataFrame] = []
    for idx, row in metadata.iterrows():
        scorer = full_scorer
        reference_mode = "all_HA_reference"
        if row["group"] == "HA":
            reference_mode = "leave_subject_out_HA_reference"
            keep = ~metadata["subject_id"].eq(row["subject_id"]).to_numpy()
            scorer = GaitNormalityScorer().fit(
                dataset.matrices[keep],
                groups[keep],
                metadata.loc[keep, "subject_id"].to_numpy(dtype=str),
            )
            ha_scorers[str(row["subject_id"])] = scorer
        scored = scorer.score_matrices(dataset.matrices[idx])
        scored.insert(0, "reference_mode", reference_mode)
        scored.insert(0, "session_group", row["group"])
        scored.insert(0, "subject_id", row["subject_id"])
        scored.insert(0, "identity_id", row["identity_id"])
        scored["n_cycles"] = int(dataset.cycle_counts[idx].sum())
        scored["n_trials"] = int(dataset.trial_counts[idx].sum())
        scored["top_waveform_regions"] = scorer.explain_matrix(dataset.matrices[idx])
        score_rows.append(scored)
    return full_scorer, pd.concat(score_rows, ignore_index=True), ha_scorers


def add_bootstrap_intervals(
    scores: pd.DataFrame,
    cycle_frame: pd.DataFrame,
    full_scorer: GaitNormalityScorer,
    ha_scorers: dict[str, GaitNormalityScorer],
    iterations: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    if iterations < 20:
        raise ValueError("at least 20 bootstrap iterations are required for a 95% interval")
    output = scores.copy()
    intervals: dict[str, tuple[float, float, float]] = {}
    for subject_idx, subject_id in enumerate(output["subject_id"].astype(str)):
        session = cycle_frame[cycle_frame["subject_id"].astype(str).eq(subject_id)]
        scorer = ha_scorers.get(subject_id, full_scorer)
        rng = np.random.default_rng(seed + subject_idx * 1009)
        values = np.empty(iterations, dtype=np.float64)
        for bootstrap_idx in range(iterations):
            matrix = hierarchical_bootstrap_matrix(session, rng)
            values[bootstrap_idx] = float(scorer.score_matrices(matrix)["normality_score"].iloc[0])
        intervals[subject_id] = (
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
            float(values.std(ddof=1)),
        )
    output["score_ci_low"] = output["subject_id"].map(lambda sid: intervals[str(sid)][0])
    output["score_ci_high"] = output["subject_id"].map(lambda sid: intervals[str(sid)][1])
    output["score_bootstrap_sd"] = output["subject_id"].map(lambda sid: intervals[str(sid)][2])
    output["bootstrap_iterations"] = iterations
    return output


def add_point_metric_explanations(
    scores: pd.DataFrame,
    features_path: str | Path,
    limit: int = 5,
) -> pd.DataFrame:
    features = pd.read_csv(features_path)
    meta_cols = {"subject_id", "group", "speed", "injured_leg"}
    numeric_cols = [
        col for col in features.columns if col not in meta_cols and pd.api.types.is_numeric_dtype(features[col])
    ]
    explanations: dict[str, str] = {}
    for _, score_row in scores.iterrows():
        subject_id = str(score_row["subject_id"])
        group = str(score_row["session_group"])
        subject_rows = features[features["subject_id"].astype(str).eq(subject_id)]
        candidates: list[dict[str, float | str]] = []
        for _, subject_row in subject_rows.iterrows():
            speed = str(subject_row["speed"])
            reference = features[(features["group"] == "HA") & (features["speed"] == speed)]
            if group == "HA":
                reference = reference[~reference["subject_id"].astype(str).eq(subject_id)]
            means = reference[numeric_cols].mean(axis=0, skipna=True)
            stds = reference[numeric_cols].std(axis=0, ddof=1, skipna=True)
            values = pd.to_numeric(subject_row[numeric_cols], errors="coerce")
            z_values = (values - means) / stds.where(stds > EPSILON)
            for feature, z_value in z_values.items():
                if np.isfinite(z_value):
                    candidates.append(
                        {
                            "speed": speed,
                            "feature": feature,
                            "z_vs_HA": round(float(z_value), 5),
                            "absolute_z": round(abs(float(z_value)), 5),
                        }
                    )
        candidates.sort(key=lambda item: float(item["absolute_z"]), reverse=True)
        explanations[subject_id] = json.dumps(candidates[:limit], ensure_ascii=False)
    output = scores.copy()
    output["top_point_metric_deviations"] = output["subject_id"].map(explanations)
    return output


def _cohen_d(patient: np.ndarray, healthy: np.ndarray) -> float:
    pooled = np.sqrt(
        ((len(patient) - 1) * patient.var(ddof=1) + (len(healthy) - 1) * healthy.var(ddof=1))
        / (len(patient) + len(healthy) - 2)
    )
    return float((patient.mean() - healthy.mean()) / pooled) if pooled > EPSILON else float("nan")


def demographic_sensitivity(scores: pd.DataFrame, id_path: str | Path) -> dict[str, Any]:
    demographics = pd.read_csv(id_path).rename(columns={"ID": "subject_id"})
    merged = scores.merge(demographics, on="subject_id", how="left")
    ha = merged[merged["session_group"] == "HA"].copy()
    result: dict[str, Any] = {"n_HA": int(len(ha)), "angular_waveform_score_only": True}
    for column in ("Age", "Weight", "Height"):
        values = pd.to_numeric(ha[column], errors="coerce")
        valid = values.notna() & ha["normality_score"].notna()
        if valid.sum() >= 5:
            rho, p_value = spearmanr(values[valid], ha.loc[valid, "normality_score"])
            result[column] = {"spearman_rho": float(rho), "p_value": float(p_value), "n": int(valid.sum())}
    sex = ha.assign(Sex=ha["Sex"].astype(str)).groupby("Sex", observed=True)["normality_score"].agg(["count", "mean", "std"])
    result["Sex"] = {
        str(index): {key: float(value) for key, value in row.items()}
        for index, row in sex.to_dict(orient="index").items()
    }
    return result


def build_validation_summary(
    scores: pd.DataFrame,
    oof: pd.DataFrame,
    oof_aggregate: pd.DataFrame,
    id_path: str | Path,
) -> dict[str, Any]:
    group_summary = (
        oof_aggregate.groupby("session_group", observed=True)["normality_score"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .round(6)
    )
    labels = (oof_aggregate["session_group"] != "HA").astype(int).to_numpy()
    aucs = {
        "normative_z_deviation": float(roc_auc_score(labels, oof_aggregate["z_deviation"])),
        "shrinkage_mahalanobis": float(roc_auc_score(labels, oof_aggregate["mahalanobis_distance"])),
        "ridge_logit_margin": float(roc_auc_score(labels, oof_aggregate["ridge_logit_margin"])),
    }
    subscore_aucs = {
        key: float(roc_auc_score(labels, oof_aggregate[f"{key}_z"]))
        for key in ("hip", "knee", "ankle", "slow", "normal", "fast", "bilateral", "asymmetry")
    }
    acl = oof_aggregate[oof_aggregate["session_group"] != "HA"]["z_deviation"].to_numpy()
    ha = oof_aggregate[oof_aggregate["session_group"] == "HA"]["z_deviation"].to_numpy()
    known_group_d = _cohen_d(acl, ha)
    primary_supported = not (
        abs(aucs["normative_z_deviation"] - 0.5) < 0.1 and abs(known_group_d) < 0.2
    )

    pre = oof_aggregate[oof_aggregate["session_group"] == "ACLD"][
        ["identity_id", "normality_score"]
    ].rename(columns={"normality_score": "ACLD_score"})
    post = oof_aggregate[oof_aggregate["session_group"] == "ACLR"][
        ["identity_id", "normality_score"]
    ].rename(columns={"normality_score": "ACLR_score"})
    pairs = pre.merge(post, on="identity_id", how="inner")
    paired_delta = pairs["ACLR_score"] - pairs["ACLD_score"]
    wilcoxon_result = wilcoxon(paired_delta) if len(pairs) else None
    paired_sd = float(paired_delta.std(ddof=1)) if len(pairs) > 1 else float("nan")
    paired = {
        "n_pairs": int(len(pairs)),
        "mean_ACLR_minus_ACLD": float(paired_delta.mean()),
        "median_ACLR_minus_ACLD": float(paired_delta.median()),
        "moved_toward_HA_count": int((paired_delta > 0).sum()),
        "moved_toward_HA_fraction": float((paired_delta > 0).mean()),
        "paired_effect_dz": float(paired_delta.mean() / paired_sd) if paired_sd > EPSILON else float("nan"),
        "wilcoxon_statistic": float(wilcoxon_result.statistic) if wilcoxon_result else float("nan"),
        "wilcoxon_p_value": float(wilcoxon_result.pvalue) if wilcoxon_result else float("nan"),
        "training_monotonicity_constraint": False,
    }
    ha_oof = oof_aggregate[oof_aggregate["session_group"] == "HA"]["normality_score"]
    return {
        "score_name": "Gait Normality Score",
        "score_definition": "100 - 10 * z(log(GPS distance)); higher is closer to HA",
        "validation_unit": "52 biological identities: 25 HA + 27 ACL longitudinal pairs",
        "sessions": int(len(oof_aggregate)),
        "identity_overlap_max": int(oof["identity_overlap"].max()),
        "group_summary": group_summary.to_dict(orient="index"),
        "HA_cross_fitted_calibration": {
            "mean": float(ha_oof.mean()),
            "sd": float(ha_oof.std(ddof=1)),
        },
        "known_group_AUC_secondary": aucs,
        "subscore_known_group_AUC_secondary": subscore_aucs,
        "known_group_cohen_d_z_deviation": known_group_d,
        "primary_known_group_validation": {
            "supported": primary_supported,
            "status": (
                "supported_by_current_known-group check"
                if primary_supported
                else "not supported: overall GPS distance is near chance with negligible effect"
            ),
        },
        "paired_responsiveness": paired,
        "bootstrap": {
            "iterations": int(scores["bootstrap_iterations"].iloc[0]),
            "median_CI_width": float((scores["score_ci_high"] - scores["score_ci_low"]).median()),
        },
        "demographic_sensitivity": demographic_sensitivity(scores, id_path),
        "interpretation_limits": [
            "Not a clinical recovery, MDC, MCID, or return-to-sport score.",
            "AUC is a secondary known-group validity result, not a score-selection target.",
            "Point metrics explain the waveform score and are not added to the total.",
        ],
    }


def _write_figures(scores: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {"HA": "#2E7D32", "ACLD": "#C62828", "ACLR": "#1565C0"}
    groups = ["HA", "ACLD", "ACLR"]
    values = [scores.loc[scores["session_group"] == group, "normality_score"] for group in groups]
    fig, ax = plt.subplots(figsize=(8, 5))
    box = ax.boxplot(values, labels=groups, patch_artist=True)
    for patch, group in zip(box["boxes"], groups):
        patch.set_facecolor(colors[group])
        patch.set_alpha(0.7)
    ax.axhline(100, color="#333", linestyle="--", linewidth=1)
    ax.set_ylabel("Gait Normality Score")
    ax.set_title("HA-referenced gait normality by session group")
    fig.tight_layout()
    group_path = figure_dir / "01_group_score_distribution.png"
    fig.savefig(group_path, dpi=180)
    plt.close(fig)

    pre = scores[scores["session_group"] == "ACLD"][["identity_id", "normality_score"]]
    post = scores[scores["session_group"] == "ACLR"][["identity_id", "normality_score"]]
    pairs = pre.merge(post, on="identity_id", suffixes=("_ACLD", "_ACLR"))
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in pairs.iterrows():
        ax.plot([0, 1], [row["normality_score_ACLD"], row["normality_score_ACLR"]], color="#777", alpha=0.45)
    ax.scatter(np.zeros(len(pairs)), pairs["normality_score_ACLD"], color=colors["ACLD"], label="ACLD")
    ax.scatter(np.ones(len(pairs)), pairs["normality_score_ACLR"], color=colors["ACLR"], label="ACLR")
    ax.set_xticks([0, 1], ["ACLD", "ACLR"])
    ax.set_ylabel("Gait Normality Score")
    ax.set_title("Longitudinal paired score change")
    ax.legend()
    fig.tight_layout()
    pair_path = figure_dir / "02_paired_score_change.png"
    fig.savefig(pair_path, dpi=180)
    plt.close(fig)
    return {
        "group_distribution": str(group_path.relative_to(output_dir)),
        "paired_change": str(pair_path.relative_to(output_dir)),
    }


def _html_table(frame: pd.DataFrame, rows: int = 30) -> str:
    return frame.head(rows).to_html(index=False, border=0, classes="data-table", float_format=lambda x: f"{x:.4f}")


def write_html_report(
    output_dir: Path,
    scores: pd.DataFrame,
    oof_aggregate: pd.DataFrame,
    summary: dict[str, Any],
    figures: dict[str, str],
) -> Path:
    group_table = (
        oof_aggregate.groupby("session_group", observed=True)["normality_score"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
    paired = pd.DataFrame([summary["paired_responsiveness"]])
    auc_table = pd.DataFrame(
        [{"method": key, "AUC_secondary": value} for key, value in summary["known_group_AUC_secondary"].items()]
    )
    subscore_auc_table = pd.DataFrame(
        [{"subscore": key, "AUC_secondary": value} for key, value in summary["subscore_known_group_AUC_secondary"].items()]
    )
    validation_status = html.escape(summary["primary_known_group_validation"]["status"])
    css = """
    body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1180px;margin:30px auto;padding:0 24px;color:#1e293b}
    h1,h2{color:#0f172a}.note{background:#eef6ff;border-left:5px solid #2563eb;padding:14px;margin:14px 0}.warn{background:#fff7ed;border-color:#ea580c}
    .data-table{border-collapse:collapse;width:100%;font-size:13px}.data-table th,.data-table td{border:1px solid #cbd5e1;padding:7px;text-align:right}.data-table th{background:#e2e8f0}.data-table td:first-child,.data-table th:first-child{text-align:left}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.grid img{width:100%;border:1px solid #cbd5e1}@media(max-width:800px){.grid{grid-template-columns:1fr}}
    """
    body = f"""<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>Gait Normality Score</title><link rel='icon' href='data:,'><style>{css}</style></head><body>
    <h1>ACL Gait Normality Score 결과</h1>
    <div class='note'><strong>정의:</strong> 100은 leave-one-out HA 평균이며, 10점 감소할 때 정상군 기준 log-distance가 1 SD 증가한다. 점수는 clipping하지 않았다.</div>
    <div class='note warn'><strong>해석 제한:</strong> 정상 보행 이탈도이며 임상 회복, MDC, MCID 또는 return-to-sport 판정 점수가 아니다.</div>
    <h2>검증 설계</h2><ul><li>52 biological identities: 25 HA + 27 ACLD/ACLR longitudinal pairs</li><li>Repeated pair-aware 5-fold OOF; identity overlap max={summary['identity_overlap_max']}</li><li>HA 기준·보정값·comparator 전처리는 training fold에서만 학습</li></ul>
    <h2>OOF 그룹 결과</h2>{_html_table(group_table)}
    <h2>ACLD→ACLR paired responsiveness</h2>{_html_table(paired)}
    <h2>Known-group comparator</h2><div class='note warn'><strong>Primary validation:</strong> {validation_status}. AUC는 보조 지표이며 총점 선택이나 임상 중증도 해석에 사용하지 않는다.</div>{_html_table(auc_table)}
    <h2>하위점수 known-group 확인</h2>{_html_table(subscore_auc_table)}
    <h2>시각화</h2><div class='grid'><img src='{html.escape(figures['group_distribution'])}' alt='group distribution'><img src='{html.escape(figures['paired_change'])}' alt='paired change'></div>
    <h2>개인별 총점·하위점수</h2>{_html_table(scores[['subject_id','session_group','normality_score','score_ci_low','score_ci_high','hip_score','knee_score','ankle_score','slow_score','normal_score','fast_score','asymmetry_score']], rows=100)}
    </body></html>"""
    path = output_dir / "gait_normality_report.html"
    path.write_text(body, encoding="utf-8")
    return path


def run_scoring_pipeline(
    cycle_path: str | Path,
    features_path: str | Path,
    id_path: str | Path,
    output_dir: str | Path,
    n_splits: int = 5,
    n_repeats: int = 10,
    bootstrap_iterations: int = 200,
    seed: int = 42,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cycle_frame = load_cycle_waveforms(cycle_path)
    dataset = aggregate_cycle_waveforms(cycle_frame)
    full_scorer, cohort_scores, ha_scorers = score_reference_cohort(dataset)
    cohort_scores = add_bootstrap_intervals(
        cohort_scores,
        cycle_frame,
        full_scorer,
        ha_scorers,
        iterations=bootstrap_iterations,
        seed=seed,
    )
    cohort_scores = add_point_metric_explanations(cohort_scores, features_path)
    oof = run_oof_validation(dataset, n_splits=n_splits, n_repeats=n_repeats, seed=seed)
    oof_aggregate = aggregate_oof_scores(oof)
    summary = build_validation_summary(cohort_scores, oof, oof_aggregate, id_path)
    group_summary = (
        oof_aggregate.groupby("session_group", observed=True)["normality_score"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )

    paths = {
        "cohort_scores": output_dir / "gait_normality_scores.csv",
        "oof_repeated": output_dir / "oof_repeated_scores.csv",
        "oof_scores": output_dir / "oof_scores.csv",
        "group_summary": output_dir / "group_summary.csv",
        "validation_summary": output_dir / "validation_summary.json",
        "model": output_dir / "gait_normality_model.json",
    }
    cohort_scores.to_csv(paths["cohort_scores"], index=False)
    oof.to_csv(paths["oof_repeated"], index=False)
    oof_aggregate.to_csv(paths["oof_scores"], index=False)
    group_summary.to_csv(paths["group_summary"], index=False)
    paths["validation_summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    full_scorer.save(paths["model"])
    figures = _write_figures(oof_aggregate, output_dir)
    paths["report"] = write_html_report(output_dir, cohort_scores, oof_aggregate, summary, figures)
    return paths
