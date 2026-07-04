"""Generate a compact, self-contained v3.1 validation report."""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd


def _table(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, border=0, classes="data", float_format=lambda x: f"{x:.4f}")


def generate(artifact_dir: Path) -> Path:
    artifact_dir = Path(artifact_dir)
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(artifact_dir / "summary_metrics.csv")
    scores = pd.read_parquet(artifact_dir / "oof_session_scores_averaged.parquet")
    diagnostics = []
    for profile in manifest["input_profiles"]:
        path = artifact_dir / "profiles" / profile / "fold_diagnostics.csv"
        frame = pd.read_csv(path)
        diagnostics.append(frame.groupby("profile", as_index=False).agg(
            inner_auc_mean=("inner_best_auc", "mean"),
            condition_max=("condition_number", "max"),
            active_features_min=("active_features", "min"),
            active_features_max=("active_features", "max"),
        ))
    diagnostics_frame = pd.concat(diagnostics, ignore_index=True)
    group_summary = scores.groupby(["profile", "group"], as_index=False).agg(
        n_sessions=("subject_id", "nunique"),
        z_mean=("overall_z_deviation", "mean"),
        z_sd=("overall_z_deviation", "std"),
        normality_mean=("normality_score", "mean"),
    )
    side_sensitivity = pd.read_csv(artifact_dir / "ha_side_swap_summary.csv")
    title = "Mahalanobis v3.1 mean-only validation report"
    body = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>{title}</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1200px;margin:36px auto;padding:0 22px;color:#17202a;line-height:1.55}}
h1,h2{{color:#123b5d}} .callout{{background:#eef6fb;border-left:5px solid #2878a8;padding:16px;margin:20px 0}}
table.data{{border-collapse:collapse;width:100%;font-size:14px;margin:12px 0 28px}} table.data th,table.data td{{border:1px solid #ccd6dd;padding:7px;text-align:right}} table.data th:first-child,table.data td:first-child{{text-align:left}}
code{{background:#eef1f3;padding:2px 5px}} .small{{font-size:13px;color:#52616b}}
</style></head><body>
<h1>{title}</h1>
<div class="callout"><strong>Primary:</strong> <code>{html.escape(manifest['primary_profile'])}</code><br>
Mean-only session representation · {manifest['analysis_sessions']} sessions · {manifest['analysis_identities']} biological identities ·
outer {manifest['outer_folds']} / inner {manifest['inner_folds']} · repeats {manifest['cv_repeats']}.</div>
<h2>사전 지정 분석 계약</h2>
<ul><li>ACLD·ACLR longitudinal pair는 모든 fold에서 동일 biological identity로 묶었다.</li>
<li>HA reference, scaling, PCA, covariance와 GVS는 training HA에서만 적합했다.</li>
<li>Scalar primary는 comparator 성능을 본 뒤 교체하지 않는다.</li>
<li>HA side convention: {html.escape(manifest['side_contract']['ha_side_basis'])}.</li></ul>
<h2>주요 지표</h2>{_table(summary)}
<h2>그룹별 OOF 점수</h2>{_table(group_summary)}
<h2>수치 안정성</h2>{_table(diagnostics_frame)}
<h2>HA pseudo-side sensitivity</h2>{_table(side_sensitivity)}
<h2>재현성</h2><p class="small">Split SHA-256: <code>{manifest['split_sha256']}</code><br>
Scalar schema SHA-256: <code>{manifest['scalar_schema_sha256']}</code><br>
Engine SHA-256: <code>{manifest['engine_source_sha256']}</code></p>
</body></html>"""
    output = artifact_dir / "report.html"
    output.write_text(body, encoding="utf-8")
    return output
