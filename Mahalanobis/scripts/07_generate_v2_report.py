"""Generate a run-scoped HTML report for the v2 Mahalanobis pipeline."""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd


def generate(artifact_dir: Path) -> Path:
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(artifact_dir / "summary_metrics.csv")
    scores = pd.read_parquet(artifact_dir / "oof_session_scores.parquet")
    influence = pd.read_csv(artifact_dir / "ha_leave_one_out_influence.csv")
    diagnostics = []
    for mode in sorted(scores["balance_mode"].unique()):
        diagnostics.append(pd.read_csv(artifact_dir / mode / "fold_diagnostics.csv"))
    diagnostic = pd.concat(diagnostics, ignore_index=True)

    metric_table = metrics.to_html(index=False, float_format=lambda x: f"{x:.4f}", classes="data")
    group_table = (
        scores.groupby(["balance_mode", "group"])[["slow_distance", "normal_distance", "fast_distance", "total_distance"]]
        .agg(["mean", "std", "median"]).round(3).to_html(classes="data")
    )
    diagnostic_table = diagnostic.to_html(index=False, float_format=lambda x: f"{x:.4g}", classes="data")
    influence_table = influence.sort_values("auc_change", key=lambda x: x.abs(), ascending=False).head(20).to_html(index=False, float_format=lambda x: f"{x:.4f}", classes="data")
    manifest_text = html.escape(json.dumps(manifest, ensure_ascii=False, indent=2))
    run_id = artifact_dir.name
    document = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>Mahalanobis v2 — {html.escape(run_id)}</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Apple SD Gothic Neo',sans-serif;max-width:1400px;margin:auto;padding:32px;color:#172033;line-height:1.55}}
h1,h2{{color:#0b3a66}} .callout{{padding:16px;border-left:5px solid #2474b5;background:#eef7ff;margin:18px 0}}
.data{{border-collapse:collapse;width:100%;font-size:13px}}.data th,.data td{{border:1px solid #ccd6e0;padding:7px;text-align:right}}.data th{{background:#edf3f8}}
pre{{background:#101827;color:#e7edf5;padding:16px;overflow:auto}} code{{font-family:ui-monospace,monospace}}
</style></head><body>
<h1>HA-referenced gait deviation — Mahalanobis v2</h1>
<div class="callout"><strong>Run ID:</strong> {html.escape(run_id)}<br>
이 보고서는 정확히 이 실행의 산출물만 사용합니다. Primary balance mode는 <code>inverse_weight</code>이며
<code>mean_aggregate</code>는 robustness comparator입니다. 외부 임상 anchor가 없으므로 impairment/severity로 해석하지 않습니다.</div>
<h2>방법</h2><p>매칭 ACLD/ACLR을 동일 biological identity로 고정한 nested CV를 사용했습니다. 각 속도는 outer-training HA로만 적합한 weighted PCA와 shrinkage Mahalanobis reference를 사용합니다. Signed deviation은 HA log-distance median/MAD 기준이며 clipping이나 절대값을 적용하지 않았습니다.</p>
<p><strong>mean_aggregate:</strong> cycle→trial 평균 후 trial 동일가중 조건 파형. <strong>inverse_weight:</strong> 모든 cycle을 유지하고 cycle→trial→side→speed→session inverse-count weight 적용.</p>
<h2>Nested OOF 결과</h2>{metric_table}
<h2>그룹별 거리</h2>{group_table}
<h2>Fold 수치 진단</h2>{diagnostic_table}
<h2>HA leave-one-out influence</h2><p>HA를 결과를 보고 제거하지 않고, 각 HA identity 제외 시 AUC 변화와 해당 OOF total distance를 진단합니다.</p>{influence_table}
<h2>직접 거리분해</h2><p>각 mode 폴더의 <code>top_contributions.parquet</code>은 별도 proxy 없이 실제 outer-fold quadratic form을 분해합니다. 전체 feature contribution의 합은 해당 cycle 또는 집계 대상의 D²와 일치하도록 구현되었습니다.</p>
<h2>재현성 Manifest</h2><pre>{manifest_text}</pre>
</body></html>"""
    out = artifact_dir / "report.html"
    out.write_text(document, encoding="utf-8")
    return out
