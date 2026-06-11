"""
실험 결과 HTML 리포트 자동 생성.
artifacts/logs/runs_*.csv → reports/report_{datetime}.html
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import glob

import numpy as np
import pandas as pd

ML = Path(__file__).parent.parent


def _load_runs(logs_dir: Path) -> pd.DataFrame:
    csvs = sorted(glob.glob(str(logs_dir / "runs_*.csv")))
    if not csvs:
        raise FileNotFoundError(f"실험 로그 없음: {logs_dir}/runs_*.csv")
    frames = []
    for c in csvs:
        try:
            frames.append(pd.read_csv(c, on_bad_lines="skip"))
        except Exception as e:
            print(f"[report] {Path(c).name} 파싱 실패, 건너뜀: {e}")
    if not frames:
        raise FileNotFoundError("파싱 가능한 실험 로그 없음")
    return pd.concat(frames, ignore_index=True)


def _model_summary(df: pd.DataFrame) -> pd.DataFrame:
    """모델별 outer fold 평균 macro_f1 ± std."""
    g = df.groupby(["model_name", "target", "waveform_type", "waveform_norm"])
    agg = g["macro_f1"].agg(["mean", "std", "count"]).reset_index()
    agg.columns = ["model", "target", "waveform_type", "waveform_norm",
                   "macro_f1_mean", "macro_f1_std", "n_folds"]
    agg["macro_f1_mean"] = agg["macro_f1_mean"].round(4)
    agg["macro_f1_std"]  = agg["macro_f1_std"].round(4)
    return agg.sort_values("macro_f1_mean", ascending=False)


def _cm_table(df: pd.DataFrame, model_name: str) -> str:
    """특정 모델의 혼동행렬 HTML."""
    sub = df[df["model_name"] == model_name]
    cm_cols = [c for c in sub.columns if c.startswith("cm_")]
    if not cm_cols:
        return "<p>혼동행렬 없음</p>"
    cm_mean = sub[cm_cols].mean().round(1)
    return cm_mean.to_frame("mean_count").to_html(classes="table table-sm", border=0)


def _waveform_comparison(df: pd.DataFrame) -> str:
    """norm_101 vs raw_padded 비교 표 HTML."""
    dl_models = ["cnn1d", "transformer", "fpca"]
    sub = df[df["model_name"].isin(dl_models)]
    if sub.empty:
        return "<p>DL 모델 결과 없음</p>"
    pivot = sub.groupby(["model_name", "waveform_type"])["macro_f1"].mean().unstack()
    return pivot.round(4).to_html(classes="table table-bordered", border=0)


def _html_table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, classes="table table-striped table-hover", border=0,
                      float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x))


def generate(output_path: str | Path | None = None, version: str = "v1") -> Path:
    logs_dir = ML / f"artifacts-{version}" / "logs"
    report_dir = ML / f"artifacts-{version}" / "reports"

    df = _load_runs(logs_dir)
    summary = _model_summary(df)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_path is None:
        report_dir.mkdir(parents=True, exist_ok=True)
        output_path = report_dir / f"report_{ts}.html"
    output_path = Path(output_path)

    html_parts = [
        "<!DOCTYPE html><html lang='ko'><head>",
        "<meta charset='UTF-8'>",
        "<title>ACL Gait ML 실험 결과</title>",
        "<link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css'>",
        "<style>body{font-family: 'Nanum Gothic', sans-serif; padding: 20px;}"
        "h2{color:#2c3e50;} h3{color:#34495e; margin-top:30px;}</style>",
        "</head><body>",
        f"<h1 class='mb-4'>ACL Gait ML 실험 결과 리포트</h1>",
        f"<p class='text-muted'>생성일시: {ts} &nbsp;|&nbsp; "
        f"총 실험 수: {len(df)} runs, {df['model_name'].nunique()} models</p>",

        "<h2>1. 모델별 macro-F1 요약</h2>",
        _html_table(summary),

        "<h2>2. waveform_type 비교 (DL 모델)</h2>",
        "<p>norm_101 vs raw_padded macro-F1 평균</p>",
        _waveform_comparison(df),

        "<h2>3. 전체 실험 로그</h2>",
        "<details><summary>펼치기</summary>",
        _html_table(df[[
            "datetime", "model_name", "fold", "target",
            "waveform_type", "waveform_norm", "macro_f1",
            "balanced_acc", "roc_auc", "train_time_sec",
        ]].sort_values(["model_name", "fold"])),
        "</details>",

        "</body></html>",
    ]

    output_path.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"[report] 저장 완료: {output_path}")
    return output_path


if __name__ == "__main__":
    generate()
