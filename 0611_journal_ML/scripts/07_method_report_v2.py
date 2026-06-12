#!/usr/bin/env python3
"""
07_method_report_v2.py — 쉬운 설명형 방법론 보고서 (side-fixed, multi-model)

기존 reports/02_pipeline_training_method_report.html은 보존하고,
새 버전 reports/03_pipeline_explained_v2.html을 생성한다.

설계 목표: 결과 요약이 아니라 "독자가 파이프라인을 이해하도록" 단계마다
실제 값 예시를 보여준다. 무엇을 입력 → 어떻게 변환 → 어떤 모델이 학습 →
결과가 무슨 뜻인지의 흐름.

데이터/결과는 읽기 전용으로만 사용한다.
"""
import warnings; warnings.filterwarnings("ignore")
import base64, json, html
from pathlib import Path

import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parents[2]
SANDBOX = Path(__file__).resolve().parents[1]
DATA    = ROOT / "data" / "processed"
RESULTS = SANDBOX / "results"
FIGURES = SANDBOX / "figures"
REPORTS = SANDBOX / "reports"
GAI     = ROOT / "GAI 중간 결과"
REPORTS.mkdir(exist_ok=True)

OUT = REPORTS / "03_pipeline_explained_v2.html"


# ── small helpers ───────────────────────────────────────────────────────────────
def esc(x):
    return html.escape(str(x))


def load_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def load_csv(p: Path):
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def img_b64(p: Path, style="max-width:100%;border-radius:8px;border:1px solid #e5e7eb;"):
    if not p.exists():
        return f'<div class="missing">[그림 없음: {esc(p.name)}]</div>'
    b = base64.b64encode(p.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b}" style="{style}"/>'


def df_to_html(df: pd.DataFrame, highlight_baseline=True):
    if df is None or len(df) == 0:
        return '<div class="missing">[결과 파일 없음 — 재실행 필요]</div>'
    cols = list(df.columns)
    head = "".join(f"<th>{esc(c)}</th>" for c in cols)
    rows = []
    for _, r in df.iterrows():
        role = str(r.get("role", ""))
        cls = ' class="baseline"' if (highlight_baseline and role == "baseline") else ""
        cells = "".join(f"<td>{esc(r[c])}</td>" for c in cols)
        rows.append(f"<tr{cls}>{cells}</tr>")
    return f'<table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


# ── live data examples ───────────────────────────────────────────────────────────
def scalar_example():
    """한 subject의 injured vs contralateral scalar 실제 값."""
    try:
        sc = pd.read_csv(DATA / "features_scalar.csv")
        sub = sc[(sc["group"] != "HA")].dropna(subset=["injured_leg"])
        row = sub[sub["speed"] == "normal"].iloc[0]
        feats = ["hip_flexion_peak", "knee_flexion_peak", "knee_adduction_ROM"]
        items = []
        for f in feats:
            ic = f"{f}_injured"; cc = f"{f}_contralateral"
            if ic in sc.columns and cc in sc.columns:
                items.append((f, round(float(row[ic]), 2), round(float(row[cc]), 2)))
        return esc(row["subject_id"]), esc(row["group"]), esc(row["injured_leg"]), items
    except Exception:
        return None, None, None, []


def stride_example():
    """한 subject·injured side의 stride 값 → std / cv 계산 과정."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _side_utils import to_inj_con
        st = pd.read_parquet(DATA / "stride_level_peaks.parquet")
        st = st[st["subject_id"].notna()].copy()
        st["side_std"] = st["side"].apply(lambda s: to_inj_con(s, "Right"))
        feat = "knee_flexion_peak"
        g = st[(st["side_std"] == "inj") & (st["speed"] == "normal")].groupby("subject_id")
        for subj, grp in g:
            v = grp[feat].dropna().values
            if len(v) >= 5:
                v5 = [round(float(x), 2) for x in v[:6]]
                std = float(np.std(v)); mean = float(np.mean(v))
                cv = std / abs(mean) if abs(mean) > 1e-6 else 0
                return esc(subj), feat, v5, round(std, 3), round(cv, 4), len(v)
    except Exception:
        pass
    return None, None, [], None, None, None


def gai_hypotheses():
    """GAI intro.md에서 H1/H2/H3 가설 문단 추출."""
    p = GAI / "intro.md"
    if not p.exists():
        return []
    txt = p.read_text(errors="ignore")
    out = []
    for tag in ["H1", "H2", "H3"]:
        i = txt.find(f"{tag} —")
        if i == -1:
            i = txt.find(f"{tag} -")
        if i != -1:
            seg = txt[i:i + 600].split("\n\n")[0].strip()
            out.append(seg)
    return out


# ── build HTML ───────────────────────────────────────────────────────────────────
def build_html():
    bench_02b = load_csv(RESULTS / "02b_benchmark.csv")
    res_02b   = load_json(RESULTS / "02b_optimal_results.json")
    res_01    = load_csv(RESULTS / "01_speed_ablation_results.csv")
    bench_04b = load_csv(RESULTS / "04b_benchmark.csv")
    res_04b   = load_json(RESULTS / "04b_multiclass_results.json")
    bench_04c = load_csv(RESULTS / "04c_model_benchmark.csv")
    grid_04c  = load_csv(RESULTS / "04c_speed_analysis.csv")
    perm_04c  = load_json(RESULTS / "04c_permutation.json")
    wave_csv  = load_csv(RESULTS / "02_waveform_results.csv")

    s_subj, s_grp, s_leg, s_items = scalar_example()
    t_subj, t_feat, t_vals, t_std, t_cv, t_n = stride_example()
    hyps = gai_hypotheses()

    def headline_block():
        if not res_02b or "headline" not in res_02b:
            return '<div class="missing">[02b 결과 없음 — 재실행 필요]</div>'
        h = res_02b["headline"]
        tag = "✓ ≥0.98" if h.get("target_achieved") else "현 공식 결과"
        return (f'<div class="big">{h["display"]} · AUC = {h["ens_oof_auc"]:.4f} '
                f'<span class="pill">{tag}</span></div>'
                f'<div class="sub">95% CI [{h["ci_95_lo"]:.3f}, {h["ci_95_hi"]:.3f}] · '
                f'2-seed ensemble · side-fixed</div>')

    hyp_html = "".join(f"<li>{esc(h)}</li>" for h in hyps) or "<li>(intro.md 미발견)</li>"

    scalar_rows = "".join(
        f"<tr><td>{esc(f)}</td><td>{i}</td><td>{c}</td><td>{round(i-c,2)}</td></tr>"
        for f, i, c in s_items) or '<tr><td colspan=4 class="missing">예시 없음</td></tr>'

    stride_calc = (
        f"<b>{esc(t_subj)}</b> (injured side, normal 속도) 의 <code>{esc(t_feat)}</code> "
        f"stride 값들: {esc(t_vals)} … (총 {t_n}개)<br>"
        f"→ <b>std</b> = {t_std} , <b>cv</b> = std/|mean| = {t_cv}"
        if t_subj else '<span class="missing">stride 예시 없음</span>')

    css = """
    :root{--p:#1a2a4a;--ok:#27ae60;--warn:#e67e22;--bg:#f7f8fa;--bd:#e5e7eb;}
    *{box-sizing:border-box}body{font-family:-apple-system,'Apple SD Gothic Neo',sans-serif;
      margin:0;background:var(--bg);color:#1f2937;line-height:1.65}
    .wrap{max-width:980px;margin:0 auto;padding:32px 22px 80px}
    h1{font-size:24px;margin:0 0 6px}h2{font-size:18px;margin:34px 0 10px;
      padding-left:10px;border-left:4px solid var(--p)}
    h3{font-size:15px;margin:18px 0 6px;color:var(--p)}
    .lead{color:#4b5563;font-size:14px}
    .card{background:#fff;border:1px solid var(--bd);border-radius:12px;padding:18px 20px;margin:14px 0}
    .big{font-size:20px;font-weight:800;color:var(--p)}.sub{color:#6b7280;font-size:13px;margin-top:4px}
    .pill{background:var(--ok);color:#fff;font-size:11px;padding:2px 8px;border-radius:10px;vertical-align:middle}
    table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}
    th,td{border:1px solid var(--bd);padding:6px 9px;text-align:center}
    th{background:#f1f5f9}tr.baseline{background:#fff7ed;font-weight:600}
    code{background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:12px}
    .missing{color:#b91c1c;font-size:12px;font-style:italic}
    .flow{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0;font-size:12px}
    .flow span{background:#eef2ff;border:1px solid #c7d2fe;border-radius:8px;padding:5px 10px}
    .note{background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;padding:10px 14px;font-size:13px}
    .bad{background:#fef2f2;border:1px solid #fecaca}.good{background:#f0fdf4;border:1px solid #bbf7d0}
    .grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
    @media(max-width:760px){.grid2{grid-template-columns:1fr}}
    """

    waveform_block = (df_to_html(wave_csv, highlight_baseline=False)
                      if wave_csv is not None
                      else '<div class="missing">02_waveform_results.csv 없음 — '
                           'waveform 분기는 미완료(보고서에 완료로 표기 금지)</div>')

    perm_txt = (f"실제 ACLD vs ACLR AUC={perm_04c.get('real_auc_light','?')}, "
                f"permutation p={perm_04c.get('p_value','?')} "
                f"(null 평균={perm_04c.get('null_mean','?')})"
                if perm_04c else "[04c permutation 결과 없음]")

    h = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ACL 보행 분류 — 파이프라인 설명 보고서 v2</title><style>{css}</style></head>
<body><div class="wrap">

<h1>ACL 보행 분류 — 파이프라인 설명 보고서 <span style="color:#9ca3af">v2 (side-fixed)</span></h1>
<p class="lead">입력 → 변환 → 학습 → 해석의 흐름을 <b>실제 값 예시</b>와 함께 설명합니다.
이 보고서는 side mapping을 <code>inj/con</code>으로 통일해 재실행한 결과를 다룹니다.
모든 모델은 동일 파이프라인에서 Optuna로 최적화되어 직접 비교 가능합니다.</p>

<div class="card">{headline_block()}</div>

<h2>0. 연구 맥락 (가설)</h2>
<div class="card"><ul>{hyp_html}</ul>
<p class="sub">출처: <code>GAI 중간 결과/intro.md</code> — 연구 배경/가설이며, 아래 0611 모델 evidence와는 구분됩니다.</p></div>

<h2>1. 무엇을 입력하는가 — 원본 데이터</h2>
<div class="card">
<div class="flow"><span>features_scalar.csv (237×148)</span><span>stride_level_peaks.parquet (15,135×50)</span><span>waveforms_stride.parquet (474×914)</span></div>
<h3>scalar 원본 예시 — injured vs contralateral</h3>
<p class="sub">subject <b>{s_subj or '?'}</b> ({s_grp or '?'}, injured leg = {s_leg or '?'}, normal 속도). 각 feature는 손상측/반대측 두 값으로 들어옵니다.</p>
<table><thead><tr><th>feature</th><th>injured</th><th>contralateral</th><th>차이(inj−con)</th></tr></thead>
<tbody>{scalar_rows}</tbody></table>
</div>

<h2>2. Side 표준화 — inj/con 통일 (이번 핵심 수정)</h2>
<div class="card">
<p>두 파일의 <code>side</code> 인코딩이 다릅니다:
stride는 <code>injured/contralateral</code>, waveform은 <code>Right/Left</code>.
공통 헬퍼 <code>_side_utils.to_inj_con</code>로 모두 <code>inj/con</code>으로 변환합니다
(HA는 Right=inj 관례).</p>
<div class="note bad"><b>수정 전 (pre-fix exploratory):</b> 02b가 stride의 <code>injured/contralateral</code>을
<code>injured_leg=Right/Left</code>와 직접 비교 → 항상 불일치 → 모든 stride가 <code>con</code>으로 처리되어
<b>injured-side 변동성 피처가 통째로 누락</b>된 상태로 AUC가 산출됨.</div>
<div class="note good" style="margin-top:8px"><b>수정 후 (current official):</b> inj/con 정상 분리(예: inj≈7,674 / con≈7,461 strides).
아래 모든 결과는 이 side-fixed 기준입니다.</div>
</div>

<h2>3. 어떻게 변환되는가 — 단계별 실제 예시</h2>
<div class="grid2">
<div class="card"><h3>① scalar pivot</h3>
<p class="sub">subject당 3속도를 한 행으로: <code>slow_X, normal_X, fast_X</code> +
<code>dfs_X</code>(fast−slow), <code>dfn_X</code>(fast−normal), <code>mean_X</code>.
237행(79×3속도) → 79행(subject 단위), 수백~864 피처.</p></div>
<div class="card"><h3>② stride variability (std/cv)</h3>
<p class="sub">{stride_calc}</p>
<p class="sub">속도×(inj/con)별로 계산 후 subject 단위 평균.</p></div>
<div class="card"><h3>③ Fold Scaling</h3>
<p class="sub">StandardScaler를 <b>train fold에만 fit</b>, test fold는 transform만.
test 통계가 학습에 새지 않도록(leakage 방지).</p></div>
<div class="card"><h3>④ RF feature selection</h3>
<p class="sub">각 fold 내부에서 RandomForest 중요도 상위 <b>20개</b> 선택.
작은 표본(N≈78)에서 수백 피처 과적합을 억제.</p></div>
<div class="card"><h3>⑤ Interactions</h3>
<p class="sub">상위 20개의 <code>C(20,2)=190</code>개 곱(feature_A×feature_B).
단일 피처로 안 보이는 결합 패턴(예: 굴곡각×변동성)을 표현.</p></div>
<div class="card"><h3>⑥ 모델 학습 + Optuna</h3>
<p class="sub">최종 분류기를 inner-CV Optuna로 튜닝.
baseline=Logistic Regression, 그 외는 벤치마크 모델.
필요시 2-seed(42/88) 확률 평균으로 headline 산출.</p></div>
</div>

<h2>4. 모델 벤치마크 (side-fixed)</h2>
<h3>4-1. 이진 ACL vs HA — 02b (주황=baseline LogReg)</h3>
<div class="card">{df_to_html(bench_02b)}</div>

<h3>4-2. 속도 ablation — 01 (조건별 모델 성능)</h3>
<div class="card">{df_to_html(res_01, highlight_baseline=False)}</div>

<h3>4-3. 3분류 ACLD/ACLR/HA — 04b</h3>
<div class="card">{df_to_html(bench_04b)}</div>

<h3>4-4. 3분류 통합조건 모델 벤치마크 — 04c</h3>
<div class="card">{df_to_html(bench_04c)}</div>

<h2>5. 실험별 결과 해석</h2>
<div class="card">
<h3>01 — 속도 ablation (가설 H2)</h3>
<p class="sub">단일 속도(slow/normal/fast)만 쓰면 보상 전략이 충분히 드러나지 않습니다.
3속도 통합이 단일 속도 대비 더 높은 AUC를 보이면 H2(다속도 우월성)를 지지합니다.</p>
<h3>02 — Raw Waveform vs Scalar</h3>
{waveform_block}
<p class="sub">파형(9채널×101시점) 직접 입력은 scalar 통계와 다른 정보 형태입니다.
산출물이 생성된 경우에만 결과로 인정합니다(미생성 시 위에 명시).</p>
<h3>02b — 최종 이진 모델</h3>
<p class="sub">위 4-1 표가 최종 이진 결과입니다. AUC는 순위지표, 임계값 0.5 분류 요약과는 별개로 봅니다.</p>
<h3>03 — SHAP (어떤 보행 구간이 기여했나)</h3>
<div class="grid2">
<div>{img_b64(FIGURES/'fig_B_shap_heatmap_mean.png')}</div>
<div>{img_b64(FIGURES/'fig_C_channel_importance.png')}</div>
</div>
<h3>04b/04c — 왜 3분류가 더 어려운가</h3>
<p class="sub">ACLD와 ACLR는 보행상 거의 겹쳐 분리가 어렵습니다. {esc(perm_txt)}.
HA는 상대적으로 분리되어, 임상적으로 의미 있는 과제는 ACL vs HA 이진 분류임을 시사합니다.</p>
<div class="grid2">
<div>{img_b64(FIGURES/'fig_04c_pca_overlap.png')}</div>
<div>{img_b64(FIGURES/'fig_04c_permutation.png')}</div>
</div>
</div>

<h2>6. 결과 구분 (pre-fix vs current)</h2>
<div class="card">
<p><b>pre-fix exploratory:</b> side 버그가 있던 이전 수치(예: 02b AUC 0.983)는 참고용 탐색 결과로만 봅니다.</p>
<p><b>current official:</b> 이 보고서의 모든 표/그림은 side-fixed 재실행 결과이며 공식 결과로 채택합니다
(가정상 기존보다 낮아도 채택).</p>
</div>

<h2>7. Evidence 파일</h2>
<div class="card"><p class="sub">
results/02b_benchmark.csv · 02b_optimal_results.json · 01_speed_ablation_results.csv ·
04b_benchmark.csv · 04b_multiclass_results.json · 04c_model_benchmark.csv ·
04c_speed_analysis.csv · 04c_permutation.json · figures/fig_*.png<br>
scripts/_side_utils.py (side 표준화) · scripts/_models.py (8모델 레지스트리)
</p></div>

</div></body></html>"""
    return h


def main():
    OUT.write_text(build_html())
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
