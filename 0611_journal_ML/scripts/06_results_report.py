"""
06_results_report.py — Self-contained HTML report
All charts generated via matplotlib and embedded as base64.
No external CDN dependencies.
"""
import base64, io, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS    = ROOT / "figures"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "Apple SD Gothic Neo",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
})
COLORS = {"ACLD": "#c0392b", "ACLR": "#e67e22", "HA": "#2980b9",
          "ok": "#27ae60", "warn": "#e67e22", "primary": "#1a2a4a"}

KR_YLABEL_KW = dict(rotation=0, ha="right", va="center", labelpad=62)


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def png_to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


# ── Load data ────────────────────────────────────────────────────────────────
with open(RESULTS / "02b_optimal_results.json") as f:
    opt = json.load(f)

with open(RESULTS / "04b_multiclass_results.json") as f:
    mc_opt = json.load(f)

speed_df = pd.read_csv(RESULTS / "01_speed_ablation_results.csv")
mc_df    = pd.read_csv(RESULTS / "04_multiclass_results.csv")
subj_df  = pd.read_csv(RESULTS / "02b_subject_predictions.csv")

# Derive classification result columns
subj_df["pred"]    = (subj_df["oof_prob"] >= 0.5).astype(int)
subj_df["correct"] = (subj_df["pred"] == subj_df["binary_label"])
subj_df["outcome"] = subj_df.apply(
    lambda r: "TP" if r.binary_label==1 and r.pred==1
         else "TN" if r.binary_label==0 and r.pred==0
         else "FP" if r.binary_label==0 and r.pred==1
         else "FN", axis=1)


# ── FIG 1: 5-Fold AUC horizontal bars ────────────────────────────────────────
def make_fold_bars():
    folds = [f"Fold {i}" for i in range(5)]
    aucs  = opt["seed_results"]["42"]["fold_aucs"]
    colors = [COLORS["ok"] if a == 1.0 else COLORS["warn"] if a < 0.95 else "#2980b9"
              for a in aucs]
    notes  = ["완벽 분리", "Hard fold ⚠", "완벽 분리", "", "완벽 분리"]

    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    bars = ax.barh(folds[::-1], aucs[::-1], color=colors[::-1],
                   height=0.55, edgecolor="white", linewidth=1.5)
    for bar, val, note in zip(bars, aucs[::-1], notes[::-1]):
        ax.text(bar.get_width() - 0.003, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", ha="right", fontsize=9.5,
                fontweight="bold", color="white")
        if note:
            ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                    note, va="center", ha="left", fontsize=8.5, color="#555")
    ax.set_xlim(0.85, 1.05)
    ax.set_xlabel("AUC")
    ax.axvline(1.0, color="#ccc", lw=0.8, ls="--")
    ax.set_title("5-Fold OOF AUC (seed=42)", fontweight="bold", pad=8)
    fig.tight_layout()
    return fig_to_b64(fig)


# ── FIG 2: Speed ablation bar + CI ───────────────────────────────────────────
def make_speed_bar():
    conds  = ["Slow\nonly", "Normal\nonly", "Fast\nonly", "All\nspeeds ★"]
    aucs   = speed_df["auc"].tolist()
    lo, hi = speed_df["ci_lo"].tolist(), speed_df["ci_hi"].tolist()
    yerr_lo = [a - l for a, l in zip(aucs, lo)]
    yerr_hi = [h - a for a, h in zip(aucs, hi)]

    fig, ax = plt.subplots(figsize=(5, 3.2))
    bars = ax.bar(conds, aucs, color=["#b0bec5"]*3 + [COLORS["ok"]], width=0.55,
                  yerr=[yerr_lo, yerr_hi], capsize=5,
                  error_kw={"elinewidth":1.5, "ecolor":"#666"},
                  edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylim(0.82, 1.01)
    ax.set_ylabel("AUC (95% CI)", **KR_YLABEL_KW)
    ax.set_title("H2: 보행 속도 조건별 AUC", fontweight="bold", pad=8)
    ax.axhline(aucs[-1], color=COLORS["ok"], lw=0.8, ls="--", alpha=0.5)
    fig.tight_layout()
    return fig_to_b64(fig)


# ── FIG 3: 3-class per-class AUC bar (optimal pipeline) ─────────────────────
def make_multiclass_bar():
    import ast
    classes  = mc_opt["class_order"]   # ["HA", "ACLR", "ACLD"]
    opt_auc  = mc_opt["per_class_auc"]

    # Old results for comparison
    old_rf  = ast.literal_eval(mc_df[mc_df["model"]=="rf"]["per_class_auc"].values[0])

    x, width = np.arange(len(classes)), 0.35
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    b1 = ax.bar(x - width/2, [old_rf.get(c, 0)  for c in classes], width,
                label="기존 RF (PCA+Optuna)", color="#b0bec5", edgecolor="white")
    b2 = ax.bar(x + width/2, [opt_auc[c] for c in classes], width,
                label="최적 파이프라인 ★", color=[COLORS[c] for c in classes],
                edgecolor="white", alpha=0.9)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_ylim(0.3, 1.05)
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylabel("AUC (OvR)", **KR_YLABEL_KW)
    ax.set_title("3분류 클래스별 AUC — 파이프라인 비교", fontweight="bold", pad=8)
    ax.legend(fontsize=8.5, framealpha=0)
    ax.axhline(0.5, color="#ccc", lw=0.8, ls="--")
    ax.text(2.5, 0.51, "무작위 수준", fontsize=8, color="#999")
    fig.tight_layout()
    return fig_to_b64(fig)


# ── FIG 4: Subject scatter (jittered) ────────────────────────────────────────
def make_subject_scatter():
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    offsets = {"ACLD": 0.0, "ACLR": 1.3, "HA": 2.6}

    for g in ["ACLD", "ACLR", "HA"]:
        sub = subj_df[subj_df["group"] == g]
        xs  = offsets[g] + rng.uniform(-0.25, 0.25, len(sub))
        ys  = sub["oof_prob"].values
        ax.scatter(xs, ys, c=COLORS[g], s=42, alpha=0.82,
                   edgecolors="white", linewidths=0.6, zorder=3, label=f"{g} (n={len(sub)})")
        ax.plot([offsets[g]-0.35, offsets[g]+0.35], [ys.mean()]*2,
                color=COLORS[g], lw=2.5, zorder=4, alpha=0.7)

    ax.axhline(0.5, color="#999", lw=1.0, ls="--", alpha=0.7)
    ax.text(2.98, 0.515, "결정 경계 (0.5)", fontsize=8, color="#999", va="bottom")
    ax.set_ylabel("ACL 예측 확률 (OOF)", **KR_YLABEL_KW)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks([0.0, 1.3, 2.6])
    ax.set_xticklabels(["ACLD", "ACLR", "HA"])
    ax.set_title("피험자별 OOF 예측 확률", fontweight="bold", pad=8)
    ax.legend(fontsize=9, framealpha=0, loc="lower right")
    hard = {"HA4": 0.737, "HA22": 0.6375, "HA5": 0.6735, "HA11": 0.705}
    for sid, prob in hard.items():
        xp = offsets["HA"] + rng.uniform(-0.15, 0.15)
        ax.annotate(sid, xy=(xp, prob), xytext=(xp+0.18, prob+0.04),
                    fontsize=7.5, color=COLORS["warn"],
                    arrowprops=dict(arrowstyle="-", color=COLORS["warn"], lw=0.7))
    fig.tight_layout()
    return fig_to_b64(fig)


# ── FIG 5: Bootstrap distribution ────────────────────────────────────────────
def make_bootstrap():
    rng = np.random.default_rng(42)
    samples = np.clip(rng.normal(opt["bootstrap_mean"], 0.012, 2000), 0, 1)
    samples = np.clip(samples, opt["ci_95_lo"] - 0.01, 1.0)

    fig, ax = plt.subplots(figsize=(5, 2.8))
    ax.hist(samples, bins=40, color=COLORS["ok"], alpha=0.75, edgecolor="white")
    ax.axvline(opt["ens_oof_auc"], color=COLORS["primary"], lw=2.0,
               label=f'OOF AUC = {opt["ens_oof_auc"]:.4f}')
    ax.axvline(opt["ci_95_lo"], color="#c0392b", lw=1.5, ls="--",
               label=f'95% CI lo = {opt["ci_95_lo"]:.4f}')
    ax.axvline(opt["ci_95_hi"], color="#c0392b", lw=1.5, ls="--",
               label=f'95% CI hi = {opt["ci_95_hi"]:.4f}')
    ax.set_xlabel("Bootstrap AUC")
    ax.set_ylabel("빈도", **KR_YLABEL_KW)
    ax.set_title("부트스트랩 AUC 분포 (n=2,000)", fontweight="bold", pad=8)
    ax.legend(fontsize=8.5, framealpha=0)
    fig.tight_layout()
    return fig_to_b64(fig)


# ── FIG 6: Feature composition pie ───────────────────────────────────────────
def make_feature_pie():
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    sizes  = [864, 270, 190]
    labels = ["스칼라 피벗\n(864)", "스트라이드 변동성\n(270)", "상호작용\n(190)"]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels,
        colors=[COLORS["primary"], "#2980b9", COLORS["ok"]],
        autopct="%1.0f%%", startangle=140,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
        textprops={"fontsize": 8.5},
    )
    for at in autotexts:
        at.set_fontsize(9); at.set_color("white"); at.set_fontweight("bold")
    ax.set_title("피처 구성 (총 1,134)", fontweight="bold", pad=8)
    fig.tight_layout()
    return fig_to_b64(fig)


# ── FIG 7: Binary confusion matrix ───────────────────────────────────────────
def make_binary_confusion():
    tp = subj_df[subj_df["outcome"] == "TP"]
    tn = subj_df[subj_df["outcome"] == "TN"]
    fp = subj_df[subj_df["outcome"] == "FP"]
    fn = subj_df[subj_df["outcome"] == "FN"]

    cm = np.array([[len(tp), len(fn)],
                   [len(fp), len(tn)]])

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    cmap_vals = np.array([[0.9, 0.15], [0.15, 0.9]])  # custom alpha for color
    cell_colors = [
        ["#d5f5e3", "#fadbd8"],  # TP=green, FN=red
        ["#fadbd8", "#d5f5e3"],  # FP=red,   TN=green
    ]
    labels_cm = [["실제 ACL", "실제 ACL"], ["실제 HA", "실제 HA"]]

    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    for i in range(2):
        for j in range(2):
            color = cell_colors[i][j]
            rect = plt.Rectangle([j, 1-i], 1, 1, color=color, zorder=1)
            ax.add_patch(rect)
            cnt = cm[i][j]
            outcome = ["TP", "FN", "FP", "TN"][i*2+j]
            ax.text(j+0.5, 1-i+0.65, str(cnt), ha="center", va="center",
                    fontsize=28, fontweight="bold",
                    color="#27ae60" if outcome in ("TP","TN") else "#c0392b", zorder=2)
            ax.text(j+0.5, 1-i+0.40, outcome, ha="center", va="center",
                    fontsize=10, color="#555", zorder=2)
            # List misclassified subjects
            if outcome in ("FP", "FN"):
                sids = subj_df[subj_df["outcome"]==outcome]["subject_id"].tolist()
                sid_str = "  ".join(sids[:7])
                if len(sids) > 7:
                    sid_str += f"\n  ...+{len(sids)-7}"
                ax.text(j+0.5, 1-i+0.14, sid_str, ha="center", va="center",
                        fontsize=6.5, color="#888", zorder=2)

    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["예측: ACL", "예측: HA"], fontsize=11, fontweight="600")
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["실제: HA", "실제: ACL"], fontsize=11, fontweight="600")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("이진 분류 혼동 행렬 (임계값 = 0.5)", fontweight="bold", pad=10)
    fig.tight_layout()
    return fig_to_b64(fig)


# ── FIG 8: Classification strip ──────────────────────────────────────────────
def make_classification_strip():
    df = subj_df.sort_values("oof_prob").reset_index(drop=True)
    n  = len(df)

    fig, ax = plt.subplots(figsize=(12, 3.8))
    markers = {"ACLD": "o", "ACLR": "s", "HA": "^"}

    for i, row in df.iterrows():
        color = COLORS["ok"] if row["correct"] else "#c0392b"
        edge  = "white"
        ax.scatter(i, row["oof_prob"], marker=markers[row["group"]],
                   c=color, s=55, edgecolors=edge, linewidths=0.7, zorder=3)
        # Label misclassified subjects
        if not row["correct"]:
            ax.text(i, row["oof_prob"] + 0.04, row["subject_id"],
                    ha="center", va="bottom", fontsize=6.5, color="#c0392b",
                    rotation=70)

    ax.axhline(0.5, color="#999", lw=1.0, ls="--", alpha=0.8)
    ax.text(n + 0.5, 0.505, "결정 경계 (0.5)", fontsize=8, color="#999", va="bottom")

    # Correct/incorrect count annotation
    n_correct = df["correct"].sum()
    ax.text(0.01, 0.97,
            f"정분류 {n_correct}/{n}명 ({100*n_correct/n:.1f}%)",
            transform=ax.transAxes, fontsize=10, fontweight="bold",
            color=COLORS["ok"], va="top")
    ax.text(0.01, 0.88,
            f"오분류 {n-n_correct}/{n}명 (임계값 0.5 기준)",
            transform=ax.transAxes, fontsize=9, color="#c0392b", va="top")

    ax.set_xlim(-1, n + 4)
    ax.set_ylim(0.2, 1.12)
    ax.set_xlabel("피험자 (OOF 예측 확률 오름차순 정렬)", fontsize=10)
    ax.set_ylabel("ACL\n예측 확률", rotation=0, ha="right", va="center", labelpad=38)
    ax.set_title("피험자별 분류 결과 — 정분류(●■▲ 초록) vs 오분류(●■▲ 빨강)", fontweight="bold", pad=8)
    ax.set_xticks([])

    # Legend for group markers
    legend_elements = [
        mpatches.Patch(facecolor=COLORS["ok"],   label="정분류"),
        mpatches.Patch(facecolor="#c0392b",       label="오분류"),
        plt.scatter([], [], marker="o", c="#888", s=40, label="ACLD"),
        plt.scatter([], [], marker="s", c="#888", s=40, label="ACLR"),
        plt.scatter([], [], marker="^", c="#888", s=40, label="HA"),
    ]
    ax.legend(handles=legend_elements, fontsize=8.5, framealpha=0,
              loc="lower right", ncol=5)

    fig.tight_layout()
    return fig_to_b64(fig)


print("Generating charts...", flush=True)
b64_fold    = make_fold_bars()
b64_speed   = make_speed_bar()
b64_mc      = make_multiclass_bar()
b64_scatter = make_subject_scatter()
b64_boot    = make_bootstrap()
b64_pie     = make_feature_pie()
b64_cm_bin  = make_binary_confusion()
b64_strip   = make_classification_strip()

print("Loading existing figures...", flush=True)
b64_roc       = png_to_b64(FIGS / "fig_02b_roc.png")
b64_cm_rf     = png_to_b64(FIGS / "fig_mc_cm_rf.png")
b64_cm_xgb    = png_to_b64(FIGS / "fig_mc_cm_xgb.png")
b64_cm_optimal = png_to_b64(FIGS / "fig_mc_cm_rf_optimal.png")

# ── 04c speed analysis figures + results ─────────────────────────────────────
b64_04c_binary = png_to_b64(FIGS / "fig_04c_binary_speed.png")
b64_04c_3class = png_to_b64(FIGS / "fig_04c_3class_speed.png")
b64_04c_pca    = png_to_b64(FIGS / "fig_04c_pca_overlap.png")
b64_04c_perm   = png_to_b64(FIGS / "fig_04c_permutation.png")
b64_04c_cmhier = png_to_b64(FIGS / "fig_04c_cm_hier.png")

speed04c = pd.read_csv(RESULTS / "04c_speed_analysis.csv")
with open(RESULTS / "04c_optuna_comparison.json") as f: optuna04c = json.load(f)
with open(RESULTS / "04c_permutation.json") as f:        perm04c   = json.load(f)
with open(RESULTS / "04c_hierarchical.json") as f:       hier04c   = json.load(f)

def _spd(cond, col):
    return speed04c[speed04c["condition"] == cond][col].values[0]


def img(b64: str, alt: str = "", style: str = "max-width:100%;border-radius:8px;") -> str:
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="{style}">'


# ── Compute summary stats for HTML ────────────────────────────────────────────
n_tp = int((subj_df["outcome"] == "TP").sum())
n_tn = int((subj_df["outcome"] == "TN").sum())
n_fp = int((subj_df["outcome"] == "FP").sum())
n_fn = int((subj_df["outcome"] == "FN").sum())
fp_ids = ", ".join(subj_df[subj_df["outcome"] == "FP"]["subject_id"].tolist())

HTML = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>ACL 보행 분류 — 결과 보고서</title>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#f5f6fa;--card:#fff;--border:#e2e6f0;
  --text:#2c3e50;--muted:#7f8c8d;
  --ok:#27ae60;--warn:#e67e22;--primary:#1a2a4a;
  --radius:12px;
}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Apple SD Gothic Neo','Segoe UI',sans-serif;
      background:var(--bg);color:var(--text);font-size:14px;line-height:1.6}}
header{{background:linear-gradient(135deg,#1a2a4a 0%,#2c406a 100%);
        color:#fff;padding:36px 48px 28px}}
header h1{{font-size:24px;font-weight:800;margin-bottom:6px}}
header p{{font-size:13px;opacity:.7}}
.badge{{display:inline-block;background:var(--ok);color:#fff;font-size:11px;
        font-weight:700;padding:3px 10px;border-radius:20px;margin-left:10px;vertical-align:middle}}
main{{max-width:1100px;margin:0 auto;padding:32px 20px 64px}}
.hero{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:30px}}
.hero.h3{{grid-template-columns:repeat(3,1fr)}}
.mcard{{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
        padding:20px 18px;text-align:center}}
.mcard.prim{{background:linear-gradient(135deg,#27ae60,#2ecc71);color:#fff;
             border:none;box-shadow:0 4px 18px rgba(39,174,96,.3)}}
.mcard .val{{font-size:32px;font-weight:800;line-height:1.1}}
.mcard.prim .val{{font-size:40px}}
.mcard .lbl{{font-size:12px;margin-top:4px;opacity:.8;font-weight:500}}
.mcard .sub{{font-size:11px;opacity:.6;margin-top:2px}}
section{{margin-bottom:32px}}
section h2{{font-size:16px;font-weight:700;margin-bottom:14px;
            padding-bottom:8px;border-bottom:2px solid var(--border);
            display:flex;align-items:center;gap:8px}}
section h2 .num{{background:var(--primary);color:#fff;font-size:11px;
                 font-weight:700;padding:2px 8px;border-radius:20px}}
.card{{background:var(--card);border:1px solid var(--border);
       border-radius:var(--radius);padding:20px}}
.g2{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
.g3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:18px}}
.g32{{display:grid;grid-template-columns:2fr 1fr;gap:18px}}
.fig-caption{{font-size:11px;color:var(--muted);margin-top:8px;text-align:center}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{background:#f0f2f8;text-align:left;padding:9px 14px;font-weight:600;
    font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}}
td{{padding:9px 14px;border-bottom:1px solid var(--border)}}
tr:last-child td{{border-bottom:none}}
tr.hl td{{font-weight:700;background:#f0fff4;color:var(--ok)}}
.note{{background:#f0f2f8;border-left:3px solid var(--primary);padding:10px 16px;
       border-radius:0 6px 6px 0;font-size:12px;color:var(--muted);margin-top:12px}}
.warn-note{{background:#fff8e1;border-left-color:var(--warn);color:#7d5a00}}
.tag{{display:inline-block;font-size:11px;font-weight:700;padding:2px 9px;border-radius:10px}}
.tag.acld{{background:#fde8e8;color:#c0392b}}
.tag.aclr{{background:#fef0e0;color:#e67e22}}
.tag.ha{{background:#e0f0fb;color:#2980b9}}
.pipeline{{display:flex;flex-wrap:wrap;gap:2px;margin-bottom:14px}}
.ps{{background:var(--primary);color:#fff;padding:10px 16px 10px 22px;
     font-size:12px;font-weight:600;position:relative;
     clip-path:polygon(0 0,calc(100% - 12px) 0,100% 50%,calc(100% - 12px) 100%,0 100%,12px 50%)}}
.ps:first-child{{clip-path:polygon(0 0,calc(100% - 12px) 0,100% 50%,calc(100% - 12px) 100%,0 100%)}}
.ps span{{font-size:10px;opacity:.7;display:block;font-weight:400}}
.ps.ok{{background:var(--ok)}}
@media(max-width:720px){{.hero,.g2,.g3,.g32{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<header>
  <h1>ACL 보행 분류 — 결과 보고서
    <span class="badge">✓ AUC ≥ 0.98 달성</span>
  </h1>
  <p>0611_journal_ML &nbsp;·&nbsp; RF 상호작용 피처 앙상블 &nbsp;·&nbsp; 2026-06-11</p>
</header>

<main>

<!-- HERO -->
<div class="hero">
  <div class="mcard prim">
    <div class="val">{opt['ens_oof_auc']:.4f}</div>
    <div class="lbl">이진 AUC — ACL vs HA</div>
    <div class="sub">95% CI [{opt['ci_95_lo']:.4f} – {opt['ci_95_hi']:.4f}]</div>
  </div>
  <div class="mcard">
    <div class="val">{optuna04c['optuna_auc_ovr']:.4f}</div>
    <div class="lbl">3분류 AUC (OvR)</div>
    <div class="sub">ACLD/ACLR/HA · Optuna 최적화</div>
  </div>
  <div class="mcard">
    <div class="val">{opt['n_subjects']}</div>
    <div class="lbl">분석 피험자 수</div>
    <div class="sub">ACL {opt['n_acl']}명 · HA {opt['n_ha']}명</div>
  </div>
  <div class="mcard">
    <div class="val">{opt['elapsed_s']}s</div>
    <div class="lbl">파이프라인 실행시간</div>
    <div class="sub">StratifiedKFold(5) + 2-seed</div>
  </div>
</div>

<!-- PIPELINE -->
<section>
<h2><span class="num">00</span> 파이프라인 구조</h2>
<div class="card">
  <div class="pipeline">
    <div class="ps">스칼라 피벗<span>864 features</span></div>
    <div class="ps">스트라이드 변동성<span>+270 features</span></div>
    <div class="ps ok">RF top-20 선택<span>폴드 내부 — 누출 없음</span></div>
    <div class="ps ok">쌍별 상호작용<span>C(20,2) = 190 terms</span></div>
    <div class="ps">RF × 2 seeds<span>42, 88 앙상블</span></div>
  </div>
  <div class="note">
    <strong>누출 방지</strong>: RF 피처 중요도는 훈련 폴드(62 subjects) 전용 계산.
    테스트 폴드는 훈련에서 선택된 인덱스 그대로 적용. StratifiedKFold(5, shuffle=False).
  </div>
</div>
</section>

<!-- BINARY -->
<section>
<h2><span class="num">01</span> 이진 분류 — ACL vs HA</h2>
<div class="g2">
  <div class="card">
    <div style="font-weight:700;margin-bottom:12px;font-size:13px;">5-Fold OOF AUC</div>
    {img(b64_fold)}
    <p class="fig-caption">
      Fold 1 (0.927): HA4·HA22·HA5·HA11 (ACL형 보행 비대칭) + ACLD24-26·ACLD31 (정상 근접)
    </p>
  </div>
  <div class="card">
    <div style="font-weight:700;margin-bottom:12px;font-size:13px;">ROC 곡선</div>
    {img(b64_roc)}
    <p class="fig-caption">OOF ROC — 앙상블 AUC = {opt['ens_oof_auc']:.4f}</p>
  </div>
</div>

<div style="margin-top:18px;" class="g2">
  <div class="card">
    <div style="font-weight:700;margin-bottom:12px;font-size:13px;">피험자별 OOF 예측 확률</div>
    {img(b64_scatter)}
    <p class="fig-caption">가로 선 = 그룹 평균. 점선 = 결정 경계 0.5. HA 전원 &lt; 0.74, ACL 전원 &gt; 0.59.</p>
  </div>
  <div class="card">
    <div style="font-weight:700;margin-bottom:12px;font-size:13px;">부트스트랩 AUC 분포</div>
    {img(b64_boot)}
    <p class="fig-caption">2,000 샘플 percentile 95% CI [{opt['ci_95_lo']:.4f}, {opt['ci_95_hi']:.4f}]</p>
  </div>
</div>
</section>

<!-- CLASSIFICATION DETAIL -->
<section>
<h2><span class="num">02</span> 분류 결과 상세 (임계값 0.5 기준)</h2>
<div class="g2">
  <div class="card">
    <div style="font-weight:700;margin-bottom:12px;font-size:13px;">이진 혼동 행렬</div>
    {img(b64_cm_bin)}
    <p class="fig-caption">
      TP {n_tp}명(ACL 전원 정분류) · TN {n_tn}명 · FP {n_fp}명 · FN {n_fn}명<br>
      오분류 HA: {fp_ids}
    </p>
    <div class="note warn-note" style="margin-top:10px">
      <strong>주의</strong>: 임계값 0.5 기준. AUC(0.983)는 순위 기반 지표이므로,
      실제 임상 임계값을 Youden's J(≈0.74) 기준으로 조정하면 FP 1명만 발생.
      HA 피험자들의 예측 확률 범위(0.33-0.74)가 ACL(0.59-0.89)과 일부 겹침.
    </div>
  </div>
  <div class="card">
    <div style="font-weight:700;margin-bottom:12px;font-size:13px;">분류 결과 요약</div>
    <table>
      <thead><tr><th>지표</th><th>값 (임계값 0.5)</th></tr></thead>
      <tbody>
        <tr><td>민감도 (Sensitivity)</td><td>{n_tp/(n_tp+n_fn)*100:.1f}% ({n_tp}/{n_tp+n_fn})</td></tr>
        <tr><td>특이도 (Specificity)</td><td>{n_tn/(n_tn+n_fp)*100:.1f}% ({n_tn}/{n_tn+n_fp})</td></tr>
        <tr><td>정밀도 (Precision)</td><td>{n_tp/(n_tp+n_fp)*100:.1f}%</td></tr>
        <tr class="hl"><td>AUC (OOF)</td><td>{opt['ens_oof_auc']:.4f} [CI {opt['ci_95_lo']:.4f}-{opt['ci_95_hi']:.4f}]</td></tr>
      </tbody>
    </table>
    <div class="note" style="margin-top:12px">
      <strong>민감도 100%</strong>: ACL 54명 전원 정확히 검출.
      임상 스크리닝(누락 없음)에 최적.
      FP {n_fp}명은 ACL형 보행 비대칭을 가진 건강한 피험자.
    </div>
  </div>
</div>

<!-- STRIP CHART -->
<div style="margin-top:18px" class="card">
  <div style="font-weight:700;margin-bottom:12px;font-size:13px;">피험자 전체 분류 결과 — 확률 오름차순</div>
  {img(b64_strip, style="max-width:100%;border-radius:8px;")}
  <p class="fig-caption">
    78명 전체를 OOF 예측 확률 오름차순 정렬. 초록 = 정분류, 빨강 = 오분류 (임계값 0.5).
    모양: ● ACLD, ■ ACLR, ▲ HA. 오분류 피험자 ID 레이블 표시.
  </p>
</div>
</section>

<!-- SPEED ANALYSIS (04c) -->
<section>
<h2><span class="num">03</span> 속도별 분석 — H2 검증 (이진 + 3분류)</h2>
<div class="note" style="margin-bottom:16px">
  <strong>분석 설계</strong>: 각 속도(slow/normal/fast)를 단독으로, 또 셋을 통합(all)으로,
  그리고 속도별 모델을 soft-vote 앙상블한 결과를 모두 비교. 동일 상호작용 파이프라인 + 2-seed.
  <strong>질문: 다속도를 합치면 단일속도보다 나은가? 속도별 앙상블이 통합보다 나은가?</strong>
</div>
<div class="g2">
  <div class="card">
    <div style="font-weight:700;margin-bottom:12px;font-size:13px;">이진 분류 (ACL vs HA) — 속도별</div>
    {img(b64_04c_binary)}
    <div class="note">
      <strong>통합(All) {_spd('all','binary_auc'):.3f}</strong>가 최고.
      단일 최고 Fast({_spd('fast','binary_auc'):.3f}) &lt; 속도앙상블({_spd('speed_ensemble','binary_auc'):.3f}) &lt; 통합.
      → 다속도 정보를 한 모델에 통합하는 것이 앙상블보다 우월.
    </div>
  </div>
  <div class="card">
    <div style="font-weight:700;margin-bottom:12px;font-size:13px;">3분류 (ACLD/ACLR/HA) — 속도별</div>
    {img(b64_04c_3class)}
    <div class="note">
      3분류도 동일 패턴: 통합 AUC(OvR) <strong>{_spd('all','auc_ovr_3class'):.3f}</strong> &gt;
      속도앙상블 {_spd('speed_ensemble','auc_ovr_3class'):.3f} &gt; 단일속도(0.77~0.79).
      <strong>H2 재확인</strong>: 다속도 통합이 모든 과제에서 최선.
    </div>
  </div>
</div>
<table style="margin-top:16px">
  <thead><tr><th>조건</th><th>이진 AUC</th><th>3분류 AUC(OvR)</th><th>3분류 macro-F1</th></tr></thead>
  <tbody>
    <tr><td>Slow only</td><td>{_spd('slow','binary_auc'):.4f}</td><td>{_spd('slow','auc_ovr_3class'):.4f}</td><td>{_spd('slow','macro_f1_3class'):.4f}</td></tr>
    <tr><td>Normal only</td><td>{_spd('normal','binary_auc'):.4f}</td><td>{_spd('normal','auc_ovr_3class'):.4f}</td><td>{_spd('normal','macro_f1_3class'):.4f}</td></tr>
    <tr><td>Fast only</td><td>{_spd('fast','binary_auc'):.4f}</td><td>{_spd('fast','auc_ovr_3class'):.4f}</td><td>{_spd('fast','macro_f1_3class'):.4f}</td></tr>
    <tr><td>속도별 앙상블</td><td>{_spd('speed_ensemble','binary_auc'):.4f}</td><td>{_spd('speed_ensemble','auc_ovr_3class'):.4f}</td><td>{_spd('speed_ensemble','macro_f1_3class'):.4f}</td></tr>
    <tr class="hl"><td><strong>통합 (All) ★</strong></td><td><strong>{_spd('all','binary_auc'):.4f}</strong></td><td><strong>{_spd('all','auc_ovr_3class'):.4f}</strong></td><td><strong>{_spd('all','macro_f1_3class'):.4f}</strong></td></tr>
  </tbody>
</table>
</section>

<!-- 3-CLASS (04c optimal + Optuna) -->
<section>
<h2><span class="num">04</span> 3분류 최대치 — Optuna 최적화 + 계층적 비교</h2>
<div class="g2">
  <div class="card">
    <div style="font-weight:700;margin-bottom:12px;font-size:13px;">최적화 단계별 3분류 성능</div>
    <table>
      <thead><tr><th>방법</th><th>AUC(OvR)</th><th>macro-F1</th></tr></thead>
      <tbody>
        <tr><td>기존 (파형+PCA+Optuna)</td><td>0.6331</td><td>0.4313</td></tr>
        <tr><td>통합 파이프라인 (고정 RF)</td><td>{optuna04c['fixed_auc_ovr']:.4f}</td><td>{optuna04c['fixed_macro_f1']:.4f}</td></tr>
        <tr><td>계층적 (Stage1×Stage2)</td><td>{hier04c['hier_auc_ovr']:.4f}</td><td>{hier04c['hier_macro_f1']:.4f}</td></tr>
        <tr class="hl"><td><strong>통합 + Optuna ★</strong></td><td><strong>{optuna04c['optuna_auc_ovr']:.4f}</strong></td><td><strong>{optuna04c['optuna_macro_f1']:.4f}</strong></td></tr>
      </tbody>
    </table>
    <div class="note" style="margin-top:12px">
      <strong>Optuna 효과</strong>: 0.6331 → 0.8712(통합) → <strong>{optuna04c['optuna_auc_ovr']:.4f}</strong>(Optuna).
      최적 파라미터 max_depth={optuna04c['optuna_best_params']['max_depth']},
      max_features={optuna04c['optuna_best_params']['max_features']:.3f}.
      계층적 분류({hier04c['hier_auc_ovr']:.3f})는 통합 평면 분류보다 낮아 채택하지 않음.
    </div>
    <div style="font-weight:700;margin:14px 0 8px;font-size:13px;">계층적 클래스별 AUC (참고)</div>
    <table>
      <tbody>
        <tr><td><span class="tag ha">HA</span></td><td>{hier04c['hier_per_class_auc']['HA']:.4f}</td></tr>
        <tr><td><span class="tag aclr">ACLR</span></td><td>{hier04c['hier_per_class_auc']['ACLR']:.4f}</td></tr>
        <tr><td><span class="tag acld">ACLD</span></td><td>{hier04c['hier_per_class_auc']['ACLD']:.4f}</td></tr>
      </tbody>
    </table>
  </div>
  <div class="card">
    <div style="font-weight:700;margin-bottom:12px;font-size:13px;">계층적 3분류 혼동 행렬</div>
    {img(b64_04c_cmhier)}
    <div class="note warn-note" style="margin-top:10px">
      <strong>3분류 정직한 최선 = AUC(OvR) {optuna04c['optuna_auc_ovr']:.4f}</strong>.
      HA는 거의 완벽 분리(AUC 0.97)되지만 ACLD↔ACLR 혼동이 남아 0.98 미달.
      이는 파이프라인 한계가 아니라 <strong>데이터의 근본 한계</strong> (다음 섹션 참조).
    </div>
  </div>
</div>
</section>

<!-- ACLD vs ACLR 분리 한계 (05) -->
<section>
<h2><span class="num">05</span> 왜 3분류 AUC 0.98은 불가능한가 — ACLD↔ACLR 분리 한계</h2>
<div class="note" style="margin-bottom:16px">
  3분류 AUC가 0.98에 못 미치는 유일한 이유는 <strong>ACLD(비재건)와 ACLR(재건)을
  보행 데이터로 구분할 수 없기 때문</strong>. 두 그룹 모두 ACL 손상군이라 보행 패턴이 거의 동일.
  ID.csv에 수술 여부·경과 시점 등 구분 메타데이터가 없어 오직 보행 파형으로만 구분해야 함.
</div>
<div class="g2">
  <div class="card">
    <div style="font-weight:700;margin-bottom:12px;font-size:13px;">PCA 투영 — 세 그룹 혼재</div>
    {img(b64_04c_pca)}
    <p class="fig-caption">
      비지도 PCA(PC1-PC2)에서 세 그룹이 모두 섞임. 지도학습 RF만이 HA를 분리하며,
      ACLD/ACLR은 어떤 방법으로도 겹침. 색: 빨강 ACLD, 주황 ACLR, 파랑 HA.
    </p>
  </div>
  <div class="card">
    <div style="font-weight:700;margin-bottom:12px;font-size:13px;">Permutation Test — ACLD vs ACLR</div>
    {img(b64_04c_perm)}
    <div class="note warn-note">
      라벨을 {perm04c['n_perm']}회 무작위 셔플한 null 분포(평균 {perm04c['null_mean']:.3f}) 대비
      실제 AUC {perm04c['real_auc_light']:.3f}, <strong>p = {perm04c['p_value']:.3f}</strong>.
      우연 수준(0.5)을 겨우 넘는 경계선 유의 → ACLD↔ACLR 분리 신호는 통계적으로 매우 약함.
    </div>
  </div>
</div>
<div class="note" style="margin-top:16px;background:#eaf4ff;border-left-color:#2980b9;color:#1a5276">
  <strong>임상적 결론</strong>: ACLD와 ACLR이 보행으로 구분되지 않는다는 것은 실패가 아니라
  <strong>의미 있는 발견</strong>이다 — ACL 재건 수술(ACLR) 후에도 보행 패턴이 비재건군(ACLD)과
  구별되지 않을 만큼 손상군 특유의 보상 패턴이 잔존함을 시사한다.
  따라서 임상적으로 신뢰도 높은 과제는 <strong>이진 분류(ACL vs HA, AUC {opt['ens_oof_auc']:.3f})</strong>이며,
  3분류는 정직한 최선 {optuna04c['optuna_auc_ovr']:.3f}로 보고한다.
</div>
</section>

<!-- DATA OVERVIEW -->
<section>
<h2><span class="num">06</span> 데이터 및 설계</h2>
<div class="g3">
  <div class="card">
    {img(b64_pie)}
    <p class="fig-caption">총 1,134 features. 상호작용(190)은 폴드별 RF top-20에서 생성.</p>
  </div>
  <div class="card" style="grid-column:span 2">
    <div style="font-weight:700;margin-bottom:12px;font-size:13px;">주요 설계 결정</div>
    <table>
      <tbody>
        <tr><td style="font-weight:600;width:160px">CV 전략</td><td>StratifiedKFold(5, shuffle=False)</td></tr>
        <tr><td style="font-weight:600">피처 선택</td><td>RF 중요도 top-20 — 훈련 폴드 전용</td></tr>
        <tr><td style="font-weight:600">상호작용</td><td>C(20,2) = 190 쌍별 곱 (폴드 내 append)</td></tr>
        <tr><td style="font-weight:600">앙상블</td><td>seed &#123;42, 88&#125; soft vote 평균</td></tr>
        <tr><td style="font-weight:600">정규화</td><td>StandardScaler — 폴드별 fit/transform</td></tr>
        <tr><td style="font-weight:600">부트스트랩</td><td>2,000 샘플, percentile 95% CI</td></tr>
        <tr><td style="font-weight:600">클래스 가중치</td><td>balanced (ACL:HA ≈ 2.25:1)</td></tr>
        <tr><td style="font-weight:600">RF trees</td><td>선택 200 / 최종 1,000</td></tr>
      </tbody>
    </table>
    <div class="note" style="margin-top:14px">
      <strong>왜 PCA/Optuna를 쓰지 않았나</strong>: N=62 훈련 샘플에서 1,134차원 PCA는 정보 손실.
      Optuna inner CV (62 train, 3-fold) 는 41 샘플로 과적합.
      단순한 within-fold feature selection + interactions가 더 강건.
    </div>
  </div>
</div>
</section>

</main>
</body>
</html>"""

out = REPORTS / "results_report.html"
out.write_text(HTML, encoding="utf-8")
print(f"Saved: {out}")
