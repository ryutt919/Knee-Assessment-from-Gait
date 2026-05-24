"""
step019: Feature ranking + Cohen's d heatmap 그래프
- eta² 내림차순 bar chart (top 20)
- pairwise Cohen's d heatmap (feature × comparison)

출력:
  figures/feature_ranking_eta2.png
  figures/feature_ranking_cohens_d.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).parent.parent
RANKING = ROOT / "data" / "processed" / "feature_ranking.csv"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


def short_name(feat: str) -> str:
    parts = feat.split("_")
    joint = parts[0][0].upper() + parts[1][0].upper() if len(parts) >= 2 else feat
    scalar = parts[-2] if len(parts) >= 3 else ""
    side = "I" if "injured" in feat else ("C" if "contra" in feat else "")
    return f"{joint}.{scalar}.{side}"


def plot_eta2_bar(df: pd.DataFrame):
    top = df.drop_duplicates("feature").nlargest(20, "eta2")
    top = top.reset_index(drop=True)
    labels = [short_name(f) for f in top["feature"]]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top)))
    bars = ax.barh(labels[::-1], top["eta2"].values[::-1], color=colors[::-1])
    ax.set_xlabel("Partial η²", fontsize=12)
    ax.set_title("Feature Importance — Partial η² (Top 20, all speeds)", fontsize=13)
    ax.axvline(0.01, color="gray", linestyle="--", alpha=0.5, label="small (0.01)")
    ax.axvline(0.06, color="orange", linestyle="--", alpha=0.5, label="medium (0.06)")
    ax.axvline(0.14, color="red", linestyle="--", alpha=0.5, label="large (0.14)")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out = FIGURES / "feature_ranking_eta2.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[019] eta² bar chart 저장: {out}")


def plot_cohens_d_heatmap(df: pd.DataFrame):
    d_cols = [c for c in df.columns if c.startswith("d_")]
    if not d_cols:
        print("[019] Cohen's d 컬럼 없음 — heatmap 스킵")
        return

    top_feats = df.drop_duplicates("feature").nlargest(20, "eta2")["feature"].tolist()
    pivot_rows = []
    for feat in top_feats:
        sub = df[df["feature"] == feat].head(1)
        row = {"feature": short_name(feat)}
        for c in d_cols:
            row[c.replace("d_", "")] = float(sub[c].values[0]) if len(sub) > 0 else np.nan
        pivot_rows.append(row)

    hm_df = pd.DataFrame(pivot_rows).set_index("feature")

    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(
        hm_df.astype(float),
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-2,
        vmax=2,
        ax=ax,
        cbar_kws={"label": "Cohen's d"},
    )
    ax.set_title("Pairwise Cohen's d — Top 20 Features", fontsize=13)
    plt.tight_layout()
    out = FIGURES / "feature_ranking_cohens_d.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[019] Cohen's d heatmap 저장: {out}")


def run():
    df = pd.read_csv(RANKING)
    print(f"[019] feature_ranking.csv 로드: {len(df)}행")

    plot_eta2_bar(df)
    plot_cohens_d_heatmap(df)

    n_figures = len(list(FIGURES.glob("feature_ranking*.png")))
    assert n_figures >= 1, f"feature ranking figure 없음"
    print(f"[019] ✅ ranking 그래프 {n_figures}개 저장 완료")


if __name__ == "__main__":
    run()
