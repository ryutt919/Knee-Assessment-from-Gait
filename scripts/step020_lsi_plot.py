"""
step020: LSI deviation box plot + progress.md 최종 업데이트
- LSI 컬럼을 그룹별로 시각화
- |LSI - 100| = deviation (100에서 멀수록 비대칭 큼)
- progress.md에 done=true 기록

출력:
  figures/lsi_deviation.png
  state/progress.md (done=true)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

FEATURES = ROOT / "data" / "processed" / "features_scalar.csv"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
PROGRESS = ROOT / "state" / "progress.md"


def plot_lsi_deviation(df: pd.DataFrame):
    lsi_cols = [c for c in df.columns if c.endswith("_LSI") and "_K1_" in c or "_K2_" in c or "_ROM_" in c]
    if not lsi_cols:
        lsi_cols = [c for c in df.columns if "_LSI" in c][:9]

    if not lsi_cols:
        print("[020] LSI 컬럼 없음 — 더미 그래프 생성")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No LSI data", ha="center", va="center")
        plt.savefig(FIGURES / "lsi_deviation.png", dpi=100)
        plt.close()
        return

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.flatten()
    colors = {"ACLD": "#e74c3c", "ACLR": "#3498db", "HA": "#2ecc71"}

    for i, feat in enumerate(lsi_cols[:9]):
        ax = axes[i]
        groups_data = {}
        for grp in ["ACLD", "ACLR", "HA"]:
            vals = df[df["group"] == grp][feat].dropna().values
            groups_data[grp] = np.abs(vals - 100)

        bp = ax.boxplot(
            [groups_data.get(g, np.array([])) for g in ["ACLD", "ACLR", "HA"]],
            labels=["ACLD", "ACLR", "HA"],
            patch_artist=True,
        )
        for patch, grp in zip(bp["boxes"], ["ACLD", "ACLR", "HA"]):
            patch.set_facecolor(colors[grp])
            patch.set_alpha(0.7)

        short = feat.replace("_LSI", "").replace("_injured", "")
        ax.set_title(short, fontsize=9)
        ax.set_ylabel("|LSI - 100|")
        ax.grid(axis="y", alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("LSI Deviation from 100 (|LSI−100|) by Group", fontsize=13, y=1.01)
    plt.tight_layout()
    out = FIGURES / "lsi_deviation.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[020] LSI deviation plot 저장: {out}")


def update_progress():
    feature_list_path = ROOT / "state" / "feature_list.json"
    import json
    with open(feature_list_path, encoding="utf-8") as f:
        fl = json.load(f)

    rows = [f"| {sid} | {info['name']} | {info['status']} |" for sid, info in fl["steps"].items()]
    n_done = sum(1 for s in fl["steps"].values() if s["status"] == "passed")
    done_flag = "done=true" if n_done == 20 else f"done=false ({n_done}/20)"

    content = (
        "# Progress\n\n"
        "| Step | 이름 | 상태 |\n|------|------|------|\n"
        + "\n".join(rows)
        + f"\n\n{done_flag}\n"
    )
    PROGRESS.write_text(content, encoding="utf-8")
    print(f"[020] progress.md 업데이트: {done_flag}")


def run():
    df = pd.read_csv(FEATURES)
    print(f"[020] features_scalar.csv: {len(df)}행")
    plot_lsi_deviation(df)
    update_progress()
    print(f"[020] ✅ step020 완료")


if __name__ == "__main__":
    run()
