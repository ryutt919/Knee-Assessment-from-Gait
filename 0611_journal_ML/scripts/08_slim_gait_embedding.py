"""
08_slim_gait_embedding.py — UMAP figure (v3)

데이터: misc/data_misc/embedding_results_cnn_Master_Gait_Dataset_lower.parquet
        CNN 1D-GAP 128-dim 벡터 (이미 생성됨)
필터:   slim_gait.parquet 피험자와 겹치는 78명만
집계:   subject × speed 평균 (~234 포인트)
UMAP:   서브셋으로 재계산 (기존 umap_x/y는 114명 전체 기준)
출력:
  figures/sandbox/20260617/slim_gait_umap_v3_cnn.png
  data/processed/slim_gait_umap_v3_coords.parquet
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

ROOT    = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "figures" / "sandbox" / "20260617"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GROUP_COLORS  = {"ACLD": "#c0392b", "ACLR": "#e67e22", "HA": "#2980b9"}
SPEED_COLORS  = {"slow": "#5d8aa8", "normal": "#6aab6a", "fast": "#c9773b"}
SPEED_MARKERS = {"slow": "o", "normal": "^", "fast": "s"}
GROUP_MARKERS = {"ACLD": "o", "ACLR": "^", "HA": "s"}

plt.rcParams.update({
    "font.family": "Apple SD Gothic Neo",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "axes.unicode_minus": False,
})


# ── 1. CNN 임베딩 로드 + slim_gait 피험자 필터 ────────────────────────────────
def load_cnn_embedding(root: Path) -> pd.DataFrame:
    emb_path = root / "misc" / "data_misc" / "embedding_results_cnn_Master_Gait_Dataset_lower.parquet"
    slim_path = root / "data" / "processed" / "slim_gait.parquet"

    slim_subjects = set(pd.read_parquet(slim_path, columns=["subject_id"])["subject_id"].unique())

    df = pd.read_parquet(emb_path)

    # slim_gait 피험자 필터
    df = df[df["subject_id"].isin(slim_subjects)].copy()

    # 그룹 이름 통일
    df["group"] = df["group"].replace({"Healthy adults": "HA"})

    # Healthy adolescents 등 slim_gait에 없는 그룹 제거 (필터 후 자동 없음)
    df = df[df["group"].isin(GROUP_COLORS)].copy()

    print(f"  [CNN] 필터 후: {df['subject_id'].nunique()}명, {len(df)} trials")
    return df


# ── 2. Subject × Speed 평균 집계 ─────────────────────────────────────────────
def aggregate_by_subject_speed(df: pd.DataFrame) -> pd.DataFrame:
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    agg = (
        df.groupby(["subject_id", "speed"])
        .agg(group=("group", "first"), **{c: (c, "mean") for c in emb_cols})
        .reset_index()
    )
    print(f"  [집계] {len(agg)} 포인트 (subject × speed)")
    return agg, emb_cols


# ── 3. UMAP 재계산 ────────────────────────────────────────────────────────────
def run_umap(agg: pd.DataFrame, emb_cols: list) -> pd.DataFrame:
    X = StandardScaler().fit_transform(agg[emb_cols].values)
    print("  [UMAP] 2D 축소 중...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(X)
    agg = agg.copy()
    agg["umap_x"] = coords[:, 0]
    agg["umap_y"] = coords[:, 1]
    return agg


# ── 4. Figure ─────────────────────────────────────────────────────────────────
def draw_group_panel(ax, df):
    for group, gdf in df.groupby("group"):
        ax.scatter(
            gdf["umap_x"], gdf["umap_y"],
            c=GROUP_COLORS[group], marker="o",
            s=70, alpha=0.82, linewidths=0.3, edgecolors="white",
            label=group,
        )
    ax.set_xlabel("UMAP 1", labelpad=6)
    ax.set_ylabel("UMAP 2", labelpad=6)
    ax.legend(loc="upper right", framealpha=0.85, fontsize=10, title="Group")


# ── 메인 ─────────────────────────────────────────────────────────────────────
print("▶ CNN 임베딩 로드 중 (misc/data_misc/)...")
df_raw = load_cnn_embedding(ROOT)

agg, emb_cols = aggregate_by_subject_speed(df_raw)
print(f"  그룹 분포: {agg['group'].value_counts().to_dict()}")

agg = run_umap(agg, emb_cols)

# 좌표 저장
coord_path = ROOT / "data" / "processed" / "slim_gait_umap_v3_coords.parquet"
agg[["subject_id", "speed", "group", "umap_x", "umap_y"]].to_parquet(coord_path, index=False)
print(f"  좌표 저장: {coord_path.name}")

fig, ax = plt.subplots(figsize=(7, 6))

draw_group_panel(ax, agg)

n_subj = agg["subject_id"].nunique()
ax.set_title(f"UMAP  -  CNN Embedding  (n={n_subj} subjects)", fontsize=12, fontweight="bold", pad=10)

out_path = OUT_DIR / "slim_gait_umap_v3_cnn.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\n✅ 저장 완료 → {out_path}")
