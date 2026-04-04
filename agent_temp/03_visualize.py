"""
03_visualize.py
ACL 보행 생체역학 시각화 파이프라인

생성 도표:
1. Boxplot — 그룹 × 속도 조건별 피처 분포 (유의성 표기 포함)
2. Correlation Heatmap — Z-score 정규화된 피처 × 그룹 더미 Point-Biserial 상관계수
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 헤드리스 환경 대응
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import logging
import warnings

warnings.filterwarnings("ignore")

# ───────────────────────────────────────────────
# 로깅 설정
# ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ───────────────────────────────────────────────
# 경로 설정
# ───────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MDS_DIR     = os.path.join(BASE_DIR, "mds")
FIG_DIR     = os.path.join(MDS_DIR, "figures")
PATH_DATA   = os.path.join(MDS_DIR, "analysis_data.csv")
PATH_STATS  = os.path.join(MDS_DIR, "stats_result.csv")
PATH_CORR   = os.path.join(MDS_DIR, "correlation_result.csv")
os.makedirs(FIG_DIR, exist_ok=True)

# ───────────────────────────────────────────────
# 시각화 스타일 설정
# ───────────────────────────────────────────────
PALETTE     = {"ACLD": "#E07B54", "ACLR": "#5B84C4", "Healthy": "#62A87C"}
GROUP_ORDER = ["Healthy", "ACLD", "ACLR"]
SPEED_ORDER = ["slow", "normal", "fast"]
FIG_DPI     = 150
FIG_W, FIG_H = 14, 5   # Boxplot 기준 (속도 3개 × 그룹 3개)

# 피처별 한글/표기 레이블
FEAT_LABEL = {
    "hip_flexion_injured"           : "Hip Flexion\n(Injured, rad)",
    "hip_adduction_injured"         : "Hip Adduction\n(Injured, rad)",
    "knee_flexion_injured"          : "Knee Flexion\n(Injured, rad)",
    "knee_adduction_injured"        : "Knee Adduction\n(Injured, rad)",
    "knee_int_rotation_injured"     : "Knee Int.Rot\n(Injured, rad)",
    "ankle_dorsiflexion_injured"    : "Ankle Dorsi.\n(Injured, rad)",
    "hip_flexion_contralateral"     : "Hip Flexion\n(Contra, rad)",
    "knee_flexion_contralateral"    : "Knee Flexion\n(Contra, rad)",
    "ankle_dorsiflexion_contralateral": "Ankle Dorsi.\n(Contra, rad)",
    "hip_flexion_LSI"               : "Hip Flex LSI (%)",
    "knee_flexion_LSI"              : "Knee Flex LSI (%)",
    "ankle_dorsiflexion_LSI"        : "Ankle Dorsi. LSI (%)",
    "gait_speed_mps"                : "Gait Speed (m/s)",
    "cadence_spm"                   : "Cadence (spm)",
    "stride_length_mean_m"          : "Stride Length (m)",
    "step_width_mean_m_orth"        : "Step Width (m)",
    "double_support_pct"            : "Double Support (%)",
    "single_support_L_pct"         : "Single Support L (%)",
    "single_support_R_pct"         : "Single Support R (%)",
}

ALPHA = 0.05


# ───────────────────────────────────────────────
# 유의성 브래킷 표기 헬퍼
# ───────────────────────────────────────────────
def significance_mark(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"


def add_stat_bracket(ax, x1: float, x2: float, y: float, h: float, p: float):
    """두 그룹 간 유의성 브래킷 그리기"""
    mark = significance_mark(p)
    if not mark or mark == "ns":
        return
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c="black")
    ax.text((x1 + x2) / 2, y + h, mark, ha="center", va="bottom", fontsize=9)


# ───────────────────────────────────────────────
# Plot 1: Boxplot (피처별 × 속도 조건 Facet)
# ───────────────────────────────────────────────
def plot_boxplot(data: pd.DataFrame, stats_df: pd.DataFrame, feature: str):
    """
    한 피처에 대해 3개 속도 조건(slow/normal/fast)을 열(Column)로 나누어
    그룹(Healthy/ACLD/ACLR)의 분포를 Boxplot으로 시각화.
    사후 검정 유의성 브래킷을 자동으로 추가.
    """
    if feature not in data.columns:
        return

    label = FEAT_LABEL.get(feature, feature)
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H), sharey=True)
    fig.suptitle(f"{label}", fontsize=13, fontweight="bold", y=1.02)

    # 유의성 p값 로드
    feat_stat = stats_df[stats_df["feature"] == feature]
    p_pairs = {}
    if not feat_stat.empty:
        row = feat_stat.iloc[0]
        for col in row.index:
            if "tukey_p_" in col or "dunn_p_" in col:
                # 컬럼명에서 두 그룹 추출 (예: tukey_p_ACLD_vs_ACLR)
                parts = col.replace("tukey_p_", "").replace("dunn_p_", "").split("_vs_")
                if len(parts) == 2:
                    p_pairs[(parts[0], parts[1])] = row[col]

    for ax, speed in zip(axes, SPEED_ORDER):
        speed_data = data[data["speed"] == speed]

        if speed_data.empty:
            ax.set_title(speed, fontsize=11)
            continue

        sns.boxplot(
            data      = speed_data,
            x         = "group",
            y         = feature,
            order     = GROUP_ORDER,
            palette   = PALETTE,
            ax        = ax,
            width     = 0.55,
            linewidth = 1.2,
            flierprops= dict(marker="o", markersize=3, alpha=0.4),
        )
        ax.set_title(speed.capitalize(), fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel(label if ax == axes[0] else "")
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        # 유의성 브래킷 추가
        y_max = speed_data[feature].dropna().quantile(0.95)
        y_range = speed_data[feature].dropna().std() * 0.4
        h = y_range * 0.2
        offsets = 0

        for (g1, g2), p_val in p_pairs.items():
            if significance_mark(p_val) and significance_mark(p_val) != "ns":
                x1 = GROUP_ORDER.index(g1) if g1 in GROUP_ORDER else None
                x2 = GROUP_ORDER.index(g2) if g2 in GROUP_ORDER else None
                if x1 is not None and x2 is not None:
                    add_stat_bracket(ax, x1, x2, y_max + offsets * h * 2, h, p_val)
                    offsets += 1

    # 범례 추가
    legend_patches = [
        mpatches.Patch(color=v, label=k) for k, v in PALETTE.items()
    ]
    fig.legend(handles=legend_patches, loc="upper right",
               bbox_to_anchor=(1.0, 1.0), fontsize=9)

    plt.tight_layout()
    safe_name = feature.replace("/", "_").replace(" ", "_")
    out_path  = os.path.join(FIG_DIR, f"boxplot_{safe_name}.png")
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  저장: {out_path}")


# ───────────────────────────────────────────────
# Plot 2: Correlation Heatmap
# ───────────────────────────────────────────────
def plot_correlation_heatmap(corr_df: pd.DataFrame):
    """
    Z-score 정규화된 피처 × 그룹 더미 변수 간
    Point-Biserial 상관계수 히트맵
    """
    dummy_cols  = ["is_ACLD", "is_ACLR", "is_Healthy"]
    r_cols      = [f"r_{d}" for d in dummy_cols]
    sig_cols    = [f"sig_{d}" for d in dummy_cols]
    feat_labels = [FEAT_LABEL.get(f, f) for f in corr_df["feature"]]

    r_matrix    = corr_df[r_cols].values
    sig_matrix  = corr_df[sig_cols].values

    fig, ax = plt.subplots(figsize=(7, max(8, len(corr_df) * 0.42)))

    im = ax.imshow(
        r_matrix,
        aspect = "auto",
        cmap   = "RdBu_r",
        vmin   = -0.7, vmax = 0.7,
    )

    # 컬럼/로우 레이블
    ax.set_xticks(range(len(r_cols)))
    ax.set_xticklabels(["ACLD", "ACLR", "Healthy"], fontsize=11)
    ax.set_yticks(range(len(feat_labels)))
    ax.set_yticklabels(feat_labels, fontsize=8)

    # 셀 내 r값 + 유의성 기호
    for i in range(len(corr_df)):
        for j in range(len(r_cols)):
            r_val = r_matrix[i, j]
            sig   = sig_matrix[i, j]
            color = "white" if abs(r_val) > 0.35 else "black"
            text  = f"{r_val:.2f}\n{sig}"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=7.5, color=color, fontweight="bold" if sig != "ns" else "normal")

    # 컬러바
    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.6)
    cbar.set_label("Point-Biserial r", fontsize=9)

    ax.set_title("Feature–Group Correlation Heatmap\n(Z-score normalized features)",
                 fontsize=12, fontweight="bold", pad=14)
    ax.set_xlabel("Group (dummy-coded)", fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "correlation_heatmap.png")
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  저장: {out_path}")


# ───────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────
def main():
    data     = pd.read_csv(PATH_DATA)
    stats_df = pd.read_csv(PATH_STATS)
    corr_df  = pd.read_csv(PATH_CORR)

    log.info(f"데이터 로드: {data.shape}, 통계: {stats_df.shape}")

    # Plot 1: 모든 피처의 Boxplot
    log.info("\n=== Boxplot 생성 ===")
    all_features = list(FEAT_LABEL.keys())
    for feat in all_features:
        if feat in data.columns:
            log.info(f"  Boxplot: {feat}")
            plot_boxplot(data, stats_df, feat)

    # Plot 2: Correlation Heatmap
    log.info("\n=== Correlation Heatmap 생성 ===")
    plot_correlation_heatmap(corr_df)

    log.info(f"\n✅ 모든 그림 저장 완료 → {FIG_DIR}")


if __name__ == "__main__":
    main()
