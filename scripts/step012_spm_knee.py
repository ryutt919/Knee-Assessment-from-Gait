"""
step012: SPM1D knee_flexion 3그룹 ANOVA (smoke test)
- ACLD vs ACLR vs HA one-way ANOVA
- 3속도 × 1채널 = 3개 검정
- 결과: F-stat 출력, figure 저장

출력: figures/spm_knee_flexion_*.png
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

WAVES = ROOT / "data" / "processed" / "waveforms_normalized.parquet"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

CHANNEL = "knee_flexion"
SPEEDS = ["normal", "slow", "fast"]
GROUPS = ["ACLD", "ACLR", "HA"]

try:
    import spm1d
    HAS_SPM = True
except ImportError:
    HAS_SPM = False
    print("[012] 경고: spm1d 미설치 — numpy fallback 사용")


def get_group_waveforms(df: pd.DataFrame, group: str, speed: str, channel: str, side: str = "Right"):
    sub = df[(df["group"] == group) & (df["speed"] == speed) & (df["side"] == side)]
    wave_cols = [f"{channel}_{i:03d}" for i in range(101)]
    avail = [c for c in wave_cols if c in sub.columns]
    if not avail or sub.empty:
        return np.array([])
    Y = sub[avail].values.astype(float)
    valid = ~np.any(np.isnan(Y), axis=1)
    return Y[valid]


def plot_spm(channel: str, speed: str, Y_groups: dict, fi=None, side: str = "Right"):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    colors = {"ACLD": "#e74c3c", "ACLR": "#3498db", "HA": "#2ecc71"}

    ax = axes[0]
    for grp, Y in Y_groups.items():
        if len(Y) == 0:
            continue
        mean = np.nanmean(Y, axis=0)
        sd = np.nanstd(Y, axis=0)
        x = np.arange(101)
        ax.plot(x, mean, color=colors[grp], label=grp)
        ax.fill_between(x, mean - sd, mean + sd, color=colors[grp], alpha=0.2)
    ax.set_xlabel("Stance phase (%)")
    ax.set_ylabel("Angle (°)")
    ax.set_title(f"{channel.replace('_', ' ').title()} — {speed} ({side})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    if fi is not None and hasattr(fi, "z"):
        ax2.plot(np.arange(101), fi.z, "k-", linewidth=1.5)
        ax2.axhline(y=fi.zstar, color="r", linestyle="--", label=f"threshold (α=0.05)")
        sig_regions = np.where(np.abs(fi.z) >= fi.zstar)[0]
        if len(sig_regions) > 0:
            ax2.fill_between(np.arange(101),
                             np.zeros(101),
                             [fi.zstar if i in set(sig_regions) else 0 for i in range(101)],
                             color="red", alpha=0.3, label="p<0.05")
        ax2.set_ylabel("SPM F")
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "SPM1D not available", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_ylabel("SPM F (N/A)")
    ax2.set_xlabel("Stance phase (%)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = FIGURES / f"spm_{channel}_{speed}_{side}.png"
    plt.savefig(fname, dpi=100)
    plt.close()
    return fname


def run():
    df = pd.read_parquet(WAVES)
    spm_rows = []

    for speed in SPEEDS:
        Y_groups = {g: get_group_waveforms(df, g, speed, CHANNEL) for g in GROUPS}
        sizes = {g: len(Y) for g, Y in Y_groups.items()}
        print(f"[012] {speed}: {sizes}")

        fi = None
        f_stat = np.nan
        if HAS_SPM:
            valid_groups = [Y for Y in Y_groups.values() if len(Y) >= 3]
            if len(valid_groups) == 3:
                try:
                    F = spm1d.stats.anova1(*[Y_groups[g] for g in GROUPS if len(Y_groups[g]) >= 3])
                    fi = F.inference(alpha=0.05)
                    f_stat = float(np.max(fi.z))
                    print(f"[012] {speed} F_max={f_stat:.2f}, significant={fi.h0reject}")
                except Exception as e:
                    print(f"[012] SPM1D 오류: {e}")

        fname = plot_spm(CHANNEL, speed, Y_groups, fi)
        print(f"[012] 그래프 저장: {fname}")

        spm_rows.append({
            "comparison": "ACLD_ACLR_HA",
            "channel": CHANNEL,
            "speed": speed,
            "side": "Right",
            "f_max": f_stat,
            "significant": fi.h0reject if fi else None,
        })

    spm_df = pd.DataFrame(spm_rows)
    spm_df.to_csv(ROOT / "data" / "processed" / "spm_knee_test.csv", index=False)
    print(f"[012] ✅ knee_flexion SPM 완료 — {FIGURES}에 {len(SPEEDS)}개 figure 저장")


if __name__ == "__main__":
    run()
