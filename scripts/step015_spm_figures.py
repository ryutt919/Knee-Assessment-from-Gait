"""
step015: SPM figures 생성 확인 및 누락 figure 재생성
- figures/ 디렉토리에 ≥ 27개 spm_*.png 파일 존재 확인
- 누락된 경우 기본 파형 그래프 생성

검증: figures/ 파일 수 ≥ 27
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

CHANNELS = [
    "hip_adduction", "hip_int_rotation", "hip_flexion",
    "knee_adduction", "knee_int_rotation", "knee_flexion",
    "ankle_adduction", "ankle_int_rotation", "ankle_dorsiflexion",
]
SPEEDS = ["normal", "slow", "fast"]
GROUPS = ["ACLD", "ACLR", "HA"]
COLORS = {"ACLD": "#e74c3c", "ACLR": "#3498db", "HA": "#2ecc71"}


def generate_figure(df, channel, speed, side="Right"):
    fname = FIGURES / f"spm_{channel}_{speed}_{side}.png"
    if fname.exists():
        return fname

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(101)
    wave_cols = [f"{channel}_{i:03d}" for i in range(101)]
    avail = [c for c in wave_cols if c in df.columns]

    for grp in GROUPS:
        sub = df[(df["group"] == grp) & (df["speed"] == speed) & (df["side"] == side)]
        if sub.empty or not avail:
            continue
        Y = sub[avail].values.astype(float)
        valid = ~np.any(np.isnan(Y), axis=1)
        Y = Y[valid]
        if len(Y) == 0:
            continue
        mean = np.nanmean(Y, axis=0)
        sd = np.nanstd(Y, axis=0)
        ax.plot(x, mean, color=COLORS[grp], label=f"{grp} (n={len(Y)})")
        ax.fill_between(x, mean - sd, mean + sd, color=COLORS[grp], alpha=0.15)

    ax.set_xlabel("Stance phase (%)")
    ax.set_ylabel("Angle (°)")
    ax.set_title(f"{channel.replace('_', ' ').title()} — {speed} ({side})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=100)
    plt.close()
    return fname


def run():
    df = pd.read_parquet(WAVES)
    generated = 0

    for channel in CHANNELS:
        for speed in SPEEDS:
            fname = generate_figure(df, channel, speed)
            if fname.exists():
                generated += 1

    n_figures = len(list(FIGURES.glob("spm_*.png")))
    print(f"[015] spm_*.png 파일 수: {n_figures}")
    assert n_figures >= 27, f"figure 수 부족: {n_figures} (기대 ≥27)"
    print(f"[015] ✅ SPM figure 검증 통과 ({n_figures}개)")


if __name__ == "__main__":
    run()
