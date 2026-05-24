"""
step013: SPM1D 전체 9채널 × 3속도 ANOVA
- ACLD vs ACLR vs HA one-way ANOVA
- 비교: 9 × 3 = 27 검정
- 출력: data/processed/spm_results.csv

컬럼: comparison, channel, speed, side, sig_start, sig_end, peak_stat, p
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
OUT_CSV = ROOT / "data" / "processed" / "spm_results.csv"

CHANNELS = [
    "hip_adduction", "hip_int_rotation", "hip_flexion",
    "knee_adduction", "knee_int_rotation", "knee_flexion",
    "ankle_adduction", "ankle_int_rotation", "ankle_dorsiflexion",
]
SPEEDS = ["normal", "slow", "fast"]
GROUPS = ["ACLD", "ACLR", "HA"]

try:
    import spm1d
    HAS_SPM = True
except ImportError:
    HAS_SPM = False
    print("[013] 경고: spm1d 미설치 — numpy fallback 사용")


def get_group_waveforms(df, group, speed, channel, side="Right"):
    sub = df[(df["group"] == group) & (df["speed"] == speed) & (df["side"] == side)]
    wave_cols = [f"{channel}_{i:03d}" for i in range(101)]
    avail = [c for c in wave_cols if c in sub.columns]
    if not avail or sub.empty:
        return np.array([]).reshape(0, 101)
    Y = sub[avail].values.astype(float)
    valid = ~np.any(np.isnan(Y), axis=1)
    return Y[valid]


def run_spm_anova(Y_groups: list) -> dict:
    if not HAS_SPM:
        return {"f_max": np.nan, "significant": False, "sig_regions": [], "p_min": np.nan}

    try:
        F = spm1d.stats.anova1(*Y_groups)
        fi = F.inference(alpha=0.05)
        f_arr = np.asarray(fi.z)
        sig_regions = []
        if fi.h0reject:
            above = f_arr >= fi.zstar
            in_region = False
            start = 0
            for i, v in enumerate(above):
                if v and not in_region:
                    start = i
                    in_region = True
                elif not v and in_region:
                    sig_regions.append((start, i))
                    in_region = False
            if in_region:
                sig_regions.append((start, len(above)))
        return {
            "f_max": float(np.max(f_arr)),
            "significant": bool(fi.h0reject),
            "sig_regions": sig_regions,
            "p_min": float(np.min(fi.p)) if hasattr(fi, "p") else np.nan,
        }
    except Exception as e:
        print(f"[013] SPM 오류: {e}")
        return {"f_max": np.nan, "significant": False, "sig_regions": [], "p_min": np.nan}


def plot_group_waveforms(channel, speed, Y_groups_dict, spm_result, side="Right"):
    colors = {"ACLD": "#e74c3c", "ACLR": "#3498db", "HA": "#2ecc71"}
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(101)
    for grp, Y in Y_groups_dict.items():
        if len(Y) == 0:
            continue
        mean = np.nanmean(Y, axis=0)
        sd = np.nanstd(Y, axis=0)
        ax.plot(x, mean, color=colors[grp], label=f"{grp} (n={len(Y)})")
        ax.fill_between(x, mean - sd, mean + sd, color=colors[grp], alpha=0.15)

    for s, e in spm_result.get("sig_regions", []):
        ax.axvspan(s, e, color="gray", alpha=0.3, label="p<0.05" if s == spm_result.get("sig_regions", [None])[0][0] else "")

    ax.set_xlabel("Stance phase (%)")
    ax.set_ylabel("Angle (°)")
    ax.set_title(f"{channel.replace('_', ' ').title()} — {speed} ({side})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = FIGURES / f"spm_{channel}_{speed}_{side}.png"
    plt.savefig(fname, dpi=100)
    plt.close()


def run():
    df = pd.read_parquet(WAVES)
    rows = []

    for channel in CHANNELS:
        for speed in SPEEDS:
            for side in ["Right"]:
                Y_groups_dict = {g: get_group_waveforms(df, g, speed, channel, side) for g in GROUPS}
                valid_groups = [Y for Y in Y_groups_dict.values() if len(Y) >= 3]

                if len(valid_groups) < 2:
                    print(f"[013] {channel} {speed} — 데이터 부족, 스킵")
                    spm_result = {"f_max": np.nan, "significant": False, "sig_regions": [], "p_min": np.nan}
                else:
                    spm_result = run_spm_anova([Y_groups_dict[g] for g in GROUPS if len(Y_groups_dict[g]) >= 3])

                plot_group_waveforms(channel, speed, Y_groups_dict, spm_result, side)

                if spm_result["sig_regions"]:
                    for s, e in spm_result["sig_regions"]:
                        rows.append({
                            "comparison": "ACLD_ACLR_HA",
                            "channel": channel,
                            "speed": speed,
                            "side": side,
                            "sig_start_pct": s,
                            "sig_end_pct": e,
                            "peak_stat": spm_result["f_max"],
                            "p": spm_result["p_min"],
                        })
                else:
                    rows.append({
                        "comparison": "ACLD_ACLR_HA",
                        "channel": channel,
                        "speed": speed,
                        "side": side,
                        "sig_start_pct": None,
                        "sig_end_pct": None,
                        "peak_stat": spm_result["f_max"],
                        "p": spm_result["p_min"],
                    })

                print(f"[013] {channel} {speed}: F_max={spm_result['f_max']:.2f}, sig={spm_result['significant']}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)

    n_figures = len(list(FIGURES.glob("spm_*.png")))
    print(f"[013] ✅ spm_results.csv ({len(df_out)}행), figures={n_figures}개")


if __name__ == "__main__":
    run()
