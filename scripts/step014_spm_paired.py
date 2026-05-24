"""
step014: SPM1D ACLD vs ACLR paired t-test
- 동일 27쌍 비교 (paired t-test)
- 9채널 × 3속도 = 27 검정
- 결과를 spm_results.csv에 추가 (append)
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

WAVES = ROOT / "data" / "processed" / "waveforms_normalized.parquet"
PAIRING = ROOT / "data" / "processed" / "id_pairing_summary.csv"
OUT_CSV = ROOT / "data" / "processed" / "spm_results.csv"

CHANNELS = [
    "hip_adduction", "hip_int_rotation", "hip_flexion",
    "knee_adduction", "knee_int_rotation", "knee_flexion",
    "ankle_adduction", "ankle_int_rotation", "ankle_dorsiflexion",
]
SPEEDS = ["normal", "slow", "fast"]

try:
    import spm1d
    HAS_SPM = True
except ImportError:
    HAS_SPM = False
    print("[014] 경고: spm1d 미설치")


def get_paired_waveforms(df: pd.DataFrame, pairing: pd.DataFrame, channel: str, speed: str, side: str = "Right"):
    wave_cols = [f"{channel}_{i:03d}" for i in range(101)]
    avail = [c for c in wave_cols if c in df.columns]
    if not avail:
        return np.array([]).reshape(0, 101), np.array([]).reshape(0, 101)

    paired = pairing[pairing["pair_status"] == "paired"]
    Y_acld, Y_aclr = [], []

    for _, row in paired.iterrows():
        acld_id = row["ID_ACLD"]
        aclr_id = row["ID_ACLR"]

        acld_row = df[(df["subject_id"] == acld_id) & (df["speed"] == speed) & (df["side"] == side)]
        aclr_row = df[(df["subject_id"] == aclr_id) & (df["speed"] == speed) & (df["side"] == side)]

        if acld_row.empty or aclr_row.empty:
            continue

        acld_wave = acld_row[avail].values[0].astype(float)
        aclr_wave = aclr_row[avail].values[0].astype(float)

        if np.any(np.isnan(acld_wave)) or np.any(np.isnan(aclr_wave)):
            continue

        Y_acld.append(acld_wave)
        Y_aclr.append(aclr_wave)

    return np.array(Y_acld), np.array(Y_aclr)


def run():
    df = pd.read_parquet(WAVES)
    pairing = pd.read_csv(PAIRING)

    existing_csv = pd.read_csv(OUT_CSV) if OUT_CSV.exists() else pd.DataFrame()
    new_rows = []

    for channel in CHANNELS:
        for speed in SPEEDS:
            Y_acld, Y_aclr = get_paired_waveforms(df, pairing, channel, speed)
            n_pairs = len(Y_acld)
            print(f"[014] {channel} {speed}: {n_pairs}쌍")

            if n_pairs < 5:
                new_rows.append({
                    "comparison": "ACLD_vs_ACLR_paired",
                    "channel": channel, "speed": speed, "side": "Right",
                    "sig_start_pct": None, "sig_end_pct": None,
                    "peak_stat": np.nan, "p": np.nan,
                })
                continue

            spm_result = {"t_max": np.nan, "significant": False, "sig_regions": [], "p_min": np.nan}
            if HAS_SPM:
                try:
                    t = spm1d.stats.ttest_paired(Y_acld, Y_aclr)
                    ti = t.inference(alpha=0.05)
                    t_arr = np.asarray(ti.z)
                    sig_regions = []
                    if ti.h0reject:
                        above = np.abs(t_arr) >= ti.zstar
                        in_region, start = False, 0
                        for i, v in enumerate(above):
                            if v and not in_region:
                                start, in_region = i, True
                            elif not v and in_region:
                                sig_regions.append((start, i))
                                in_region = False
                        if in_region:
                            sig_regions.append((start, len(above)))
                    spm_result = {
                        "t_max": float(np.max(np.abs(t_arr))),
                        "significant": bool(ti.h0reject),
                        "sig_regions": sig_regions,
                        "p_min": float(np.min(ti.p)) if hasattr(ti, "p") else np.nan,
                    }
                    print(f"[014] {channel} {speed}: T_max={spm_result['t_max']:.2f}, sig={spm_result['significant']}")
                except Exception as e:
                    print(f"[014] SPM 오류: {e}")

            if spm_result["sig_regions"]:
                for s, e in spm_result["sig_regions"]:
                    new_rows.append({
                        "comparison": "ACLD_vs_ACLR_paired",
                        "channel": channel, "speed": speed, "side": "Right",
                        "sig_start_pct": s, "sig_end_pct": e,
                        "peak_stat": spm_result["t_max"], "p": spm_result["p_min"],
                    })
            else:
                new_rows.append({
                    "comparison": "ACLD_vs_ACLR_paired",
                    "channel": channel, "speed": speed, "side": "Right",
                    "sig_start_pct": None, "sig_end_pct": None,
                    "peak_stat": spm_result["t_max"], "p": spm_result["p_min"],
                })

    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([existing_csv, new_df], ignore_index=True) if not existing_csv.empty else new_df
    combined.to_csv(OUT_CSV, index=False)

    print(f"[014] ✅ paired t-test {len(CHANNELS)*len(SPEEDS)}개 검정 완료, spm_results.csv 업데이트")


if __name__ == "__main__":
    run()
