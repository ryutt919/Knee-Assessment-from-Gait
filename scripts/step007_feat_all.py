"""
step007: 전체 피험자 scalar feature + LSI 계산
- slim_gait.parquet에서 모든 subject × speed 조합
- K1, K2, ROM, IC_angle, K1_timing × 9채널 × 2측 = 90 feature
- LSI = (involved / contralateral) × 100 (ACLD/ACLR)
  Healthy: Left/Right로 계산 (Right=reference)

검증: 결측률 < 20%, LSI ∈ [30, 200]
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.analysis.preprocess import (
    get_stance_segments, build_stance_contact_signal, detect_peaks_with_iqr
)
from utils.peak_utils import get_stance_scalar_features

SLIM = ROOT / "data" / "processed" / "slim_gait.parquet"
PAIRING = ROOT / "data" / "processed" / "id_pairing_summary.csv"
OUT_FEATURES = ROOT / "data" / "processed" / "features_scalar.csv"
OUT_PEAKS = ROOT / "data" / "processed" / "peak_records_v2.csv"

JOINT_COLS = {
    "hip_adduction":    ("jointAngle_42", "jointAngle_54"),
    "hip_int_rotation": ("jointAngle_43", "jointAngle_55"),
    "hip_flexion":      ("jointAngle_44", "jointAngle_56"),
    "knee_adduction":   ("jointAngle_45", "jointAngle_57"),
    "knee_int_rotation":("jointAngle_46", "jointAngle_58"),
    "knee_flexion":     ("jointAngle_47", "jointAngle_59"),
    "ankle_adduction":  ("jointAngle_48", "jointAngle_60"),
    "ankle_int_rotation":("jointAngle_49", "jointAngle_61"),
    "ankle_dorsiflexion":("jointAngle_50", "jointAngle_62"),
}

PEAK_DIRECTION = {
    "hip_flexion": "max", "hip_adduction": "min", "hip_int_rotation": "max",
    "knee_flexion": "max", "knee_adduction": "min", "knee_int_rotation": "max",
    "ankle_dorsiflexion": "max", "ankle_adduction": "min", "ankle_int_rotation": "max",
}

LIMITS = {
    "distance": 30, "prominence": 1.0,
    "iqr_lower_bound": 1.5, "iqr_upper_bound": 2.5,
    "peak_method": "argextrema", "butter_order": 2, "butter_cutoff": 0.1,
}


def extract_subject_features(speed_df: pd.DataFrame, subject_id: str,
                               group: str, injured_leg: str, speed: str) -> dict:
    contra_leg = "Left" if injured_leg == "Right" else "Right"
    rec = {"subject_id": subject_id, "group": group, "speed": speed, "injured_leg": injured_leg}

    inj_contact, _ = build_stance_contact_signal(speed_df, injured_leg)
    contra_contact, _ = build_stance_contact_signal(speed_df, contra_leg)

    for feat, (r_col, l_col) in JOINT_COLS.items():
        direction = PEAK_DIRECTION[feat]
        inj_col = r_col if injured_leg == "Right" else l_col
        contra_col = l_col if injured_leg == "Right" else r_col

        for side_label, col, contact in [
            ("injured", inj_col, inj_contact),
            ("contralateral", contra_col, contra_contact),
        ]:
            signal = speed_df[col].values if col in speed_df.columns else np.full(len(speed_df), np.nan)
            segs = get_stance_segments(contact) if contact is not None else []

            scalar_vals = {"K1": [], "K2": [], "ROM": [], "IC_angle": [], "K1_timing": []}
            for seg_start, seg_end in segs:
                seg = signal[seg_start:seg_end]
                if len(seg) < 5:
                    continue
                sf = get_stance_scalar_features(seg, direction)
                for k, v in sf.items():
                    if not np.isnan(v):
                        scalar_vals[k].append(v)

            for feat_name, vals in scalar_vals.items():
                col_name = f"{feat}_{feat_name}_{side_label}"
                rec[col_name] = float(np.mean(vals)) if vals else np.nan

    for feat in JOINT_COLS:
        for scalar in ["K1", "K2", "ROM", "IC_angle", "K1_timing"]:
            inj_val = rec.get(f"{feat}_{scalar}_injured", np.nan)
            contra_val = rec.get(f"{feat}_{scalar}_contralateral", np.nan)
            if (not np.isnan(inj_val) if isinstance(inj_val, float) else False) and \
               (not np.isnan(contra_val) if isinstance(contra_val, float) else False) and \
               abs(contra_val) > 1e-6:
                rec[f"{feat}_{scalar}_LSI"] = 100.0 * inj_val / contra_val
            else:
                rec[f"{feat}_{scalar}_LSI"] = np.nan

    return rec


def run():
    print("[007] slim_gait.parquet 로드...")
    slim = pd.read_parquet(SLIM)
    pairing = pd.read_csv(PAIRING)
    paired = pairing[pairing["pair_status"] == "paired"]

    injured_map = {}
    id_csv = pd.read_csv(ROOT / "data" / "ID.csv")
    for _, row in id_csv.iterrows():
        injured_map[row["ID"]] = row.get("Injured leg", "Right")

    records = []
    total = 0

    for group in ["ACLD", "ACLR", "HA"]:
        group_df = slim[slim["group"] == group]
        subjects = group_df["subject_id"].unique()

        if group in ("ACLD", "ACLR"):
            id_col = "ID_ACLD" if group == "ACLD" else "ID_ACLR"
            subjects = paired[id_col].dropna().tolist()

        for sub_id in subjects:
            sub_df = group_df[group_df["subject_id"] == sub_id]
            if sub_df.empty:
                continue

            injured_leg = injured_map.get(sub_id, "Right")
            if not isinstance(injured_leg, str) or injured_leg not in ("Right", "Left"):
                injured_leg = "Right"

            for speed in ["normal", "slow", "fast"]:
                speed_df = sub_df[sub_df["speed"] == speed].reset_index(drop=True)
                if len(speed_df) < 100:
                    continue

                rec = extract_subject_features(speed_df, sub_id, group, injured_leg, speed)
                records.append(rec)
                total += 1

        print(f"[007] {group}: {len(subjects)}명 처리")

    df_out = pd.DataFrame(records)
    missing_rate = df_out.isnull().mean().mean()
    print(f"[007] 총 {total}행, 결측률={missing_rate:.1%}")

    df_out.to_csv(OUT_FEATURES, index=False)
    print(f"[007] ✅ features_scalar.csv 저장: {OUT_FEATURES}")


if __name__ == "__main__":
    run()
