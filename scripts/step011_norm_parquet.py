"""
step011: waveforms_normalized.parquet 생성
- slim_gait.parquet에서 각 subject × speed × side별 평균 파형 계산
- 각 joint channel별 101pt 정규화 파형 저장

출력: data/processed/waveforms_normalized.parquet
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.analysis.preprocess import get_stance_segments, build_stance_contact_signal
from utils.norm_utils import mean_waveform

SLIM = ROOT / "data" / "processed" / "slim_gait.parquet"
OUT = ROOT / "data" / "processed" / "waveforms_normalized.parquet"

JOINT_COLS = {
    "hip_adduction":     ("jointAngle_42", "jointAngle_54"),
    "hip_int_rotation":  ("jointAngle_43", "jointAngle_55"),
    "hip_flexion":       ("jointAngle_44", "jointAngle_56"),
    "knee_adduction":    ("jointAngle_45", "jointAngle_57"),
    "knee_int_rotation": ("jointAngle_46", "jointAngle_58"),
    "knee_flexion":      ("jointAngle_47", "jointAngle_59"),
    "ankle_adduction":   ("jointAngle_48", "jointAngle_60"),
    "ankle_int_rotation":("jointAngle_49", "jointAngle_61"),
    "ankle_dorsiflexion":("jointAngle_50", "jointAngle_62"),
}


def run():
    print("[011] slim_gait.parquet 로드...")
    slim = pd.read_parquet(SLIM)

    records = []
    for (sub_id, speed), sub_df in slim.groupby(["subject_id", "speed"]):
        sub_df = sub_df.reset_index(drop=True)
        group = sub_df["group"].iloc[0]

        for side in ["Right", "Left"]:
            try:
                contact, _ = build_stance_contact_signal(sub_df, side)
            except Exception:
                continue

            segs = get_stance_segments(contact)
            if len(segs) < 3:
                continue

            rec = {
                "subject_id": sub_id,
                "group": group,
                "speed": speed,
                "side": side,
                "n_strides": len(segs),
            }

            for feat, (r_col, l_col) in JOINT_COLS.items():
                col = r_col if side == "Right" else l_col
                signal = sub_df[col].values if col in sub_df.columns else np.zeros(len(sub_df))
                wave = mean_waveform(signal, segs)
                for i, v in enumerate(wave):
                    rec[f"{feat}_{i:03d}"] = float(v)

            records.append(rec)

    df_out = pd.DataFrame(records)
    print(f"[011] 파형 레코드: {len(df_out)}행")

    meta_cols = ["subject_id", "group", "speed", "side", "n_strides"]
    wave_cols = [c for c in df_out.columns if c not in meta_cols]
    print(f"[011] 파형 컬럼 수: {len(wave_cols)} (기대: 9채널 × 101 = 909)")

    df_out.to_parquet(OUT, index=False)
    size_mb = OUT.stat().st_size / 1e6
    print(f"[011] ✅ waveforms_normalized.parquet 저장: {OUT} ({size_mb:.1f}MB)")


if __name__ == "__main__":
    run()
