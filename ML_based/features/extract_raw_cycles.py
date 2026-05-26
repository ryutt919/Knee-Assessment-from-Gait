"""
Phase 1: slim_gait → per-stride raw waveform (패딩+마스크)

출력: data/processed/stride_raw_waveforms.parquet
- 메타: subject_id, group, speed, trial_id, stride_idx, cycle_len
- 파형: {joint}_{t:04d} (t=0~max_len, 초과분 NaN)
- 마스크: mask_{joint}_{t:04d} (True=유효)

Stride Trim: trial별 앞 2개·뒤 2개 stride 제거 (config.features.stride_trim)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).parent.parent.parent
ML = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ML))

from utils.preprocess import get_stance_segments, build_stance_contact_signal

SLIM = ROOT / "data" / "processed" / "slim_gait.parquet"
OUT  = ROOT / "data" / "processed" / "stride_raw_waveforms.parquet"

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

SIDES = {
    "injured":      {"heel": "footContacts_2", "toe": "footContacts_3", "suffix": 1},
    "contralateral":{"heel": "footContacts_0", "toe": "footContacts_1", "suffix": 0},
}


def extract_strides_for_trial(
    trial_df: pd.DataFrame,
    joint_cols: dict,
    side: str,
    stride_trim: int,
) -> list[dict]:
    """단일 trial(file_name)에서 stride별 raw 파형 추출."""
    heel_col = SIDES[side]["heel"]
    toe_col  = SIDES[side]["toe"]

    if heel_col not in trial_df.columns or toe_col not in trial_df.columns:
        return []

    contact = build_stance_contact_signal(
        trial_df[heel_col].values.astype(int),
        trial_df[toe_col].values.astype(int),
        mode="heel_toe_or",
    )
    segments = get_stance_segments(contact, max_gap=8)

    if len(segments) <= stride_trim * 2:
        return []

    segments = segments[stride_trim: len(segments) - stride_trim]

    records = []
    for idx, (start, end) in enumerate(segments):
        seg_data = {"stride_idx": idx, "cycle_len": end - start}
        for joint, (right_col, left_col) in joint_cols.items():
            col = right_col if SIDES[side]["suffix"] == 1 else left_col
            if col in trial_df.columns:
                seg_data[f"_raw_{joint}"] = trial_df[col].values[start:end].tolist()
            else:
                seg_data[f"_raw_{joint}"] = []
        records.append(seg_data)

    return records


def run(cfg_path: str | None = None):
    cfg_path = cfg_path or str(ML / "configs" / "config.yaml")
    cfg = OmegaConf.load(cfg_path)
    stride_trim: int = cfg.features.stride_trim

    print(f"[extract_raw_cycles] slim_gait 로드: {SLIM}")
    slim = pd.read_parquet(SLIM)
    print(f"  shape={slim.shape}, subjects={slim['subject_id'].nunique()}")

    all_records: list[dict] = []

    for (subj, grp, spd, fn), trial_df in slim.groupby(
        ["subject_id", "group", "speed", "file_name"]
    ):
        trial_df = trial_df.sort_values("time_ms").reset_index(drop=True)
        for side in ("injured", "contralateral"):
            strides = extract_strides_for_trial(trial_df, JOINT_COLS, side, stride_trim)
            for s in strides:
                rec = {
                    "subject_id": subj,
                    "group": grp,
                    "speed": spd,
                    "trial_id": fn,
                    "side": side,
                    **{k: v for k, v in s.items() if not k.startswith("_raw_")},
                }
                for joint in JOINT_COLS:
                    rec[f"_raw_{joint}"] = s.get(f"_raw_{joint}", [])
                all_records.append(rec)

    if not all_records:
        raise RuntimeError("추출된 stride가 없습니다.")

    cycle_lens = [r["cycle_len"] for r in all_records]
    max_len = int(np.percentile(cycle_lens, 99))
    print(f"  stride 수={len(all_records)}, cycle 길이 99th percentile={max_len}")

    # 이상 stride 제거 (cycle_len > max_len)
    all_records = [r for r in all_records if r["cycle_len"] <= max_len]
    print(f"  99th percentile 초과 제거 후: {len(all_records)} strides")

    # 행 기반 Parquet 컬럼 빌드 (패딩 NaN)
    rows = []
    for r in all_records:
        row = {k: v for k, v in r.items() if not k.startswith("_raw_")}
        for joint in JOINT_COLS:
            raw = r[f"_raw_{joint}"]
            for t in range(max_len):
                row[f"{joint}_{t:04d}"] = float(raw[t]) if t < len(raw) else float("nan")
                row[f"mask_{joint}_{t:04d}"] = t < len(raw)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False)
    size_mb = OUT.stat().st_size / 1e6
    print(f"[extract_raw_cycles] ✅ 저장: {OUT} ({size_mb:.1f}MB, {len(df):,}행)")
    return df


if __name__ == "__main__":
    run()
