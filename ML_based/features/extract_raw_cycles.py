"""
Phase 1: slim_gait → per-stride raw waveform (패딩+마스크)

출력: data/processed/stride_raw_waveforms.parquet
- 메타: subject_id, group, speed, trial_id, stride_idx, cycle_len
- 파형: {joint}_{t:04d} (t=0~max_len, 초과분 NaN)
- 마스크: mask_{joint}_{t:04d} (True=유효)

Stride Trim: trial별 앞 2개·뒤 2개 stride 제거 (config.features.stride_trim)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).parent.parent.parent
ML = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ML))

ID_CSV = ROOT / "data" / "ID.csv"

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

HEEL_COLS = {"Left": "footContacts_0", "Right": "footContacts_2"}


def _load_cfg(cfg: DictConfig | str | Path | None) -> DictConfig:
    if cfg is None:
        return OmegaConf.load(ML / "configs" / "config.yaml")
    if isinstance(cfg, (str, Path)):
        return OmegaConf.load(cfg)
    return cfg


def _data_root(cfg: DictConfig) -> Path:
    root = Path(str(cfg.data.root))
    if not root.is_absolute():
        root = (ML / root).resolve()
    return root


def _input_output_paths(cfg: DictConfig) -> tuple[Path, Path]:
    data_root = _data_root(cfg)
    slim = data_root / cfg.data.get("raw_slim", "slim_gait.parquet")
    out = data_root / cfg.data.get("raw_cycles", "stride_raw_waveforms.parquet")
    return slim, out


def _injured_map() -> dict[str, str]:
    id_df = pd.read_csv(ID_CSV)
    mapping: dict[str, str] = {}
    for _, row in id_df.iterrows():
        leg = row.get("Injured leg", "")
        leg = str(leg).strip() if pd.notna(leg) else ""
        if leg in ("Left", "Right"):
            mapping[str(row["ID"])] = leg
    return mapping


def _injured_leg_for(subject_id: str, group: str, injured: dict[str, str]) -> str:
    if group in ("ACLD", "ACLR"):
        if subject_id not in injured:
            raise ValueError(f"{subject_id} ({group})의 Injured leg metadata가 없습니다.")
        return injured[subject_id]
    return "Right"


def _side_basis(group: str, actual_leg: str, injured_leg: str) -> str:
    if group in ("ACLD", "ACLR"):
        return "injured" if actual_leg == injured_leg else "contralateral"
    return "injured" if actual_leg == "Right" else "contralateral"


def _cycle_segments(heel_signal: np.ndarray, min_samples: int = 40) -> list[tuple[int, int]]:
    """Heel-strike to next heel-strike full gait cycles."""
    binary = np.asarray(pd.to_numeric(pd.Series(heel_signal), errors="coerce").fillna(0), dtype=int)
    heel_strikes = np.where(np.diff(binary) == 1)[0] + 1
    segments: list[tuple[int, int]] = []
    for i in range(len(heel_strikes) - 1):
        start = int(heel_strikes[i])
        end = int(heel_strikes[i + 1])
        if end - start >= min_samples:
            segments.append((start, end))
    return segments


def extract_strides_for_trial(
    trial_df: pd.DataFrame,
    joint_cols: dict,
    actual_leg: str,
    stride_trim: int,
) -> list[dict]:
    """단일 trial(file_name)에서 stride별 raw 파형 추출."""
    heel_col = HEEL_COLS[actual_leg]
    if heel_col not in trial_df.columns:
        return []

    segments = _cycle_segments(trial_df[heel_col].to_numpy())
    n_raw = len(segments)

    if n_raw <= stride_trim * 2:
        return []

    segments = segments[stride_trim:n_raw - stride_trim]
    n_trimmed = len(segments)

    records = []
    for idx, (start, end) in enumerate(segments):
        seg_data = {
            "stride_idx": idx,
            "stride_idx_original": idx + stride_trim,
            "cycle_len": end - start,
            "n_cycles_raw": n_raw,
            "n_cycles_trimmed": n_trimmed,
        }
        for joint, (right_col, left_col) in joint_cols.items():
            col = right_col if actual_leg == "Right" else left_col
            if col in trial_df.columns:
                seg_data[f"_raw_{joint}"] = trial_df[col].values[start:end].tolist()
            else:
                seg_data[f"_raw_{joint}"] = []
        records.append(seg_data)

    return records


def run(cfg_path: str | None = None):
    cfg = _load_cfg(cfg_path)
    stride_trim: int = cfg.features.stride_trim
    slim_path, out_path = _input_output_paths(cfg)

    print(f"[extract_raw_cycles] slim_gait 로드: {slim_path}")
    slim = pd.read_parquet(slim_path)
    print(f"  shape={slim.shape}, subjects={slim['subject_id'].nunique()}")

    injured = _injured_map()
    all_records: list[dict] = []

    for (subj, grp, spd, fn), trial_df in slim.groupby(
        ["subject_id", "group", "speed", "file_name"]
    ):
        trial_df = trial_df.sort_values("time_ms").reset_index(drop=True)
        injured_leg = _injured_leg_for(str(subj), str(grp), injured)
        for actual_leg in ("Right", "Left"):
            side = _side_basis(str(grp), actual_leg, injured_leg)
            strides = extract_strides_for_trial(trial_df, JOINT_COLS, actual_leg, stride_trim)
            for s in strides:
                rec = {
                    "subject_id": subj,
                    "group": grp,
                    "speed": spd,
                    "trial_id": fn,
                    "side": side,
                    "side_basis": side,
                    "actual_leg": actual_leg,
                    "injured_leg": injured_leg,
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    size_mb = out_path.stat().st_size / 1e6
    print(f"[extract_raw_cycles] ✅ 저장: {out_path} ({size_mb:.1f}MB, {len(df):,}행)")
    return df


def extract(cfg: DictConfig | str | Path | None = None):
    return run(cfg)


if __name__ == "__main__":
    run()
