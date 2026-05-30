"""
Cycle-level 101pt waveform extraction.

Input:
  data/processed/slim_gait.parquet

Output:
  data/processed/cycle_waveforms_101.parquet
  data/processed/cycle_waveforms_101_summary.json

Each row is one trimmed full gait cycle:
  subject_id x speed x trial_id x actual_leg x cycle_idx

Trim rule:
  drop the first 2 and last 2 cycles per trial/actual_leg by default
  (config.features.stride_trim).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
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
    "hip_adduction":      ("jointAngle_42", "jointAngle_54"),
    "hip_int_rotation":   ("jointAngle_43", "jointAngle_55"),
    "hip_flexion":        ("jointAngle_44", "jointAngle_56"),
    "knee_adduction":     ("jointAngle_45", "jointAngle_57"),
    "knee_int_rotation":  ("jointAngle_46", "jointAngle_58"),
    "knee_flexion":       ("jointAngle_47", "jointAngle_59"),
    "ankle_adduction":    ("jointAngle_48", "jointAngle_60"),
    "ankle_int_rotation": ("jointAngle_49", "jointAngle_61"),
    "ankle_dorsiflexion": ("jointAngle_50", "jointAngle_62"),
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


def _output_paths(cfg: DictConfig) -> tuple[Path, Path]:
    data_root = _data_root(cfg)
    name = cfg.data.get("cycle_waveforms_101", "cycle_waveforms_101.parquet")
    out = data_root / name
    summary = out.with_name(f"{out.stem}_summary.json")
    return out, summary


def _slim_path(cfg: DictConfig) -> Path:
    data_root = _data_root(cfg)
    return data_root / cfg.data.get("raw_slim", "slim_gait.parquet")


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


def _stride_segments(heel_signal: np.ndarray, min_samples: int = 40) -> list[tuple[int, int]]:
    binary = np.asarray(pd.to_numeric(pd.Series(heel_signal), errors="coerce").fillna(0), dtype=int)
    heel_strikes = np.where(np.diff(binary) == 1)[0] + 1
    segments: list[tuple[int, int]] = []
    for i in range(len(heel_strikes) - 1):
        start = int(heel_strikes[i])
        end = int(heel_strikes[i + 1])
        if end - start >= min_samples:
            segments.append((start, end))
    return segments


def _normalize_to_101(values: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.full(101, np.nan, dtype=float)
    x_old = np.linspace(0, 100, len(values))
    x_new = np.arange(101)
    return np.interp(x_new, x_old, values.astype(float))


def build_cycle_waveforms(cfg: DictConfig | str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    cfg = _load_cfg(cfg)
    stride_trim = int(cfg.features.get("stride_trim", 2))
    out_path, summary_path = _output_paths(cfg)
    slim_path = _slim_path(cfg)

    slim = pd.read_parquet(slim_path)
    injured = _injured_map()
    records: list[dict] = []
    skipped_short_groups = 0
    raw_cycle_total = 0

    group_cols = ["subject_id", "group", "speed", "file_name"]
    for (subject_id, group, speed, trial_id), trial_df in slim.groupby(group_cols, sort=False):
        trial_df = trial_df.sort_values("time_ms").reset_index(drop=True)
        injured_leg = _injured_leg_for(str(subject_id), str(group), injured)

        for actual_leg, heel_col in HEEL_COLS.items():
            if heel_col not in trial_df.columns:
                continue
            segments = _stride_segments(trial_df[heel_col].to_numpy())
            n_raw = len(segments)
            raw_cycle_total += n_raw
            if n_raw <= stride_trim * 2:
                skipped_short_groups += 1
                continue

            trimmed = segments[stride_trim:n_raw - stride_trim]
            for local_idx, (start, end) in enumerate(trimmed):
                original_idx = local_idx + stride_trim
                row = {
                    "subject_id": subject_id,
                    "group": group,
                    "speed": speed,
                    "trial_id": trial_id,
                    "actual_leg": actual_leg,
                    "side_basis": _side_basis(str(group), actual_leg, injured_leg),
                    "injured_leg": injured_leg,
                    "cycle_idx": local_idx,
                    "cycle_idx_original": original_idx,
                    "cycle_len": int(end - start),
                    "n_cycles_raw": int(n_raw),
                    "n_cycles_trimmed": int(len(trimmed)),
                }

                for joint, (right_col, left_col) in JOINT_COLS.items():
                    col = right_col if actual_leg == "Right" else left_col
                    signal = trial_df[col].to_numpy() if col in trial_df.columns else np.full(len(trial_df), np.nan)
                    wave = _normalize_to_101(signal[start:end])
                    for i, value in enumerate(wave):
                        row[f"{joint}_{i:03d}"] = float(value)
                records.append(row)

    if not records:
        raise RuntimeError("추출된 cycle waveform이 없습니다.")

    df = pd.DataFrame(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    wave_cols = [c for c in df.columns if any(c.startswith(f"{joint}_") for joint in JOINT_COLS)]
    lens = df["cycle_len"]
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(slim_path),
        "output": str(out_path),
        "stride_trim": stride_trim,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "n_subjects": int(df["subject_id"].nunique()),
        "n_trials": int(df[["subject_id", "speed", "trial_id"]].drop_duplicates().shape[0]),
        "raw_cycle_total_before_trim": int(raw_cycle_total),
        "trimmed_cycle_rows": int(len(df)),
        "skipped_trial_leg_groups_too_short": int(skipped_short_groups),
        "waveform_columns": int(len(wave_cols)),
        "cycle_len": {
            "mean": float(lens.mean()),
            "std": float(lens.std()),
            "min": int(lens.min()),
            "p05": float(lens.quantile(0.05)),
            "p50": float(lens.quantile(0.50)),
            "p95": float(lens.quantile(0.95)),
            "max": int(lens.max()),
        },
        "group_counts": {str(k): int(v) for k, v in df["group"].value_counts().to_dict().items()},
        "speed_counts": {str(k): int(v) for k, v in df["speed"].value_counts().to_dict().items()},
        "side_basis_counts": {str(k): int(v) for k, v in df["side_basis"].value_counts().to_dict().items()},
        "actual_leg_counts": {str(k): int(v) for k, v in df["actual_leg"].value_counts().to_dict().items()},
        "max_waveform_nan_rate": float(df[wave_cols].isna().mean().max()),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[extract_cycle_waveforms_101] 저장: {out_path} {df.shape}")
    print(f"[extract_cycle_waveforms_101] 요약: {summary_path}")
    return df, summary


def run(cfg_path: str | None = None) -> pd.DataFrame:
    df, _summary = build_cycle_waveforms(cfg_path)
    return df


def extract(cfg: DictConfig | str | Path | None = None) -> pd.DataFrame:
    df, _summary = build_cycle_waveforms(cfg)
    return df


if __name__ == "__main__":
    run()
