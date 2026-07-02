"""
01_data_preprocessing.py — Stride 분할, 101pt 정규화, Waveform 특징 벡터 생성

입력: data/processed/raw_subset_mahalanobis.parquet  (or test variant)
출력: data/processed/mahalanobis_features.parquet

각 행(row) = 1 Stride × (좌 또는 우) 발:
  - 메타: subject_id, group, speed, trial_id, actual_leg, side_basis, stride_idx
  - 특징: <channel>_<001..101> 형태의 플랫 waveform (채널 수 × 101)

처리 흐름:
  1. raw_subset 로드
  2. (subject_id, group, speed, file_name) 단위로 그룹핑 → Trial
  3. footContacts_0/2 heel-strike 엣지 감지 → Stride 분할
  4. 앞뒤 STRIDE_TRIM개 제거
  5. 채널별 선형 보간 → 101 포인트
  6. 피처 행 생성 및 저장
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent  # Walking/
SANDBOX = Path(__file__).resolve().parent.parent       # 0702_Mahalanobis/
DATA_PROCESSED = ROOT / "data" / "processed"
ID_CSV = ROOT / "data" / "ID.csv"

STRIDE_TRIM = 2   # 앞뒤 제거 stride 수

# ─── 관절 컬럼 매핑 (오른발 / 왼발) ─────────────────────────────────────────
JOINT_DEF = {
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

# IMU 채널: 센서 인덱스는 양쪽 다리를 포함하므로 좌/우 구분 없이 모두 사용
ACCEL_NAMES  = [f"sensorFreeAcceleration_{i}" for i in range(21)]
ORIENT_NAMES = [f"sensorOrientation_{i}" for i in range(28)]
MAG_NAMES    = [f"sensorMagneticField_{i}" for i in range(21)]

HEEL_COLS = {"Left": "footContacts_0", "Right": "footContacts_2"}


# ─── 유틸 함수 ───────────────────────────────────────────────────────────────

def load_injured_map() -> dict[str, str]:
    """data/ID.csv에서 피험자별 손상 다리 정보 로드."""
    if not ID_CSV.exists():
        return {}
    id_df = pd.read_csv(ID_CSV)
    mapping: dict[str, str] = {}
    for _, row in id_df.iterrows():
        leg = row.get("Injured leg", "")
        leg = str(leg).strip() if pd.notna(leg) else ""
        if leg in ("Left", "Right"):
            mapping[str(row["ID"])] = leg
    return mapping


def injured_leg_for(subject_id: str, group: str, injured_map: dict[str, str]) -> str:
    """그룹별 손상 다리 반환. HA는 Right를 pseudo-injured로 사용."""
    if group in ("ACLD", "ACLR"):
        return injured_map.get(subject_id, "Right")
    return "Right"


def side_basis(group: str, actual_leg: str, injured_leg: str) -> str:
    if group in ("ACLD", "ACLR"):
        return "injured" if actual_leg == injured_leg else "contralateral"
    return "injured" if actual_leg == "Right" else "contralateral"


def detect_heel_strikes(signal: np.ndarray, min_samples: int = 40) -> list[tuple[int, int]]:
    """0→1 상승 엣지를 Heel Strike로 감지하여 (start, end) stride 구간 반환."""
    binary = pd.to_numeric(pd.Series(signal), errors="coerce").fillna(0).to_numpy(dtype=int)
    edges = np.where(np.diff(binary) == 1)[0] + 1
    segments: list[tuple[int, int]] = []
    for i in range(len(edges) - 1):
        s, e = int(edges[i]), int(edges[i + 1])
        if e - s >= min_samples:
            segments.append((s, e))
    return segments


def normalize_101(values: np.ndarray) -> np.ndarray:
    """시계열을 101 포인트로 선형 보간하여 반환."""
    if len(values) < 2:
        return np.full(101, np.nan, dtype=float)
    x_old = np.linspace(0, 100, len(values))
    x_new = np.arange(101)
    return np.interp(x_new, x_old, values.astype(float))


def build_channel_names(
    use_joint: bool,
    use_accel: bool,
    use_orient: bool,
    use_mag: bool,
) -> list[str]:
    """활성화된 채널의 이름 목록 반환."""
    channels: list[str] = []
    if use_joint:
        channels.extend(JOINT_DEF.keys())
    if use_accel:
        channels.extend(ACCEL_NAMES)
    if use_orient:
        channels.extend(ORIENT_NAMES)
    if use_mag:
        channels.extend(MAG_NAMES)
    return channels


# ─── 핵심 추출 함수 ──────────────────────────────────────────────────────────

def extract_features(
    df: pd.DataFrame,
    use_joint: bool = True,
    use_accel: bool = True,
    use_orient: bool = True,
    use_mag: bool = True,
    stride_trim: int = STRIDE_TRIM,
) -> pd.DataFrame:
    """
    raw subset DataFrame으로부터 stride-level waveform feature DataFrame 생성.

    반환 DataFrame 컬럼:
        subject_id, group, speed, trial_id, actual_leg, side_basis,
        stride_idx, cycle_len,
        {channel}_{001..101}  (선형 보간 101pt 플랫 벡터)
    """
    injured_map = load_injured_map()
    records: list[dict[str, Any]] = []

    group_cols = ["subject_id", "group", "speed", "file_name"]
    total_trials = df.groupby(group_cols, sort=False).ngroups
    processed = 0

    for (subject_id, group, speed, file_name), trial_df in df.groupby(group_cols, sort=False):
        trial_df = trial_df.sort_values("time_ms").reset_index(drop=True)
        inj_leg = injured_leg_for(str(subject_id), str(group), injured_map)
        processed += 1

        for actual_leg, heel_col in HEEL_COLS.items():
            if heel_col not in trial_df.columns:
                continue
            segments = detect_heel_strikes(trial_df[heel_col].to_numpy())
            n_raw = len(segments)

            if n_raw <= stride_trim * 2:
                continue

            trimmed = segments[stride_trim: n_raw - stride_trim]

            for local_idx, (start, end) in enumerate(trimmed):
                row: dict[str, Any] = {
                    "subject_id": subject_id,
                    "group": group,
                    "speed": speed,
                    "trial_id": file_name,
                    "actual_leg": actual_leg,
                    "side_basis": side_basis(str(group), actual_leg, inj_leg),
                    "stride_idx": local_idx,
                    "cycle_len": int(end - start),
                }
                seg = trial_df.iloc[start:end]

                # ── 관절 각도 채널 ────────────────────────────────
                if use_joint:
                    for jname, (r_col, l_col) in JOINT_DEF.items():
                        col = r_col if actual_leg == "Right" else l_col
                        vals = seg[col].to_numpy() if col in seg.columns else np.full(end - start, np.nan)
                        wave = normalize_101(vals)
                        for t, v in enumerate(wave):
                            row[f"{jname}_{t+1:03d}"] = float(v)

                # ── IMU 원본 채널 ────────────────────────────────
                def _add_imu(col_names: list[str]) -> None:
                    for col in col_names:
                        if col in seg.columns:
                            vals = seg[col].to_numpy()
                        else:
                            vals = np.full(end - start, np.nan)
                        wave = normalize_101(vals)
                        for t, v in enumerate(wave):
                            row[f"{col}_{t+1:03d}"] = float(v)

                if use_accel:
                    _add_imu([c for c in ACCEL_NAMES if c in trial_df.columns])
                if use_orient:
                    _add_imu([c for c in ORIENT_NAMES if c in trial_df.columns])
                if use_mag:
                    _add_imu([c for c in MAG_NAMES if c in trial_df.columns])

                records.append(row)

        if processed % 50 == 0 or processed == total_trials:
            print(f"  [01] 진행: {processed}/{total_trials} trials, 누적 strides: {len(records)}")

    if not records:
        raise RuntimeError("추출된 stride가 없습니다. 데이터를 확인하세요.")

    return pd.DataFrame(records)


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main(test_mode: bool = False) -> pd.DataFrame:
    # 입력 파일 선택
    if test_mode:
        src = DATA_PROCESSED / "raw_subset_mahalanobis_test.parquet"
    else:
        src = DATA_PROCESSED / "raw_subset_mahalanobis.parquet"

    if not src.exists():
        print(f"[01] ❌ 입력 파일 없음: {src}")
        print("[01] 먼저 00_extract_subset.py를 실행하세요.")
        sys.exit(1)

    print(f"[01] 로드: {src}")
    df = pd.read_parquet(src)
    print(f"[01] 로드 완료: shape={df.shape}")
    print(f"[01] 그룹 분포: {df['group'].value_counts().to_dict()}")

    print("[01] Stride 분할 및 101pt 보간 시작...")
    feat_df = extract_features(df)

    print(f"[01] 추출 완료: {feat_df.shape}")
    print(f"[01] 그룹별 stride 수:\n{feat_df.groupby('group').size().to_string()}")

    # 저장
    out = DATA_PROCESSED / ("mahalanobis_features_test.parquet" if test_mode else "mahalanobis_features.parquet")
    feat_df.to_parquet(out, index=False)
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"[01] 저장: {out}  ({size_mb:.1f} MB)")

    return feat_df


if __name__ == "__main__":
    test_mode = "--test" in sys.argv
    main(test_mode=test_mode)
