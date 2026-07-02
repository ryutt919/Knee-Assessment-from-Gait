"""
00_extract_subset.py — raw_merged.parquet에서 분석용 서브셋 추출

raw_merged.parquet (약 6.29 GB)에서 마할라노비스 Impairment Score 분석에
필요한 컬럼만 선별 추출하여 경량화된 parquet을 1회 생성합니다.

출력: data/processed/raw_subset_mahalanobis.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent  # Walking/
DATA_PROCESSED = ROOT / "data" / "processed"

INPUT_PATH  = DATA_PROCESSED / "raw_merged.parquet"
OUTPUT_PATH = DATA_PROCESSED / "raw_subset_mahalanobis.parquet"

# ─── 추출할 컬럼 정의 ────────────────────────────────────────────────────────
META_COLS = ["subject_id", "group", "speed", "file_name", "time_ms"]

FOOT_COLS = [
    "footContacts_0",  # Left Heel
    "footContacts_1",  # Left Toe
    "footContacts_2",  # Right Heel
    "footContacts_3",  # Right Toe
]

# jointAngle_42~62 : 오른쪽(42~50) + 왼쪽(54~62) 총 21채널
#   hip_adduction    : 42(R), 54(L)
#   hip_int_rotation : 43(R), 55(L)
#   hip_flexion      : 44(R), 56(L)
#   knee_adduction   : 45(R), 57(L)
#   knee_int_rotation: 46(R), 58(L)
#   knee_flexion     : 47(R), 59(L)
#   ankle_adduction  : 48(R), 60(L)
#   ankle_int_rotation:49(R), 61(L)
#   ankle_dorsiflexion:50(R), 62(L)
JOINT_COLS = [f"jointAngle_{i}" for i in list(range(42, 63))]   # 42~62 (21개)

# IMU 원본 센서 채널
ACCEL_COLS  = [f"sensorFreeAcceleration_{i}" for i in range(21)]   # 21개
ORIENT_COLS = [f"sensorOrientation_{i}" for i in range(28)]        # 28개
MAG_COLS    = [f"sensorMagneticField_{i}" for i in range(21)]      # 21개

ALL_COLS = META_COLS + FOOT_COLS + JOINT_COLS + ACCEL_COLS + ORIENT_COLS + MAG_COLS

# ─── group 정규화 ────────────────────────────────────────────────────────────
GROUP_MAP = {
    "Healthy adults":      "HA",
    "Healthy adolescents": None,  # 제외
    "ACLD":                "ACLD",
    "ACLR":                "ACLR",
}


def _normalize_groups(df: pd.DataFrame) -> pd.DataFrame:
    """raw_merged의 group 컬럼을 HA / ACLD / ACLR 으로 통일.
    Healthy adolescents 행은 제외한다."""
    df = df.copy()
    df["group"] = df["group"].map(GROUP_MAP)
    df = df[df["group"].notna()].copy()
    return df


def main(test_mode: bool = False) -> None:
    """
    Args:
        test_mode: True면 처음 5만 행만 추출 (기능 검증용)
    """
    if OUTPUT_PATH.exists() and not test_mode:
        print(f"[00] 이미 존재: {OUTPUT_PATH}  → 재실행 불필요")
        return

    print(f"[00] raw_merged.parquet 로드 중... ({INPUT_PATH})")
    print(f"[00] 추출 컬럼 수: {len(ALL_COLS)}")

    # PyArrow로 열 선택 로드 (메모리 효율적)
    import pyarrow.parquet as pq

    # 실제로 존재하는 컬럼만 추출 (스키마 확인 후 필터)
    schema = pq.read_schema(INPUT_PATH)
    available = set(schema.names)
    selected = [c for c in ALL_COLS if c in available]
    missing  = [c for c in ALL_COLS if c not in available]

    if missing:
        print(f"[00] ⚠️  누락 컬럼 ({len(missing)}개): {missing}")
    print(f"[00] 선택된 컬럼 수: {len(selected)}")

    if test_mode:
        print("[00] 테스트 모드: 그룹별 3명씩 샘플링 (HA / ACLD / ACLR)")
        df = pd.read_parquet(INPUT_PATH, columns=selected)
        # group 마핑 정규화: Healthy adults → HA, Healthy adolescents 제외
        df = _normalize_groups(df)
        # 그룹별 최대 3명의 subject_id만 선택
        sample_ids: list[str] = []
        for grp in ["HA", "ACLD", "ACLR"]:
            grp_ids = df.loc[df["group"] == grp, "subject_id"].unique()[:3].tolist()
            sample_ids.extend([str(sid) for sid in grp_ids])
        df = df[df["subject_id"].astype(str).isin(sample_ids)].copy()
        print(f"[00] 테스트 선택: {df['group'].value_counts().to_dict()}")
    else:
        df = pd.read_parquet(INPUT_PATH, columns=selected)
        df = _normalize_groups(df)

    print(f"[00] 로드 완료: shape={df.shape}")

    # 그룹별 행 수 확인
    if "group" in df.columns:
        print("[00] 그룹별 행 수:")
        print(df["group"].value_counts().to_string())

    # 저장
    out = OUTPUT_PATH.with_name("raw_subset_mahalanobis_test.parquet") if test_mode else OUTPUT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"[00] 저장 완료: {out}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    test_mode = "--test" in sys.argv
    main(test_mode=test_mode)
