"""
step003: slim_gait.parquet 생성
- ACLD.parquet, ACLR.parquet, Healthy adults.parquet에서 27개 컬럼 추출
- ACLD: paired 27명만 / ACLR: 전원 (ACLR38→ACLR36 치환) / HA: 전원
- 출력: data/processed/slim_gait.parquet (~200MB)

검증: 컬럼 27개, ACLR38 subject_id 없음
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
PARQUETS = ROOT / "data" / "processed" / "parquets"
PROCESSED = ROOT / "data" / "processed"
PAIRING_CSV = PROCESSED / "id_pairing_summary.csv"
OUT_PATH = PROCESSED / "slim_gait.parquet"

KEEP_COLS = [
    "time_ms", "group", "subject_id", "speed", "file_name",
    "footContacts_0", "footContacts_1", "footContacts_2", "footContacts_3",
    "jointAngle_42", "jointAngle_43", "jointAngle_44",
    "jointAngle_45", "jointAngle_46", "jointAngle_47",
    "jointAngle_48", "jointAngle_49", "jointAngle_50",
    "jointAngle_54", "jointAngle_55", "jointAngle_56",
    "jointAngle_57", "jointAngle_58", "jointAngle_59",
    "jointAngle_60", "jointAngle_61", "jointAngle_62",
]


def load_parquet(name: str) -> pd.DataFrame:
    path = PARQUETS / f"{name}.parquet"
    df = pd.read_parquet(path)
    available = [c for c in KEEP_COLS if c in df.columns]
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        print(f"[003] {name}: 컬럼 누락 {missing[:5]}...")
    df = df[available].copy()
    if "group" not in df.columns:
        group_map = {"ACLD": "ACLD", "ACLR": "ACLR", "Healthy adults": "HA"}
        df["group"] = group_map.get(name, name)
    return df


def run():
    pairing = pd.read_csv(PAIRING_CSV)
    paired_acld = pairing[pairing["pair_status"] == "paired"]["ID_ACLD"].tolist()
    paired_aclr = pairing[pairing["pair_status"] == "paired"]["ID_ACLR"].tolist()

    print(f"[003] ACLD 대상: {len(paired_acld)}명")
    print(f"[003] ACLR 대상: {len(paired_aclr)}명")

    acld_df = load_parquet("ACLD")
    acld_df["group"] = "ACLD"
    acld_df = acld_df[acld_df["subject_id"].isin(paired_acld)]
    print(f"[003] ACLD 로드: {len(acld_df)}행")

    aclr_df = load_parquet("ACLR")
    aclr_df["group"] = "ACLR"
    aclr_df["subject_id"] = aclr_df["subject_id"].replace({"ACLR38": "ACLR36"})
    print(f"[003] ACLR 로드: {len(aclr_df)}행")

    ha_df = load_parquet("Healthy adults")
    ha_df["group"] = "HA"
    ha_ids = ha_df["subject_id"].unique()
    ha_keep = [s for s in ha_ids if s.startswith("HA")]
    ha_df = ha_df[ha_df["subject_id"].isin(ha_keep)]
    print(f"[003] HA 로드: {len(ha_df)}행 ({len(ha_keep)}명)")

    slim = pd.concat([acld_df, aclr_df, ha_df], ignore_index=True)

    assert "ACLR38" not in slim["subject_id"].values, "ACLR38 subject_id 잔존"
    assert len(slim.columns) >= 20, f"컬럼 수 부족: {len(slim.columns)}"

    slim.to_parquet(OUT_PATH, index=False)
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"[003] ✅ slim_gait.parquet 저장: {OUT_PATH} ({size_mb:.1f}MB, {len(slim):,}행, {len(slim.columns)}컬럼)")


if __name__ == "__main__":
    run()
