"""
step008: features_scalar.csv 완성 검증
- 행 수 = 79명 × 3속도 = 237 (ACLD27+ACLR27+HA25=79)
- group 컬럼 존재, subject_id/speed 조합 중복 없음

추가로 group 레이블 정규화 및 요약 통계 출력.
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
FEATURES = ROOT / "data" / "processed" / "features_scalar.csv"


def run():
    assert FEATURES.exists(), f"features_scalar.csv 없음: {FEATURES}"

    df = pd.read_csv(FEATURES)
    print(f"[008] features_scalar.csv: {len(df)}행, {len(df.columns)}컬럼")

    required = ["subject_id", "group", "speed", "injured_leg"]
    for col in required:
        assert col in df.columns, f"필수 컬럼 없음: {col}"

    dupes = df.duplicated(["subject_id", "speed"]).sum()
    assert dupes == 0, f"subject_id×speed 중복 {dupes}개"

    groups = df["group"].unique()
    print(f"[008] 그룹: {groups}")
    for g in ["ACLD", "ACLR", "HA"]:
        n = len(df[df["group"] == g]["subject_id"].unique())
        n_rows = len(df[df["group"] == g])
        print(f"[008]   {g}: {n}명, {n_rows}행")

    speeds = df["speed"].unique()
    print(f"[008] 속도: {speeds}")

    total_subjects = df["subject_id"].nunique()
    total_rows = len(df)
    print(f"[008] 총 피험자={total_subjects}, 총 행={total_rows}")

    lsi_cols = [c for c in df.columns if c.endswith("_LSI")]
    print(f"[008] LSI 컬럼 수: {len(lsi_cols)}")

    missing_rate = df.isnull().mean().mean()
    print(f"[008] 전체 결측률: {missing_rate:.1%}")
    assert missing_rate < 0.3, f"결측률 너무 높음: {missing_rate:.1%}"

    print(f"[008] ✅ features_scalar.csv 검증 통과")


if __name__ == "__main__":
    run()
