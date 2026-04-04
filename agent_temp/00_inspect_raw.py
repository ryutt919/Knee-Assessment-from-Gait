"""
00_inspect_raw.py  
raw_merged.parquet의 구조 및 샘플링 주파수 파악용 탐색 스크립트
"""
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_RAW = os.path.join(BASE_DIR, "data", "processed", "raw_merged.parquet")

dataset = ds.dataset(PATH_RAW, format="parquet")

# 컬럼 목록 확인
schema = dataset.schema
col_names = schema.names
print("=== 전체 컬럼 목록 ===")
print(col_names[:30], "... (총", len(col_names), "개)")

# ACLD1, normal 속도 데이터 일부 샘플 추출
filt = (pc.field("subject_id") == "ACLD1") & (pc.field("speed") == "normal")
table = dataset.to_table(
    columns=["subject_id", "speed", "frame", "time", "jointAngle_45"] if "frame" in col_names else
            ["subject_id", "speed", "jointAngle_45"],
    filter=filt
)
df = table.to_pandas()

print("\n=== ACLD1 / normal speed 샘플 ===")
print(df.head(20).to_string())
print(f"\n총 행 수: {len(df)}")
print(f"jointAngle_45 기술통계:\n{df['jointAngle_45'].describe()}")

# 시간 컬럼 확인
print(f"\n=== 사용 가능한 시간/프레임 관련 컬럼 ===")
time_cols = [c for c in col_names if any(k in c.lower() for k in ["time", "frame", "index", "sample"])]
print(time_cols)
