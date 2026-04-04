"""
00b_inspect_sampling.py  
샘플링 주파수, 각도 단위(radian vs degree), 보행 주기 패턴 탐색
"""
import pandas as pd
import numpy as np
import pyarrow.dataset as ds
import pyarrow.compute as pc
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_RAW = os.path.join(BASE_DIR, "data", "processed", "raw_merged.parquet")

dataset = ds.dataset(PATH_RAW, format="parquet")

# ACLD1, normal 속도, time_ms + 관절 각도 모두 추출
filt = (pc.field("subject_id") == "ACLD1") & (pc.field("speed") == "normal")
table = dataset.to_table(
    columns=["subject_id", "speed", "time_ms",
             "jointAngle_45", "jointAngle_57"],  # R/L Knee flexion
    filter=filt
)
df = table.to_pandas()
df["time_ms"] = pd.to_numeric(df["time_ms"], errors="coerce")
df = df.sort_values("time_ms").reset_index(drop=True)

# 1) 샘플링 주파수 계산
dt = df["time_ms"].diff().dropna()
print("=== 샘플링 주기 (ms) ===")
print(f"  dt 평균: {dt.mean():.2f} ms  →  약 {1000/dt.mean():.1f} Hz")
print(f"  dt 중앙값: {dt.median():.2f} ms")
print(f"  dt 최소: {dt.min():.2f} ms, 최대: {dt.max():.2f} ms")

# 2) 전체 녹화 시간
total_sec = (df["time_ms"].max() - df["time_ms"].min()) / 1000
print(f"\n=== 총 녹화 시간 ===")
print(f"  {total_sec:.2f} 초  ({len(df)} 샘플)")

# 3) 각도 단위 추정 (radian이면 최대치가 π=3.14 수준)
max_val = df["jointAngle_45"].abs().max()
print(f"\n=== jointAngle_45 절대값 최대 ===")
print(f"  {max_val:.4f}")
if max_val < 6.3:
    print("  → 라디안(radian) 단위로 추정됨 (최대 ≈ 2π)")
    print(f"  → degree 환산 시 최대값: {np.degrees(max_val):.1f}°")
else:
    print("  → 도(degree) 단위로 추정됨")

# 4) 전체 시계열의 첫 3초 출력하여 보행 주기 패턴 확인
hz = 1000 / dt.median()
n_3sec = int(hz * 3)
print(f"\n=== 첫 3초 데이터 ({n_3sec}개 샘플) jointAngle_45 (R_Knee_flex) ===")
print(np.round(df["jointAngle_45"].values[:n_3sec], 4))

# 5) 다른 피험자/속도 조건 확인 (빠르게)
print("\n=== subject_id, speed 별 샘플 수 ===")
filt2 = pc.field("subject_id").isin(["ACLD1", "ACLR1", "Healthy1"])
t2 = dataset.to_table(columns=["subject_id", "speed"], filter=filt2)
df2 = t2.to_pandas()
print(df2.groupby(["subject_id", "speed"]).size())
