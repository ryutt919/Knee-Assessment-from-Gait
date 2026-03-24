"""
utils/embedding_utils.py
공유 데이터 전처리 및 Trial 단위 분리 유틸리티 모듈
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict


# 반드시 제외할 비분석용 메타 컬럼
META_COLS = ["time_ms", "group", "subject_id", "speed", "file_name"]

# sensorFreeAcceleration / sensorOrientation 계열 결측 컬럼 제거
SENSOR_PREFIXES = ("sensorFreeAcceleration", "sensorOrientation")


def load_parquet(input_path: Path) -> pd.DataFrame:
    """
    Parquet 파일을 로드하고 결측치가 많은 센서 계열 컬럼을 자동 제거.
    """
    print(f"⏳ 데이터 로딩 중: {input_path.name}")
    df = pd.read_parquet(input_path)
    
    # 결측치 70개 이상인 센서 계열 컬럼 자동 제외
    sensor_cols = [c for c in df.columns if c.startswith(SENSOR_PREFIXES)]
    if sensor_cols:
        df = df.drop(columns=sensor_cols)
        print(f"  ✂️  센서 Raw 컬럼 {len(sensor_cols)}개 제거 완료 (결측 처리)")
    
    # 남아있는 결측치 전방/후방 보간 처리
    feature_cols = [c for c in df.columns if c not in META_COLS]
    df[feature_cols] = df[feature_cols].interpolate(method="linear", limit_direction="both")
    
    print(f"  ✅ 로딩 완료: {df.shape[0]:,}행 × {df.shape[1]}열")
    return df


def split_by_trial(df: pd.DataFrame) -> List[Dict]:
    """
    file_name 기준으로 데이터를 Trial 단위로 분리.
    각 Trial을 dict 형태로 반환 (data, meta).
    """
    trials = []
    feature_cols = [c for c in df.columns if c not in META_COLS]
    
    for file_name, group_df in df.groupby("file_name"):
        meta = {
            "file_name": file_name,
            "subject_id": group_df["subject_id"].iloc[0],
            "group": group_df["group"].iloc[0],
        }
        if "speed" in group_df.columns:
            meta["speed"] = group_df["speed"].iloc[0]
        
        data = group_df[feature_cols].values.astype(np.float32)
        trials.append({"meta": meta, "data": data})
    
    return trials


def downsample(data: np.ndarray, source_hz: int = 100, target_hz: int = 100) -> np.ndarray:
    """
    시간 축(axis=0)을 대상으로 stride 다운샘플링.
    예: source_hz=100, target_hz=50 → stride=2 (2프레임당 1개 선택)
    """
    if target_hz >= source_hz:
        return data
    stride = int(source_hz / target_hz)
    return data[::stride]


def build_windows(data: np.ndarray, window_size: int = 100, step: int = 50) -> np.ndarray:
    """
    Sliding Window로 시계열 데이터를 고정 길이 조각으로 분할.
    Returns: [N개 조각, window_size, 피처 수]
    """
    n_frames, n_features = data.shape
    windows = []
    for start in range(0, n_frames - window_size + 1, step):
        windows.append(data[start:start + window_size])
    if not windows:
        # 데이터가 window_size 보다 짧을 경우 패딩
        pad = np.zeros((window_size - n_frames, n_features), dtype=np.float32)
        windows.append(np.concatenate([data, pad], axis=0))
    return np.stack(windows, axis=0)  # (N, window_size, features)
