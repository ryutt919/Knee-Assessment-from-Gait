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


import pyarrow.parquet as pq
from collections import defaultdict
import gc

def load_trials_chunked(input_path: Path) -> List[Dict]:
    """
    Parquet 파일을 Row Group(Chunk) 단위로 읽어 메모리를 최소화하며
    Trial(subject_id + file_name) 단위의 시계열 데이터(Numpy) 리스트로 실시간 변환.
    """
    print(f"⏳ 데이터 로딩 중 (Chunk Streaming): {input_path.name}")
    pf = pq.ParquetFile(input_path)
    
    trials_buffer = defaultdict(list)
    meta_buffer = {}
    
    for i in range(pf.num_row_groups):
        print(f"  [Chunking] Row Group {i+1}/{pf.num_row_groups} 읽기 및 처리...")
        df = pf.read_row_group(i).to_pandas()
        
        # 메모리 최적화: 수치형 float32 캐스팅
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].astype('float32')
        
        # 센서 결측치 제거
        sensor_cols = [c for c in df.columns if c.startswith(SENSOR_PREFIXES)]
        if sensor_cols:
            df = df.drop(columns=sensor_cols)
            
        feature_cols = [c for c in df.columns if c not in META_COLS]
        
        # 보간 처리 (float32 유지)
        df[feature_cols] = df[feature_cols].interpolate(method="linear", limit_direction="both").astype('float32')
        
        # 추출 및 버퍼링
        for (sub_id, f_name), group_df in df.groupby(["subject_id", "file_name"]):
            key = (sub_id, f_name)
            if key not in meta_buffer:
                meta = {
                    "file_name": f_name,
                    "subject_id": sub_id,
                    "group": group_df["group"].iloc[0],
                }
                if "speed" in group_df.columns:
                    meta["speed"] = group_df["speed"].iloc[0]
                meta_buffer[key] = meta
                
            # 순수 Numpy 데이터(float32)만 버퍼에 추가하여 Pandas 객체 의존성 탈피
            data = group_df[feature_cols].values.astype(np.float32)
            trials_buffer[key].append(data)
            
        # 명시적 메모리 해제 - 6.3GB 데이터가 한 방에 터지는 것을 방지
        del df
        gc.collect()
        
    print(f"  ✅ 청크 스캔 완료! 분할된 Trial 병합 중...")
    
    # 조각난 Numpy 배열들을 하나로 이어 붙이기
    final_trials = []
    for key, data_chunks in trials_buffer.items():
        merged_data = np.concatenate(data_chunks, axis=0)
        final_trials.append({
            "meta": meta_buffer[key],
            "data": merged_data
        })
        
    return final_trials


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
