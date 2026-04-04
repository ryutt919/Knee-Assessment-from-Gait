import pandas as pd
import numpy as np
import pyarrow.dataset as ds
import pyarrow.compute as pc
import os
from scipy.signal import find_peaks

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PATH_RAW = os.path.join(DATA_DIR, "processed", "raw_merged.parquet")

def test_peaks():
    # 샘플로 1명의 대상자 추출
    dataset = ds.dataset(PATH_RAW, format="parquet")
    sample_id = "ACLD1"
    
    # knee_flexion (45번 컬럼)을 대상으로 테스트
    filter_expr = (pc.field("subject_id") == sample_id) & (pc.field("speed") == "normal")
    table = dataset.to_table(columns=["subject_id", "speed", "jointAngle_45"], filter=filter_expr)
    df = table.to_pandas()
    
    if df.empty:
        print("샘플 데이터를 불러올 수 없습니다.")
        return
        
    y = df["jointAngle_45"].values
    
    # 파라미터 탐색 테스트
    # 일반적인 보행 주기 1 stride = 약 1 ~ 1.2초. 60Hz 기준 60~72프레임.
    # 안전하게 최소 30프레임 이상 떨어져 있다고 가정
    peaks, properties = find_peaks(y, distance=30, prominence=5)
    
    print(f"Total samples: {len(y)}")
    print(f"Number of peaks found: {len(peaks)}")
    print(f"Peak values: {y[peaks]}")
    print(f"Mean peak value: {np.mean(y[peaks]):.2f}")
    print(f"Global max value: {np.max(y):.2f}")
    
    # 5번째 피크 전후의 값들을 살짝 출력하여 주기성 확인
    if len(peaks) > 0:
        idx = peaks[len(peaks)//2]
        start = max(0, idx - 10)
        end = min(len(y), idx + 10)
        print("\n중간 지점 피크 주변 값:")
        print(np.round(y[start:end], 2))

if __name__ == "__main__":
    test_peaks()
