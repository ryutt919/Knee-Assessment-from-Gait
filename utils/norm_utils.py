"""파형 정규화 유틸리티."""
import numpy as np
from scipy.interpolate import interp1d


def normalize_to_101(seg: np.ndarray) -> np.ndarray:
    """stance segment를 101점(0-100%) 으로 선형 보간 정규화."""
    if len(seg) < 2:
        return np.full(101, np.nan)
    f = interp1d(np.linspace(0, 100, len(seg)), seg.astype(float), kind="linear")
    return f(np.arange(101))


def normalize_segments(signal: np.ndarray, segments: list) -> list[np.ndarray]:
    """stance segment 목록을 각각 101pt로 정규화하여 리스트로 반환."""
    normalized = []
    for seg_start, seg_end in segments:
        seg = signal[seg_start:seg_end]
        if len(seg) < 5:
            continue
        normalized.append(normalize_to_101(seg))
    return normalized


def mean_waveform(signal: np.ndarray, segments: list) -> np.ndarray:
    """모든 stance segment의 평균 파형(101pt)을 반환."""
    norms = normalize_segments(signal, segments)
    if not norms:
        return np.full(101, np.nan)
    return np.nanmean(norms, axis=0)
