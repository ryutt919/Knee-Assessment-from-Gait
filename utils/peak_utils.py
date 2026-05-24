"""첫 번째 peak 탐지 유틸리티."""
import numpy as np
from scipy.signal import find_peaks


def get_first_peak(seg: np.ndarray, direction: str,
                   min_prominence: float = 1.0) -> int:
    """stance segment에서 첫 번째 peak index를 반환.

    direction: "max" (flexion peak) or "min" (adduction trough)
    fallback: find_peaks가 없으면 argmax/argmin으로 대체
    """
    if len(seg) == 0:
        return 0

    arr = seg if direction == "max" else -seg
    indices, _ = find_peaks(arr, prominence=min_prominence)

    if len(indices) > 0:
        return int(indices[0])
    return int(np.argmax(seg)) if direction == "max" else int(np.argmin(seg))


def get_stance_scalar_features(seg: np.ndarray, direction: str,
                                min_prominence: float = 1.0) -> dict:
    """stance segment에서 K1, K2, ROM, IC_angle, K1_timing을 계산."""
    if len(seg) == 0:
        return {k: float("nan") for k in ["K1", "K2", "ROM", "IC_angle", "K1_timing"]}

    k1_idx = get_first_peak(seg, direction, min_prominence)
    k1_val = float(seg[k1_idx])
    k2_val = float(np.min(seg))
    true_max = float(np.max(seg))
    true_min = float(np.min(seg))
    rom = true_max - true_min
    ic_angle = float(seg[0])
    k1_timing = k1_idx / max(len(seg) - 1, 1) * 100.0

    return {
        "K1": k1_val,
        "K2": k2_val,
        "ROM": rom,
        "IC_angle": ic_angle,
        "K1_timing": k1_timing,
    }
