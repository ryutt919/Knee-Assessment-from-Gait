"""Stance phase peak detection utilities."""
import numpy as np
from scipy.signal import find_peaks


def iqr_filter_values(vals: list,
                      lower_k: float = 1.5,
                      upper_k: float = 2.5,
                      min_n: int = 4) -> list:
    """IQR 기반 이상치 제거 후 유효 값 반환.

    min_n 미만이면 필터 없이 전체 반환 (데이터 부족).
    lower_k=1.5, upper_k=2.5는 preprocess.py detect_peaks_with_iqr 기본값과 동일.
    """
    arr = np.array([v for v in vals if not np.isnan(v)], dtype=float)
    if len(arr) < min_n:
        return list(arr)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    mask = (arr >= q1 - lower_k * iqr) & (arr <= q3 + upper_k * iqr)
    return list(arr[mask])


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
    """stance segment에서 peak, min, ROM, IC_angle, peak_timing을 계산.

      peak        — first prominence peak during stance (loading response peak)
      min         — minimum value during stance (maximum extension / opposite extreme)
      ROM         — range of motion within stance (max - min)
      IC_angle    — joint angle at initial contact (seg[0])
      peak_timing — timing of peak as % of stance duration
    """
    if len(seg) == 0:
        return {k: float("nan") for k in ["peak", "min", "ROM", "IC_angle", "peak_timing"]}

    peak_idx = get_first_peak(seg, direction, min_prominence)
    peak_val = float(seg[peak_idx])
    min_val = float(np.min(seg))
    true_max = float(np.max(seg))
    true_min = float(np.min(seg))
    rom = true_max - true_min
    ic_angle = float(seg[0])
    peak_timing = peak_idx / max(len(seg) - 1, 1) * 100.0

    return {
        "peak": peak_val,
        "min": min_val,
        "ROM": rom,
        "IC_angle": ic_angle,
        "peak_timing": peak_timing,
    }
