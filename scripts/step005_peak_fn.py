"""
step005: get_first_peak() 구현 및 smoke check
- stance 내 첫 번째 peak (loading response peak) 탐지
- argmax 대신 find_peaks 사용해 2nd peak 포착 방지

출력: utils/peak_utils.py에 함수 저장
"""
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
UTILS_DIR = ROOT / "utils"
UTILS_DIR.mkdir(exist_ok=True)

PEAK_UTILS = UTILS_DIR / "peak_utils.py"

CODE = '''"""첫 번째 peak 탐지 유틸리티."""
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
'''

def write_utils():
    PEAK_UTILS.write_text(CODE, encoding="utf-8")
    print(f"[005] peak_utils.py 작성: {PEAK_UTILS}")


def smoke_check():
    sys.path.insert(0, str(ROOT))
    from utils.peak_utils import get_first_peak, get_stance_scalar_features

    seg_max = np.array([5.0, 15.0, 10.0, 8.0, 12.0, 6.0])
    idx = get_first_peak(seg_max, "max", min_prominence=0.5)
    assert idx == 1, f"첫 번째 max peak가 idx=1이어야 함, 실제: {idx}"

    seg_min = np.array([5.0, 2.0, 4.0, 1.0, 3.0])
    idx = get_first_peak(seg_min, "min", min_prominence=0.5)
    assert idx == 1, f"첫 번째 min peak가 idx=1이어야 함, 실제: {idx}"

    feats = get_stance_scalar_features(seg_max, "max")
    assert feats["K1"] == 15.0
    assert feats["ROM"] == 15.0 - 5.0
    assert feats["IC_angle"] == 5.0
    assert 0 <= feats["K1_timing"] <= 100

    print("[005] ✅ get_first_peak() smoke check 통과")


if __name__ == "__main__":
    write_utils()
    smoke_check()
