"""
step010: waveform 101pt 정규화 함수 구현 및 smoke check
- 각 stance segment를 interp1d로 0~100% (101pt) 정규화
- 함수는 utils/norm_utils.py에 저장

출력: utils/norm_utils.py
"""
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
UTILS_DIR = ROOT / "utils"
UTILS_DIR.mkdir(exist_ok=True)

CODE = '''"""파형 정규화 유틸리티."""
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
'''


def write_utils():
    norm_path = UTILS_DIR / "norm_utils.py"
    norm_path.write_text(CODE, encoding="utf-8")
    print(f"[010] norm_utils.py 작성: {norm_path}")


def smoke_check():
    sys.path.insert(0, str(ROOT))
    from utils.norm_utils import normalize_to_101, normalize_segments, mean_waveform
    from scripts.analysis.preprocess import get_stance_segments

    seg = np.sin(np.linspace(0, np.pi, 50)) * 10
    norm = normalize_to_101(seg)
    assert len(norm) == 101, f"정규화 결과 길이 {len(norm)} ≠ 101"
    assert not np.any(np.isnan(norm)), "정규화 결과에 NaN"
    assert abs(norm[0] - seg[0]) < 0.01

    contact = np.array([0]*10 + [1]*50 + [0]*10 + [1]*40 + [0]*10)
    segs = get_stance_segments(contact)
    signal = np.random.randn(120)
    norms = normalize_segments(signal, segs)
    assert all(len(n) == 101 for n in norms), "normalize_segments 결과 길이 오류"

    mean = mean_waveform(signal, segs)
    assert len(mean) == 101, f"mean_waveform 길이 {len(mean)} ≠ 101"

    print("[010] ✅ normalize_to_101() smoke check 통과")


if __name__ == "__main__":
    write_utils()
    smoke_check()
