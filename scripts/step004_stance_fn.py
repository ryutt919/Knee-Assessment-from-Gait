"""
step004: get_stance_segments() 구현 검증
- preprocess.py의 get_stance_segments를 공유 utils로 확인
- 단위 테스트는 tests/smoke/test_stance.py에서 실행

이 스크립트는 함수가 정상 임포트/실행되는지 smoke check 수행.
"""
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.analysis.preprocess import get_stance_segments


def smoke_check():
    signal = np.array([0,0,1,1,1,0,0,1,1,0])
    segs = get_stance_segments(signal)
    assert len(segs) == 2, f"기대 2개 구간, 실제: {segs}"
    assert segs[0] == (2, 5), f"첫 번째 구간 오류: {segs[0]}"
    assert segs[1] == (7, 9), f"두 번째 구간 오류: {segs[1]}"

    empty_segs = get_stance_segments(np.zeros(10))
    assert len(empty_segs) == 0

    single = np.array([1,1,1,1,1])
    single_segs = get_stance_segments(single)
    assert len(single_segs) == 1
    assert single_segs[0] == (0, 5)

    print("[004] ✅ get_stance_segments() smoke check 통과")


if __name__ == "__main__":
    smoke_check()
