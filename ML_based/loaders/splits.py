"""
GroupKFold 분할 생성·저장·재현
- group_col(subject_id) 기준으로 동일 피험자의 모든 stride가 같은 fold에 배정
- fold 인덱스를 data/splits/ 에 pkl로 캐시
"""
import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

SPLITS_DIR = Path(__file__).parent.parent / "data" / "splits"


def make_outer_splits(
    groups: np.ndarray,
    n_splits: int,
    seed: int,
    y: np.ndarray = None,
    name: str = "outer",
) -> list[tuple[np.ndarray, np.ndarray]]:
    # 캐시 키에 샘플 수 포함 — 다른 크기 배열이 같은 캐시를 로드하지 않도록
    cache = SPLITS_DIR / f"{name}_k{n_splits}_seed{seed}_n{len(groups)}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            splits = pickle.load(f)
        print(f"[splits] 캐시 로드: {cache}")
        return splits

    gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # y가 주어지지 않으면 더미로 동작 (StratifiedGroupKFold는 y가 필수지만, 없으면 GroupKFold처럼 동작하도록 처리)
    if y is None:
        y = np.zeros(len(groups))
    splits = list(gkf.split(X=np.zeros((len(groups), 1)), y=y, groups=groups))

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(splits, f)
    print(f"[splits] 분할 생성·저장: {cache} (n_splits={n_splits}, samples={len(groups)})")
    return splits


def verify_no_leakage(splits: list, groups: np.ndarray) -> bool:
    """동일 subject가 train/test에 동시 등장하지 않는지 확인."""
    for fold, (tr, te) in enumerate(splits):
        train_subjects = set(groups[tr])
        test_subjects  = set(groups[te])
        overlap = train_subjects & test_subjects
        if overlap:
            print(f"[splits] ❌ fold {fold} 누수: {overlap}")
            return False
    print("[splits] ✅ 누수 없음 확인")
    return True
