"""
GroupKFold 분할 생성·저장·재현
- 클래스별 라운드로빈 피험자 배정으로 fold 간 클래스 균형 보장 (±1명 이내)
- group_col(subject_id) 기준으로 동일 피험자의 모든 stride가 같은 fold에 배정
- fold 인덱스를 data/splits/ 에 pkl로 캐시 (_bal_ 접두사로 기존 캐시와 분리)
"""
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

SPLITS_DIR = Path(__file__).parent.parent / "data" / "splits"


def _subject_labels(groups: np.ndarray, y: np.ndarray) -> dict:
    """각 피험자의 majority-vote 클래스 레이블 반환."""
    return {s: int(np.bincount(y[groups == s]).argmax()) for s in np.unique(groups)}


def make_outer_splits(
    groups: np.ndarray,
    n_splits: int,
    seed: int,
    y: np.ndarray = None,
    name: str = "outer",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    클래스별 라운드로빈 피험자 배정 분할.

    각 클래스 내 피험자를 seeded shuffle 후 i % n_splits 로 fold 배정
    → fold 간 클래스당 피험자 수 차이 최대 ±1명 보장.
    """
    cache = SPLITS_DIR / f"{name}_bal_k{n_splits}_seed{seed}_n{len(groups)}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            splits = pickle.load(f)
        print(f"[splits] 캐시 로드: {cache}")
        return splits

    if y is None:
        y = np.zeros(len(groups), dtype=int)

    subj_label = _subject_labels(groups, y)

    rng = np.random.default_rng(seed)
    by_class: dict[int, list] = defaultdict(list)
    for s, lbl in subj_label.items():
        by_class[lbl].append(s)

    # fold별 피험자 목록 구성 (클래스 내 shuffle → 라운드로빈)
    fold_assignment: dict = {}
    for lbl in sorted(by_class.keys()):
        subjects = list(by_class[lbl])
        rng.shuffle(subjects)
        for i, s in enumerate(subjects):
            fold_assignment[s] = i % n_splits

    row_fold = np.array([fold_assignment[g] for g in groups])
    all_idx = np.arange(len(groups))

    splits = []
    for fold in range(n_splits):
        te_idx = all_idx[row_fold == fold]
        tr_idx = all_idx[row_fold != fold]
        splits.append((tr_idx, te_idx))

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(splits, f)
    print(f"[splits] 분할 생성·저장: {cache} (n_splits={n_splits}, samples={len(groups)})")

    for fold, (tr, te) in enumerate(splits):
        subj_te = np.unique(groups[te])
        cls_cnt = defaultdict(int)
        for s in subj_te:
            cls_cnt[subj_label[s]] += 1
        print(f"  fold {fold}: test {len(subj_te)}명 {dict(sorted(cls_cnt.items()))}")

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
