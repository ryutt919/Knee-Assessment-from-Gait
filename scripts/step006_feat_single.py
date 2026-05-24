"""
step006: 1명(ACLD3, normal) feature 추출 검증
- slim_gait.parquet에서 ACLD3 normal 데이터 로드
- get_stance_scalar_features 적용
- 검증: peak 개수 ≥ 5, IQR pass rate ≥ 0.8

출력: data/processed/feat_single_test.csv (검증용)
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.analysis.preprocess import get_stance_segments, build_stance_contact_signal, detect_peaks_with_iqr
from utils.peak_utils import get_stance_scalar_features

SLIM = ROOT / "data" / "processed" / "slim_gait.parquet"
OUT = ROOT / "data" / "processed" / "feat_single_test.csv"

JOINT_COLS = {
    "hip_adduction":    ("jointAngle_42", "jointAngle_54"),
    "hip_int_rotation": ("jointAngle_43", "jointAngle_55"),
    "hip_flexion":      ("jointAngle_44", "jointAngle_56"),
    "knee_adduction":   ("jointAngle_45", "jointAngle_57"),
    "knee_int_rotation":("jointAngle_46", "jointAngle_58"),
    "knee_flexion":     ("jointAngle_47", "jointAngle_59"),
    "ankle_adduction":  ("jointAngle_48", "jointAngle_60"),
    "ankle_int_rotation":("jointAngle_49", "jointAngle_61"),
    "ankle_dorsiflexion":("jointAngle_50", "jointAngle_62"),
}

PEAK_DIRECTION = {
    "hip_flexion": "max", "hip_adduction": "min", "hip_int_rotation": "max",
    "knee_flexion": "max", "knee_adduction": "min", "knee_int_rotation": "max",
    "ankle_dorsiflexion": "max", "ankle_adduction": "min", "ankle_int_rotation": "max",
}


def run():
    df = pd.read_parquet(SLIM)
    sub = df[(df["subject_id"] == "ACLD3") & (df["speed"] == "normal")].reset_index(drop=True)
    assert len(sub) > 0, "ACLD3 normal 데이터 없음"

    contact, _ = build_stance_contact_signal(sub, "Right")
    segs = get_stance_segments(contact)
    print(f"[006] ACLD3 normal: {len(sub)}행, stance 구간 {len(segs)}개")
    assert len(segs) >= 5, f"stance 구간 수 부족: {len(segs)}"

    limits = {
        "distance": 30, "prominence": 1.0,
        "iqr_lower_bound": 1.5, "iqr_upper_bound": 2.5,
        "peak_method": "argextrema", "butter_order": 2, "butter_cutoff": 0.1,
    }

    results = []
    total, iqr_pass = 0, 0
    for feat, (r_col, l_col) in JOINT_COLS.items():
        signal = sub[r_col].values
        direction = PEAK_DIRECTION[feat]
        det = detect_peaks_with_iqr(
            signal=signal, direction=direction,
            distance=limits["distance"], prominence=limits["prominence"],
            iqr_lower_bound=limits["iqr_lower_bound"], iqr_upper_bound=limits["iqr_upper_bound"],
            contact_signal=contact, peak_method=limits["peak_method"],
            butter_order=limits["butter_order"], butter_cutoff=limits["butter_cutoff"],
        )
        n_all = len(det["all_peaks"])
        n_valid = len(det["valid_peaks"])
        total += n_all
        iqr_pass += n_valid

        for seg_start, seg_end in segs[:3]:
            seg = signal[seg_start:seg_end]
            if len(seg) < 5:
                continue
            feats = get_stance_scalar_features(seg, direction)
            feats["feature"] = feat
            feats["seg_start"] = seg_start
            results.append(feats)

    iqr_rate = iqr_pass / total if total > 0 else 0
    print(f"[006] 총 peak={total}, IQR 통과={iqr_pass}, pass_rate={iqr_rate:.2f}")
    assert total >= 5, f"peak 수 부족: {total}"
    assert iqr_rate >= 0.5, f"IQR pass rate 낮음: {iqr_rate:.2f}"

    pd.DataFrame(results).to_csv(OUT, index=False)
    print(f"[006] ✅ feat_single_test.csv 저장: {OUT}")


if __name__ == "__main__":
    run()
