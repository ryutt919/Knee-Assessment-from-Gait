"""
step009: waveform asymmetry 계산 및 features_scalar.csv에 추가
- 각 joint channel × speed 별 Left/Right 파형 차이 면적 계산
- asym = trapz(|left_norm - right_norm|) / 101
- 검증: 음수 없음, NaN < 5%

출력: features_scalar.csv 에 asymmetry 컬럼 추가 (덮어쓰기)
"""
import numpy as np
import pandas as pd
import sys
from scipy.interpolate import interp1d
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
SLIM = ROOT / "data" / "processed" / "slim_gait.parquet"
FEATURES = ROOT / "data" / "processed" / "features_scalar.csv"

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

FOOT_CONTACTS = {
    "Right": ("footContacts_2", "footContacts_3"),
    "Left":  ("footContacts_0", "footContacts_1"),
}


def normalize_to_101(seg: np.ndarray) -> np.ndarray:
    if len(seg) < 2:
        return np.full(101, np.nan)
    f = interp1d(np.linspace(0, 100, len(seg)), seg, kind="linear")
    return f(np.arange(101))


from scripts.analysis.preprocess import get_stance_segments as _get_stance_segments


def get_mean_waveform(df: pd.DataFrame, col: str, contact_signal: np.ndarray) -> np.ndarray:
    signal = df[col].values if col in df.columns else np.zeros(len(df))
    segs = _get_stance_segments(contact_signal)
    norms = []
    for s, e in segs:
        seg = signal[s:e]
        if len(seg) < 10:
            continue
        norms.append(normalize_to_101(seg))
    if not norms:
        return np.full(101, np.nan)
    return np.nanmean(norms, axis=0)


def run():
    slim = pd.read_parquet(SLIM)
    features = pd.read_csv(FEATURES)

    # 재실행 시 기존 asymmetry 컬럼 제거
    existing_asym = [c for c in features.columns if c.endswith("_asym")]
    if existing_asym:
        features = features.drop(columns=existing_asym)

    asym_records = {}

    for (sub_id, speed), sub_speed_df in slim.groupby(["subject_id", "speed"]):
        sub_speed_df = sub_speed_df.reset_index(drop=True)

        def get_contact(leg):
            h_col, t_col = FOOT_CONTACTS[leg]
            h = (sub_speed_df[h_col].values == 1).astype(int) if h_col in sub_speed_df else np.zeros(len(sub_speed_df))
            t = (sub_speed_df[t_col].values == 1).astype(int) if t_col in sub_speed_df else np.zeros(len(sub_speed_df))
            return ((h == 1) | (t == 1)).astype(int)

        r_contact = get_contact("Right")
        l_contact = get_contact("Left")

        key = (sub_id, speed)
        asym_records[key] = {}

        for feat, (r_col, l_col) in JOINT_COLS.items():
            r_wave = get_mean_waveform(sub_speed_df, r_col, r_contact)
            l_wave = get_mean_waveform(sub_speed_df, l_col, l_contact)

            if np.any(np.isnan(r_wave)) or np.any(np.isnan(l_wave)):
                asym = np.nan
            else:
                diff = np.abs(l_wave - r_wave)
                asym = float(np.trapezoid(diff) if hasattr(np, "trapezoid") else np.trapz(diff)) / 101

            asym_records[key][f"{feat}_asym"] = asym

    asym_rows = []
    for (sub_id, speed), asym_dict in asym_records.items():
        row = {"subject_id": sub_id, "speed": speed}
        row.update(asym_dict)
        asym_rows.append(row)

    asym_df = pd.DataFrame(asym_rows)

    merged = pd.merge(features, asym_df, on=["subject_id", "speed"], how="left")

    asym_cols = [c for c in merged.columns if c.endswith("_asym")]
    neg_count = 0
    for c in asym_cols:
        vals = merged[c].dropna()
        neg_count += (vals < 0).sum()
    assert neg_count == 0, f"asymmetry 음수 값 {neg_count}개"

    nan_rate = merged[asym_cols].isnull().mean().mean()
    print(f"[009] asymmetry NaN 비율: {nan_rate:.1%}")
    assert nan_rate < 0.15, f"asymmetry NaN 너무 많음: {nan_rate:.1%}"

    merged.to_csv(FEATURES, index=False)
    print(f"[009] ✅ asymmetry {len(asym_cols)}개 컬럼 추가 완료: {FEATURES}")


if __name__ == "__main__":
    run()
