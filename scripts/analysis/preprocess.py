import pandas as pd
import numpy as np
import pyarrow.dataset as ds
import pyarrow.compute as pc
import os
import logging
from scipy.signal import find_peaks

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# 경로 참조 (notebooks/에서 실행될 것을 고려해 절대 경로 세팅 유도, 여기선 명시적 상대/절대 사용)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if not os.path.exists(os.path.join(BASE_DIR, "data")):
    # 혹시 scripts 안이 아니면 현재 디렉토리 기준
    BASE_DIR = os.getcwd()

DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "mds")
os.makedirs(OUT_DIR, exist_ok=True)

PATH_ID          = os.path.join(DATA_DIR, "ID.csv")
PATH_RAW         = os.path.join(DATA_DIR, "processed", "raw_merged.parquet")
PATH_GAIT_GLOBAL = os.path.join(DATA_DIR, "gait_analysis_global.csv")
PATH_OUT         = os.path.join(OUT_DIR, "analysis_data.csv")

JOINT_COLS = {
    "hip_flexion"        : ("jointAngle_42", "jointAngle_54"),
    "hip_adduction"      : ("jointAngle_43", "jointAngle_55"),
    "knee_flexion"       : ("jointAngle_45", "jointAngle_57"),
    "knee_adduction"     : ("jointAngle_46", "jointAngle_58"),
    "knee_int_rotation"  : ("jointAngle_47", "jointAngle_59"),
    "ankle_dorsiflexion" : ("jointAngle_48", "jointAngle_60"),
}

# footContacts 컬럼 매핑
# mvnx footContactDefinition 순서: 0=LeftHeel, 1=LeftToe, 2=RightHeel, 3=RightToe
FOOT_CONTACT_COLS = {
    "Right": "footContacts_2",  # RightFoot_Heel
    "Left" : "footContacts_0",  # LeftFoot_Heel
}

PEAK_DIRECTION = {
    "hip_flexion"        : "max",
    "hip_adduction"      : "min",
    "knee_flexion"       : "max",
    "knee_adduction"     : "min",
    "knee_int_rotation"  : "max",
    "ankle_dorsiflexion" : "max",
}

LSI_FEATURES = ["hip_flexion", "knee_flexion", "ankle_dorsiflexion"]
SPATIOTEMPORAL_COLS = [
    "gait_speed_mps", "cadence_spm", "stride_length_mean_m",
    "step_width_mean_m_orth", "double_support_pct",
    "single_support_L_pct", "single_support_R_pct"
]

def extract_mean_peak(signal: np.ndarray, direction: str,
                      distance: int = 100,
                      prominence: float = 1.0,
                      iqr_lower_bound: float = 1.5,
                      iqr_upper_bound: float = 2.5,
                      contact_signal: np.ndarray = None) -> float:
    """
    contact_signal: footContacts 배열 (1=stance, 0=swing).
                    주어지면 stance phase(1인 구간) 내 피크만 추출.
                    None이면 전체 신호에서 find_peaks로 추정 (기존 방식).
    """
    if len(signal) < distance * 2:
        return np.nan

    if contact_signal is not None and len(contact_signal) == len(signal):
        # ── Stance phase 기반 피크 추출 ──────────────────────────────
        # contact == 1 인 인덱스만 추려서, 그 위에서 find_peaks 실행
        stance_idx = np.where(contact_signal == 1)[0]

        if len(stance_idx) < distance * 2:
            return np.nan

        stance_signal = signal[stance_idx]

        if direction == "max":
            local_peaks, _ = find_peaks(stance_signal, distance=distance, prominence=prominence)
        else:
            local_peaks, _ = find_peaks(-stance_signal, distance=distance, prominence=prominence)

        if len(local_peaks) == 0:
            return np.nan

        # stance_signal 내 local index → 원래 signal의 global index로 변환
        peaks = stance_idx[local_peaks]

    else:
        # ── 기존 방식: find_peaks로 전체 신호에서 추정 ───────────────
        if direction == "max":
            peaks, _ = find_peaks(signal, distance=distance, prominence=prominence)
        else:
            peaks, _ = find_peaks(-signal, distance=distance, prominence=prominence)

        if len(peaks) == 0:
            return np.nan

    # 비대칭 IQR 필터링 (피크 수 >= 4 일 때)
    if len(peaks) >= 4:
        target_signal = signal if direction == "max" else -signal
        peak_heights = target_signal[peaks]

        Q1 = np.percentile(peak_heights, 25)
        Q3 = np.percentile(peak_heights, 75)
        IQR = Q3 - Q1

        lower_threshold = Q1 - (iqr_lower_bound * IQR)
        upper_threshold = Q3 + (iqr_upper_bound * IQR)

        valid_peaks = [p for p in peaks if lower_threshold <= target_signal[p] <= upper_threshold]

        if len(valid_peaks) > 0:
            return float(np.mean(signal[valid_peaks]))
        else:
            return np.nan
    else:
        return float(np.mean(signal[peaks]))

def match_subjects(path_id: str):
    id_df = pd.read_csv(path_id)
    acld = id_df[id_df["Group"] == 3].copy()
    aclr = id_df[id_df["Group"] == 4].copy()
    acld["num"] = acld["ID"].str.extract(r"(\d+)$").astype(int)
    aclr["num"] = aclr["ID"].str.extract(r"(\d+)$").astype(int)
    matched = pd.merge(
        acld[["num", "ID", "Sex", "Age", "Weight", "Height", "Injured leg"]],
        aclr[["num", "ID", "Sex", "Age", "Weight", "Height", "Injured leg"]],
        on=["num", "Sex", "Age", "Weight", "Height", "Injured leg"],
        suffixes=("_ACLD", "_ACLR")
    )
    healthy = id_df[id_df["Group"] == 1][["ID"]].rename(columns={"ID": "ID_HA"})
    return matched, healthy

def load_joint_data(path_raw: str, subject_ids: list, needed_cols: list) -> pd.DataFrame:
    contact_cols = list(FOOT_CONTACT_COLS.values())  # footContacts_0, footContacts_2
    select_cols = ["subject_id", "speed", "time_ms"] + needed_cols + contact_cols
    # 중복 제거
    select_cols = list(dict.fromkeys(select_cols))
    dataset = ds.dataset(path_raw, format="parquet")
    filter_expr = pc.field("subject_id").isin(subject_ids)
    table = dataset.to_table(columns=select_cols, filter=filter_expr)
    df = table.to_pandas()
    df["time_ms"] = pd.to_numeric(df["time_ms"], errors="coerce")
    return df.sort_values(["subject_id", "speed", "time_ms"]).reset_index(drop=True)

def compute_features_for_subject(sub_df: pd.DataFrame, subject_id: str, group_label: str, injured_leg: str, limits: dict) -> list:
    records = []
    contra_leg = "Left" if injured_leg == "Right" else "Right"

    for speed in ["fast", "normal", "slow"]:
        speed_df = sub_df[sub_df["speed"] == speed].reset_index(drop=True)
        if len(speed_df) < limits['distance'] * 2:
            continue

        rec = {"subject_id": subject_id, "group": group_label, "speed": speed, "injured_leg": injured_leg}

        # 환측/건측 footContacts 신호 (Heel 기준)
        inj_contact_col   = FOOT_CONTACT_COLS[injured_leg]
        contra_contact_col = FOOT_CONTACT_COLS[contra_leg]
        inj_contact   = speed_df[inj_contact_col].values   if inj_contact_col   in speed_df.columns else None
        contra_contact = speed_df[contra_contact_col].values if contra_contact_col in speed_df.columns else None

        for feat, (r_col, l_col) in JOINT_COLS.items():
            direction = PEAK_DIRECTION[feat]
            injured_col = r_col if injured_leg == "Right" else l_col
            contra_col  = l_col if injured_leg == "Right" else r_col

            inj_signal    = speed_df[injured_col].values
            contra_signal = speed_df[contra_col].values

            # stance phase 기반 피크 추출 (contact_signal 전달)
            rec[f"{feat}_injured"]       = extract_mean_peak(inj_signal,    direction, **limits, contact_signal=inj_contact)
            rec[f"{feat}_contralateral"] = extract_mean_peak(contra_signal, direction, **limits, contact_signal=contra_contact)

        # LSI
        for feat in LSI_FEATURES:
            I = rec.get(f"{feat}_injured", np.nan)
            C = rec.get(f"{feat}_contralateral", np.nan)
            if C is not None and not np.isnan(C) and abs(C) > 1e-6:
                rec[f"{feat}_LSI"] = 100.0 * (I / C)
            else:
                rec[f"{feat}_LSI"] = np.nan

        records.append(rec)
    return records

def extract_all_features(joint_df: pd.DataFrame, matched: pd.DataFrame, healthy: pd.DataFrame, limits: dict) -> pd.DataFrame:
    all_records = []
    for group_label, id_col in [("ACLD", "ID_ACLD"), ("ACLR", "ID_ACLR")]:
        for _, row in matched.iterrows():
            sub_id, injured_leg = row[id_col], row["Injured leg"]
            sub_df = joint_df[joint_df["subject_id"] == sub_id]
            if not sub_df.empty:
                all_records.extend(compute_features_for_subject(sub_df, sub_id, group_label, injured_leg, limits))
                
    for _, row in healthy.iterrows():
        sub_id = row["ID_HA"]
        sub_df = joint_df[joint_df["subject_id"] == sub_id]
        if not sub_df.empty:
            all_records.extend(compute_features_for_subject(sub_df, sub_id, "Healthy", "Right", limits))
            
    return pd.DataFrame(all_records)

def load_and_merge_spatiotemporal(kinematic_df: pd.DataFrame, path_gait: str) -> pd.DataFrame:
    gait = pd.read_csv(path_gait)
    gait["pace_condition"] = gait["pace_condition"].str.lower().str.strip()
    gait = gait.rename(columns={"participant": "subject_id", "pace_condition": "speed"})
    spatio = gait.groupby(["subject_id", "speed"])[SPATIOTEMPORAL_COLS].mean().reset_index()
    return pd.merge(kinematic_df, spatio, on=["subject_id", "speed"], how="left")

def run_preprocessing(distance_val=100, prominence_val=1, iqr_upper=2.5, iqr_lower=1.5):
    log.info("▶ 전처리 파이프라인 시작 (분리된 모듈)")
    limits = {'distance': distance_val, 'prominence': prominence_val, 'iqr_lower_bound': iqr_lower, 'iqr_upper_bound': iqr_upper}
    
    matched, healthy = match_subjects(PATH_ID)
    all_ids = matched["ID_ACLD"].tolist() + matched["ID_ACLR"].tolist() + healthy["ID_HA"].tolist()
    needed_cols = list({col for pair in JOINT_COLS.values() for col in pair})
    
    joint_df = load_joint_data(PATH_RAW, all_ids, needed_cols)
    kinematic_df = extract_all_features(joint_df, matched, healthy, limits)
    full_df = load_and_merge_spatiotemporal(kinematic_df, PATH_GAIT_GLOBAL)
    
    full_df.to_csv(PATH_OUT, index=False, encoding="utf-8-sig")
    log.info(f"✅ 전처리 완료. 결과 저장됨: {PATH_OUT}")
    return full_df
