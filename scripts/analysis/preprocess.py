import pandas as pd
import numpy as np
import pyarrow.dataset as ds
import pyarrow.compute as pc
import os
import logging
from scipy.signal import butter, filtfilt, find_peaks

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# 경로 참조 (notebooks/에서 실행될 것을 고려해 절대 경로 세팅 유도, 여기선 명시적 상대/절대 사용)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if not os.path.exists(os.path.join(BASE_DIR, "data")):
    # 혹시 scripts 안이 아니면 현재 디렉토리 기준
    BASE_DIR = os.getcwd()

DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

PATH_ID = os.path.join(DATA_DIR, "ID.csv")
PATH_RAW = os.path.join(DATA_DIR, "processed", "raw_merged.parquet")
PATH_GAIT_GLOBAL = os.path.join(DATA_DIR, "gait_analysis_global.csv")
PATH_OUT = os.path.join(OUT_DIR, "analysis_data.csv")

JOINT_COLS = {
    "hip_flexion": ("jointAngle_42", "jointAngle_54"),
    "hip_adduction": ("jointAngle_43", "jointAngle_55"),
    "hip_int_rotation": ("jointAngle_44", "jointAngle_56"),
    "knee_flexion": ("jointAngle_45", "jointAngle_57"),
    "knee_adduction": ("jointAngle_46", "jointAngle_58"),
    "knee_int_rotation": ("jointAngle_47", "jointAngle_59"),
    "ankle_dorsiflexion": ("jointAngle_48", "jointAngle_60"),
    "ankle_adduction": ("jointAngle_49", "jointAngle_61"),
    "ankle_int_rotation": ("jointAngle_50", "jointAngle_62"),
}

# footContacts 컬럼 매핑 (오른쪽/왼쪽 발뒤꿈치 접촉)
# mvnx footContactDefinition 순서: 0=LeftHeel, 1=LeftToe, 2=RightHeel, 3=RightToe
FOOT_CONTACT_COLS = {
    "Right": "footContacts_2",  # RightFoot_Heel
    "Left": "footContacts_0",  # LeftFoot_Heel
}

# 피크 추출 방향 (최대값 또는 최소값)
PEAK_DIRECTION = {
    "hip_flexion": "max",
    "hip_adduction": "min",
    "hip_int_rotation": "max",
    "knee_flexion": "max",
    "knee_adduction": "min",
    "knee_int_rotation": "max",
    "ankle_dorsiflexion": "max",
    "ankle_adduction": "min",
    "ankle_int_rotation": "max",
}

# LSI (Limb Symmetry Index) 계산에 사용될 피처 목록
LSI_FEATURES = [
    "hip_flexion",
    "hip_adduction",
    "hip_int_rotation",
    "knee_flexion",
    "knee_adduction",
    "knee_int_rotation",
    "ankle_dorsiflexion",
    "ankle_adduction",
    "ankle_int_rotation",
]
# 시공간(Spatio-temporal) 변수 목록
SPATIOTEMPORAL_COLS = [
    "gait_speed_mps",
    "cadence_spm",
    "stride_length_mean_m",
    "step_width_mean_m_orth",
    "double_support_pct",
    "single_support_L_pct",
    "single_support_R_pct",
]


def extract_mean_peak(
    signal: np.ndarray,
    direction: str,
    distance: int = 100,
    prominence: float = 1.0,
    iqr_lower_bound: float = 1.5,
    iqr_upper_bound: float = 2.5,
    contact_signal: np.ndarray = None,
    peak_method: str = "argextrema",
    butter_order: int = 2,
    butter_cutoff: float = 0.1,
    debug_info: dict = None,
) -> float:
    # 신호 길이가 피크 탐색 최소 거리의 두 배보다 짧으면 NaN 반환
    if len(signal) < distance * 2:
        return np.nan
    if len(signal) < distance * 2:
        raise ValueError("Signal length is too short for peak detection.")

    stance_len = None

    if contact_signal is not None and len(contact_signal) == len(signal):  # 발 접촉 신호가 주어지고 길이가 같으면
        # ── Stance phase 기반 피크 추출 ──────────────────────────────
        # contact == 1 인 인덱스만 추림
        stance_idx = np.where(contact_signal == 1)[0]  # 발 접촉(stance) 구간의 인덱스 추출

        if len(stance_idx) < distance * 2:  # stance 구간 길이가 너무 짧으면 NaN 반환
            return np.nan

        stance_signal = signal[stance_idx]  # stance 구간에 해당하는 신호만 추출
        stance_len = len(stance_idx)

        if peak_method == "argextrema":
            # 연속된 stance 구간마다 argmax/argmin 1개씩 추출
            contact_binary = np.asarray(contact_signal, dtype=int)
            segments = []
            seg_start = None
            for idx, val in enumerate(contact_binary):
                if val == 1 and seg_start is None:
                    seg_start = idx
                elif val != 1 and seg_start is not None:
                    segments.append((seg_start, idx))
                    seg_start = None
            if seg_start is not None:
                segments.append((seg_start, len(contact_binary)))

            peaks = []
            for seg_start, seg_end in segments:
                if seg_end <= seg_start:
                    continue
                seg = signal[seg_start:seg_end]
                if len(seg) == 0:
                    continue
                local_idx = int(np.argmax(seg)) if direction == "max" else int(np.argmin(seg))
                peaks.append(seg_start + local_idx)

            peaks = np.asarray(peaks, dtype=int)
            if len(peaks) == 0:
                return np.nan

        elif peak_method == "butterworth":
            # 버터워스 저역통과 후 기존 find_peaks 적용
            filtered_stance_signal = stance_signal
            try:
                if 0 < butter_cutoff < 1:
                    b, a = butter(butter_order, butter_cutoff, btype="low")
                    pad_len = 3 * (max(len(a), len(b)) - 1)
                    if len(stance_signal) > pad_len:
                        filtered_stance_signal = filtfilt(b, a, stance_signal)
            except ValueError:
                filtered_stance_signal = stance_signal

            if direction == "max":  # 최대 피크를 찾는 경우
                local_peaks, _ = find_peaks(filtered_stance_signal, distance=distance, prominence=prominence)
            else:  # 최소 피크를 찾는 경우 (신호를 뒤집어서 최대 피크 찾기)
                local_peaks, _ = find_peaks(-filtered_stance_signal, distance=distance, prominence=prominence)

            if len(local_peaks) == 0:  # 피크가 없으면 NaN 반환
                return np.nan

            # stance_signal 내 local index → 원래 signal의 global index로 변환
            peaks = stance_idx[local_peaks]  # 원래 신호에서의 피크 인덱스

        else:
            raise ValueError(f"지원하지 않는 peak_method: {peak_method}")

    else:
        raise ValueError("발 접촉 정보 없음.")
        """ 
        # ── 기존 방식: find_peaks로 전체 신호에서 추정 ───────────────
        if direction == "max":  # 최대 피크를 찾는 경우
            peaks, _ = find_peaks(signal, distance=distance, prominence=prominence)
        else:  # 최소 피크를 찾는 경우
            peaks, _ = find_peaks(-signal, distance=distance, prominence=prominence)

        if len(peaks) == 0:  # 피크가 없으면 NaN 반환
            return np.nan
        """

    # 비대칭 IQR 필터링 (피크 수 >= 4 일 때)
    if len(peaks) >= 4:  # 피크가 4개 이상일 경우에만 IQR 필터링 적용
        target_signal = signal if direction == "max" else -signal  # 피크 높이 비교를 위한 신호 (최소 피크는 뒤집은 신호)
        peak_heights = target_signal[peaks]  # 추출된 피크들의 높이

        Q1 = np.percentile(peak_heights, 25)  # 1사분위수
        Q3 = np.percentile(peak_heights, 75)  # 3사분위수
        IQR = Q3 - Q1  # 사분위 범위

        lower_threshold = Q1 - (iqr_lower_bound * IQR)  # 하한 임계값
        upper_threshold = Q3 + (iqr_upper_bound * IQR)  # 상한 임계값

        valid_peaks = [p for p in peaks if lower_threshold <= target_signal[p] <= upper_threshold]  # 임계값 범위 내의 유효한 피크

        if len(valid_peaks) > 0:  # 유효한 피크가 있으면 평균 반환
            return float(np.mean(signal[valid_peaks]))
        else:  # 유효한 피크가 없으면 NaN 반환
            return np.nan
    else:
        info = debug_info or {}
        print(
            "[PEAK_DEBUG] reason=too_few_peaks "
            f"subject_id={info.get('subject_id', 'NA')} "
            f"group={info.get('group', 'NA')} "
            f"speed={info.get('speed', 'NA')} "
            f"feature={info.get('feature', 'NA')} "
            f"side={info.get('side', 'NA')} "
            f"leg={info.get('leg', 'NA')} "
            f"direction={direction} "
            f"peak_method={peak_method} "
            f"peak_count={len(peaks)} "
            f"signal_len={len(signal)} "
            f"stance_len={stance_len if stance_len is not None else 'NA'} "
            f"distance={distance} "
            f"prominence={prominence}"
        )
        return np.nan


# ID 매칭 함수 (ACLD, ACLR, Healthy 그룹 간 매칭)
def match_subjects(path_id: str):
    # ID CSV 파일 로드
    id_df = pd.read_csv(path_id)
    acld = id_df[id_df["Group"] == 3].copy()  # ACLD 그룹 필터링
    aclr = id_df[id_df["Group"] == 4].copy()  # ACLR 그룹 필터링
    acld["num"] = acld["ID"].str.extract(r"(\d+)$").astype(int)  # ACLD ID에서 숫자 부분 추출
    aclr["num"] = aclr["ID"].str.extract(r"(\d+)$").astype(int)  # ACLR ID에서 숫자 부분 추출
    matched = pd.merge(  # ACLD와 ACLR 그룹을 공통 정보(num, Sex, Age 등)를 기준으로 병합
        acld[["num", "ID", "Sex", "Age", "Weight", "Height", "Injured leg"]],
        aclr[["num", "ID", "Sex", "Age", "Weight", "Height", "Injured leg"]],
        on=["num", "Sex", "Age", "Weight", "Height", "Injured leg"],
        suffixes=("_ACLD", "_ACLR"),  # 컬럼 이름 충돌 방지
    )
    print(f"==={len(matched)}명 매치됨===")
    healthy = id_df[id_df["Group"] == 1][["ID"]].rename(columns={"ID": "ID_HA"})  # Healthy 그룹 필터링 및 컬럼 이름 변경
    return matched, healthy  # 매칭된 데이터와 Healthy 그룹 데이터 반환


# 관절 데이터 로드 함수 (PyArrow Dataset을 사용하여 효율적으로 로드)
def load_joint_data(path_raw: str, subject_ids: list, needed_cols: list) -> pd.DataFrame:
    # 발 접촉 컬럼 목록
    contact_cols = list(FOOT_CONTACT_COLS.values())
    # 선택할 컬럼 목록 구성
    select_cols = ["subject_id", "speed", "time_ms"] + needed_cols + contact_cols
    # 중복 제거 (set을 이용한 후 다시 list로 변환)
    select_cols = list(dict.fromkeys(select_cols))
    # Parquet 데이터셋 생성
    dataset = ds.dataset(path_raw, format="parquet")
    # subject_id로 필터링할 표현식 생성
    filter_expr = pc.field("subject_id").isin(subject_ids)
    # 필터링된 데이터와 선택된 컬럼만 PyArrow Table로 로드
    table = dataset.to_table(columns=select_cols, filter=filter_expr)
    # PyArrow Table을 Pandas DataFrame으로 변환
    df = table.to_pandas()
    df["time_ms"] = pd.to_numeric(df["time_ms"], errors="coerce")  # time_ms 컬럼을 숫자형으로 변환
    return df.sort_values(["subject_id", "speed", "time_ms"]).reset_index(drop=True)  # 정렬 후 인덱스 재설정하여 반환


# 각 피험자에 대한 피처 계산 함수
def compute_features_for_subject(
    sub_df: pd.DataFrame,
    subject_id: str,
    group_label: str,
    injured_leg: str,
    limits: dict,
) -> list:
    # 결과를 저장할 리스트
    records = []
    contra_leg = "Left" if injured_leg == "Right" else "Right"  # 건측(반대쪽 다리) 결정

    for speed in ["fast", "normal", "slow"]:  # 각 속도(fast, normal, slow)에 대해 반복
        speed_df = sub_df[sub_df["speed"] == speed].reset_index(drop=True)  # 해당 속도의 데이터 필터링
        if len(speed_df) < limits["distance"] * 2:  # 데이터 길이가 피크 탐색에 충분하지 않으면 건너뛰기
            raise ValueError("데이터 길이가 피크 탐색에 충분하지 않음.")

        rec = {
            "subject_id": subject_id,
            "group": group_label,
            "speed": speed,
            "injured_leg": injured_leg,
        }  # 레코드 초기화

        # 환측/건측 footContacts 신호 (Heel 기준)
        inj_contact_col = FOOT_CONTACT_COLS[injured_leg]  # 환측 발 접촉 컬럼
        contra_contact_col = FOOT_CONTACT_COLS[contra_leg]  # 건측 발 접촉 컬럼
        inj_contact = speed_df[inj_contact_col].values if inj_contact_col in speed_df.columns else None  # 환측 발 접촉 신호
        contra_contact = speed_df[contra_contact_col].values if contra_contact_col in speed_df.columns else None  # 건측 발 접촉 신호

        for feat, (r_col, l_col) in JOINT_COLS.items():  # 각 관절 피처에 대해 반복
            direction = PEAK_DIRECTION[feat]  # 피크 추출 방향
            injured_col = r_col if injured_leg == "Right" else l_col  # 환측 관절 각도 컬럼
            contra_col = l_col if injured_leg == "Right" else r_col  # 건측 관절 각도 컬럼

            inj_signal = speed_df[injured_col].values  # 환측 신호
            contra_signal = speed_df[contra_col].values  # 건측 신호

            # stance phase 기반 피크 추출 (contact_signal 전달)
            rec[f"{feat}_injured"] = extract_mean_peak(
                inj_signal,
                direction,
                **limits,
                contact_signal=inj_contact,
                debug_info={
                    "subject_id": subject_id,
                    "group": group_label,
                    "speed": speed,
                    "feature": feat,
                    "side": "injured",
                    "leg": injured_leg,
                },
            )  # 환측 피크 추출
            rec[f"{feat}_contralateral"] = extract_mean_peak(
                contra_signal,
                direction,
                **limits,
                contact_signal=contra_contact,
                debug_info={
                    "subject_id": subject_id,
                    "group": group_label,
                    "speed": speed,
                    "feature": feat,
                    "side": "contralateral",
                    "leg": contra_leg,
                },
            )  # 건측 피크 추출

        # LSI (Limb Symmetry Index) 계산
        for feat in LSI_FEATURES:  # LSI 계산 대상 피처에 대해 반복
            injured_value = rec.get(f"{feat}_injured", np.nan)  # 환측 값
            contra_value = rec.get(f"{feat}_contralateral", np.nan)  # 건측 값
            if contra_value is not None and not np.isnan(contra_value) and abs(contra_value) > 1e-6:  # 건측 값이 유효하고 0이 아니면
                rec[f"{feat}_LSI"] = 100.0 * (injured_value / contra_value)  # LSI 계산
            else:  # 그렇지 않으면 NaN
                rec[f"{feat}_LSI"] = np.nan

        records.append(rec)  # 현재 속도에 대한 레코드 추가
    return records  # 모든 레코드 반환


# 모든 피처를 추출하는 메인 함수
def extract_all_features(joint_df: pd.DataFrame, matched: pd.DataFrame, healthy: pd.DataFrame, limits: dict) -> pd.DataFrame:
    # 모든 피험자의 레코드를 저장할 리스트
    all_records = []
    for group_label, id_col in [
        ("ACLD", "ID_ACLD"),
        ("ACLR", "ID_ACLR"),
    ]:  # ACLD, ACLR 그룹에 대해 반복
        for _, row in matched.iterrows():  # 매칭된 피험자 데이터에 대해 반복
            sub_id, injured_leg = (
                row[id_col],
                row["Injured leg"],
            )  # 피험자 ID와 손상된 다리 정보 추출
            sub_df = joint_df[joint_df["subject_id"] == sub_id]  # 해당 피험자의 관절 데이터 필터링
            if not sub_df.empty:  # 데이터가 비어있지 않으면
                all_records.extend(compute_features_for_subject(sub_df, sub_id, group_label, injured_leg, limits))  # 피처 계산 및 추가

    for _, row in healthy.iterrows():  # Healthy 그룹에 대해 반복
        sub_id = row["ID_HA"]  # 피험자 ID 추출
        sub_df = joint_df[joint_df["subject_id"] == sub_id]  # 해당 피험자의 관절 데이터 필터링
        if not sub_df.empty:  # 데이터가 비어있지 않으면
            all_records.extend(
                compute_features_for_subject(sub_df, sub_id, "Healthy", "Right", limits)
            )  # 피처 계산 및 추가 (Healthy는 손상된 다리 없음, 임의로 "Right" 설정)

    return pd.DataFrame(all_records)  # 모든 레코드를 포함하는 데이터프레임 반환


# 시공간 데이터 로드 및 병합 함수
def load_and_merge_spatiotemporal(kinematic_df: pd.DataFrame, path_gait: str) -> pd.DataFrame:
    # 보행 분석 글로벌 데이터 로드
    gait = pd.read_csv(path_gait)
    gait["pace_condition"] = gait["pace_condition"].str.lower().str.strip()  # 'pace_condition' 컬럼 값 소문자 및 공백 제거
    gait = gait.rename(columns={"participant": "subject_id", "pace_condition": "speed"})  # 컬럼 이름 변경
    spatio = gait.groupby(["subject_id", "speed"])[SPATIOTEMPORAL_COLS].mean().reset_index()  # subject_id와 speed별로 시공간 피처 평균 집계
    return pd.merge(kinematic_df, spatio, on=["subject_id", "speed"], how="left")  # 운동학적 데이터와 시공간 데이터 병합


# 전처리 파이프라인 실행 메인 함수
def run_preprocessing(
    distance_val=100,
    prominence_val=1,
    iqr_upper=2.5,
    iqr_lower=1.5,
    peak_method="argextrema",
    butter_order=2,
    butter_cutoff=0.1,
):
    # 파이프라인 시작 로깅
    log.info("▶ 전처리 파이프라인 시작 (분리된 모듈)")
    limits = {
        "distance": distance_val,
        "prominence": prominence_val,
        "iqr_lower_bound": iqr_lower,
        "iqr_upper_bound": iqr_upper,
        "peak_method": peak_method,
        "butter_order": butter_order,
        "butter_cutoff": butter_cutoff,
    }  # 피크 추출 파라미터 딕셔너리

    matched, healthy = match_subjects(PATH_ID)  # ID 매칭
    all_ids = matched["ID_ACLD"].tolist() + matched["ID_ACLR"].tolist() + healthy["ID_HA"].tolist()  # 모든 피험자 ID 목록
    needed_cols = list({col for pair in JOINT_COLS.values() for col in pair})  # 필요한 관절 각도 컬럼 목록

    joint_df = load_joint_data(PATH_RAW, all_ids, needed_cols)  # 관절 데이터 로드
    kinematic_df = extract_all_features(joint_df, matched, healthy, limits)  # 운동학적 피처 추출
    full_df = load_and_merge_spatiotemporal(kinematic_df, PATH_GAIT_GLOBAL)  # 시공간 데이터 병합

    full_df.to_csv(PATH_OUT, index=False, encoding="utf-8-sig")  # 최종 데이터 CSV 파일로 저장
    log.info(f"✅ 전처리 완료. 결과 저장됨: {PATH_OUT}")  # 전처리 완료 로깅
    return full_df  # 최종 데이터프레임 반환


run_preprocessing()
