import pandas as pd  # 표 형태 데이터 처리를 위한 핵심 라이브러리
import numpy as np  # 수치 연산(배열, 결측치, 통계값 계산)에 사용
import pyarrow.dataset as ds  # Parquet 데이터셋을 컬럼/필터 단위로 효율 로딩할 때 사용
import pyarrow.compute as pc  # PyArrow 필터 표현식 생성에 사용
import os  # 파일/경로 조작 및 디렉터리 생성에 사용
import logging  # 전처리 진행 상황을 콘솔에 기록하기 위해 사용
from scipy.signal import butter, filtfilt, find_peaks  # 신호 필터링 및 피크 검출 함수

# 로그 출력 형식을 단순 메시지로 지정해 노트북/터미널에서 가독성을 높인다.
logging.basicConfig(level=logging.INFO, format="%(message)s")
# 현재 모듈 이름으로 로거 인스턴스를 생성한다.
log = logging.getLogger(__name__)

# 현재 파일 위치(preprocess.py)를 기준으로 프로젝트 루트(../..)를 계산한다.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 계산된 경로 아래에 data 폴더가 없으면, 실행 위치(CWD)를 루트로 간주한다.
if not os.path.exists(os.path.join(BASE_DIR, "data")):
    # scripts 폴더 외부(예: notebooks)에서 실행될 때를 대비한 폴백 경로 처리다.
    BASE_DIR = os.getcwd()

# 원본/중간/결과 데이터가 들어있는 최상위 data 폴더 경로다.
DATA_DIR = os.path.join(BASE_DIR, "data")
# 최종 분석 결과를 저장할 processed 폴더 경로다.
OUT_DIR = os.path.join(BASE_DIR, "data", "processed")
# 출력 폴더가 없으면 생성하고, 이미 있으면 오류 없이 계속 진행한다.
os.makedirs(OUT_DIR, exist_ok=True)

# 피험자 메타 정보(ID, 그룹, 성별 등)가 담긴 입력 CSV 경로다.
PATH_ID = os.path.join(DATA_DIR, "ID.csv")
# 센서 기반 원시 시계열이 병합된 Parquet 파일 경로다.
PATH_RAW = os.path.join(DATA_DIR, "processed", "raw_merged.parquet")
# 시공간(spatio-temporal) 보행 지표 CSV 경로다.
PATH_GAIT_GLOBAL = os.path.join(DATA_DIR, "gait_analysis_global.csv")
# 최종 전처리 산출물 CSV 저장 경로다.
PATH_OUT = os.path.join(OUT_DIR, "analysis_data.csv")
# 피크 개별 레코드(메타데이터 + IQR 통과 여부) CSV 저장 경로다.
PATH_PEAKS = os.path.join(OUT_DIR, "peak_records.csv")

# 동일 인원으로 간주할 subject_id 별칭 맵이다(로딩 후 canonical ID로 통일한다).
SUBJECT_ID_ALIASES = {
    "ACLR38": "ACLR36",
}

# 관절 피처명과 원본 컬럼(오른쪽, 왼쪽) 매핑 테이블이다.
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

# footContacts 컬럼 매핑(발뒤꿈치 접촉 기준)이다.
# mvnx footContactDefinition 순서: 0=LeftHeel, 1=LeftToe, 2=RightHeel, 3=RightToe
FOOT_CONTACT_COLS = {
    "Right": "footContacts_2",  # 오른발 뒤꿈치 접촉(환측/건측 분기에서 사용)
    "Left": "footContacts_0",  # 왼발 뒤꿈치 접촉(환측/건측 분기에서 사용)
}

# 피처별 피크 방향 규칙(최대값/최소값)을 정의한다.
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

# LSI(Limb Symmetry Index)를 계산할 피처 목록이다.
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

# 시공간(spatio-temporal) 평균 집계에 사용할 컬럼 목록이다.
SPATIOTEMPORAL_COLS = [
    "gait_speed_mps",
    "cadence_spm",
    "stride_length_mean_m",
    "step_width_mean_m_orth",
    "double_support_pct",
    "single_support_L_pct",
    "single_support_R_pct",
]


def get_stance_segments(contact_signal: np.ndarray) -> list:
    """contact(0/1) 신호에서 연속된 stance 구간 [start, end) 목록을 반환한다."""
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

    return segments


def detect_peaks_with_iqr(
    signal: np.ndarray,
    direction: str,
    distance: int,
    prominence: float,
    iqr_lower_bound: float,
    iqr_upper_bound: float,
    contact_signal: np.ndarray,
    peak_method: str,
    butter_order: int,
    butter_cutoff: float,
) -> dict:
    """stance 기반 피크를 검출하고 IQR 필터 결과(통과/탈락)를 함께 반환한다."""
    result = {
        "all_peaks": np.array([], dtype=int),
        "valid_peaks": np.array([], dtype=int),
        "rejected_peaks": np.array([], dtype=int),
        "reason": "",
        "stance_len": None,
        "q1": np.nan,
        "q3": np.nan,
        "iqr": np.nan,
        "lower_threshold": np.nan,
        "upper_threshold": np.nan,
    }

    if contact_signal is None or len(contact_signal) != len(signal):
        result["reason"] = "invalid_contact_signal"
        return result

    stance_idx = np.where(contact_signal == 1)[0]
    result["stance_len"] = int(len(stance_idx))

    if len(stance_idx) < distance * 2:
        result["reason"] = "stance_too_short"
        return result

    if peak_method == "argextrema":
        peaks = []
        for seg_start, seg_end in get_stance_segments(contact_signal):
            if seg_end <= seg_start:
                continue
            seg = signal[seg_start:seg_end]
            if len(seg) == 0:
                continue
            local_idx = int(np.argmax(seg)) if direction == "max" else int(np.argmin(seg))
            peaks.append(seg_start + local_idx)
        peaks = np.asarray(peaks, dtype=int)

    elif peak_method == "butterworth":
        stance_signal = signal[stance_idx]
        filtered_stance_signal = stance_signal

        try:
            if 0 < butter_cutoff < 1:
                b, a = butter(butter_order, butter_cutoff, btype="low")
                pad_len = 3 * (max(len(a), len(b)) - 1)
                if len(stance_signal) > pad_len:
                    filtered_stance_signal = filtfilt(b, a, stance_signal)
        except ValueError:
            filtered_stance_signal = stance_signal

        if direction == "max":
            local_peaks, _ = find_peaks(filtered_stance_signal, distance=distance, prominence=prominence)
        else:
            local_peaks, _ = find_peaks(-filtered_stance_signal, distance=distance, prominence=prominence)

        peaks = stance_idx[local_peaks] if len(local_peaks) > 0 else np.array([], dtype=int)

    else:
        raise ValueError(f"지원하지 않는 peak_method: {peak_method}")

    result["all_peaks"] = peaks

    if len(peaks) == 0:
        result["reason"] = "no_peaks"
        return result

    target_signal = signal if direction == "max" else -signal

    if len(peaks) >= 4:
        peak_heights = target_signal[peaks]
        q1 = np.percentile(peak_heights, 25)
        q3 = np.percentile(peak_heights, 75)
        iqr = q3 - q1
        lower_threshold = q1 - (iqr_lower_bound * iqr)
        upper_threshold = q3 + (iqr_upper_bound * iqr)

        valid_mask = (target_signal[peaks] >= lower_threshold) & (target_signal[peaks] <= upper_threshold)
        valid_peaks = peaks[valid_mask]
        rejected_peaks = peaks[~valid_mask]

        result["valid_peaks"] = np.asarray(valid_peaks, dtype=int)
        result["rejected_peaks"] = np.asarray(rejected_peaks, dtype=int)
        result["q1"] = float(q1)
        result["q3"] = float(q3)
        result["iqr"] = float(iqr)
        result["lower_threshold"] = float(lower_threshold)
        result["upper_threshold"] = float(upper_threshold)
        result["reason"] = "ok" if len(valid_peaks) > 0 else "all_filtered_by_iqr"
    else:
        result["valid_peaks"] = np.array([], dtype=int)
        result["rejected_peaks"] = np.asarray(peaks, dtype=int)
        result["reason"] = "too_few_peaks_for_iqr"

    return result


def _mean_from_valid_peaks(signal: np.ndarray, valid_peaks: np.ndarray) -> float:
    """IQR 통과 피크 평균을 계산한다. 통과 피크가 없으면 NaN을 반환한다."""
    if valid_peaks is None or len(valid_peaks) == 0:
        return np.nan
    return float(np.mean(signal[valid_peaks]))


def _log_too_few_peaks(
    debug_info: dict,
    direction: str,
    peak_method: str,
    peak_count: int,
    signal_len: int,
    stance_len,
    distance: int,
    prominence: float,
):
    """피크 수 부족 케이스를 기존 포맷으로 로깅한다."""
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
        f"peak_count={peak_count} "
        f"signal_len={signal_len} "
        f"stance_len={stance_len if stance_len is not None else 'NA'} "
        f"distance={distance} "
        f"prominence={prominence}"
    )


def build_peak_records(
    subject_id: str,
    group_label: str,
    speed: str,
    injured_leg: str,
    side: str,
    leg: str,
    feature: str,
    direction: str,
    peak_method: str,
    signal_col: str,
    contact_col: str,
    time_ms: np.ndarray,
    signal: np.ndarray,
    detection: dict,
    limits: dict,
) -> list:
    """피크 개별 레코드(메타데이터 + IQR 통과 여부)를 행 리스트로 만든다."""
    rows = []
    all_peaks = detection["all_peaks"]
    valid_set = {int(v) for v in detection["valid_peaks"]}
    rejected_set = {int(v) for v in detection["rejected_peaks"]}
    iqr_applied = len(all_peaks) >= 4

    for peak_idx in all_peaks:
        idx = int(peak_idx)
        time_val = np.nan
        signal_val = np.nan

        if 0 <= idx < len(time_ms):
            time_val = float(time_ms[idx])
        if 0 <= idx < len(signal):
            signal_val = float(signal[idx])

        rows.append(
            {
                "subject_id": subject_id,
                "group": group_label,
                "speed": speed,
                "injured_leg": injured_leg,
                "side": side,
                "leg": leg,
                "feature": feature,
                "direction": direction,
                "peak_method": peak_method,
                "signal_col": signal_col,
                "contact_col": contact_col,
                "signal_len": int(len(signal)),
                "stance_len": detection["stance_len"],
                "peak_index": idx,
                "time_ms": time_val,
                "peak_value": signal_val,
                "iqr_applied": iqr_applied,
                "iqr_pass": idx in valid_set,
                "is_rejected": idx in rejected_set,
                "reason": detection["reason"],
                "q1": detection["q1"],
                "q3": detection["q3"],
                "iqr": detection["iqr"],
                "lower_threshold": detection["lower_threshold"],
                "upper_threshold": detection["upper_threshold"],
                "distance": limits["distance"],
                "prominence": limits["prominence"],
                "iqr_lower_bound": limits["iqr_lower_bound"],
                "iqr_upper_bound": limits["iqr_upper_bound"],
                "butter_order": limits["butter_order"],
                "butter_cutoff": limits["butter_cutoff"],
            }
        )

    return rows


def extract_mean_peak(
    signal: np.ndarray,
    direction: str,
    distance: int = 50,
    prominence: float = 1.0,
    iqr_lower_bound: float = 1.5,
    iqr_upper_bound: float = 2.5,
    contact_signal: np.ndarray = None,
    peak_method: str = "argextrema",
    butter_order: int = 2,
    butter_cutoff: float = 0.1,
    debug_info: dict = None,
) -> float:
    """stance phase(발 접촉 구간) 기반으로 평균 피크 값을 계산한다."""
    if len(signal) < distance * 2:
        raise ValueError("Signal length is too short for peak detection.")

    if contact_signal is None or len(contact_signal) != len(signal):
        raise ValueError("발 접촉 정보 없음.")

    detection = detect_peaks_with_iqr(
        signal=signal,
        direction=direction,
        distance=distance,
        prominence=prominence,
        iqr_lower_bound=iqr_lower_bound,
        iqr_upper_bound=iqr_upper_bound,
        contact_signal=contact_signal,
        peak_method=peak_method,
        butter_order=butter_order,
        butter_cutoff=butter_cutoff,
    )

    # 기존과 동일하게 피크 수가 적어 IQR 적용이 불가한 케이스는 디버그 로그를 남긴다.
    if detection["reason"] == "too_few_peaks_for_iqr":
        _log_too_few_peaks(
            debug_info=debug_info,
            direction=direction,
            peak_method=peak_method,
            peak_count=len(detection["all_peaks"]),
            signal_len=len(signal),
            stance_len=detection["stance_len"],
            distance=distance,
            prominence=prominence,
        )

    return _mean_from_valid_peaks(signal, detection["valid_peaks"])


def match_subjects(path_id: str):
    """ID.csv를 이용해 ACLD/ACLR 매칭 테이블과 Healthy 목록을 생성한다."""
    # ID 메타데이터 CSV를 읽어 전체 피험자 표를 만든다.
    id_df = pd.read_csv(path_id)
    # Group==3(ACLD) 행만 복사해 별도 테이블로 분리한다.
    acld = id_df[id_df["Group"] == 3].copy()
    # Group==4(ACLR) 행만 복사해 별도 테이블로 분리한다.
    aclr = id_df[id_df["Group"] == 4].copy()
    # ACLD ID 끝 숫자(예: ACLD36 -> 36)를 추출해 정수형 num 컬럼을 만든다.
    acld["num"] = acld["ID"].str.extract(r"(\d+)$").astype(int)
    # ACLR도 동일하게 끝 숫자를 추출해 정수형 num 컬럼을 만든다.
    aclr["num"] = aclr["ID"].str.extract(r"(\d+)$").astype(int)

    # ACLD/ACLR를 num+인구통계+손상측 기준으로 inner merge해 짝을 만든다.
    matched = pd.merge(
        acld[["num", "ID", "Sex", "Age", "Weight", "Height", "Injured leg"]],
        aclr[["num", "ID", "Sex", "Age", "Weight", "Height", "Injured leg"]],
        on=["num", "Sex", "Age", "Weight", "Height", "Injured leg"],
        suffixes=("_ACLD", "_ACLR"),
    )

    # 매칭 인원 수를 즉시 출력해 데이터 상태를 빠르게 확인한다.
    print(f"==={len(matched)}명 매치됨===")

    # Healthy 그룹(Group==1) ID를 추출해 컬럼명을 ID_HA로 맞춘다.
    healthy = id_df[id_df["Group"] == 1][["ID"]].rename(columns={"ID": "ID_HA"})

    # ACLD/ACLR 매칭 표와 Healthy ID 표를 함께 반환한다.
    return matched, healthy


def load_joint_data(path_raw: str, subject_ids: list, needed_cols: list) -> pd.DataFrame:
    """Parquet 원시 관절 데이터를 subject_id 기준으로 필터링해 로드한다."""
    # 발 접촉 신호 컬럼(좌/우 heel) 목록을 준비한다.
    contact_cols = list(FOOT_CONTACT_COLS.values())

    # 기본 대상 ID 집합을 set으로 만든다(빠른 포함 검사/중복 제거 목적).
    load_subject_ids = set(subject_ids)

    # canonical ID가 요청되었으면 별칭 원본 ID도 함께 읽어오도록 집합에 추가한다.
    for src_id, canonical_id in SUBJECT_ID_ALIASES.items():
        if canonical_id in load_subject_ids:
            load_subject_ids.add(src_id)

    # 로드할 컬럼 목록(기본 키 + 관절각 + 발접촉)을 조합한다.
    select_cols = ["subject_id", "speed", "time_ms"] + needed_cols + contact_cols
    # dict.fromkeys를 이용해 순서를 유지하면서 중복 컬럼을 제거한다.
    select_cols = list(dict.fromkeys(select_cols))

    # Parquet 파일/폴더를 데이터셋으로 연다.
    dataset = ds.dataset(path_raw, format="parquet")
    # subject_id IN (...) 형태의 PyArrow 필터 표현식을 만든다.
    filter_expr = pc.field("subject_id").isin(sorted(load_subject_ids))
    # 필요한 컬럼 + 필터 조건만 반영해 테이블을 읽어 메모리 사용을 줄인다.
    table = dataset.to_table(columns=select_cols, filter=filter_expr)
    # PyArrow Table을 pandas DataFrame으로 변환한다.
    df = table.to_pandas()

    # 별칭 ID가 몇 행인지 계산해 치환 로그를 제어한다.
    alias_hits = int(df["subject_id"].isin(SUBJECT_ID_ALIASES.keys()).sum())
    # 별칭이 실제로 존재하면 canonical ID로 일괄 치환한다.
    if alias_hits > 0:
        df["subject_id"] = df["subject_id"].replace(SUBJECT_ID_ALIASES)
        log.info(f"ℹ subject_id 별칭 치환 적용 (raw): {alias_hits} rows")

    # time_ms를 숫자형으로 강제 변환하고 실패 값은 NaN으로 둔다.
    df["time_ms"] = pd.to_numeric(df["time_ms"], errors="coerce")
    # subject/speed/time 순으로 정렬해 후속 피크 연산 시 시간축 일관성을 보장한다.
    return df.sort_values(["subject_id", "speed", "time_ms"]).reset_index(drop=True)


def compute_features_for_subject(
    sub_df: pd.DataFrame,
    subject_id: str,
    group_label: str,
    injured_leg: str,
    limits: dict,
) -> tuple[list, list]:
    """피험자 1명에 대해 속도별 평균 피크 피처와 개별 피크 레코드를 함께 계산한다."""
    # speed별 평균 피처 레코드를 누적할 리스트다.
    records = []
    # 개별 피크(메타데이터 포함) 레코드를 누적할 리스트다.
    peak_rows = []
    # 환측의 반대쪽 다리를 건측으로 정의한다.
    contra_leg = "Left" if injured_leg == "Right" else "Right"

    # fast/normal/slow 세 가지 속도 조건을 모두 순회한다.
    for speed in ["fast", "normal", "slow"]:
        # 현재 속도에 해당하는 시계열만 선택하고 인덱스를 재정렬한다.
        speed_df = sub_df[sub_df["speed"] == speed].reset_index(drop=True)

        # 피크 탐지 최소 길이를 만족하지 못하면 해당 피험자/속도는 예외 처리한다.
        if len(speed_df) < limits["distance"] * 2:
            raise ValueError("데이터 길이가 피크 탐색에 충분하지 않음.")

        # 현재 속도 레코드의 기본 메타 필드를 초기화한다.
        rec = {
            "subject_id": subject_id,
            "group": group_label,
            "speed": speed,
            "injured_leg": injured_leg,
        }

        # 환측 heel contact 컬럼명을 가져온다.
        inj_contact_col = FOOT_CONTACT_COLS[injured_leg]
        # 건측 heel contact 컬럼명을 가져온다.
        contra_contact_col = FOOT_CONTACT_COLS[contra_leg]
        # 환측 contact 신호 배열을 추출하고 컬럼이 없으면 None으로 둔다.
        inj_contact = speed_df[inj_contact_col].values if inj_contact_col in speed_df.columns else None
        # 건측 contact 신호 배열을 추출하고 컬럼이 없으면 None으로 둔다.
        contra_contact = speed_df[contra_contact_col].values if contra_contact_col in speed_df.columns else None
        # 개별 피크 레코드에 넣기 위한 시간축 배열을 뽑는다.
        time_ms = pd.to_numeric(speed_df["time_ms"], errors="coerce").values

        # 정의된 관절 피처를 하나씩 순회하며 환측/건측 피크를 계산한다.
        for feat, (r_col, l_col) in JOINT_COLS.items():
            # 해당 피처의 피크 방향(max/min)을 조회한다.
            direction = PEAK_DIRECTION[feat]
            # 환측이 오른쪽이면 r_col, 왼쪽이면 l_col을 환측 신호 컬럼으로 사용한다.
            injured_col = r_col if injured_leg == "Right" else l_col
            # 건측은 환측의 반대 컬럼을 사용한다.
            contra_col = l_col if injured_leg == "Right" else r_col

            # 환측 관절 각도 시계열을 numpy 배열로 뽑는다.
            inj_signal = speed_df[injured_col].values
            # 건측 관절 각도 시계열을 numpy 배열로 뽑는다.
            contra_signal = speed_df[contra_col].values

            # 환측 피크 검출 + IQR 판정을 수행한다.
            inj_detection = detect_peaks_with_iqr(
                signal=inj_signal,
                direction=direction,
                distance=limits["distance"],
                prominence=limits["prominence"],
                iqr_lower_bound=limits["iqr_lower_bound"],
                iqr_upper_bound=limits["iqr_upper_bound"],
                contact_signal=inj_contact,
                peak_method=limits["peak_method"],
                butter_order=limits["butter_order"],
                butter_cutoff=limits["butter_cutoff"],
            )
            # 환측 평균 피크(= IQR 통과 피크 평균)를 저장한다.
            rec[f"{feat}_injured"] = _mean_from_valid_peaks(inj_signal, inj_detection["valid_peaks"])

            # 건측 피크 검출 + IQR 판정을 수행한다.
            contra_detection = detect_peaks_with_iqr(
                signal=contra_signal,
                direction=direction,
                distance=limits["distance"],
                prominence=limits["prominence"],
                iqr_lower_bound=limits["iqr_lower_bound"],
                iqr_upper_bound=limits["iqr_upper_bound"],
                contact_signal=contra_contact,
                peak_method=limits["peak_method"],
                butter_order=limits["butter_order"],
                butter_cutoff=limits["butter_cutoff"],
            )
            # 건측 평균 피크(= IQR 통과 피크 평균)를 저장한다.
            rec[f"{feat}_contralateral"] = _mean_from_valid_peaks(contra_signal, contra_detection["valid_peaks"])

            # 기존과 동일하게 "피크 수 부족" 케이스는 디버그 로그를 남긴다.
            if inj_detection["reason"] == "too_few_peaks_for_iqr":
                _log_too_few_peaks(
                    debug_info={
                        "subject_id": subject_id,
                        "group": group_label,
                        "speed": speed,
                        "feature": feat,
                        "side": "injured",
                        "leg": injured_leg,
                    },
                    direction=direction,
                    peak_method=limits["peak_method"],
                    peak_count=len(inj_detection["all_peaks"]),
                    signal_len=len(inj_signal),
                    stance_len=inj_detection["stance_len"],
                    distance=limits["distance"],
                    prominence=limits["prominence"],
                )

            if contra_detection["reason"] == "too_few_peaks_for_iqr":
                _log_too_few_peaks(
                    debug_info={
                        "subject_id": subject_id,
                        "group": group_label,
                        "speed": speed,
                        "feature": feat,
                        "side": "contralateral",
                        "leg": contra_leg,
                    },
                    direction=direction,
                    peak_method=limits["peak_method"],
                    peak_count=len(contra_detection["all_peaks"]),
                    signal_len=len(contra_signal),
                    stance_len=contra_detection["stance_len"],
                    distance=limits["distance"],
                    prominence=limits["prominence"],
                )

            # 환측 개별 피크를 메타데이터와 함께 기록한다.
            peak_rows.extend(
                build_peak_records(
                    subject_id=subject_id,
                    group_label=group_label,
                    speed=speed,
                    injured_leg=injured_leg,
                    side="injured",
                    leg=injured_leg,
                    feature=feat,
                    direction=direction,
                    peak_method=limits["peak_method"],
                    signal_col=injured_col,
                    contact_col=inj_contact_col,
                    time_ms=time_ms,
                    signal=inj_signal,
                    detection=inj_detection,
                    limits=limits,
                )
            )

            # 건측 개별 피크를 메타데이터와 함께 기록한다.
            peak_rows.extend(
                build_peak_records(
                    subject_id=subject_id,
                    group_label=group_label,
                    speed=speed,
                    injured_leg=injured_leg,
                    side="contralateral",
                    leg=contra_leg,
                    feature=feat,
                    direction=direction,
                    peak_method=limits["peak_method"],
                    signal_col=contra_col,
                    contact_col=contra_contact_col,
                    time_ms=time_ms,
                    signal=contra_signal,
                    detection=contra_detection,
                    limits=limits,
                )
            )

        # LSI 대상 피처를 순회하며 100 * (환측 / 건측) 값을 계산한다.
        for feat in LSI_FEATURES:
            # 현재 피처의 환측 값을 가져온다(없으면 NaN).
            injured_value = rec.get(f"{feat}_injured", np.nan)
            # 현재 피처의 건측 값을 가져온다(없으면 NaN).
            contra_value = rec.get(f"{feat}_contralateral", np.nan)

            # 건측 값이 유효하고 0에 매우 가깝지 않을 때만 LSI를 계산한다.
            if contra_value is not None and not np.isnan(contra_value) and abs(contra_value) > 1e-6:
                rec[f"{feat}_LSI"] = 100.0 * (injured_value / contra_value)
            # 그 외에는 분모 불안정/결측으로 간주하고 NaN 처리한다.
            else:
                rec[f"{feat}_LSI"] = np.nan

        # 한 속도 조건의 평균 피처 레코드를 결과 리스트에 추가한다.
        records.append(rec)

    # 평균 피처 레코드와 개별 피크 레코드를 함께 반환한다.
    return records, peak_rows


def extract_all_features(joint_df: pd.DataFrame, matched: pd.DataFrame, healthy: pd.DataFrame, limits: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """전체 피험자의 평균 피처 테이블과 개별 피크 테이블을 함께 생성한다."""
    # 모든 피험자/속도의 평균 피처 레코드를 누적할 리스트다.
    all_records = []
    # 모든 피험자/속도의 개별 피크 레코드를 누적할 리스트다.
    all_peak_rows = []

    # ACLD와 ACLR 그룹을 순회하며 matched 테이블의 대응 ID 컬럼을 사용한다.
    for group_label, id_col in [
        ("ACLD", "ID_ACLD"),
        ("ACLR", "ID_ACLR"),
    ]:
        # 매칭된 각 행(동일 조건의 ACLD/ACLR 쌍)을 순회한다.
        for _, row in matched.iterrows():
            # 현재 그룹에 해당하는 subject_id와 손상측 정보를 가져온다.
            sub_id, injured_leg = (
                row[id_col],
                row["Injured leg"],
            )

            # 전체 관절 데이터에서 현재 피험자의 시계열만 필터링한다.
            sub_df = joint_df[joint_df["subject_id"] == sub_id]

            # 데이터가 비어 있지 않으면 speed별 피처를 계산해 누적한다.
            if not sub_df.empty:
                recs, peak_rows = compute_features_for_subject(sub_df, sub_id, group_label, injured_leg, limits)
                all_records.extend(recs)
                all_peak_rows.extend(peak_rows)

    # Healthy 피험자 목록을 순회한다.
    for _, row in healthy.iterrows():
        # Healthy ID 컬럼에서 subject_id를 꺼낸다.
        sub_id = row["ID_HA"]
        # 전체 관절 데이터에서 해당 Healthy 피험자 데이터만 필터링한다.
        sub_df = joint_df[joint_df["subject_id"] == sub_id]

        # 데이터가 비어 있지 않으면 동일 함수로 피처를 계산해 누적한다.
        if not sub_df.empty:
            recs, peak_rows = (
                # Healthy는 손상측 개념이 없어 관례적으로 "Right"를 전달한다.
                compute_features_for_subject(sub_df, sub_id, "Healthy", "Right", limits)
            )
            all_records.extend(recs)
            all_peak_rows.extend(peak_rows)

    # 누적 결과를 DataFrame으로 변환해 (평균 피처, 개별 피크) 순으로 반환한다.
    return pd.DataFrame(all_records), pd.DataFrame(all_peak_rows)


def load_and_merge_spatiotemporal(kinematic_df: pd.DataFrame, path_gait: str) -> pd.DataFrame:
    """시공간 보행 지표를 로드해 운동학 피처 테이블과 병합한다."""
    # 보행 분석 글로벌 CSV를 읽는다.
    gait = pd.read_csv(path_gait)
    # pace_condition 문자열을 소문자+trim 처리해 speed 키와 일치시킨다.
    gait["pace_condition"] = gait["pace_condition"].str.lower().str.strip()
    # 병합 키를 맞추기 위해 participant->subject_id, pace_condition->speed로 rename한다.
    gait = gait.rename(columns={"participant": "subject_id", "pace_condition": "speed"})

    # 별칭 subject_id가 몇 행인지 계산한다.
    alias_hits = int(gait["subject_id"].isin(SUBJECT_ID_ALIASES.keys()).sum())

    # 별칭이 있으면 canonical ID로 치환해 raw 파트와 동일한 기준으로 맞춘다.
    if alias_hits > 0:
        gait["subject_id"] = gait["subject_id"].replace(SUBJECT_ID_ALIASES)
        log.info(f"ℹ subject_id 별칭 치환 적용 (spatio): {alias_hits} rows")

    # subject_id/speed 단위로 시공간 지표 평균을 집계한다.
    spatio = gait.groupby(["subject_id", "speed"])[SPATIOTEMPORAL_COLS].mean().reset_index()
    # 운동학 피처 테이블과 left join해 없는 시공간 값은 NaN으로 유지한다.
    return pd.merge(kinematic_df, spatio, on=["subject_id", "speed"], how="left")


def run_preprocessing(
    distance_val=50,
    prominence_val=1,
    iqr_upper=2.5,
    iqr_lower=1.5,
    peak_method="argextrema",
    butter_order=2,
    butter_cutoff=0.1,
):
    """전처리 파이프라인 전체를 실행하고 결과 DataFrame을 반환한다."""
    # 파이프라인 시작 로그를 출력한다.
    log.info("▶ 전처리 파이프라인 시작 (분리된 모듈)")

    # 피크 추출에 필요한 하이퍼파라미터를 한 dict로 구성한다.
    limits = {
        "distance": distance_val,
        "prominence": prominence_val,
        "iqr_lower_bound": iqr_lower,
        "iqr_upper_bound": iqr_upper,
        "peak_method": peak_method,
        "butter_order": butter_order,
        "butter_cutoff": butter_cutoff,
    }

    # ID 메타를 기반으로 ACLD/ACLR 매칭과 Healthy 목록을 만든다.
    matched, healthy = match_subjects(PATH_ID)
    # 실제 로드 대상 피험자 ID 목록을 ACLD+ACLR+Healthy 순으로 합친다.
    all_ids = matched["ID_ACLD"].tolist() + matched["ID_ACLR"].tolist() + healthy["ID_HA"].tolist()
    # JOINT_COLS 값(튜플)에서 필요한 원본 관절 컬럼명을 집합으로 펼쳐 추출한다.
    needed_cols = list({col for pair in JOINT_COLS.values() for col in pair})

    # 원시 Parquet에서 필요한 피험자/컬럼만 불러와 관절 시계열 테이블을 만든다.
    joint_df = load_joint_data(PATH_RAW, all_ids, needed_cols)
    # 관절 시계열로부터 (1) 평균 피크 피처, (2) 개별 피크 레코드를 함께 계산한다.
    kinematic_df, peak_df = extract_all_features(joint_df, matched, healthy, limits)
    # 계산된 운동학 피처에 시공간 지표를 병합한다.
    full_df = load_and_merge_spatiotemporal(kinematic_df, PATH_GAIT_GLOBAL)

    # 최종 결과를 UTF-8 BOM 포함 CSV로 저장해 엑셀 호환성을 높인다.
    full_df.to_csv(PATH_OUT, index=False, encoding="utf-8-sig")
    # 개별 피크 레코드도 CSV로 저장해 시각화/검증 시 바로 사용할 수 있게 한다.
    peak_df.to_csv(PATH_PEAKS, index=False, encoding="utf-8-sig")
    # 저장 완료 로그를 출력한다.
    log.info(f"✅ 전처리 완료. 결과 저장됨: {PATH_OUT}")
    log.info(f"✅ 피크 레코드 저장됨: {PATH_PEAKS} (rows={len(peak_df)})")
    # 후속 분석에서 재사용할 수 있도록 DataFrame을 반환한다.
    return full_df


# 스크립트 직접 실행 시 전처리 파이프라인을 즉시 수행한다.
run_preprocessing()
