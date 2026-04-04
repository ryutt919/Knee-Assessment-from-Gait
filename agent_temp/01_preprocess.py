"""
01_preprocess.py
ACL 보행 생체역학 분석  전처리 파이프라인 (find_peaks 기반)

처리 흐름:
1. ID.csv → ACLR-ACLD 27명 매칭 (Sex, Age, Weight, Height, Injured leg)
2. raw_merged.parquet → PyArrow Lazy Loading (필요 피험자 × 컬럼만 추출)
3. Trial 내 연속 시계열에서 find_peaks로 보행 주기별 Local Peak 탐지 후 평균 산출
4. 환측/건측 기반 관절 각도 Peak 피처 및 LSI 비대칭 지수 계산
5. gait_analysis_global.csv 시공간 지표 병합
6. 최종 분석용 테이블 → mds/analysis_data.csv 저장

핵심 용어:
- Local Peak: 하나의 보행 주기에서 발생하는 극대/극소값
- Mean Peak: 해당 조건(피험자 × 속도) 내 모든 Local Peak의 평균값
- distance: 두 피크 사이에 보장되어야 하는 최소 샘플 수 (보행 주기 최소 간격)
- prominence: 피크가 주변 계곡 바닥으로부터 최소 이 값만큼 높아야 진짜 피크로 인정
"""

import pandas as pd
import numpy as np
import pyarrow.dataset as ds
import pyarrow.compute as pc
import os
import logging
from scipy.signal import find_peaks

# ───────────────────────────────────────────────
# 로깅 설정
# ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ───────────────────────────────────────────────
# 경로 설정
# ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "mds")
os.makedirs(OUT_DIR, exist_ok=True)

PATH_ID          = os.path.join(DATA_DIR, "ID.csv")
PATH_RAW         = os.path.join(DATA_DIR, "processed", "raw_merged.parquet")
PATH_GAIT_GLOBAL = os.path.join(DATA_DIR, "gait_analysis_global.csv")
PATH_OUT         = os.path.join(OUT_DIR, "analysis_data.csv")

# ───────────────────────────────────────────────
# 분석 피처 정의
# ───────────────────────────────────────────────
# (Right 컬럼, Left 컬럼) 튜플
JOINT_COLS = {
    "hip_flexion"        : ("jointAngle_42", "jointAngle_54"),
    "hip_adduction"      : ("jointAngle_43", "jointAngle_55"),
    "knee_flexion"       : ("jointAngle_45", "jointAngle_57"),
    "knee_adduction"     : ("jointAngle_46", "jointAngle_58"),
    "knee_int_rotation"  : ("jointAngle_47", "jointAngle_59"),
    "ankle_dorsiflexion" : ("jointAngle_48", "jointAngle_60"),
}

# 추출 방향: "max" → Local Maxima, "min" → Local Minima (신호를 뒤집어 find_peaks 적용)
PEAK_DIRECTION = {
    "hip_flexion"        : "max",
    "hip_adduction"      : "min",
    "knee_flexion"       : "max",
    "knee_adduction"     : "min",
    "knee_int_rotation"  : "max",
    "ankle_dorsiflexion" : "max",
}

# LSI를 산출할 피처 목록 (환측과 건측 굴곡 비교)
LSI_FEATURES = ["hip_flexion", "knee_flexion", "ankle_dorsiflexion"]

# 시공간 피처 7종
SPATIOTEMPORAL_COLS = [
    "gait_speed_mps", "cadence_spm", "stride_length_mean_m",
    "step_width_mean_m_orth", "double_support_pct",
    "single_support_L_pct", "single_support_R_pct"
]

# ───────────────────────────────────────────────
# find_peaks 파라미터
# 데이터: 100Hz 기준 (dt 중앙값 = 10ms)
# 정상 보행 주기: 약 1.0~1.2초 → 최소 피크 간격 0.5초 = 50 샘플
# prominence: 각도 신호의 실제 피크 돌출도 (라디안 기준 0.1 rad ≈ 5.7°)
# ───────────────────────────────────────────────
SAMPLING_HZ      = 100
MIN_STRIDE_SEC   = 0.5      # 정상 보행 주기 최소값의 절반 (초)
PEAK_DISTANCE    = int(SAMPLING_HZ * MIN_STRIDE_SEC)  # = 50 샘플
PEAK_PROMINENCE  = 0.05     # 라디안 기준 (탐색 후 조정 가능)


# ───────────────────────────────────────────────
# 핵심 함수: 시계열 신호에서 Mean Peak 산출
# ───────────────────────────────────────────────
def extract_mean_peak(signal: np.ndarray, direction: str,
                      distance: int = PEAK_DISTANCE,
                      prominence: float = PEAK_PROMINENCE) -> float:
    """
    연속 보행 시계열에서 보행 주기별 Local Peak들을 탐지하고 그 평균을 반환.

    Parameters
    ----------
    signal    : 1D np.ndarray — 관절 각도 연속 시계열
    direction : 'max' 또는 'min' — 극대(굴곡) 또는 극소(내전) 방향
    distance  : 두 피크 사이 최소 샘플 수 (보행 주기 최소 간격 보장)
    prominence: 피크가 주변 계곡 대비 최소 돌출 높이 (노이즈 제거)

    Returns
    -------
    float — 탐지된 모든 Local Peak 값들의 평균. 피크 미발견 시 np.nan 반환.
    """
    if len(signal) < distance * 2:
        return np.nan

    if direction == "max":
        peaks, _ = find_peaks(signal, distance=distance, prominence=prominence)
    else:
        # 극소값 탐지: 신호를 뒤집어 find_peaks 적용
        peaks, _ = find_peaks(-signal, distance=distance, prominence=prominence)

    if len(peaks) == 0:
        return np.nan

    return float(np.mean(signal[peaks]))


# ───────────────────────────────────────────────
# Step 1: ID.csv 기반 27명 피험자 매칭
# ───────────────────────────────────────────────
def match_subjects(path_id: str):
    """
    ACLR-ACLD 집단 간 Sex, Age, Weight, Height, Injured leg이
    완전히 일치하는 27명 subject_id와 Injured leg 정보를 반환.
    """
    log.info("ID.csv 로드 및 피험자 매칭 시작...")
    id_df = pd.read_csv(path_id)

    acld = id_df[id_df["Group"] == 3].copy()
    aclr = id_df[id_df["Group"] == 4].copy()

    # 피험자 번호 추출 (예: ACLD3 → 3)
    acld["num"] = acld["ID"].str.extract(r"(\d+)$").astype(int)
    aclr["num"] = aclr["ID"].str.extract(r"(\d+)$").astype(int)

    # 번호 + 인구통계 모두 일치하는 27명 Inner Join
    matched = pd.merge(
        acld[["num", "ID", "Sex", "Age", "Weight", "Height", "Injured leg"]],
        aclr[["num", "ID", "Sex", "Age", "Weight", "Height", "Injured leg"]],
        on=["num", "Sex", "Age", "Weight", "Height", "Injured leg"],
        suffixes=("_ACLD", "_ACLR")
    )
    log.info(f"  ACLD-ACLR 매칭 완료: {len(matched)}명")

    healthy = id_df[id_df["Group"] == 1][["ID"]].rename(columns={"ID": "ID_HA"})
    log.info(f"  Healthy 피험자: {len(healthy)}명")

    return matched, healthy


# ───────────────────────────────────────────────
# Step 2: PyArrow Lazy Loading — 필요 컬럼만 추출
# ───────────────────────────────────────────────
def load_joint_data(path_raw: str, subject_ids: list, needed_cols: list) -> pd.DataFrame:
    """
    raw_merged.parquet에서 대상 피험자의 time_ms + 관절 각도 컬럼만 Lazy Load.
    """
    log.info(f"raw_merged.parquet Lazy Loading (대상 {len(subject_ids)}명)...")

    select_cols = ["subject_id", "speed", "time_ms"] + needed_cols
    dataset = ds.dataset(path_raw, format="parquet")
    filter_expr = pc.field("subject_id").isin(subject_ids)

    table = dataset.to_table(columns=select_cols, filter=filter_expr)
    df = table.to_pandas()

    # time_ms → 숫자 변환 후 정렬
    df["time_ms"] = pd.to_numeric(df["time_ms"], errors="coerce")
    df = df.sort_values(["subject_id", "speed", "time_ms"]).reset_index(drop=True)

    log.info(f"  로드 완료: {df.shape}")
    return df


# ───────────────────────────────────────────────
# Step 3: 피험자 × 속도 조건별 Mean Peak 산출
# ───────────────────────────────────────────────
def compute_features_for_subject(sub_df: pd.DataFrame,
                                  subject_id: str,
                                  group_label: str,
                                  injured_leg: str) -> list:
    """
    한 명의 피험자에 대해 속도 조건(fast/normal/slow) × 피처별
    Mean Peak을 계산하고 레코드 리스트로 반환.

    injured_leg: 'Right' 또는 'Left'
    Healthy 그룹은 injured_leg='Right' 고정 (우측=환측 대체)
    """
    records = []

    for speed in ["fast", "normal", "slow"]:
        speed_df = sub_df[sub_df["speed"] == speed]

        # 속도 조건이 없는 경우 건너뜀
        if len(speed_df) < PEAK_DISTANCE * 2:
            log.debug(f"    {subject_id}/{speed}: 샘플 부족 ({len(speed_df)}개) — 건너뜀")
            continue

        rec = {
            "subject_id"  : subject_id,
            "group"       : group_label,
            "speed"       : speed,
            "injured_leg" : injured_leg,
        }

        for feat, (r_col, l_col) in JOINT_COLS.items():
            direction = PEAK_DIRECTION[feat]

            # 환측/건측 컬럼 배치
            if injured_leg == "Right":
                injured_col     = r_col
                contra_col      = l_col
            else:
                injured_col     = l_col
                contra_col      = r_col

            # 결측치 제거 후 numpy 배열 변환
            inj_signal   = speed_df[injured_col].dropna().values
            contra_signal= speed_df[contra_col].dropna().values

            rec[f"{feat}_injured"]       = extract_mean_peak(inj_signal, direction)
            rec[f"{feat}_contralateral"] = extract_mean_peak(contra_signal, direction)

        # LSI 계산: LSI = 100 × (환측 / 건측)
        for feat in LSI_FEATURES:
            I = rec.get(f"{feat}_injured", np.nan)
            C = rec.get(f"{feat}_contralateral", np.nan)
            if C is not None and not np.isnan(C) and abs(C) > 1e-6:
                rec[f"{feat}_LSI"] = 100.0 * (I / C)
            else:
                rec[f"{feat}_LSI"] = np.nan

        records.append(rec)

    return records


# ───────────────────────────────────────────────
# Step 4: 전체 그룹 순회 및 피처 취합
# ───────────────────────────────────────────────
def extract_all_features(joint_df: pd.DataFrame,
                          matched: pd.DataFrame,
                          healthy: pd.DataFrame) -> pd.DataFrame:
    """
    ACLD, ACLR, Healthy 그룹 전체 피험자의 Mean Peak 피처를 산출.
    """
    all_records = []

    # ACLD / ACLR 그룹
    for group_label, id_col in [("ACLD", "ID_ACLD"), ("ACLR", "ID_ACLR")]:
        log.info(f"[{group_label}] 피처 추출 중...")
        for _, row in matched.iterrows():
            sub_id      = row[id_col]
            injured_leg = row["Injured leg"]
            sub_df      = joint_df[joint_df["subject_id"] == sub_id]

            if sub_df.empty:
                log.warning(f"  [{group_label}] {sub_id}: 데이터 없음 — 건너뜀")
                continue

            recs = compute_features_for_subject(sub_df, sub_id, group_label, injured_leg)
            all_records.extend(recs)

    # Healthy 그룹 (Right=환측 고정)
    log.info("[Healthy] 피처 추출 중...")
    for _, row in healthy.iterrows():
        sub_id = row["ID_HA"]
        sub_df = joint_df[joint_df["subject_id"] == sub_id]

        if sub_df.empty:
            log.warning(f"  [Healthy] {sub_id}: 데이터 없음 — 건너뜀")
            continue

        recs = compute_features_for_subject(sub_df, sub_id, "Healthy", "Right")
        all_records.extend(recs)

    result = pd.DataFrame(all_records)
    log.info(f"운동학 피처 테이블 완성: {result.shape}")
    return result


# ───────────────────────────────────────────────
# Step 5: Spatiotemporal 병합
# ───────────────────────────────────────────────
def load_and_merge_spatiotemporal(kinematic_df: pd.DataFrame,
                                   path_gait: str) -> pd.DataFrame:
    """
    gait_analysis_global.csv를 로드하고 피험자 × 속도 조건별 평균을 병합.
    """
    log.info("gait_analysis_global.csv 로드 및 병합...")
    gait = pd.read_csv(path_gait)

    gait["pace_condition"] = gait["pace_condition"].str.lower().str.strip()
    gait = gait.rename(columns={"participant": "subject_id",
                                 "pace_condition": "speed"})

    # 같은 피험자 × 속도 내 여러 Trial의 평균값 사용
    spatio = gait.groupby(["subject_id", "speed"])[SPATIOTEMPORAL_COLS].mean().reset_index()

    merged = pd.merge(kinematic_df, spatio, on=["subject_id", "speed"], how="left")
    log.info(f"  최종 병합 완료: {merged.shape}")
    return merged


# ───────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────
def main():
    # Step 1: 피험자 매칭
    matched, healthy = match_subjects(PATH_ID)

    all_subject_ids = (
        matched["ID_ACLD"].tolist() +
        matched["ID_ACLR"].tolist() +
        healthy["ID_HA"].tolist()
    )
    needed_cols = list({col for pair in JOINT_COLS.values() for col in pair})

    # Step 2: Lazy Load
    joint_df = load_joint_data(PATH_RAW, all_subject_ids, needed_cols)

    # Step 3 & 4: 전체 피처 추출
    kinematic_df = extract_all_features(joint_df, matched, healthy)

    # Step 5: Spatiotemporal 병합
    full_df = load_and_merge_spatiotemporal(kinematic_df, PATH_GAIT_GLOBAL)

    # 저장
    full_df.to_csv(PATH_OUT, index=False, encoding="utf-8-sig")
    log.info(f"\n✅ 전처리 완료 → {PATH_OUT}")

    # 결과 요약
    log.info(f"\n그룹 × 속도 샘플 수:\n{full_df.groupby(['group', 'speed']).size().to_string()}")

    # 피처별 결측치 비율 간단 확인
    feat_cols = [c for c in full_df.columns if any(
        c.endswith(x) for x in ["_injured", "_contralateral", "_LSI"]
    )]
    missing = full_df[feat_cols].isnull().mean().round(3)
    log.info(f"\n피처별 결측 비율:\n{missing.to_string()}")


if __name__ == "__main__":
    main()
