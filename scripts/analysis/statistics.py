import pandas as pd
import numpy as np
import os
import warnings
import logging
from itertools import combinations

from scipy import stats
from scipy.stats import shapiro, levene, kruskal, pointbiserialr
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if not os.path.exists(os.path.join(BASE_DIR, "mds")):
    BASE_DIR = os.getcwd()
MDS_DIR = os.path.join(BASE_DIR, "mds")
PATH_DATA = os.path.join(MDS_DIR, "analysis_data.csv")
PATH_STATS = os.path.join(MDS_DIR, "stats_result.csv")
PATH_CORR = os.path.join(MDS_DIR, "correlation_result.csv")

ALPHA = 0.05

KINEMATIC_PEAKS = [
    "hip_flexion_injured",
    "hip_adduction_injured",
    "hip_int_rotation_injured",
    "knee_flexion_injured",
    "knee_adduction_injured",
    "knee_int_rotation_injured",
    "ankle_dorsiflexion_injured",
    "ankle_adduction_injured",
    "ankle_int_rotation_injured",
    "hip_flexion_contralateral",
    "hip_adduction_contralateral",
    "hip_int_rotation_contralateral",
    "knee_flexion_contralateral",
    "knee_adduction_contralateral",
    "knee_int_rotation_contralateral",
    "ankle_dorsiflexion_contralateral",
    "ankle_adduction_contralateral",
    "ankle_int_rotation_contralateral",
]
LSI_FEATURES = [
    "hip_flexion_LSI",
    "hip_adduction_LSI",
    "hip_int_rotation_LSI",
    "knee_flexion_LSI",
    "knee_adduction_LSI",
    "knee_int_rotation_LSI",
    "ankle_dorsiflexion_LSI",
    "ankle_adduction_LSI",
    "ankle_int_rotation_LSI",
]
SPATIOTEMPORAL = [
    "gait_speed_mps",
    "cadence_spm",
    "stride_length_mean_m",
    "step_width_mean_m_orth",
    "double_support_pct",
    "single_support_L_pct",
    "single_support_R_pct",
]
ALL_FEATURES = KINEMATIC_PEAKS + LSI_FEATURES + SPATIOTEMPORAL


# 부분 에타 제곱 (Partial Eta Squared) 계산 함수 (효과 크기)
def partial_eta_squared(df: pd.DataFrame, feature: str) -> float:
    # 전체 평균 계산
    grand_mean = df[feature].mean()
    # 그룹별 데이터 추출
    groups = [g[feature].values for _, g in df.groupby("group")]
    # 그룹 간 제곱합 (SS_between) 계산
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    # 총 제곱합 (SS_total) 계산
    ss_total = sum((x - grand_mean) ** 2 for g in groups for x in g)
    return (
        round(ss_between / ss_total, 4) if ss_total > 0 else np.nan
    )  # SS_between / SS_total 반환, 0으로 나누는 경우 방지


# 엡실론 제곱 (Epsilon Squared) 계산 함수 (크루스칼-월리스의 효과 크기)
def epsilon_squared(h_stat: float, n: int, k: int = 3) -> float:
    # (H - k + 1) / (n - k) 공식 적용, 0으로 나누는 경우 방지
    return round((h_stat - k + 1) / (n - k), 4) if (n - k) > 0 else np.nan


# 예비 검정 (정규성, 등분산성) 실행 함수
def run_preliminary_tests(df: pd.DataFrame, feature: str) -> dict:
    # 그룹별 피처 데이터 추출 (NaN 값 제거)
    groups = [g[feature].dropna().values for _, g in df.groupby("group")]
    # 각 그룹의 샤피로-윌크 검정 (정규성) p-value가 ALPHA보다 큰지 확인
    is_normal = all((shapiro(g)[1] > ALPHA if len(g) >= 3 else False) for g in groups)
    # 레빈 검정 (등분산성) p-value 추출
    _, p_lev = levene(*groups)
    return {
        "is_normal": is_normal,
        "is_equal_var": (p_lev > ALPHA),
    }  # 정규성 및 등분산성 결과 반환


# 터키의 HSD 사후 분석 실행 함수
def run_post_hoc_tukey(df: pd.DataFrame, feature: str) -> dict:
    # 결과를 저장할 딕셔너리 초기화
    result = {}
    try:  # 터키의 HSD 검정 실행
        tukey = pairwise_tukeyhsd(df[feature], df["group"])
        # 검정 결과를 데이터프레임으로 변환
        t_df = pd.DataFrame(
            data=tukey._results_table.data[1:], columns=tukey._results_table.data[0]
        )
        # 각 그룹 쌍에 대한 조정된 p-value 저장
        for _, r in t_df.iterrows():
            result[f"tukey_p_{r['group1']}_vs_{r['group2']}"] = round(
                float(r["p-adj"]), 4
            )
    except Exception as e:  # 오류 발생 시 오류 메시지 저장
        result["tukey_error"] = str(e)
    return result  # 결과 딕셔너리 반환


# 던(Dunn)의 사후 분석 실행 함수 (크루스칼-월리스 후 비모수 사후 분석)
def run_post_hoc_dunn(df: pd.DataFrame, feature: str) -> dict:
    # 결과를 저장할 딕셔너리 초기화
    result = {}
    # 고유한 그룹 목록 추출
    groups = df["group"].unique().tolist()
    # 총 그룹 쌍의 수 계산 (본페로니 보정용)
    n_pairs = len(list(combinations(groups, 2)))
    # 모든 그룹 쌍에 대해 반복
    for g1, g2 in combinations(groups, 2):
        # 각 그룹의 피처 데이터 추출
        a = df[df["group"] == g1][feature].dropna().values
        b = df[df["group"] == g2][feature].dropna().values
        # 샘플 수가 너무 적으면 NaN 반환
        if len(a) < 2 or len(b) < 2:
            result[f"dunn_p_{g1}_vs_{g2}"] = np.nan
            continue
        # Mann-Whitney U 검정 실행 (비모수 비교)
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        # 본페로니 보정 적용 후 p-value 저장
        result[f"dunn_p_{g1}_vs_{g2}"] = round(min(p * n_pairs, 1.0), 4)
    return result  # 결과 딕셔너리 반환


# 단일 피처에 대한 통계 분석 실행 함수
def analyze_feature(df: pd.DataFrame, feature: str) -> dict:
    # 결과를 저장할 딕셔너리 초기화
    result = {"feature": feature}
    # 필요한 컬럼만 선택하고 NaN 값 제거
    sub_df = df[["subject_id", "group", "speed", feature]].dropna()
    # 샘플 수가 9개 미만이면 분석하지 않고 반환
    if len(sub_df) < 9:
        result["note"] = "샘플 부족 (<9)"
        return result

    # 예비 검정 실행 (정규성, 등분산성)
    pre = run_preliminary_tests(sub_df, feature)
    # 예비 검정 결과 업데이트
    result.update(
        {"shapiro_normal": pre["is_normal"], "levene_equal_var": pre["is_equal_var"]}
    )

    # 정규성과 등분산성을 만족하면 혼합 효과 모델 (LMM) 사용
    if pre["is_normal"] and pre["is_equal_var"]:
        result["model"] = "LMM"
        try:  # 피처 이름에 하이픈이 있으면 언더스코어로 변경 (모델 공식에 사용하기 위함)
            clean_feat = feature.replace("-", "_")
            sub_df2 = sub_df.rename(columns={feature: clean_feat})
            # 혼합 효과 모델 정의: 그룹과 속도 간의 상호작용 효과, subject_id를 랜덤 효과로
            model = mixedlm(
                f"{clean_feat} ~ C(group) * C(speed)",
                sub_df2,
                groups=sub_df2["subject_id"],
            )
            # 모델 적합 (REML 추정, L-BFGS 최적화)
            fitted = model.fit(reml=True, method="lbfgs", disp=False)

            # 그룹 주 효과의 p-value 추출
            group_keys = [
                k
                for k in fitted.pvalues.index
                if "group" in k.lower() and "speed" not in k.lower()
            ]
            result["p_group_main"] = (
                round(float(np.min([fitted.pvalues[k] for k in group_keys])), 4)
                if group_keys
                else np.nan
            )

            # 상호작용 효과의 p-value 추출
            int_keys = [
                k
                for k in fitted.pvalues.index
                if "group" in k.lower() and "speed" in k.lower()
            ]
            result["p_interaction"] = (
                round(float(np.min([fitted.pvalues[k] for k in int_keys])), 4)
                if int_keys
                else np.nan
            )

            # 효과 크기 (부분 에타 제곱) 계산 및 업데이트
            result.update(
                {
                    "effect_metric": "partial_eta_sq",
                    "effect_size": partial_eta_squared(sub_df, feature),
                }
            )
        except Exception as e:  # LMM 실행 중 오류 발생 시
            result.update({"lmm_error": str(e), "model": "LMM_FAILED"})

        # LMM 후 터키의 HSD 사후 분석 실행
        result.update(run_post_hoc_tukey(sub_df, feature))
    else:  # 정규성 또는 등분산성을 만족하지 않으면 크루스칼-월리스 검정 사용
        result["model"] = "Kruskal-Wallis"
        # 그룹별 데이터 추출
        groups_data = [g[feature].dropna().values for _, g in sub_df.groupby("group")]
        # 크루스칼-월리스 검정 실행
        h_stat, p_kw = kruskal(*groups_data)
        # p-value와 효과 크기 (엡실론 제곱) 업데이트
        result.update(
            {
                "p_group_main": round(p_kw, 4),
                "effect_metric": "epsilon_sq",
                "effect_size": epsilon_squared(h_stat, len(sub_df)),
            }
        )
        # 크루스칼-월리스 후 던(Dunn)의 사후 분석 실행
        result.update(run_post_hoc_dunn(sub_df, feature))
    return result  # 최종 결과 딕셔너리 반환


# 상관관계 분석 실행 함수
def run_correlation_analysis(df: pd.DataFrame, features: list) -> pd.DataFrame:
    # 데이터프레임에 존재하는 피처 컬럼만 선택
    feat_cols = [f for f in features if f in df.columns]
    # subject_id별로 피처의 평균과 그룹 정보를 집계
    agg = (
        df.groupby("subject_id")[feat_cols + ["group"]]
        .agg({**{f: "mean" for f in feat_cols}, "group": "first"})
        .reset_index()
    )

    # StandardScaler 초기화
    scaler = StandardScaler()
    # 피처 컬럼의 결측치를 평균으로 채우고 표준화
    agg[feat_cols] = scaler.fit_transform(agg[feat_cols].fillna(agg[feat_cols].mean()))

    # 그룹별 더미 변수 생성 (예: is_ACLD, is_ACLR, is_Healthy)
    for grp in ["ACLD", "ACLR", "Healthy"]:
        agg[f"is_{grp}"] = (agg["group"] == grp).astype(int)

    records = []  # 결과를 저장할 리스트
    for feat in feat_cols:  # 각 피처에 대해 반복
        rec = {"feature": feat}
        for dummy in ["is_ACLD", "is_ACLR", "is_Healthy"]:  # 각 더미 변수에 대해 반복
            # 점이분 상관계수 (point-biserial correlation) 계산
            r, p = pointbiserialr(agg[dummy], agg[feat])
            # 유의 수준에 따른 별표 표시
            sig = (
                "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            )
            # 결과 딕셔너리 업데이트
            rec.update(
                {
                    f"r_{dummy}": round(r, 4),
                    f"p_{dummy}": round(p, 4),
                    f"sig_{dummy}": sig,
                }
            )
        records.append(rec)  # 레코드 추가
    return pd.DataFrame(records)  # 결과 데이터프레임 반환


# 통계 분석 파이프라인 실행 메인 함수
def run_statistics():
    # 파이프라인 시작 로깅
    log.info("▶ 통계 분석 파이프라인 시작 (분리된 모듈)")
    # 데이터 파일 존재 여부 확인
    if not os.path.exists(PATH_DATA):
        log.error(f"데이터 파일 누락: {PATH_DATA}")
        return

    # 데이터 파일 로드
    df = pd.read_csv(PATH_DATA)
    # 모든 피처에 대해 analyze_feature 함수 실행
    stat_records = [
        analyze_feature(df, feat) for feat in ALL_FEATURES if feat in df.columns
    ]
    # 통계 결과 데이터프레임 생성
    stats_df = pd.DataFrame(stat_records)
    # 통계 결과 CSV 파일로 저장
    stats_df.to_csv(PATH_STATS, index=False, encoding="utf-8-sig")
    log.info(f"✅ 통계 분석 완료. 결과 저장됨: {PATH_STATS}")

    # 상관관계 분석 실행
    corr_df = run_correlation_analysis(df, ALL_FEATURES)
    # 상관관계 결과 CSV 파일로 저장
    corr_df.to_csv(PATH_CORR, index=False, encoding="utf-8-sig")
    log.info(f"✅ 상관관계 분석 완료. 결과 저장됨: {PATH_CORR}")

    return stats_df, corr_df  # 통계 및 상관관계 데이터프레임 반환
