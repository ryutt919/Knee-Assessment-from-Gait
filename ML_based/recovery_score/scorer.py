"""
Recovery Score 계산기.
5개 컴포넌트를 균등 가중치(기본) 또는 SHAP 기반 가중치로 합산.
score = 100 - sigmoid(weighted_sum) × 100  (100 = 완전 정상)
단조성 검증: ACLD_mean < ACLR_mean < HA_mean 를 확인한다.
"""
from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .components import (
    waveform_deviation,
    lsi_asymmetry,
    compensation_strategy,
    spatiotemporal_deviation,
)

COMPONENT_NAMES = [
    "waveform_dev",
    "lsi_asym",
    "compensation",
    "spatiotemporal",
]

DEFAULT_WEIGHTS = {name: 1.0 / len(COMPONENT_NAMES) for name in COMPONENT_NAMES}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class RecoveryScorer:
    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self._components: dict[str, np.ndarray] = {}
        self._groups: np.ndarray | None = None

    def compute(
        self,
        features_df: pd.DataFrame,
        groups: np.ndarray,
        subject_ids: np.ndarray,
        waveforms: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        모든 컴포넌트를 계산하고 최종 Recovery Score를 반환.
        Returns DataFrame: subject_id, group, comp_*, recovery_score
        """
        self._groups = groups
        n = len(subject_ids)

        # 컴포넌트 계산
        comps = {}

        if waveforms is not None:
            comps["waveform_dev"] = waveform_deviation(subject_ids, groups, waveforms)
        else:
            comps["waveform_dev"] = np.zeros(n)

        comps["lsi_asym"] = lsi_asymmetry(features_df, groups, subject_ids)
        comps["compensation"] = compensation_strategy(features_df, groups)
        comps["spatiotemporal"] = spatiotemporal_deviation(features_df, groups)

        self._components = comps

        # 가중 합산
        w_sum = np.zeros(n)
        total_w = sum(self.weights.get(k, 0.0) for k in comps)
        if total_w < 1e-9:
            total_w = 1.0

        for name, vals in comps.items():
            w = self.weights.get(name, 0.0) / total_w
            w_sum += w * vals

        # 0~100 매핑 (높을수록 정상에 가까움)
        scores = 100.0 - _sigmoid(w_sum) * 100.0

        result = pd.DataFrame({
            "subject_id":     subject_ids,
            "group":          groups,
            **{f"comp_{k}": v for k, v in comps.items()},
            "recovery_score": scores,
        })

        self._validate_monotonicity(result)
        return result

    def update_weights_from_shap(self, shap_importance: dict[str, float]) -> None:
        """
        SHAP feature importance를 컴포넌트 그룹별로 합산해 가중치 재조정.
        shap_importance: {feature_name: mean_abs_shap_value}
        """
        component_keywords = {
            "waveform_dev":   ["waveform", "wave", "rmsd"],
            "lsi_asym":       ["lsi", "asym"],
            "compensation":   ["hip", "ankle", "comp"],
            "spatiotemporal": ["step", "cadence", "stride", "stance", "swing", "velocity"],
        }

        group_sums: dict[str, float] = {k: 0.0 for k in component_keywords}
        for feat, imp in shap_importance.items():
            feat_l = feat.lower()
            for comp, keywords in component_keywords.items():
                if any(kw in feat_l for kw in keywords):
                    group_sums[comp] += imp
                    break

        total = sum(group_sums.values())
        if total < 1e-9:
            warnings.warn("SHAP 기반 가중치 계산 실패: feature 이름이 컴포넌트와 매칭되지 않음")
            return

        self.weights = {k: v / total for k, v in group_sums.items()}
        print("[scorer] SHAP 기반 가중치 업데이트:", self.weights)

    def _validate_monotonicity(self, result: pd.DataFrame) -> None:
        """ACLD_mean < ACLR_mean < HA_mean 단조성 검증."""
        means = result.groupby("group")["recovery_score"].mean()
        groups_present = set(means.index)

        if {"ACLD", "ACLR", "HA"} <= groups_present:
            ok = means["ACLD"] < means["ACLR"] < means["HA"]
            if not ok:
                warnings.warn(
                    f"[scorer] 단조성 위반 — "
                    f"ACLD={means['ACLD']:.1f}, ACLR={means['ACLR']:.1f}, HA={means['HA']:.1f}"
                )
            else:
                print(f"[scorer] 단조성 검증 통과 — "
                      f"ACLD={means['ACLD']:.1f}, ACLR={means['ACLR']:.1f}, HA={means['HA']:.1f}")
        elif {"ACL", "HA"} <= groups_present:
            ok = means["ACL"] < means["HA"]
            if not ok:
                warnings.warn("[scorer] 단조성 위반 — ACL > HA")
