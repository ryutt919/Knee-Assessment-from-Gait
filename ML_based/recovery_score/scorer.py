"""GDI/GPS-inspired normality scoring against healthy gait waveforms."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .components import EPSILON, raw_distances, top_waveform_regions

SCORE_KEYS = (
    "overall",
    "hip",
    "knee",
    "ankle",
    "slow",
    "normal",
    "fast",
    "bilateral",
    "asymmetry",
)


class GaitNormalityScorer:
    """
    Score trial-balanced session waveforms against an HA reference.

    Scientific outputs are not clipped. A normality score of 100 is the
    leave-one-out HA mean and each 10-point decrease is one HA SD farther away.
    """

    def __init__(self) -> None:
        self.reference_mean: np.ndarray | None = None
        self.calibration: dict[str, dict[str, float]] = {}
        self.reference_subject_ids: list[str] = []
        self.mahalanobis_location: np.ndarray | None = None
        self.mahalanobis_precision: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self.reference_mean is not None and bool(self.calibration)

    def fit(
        self,
        matrices: np.ndarray,
        groups: np.ndarray | pd.Series,
        subject_ids: np.ndarray | pd.Series | None = None,
    ) -> "GaitNormalityScorer":
        matrices = np.asarray(matrices, dtype=np.float64)
        groups = np.asarray(groups, dtype=str)
        if len(matrices) != len(groups):
            raise ValueError("matrices and groups must have the same length")
        ha_indices = np.flatnonzero(groups == "HA")
        if len(ha_indices) < 5:
            raise ValueError(f"at least 5 HA reference sessions are required, found {len(ha_indices)}")
        references = matrices[ha_indices]
        self.reference_mean = references.mean(axis=0)
        if subject_ids is None:
            self.reference_subject_ids = [str(idx) for idx in ha_indices]
        else:
            ids = np.asarray(subject_ids, dtype=str)
            self.reference_subject_ids = ids[ha_indices].tolist()

        loo_values: dict[str, list[float]] = {key: [] for key in SCORE_KEYS}
        loo_log_gvs: list[np.ndarray] = []
        for reference_idx in range(len(references)):
            keep = np.arange(len(references)) != reference_idx
            loo_mean = references[keep].mean(axis=0)
            distances, gvs = raw_distances(references[reference_idx], loo_mean)
            for key in SCORE_KEYS:
                loo_values[key].append(float(distances[key][0]))
            loo_log_gvs.append(np.log(np.maximum(gvs[0].ravel(), EPSILON)))

        calibration: dict[str, dict[str, float]] = {}
        for key, values in loo_values.items():
            logged = np.log(np.maximum(np.asarray(values, dtype=np.float64), EPSILON))
            sd = float(logged.std(ddof=1))
            if not np.isfinite(sd) or sd < EPSILON:
                raise ValueError(f"HA leave-one-out calibration is degenerate for {key}")
            calibration[key] = {
                "mean_log_distance": float(logged.mean()),
                "sd_log_distance": sd,
                "n_reference": int(len(logged)),
            }
        self.calibration = calibration

        covariance = LedoitWolf().fit(np.vstack(loo_log_gvs))
        self.mahalanobis_location = covariance.location_.copy()
        self.mahalanobis_precision = covariance.precision_.copy()
        return self

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("GaitNormalityScorer.fit must be called before scoring")

    def score_matrices(self, matrices: np.ndarray) -> pd.DataFrame:
        self._require_fitted()
        assert self.reference_mean is not None
        matrices = np.asarray(matrices, dtype=np.float64)
        if matrices.ndim == 4:
            matrices = matrices[np.newaxis, ...]
        distances, gvs = raw_distances(matrices, self.reference_mean)
        output: dict[str, np.ndarray] = {}
        for key in SCORE_KEYS:
            raw = distances[key]
            parameters = self.calibration[key]
            z = (
                np.log(np.maximum(raw, EPSILON)) - parameters["mean_log_distance"]
            ) / parameters["sd_log_distance"]
            if key == "overall":
                output["raw_distance"] = raw
                output["z_deviation"] = z
                output["normality_score"] = 100.0 - 10.0 * z
            else:
                output[f"{key}_score"] = 100.0 - 10.0 * z
                output[f"{key}_z"] = z

        if self.mahalanobis_location is not None and self.mahalanobis_precision is not None:
            log_gvs = np.log(np.maximum(gvs.reshape(len(matrices), -1), EPSILON))
            centered = log_gvs - self.mahalanobis_location
            squared = np.einsum("ij,jk,ik->i", centered, self.mahalanobis_precision, centered)
            output["mahalanobis_distance"] = np.sqrt(np.maximum(squared, 0.0))
        result = pd.DataFrame(output)
        if not np.isfinite(result.to_numpy(dtype=float)).all():
            raise ValueError("normality scoring produced non-finite values")
        return result

    def gvs_features(self, matrices: np.ndarray) -> np.ndarray:
        """Log-GVS features used only by research comparators."""
        self._require_fitted()
        assert self.reference_mean is not None
        _, gvs = raw_distances(np.asarray(matrices, dtype=np.float64), self.reference_mean)
        return np.log(np.maximum(gvs.reshape(len(gvs), -1), EPSILON))

    def explain_matrix(self, matrix: np.ndarray, limit: int = 5) -> str:
        self._require_fitted()
        assert self.reference_mean is not None
        return top_waveform_regions(matrix, self.reference_mean, limit=limit)

    def to_dict(self) -> dict[str, Any]:
        self._require_fitted()
        assert self.reference_mean is not None
        return {
            "model_type": "GDI_GPS_HA_normative_waveform",
            "score_definition": "normality_score = 100 - 10 * z(log(GPS_distance))",
            "reference_subject_ids": self.reference_subject_ids,
            "reference_mean": self.reference_mean.tolist(),
            "calibration": self.calibration,
            "mahalanobis_location": (
                self.mahalanobis_location.tolist() if self.mahalanobis_location is not None else None
            ),
            "mahalanobis_precision": (
                self.mahalanobis_precision.tolist() if self.mahalanobis_precision is not None else None
            ),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "GaitNormalityScorer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        scorer = cls()
        scorer.reference_subject_ids = [str(value) for value in payload["reference_subject_ids"]]
        scorer.reference_mean = np.asarray(payload["reference_mean"], dtype=np.float64)
        scorer.calibration = payload["calibration"]
        if payload.get("mahalanobis_location") is not None:
            scorer.mahalanobis_location = np.asarray(payload["mahalanobis_location"], dtype=np.float64)
        if payload.get("mahalanobis_precision") is not None:
            scorer.mahalanobis_precision = np.asarray(payload["mahalanobis_precision"], dtype=np.float64)
        scorer._require_fitted()
        return scorer


# Import compatibility only. The former scalar-component compute API is intentionally removed.
RecoveryScorer = GaitNormalityScorer
