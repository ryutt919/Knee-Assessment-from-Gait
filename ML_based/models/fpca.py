"""
FPCA (Functional PCA) + sklearn classifier
입력: (N, 9, 101) norm_101 파형
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    from skfda import FDataGrid
    from skfda.preprocessing.dim_reduction import FPCA
    _HAS_SKFDA = True
except ImportError:
    _HAS_SKFDA = False

from .base import BaseModel


class FPCAClassifier(BaseModel):
    name = "fpca"

    def __init__(self, n_components: int = 50, clf_C: float = 1.0, **kw):
        if not _HAS_SKFDA:
            raise ImportError("skfda 설치 필요: pip install scikit-fda")
        self.n_components = n_components
        self.clf_C = clf_C
        self._fpca_list: list[FPCA] = []
        self._scaler = StandardScaler()
        self._clf = LogisticRegression(C=clf_C, max_iter=1000, random_state=42,
                                       class_weight="balanced")

    def _to_fd(self, X: np.ndarray, channel: int) -> FDataGrid:
        data = X[:, channel, :]  # (N, 101)
        grid = np.linspace(0, 1, data.shape[1])
        return FDataGrid(data_matrix=data[:, :, np.newaxis], grid_points=grid)

    def _fit_transform(self, X: np.ndarray) -> np.ndarray:
        parts = []
        for ch in range(X.shape[1]):
            fd = self._to_fd(X, ch)
            n_time = X.shape[2]  # 101
            n_comp = min(self.n_components, X.shape[0] - 1, n_time - 1)
            fpca = FPCA(n_components=n_comp)
            scores = fpca.fit_transform(fd)
            self._fpca_list.append(fpca)
            parts.append(scores)
        return np.hstack(parts)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        parts = []
        for ch, fpca in enumerate(self._fpca_list):
            fd = self._to_fd(X, ch)
            scores = fpca.transform(fd)
            parts.append(scores)
        return np.hstack(parts)

    def fit(self, X: np.ndarray, y: np.ndarray, **kw):
        self._fpca_list = []
        scores = self._fit_transform(X)
        scores_s = self._scaler.fit_transform(scores)
        self._clf.fit(scores_s, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self._transform(X)
        return self._clf.predict(self._scaler.transform(scores))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self._transform(X)
        return self._clf.predict_proba(self._scaler.transform(scores))
