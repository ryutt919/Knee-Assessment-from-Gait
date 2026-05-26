"""모든 모델이 구현해야 하는 인터페이스."""
from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kw) -> "BaseModel":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ...
