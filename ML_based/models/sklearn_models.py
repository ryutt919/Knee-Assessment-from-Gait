"""
sklearn/boosting 모델 래퍼 — BaseModel 인터페이스 구현
모델: LogReg, LinearSVC, SVM-RBF, RF, GBT, XGBoost, LightGBM
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

from .base import BaseModel


class LogReg(BaseModel):
    name = "logreg"

    def __init__(self, C=1.0, penalty="l2", l1_ratio=None, max_iter=1000,
                 class_weight="balanced", **kw):
        kwargs = dict(C=C, penalty=penalty, max_iter=max_iter,
                      class_weight=class_weight, solver="saga", random_state=42)
        if penalty == "elasticnet" and l1_ratio is not None:
            kwargs["l1_ratio"] = l1_ratio
        self.model = LogisticRegression(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y); return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class LinearSVCModel(BaseModel):
    name = "linearsvc"

    def __init__(self, C=1.0, class_weight="balanced", max_iter=2000, **kw):
        base = LinearSVC(C=C, class_weight=class_weight, max_iter=max_iter, random_state=42)
        self.model = CalibratedClassifierCV(base, cv=3)

    def fit(self, X, y):
        self.model.fit(X, y); return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class SVMRBF(BaseModel):
    name = "svm_rbf"

    def __init__(self, C=1.0, gamma="scale", class_weight="balanced", **kw):
        self.model = SVC(C=C, gamma=gamma, class_weight=class_weight,
                         kernel="rbf", probability=True, random_state=42)

    def fit(self, X, y):
        self.model.fit(X, y); return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class RandomForest(BaseModel):
    name = "rf"

    def __init__(self, n_estimators=300, max_depth=None, max_features="sqrt",
                 min_samples_split=2, min_samples_leaf=1,
                 class_weight="balanced", n_jobs=-1, **kw):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, max_features=max_features,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            class_weight=class_weight, n_jobs=n_jobs, random_state=42,
        )

    def fit(self, X, y):
        self.model.fit(X, y); return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class GBT(BaseModel):
    name = "gbt"

    def __init__(self, n_estimators=200, learning_rate=0.05, max_depth=3,
                 subsample=0.8, min_samples_leaf=5, **kw):
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, subsample=subsample,
            min_samples_leaf=min_samples_leaf, random_state=42,
        )

    def fit(self, X, y):
        self.model.fit(X, y); return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class XGBoost(BaseModel):
    name = "xgboost"

    def __init__(self, n_estimators=300, learning_rate=0.05, max_depth=3,
                 subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                 early_stopping_rounds=50, n_jobs=-1, **kw):
        if not _HAS_XGB:
            raise ImportError("xgboost 설치 필요: pip install xgboost")
        self._es_rounds = early_stopping_rounds
        self.model = XGBClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, subsample=subsample,
            colsample_bytree=colsample_bytree, min_child_weight=min_child_weight,
            n_jobs=n_jobs, random_state=42,
            eval_metric="logloss", use_label_encoder=False,
        )

    def fit(self, X, y, eval_set=None):
        fit_kw = {}
        if eval_set and self._es_rounds:
            fit_kw["eval_set"] = eval_set
            fit_kw["early_stopping_rounds"] = self._es_rounds
            fit_kw["verbose"] = False
        self.model.fit(X, y, **fit_kw)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class LightGBM(BaseModel):
    name = "lightgbm"

    def __init__(self, n_estimators=300, learning_rate=0.05, num_leaves=63,
                 subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                 early_stopping_rounds=50, n_jobs=-1, verbose=-1, **kw):
        if not _HAS_LGB:
            raise ImportError("lightgbm 설치 필요: pip install lightgbm")
        self._es_rounds = early_stopping_rounds
        self.model = LGBMClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate,
            num_leaves=num_leaves, subsample=subsample,
            colsample_bytree=colsample_bytree, min_child_samples=min_child_samples,
            n_jobs=n_jobs, random_state=42, verbose=verbose,
        )

    def fit(self, X, y, eval_set=None):
        fit_kw = {"callbacks": []}
        if eval_set and self._es_rounds:
            import lightgbm as lgb
            fit_kw["eval_set"] = eval_set
            fit_kw["callbacks"].append(lgb.early_stopping(self._es_rounds, verbose=False))
            fit_kw["callbacks"].append(lgb.log_evaluation(-1))
        self.model.fit(X, y, **fit_kw)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
