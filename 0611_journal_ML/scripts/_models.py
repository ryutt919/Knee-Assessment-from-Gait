"""
_models.py — Shared model registry for the ACL benchmark (binary + multiclass).

True baseline:  logreg (Logistic Regression)
Benchmark set:  svm_rbf, rf, gbt, xgboost, lightgbm, catboost, tabpfn

Every model is tuned with Optuna (where it has hyperparameters) and trained
through the SAME pipeline as the rest of the script, so results are directly
comparable. No model is feature-capped:
  - tree / boosting models (rf, gbt, xgboost, lightgbm, catboost) take the full
    feature matrix natively;
  - TabPFN is run with ignore_pretraining_limits=True so it also accepts the
    full feature matrix (no artificial cap).

Usage (per fold)::

    from _models import ALL_MODELS, tune_and_build, is_available
    est, best_params, best_val = tune_and_build(
        "xgboost", X_tr, y_tr, inner_cv, n_trials=30, task="binary")
    est.fit(X_tr, y_tr)
    proba = est.predict_proba(X_te)        # binary: [:, 1]; multiclass: full matrix
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

import xgboost as xgb
import lightgbm as lgb

RANDOM_STATE = 42

# ── Optional models ────────────────────────────────────────────────────────────
try:
    from catboost import CatBoostClassifier
    _HAS_CATBOOST = True
except Exception:                       # pragma: no cover - import guard only
    _HAS_CATBOOST = False

try:
    from tabpfn import TabPFNClassifier
    _HAS_TABPFN = True
except Exception:                       # pragma: no cover - import guard only
    _HAS_TABPFN = False


def _tabpfn_runnable() -> bool:
    """TabPFN can run locally only if weights are obtainable: either a license
    token is set (TABPFN_TOKEN) or model weights are already cached on disk.
    Importable-but-no-weights → not runnable (would raise on .fit)."""
    import os
    from pathlib import Path
    if not _HAS_TABPFN:
        return False
    if os.environ.get("TABPFN_TOKEN"):
        return True
    cache_dirs = [Path.home() / ".cache" / "tabpfn",
                  Path.home() / "Library" / "Caches" / "tabpfn"]
    for d in cache_dirs:
        if d.exists() and any(d.rglob("*.ckpt")):
            return True
    return False


_TABPFN_RUNNABLE = _tabpfn_runnable()


# ── Registry metadata ───────────────────────────────────────────────────────────
BASELINE = "logreg"
ALL_MODELS = ["logreg", "svm_rbf", "rf", "gbt", "xgboost", "lightgbm", "catboost", "tabpfn"]

DISPLAY = {
    "logreg":   "Logistic Regression",
    "svm_rbf":  "SVM (RBF)",
    "rf":       "Random Forest",
    "gbt":      "Gradient Boosting",
    "xgboost":  "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "tabpfn":   "TabPFN",
}

ROLE = {m: ("baseline" if m == BASELINE else "benchmark") for m in ALL_MODELS}

# Models that need standardized inputs → wrapped in a StandardScaler pipeline.
_NEEDS_SCALING = {"logreg", "svm_rbf"}
# Models without an Optuna search space (run with fixed config).
_NO_TUNING = {"tabpfn"}
# Per-model Optuna trial caps for the slower learners (keeps wall-time bounded
# without hurting quality much). Others use the caller's n_trials.
_TRIAL_CAP = {"catboost": 12, "svm_rbf": 15}


def is_available(name: str) -> bool:
    if name == "catboost":
        return _HAS_CATBOOST
    if name == "tabpfn":
        return _TABPFN_RUNNABLE
    return name in ALL_MODELS


def available_models(names=None) -> list:
    """Filter a model list (default ALL_MODELS) down to importable ones."""
    return [m for m in (names or ALL_MODELS) if is_available(m)]


def scoring_for(task: str) -> str:
    return "roc_auc" if task == "binary" else "roc_auc_ovr"


# ── Per-model Optuna search spaces ───────────────────────────────────────────────
def _suggest_params(name: str, trial) -> dict:
    if name == "logreg":
        return dict(C=trial.suggest_float("C", 1e-3, 1e2, log=True))
    if name == "svm_rbf":
        return dict(
            C=trial.suggest_float("C", 1e-2, 1e3, log=True),
            gamma=trial.suggest_float("gamma", 1e-5, 1.0, log=True),
        )
    if name == "rf":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600),
            max_depth=trial.suggest_int("max_depth", 2, 12),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_float("max_features", 0.1, 1.0),
        )
    if name == "gbt":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 400),
            max_depth=trial.suggest_int("max_depth", 2, 6),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
        )
    if name == "xgboost":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        )
    if name == "lightgbm":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        )
    if name == "catboost":
        return dict(
            iterations=trial.suggest_int("iterations", 100, 300),
            depth=trial.suggest_int("depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        )
    raise ValueError(f"No search space for model: {name}")


# ── Estimator construction ───────────────────────────────────────────────────────
def _make_estimator(name: str, params: dict, y_tr=None, task: str = "binary"):
    """Return a bare (unscaled) estimator with the given params."""
    p = dict(params or {})
    if name == "logreg":
        return LogisticRegression(**p, class_weight="balanced", max_iter=5000,
                                  random_state=RANDOM_STATE)
    if name == "svm_rbf":
        return SVC(**p, kernel="rbf", class_weight="balanced", probability=True,
                   random_state=RANDOM_STATE)
    if name == "rf":
        return RandomForestClassifier(**p, class_weight="balanced",
                                      random_state=RANDOM_STATE, n_jobs=-1)
    if name == "gbt":
        # GradientBoosting has no class_weight; small imbalance handled by CV metric.
        return GradientBoostingClassifier(**p, random_state=RANDOM_STATE)
    if name == "xgboost":
        kw = {}
        if task == "binary" and y_tr is not None:
            pos = max(float((np.asarray(y_tr) == 1).sum()), 1.0)
            kw["scale_pos_weight"] = float((np.asarray(y_tr) == 0).sum()) / pos
        return xgb.XGBClassifier(**p, **kw, random_state=RANDOM_STATE,
                                 verbosity=0, eval_metric="auc", n_jobs=-1)
    if name == "lightgbm":
        return lgb.LGBMClassifier(**p, class_weight="balanced",
                                  random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
    if name == "catboost":
        if not _HAS_CATBOOST:
            raise RuntimeError("catboost not installed")
        return CatBoostClassifier(**p, loss_function=("Logloss" if task == "binary"
                                                       else "MultiClass"),
                                  boosting_type="Plain",   # default 'Ordered' is very slow on small N
                                  auto_class_weights="Balanced", random_state=RANDOM_STATE,
                                  verbose=0, allow_writing_files=False)
    if name == "tabpfn":
        if not _HAS_TABPFN:
            raise RuntimeError("tabpfn not installed")
        # ignore_pretraining_limits=True → accept the full feature matrix (no cap).
        return TabPFNClassifier(ignore_pretraining_limits=True, device="cpu",
                                random_state=RANDOM_STATE)
    raise ValueError(f"Unknown model: {name}")


def build_model(name: str, params: dict, y_tr=None, task: str = "binary"):
    """Build a fit-ready estimator. Scaling-sensitive models are wrapped in a
    StandardScaler pipeline so all call sites can treat models uniformly."""
    est = _make_estimator(name, params, y_tr=y_tr, task=task)
    if name in _NEEDS_SCALING:
        return Pipeline([("scaler", StandardScaler()), ("clf", est)])
    return est


# ── Optuna objective + convenience tuner ─────────────────────────────────────────
def make_objective(name: str, X_tr, y_tr, inner_cv, task: str = "binary"):
    """Return an Optuna objective, or None if the model has no search space."""
    if name in _NO_TUNING:
        return None
    scoring = scoring_for(task)
    # boosting libs already parallelize internally → keep CV serial to avoid oversubscription
    cv_jobs = 1 if name in {"xgboost", "lightgbm", "catboost", "gbt"} else -1

    def objective(trial):
        params = _suggest_params(name, trial)
        est = build_model(name, params, y_tr=y_tr, task=task)
        return cross_val_score(est, X_tr, y_tr, cv=inner_cv,
                               scoring=scoring, n_jobs=cv_jobs).mean()
    return objective


def tune_and_build(name: str, X_tr, y_tr, inner_cv, n_trials: int = 30,
                   task: str = "binary", seed: int = RANDOM_STATE):
    """Run Optuna (if applicable) and return (estimator, best_params, best_val).

    The returned estimator is unfitted; the caller fits it on the full training
    fold. Models without a search space (tabpfn) return ({}, nan).
    """
    obj = make_objective(name, X_tr, y_tr, inner_cv, task=task)
    if obj is None:
        return build_model(name, {}, y_tr=y_tr, task=task), {}, float("nan")

    n_trials = min(n_trials, _TRIAL_CAP.get(name, n_trials))
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    return build_model(name, best, y_tr=y_tr, task=task), best, float(study.best_value)
