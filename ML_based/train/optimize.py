"""
Optuna inner GroupKFold HPO.
- MedianPruner(n_warmup_steps=10)
- SQLite 저장 → 재시작 시 이어서 실행
- 모델별 search_space.yaml 파라미터 탐색
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import optuna
import yaml
from omegaconf import DictConfig
from sklearn.metrics import f1_score

ML = Path(__file__).parent.parent
SEARCH_SPACE_PATH = ML / "configs" / "search_space.yaml"

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _load_search_space() -> dict:
    with open(SEARCH_SPACE_PATH) as f:
        return yaml.safe_load(f)


def _suggest(trial: optuna.Trial, name: str, spec) -> object:
    """search_space.yaml 명세를 Optuna suggest 호출로 변환."""
    if isinstance(spec, list):
        if len(spec) == 3 and spec[2] == "log":
            return trial.suggest_float(name, float(spec[0]), float(spec[1]), log=True)
        elif len(spec) == 2 and all(isinstance(v, (int, float)) for v in spec):
            if isinstance(spec[0], int) and isinstance(spec[1], int):
                return trial.suggest_int(name, spec[0], spec[1])
            return trial.suggest_float(name, float(spec[0]), float(spec[1]))
        else:
            return trial.suggest_categorical(name, spec)
    elif isinstance(spec, dict):
        # {low: x, high: y, log: bool} 형식
        low   = spec.get("low", spec.get("min"))
        high  = spec.get("high", spec.get("max"))
        log   = spec.get("log", False)
        step  = spec.get("step", None)
        if isinstance(low, int) and isinstance(high, int) and not log:
            return trial.suggest_int(name, low, high, step=step or 1)
        return trial.suggest_float(name, float(low), float(high), log=log)
    else:
        return spec


def _build_params(trial: optuna.Trial, model_name: str, cfg: DictConfig,
                  search_space: dict, feat_count_cap: int = None) -> dict:
    """
    Optuna trial에서 하이퍼파라미터를 샘플링.

    feat_count_cap: n_features의 상한을 실제 feature 수로 제한 (None이면 무제한).
    n_features는 모델 생성자 인자가 아닌 특수 키로 반환 — objective에서 pop 처리.
    """
    model_space = search_space.get(model_name, {})
    params: dict = {}

    n_classes = 2 if cfg.targets.mode == "binary" else 3
    params["n_classes"] = n_classes

    for key, spec in model_space.items():
        if key == "n_features":
            low  = int(spec[0])
            high = int(spec[1])
            if feat_count_cap is not None:
                high = min(high, feat_count_cap)
            high = max(high, low)  # low > cap 방지
            params["n_features"] = trial.suggest_int("n_features", low, high)
        else:
            params[key] = _suggest(trial, key, spec)

    model_cfg = cfg.models.get(model_name, {})
    for k in ("max_iter", "class_weight", "max_epochs", "batch_size", "patience",
              "min_delta", "early_stopping_rounds"):
        if k in model_cfg:
            params.setdefault(k, model_cfg[k])

    if model_name in ("cnn1d", "transformer"):
        params.setdefault("speed_dim", 3 if cfg.features.speed_as_feature else 0)

    return params


def run_optuna(
    model_name: str,
    model_cls,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    groups_tr: np.ndarray,
    cfg: DictConfig,
    fold: int,
    sp_tr: np.ndarray | None = None,
    mk_tr: np.ndarray | None = None,
    is_dl: bool = False,
) -> dict:
    """
    Inner GroupKFold CV + Optuna HPO.
    Returns best_params dict.
    """
    from loaders.splits import make_outer_splits
    from sklearn.preprocessing import StandardScaler

    version = cfg.get("version", "v1")
    optuna_dir = ML / f"artifacts-{version}" / "optuna"
    optuna_dir.mkdir(parents=True, exist_ok=True)
    search_space = _load_search_space()

    n_inner  = cfg.cv.n_inner
    n_trials = cfg.optuna.n_trials.dl if is_dl else cfg.optuna.n_trials.sklearn
    inner_splits = make_outer_splits(groups_tr, n_inner, cfg.cv.seed + fold * 100, y=y_tr)

    def objective(trial: optuna.Trial) -> float:
        params = _build_params(trial, model_name, cfg, search_space,
                               feat_count_cap=X_tr.shape[1])
        # n_features는 모델 생성자 인자가 아님 — 먼저 pop
        n_features = params.pop("n_features", None)
        fold_f1s = []
        fold_bacs = []
        fold_precs = []
        fold_recs = []
        fold_aucs = []
        fold_f1_per_cls = []

        for inner_fold_idx, (i_tr, i_va) in enumerate(inner_splits):
            Xi_tr, Xi_va = X_tr[i_tr], X_tr[i_va]
            yi_tr, yi_va = y_tr[i_tr], y_tr[i_va]

            # n_features 슬라이싱: 상위 n_features개 피처 컬럼 사용
            # speed one-hot (마지막 3열)은 항상 포함
            if n_features is not None and not is_dl:
                n_speed = 3 if cfg.features.speed_as_feature else 0
                nf = min(n_features, Xi_tr.shape[1] - n_speed)
                col_idx = list(range(nf)) + list(range(Xi_tr.shape[1] - n_speed, Xi_tr.shape[1]))
                Xi_tr = Xi_tr[:, col_idx]
                Xi_va = Xi_va[:, col_idx]

            sp_i_tr = sp_tr[i_tr] if sp_tr is not None else None
            sp_i_va = sp_tr[i_va] if sp_tr is not None else None
            mk_i_tr = mk_tr[i_tr] if mk_tr is not None else None
            mk_i_va = mk_tr[i_va] if mk_tr is not None else None

            if Xi_tr.ndim == 3:
                from loaders.waveform_loader import WaveformScaler
                sc = WaveformScaler(mode=cfg.features.waveform_norm)
                if cfg.features.waveform_norm == "ha_centered":
                    ha_mask = (yi_tr == 2) if cfg.targets.mode == "multiclass" else (yi_tr == 0)
                    sc.fit(Xi_tr, ha_mask=ha_mask)
                else:
                    sc.fit(Xi_tr)
                Xi_tr = sc.transform(Xi_tr)
                Xi_va = sc.transform(Xi_va)
            elif not is_dl:
                sc = StandardScaler()
                Xi_tr = sc.fit_transform(Xi_tr)
                Xi_va = sc.transform(Xi_va)

            model = model_cls(**params)

            if is_dl:
                val_split = max(1, int(len(Xi_tr) * 0.15))
                val_data  = (Xi_tr[:val_split],
                             sp_i_tr[:val_split] if sp_i_tr is not None else None,
                             mk_i_tr[:val_split] if mk_i_tr is not None else None,
                             yi_tr[:val_split])
                X_fit, y_fit = Xi_tr[val_split:], yi_tr[val_split:]
                sp_fit = sp_i_tr[val_split:] if sp_i_tr is not None else None
                mk_fit = mk_i_tr[val_split:] if mk_i_tr is not None else None
                model.fit(X_fit, y_fit, speed=sp_fit, mask=mk_fit,
                          val_data=val_data, trial=trial)
                preds = model.predict(Xi_va, speed=sp_i_va, mask=mk_i_va)
            else:
                model.fit(Xi_tr, yi_tr)
                preds = model.predict(Xi_va)

            from sklearn.metrics import f1_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score
            
            fold_f1s.append(f1_score(yi_va, preds, average="macro", zero_division=0))
            fold_bacs.append(balanced_accuracy_score(yi_va, preds))
            fold_precs.append(precision_score(yi_va, preds, average="macro", zero_division=0))
            fold_recs.append(recall_score(yi_va, preds, average="macro", zero_division=0))
            fold_f1_per_cls.append(f1_score(yi_va, preds, average=None, zero_division=0))
            
            try:
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(Xi_va)
                    if y_prob.ndim == 2 and y_prob.shape[1] >= 2:
                        if len(np.unique(yi_va)) == 2:
                            auc = roc_auc_score(yi_va, y_prob[:, 1])
                        else:
                            auc = roc_auc_score(yi_va, y_prob, multi_class="ovr", average="macro")
                    else:
                        auc = 0.0
                else:
                    auc = 0.0
            except Exception:
                auc = 0.0
            fold_aucs.append(auc)

            # step별 pruning (DL은 fit 내부에서 처리, sklearn은 fold 단위)
            # step에 순차 인덱스 사용 — i_tr[0]은 비단조적이라 Optuna가 report를 무시함
            if not is_dl:
                trial.report(np.mean(fold_f1s), step=inner_fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        trial.set_user_attr("bacc", float(np.mean(fold_bacs)))
        trial.set_user_attr("prec", float(np.mean(fold_precs)))
        trial.set_user_attr("rec", float(np.mean(fold_recs)))
        trial.set_user_attr("auc", float(np.mean(fold_aucs)))
        
        # per_class f1 list
        f1_cls_mean = np.mean(fold_f1_per_cls, axis=0)
        f1_cls_str = ",".join([f"{x:.4f}" for x in f1_cls_mean])
        trial.set_user_attr("f1_cls", f1_cls_str)
        
        return float(np.mean(fold_f1s))

    # feature 수를 study 이름에 포함 → feature set 변경 시 자동으로 새 study 생성
    n_feat = X_tr.shape[1]
    study_key = f"{model_name}_fold{fold}_nf{n_feat}"
    storage = f"sqlite:///{optuna_dir}/study_{study_key}.db"
    pruner  = optuna.pruners.MedianPruner(
        n_warmup_steps=cfg.optuna.pruner_warmup_steps
    )
    study = optuna.create_study(
        study_name=study_key,
        storage=storage,
        direction="maximize",
        pruner=pruner,
        load_if_exists=True,
    )
    existing = len(study.trials)
    remaining = max(0, n_trials - existing)
    print(f"  [Optuna] {model_name} fold={fold}: nf={n_feat}, 기존 {existing}trials, {remaining} 추가 예정")
    if remaining == 0:
        print(f"  [Optuna] 이미 {n_trials} trials 완료 — best 파라미터 재사용")
    else:
        import optuna as _optuna

        def _trial_callback(s, t):
            if t.value is None:
                return
            # pruned trial: user_attrs 미설정 (bacc 없음) → [PRUNED] 표시
            is_pruned = 'bacc' not in t.user_attrs
            if is_pruned:
                print(f"    trial {len(s.trials)}/{n_trials}  [PRUNED] F1_fold1={t.value:.4f} | params={t.params}")
            else:
                print(
                    f"    trial {len(s.trials)}/{n_trials}  "
                    f"F1={t.value:.4f} BAcc={t.user_attrs['bacc']:.4f} "
                    f"Prec={t.user_attrs.get('prec', 0):.4f} Rec={t.user_attrs.get('rec', 0):.4f} "
                    f"AUC={t.user_attrs.get('auc', 0):.4f} f1_cls=[{t.user_attrs.get('f1_cls', '')}] | "
                    f"params={t.params}"
                )

        study.optimize(objective, n_trials=remaining, show_progress_bar=False,
                       callbacks=[_trial_callback])

    best = study.best_params.copy()
    # n_classes 재삽입 (best_params에서 제외되는 경우 대비)
    best.setdefault("n_classes", 2 if cfg.targets.mode == "binary" else 3)

    if is_dl:
        best.setdefault("speed_dim", 3 if cfg.features.speed_as_feature else 0)

    # config의 고정 파라미터 병합
    model_cfg = cfg.models.get(model_name, {})
    for k in ("max_iter", "class_weight", "max_epochs", "batch_size", "patience",
              "min_delta", "early_stopping_rounds"):
        if k in model_cfg:
            best.setdefault(k, model_cfg[k])

    return best
