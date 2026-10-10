"""
Leakage-free experiment engine shared by the diagnosis, prognosis and
chemoresistance experiments.

For every outer CV fold (repeated stratified K-fold):
  1. scale on the training fold only
  2. Phase 1  candidate pool + hybrid filter           (training fold only)
  3. Phase 2  SMOTE                                     (classification only)
  4. Phase 3  feature selection (NPO or a comparison method)
  5. Phase 4  final model -> metrics on the untouched test fold
"""

from __future__ import annotations

import time as _time

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, f1_score, matthews_corrcoef,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import config
from . import baselines as B
from .fitness import ClassificationFitness, SurvivalFitness
from .hybrid_filter import HybridFilter, SurvivalHybridFilter
from .models import make_cox, make_ensemble, smote_balance
from .npo_optimizer import NPOOptimizer
from .progress import log
from .survival import concordance_index

PROPOSED = "HFW-NPO-AML"
# co-primary methods (config.PRIMARY_METHODS): HFW-NPO (original) and HFW-NPO-AML
PRIMARY = list(config.PRIMARY_METHODS)
MONOCYTIC = "Monocytic markers only"

# method -> (candidate pool, anchored?, selector)
CLASSIFICATION_METHODS = {
    PROPOSED:                      ("centered", True,  "npo"),
    "HFW-NPO (original)":          ("genome",   False, "npo"),
    "HFW-NPO-AML w/o centred pool": ("genome",  True,  "npo"),
    "GA":                          ("centered", True,  "ga"),
    "PSO":                         ("centered", True,  "pso"),
    "mRMR":                        ("centered", True,  "mrmr"),
    "RFE-SVM":                     ("centered", True,  "rfe"),
    "L1-logistic":                 ("centered", True,  "l1"),
    "Filter only":                 ("centered", True,  "all"),
    "PD-1/PD-L1 only":             ("centered", True,  "anchors"),
    # reference panel (chemoresistance only): config.MONOCYTIC_GENES, no filter
    MONOCYTIC:                     ("fixed",    False, "fixed"),
}

# Prognosis design (decided before the full runs, see config.SURVIVAL_POOL):
# PD-1/PD-L1 showed no univariate association with overall survival, so for
# survival the proposed method searches the whole genome while keeping PD-1
# and PD-L1 anchored; all comparison methods use the same pool. The centred
# pool is kept as an ablation.
SURVIVAL_METHODS = {
    PROPOSED:                          ("genome",   True,  "npo"),
    "HFW-NPO (original)":              ("genome",   False, "npo"),
    "HFW-NPO-AML with centred pool":   ("centered", True,  "npo"),
    "GA":                              ("genome",   True,  "ga"),
    "PSO":                             ("genome",   True,  "pso"),
    "Cox-LASSO":                       ("genome",   True,  "coxlasso"),
    "Univariate Cox":                  ("genome",   True,  "unicox"),
    "Filter only":                     ("genome",   True,  "all"),
    "PD-1/PD-L1 only":                 ("genome",   True,  "anchors"),
}

_CFG_KEYS = ["MONOCYTIC_GENES", "FILTER_K", "NPO_CLANS", "NPO_FAMILIES", "NPO_ITER", "NPO_LB", "NPO_UB",
             "ALPHA_FIT", "BASELINE_POP", "BASELINE_ITER", "N_FIXED_FEATURES",
             "CANDIDATE_MODE", "N_COEXPRESSED", "ANCHOR_GENES", "PDL1_AXIS_GENES",
             "FILTER_WEIGHTS", "SURV_FILTER_WEIGHTS", "RANDOM_STATE", "LOG_FILE"]


def cfg_snapshot() -> dict:
    return {k: getattr(config, k) for k in _CFG_KEYS}


def _apply_cfg(cfg: dict) -> None:
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    for k, v in cfg.items():
        setattr(config, k, v)


def classification_metrics(y, proba, thr=0.5) -> dict:
    pred = (proba >= thr).astype(int)
    out = {
        "AUC": roc_auc_score(y, proba) if len(np.unique(y)) > 1 else np.nan,
        "BalAcc": balanced_accuracy_score(y, pred),
        "Accuracy": accuracy_score(y, pred),
        "Sensitivity": recall_score(y, pred, pos_label=1, zero_division=0),
        "Specificity": recall_score(y, pred, pos_label=0, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
        "F1": f1_score(y, pred, zero_division=0),
        "Kappa": cohen_kappa_score(y, pred),
    }
    return {k: float(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# selection step (shared)
# ---------------------------------------------------------------------------
def _select(kind, selector, F, target, anchors, seed, tag=""):
    """Return (selected local indices, convergence history, n_evaluations)."""
    k = F.shape[1]
    m = config.N_FIXED_FEATURES
    if selector == "all":
        return np.arange(k), [], 0
    if selector == "anchors":
        return np.array(anchors, int), [], 0
    if selector in ("npo", "ga", "pso"):
        if kind == "clf":
            fit = ClassificationFitness(F, target, k, anchors, seed=seed)
        else:
            fit = SurvivalFitness(F, target[0], target[1], k, anchors, seed=seed)
        if selector == "npo":
            opt = NPOOptimizer(fit, dim=k, lb=config.NPO_LB, ub=config.NPO_UB,
                               n_clans=config.NPO_CLANS, n_families=config.NPO_FAMILIES,
                               max_iter=config.NPO_ITER, seed=seed,
                               log_fn=(lambda m: log(f"{tag} {m}")) if tag else None)
            opt.optimize()
            sel, hist = opt.get_selected_indices(), opt.history
        elif selector == "ga":
            sel, hist = B.genetic_algorithm(fit, k, config.BASELINE_POP, config.BASELINE_ITER, seed)
        else:
            sel, hist = B.binary_pso(fit, k, config.BASELINE_POP, config.BASELINE_ITER, seed)
        sel = np.array(sorted(set(sel.tolist()) | set(anchors)), int)
        if len(sel) == 0:
            sel = np.arange(min(10, k))
        return sel, list(map(float, hist)), fit.n_calls
    if kind == "clf":
        y = target
        if selector == "mrmr":
            return B.mrmr(F, y, m, anchors, seed), [], 0
        if selector == "rfe":
            return B.rfe_svm(F, y, m, anchors), [], 0
        if selector == "l1":
            return B.l1_logistic(F, y, anchors, seed), [], 0
    else:
        t, e = target
        if selector == "unicox":
            return B.univariate_cox_top(F, t, e, m, anchors), [], 0
        if selector == "coxlasso":
            return B.cox_lasso(F, t, e, anchors, seed), [], 0
    raise ValueError(selector)


def _filters_needed(methods, table):
    return sorted({(table[m][0], table[m][1]) for m in methods})


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------
def _fit_fixed_panel(Xtr, ytr, genes, seed):
    """Reference panel used as is (no filter, no wrapper)."""
    pos = {g: i for i, g in enumerate(genes)}
    cols = np.array([pos[g] for g in config.MONOCYTIC_GENES if g in pos], int)
    Fb, yb = smote_balance(Xtr[:, cols], ytr, seed)
    model = make_ensemble(seed).fit(Fb, yb)
    return dict(model=model, cols=cols, genes=[genes[c] for c in cols],
                history=[], n_evals=0, sel_time=0.0)


def _fit_classification(Xtr, ytr, genes, method, seed, filters=None, tag=""):
    pool_mode, anchored, selector = CLASSIFICATION_METHODS[method]
    if selector == "fixed":
        return _fit_fixed_panel(Xtr, ytr, genes, seed)
    key = (pool_mode, anchored)
    if filters is None or key not in filters:
        from .hybrid_filter import candidate_pool
        hf = HybridFilter().fit(Xtr, ytr, genes, pool=candidate_pool(Xtr, genes, pool_mode),
                                use_anchors=anchored, random_state=seed)
        if filters is not None:
            filters[key] = hf
    else:
        hf = filters[key]
    Ftr = hf.transform(Xtr)
    Fb, yb = smote_balance(Ftr, ytr, seed)
    t0 = _time.time()
    sel, hist, n_ev = _select("clf", selector, Fb, yb, hf.anchor_local, seed, tag)
    sel_time = _time.time() - t0
    model = make_ensemble(seed).fit(Fb[:, sel], yb)
    cols = hf.selected_indices[sel]
    return dict(model=model, cols=cols, genes=[genes[c] for c in cols],
                history=hist, n_evals=n_ev, sel_time=sel_time)


def _classification_fold(X, y, genes, tr, te, methods, cfg, seed, rep, fold):
    _apply_cfg(cfg)
    sc = MinMaxScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), np.clip(sc.transform(X[te]), -1, 2)
    filters = {}
    rows, preds = [], []
    log(f"rep {rep} fold {fold}: started ({len(tr)} train / {len(te)} test samples)")
    for method in methods:
        log(f"rep {rep} fold {fold} | {method}: running ...")
        r = _fit_classification(Xtr, y[tr], genes, method, seed, filters,
                                tag=f"rep {rep} fold {fold} | {method}:")
        proba = r["model"].predict_proba(Xte[:, r["cols"]])[:, 1]
        met = classification_metrics(y[te], proba)
        rows.append(dict(method=method, repeat=rep, fold=fold, n_genes=len(r["cols"]),
                         genes=";".join(r["genes"]), n_evals=r["n_evals"],
                         selection_seconds=round(r["sel_time"], 2),
                         history=r["history"], **met))
        preds += [dict(method=method, repeat=rep, fold=fold, sample=int(i),
                       y=int(yy), proba=float(p)) for i, yy, p in zip(te, y[te], proba)]
        log(f"rep {rep} fold {fold} | {method:<30} AUC={met['AUC']:.3f} "
            f"BalAcc={met['BalAcc']:.3f} genes={len(r['cols'])} "
            f"({r['sel_time'] / 60:.1f} min)")
    return rows, preds


def run_classification_cv(X, y, genes, methods=None):
    methods = methods or list(CLASSIFICATION_METHODS)
    cv = RepeatedStratifiedKFold(n_splits=config.N_SPLITS, n_repeats=config.N_REPEATS,
                                 random_state=config.RANDOM_STATE)
    jobs = []
    for i, (tr, te) in enumerate(cv.split(X, y)):
        rep, fold = divmod(i, config.N_SPLITS)
        jobs.append(delayed(_classification_fold)(
            X, y, genes, tr, te, methods, cfg_snapshot(),
            config.RANDOM_STATE + i, rep + 1, fold + 1))
    out = Parallel(n_jobs=config.N_JOBS, verbose=10)(jobs)
    rows = [r for o in out for r in o[0]]
    preds = [p for o in out for p in o[1]]
    return rows, preds


def fit_final_classifier(X, y, genes, method=PROPOSED):
    sc = MinMaxScaler().fit(X)
    r = _fit_classification(sc.transform(X), y, genes, method, config.RANDOM_STATE)
    r["scaler"] = sc
    return r


def _final_job(kind, X, target, genes, method, cfg, label):
    _apply_cfg(cfg)
    log(f"final model | {label} | {method}: running ...")
    if kind == "clf":
        r = fit_final_classifier(X, target, genes, method)
    else:
        r = fit_final_survival(X, target[0], target[1], genes, method)
    log(f"final model | {label} | {method}: done, {len(r['cols'])} genes "
        f"({r['sel_time'] / 60:.1f} min)")
    return r


def fit_final_parallel(kind, specs):
    """Fit several final models at once.

    specs: list of (key, X, target, genes, method); target is y for
    kind="clf" and (time, event) for kind="surv". Returns {key: result}.
    """
    cfg = cfg_snapshot()
    jobs = [delayed(_final_job)(kind, X, target, genes, method, cfg, str(key[0]))
            for key, X, target, genes, method in specs]
    out = Parallel(n_jobs=config.N_JOBS, verbose=0)(jobs)
    return {spec[0]: r for spec, r in zip(specs, out)}


# ---------------------------------------------------------------------------
# survival
# ---------------------------------------------------------------------------
def _fit_survival(Xtr, ttr, etr, genes, method, seed, filters=None, tag=""):
    pool_mode, anchored, selector = SURVIVAL_METHODS[method]
    key = (pool_mode, anchored)
    if filters is None or key not in filters:
        from .hybrid_filter import candidate_pool
        hf = SurvivalHybridFilter().fit(Xtr, ttr, etr, genes,
                                        pool=candidate_pool(Xtr, genes, pool_mode),
                                        use_anchors=anchored)
        if filters is not None:
            filters[key] = hf
    else:
        hf = filters[key]
    Ftr = hf.transform(Xtr)
    t0 = _time.time()
    sel, hist, n_ev = _select("surv", selector, Ftr, (ttr, etr), hf.anchor_local, seed, tag)
    sel_time = _time.time() - t0
    model = make_cox().fit(Ftr[:, sel], ttr, etr)
    cols = hf.selected_indices[sel]
    return dict(model=model, cols=cols, genes=[genes[c] for c in cols],
                history=hist, n_evals=n_ev, sel_time=sel_time)


def _survival_fold(X, t, e, genes, tr, te, methods, cfg, seed, rep, fold):
    _apply_cfg(cfg)
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    filters = {}
    rows, preds = [], []
    log(f"rep {rep} fold {fold}: started ({len(tr)} train / {len(te)} test patients)")
    for method in methods:
        log(f"rep {rep} fold {fold} | {method}: running ...")
        r = _fit_survival(Xtr, t[tr], e[tr], genes, method, seed, filters,
                          tag=f"rep {rep} fold {fold} | {method}:")
        risk = r["model"].predict(Xte[:, r["cols"]])
        c = concordance_index(t[te], e[te], risk)
        rows.append(dict(method=method, repeat=rep, fold=fold, n_genes=len(r["cols"]),
                         genes=";".join(r["genes"]), n_evals=r["n_evals"],
                         selection_seconds=round(r["sel_time"], 2),
                         history=r["history"], C_index=float(c)))
        z = (risk - risk.mean()) / (risk.std() + 1e-12)
        preds += [dict(method=method, repeat=rep, fold=fold, sample=int(i), time=float(tt),
                       event=int(ee), risk_z=float(rz))
                  for i, tt, ee, rz in zip(te, t[te], e[te], z)]
        log(f"rep {rep} fold {fold} | {method:<30} C-index={c:.3f} "
            f"genes={len(r['cols'])} ({r['sel_time'] / 60:.1f} min)")
    return rows, preds


def run_survival_cv(X, t, e, genes, methods=None):
    methods = methods or list(SURVIVAL_METHODS)
    cv = RepeatedStratifiedKFold(n_splits=config.N_SPLITS, n_repeats=config.N_REPEATS,
                                 random_state=config.RANDOM_STATE)
    jobs = []
    for i, (tr, te) in enumerate(cv.split(X, e)):
        rep, fold = divmod(i, config.N_SPLITS)
        jobs.append(delayed(_survival_fold)(
            X, t, e, genes, tr, te, methods, cfg_snapshot(),
            config.RANDOM_STATE + i, rep + 1, fold + 1))
    out = Parallel(n_jobs=config.N_JOBS, verbose=10)(jobs)
    return [r for o in out for r in o[0]], [p for o in out for p in o[1]]


def fit_final_survival(X, t, e, genes, method=PROPOSED):
    sc = StandardScaler().fit(X)
    r = _fit_survival(sc.transform(X), t, e, genes, method, config.RANDOM_STATE)
    r["scaler"] = sc
    return r
