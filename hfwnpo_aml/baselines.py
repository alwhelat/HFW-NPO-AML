"""
Comparison feature-selection methods. All of them work on the same filtered
gene pool as HFW-NPO-AML and all keep the PD-1/PD-L1 anchors, so differences
come from the selection strategy only.

Wrapper baselines (GA, PSO) receive the same fitness object and the same
evaluation budget (population x iterations) as NPO.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .survival import LassoCox, cox_score_z


def _with_anchors(sel, anchors) -> np.ndarray:
    return np.array(sorted(set(np.asarray(sel, int).tolist()) | set(anchors)))


# ---------------------------------------------------------------------------
# metaheuristic wrappers
# ---------------------------------------------------------------------------
def genetic_algorithm(fitness, dim, pop=200, iters=300, seed=42,
                      p_cross=0.9, elite=2):
    """Binary GA: tournament selection, uniform crossover, bit-flip mutation."""
    rng = np.random.RandomState(seed)
    P = (rng.rand(pop, dim) < 0.5).astype(float)
    F = np.array([fitness(p) for p in P])
    history = []
    p_mut = 1.0 / dim
    for _ in range(iters):
        order = np.argsort(F)
        new = [P[i].copy() for i in order[:elite]]
        while len(new) < pop:
            a, b = rng.randint(0, pop, 2), rng.randint(0, pop, 2)
            pa = P[a[np.argmin(F[a])]]
            pb = P[b[np.argmin(F[b])]]
            if rng.rand() < p_cross:
                mask = rng.rand(dim) < 0.5
                child = np.where(mask, pa, pb)
            else:
                child = pa.copy()
            flip = rng.rand(dim) < p_mut
            child[flip] = 1 - child[flip]
            new.append(child)
        P = np.array(new)
        F = np.array([fitness(p) for p in P])
        history.append(float(F.min()))
    best = P[np.argmin(F)]
    return np.where(best > 0.5)[0], history


def binary_pso(fitness, dim, pop=200, iters=300, seed=42,
               c1=2.0, c2=2.0, w_max=0.9, w_min=0.4, v_max=6.0):
    """Binary PSO with a sigmoid transfer function (Kennedy & Eberhart)."""
    rng = np.random.RandomState(seed)
    X = (rng.rand(pop, dim) < 0.5).astype(float)
    V = rng.uniform(-1, 1, (pop, dim))
    F = np.array([fitness(x) for x in X])
    pbest, pbest_f = X.copy(), F.copy()
    g = pbest[np.argmin(pbest_f)].copy()
    g_f = pbest_f.min()
    history = []
    for t in range(iters):
        w = w_max - (w_max - w_min) * t / max(iters - 1, 1)
        r1, r2 = rng.rand(pop, dim), rng.rand(pop, dim)
        V = np.clip(w * V + c1 * r1 * (pbest - X) + c2 * r2 * (g - X), -v_max, v_max)
        X = (rng.rand(pop, dim) < 1 / (1 + np.exp(-V))).astype(float)
        F = np.array([fitness(x) for x in X])
        better = F < pbest_f
        pbest[better], pbest_f[better] = X[better], F[better]
        if pbest_f.min() < g_f:
            g_f = pbest_f.min()
            g = pbest[np.argmin(pbest_f)].copy()
        history.append(float(g_f))
    return np.where(g > 0.5)[0], history


# ---------------------------------------------------------------------------
# classification baselines
# ---------------------------------------------------------------------------
def mrmr(X, y, m, anchors=(), seed=42):
    """Minimum-redundancy maximum-relevance (MI relevance, |r| redundancy)."""
    rel = mutual_info_classif(X, y, random_state=seed)
    C = np.abs(np.corrcoef(X, rowvar=False))
    C = np.nan_to_num(C)
    sel = list(anchors)
    if not sel:
        sel = [int(np.argmax(rel))]
    while len(sel) < min(m, X.shape[1]):
        red = C[:, sel].mean(1)
        score = rel - red
        score[sel] = -np.inf
        sel.append(int(np.argmax(score)))
    return np.array(sorted(sel))


def rfe_svm(X, y, m, anchors=()):
    Xs = StandardScaler().fit_transform(X)
    est = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, dual="auto")
    r = RFE(est, n_features_to_select=min(m, X.shape[1]), step=0.1).fit(Xs, y)
    return _with_anchors(np.where(r.support_)[0], anchors)


def l1_logistic(X, y, anchors=(), seed=42):
    Xs = StandardScaler().fit_transform(X)
    lr = LogisticRegressionCV(Cs=10, cv=3, penalty="l1", solver="liblinear",
                              scoring="roc_auc", class_weight="balanced",
                              random_state=seed, max_iter=5000).fit(Xs, y)
    sel = np.where(np.abs(lr.coef_[0]) > 1e-8)[0]
    if len(sel) == 0:
        sel = np.argsort(np.abs(lr.coef_[0]))[::-1][:5]
    return _with_anchors(sel, anchors)


# ---------------------------------------------------------------------------
# survival baselines
# ---------------------------------------------------------------------------
def univariate_cox_top(X, time, event, m, anchors=()):
    z = np.abs(cox_score_z(X, time, event))
    return _with_anchors(np.argsort(z)[::-1][:m], anchors)


def cox_lasso(X, time, event, anchors=(), seed=42):
    lc = LassoCox(random_state=seed).fit(X, time, event)
    sel = lc.selected()
    if len(sel) == 0:
        sel = np.argsort(np.abs(cox_score_z(X, time, event)))[::-1][:5]
    return _with_anchors(sel, anchors)
