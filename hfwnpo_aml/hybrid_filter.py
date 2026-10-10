"""
Phase 1 of HFW-NPO-AML: PD-1/PD-L1-centred candidate pool + hybrid filter.

Adapted from the HFW-NPO RealHybridFilter (Alwhelat & Abiyev, Electronics
2026, 15(13), 2970). Additions for this study:
  * a PD-1/PD-L1-centred candidate pool (anchors, curated axis genes and the
    genes most co-expressed with the anchors in the training fold);
  * anchors are always kept in the filter output;
  * a survival version of the omega score (Cox score |z|, |C - 0.5|, log-rank).
All scores are computed on the training fold only.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2, ttest_ind
from sklearn.feature_selection import mutual_info_classif

import config
from .survival import concordance_many, cox_score_z, logrank_many


def _minmax(a: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(np.asarray(a, float))
    lo, hi = a.min(), a.max()
    return np.zeros_like(a) if hi - lo < 1e-12 else (a - lo) / (hi - lo)


def anchor_indices(genes: list[str]) -> list[int]:
    pos = {g: i for i, g in enumerate(genes)}
    return [pos[g] for g in config.ANCHOR_GENES if g in pos]


def candidate_pool(X: np.ndarray, genes: list[str], mode: str | None = None) -> np.ndarray:
    """Column indices that enter the filter (training data only)."""
    mode = mode or config.CANDIDATE_MODE
    if mode == "genome":
        return np.arange(X.shape[1])
    pos = {g: i for i, g in enumerate(genes)}
    anchors = anchor_indices(genes)
    axis = [pos[g] for g in dict.fromkeys(config.PDL1_AXIS_GENES) if g in pos]
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    co = set()
    for a in anchors:
        r = np.abs(Z.T @ Z[:, a]) / len(Z)
        r[a] = -1
        co.update(np.argsort(r)[::-1][:config.N_COEXPRESSED].tolist())
    return np.array(sorted(set(anchors) | set(axis) | co))


class _BaseFilter:
    def __init__(self, k: int | None = None):
        self.k = k
        self.selected_indices: np.ndarray | None = None
        self.anchor_local: list[int] = []

    def _finish(self, omega: np.ndarray, pool: np.ndarray, anchors: list[int]):
        k = min(self.k or config.FILTER_K, len(pool))
        order = pool[np.argsort(omega)[::-1]]
        chosen = [a for a in anchors]
        for idx in order:
            if len(chosen) >= k:
                break
            if idx not in chosen:
                chosen.append(int(idx))
        self.selected_indices = np.array(chosen)
        self.anchor_local = list(range(len(anchors)))    # anchors come first
        self.omega_ = dict(zip(pool.tolist(), omega.tolist()))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_indices]


class HybridFilter(_BaseFilter):
    """omega = 0.40 Fisher + 0.30 MI + 0.30 Welch(-log10 p)  (classification)."""

    def fit(self, X, y, genes, pool=None, use_anchors=True, random_state=42):
        pool = candidate_pool(X, genes) if pool is None else pool
        Xp = X[:, pool]
        m1, m0 = y == 1, y == 0
        denom = Xp[m1].var(0) + Xp[m0].var(0)
        denom[denom < 1e-12] = 1e-12
        fisher = (Xp[m1].mean(0) - Xp[m0].mean(0)) ** 2 / denom
        mi = mutual_info_classif(Xp, y, random_state=random_state)
        _, p = ttest_ind(Xp[m1], Xp[m0], equal_var=False, axis=0)
        welch = -np.log10(np.clip(np.nan_to_num(p, nan=1.0), 1e-300, 1.0))
        w1, w2, w3 = config.FILTER_WEIGHTS
        omega = w1 * _minmax(fisher) + w2 * _minmax(mi) + w3 * _minmax(welch)
        anchors = anchor_indices(genes) if use_anchors else []
        return self._finish(omega, pool, anchors)


class SurvivalHybridFilter(_BaseFilter):
    """omega = 0.40 |Cox score z| + 0.30 |C-index - 0.5| + 0.30 log-rank(-log10 p)."""

    def fit(self, X, time, event, genes, pool=None, use_anchors=True, prescreen=2000):
        pool = candidate_pool(X, genes) if pool is None else pool
        Xp = X[:, pool]
        z = np.abs(cox_score_z(Xp, time, event))
        lr = -np.log10(np.clip(chi2.sf(logrank_many(Xp, time, event), 1), 1e-300, 1))
        cidx = np.zeros(len(pool))
        top = np.argsort(z)[::-1][:prescreen]              # C-index is O(n^2)
        cidx[top] = np.abs(concordance_many(time, event, Xp[:, top]) - 0.5)
        w1, w2, w3 = config.SURV_FILTER_WEIGHTS
        omega = w1 * _minmax(z) + w2 * _minmax(cidx) + w3 * _minmax(lr)
        anchors = anchor_indices(genes) if use_anchors else []
        return self._finish(omega, pool, anchors)
