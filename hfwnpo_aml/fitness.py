"""
Wrapper fitness functions (lower is better).

    f(S) = ALPHA * (1 - score(S)) + (1 - ALPHA) * sqrt(|S \\ anchors| / K)

score = balanced accuracy of a RidgeClassifier (classification, as in the
published HFW-NPO) or the C-index of a ridge Cox model (survival, new in this
study), both from 3-fold CV on the training fold. Anchor genes (PD-1, PD-L1)
are forced into every mask and are not counted in the compactness penalty.
Results are cached, because metaheuristics re-visit the same masks often.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import config
from .survival import RidgeCox, concordance_index


class _Base:
    def __init__(self, k_retained: int, anchor_idx=(), alpha: float | None = None):
        self.k_retained = k_retained
        self.anchor_idx = np.asarray(list(anchor_idx), int)
        self.alpha = config.ALPHA_FIT if alpha is None else alpha
        self.cache: dict[bytes, float] = {}
        self.n_calls = 0

    def _mask(self, x) -> np.ndarray:
        m = np.asarray(x, float) > 0.5
        if len(self.anchor_idx):
            m[self.anchor_idx] = True
        return m

    def __call__(self, x) -> float:
        self.n_calls += 1
        m = self._mask(x)
        key = np.packbits(m).tobytes()
        if key in self.cache:
            return self.cache[key]
        sel = np.where(m)[0]
        if len(sel) == 0:
            val = 1.0
        else:
            try:
                score = self.score(sel)
            except Exception:
                score = 0.0
            n_free = len(sel) - len(self.anchor_idx)
            ratio = max(n_free, 0) / self.k_retained
            val = float(self.alpha * (1.0 - score) + (1.0 - self.alpha) * np.sqrt(ratio))
        self.cache[key] = val
        return val

    evaluate = __call__

    def selected(self, x) -> np.ndarray:
        return np.where(self._mask(x))[0]


class ClassificationFitness(_Base):
    def __init__(self, X, y, k_retained, anchor_idx=(), alpha=None, seed=42):
        super().__init__(k_retained, anchor_idx, alpha)
        self.X, self.y = X, y
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        self.splits = list(cv.split(X, y))

    def score(self, sel) -> float:
        Xs = self.X[:, sel]
        scores = []
        for tr, va in self.splits:
            sc = StandardScaler().fit(Xs[tr])
            clf = RidgeClassifier(alpha=1.0, class_weight="balanced")
            clf.fit(sc.transform(Xs[tr]), self.y[tr])
            scores.append(balanced_accuracy_score(self.y[va], clf.predict(sc.transform(Xs[va]))))
        return float(np.mean(scores))


class SurvivalFitness(_Base):
    def __init__(self, X, time, event, k_retained, anchor_idx=(), alpha=None, seed=42):
        super().__init__(k_retained, anchor_idx, alpha)
        self.X, self.time, self.event = X, np.asarray(time, float), np.asarray(event, int)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        self.splits = list(cv.split(X, self.event))

    def score(self, sel) -> float:
        Xs = self.X[:, sel]
        scores = []
        for tr, va in self.splits:
            m = RidgeCox(l2=1.0, max_iter=15).fit(Xs[tr], self.time[tr], self.event[tr])
            scores.append(concordance_index(self.time[va], self.event[va], m.predict(Xs[va])))
        return float(np.mean(scores))
