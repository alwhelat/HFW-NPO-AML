"""
Light-weight survival analysis in NumPy (no external survival library needed).

- Harrell's concordance index
- vectorised Cox score test for thousands of genes (continuous and log-rank)
- ridge-penalised Cox regression (Newton-Raphson, Breslow ties)
- L1-penalised Cox regression (proximal gradient) for the Cox-LASSO baseline
- Kaplan-Meier estimator and univariate hazard ratios
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2, norm


# ---------------------------------------------------------------------------
# concordance
# ---------------------------------------------------------------------------
def concordance_index(time, event, risk) -> float:
    """Harrell's C: higher risk should mean shorter survival."""
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    risk = np.asarray(risk, float)
    ev = event == 1
    # comparable pairs: i had the event and t_i < t_j
    comp = ev[:, None] & (time[:, None] < time[None, :])
    if not comp.any():
        return 0.5
    d = risk[:, None] - risk[None, :]
    conc = (d > 0) & comp
    ties = (d == 0) & comp
    return float((conc.sum() + 0.5 * ties.sum()) / comp.sum())


def concordance_many(time, event, R: np.ndarray, chunk: int = 256) -> np.ndarray:
    """C-index for every column of R (n x p)."""
    time = np.asarray(time, float)
    ev = np.asarray(event, int) == 1
    comp = ev[:, None] & (time[:, None] < time[None, :])
    n_comp = comp.sum()
    out = np.full(R.shape[1], 0.5)
    if n_comp == 0:
        return out
    ii, jj = np.nonzero(comp)
    for s in range(0, R.shape[1], chunk):
        block = R[:, s:s + chunk]
        d = block[ii] - block[jj]
        out[s:s + chunk] = ((d > 0).sum(0) + 0.5 * (d == 0).sum(0)) / n_comp
    return out


# ---------------------------------------------------------------------------
# score tests (vectorised over genes)
# ---------------------------------------------------------------------------
def _risk_sets(time):
    order = np.argsort(time, kind="mergesort")
    t_sorted = np.asarray(time, float)[order]
    first = np.searchsorted(t_sorted, t_sorted, side="left")
    return order, first


def cox_score_z(X: np.ndarray, time, event) -> np.ndarray:
    """Score-test z statistic of a univariate Cox model at beta = 0, per column."""
    order, first = _risk_sets(time)
    Xs = X[order].astype(float)
    ev = np.asarray(event, int)[order] == 1
    n = len(Xs)
    S0 = (n - np.arange(n)).astype(float)                   # risk-set sizes
    S1 = np.cumsum(Xs[::-1], axis=0)[::-1]
    S2 = np.cumsum((Xs ** 2)[::-1], axis=0)[::-1]
    S0, S1, S2 = S0[first], S1[first], S2[first]
    mean = S1 / S0[:, None]
    U = (Xs[ev] - mean[ev]).sum(0)
    I = (S2[ev] / S0[ev, None] - mean[ev] ** 2).sum(0)
    I[I < 1e-12] = 1e-12
    return U / np.sqrt(I)


def logrank_many(X: np.ndarray, time, event) -> np.ndarray:
    """Log-rank chi-square for a median split of every column."""
    B = (X > np.median(X, axis=0)).astype(float)
    return cox_score_z(B, time, event) ** 2


def logrank_test(time, event, group) -> tuple[float, float]:
    z = cox_score_z(np.asarray(group, float)[:, None], time, event)[0]
    stat = float(z ** 2)
    return stat, float(chi2.sf(stat, 1))


# ---------------------------------------------------------------------------
# Cox regression
# ---------------------------------------------------------------------------
class RidgeCox:
    """Cox proportional-hazards model with an L2 penalty (Breslow ties)."""

    def __init__(self, l2: float = 1.0, max_iter: int = 30, tol: float = 1e-7,
                 standardize: bool = True):
        self.l2, self.max_iter, self.tol = l2, max_iter, tol
        self.standardize = standardize

    def _prep(self, X):
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X[:, None]
        return (X - self.mu_) / self.sd_

    def _loglik_grad_hess(self, X, ev, first, b, need_hess=True):
        eta = X @ b
        eta = eta - eta.max()
        w = np.exp(eta)
        S0 = np.cumsum(w[::-1])[::-1][first]
        wx = w[:, None] * X
        S1 = np.cumsum(wx[::-1], axis=0)[::-1][first]
        ll = float((eta[ev] - np.log(S0[ev])).sum() - 0.5 * self.l2 * b @ b)
        m = S1[ev] / S0[ev, None]
        grad = (X[ev] - m).sum(0) - self.l2 * b
        if not need_hess:
            return ll, grad, None
        # sum_events S2_i / S0_i  ==  X^T diag(w * c) X, with
        # c_j = sum of 1/S0_i over events i whose risk set contains j
        contrib = np.zeros(len(w))
        np.add.at(contrib, first[ev], 1.0 / S0[ev])
        c = np.cumsum(contrib)
        H = (X * (w * c)[:, None]).T @ X - m.T @ m
        H = H + self.l2 * np.eye(X.shape[1])
        return ll, grad, H

    def fit(self, X, time, event):
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X[:, None]
        if self.standardize:
            self.mu_ = X.mean(0)
            self.sd_ = X.std(0)
            self.sd_[self.sd_ < 1e-12] = 1.0
        else:
            self.mu_, self.sd_ = np.zeros(X.shape[1]), np.ones(X.shape[1])
        order, first = _risk_sets(time)
        Xs = self._prep(X)[order]
        ev = np.asarray(event, int)[order] == 1
        b = np.zeros(X.shape[1])
        ll, g, H = self._loglik_grad_hess(Xs, ev, first, b)
        for _ in range(self.max_iter):
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(H, g, rcond=None)[0]
            t = 1.0
            while t > 1e-4:
                nb = b + t * step
                nll, _, _ = self._loglik_grad_hess(Xs, ev, first, nb, need_hess=False)
                if nll >= ll - 1e-12:
                    break
                t *= 0.5
            b = nb
            if abs(nll - ll) < self.tol:
                ll = nll
                break
            ll, g, H = self._loglik_grad_hess(Xs, ev, first, b)
        self.coef_ = b
        self.loglik_ = ll
        _, _, H = self._loglik_grad_hess(Xs, ev, first, b)
        try:
            self.cov_ = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            self.cov_ = np.full((len(b), len(b)), np.nan)
        return self

    def predict(self, X) -> np.ndarray:
        return self._prep(X) @ self.coef_


class LassoCox:
    """L1-penalised Cox by proximal gradient; lambda chosen by inner CV."""

    def __init__(self, n_lambdas: int = 12, inner_folds: int = 3,
                 max_iter: int = 300, random_state: int = 42):
        self.n_lambdas, self.inner_folds = n_lambdas, inner_folds
        self.max_iter, self.random_state = max_iter, random_state

    @staticmethod
    def _neg_ll_grad(Xs, ev, first, b):
        eta = Xs @ b
        c = eta.max()
        w = np.exp(eta - c)
        S0 = np.cumsum(w[::-1])[::-1][first]
        S1 = np.cumsum((w[:, None] * Xs)[::-1], axis=0)[::-1][first]
        n = len(Xs)
        nll = -((eta[ev] - c - np.log(S0[ev])).sum()) / n
        grad = -((Xs[ev] - S1[ev] / S0[ev, None]).sum(0)) / n
        return nll, grad

    def _fit_path(self, X, time, event, lambdas):
        order, first = _risk_sets(time)
        Xs = X[order]
        ev = np.asarray(event, int)[order] == 1
        b = np.zeros(X.shape[1])
        path = []
        for lam in lambdas:                      # warm starts, large -> small
            step = 1.0
            for _ in range(self.max_iter):
                f, g = self._neg_ll_grad(Xs, ev, first, b)
                while True:
                    z = b - step * g
                    nb = np.sign(z) * np.maximum(np.abs(z) - step * lam, 0.0)
                    nf, _ = self._neg_ll_grad(Xs, ev, first, nb)
                    d = nb - b
                    if nf <= f + g @ d + (d @ d) / (2 * step) + 1e-12 or step < 1e-8:
                        break
                    step *= 0.5
                if np.max(np.abs(nb - b)) < 1e-6:
                    b = nb
                    break
                b = nb
            path.append(b.copy())
        return path

    def fit(self, X, time, event):
        X = np.asarray(X, float)
        self.mu_, self.sd_ = X.mean(0), X.std(0)
        self.sd_[self.sd_ < 1e-12] = 1.0
        Z = (X - self.mu_) / self.sd_
        order, first = _risk_sets(time)
        _, g0 = self._neg_ll_grad(Z[order], np.asarray(event, int)[order] == 1,
                                  first, np.zeros(Z.shape[1]))
        lam_max = float(np.abs(g0).max())
        lambdas = lam_max * np.logspace(0, -2, self.n_lambdas)
        rng = np.random.RandomState(self.random_state)
        folds = rng.randint(0, self.inner_folds, len(Z))
        scores = np.zeros(len(lambdas))
        for k in range(self.inner_folds):
            tr, va = folds != k, folds == k
            if va.sum() < 5 or np.asarray(event)[va].sum() < 2:
                continue
            path = self._fit_path(Z[tr], np.asarray(time)[tr], np.asarray(event)[tr], lambdas)
            for i, b in enumerate(path):
                scores[i] += concordance_index(np.asarray(time)[va], np.asarray(event)[va], Z[va] @ b)
        best = int(np.argmax(scores[1:]) + 1)    # skip the all-zero solution
        self.lambda_ = lambdas[best]
        self.coef_ = self._fit_path(Z, time, event, lambdas[:best + 1])[-1]
        return self

    def selected(self) -> np.ndarray:
        return np.where(np.abs(self.coef_) > 1e-8)[0]

    def predict(self, X):
        return ((np.asarray(X, float) - self.mu_) / self.sd_) @ self.coef_


def univariate_cox(x, time, event) -> dict:
    """Hazard ratio per 1-SD increase, 95% CI and Wald p-value."""
    m = RidgeCox(l2=1e-6).fit(np.asarray(x, float)[:, None], time, event)
    b = float(m.coef_[0])
    se = float(np.sqrt(m.cov_[0, 0]))
    z = b / se if se > 0 else np.nan
    return {"HR": np.exp(b), "CI_low": np.exp(b - 1.96 * se),
            "CI_high": np.exp(b + 1.96 * se), "p": float(2 * norm.sf(abs(z)))}


def multivariable_cox(X, names, time, event, standardize=True) -> list[dict]:
    """standardize=True: HR per 1 SD of each column; False: HR per unit
    (use False when the matrix holds 0/1 indicator variables)."""
    m = RidgeCox(l2=1e-6, standardize=standardize).fit(np.asarray(X, float), time, event)
    out = []
    for i, name in enumerate(names):
        b, se = float(m.coef_[i]), float(np.sqrt(m.cov_[i, i]))
        out.append({"variable": name, "HR": np.exp(b), "CI_low": np.exp(b - 1.96 * se),
                    "CI_high": np.exp(b + 1.96 * se), "p": float(2 * norm.sf(abs(b / se)))})
    return out


def kaplan_meier(time, event):
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    ts = np.unique(time[event == 1])
    surv, s = [], 1.0
    for t in ts:
        at_risk = (time >= t).sum()
        d = ((time == t) & (event == 1)).sum()
        s *= 1 - d / at_risk
        surv.append(s)
    return np.concatenate([[0], ts]), np.concatenate([[1.0], surv])
