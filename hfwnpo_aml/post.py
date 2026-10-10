"""
Post-hoc statistics for the manuscript. Everything here works on the saved
results (fold metrics, out-of-fold predictions, external predictions, final
models) and on the prepared data; nothing re-tunes a model.

Contents
  ranking      Nemenyi critical difference
  effect size  bootstrap CI of paired differences, Hedges g
  stability    pairwise Jaccard of selected genes vs chance
  AUC          DeLong variance, CI and paired test
  calibration  Brier score, calibration intercept / slope, reliability curve
  decision     decision-curve net benefit
  survival     IPCW time-dependent AUC, paired bootstrap of C-index
  meta         DerSimonian-Laird random-effects pooling with I^2
  correlation  partial Spearman correlation
  enrichment   Enrichr API (needs internet; skipped when offline)
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, studentized_range, t as t_dist

from .survival import concordance_index, kaplan_meier


# ---------------------------------------------------------------------------
# rankings and effect sizes
# ---------------------------------------------------------------------------
def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    """Critical difference of mean ranks for k methods over n paired blocks."""
    q = studentized_range.ppf(1 - alpha, k, 1e6) / math.sqrt(2)
    return float(q * math.sqrt(k * (k + 1) / (6.0 * n)))


def mean_ranks(df: pd.DataFrame, metric: str, higher_is_better=True):
    wide = df.pivot_table(index=["repeat", "fold"], columns="method", values=metric).dropna(axis=1)
    ranks = wide.rank(axis=1, ascending=not higher_is_better).mean().sort_values()
    return ranks, wide.shape[0]


def bootstrap_paired_diff(a, b, n_boot=5000, seed=42):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.RandomState(seed)
    boots = d[rng.randint(0, len(d), (n_boot, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def hedges_g(a, b):
    """Standardised mean difference (a - b) with small-sample correction and its variance."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    sp = math.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    d = (a.mean() - b.mean()) / sp if sp > 0 else 0.0
    j = 1 - 3 / (4 * (n1 + n2) - 9)
    g = j * d
    var = (n1 + n2) / (n1 * n2) + g ** 2 / (2 * (n1 + n2))
    return float(g), float(var)


# ---------------------------------------------------------------------------
# feature-selection stability
# ---------------------------------------------------------------------------
def selection_stability(fold_df: pd.DataFrame, pool_size: int) -> pd.DataFrame:
    """Mean pairwise Jaccard index of the gene sets selected in the outer folds,
    with the Jaccard expected for random sets of the same sizes drawn from a
    pool of `pool_size` genes (the filter output) as a reference."""
    rows = []
    for m, d in fold_df.groupby("method", sort=False):
        sets = [set(str(g).split(";")) for g in d["genes"] if isinstance(g, str) and g]
        if len(sets) < 2:
            continue
        jac, chance = [], []
        for s1, s2 in itertools.combinations(sets, 2):
            jac.append(len(s1 & s2) / len(s1 | s2))
            k1, k2 = min(len(s1), pool_size), min(len(s2), pool_size)
            e_inter = k1 * k2 / pool_size
            chance.append(e_inter / (k1 + k2 - e_inter))
        rows.append({"method": m, "n_sets": len(sets),
                     "mean_size": float(np.mean([len(s) for s in sets])),
                     "jaccard_mean": float(np.mean(jac)), "jaccard_sd": float(np.std(jac)),
                     "jaccard_expected_by_chance": float(np.mean(chance)),
                     "stability_ratio": float(np.mean(jac) / np.mean(chance))})
    return pd.DataFrame(rows).set_index("method")


# ---------------------------------------------------------------------------
# AUC: DeLong
# ---------------------------------------------------------------------------
def _midrank(x):
    return rankdata(x, method="average")


def _delong_components(y, scores):
    """scores: (k, n). Returns aucs (k,), covariance (k, k)."""
    y = np.asarray(y).astype(int)
    pos, neg = scores[:, y == 1], scores[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = scores.shape[0]
    tx = np.array([_midrank(p) for p in pos])
    ty = np.array([_midrank(q) for q in neg])
    tz = np.array([_midrank(np.r_[p, q]) for p, q in zip(pos, neg)])
    aucs = tz[:, :m].sum(1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01)) if k > 1 else np.array([[np.var(v01, ddof=1)]])
    sy = np.atleast_2d(np.cov(v10)) if k > 1 else np.array([[np.var(v10, ddof=1)]])
    return aucs, sx / m + sy / n


def delong_ci(y, p, alpha=0.05):
    aucs, cov = _delong_components(y, np.atleast_2d(p))
    se = math.sqrt(cov[0, 0])
    z = norm.ppf(1 - alpha / 2)
    return float(aucs[0]), float(max(0, aucs[0] - z * se)), float(min(1, aucs[0] + z * se))


def delong_test(y, p1, p2):
    """Two-sided DeLong test for two correlated AUCs on the same samples."""
    aucs, cov = _delong_components(y, np.vstack([p1, p2]))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), 0.0, 1.0
    zval = (aucs[0] - aucs[1]) / math.sqrt(var)
    return float(aucs[0]), float(aucs[1]), float(zval), float(2 * norm.sf(abs(zval)))


# ---------------------------------------------------------------------------
# calibration and decision curves
# ---------------------------------------------------------------------------
def calibration_stats(y, p, bins=10):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    brier = float(np.mean((p - y) ** 2))
    lp = np.log(p / (1 - p))
    import statsmodels.api as sm
    try:
        slope_fit = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
        slope = float(slope_fit.params[1])
        int_fit = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit()
        intercept = float(int_fit.params[0])
    except Exception:
        slope, intercept = np.nan, np.nan
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, bins - 1)
    curve = pd.DataFrame({"bin": idx, "p": p, "y": y}).groupby("bin").mean()
    return {"Brier": brier, "cal_intercept": intercept, "cal_slope": slope}, curve


def decision_curve(y, p, thresholds=None):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    thresholds = np.arange(0.05, 0.96, 0.01) if thresholds is None else thresholds
    n = len(y)
    prev = y.mean()
    rows = []
    for t in thresholds:
        pred = p >= t
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        w = t / (1 - t)
        rows.append({"threshold": t, "net_benefit": tp / n - fp / n * w,
                     "treat_all": prev - (1 - prev) * w, "treat_none": 0.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# survival extras
# ---------------------------------------------------------------------------
def _censor_survival(time, event):
    """Kaplan-Meier estimate of the censoring distribution G(t)."""
    tt, gg = kaplan_meier(time, 1 - np.asarray(event, int))
    def G(x):
        i = np.searchsorted(tt, x, side="right") - 1
        return gg[np.clip(i, 0, len(gg) - 1)]
    return G


def td_auc(time, event, risk, t):
    """Cumulative/dynamic AUC at time t with inverse-probability-of-censoring
    weights (Uno et al. 2007; Hung & Chiang 2010)."""
    time, event, risk = (np.asarray(a, float) for a in (time, event, risk))
    G = _censor_survival(time, event)
    cases = (time <= t) & (event == 1)
    controls = time > t
    if cases.sum() < 3 or controls.sum() < 3:
        return np.nan
    w = 1.0 / np.maximum(G(time[cases] - 1e-9), 1e-6)
    rc, rn = risk[cases], risk[controls]
    conc = (rc[:, None] > rn[None, :]).astype(float) + 0.5 * (rc[:, None] == rn[None, :])
    return float((w[:, None] * conc).sum() / (w.sum() * controls.sum()))


def paired_cindex_bootstrap(time, event, r1, r2, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(time)
    d0 = concordance_index(time, event, r1) - concordance_index(time, event, r2)
    ds = []
    for _ in range(n_boot):
        i = rng.randint(0, n, n)
        ds.append(concordance_index(time[i], event[i], r1[i]) -
                  concordance_index(time[i], event[i], r2[i]))
    ds = np.array(ds)
    p = 2 * min((ds <= 0).mean(), (ds >= 0).mean())
    return float(d0), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)), float(min(1.0, p))


# ---------------------------------------------------------------------------
# meta-analysis
# ---------------------------------------------------------------------------
def random_effects(est, var):
    """DerSimonian-Laird random-effects pooling. est/var on the analysis scale
    (log HR or Hedges g)."""
    est, var = np.asarray(est, float), np.asarray(var, float)
    w = 1 / var
    fixed = np.sum(w * est) / w.sum()
    q = np.sum(w * (est - fixed) ** 2)
    df = len(est) - 1
    c = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    wr = 1 / (var + tau2)
    pooled = np.sum(wr * est) / wr.sum()
    se = math.sqrt(1 / wr.sum())
    i2 = max(0.0, (q - df) / q) * 100 if q > 0 else 0.0
    return {"pooled": float(pooled), "ci_low": float(pooled - 1.96 * se),
            "ci_high": float(pooled + 1.96 * se), "p": float(2 * norm.sf(abs(pooled / se))),
            "tau2": float(tau2), "I2": float(i2), "Q": float(q), "k": int(len(est))}


# ---------------------------------------------------------------------------
# correlation
# ---------------------------------------------------------------------------
def partial_spearman(x, y, z):
    """Spearman correlation of x and y adjusted for covariate(s) z."""
    x, y = rankdata(x), rankdata(y)
    Z = np.column_stack([np.ones(len(x))] + [rankdata(c) for c in np.atleast_2d(np.asarray(z).T)])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    r = float(np.corrcoef(rx, ry)[0, 1])
    df = len(x) - 2 - (Z.shape[1] - 1)
    tval = r * math.sqrt(df / max(1e-12, 1 - r * r))
    return r, float(2 * t_dist.sf(abs(tval), df))


# ---------------------------------------------------------------------------
# enrichment (Enrichr web API)
# ---------------------------------------------------------------------------
ENRICHR = "https://maayanlab.cloud/Enrichr"
LIBRARIES = ["GO_Biological_Process_2023", "KEGG_2021_Human", "Reactome_2022"]


def enrichr(genes, description="HFW-NPO-AML", libraries=LIBRARIES, timeout=60):
    try:
        import requests
    except ImportError:
        return None
    try:
        r = requests.post(f"{ENRICHR}/addList", timeout=timeout,
                          files={"list": (None, "\n".join(genes)),
                                 "description": (None, description)})
        r.raise_for_status()
        uid = r.json()["userListId"]
        rows = []
        for lib in libraries:
            q = requests.get(f"{ENRICHR}/enrich", timeout=timeout,
                             params={"userListId": uid, "backgroundType": lib})
            q.raise_for_status()
            for rec in q.json().get(lib, []):
                rows.append({"library": lib, "term": rec[1], "p": rec[2], "z": rec[3],
                             "combined_score": rec[4], "genes": ";".join(rec[5]),
                             "p_adj": rec[6]})
        return pd.DataFrame(rows)
    except Exception as exc:  # offline or API change
        print(f"  [enrichment skipped] {type(exc).__name__}: {exc}")
        return None
