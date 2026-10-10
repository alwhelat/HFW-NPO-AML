"""Statistical comparison of methods and summary tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests


def summarise(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    g = df.groupby("method", sort=False)
    out = pd.DataFrame(index=g.size().index)
    for m in metrics:
        out[f"{m}_mean"] = g[m].mean()
        out[f"{m}_sd"] = g[m].std()
    out["n_genes_mean"] = g["n_genes"].mean()
    out["n_genes_sd"] = g["n_genes"].std()
    out["selection_seconds_mean"] = g["selection_seconds"].mean()
    return out.round(4)


def _wilcoxon(a, b):
    d = a - b
    if np.allclose(d, 0):
        return np.nan, 1.0
    return wilcoxon(a, b, alternative="two-sided", zero_method="zsplit")


def compare_methods(df: pd.DataFrame, metric: str, primaries: list[str]) -> pd.DataFrame:
    """Friedman test over all methods, then two-sided paired Wilcoxon tests of
    each co-primary method against every other method and of the co-primary
    methods against each other. Holm correction over all pairwise tests of
    the metric. mean_diff = first method minus second method."""
    wide = df.pivot_table(index=["repeat", "fold"], columns="method", values=metric)
    wide = wide.dropna(axis=1, how="any")
    rows = []
    if wide.shape[1] >= 3:
        stat, p = friedmanchisquare(*[wide[c].values for c in wide.columns])
        ranks = wide.rank(axis=1, ascending=False).mean()
        rows.append({"comparison": "Friedman (all methods)", "family": "global",
                     "statistic": stat, "p": p, "p_holm": np.nan, "mean_diff": np.nan})
    else:
        ranks = pd.Series(dtype=float)
    prim = [m for m in primaries if m in wide.columns]
    pairs = []
    for i, a in enumerate(prim):                       # head-to-head first
        for b in prim[i + 1:]:
            pairs.append((a, b, "primary vs primary"))
    for a in prim:
        for b in wide.columns:
            if b not in prim:
                pairs.append((a, b, f"{a} vs others"))
    tmp = []
    for a, b, fam in pairs:
        stat, p = _wilcoxon(wide[a].values, wide[b].values)
        tmp.append({"comparison": f"{a} vs {b}", "family": fam, "statistic": stat, "p": p,
                    "mean_diff": float((wide[a] - wide[b]).mean())})
    if tmp:
        holm = multipletests([r["p"] for r in tmp], method="holm")[1]
        for r, ph in zip(tmp, holm):
            r["p_holm"] = ph
        rows += tmp
    res = pd.DataFrame(rows)
    if len(ranks):
        rank_row = pd.DataFrame({"comparison": [f"mean rank: {k}" for k in ranks.index],
                                 "family": "rank", "statistic": ranks.values})
        res = pd.concat([res, rank_row], ignore_index=True)
    return res


def bootstrap_ci(fn, *arrays, n_boot=1000, seed=42, stratify=None):
    """Percentile 95% CI of a statistic fn(*arrays) by resampling rows."""
    rng = np.random.RandomState(seed)
    n = len(arrays[0])
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if stratify is not None and len(np.unique(np.asarray(stratify)[idx])) < 2:
            continue
        try:
            vals.append(fn(*[np.asarray(a)[idx] for a in arrays]))
        except Exception:
            continue
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def gene_frequency(df: pd.DataFrame, method: str) -> pd.Series:
    genes = df.loc[df["method"] == method, "genes"].str.split(";")
    n = len(genes)
    return (pd.Series([g for lst in genes for g in lst]).value_counts() / n).rename("frequency")
