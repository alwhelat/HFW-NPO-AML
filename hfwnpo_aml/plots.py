"""Publication figures (matplotlib, 300 dpi PNG + vector PDF)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from .survival import kaplan_meier, logrank_test  # noqa: E402

# categorical order (validated colour-blind-safe reference palette)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GREY = "#8c8b86"
INK = "#0b0b0b"
INK2 = "#52514e"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
    "axes.labelcolor": INK2, "axes.edgecolor": "#c8c7c2", "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#ececea", "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def method_bars(summary: pd.DataFrame, metric: str, primaries, path: Path, label=None):
    """primaries: method name or list of co-primary methods (coloured)."""
    primaries = [primaries] if isinstance(primaries, str) else list(primaries)
    s = summary.sort_values(f"{metric}_mean")
    fig, ax = plt.subplots(figsize=(6.2, 0.32 * len(s) + 1.0))
    colors = [C[primaries.index(m)] if m in primaries else GREY for m in s.index]
    ax.barh(s.index, s[f"{metric}_mean"], xerr=s[f"{metric}_sd"], color=colors,
            height=0.6, error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
    lo = max(0.0, float((s[f"{metric}_mean"] - s[f"{metric}_sd"]).min()) - 0.05)
    ax.set_xlim(lo, min(1.1, float((s[f"{metric}_mean"] + s[f"{metric}_sd"]).max()) + 0.07))
    for y, (v, sd) in enumerate(zip(s[f"{metric}_mean"], s[f"{metric}_sd"].fillna(0))):
        ax.text(v + sd + 0.004, y, f"{v:.3f}", va="center", ha="left", fontsize=8, color=INK)
    ax.set_xlabel(f"{label or metric} (mean ± SD over CV folds)")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def roc_plot(preds: pd.DataFrame, methods: list[str], path: Path, title=""):
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color=GREY)
    for i, m in enumerate(methods[:4]):
        d = preds[(preds["method"] == m) & (preds["repeat"] == 1)]
        if d.empty:
            continue
        fpr, tpr, _ = roc_curve(d["y"], d["proba"])
        from sklearn.metrics import roc_auc_score
        ax.plot(fpr, tpr, lw=2, color=C[i], label=f"{m} (AUC {roc_auc_score(d['y'], d['proba']):.3f})")
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right", fontsize=7.5)
    _save(fig, path)


def convergence_plot(folds: pd.DataFrame, methods: list[str], path: Path):
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for i, m in enumerate(methods[:4]):
        H = [h for h in folds.loc[folds["method"] == m, "history"] if len(h)]
        if not H:
            continue
        L = min(len(h) for h in H)
        A = np.array([h[:L] for h in H])
        mu = A.mean(0)
        ax.plot(np.arange(1, L + 1), mu, lw=2, color=C[i], label=m)
        ax.fill_between(np.arange(1, L + 1), A.min(0), A.max(0), color=C[i], alpha=0.12, lw=0)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best fitness (lower is better)")
    ax.legend()
    _save(fig, path)


def gene_frequency_plot(freq: pd.Series, anchors: list[str], path: Path, top=25):
    f = freq.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(4.8, 0.22 * len(f) + 0.9))
    colors = [C[1] if g in anchors else C[0] for g in f.index]
    ax.barh(f.index, f.values, color=colors, height=0.65)
    ax.set_xlabel("Selection frequency across outer folds")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="y", visible=False)
    ax.set_title("Orange = PD-1/PD-L1 anchors", loc="left", fontsize=8, color=INK2)
    _save(fig, path)


def km_plot(time, event, risk, path: Path, title=""):
    time, event, risk = map(np.asarray, (time, event, risk))
    high = risk > np.median(risk)
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    for grp, lab, col in ((~high, "Low risk", C[0]), (high, "High risk", C[1])):
        tt, ss = kaplan_meier(time[grp], event[grp])
        ax.step(tt, ss, where="post", lw=2, color=col, label=f"{lab} (n={grp.sum()})")
    _, p = logrank_test(time, event, high.astype(float))
    ax.set_xlabel("Months")
    ax.set_ylabel("Overall survival probability")
    ax.set_ylim(0, 1.02)
    ax.set_title(title, loc="left")
    ax.text(0.98, 0.95, f"log-rank p = {p:.2g}", transform=ax.transAxes, ha="right",
            va="top", color=INK)
    ax.legend(loc="lower left")
    _save(fig, path)
    return p


def forest_plot(tab: pd.DataFrame, path: Path, label_col="label"):
    t = tab.iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.6, 0.32 * len(t) + 1.0))
    y = np.arange(len(t))
    ax.hlines(y, t["CI_low"], t["CI_high"], color=INK2, lw=1.2)
    ax.plot(t["HR"], y, "s", color=C[0], ms=6)
    ax.axvline(1, ls="--", lw=0.8, color=GREY)
    ax.set_yticks(y, t[label_col])
    ax.set_xscale("log")
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([0.5, 0.75, 1, 1.5, 2])
    ax.set_xlim(min(0.5, t["CI_low"].min() * 0.9), max(2, t["CI_high"].max() * 1.1))
    ax.set_xlabel("Hazard ratio per 1 SD (95% CI)")
    for yy, (hr, p) in enumerate(zip(t["HR"], t["p"])):
        ax.text(1.01, yy, f"{hr:.2f}, p={p:.2g}", transform=ax.get_yaxis_transform(),
                va="center", fontsize=7.5, color=INK)
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def expression_boxplot(X: pd.DataFrame, y, genes: list[str], labels: tuple, path: Path):
    genes = [g for g in genes if g in X.columns]
    fig, axes = plt.subplots(1, len(genes), figsize=(2.3 * len(genes), 3.2), squeeze=False)
    y = np.asarray(y)
    from scipy.stats import mannwhitneyu
    for ax, g in zip(axes[0], genes):
        data = [X.loc[y == 0, g].values, X.loc[y == 1, g].values]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False)
        for patch, col in zip(bp["boxes"], (C[0], C[1])):
            patch.set_facecolor(col)
            patch.set_alpha(0.35)
            patch.set_edgecolor(col)
        for k, d in enumerate(data):
            ax.plot(np.random.normal(k + 1, 0.06, len(d)), d, ".", ms=3,
                    color=(C[0], C[1])[k], alpha=0.6)
        p = mannwhitneyu(*data).pvalue
        ax.set_xticks([1, 2], labels)
        ax.set_title(f"{g}\np = {p:.2g}", fontsize=9)
        ax.set_ylabel("log2 expression")
    fig.tight_layout()
    _save(fig, path)


def drug_correlation_plot(tab: pd.DataFrame, path: Path):
    """tab: rows = drugs, columns rho_<gene>, p_<gene>."""
    genes = [c[4:] for c in tab.columns if c.startswith("rho_")]
    fig, ax = plt.subplots(figsize=(5.2, 0.35 * len(tab) + 1.0))
    y = np.arange(len(tab))
    h = 0.8 / max(len(genes), 1)
    for i, g in enumerate(genes):
        ax.barh(y + i * h - 0.4 + h / 2, tab[f"rho_{g}"], height=h * 0.9, color=C[i], label=g)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, tab.index)
    ax.set_xlabel("Spearman rho with drug AUC (positive = more resistant)")
    ax.legend()
    ax.grid(axis="y", visible=False)
    _save(fig, path)


# ---------------------------------------------------------------------------
# post-hoc figures (scripts/05_post_analysis.py)
# ---------------------------------------------------------------------------
def cd_diagram(ranks: pd.Series, cd: float, primaries, path: Path, title=""):
    """Critical-difference diagram: mean rank per method (1 = best) with the
    Nemenyi critical difference drawn as a bar."""
    r = ranks.sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 0.34 * len(r) + 1.3))
    y = np.arange(len(r))[::-1]
    for yy, (m, v) in zip(y, r.items()):
        col = C[0] if m in primaries else GREY
        ax.plot([v], [yy], "o", color=col, ms=7)
        ax.text(v + 0.12, yy, f"{m} ({v:.2f})", va="center", fontsize=8, color=INK)
    best = r.iloc[0]
    ax.axvspan(best, best + cd, color=C[0], alpha=0.08, lw=0)
    ax.annotate("", xy=(best + cd, len(r) - 0.3), xytext=(best, len(r) - 0.3),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=1))
    ax.text(best + cd / 2, len(r) - 0.1, f"CD = {cd:.2f}", ha="center", fontsize=8, color=INK2)
    ax.set_yticks([])
    ax.set_xlabel("Mean rank across outer folds (lower is better)")
    ax.set_xlim(0.5, len(r) + 2.5)
    ax.set_ylim(-0.8, len(r) + 0.4)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def pareto_plot(summary: pd.DataFrame, metric: str, primaries, path: Path, label=None):
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for m, row in summary.iterrows():
        col = C[0] if m in primaries else GREY
        ax.errorbar(row["n_genes_mean"], row[f"{metric}_mean"], yerr=row[f"{metric}_sd"],
                    fmt="o", color=col, ms=6, elinewidth=0.8, capsize=2)
        ax.annotate(m, (row["n_genes_mean"], row[f"{metric}_mean"]), fontsize=7,
                    xytext=(4, 3), textcoords="offset points", color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("Signature size (mean number of genes, log scale)")
    ax.set_ylabel(f"{label or metric} (mean ± SD)")
    _save(fig, path)


def calibration_plot(curves: dict, path: Path, title=""):
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=0.8)
    for i, (name, cur) in enumerate(curves.items()):
        ax.plot(cur["p"], cur["y"], "o-", color=C[i % len(C)], lw=1.6, ms=4, label=name)
    ax.set_xlabel("Predicted probability of AML")
    ax.set_ylabel("Observed proportion")
    ax.set_title(title, loc="left")
    ax.legend(fontsize=7, loc="upper left")
    _save(fig, path)


def dca_plot(curves: dict, path: Path, title=""):
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    first = next(iter(curves.values()))
    ax.plot(first["threshold"], first["treat_all"], color=GREY, lw=1, ls="--", label="Treat all")
    ax.axhline(0, color=INK2, lw=0.8, label="Treat none")
    for i, (name, d) in enumerate(curves.items()):
        ax.plot(d["threshold"], d["net_benefit"], color=C[i % len(C)], lw=1.8, label=name)
    ymax = max(d["net_benefit"].max() for d in curves.values())
    ax.set_ylim(-0.05, ymax + 0.05)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title(title, loc="left")
    ax.legend(fontsize=7)
    _save(fig, path)


def td_auc_plot(tab: pd.DataFrame, path: Path):
    cohorts = list(dict.fromkeys(tab["cohort"]))
    fig, axes = plt.subplots(1, len(cohorts), figsize=(3.4 * len(cohorts), 3.2), squeeze=False)
    for ax, coh in zip(axes[0], cohorts):
        d = tab[tab["cohort"] == coh]
        for i, (m, dm) in enumerate(d.groupby("method", sort=False)):
            ax.plot(dm["months"], dm["tdAUC"], "o-", color=C[i % len(C)], lw=1.6, ms=4, label=m)
        ax.axhline(0.5, color=GREY, ls="--", lw=0.8)
        ax.set_title(coh, fontsize=9, loc="left")
        ax.set_xlabel("Months")
        ax.set_ylabel("Time-dependent AUC")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    _save(fig, path)


def meta_forest(tab: pd.DataFrame, pooled: dict, path: Path, xlabel, log=True, ref=1.0):
    t = tab.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(5.6, 0.36 * (len(t) + 1) + 1.0))
    y = np.arange(1, len(t) + 1)
    ax.hlines(y, t["ci_low"], t["ci_high"], color=INK2, lw=1.2)
    ax.plot(t["est"], y, "s", color=C[0], ms=6)
    ax.plot([pooled["est"]], [0], "D", color=C[1], ms=8)
    ax.hlines(0, pooled["ci_low"], pooled["ci_high"], color=C[1], lw=2)
    ax.axvline(ref, ls="--", color=GREY, lw=0.8)
    ax.set_yticks(list(y) + [0], list(t["label"]) +
                  [f"Pooled (random effects), I² = {pooled['I2']:.0f}%"])
    if log:
        ax.set_xscale("log")
        from matplotlib.ticker import NullFormatter, ScalarFormatter
        lo = min(t["ci_low"].min(), pooled["ci_low"]) * 0.95
        hi = max(t["ci_high"].max(), pooled["ci_high"]) * 1.05
        ticks = [v for v in (0.25, 0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0) if lo <= v <= hi]
        ax.set_xlim(min(lo, 0.95), max(hi, 1.05))
        ax.set_xticks(ticks or [1.0])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
    for yy, (est, l, h) in zip(y, zip(t["est"], t["ci_low"], t["ci_high"])):
        ax.text(1.01, yy, f"{est:.2f} [{l:.2f}-{h:.2f}]", transform=ax.get_yaxis_transform(),
                va="center", fontsize=7.5, color=INK)
    ax.set_xlabel(xlabel)
    ax.text(1.01, 0, f"{pooled['est']:.2f} [{pooled['ci_low']:.2f}-{pooled['ci_high']:.2f}], "
            f"p = {pooled['p']:.2g}", transform=ax.get_yaxis_transform(),
            va="center", fontsize=8, color=INK)
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def importance_plot(imp: pd.DataFrame, path: Path, title="", anchors=()):
    d = imp.sort_values("importance_mean").tail(25)
    fig, ax = plt.subplots(figsize=(4.8, 0.24 * len(d) + 1.0))
    cols = [C[1] if g in anchors else C[0] for g in d.index]
    ax.barh(d.index, d["importance_mean"], xerr=d["importance_sd"], color=cols, height=0.65,
            error_kw=dict(ecolor=INK2, lw=0.7, capsize=1.5))
    ax.set_xlabel("Permutation importance (drop in AUC)")
    ax.set_title(title, loc="left", fontsize=9)
    ax.grid(axis="y", visible=False)
    _save(fig, path)
