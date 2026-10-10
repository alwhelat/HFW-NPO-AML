"""Helpers that write the CV results of one experiment to disk."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
from . import plots
from .pipeline import PRIMARY
from .stats import compare_methods, gene_frequency, summarise


def slug(method: str) -> str:
    """File-name friendly method name, e.g. 'HFW-NPO (original)' -> 'HFW-NPO_original'."""
    return "".join(ch if ch.isalnum() or ch == "-" else "_" for ch in method).strip("_").replace("__", "_")


def zscore_dataset(X: pd.DataFrame) -> pd.DataFrame:
    """Per-gene standardisation within one cohort (removes platform offsets
    so that models can be transferred between platforms)."""
    sd = X.std().replace(0, 1.0)
    return (X - X.mean()) / sd


def save_cv(rows, preds, out: Path, fig: Path, metrics: list[str], main_metric: str,
            label: str) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    hist = {f"{r['method']}|rep{r['repeat']}|fold{r['fold']}": r["history"] for r in rows}
    (out / "convergence_histories.json").write_text(json.dumps(hist))
    df.drop(columns="history").to_csv(out / "fold_metrics.csv", index=False)
    pd.DataFrame(preds).to_csv(out / "oof_predictions.csv", index=False)

    summary = summarise(df, metrics)
    summary.to_csv(out / "summary.csv")
    stats = pd.concat([compare_methods(df, m, PRIMARY).assign(metric=m) for m in metrics
                       if m in df.columns], ignore_index=True)
    stats.to_csv(out / "statistical_tests.csv", index=False)

    primaries = [m for m in PRIMARY if m in set(df["method"])]
    freqs = []
    for m in primaries:
        freq = gene_frequency(df, m)
        freqs.append(freq.rename_axis("gene").reset_index().assign(method=m))
        plots.gene_frequency_plot(freq, config.ANCHOR_GENES,
                                  fig / f"{label}_gene_frequency_{slug(m)}")
    if freqs:
        pd.concat(freqs, ignore_index=True)[["method", "gene", "frequency"]].to_csv(
            out / "gene_selection_frequency.csv", index=False)

    plots.method_bars(summary, main_metric, primaries, fig / f"{label}_method_comparison",
                      label=main_metric.replace("_", "-"))
    plots.convergence_plot(df, primaries + ["GA", "PSO"], fig / f"{label}_convergence")

    print("\n  Summary (mean over folds):")
    cols = [c for c in summary.columns if c.endswith("_mean")]
    print(summary[cols].to_string())
    return df


def reproducibility_info() -> dict:
    """Git commit of the code and versions of Python and the key libraries."""
    import platform
    import subprocess
    from importlib.metadata import PackageNotFoundError, version

    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=config.PROJECT_ROOT, capture_output=True,
                                  text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    pkgs = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn",
                 "statsmodels", "joblib", "matplotlib"):
        try:
            pkgs[name] = version(name)
        except PackageNotFoundError:
            pkgs[name] = None
    return {"git_commit": git("rev-parse", "HEAD") or None,
            "git_uncommitted_changes": bool(git("status", "--porcelain", "--untracked-files=no")),
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": pkgs}


def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    path.write_text(json.dumps(obj, indent=2, default=conv))
