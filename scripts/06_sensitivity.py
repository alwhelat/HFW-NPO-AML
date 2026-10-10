"""
Step 6 - sensitivity analysis of the two HFW-NPO settings that a reviewer is
most likely to question: the filter size K and the fitness weight alpha.

Only the two co-primary methods are rerun, on the discovery cohorts of the
diagnosis and prognosis tasks, with 5-fold CV (one repeat). The reference
setting (K = 80, alpha = 0.85) comes from the main run. Each setting is
varied on its own while the other stays at the reference value.

    python scripts/06_sensitivity.py --tag v2            # about 2-4 h on 16 cores
    python scripts/06_sensitivity.py --tag v2 --quick    # code check only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

import config  # noqa: E402
from hfwnpo_aml import datasets  # noqa: E402
from hfwnpo_aml.pipeline import PRIMARY, run_classification_cv, run_survival_cv  # noqa: E402
from hfwnpo_aml.progress import log  # noqa: E402
from hfwnpo_aml.report import zscore_dataset  # noqa: E402

K_VALUES = [40, 120]
ALPHA_VALUES = [0.75, 0.95]


def settings(quick: bool):
    ref_k, ref_a = config.FILTER_K, config.ALPHA_FIT
    grid = [("K", k, k, ref_a) for k in K_VALUES] + [("alpha", a, ref_k, a) for a in ALPHA_VALUES]
    if quick:
        grid = grid[:1] + grid[2:3]
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args()
    if args.quick:
        config.apply_quick_mode()
    res = config.PROJECT_ROOT / f"results_{args.tag}"
    out = res / ("sensitivity_quick" if args.quick else "sensitivity")
    out.mkdir(parents=True, exist_ok=True)
    config.LOG_FILE = res / "progress.log"
    if args.jobs:
        config.N_JOBS = args.jobs
    config.N_REPEATS = 1

    Xd, phd = datasets.load(config.DIAGNOSIS_DISCOVERY)
    Xdz, yd = zscore_dataset(Xd), phd["label"].values.astype(int)
    Xp, php = datasets.load(config.PROGNOSIS_DISCOVERY)
    Xpz = zscore_dataset(Xp)
    tp, ep = php["time"].values, php["event"].values.astype(int)

    rows = []
    for param, value, k, alpha in settings(args.quick):
        config.FILTER_K, config.ALPHA_FIT = k, alpha
        log(f"[sensitivity] {param} = {value} (K={k}, alpha={alpha})")
        r1, _ = run_classification_cv(Xdz.values, yd, list(Xdz.columns), PRIMARY)
        r2, _ = run_survival_cv(Xpz.values, tp, ep, list(Xpz.columns), PRIMARY)
        for task, rr, metric in (("diagnosis", r1, "AUC"), ("prognosis", r2, "C_index")):
            df = pd.DataFrame(rr)
            for m, d in df.groupby("method"):
                rows.append({"task": task, "parameter": param, "value": value, "K": k,
                             "alpha": alpha, "method": m, "metric": metric,
                             "mean": d[metric].mean(), "sd": d[metric].std(),
                             "n_genes_mean": d["n_genes"].mean()})
        pd.DataFrame(rows).to_csv(out / "sensitivity.csv", index=False)
    log(f"[sensitivity] finished: {out / 'sensitivity.csv'}")


if __name__ == "__main__":
    main()
