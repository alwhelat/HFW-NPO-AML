"""
Step 4 - Chemoresistance: BeatAML ex-vivo drug response.

(a) Spearman correlation of PD-1 / PD-L1 expression with drug AUC
    (higher AUC = more resistant) for every drug with enough samples.
(b) For each drug in config.DRUGS: resistant (top third of AUC) vs sensitive
    (bottom third) classification with the co-primary methods (HFW-NPO
    original, HFW-NPO-AML), the baselines and a monocytic-marker reference
    panel. BeatAML holds one specimen per patient (see datasets.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

import config  # noqa: E402
from hfwnpo_aml import datasets, plots  # noqa: E402
from hfwnpo_aml.progress import log  # noqa: E402
from hfwnpo_aml.pipeline import MONOCYTIC, PRIMARY, run_classification_cv  # noqa: E402
from hfwnpo_aml.report import (reproducibility_info, save_cv, write_json,  # noqa: E402
                               zscore_dataset)

METRICS = ["AUC", "BalAcc", "MCC", "F1"]
DRUG_METHODS = PRIMARY + ["GA", "PSO", "L1-logistic", "Filter only", "PD-1/PD-L1 only",
                          MONOCYTIC]
MIN_SAMPLES = 60


def main(methods=None):
    out = config.RESULTS_DIR / "chemoresistance"
    fig = config.FIGURES_DIR
    data = datasets.load_drug()
    if data is None:
        print("[chemoresistance] BeatAML drug data not prepared. Put "
              "beataml_probit_curve_fits_v4_dbgap.txt and beataml_wv1to4_clinical.xlsx "
              "in the raw data folder and re-run 01_prepare_data.py")
        return
    X, auc = data
    out.mkdir(parents=True, exist_ok=True)
    anchors = [g for g in config.ANCHOR_GENES + ["PDCD1LG2"] if g in X.columns]

    # (a) correlation screen over all drugs
    rows = []
    for drug in auc.columns:
        ok = auc[drug].notna()
        if ok.sum() < MIN_SAMPLES:
            continue
        r = {"drug": drug, "n": int(ok.sum())}
        for g in anchors:
            rho, p = spearmanr(X.loc[ok, g], auc.loc[ok, drug])
            r[f"rho_{g}"], r[f"p_{g}"] = rho, p
        rows.append(r)
    cor = pd.DataFrame(rows).set_index("drug")
    for g in anchors:
        cor[f"fdr_{g}"] = multipletests(cor[f"p_{g}"], method="fdr_bh")[1]
    cor.sort_values(f"rho_{anchors[-1] if 'CD274' not in anchors else 'CD274'}",
                    ascending=False).to_csv(out / "pd1_pdl1_drug_auc_correlation.csv")
    sel = [d for d in cor.index if d.lower() in {x.lower() for x in config.DRUGS}]
    if sel:
        plots.drug_correlation_plot(cor.loc[sel, [f"rho_{g}" for g in anchors if g != "PDCD1LG2"]],
                                    fig / "chemoresistance_pd1_pdl1_correlation")
    print(f"[chemoresistance] correlation screen: {len(cor)} drugs")

    # (b) resistant vs sensitive classification
    lower = {c.lower(): c for c in auc.columns}
    methods = methods or DRUG_METHODS
    run_drugs = {}
    for d in config.DRUGS:
        col = lower.get(d.lower())
        if col is None:
            print(f"  [skip] {d}: not in BeatAML inhibitor list")
            continue
        a = auc[col].dropna()
        lo, hi = a.quantile(config.DRUG_QUANTILES[0]), a.quantile(config.DRUG_QUANTILES[1])
        keep = a[(a <= lo) | (a >= hi)]
        if len(keep) < MIN_SAMPLES:
            print(f"  [skip] {d}: only {len(keep)} samples after tertile split")
            continue
        y = (keep >= hi).astype(int).values
        Xd = zscore_dataset(X.loc[keep.index])
        log(f"[chemoresistance] {d}: {len(y)} samples (resistant={y.sum()}, "
              f"sensitive={(y == 0).sum()})")
        run_drugs[d] = {"n": int(len(y)), "resistant": int(y.sum()),
                        "sensitive": int((y == 0).sum())}
        rows, preds = run_classification_cv(Xd.values, y, list(Xd.columns), methods)
        save_cv(rows, preds, out / d, fig, METRICS, "AUC", f"chemoresistance_{d}")
    write_json({"cohort": "BeatAML (one specimen per patient)", "n_patients": int(len(X)),
                "drugs": run_drugs, "settings": {
                    k: getattr(config, k) for k in ("FILTER_K", "NPO_CLANS", "NPO_FAMILIES",
                                                     "NPO_ITER", "ALPHA_FIT", "N_SPLITS",
                                                     "N_REPEATS", "DRUG_QUANTILES",
                                                     "MONOCYTIC_GENES")},
                "reproducibility": reproducibility_info()}, out / "run_info.json")
    print(f"\n[chemoresistance] results in {out}")


if __name__ == "__main__":
    main()
