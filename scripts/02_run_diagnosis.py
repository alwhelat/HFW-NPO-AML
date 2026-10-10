"""
Step 2 - Diagnosis: AML vs non-leukaemic bone marrow.

Discovery cohort (GSE13159): repeated stratified CV comparing the co-primary
methods HFW-NPO (original) and HFW-NPO-AML with the baselines. External
cohorts (GSE114868, GSE9476): a final model trained on the whole discovery
cohort is applied unchanged to each validation cohort. All final models are
fitted in parallel.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from hfwnpo_aml import datasets, plots  # noqa: E402
from hfwnpo_aml.progress import log  # noqa: E402
from hfwnpo_aml.pipeline import (CLASSIFICATION_METHODS, MONOCYTIC, PRIMARY,  # noqa: E402
                                 classification_metrics, fit_final_parallel,
                                 run_classification_cv)
from hfwnpo_aml.report import (reproducibility_info, save_cv, slug,  # noqa: E402
                               write_json, zscore_dataset)
from hfwnpo_aml.stats import bootstrap_ci  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

METRICS = ["AUC", "BalAcc", "MCC", "F1", "Sensitivity", "Specificity"]
EXTERNAL_METHODS = PRIMARY + ["L1-logistic", "PD-1/PD-L1 only"]
# the monocytic reference panel belongs to the chemoresistance experiment only
DIAGNOSIS_METHODS = [m for m in CLASSIFICATION_METHODS if m != MONOCYTIC]


def main(methods=None):
    out = config.RESULTS_DIR / "diagnosis"
    fig = config.FIGURES_DIR

    candidates = [config.DIAGNOSIS_DISCOVERY] + config.DIAGNOSIS_VALIDATION
    disc = next((c for c in candidates if datasets.available(c)), None)
    if disc is None:
        print("[diagnosis] no diagnosis cohort prepared - run 01_prepare_data.py first")
        return
    if disc != config.DIAGNOSIS_DISCOVERY:
        print(f"[diagnosis] WARNING: {config.DIAGNOSIS_DISCOVERY} not available, "
              f"using {disc} as discovery cohort")
    X, ph = datasets.load(disc)
    y = ph["label"].values.astype(int)
    missing = [g for g in config.ANCHOR_GENES if g not in X.columns]
    log(f"[diagnosis] discovery {disc}: {X.shape[0]} samples, {X.shape[1]} genes, "
          f"AML={y.sum()} control={(y == 0).sum()}"
          + (f" | anchors missing: {missing}" if missing else ""))

    plots.expression_boxplot(X, y, config.ANCHOR_GENES + ["PDCD1LG2"], ("Control", "AML"),
                             fig / f"diagnosis_{disc}_anchor_expression")

    Xz = zscore_dataset(X)
    genes = list(Xz.columns)
    rows, preds = run_classification_cv(Xz.values, y, genes, methods or DIAGNOSIS_METHODS)
    df = save_cv(rows, preds, out, fig, METRICS, "AUC", "diagnosis")
    top = df.groupby("method")["AUC"].mean().sort_values(ascending=False).index.tolist()
    order = [m for m in PRIMARY if m in top] + [m for m in top if m not in PRIMARY]
    plots.roc_plot(pd.DataFrame(preds), order, fig / "diagnosis_roc_cv",
                   title=f"{disc} - out-of-fold ROC (repeat 1)")

    # ------------------------------------------------------------------
    # final signatures (full discovery cohort) + external validation:
    # every final model is an independent job, all run in parallel
    # ------------------------------------------------------------------
    primaries = [m for m in PRIMARY if not methods or m in methods]
    ext_methods = (methods and [m for m in EXTERNAL_METHODS if m in methods]) or EXTERNAL_METHODS
    specs = [(("discovery", m), Xz.values, y, genes, m) for m in primaries]
    cohorts = {}
    for coh in candidates:
        if coh == disc or not datasets.available(coh):
            continue
        Xv, phv = datasets.load(coh)
        shared = [g for g in Xz.columns if g in Xv.columns]
        cohorts[coh] = (zscore_dataset(Xv[shared]), phv["label"].values.astype(int), shared)
        specs += [((coh, m), Xz[shared].values, y, shared, m) for m in ext_methods]
    log(f"training {len(specs)} final models in parallel (full discovery cohort) ...")
    fits = fit_final_parallel("clf", specs)

    sig_rows = []
    for m in primaries:
        final = fits[("discovery", m)]
        sig_rows += [dict(method=m, gene=g) for g in final["genes"]]
        joblib.dump({"method": m, "model": final["model"], "genes": final["genes"],
                     "scaler": final["scaler"], "cols": final["cols"], "discovery": disc},
                    out / f"final_model_{slug(m)}.joblib")
        print(f"\n  Final {m} signature ({len(final['genes'])} genes): {', '.join(final['genes'])}")
    pd.DataFrame(sig_rows).to_csv(out / "final_signature_genes.csv", index=False)

    ext_rows = []
    for coh, (Xvz, yv, shared) in cohorts.items():
        for m in ext_methods:
            r = fits[(coh, m)]
            proba = r["model"].predict_proba(
                np.clip(r["scaler"].transform(Xvz.values), -1, 2)[:, r["cols"]])[:, 1]
            met = classification_metrics(yv, proba)
            lo, hi = bootstrap_ci(roc_auc_score, yv, proba, stratify=yv)
            ext_rows.append(dict(cohort=coh, method=m, n=len(yv), n_genes=len(r["cols"]),
                                 AUC_CI_low=lo, AUC_CI_high=hi, genes=";".join(r["genes"]),
                                 anchors_missing=";".join(g for g in config.ANCHOR_GENES
                                                          if g not in shared), **met))
            log(f"external {coh:<10} {m:<30} AUC={met['AUC']:.3f} [{lo:.3f}-{hi:.3f}]")
    if ext_rows:
        pd.DataFrame(ext_rows).to_csv(out / "external_validation.csv", index=False)
    write_json({"discovery": disc, "n": int(len(y)), "settings": {
        k: getattr(config, k) for k in ("FILTER_K", "NPO_CLANS", "NPO_FAMILIES", "NPO_ITER",
                                         "ALPHA_FIT", "N_SPLITS", "N_REPEATS", "CANDIDATE_MODE")},
                "validation": [c for c in cohorts], "reproducibility": reproducibility_info()},
               out / "run_info.json")
    print(f"\n[diagnosis] results in {out}")


if __name__ == "__main__":
    main()
