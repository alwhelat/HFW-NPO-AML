"""
Step 3 - Prognosis: overall survival.

Discovery cohort (GSE37642, U133A+B): repeated CV of HFW-NPO-AML vs
baselines with the C-index. The final signature is then transferred to the
independent validation cohorts (GSE37642 U133 Plus 2, TCGA-LAML, BeatAML when
available): C-index, Kaplan-Meier (median split), log-rank test and a
multivariable Cox model adjusted for age. Univariate hazard ratios of PD-1
and PD-L1 are reported for every cohort.
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
from hfwnpo_aml.pipeline import PRIMARY, fit_final_parallel, run_survival_cv  # noqa: E402
from hfwnpo_aml.report import (reproducibility_info, save_cv, slug,  # noqa: E402
                               write_json, zscore_dataset)
from hfwnpo_aml.stats import bootstrap_ci  # noqa: E402
from hfwnpo_aml.survival import (concordance_index, logrank_test,  # noqa: E402
                                 multivariable_cox, univariate_cox)

EXTERNAL_METHODS = PRIMARY + ["Cox-LASSO", "PD-1/PD-L1 only"]


def anchor_hazard_table() -> pd.DataFrame:
    rows = []
    cohorts = [config.PROGNOSIS_DISCOVERY] + config.PROGNOSIS_VALIDATION
    for coh in cohorts:
        if not datasets.available(coh):
            continue
        X, ph = datasets.load(coh)
        for g in config.ANCHOR_GENES + ["PDCD1LG2"]:
            if g not in X.columns:
                continue
            r = univariate_cox(X[g].values, ph["time"].values, ph["event"].values)
            _, p_lr = logrank_test(ph["time"].values, ph["event"].values,
                                   (X[g] > X[g].median()).astype(float).values)
            rows.append(dict(cohort=coh, gene=g, n=len(ph), events=int(ph["event"].sum()),
                             logrank_p_median_split=p_lr, label=f"{g} | {coh}", **r))
    return pd.DataFrame(rows)


def main(methods=None):
    out = config.RESULTS_DIR / "prognosis"
    fig = config.FIGURES_DIR
    out.mkdir(parents=True, exist_ok=True)

    hz = anchor_hazard_table()
    if len(hz):
        hz.to_csv(out / "pd1_pdl1_univariate_cox.csv", index=False)
        plots.forest_plot(hz, fig / "prognosis_pd1_pdl1_forest")
        print(hz[["cohort", "gene", "HR", "CI_low", "CI_high", "p"]].round(4).to_string(index=False))

    disc = config.PROGNOSIS_DISCOVERY
    if not datasets.available(disc):
        print(f"[prognosis] {disc} not prepared - run 01_prepare_data.py first")
        return
    X, ph = datasets.load(disc)
    t, e = ph["time"].values, ph["event"].values.astype(int)
    log(f"[prognosis] discovery {disc}: {X.shape[0]} patients, {int(e.sum())} deaths, "
          f"{X.shape[1]} genes")
    Xz = zscore_dataset(X)
    genes = list(Xz.columns)
    rows, preds = run_survival_cv(Xz.values, t, e, genes, methods)
    save_cv(rows, preds, out, fig, ["C_index"], "C_index", "prognosis")

    # final signatures + external validation: all final models in parallel
    primaries = [m for m in PRIMARY if not methods or m in methods]
    ext_methods = (methods and [m for m in EXTERNAL_METHODS if m in methods]) or EXTERNAL_METHODS
    specs = [(("discovery", m), Xz.values, (t, e), genes, m) for m in primaries]
    cohorts = {}
    for coh in config.PROGNOSIS_VALIDATION:
        if not datasets.available(coh):
            continue
        Xv, phv = datasets.load(coh)
        shared = [g for g in Xz.columns if g in Xv.columns]
        cohorts[coh] = (zscore_dataset(Xv[shared]), phv, shared)
        specs += [((coh, m), Xz[shared].values, (t, e), shared, m) for m in ext_methods]
    log(f"training {len(specs)} final models in parallel (full discovery cohort) ...")
    fits = fit_final_parallel("surv", specs)

    coef_rows = []
    for m in primaries:
        final = fits[("discovery", m)]
        coef_rows += [dict(method=m, gene=g, coef_per_SD=c)
                      for g, c in zip(final["genes"], final["model"].coef_)]
        joblib.dump({"method": m, "model": final["model"], "genes": final["genes"],
                     "scaler": final["scaler"], "cols": final["cols"], "discovery": disc},
                    out / f"final_model_{slug(m)}.joblib")
        print(f"\n  Final {m} signature ({len(final['genes'])} genes): {', '.join(final['genes'])}")
    pd.DataFrame(coef_rows).to_csv(out / "final_signature_coefficients.csv", index=False)

    ext_rows, mv_rows = [], []
    for coh, (Xvz, phv, shared) in cohorts.items():
        tv, ev = phv["time"].values, phv["event"].values.astype(int)
        for m in ext_methods:
            r = fits[(coh, m)]
            risk = r["model"].predict(r["scaler"].transform(Xvz.values)[:, r["cols"]])
            c = concordance_index(tv, ev, risk)
            lo, hi = bootstrap_ci(concordance_index, tv, ev, risk)
            _, p_lr = logrank_test(tv, ev, (risk > np.median(risk)).astype(float))
            ext_rows.append(dict(cohort=coh, method=m, n=len(tv), events=int(ev.sum()),
                                 n_genes=len(r["cols"]), C_index=c, C_CI_low=lo, C_CI_high=hi,
                                 logrank_p=p_lr, genes=";".join(r["genes"])))
            log(f"external {coh:<16} {m:<30} C={c:.3f} [{lo:.3f}-{hi:.3f}] log-rank p={p_lr:.3g}")
            if m in PRIMARY:
                plots.km_plot(tv, ev, risk, fig / f"prognosis_km_{coh}_{slug(m)}",
                              title=f"{coh}: {m} risk score")
                if "age" in phv.columns and phv["age"].notna().sum() > 30:
                    ok = phv["age"].notna().values       # complete cases only
                    mv = multivariable_cox(np.c_[risk[ok], phv["age"].values[ok]],
                                           ["Signature risk score", "Age"], tv[ok], ev[ok])
                    mv_rows += [dict(cohort=coh, method=m, **row) for row in mv]
    if ext_rows:
        pd.DataFrame(ext_rows).to_csv(out / "external_validation.csv", index=False)
    if mv_rows:
        pd.DataFrame(mv_rows).to_csv(out / "multivariable_cox_age_adjusted.csv", index=False)
    write_json({"discovery": disc, "n": int(len(t)), "events": int(e.sum()),
                "validation": list(cohorts), "settings": {
                    k: getattr(config, k) for k in ("FILTER_K", "NPO_CLANS", "NPO_FAMILIES",
                                                     "NPO_ITER", "ALPHA_FIT", "N_SPLITS",
                                                     "N_REPEATS", "SURVIVAL_POOL")},
                "reproducibility": reproducibility_info()}, out / "run_info.json")
    print(f"\n[prognosis] results in {out}")


if __name__ == "__main__":
    main()
