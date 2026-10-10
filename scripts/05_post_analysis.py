"""
Step 5 - additional statistics and biology for the manuscript.

Runs on the results of steps 2-4 (no CV is repeated, nothing is re-tuned):

  A. CV comparisons     Nemenyi critical-difference diagrams, bootstrap 95% CI
                        of paired differences, accuracy-vs-size (Pareto) plots,
                        stability of the selected gene sets (Jaccard vs chance)
  B. PD-1/PD-L1 axis    random-effects meta-analysis of differential expression
                        (Hedges g, 3 cohorts) and of overall-survival hazard
                        ratios (4 cohorts); drug-response correlations adjusted
                        for a monocytic score; PD-L1 by ELN-2017 risk (BeatAML)
  C. External cohorts   the external-validation models are refitted exactly as
                        in steps 2-3 (same seeds) to keep their predictions,
                        then: DeLong CIs and paired tests, calibration (Brier,
                        intercept, slope), decision curves, IPCW time-dependent
                        AUC at 12/24/36 months, paired bootstrap of C-index,
                        multivariable Cox (age, ELN-2017), meta-analysis of the
                        signature hazard ratio, permutation importance
  D. Enrichment         GO / KEGG / Reactome over-representation (Enrichr API);
                        HFW-NPO-AML lists are run with and without the forced
                        anchors (PDCD1, CD274), and terms whose overlap is only
                        anchor genes are flagged (driven_by_anchors_only)
  E. Workbook           every table in results_<tag>/post/manuscript_tables.xlsx

    python scripts/05_post_analysis.py --tag v2
    python scripts/05_post_analysis.py --tag v2 --skip-external     # fast parts only
    python scripts/05_post_analysis.py --tag v2 --enrichment-only   # section D only, other
                                                                    # workbook sheets kept
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kruskal, spearmanr  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

import config  # noqa: E402
from hfwnpo_aml import datasets, plots  # noqa: E402
from hfwnpo_aml import post as P  # noqa: E402
from hfwnpo_aml.pipeline import PRIMARY, fit_final_parallel  # noqa: E402
from hfwnpo_aml.progress import log  # noqa: E402
from hfwnpo_aml.report import slug, write_json, zscore_dataset  # noqa: E402
from hfwnpo_aml.stats import gene_frequency  # noqa: E402
from hfwnpo_aml.survival import (concordance_index, multivariable_cox,  # noqa: E402
                                 univariate_cox)

TABLES: dict[str, pd.DataFrame] = {}


def keep(name: str, df: pd.DataFrame, out: Path, index=False):
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=index)
    TABLES[name] = df.reset_index() if index else df
    return df


# ===========================================================================
# A. cross-validation comparisons
# ===========================================================================
def cv_tasks(res: Path):
    tasks = [("diagnosis", res / "diagnosis", "AUC"),
             ("prognosis", res / "prognosis", "C_index")]
    chem = res / "chemoresistance"
    if chem.exists():
        for d in sorted(p for p in chem.iterdir() if (p / "fold_metrics.csv").exists()):
            tasks.append((f"chemoresistance_{d.name}", d, "AUC"))
    return [t for t in tasks if (t[1] / "fold_metrics.csv").exists()]


def section_a(res: Path, out: Path, fig: Path):
    log("[post] A. CV rankings, effect sizes, Pareto and stability")
    rank_rows, eff_rows, stab_rows = [], [], []
    n_genes_total = {"diagnosis": None, "prognosis": None}
    for task, d, metric in cv_tasks(res):
        df = pd.read_csv(d / "fold_metrics.csv")
        ranks, n_blocks = P.mean_ranks(df, metric)
        cd = P.nemenyi_cd(len(ranks), n_blocks)
        plots.cd_diagram(ranks, cd, PRIMARY, fig / f"post_cd_{task}",
                         title=f"{task}: {metric} (Nemenyi, alpha = 0.05)")
        best = ranks.iloc[0]
        rank_rows += [{"task": task, "metric": metric, "method": m, "mean_rank": r,
                       "CD": cd, "within_CD_of_best": bool(r - best <= cd)}
                      for m, r in ranks.items()]
        wide = df.pivot_table(index=["repeat", "fold"], columns="method", values=metric)
        for a in [p for p in PRIMARY if p in wide.columns]:
            for b in wide.columns:
                if b == a:
                    continue
                diff, lo, hi = P.bootstrap_paired_diff(wide[a], wide[b])
                eff_rows.append({"task": task, "metric": metric, "method": a, "vs": b,
                                 "mean_diff": diff, "ci_low": lo, "ci_high": hi,
                                 "excludes_zero": bool(lo > 0 or hi < 0)})
        summ = pd.read_csv(d / "summary.csv", index_col=0)
        plots.pareto_plot(summ, metric, PRIMARY, fig / f"post_pareto_{task}",
                          label=metric.replace("_", "-"))
        info = d / "run_info.json"
        if not info.exists():
            info = d.parent / "run_info.json"
        n_feat = (pd.read_json(info, typ="series").get("settings", {}) or {}).get(
            "FILTER_K", 80) if info.exists() else 80
        st = P.selection_stability(df, n_feat).reset_index()
        st.insert(0, "task", task)
        stab_rows.append(st)
    keep("A1_mean_ranks_nemenyi", pd.DataFrame(rank_rows), out)
    keep("A2_paired_differences_bootstrap", pd.DataFrame(eff_rows), out)
    keep("A3_selection_stability", pd.concat(stab_rows, ignore_index=True), out)


# ===========================================================================
# B. PD-1 / PD-L1 axis
# ===========================================================================
AXIS = ["PDCD1", "CD274", "PDCD1LG2"]


def section_b(out: Path, fig: Path):
    log("[post] B. PD-1/PD-L1 meta-analyses and drug correlations")
    # B1 differential expression (AML - control), Hedges g
    rows = []
    for coh in [config.DIAGNOSIS_DISCOVERY] + config.DIAGNOSIS_VALIDATION:
        if not datasets.available(coh):
            continue
        X, ph = datasets.load(coh)
        y = ph["label"].values
        for g in AXIS:
            if g in X.columns:
                gg, var = P.hedges_g(X.loc[y == 1, g], X.loc[y == 0, g])
                rows.append({"gene": g, "cohort": coh, "n_AML": int(y.sum()),
                             "n_control": int((y == 0).sum()), "hedges_g": gg, "var": var,
                             "ci_low": gg - 1.96 * np.sqrt(var), "ci_high": gg + 1.96 * np.sqrt(var)})
    de = pd.DataFrame(rows)
    meta_rows = []
    for g, d in de.groupby("gene"):
        if len(d) >= 2:
            re = P.random_effects(d["hedges_g"], d["var"])
            meta_rows.append({"gene": g, "analysis": "differential expression (Hedges g)", **re})
            t = d.assign(est=d["hedges_g"], label=d["cohort"])
            plots.meta_forest(t, {"est": re["pooled"], **re}, fig / f"post_meta_DE_{g}",
                              f"{g}: Hedges g, AML minus control", log=False, ref=0.0)
    keep("B1_pd1_pdl1_differential_expression", de, out)

    # B2 overall survival, log HR per SD
    rows = []
    for coh in [config.PROGNOSIS_DISCOVERY] + config.PROGNOSIS_VALIDATION:
        if not datasets.available(coh):
            continue
        X, ph = datasets.load(coh)
        for g in AXIS:
            if g in X.columns:
                r = univariate_cox(X[g].values, ph["time"].values, ph["event"].values)
                lhr = np.log(r["HR"])
                se = (np.log(r["CI_high"]) - np.log(r["CI_low"])) / (2 * 1.96)
                rows.append({"gene": g, "cohort": coh, "n": len(ph), "events": int(ph["event"].sum()),
                             "HR": r["HR"], "ci_low": r["CI_low"], "ci_high": r["CI_high"],
                             "p": r["p"], "logHR": lhr, "var": se ** 2})
    hz = pd.DataFrame(rows)
    for g, d in hz.groupby("gene"):
        if len(d) >= 2:
            re = P.random_effects(d["logHR"], d["var"])
            meta_rows.append({"gene": g, "analysis": "overall survival (pooled HR per SD)",
                              **{**re, "pooled": np.exp(re["pooled"]),
                                 "ci_low": np.exp(re["ci_low"]), "ci_high": np.exp(re["ci_high"])}})
            t = d.assign(est=d["HR"], label=d["cohort"])
            plots.meta_forest(t, {"est": np.exp(re["pooled"]), "ci_low": np.exp(re["ci_low"]),
                                  "ci_high": np.exp(re["ci_high"]), "I2": re["I2"], "p": re["p"]},
                              fig / f"post_meta_OS_{g}", f"{g}: hazard ratio per 1 SD")
    keep("B2_pd1_pdl1_survival_by_cohort", hz, out)
    keep("B3_pd1_pdl1_meta_analysis", pd.DataFrame(meta_rows), out)

    # B4 drug response: raw and monocytic-adjusted Spearman correlations
    data = datasets.load_drug()
    if data is not None:
        X, auc = data
        mono_genes = [g for g in config.MONOCYTIC_GENES if g in X.columns]
        mono = zscore_dataset(X[mono_genes]).mean(1)
        rows = []
        for drug in auc.columns:
            ok = auc[drug].notna().values
            if ok.sum() < 60:
                continue
            for g in [a for a in AXIS if a in X.columns]:
                rho, p = spearmanr(X.loc[ok, g], auc.loc[ok, drug])
                prho, pp = P.partial_spearman(X.loc[ok, g].values, auc.loc[ok, drug].values,
                                              mono[ok].values)
                rows.append({"drug": drug, "gene": g, "n": int(ok.sum()), "rho": rho, "p": p,
                             "partial_rho_adj_monocytic": prho, "partial_p": pp})
        dc = pd.DataFrame(rows)
        for col, new in (("p", "fdr"), ("partial_p", "partial_fdr")):
            dc[new] = np.nan
            for g, idx in dc.groupby("gene").groups.items():
                dc.loc[idx, new] = multipletests(dc.loc[idx, col], method="fdr_bh")[1]
        main = dc[dc["drug"].str.lower().isin([d.lower() for d in config.DRUGS])]
        keep("B4_drug_correlations_all", dc.sort_values("p"), out)
        keep("B5_drug_correlations_study_drugs", main, out)

    # B6 PD-L1 by ELN-2017 risk group (BeatAML)
    clin_p = next(iter(sorted(config.RAW_DIR.glob("beataml*clinical*.xlsx"))), None)
    if clin_p is not None and datasets.available("BeatAML"):
        clin = pd.read_excel(clin_p)
        X, ph = datasets.load("BeatAML")
        if "dbgap_rnaseq_sample" in clin.columns and "ELN2017" in clin.columns:
            eln = clin.dropna(subset=["dbgap_rnaseq_sample"]).drop_duplicates("dbgap_rnaseq_sample") \
                      .set_index("dbgap_rnaseq_sample")["ELN2017"]
            eln = eln.reindex(X.index)
            groups = {k: v for k, v in eln.groupby(eln).groups.items()
                      if k in ("Favorable", "Intermediate", "Adverse") and len(v) >= 10}
            rows = []
            for g in [a for a in AXIS if a in X.columns]:
                vals = [X.loc[idx, g].values for idx in groups.values()]
                if len(vals) >= 2:
                    h, p = kruskal(*vals)
                    rows.append({"gene": g, **{f"median_{k}": float(np.median(X.loc[i, g]))
                                               for k, i in groups.items()},
                                 **{f"n_{k}": len(i) for k, i in groups.items()},
                                 "kruskal_H": h, "p": p})
            keep("B6_pd1_pdl1_by_ELN2017", pd.DataFrame(rows), out)


# ===========================================================================
# C. external cohorts
# ===========================================================================
EXT_DIAG = PRIMARY + ["L1-logistic", "PD-1/PD-L1 only"]
EXT_PROG = PRIMARY + ["Cox-LASSO", "PD-1/PD-L1 only"]


def _refit_external(kind):
    """Rebuild the external-validation models exactly as in steps 2-3."""
    if kind == "clf":
        disc, vals, methods = config.DIAGNOSIS_DISCOVERY, config.DIAGNOSIS_VALIDATION, EXT_DIAG
    else:
        disc, vals, methods = config.PROGNOSIS_DISCOVERY, config.PROGNOSIS_VALIDATION, EXT_PROG
    X, ph = datasets.load(disc)
    Xz = zscore_dataset(X)
    target = ph["label"].values.astype(int) if kind == "clf" else \
        (ph["time"].values, ph["event"].values.astype(int))
    specs, cohorts = [], {}
    for coh in vals:
        if not datasets.available(coh):
            continue
        Xv, phv = datasets.load(coh)
        shared = [g for g in Xz.columns if g in Xv.columns]
        cohorts[coh] = (zscore_dataset(Xv[shared]), phv)
        specs += [((coh, m), Xz[shared].values, target, shared, m) for m in methods]
    log(f"[post] refitting {len(specs)} external-validation models ({kind}) in parallel ...")
    return fit_final_parallel(kind, specs), cohorts, methods


def section_c_diagnosis(res: Path, out: Path, fig: Path):
    fits, cohorts, methods = _refit_external("clf")
    pred_rows, ci_rows, test_rows, cal_rows = [], [], [], []
    reported = pd.read_csv(res / "diagnosis" / "external_validation.csv") \
        if (res / "diagnosis" / "external_validation.csv").exists() else None
    for coh, (Xvz, phv) in cohorts.items():
        yv = phv["label"].values.astype(int)
        probs, curves, dcas = {}, {}, {}
        for m in methods:
            r = fits[(coh, m)]
            p = r["model"].predict_proba(
                np.clip(r["scaler"].transform(Xvz.values), -1, 2)[:, r["cols"]])[:, 1]
            probs[m] = p
            pred_rows += [{"cohort": coh, "method": m, "sample": s, "y": int(a), "proba": float(b)}
                          for s, a, b in zip(Xvz.index, yv, p)]
            auc, lo, hi = P.delong_ci(yv, p)
            stats, curve = P.calibration_stats(yv, p)
            ci_rows.append({"cohort": coh, "method": m, "n": len(yv), "AUC": auc,
                            "DeLong_ci_low": lo, "DeLong_ci_high": hi, **stats})
            if reported is not None:
                rep = reported[(reported.cohort == coh) & (reported.method == m)]
                if len(rep) and abs(rep["AUC"].iloc[0] - auc) > 1e-3:
                    log(f"[post] note: {coh} {m} refit AUC {auc:.3f} differs from "
                        f"reported {rep['AUC'].iloc[0]:.3f} (expected with --quick; otherwise check library versions)")
            if m in PRIMARY or m == "L1-logistic":
                curves[m] = curve
                dcas[m] = P.decision_curve(yv, p)
                for _, row in dcas[m].iterrows():
                    cal_rows.append({"cohort": coh, "method": m, **row.to_dict()})
            if m in PRIMARY:
                Z = np.clip(r["scaler"].transform(Xvz.values), -1, 2)[:, r["cols"]]
                pi = permutation_importance(r["model"], Z, yv, scoring="roc_auc",
                                            n_repeats=30, random_state=config.RANDOM_STATE)
                imp = pd.DataFrame({"importance_mean": pi.importances_mean,
                                    "importance_sd": pi.importances_std},
                                   index=r["genes"]).sort_values("importance_mean", ascending=False)
                keep(f"C_importance_{coh}_{slug(m)}", imp, out, index=True)
                plots.importance_plot(imp, fig / f"post_importance_{coh}_{slug(m)}",
                                      title=f"{m} on {coh}", anchors=config.ANCHOR_GENES)
        pairs = [(PRIMARY[0], PRIMARY[1])] + [(a, b) for a in PRIMARY for b in methods if b not in PRIMARY]
        for a, b in pairs:
            if a in probs and b in probs:
                a1, a2, z, p = P.delong_test(yv, probs[a], probs[b])
                test_rows.append({"cohort": coh, "method": a, "vs": b, "AUC_method": a1,
                                  "AUC_vs": a2, "z": z, "p": p})
        plots.calibration_plot(curves, fig / f"post_calibration_{coh}", title=f"{coh}: calibration")
        plots.dca_plot(dcas, fig / f"post_dca_{coh}", title=f"{coh}: decision curve")
    tests = pd.DataFrame(test_rows)
    if len(tests):
        tests["p_holm"] = multipletests(tests["p"], method="holm")[1]
    keep("C1_diagnosis_external_predictions", pd.DataFrame(pred_rows), out)
    keep("C2_diagnosis_external_AUC_DeLong_calibration", pd.DataFrame(ci_rows), out)
    keep("C3_diagnosis_external_DeLong_tests", tests, out)
    keep("C4_diagnosis_decision_curves", pd.DataFrame(cal_rows), out)


def _eln_dummies(index):
    clin_p = next(iter(sorted(config.RAW_DIR.glob("beataml*clinical*.xlsx"))), None)
    if clin_p is None:
        return None
    clin = pd.read_excel(clin_p)
    if "dbgap_rnaseq_sample" not in clin.columns or "ELN2017" not in clin.columns:
        return None
    eln = clin.dropna(subset=["dbgap_rnaseq_sample"]).drop_duplicates("dbgap_rnaseq_sample") \
              .set_index("dbgap_rnaseq_sample")["ELN2017"].reindex(index)
    return pd.DataFrame({"ELN intermediate vs favourable": (eln == "Intermediate").astype(float),
                         "ELN adverse vs favourable": (eln == "Adverse").astype(float),
                         "_known": eln.isin(["Favorable", "Intermediate", "Adverse"])},
                        index=index)


def section_c_prognosis(out: Path, fig: Path):
    fits, cohorts, methods = _refit_external("surv")
    pred_rows, td_rows, diff_rows, mv_rows, hr_rows = [], [], [], [], []
    for coh, (Xvz, phv) in cohorts.items():
        tv, ev = phv["time"].values, phv["event"].values.astype(int)
        risks = {}
        for m in methods:
            r = fits[(coh, m)]
            risk = r["model"].predict(r["scaler"].transform(Xvz.values)[:, r["cols"]])
            risks[m] = risk
            pred_rows += [{"cohort": coh, "method": m, "sample": s, "time": a, "event": int(b),
                           "risk": float(c)} for s, a, b, c in zip(Xvz.index, tv, ev, risk)]
            for months in (12, 24, 36):
                td_rows.append({"cohort": coh, "method": m, "months": months,
                                "tdAUC": P.td_auc(tv, ev, risk, months)})
            z = (risk - risk.mean()) / risk.std()
            u = univariate_cox(z, tv, ev)
            hr_rows.append({"cohort": coh, "method": m, "HR": u["HR"], "ci_low": u["CI_low"],
                            "ci_high": u["CI_high"], "p": u["p"],
                            "logHR": np.log(u["HR"]),
                            "var": ((np.log(u["CI_high"]) - np.log(u["CI_low"])) / 3.92) ** 2})
            if m in PRIMARY:
                cov = pd.DataFrame({"Signature risk (per SD)": z}, index=Xvz.index)
                if "age" in phv.columns:
                    cov["Age (per SD)"] = (phv["age"] - phv["age"].mean()) / phv["age"].std()
                if coh == "BeatAML":
                    eln = _eln_dummies(Xvz.index)
                    if eln is not None:
                        cov = cov.join(eln)
                        cov = cov[cov.pop("_known")]
                cov = cov.loc[:, cov.notna().mean() > 0.5].dropna()
                cov = cov.loc[:, cov.std() > 0]
                ok = Xvz.index.isin(cov.index)
                if ok.sum() >= 30 and cov.shape[1] >= 1:
                    cov = cov.loc[Xvz.index[ok]]
                    res = multivariable_cox(cov.values, list(cov.columns), tv[ok], ev[ok],
                                           standardize=False)
                    mv_rows += [{"cohort": coh, "method": m, "n": int(ok.sum()), **row}
                                for row in res]
        for a in PRIMARY:
            for b in methods:
                if b != a and (b not in PRIMARY or a == PRIMARY[0]):
                    d, lo, hi, p = P.paired_cindex_bootstrap(tv, ev, risks[a], risks[b])
                    diff_rows.append({"cohort": coh, "method": a, "vs": b, "C_diff": d,
                                      "ci_low": lo, "ci_high": hi, "p_bootstrap": p})
    hr = pd.DataFrame(hr_rows)
    meta = []
    for m, d in hr.groupby("method", sort=False):
        re = P.random_effects(d["logHR"], d["var"])
        meta.append({"method": m, "pooled_HR": np.exp(re["pooled"]), "ci_low": np.exp(re["ci_low"]),
                     "ci_high": np.exp(re["ci_high"]), "p": re["p"], "I2": re["I2"], "k": re["k"]})
        if m in PRIMARY:
            t = d.assign(est=d["HR"], label=d["cohort"])
            plots.meta_forest(t, {"est": np.exp(re["pooled"]), "ci_low": np.exp(re["ci_low"]),
                                  "ci_high": np.exp(re["ci_high"]), "I2": re["I2"], "p": re["p"]},
                              fig / f"post_meta_signature_HR_{slug(m)}",
                              f"{m}: hazard ratio per 1 SD of risk score")
    td = pd.DataFrame(td_rows)
    plots.td_auc_plot(td, fig / "post_time_dependent_auc")
    diffs = pd.DataFrame(diff_rows)
    if len(diffs):
        diffs["p_holm"] = multipletests(diffs["p_bootstrap"], method="holm")[1]
    keep("C5_prognosis_external_risk_scores", pd.DataFrame(pred_rows), out)
    keep("C6_prognosis_time_dependent_AUC", td, out)
    keep("C7_prognosis_Cindex_paired_bootstrap", diffs, out)
    keep("C8_prognosis_multivariable_cox", pd.DataFrame(mv_rows), out)
    keep("C9_prognosis_signature_HR_by_cohort", hr, out)
    keep("C10_prognosis_signature_HR_meta", pd.DataFrame(meta), out)


# ===========================================================================
# D. enrichment
# ===========================================================================
def section_d(res: Path, out: Path, fig: Path):
    log("[post] D. functional enrichment (Enrichr)")
    lists = {}
    for task, d, _ in cv_tasks(res):
        df = pd.read_csv(d / "fold_metrics.csv")
        for m in PRIMARY:
            f = gene_frequency(df, m)
            lists[f"{task} | {m} | genes chosen in >=30% of folds"] = list(f[f >= 0.3].index)
    for task in ("diagnosis", "prognosis"):
        sig = res / task / ("final_signature_genes.csv" if task == "diagnosis"
                            else "final_signature_coefficients.csv")
        if sig.exists():
            s = pd.read_csv(sig)
            for m, d in s.groupby("method"):
                lists[f"{task} | {m} | final signature"] = list(d["gene"])
    # HFW-NPO-AML forces the anchors into every signature, so its lists are
    # also enriched without them: terms carried by PDCD1/CD274 alone are circular
    anchors = set(config.ANCHOR_GENES)
    versions = []
    for name, genes in lists.items():
        genes = [g for g in dict.fromkeys(genes) if isinstance(g, str)]
        versions.append((name, "full list", genes))
        if "| HFW-NPO-AML |" in name:
            versions.append((name, "without PDCD1/CD274", [g for g in genes if g not in anchors]))
    rows, list_rows = [], []
    for name, version, genes in versions:
        list_rows.append({"gene_list": name, "gene_set_version": version,
                          "n_genes": len(genes), "genes": ";".join(genes)})
        if len(genes) < 5:
            continue
        e = P.enrichr(genes, description=f"{name} | {version}")
        if e is None:
            break
        e.insert(0, "gene_list", name)
        e.insert(1, "gene_set_version", version)
        e.insert(2, "list_size", len(genes))
        rows.append(e)
    keep("D1_gene_lists", pd.DataFrame(list_rows), out)
    if rows:
        enr = pd.concat(rows, ignore_index=True)
        overlap = enr["genes"].fillna("").apply(lambda s: {g for g in s.split(";") if g})
        enr["driven_by_anchors_only"] = overlap.apply(lambda gs: bool(gs) and gs <= anchors)
        keep("D2_enrichment_all", enr, out)
        keep("D3_enrichment_significant", enr[enr["p_adj"] < 0.05], out)


# ===========================================================================
def write_workbook(out: Path, update: bool = False):
    """Write every table; with update=True only the tables of this run are
    replaced or added in the existing workbook and the other sheets are kept."""
    path = out / "manuscript_tables.xlsx"
    if update and path.exists():
        with pd.ExcelWriter(path, mode="a", engine="openpyxl", if_sheet_exists="replace") as xw:
            for name, df in TABLES.items():
                df.to_excel(xw, sheet_name=name[:31], index=False)
        n = len(pd.ExcelFile(path).sheet_names)
        log(f"[post] workbook updated: {path} ({len(TABLES)} sheets replaced/added, {n} in total)")
        return
    with pd.ExcelWriter(path) as xw:
        for name, df in TABLES.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
    log(f"[post] workbook: {path} ({len(TABLES)} sheets)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--skip-external", action="store_true")
    ap.add_argument("--skip-enrichment", action="store_true")
    ap.add_argument("--enrichment-only", action="store_true",
                    help="re-run section D only and update its sheets in the existing workbook")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--quick", action="store_true",
                    help="code check only: small NPO budget for the refits, output in post_quick/")
    args = ap.parse_args()
    if args.quick:
        config.apply_quick_mode()
    res = config.PROJECT_ROOT / f"results_{args.tag}"
    config.RESULTS_DIR = res
    config.FIGURES_DIR = config.PROJECT_ROOT / f"figures_{args.tag}"
    config.LOG_FILE = res / "progress.log"
    if args.jobs:
        config.N_JOBS = args.jobs
    out = res / ("post_quick" if args.quick else "post")
    fig = config.FIGURES_DIR / ("post_quick" if args.quick else "")
    log(f"[post] started on {res}")
    if args.enrichment_only:
        section_d(res, out, fig)
        if "D2_enrichment_all" not in TABLES:
            log("[post] enrichment produced no results (offline or requests missing); "
                "workbook left unchanged")
            return
        write_workbook(out, update=True)
        info = out / "post_run_info.json"
        sections = json.loads(info.read_text())["sections"] if info.exists() else []
        write_json({"sections": sections + [t for t in TABLES if t not in sections]}, info)
        log("[post] finished (enrichment only)")
        return
    section_a(res, out, fig)
    section_b(out, fig)
    if not args.skip_external:
        log("[post] C. external cohorts (refits the external-validation models)")
        section_c_diagnosis(res, out, fig)
        section_c_prognosis(out, fig)
    if not args.skip_enrichment:
        section_d(res, out, fig)
    write_workbook(out)
    write_json({"sections": list(TABLES)}, out / "post_run_info.json")
    log("[post] finished")


if __name__ == "__main__":
    main()
