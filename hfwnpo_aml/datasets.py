"""
Builders that turn the raw public downloads into clean analysis tables.

Every builder writes two files into config.PROCESSED_DIR:
    <name>_expr.csv.gz   samples x genes (log2 expression)
    <name>_pheno.csv     samples x phenotype columns
        diagnosis  : label (1 = AML, 0 = non-leukaemic)
        survival   : time (months), event (1 = death), age
A builder whose raw files are missing prints what to download and returns
False, so the rest of the pipeline keeps running.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

import config
from .geo import (collapse_to_genes, ensure_log2, read_gpl_annotation,
                  read_series_expr, read_series_matrix, read_series_meta,
                  title_number)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _find(pattern: str, prefer: tuple[str, ...] = ()) -> Path | None:
    """First file matching the pattern. Browser duplicates such as
    'name (1).txt' are ignored and the extensions in `prefer` win."""
    hits = [h for h in config.RAW_DIR.rglob(pattern)
            if h.is_file() and not re.search(r"\(\d+\)", h.name)]
    if not hits:
        return None
    def rank(h: Path):
        ext = "".join(h.suffixes).lower()
        return (0 if any(ext.endswith(p) for p in prefer) else 1, len(h.name), h.name)
    return sorted(hits, key=rank)[0]


def _missing(name: str, what: str) -> bool:
    print(f"  [skip] {name}: {what}")
    return False


def _save(name: str, expr: pd.DataFrame, pheno: pd.DataFrame) -> bool:
    """expr: genes x samples, pheno: samples x columns (index = samples)."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    common = [s for s in pheno.index if s in expr.columns]
    if len(common) < 10:
        print(f"  [skip] {name}: only {len(common)} samples match between expression "
              f"and phenotype files - check that both files belong to this cohort")
        return False
    expr = expr[common]
    pheno = pheno.loc[common]
    expr = expr.loc[~expr.index.duplicated()]
    expr = expr.dropna(axis=0, thresh=int(0.9 * expr.shape[1]))
    expr = expr.T.astype("float32")
    expr.index.name = "sample"
    expr.to_csv(config.PROCESSED_DIR / f"{name}_expr.csv.gz", float_format="%.4f")
    pheno.index.name = "sample"
    pheno.to_csv(config.PROCESSED_DIR / f"{name}_pheno.csv")
    desc = (f"label counts {pheno['label'].value_counts().to_dict()}"
            if "label" in pheno else
            f"events {int(pheno['event'].sum())}/{len(pheno)}")
    anchors = [g for g in config.ANCHOR_GENES if g in expr.columns]
    print(f"  [ok]   {name}: {expr.shape[0]} samples x {expr.shape[1]} genes | "
          f"{desc} | anchors present: {anchors}")
    return True


_GPL570_CACHE: dict[str, pd.Series] = {}


def _affy_annotation() -> pd.Series | None:
    """GPL570 table also covers every GPL96 and most GPL97 probe IDs."""
    if "gpl570" not in _GPL570_CACHE:
        p = _find("GPL570*.txt*")
        if p is None:
            return None
        _GPL570_CACHE["gpl570"] = read_gpl_annotation(p)
    return _GPL570_CACHE["gpl570"]


def _to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace({"NA": np.nan, "": np.nan}), errors="coerce")


# ---------------------------------------------------------------------------
# Diagnosis cohorts
# ---------------------------------------------------------------------------
def build_gse114868() -> bool:
    name = "GSE114868"
    mat = _find("GSE114868*series_matrix*")
    gpl = _find("GPL17586*", prefer=(".txt",))
    if mat is None:
        return _missing(name, "GSE114868_series_matrix.txt.gz not found")
    if gpl is None:
        return _missing(name, "annotation GPL17586 not found. Download the full "
                        "table from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL17586")
    expr, meta = read_series_matrix(mat)
    genes = collapse_to_genes(ensure_log2(expr), read_gpl_annotation(gpl))
    status = meta["subject status"].str.lower()
    label = np.where(status.str.contains("aml|leuk"), 1,
                     np.where(status.str.contains("healthy|normal"), 0, -1))
    pheno = pd.DataFrame({"label": label}, index=meta.index)
    return _save(name, genes, pheno[pheno["label"] >= 0])


def build_gse9476() -> bool:
    name = "GSE9476"
    mat = _find("GSE9476*series_matrix*")
    ann = _affy_annotation()
    if mat is None or ann is None:
        return _missing(name, "series matrix or GPL570 annotation not found")
    expr, meta = read_series_matrix(mat)
    genes = collapse_to_genes(ensure_log2(expr), ann)
    title = meta["title"].str.upper()
    source = meta["source_name_ch1"].str.lower()
    label = np.where(title.str.startswith("FHCRCAML") | source.str.contains("leuk"), 1, 0)
    return _save(name, genes, pd.DataFrame({"label": label}, index=meta.index))


def build_gse13159() -> bool:
    """MILE study (optional). All AML subclasses vs non-leukaemia / healthy BM."""
    name = "GSE13159"
    mat = _find("GSE13159*series_matrix*")
    ann = _affy_annotation()
    if mat is None or ann is None:
        return _missing(name, "optional; series matrix not found")
    meta, n = read_series_meta(mat)
    col = next((c for c in meta.columns if "leukemia class" in c), None)
    if col is None:
        return _missing(name, "no 'leukemia class' characteristic found")
    cls = meta[col].str.lower()
    label = np.where(cls.str.startswith("aml"), 1,
                     np.where(cls.str.contains("non-leukemia|healthy"), 0, -1))
    pheno = pd.DataFrame({"label": label}, index=meta.index)
    pheno = pheno[pheno["label"] >= 0]
    # read only the AML and control columns (the full file has 2096 samples)
    expr = read_series_expr(mat, n, pheno.index)
    genes = collapse_to_genes(ensure_log2(expr), ann)
    return _save(name, genes, pheno)


# ---------------------------------------------------------------------------
# Survival cohorts
# ---------------------------------------------------------------------------
def _gse37642_pheno(meta: pd.DataFrame) -> pd.DataFrame:
    os_col = next(c for c in meta.columns if c.startswith("overall survival"))
    st_col = next(c for c in meta.columns if c.startswith("life status"))
    pheno = pd.DataFrame(index=meta.index)
    pheno["time"] = _to_float(meta[os_col]) / 30.4375            # days -> months
    pheno["event"] = meta[st_col].str.lower().map({"dead": 1, "alive": 0})
    if "age" in meta.columns:
        pheno["age"] = _to_float(meta["age"])
    pheno["title"] = meta["title"]
    return pheno.dropna(subset=["time", "event"])


def build_gse37642_u133ab() -> bool:
    """422 AMLCG patients hybridised on U133A (GPL96) + U133B (GPL97)."""
    name = "GSE37642_U133AB"
    a, b = _find("GSE37642-GPL96*"), _find("GSE37642-GPL97*")
    ann = _affy_annotation()
    if a is None or b is None or ann is None:
        return _missing(name, "GPL96/GPL97 series matrices or GPL570 table missing")
    ea, ma = read_series_matrix(a)
    eb, mb = read_series_matrix(b)
    ga = collapse_to_genes(ensure_log2(ea), ann)
    gb = collapse_to_genes(ensure_log2(eb), ann)
    # pair chips by patient number in the sample title
    ga.columns = [title_number(t) for t in ma.loc[ga.columns, "title"]]
    gb.columns = [title_number(t) for t in mb.loc[gb.columns, "title"]]
    common = ga.columns.intersection(gb.columns)
    both = pd.concat([ga[common], gb[common]])
    both["__mean"] = both.mean(axis=1)
    both = (both.sort_values("__mean", ascending=False)
                .loc[lambda d: ~d.index.duplicated()].drop(columns="__mean"))
    pheno = _gse37642_pheno(ma)
    pheno.index = [title_number(t) for t in pheno["title"]]
    pheno = pheno.drop(columns="title")
    both.columns = [f"AMLCG_{c}" for c in both.columns]
    pheno.index = [f"AMLCG_{c}" for c in pheno.index]
    return _save(name, both, pheno)


def build_gse37642_u133p2() -> bool:
    name = "GSE37642_U133P2"
    mat = _find("GSE37642-GPL570*")
    ann = _affy_annotation()
    if mat is None or ann is None:
        return _missing(name, "GPL570 series matrix or annotation missing")
    expr, meta = read_series_matrix(mat)
    genes = collapse_to_genes(ensure_log2(expr), ann)
    return _save(name, genes, _gse37642_pheno(meta).drop(columns="title"))


def build_tcga_laml() -> bool:
    name = "TCGA_LAML"
    clin_p = _find("laml_tcga*clinical*.tsv") or _find("data_clinical_patient.txt")
    expr_p = _find("data_mrna_seq_v2_rsem.txt") or _find("data_mrna_seq*.txt")
    if expr_p is None or clin_p is None:
        return _missing(name, "needs data_mrna_seq_v2_rsem.txt from the cBioPortal "
                        "download of laml_tcga_pan_can_atlas_2018 (plus clinical tsv)")
    expr = pd.read_csv(expr_p, sep="\t", low_memory=False)
    sym_col = "Hugo_Symbol" if "Hugo_Symbol" in expr.columns else expr.columns[0]
    expr = expr.dropna(subset=[sym_col]).drop(
        columns=[c for c in expr.columns if c.startswith("Entrez")], errors="ignore")
    expr = expr.groupby(sym_col).mean(numeric_only=True)
    expr = np.log2(expr.clip(lower=0) + 1.0)
    clin = pd.read_csv(clin_p, sep="\t", comment=None)
    sid = "Sample ID" if "Sample ID" in clin.columns else "SAMPLE_ID"
    os_m = "Overall Survival (Months)" if "Overall Survival (Months)" in clin.columns else "OS_MONTHS"
    os_s = "Overall Survival Status" if "Overall Survival Status" in clin.columns else "OS_STATUS"
    pheno = pd.DataFrame(index=clin[sid].values)
    pheno["time"] = _to_float(clin[os_m]).values
    pheno["event"] = clin[os_s].astype(str).str.startswith("1").astype(int).values
    age_col = next((c for c in clin.columns if c.lower() in ("diagnosis age", "age")), None)
    if age_col:
        pheno["age"] = _to_float(clin[age_col]).values
    pheno = pheno.dropna(subset=["time"])
    pheno = pheno[pheno["time"] > 0]
    return _save(name, expr, pheno)


# ---------------------------------------------------------------------------
# BeatAML (expression + survival + ex-vivo drug response)
# ---------------------------------------------------------------------------
def _read_table(p: Path) -> pd.DataFrame:
    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p)
    sep = "," if p.suffix.lower() == ".csv" else "\t"
    return pd.read_csv(p, sep=sep, low_memory=False)


def _beataml_expr() -> pd.DataFrame | None:
    p = _find("beataml*norm_exp*")
    if p is None:
        return None
    e = pd.read_csv(p, sep="\t", low_memory=False)
    sym = "display_label" if "display_label" in e.columns else e.columns[1]
    e = e.drop(columns=[c for c in ("stable_id", "description", "biotype")
                        if c in e.columns]).groupby(sym).mean(numeric_only=True)
    return e


def _match_sample_col(df: pd.DataFrame, ids: set[str]) -> str | None:
    best, best_n = None, 0
    for c in df.columns:
        n = df[c].astype(str).isin(ids).sum()
        if n > best_n:
            best, best_n = c, n
    return best if best_n > 0 else None


def _one_sample_per_patient(samples, clin: pd.DataFrame | None, scol: str | None) -> list[str]:
    """Keep one specimen per BeatAML patient (dbgap_subject_id): the
    initial-diagnosis specimen when there is one, otherwise the earliest
    (timeOfSampleCollectionRelativeToInclusion); ties broken by sample id."""
    samples = [str(x) for x in samples]
    if clin is None or scol is None or "dbgap_subject_id" not in clin.columns:
        return samples
    c = clin.dropna(subset=[scol]).copy()
    c[scol] = c[scol].astype(str)
    c = c.drop_duplicates(scol).set_index(scol).reindex(samples)
    c["sample"] = samples
    stage = c.get("diseaseStageAtSpecimenCollection", pd.Series(index=c.index, dtype=object))
    c["_not_dx"] = (stage.astype(str).str.lower() != "initial diagnosis").astype(int)
    c["_time"] = _to_float(c.get("timeOfSampleCollectionRelativeToInclusion",
                                 pd.Series(index=c.index, dtype=float))).fillna(np.inf)
    # samples without a subject id are kept as their own patient
    c["_subject"] = c["dbgap_subject_id"].astype(str).where(c["dbgap_subject_id"].notna(),
                                                           "sample_" + c["sample"])
    c = c.sort_values(["_subject", "_not_dx", "_time", "sample"])
    keep = c.drop_duplicates("_subject")["sample"].tolist()
    n_drop = len(samples) - len(keep)
    if n_drop:
        print(f"  [info] BeatAML: kept 1 specimen per patient ({len(keep)} patients, "
              f"{n_drop} extra specimens removed)")
    return [x for x in samples if x in set(keep)]


def build_beataml() -> bool:
    name = "BeatAML"
    expr = _beataml_expr()
    if expr is None:
        return _missing(name, "beataml_waves1to4_norm_exp_dbgap.txt not found")
    ids = set(expr.columns)
    ok = False

    clin_p = _find("beataml*clinical*", prefer=(".xlsx",))
    clin = _read_table(clin_p) if clin_p is not None else None
    if clin is not None:
        scol = _match_sample_col(clin, ids)
        lower = {c.lower(): c for c in clin.columns}
        os_c = lower.get("overallsurvival")
        vs_c = lower.get("vitalstatus")
        if scol and os_c and vs_c:
            pheno = pd.DataFrame(index=clin[scol].astype(str).values)
            pheno["time"] = _to_float(clin[os_c]).values / 30.4375
            pheno["event"] = clin[vs_c].astype(str).str.lower().str.startswith("dead").astype(int).values
            age_c = lower.get("ageatspecimenacquisition") or lower.get("ageatdiagnosis")
            if age_c:
                pheno["age"] = _to_float(clin[age_c]).values
            if "isRelapse" in clin.columns:     # keep diagnosis-time specimens
                rel = clin["isRelapse"].astype(str).str.lower().isin(["true", "1", "yes"]).values
                pheno = pheno[~rel]
            pheno = pheno[~pheno.index.duplicated()].dropna(subset=["time"])
            pheno = pheno[pheno["time"] > 0]
            pheno = pheno[pheno.index.isin(ids)]
            pheno = pheno.loc[_one_sample_per_patient(pheno.index, clin, scol)]
            ok = _save(name, expr, pheno)
        else:
            _missing(name + " survival", "clinical columns overallSurvival/vitalStatus not found")
    else:
        _missing(name + " survival", "beataml_wv1to4_clinical.xlsx not found")

    drug_p = _find("beataml*probit*")
    if drug_p is None:
        _missing(name + " drug", "beataml_probit_curve_fits_v4_dbgap.txt not found")
        return ok
    drug = _read_table(drug_p)
    lower = {c.lower(): c for c in drug.columns}
    inh_c = lower.get("inhibitor")
    auc_c = lower.get("auc")
    scol = _match_sample_col(drug, ids)
    if scol is None and clin is not None:
        # map through a subject id shared by the drug and clinical tables
        csample = _match_sample_col(clin, ids)
        shared = [c for c in drug.columns if c in clin.columns and c != csample]
        if csample and shared:
            key = shared[0]
            m = clin[[key, csample]].drop_duplicates(key)
            drug = drug.merge(m, on=key, how="inner")
            scol = csample
    if not (inh_c and auc_c and scol):
        return _missing(name + " drug", "could not find sample / inhibitor / auc columns")
    wide = drug.pivot_table(index=scol, columns=inh_c, values=auc_c, aggfunc="mean")
    wide.index = wide.index.astype(str)
    wide = wide.loc[wide.index.isin(ids)]
    if clin is not None:
        wide = wide.loc[_one_sample_per_patient(wide.index, clin, _match_sample_col(clin, ids))]
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    wide.index.name = "sample"
    wide.to_csv(config.PROCESSED_DIR / "BeatAML_drug_auc.csv")
    e = expr[[s for s in wide.index]]
    e = e.T.astype("float32")
    e.index.name = "sample"
    e.to_csv(config.PROCESSED_DIR / "BeatAML_drug_expr.csv.gz", float_format="%.4f")
    print(f"  [ok]   BeatAML drug: {wide.shape[0]} samples x {wide.shape[1]} inhibitors")
    return True


BUILDERS = {
    "GSE114868": build_gse114868,
    "GSE13159": build_gse13159,
    "GSE9476": build_gse9476,
    "GSE37642_U133AB": build_gse37642_u133ab,
    "GSE37642_U133P2": build_gse37642_u133p2,
    "TCGA_LAML": build_tcga_laml,
    "BeatAML": build_beataml,
}


# ---------------------------------------------------------------------------
# loaders used by the experiments
# ---------------------------------------------------------------------------
def available(name: str) -> bool:
    return (config.PROCESSED_DIR / f"{name}_expr.csv.gz").exists()


def load(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = pd.read_csv(config.PROCESSED_DIR / f"{name}_expr.csv.gz", index_col=0)
    pheno = pd.read_csv(config.PROCESSED_DIR / f"{name}_pheno.csv", index_col=0)
    X = X.loc[pheno.index]
    X = X.fillna(X.mean())
    return X, pheno


def load_drug() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    p = config.PROCESSED_DIR / "BeatAML_drug_auc.csv"
    if not p.exists():
        return None
    auc = pd.read_csv(p, index_col=0)
    X = pd.read_csv(config.PROCESSED_DIR / "BeatAML_drug_expr.csv.gz", index_col=0)
    X = X.fillna(X.mean())
    return X, auc.loc[X.index]
