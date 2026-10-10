"""Readers for GEO series-matrix files and GPL annotation tables."""

from __future__ import annotations

import gzip
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _open_text(path: Path):
    path = Path(path)
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _split(line: str) -> list[str]:
    return [v.strip().strip('"') for v in line.rstrip("\n").split("\t")]


def read_series_meta(path: Path) -> tuple[pd.DataFrame, int]:
    """
    Parse the sample annotation block of a GEO series matrix.

    Returns (meta, n_header_lines). meta is samples x fields; characteristic
    lines of the form "key: value" become one column per key.
    """
    meta_rows: dict[str, list[str]] = {}
    char_cols: dict[str, list[str]] = {}
    n_lines = 0
    with _open_text(path) as fh:
        for line in fh:
            n_lines += 1
            if line.startswith("!series_matrix_table_begin"):
                break
            if not line.startswith("!Sample_"):
                continue
            parts = _split(line)
            key, values = parts[0][len("!Sample_"):], parts[1:]
            if key.startswith("characteristics"):
                # each characteristics line may hold a different key per sample
                for j, v in enumerate(values):
                    if ":" in v:
                        k, val = v.split(":", 1)
                        k, val = k.strip().lower(), val.strip()
                    else:
                        k, val = f"{key}_{len(char_cols)}", v
                    col = char_cols.setdefault(k, [""] * len(values))
                    if len(col) < len(values):
                        col.extend([""] * (len(values) - len(col)))
                    col[j] = val
            else:
                if key in meta_rows:
                    key = f"{key}_{sum(k.startswith(key) for k in meta_rows)}"
                meta_rows[key] = values
    meta = pd.DataFrame(meta_rows)
    extra = {k: v for k, v in char_cols.items() if len(v) == len(meta)}
    if extra:
        meta = pd.concat([meta, pd.DataFrame(extra)], axis=1)
    if "geo_accession" in meta.columns:
        meta.index = meta["geo_accession"]
    return meta, n_lines


def read_series_expr(path: Path, n_header_lines: int, samples=None) -> pd.DataFrame:
    """Read the expression table (probes x samples); optionally only some samples."""
    keep = None
    if samples is not None:
        wanted = {str(s) for s in samples}
        keep = lambda c: c == "ID_REF" or c in wanted  # noqa: E731
    # read in chunks so that very large files (e.g. GSE13159) fit in memory
    chunks = pd.read_csv(path, sep="\t", skiprows=n_header_lines, index_col=0,
                         usecols=keep, chunksize=5000, low_memory=True)
    expr = pd.concat([c.apply(pd.to_numeric, errors="coerce").astype("float32")
                      for c in chunks])
    expr.index = expr.index.astype(str).str.strip('"')
    expr = expr[~expr.index.str.startswith("!")]
    return expr.astype("float64")


def read_series_matrix(path: Path, samples=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """GEO series matrix -> (expr probes x samples, meta samples x fields)."""
    meta, n = read_series_meta(path)
    return read_series_expr(path, n, samples), meta


def read_gpl_annotation(path: Path) -> pd.Series:
    """
    Return a Series mapping probe ID -> single gene symbol.

    Supports the standard Affymetrix 3' tables ("Gene Symbol" column, e.g.
    GPL570/GPL96/GPL97) and the HTA/Clariom tables ("gene_assignment"
    column, e.g. GPL17586). Probes mapping to several genes are dropped.
    """
    with _open_text(path) as fh:
        lines = [ln for ln in fh if not ln.startswith(("#", "!", "^"))]
    ann = pd.read_csv(io.StringIO("".join(lines)), sep="\t", dtype=str,
                      low_memory=False)
    ann.columns = [c.strip() for c in ann.columns]
    id_col = "ID" if "ID" in ann.columns else ann.columns[0]

    if "Gene Symbol" in ann.columns:
        sym = ann["Gene Symbol"].fillna("")
        sym = sym.where(~sym.str.contains("///"), "")
    elif "gene_assignment" in ann.columns:
        def first_symbol(s: str) -> str:
            if not isinstance(s, str) or s.strip() in ("", "---"):
                return ""
            symbols = {p.split("//")[1].strip() for p in s.split("///")
                       if len(p.split("//")) > 1}
            symbols.discard("---")
            return symbols.pop() if len(symbols) == 1 else ""
        sym = ann["gene_assignment"].map(first_symbol)
    elif "GENE_SYMBOL" in ann.columns:
        sym = ann["GENE_SYMBOL"].fillna("")
    else:
        raise ValueError(f"No gene-symbol column found in {path}")

    out = pd.Series(sym.str.strip().values, index=ann[id_col].astype(str).values)
    return out[out != ""]


def collapse_to_genes(expr: pd.DataFrame, probe2gene: pd.Series) -> pd.DataFrame:
    """
    Probe x sample -> gene x sample. For genes with several probes the probe
    with the highest mean expression is kept (a common, simple rule).
    """
    common = expr.index.intersection(probe2gene.index)
    e = expr.loc[common].copy()
    e["__gene"] = probe2gene.loc[common].values
    e["__mean"] = e.drop(columns="__gene").mean(axis=1)
    e = e.sort_values("__mean", ascending=False)
    e = e.drop_duplicates("__gene").set_index("__gene").drop(columns="__mean")
    e.index.name = "gene"
    return e


def ensure_log2(expr: pd.DataFrame) -> pd.DataFrame:
    """Log2-transform data that still look like raw intensities."""
    q99 = float(np.nanpercentile(expr.values, 99))
    if q99 > 100:
        return np.log2(expr.clip(lower=0) + 1.0)
    return expr


def title_number(title: str) -> str:
    """'AML_patient_12' -> '12' (used to pair U133A and U133B chips)."""
    m = re.search(r"(\d+)\s*$", str(title))
    return m.group(1) if m else str(title)
