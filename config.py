"""
Central configuration for the HFW-NPO-AML experiments.

Every experiment script reads its settings from here, so a full re-run of the
paper only needs edits in this one file.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# make the project importable in parallel worker processes (joblib/loky)
if str(PROJECT_ROOT) not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])

# Folder holding the raw downloads (GEO series matrices, GPL tables, BeatAML,
# cBioPortal files). By default it is the "dataset" folder next to this
# project; change it if your raw files live elsewhere.
RAW_DIR = Path(os.environ.get("HFWNPO_RAW_DIR", PROJECT_ROOT.parent / "dataset"))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
# live progress log (open it in VS Code to watch the experiments)
LOG_FILE = RESULTS_DIR / "progress.log"

# ---------------------------------------------------------------------------
# PD-1 / PD-L1 anchoring
# ---------------------------------------------------------------------------
# Genes that are forced into every candidate solution of the optimiser.
ANCHOR_GENES = ["PDCD1", "CD274"]

# How the candidate pool entering the hybrid filter is built:
#   "centered" -> anchors + curated PD-1/PD-L1 axis genes + the genes most
#                 co-expressed with the anchors (computed on the training fold)
#   "genome"   -> all genes (classic HFW-NPO); anchors are still forced in
CANDIDATE_MODE = "centered"
N_COEXPRESSED = 1500  # co-expressed genes added per anchor in "centered" mode

# Candidate pool per task for the proposed method and its comparators.
#   diagnosis / chemoresistance: "centered" (PD-1/PD-L1-centred pool)
#   prognosis: "genome" with PD-1/PD-L1 anchored. Decided before the full
#   runs because PD-1 and PD-L1 showed no univariate association with overall
#   survival in any cohort, so restricting the search to their co-expression
#   neighbourhood would discard the survival signal.
SURVIVAL_POOL = "genome"

# Curated immune-checkpoint / PD-1-PD-L1 signalling genes. Only genes present
# in a dataset are used; missing ones are ignored silently.
PDL1_AXIS_GENES = [
    # checkpoint receptors and ligands
    "PDCD1", "CD274", "PDCD1LG2", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "CD96",
    "VSIR", "BTLA", "CD276", "VTCN1", "IDO1", "IDO2", "CD80", "CD86", "CD28",
    "ICOS", "ICOSLG", "LGALS9", "PVR", "NECTIN2", "CD47", "SIRPA", "ADORA2A",
    "NT5E", "ENTPD1",
    # regulators of PD-L1 expression and stability
    "CMTM6", "CMTM4", "IFNG", "IFNGR1", "IFNGR2", "JAK1", "JAK2", "STAT1",
    "STAT3", "IRF1", "HIF1A", "MYC", "PTEN", "PIK3CA", "AKT1", "MTOR",
    "NFKB1", "RELA", "MAPK1", "MAPK3", "CSN5", "COPS5", "CDK4", "SPOP",
    "BRD4", "EGFR", "TNF", "IL6", "IL10", "TGFB1",
    # T-cell / exhaustion context
    "CD3E", "CD8A", "CD4", "GZMB", "PRF1", "TOX", "EOMES", "TBX21", "FOXP3",
    "IL2RA", "ZAP70", "LCK",
    # AML immunophenotype markers
    "CD33", "CD34", "KIT", "HLA-DRA", "CD38", "CD14", "ITGAM", "CD4",
]

# ---------------------------------------------------------------------------
# Primary methods (decided before the full runs)
# ---------------------------------------------------------------------------
# Two HFW-NPO variants are reported as co-primary methods with equal standing:
# the published HFW-NPO (genome-wide pool, no anchors) and the
# PD-1/PD-L1-anchored HFW-NPO-AML. Each is compared with every other method
# and with each other (two-sided Wilcoxon, Holm-corrected).
PRIMARY_METHODS = ["HFW-NPO (original)", "HFW-NPO-AML"]

# Reference panel for the chemoresistance experiment: monocytic markers
# (venetoclax resistance is linked to monocytic differentiation), used
# directly without filter or wrapper.
MONOCYTIC_GENES = ["CD14", "CD68", "LYZ", "CD163", "ITGAM"]

# ---------------------------------------------------------------------------
# Hybrid filter (Phase 1)
# ---------------------------------------------------------------------------
FILTER_K = 80              # genes kept by the filter (Electronics paper: 80)
FILTER_WEIGHTS = (0.40, 0.30, 0.30)   # Fisher, MI, Welch  (classification)
SURV_FILTER_WEIGHTS = (0.40, 0.30, 0.30)  # Cox score z, |C-0.5|, log-rank

# ---------------------------------------------------------------------------
# NPO wrapper (Phase 3) - defaults follow the published HFW-NPO settings
# ---------------------------------------------------------------------------
NPO_CLANS = 10
NPO_FAMILIES = 20
NPO_ITER = 300
NPO_LB, NPO_UB = -10.0, 10.0

# Fitness = ALPHA_FIT * (1 - score) + (1 - ALPHA_FIT) * sqrt(|S| / FILTER_K)
# score = balanced accuracy (classification) or C-index (survival).
ALPHA_FIT = 0.85

# ---------------------------------------------------------------------------
# Baseline metaheuristics use the same evaluation budget as NPO
# ---------------------------------------------------------------------------
BASELINE_POP = NPO_CLANS * NPO_FAMILIES
BASELINE_ITER = NPO_ITER
N_FIXED_FEATURES = 20      # size used by mRMR / RFE / univariate baselines

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
N_SPLITS = 5
N_REPEATS = 2              # repeated CV -> 10 paired values for statistics
RANDOM_STATE = 42
N_JOBS = -2                # parallel CV folds (joblib): -2 = all CPU cores but one

# ---------------------------------------------------------------------------
# Quick mode (python run_all.py --quick) for a fast end-to-end check
# ---------------------------------------------------------------------------
QUICK = dict(
    NPO_CLANS=4, NPO_FAMILIES=5, NPO_ITER=15,
    BASELINE_POP=20, BASELINE_ITER=15,
    N_SPLITS=3, N_REPEATS=1, FILTER_K=40, N_COEXPRESSED=300,
)

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
# Diagnosis (AML vs non-leukaemic bone marrow)
# GSE13159 (MILE, 542 AML / 74 controls) is the discovery cohort; the small
# GSE114868 (HTA 2.0) and GSE9476 (U133A) cohorts are external validation.
DIAGNOSIS_DISCOVERY = "GSE13159"
DIAGNOSIS_VALIDATION = ["GSE114868", "GSE9476"]   # used when present

# Prognosis (overall survival)
PROGNOSIS_DISCOVERY = "GSE37642_U133AB"
PROGNOSIS_VALIDATION = ["GSE37642_U133P2", "TCGA_LAML", "BeatAML"]

# Chemoresistance (BeatAML ex-vivo drug response, area under the curve).
# Present in BeatAML: Cytarabine, Venetoclax, Azacytidine (the others are
# skipped automatically). BeatAML tables keep one specimen per patient
# (dbgap_subject_id): the initial-diagnosis specimen, else the earliest one.
DRUGS = ["Cytarabine", "Venetoclax", "Daunorubicin", "Idarubicin",
         "Azacytidine", "Decitabine"]
# Samples in the top third of AUC are "resistant", bottom third "sensitive".
DRUG_QUANTILES = (1 / 3, 2 / 3)


def apply_quick_mode() -> None:
    """Overwrite the heavy settings with the QUICK values (in place)."""
    g = globals()
    for key, value in QUICK.items():
        g[key] = value
