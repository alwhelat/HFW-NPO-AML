<h1 align="center">HFW-NPO-AML</h1>

<p align="center">
<b>HFW-NPO: Compact Gene Expression Signatures for Diagnosis, Prognosis<br>
and Drug Response in Acute Myeloid Leukemia</b>
</p>

<p align="center">
<img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue">
<img alt="License" src="https://img.shields.io/badge/license-MIT-green">
<img alt="Status" src="https://img.shields.io/badge/status-manuscript%20in%20preparation-orange">
</p>

Feature selection for acute myeloid leukemia (AML) transcriptomes, extending the published
HFW-NPO framework from Alzheimer's disease to oncology: a three-score hybrid filter, adaptive SMOTE,
a binary Nomadic People Optimizer (NPO) wrapper and a soft-voting ensemble, with two AML-specific
extensions (PD-1/PD-L1 anchoring and a survival objective), evaluated on seven public cohorts.

> **Status.** The accompanying manuscript is in preparation. This repository contains the code only.
> Result tables, signatures and figures will be released here when the article is published.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Compared methods](#compared-methods)
- [Datasets](#datasets)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Outputs](#outputs)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [Authors](#authors)
- [License](#license)

## Overview

Gene expression profiles measure tens of thousands of genes on a few hundred patients. Clinical tests,
in contrast, need small panels that transfer across platforms and cohorts. This project applies
HFW-NPO (Alwhelat & Abiyev, *Electronics* 2026, 15(13), 2970,
[doi:10.3390/electronics15132970](https://doi.org/10.3390/electronics15132970)) to three AML tasks:

| Task | Outcome | Model |
|---|---|---|
| Diagnosis | AML vs. non-leukaemic marrow | soft-voting ensemble |
| Prognosis | overall survival | ridge Cox model |
| Drug response | resistant vs. sensitive (BeatAML ex vivo AUC tertiles): cytarabine, venetoclax, azacitidine | soft-voting ensemble |

Two variants are evaluated as **co-primary methods** with equal standing:

| Variant | Candidate pool | PD-1/PD-L1 anchored |
|---|---|---|
| `HFW-NPO (original)` | whole genome | no |
| `HFW-NPO-AML` | PD-1/PD-L1-centred (diagnosis, drugs); whole genome (prognosis) | yes |

The AML-specific extensions in `HFW-NPO-AML` are:

1. **PD-1/PD-L1 anchoring.** `PDCD1` (PD-1) and `CD274` (PD-L1) are forced into every candidate gene subset and are not counted in the compactness penalty.
2. **PD-1/PD-L1-centred candidate pool.** The filter scores the anchors, a curated checkpoint/interferon/T-cell/myeloid gene set and the genes most co-expressed with each anchor in the training fold.
3. **Survival objective.** A survival filter (Cox score, |C − 0.5|, log-rank) and a wrapper fitness based on the three-fold C-index of a ridge Cox model.

The NPO search itself (clan–family hierarchy, adaptive Lévy flight, anti-stagnation restarts, hill climbing) is taken unchanged from the published implementation.

## Architecture

```
Public expression cohorts (z-scored within each cohort)
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 1 · Candidate pool + hybrid filter  (TRAIN only)     │
│  classification: ω = 0.40·Fisher + 0.30·MI + 0.30·Welch     │
│  survival:       ω = 0.40·|Cox z| + 0.30·|C−0.5| + 0.30·LR  │
│  → keep top K = 80 genes (anchors always kept)              │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 2 · Adaptive SMOTE  (train split, classification)    │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 3 · Binary NPO wrapper                               │
│  10 clans × 20 families × 300 iterations                    │
│  f(S) = α·(1 − score) + (1 − α)·√(|S \ anchors| / K), α=0.85│
│  score = balanced accuracy (ridge) or C-index (ridge Cox)   │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 4 · Final model                                      │
│  classification: ET + SVM-RBF + LR + RF + GB (soft voting)  │
│  survival:       ridge Cox risk score                       │
└─────────────────────────────────────────────────────────────┘
   Repeated stratified 5-fold CV × 2 (outer loop); external cohorts
   scored with models fitted on the full discovery cohort, no re-tuning
```

## Compared methods

| Method | Candidate pool | Anchored | Selector |
|---|---|---|---|
| **HFW-NPO (original)** (co-primary) | whole genome | no | NPO |
| **HFW-NPO-AML** (co-primary) | centred | yes | NPO |
| HFW-NPO-AML w/o centred pool (ablation, diagnosis) | whole genome | yes | NPO |
| GA, PSO | centred | yes | same fitness and evaluation budget |
| mRMR, RFE-SVM, L1-logistic (diagnosis, drugs) | centred | yes | classic |
| Cox-LASSO, univariate Cox (prognosis) | genome | yes | classic |
| Filter only | centred | yes | all K filtered genes |
| PD-1/PD-L1 only | – | – | the two anchors |
| Monocytic markers only (drugs) | – | – | CD14, CD68, LYZ, CD163, ITGAM |

For prognosis, all methods search the whole genome with the anchors forced in, and the centred pool is an ablation. This choice was fixed before the full runs (`config.SURVIVAL_POOL`).

**Statistics.** Friedman test, two-sided paired Wilcoxon tests with Holm correction, Nemenyi critical differences, Jaccard selection stability; DeLong confidence intervals, calibration and decision curves (diagnosis); C-index, time-dependent AUC, multivariable Cox and random-effects meta-analysis (prognosis); partial Spearman correlations with drug AUC; Enrichr enrichment with and without the anchors; sensitivity analysis for K and α.

## Datasets

All datasets are public. Raw files are **not** included in this repository because of their size and redistribution terms; download them from the links in [`data/README.md`](data/README.md) and place them in `../dataset` (`RAW_DIR` in `config.py`).

| Role | Cohort | Platform |
|---|---|---|
| Diagnosis, discovery | [GSE13159](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE13159) (MILE) | Affymetrix HG-U133 Plus 2.0 |
| Diagnosis, external | [GSE114868](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE114868) | Affymetrix HTA 2.0 |
| Diagnosis, external | [GSE9476](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE9476) | Affymetrix HG-U133A |
| Prognosis, discovery | [GSE37642](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE37642) U133A+B (AMLCG) | Affymetrix HG-U133A/B |
| Prognosis, external | GSE37642 U133 Plus 2.0 | Affymetrix HG-U133 Plus 2.0 |
| Prognosis, external | [TCGA-LAML](https://www.cbioportal.org/study/summary?id=laml_tcga_pan_can_atlas_2018) | RNA-seq |
| Prognosis, external; drug response | [BeatAML waves 1–4](https://biodev.github.io/BeatAML2/) | RNA-seq, ex vivo drug screen |

BeatAML keeps one specimen per patient (initial-diagnosis specimen, otherwise the earliest).

## Results

Results will be added after publication of the accompanying article.

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/alwhelat/HFW-NPO-AML.git
cd HFW-NPO-AML

# 2. Create a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

| Package | Version | Purpose |
|---|---|---|
| numpy | ≥ 1.24 | array operations |
| pandas | ≥ 2.0 | data loading |
| scipy | ≥ 1.10 | statistics, Lévy flight |
| scikit-learn | ≥ 1.3 | classifiers, CV, metrics |
| imbalanced-learn | ≥ 0.11 | SMOTE |
| statsmodels | ≥ 0.14 | calibration, statistical tests |
| matplotlib | ≥ 3.7 | figures |
| joblib | ≥ 1.3 | parallel folds |
| openpyxl | ≥ 3.1 | Excel tables |
| requests | ≥ 2.28 | Enrichr API |

## Usage

```bash
python run_all.py --quick                    # end-to-end check with small settings (~10 min)
python run_all.py                            # full experiments (several hours)
python run_all.py --only prognosis           # one task
python run_all.py --skip-prepare --jobs 4    # reuse prepared data, 4 parallel folds
python run_all.py --tag v2                   # write to results_v2/ and figures_v2/
```

Individual steps:

```bash
python scripts/01_prepare_data.py            # build clean tables in data/processed/
python scripts/02_run_diagnosis.py
python scripts/03_run_prognosis.py
python scripts/04_run_chemoresistance.py
python scripts/05_post_analysis.py --tag v2  # post-hoc statistics, external analyses, enrichment
python scripts/06_sensitivity.py --tag v2    # sensitivity to K and alpha
```

**VS Code.** Open the folder, run the task *Setup: create .venv and install requirements*, then pick an experiment in *Run and Debug*. The task *Watch progress log* shows live progress.

**Run time.** One wrapper fitness evaluation takes about 10 ms; the NPO budget (10 × 20 agents × 300 iterations) is about 60,000 evaluations per fold. Folds run in parallel on all but one CPU core (`N_JOBS` in `config.py`).

## Outputs

`results_<tag>/<task>/` and `figures_<tag>/` (not tracked by git):

| File | Content |
|---|---|
| `fold_metrics.csv` | every method × repeat × fold: metrics, genes, run time |
| `summary.csv` | mean ± SD per method |
| `statistical_tests.csv` | Friedman, Wilcoxon-Holm, mean ranks |
| `gene_selection_frequency.csv` | selection frequency of each gene |
| `external_validation.csv` | external cohorts with 95% CI |
| `final_signature_*.csv` | signatures fitted on the full discovery cohort |
| `run_info.json` | settings, git commit, Python and library versions |
| `post/manuscript_tables.xlsx` | all post-hoc tables |

## Project Structure

```
HFW-NPO-AML/
├── config.py                  ← all settings (anchors, K, NPO budget, α, CV)
├── run_all.py                 ← master runner
├── hfwnpo_aml/
│   ├── geo.py                 ← chunked GEO series-matrix reader, probe → gene
│   ├── datasets.py            ← cohort builders
│   ├── hybrid_filter.py       ← Phase 1 filters (classification, survival)
│   ├── npo_optimizer.py       ← binary NPO (published search, unchanged)
│   ├── fitness.py             ← anchored fitness (balanced accuracy, C-index)
│   ├── models.py              ← SMOTE, ensemble, ridge Cox
│   ├── baselines.py           ← GA, PSO, mRMR, RFE-SVM, L1, Cox-LASSO
│   ├── survival.py            ← NumPy Cox models, C-index, Kaplan–Meier
│   ├── pipeline.py            ← method tables and CV engine
│   ├── stats.py, post.py      ← statistical tests and post-hoc analyses
│   ├── plots.py, report.py    ← figures and result files
│   └── progress.py            ← live progress log
├── scripts/                   ← steps 01–06
├── notebooks/                 ← data overview, results viewer
├── data/README.md             ← raw files and download links
├── .vscode/                   ← launch configurations and tasks
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Citation

The AML article is in preparation. Until it is published, please cite the HFW-NPO method:

```bibtex
@article{alwhelat2026hfwnpo,
  title   = {HFW-NPO: A Dual Paradigm Hybrid Filter--Wrapper Nomadic People Optimizer Framework
             for High-Dimensional Alzheimer's Gene Expression Classification},
  author  = {Alwhelat, Almuntadher Mahmood and Abiyev, Rahib H.},
  journal = {Electronics},
  volume  = {15},
  number  = {13},
  pages   = {2970},
  year    = {2026},
  doi     = {10.3390/electronics15132970}
}
```

A `CITATION.cff` file is provided for GitHub's "Cite this repository".

## Authors

**Almuntadher Mahmood Alwhelat**
Near East University / Al-Farabi University College, Baghdad, Iraq
📧 almuntadher.mahmood@alfarabiuc.edu.iq · ORCID [0009-0002-1511-3991](https://orcid.org/0009-0002-1511-3991) · 🔗 [github.com/alwhelat](https://github.com/alwhelat)

**Prof. Rahib H. Abiyev**
Near East University, Nicosia, North Cyprus

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).
