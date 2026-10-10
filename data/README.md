# Raw data

Raw files are not part of this repository. Download them from the sources below and put them in the raw data folder (`RAW_DIR` in `config.py`, default `../dataset`). File names are matched with wildcards, so version numbers in the names can differ.

## Diagnosis

| File | Source | Used for |
|---|---|---|
| `GSE13159_series_matrix.txt.gz` | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE13159 | discovery (U133 Plus 2.0) |
| `GSE114868_series_matrix.txt.gz` | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE114868 | external validation (HTA 2.0) |
| `GPL17586-*.txt` (full annotation table) | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL17586 → "Download full table" | gene symbols for GSE114868 |
| `GSE9476_series_matrix.txt.gz` | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE9476 | external validation (U133A, no CD274 probe) |
| `GPL570-*.txt` (full annotation table) | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL570 | gene symbols for all Affymetrix U133 files |

## Prognosis

| File | Source | Used for |
|---|---|---|
| `GSE37642-GPL96_series_matrix.txt.gz`, `GSE37642-GPL97_series_matrix.txt.gz` | https://ftp.ncbi.nlm.nih.gov/geo/series/GSE37nnn/GSE37642/matrix/ | discovery (U133A + U133B, same patients) |
| `GSE37642-GPL570_series_matrix.txt.gz` | same folder | external validation |
| `data_mrna_seq_v2_rsem.txt`, `laml_tcga_pan_can_atlas_2018_clinical_data.tsv` | https://www.cbioportal.org/study/summary?id=laml_tcga_pan_can_atlas_2018 | external validation (TCGA-LAML) |
| `beataml_waves1to4_norm_exp_dbgap.txt`, `beataml_wv1to4_clinical.xlsx` | https://biodev.github.io/BeatAML2/ | external validation (BeatAML) |

## Drug response

| File | Source | Used for |
|---|---|---|
| `beataml_probit_curve_fits_v4_dbgap.txt` | https://biodev.github.io/BeatAML2/ | ex vivo drug AUC |

`GSE37642_RAW.tar` (CEL files) and `GPL17586_family.soft.gz` are **not** needed.

`python scripts/01_prepare_data.py` writes the clean tables to `data/processed/` (not tracked by git).
