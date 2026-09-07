# TAPAS

Source code and benchmark inputs used to reproduce TAPAS, a TabPFN model for
TCR–pMHC binding prediction. 

The final TAPAS input contains 303 features:

- 4 AF3 confidence features: `avgipae_pmhc`, `avgipae_tcr`,
  `pdockq2_pmhc`, and `pdockq2_tcr`.
- 11 AF3 geometry features describing CDR3–peptide contacts and the predicted
  TCR–pMHC pose. Their exact names are defined in the dataset-specific training
  scripts.
- 288 ESM-2 features. Mean-pooled 1,280-dimensional embeddings are generated
  for the peptide and the six TCR CDRs, then reduced by PCA.

For every complex, the training scripts use confidence and geometry features
from the structure with the highest AF3 `ranking_score` among its five
diffusion samples.

## Environment

```bash
conda env create -f environment.yml
conda activate tabpfn
```

The ePytope and ImmRep25 ESM preparation scripts use ANARCI. Install it with
the following commands:

```bash
git clone https://github.com/oxpig/ANARCI.git
cd ANARCI
python setup.py install
```

## Expected AF3 output layout

By default, feature extraction searches under the following untracked paths:

```text
af3_outputs/
├── vdjdb/
├── epytope_tcr_viral/
└── immrep25/
```

Each root must contain one AF3 job directory per pair. Each job directory must
contain the standard AF3 sample directories and their `model.cif`,
`summary_confidences.json`, and `confidences.json` files. The scripts also read
AF3 `ranking_score` from the job's ranking CSV or sample summary JSON.
If AF3 jobs are distributed across multiple directories, supply each root by
repeating `--structure-root` for confidence extraction or `--output-dir` for
geometry extraction.

## Reproducing the feature tables

Run commands from the repository root.

### VDJdb

```bash
python af3_confidence/analyze_model_quality_metrics_vdjdb.py
python af3_geometry/extract_af3_geometry_features_vdjdb.py

python tapas/vdjdb/get_esm.py
```

The VDJdb geometry and ESM scripts read their source and lookup tables from
`tapas/vdjdb/data/`.
`get_esm.py` reconstructs
the required peptide and CDR table directly from `parsed_data_final.csv`,
`negatives.csv`, and `dic_full_vavb.csv`.

### ePytope viral benchmark

```bash
python af3_confidence/analyze_model_quality_metrics_epytope_tcr_viral.py
python af3_geometry/extract_af3_geometry_features_epytope_tcr_viral.py

python tapas/epytope/get_esm.py
```

The ePytope geometry and ESM scripts read the active pair manifest and
full-chain TCR sequences from `tapas/epytope/data/manifest.csv` and
`tapas/epytope/data/tcr_sequences.csv`. `get_esm.py` uses ANARCI/IMGT to derive
CDR1/2 and validates the manifest-provided CDR3 sequences against the ANARCI
assignments.

### ImmRep25

```bash
python af3_confidence/analyze_model_quality_metrics_immrep25.py
python af3_geometry/extract_af3_geometry_features_immrep25.py

python tapas/immrep25/get_esm.py
```

The ImmRep25 geometry script reads `immrep25_pairs.csv`, `immrep25_tcrs.csv`,
and `mhc_i_protein_seq.csv` from `tapas/immrep25/data/`. `get_esm.py`
reconstructs the peptide and six TCR CDR inputs directly from
`data/immrep25.tsv` and `data/immrep25_pairs.csv` using ANARCI/IMGT.

## Training and evaluation

After generating `esm_embeddings_map_vdjdb.npy` and the external-dataset ESM
maps, run the following dataset-specific scripts from the repository root.

### VDJdb

```bash
python tapas/vdjdb/train_tabpfn.py
```

`train_tabpfn.py` evaluates five-fold VDJdb RS and SS.

### ePytope viral benchmark

```bash
python tapas/epytope/train_tabpfn_ensemble.py
```

The reported TAPAS value uses the ten-model ensemble trained on the five RS
and five SS full-fold datasets. 

### ImmRep25

```bash
python tapas/immrep25/train_tabpfn_ensemble.py
```

## SHAP analysis

The SHAP scripts interpret the structural-only TAPAS submodel using the 4 AF3
confidence and 11 geometry features; ESM-2 features are excluded. Run them from
the repository root after generating the confidence and geometry feature
tables:

```bash
python tapas/vdjdb/shap_rs_ss_fold0.py
python tapas/epytope/shap_epytope_tcr.py
python tapas/immrep25/shap_immrep25.py
```

The scripts save numeric SHAP summaries and figures under their respective
dataset directories.
