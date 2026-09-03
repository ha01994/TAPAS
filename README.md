# TAPAS

Source code and benchmark inputs used to reproduce TAPAS, a TabPFN model for
TCR–pMHC binding prediction. 

## Model features

The final TAPAS input contains 303 features:

- 4 AF3 confidence features: `avgipae_pmhc`, `avgipae_tcr`,
  `pdockq2_pmhc`, and `pdockq2_tcr`.
- 11 AF3 geometry features describing CDR3–peptide contacts and the predicted
  TCR–pMHC pose. Their exact names are defined by `FINAL_GEOMETRY_COLS` in each
  `train_tabpfn_best.py`.
- 288 ESM-2 features. Mean-pooled 1,280-dimensional embeddings are generated
  for the peptide and the six TCR CDRs, then reduced by PCA. PCA is fitted only
  on the relevant VDJdb training rows.

For every complex, the confidence and geometry tables use the structure with
the highest AF3 `ranking_score` among its five diffusion samples.

## Repository layout

```text
.
├── af3_confidence/
│   ├── analyze_model_quality_metrics_common.py
│   ├── analyze_model_quality_metrics_vdjdb.py
│   ├── analyze_model_quality_metrics_epytope_tcr_viral.py
│   ├── analyze_model_quality_metrics_immrep25.py
│   └── pdockq2_json_interface.py
├── af3_geometry/
│   ├── extract_af3_geometry_samples_common.py
│   ├── extract_af3_geometry_features_vdjdb.py
│   ├── extract_af3_geometry_features_epytope_tcr_viral.py
│   └── extract_af3_geometry_features_immrep25.py
└── tabpfn_developed/
    ├── tabpfn_vdjdb_combined_af3/
    │   ├── dataset_rs/                     # VDJdb random split, folds 0–4
    │   ├── dataset_ss/                     # VDJdb strict split, folds 0–4
    │   └── af3_inputs/                     # VDJdb source/lookup tables
    ├── tabpfn_epytope_af3/
    │   ├── original_data/viral_8peptide.csv
    │   └── af3_inputs/                     # 3,560-pair manifest and TCRs
    └── tabpfn_immrep25_af3/
        ├── immrep25.tsv                    # original benchmark table
        ├── immrep25_pairs.csv
        └── immrep25_tcrs.csv
```

Precomputed AF3 confidence and geometry feature tables are not distributed.
The extraction scripts write them to dataset-specific subdirectories under
`af3_confidence/` and `af3_geometry/`.

The ePytope benchmark in this repository contains exactly 3,560 pairs from 445
TCRs and these eight viral peptides:
`AYAQKIFKI`, `CTELKLSDY`, `FPQSAPHGV`, `KCYGVSPTK`, `LTDEMIAQY`,
`NYNYLYRLF`, `SPRRARSVA`, and `TYGPVFMCL`.
`original_data/viral_8peptide.csv` contains the corresponding 445 cognate TCR
rows in the source ePytope table format. ImmRep25 retains its original
10,000-row `immrep25.tsv`; the pair and TCR CSVs are deterministic downstream
tables used by feature extraction and evaluation.

## Environment

```bash
conda env create -f environment.yml
conda activate tabpfn
```

The ESM preparation scripts use ANARCI. If it is not already available in the
environment, install it from its upstream repository:

```bash
git clone https://github.com/oxpig/ANARCI.git
cd ANARCI
python setup.py install
```

TabPFN and ESM-2 download pretrained weights on first use. A CUDA-capable GPU is
recommended for training and embedding generation. The scripts default to
`cuda:0`.

## Expected AF3 output layout

By default, feature extraction searches under the following untracked paths:

```text
af3_outputs/
├── vdjdb_positive/
├── vdjdb_negative_part1/
├── vdjdb_negative_part2/
├── epytope_tcr_viral/
└── immrep25/
```

Each root must contain one AF3 job directory per pair. Each job directory must
contain the standard AF3 sample directories and their `model.cif`,
`summary_confidences.json`, and `confidences.json` files. The scripts also read
AF3 `ranking_score` from the job's ranking CSV or sample summary JSON.
Alternative roots can be supplied with repeated `--structure-root` for
confidence extraction and repeated `--output-dir` for geometry extraction.

## Reproducing the feature tables

Run commands from the repository root.

#### VDJdb

Confidence extraction must run before geometry extraction because the latter
also uses the generated median-sample selection table.

```bash
python af3_confidence/analyze_model_quality_metrics_vdjdb.py
python af3_geometry/extract_af3_geometry_features_vdjdb.py

cd tabpfn_developed/tabpfn_vdjdb_combined_af3
python get_esm.py
```

The input metadata used by the VDJdb geometry extractor is in
`tabpfn_developed/tabpfn_vdjdb_combined_af3/af3_inputs/`.

#### ePytope viral set

```bash
python af3_confidence/analyze_model_quality_metrics_epytope_tcr_viral.py
python af3_geometry/extract_af3_geometry_features_epytope_tcr_viral.py

cd tabpfn_developed/tabpfn_epytope_af3
python get_esm.py --device cuda:0
```

Use `python get_esm.py --prepare-only` to build and validate the 3,560-row CDR
table in memory without loading ESM-2. Add `--cdr-csv PATH` only when a copy of
that intermediate table is needed.

#### ImmRep25

```bash
python af3_confidence/analyze_model_quality_metrics_immrep25.py
python af3_geometry/extract_af3_geometry_features_immrep25.py

cd tabpfn_developed/tabpfn_immrep25_af3
python get_esm.py
```

The ImmRep25 ESM script reconstructs IMGT CDR1/2/3 features directly from
`immrep25.tsv` and takes pair IDs and labels from `immrep25_pairs.csv`; no
VDJdb-like intermediate CSV is required.

## Training and evaluation

After generating `esm_embeddings_map_vdjdb.npy` and the external-dataset ESM
maps, run the dataset-specific scripts from their own directories.

#### VDJdb random and strict splits

```bash
cd tabpfn_developed/tabpfn_vdjdb_combined_af3
python train_tabpfn_best.py
```

`train_tabpfn_best.py` evaluates five-fold VDJdb RS and SS. 

#### ePytope viral benchmark

```bash
cd tabpfn_developed/tabpfn_epytope_af3
python train_tabpfn_ensemble.py
```

The reported TAPAS value uses the ten-model ensemble trained on the five RS
and five SS full-fold datasets. 

#### ImmRep25

```bash
cd tabpfn_developed/tabpfn_immrep25_af3
python train_tabpfn_ensemble.py
```

ImmRep25 postprocessing is implemented directly in `train_tabpfn_best.py` and
reused by the ensemble and ablation scripts. Scores are double-centered within
each HLA-specific TCR-by-peptide matrix. TCRdist3 is then computed from the
provided CDR1, CDR2, and CDR3 sequences of both alpha and beta chains. Connected
components at distance 120 implement single-linkage clusters. For each HLA,
cluster, and peptide, the final score is the cluster mean multiplied by the
square root of cluster size. 
