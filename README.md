# TAPAS: TCR-Antigen Prediction via AlphaFold Structural confidence

> **Paper**: *Structure-Informed TCR-pMHC Binding Prediction Generalizes to Unseen Peptides*

## Overview

TAPAS predicts TCR–pMHC binding by combining AlphaFold3 structural confidence signals with a TabPFN classifier.
AlphaFast is used to efficiently generate AF3 confidence metrics (ipTM, Conf) and PAE interface features for large TCR–pMHC datasets.

**Feature set**: AF3 Conf + PAE interface features + ESM-2 embeddings

**Training data**: VDJdb + IEDB (score=0 included), filtered to ipTM ≥ 0.7 → 2,194 positives

## Environment

```bash
conda env create -f environment.yml
```

## Pipeline

1. `vdjdbiedb/`
   - Build dataset splits (RS/SS, folds)
   - Generate negatives

2. `pos/` and `neg/` (after ipTM filtering)
   - Compute AF3 confidence metrics (Conf)
   - Extract PAE interface features
   - Compute ESM-2 embeddings

3. `combined/` (after ipTM filtering)
   - Train and evaluate the TAPAS TabPFN classifier

## Repository Structure

```
.
├── vdjdbiedb/                          # Dataset preparation
│   ├── dataset_rs/                     # Random-split fold CSVs (fold0~4)
│   ├── dataset_ss/                     # Shared-split fold CSVs (fold0~4)
│   ├── dataset_iptm_filtered_rs/       # ipTM-filtered random-split folds
│   └── dataset_iptm_filtered_ss/       # ipTM-filtered shared-split folds
│
├── tapas_vdjdbiedb_after_iptm_pos/     # Positive sample feature extraction
│   ├── make_sites_file.py              # Generate AlphaFast sites input
│   ├── extract_pae_matrix.py           # Extract PAE matrices from AF3 outputs
│   ├── extract_pae_features.py         # Compute PAE interface features
│   ├── get_esm_emb.py                  # Extract ESM-2 embeddings
│   ├── analyze_model_quality_metrics.py
│   ├── only_leave_ones_with_best_iptm_tcrpmhc.py
│   ├── pae_feat_vdjdbiedb.csv          # Extracted PAE features (positives)
│   └── results_model_quality_metrics*.csv
│
├── tapas_vdjdbiedb_after_iptm_neg/     # Negative sample feature extraction
│   ├── make_sites_file.py
│   ├── extract_pae_matrix.py
│   ├── extract_pae_features.py
│   ├── analyze_model_quality_metrics_*.py
│   ├── only_leave_ones_with_best_iptm_tcrpmhc.py
│   ├── pae_feat_vdjdbiedb_neg.csv      # Extracted PAE features (negatives)
│   └── results_model_quality_metrics*.csv
│
├── tapas_vdjdbiedb_after_iptm_combined/   # TAPAS training & evaluation
│   ├── train_tapas_conf.py             # Train: Conf features only
│   ├── train_tapas_pae.py              # Train: PAE features only
│   ├── train_tapas_conf_pae.py         # Train: Conf + PAE
│   ├── train_tapas_esm_conf_pae.py     # Train: Conf + PAE + ESM-2
│   ├── test_zeroshot_iptm.py           # Zero-shot ipTM evaluation
│   ├── test_zeroshot_individual.py     # Zero-shot per-feature evaluation
│   └── results_auc/                    # AUC result CSVs per feature set & split
│
├── environment.yml                     # Conda environment
└── README.md
```

