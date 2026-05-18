# TAPAS 

Code for predicting TCR–pMHC binding using AlphaFold3 (AF3) confidence and contact metrics, summarized PAE interface features, and ESM-2 sequence embeddings.

## Environment

```bash
conda env create -f environment.yml
conda activate tabpfn

# Install ANARCI
git clone https://github.com/oxpig/ANARCI.git
cd ANARCI
python setup.py install
```

## Layout

```
.
├── environment.yml
├── README.md
│
├── tapas_vdjdb_pos_af3/                 # VDJdb positives
│   ├── vdjdb123.csv
│   ├── analyze_model_quality_metrics.py
│   ├── analyze_crystal_vs_model.py
│   ├── pdockq2_json_interface.py
│   ├── only_leave_ones_with_best_iptm_tcrpmhc.py
│   ├── make_sites_file.py, script_make_sites.py
│   ├── extract_pae_matrix.py, extract_pae_features.py
│   ├── get_esm.py
│   ├── results_model_quality_metrics*.csv
│   └── …
│
├── tapas_vdjdb_neg_af3_PART1/          # negatives, part 1 
├── tapas_vdjdb_neg_af3_PART2/          # negatives, part 2
│
└── test_tapas/
    ├── dataset_rs/                      # random split, fold0–4
    ├── dataset_ss/                      # shared split, fold0–4
    ├── pae_feat_af3_vdjdb.csv
    ├── pae_feat_af3_vdjdb_neg_part1.csv
    ├── pae_feat_af3_vdjdb_neg_part2.csv
    ├── results_model_quality_metrics_pos_best.csv
    ├── results_model_quality_metrics_neg1_best.csv
    ├── results_model_quality_metrics_neg2_best.csv
    ├── test_tapas.py          
    ├── test_zeroshot_all.py
    └── results_auc/                         
```

## Suggested workflow

1. **AF3 Confidence Metrics**:<br>
- In each positive / negative folder, build `results_model_quality_metrics.csv` from AF3 outputs (`analyze_model_quality_metrics.py`).<br>
- Run `only_leave_ones_with_best_iptm_tcrpmhc.py` to write `results_model_quality_metrics_best.csv`. That step selects the best-scoring structure per id. Copy it to `train_tapas/`.<br>

2. **PAE features (reference code only in `tapas_vdjdb_pos_af3/`)**:<br>
- Run `extract_pae_matrix.py` to generate `*_interface_pae.npy` under `pae_matrix_af3_vdjdb/`.<br>
- Then run `extract_pae_features.py` to generate `pae_feat_af3_vdjdb.csv`, then copy it to `train_tapas/`.<br>
    
3. **ESM-2 Features (reference code only in `tapas_vdjdb_pos_af3/`)**:<br>
- Run `get_esm.py` to produce `esm_embeddings_map_vdjdb.npy`, then copy it to `train_tapas/`.<br>

4. **Inference**:<br>
- From `test_tapas/`, run `python test_tapas.py`.<br>
- Use `test_zeroshot_all.py` for single-score zero-shot summaries.<br>


## Acknowledgements

AF3 confidence metric extraction in this repository drew heavily on structure from [AF3TCRpMHC](https://github.com/Alexasparis/AF3TCRpMHC). 

Reference: Ascunce-París *et al.*, *A Unified Framework for TCR-pMHC Structural Model Assessment*, [doi:10.1101/2025.10.09.681411](https://doi.org/10.1101/2025.10.09.681411).

