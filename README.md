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
└── train_tapas/
    ├── dataset_rs/                      # random split, fold0–4
    ├── dataset_ss/                      # shared split, fold0–4
    ├── pae_feat_af3_vdjdb.csv
    ├── pae_feat_af3_vdjdb_neg_part1.csv
    ├── pae_feat_af3_vdjdb_neg_part2.csv
    ├── results_model_quality_metrics_pos_best.csv
    ├── results_model_quality_metrics_neg1_best.csv
    ├── results_model_quality_metrics_neg2_best.csv
    ├── train_tapas_esm_conf_pae.py          
    ├── test_zeroshot_all.py
    └── results_auc/                         
```

## Suggested workflow

1. **AF3 Confidence Metrics**: In each positive / negative folder, build `results_model_quality_metrics.csv` from AF3 outputs (`analyze_model_quality_metrics.py`). If you have multiple decoys per `pdb_id`, run `only_leave_ones_with_best_iptm_tcrpmhc.py` to write `*_best.csv`. That step selects the best-scoring structure per id. 

2. **PAE features (reference code only in `tapas_vdjdb_pos_af3/`)**: after you have `results_model_quality_metrics_best.csv` and a sites file for interface residues (see `make_sites_file.py` / `script_make_sites.py`), run `extract_pae_matrix.py` to slice AF3 full PAE JSON into per-complex `*_interface_pae.npy` under `pae_matrix_af3_vdjdb/` (paths such as `AF_OUTPUT_DIR`, `BEST_CSV`, `SITES_TXT`, and `OUT_DIR` are set at the top of that script—point them at your AF3 output tree). Then run `extract_pae_features.py` to aggregate those `.npy` files into `pae_feat_af3_vdjdb.csv` (adjust `pae_directory` in the script if needed). Copy the resulting `pae_feat_*.csv` files into `train_tapas/` next to the training scripts (negative parts use the same column schema if you generate them with an equivalent pipeline).

3. **ESM-2 Features**: For ESM inference, run `tapas_vdjdb_pos_af3/get_esm.py` to produce `esm_embeddings_map_vdjdb.npy`, then copy or symlink it into `train_tapas/`. 

4. From `train_tapas/`, run `python train_tapas_esm_conf_pae.py`.

5. Use `test_zeroshot_all.py` for single-score zero-shot summaries.

## Acknowledgements

AF3 confidence metric extraction in this repository drew heavily on structure from [AF3TCRpMHC](https://github.com/Alexasparis/AF3TCRpMHC). 

Reference: Ascunce-París *et al.*, *A Unified Framework for TCR-pMHC Structural Model Assessment*, [doi:10.1101/2025.10.09.681411](https://doi.org/10.1101/2025.10.09.681411).

