import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# The three folders correspond to OUT_DIR in extract_pae_matrix.py
PAE_DIRECTORIES = [
    SCRIPT_DIR / 'pae_matrix_vdjdbiedb_neg_0',
    SCRIPT_DIR / 'pae_matrix_vdjdbiedb_neg_1',
    SCRIPT_DIR / 'pae_matrix_vdjdbiedb_neg_2',
]

OUTPUT_CSV = SCRIPT_DIR / 'pae_feat_vdjdbiedb_neg.csv'


def extract_pae_features(pae_dir):
    file_pattern = os.path.join(pae_dir, '*_interface_pae.npy')
    pae_files = glob.glob(file_pattern)

    features_list = []

    for file_path in pae_files:
        file_name = os.path.basename(file_path)
        sample_id = file_name.split('_interface_pae')[0]

        pae_matrix = np.load(file_path)

        if np.isnan(pae_matrix).any() or np.isinf(pae_matrix).any():
            pae_matrix = np.nan_to_num(pae_matrix, nan=31.75, posinf=31.75)

        pae_flat = pae_matrix.flatten()

        mean_pae = np.mean(pae_flat)
        max_pae = np.max(pae_flat)
        std_pae = np.std(pae_flat)

        median_pae = np.median(pae_flat)
        p10_pae = np.percentile(pae_flat, 10)
        p90_pae = np.percentile(pae_flat, 90)

        frac_less_5 = np.mean(pae_flat < 5.0)
        frac_greater_15 = np.mean(pae_flat > 15.0)

        asymmetry = 0.0
        if pae_matrix.shape[0] == pae_matrix.shape[1]:
            asymmetry = np.mean(np.abs(pae_matrix - pae_matrix.T))

        features = {
            'sample_id': sample_id,
            'pae_mean': mean_pae,
            'pae_max': max_pae,
            'pae_std': std_pae,
            'pae_median': median_pae,
            'pae_p10': p10_pae,
            'pae_p90': p90_pae,
            'pae_frac_lt_5': frac_less_5,
            'pae_frac_gt_15': frac_greater_15,
            'pae_asymmetry': asymmetry,
        }

        features_list.append(features)

    return pd.DataFrame(features_list)


def main():
    dfs = []
    for pae_run, pae_dir in enumerate(PAE_DIRECTORIES):
        pae_dir = str(pae_dir)
        if not os.path.isdir(pae_dir):
            print(f"[WARN] directory missing, skipping: {pae_dir}")
            continue
        df = extract_pae_features(pae_dir)
        if df.empty:
            print(f"[WARN] npy not found: {pae_dir}")
            continue
        df['pae_run'] = pae_run
        dfs.append(df)
        print(f"[OK] {pae_dir} → {len(df)} rows")

    if not dfs:
        print("No rows to save.")
        return

    pae_features_df = pd.concat(dfs, ignore_index=True)
    pae_features_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaving PAE features: {OUTPUT_CSV} (total {len(pae_features_df)} rows)")
    print(pae_features_df.head(10))


if __name__ == '__main__':
    main()
