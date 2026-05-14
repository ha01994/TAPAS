import os
import glob
import numpy as np
import pandas as pd

def extract_pae_features(pae_dir):
    # Collect all interface PAE .npy files under pae_dir
    file_pattern = os.path.join(pae_dir, '*_interface_pae.npy')
    pae_files = glob.glob(file_pattern)
    
    features_list = []
    
    for file_path in pae_files:
        # Sample id from filename prefix before _interface_pae (e.g. v0, v1000)
        file_name = os.path.basename(file_path)
        sample_id = file_name.split('_interface_pae')[0]
        
        # Load PAE matrix
        pae_matrix = np.load(file_path)
        
        # Replace NaN/Inf for stable statistics (clip near AF3 PAE upper range)
        if np.isnan(pae_matrix).any() or np.isinf(pae_matrix).any():
            pae_matrix = np.nan_to_num(pae_matrix, nan=31.75, posinf=31.75)

        # Flatten for distribution summaries
        pae_flat = pae_matrix.flatten()
        
        # ---------------------------------------------------------
        # 1. Basic moments (mean, min, max, std)
        # ---------------------------------------------------------
        mean_pae = np.mean(pae_flat)
        min_pae = np.min(pae_flat)
        max_pae = np.max(pae_flat)
        std_pae = np.std(pae_flat)
        
        # ---------------------------------------------------------
        # 2. Percentiles and median (outlier-robust)
        # ---------------------------------------------------------
        median_pae = np.median(pae_flat)
        p10_pae = np.percentile(pae_flat, 10)
        p90_pae = np.percentile(pae_flat, 90)
        
        # ---------------------------------------------------------
        # 3. Threshold fractions (e.g. strong contact vs. broken interface)
        # ---------------------------------------------------------
        # Fraction of pairs with PAE below 5 Å (tight interface)
        frac_less_5 = np.mean(pae_flat < 5.0)

        # Fraction of pairs with PAE above 15 Å (effectively no contact)
        frac_greater_15 = np.mean(pae_flat > 15.0)
        
        # ---------------------------------------------------------
        # 4. Optional asymmetry (square matrices only)
        # ---------------------------------------------------------
        # Only valid when pae_matrix is square (full interface block stored).
        # If you only stored a rectangular slice (e.g. TCR x peptide), keep asymmetry at 0.
        asymmetry = 0.0
        if pae_matrix.shape[0] == pae_matrix.shape[1]:
            # Mean |PAE_ij - PAE_ji|
            asymmetry = np.mean(np.abs(pae_matrix - pae_matrix.T))

        # One row of scalar features
        features = {
            'sample_id': sample_id,
            'pae_mean': mean_pae,
            'pae_min': min_pae,
            'pae_max': max_pae,
            'pae_std': std_pae,
            'pae_median': median_pae,
            'pae_p10': p10_pae,
            'pae_p90': p90_pae,
            'pae_frac_lt_5': frac_less_5,
            'pae_frac_gt_15': frac_greater_15,
            'pae_asymmetry': asymmetry  # non-zero only for square interface matrices
        }
        
        features_list.append(features)
        
    # Build feature table
    df_features = pd.DataFrame(features_list)
    return df_features

# ==========================================
# Script entry (adjust paths for your machine)
# ==========================================
pae_directory = './pae_matrix_af3_vdjdb'

# Build features from saved interface PAE tensors
pae_features_df = extract_pae_features(pae_directory)

# Write CSV for merging with AF3 confidence / ipTM columns downstream
pae_features_df.to_csv('pae_feat_af3_vdjdb.csv', index=False)

print("PAE feature extraction done. Preview (10 rows):")
print(pae_features_df.head(10))
