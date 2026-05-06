import os
import glob
import numpy as np
import pandas as pd

def extract_pae_features(pae_dir):
    # Find paths to all PAE npy files in the folder
    file_pattern = os.path.join(pae_dir, '*_interface_pae.npy')
    pae_files = glob.glob(file_pattern)
    
    features_list = []
    
    for file_path in pae_files:
        # Extract ID from filename (e.g., v0, v1000)
        file_name = os.path.basename(file_path)
        sample_id = file_name.split('_interface_pae')[0]
        
        # Load PAE matrix
        pae_matrix = np.load(file_path)
        
        # Check and handle NaN/Inf in the matrix (robustness)
        if np.isnan(pae_matrix).any() or np.isinf(pae_matrix).any():
            pae_matrix = np.nan_to_num(pae_matrix, nan=31.75, posinf=31.75) # clamp near AF3 max PAE
            
        # Flatten 2D matrix to 1D for statistics
        pae_flat = pae_matrix.flatten()
        
        # ---------------------------------------------------------
        # 1. Basic statistical features (mean, min, max, std)
        # ---------------------------------------------------------
        mean_pae = np.mean(pae_flat)
        min_pae = np.min(pae_flat)
        max_pae = np.max(pae_flat)
        std_pae = np.std(pae_flat)
        
        # ---------------------------------------------------------
        # 2. Percentiles and median (outlier-robust features)
        # ---------------------------------------------------------
        median_pae = np.median(pae_flat)
        p10_pae = np.percentile(pae_flat, 10)
        p90_pae = np.percentile(pae_flat, 90)
        
        # ---------------------------------------------------------
        # 3. Threshold-based fraction features
        # ---------------------------------------------------------
        # Fraction of regions considered strongly bound (e.g., <= 5Å)
        frac_less_5 = np.mean(pae_flat < 5.0)
        
        # Fraction of regions considered effectively unbound (e.g., >= 15Å)
        frac_greater_15 = np.mean(pae_flat > 15.0)
        
        # ---------------------------------------------------------
        # 4. Asymmetry (optional)
        # ---------------------------------------------------------
        # *Note*: this part only works when pae_matrix is square.
        # i.e., when the full [TCR+Pep] x [TCR+Pep] interface is stored.
        # If only a rectangular matrix (TCR x Pep) is stored, skip this part.
        asymmetry = 0.0
        if pae_matrix.shape[0] == pae_matrix.shape[1]:
            # Mean of |PAE_ij - PAE_ji|
            asymmetry = np.mean(np.abs(pae_matrix - pae_matrix.T))
            
        # Store results in a dictionary
        features = {
            'sample_id': sample_id,
            'pae_mean': mean_pae,
            #'pae_min': min_pae,
            'pae_max': max_pae,
            'pae_std': std_pae,
            'pae_median': median_pae,
            'pae_p10': p10_pae,
            'pae_p90': p90_pae,
            'pae_frac_lt_5': frac_less_5,
            'pae_frac_gt_15': frac_greater_15,
            'pae_asymmetry': asymmetry  # uncomment if matrix is square
        }
        
        features_list.append(features)
        
    # Convert to a dataframe
    df_features = pd.DataFrame(features_list)
    return df_features

# ==========================================
# Execution
# ==========================================
# Please set this to the actual folder path.
pae_directory = './pae_matrix_vdjdbiedb' 

# Feature extraction
pae_features_df = extract_pae_features(pae_directory)

# So it can be merged with other feature dataframes (ipTM, etc.) for TabPFN input
# Save as a CSV file.
pae_features_df.to_csv('pae_feat_vdjdbiedb.csv', index=False)

print("PAE feature extraction complete! Preview of 10 samples:")
print(pae_features_df.head(10))
