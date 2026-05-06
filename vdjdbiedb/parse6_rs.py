import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import random
import os

PARSED_CSV_COLS = ['id', 'pmhc', 'tcr', 'label']


def load_parsed_csv(path):
    df = pd.read_csv(path)
    if 'pmhc' not in df.columns:
        df = pd.read_csv(path, header=None, names=PARSED_CSV_COLS)
    return df


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]



def generate_negatives(df_subset, ratio, split_type, fold, dist_threshold=3):
    # Copy original data (prevent data contamination)
    data = df_subset.copy()
    
    # Group TCRs by peptide for efficiency
    # { 'Peptide_A': ['tcr1', 'tcr2', ...], ... }
    pep_to_tcrs = data.groupby('peptide')['tcr'].apply(list).to_dict()
    unique_peps = list(pep_to_tcrs.keys())
    
    # Precompute, for each peptide, a list of other compatible peptides with distance > 3
    compatible_peps_map = {}
    for p1 in unique_peps:
        compatible = []
        for p2 in unique_peps:
            if p1 == p2: continue
            if levenshtein_distance(p1, p2) > dist_threshold:
                compatible.append(p2)
        compatible_peps_map[p1] = compatible
    
    final_rows = []
    discard_count = 0
    
    # Iterate over rows and generate negatives
    for idx, row in data.iterrows():
        target_pep = row['peptide']
        target_tcr = row['tcr']
        
        # 1. Get a list of compatible (distant) peptides
        candidate_peps = compatible_peps_map.get(target_pep, [])
        
        # 2. Collect all TCRs from compatible peptides as candidates
        candidate_tcrs = []
        for cp in candidate_peps:
            candidate_tcrs.extend(pep_to_tcrs[cp])
        
        # 3. Exclude TCRs identical to the original (swapping constraint)
        candidate_tcrs = [t for t in candidate_tcrs if t != target_tcr]
        
        # 4. Check whether candidate pool is sufficient (to keep ratios)
        if len(candidate_tcrs) >= ratio:
            # (1) Add positive data
            final_rows.append(row)
            
            # (2) Sample and add negative data
            chosen_neg_tcrs = random.sample(candidate_tcrs, ratio)
            for i, neg_tcr in enumerate(chosen_neg_tcrs):
                neg_row = row.copy()
                neg_row['tcr'] = neg_tcr
                neg_row['label'] = 0  # set label to 0
                neg_row['id'] = f"{row['id']}_{split_type}_{fold}_n{i}_iptm_filtered" 
                final_rows.append(neg_row)
        else:
            # If a valid negative cannot be generated, drop this positive.
            discard_count += 1
            
    #print(f"  - Processed: {len(data)} positives -> Discarded: {discard_count} (Low candidates)")
    return pd.DataFrame(final_rows)




def main():
    split_type = 'rs'
    out_folder = f'dataset_iptm_filtered_{split_type}'
    os.makedirs(out_folder, exist_ok=True)    
    filename = 'iptm_filtered_vdjdbiedb.csv'
    ratio=1
    
    df = load_parsed_csv(filename)
    df['peptide'] = df['pmhc'].apply(lambda x: str(x).split('_')[0])
    
    # Shuffle all data randomly
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_full_idx, test_idx) in enumerate(kf.split(df)):
        print(f"\n[Fold {fold}] Processing...")
        
        train_full = df.iloc[train_full_idx]
        test_pos = df.iloc[test_idx]
        
        # Train : Val : Test = 72 : 8 : 20
        train_pos, val_pos = train_test_split(train_full, test_size=0.1, random_state=42) 
        
        print(len(train_pos))
        print(len(val_pos))
        print(len(test_pos))
            
        # Generate negatives (independently per split to prevent leakage)
        train_final = generate_negatives(train_pos, ratio, split_type, fold, dist_threshold=3)
        val_final = generate_negatives(val_pos, ratio, split_type, fold, dist_threshold=3)
        test_final = generate_negatives(test_pos, ratio, split_type, fold, dist_threshold=3)
        
        train_final.drop(columns=['peptide']).to_csv(f'{out_folder}/fold{fold}_train.csv', index=False)
        val_final.drop(columns=['peptide']).to_csv(f'{out_folder}/fold{fold}_val.csv', index=False)
        test_final.drop(columns=['peptide']).to_csv(f'{out_folder}/fold{fold}_test.csv', index=False)
        
        print(f"  Saved: fold{fold}_train.csv ({len(train_final)} rows), "
              f"fold{fold}_val.csv ({len(val_final)} rows), "
              f"fold{fold}_test.csv ({len(test_final)} rows)")

        
        
        
if __name__ == "__main__":
    main()