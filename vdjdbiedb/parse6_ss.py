import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import random
import os

PARSED_CSV_COLS = ['id', 'pmhc', 'tcr', 'label']


def load_parsed_csv(path):
    """헤더가 있으면 그대로 사용하고, 없으면 id/pmhc/tcr/label 컬럼을 부여한다."""
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
    """
    Strict Split 환경에서는 df_subset 안에 있는 Peptide들끼리만
    TCR Swapping이 일어납니다. (Data Leakage 방지)
    """
    data = df_subset.copy()
    
    # TCR을 Peptide별로 그룹화
    pep_to_tcrs = data.groupby('peptide')['tcr'].apply(list).to_dict()
    unique_peps = list(pep_to_tcrs.keys())
    
    # 거리 계산 및 호환성 맵 생성
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
    
    for idx, row in data.iterrows():
        target_pep = row['peptide']
        target_tcr = row['tcr']
        
        candidate_peps = compatible_peps_map.get(target_pep, [])
        
        candidate_tcrs = []
        for cp in candidate_peps:
            candidate_tcrs.extend(pep_to_tcrs[cp])
        
        candidate_tcrs = [t for t in candidate_tcrs if t != target_tcr]
        
        if len(candidate_tcrs) >= ratio:
            # Positive
            final_rows.append(row)
            # Negative
            chosen_neg_tcrs = random.sample(candidate_tcrs, ratio)
            for i, neg_tcr in enumerate(chosen_neg_tcrs):
                neg_row = row.copy()
                neg_row['tcr'] = neg_tcr
                neg_row['label'] = 0
                neg_row['id'] = f"{row['id']}_{split_type}_{fold}_n{i}_iptm_filtered" 
                final_rows.append(neg_row)
        else:
            discard_count += 1
            
    #print(f"  - Processed: {len(data)} positives -> Discarded: {discard_count} (Low candidates)")
    return pd.DataFrame(final_rows)




def main():
    split_type = 'ss'
    out_folder = f'dataset_iptm_filtered_{split_type}'
    os.makedirs(out_folder, exist_ok=True)    
    filename = 'iptm_filtered_vdjdbiedb.csv'
    ratio=1
    
    df = load_parsed_csv(filename)
    df['peptide'] = df['pmhc'].apply(lambda x: str(x).split('_')[0])
    
    unique_peptides = df['peptide'].unique()
    print(f"Total Unique Peptides: {len(unique_peptides)}")
    
    # Peptide 기준 5-Fold Split
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_val_pep_idx, test_pep_idx) in enumerate(kf.split(unique_peptides)):
        print(f"\n[Fold {fold}] Processing Strict Split...")
        
        # Peptide ID 분리
        test_peps = unique_peptides[test_pep_idx]
        train_val_peps = unique_peptides[train_val_pep_idx]
        
        # Train : Val : Test = 72 : 8 : 20
        train_peps, val_peps = train_test_split(train_val_peps, test_size=0.1, random_state=42)
        
        train_pos = df[df['peptide'].isin(train_peps)]
        val_pos = df[df['peptide'].isin(val_peps)]
        test_pos = df[df['peptide'].isin(test_peps)]        
        print(f"  Peptide Count - Train: {len(train_peps)}, Val: {len(val_peps)}, Test: {len(test_peps)}")
        
        print(len(train_pos))
        print(len(val_pos))
        print(len(test_pos))
        
        # Negative 생성 (각 Split 별로 독립적으로 수행하여 Leakage 방지)        
        train_final = generate_negatives(train_pos, ratio, split_type, fold, dist_threshold=3)
        val_final = generate_negatives(val_pos, ratio, split_type, fold, dist_threshold=3)
        test_final = generate_negatives(test_pos, ratio, split_type, fold, dist_threshold=3)
        
        train_final.drop(columns=['peptide']).to_csv(f'{out_folder}/fold{fold}_train.csv', index=False)
        val_final.drop(columns=['peptide']).to_csv(f'{out_folder}/fold{fold}_val.csv', index=False)
        test_final.drop(columns=['peptide']).to_csv(f'{out_folder}/fold{fold}_test.csv', index=False)
        
        print(f"  Saved to {out_folder}: Train({len(train_final)}), Val({len(val_final)}), Test({len(test_final)})")

        
if __name__ == "__main__":
    main()