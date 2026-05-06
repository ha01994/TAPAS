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
    df_subset: 처리할 데이터프레임 (Positive만 존재한다고 가정)
    ratio: Positive 1개당 생성할 Negative 개수
    dist_threshold: Peptide 간 거리 임계값 (기본 3)
    """
    # 원본 데이터 복사 (데이터 오염 방지)
    data = df_subset.copy()
    
    # 최적화를 위해 TCR을 Peptide별로 그룹화
    # { 'Peptide_A': ['tcr1', 'tcr2', ...], ... }
    pep_to_tcrs = data.groupby('peptide')['tcr'].apply(list).to_dict()
    unique_peps = list(pep_to_tcrs.keys())
    
    # 각 Peptide별로 "거리가 3보다 큰" 호환 가능한 다른 Peptide 목록을 미리 계산
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
    
    # 각 행(Row)을 순회하며 Negative 생성
    for idx, row in data.iterrows():
        target_pep = row['peptide']
        target_tcr = row['tcr']
        
        # 1. 호환되는(거리가 먼) 펩타이드 목록 가져오기
        candidate_peps = compatible_peps_map.get(target_pep, [])
        
        # 2. 호환되는 펩타이드들의 TCR을 모두 후보군으로 수집
        candidate_tcrs = []
        for cp in candidate_peps:
            candidate_tcrs.extend(pep_to_tcrs[cp])
        
        # 3. 자기 자신과 같은 TCR은 제외 (Swapping 조건)
        candidate_tcrs = [t for t in candidate_tcrs if t != target_tcr]
        
        # 4. 후보군이 충분한지 확인 (비율 유지를 위해)
        if len(candidate_tcrs) >= ratio:
            # (1) Positive 데이터 추가
            final_rows.append(row)
            
            # (2) Negative 데이터 샘플링 및 추가
            chosen_neg_tcrs = random.sample(candidate_tcrs, ratio)
            for i, neg_tcr in enumerate(chosen_neg_tcrs):
                neg_row = row.copy()
                neg_row['tcr'] = neg_tcr
                neg_row['label'] = 0  # Label 0으로 변경
                neg_row['id'] = f"{row['id']}_{split_type}_{fold}_n{i}_iptm_filtered" 
                final_rows.append(neg_row)
        else:
            # 조건을 만족하는 Negative를 만들 수 없으면 해당 Positive는 버림
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
    
    # 전체 데이터를 랜덤하게 섞음
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
            
        # Negative 생성 (각 Split 별로 독립적으로 수행하여 Leakage 방지)
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