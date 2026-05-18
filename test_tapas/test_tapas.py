import pandas as pd
import numpy as np
import os
import warnings
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier
warnings.filterwarnings('ignore')


DEVICE       = 'cuda:0'
TARGET_COLS  = ['peptide', 'A1', 'A2', 'A3', 'B1', 'B2', 'B3']
PCA_DIMS     = {
    'peptide': 64,
    'A1': 64, 'A2': 64, 'A3': 200,
    'B1': 64, 'B2': 64, 'B3': 200,
}   
N_ESM        = sum(PCA_DIMS.values())   

AF3_COLS     = [
    'iptm_tcrpmhc', 'global_plddt',
    'cdr1_A', 'cdr2_A', 'cdr3_A',
    'cdr1_B', 'cdr2_B', 'cdr3_B',
    'iptm_mean', 'pdockq',
    'avgipae_pmhc', 'avgipae_tcr',
    'pdockq2_pmhc', 'pdockq2_tcr',
]

# Summarized PAE interface feature columns
PAE_COLS = [
    'pae_mean', 'pae_max', 'pae_std', 
    'pae_median', 'pae_p10', 'pae_p90', 
    'pae_frac_lt_5', 'pae_frac_gt_15', 'pae_asymmetry'
]

ESM_COLS     = [f'esm_pca_{i}' for i in range(N_ESM)]
FEATURES     = AF3_COLS + PAE_COLS + ESM_COLS  # includes PAE block


# --- 1. Load AF3 / quality metric tables ---
print("Loading quality metrics...")
metric_files = [
    'results_model_quality_metrics_pos_best.csv',
    'results_model_quality_metrics_neg1_best.csv',
    'results_model_quality_metrics_neg2_best.csv',
]
df_metrics = pd.concat(
    [pd.read_csv(f) for f in metric_files if os.path.exists(f)],
    ignore_index=True
)
print(f"  - Total combined metrics: {df_metrics.shape}")



# --- 1.5 Load PAE features and merge ---
print("Loading PAE features...")
pae_files = [
    'pae_feat_af3_vdjdb.csv',
    'pae_feat_af3_vdjdb_neg_part1.csv',
    'pae_feat_af3_vdjdb_neg_part2.csv'
]
df_pae = pd.concat(
    [pd.read_csv(f) for f in pae_files if os.path.exists(f)],
    ignore_index=True
)
print(f"  - Total combined PAE features: {df_pae.shape}")

# Merge metrics with PAE rows on pdb_id == sample_id (inner join)
df_metrics = pd.merge(df_metrics, df_pae, left_on='pdb_id', right_on='sample_id', how='inner')
print(f"  - Metrics after merging PAE: {df_metrics.shape}")



# --- 2. Load raw ESM embeddings (1280-d per part, before PCA) ---
print("Loading raw ESM embeddings...")
raw_map = np.load('esm_embeddings_map_vdjdb.npy', allow_pickle=True).item()
# Layout: {pdb_id: {'peptide': (1280,), 'A1': (1280,), ..., 'B3': (1280,)}}
print(f"  - {len(raw_map)} entries loaded")



# --- 3. Per-fold, leakage-free PCA on ESM columns ---
def get_pca_embeddings_for_fold(train_ids, test_ids):
    zero = np.zeros(1280, dtype=np.float32)

    reduced_train, reduced_test = [], []

    for col in TARGET_COLS:
        n_comp = PCA_DIMS[col]

        X_train_raw = np.stack([
            raw_map[pid][col] if (pid in raw_map and col in raw_map[pid])
            else zero
            for pid in train_ids
        ])   # (n_train, 1280)

        X_test_raw = np.stack([
            raw_map[pid][col] if (pid in raw_map and col in raw_map[pid])
            else zero
            for pid in test_ids
        ])   # (n_test, 1280)

        # Fit PCA on training rows only
        pca = PCA(n_components=n_comp, random_state=42)
        reduced_train.append(pca.fit_transform(X_train_raw))  # fit + transform
        reduced_test.append(pca.transform(X_test_raw))        # transform only

    X_train_esm = np.concatenate(reduced_train, axis=1)  # (n_train, 200)
    X_test_esm  = np.concatenate(reduced_test,  axis=1)  # (n_test,  200)

    df_train_esm = pd.DataFrame(X_train_esm, columns=ESM_COLS)
    df_test_esm  = pd.DataFrame(X_test_esm,  columns=ESM_COLS)
    return df_train_esm, df_test_esm


def macro_auc_max_fpr(y_true, y_score, pmhc, max_fpr=0.1):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pmhc = np.asarray(pmhc)
    peptide = np.array([str(x).split("_")[0] for x in pmhc], dtype=object)
    per_pep = []
    for p in np.unique(peptide):
        mask = peptide == p
        y_p = y_true[mask]
        if np.unique(y_p).size < 2:
            continue
        s_p = y_score[mask]
        try:
            per_pep.append(roc_auc_score(y_p, s_p, max_fpr=max_fpr))
        except ValueError:
            continue
    if not per_pep:
        return np.nan
    return float(np.mean(per_pep))


# --- 4. Cross-validated evaluation ---
def run_cv_evaluation(folder_name, df_metrics):
    fold_aucs = []
    fold_aucs_01 = []
    print(f"\n>>> Evaluating Dataset Split: {folder_name}")

    model = TabPFNClassifier(device=DEVICE, random_state=42)

    for i in range(5):
        train_file = os.path.join(folder_name, f'fold{i}_train.csv')
        test_file  = os.path.join(folder_name, f'fold{i}_test.csv')
        if not (os.path.exists(train_file) and os.path.exists(test_file)):
            continue

        df_train_raw = pd.read_csv(train_file)
        df_test_raw  = pd.read_csv(test_file)

        # AF3 + PAE columns for the merge
        merge_cols = ['pdb_id'] + AF3_COLS + PAE_COLS
        train_df = pd.merge(df_train_raw, df_metrics[merge_cols],
                            left_on='id', right_on='pdb_id', how='inner')
        test_df  = pd.merge(df_test_raw,  df_metrics[merge_cols],
                            left_on='id', right_on='pdb_id', how='inner')

        n_tr, n_te = len(df_train_raw), len(df_test_raw)
        if len(train_df) != n_tr:
            raise ValueError(
                f"{folder_name} fold{i} train: inner merge row count mismatch "
                f"(fold CSV {n_tr} vs merged {len(train_df)}). "
                f"Some fold 'id' values are missing from df_metrics['pdb_id']."
            )
        if len(test_df) != n_te:
            raise ValueError(
                f"{folder_name} fold{i} test: inner merge row count mismatch "
                f"(fold CSV {n_te} vs merged {len(test_df)}). "
                f"Some fold 'id' values are missing from df_metrics['pdb_id']."
            )

        # Leakage-free ESM PCA: refit on each fold's training ids only
        train_ids = train_df['pdb_id'].astype(str).tolist()
        test_ids  = test_df['pdb_id'].astype(str).tolist()

        df_train_esm, df_test_esm = get_pca_embeddings_for_fold(train_ids, test_ids)
        # ─────────────────────────────────────────────────────────────────

        # Concatenate AF3 + PAE numeric block with PCA-reduced ESM
        feature_cols = AF3_COLS + PAE_COLS
        X_train = pd.concat(
            [train_df[feature_cols].reset_index(drop=True), df_train_esm],
            axis=1
        )
        X_test  = pd.concat(
            [test_df[feature_cols].reset_index(drop=True),  df_test_esm],
            axis=1
        )
        
        y_train = train_df['label'].reset_index(drop=True)
        y_test  = test_df['label'].reset_index(drop=True)
        pmhc_test = test_df['pmhc'].reset_index(drop=True)
        print(f"[Fold {i}] Train: {X_train.shape}, Test: {X_test.shape}")

        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        fold_aucs.append(auc)
        auc01 = macro_auc_max_fpr(y_test, y_proba, pmhc_test, max_fpr=0.1)
        fold_aucs_01.append(auc01)
        print(f"fold{i},auc={auc:.4f},auc_0.1(macro)={auc01:.4f}")

    mean_auc = np.mean(fold_aucs)
    mean_auc_01 = np.nanmean(fold_aucs_01) if fold_aucs_01 else np.nan
    print(f"average,auc={mean_auc:.4f},auc_0.1(macro)={mean_auc_01:.4f}")
    return mean_auc, fold_aucs, mean_auc_01, fold_aucs_01


# --- 5. Main ---
if __name__ == "__main__":
    os.makedirs('results_auc', exist_ok=True)

    rs_avg, rs_fold_aucs, rs_avg_01, rs_fold_aucs_01 = run_cv_evaluation(
        'dataset_rs', df_metrics
    )
    ss_avg, ss_fold_aucs, ss_avg_01, ss_fold_aucs_01 = run_cv_evaluation(
        'dataset_ss', df_metrics
    )

    def _results_metric_table(fold_aucs, mean_auc, fold_aucs_01, mean_auc_01):
        n = len(fold_aucs)
        cols = [''] + [f'fold{i}' for i in range(n)] + ['average']
        row_auc = ['auc'] + list(fold_aucs) + [mean_auc]
        row_01 = ['auc_0.1'] + list(fold_aucs_01) + [mean_auc_01]
        return pd.DataFrame([row_auc, row_01], columns=cols)

    rs_out = _results_metric_table(
        rs_fold_aucs, rs_avg, rs_fold_aucs_01, rs_avg_01
    )
    ss_out = _results_metric_table(
        ss_fold_aucs, ss_avg, ss_fold_aucs_01, ss_avg_01
    )
    rs_out.to_csv(
        os.path.join('results_auc', 'rs_esm_conf_pae.csv'),
        index=False,
        float_format='%.8f',
    )
    ss_out.to_csv(
        os.path.join('results_auc', 'ss_esm_conf_pae.csv'),
        index=False,
        float_format='%.8f',
    )

    print("\n" + "="*45)
    print(f"Final Results for TAPAS (with PAE Features)")
    print(f"Random Split Avg ROC-AUC: {rs_avg:.3f} | macro AUC@FPR≤0.1: {rs_avg_01:.3f}")
    print(f"Strict Split Avg ROC-AUC : {ss_avg:.3f} | macro AUC@FPR≤0.1: {ss_avg_01:.3f}")
    print("="*45)
