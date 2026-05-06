import pandas as pd
import numpy as np
import os
import sys
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
PAE_COLS = [
    'pae_mean', 'pae_max', 'pae_std',
    'pae_median', 'pae_p10', 'pae_p90',
    'pae_frac_lt_5', 'pae_frac_gt_15', 'pae_asymmetry'
]
ESM_COLS     = [f'esm_pca_{i}' for i in range(N_ESM)]
FEATURE_COLS = AF3_COLS + PAE_COLS

# Per-peptide iptm baseline (same signal as test_zeroshot_iptm.py)
IPTM_SCORE_COL = 'iptm_tcrpmhc'

PER_PEPTIDE_COLS = [
    'peptide', 'fold', 'n_samples', 'n_pos', 'n_neg',
    'tabpfn_auc', 'iptm_auc', 'delta', 'iptm_mean', 'iptm_std',
]


# ── 1. Load metrics ───────────────────────────────────────────
print("Loading quality metrics...")
metric_files = [
    'results_vdjdbiedb_iptm_filtered_pos_best.csv',
    'results_vdjdbiedb_iptm_filtered_neg_best.csv',
]
metric_dfs = [pd.read_csv(f) for f in metric_files if os.path.exists(f)]
if not metric_dfs:
    print("ERROR: metrics CSV file not found.", file=sys.stderr)
    sys.exit(1)
df_metrics = pd.concat(metric_dfs, ignore_index=True)
if len(df_metrics) == 0:
    print("ERROR: metrics dataframe is empty.", file=sys.stderr)
    sys.exit(1)

print(df_metrics.shape)

print("Loading PAE features...")
pae_files = [
    'pae_feat_vdjdbiedb_after_iptm_pos.csv',
    'pae_feat_vdjdbiedb_after_iptm_neg.csv',
]
pae_dfs = [pd.read_csv(f) for f in pae_files if os.path.exists(f)]
if not pae_dfs:
    print("ERROR: PAE CSV file not found.", file=sys.stderr)
    sys.exit(1)
df_pae = pd.concat(pae_dfs, ignore_index=True)
if len(df_pae) == 0:
    print("ERROR: PAE dataframe is empty.", file=sys.stderr)
    sys.exit(1)

print(df_pae.shape)


df_metrics = pd.merge(df_metrics, df_pae,
                      left_on='pdb_id', right_on='sample_id', how='inner')

if len(df_metrics) == 0:
    print("ERROR: No samples after merging metrics/PAE. Please check the input CSVs.", file=sys.stderr)
    sys.exit(1)
print(f"  Metrics rows (all samples with features): {len(df_metrics)}")

# ── 2. Load ESM embeddings ────────────────────────────────────
print("Loading raw ESM embeddings...")
raw_map = np.load('esm_embeddings_map_vdjdbiedb_after_iptm.npy', allow_pickle=True).item()
print(f"  {len(raw_map)} entries loaded (ESM embeddings)")


# ── 3. PCA function ───────────────────────────────────────────
def get_pca_embeddings_for_fold(train_ids, test_ids):
    zero = np.zeros(1280, dtype=np.float32)
    reduced_train, reduced_test = [], []
    for col in TARGET_COLS:
        n_comp = PCA_DIMS[col]
        X_tr = np.stack([
            raw_map[p][col] if (p in raw_map and col in raw_map[p]) else zero
            for p in train_ids
        ])
        X_te = np.stack([
            raw_map[p][col] if (p in raw_map and col in raw_map[p]) else zero
            for p in test_ids
        ])
        pca = PCA(n_components=n_comp, random_state=42)
        reduced_train.append(pca.fit_transform(X_tr))
        reduced_test.append(pca.transform(X_te))
    df_tr = pd.DataFrame(np.concatenate(reduced_train, axis=1), columns=ESM_COLS)
    df_te = pd.DataFrame(np.concatenate(reduced_test,  axis=1), columns=ESM_COLS)
    return df_tr, df_te


def pmhc_to_peptide(pmhc):
    return str(pmhc).split('_', 1)[0]


def per_peptide_auroc_rows(fold_idx, y_true, y_tabpfn, y_iptm, peptides):
    y_true = np.asarray(y_true)
    y_tabpfn = np.asarray(y_tabpfn)
    y_iptm = np.asarray(y_iptm)
    peptides = np.asarray(peptides)
    rows = []
    for pep in np.unique(peptides):
        m = peptides == pep
        yt = y_true[m]
        if np.unique(yt).size < 2:
            continue
        try:
            auc_tab = float(roc_auc_score(yt, y_tabpfn[m]))
            auc_iptm = float(roc_auc_score(yt, y_iptm[m]))
        except ValueError:
            continue
        n_pos = int(np.sum(yt == 1))
        n_neg = int(np.sum(yt == 0))
        iptm_vals = y_iptm[m]
        rows.append({
            'peptide': pep,
            'fold': fold_idx,
            'n_samples': int(m.sum()),
            'n_pos': n_pos,
            'n_neg': n_neg,
            'tabpfn_auc': auc_tab,
            'iptm_auc': auc_iptm,
            'delta': auc_tab - auc_iptm,
            'iptm_mean': float(np.mean(iptm_vals)),
            'iptm_std': float(np.std(iptm_vals, ddof=0)),
        })
    return rows


# ── 4. macro AUC@0.1 ─────────────────────────────────────────
def macro_auc_max_fpr(y_true, y_score, pmhc, max_fpr=0.1):
    y_true  = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pmhc    = np.asarray(pmhc)
    per_pmhc = []
    for p in np.unique(pmhc):
        mask = pmhc == p
        y_p  = y_true[mask]
        if np.unique(y_p).size < 2:
            continue
        try:
            per_pmhc.append(roc_auc_score(y_p, y_score[mask], max_fpr=max_fpr))
        except ValueError:
            continue
    return float(np.mean(per_pmhc)) if per_pmhc else np.nan


# ── 5. CV evaluation ──────────────────────────────────────────
def run_cv_evaluation(folder_name, df_metrics):
    fold_aucs, fold_aucs_01 = [], []
    per_peptide_records = []
    merge_cols = ['pdb_id'] + FEATURE_COLS

    print(f"\n>>> Evaluating: {folder_name}")
    model = TabPFNClassifier(device=DEVICE, random_state=42)

    for i in range(5):
        train_file = os.path.join(folder_name, f'fold{i}_train.csv')
        test_file  = os.path.join(folder_name, f'fold{i}_test.csv')
        if not (os.path.exists(train_file) and os.path.exists(test_file)):
            print(
                f"ERROR: {folder_name} fold{i} train/test CSV not found.",
                file=sys.stderr,
            )
            sys.exit(1)

        df_train_raw = pd.read_csv(train_file)
        df_test_raw  = pd.read_csv(test_file)

        if len(df_train_raw) == 0 or len(df_test_raw) == 0:
            print(
                f"ERROR: {folder_name} fold{i} has no train or test rows.",
                file=sys.stderr,
            )
            sys.exit(1)

        df_train_raw['id'] = df_train_raw['id'].astype(str)
        df_test_raw['id'] = df_test_raw['id'].astype(str)

        df_train_f = df_train_raw.copy()
        df_test_f = df_test_raw.copy()

        if df_train_f['label'].nunique() < 2:
            print(
                f"ERROR: {folder_name} fold{i} train requires both label 0 and 1.",
                file=sys.stderr,
            )
            sys.exit(1)
        if df_test_f['label'].nunique() < 2:
            print(
                f"ERROR: {folder_name} fold{i} test requires both label 0 and 1.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"  fold{i} train: n={len(df_train_f)}")
        print(f"  fold{i} test:  n={len(df_test_f)}")

        # ── metrics merge ─────────────────────────────────────
        train_df = pd.merge(df_train_f, df_metrics[merge_cols],
                            left_on='id', right_on='pdb_id', how='inner')
        test_df = pd.merge(df_test_f, df_metrics[merge_cols],
                           left_on='id', right_on='pdb_id', how='inner')

        if len(train_df) == 0 or len(test_df) == 0:
            print(
                f"ERROR: {folder_name} fold{i} became empty (train or test) after merging metrics.",
                file=sys.stderr,
            )
            sys.exit(1)
        if len(train_df) != len(df_train_f) or len(test_df) != len(df_test_f):
            print(
                f"ERROR: {folder_name} fold{i}: row count mismatch after merging metrics "
                f"(train {len(df_train_f)}→{len(train_df)}, test {len(df_test_f)}→{len(test_df)}). "
                'fold ids may be missing from df_metrics, or pdb_id may be duplicated.',
                file=sys.stderr,
            )
            sys.exit(1)

        # ── PCA ───────────────────────────────────────────────
        train_ids = train_df['pdb_id'].astype(str).tolist()
        test_ids  = test_df['pdb_id'].astype(str).tolist()
        df_train_esm, df_test_esm = get_pca_embeddings_for_fold(train_ids, test_ids)

        X_train = pd.concat([train_df[FEATURE_COLS].reset_index(drop=True),
                              df_train_esm], axis=1)
        X_test  = pd.concat([test_df[FEATURE_COLS].reset_index(drop=True),
                              df_test_esm],  axis=1)
        y_train = train_df['label'].reset_index(drop=True)
        y_test  = test_df['label'].reset_index(drop=True)
        pmhc_test = test_df['pmhc'].reset_index(drop=True)

        print(f"  fold{i} X_train={X_train.shape}, X_test={X_test.shape}")

        # ── Train & Predict ───────────────────────────────────
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]

        auc   = roc_auc_score(y_test, y_proba)
        auc01 = macro_auc_max_fpr(y_test, y_proba, pmhc_test)
        fold_aucs.append(auc)
        fold_aucs_01.append(auc01)
        print(f"  fold{i} → AUC={auc:.4f}, macro_AUC_0.1={auc01:.4f}")

        peptides_test = test_df['pmhc'].map(pmhc_to_peptide)
        iptm_scores = test_df[IPTM_SCORE_COL].to_numpy()
        per_peptide_records.extend(
            per_peptide_auroc_rows(i, y_test.to_numpy(), y_proba, iptm_scores, peptides_test)
        )

    if len(fold_aucs) == 0:
        print(
            f"ERROR: {folder_name} has no valid fold evaluation results.",
            file=sys.stderr,
        )
        sys.exit(1)

    mean_auc   = np.mean(fold_aucs)
    mean_auc01 = np.nanmean(fold_aucs_01)
    print(f"  Average → AUC={mean_auc:.4f}, macro_AUC_0.1={mean_auc01:.4f}")
    return mean_auc, fold_aucs, mean_auc01, fold_aucs_01, per_peptide_records


# ── 6. Run ───────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs('results_auc', exist_ok=True)

    import time
    a = time.time()
    ss_avg, ss_folds, ss_avg01, ss_folds01, ss_per_peptide = run_cv_evaluation(
        'dataset_iptm_filtered_ss', df_metrics)
    b = time.time()
    print('time taken in minutes: %.2f'%((b-a)/60))

    def save_results(fold_aucs, mean_auc, fold_aucs_01, mean_auc01, path):
        n = len(fold_aucs)
        cols = [''] + [f'fold{i}' for i in range(n)] + ['average']
        pd.DataFrame(
            [['auc']    + list(fold_aucs)    + [mean_auc],
             ['auc_0.1'] + list(fold_aucs_01) + [mean_auc01]],
            columns=cols
        ).to_csv(path, index=False, float_format='%.8f')

    save_results(ss_folds, ss_avg, ss_folds01, ss_avg01,
                 'results_auc/ss_esm_conf_pae_all.csv')

    def save_per_peptide_csv(records, path):
        df_pp = pd.DataFrame.from_records(records)
        if len(df_pp) == 0:
            print(f"WARNING: no per-peptide records; {path} will not be used.", file=sys.stderr)
            return
        df_pp = df_pp.sort_values(
            ['delta', 'peptide'], ascending=[False, True], kind='stable'
        ).reset_index(drop=True)
        out = df_pp[PER_PEPTIDE_COLS].copy()
        out['fold'] = out['fold'].astype(int)
        out.to_csv(path, index=False, float_format='%.4f')

    save_per_peptide_csv(ss_per_peptide, 'results_auc/per_peptide_auroc_iptm_filtered_ss.csv')

    print("\n" + "="*50)
    print("TabPFN (AF3+PAE+ESM) — dataset_iptm_filtered_ss only")
    print(f"SS: AUC={ss_avg:.3f}, macro_AUC_0.1={ss_avg01:.3f}")
    print("Per-peptide AUROC CSV:")
    print("  results_auc/per_peptide_auroc_iptm_filtered_ss.csv")
    print("="*50)
