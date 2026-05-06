import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

AF3_COLS = [
    'iptm_tcrpmhc', 'global_plddt',
    'cdr1_A', 'cdr2_A', 'cdr3_A',
    'cdr1_B', 'cdr2_B', 'cdr3_B',
    'iptm_mean', 'pdockq',
    'avgipae_pmhc', 'avgipae_tcr', 'avgipae_average',
    'pdockq2_pmhc', 'pdockq2_tcr',
]
PAE_COLS = [
    'pae_mean', 'pae_max', 'pae_std',
    'pae_median', 'pae_p10', 'pae_p90',
    'pae_frac_lt_5', 'pae_frac_gt_15', 'pae_asymmetry',
]
FEATURE_COLS = AF3_COLS + PAE_COLS


def macro_auc_max_fpr(y_true, y_score, pmhc, max_fpr=0.1):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pmhc = np.asarray(pmhc)
    per_pmhc = []
    for p in np.unique(pmhc):
        mask = pmhc == p
        y_p = y_true[mask]
        if np.unique(y_p).size < 2:
            continue
        try:
            per_pmhc.append(roc_auc_score(y_p, y_score[mask], max_fpr=max_fpr))
        except ValueError:
            continue
    return float(np.mean(per_pmhc)) if per_pmhc else np.nan


def evaluate_feature_zeroshot(folder_name, df_metrics, feature_col):
    if feature_col not in df_metrics.columns:
        print(
            f"ERROR: df_metrics is missing column '{feature_col}' not found.",
            file=sys.stderr,
        )
        sys.exit(1)

    merge_cols = ['pdb_id', feature_col]
    fold_aucs, fold_aucs_01 = [], []

    print(f"\n>>> [{feature_col}] zeroshot — {folder_name}")

    for i in range(5):
        test_file = os.path.join(folder_name, f'fold{i}_test.csv')
        if not os.path.exists(test_file):
            print(
                f"ERROR: {folder_name} fold{i} test CSV not found: {test_file}",
                file=sys.stderr,
            )
            sys.exit(1)

        df_test_raw = pd.read_csv(test_file)
        if len(df_test_raw) == 0:
            print(
                f"ERROR: {folder_name} fold{i} test CSV has no rows.",
                file=sys.stderr,
            )
            sys.exit(1)

        df_test_f = df_test_raw.copy()
        df_test_f['id'] = df_test_f['id'].astype(str)

        if df_test_f['label'].nunique() < 2:
            print(
                f"ERROR: {folder_name} fold{i}: "
                'test must contain both label 0 and 1.',
                file=sys.stderr,
            )
            sys.exit(1)

        test_df = pd.merge(
            df_test_f,
            df_metrics[merge_cols],
            left_on='id',
            right_on='pdb_id',
            how='inner',
        )

        if len(test_df) == 0:
            print(
                f"ERROR: {folder_name} fold{i} ({feature_col}): "
                'test became empty after merging metrics.',
                file=sys.stderr,
            )
            sys.exit(1)

        if len(test_df) != len(df_test_f):
            print(
                f"ERROR: {folder_name} fold{i} ({feature_col}): "
                f"row count mismatch after merging metrics (test {len(df_test_f)} vs merge {len(test_df)}). "
                'some ids may be missing from df_metrics, or pdb_id may be duplicated.',
                file=sys.stderr,
            )
            sys.exit(1)

        y_scores = pd.to_numeric(test_df[feature_col], errors='coerce')
        if y_scores.isnull().any():
            n_bad = int(y_scores.isnull().sum())
            print(
                f"ERROR: {folder_name} fold{i} ({feature_col}): "
                f'NaN found in score column {n_bad}items present.',
                file=sys.stderr,
            )
            sys.exit(1)

        y_true = test_df['label'].reset_index(drop=True)
        y_scores = y_scores.reset_index(drop=True)
        pmhc_test = (
            test_df['pmhc'].reset_index(drop=True)
            if 'pmhc' in test_df.columns
            else None
        )

        if y_scores.nunique() < 2:
            print(
                f"ERROR: {folder_name} fold{i} ({feature_col}): "
                "in test, this feature is constant so AUC cannot be defined.",
                file=sys.stderr,
            )
            sys.exit(1)

        if len(y_true) == 0 or y_true.nunique() < 2:
            print(
                f"ERROR: {folder_name} fold{i} ({feature_col}): "
                "not enough labels.",
                file=sys.stderr,
            )
            sys.exit(1)

        try:
            auc = roc_auc_score(y_true, y_scores)
        except ValueError as e:
            print(
                f"ERROR: {folder_name} fold{i} ({feature_col}) ROC-AUC: {e}",
                file=sys.stderr,
            )
            sys.exit(1)

        auc01 = np.nan
        if pmhc_test is not None:
            auc01 = macro_auc_max_fpr(
                y_true.values, y_scores.values, pmhc_test.values
            )

        fold_aucs.append(auc)
        fold_aucs_01.append(auc01)
        print(f"  fold{i} → AUC={auc:.4f}, macro_AUC_0.1={auc01:.4f}")

    mean_auc = float(np.mean(fold_aucs))
    mean_auc01 = float(np.nanmean(fold_aucs_01))
    print(f"  Average → AUC={mean_auc:.4f}, macro_AUC_0.1={mean_auc01:.4f}")
    return mean_auc, fold_aucs, mean_auc01, fold_aucs_01


def _row_for_split(feature_col, split_name, mean_auc, folds_auc, mean_01, folds_01):
    row = {'feature': feature_col, 'split': split_name}
    for j, v in enumerate(folds_auc):
        row[f'auc_fold{j}'] = v
    row['auc_mean'] = mean_auc
    for j, v in enumerate(folds_01):
        row[f'auc01_fold{j}'] = v
    row['auc01_mean'] = mean_01
    return row


if __name__ == '__main__':
    print('Loading quality metrics...')
    metric_files = [
        'results_vdjdbiedb_iptm_filtered_pos_best.csv',
        'results_vdjdbiedb_iptm_filtered_neg_best.csv',
    ]
    metric_dfs = [pd.read_csv(f) for f in metric_files if os.path.exists(f)]
    if not metric_dfs:
        print('ERROR: metrics CSV file not found.', file=sys.stderr)
        sys.exit(1)
    df_metrics = pd.concat(metric_dfs, ignore_index=True)
    if len(df_metrics) == 0:
        print('ERROR: metrics dataframe is empty.', file=sys.stderr)
        sys.exit(1)
    print(df_metrics.shape)

    print('Loading PAE features...')
    pae_files = [
        'pae_feat_vdjdbiedb_after_iptm_pos.csv',
        'pae_feat_vdjdbiedb_after_iptm_neg.csv',
    ]
    pae_dfs = [pd.read_csv(f) for f in pae_files if os.path.exists(f)]
    if not pae_dfs:
        print('ERROR: PAE CSV file not found.', file=sys.stderr)
        sys.exit(1)
    df_pae = pd.concat(pae_dfs, ignore_index=True)
    if len(df_pae) == 0:
        print('ERROR: PAE dataframe is empty.', file=sys.stderr)
        sys.exit(1)
    print(df_pae.shape)

    df_metrics = pd.merge(
        df_metrics,
        df_pae,
        left_on='pdb_id',
        right_on='sample_id',
        how='inner',
    )
    if len(df_metrics) == 0:
        print(
            'ERROR: No samples after merging metrics/PAE. Please check the input CSVs.',
            file=sys.stderr,
        )
        sys.exit(1)

    if 'avgipae_pmhc' not in df_metrics.columns or 'avgipae_tcr' not in df_metrics.columns:
        print(
            "ERROR: Missing columns required to compute avgipae_average: 'avgipae_pmhc', 'avgipae_tcr' "
            'columns not found.',
            file=sys.stderr,
        )
        sys.exit(1)
    df_metrics = df_metrics.copy()
    df_metrics['avgipae_average'] = (
        pd.to_numeric(df_metrics['avgipae_pmhc'], errors='coerce')
        + pd.to_numeric(df_metrics['avgipae_tcr'], errors='coerce')
    ) / 2.0

    print(f"  Metrics rows (all samples with features): {len(df_metrics)}")

    os.makedirs('results_auc', exist_ok=True)

    rs_folder = 'dataset_iptm_filtered_rs'
    ss_folder = 'dataset_iptm_filtered_ss'

    out_rows = []
    t0 = time.time()

    for k, feat in enumerate(FEATURE_COLS):
        print('\n' + '=' * 60)
        print(f'Feature {k + 1}/{len(FEATURE_COLS)}: {feat}')
        print('=' * 60)

        rs_m, rs_f, rs_m01, rs_f01 = evaluate_feature_zeroshot(
            rs_folder, df_metrics, feat
        )
        ss_m, ss_f, ss_m01, ss_f01 = evaluate_feature_zeroshot(
            ss_folder, df_metrics, feat
        )

        out_rows.append(_row_for_split(feat, 'rs', rs_m, rs_f, rs_m01, rs_f01))
        out_rows.append(_row_for_split(feat, 'ss', ss_m, ss_f, ss_m01, ss_f01))

    elapsed_min = (time.time() - t0) / 60.0
    print('\ntime taken in minutes: %.2f' % elapsed_min)

    out_df = pd.DataFrame(out_rows)
    out_path = 'results_auc/zeroshot_individual_by_feature.csv'
    out_df.to_csv(out_path, index=False, float_format='%.8f')
    print(f"\nSaved: {out_path}")
    print('Columns: feature, split, auc_fold0..4, auc_mean, auc01_fold0..4, auc01_mean')
