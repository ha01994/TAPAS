"""
train_tabpfn_conf_pae.py와 동일한 CV/TabPFN 설정이나,
AF3 품질(confidence) 메트릭만 특징으로 사용하고 PAE는 쓰지 않는다.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier

warnings.filterwarnings('ignore')

DEVICE = 'cuda:0'

AF3_COLS = [
    'iptm_tcrpmhc', 'global_plddt',
    'cdr1_A', 'cdr2_A', 'cdr3_A',
    'cdr1_B', 'cdr2_B', 'cdr3_B',
    'iptm_mean', 'pdockq',
    'avgipae_pmhc', 'avgipae_tcr',
    'pdockq2_pmhc', 'pdockq2_tcr',
]
FEATURE_COLS = AF3_COLS


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


def run_cv_evaluation(folder_name, df_metrics):
    fold_aucs, fold_aucs_01 = [], []
    merge_cols = ['pdb_id'] + FEATURE_COLS

    print(f"\n>>> Evaluating: {folder_name}")
    model = TabPFNClassifier(device=DEVICE, random_state=42)

    for i in range(5):
        train_file = os.path.join(folder_name, f'fold{i}_train.csv')
        test_file = os.path.join(folder_name, f'fold{i}_test.csv')
        if not (os.path.exists(train_file) and os.path.exists(test_file)):
            print(
                f"ERROR: {folder_name} fold{i} train/test CSV가 없습니다.",
                file=sys.stderr,
            )
            sys.exit(1)

        df_train_raw = pd.read_csv(train_file)
        df_test_raw = pd.read_csv(test_file)

        if len(df_train_raw) == 0 or len(df_test_raw) == 0:
            print(
                f"ERROR: {folder_name} fold{i}에 train 또는 test 행이 없습니다.",
                file=sys.stderr,
            )
            sys.exit(1)

        df_train_raw['id'] = df_train_raw['id'].astype(str)
        df_test_raw['id'] = df_test_raw['id'].astype(str)

        df_train_f = df_train_raw.copy()
        df_test_f = df_test_raw.copy()

        if df_train_f['label'].nunique() < 2:
            print(
                f"ERROR: {folder_name} fold{i} train에 label 0과 1이 모두 필요합니다.",
                file=sys.stderr,
            )
            sys.exit(1)
        if df_test_f['label'].nunique() < 2:
            print(
                f"ERROR: {folder_name} fold{i} test에 label 0과 1이 모두 필요합니다.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"  fold{i} train: n={len(df_train_f)}")
        print(f"  fold{i} test:  n={len(df_test_f)}")

        train_df = pd.merge(
            df_train_f,
            df_metrics[merge_cols],
            left_on='id',
            right_on='pdb_id',
            how='inner',
        )
        test_df = pd.merge(
            df_test_f,
            df_metrics[merge_cols],
            left_on='id',
            right_on='pdb_id',
            how='inner',
        )

        if len(train_df) == 0 or len(test_df) == 0:
            print(
                f"ERROR: {folder_name} fold{i}에서 메트릭 병합 후 train 또는 test가 비었습니다.",
                file=sys.stderr,
            )
            sys.exit(1)
        if len(train_df) != len(df_train_f) or len(test_df) != len(df_test_f):
            print(
                f"ERROR: {folder_name} fold{i}: 메트릭 merge 후 행 수 불일치 "
                f"(train {len(df_train_f)}→{len(train_df)}, test {len(df_test_f)}→{len(test_df)}). "
                'fold id가 df_metrics에 없거나 pdb_id 중복일 수 있습니다.',
                file=sys.stderr,
            )
            sys.exit(1)

        X_train = train_df[FEATURE_COLS].reset_index(drop=True)
        X_test = test_df[FEATURE_COLS].reset_index(drop=True)
        y_train = train_df['label'].reset_index(drop=True)
        y_test = test_df['label'].reset_index(drop=True)
        pmhc_test = test_df['pmhc'].reset_index(drop=True)

        print(f"  fold{i} X_train={X_train.shape}, X_test={X_test.shape}")

        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        auc01 = macro_auc_max_fpr(y_test, y_proba, pmhc_test)
        fold_aucs.append(auc)
        fold_aucs_01.append(auc01)
        print(f"  fold{i} → AUC={auc:.4f}, macro_AUC_0.1={auc01:.4f}")

    if len(fold_aucs) == 0:
        print(
            f"ERROR: {folder_name}에서 유효한 fold 평가 결과가 없습니다.",
            file=sys.stderr,
        )
        sys.exit(1)

    mean_auc = np.mean(fold_aucs)
    mean_auc01 = np.nanmean(fold_aucs_01)
    print(f"  Average → AUC={mean_auc:.4f}, macro_AUC_0.1={mean_auc01:.4f}")
    return mean_auc, fold_aucs, mean_auc01, fold_aucs_01


if __name__ == '__main__':
    print('Loading quality metrics...')
    metric_files = [
        'results_vdjdbiedb_iptm_filtered_pos_best.csv',
        'results_vdjdbiedb_iptm_filtered_neg_best.csv',
    ]
    metric_dfs = [pd.read_csv(f) for f in metric_files if os.path.exists(f)]
    if not metric_dfs:
        print('ERROR: 메트릭 CSV 파일이 없습니다.', file=sys.stderr)
        sys.exit(1)
    df_metrics = pd.concat(metric_dfs, ignore_index=True)
    if len(df_metrics) == 0:
        print('ERROR: 메트릭 데이터프레임이 비어 있습니다.', file=sys.stderr)
        sys.exit(1)
    print(df_metrics.shape)

    need = set(['pdb_id'] + FEATURE_COLS)
    miss = need - set(df_metrics.columns)
    if miss:
        print(f'ERROR: 메트릭 CSV에 필요한 컬럼이 없습니다: {sorted(miss)}', file=sys.stderr)
        sys.exit(1)

    print(f"  Rows (confidence metrics only): {len(df_metrics)}")

    os.makedirs('results_auc', exist_ok=True)

    a = time.time()
    rs_avg, rs_folds, rs_avg01, rs_folds01 = run_cv_evaluation(
        'dataset_iptm_filtered_rs', df_metrics
    )
    ss_avg, ss_folds, ss_avg01, ss_folds01 = run_cv_evaluation(
        'dataset_iptm_filtered_ss', df_metrics
    )
    b = time.time()
    print('time taken in minutes: %.2f' % ((b - a) / 60))

    def save_results(fold_aucs, mean_auc, fold_aucs_01, mean_auc01, path):
        n = len(fold_aucs)
        cols = [''] + [f'fold{i}' for i in range(n)] + ['average']
        pd.DataFrame(
            [['auc'] + list(fold_aucs) + [mean_auc],
             ['auc_0.1'] + list(fold_aucs_01) + [mean_auc01]],
            columns=cols,
        ).to_csv(path, index=False, float_format='%.8f')

    save_results(rs_folds, rs_avg, rs_folds01, rs_avg01,
                 'results_auc/rs_conf_all.csv')
    save_results(ss_folds, ss_avg, ss_folds01, ss_avg01,
                 'results_auc/ss_conf_all.csv')

    print('\n' + '=' * 50)
    print('TabPFN (AF3 confidence only)')
    print(f'RS: AUC={rs_avg:.3f}, macro_AUC_0.1={rs_avg01:.3f}')
    print(f'SS: AUC={ss_avg:.3f}, macro_AUC_0.1={ss_avg01:.3f}')
    print('=' * 50)
