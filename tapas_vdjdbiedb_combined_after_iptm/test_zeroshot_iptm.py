import pandas as pd
import numpy as np
import os
import sys
from sklearn.metrics import roc_auc_score


# ── 1. score map 로드 ─────────────────────────────────────────
def load_score_map():
    metric_files = [
    'results_vdjdbiedb_iptm_filtered_pos_best.csv',
    'results_vdjdbiedb_iptm_filtered_neg_best.csv',
    ]
    dfs = [pd.read_csv(f) for f in metric_files if os.path.exists(f)]
    if not dfs:
        print("ERROR: 메트릭 CSV 파일이 없습니다.", file=sys.stderr)
        sys.exit(1)
    all_results = pd.concat(dfs, ignore_index=True)
    if len(all_results) == 0:
        print("ERROR: 메트릭 데이터가 비어 있습니다.", file=sys.stderr)
        sys.exit(1)
    return dict(zip(all_results['pdb_id'].astype(str),
                    all_results['iptm_tcrpmhc']))


# ── 2. macro AUC@FPR≤0.1 ─────────────────────────────────────
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


# ── 3. 폴드 평가 ─────────────────────────────────────────────
def evaluate_folds(folder_path, score_map):
    fold_aucs, fold_aucs_01 = [], []
    print(f"\n--- Evaluating: {folder_path} ---")

    for i in range(5):
        test_file = os.path.join(folder_path, f'fold{i}_test.csv')
        if not os.path.exists(test_file):
            print(
                f"ERROR: {folder_path} fold{i} test CSV가 없습니다: {test_file}",
                file=sys.stderr,
            )
            sys.exit(1)

        df_test_raw = pd.read_csv(test_file)
        if len(df_test_raw) == 0:
            print(
                f"ERROR: {folder_path} fold{i} test CSV에 행이 없습니다.",
                file=sys.stderr,
            )
            sys.exit(1)

        df_test = df_test_raw.copy()
        df_test['id'] = df_test['id'].astype(str)

        print(f"  Fold {i}: n={len(df_test)}")

        y_scores = df_test['id'].map(score_map)
        y_true   = df_test['label']
        pmhc     = df_test['pmhc'] if 'pmhc' in df_test.columns else None

        if y_scores.isnull().any():
            missing = df_test.loc[y_scores.isnull(), 'id'].tolist()
            print(
                f"ERROR: {folder_path} fold{i}: score_map에 없는 id가 "
                f"{y_scores.isnull().sum()}개입니다 (예: {missing[:5]}).",
                file=sys.stderr,
            )
            sys.exit(1)

        if len(y_true) == 0 or y_true.nunique() < 2:
            print(
                f"ERROR: {folder_path} fold{i}: AUC 계산에 필요한 라벨이 부족합니다.",
                file=sys.stderr,
            )
            sys.exit(1)

        auc = roc_auc_score(y_true, y_scores)
        fold_aucs.append(auc)

        auc01 = np.nan
        if pmhc is not None:
            auc01 = macro_auc_max_fpr(y_true.values, y_scores.values, pmhc.values)
        fold_aucs_01.append(auc01)

        print(f"  Fold {i}: AUC={auc:.4f}, macro_AUC_0.1={auc01:.4f}")

    if len(fold_aucs) == 0:
        print(
            f"ERROR: {folder_path}에서 유효한 fold 결과가 없습니다.",
            file=sys.stderr,
        )
        sys.exit(1)

    avg_auc   = np.mean(fold_aucs)
    avg_auc01 = np.nanmean(fold_aucs_01)
    print(f"  >> Average: AUC={avg_auc:.4f}, macro_AUC_0.1={avg_auc01:.4f}")
    return avg_auc, fold_aucs, avg_auc01, fold_aucs_01


# ── 4. 메인 ──────────────────────────────────────────────────
if __name__ == "__main__":
    score_map = load_score_map()
    if len(score_map) == 0:
        print("ERROR: score_map이 비어 있습니다.", file=sys.stderr)
        sys.exit(1)
    print(f"Score map entries: {len(score_map)}")

    rs_avg, rs_folds, rs_avg01, rs_folds01 = evaluate_folds(
        'dataset_iptm_filtered_rs', score_map)
    ss_avg, ss_folds, ss_avg01, ss_folds01 = evaluate_folds(
        'dataset_iptm_filtered_ss', score_map)

    os.makedirs('results_auc', exist_ok=True)

    def save_results(fold_aucs, mean_auc, fold_aucs_01, mean_auc01, path):
        n    = len(fold_aucs)
        cols = [''] + [f'fold{i}' for i in range(n)] + ['average']
        pd.DataFrame(
            [['auc']     + list(fold_aucs)    + [mean_auc],
             ['auc_0.1'] + list(fold_aucs_01) + [mean_auc01]],
            columns=cols
        ).to_csv(path, index=False, float_format='%.8f')

    save_results(rs_folds, rs_avg, rs_folds01, rs_avg01,
                 'results_auc/rs_zeroshot_all.csv')
    save_results(ss_folds, ss_avg, ss_folds01, ss_avg01,
                 'results_auc/ss_zeroshot_all.csv')

    print("\n" + "="*40)
    print("Zero-shot ipTM")
    print(f"RS: AUC={rs_avg:.3f}, macro_AUC_0.1={rs_avg01:.3f}")
    print(f"SS: AUC={ss_avg:.3f}, macro_AUC_0.1={ss_avg01:.3f}")
    print("="*40)