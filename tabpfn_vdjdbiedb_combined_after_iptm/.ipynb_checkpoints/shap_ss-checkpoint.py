import pandas as pd
import numpy as np
import os
import warnings
import time
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier
warnings.filterwarnings('ignore')
import torch

start_time = time.time()


a = time.time()

DEVICE      = 'cuda:0'
FOLDER_NAME = 'dataset_iptm_filtered_ss'
FOLD_IDX    = 0

AF3_COLS = [
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

FEATURES = AF3_COLS + PAE_COLS   # 14 + 10 = 24


# ─────────────────────────────────────────────────────────────
# [1] Load
# ─────────────────────────────────────────────────────────────
print("Step 1: Loading quality metrics...")
metric_files = ['results_model_iptm_filtered_pos_best.csv',
                'results_model_iptm_filtered_neg_best.csv']

df_metrics = pd.concat(
    [pd.read_csv(f) for f in metric_files if os.path.exists(f)],
    ignore_index=True
)

print("Loading PAE features...")
pae_files = [
    'pae_feat_vdjdbiedb_after_iptm_pos.csv',
    'pae_feat_vdjdbiedb_after_iptm_neg.csv'
]
df_pae = pd.concat(
    [pd.read_csv(f) for f in pae_files if os.path.exists(f)],
    ignore_index=True
)
df_metrics = pd.merge(df_metrics, df_pae, left_on='pdb_id', right_on='sample_id', how='inner')
print(f"  - Metrics after merging PAE: {df_metrics.shape}")


# ─────────────────────────────────────────────────────────────
# [2] Prepare fold data
# ─────────────────────────────────────────────────────────────
print(f"Step 2: Preparing data for Fold {FOLD_IDX}...")
df_train_raw = pd.read_csv(os.path.join(FOLDER_NAME, f'fold{FOLD_IDX}_train.csv'))
df_test_raw  = pd.read_csv(os.path.join(FOLDER_NAME, f'fold{FOLD_IDX}_test.csv'))

merge_cols = ['pdb_id'] + FEATURES
train_df = pd.merge(df_train_raw, df_metrics[merge_cols], left_on='id', right_on='pdb_id', how='inner')
test_df  = pd.merge(df_test_raw,  df_metrics[merge_cols], left_on='id', right_on='pdb_id', how='inner')

X_train = train_df[FEATURES].reset_index(drop=True)
X_test  = test_df[FEATURES].reset_index(drop=True)
y_train = train_df['label'].reset_index(drop=True)
y_test  = test_df['label'].reset_index(drop=True)
print(f" -> Train: {X_train.shape}, Test: {X_test.shape}")


# ─────────────────────────────────────────────────────────────
# [3] Train TabPFN
# ─────────────────────────────────────────────────────────────
print("Step 3: Training TabPFN model...")
model = TabPFNClassifier(device=DEVICE)
model.fit(X_train, y_train)

# SHAP용 모델 = 전체 train으로 fit (subsampling 없음)
model_shap = model

y_proba = model.predict_proba(X_test)[:, 1]
print(f"Model AUC:  {roc_auc_score(y_test, y_proba):.4f}")


# ─────────────────────────────────────────────────────────────
# [4] SHAP — Permutation explainer (feature-level, no grouping)
# ─────────────────────────────────────────────────────────────
print("\nStep 4: Running SHAP...")

X_bg = X_train.sample(50, random_state=42)

def model_predict(X_arr):
    results = []
    batch_size = 200
    n_batches  = (len(X_arr) + batch_size - 1) // batch_size
    for i in range(0, len(X_arr), batch_size):
        batch = pd.DataFrame(X_arr[i:i+batch_size], columns=FEATURES)
        with torch.no_grad():
            proba = model_shap.predict_proba(batch)[:, 1]
        results.append(proba)
        torch.cuda.empty_cache()

        batch_idx = i // batch_size + 1
        pct       = batch_idx / n_batches * 100
        filled    = int(30 * batch_idx / n_batches)
        bar       = '█' * filled + '░' * (30 - filled)
        elapsed   = time.time() - start_time
        eta       = elapsed / max(batch_idx - 1, 1) * (n_batches - batch_idx)
        print(f"\r  [{bar}] {pct:5.1f}% | batch {batch_idx}/{n_batches} | "
              f"elapsed {elapsed:5.1f}s | ETA {eta:5.1f}s",
              end='', flush=True)
    print()
    return np.concatenate(results)

masker    = shap.maskers.Independent(X_bg.values, max_samples=100)
explainer = shap.explainers.Permutation(
    model_predict,
    masker=masker,
    feature_names=FEATURES,
)

# Stratified sampling: 100 pos + 100 neg
pos_idx = y_test[y_test == 1].index.tolist()
neg_idx = y_test[y_test == 0].index.tolist()
rng = np.random.default_rng(42)
sampled_pos = rng.choice(pos_idx, size=100, replace=False)
sampled_neg = rng.choice(neg_idx, size=100, replace=False)
explain_idx = np.concatenate([sampled_pos, sampled_neg])
print(f" -> Explain set: 100 pos + 100 neg = 200 total")

X_explain = X_test.loc[explain_idx].values   # (200, 24)

max_evals = 2 * len(FEATURES) + 1   # 49
print(f" -> n_features: {len(FEATURES)}, max_evals: {max_evals}")
print(f" -> Total model calls (est.): {max_evals} × {len(X_explain)} = {max_evals * len(X_explain)}")

shap_values = explainer(X_explain, max_evals=max_evals)

print(f" -> SHAP shape: {shap_values.values.shape}")
print(f" -> Time: {(time.time() - start_time)/60:.2f} min")

import pickle
with open(f'shap_values_ss_fold{FOLD_IDX}.pkl', 'wb') as f:
    pickle.dump(shap_values, f)


# ─────────────────────────────────────────────────────────────
# [5] Plots
# ─────────────────────────────────────────────────────────────
print("\nStep 5: Generating plots...")

plt.figure(figsize=(12, 8))
shap.plots.beeswarm(shap_values, max_display=24, show=False)
plt.title(f"SHAP Beeswarm — Conf + PAE Features (Fold {FOLD_IDX})", fontsize=14)
plt.tight_layout()
plt.savefig(f'shap_beeswarm_ss_fold{FOLD_IDX}.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 8))
shap.plots.bar(shap_values, max_display=24, show=False)
plt.title(f"Feature Importance (Mean |SHAP|) — Conf + PAE (Fold {FOLD_IDX})", fontsize=14)
plt.tight_layout()
plt.savefig(f'shap_bar_ss_fold{FOLD_IDX}.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 7))
shap.plots.waterfall(shap_values[0], show=False)
plt.title(f"SHAP Waterfall — Sample 0, Fold {FOLD_IDX}", fontsize=14)
plt.tight_layout()
plt.savefig(f'shap_waterfall_ss_fold{FOLD_IDX}.png', dpi=300, bbox_inches='tight')
plt.close()


# ─────────────────────────────────────────────────────────────
# [6] Numeric summary
# ─────────────────────────────────────────────────────────────
summary_df = pd.DataFrame({
    'feature':       FEATURES,
    'group':         ['Conf'] * len(AF3_COLS) + ['PAE'] * len(PAE_COLS),
    'mean_abs_shap': np.abs(shap_values.values).mean(axis=0),
    'mean_shap':     shap_values.values.mean(axis=0),
}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

print("\n── Feature-level SHAP Summary ────────────────────────────")
print(summary_df.to_string(index=False))

summary_df.to_csv(f'shap_summary_ss_fold{FOLD_IDX}.csv', index=False)

print("\nAll done!")
b = time.time()
print((b-a)/60., 'minutes')
