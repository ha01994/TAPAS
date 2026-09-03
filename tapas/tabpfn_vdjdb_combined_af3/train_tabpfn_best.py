"""Final TabPFN VDJdb RS/SS cross-validation model.

Selected feature set:
  extra_interface_quality_geometry_cdr3_contacts_pose_esm

Features:
  - base AF3: avgipae_pmhc, avgipae_tcr, pdockq2_pmhc, pdockq2_tcr from
    the best-AF3-ranking-score model
  - geometry: CDR3 contact + pose features (11 cols) from the same model
  - ESM: 288 PCA features (pep/A1/A2/B1/B2=32, A3/B3=64), leakage-free fold-wise
"""

from __future__ import annotations

import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier


warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = BASE_DIR
DEVELOPED_DIR = os.path.dirname(BASE_DIR)
DATASET_ROOT = os.path.join(BASE_DIR, "data")
REPO_ROOT = os.path.dirname(DEVELOPED_DIR)
DEVICE = "cuda:0"
SEED = 42
RESULTS_DIR = os.path.join(BASE_DIR, "results_auc")

METRICS_PATH = os.path.join(
    REPO_ROOT, "af3_confidence", "vdjdb", "model_quality_metrics_best_af3_ranking_score.csv"
)
GEOMETRY_FEATURE_PATH = os.path.join(
    REPO_ROOT, "af3_geometry", "vdjdb", "geometry_features_best_af3_ranking_score.csv"
)
TARGET_COLS = ["peptide", "A1", "A2", "A3", "B1", "B2", "B3"]
PCA_DIMS = {
    "peptide": 32,
    "A1": 32,
    "A2": 32,
    "A3": 64,
    "B1": 32,
    "B2": 32,
    "B3": 64,
}
N_ESM = sum(PCA_DIMS.values())
ESM_COLS = [f"esm_pca_{i}" for i in range(N_ESM)]

BASE_FEATURE_COLS = [
    "avgipae_pmhc",
    "avgipae_tcr",
    "pdockq2_pmhc",
    "pdockq2_tcr",
]
GEOMETRY_META_COLS = ["pair_id", "dataset", "label", "condition"]

FINAL_SUBSET = "extra_interface_quality_geometry_cdr3_contacts_pose_esm"
FINAL_BASE_FAMILY = "interface_quality"
FINAL_GEOMETRY_FAMILY = "cdr3_contacts_pose"
FINAL_GEOMETRY_COLS = [
    "cdr3_all_pep_centroid_dist",
    "cdr3_all_pep_confident_residue_contacts_5a",
    "cdr3_all_pep_group1_contact_fraction_5a",
    "cdr3_all_pep_group2_contact_fraction_5a",
    "cdr3_all_pep_residue_contacts_5a",
    "cdr3a_pep_confident_residue_contacts_5a",
    "cdr3a_pep_residue_contacts_5a",
    "cdr3b_pep_confident_residue_contacts_5a",
    "cdr3b_pep_residue_contacts_5a",
    "tcr_over_peptide_angle_proxy",
    "tcr_pep_centroid_dist",
]


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def keep_unique(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def load_metrics() -> pd.DataFrame:
    print("Loading quality metrics...")
    df_metrics = pd.read_csv(METRICS_PATH)
    print(f"  - Total combined metrics: {df_metrics.shape}")
    return df_metrics


def load_geometry() -> tuple[pd.DataFrame, list[str]]:
    print("Loading AF3 geometry features...")
    df_geometry = pd.read_csv(GEOMETRY_FEATURE_PATH)
    missing_meta = [col for col in GEOMETRY_META_COLS if col not in df_geometry.columns]
    if missing_meta:
        raise ValueError("Geometry feature file is missing required columns: " + ", ".join(missing_meta))

    dup_count = df_geometry["pair_id"].astype(str).duplicated().sum()
    if dup_count:
        raise ValueError(f"Geometry feature file has duplicated pair_id rows: {dup_count}")

    geometry_cols = [col for col in df_geometry.columns if col not in GEOMETRY_META_COLS]
    missing = [col for col in FINAL_GEOMETRY_COLS if col not in geometry_cols]
    if missing:
        raise ValueError(f"Missing final geometry features: {missing}")

    df_geometry = df_geometry[["pair_id"] + geometry_cols].copy()
    df_geometry["pair_id"] = df_geometry["pair_id"].astype(str)
    df_geometry[geometry_cols] = df_geometry[geometry_cols].apply(pd.to_numeric, errors="coerce")
    if df_geometry[geometry_cols].isna().any().any():
        nan_counts = df_geometry[geometry_cols].isna().sum()
        nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)
        print("  - Filling NaN geometry values with 0.0:\n" + nan_counts.to_string())
        df_geometry[geometry_cols] = df_geometry[geometry_cols].fillna(0.0)

    print(f"  - Geometry rows: {len(df_geometry)}")
    print(f"  - Selected geometry columns: {len(FINAL_GEOMETRY_COLS)}")
    return df_geometry, FINAL_GEOMETRY_COLS.copy()


def load_raw_esm() -> dict:
    print("Loading raw ESM embeddings...")
    path = os.path.join(SOURCE_DIR, "esm_embeddings_map_vdjdb.npy")
    raw_map = np.load(path, allow_pickle=True).item()
    print(f"  - {len(raw_map)} entries loaded")
    return raw_map


def get_pca_embeddings_for_fold(train_ids, test_ids, raw_map):
    zero = np.zeros(1280, dtype=np.float32)
    reduced_train, reduced_test = [], []
    for col in TARGET_COLS:
        n_comp = PCA_DIMS[col]
        x_train = np.stack(
            [
                raw_map[pid][col] if pid in raw_map and col in raw_map[pid] else zero
                for pid in train_ids
            ]
        )
        x_test = np.stack(
            [
                raw_map[pid][col] if pid in raw_map and col in raw_map[pid] else zero
                for pid in test_ids
            ]
        )
        pca = PCA(n_components=n_comp, random_state=SEED)
        reduced_train.append(pca.fit_transform(x_train))
        reduced_test.append(pca.transform(x_test))
    return (
        pd.DataFrame(np.concatenate(reduced_train, axis=1), columns=ESM_COLS),
        pd.DataFrame(np.concatenate(reduced_test, axis=1), columns=ESM_COLS),
    )


def macro_auc(y_true, y_score, pmhc, max_fpr=None):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pmhc = np.asarray(pmhc)
    peptide = np.array([str(x).split("_")[0] for x in pmhc], dtype=object)
    per_peptide = []
    for pep in np.unique(peptide):
        mask = peptide == pep
        y_pep = y_true[mask]
        if np.unique(y_pep).size < 2:
            continue
        try:
            per_peptide.append(roc_auc_score(y_pep, y_score[mask], max_fpr=max_fpr))
        except ValueError:
            continue
    return float(np.mean(per_peptide)) if per_peptide else np.nan


def run_cv_evaluation(folder_name, df_metrics, df_geometry, geometry_cols, raw_map):
    fold_macro_aucs = []
    fold_aucs_01 = []
    prediction_frames = []
    feature_cols = keep_unique(BASE_FEATURE_COLS + geometry_cols)

    print(f"\n>>> Evaluating Dataset Split: {folder_name}")
    print(f"Feature set: {FINAL_SUBSET}")
    print(f"  - base={len(BASE_FEATURE_COLS)} geometry={len(geometry_cols)} esm={N_ESM}")

    for fold_idx in range(5):
        train_file = os.path.join(DATASET_ROOT, folder_name, f"fold{fold_idx}_train.csv")
        test_file = os.path.join(DATASET_ROOT, folder_name, f"fold{fold_idx}_test.csv")
        if not (os.path.exists(train_file) and os.path.exists(test_file)):
            continue

        df_train_raw = pd.read_csv(train_file)
        df_test_raw = pd.read_csv(test_file)
        n_train_raw, n_test_raw = len(df_train_raw), len(df_test_raw)

        merge_cols = ["pdb_id"] + BASE_FEATURE_COLS
        train_df = pd.merge(
            df_train_raw,
            df_metrics[merge_cols],
            left_on="id",
            right_on="pdb_id",
            how="inner",
        )
        test_df = pd.merge(
            df_test_raw,
            df_metrics[merge_cols],
            left_on="id",
            right_on="pdb_id",
            how="inner",
        )
        if len(train_df) != n_train_raw:
            raise ValueError(f"{folder_name} fold{fold_idx} train metrics merge mismatch")
        if len(test_df) != n_test_raw:
            raise ValueError(f"{folder_name} fold{fold_idx} test metrics merge mismatch")

        train_df = pd.merge(train_df, df_geometry, left_on="id", right_on="pair_id", how="inner")
        test_df = pd.merge(test_df, df_geometry, left_on="id", right_on="pair_id", how="inner")
        if len(train_df) != n_train_raw:
            raise ValueError(f"{folder_name} fold{fold_idx} train geometry merge mismatch")
        if len(test_df) != n_test_raw:
            raise ValueError(f"{folder_name} fold{fold_idx} test geometry merge mismatch")
        train_ids = train_df["pdb_id"].astype(str).tolist()
        test_ids = test_df["pdb_id"].astype(str).tolist()
        train_esm, test_esm = get_pca_embeddings_for_fold(train_ids, test_ids, raw_map)

        x_train = pd.concat(
            [train_df[feature_cols].reset_index(drop=True), train_esm],
            axis=1,
        )
        x_test = pd.concat(
            [test_df[feature_cols].reset_index(drop=True), test_esm],
            axis=1,
        )
        y_train = train_df["label"].reset_index(drop=True)
        y_test = test_df["label"].reset_index(drop=True)
        pmhc_test = test_df["pmhc"].reset_index(drop=True)

        print(f"[{folder_name} fold {fold_idx}] Train: {x_train.shape}, Test: {x_test.shape}")
        set_global_seed(SEED)
        model = TabPFNClassifier(device=DEVICE, random_state=SEED)
        model.fit(x_train, y_train)
        y_proba = model.predict_proba(x_test)[:, 1]

        auc = macro_auc(y_test, y_proba, pmhc_test)
        auc_01 = macro_auc(y_test, y_proba, pmhc_test, max_fpr=0.1)
        fold_macro_aucs.append(auc)
        fold_aucs_01.append(auc_01)
        print(f"fold{fold_idx},macro_auc={auc:.4f},macro_auc_0.1={auc_01:.4f}")

        pred = test_df[["id", "pmhc", "label"]].copy()
        pred["split"] = folder_name
        pred["fold"] = fold_idx
        pred["subset"] = FINAL_SUBSET
        pred["y_proba"] = y_proba
        prediction_frames.append(pred)

    mean_auc = float(np.nanmean(fold_macro_aucs)) if fold_macro_aucs else np.nan
    mean_auc_01 = float(np.nanmean(fold_aucs_01)) if fold_aucs_01 else np.nan
    print(f"average,macro_auc={mean_auc:.4f},macro_auc_0.1={mean_auc_01:.4f}")

    if prediction_frames:
        pred_out = pd.concat(prediction_frames, ignore_index=True)
        pred_path = os.path.join(RESULTS_DIR, f"{folder_name}_tabpfn_best__predictions.csv")
        pred_out.to_csv(pred_path, index=False)
        print(f"  - Fold predictions saved to {pred_path}")

    return mean_auc, fold_macro_aucs, mean_auc_01, fold_aucs_01


def results_metric_table(fold_aucs, mean_auc, fold_aucs_01, mean_auc_01):
    n = len(fold_aucs)
    cols = [""] + [f"fold{i}" for i in range(n)] + ["average"]
    row_auc = ["macro_auc"] + list(fold_aucs) + [mean_auc]
    row_auc_01 = ["macro_auc_0.1"] + list(fold_aucs_01) + [mean_auc_01]
    return pd.DataFrame([row_auc, row_auc_01], columns=cols)


def main() -> None:
    set_global_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 80)
    print("Confidence selection: best AF3 ranking_score")
    print(f"Confidence file     : {METRICS_PATH}")
    print("Geometry selection  : best AF3 ranking_score")
    print(f"Geometry file       : {GEOMETRY_FEATURE_PATH}")
    print("=" * 80)

    df_metrics = load_metrics()
    df_geometry, geometry_cols = load_geometry()
    raw_map = load_raw_esm()

    rs_avg, rs_fold_aucs, rs_avg_01, rs_fold_aucs_01 = run_cv_evaluation(
        "dataset_rs",
        df_metrics,
        df_geometry,
        geometry_cols,
        raw_map,
    )
    ss_avg, ss_fold_aucs, ss_avg_01, ss_fold_aucs_01 = run_cv_evaluation(
        "dataset_ss",
        df_metrics,
        df_geometry,
        geometry_cols,
        raw_map,
    )

    rs_out = results_metric_table(rs_fold_aucs, rs_avg, rs_fold_aucs_01, rs_avg_01)
    ss_out = results_metric_table(ss_fold_aucs, ss_avg, ss_fold_aucs_01, ss_avg_01)
    rs_path = os.path.join(RESULTS_DIR, "rs_tabpfn_best_.csv")
    ss_path = os.path.join(RESULTS_DIR, "ss_tabpfn_best_.csv")
    rs_out.to_csv(rs_path, index=False, float_format="%.8f")
    ss_out.to_csv(ss_path, index=False, float_format="%.8f")

    summary = pd.DataFrame(
        [
            {
                "subset": FINAL_SUBSET,
                "split": "dataset_rs",
                "mean_macro_auc": rs_avg,
                "mean_macro_auc_0.1": rs_avg_01,
                "n_base_features": len(BASE_FEATURE_COLS),
                "n_geometry_features": len(geometry_cols),
                "n_esm_features": N_ESM,
                "n_total_features": len(BASE_FEATURE_COLS) + len(geometry_cols) + N_ESM,
            },
            {
                "subset": FINAL_SUBSET,
                "split": "dataset_ss",
                "mean_macro_auc": ss_avg,
                "mean_macro_auc_0.1": ss_avg_01,
                "n_base_features": len(BASE_FEATURE_COLS),
                "n_geometry_features": len(geometry_cols),
                "n_esm_features": N_ESM,
                "n_total_features": len(BASE_FEATURE_COLS) + len(geometry_cols) + N_ESM,
            },
        ]
    )
    summary_path = os.path.join(RESULTS_DIR, "tabpfn_best__summary.csv")
    summary.to_csv(summary_path, index=False)

    print("###################################################################")
    print("Final best Results for TabPFN")
    print(f"Subset: {FINAL_SUBSET}")
    print(f"Random Split Avg Macro AUC: {rs_avg:.3f} | macro AUC@FPR<=0.1: {rs_avg_01:.3f}")
    print(f"Strict Split Avg Macro AUC : {ss_avg:.3f} | macro AUC@FPR<=0.1: {ss_avg_01:.3f}")
    print(f"Saved RS metrics -> {rs_path}")
    print(f"Saved SS metrics -> {ss_path}")
    print(f"Saved summary -> {summary_path}")
    print("###################################################################")


if __name__ == "__main__":
    main()
