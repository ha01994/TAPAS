"""Final TabPFN model for ImmRep25 using the selected AF3 geometry feature set.

Selected feature set:
  extra_interface_quality_geometry_cdr3_contacts_pose_esm

Features:
  - base AF3: avgipae_pmhc, avgipae_tcr, pdockq2_pmhc, pdockq2_tcr from
    the best-AF3-ranking-score model
  - geometry: CDR3 contact + pose features (11 cols) from the same model
  - ESM: 288 PCA features (pep/A1/A2/B1/B2=32, A3/B3=64) fit on VDJdb train rows

Outputs:
  - results_auc/immrep25_tabpfn_best__predictions.csv
  - results_auc/immrep25_tabpfn_best__summary.csv
"""

from __future__ import annotations

import os
import random
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier


warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = SCRIPT_DIR
DATA_DIR = os.path.join(SOURCE_DIR, "data")
DATASET_ROOT = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(DATASET_ROOT)
VDJDB_DIR = os.path.join(DATASET_ROOT, "vdjdb")
RESULTS_AUC_DIR = os.path.join(SCRIPT_DIR, "results_auc")
SEED = 42
DEVICE = "cuda:0"

os.environ.setdefault("PYTHONHASHSEED", str(SEED))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

IMMREP25_PAIRS_CSV = os.path.join(DATA_DIR, "immrep25_pairs.csv")
METRICS_BEST_CSV = os.path.join(
    REPO_ROOT, "af3_confidence", "immrep25", "model_quality_metrics_best_af3_ranking_score.csv"
)
ESM_IMMREP25_PATH = os.path.join(SOURCE_DIR, "esm_embeddings_map.npy")
GEOMETRY_IMMREP25_CSV = os.path.join(
    REPO_ROOT, "af3_geometry", "immrep25", "geometry_features_best_af3_ranking_score.csv"
)

VDJDB_METRICS_CSV = os.path.join(
    REPO_ROOT, "af3_confidence", "vdjdb", "model_quality_metrics_best_af3_ranking_score.csv"
)
ESM_VDJDB_PATH = os.path.join(
    VDJDB_DIR, "esm_embeddings_map_vdjdb.npy"
)
GEOMETRY_VDJDB_CSV = os.path.join(
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

BASE_COLS = [
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


def _dataset_dir(name: str) -> str:
    return os.path.join(VDJDB_DIR, "data", name)


def peptide_from_pmhc(pmhc: str) -> str:
    value = str(pmhc)
    if "_HLA-" in value:
        return value.split("_HLA-", 1)[0]
    return value.split("_", 1)[0]


def keep_unique(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def load_geometry_features(path: str, name: str) -> pd.DataFrame:
    print(f"Loading {name} geometry features...")
    if not os.path.exists(path):
        print(f"Missing {name} geometry file: {path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(path)
    missing_meta = [col for col in GEOMETRY_META_COLS if col not in df.columns]
    if missing_meta:
        raise ValueError(f"{name} geometry missing columns: {missing_meta}")
    if df["pair_id"].astype(str).duplicated().any():
        dup_count = int(df["pair_id"].astype(str).duplicated().sum())
        raise ValueError(f"{name} geometry has duplicated pair_id rows: {dup_count}")

    geometry_cols = [col for col in df.columns if col not in GEOMETRY_META_COLS]
    df = df[["pair_id"] + geometry_cols].copy()
    df["pair_id"] = df["pair_id"].astype(str)
    df[geometry_cols] = df[geometry_cols].apply(pd.to_numeric, errors="coerce")
    if df[geometry_cols].isna().any().any():
        nan_counts = df[geometry_cols].isna().sum()
        nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)
        print("  - Filling NaN geometry values with 0.0:\n" + nan_counts.to_string())
        df[geometry_cols] = df[geometry_cols].fillna(0.0)

    print(f"  - {name} geometry rows: {len(df)}")
    print(f"  - {name} geometry feature columns: {len(geometry_cols)}")
    return df


def selected_geometry_cols(geometry_cols: list[str]) -> list[str]:
    missing = [col for col in FINAL_GEOMETRY_COLS if col not in geometry_cols]
    if missing:
        raise ValueError(f"Missing final geometry features: {missing}")
    return FINAL_GEOMETRY_COLS.copy()


def get_pca_embeddings(train_ids, test_ids, train_raw_map, test_raw_map):
    zero = np.zeros(1280, dtype=np.float32)
    reduced_train, reduced_test = [], []
    for col in TARGET_COLS:
        n_comp = PCA_DIMS[col]
        x_train = np.stack(
            [
                train_raw_map[sid][col]
                if sid in train_raw_map and col in train_raw_map[sid]
                else zero
                for sid in train_ids
            ]
        )
        x_test = np.stack(
            [
                test_raw_map[sid][col]
                if sid in test_raw_map and col in test_raw_map[sid]
                else zero
                for sid in test_ids
            ]
        )
        pca = PCA(n_components=n_comp, random_state=SEED)
        reduced_train.append(pca.fit_transform(x_train))
        reduced_test.append(pca.transform(x_test))
    train_esm = pd.DataFrame(np.concatenate(reduced_train, axis=1), columns=ESM_COLS)
    test_esm = pd.DataFrame(np.concatenate(reduced_test, axis=1), columns=ESM_COLS)
    return train_esm, test_esm


def macro_auc(y_true, y_score, peptide, max_fpr=None):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    peptide = np.asarray(peptide)
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


def load_vdjdb_metrics() -> pd.DataFrame:
    print("Loading VDJdb quality metrics...")
    df_metrics = pd.read_csv(VDJDB_METRICS_CSV)
    print(f"  - VDJdb metrics: {df_metrics.shape}")
    return df_metrics


def load_train_rows() -> pd.DataFrame:
    dataset_ss = _dataset_dir("dataset_ss")
    print(f"\nLoading VDJdb training rows ({dataset_ss} fold0 train/val/test)...")
    parts = []
    for part in ["train", "val", "test"]:
        path = os.path.join(dataset_ss, f"fold0_{part}.csv")
        if os.path.exists(path):
            parts.append(pd.read_csv(path))
    if not parts:
        raise FileNotFoundError(f"No fold0 train/val/test CSVs found under {dataset_ss}")
    df = pd.concat(parts, ignore_index=True).drop_duplicates(subset="id").reset_index(drop=True)
    print(f"  - Unique train rows: {len(df)}")
    print(f"  - pos: {(df['label'] == 1).sum()}, neg: {(df['label'] == 0).sum()}")
    return df


def main() -> None:
    set_global_seed(SEED)
    os.makedirs(RESULTS_AUC_DIR, exist_ok=True)

    print("=" * 80)
    print("Confidence selection: best AF3 ranking_score")
    print(f"  VDJDB file    : {VDJDB_METRICS_CSV}")
    print(f"  ImmRep25 file : {METRICS_BEST_CSV}")
    print("Geometry selection  : best AF3 ranking_score")
    print(f"  VDJDB file    : {GEOMETRY_VDJDB_CSV}")
    print(f"  ImmRep25 file : {GEOMETRY_IMMREP25_CSV}")
    print("=" * 80)

    df_train_raw = load_train_rows()
    df_metrics_vdjdb = load_vdjdb_metrics()
    merge_cols = ["pdb_id"] + BASE_COLS

    train_df = pd.merge(
        df_train_raw,
        df_metrics_vdjdb[merge_cols],
        left_on="id",
        right_on="pdb_id",
        how="inner",
    )
    if len(train_df) != len(df_train_raw):
        raise ValueError(f"VDJdb metrics merge mismatch: {len(df_train_raw)} vs {len(train_df)}")

    geom_vdjdb = load_geometry_features(GEOMETRY_VDJDB_CSV, "VDJdb")
    geometry_cols = [col for col in geom_vdjdb.columns if col != "pair_id"]
    geom_cols = selected_geometry_cols(geometry_cols)
    train_df = pd.merge(train_df, geom_vdjdb, left_on="id", right_on="pair_id", how="inner")
    if len(train_df) != len(df_train_raw):
        raise ValueError(f"VDJdb geometry merge mismatch: {len(df_train_raw)} vs {len(train_df)}")

    print(f"\nSelected final feature set: {FINAL_SUBSET}")
    print(f"  - base features: {len(BASE_COLS)}")
    print(f"  - geometry features: {len(geom_cols)}")
    print(f"  - ESM features: {N_ESM}")

    df_immrep = pd.read_csv(IMMREP25_PAIRS_CSV)
    df_immrep["peptide"] = df_immrep["pmhc"].map(peptide_from_pmhc)
    df_immrep["pdb_id"] = df_immrep["id"].astype(str)
    df_metrics_immrep = pd.read_csv(METRICS_BEST_CSV)
    test_df = pd.merge(df_immrep, df_metrics_immrep[merge_cols], on="pdb_id", how="inner")
    if len(test_df) != len(df_immrep):
        raise ValueError(f"ImmRep25 metrics merge mismatch: {len(df_immrep)} vs {len(test_df)}")

    geom_immrep = load_geometry_features(GEOMETRY_IMMREP25_CSV, "ImmRep25")
    immrep_geometry_cols = [col for col in geom_immrep.columns if col != "pair_id"]
    if immrep_geometry_cols != geometry_cols:
        raise ValueError("VDJdb and ImmRep25 geometry columns differ.")
    test_df = pd.merge(test_df, geom_immrep, left_on="id", right_on="pair_id", how="inner")
    if len(test_df) != len(df_immrep):
        raise ValueError(f"ImmRep25 geometry merge mismatch: {len(df_immrep)} vs {len(test_df)}")
    print("\nLoading ESM maps and fitting PCA...")
    raw_vdjdb = np.load(ESM_VDJDB_PATH, allow_pickle=True).item()
    raw_immrep = np.load(ESM_IMMREP25_PATH, allow_pickle=True).item()
    train_ids = train_df["id"].astype(str).tolist()
    test_ids = test_df["id"].astype(str).tolist()
    df_train_esm, df_test_esm = get_pca_embeddings(train_ids, test_ids, raw_vdjdb, raw_immrep)

    model_feature_cols = keep_unique(BASE_COLS + geom_cols)
    x_train = pd.concat(
        [train_df[model_feature_cols].reset_index(drop=True), df_train_esm],
        axis=1,
    )
    x_test = pd.concat(
        [test_df[model_feature_cols].reset_index(drop=True), df_test_esm],
        axis=1,
    )
    y_train = train_df["label"].reset_index(drop=True)
    y_test = test_df["label"].reset_index(drop=True)
    peptide_test = test_df["peptide"].reset_index(drop=True)

    print(f"\nTraining TabPFN final model: train={x_train.shape}, test={x_test.shape}")
    model = TabPFNClassifier(device=DEVICE, random_state=SEED)
    model.fit(x_train, y_train)
    y_proba = model.predict_proba(x_test)[:, 1]
    macro_auc_full = macro_auc(y_test, y_proba, peptide_test)
    macro_auc_01 = macro_auc(y_test, y_proba, peptide_test, max_fpr=0.1)

    pred_df = test_df[["id", "pmhc", "label", "peptide"]].copy().reset_index(drop=True)
    if "tcr_id" in test_df.columns:
        pred_df["tcr_id"] = test_df["tcr_id"].reset_index(drop=True)
    pred_df["best_subset"] = FINAL_SUBSET
    pred_df["best_base_family"] = FINAL_BASE_FAMILY
    pred_df["best_geometry_family"] = FINAL_GEOMETRY_FAMILY
    pred_df["best_use_esm"] = True
    pred_df["y_proba"] = y_proba

    final_pred_path = os.path.join(RESULTS_AUC_DIR, "immrep25_tabpfn_best__predictions.csv")
    pred_df.to_csv(final_pred_path, index=False)

    summary = pd.DataFrame(
        [
            {
                "subset": FINAL_SUBSET,
                "base_family": FINAL_BASE_FAMILY,
                "geometry_family": FINAL_GEOMETRY_FAMILY,
                "use_esm": True,
                "n_base_features": len(BASE_COLS),
                "n_geometry_features": len(geom_cols),
                "n_esm_features": N_ESM,
                "n_total_features": x_train.shape[1],
                "n_train": len(x_train),
                "n_test": len(x_test),
                "macro_auc": macro_auc_full,
                "macro_auc_0.1": macro_auc_01,
                "base_features": ",".join(BASE_COLS),
                "geometry_features": ",".join(geom_cols),
                "prediction_csv": final_pred_path,
            }
        ]
    )
    summary_path = os.path.join(RESULTS_AUC_DIR, "immrep25_tabpfn_best__summary.csv")
    summary.to_csv(summary_path, index=False)

    print("###################################################################")
    print("Final best ImmRep25 TabPFN result")
    print(f"  subset: {FINAL_SUBSET}")
    print(f"  Macro AUC: {macro_auc_full:.6f}")
    print(f"  Macro AUC@FPR<=0.1: {macro_auc_01:.6f}")
    print()
    print(f"  predictions: {final_pred_path}")
    print(f"  summary: {summary_path}")
    print("###################################################################")


if __name__ == "__main__":
    main()
