"""Train the final TAPAS model on VDJDB RS and test it on ePytope-TCR viral.

The external test set is defined by the active AF3 manifest.  The model uses
the final VDJDB feature combination (4 confidence + 11 geometry + 288 ESM-2
PCA features).  PCA is fitted on VDJDB only.  Predictions are deliberately
left raw so the result is directly comparable with the viral benchmark tools.
"""

from __future__ import annotations

import os
import random
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier


warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
DEVELOPED_DIR = SCRIPT_DIR.parent
REPO_ROOT = DEVELOPED_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results_auc"

VDJDB_DATASET_DIR = (
    DEVELOPED_DIR / "tabpfn_vdjdb_combined_af3" / "data" / "dataset_rs"
)
VDJDB_CONFIDENCE_CSV = (
    REPO_ROOT / "af3_confidence" / "vdjdb"
    / "model_quality_metrics_best_af3_ranking_score.csv"
)
VDJDB_GEOMETRY_CSV = (
    REPO_ROOT / "af3_geometry" / "vdjdb"
    / "geometry_features_best_af3_ranking_score.csv"
)
VDJDB_ESM_PATH = (
    DEVELOPED_DIR / "tabpfn_vdjdb_combined_af3" / "esm_embeddings_map_vdjdb.npy"
)

VIRAL_MANIFEST_CSV = SCRIPT_DIR / "data" / "manifest.csv"
VIRAL_CONFIDENCE_CSV = (
    REPO_ROOT / "af3_confidence" / "epytope_tcr_viral"
    / "model_quality_metrics_best_af3_ranking_score.csv"
)
VIRAL_GEOMETRY_CSV = (
    REPO_ROOT / "af3_geometry" / "epytope_tcr_viral"
    / "geometry_features_best_af3_ranking_score.csv"
)
VIRAL_ESM_PATH = SCRIPT_DIR / "esm_embeddings_map.npy"

PREDICTIONS_CSV = RESULTS_DIR / "epytope_tcr_viral_tabpfn_best__predictions.csv"
SUMMARY_CSV = RESULTS_DIR / "epytope_tcr_viral_tabpfn_best__summary.csv"

SEED = 42
DEVICE = "cuda:0"
EXPECTED_VDJDB_ROWS = 4298
EXPECTED_VIRAL_ROWS = 3560
EXPECTED_VIRAL_TCRS = 445
EXPECTED_VIRAL_PEPTIDES = 8

os.environ.setdefault("PYTHONHASHSEED", str(SEED))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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
GEOMETRY_COLS = [
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
FINAL_SUBSET = "extra_interface_quality_geometry_cdr3_contacts_pose_esm"


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required input is not ready: {path}")


def peptide_from_pmhc(pmhc: str) -> str:
    value = str(pmhc)
    if "_HLA-" in value:
        return value.split("_HLA-", 1)[0]
    return value.split("_", 1)[0]


def load_vdjdb_rows() -> pd.DataFrame:
    parts = []
    for split in ("train", "val", "test"):
        path = VDJDB_DATASET_DIR / f"fold0_{split}.csv"
        require_file(path)
        parts.append(pd.read_csv(path))
    rows = pd.concat(parts, ignore_index=True).drop_duplicates("id").reset_index(drop=True)
    if len(rows) != EXPECTED_VDJDB_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_VDJDB_ROWS} unique VDJDB rows, found {len(rows)}"
        )
    rows["id"] = rows["id"].astype(str)
    return rows


def load_viral_rows() -> pd.DataFrame:
    require_file(VIRAL_MANIFEST_CSV)
    rows = pd.read_csv(VIRAL_MANIFEST_CSV)
    required = {
        "job_name",
        "tcr_index",
        "label",
        "cognate_epitope",
        "target_epitope",
        "mhc",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Viral manifest is missing columns: {missing}")

    rows = rows.copy()
    rows["id"] = rows["job_name"].astype(str) + "_structure"
    rows["tcr_id"] = "viral_tcr_" + rows["tcr_index"].astype(str)
    rows["peptide"] = rows["target_epitope"].astype(str)
    rows["pmhc"] = rows["peptide"] + "_" + rows["mhc"].astype(str)
    rows["label"] = pd.to_numeric(rows["label"], errors="raise").astype(int)

    observed = (len(rows), rows["tcr_id"].nunique(), rows["peptide"].nunique())
    expected = (EXPECTED_VIRAL_ROWS, EXPECTED_VIRAL_TCRS, EXPECTED_VIRAL_PEPTIDES)
    if observed != expected:
        raise ValueError(
            "Unexpected active viral dataset dimensions: "
            f"rows/TCRs/peptides={observed}, expected={expected}"
        )
    if rows["id"].duplicated().any():
        raise ValueError("Viral manifest produces duplicated structure IDs")
    if set(rows["label"].unique()) != {0, 1}:
        raise ValueError("Viral labels must contain only 0 and 1")
    pair_counts = rows.groupby("tcr_id")["peptide"].nunique()
    positive_counts = rows.groupby("tcr_id")["label"].sum()
    if not (pair_counts == EXPECTED_VIRAL_PEPTIDES).all():
        raise ValueError("Each viral TCR must have all eight target peptides")
    if not (positive_counts == 1).all():
        raise ValueError("Each viral TCR must have exactly one positive pair")
    return rows


def merge_features(
    rows: pd.DataFrame,
    confidence_path: Path,
    geometry_path: Path,
    dataset_name: str,
) -> pd.DataFrame:
    require_file(confidence_path)
    require_file(geometry_path)

    confidence = pd.read_csv(confidence_path)
    required_conf = {"pdb_id", *BASE_COLS}
    missing_conf = sorted(required_conf - set(confidence.columns))
    if missing_conf:
        raise ValueError(f"{dataset_name} confidence is missing columns: {missing_conf}")
    confidence["pdb_id"] = confidence["pdb_id"].astype(str)
    if confidence["pdb_id"].duplicated().any():
        raise ValueError(f"{dataset_name} confidence contains duplicated pdb_id values")

    geometry = pd.read_csv(geometry_path)
    required_geom = {"pair_id", *GEOMETRY_COLS}
    missing_geom = sorted(required_geom - set(geometry.columns))
    if missing_geom:
        raise ValueError(f"{dataset_name} geometry is missing columns: {missing_geom}")
    geometry["pair_id"] = geometry["pair_id"].astype(str)
    if geometry["pair_id"].duplicated().any():
        raise ValueError(f"{dataset_name} geometry contains duplicated pair_id values")

    expected_ids = set(rows["id"])
    missing_conf_ids = sorted(expected_ids - set(confidence["pdb_id"]))
    missing_geom_ids = sorted(expected_ids - set(geometry["pair_id"]))
    if missing_conf_ids or missing_geom_ids:
        messages = []
        if missing_conf_ids:
            messages.append(
                f"confidence missing {len(missing_conf_ids)} IDs "
                f"(examples: {missing_conf_ids[:3]})"
            )
        if missing_geom_ids:
            messages.append(
                f"geometry missing {len(missing_geom_ids)} IDs "
                f"(examples: {missing_geom_ids[:3]})"
            )
        raise ValueError(f"{dataset_name} feature extraction is incomplete: " + "; ".join(messages))

    merged = rows.merge(
        confidence[["pdb_id", *BASE_COLS]],
        left_on="id",
        right_on="pdb_id",
        how="left",
        validate="one_to_one",
    ).merge(
        geometry[["pair_id", *GEOMETRY_COLS]],
        left_on="id",
        right_on="pair_id",
        how="left",
        validate="one_to_one",
    )
    feature_cols = BASE_COLS + GEOMETRY_COLS
    merged[feature_cols] = merged[feature_cols].apply(pd.to_numeric, errors="coerce")
    if merged[BASE_COLS].isna().any().any():
        raise ValueError(f"{dataset_name} confidence contains non-numeric or missing values")
    if merged[GEOMETRY_COLS].isna().any().any():
        nan_counts = merged[GEOMETRY_COLS].isna().sum()
        print(
            f"{dataset_name}: filling geometry NaNs with 0.0: "
            + ", ".join(f"{col}={count}" for col, count in nan_counts.items() if count)
        )
        merged[GEOMETRY_COLS] = merged[GEOMETRY_COLS].fillna(0.0)
    return merged


def load_esm_map(path: Path, expected_ids: list[str], dataset_name: str) -> dict:
    require_file(path)
    embedding_map = np.load(path, allow_pickle=True).item()
    missing_ids = sorted(set(expected_ids) - set(embedding_map))
    if missing_ids:
        raise ValueError(
            f"{dataset_name} ESM map is missing {len(missing_ids)} IDs "
            f"(examples: {missing_ids[:3]})"
        )
    for pair_id in expected_ids:
        missing_targets = [col for col in TARGET_COLS if col not in embedding_map[pair_id]]
        if missing_targets:
            raise ValueError(f"{dataset_name} ESM {pair_id} missing targets: {missing_targets}")
    return embedding_map


def pca_embeddings(
    train_ids: list[str],
    test_ids: list[str],
    train_map: dict,
    test_map: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_reduced = []
    test_reduced = []
    for target in TARGET_COLS:
        x_train = np.stack(
            [np.asarray(train_map[pair_id][target], dtype=np.float32) for pair_id in train_ids]
        )
        x_test = np.stack(
            [np.asarray(test_map[pair_id][target], dtype=np.float32) for pair_id in test_ids]
        )
        if x_train.shape[1] != x_test.shape[1]:
            raise ValueError(
                f"ESM dimension mismatch for {target}: {x_train.shape} vs {x_test.shape}"
            )
        pca = PCA(n_components=PCA_DIMS[target], random_state=SEED)
        train_reduced.append(pca.fit_transform(x_train))
        test_reduced.append(pca.transform(x_test))
    return (
        pd.DataFrame(np.concatenate(train_reduced, axis=1), columns=ESM_COLS),
        pd.DataFrame(np.concatenate(test_reduced, axis=1), columns=ESM_COLS),
    )


def macro_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    peptides: np.ndarray,
    max_fpr: Optional[float] = None,
) -> float:
    values = []
    for peptide in pd.unique(peptides):
        mask = peptides == peptide
        if np.unique(labels[mask]).size == 2:
            values.append(roc_auc_score(labels[mask], scores[mask], max_fpr=max_fpr))
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    set_global_seed()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading VDJDB RS training set...")
    train_rows = load_vdjdb_rows()
    train = merge_features(
        train_rows, VDJDB_CONFIDENCE_CSV, VDJDB_GEOMETRY_CSV, "VDJDB"
    )

    print("Loading active ePytope-TCR viral test set...")
    test_rows = load_viral_rows()
    test = merge_features(
        test_rows, VIRAL_CONFIDENCE_CSV, VIRAL_GEOMETRY_CSV, "ePytope viral"
    )

    print("Loading ESM-2 maps and fitting PCA on VDJDB only...")
    train_ids = train["id"].tolist()
    test_ids = test["id"].tolist()
    train_map = load_esm_map(VDJDB_ESM_PATH, train_ids, "VDJDB")
    test_map = load_esm_map(VIRAL_ESM_PATH, test_ids, "ePytope viral")
    train_esm, test_esm = pca_embeddings(train_ids, test_ids, train_map, test_map)

    feature_cols = BASE_COLS + GEOMETRY_COLS
    x_train = pd.concat([train[feature_cols].reset_index(drop=True), train_esm], axis=1)
    x_test = pd.concat([test[feature_cols].reset_index(drop=True), test_esm], axis=1)
    y_train = train["label"].to_numpy(dtype=int)
    y_test = test["label"].to_numpy(dtype=int)
    peptides = test["peptide"].to_numpy(dtype=str)

    print(f"Training TAPAS: train={x_train.shape}, test={x_test.shape}, device={DEVICE}")
    model = TabPFNClassifier(device=DEVICE, random_state=SEED)
    model.fit(x_train, y_train)
    raw_scores = model.predict_proba(x_test)[:, 1]

    full_macro_auc = macro_auc(y_test, raw_scores, peptides)
    macro_auc_01 = macro_auc(y_test, raw_scores, peptides, max_fpr=0.1)

    prediction_cols = [
        "id",
        "job_name",
        "tcr_index",
        "tcr_id",
        "cognate_epitope",
        "target_epitope",
        "peptide",
        "mhc",
        "pmhc",
        "label",
    ]
    predictions = test[prediction_cols].copy()
    predictions["tapas_raw"] = raw_scores
    predictions.to_csv(PREDICTIONS_CSV, index=False)

    summary = pd.DataFrame(
        [
            {
                "method": "TAPAS (raw)",
                "subset": FINAL_SUBSET,
                "training_dataset": "VDJDB RS fold0 train+val+test",
                "n_train": len(train),
                "n_test": len(test),
                "positive": int(y_test.sum()),
                "negative": int(len(y_test) - y_test.sum()),
                "n_peptides": int(test["peptide"].nunique()),
                "n_features": x_train.shape[1],
                "macro_auc": full_macro_auc,
                "macro_auc_0.1": macro_auc_01,
            }
        ]
    )
    summary.to_csv(SUMMARY_CSV, index=False, float_format="%.4f")

    print("\nePytope-TCR viral TAPAS (raw)")
    print(f"Macro-AUC     : {full_macro_auc:.4f}")
    print(f"Macro-AUC@0.1 : {macro_auc_01:.4f}")
    print(f"\nPredictions : {PREDICTIONS_CSV}")
    print(f"Summary     : {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
