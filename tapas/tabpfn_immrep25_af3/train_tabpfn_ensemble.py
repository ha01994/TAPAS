#!/usr/bin/env python3
"""Train ten full-VDJDB TAPAS models and ensemble them on IMMREP25.

One model is trained on train+val+test from each RS fold 0-4 and SS fold 0-4.
Each model fits its ESM-2 PCA on its own 4,298-row training set.  Raw external
prediction probabilities are averaged pairwise, then Bradley normalization and
TCRdist small-cluster smoothing are applied to the ensemble prediction.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier

import train_tabpfn_best as best


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SCRIPT_DIR.parent / "tabpfn_vdjdb_combined_af3" / "data"
RESULTS_DIR = SCRIPT_DIR / "results_auc"
PREDICTIONS_CSV = RESULTS_DIR / "immrep25_tabpfn_ensemble_10model__predictions.csv"
PER_PEPTIDE_CSV = (
    RESULTS_DIR / "immrep25_tabpfn_ensemble_10model__per_peptide_auc_0.1.csv"
)
SUMMARY_CSV = RESULTS_DIR / "immrep25_tabpfn_ensemble_10model__summary.csv"
SMALLCLUST_INPUT_CSV = (
    RESULTS_DIR
    / "immrep25_tabpfn_ensemble_10model__smallclust_input_predictions.csv"
)
SMALLCLUST_METRICS_CSV = (
    RESULTS_DIR / "immrep25_tabpfn_ensemble_10model__smallclust_metrics.csv"
)
SMALLCLUST_PREDICTIONS_CSV = (
    RESULTS_DIR / "immrep25_tabpfn_ensemble_10model__smallclust_predictions.csv"
)
SMALLCLUST_CLUSTERS_CSV = (
    RESULTS_DIR / "immrep25_tabpfn_ensemble_10model__smallclust_clusters.csv"
)
SMALLCLUST_CLUSTER_SUMMARY_CSV = (
    RESULTS_DIR
    / "immrep25_tabpfn_ensemble_10model__smallclust_cluster_summary.csv"
)

MODEL_SPECS = [(split, fold) for split in ("rs", "ss") for fold in range(5)]
EXPECTED_TRAIN_ROWS = 4298
EXPECTED_UNION_ROWS = 23639
EXPECTED_TEST_ROWS = 10000
EXPECTED_TEST_PEPTIDES = 20
FEATURE_COLS = best.BASE_COLS + best.FINAL_GEOMETRY_COLS


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required input: {path}")


def load_full_fold(split: str, fold: int) -> pd.DataFrame:
    frames = []
    dataset_dir = DATASET_ROOT / f"dataset_{split}"
    for part in ("train", "val", "test"):
        path = dataset_dir / f"fold{fold}_{part}.csv"
        require_file(path)
        frames.append(pd.read_csv(path))
    rows = pd.concat(frames, ignore_index=True).drop_duplicates("id").reset_index(drop=True)
    rows["id"] = rows["id"].astype(str)
    if len(rows) != EXPECTED_TRAIN_ROWS:
        raise ValueError(
            f"VDJDB {split.upper()} fold {fold}: expected {EXPECTED_TRAIN_ROWS} "
            f"unique rows, found {len(rows)}"
        )
    if rows["label"].value_counts().to_dict() != {1: 2149, 0: 2149}:
        raise ValueError(f"VDJDB {split.upper()} fold {fold} is not class-balanced")
    return rows


def build_training_feature_table(
    fold_rows: dict[tuple[str, int], pd.DataFrame]
) -> pd.DataFrame:
    all_rows = (
        pd.concat(fold_rows.values(), ignore_index=True)
        .drop_duplicates("id")
        .reset_index(drop=True)
    )
    if len(all_rows) != EXPECTED_UNION_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_UNION_ROWS} unique rows across ten VDJDB sets, "
            f"found {len(all_rows)}"
        )
    metrics = pd.read_csv(best.VDJDB_METRICS_CSV)
    required_metrics = {"pdb_id", *best.BASE_COLS}
    missing_metrics = sorted(required_metrics - set(metrics.columns))
    if missing_metrics:
        raise ValueError(f"VDJDB confidence is missing columns: {missing_metrics}")
    metrics["pdb_id"] = metrics["pdb_id"].astype(str)
    geometry = best.load_geometry_features(best.GEOMETRY_VDJDB_CSV, "VDJDB")
    missing_geometry = sorted(set(best.FINAL_GEOMETRY_COLS) - set(geometry.columns))
    if missing_geometry:
        raise ValueError(f"VDJDB geometry is missing columns: {missing_geometry}")
    merged = all_rows.merge(
        metrics[["pdb_id", *best.BASE_COLS]],
        left_on="id",
        right_on="pdb_id",
        how="inner",
        validate="one_to_one",
    ).merge(
        geometry[["pair_id", *best.FINAL_GEOMETRY_COLS]],
        left_on="id",
        right_on="pair_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(all_rows):
        raise ValueError(
            f"VDJDB feature merge mismatch: expected {len(all_rows)}, found {len(merged)}"
        )
    return merged


def select_fold_features(
    rows: pd.DataFrame, feature_table: pd.DataFrame
) -> pd.DataFrame:
    selected = rows.merge(
        feature_table[["id", *FEATURE_COLS]],
        on="id",
        how="left",
        validate="one_to_one",
    )
    if selected[FEATURE_COLS].isna().any().any():
        raise ValueError("A VDJDB fold contains missing confidence/geometry features")
    return selected


def load_test_features() -> pd.DataFrame:
    require_file(Path(best.IMMREP25_PAIRS_CSV))
    require_file(Path(best.METRICS_BEST_CSV))
    pairs = pd.read_csv(best.IMMREP25_PAIRS_CSV)
    required = {"id", "pmhc", "tcr_id", "label"}
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"IMMREP25 pairs are missing columns: {missing}")
    pairs["id"] = pairs["id"].astype(str)
    pairs["peptide"] = pairs["pmhc"].map(best.peptide_from_pmhc)
    if pairs["id"].duplicated().any():
        raise ValueError("IMMREP25 pairs contain duplicated IDs")

    metrics = pd.read_csv(best.METRICS_BEST_CSV)
    required_metrics = {"pdb_id", *best.BASE_COLS}
    missing_metrics = sorted(required_metrics - set(metrics.columns))
    if missing_metrics:
        raise ValueError(f"IMMREP25 confidence is missing columns: {missing_metrics}")
    metrics["pdb_id"] = metrics["pdb_id"].astype(str)
    geometry = best.load_geometry_features(best.GEOMETRY_IMMREP25_CSV, "IMMREP25")
    missing_geometry = sorted(set(best.FINAL_GEOMETRY_COLS) - set(geometry.columns))
    if missing_geometry:
        raise ValueError(f"IMMREP25 geometry is missing columns: {missing_geometry}")

    test = pairs.merge(
        metrics[["pdb_id", *best.BASE_COLS]],
        left_on="id",
        right_on="pdb_id",
        how="inner",
        validate="one_to_one",
    ).merge(
        geometry[["pair_id", *best.FINAL_GEOMETRY_COLS]],
        left_on="id",
        right_on="pair_id",
        how="inner",
        validate="one_to_one",
    )
    if len(test) != EXPECTED_TEST_ROWS or test["peptide"].nunique() != EXPECTED_TEST_PEPTIDES:
        raise ValueError(
            f"Unexpected IMMREP25 dimensions: rows={len(test)}, "
            f"peptides={test['peptide'].nunique()}"
        )
    if test[FEATURE_COLS].isna().any().any():
        raise ValueError("IMMREP25 contains missing confidence/geometry features")
    return test


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
        missing_targets = [
            target for target in best.TARGET_COLS if target not in embedding_map[pair_id]
        ]
        if missing_targets:
            raise ValueError(
                f"{dataset_name} ESM {pair_id} is missing targets: {missing_targets}"
            )
    return embedding_map


def per_peptide_auc(labels, scores, peptides) -> pd.DataFrame:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    peptides = np.asarray(peptides, dtype=str)
    rows = []
    for peptide in sorted(pd.unique(peptides)):
        mask = peptides == peptide
        peptide_labels = labels[mask]
        if np.unique(peptide_labels).size != 2:
            continue
        rows.append(
            {
                "peptide": peptide,
                "n": int(mask.sum()),
                "positive": int(peptide_labels.sum()),
                "negative": int(mask.sum() - peptide_labels.sum()),
                "auc": roc_auc_score(peptide_labels, scores[mask]),
                "auc_0.1": roc_auc_score(
                    peptide_labels, scores[mask], max_fpr=0.1
                ),
            }
        )
    return pd.DataFrame(rows)


def postprocess_predictions(predictions: pd.DataFrame) -> dict[str, float]:
    required = {"id", "tcr_id", "peptide", "pmhc", "label", "ensemble_probability"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Ensemble predictions are missing columns: {missing}")
    if len(predictions) != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_TEST_ROWS} ensemble predictions, "
            f"found {len(predictions)}"
        )

    labels = predictions["label"].to_numpy(dtype=int)
    peptides = predictions["peptide"].to_numpy(dtype=str)
    normalized = best.bradley_normalize_immrep25(
        predictions["ensemble_probability"].to_numpy(dtype=float),
        tcr_ids=predictions["tcr_id"].to_numpy(),
        peptide_ids=peptides,
        pmhcs=predictions["pmhc"].to_numpy(),
    )
    postprocess_input = predictions.copy()
    postprocess_input["y_proba_raw"] = postprocess_input["ensemble_probability"]
    postprocess_input["y_proba_normalized"] = normalized
    postprocess_input.to_csv(SMALLCLUST_INPUT_CSV, index=False)

    smoothed, metrics, clusters, cluster_summary = best.apply_smallclust(
        postprocess_input,
        threshold=best.SMALLCLUST_THRESHOLD,
    )
    metrics.to_csv(SMALLCLUST_METRICS_CSV, index=False, float_format="%.4f")
    smoothed.to_csv(SMALLCLUST_PREDICTIONS_CSV, index=False)
    clusters.to_csv(SMALLCLUST_CLUSTERS_CSV, index=False)
    cluster_summary.to_csv(SMALLCLUST_CLUSTER_SUMMARY_CSV, index=False)

    smoothed_row = metrics.loc[
        metrics["method"] == "smallclust_sqrt_weighted"
    ].iloc[0]
    return {
        "normalized_macro_auc": best.macro_auc(labels, normalized, peptides),
        "normalized_macro_auc_0.1": best.macro_auc(
            labels, normalized, peptides, max_fpr=0.1
        ),
        "postprocessed_macro_auc": float(smoothed_row["bradley_macro_auc"]),
        "postprocessed_macro_auc_0.1": float(
            smoothed_row["bradley_macro_auc_0.1"]
        ),
    }


def print_postprocessed(metrics: dict[str, float]) -> None:
    print(f"Normalized Macro-AUC       : {metrics['normalized_macro_auc']:.4f}")
    print(f"Normalized Macro-AUC@0.1   : {metrics['normalized_macro_auc_0.1']:.4f}")
    print(f"Postprocessed Macro-AUC    : {metrics['postprocessed_macro_auc']:.4f}")
    print(
        "Postprocessed Macro-AUC@0.1: "
        f"{metrics['postprocessed_macro_auc_0.1']:.4f}"
    )
    print(f"Postprocessed predictions  : {SMALLCLUST_PREDICTIONS_CSV}")
    print(f"Postprocessed metrics      : {SMALLCLUST_METRICS_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Postprocess an existing 10-model ensemble prediction CSV.",
    )
    args = parser.parse_args()

    best.set_global_seed()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.postprocess_only:
        require_file(PREDICTIONS_CSV)
        predictions = pd.read_csv(PREDICTIONS_CSV)
        postprocessed = postprocess_predictions(predictions)
        if SUMMARY_CSV.is_file():
            summary = pd.read_csv(SUMMARY_CSV)
            for column, value in postprocessed.items():
                summary[column] = value
            summary.to_csv(SUMMARY_CSV, index=False, float_format="%.4f")
        print("\nIMMREP25 TAPAS 10-model ensemble postprocessing")
        print_postprocessed(postprocessed)
        return

    fold_rows = {
        spec: load_full_fold(*spec)
        for spec in MODEL_SPECS
    }
    train_feature_table = build_training_feature_table(fold_rows)

    test = load_test_features()

    print("Loading ESM-2 embedding maps...")
    train_map = load_esm_map(
        Path(best.ESM_VDJDB_PATH),
        train_feature_table["id"].astype(str).tolist(),
        "VDJDB ten-fold union",
    )
    test_ids = test["id"].astype(str).tolist()
    test_map = load_esm_map(
        Path(best.ESM_IMMREP25_PATH), test_ids, "IMMREP25"
    )

    model_predictions = []
    model_names = []
    for model_index, (split, fold) in enumerate(MODEL_SPECS, start=1):
        model_name = f"{split}_fold{fold}"
        train = select_fold_features(fold_rows[(split, fold)], train_feature_table)
        train_ids = train["id"].astype(str).tolist()
        print(
            f"[{model_index}/10] {model_name}: fitting PCA and TAPAS "
            f"on {len(train)} rows...",
            flush=True,
        )
        train_esm, test_esm = best.get_pca_embeddings(
            train_ids, test_ids, train_map, test_map
        )
        x_train = pd.concat(
            [train[FEATURE_COLS].reset_index(drop=True), train_esm], axis=1
        )
        x_test = pd.concat(
            [test[FEATURE_COLS].reset_index(drop=True), test_esm], axis=1
        )
        if x_train.shape[1] != 303 or x_test.shape[1] != 303:
            raise ValueError(
                f"Unexpected feature dimensions: train={x_train.shape}, test={x_test.shape}"
            )

        best.set_global_seed()
        model = TabPFNClassifier(device=best.DEVICE, random_state=best.SEED)
        model.fit(x_train, train["label"].to_numpy(dtype=int))
        model_predictions.append(model.predict_proba(x_test)[:, 1])
        model_names.append(model_name)
        del model, x_train, x_test, train_esm, test_esm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    prediction_matrix = np.column_stack(model_predictions)
    ensemble_probability = prediction_matrix.mean(axis=1)
    labels = test["label"].to_numpy(dtype=int)
    peptides = test["peptide"].to_numpy(dtype=str)
    peptide_results = per_peptide_auc(labels, ensemble_probability, peptides)
    macro_auc = float(peptide_results["auc"].mean())
    macro_auc_01 = float(peptide_results["auc_0.1"].mean())

    output_columns = [
        "id",
        "tcr_id",
        "peptide",
        "pmhc",
        "label",
    ]
    predictions = test[output_columns].copy()
    for column_index, model_name in enumerate(model_names):
        predictions[f"probability_{model_name}"] = prediction_matrix[:, column_index]
    predictions["ensemble_probability"] = ensemble_probability
    predictions.to_csv(PREDICTIONS_CSV, index=False)
    peptide_results.to_csv(PER_PEPTIDE_CSV, index=False, float_format="%.8f")
    postprocessed = postprocess_predictions(predictions)

    summary = pd.DataFrame(
        [
            {
                "method": "TAPAS 10-model probability ensemble",
                "score_stage": "raw",
                "aggregation": "pairwise_mean_probability",
                "training_models": "RS folds 0-4 + SS folds 0-4; train+val+test",
                "n_models": len(model_names),
                "n_train_per_model": EXPECTED_TRAIN_ROWS,
                "n_test": len(test),
                "n_peptides": len(peptide_results),
                "n_features": 303,
                "macro_auc": macro_auc,
                "macro_auc_0.1": macro_auc_01,
                **postprocessed,
            }
        ]
    )
    summary.to_csv(SUMMARY_CSV, index=False, float_format="%.4f")

    print("\nIMMREP25 TAPAS 10-model ensemble")
    print(f"Macro-AUC     : {macro_auc:.4f}")
    print(f"Macro-AUC@0.1 : {macro_auc_01:.4f}")
    print_postprocessed(postprocessed)
    print(f"Predictions    : {PREDICTIONS_CSV}")
    print(f"Per-peptide    : {PER_PEPTIDE_CSV}")
    print(f"Summary        : {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
