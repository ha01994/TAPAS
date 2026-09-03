#!/usr/bin/env python3
"""Train ten full-VDJDB TAPAS models and ensemble them on ePytope viral.

One model is trained on train+val+test from each RS fold 0-4 and SS fold 0-4.
Each model fits its ESM-2 PCA on its own 4,298-row training set.  Raw external
prediction probabilities are averaged pairwise, then peptide macro-AUC and
macro-AUC@0.1 are calculated from the average.  No postprocessing is applied.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier

import train_tabpfn_best as best


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SCRIPT_DIR.parent / "tabpfn_vdjdb_combined_af3"
RESULTS_DIR = SCRIPT_DIR / "results_auc"
PREDICTIONS_CSV = (
    RESULTS_DIR / "epytope_tcr_viral_tabpfn_ensemble_10model__predictions.csv"
)
PER_PEPTIDE_CSV = (
    RESULTS_DIR / "epytope_tcr_viral_tabpfn_ensemble_10model__per_peptide_auc_0.1.csv"
)
SUMMARY_CSV = (
    RESULTS_DIR / "epytope_tcr_viral_tabpfn_ensemble_10model__summary.csv"
)

MODEL_SPECS = [(split, fold) for split in ("rs", "ss") for fold in range(5)]
EXPECTED_TRAIN_ROWS = 4298
EXPECTED_UNION_ROWS = 23639


def load_full_fold(split: str, fold: int) -> pd.DataFrame:
    frames = []
    dataset_dir = DATASET_ROOT / f"dataset_{split}"
    for part in ("train", "val", "test"):
        path = dataset_dir / f"fold{fold}_{part}.csv"
        best.require_file(path)
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
    return best.merge_features(
        all_rows,
        best.VDJDB_CONFIDENCE_CSV,
        best.VDJDB_GEOMETRY_CSV,
        "VDJDB ten-fold union",
    )


def select_fold_features(
    rows: pd.DataFrame, feature_table: pd.DataFrame
) -> pd.DataFrame:
    feature_cols = best.BASE_COLS + best.GEOMETRY_COLS
    selected = rows.merge(
        feature_table[["id", *feature_cols]],
        on="id",
        how="left",
        validate="one_to_one",
    )
    if selected[feature_cols].isna().any().any():
        raise ValueError("A VDJDB fold contains missing confidence/geometry features")
    return selected


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


def main() -> None:
    best.set_global_seed()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fold_rows = {
        spec: load_full_fold(*spec)
        for spec in MODEL_SPECS
    }
    train_feature_table = build_training_feature_table(fold_rows)

    test_rows = best.load_viral_rows()
    test = best.merge_features(
        test_rows,
        best.VIRAL_CONFIDENCE_CSV,
        best.VIRAL_GEOMETRY_CSV,
        "ePytope viral",
    )
    feature_cols = best.BASE_COLS + best.GEOMETRY_COLS

    print("Loading ESM-2 embedding maps...")
    train_map = best.load_esm_map(
        best.VDJDB_ESM_PATH,
        train_feature_table["id"].astype(str).tolist(),
        "VDJDB ten-fold union",
    )
    test_ids = test["id"].astype(str).tolist()
    test_map = best.load_esm_map(best.VIRAL_ESM_PATH, test_ids, "ePytope viral")

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
        train_esm, test_esm = best.pca_embeddings(
            train_ids, test_ids, train_map, test_map
        )
        x_train = pd.concat(
            [train[feature_cols].reset_index(drop=True), train_esm], axis=1
        )
        x_test = pd.concat(
            [test[feature_cols].reset_index(drop=True), test_esm], axis=1
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
    predictions = test[output_columns].copy()
    for column_index, model_name in enumerate(model_names):
        predictions[f"probability_{model_name}"] = prediction_matrix[:, column_index]
    predictions["ensemble_probability"] = ensemble_probability
    predictions.to_csv(PREDICTIONS_CSV, index=False)
    peptide_results.to_csv(PER_PEPTIDE_CSV, index=False, float_format="%.8f")

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
            }
        ]
    )
    summary.to_csv(SUMMARY_CSV, index=False, float_format="%.4f")

    print("\nePytope-TCR viral TAPAS 10-model ensemble")
    print(f"Macro-AUC     : {macro_auc:.4f}")
    print(f"Macro-AUC@0.1 : {macro_auc_01:.4f}")
    print(f"Predictions    : {PREDICTIONS_CSV}")
    print(f"Per-peptide    : {PER_PEPTIDE_CSV}")
    print(f"Summary        : {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
