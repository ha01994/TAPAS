#!/usr/bin/env python3
"""Complete six-group TAPAS feature ablation on VDJDB RS and SS.

Each configuration is evaluated on the existing five test folds.  ESM-2 PCA
is fitted independently on each fold's training rows and applied to that
fold's test rows.  Results are peptide Macro-AUC and standardized partial
Macro-AUC@0.1, averaged over folds exactly as in the main VDJDB evaluation.
"""

from __future__ import annotations

import gc
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tabpfn import TabPFNClassifier

import train_tabpfn_best as best


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SCRIPT_DIR
RESULTS_DIR = SCRIPT_DIR / "results_auc"
FOLD_RESULTS_CSV = RESULTS_DIR / "complete_feature_ablation_folds.csv"
SUMMARY_CSV = RESULTS_DIR / "complete_feature_ablation_summary.csv"
PREDICTIONS_CSV = RESULTS_DIR / "complete_feature_ablation_predictions.csv"

SPLITS = OrderedDict([("RS", "dataset_rs"), ("SS", "dataset_ss")])
FEATURE_SETS = OrderedDict(
    [
        ("ESM-2 only", (False, False, True)),
        ("Confidence only", (True, False, False)),
        ("Geometry only", (False, True, False)),
        ("ESM-2 + confidence", (True, False, True)),
        ("ESM-2 + geometry", (False, True, True)),
        ("Full TAPAS", (True, True, True)),
    ]
)


def merge_features(
    rows: pd.DataFrame,
    metrics: pd.DataFrame,
    geometry: pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    rows = rows.copy()
    rows["id"] = rows["id"].astype(str)
    merged = rows.merge(
        metrics[["pdb_id", *best.BASE_FEATURE_COLS]],
        left_on="id",
        right_on="pdb_id",
        how="left",
        validate="one_to_one",
    ).merge(
        geometry[["pair_id", *best.FINAL_GEOMETRY_COLS]],
        left_on="id",
        right_on="pair_id",
        how="left",
        validate="one_to_one",
    )
    feature_columns = best.BASE_FEATURE_COLS + best.FINAL_GEOMETRY_COLS
    if len(merged) != len(rows) or merged[feature_columns].isna().any().any():
        raise ValueError(f"{name}: confidence/geometry merge is incomplete")
    return merged


def feature_matrix(
    data: pd.DataFrame,
    esm: pd.DataFrame,
    use_confidence: bool,
    use_geometry: bool,
    use_esm: bool,
) -> pd.DataFrame:
    parts = []
    if use_confidence:
        parts.append(data[best.BASE_FEATURE_COLS].reset_index(drop=True))
    if use_geometry:
        parts.append(data[best.FINAL_GEOMETRY_COLS].reset_index(drop=True))
    if use_esm:
        parts.append(esm.reset_index(drop=True))
    return pd.concat(parts, axis=1)


def main() -> None:
    best.set_global_seed(best.SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = best.load_metrics()
    metrics["pdb_id"] = metrics["pdb_id"].astype(str)
    geometry, _ = best.load_geometry()
    raw_esm = best.load_raw_esm()

    fold_rows = []
    prediction_rows = []
    for split_label, dataset_name in SPLITS.items():
        dataset_dir = DATASET_ROOT / dataset_name
        for fold in range(5):
            train_raw = pd.read_csv(dataset_dir / f"fold{fold}_train.csv")
            test_raw = pd.read_csv(dataset_dir / f"fold{fold}_test.csv")
            train = merge_features(
                train_raw, metrics, geometry, f"{split_label} fold{fold} train"
            )
            test = merge_features(
                test_raw, metrics, geometry, f"{split_label} fold{fold} test"
            )
            train_ids = train["id"].astype(str).tolist()
            test_ids = test["id"].astype(str).tolist()
            train_esm, test_esm = best.get_pca_embeddings_for_fold(
                train_ids, test_ids, raw_esm
            )

            for feature_set, flags in FEATURE_SETS.items():
                x_train = feature_matrix(train, train_esm, *flags)
                x_test = feature_matrix(test, test_esm, *flags)
                best.set_global_seed(best.SEED)
                model = TabPFNClassifier(
                    device=best.DEVICE, random_state=best.SEED
                )
                model.fit(x_train, train["label"].to_numpy(dtype=int))
                probability = model.predict_proba(x_test)[:, 1]
                macro_auc = best.macro_auc(
                    test["label"], probability, test["pmhc"]
                )
                macro_auc_01 = best.macro_auc(
                    test["label"], probability, test["pmhc"], max_fpr=0.1
                )
                fold_rows.append(
                    {
                        "dataset": "VDJDB",
                        "split": split_label,
                        "fold": fold,
                        "feature_set": feature_set,
                        "n_train": len(train),
                        "n_test": len(test),
                        "n_features": x_train.shape[1],
                        "macro_auc": macro_auc,
                        "macro_auc_0.1": macro_auc_01,
                    }
                )
                prediction = test[["id", "pmhc", "label"]].copy()
                prediction.insert(0, "feature_set", feature_set)
                prediction.insert(0, "fold", fold)
                prediction.insert(0, "split", split_label)
                prediction["probability"] = probability
                prediction_rows.append(prediction)
                print(
                    f"{split_label},fold{fold},{feature_set},"
                    f"features={x_train.shape[1]},macro_auc={macro_auc:.4f},"
                    f"macro_auc_0.1={macro_auc_01:.4f}",
                    flush=True,
                )
                del model, x_train, x_test
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    folds = pd.DataFrame(fold_rows)
    summary = (
        folds.groupby(["dataset", "split", "feature_set"], sort=False)
        .agg(
            n_folds=("fold", "nunique"),
            n_features=("n_features", "first"),
            macro_auc=("macro_auc", "mean"),
            macro_auc_sd=("macro_auc", "std"),
            macro_auc_0_1=("macro_auc_0.1", "mean"),
            macro_auc_0_1_sd=("macro_auc_0.1", "std"),
        )
        .reset_index()
        .rename(
            columns={
                "macro_auc_0_1": "macro_auc_0.1",
                "macro_auc_0_1_sd": "macro_auc_0.1_sd",
            }
        )
    )
    folds.to_csv(FOLD_RESULTS_CSV, index=False, float_format="%.8f")
    summary.to_csv(SUMMARY_CSV, index=False, float_format="%.8f")
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        PREDICTIONS_CSV, index=False, float_format="%.8f"
    )

    print("\nVDJDB complete feature ablation")
    print(
        summary[
            ["split", "feature_set", "n_features", "macro_auc", "macro_auc_0.1"]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print(f"Summary: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
