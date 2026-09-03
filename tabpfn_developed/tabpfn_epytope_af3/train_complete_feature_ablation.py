#!/usr/bin/env python3
"""Complete six-group TAPAS feature ablation on ePytope-TCR viral.

For every feature set, ten models are trained on the full rows of VDJDB RS
folds 0-4 and SS folds 0-4.  Their raw external probabilities are averaged
pairwise before calculating peptide Macro-AUC and Macro-AUC@0.1.
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
import train_tabpfn_ensemble as ensemble


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results_auc"
MODEL_RESULTS_CSV = RESULTS_DIR / "complete_feature_ablation_model_results.csv"
PREDICTIONS_CSV = RESULTS_DIR / "complete_feature_ablation_predictions.csv"
PER_PEPTIDE_CSV = RESULTS_DIR / "complete_feature_ablation_per_peptide.csv"
SUMMARY_CSV = RESULTS_DIR / "complete_feature_ablation_summary.csv"

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


def feature_matrix(
    data: pd.DataFrame,
    esm: pd.DataFrame,
    use_confidence: bool,
    use_geometry: bool,
    use_esm: bool,
) -> pd.DataFrame:
    parts = []
    if use_confidence:
        parts.append(data[best.BASE_COLS].reset_index(drop=True))
    if use_geometry:
        parts.append(data[best.GEOMETRY_COLS].reset_index(drop=True))
    if use_esm:
        parts.append(esm.reset_index(drop=True))
    return pd.concat(parts, axis=1)


def main() -> None:
    best.set_global_seed()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fold_rows = {
        spec: ensemble.load_full_fold(*spec) for spec in ensemble.MODEL_SPECS
    }
    train_feature_table = ensemble.build_training_feature_table(fold_rows)
    test = best.merge_features(
        best.load_viral_rows(),
        best.VIRAL_CONFIDENCE_CSV,
        best.VIRAL_GEOMETRY_CSV,
        "ePytope viral",
    )

    print("Loading ESM-2 embedding maps...")
    train_map = best.load_esm_map(
        best.VDJDB_ESM_PATH,
        train_feature_table["id"].astype(str).tolist(),
        "VDJDB ten-fold union",
    )
    test_ids = test["id"].astype(str).tolist()
    test_map = best.load_esm_map(best.VIRAL_ESM_PATH, test_ids, "ePytope viral")

    predictions_by_feature = {name: [] for name in FEATURE_SETS}
    model_rows = []
    labels = test["label"].to_numpy(dtype=int)
    peptides = test["peptide"].to_numpy(dtype=str)

    for model_index, (split, fold) in enumerate(ensemble.MODEL_SPECS, start=1):
        model_name = f"{split}_fold{fold}"
        train = ensemble.select_fold_features(
            fold_rows[(split, fold)], train_feature_table
        )
        train_ids = train["id"].astype(str).tolist()
        train_esm, test_esm = best.pca_embeddings(
            train_ids, test_ids, train_map, test_map
        )

        for feature_set, flags in FEATURE_SETS.items():
            x_train = feature_matrix(train, train_esm, *flags)
            x_test = feature_matrix(test, test_esm, *flags)
            best.set_global_seed()
            model = TabPFNClassifier(device=best.DEVICE, random_state=best.SEED)
            model.fit(x_train, train["label"].to_numpy(dtype=int))
            probability = model.predict_proba(x_test)[:, 1]
            predictions_by_feature[feature_set].append(probability)
            peptide_result = ensemble.per_peptide_auc(labels, probability, peptides)
            model_rows.append(
                {
                    "dataset": "ePytope-TCR",
                    "model": model_name,
                    "feature_set": feature_set,
                    "n_train": len(train),
                    "n_test": len(test),
                    "n_features": x_train.shape[1],
                    "macro_auc": peptide_result["auc"].mean(),
                    "macro_auc_0.1": peptide_result["auc_0.1"].mean(),
                }
            )
            print(
                f"[{model_index}/10] {model_name},{feature_set},"
                f"features={x_train.shape[1]},"
                f"macro_auc={peptide_result['auc'].mean():.4f},"
                f"macro_auc_0.1={peptide_result['auc_0.1'].mean():.4f}",
                flush=True,
            )
            del model, x_train, x_test
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del train_esm, test_esm

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
    per_peptide_frames = []
    summary_rows = []
    for feature_set, model_predictions in predictions_by_feature.items():
        ensemble_probability = np.column_stack(model_predictions).mean(axis=1)
        score_column = feature_set.lower().replace("-", "").replace(" ", "_").replace(
            "+", "plus"
        )
        predictions[f"probability__{score_column}"] = ensemble_probability
        peptide_result = ensemble.per_peptide_auc(
            labels, ensemble_probability, peptides
        )
        peptide_result.insert(0, "feature_set", feature_set)
        per_peptide_frames.append(peptide_result)
        summary_rows.append(
            {
                "dataset": "ePytope-TCR",
                "feature_set": feature_set,
                "score_stage": "raw",
                "aggregation": "mean_probability_over_10_models",
                "n_models": len(model_predictions),
                "n_train_per_model": ensemble.EXPECTED_TRAIN_ROWS,
                "n_test": len(test),
                "n_peptides": len(peptide_result),
                "n_features": {
                    "ESM-2 only": best.N_ESM,
                    "Confidence only": len(best.BASE_COLS),
                    "Geometry only": len(best.GEOMETRY_COLS),
                    "ESM-2 + confidence": best.N_ESM + len(best.BASE_COLS),
                    "ESM-2 + geometry": best.N_ESM + len(best.GEOMETRY_COLS),
                    "Full TAPAS": best.N_ESM
                    + len(best.BASE_COLS)
                    + len(best.GEOMETRY_COLS),
                }[feature_set],
                "macro_auc": peptide_result["auc"].mean(),
                "macro_auc_0.1": peptide_result["auc_0.1"].mean(),
            }
        )

    model_results = pd.DataFrame(model_rows)
    summary = pd.DataFrame(summary_rows)
    model_results.to_csv(MODEL_RESULTS_CSV, index=False, float_format="%.8f")
    predictions.to_csv(PREDICTIONS_CSV, index=False, float_format="%.8f")
    pd.concat(per_peptide_frames, ignore_index=True).to_csv(
        PER_PEPTIDE_CSV, index=False, float_format="%.8f"
    )
    summary.to_csv(SUMMARY_CSV, index=False, float_format="%.8f")

    print("\nePytope-TCR complete feature ablation")
    print(
        summary[
            ["feature_set", "n_features", "macro_auc", "macro_auc_0.1"]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print(f"Summary: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
