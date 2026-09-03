#!/usr/bin/env python3
"""Complete six-group TAPAS feature ablation on IMMREP25.

For each feature set, ten full-fold VDJDB models are ensembled by averaging
raw pair probabilities.  The script reports raw, Bradley-normalized, and
small-cluster-smoothed peptide Macro-AUC and Macro-AUC@0.1.
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
CLUSTERS_CSV = RESULTS_DIR / "immrep25_tabpfn_best__smallclust_clusters.csv"
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
        parts.append(data[best.FINAL_GEOMETRY_COLS].reset_index(drop=True))
    if use_esm:
        parts.append(esm.reset_index(drop=True))
    return pd.concat(parts, axis=1)


def load_clusters() -> pd.DataFrame:
    ensemble.require_file(CLUSTERS_CSV)
    clusters = pd.read_csv(CLUSTERS_CSV)
    required = {"tcr_id", "hla", "threshold", "cluster_id", "cluster_size"}
    missing = sorted(required - set(clusters.columns))
    if missing:
        raise ValueError(f"Small-cluster CSV is missing columns: {missing}")
    clusters = clusters.loc[
        clusters["threshold"] == best.SMALLCLUST_THRESHOLD
    ].copy()
    if clusters["tcr_id"].nunique() != 1000:
        raise ValueError("Expected cluster assignments for 1,000 IMMREP25 TCRs")
    return clusters


def postprocess(
    test: pd.DataFrame,
    raw_probability: np.ndarray,
    clusters: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    normalized = best.bradley_normalize_immrep25(
        raw_probability,
        tcr_ids=test["tcr_id"].to_numpy(),
        peptide_ids=test["peptide"].to_numpy(),
        pmhcs=test["pmhc"].to_numpy(),
    )
    scored = test[["id", "tcr_id", "pmhc", "peptide", "label"]].copy()
    scored["_row_order"] = np.arange(len(scored))
    scored["hla"] = scored["pmhc"].map(best.hla_from_pmhc)
    scored["normalized"] = normalized
    scored = scored.merge(
        clusters[["tcr_id", "hla", "cluster_id", "cluster_size"]],
        on=["tcr_id", "hla"],
        how="left",
        validate="many_to_one",
    )
    scored = scored.sort_values("_row_order")
    if scored["cluster_id"].isna().any():
        raise ValueError("Some IMMREP25 pairs lack small-cluster assignments")
    cluster_means = (
        scored.groupby(["hla", "cluster_id", "peptide"], sort=False)["normalized"]
        .mean()
        .rename("cluster_mean")
        .reset_index()
    )
    scored = scored.merge(
        cluster_means,
        on=["hla", "cluster_id", "peptide"],
        how="left",
        validate="many_to_one",
    )
    smoothed = scored["cluster_mean"].to_numpy() * np.sqrt(
        scored["cluster_size"].to_numpy(dtype=float)
    )
    return normalized, smoothed


def feature_count(feature_set: str) -> int:
    return {
        "ESM-2 only": best.N_ESM,
        "Confidence only": len(best.BASE_COLS),
        "Geometry only": len(best.FINAL_GEOMETRY_COLS),
        "ESM-2 + confidence": best.N_ESM + len(best.BASE_COLS),
        "ESM-2 + geometry": best.N_ESM + len(best.FINAL_GEOMETRY_COLS),
        "Full TAPAS": best.N_ESM
        + len(best.BASE_COLS)
        + len(best.FINAL_GEOMETRY_COLS),
    }[feature_set]


def main() -> None:
    best.set_global_seed()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    clusters = load_clusters()

    fold_rows = {
        spec: ensemble.load_full_fold(*spec) for spec in ensemble.MODEL_SPECS
    }
    train_feature_table = ensemble.build_training_feature_table(fold_rows)
    test = ensemble.load_test_features()

    print("Loading ESM-2 embedding maps...")
    train_map = ensemble.load_esm_map(
        Path(best.ESM_VDJDB_PATH),
        train_feature_table["id"].astype(str).tolist(),
        "VDJDB ten-fold union",
    )
    test_ids = test["id"].astype(str).tolist()
    test_map = ensemble.load_esm_map(
        Path(best.ESM_IMMREP25_PATH), test_ids, "IMMREP25"
    )

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
        train_esm, test_esm = best.get_pca_embeddings(
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
                    "dataset": "IMMREP25",
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
                f"raw_macro_auc={peptide_result['auc'].mean():.4f},"
                f"raw_macro_auc_0.1={peptide_result['auc_0.1'].mean():.4f}",
                flush=True,
            )
            del model, x_train, x_test
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del train_esm, test_esm

    predictions = test[["id", "tcr_id", "peptide", "pmhc", "label"]].copy()
    per_peptide_frames = []
    summary_rows = []
    for feature_set, model_predictions in predictions_by_feature.items():
        raw = np.column_stack(model_predictions).mean(axis=1)
        normalized, postprocessed = postprocess(test, raw, clusters)
        slug = feature_set.lower().replace("-", "").replace(" ", "_").replace(
            "+", "plus"
        )
        predictions[f"raw__{slug}"] = raw
        predictions[f"normalized__{slug}"] = normalized
        predictions[f"postprocessed__{slug}"] = postprocessed

        stage_results = {}
        for stage, scores in (
            ("raw", raw),
            ("normalized", normalized),
            ("postprocessed", postprocessed),
        ):
            peptide_result = ensemble.per_peptide_auc(labels, scores, peptides)
            peptide_result.insert(0, "score_stage", stage)
            peptide_result.insert(0, "feature_set", feature_set)
            per_peptide_frames.append(peptide_result)
            stage_results[stage] = (
                peptide_result["auc"].mean(),
                peptide_result["auc_0.1"].mean(),
            )
        summary_rows.append(
            {
                "dataset": "IMMREP25",
                "feature_set": feature_set,
                "aggregation": "mean_probability_over_10_models",
                "n_models": len(model_predictions),
                "n_train_per_model": ensemble.EXPECTED_TRAIN_ROWS,
                "n_test": len(test),
                "n_peptides": test["peptide"].nunique(),
                "n_features": feature_count(feature_set),
                "raw_macro_auc": stage_results["raw"][0],
                "raw_macro_auc_0.1": stage_results["raw"][1],
                "normalized_macro_auc": stage_results["normalized"][0],
                "normalized_macro_auc_0.1": stage_results["normalized"][1],
                "postprocessed_macro_auc": stage_results["postprocessed"][0],
                "postprocessed_macro_auc_0.1": stage_results["postprocessed"][1],
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

    print("\nIMMREP25 complete feature ablation")
    print(
        summary[
            [
                "feature_set",
                "n_features",
                "raw_macro_auc_0.1",
                "postprocessed_macro_auc_0.1",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print(f"Summary: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
