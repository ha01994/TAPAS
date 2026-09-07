#!/usr/bin/env python3
"""Permutation SHAP for ImmRep25 using confidence and geometry features.

SHAP explains the raw TabPFN probability from the structural-only submodel.
"""

from __future__ import annotations

import os
import pickle
import time
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
os.environ.setdefault("POSTHOG_DISABLED", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier

from train_tabpfn import (
    BASE_COLS,
    DEVICE,
    FINAL_GEOMETRY_COLS,
    GEOMETRY_IMMREP25_CSV,
    GEOMETRY_VDJDB_CSV,
    IMMREP25_PAIRS_CSV,
    METRICS_BEST_CSV,
    SEED,
    SOURCE_DIR,
    VDJDB_DIR,
    VDJDB_METRICS_CSV,
    load_geometry_features,
    macro_auc,
    peptide_from_pmhc,
    set_global_seed,
)


warnings.filterwarnings("ignore")

OUT_DIR = Path(SOURCE_DIR) / "shap_immrep25_best_no_esm"
FIGURES_DIR = Path(SOURCE_DIR) / "figures"
BACKGROUND_SIZE = 100
N_PER_CLASS = 100
PREDICT_BATCH_SIZE = 200
MAX_DISPLAY = 15
TRAIN_SPLIT = "dataset_rs"

FEATURES = BASE_COLS + FINAL_GEOMETRY_COLS
FEATURE_GROUPS = (
    ["Confidence"] * len(BASE_COLS)
    + ["Geometry"] * len(FINAL_GEOMETRY_COLS)
)


def prepare_data():
    print(f"Step 1: Loading VDJDB training data from {TRAIN_SPLIT}...")
    split_dir = Path(VDJDB_DIR) / "data" / TRAIN_SPLIT
    parts = []
    for part in ("train", "val", "test"):
        path = split_dir / f"fold0_{part}.csv"
        if path.exists():
            parts.append(pd.read_csv(path))
    if not parts:
        raise FileNotFoundError(f"No fold0 train/val/test files under {split_dir}")
    train_raw = (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates(subset="id")
        .reset_index(drop=True)
    )
    print(
        f"  - Unique train rows: {len(train_raw)}; "
        f"positive={(train_raw['label'] == 1).sum()}; "
        f"negative={(train_raw['label'] == 0).sum()}"
    )
    vdjdb_metrics = pd.read_csv(VDJDB_METRICS_CSV)
    merge_cols = ["pdb_id", *BASE_COLS]
    train = pd.merge(
        train_raw,
        vdjdb_metrics[merge_cols],
        left_on="id",
        right_on="pdb_id",
        how="inner",
    )
    if len(train) != len(train_raw):
        raise ValueError("VDJDB confidence merge changed the training row count")

    vdjdb_geometry = load_geometry_features(GEOMETRY_VDJDB_CSV, "VDJDB")
    missing = [feature for feature in FINAL_GEOMETRY_COLS if feature not in vdjdb_geometry]
    if missing:
        raise ValueError(f"VDJDB geometry is missing selected features: {missing}")
    train = pd.merge(
        train, vdjdb_geometry, left_on="id", right_on="pair_id", how="inner"
    )
    if len(train) != len(train_raw):
        raise ValueError("VDJDB geometry merge changed the training row count")

    print("Step 2: Loading ImmRep25 test data...")
    test_raw = pd.read_csv(IMMREP25_PAIRS_CSV)
    test_raw["peptide"] = test_raw["pmhc"].map(peptide_from_pmhc)
    test_raw["pdb_id"] = test_raw["id"].astype(str)
    immrep25_metrics = pd.read_csv(METRICS_BEST_CSV)
    test = pd.merge(test_raw, immrep25_metrics[merge_cols], on="pdb_id", how="inner")
    if len(test) != len(test_raw):
        raise ValueError("ImmRep25 confidence merge changed the test row count")

    immrep25_geometry = load_geometry_features(GEOMETRY_IMMREP25_CSV, "ImmRep25")
    missing = [feature for feature in FINAL_GEOMETRY_COLS if feature not in immrep25_geometry]
    if missing:
        raise ValueError(f"ImmRep25 geometry is missing selected features: {missing}")
    test = pd.merge(
        test, immrep25_geometry, left_on="id", right_on="pair_id", how="inner"
    )
    if len(test) != len(test_raw):
        raise ValueError("ImmRep25 geometry merge changed the test row count")

    x_train = train[FEATURES].reset_index(drop=True)
    x_test = test[FEATURES].reset_index(drop=True)
    y_train = train["label"].reset_index(drop=True)
    y_test = test["label"].reset_index(drop=True)
    peptide_test = test["peptide"].reset_index(drop=True)
    print(f"  - Train: {x_train.shape}; Test: {x_test.shape}")
    return x_train, x_test, y_train, y_test, peptide_test


def main() -> None:
    start_time = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    set_global_seed(SEED)
    x_train, x_test, y_train, y_test, peptide_test = prepare_data()

    print("Step 3: Training the confidence + geometry TabPFN model...")
    model = TabPFNClassifier(device=DEVICE, random_state=SEED)
    model.fit(x_train, y_train)
    raw_probability = model.predict_proba(x_test)[:, 1]
    print(f"  - Raw ROC-AUC:                {roc_auc_score(y_test, raw_probability):.4f}")
    print(
        "  - Raw Macro-AUC:              "
        f"{macro_auc(y_test, raw_probability, peptide_test):.4f}"
    )
    print("Step 4: Running feature-level permutation SHAP on raw probability...")
    background = x_train.sample(BACKGROUND_SIZE, random_state=SEED)

    def model_predict(values):
        predictions = []
        for start in range(0, len(values), PREDICT_BATCH_SIZE):
            batch = pd.DataFrame(
                values[start : start + PREDICT_BATCH_SIZE], columns=FEATURES
            )
            with torch.no_grad():
                predictions.append(model.predict_proba(batch)[:, 1])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return np.concatenate(predictions)

    positive = y_test[y_test == 1].index.to_numpy()
    negative = y_test[y_test == 0].index.to_numpy()
    if len(positive) < N_PER_CLASS or len(negative) < N_PER_CLASS:
        raise ValueError(
            f"Need {N_PER_CLASS} rows per class; found "
            f"positive={len(positive)}, negative={len(negative)}"
        )

    rng = np.random.default_rng(SEED)
    explain_index = np.concatenate(
        [
            rng.choice(positive, size=N_PER_CLASS, replace=False),
            rng.choice(negative, size=N_PER_CLASS, replace=False),
        ]
    )
    x_explain = x_test.loc[explain_index].to_numpy()
    max_evals = 2 * len(FEATURES) + 1
    print(
        f"  - Explain set: {N_PER_CLASS} positive + {N_PER_CLASS} negative"
        f"\n  - Features: {len(FEATURES)}; max_evals: {max_evals}"
    )

    masker = shap.maskers.Independent(background.to_numpy(), max_samples=100)
    explainer = shap.explainers.Permutation(
        model_predict,
        masker=masker,
        feature_names=FEATURES,
    )
    shap_values = explainer(x_explain, max_evals=max_evals)
    print(f"  - SHAP shape: {shap_values.values.shape}")
    print(f"  - Elapsed: {(time.time() - start_time) / 60:.2f} minutes")

    pickle_path = OUT_DIR / "shap_values_immrep25.pkl"
    with pickle_path.open("wb") as handle:
        pickle.dump(shap_values, handle)

    print("Step 5: Generating plots and numeric summary...")
    plt.figure(figsize=(12, 9))
    shap.plots.beeswarm(shap_values, max_display=MAX_DISPLAY, show=False)
    plt.title("SHAP Beeswarm — Confidence + Geometry (IMMREP25)")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "shap_beeswarm_immrep25.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    plt.figure(figsize=(12, 9))
    shap.plots.bar(shap_values, max_display=MAX_DISPLAY, show=False)
    plt.title("Mean |SHAP| — Confidence + Geometry (IMMREP25)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_bar_immrep25.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 7))
    shap.plots.waterfall(shap_values[0], max_display=MAX_DISPLAY, show=False)
    plt.title("SHAP Waterfall — Sample 0 (IMMREP25)")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "shap_waterfall_immrep25.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    summary = pd.DataFrame(
        {
            "feature": FEATURES,
            "group": FEATURE_GROUPS,
            "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
            "mean_shap": shap_values.values.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    summary_path = OUT_DIR / "shap_summary_immrep25.csv"
    summary.to_csv(summary_path, index=False)
    print("\nFeature-level SHAP summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
