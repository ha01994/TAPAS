#!/usr/bin/env python3
"""Permutation SHAP for the ePytope-TCR structural-only TAPAS submodel.

One TabPFN model is trained on all unique VDJDB RS fold-0 train, validation,
and test rows.  SHAP explains raw ePytope-TCR probabilities from the 4 AF3
confidence and 11 geometry features; ESM-2 is deliberately excluded so this
is an interpretation of the structural-only submodel, not full TAPAS.
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
from tabpfn import TabPFNClassifier

import train_tabpfn as best


warnings.filterwarnings("ignore")

OUT_DIR = best.SCRIPT_DIR / "shap_epytope_tcr_best_no_esm"
FIGURES_DIR = best.SCRIPT_DIR / "figures"
BACKGROUND_SIZE = 100
N_PER_CLASS = 100
PREDICT_BATCH_SIZE = 200
MAX_DISPLAY = 15

FEATURES = best.BASE_COLS + best.GEOMETRY_COLS
FEATURE_GROUPS = (
    ["Confidence"] * len(best.BASE_COLS)
    + ["Geometry"] * len(best.GEOMETRY_COLS)
)


def prepare_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    print("Step 1: Loading VDJDB RS fold-0 train+val+test rows...")
    train_rows = best.load_vdjdb_rows()
    train = best.merge_features(
        train_rows,
        best.VDJDB_CONFIDENCE_CSV,
        best.VDJDB_GEOMETRY_CSV,
        "VDJDB",
    )

    print("Step 2: Loading the active ePytope-TCR viral dataset...")
    test_rows = best.load_viral_rows()
    test = best.merge_features(
        test_rows,
        best.VIRAL_CONFIDENCE_CSV,
        best.VIRAL_GEOMETRY_CSV,
        "ePytope viral",
    )

    x_train = train[FEATURES].reset_index(drop=True)
    x_test = test[FEATURES].reset_index(drop=True)
    y_train = train["label"].astype(int).reset_index(drop=True)
    y_test = test["label"].astype(int).reset_index(drop=True)
    peptide_test = test["peptide"].astype(str).reset_index(drop=True)
    if x_train.isna().any().any() or x_test.isna().any().any():
        raise ValueError("Prepared confidence/geometry matrices contain missing values")
    print(
        f"  - Train: {x_train.shape}; test: {x_test.shape}; "
        f"test positive={(y_test == 1).sum()}; test negative={(y_test == 0).sum()}"
    )
    return x_train, x_test, y_train, y_test, peptide_test


def main() -> None:
    start_time = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    best.set_global_seed(best.SEED)
    x_train, x_test, y_train, y_test, peptide_test = prepare_data()

    print("Step 3: Training one confidence + geometry TabPFN model...")
    model = TabPFNClassifier(device=best.DEVICE, random_state=best.SEED)
    model.fit(x_train, y_train)
    raw_probability = model.predict_proba(x_test)[:, 1]
    print(
        "  - Raw Macro-AUC:     "
        f"{best.macro_auc(y_test.to_numpy(), raw_probability, peptide_test.to_numpy()):.4f}"
    )
    print(
        "  - Raw Macro-AUC@0.1: "
        f"{best.macro_auc(y_test.to_numpy(), raw_probability, peptide_test.to_numpy(), max_fpr=0.1):.4f}"
    )

    print("Step 4: Running permutation SHAP on raw probability...")
    background = x_train.sample(BACKGROUND_SIZE, random_state=best.SEED)

    def model_predict(values: np.ndarray) -> np.ndarray:
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

    rng = np.random.default_rng(best.SEED)
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

    masker = shap.maskers.Independent(
        background.to_numpy(), max_samples=BACKGROUND_SIZE
    )
    explainer = shap.explainers.Permutation(
        model_predict,
        masker=masker,
        feature_names=FEATURES,
    )
    shap_values = explainer(x_explain, max_evals=max_evals)
    print(f"  - SHAP shape: {shap_values.values.shape}")
    print(f"  - Elapsed: {(time.time() - start_time) / 60:.2f} minutes")

    with (OUT_DIR / "shap_values_epytope_tcr.pkl").open("wb") as handle:
        pickle.dump(shap_values, handle)

    print("Step 5: Generating figures and numeric summary...")
    plt.figure(figsize=(12, 9))
    shap.plots.beeswarm(shap_values, max_display=MAX_DISPLAY, show=False)
    plt.title("SHAP Beeswarm — Confidence + Geometry (ePytope-TCR)")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "shap_beeswarm_epytope_tcr.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(12, 9))
    shap.plots.bar(shap_values, max_display=MAX_DISPLAY, show=False)
    plt.title("Mean |SHAP| — Confidence + Geometry (ePytope-TCR)")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "shap_bar_epytope_tcr.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(10, 7))
    shap.plots.waterfall(shap_values[0], max_display=MAX_DISPLAY, show=False)
    plt.title("SHAP Waterfall — Positive Sample 0 (ePytope-TCR)")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "shap_waterfall_epytope_tcr.png",
        dpi=300,
        bbox_inches="tight",
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
    summary_path = OUT_DIR / "shap_summary_epytope_tcr.csv"
    summary.to_csv(summary_path, index=False)
    print("\nFeature-level SHAP summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved SHAP values and summary to: {OUT_DIR}")
    print(f"Saved figures to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
