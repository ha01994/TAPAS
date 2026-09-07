#!/usr/bin/env python3
"""Fold-0 structural-only SHAP analysis for VDJDB RS and SS.

For each split, one TabPFN model is trained on fold 0's training rows and
explained only on fold 0's held-out test rows. Mean absolute SHAP values are
normalized to sum to one within each split. ESM-2 is deliberately excluded:
this script interprets the 4-confidence + 11-geometry structural-only submodel.
"""

from __future__ import annotations

import argparse
import gc
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

OUT_DIR = Path(best.SOURCE_DIR) / "shap_rs_ss_fold0_no_esm"
FIGURES_DIR = Path(best.SOURCE_DIR) / "figures"
BACKGROUND_SIZE = 100
N_PER_CLASS = 100
PREDICT_BATCH_SIZE = 200
MAX_DISPLAY = 15
FOLD = 0

FEATURES = best.BASE_FEATURE_COLS + best.FINAL_GEOMETRY_COLS
FEATURE_GROUPS = (
    ["Confidence"] * len(best.BASE_FEATURE_COLS)
    + ["Geometry"] * len(best.FINAL_GEOMETRY_COLS)
)
SPLITS = {"RS": "dataset_rs", "SS": "dataset_ss"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(SPLITS),
        default=list(SPLITS),
        help="Splits to analyze (default: RS SS).",
    )
    return parser.parse_args()


def prepare_fold_data(
    dataset_folder: str,
    fold: int,
    metrics: pd.DataFrame,
    geometry: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    split_dir = Path(best.DATASET_ROOT) / dataset_folder
    train_raw = pd.read_csv(split_dir / f"fold{fold}_train.csv")
    test_raw = pd.read_csv(split_dir / f"fold{fold}_test.csv")
    merge_cols = ["pdb_id", *best.BASE_FEATURE_COLS]

    train = train_raw.merge(
        metrics[merge_cols],
        left_on="id",
        right_on="pdb_id",
        how="inner",
        validate="one_to_one",
    ).merge(
        geometry,
        left_on="id",
        right_on="pair_id",
        how="inner",
        validate="one_to_one",
    )
    test = test_raw.merge(
        metrics[merge_cols],
        left_on="id",
        right_on="pdb_id",
        how="inner",
        validate="one_to_one",
    ).merge(
        geometry,
        left_on="id",
        right_on="pair_id",
        how="inner",
        validate="one_to_one",
    )
    if len(train) != len(train_raw) or len(test) != len(test_raw):
        raise ValueError(
            f"{dataset_folder} fold {fold}: feature merge changed row counts"
        )

    x_train = train[FEATURES].reset_index(drop=True)
    x_test = test[FEATURES].reset_index(drop=True)
    y_train = train["label"].astype(int).reset_index(drop=True)
    y_test = test["label"].astype(int).reset_index(drop=True)
    pmhc_test = test["pmhc"].astype(str).reset_index(drop=True)
    if x_train.isna().any().any() or x_test.isna().any().any():
        raise ValueError(f"{dataset_folder} fold {fold}: prepared features contain NaN")
    return x_train, x_test, y_train, y_test, pmhc_test


def explain_fold(
    split_label: str,
    fold: int,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    pmhc_test: pd.Series,
) -> tuple[shap.Explanation, pd.DataFrame, dict[str, float | int | str]]:
    print(
        f"\n[{split_label} fold {fold}] "
        f"train={x_train.shape}, test={x_test.shape}"
    )
    best.set_global_seed(best.SEED + fold)
    model = TabPFNClassifier(device=best.DEVICE, random_state=best.SEED)
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]
    macro_auc = best.macro_auc(y_test, probability, pmhc_test)
    macro_auc_01 = best.macro_auc(
        y_test, probability, pmhc_test, max_fpr=0.1
    )
    print(f"  Macro-AUC={macro_auc:.4f}; Macro-AUC@0.1={macro_auc_01:.4f}")

    background = x_train.sample(
        BACKGROUND_SIZE, random_state=best.SEED + fold
    ).to_numpy()

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
    if min(len(positive), len(negative)) < N_PER_CLASS:
        raise ValueError(
            f"{split_label} fold {fold}: need {N_PER_CLASS} rows per class; "
            f"found positive={len(positive)}, negative={len(negative)}"
        )
    split_offset = 0 if split_label == "RS" else 10_000
    rng = np.random.default_rng(best.SEED + split_offset + fold)
    explain_index = np.concatenate(
        [
            rng.choice(positive, N_PER_CLASS, replace=False),
            rng.choice(negative, N_PER_CLASS, replace=False),
        ]
    )
    x_explain = x_test.loc[explain_index].to_numpy()
    explainer = shap.explainers.Permutation(
        model_predict,
        masker=shap.maskers.Independent(
            background, max_samples=BACKGROUND_SIZE
        ),
        feature_names=FEATURES,
    )
    explanation = explainer(
        x_explain,
        max_evals=2 * len(FEATURES) + 1,
    )

    mean_abs = np.abs(explanation.values).mean(axis=0)
    importance_sum = float(mean_abs.sum())
    if importance_sum <= 0:
        raise ValueError(f"{split_label} fold {fold}: zero total SHAP importance")
    fold_importance = pd.DataFrame(
        {
            "split": split_label,
            "fold": fold,
            "feature": FEATURES,
            "group": FEATURE_GROUPS,
            "mean_abs_shap": mean_abs,
            "normalized_importance": mean_abs / importance_sum,
            "mean_shap": explanation.values.mean(axis=0),
            "n_explained": len(explanation.values),
        }
    )
    fold_metrics = {
        "split": split_label,
        "fold": fold,
        "n_train": len(x_train),
        "n_test": len(x_test),
        "n_explained": len(explanation.values),
        "macro_auc": macro_auc,
        "macro_auc_0.1": macro_auc_01,
    }
    return explanation, fold_importance, fold_metrics


def summarize_fold0(fold_importance: pd.DataFrame) -> pd.DataFrame:
    summary = fold_importance[
        [
            "split",
            "fold",
            "feature",
            "group",
            "mean_abs_shap",
            "normalized_importance",
            "mean_shap",
            "n_explained",
        ]
    ].copy()
    summary["rank"] = summary.groupby("split")["normalized_importance"].rank(
        method="first", ascending=False
    ).astype(int)
    return summary.sort_values(["split", "rank"], ignore_index=True)


def importance_matrix(summary: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    matrix = summary.pivot(
        index="feature", columns="split", values="normalized_importance"
    )[["RS", "SS"]]
    order = matrix.mean(axis=1).sort_values(ascending=False).index.tolist()
    return matrix.loc[order], order


def plot_heatmap(matrix: pd.DataFrame, output: Path, title: str) -> None:
    height = max(7.5, 0.48 * len(matrix))
    fig, ax = plt.subplots(figsize=(8.5, height))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(matrix.columns)), labels=matrix.columns)
    ax.set_yticks(np.arange(len(matrix.index)), labels=matrix.index)
    ax.set_title(title)
    for row in range(len(matrix.index)):
        for column in range(len(matrix.columns)):
            value = matrix.iat[row, column]
            color = "white" if value > matrix.to_numpy().max() * 0.55 else "black"
            ax.text(column, row, f"{value:.3f}", ha="center", va="center", color=color)
    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label("Normalized mean |SHAP|")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_rs_ss_bar(summary: pd.DataFrame, order: list[str]) -> None:
    rs = summary.loc[summary["split"] == "RS"].set_index("feature").loc[order]
    ss = summary.loc[summary["split"] == "SS"].set_index("feature").loc[order]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.barh(
        y - 0.19,
        rs["normalized_importance"],
        height=0.36,
        label="VDJDB RS",
    )
    ax.barh(
        y + 0.19,
        ss["normalized_importance"],
        height=0.36,
        label="VDJDB SS",
    )
    ax.set_yticks(y, labels=order)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized mean |SHAP|")
    ax.set_title("Structural-only SHAP Importance — VDJDB RS vs SS, fold 0")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "shap_normalized_importance_bar_vdjdb_rs_ss_fold0.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    selected_splits = list(dict.fromkeys(args.splits))
    start_time = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    best.set_global_seed(best.SEED)

    print("Loading shared best-ranking-score confidence and geometry features...")
    metrics = best.load_metrics()
    geometry, _ = best.load_geometry()

    importance_frames = []
    metric_rows = []

    for split_label in selected_splits:
        dataset_folder = SPLITS[split_label]
        fold_data = prepare_fold_data(dataset_folder, FOLD, metrics, geometry)
        explanation, importance, metrics_row = explain_fold(
            split_label, FOLD, *fold_data
        )
        importance_frames.append(importance)
        metric_rows.append(metrics_row)
        with (OUT_DIR / f"shap_values_{split_label.lower()}_fold0.pkl").open(
            "wb"
        ) as handle:
            pickle.dump(explanation, handle)

        split_summary = summarize_fold0(importance)
        split_summary.to_csv(
            OUT_DIR / f"shap_summary_{split_label.lower()}_fold0.csv",
            index=False,
            float_format="%.8f",
        )
        pd.DataFrame([metrics_row]).to_csv(
            OUT_DIR / f"shap_metrics_{split_label.lower()}_fold0.csv",
            index=False,
            float_format="%.8f",
        )
        plt.figure(figsize=(12, 9))
        shap.plots.beeswarm(explanation, max_display=MAX_DISPLAY, show=False)
        plt.title(
            f"SHAP Beeswarm — Confidence + Geometry (VDJDB {split_label}, fold 0)"
        )
        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / f"shap_beeswarm_vdjdb_{split_label.lower()}_fold0.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fold_importance = pd.concat(importance_frames, ignore_index=True)
    fold_metrics = pd.DataFrame(metric_rows)
    summary = summarize_fold0(fold_importance)

    if set(selected_splits) == set(SPLITS):
        fold_importance.to_csv(
            OUT_DIR / "shap_rs_ss_fold0_feature_importance.csv",
            index=False,
            float_format="%.8f",
        )
        fold_metrics.to_csv(
            OUT_DIR / "shap_rs_ss_fold0_metrics.csv",
            index=False,
            float_format="%.8f",
        )
        summary.to_csv(
            OUT_DIR / "shap_rs_ss_fold0_summary.csv",
            index=False,
            float_format="%.8f",
        )
        rs_ss_matrix, order = importance_matrix(summary)
        rs_ss_matrix.to_csv(
            OUT_DIR / "shap_normalized_importance_vdjdb_rs_ss.csv",
            float_format="%.8f",
        )
        plot_heatmap(
            rs_ss_matrix,
            FIGURES_DIR / "shap_normalized_importance_heatmap_vdjdb_rs_ss_fold0.png",
            "Normalized Structural-only SHAP Importance — VDJDB RS vs SS, fold 0",
        )
        plot_rs_ss_bar(summary, order)

    print("\nFold-0 normalized SHAP summary:")
    print(
        summary[
            [
                "split",
                "rank",
                "feature",
                "group",
                "normalized_importance",
            ]
        ].to_string(index=False)
    )
    print(f"\nElapsed: {(time.time() - start_time) / 60:.2f} minutes")
    print(f"Saved numeric outputs to: {OUT_DIR}")
    print(f"Saved figures to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
