"""Evaluate all best-ranking-score AF3 confidence metrics on VDJdb folds."""

from __future__ import annotations

import csv
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVELOPED_DIR = os.path.dirname(SCRIPT_DIR)
DATASET_ROOT = SCRIPT_DIR
REPO_ROOT = os.path.dirname(DEVELOPED_DIR)
METRIC_FILE = os.path.join(
    REPO_ROOT, "af3_confidence", "vdjdb", "model_quality_metrics_best_af3_ranking_score.csv"
)
SUMMARY_CSV = os.path.join(SCRIPT_DIR, "summary_vdjdb.csv")

SCORE_COLS = [
    "iptm_tcrpmhc",
    "iptm_mean",
    "global_plddt",
    "pdockq",
    "pdockq2_average",
    "avgipae_average",
]


def peptide_from_pmhc(pmhc: str) -> str:
    value = str(pmhc)
    if "_HLA-" in value:
        return value.split("_HLA-", 1)[0]
    return value.split("_", 1)[0]


def load_metrics() -> pd.DataFrame:
    if not os.path.isfile(METRIC_FILE):
        raise FileNotFoundError(f"Missing metrics file: {METRIC_FILE}")
    df = pd.read_csv(METRIC_FILE)
    required = {"pdb_id", "avgipae_pmhc", "avgipae_tcr", "pdockq2_pmhc", "pdockq2_tcr"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Metrics file is missing required columns: {missing}")
    df["pdb_id"] = df["pdb_id"].astype(str)
    if df["pdb_id"].duplicated().any():
        raise ValueError("Best-ranking-score metrics contain duplicated pdb_id rows")
    df["avgipae_average"] = (df["avgipae_pmhc"] + df["avgipae_tcr"]) / 2.0
    df["pdockq2_average"] = (df["pdockq2_pmhc"] + df["pdockq2_tcr"]) / 2.0
    return df


def build_score_map(df: pd.DataFrame, score_col: str) -> dict[str, float]:
    if score_col not in df.columns:
        return {}
    sub = df[["pdb_id", score_col]].dropna(subset=[score_col])
    return dict(zip(sub["pdb_id"], sub[score_col]))


def macro_auc(y_true, y_score, pmhc, max_fpr=None) -> float:
    """Mean peptide-level ROC-AUC over peptides containing both classes."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    peptide = np.asarray([peptide_from_pmhc(value) for value in pmhc])
    per_peptide = []
    for pep in np.unique(peptide):
        mask = peptide == pep
        if np.unique(y_true[mask]).size < 2:
            continue
        try:
            per_peptide.append(
                roc_auc_score(y_true[mask], y_score[mask], max_fpr=max_fpr)
            )
        except ValueError:
            continue
    return float(np.mean(per_peptide)) if per_peptide else float("nan")


def evaluate_folds(folder_name: str, score_map: dict[str, float]) -> tuple[float, float]:
    fold_aucs = []
    fold_aucs_01 = []
    for fold_idx in range(5):
        test_file = os.path.join(DATASET_ROOT, folder_name, f"fold{fold_idx}_test.csv")
        if not os.path.isfile(test_file):
            continue
        df_test = pd.read_csv(test_file)
        scores = df_test["id"].astype(str).map(score_map)
        valid = scores.notna()
        fold_aucs.append(
            macro_auc(
                df_test.loc[valid, "label"],
                scores.loc[valid],
                df_test.loc[valid, "pmhc"],
            )
        )
        fold_aucs_01.append(
            macro_auc(
                df_test.loc[valid, "label"],
                scores.loc[valid],
                df_test.loc[valid, "pmhc"],
                max_fpr=0.1,
            )
        )
    mean_auc = float(np.nanmean(fold_aucs)) if fold_aucs else float("nan")
    mean_auc_01 = float(np.nanmean(fold_aucs_01)) if fold_aucs_01 else float("nan")
    return mean_auc, mean_auc_01


def fmt(value: float) -> str:
    return "nan" if np.isnan(value) else f"{value:.3f}"


def main() -> None:
    metrics = load_metrics()
    print("metric,RS_macro_auc,RS_macro_auc_0.1,SS_macro_auc,SS_macro_auc_0.1")
    for score_col in SCORE_COLS:
        score_map = build_score_map(metrics, score_col)
        rs_auc, rs_auc_01 = evaluate_folds("dataset_rs", score_map)
        ss_auc, ss_auc_01 = evaluate_folds("dataset_ss", score_map)
        print(
            f"{score_col},{fmt(rs_auc)},{fmt(rs_auc_01)},"
            f"{fmt(ss_auc)},{fmt(ss_auc_01)}"
        )

    comparison_cols = [
        ("ipae", "avgipae_average"),
        ("pdockq2", "pdockq2_average"),
        ("iptm", "iptm_tcrpmhc"),
        ("plddt", "global_plddt"),
    ]
    comparison = {}
    for label, score_col in comparison_cols:
        score_map = build_score_map(metrics, score_col)
        comparison[label] = {
            "RS": evaluate_folds("dataset_rs", score_map),
            "SS": evaluate_folds("dataset_ss", score_map),
        }

    rows = [["RS_Macro_AUC_0.1", "SS_Macro_AUC_0.1"]]
    print("\n-----------------------------------------")
    print("RS_Macro_AUC_0.1,SS_Macro_AUC_0.1")
    for label, _ in comparison_cols:
        rs_auc_01 = comparison[label]["RS"][1]
        ss_auc_01 = comparison[label]["SS"][1]
        print(f"{label},{fmt(rs_auc_01)},{fmt(ss_auc_01)}")
        rows.append([label, fmt(rs_auc_01), fmt(ss_auc_01)])

    print("\nRS_Macro_AUC,SS_Macro_AUC")
    rows.extend([["", ""], ["RS_Macro_AUC", "SS_Macro_AUC"]])
    for label, _ in comparison_cols:
        rs_auc = comparison[label]["RS"][0]
        ss_auc = comparison[label]["SS"][0]
        print(f"{label},{fmt(rs_auc)},{fmt(ss_auc)}")
        rows.append([label, fmt(rs_auc), fmt(ss_auc)])

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    print(f"\nSaved {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
