"""Evaluate all best-ranking-score AF3 confidence metrics on ImmRep25."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
METRIC_FILE = os.path.join(
    REPO_ROOT, "af3_confidence", "immrep25", "model_quality_metrics_best_af3_ranking_score.csv"
)
IMMREP25_PAIRS_CSV = os.path.join(SCRIPT_DIR, "immrep25_pairs.csv")
SUMMARY_CSV = os.path.join(SCRIPT_DIR, "summary_immrep25.csv")

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


def load_labels() -> pd.DataFrame:
    df = pd.read_csv(IMMREP25_PAIRS_CSV)
    df["peptide"] = df["pmhc"].map(peptide_from_pmhc)
    df["pdb_id"] = df["id"].astype(str)
    return df


def macro_auc(y_true, y_score, peptide, max_fpr=None) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    peptide = np.asarray(peptide)
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


def evaluate(df: pd.DataFrame, score_col: str) -> tuple[float, float]:
    if df.empty or score_col not in df.columns:
        return float("nan"), float("nan")
    sub = df[[score_col, "label", "peptide"]].dropna(subset=[score_col])
    if sub.empty or sub["label"].nunique() < 2:
        return float("nan"), float("nan")
    auc = macro_auc(sub["label"], sub[score_col], sub["peptide"])
    auc_01 = macro_auc(
        sub["label"], sub[score_col], sub["peptide"], max_fpr=0.1
    )
    return auc, auc_01


def fmt(value: float) -> str:
    return "nan" if np.isnan(value) else f"{value:.3f}"


def main() -> None:
    labels = load_labels()
    metrics = load_metrics()
    merged = pd.merge(labels, metrics, on="pdb_id", how="inner", validate="one_to_one")
    if len(merged) != len(labels):
        raise ValueError(f"Metrics merge mismatch: {len(labels)} labels vs {len(merged)} rows")

    print(f"ImmRep25 merged rows: {len(merged)}")
    print("metric,macro_auc,macro_auc_0.1")
    for score_col in SCORE_COLS:
        auc, auc_01 = evaluate(merged, score_col)
        print(f"{score_col},{fmt(auc)},{fmt(auc_01)}")

    comparison_cols = [
        ("ipae", "avgipae_average"),
        ("pdockq2", "pdockq2_average"),
        ("iptm", "iptm_tcrpmhc"),
        ("plddt", "global_plddt"),
    ]
    comparison = {
        label: evaluate(merged, score_col)
        for label, score_col in comparison_cols
    }
    lines = ["Macro_AUC_0.1"]
    for label, _ in comparison_cols:
        lines.append(f"{label},{fmt(comparison[label][1])}")
    lines.extend(["", "Macro_AUC"])
    for label, _ in comparison_cols:
        lines.append(f"{label},{fmt(comparison[label][0])}")

    print("\n-----------------------------------------")
    print("\n".join(lines))
    with open(SUMMARY_CSV, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"\nSaved {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
