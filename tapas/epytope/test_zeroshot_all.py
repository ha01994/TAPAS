"""Evaluate all best-ranking AF3 confidence scores on ePytope-TCR viral."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
MANIFEST_CSV = SCRIPT_DIR / "data" / "manifest.csv"
CONFIDENCE_CSV = (
    REPO_ROOT / "af3_confidence" / "epytope_tcr_viral"
    / "model_quality_metrics_best_af3_ranking_score.csv"
)
RESULTS_DIR = SCRIPT_DIR / "results_auc"
SUMMARY_CSV = RESULTS_DIR / "epytope_tcr_viral_zeroshot_all.csv"

EXPECTED_ROWS = 3560
EXPECTED_TCRS = 445
EXPECTED_PEPTIDES = 8

SCORES = [
    ("avgipae_average", "iPAE_conf"),
    ("pdockq2_average", "pDockQ2"),
    ("iptm_tcrpmhc", "ipTM_TCR_pMHC"),
    ("global_plddt", "pLDDT"),
]


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required input is not ready: {path}")


def load_data() -> pd.DataFrame:
    require_file(MANIFEST_CSV)
    require_file(CONFIDENCE_CSV)

    manifest = pd.read_csv(MANIFEST_CSV)
    required_manifest = {"job_name", "tcr_index", "label", "target_epitope"}
    missing_manifest = sorted(required_manifest - set(manifest.columns))
    if missing_manifest:
        raise ValueError(f"Viral manifest is missing columns: {missing_manifest}")
    manifest = manifest.copy()
    manifest["pdb_id"] = manifest["job_name"].astype(str) + "_structure"
    manifest["tcr_id"] = "viral_tcr_" + manifest["tcr_index"].astype(str)
    manifest["peptide"] = manifest["target_epitope"].astype(str)
    manifest["label"] = pd.to_numeric(manifest["label"], errors="raise").astype(int)

    observed = (
        len(manifest),
        manifest["tcr_id"].nunique(),
        manifest["peptide"].nunique(),
    )
    expected = (EXPECTED_ROWS, EXPECTED_TCRS, EXPECTED_PEPTIDES)
    if observed != expected:
        raise ValueError(
            f"Unexpected active viral dimensions {observed}; expected {expected}"
        )
    if manifest["pdb_id"].duplicated().any():
        raise ValueError("Viral manifest produces duplicated structure IDs")
    if set(manifest["label"].unique()) != {0, 1}:
        raise ValueError("Viral labels must contain only 0 and 1")
    pair_counts = manifest.groupby("tcr_id")["peptide"].nunique()
    positive_counts = manifest.groupby("tcr_id")["label"].sum()
    if not (pair_counts == EXPECTED_PEPTIDES).all():
        raise ValueError("Each viral TCR must have all eight target peptides")
    if not (positive_counts == 1).all():
        raise ValueError("Each viral TCR must have exactly one positive pair")

    confidence = pd.read_csv(CONFIDENCE_CSV)
    required_scores = {
        "pdb_id",
        "avgipae_pmhc",
        "avgipae_tcr",
        "pdockq2_pmhc",
        "pdockq2_tcr",
        "iptm_tcrpmhc",
        "global_plddt",
    }
    missing_scores = sorted(required_scores - set(confidence.columns))
    if missing_scores:
        raise ValueError(f"Confidence CSV is missing columns: {missing_scores}")
    confidence["pdb_id"] = confidence["pdb_id"].astype(str)
    if confidence["pdb_id"].duplicated().any():
        raise ValueError("Confidence CSV contains duplicated pdb_id values")

    missing_ids = sorted(set(manifest["pdb_id"]) - set(confidence["pdb_id"]))
    if missing_ids:
        raise ValueError(
            f"Confidence extraction is incomplete: {len(missing_ids)} of "
            f"{EXPECTED_ROWS} active IDs are missing (examples: {missing_ids[:3]})"
        )

    data = manifest.merge(confidence, on="pdb_id", how="left", validate="one_to_one")
    data["avgipae_average"] = (
        pd.to_numeric(data["avgipae_pmhc"], errors="coerce")
        + pd.to_numeric(data["avgipae_tcr"], errors="coerce")
    ) / 2.0
    data["pdockq2_average"] = (
        pd.to_numeric(data["pdockq2_pmhc"], errors="coerce")
        + pd.to_numeric(data["pdockq2_tcr"], errors="coerce")
    ) / 2.0
    return data


def macro_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    peptides: np.ndarray,
    max_fpr: Optional[float] = None,
) -> float:
    values = []
    for peptide in pd.unique(peptides):
        mask = peptides == peptide
        if np.unique(labels[mask]).size == 2:
            values.append(roc_auc_score(labels[mask], scores[mask], max_fpr=max_fpr))
    return float(np.mean(values)) if values else float("nan")


def evaluate(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for score_column, method in SCORES:
        frame = data[["label", "peptide", score_column]].copy()
        frame[score_column] = pd.to_numeric(frame[score_column], errors="coerce")
        frame = frame.dropna()
        if len(frame) != len(data):
            raise ValueError(
                f"{score_column} has {len(data) - len(frame)} missing/non-numeric scores"
            )
        labels = frame["label"].to_numpy(dtype=int)
        scores = frame[score_column].to_numpy(dtype=float)
        peptides = frame["peptide"].to_numpy(dtype=str)
        rows.append(
            {
                "method": method,
                "score_column": score_column,
                "n": len(frame),
                "positive": int(labels.sum()),
                "negative": int(len(labels) - labels.sum()),
                "n_peptides": int(frame["peptide"].nunique()),
                "macro_auc": macro_auc(labels, scores, peptides),
                "macro_auc_0.1": macro_auc(labels, scores, peptides, max_fpr=0.1),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    summary = evaluate(data)
    summary.to_csv(SUMMARY_CSV, index=False, float_format="%.4f")
    print("ePytope-TCR viral AF3 zero-shot results (raw best-ranking scores)")
    print("method,macro_auc,macro_auc_0.1")
    for _, row in summary.iterrows():
        print(
            f"{row['method']},{row['macro_auc']:.4f},"
            f"{row['macro_auc_0.1']:.4f}"
        )
    print(f"\nSaved: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
