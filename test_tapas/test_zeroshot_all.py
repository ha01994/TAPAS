import csv
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


METRIC_FILES = [
    "results_model_quality_metrics_pos_best.csv",
    "results_model_quality_metrics_neg1_best.csv",
    "results_model_quality_metrics_neg2_best.csv",
]

PAE_FILES = [
    "pae_feat_af3_vdjdb.csv",
    "pae_feat_af3_vdjdb_neg_part1.csv",
    "pae_feat_af3_vdjdb_neg_part2.csv",
]

# Same AF3_COLS + PAE_COLS definitions as train_tapas_esm_conf_pae.py (lines 20–34).
AF3_COLS = [
    "iptm_tcrpmhc",
    "global_plddt",
    "cdr1_A",
    "cdr2_A",
    "cdr3_A",
    "cdr1_B",
    "cdr2_B",
    "cdr3_B",
    "iptm_mean",
    "pdockq",
]

PAE_COLS = [
    "pae_mean",
    "pae_max",
    "pae_std",
    "pae_median",
    "pae_p10",
    "pae_p90",
    "pae_frac_lt_5",
    "pae_frac_gt_15",
    "pae_asymmetry",
]

# PAE columns where lower is better: flip sign (-score) so AUC direction matches "higher = binder"
# (e.g. low pae_mean may indicate binding but raw AUC can fall below 0.5 without flipping)
PAE_INVERT_COLS = set(PAE_COLS) - {"pae_frac_lt_5"}

SCORE_COLS = [
    *AF3_COLS,
    "avgipae_average",
    "pdockq2_average",
    *PAE_COLS,
]


def load_merged_results():
    dfs = [pd.read_csv(f) for f in METRIC_FILES if os.path.exists(f)]
    if not dfs:
        raise FileNotFoundError(
            "No metric files found. Expected one of: " + ", ".join(METRIC_FILES)
        )
    df_metrics = pd.concat(dfs, ignore_index=True)

    pae_dfs = [pd.read_csv(f) for f in PAE_FILES if os.path.exists(f)]
    if not pae_dfs:
        raise FileNotFoundError(
            "No PAE files found. Expected one of: " + ", ".join(PAE_FILES)
        )
    df_pae = pd.concat(pae_dfs, ignore_index=True)

    # pdb_id (metrics) <-> sample_id (PAE)
    df = pd.merge(
        df_metrics,
        df_pae,
        left_on="pdb_id",
        right_on="sample_id",
        how="inner",
    )

    # derived averages
    if "avgipae_pmhc" in df.columns and "avgipae_tcr" in df.columns:
        df["avgipae_average"] = (df["avgipae_pmhc"] + df["avgipae_tcr"]) / 2.0
    if "pdockq2_pmhc" in df.columns and "pdockq2_tcr" in df.columns:
        df["pdockq2_average"] = (df["pdockq2_pmhc"] + df["pdockq2_tcr"]) / 2.0

    return df


def build_score_map(df_all, score_col: str, invert: bool = False):
    if score_col not in df_all.columns:
        return {}

    sub = df_all[["pdb_id", score_col]].copy()
    sub = sub.dropna(subset=[score_col])
    if sub.empty:
        return {}

    # Multiple rows per pdb_id: pick one "best" model score per id
    # Default: take max score
    # invert=True: take min raw score, then use -score so higher is better for AUC
    if invert:
        best = sub.groupby("pdb_id", as_index=False)[score_col].min()
        return dict(zip(best["pdb_id"], -best[score_col]))
    else:
        best = sub.groupby("pdb_id", as_index=False)[score_col].max()
        return dict(zip(best["pdb_id"], best[score_col]))


def macro_auc_max_fpr(y_true, y_score, pmhc, max_fpr=0.1):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pmhc = np.asarray(pmhc)
    peptide = np.array([str(x).split("_")[0] for x in pmhc], dtype=object)
    per_pep = []
    for p in np.unique(peptide):
        mask = peptide == p
        y_p = y_true[mask]
        if np.unique(y_p).size < 2:
            continue
        s_p = y_score[mask]
        try:
            per_pep.append(roc_auc_score(y_p, s_p, max_fpr=max_fpr))
        except ValueError:
            continue
    if not per_pep:
        return np.nan
    return float(np.mean(per_pep))


def evaluate_folds(folder_path: str, score_map: dict):
    fold_aucs_01 = []

    for i in range(5):
        test_file = os.path.join(folder_path, f"fold{i}_test.csv")
        if not os.path.exists(test_file):
            continue

        df_test = pd.read_csv(test_file)
        y_true = df_test["label"]
        y_scores = df_test["id"].map(score_map)
        pmhc = df_test["pmhc"]

        if y_scores.isnull().any():
            valid_mask = y_scores.notnull()
            y_true = y_true[valid_mask]
            y_scores = y_scores[valid_mask]
            pmhc = pmhc[valid_mask]

        auc01 = macro_auc_max_fpr(y_true, y_scores, pmhc, max_fpr=0.1)
        fold_aucs_01.append(auc01)

    avg_auc_01 = float(np.nanmean(fold_aucs_01)) if fold_aucs_01 else float("nan")
    return avg_auc_01


if __name__ == "__main__":
    df_all = load_merged_results()
    print("metric,RS,SS")
    for score_col in SCORE_COLS:
        invert = score_col in PAE_INVERT_COLS
        score_map = build_score_map(df_all, score_col, invert=invert)
        if not score_map:
            print(f"{score_col},nan,nan")
            continue

        avg_rs_auc01 = evaluate_folds("dataset_rs", score_map)
        avg_ss_auc01 = evaluate_folds("dataset_ss", score_map)

        print(f"{score_col},{avg_rs_auc01:.3f},{avg_ss_auc01:.3f}")


def _zeroshot_auc01_for_folder(
    df_all, score_col: str, invert: bool, folder_path: str
) -> float:
    score_map = build_score_map(df_all, score_col, invert=invert)
    if not score_map:
        return float("nan")
    return evaluate_folds(folder_path, score_map)


def _tapas_auc01_from_summary_csv(split_name: str) -> float:
    fname = (
        "rs_esm_conf_pae.csv"
        if split_name.upper() == "RS"
        else "ss_esm_conf_pae.csv"
    )
    path = os.path.join("results_auc", fname)
    if not os.path.isfile(path):
        return float("nan")
    df = pd.read_csv(path)
    if df.empty or "average" not in df.columns:
        return float("nan")
    key_col = df.columns[0]
    row = df[df[key_col].astype(str).str.strip() == "auc_0.1"]
    if row.empty:
        return float("nan")
    return float(row["average"].iloc[0])


def _print_tapas_and_zeroshot_summary_lines() -> None:
    print("-----------------------------------------")

    df_all = load_merged_results()
    # Same invert rules as the main SCORE_COLS loop
    zeroshot_specs = [
        ("Zero-shot iPAE confidence", "avgipae_average", False),
        ("Zero-shot pDockQ2", "pdockq2_average", False),
        ("Zero-shot ipTM", "iptm_tcrpmhc", False),
        ("Zero-shot pLDDT", "global_plddt", False),
    ]

    csv_rows: list[list[str]] = []
    csv_rows.append(["SUMMARY", ""])
    csv_rows.append(["", ""])

    print("SUMMARY")
    print()

    for i, (split_title, folder) in enumerate(
        (("RS", "dataset_rs"), ("SS", "dataset_ss"))
    ):
        if i:
            print()
            csv_rows.append(["", ""])
        print(f"{split_title},macro_auc_0.1")
        csv_rows.append([split_title, "macro_auc_0.1"])

        tv = _tapas_auc01_from_summary_csv(split_title)
        if np.isnan(tv):
            print("TAPAS,nan")
            csv_rows.append(["TAPAS", "nan"])
        else:
            print(f"TAPAS,{tv:.3f}")
            csv_rows.append(["TAPAS", f"{tv:.3f}"])

        for label, col, invert in zeroshot_specs:
            v = _zeroshot_auc01_for_folder(df_all, col, invert, folder)
            if np.isnan(v):
                print(f"{label},nan")
                csv_rows.append([label, "nan"])
            else:
                print(f"{label},{v:.3f}")
                csv_rows.append([label, f"{v:.3f}"])

    summary_path = os.path.join(os.getcwd(), "summary_vdjdb.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as fp:
        csv.writer(fp).writerows(csv_rows)
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    _print_tapas_and_zeroshot_summary_lines()

