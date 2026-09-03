#!/usr/bin/env python3
"""Generate ESM-2 embeddings for the eight-peptide ePytope-TCR viral set.

The output matches the ImmRep25 map schema:
    {pair_id: {peptide, A1, A2, A3, B1, B2, B3 -> float32[1280]}}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import esm
import numpy as np
import pandas as pd
import torch
from anarci import anarci
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "data" / "manifest.csv"
DEFAULT_TCR_SEQUENCES = SCRIPT_DIR / "data" / "tcr_sequences.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "esm_embeddings_map.npy"

MODEL_NAME = "esm2_t33_650M_UR50D"
EMBEDDING_DIM = 1280
TARGET_COLS = ["peptide", "A1", "A2", "A3", "B1", "B2", "B3"]
PAIR_SUFFIX = "_structure"


def require_columns(df: pd.DataFrame, path: Path, required: set[str]) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")


def cdr_from_numbered(numbered, start: int, end: int) -> str:
    return "".join(
        aa
        for (position, _insertion), aa in numbered
        if aa not in {"-", ".", " "} and start <= position <= end
    )


def imgt_cdrs(sequence: str) -> tuple[str, str, str]:
    results = anarci(
        [("query", sequence)],
        scheme="imgt",
        assign_germline=True,
        allowed_species=["human"],
        ncpu=1,
    )
    numbered_results = results[0]
    if not numbered_results or not numbered_results[0] or not numbered_results[0][0]:
        raise ValueError(f"ANARCI failed for sequence of length {len(sequence)}")
    numbered = numbered_results[0][0][0]
    return (
        cdr_from_numbered(numbered, 27, 38),
        cdr_from_numbered(numbered, 56, 65),
        cdr_from_numbered(numbered, 104, 118),
    )


def build_cdr_table(
    manifest_path: Path,
    tcr_sequences_path: Path,
    limit: int | None,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    tcr_sequences = pd.read_csv(tcr_sequences_path, dtype=str, keep_default_na=False)
    require_columns(
        manifest,
        manifest_path,
        {"job_name", "tcr_index", "label", "target_epitope", "cdr3_alpha", "cdr3_beta"},
    )
    require_columns(
        tcr_sequences,
        tcr_sequences_path,
        {"tcr_index", "tra_sequence", "trb_sequence"},
    )
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1")
        manifest = manifest.iloc[:limit].copy()

    if manifest["job_name"].duplicated().any():
        raise ValueError("Manifest contains duplicate job_name values")
    if not set(manifest["label"]).issubset({"0", "1"}):
        raise ValueError("Manifest label values must be 0 or 1")
    if tcr_sequences["tcr_index"].duplicated().any():
        raise ValueError("TCR sequence table contains duplicate tcr_index values")

    needed_tcrs = set(manifest["tcr_index"])
    tcr_sequences = tcr_sequences[tcr_sequences["tcr_index"].isin(needed_tcrs)]
    if set(tcr_sequences["tcr_index"]) != needed_tcrs:
        missing = sorted(needed_tcrs - set(tcr_sequences["tcr_index"]))
        raise ValueError(f"Missing TCR sequences for indices: {missing[:10]}")

    cdr_rows: list[dict[str, str]] = []
    print(f"Numbering {len(tcr_sequences)} unique TCRs with ANARCI/IMGT...", flush=True)
    for row in tqdm(tcr_sequences.itertuples(index=False), total=len(tcr_sequences), desc="ANARCI"):
        alpha = imgt_cdrs(row.tra_sequence)
        beta = imgt_cdrs(row.trb_sequence)
        cdr_rows.append(
            {
                "tcr_index": row.tcr_index,
                "A1": alpha[0],
                "A2": alpha[1],
                "anarci_A3": alpha[2],
                "B1": beta[0],
                "B2": beta[1],
                "anarci_B3": beta[2],
            }
        )

    merged = manifest.merge(pd.DataFrame(cdr_rows), on="tcr_index", how="left", validate="many_to_one")
    alpha_mismatch = merged["anarci_A3"] != merged["cdr3_alpha"]
    beta_mismatch = merged["anarci_B3"] != merged["cdr3_beta"]
    if alpha_mismatch.any() or beta_mismatch.any():
        bad = merged.loc[
            alpha_mismatch | beta_mismatch,
            ["job_name", "cdr3_alpha", "anarci_A3", "cdr3_beta", "anarci_B3"],
        ]
        raise ValueError(f"ANARCI/manifest CDR3 mismatch:\n{bad.head().to_string(index=False)}")

    ids = merged["job_name"].map(
        lambda value: value if value.endswith(PAIR_SUFFIX) else f"{value}{PAIR_SUFFIX}"
    )
    output = pd.DataFrame(
        {
            "id": ids,
            "peptide": merged["target_epitope"],
            "A1": merged["A1"],
            "A2": merged["A2"],
            "A3": merged["cdr3_alpha"],
            "B1": merged["B1"],
            "B2": merged["B2"],
            "B3": merged["cdr3_beta"],
            "binder": merged["label"].astype(int),
        }
    )
    empty_counts = {column: int((output[column] == "").sum()) for column in TARGET_COLS}
    if any(empty_counts.values()):
        raise ValueError(f"Empty ESM input sequences: {empty_counts}")
    return output


def embed_sequences(
    cdr_table: pd.DataFrame,
    output_path: Path,
    device: str,
    batch_size: int,
) -> None:
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")

    unique_sequences = sorted(
        {
            str(sequence)
            for column in TARGET_COLS
            for sequence in cdr_table[column]
        }
    )
    print(
        f"Rows: {len(cdr_table)}; unique sequences to embed: {len(unique_sequences)}",
        flush=True,
    )
    print(f"Loading {MODEL_NAME} on {device}...", flush=True)
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    model.to(device)

    sequence_embeddings: dict[str, np.ndarray] = {}
    items = [(str(index), sequence) for index, sequence in enumerate(unique_sequences)]
    with torch.no_grad():
        for start in tqdm(range(0, len(items), batch_size), desc="ESM-2"):
            batch = items[start : start + batch_size]
            _, _, tokens = batch_converter(batch)
            representations = model(
                tokens.to(device),
                repr_layers=[33],
                return_contacts=False,
            )["representations"][33]
            for index, (_label, sequence) in enumerate(batch):
                sequence_embeddings[sequence] = (
                    representations[index, 1 : len(sequence) + 1]
                    .mean(0)
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

    embedding_map: dict[str, dict[str, np.ndarray]] = {}
    print("Building pair-level embedding map...", flush=True)
    for row in tqdm(cdr_table.itertuples(index=False), total=len(cdr_table), desc="Pairs"):
        pair_id = str(row.id)
        embedding_map[pair_id] = {
            column: sequence_embeddings[str(getattr(row, column))].copy()
            for column in TARGET_COLS
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embedding_map, allow_pickle=True)
    print(f"Saved {len(embedding_map)} entries to {output_path}", flush=True)
    print(
        f"Each entry has {TARGET_COLS}; each value is float32[{EMBEDDING_DIM}].",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tcr-sequences", type=Path, default=DEFAULT_TCR_SEQUENCES)
    parser.add_argument(
        "--cdr-csv",
        type=Path,
        default=None,
        help="Optional path for saving the generated CDR table",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, help="Process only the first N pairs for testing")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build and validate the CDR table in memory without loading ESM-2",
    )
    args = parser.parse_args()

    cdr_table = build_cdr_table(args.manifest, args.tcr_sequences, args.limit)
    positives = int(cdr_table["binder"].sum())
    print(
        f"Pairs: {len(cdr_table)}; positive/negative: "
        f"{positives}/{len(cdr_table) - positives}",
        flush=True,
    )
    if args.cdr_csv is not None:
        args.cdr_csv.parent.mkdir(parents=True, exist_ok=True)
        cdr_table.to_csv(args.cdr_csv, index=False)
        print(f"Saved CDR table: {args.cdr_csv}", flush=True)
    if not args.prepare_only:
        embed_sequences(cdr_table, args.output, args.device, args.batch_size)


if __name__ == "__main__":
    main()
