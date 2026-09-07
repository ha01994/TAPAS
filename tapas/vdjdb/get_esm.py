from pathlib import Path

import pandas as pd
import numpy as np
import esm
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
POSITIVE_CSV = DATA_DIR / "parsed_data_final.csv"
NEGATIVE_CSV = DATA_DIR / "negatives.csv"
TCR_LOOKUP_CSV = DATA_DIR / "dic_full_vavb.csv"
OUTPUT_FILE = SCRIPT_DIR / "esm_embeddings_map_vdjdb.npy"
MODEL_NAME = "esm2_t33_650M_UR50D"
DEVICE     = "cuda:0"
BATCH_SIZE = 16
TARGET_COLS = ['peptide', 'A1', 'A2', 'A3', 'B1', 'B2', 'B3']


def load_embedding_rows():
    """Build the ESM input rows directly from the VDJdb source tables."""
    positive = pd.read_csv(POSITIVE_CSV)
    negative = pd.read_csv(NEGATIVE_CSV)
    pairs = pd.concat([positive, negative], ignore_index=True)

    required = {"id", "pmhc", "tcr", "label"}
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"VDJdb pair tables are missing columns: {missing}")
    if pairs["id"].astype(str).duplicated().any():
        raise ValueError("VDJdb pair IDs must be unique")

    lookup = pd.read_csv(
        TCR_LOOKUP_CSV, header=None, names=["tcr", "components"], dtype=str
    )
    if lookup["tcr"].duplicated().any():
        raise ValueError("VDJdb TCR lookup contains duplicate TCR IDs")
    parts = lookup["components"].str.split("_", expand=True)
    if parts.shape[1] < 8:
        raise ValueError("VDJdb TCR lookup does not contain the expected CDR fields")
    cdr_lookup = lookup[["tcr"]].join(parts.iloc[:, 2:8].set_axis(
        ["A1", "A2", "A3", "B1", "B2", "B3"], axis=1
    ))

    rows = pairs.merge(cdr_lookup, on="tcr", how="left", validate="many_to_one")
    missing_tcrs = rows.loc[rows["A1"].isna(), "tcr"].drop_duplicates().tolist()
    if missing_tcrs:
        raise ValueError(f"Missing VDJdb TCR lookup entries: {missing_tcrs[:5]}")

    rows["peptide"] = rows["pmhc"].astype(str).str.split("_", n=1).str[0]
    rows["binder"] = rows["label"].astype(int)
    return (
        rows[["id", *TARGET_COLS, "binder"]]
        .sort_values("id")
        .reset_index(drop=True)
    )


def generate_partwise_raw_embeddings():
    print(f"Loading data from {POSITIVE_CSV}, {NEGATIVE_CSV}, and {TCR_LOOKUP_CSV}...")
    df = load_embedding_rows()

    # 1. Collect unique sequences across target columns
    unique_seqs = set()
    for col in TARGET_COLS:
        unique_seqs.update(df[col].dropna().unique())
    unique_seqs = list(unique_seqs)
    print(f"Total rows: {len(df)}, Unique sequences: {len(unique_seqs)}")

    # 2. Load ESM-2
    print(f"\nLoading {MODEL_NAME}...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter  = alphabet.get_batch_converter()
    model.eval()
    model.to(DEVICE)

    # 3. Compute raw 1280-dim embeddings
    seq_to_embedding = {}
    data_for_esm = [(str(i), seq) for i, seq in enumerate(unique_seqs)]

    print("Computing raw embeddings...")
    with torch.no_grad():
        for i in tqdm(range(0, len(data_for_esm), BATCH_SIZE)):
            batch = data_for_esm[i:i+BATCH_SIZE]
            labels, strs, tokens = batch_converter(batch)
            tokens = tokens.to(DEVICE)
            results = model(tokens, repr_layers=[33], return_contacts=False)
            token_repr = results["representations"][33]

            for j, (label, seq_str) in enumerate(batch):
                seq_len = len(seq_str)
                embedding = token_repr[j, 1:seq_len+1].mean(0).cpu().numpy()
                seq_to_embedding[seq_str] = embedding

    # 4. Build per-part raw embedding dict per complex id
    # Saved shape: {id: {'peptide': (1280,), 'A1': (1280,), ..., 'B3': (1280,)}}
    print("\nOrganizing per-part raw embeddings...")
    complex_ids = df['id'].astype(str).tolist()

    id_to_raw = {}
    for _, row in tqdm(df.iterrows(), total=len(df)):
        pid = str(row['id'])
        id_to_raw[pid] = {}
        for col in TARGET_COLS:
            seq = row[col]
            if pd.isna(seq):
                id_to_raw[pid][col] = np.zeros(1280, dtype=np.float32)
            else:
                id_to_raw[pid][col] = seq_to_embedding[seq].astype(np.float32)

    print(f"Saving raw embeddings to {OUTPUT_FILE}...")
    np.save(OUTPUT_FILE, id_to_raw, allow_pickle=True)
    print(f"Done! {len(id_to_raw)} entries saved.")
    print(f"Each entry: dict with keys {TARGET_COLS}, each value shape (1280,)")

if __name__ == "__main__":
    generate_partwise_raw_embeddings()
