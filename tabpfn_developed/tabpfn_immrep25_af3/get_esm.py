from pathlib import Path

import esm
import numpy as np
import pandas as pd
import torch
from anarci import anarci
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
IMMREP25_TSV = SCRIPT_DIR / "immrep25.tsv"
IMMREP25_PAIRS_CSV = SCRIPT_DIR / "immrep25_pairs.csv"
OUTPUT_FILE = SCRIPT_DIR / "esm_embeddings_map.npy"
MODEL_NAME = "esm2_t33_650M_UR50D"
DEVICE     = "cuda:0"
BATCH_SIZE = 16
TARGET_COLS = ['peptide', 'A1', 'A2', 'A3', 'B1', 'B2', 'B3']


def cdr_from_numbered(numbered, start, end):
    return "".join(
        aa
        for (position, _insertion), aa in numbered
        if aa not in {"-", ".", " "} and start <= position <= end
    )


def imgt_cdrs(sequence):
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


def number_unique_sequences(sequences, chain_name):
    unique_sequences = list(dict.fromkeys(sequences.astype(str)))
    print(f"Numbering {len(unique_sequences)} unique {chain_name} sequences with ANARCI/IMGT...")
    return {
        sequence: imgt_cdrs(sequence)
        for sequence in tqdm(unique_sequences, desc=f"ANARCI {chain_name}")
    }


def load_embedding_rows():
    """Build the former VDJdb-like ESM table directly from ImmRep25 inputs."""
    raw = pd.read_csv(IMMREP25_TSV, sep="\t")
    pairs = pd.read_csv(IMMREP25_PAIRS_CSV)
    raw_columns = {
        "tcra_trimmed", "tcrb_trimmed", "peptide", "label",
    }
    pair_columns = {"id", "pmhc", "label"}
    missing_raw = sorted(raw_columns - set(raw.columns))
    missing_pairs = sorted(pair_columns - set(pairs.columns))
    if missing_raw:
        raise ValueError(f"ImmRep25 TSV is missing columns: {missing_raw}")
    if missing_pairs:
        raise ValueError(f"ImmRep25 pairs CSV is missing columns: {missing_pairs}")
    if len(raw) != len(pairs):
        raise ValueError(
            f"ImmRep25 row mismatch: TSV={len(raw)}, pairs={len(pairs)}"
        )
    if pairs["id"].astype(str).duplicated().any():
        raise ValueError("ImmRep25 pairs contain duplicate IDs")

    pair_peptide = pairs["pmhc"].astype(str).str.split("_", n=1).str[0]
    if not raw["peptide"].astype(str).reset_index(drop=True).equals(
        pair_peptide.reset_index(drop=True)
    ):
        raise ValueError("ImmRep25 TSV and pairs peptide order differs")
    if not raw["label"].astype(int).reset_index(drop=True).equals(
        pairs["label"].astype(int).reset_index(drop=True)
    ):
        raise ValueError("ImmRep25 TSV and pairs labels differ")

    alpha_cdrs = number_unique_sequences(raw["tcra_trimmed"], "TRA")
    beta_cdrs = number_unique_sequences(raw["tcrb_trimmed"], "TRB")
    alpha = raw["tcra_trimmed"].astype(str).map(alpha_cdrs)
    beta = raw["tcrb_trimmed"].astype(str).map(beta_cdrs)

    return pd.DataFrame(
        {
            "id": pairs["id"].astype(str),
            "peptide": raw["peptide"],
            "A1": alpha.map(lambda cdrs: cdrs[0]),
            "A2": alpha.map(lambda cdrs: cdrs[1]),
            "A3": alpha.map(lambda cdrs: cdrs[2]),
            "B1": beta.map(lambda cdrs: cdrs[0]),
            "B2": beta.map(lambda cdrs: cdrs[1]),
            "B3": beta.map(lambda cdrs: cdrs[2]),
            "binder": pairs["label"].astype(int),
        }
    )

def generate_partwise_raw_embeddings():
    print(f"Loading data from {IMMREP25_TSV} and {IMMREP25_PAIRS_CSV}...")
    df = load_embedding_rows()

    # 1. unique sequence 추출
    unique_seqs = set()
    for col in TARGET_COLS:
        unique_seqs.update(df[col].dropna().unique())
    unique_seqs = list(unique_seqs)
    print(f"Total rows: {len(df)}, Unique sequences: {len(unique_seqs)}")

    # 2. ESM-2 로드
    print(f"\nLoading {MODEL_NAME}...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter  = alphabet.get_batch_converter()
    model.eval()
    model.to(DEVICE)

    # 3. raw 1280-dim embedding 계산
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

    # 4. 파트별 raw matrix 구성
    # 저장 형태: {id: {'peptide': (1280,), 'A1': (1280,), ..., 'A3': (1280,), ...}}
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
