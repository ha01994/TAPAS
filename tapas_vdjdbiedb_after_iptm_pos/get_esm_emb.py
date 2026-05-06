import pandas as pd
import numpy as np
import esm
import torch
from tqdm import tqdm

CSV_PATH   = 'vdjdbiedb_filtered.csv'
OUTPUT_FILE = 'esm_embeddings_map_vdjdbiedb.npy'
MODEL_NAME = "esm2_t33_650M_UR50D"
DEVICE     = "cuda:5"
BATCH_SIZE = 16

def generate_partwise_raw_embeddings():
    print(f"Loading data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)

    target_cols = ['peptide', 'A1', 'A2', 'A3', 'B1', 'B2', 'B3']

    # 1. Extract unique sequences
    unique_seqs = set()
    for col in target_cols:
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

    # 4. Build per-part raw matrices
    # Save format: {id: {'peptide': (1280,), 'A1': (1280,), ..., 'A3': (1280,), ...}}
    print("\nOrganizing per-part raw embeddings...")
    complex_ids = df['id'].astype(str).tolist()

    id_to_raw = {}
    for _, row in tqdm(df.iterrows(), total=len(df)):
        pid = str(row['id'])
        id_to_raw[pid] = {}
        for col in target_cols:
            seq = row[col]
            if pd.isna(seq):
                id_to_raw[pid][col] = np.zeros(1280, dtype=np.float32)
            else:
                id_to_raw[pid][col] = seq_to_embedding[seq].astype(np.float32)

    print(f"Saving raw embeddings to {OUTPUT_FILE}...")
    np.save(OUTPUT_FILE, id_to_raw, allow_pickle=True)
    print(f"Done! {len(id_to_raw)} entries saved.")
    print(f"Each entry: dict with keys {target_cols}, each value shape (1280,)")

if __name__ == "__main__":
    generate_partwise_raw_embeddings()