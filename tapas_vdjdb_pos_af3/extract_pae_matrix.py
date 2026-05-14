import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# --- Path configuration ---
AF_OUTPUT_DIR  = '/home/ha01994/af3_structpred/af3_vdjdb/af_output'
BEST_CSV       = 'results_model_quality_metrics_best.csv'
SITES_TXT      = 'sites_af3_vdjdb.txt'
OUT_DIR        = 'pae_matrix_af3_vdjdb'

os.makedirs(OUT_DIR, exist_ok=True)


# --- 1. Chain residue offsets from CIF ---
def get_chain_offsets(cif_path):
    chain_order = []
    chain_counts = {}
    seen = set()

    with open(cif_path) as f:
        for line in f:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            parts = line.split()
            try:
                chain  = parts[6]   # label_asym_id
                resnum = parts[8]   # label_seq_id
                key    = (chain, resnum)
                if key not in seen:
                    seen.add(key)
                    if chain not in chain_counts:
                        chain_order.append(chain)
                        chain_counts[chain] = 0
                    chain_counts[chain] += 1
            except Exception:
                pass

    offsets = {}
    cumsum = 0
    for ch in chain_order:
        offsets[ch] = cumsum
        cumsum += chain_counts[ch]

    return offsets, chain_counts


# --- 2. Parse sites file (interface residue lists) ---
def parse_sites(sites_path):
    sites = defaultdict(dict)
    with open(sites_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            pdb_id = parts[0]
            chain  = parts[1]
            resnums = [int(x) for x in parts[2:]]
            sites[pdb_id][chain] = resnums
    return dict(sites)


# --- 3. Load full PAE matrix from AF3 JSON ---
def load_pae(pdb_id, model_number, af_output_dir):
    sample_dir = os.path.join(
        af_output_dir, pdb_id,
        f'seed-1_sample-{model_number}'
    )
    json_path = os.path.join(
        sample_dir,
        f'{pdb_id}_seed-1_sample-{model_number}_confidences.json'
    )
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"PAE file not found: {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    pae = np.array(data['pae'], dtype=np.float32)
    return pae


# --- 4. Slice interface PAE submatrix ---
def extract_interface_pae(pae, offsets, site_chains):
    indices = []
    labels  = []

    for chain in ['E', 'A', 'B']:
        if chain not in site_chains:
            continue
        offset = offsets.get(chain)
        if offset is None:
            raise KeyError(f"Chain {chain} not found in CIF offsets")
        for resnum in sorted(site_chains[chain]):
            idx = offset + (resnum - 1)  # 0-based
            indices.append(idx)
            labels.append(f'{chain}{resnum}')

    indices = np.array(indices)
    submatrix = pae[np.ix_(indices, indices)]
    return submatrix, labels


# --- 5. Resolve CIF path for a sample ---
def get_cif_path(pdb_id, model_number, af_output_dir):
    sample_cif = os.path.join(
        af_output_dir, pdb_id,
        f'seed-1_sample-{model_number}',
        f'{pdb_id}_seed-1_sample-{model_number}_model.cif'
    )
    if os.path.exists(sample_cif):
        return sample_cif

    top_cif = os.path.join(af_output_dir, pdb_id, f'{pdb_id}_model.cif')
    if os.path.exists(top_cif):
        return top_cif

    raise FileNotFoundError(f"CIF not found for {pdb_id} model {model_number}")


# --- Main ---
def main():
    # Load inputs
    df_best = pd.read_csv(BEST_CSV)
    sites   = parse_sites(SITES_TXT)

    success, skipped, errors = 0, 0, []

    for _, row in df_best.iterrows():
        pdb_id       = str(row['pdb_id'])
        model_number = int(row['model_number'])

        if pdb_id not in sites:
            print(f"[SKIP] {pdb_id}: no sites entry")
            skipped += 1
            continue

        try:
            # Chain offsets from CIF atom records
            cif_path = get_cif_path(pdb_id, model_number, AF_OUTPUT_DIR)
            offsets, chain_counts = get_chain_offsets(cif_path)

            # Load PAE
            pae = load_pae(pdb_id, model_number, AF_OUTPUT_DIR)

            # Check PAE size matches total residue count
            total_res = sum(chain_counts.values())
            if pae.shape != (total_res, total_res):
                raise ValueError(
                    f"PAE shape {pae.shape} != expected ({total_res},{total_res})"
                )

            # Extract interface submatrix
            submatrix, labels = extract_interface_pae(
                pae, offsets, sites[pdb_id]
            )

            # Save .npy
            out_path = os.path.join(OUT_DIR, f'{pdb_id}_interface_pae.npy')
            np.save(out_path, submatrix)

            # Save row/column labels
            label_path = os.path.join(OUT_DIR, f'{pdb_id}_interface_labels.txt')
            with open(label_path, 'w') as f:
                f.write('\n'.join(labels))

            print(f"[OK] {pdb_id} model={model_number} | "
                  f"interface={submatrix.shape[0]} residues | "
                  f"PAE mean={submatrix.mean():.3f}")
            success += 1

        except Exception as e:
            print(f"[ERR] {pdb_id}: {e}")
            errors.append((pdb_id, str(e)))

    print(f"\nDone: {success} ok | {skipped} skipped | {len(errors)} errors")
    if errors:
        print("Errors:")
        for pid, msg in errors:
            print(f"  {pid}: {msg}")


if __name__ == '__main__':
    main()
