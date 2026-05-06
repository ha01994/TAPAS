import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ── Path configuration ─────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent

# (AF_OUTPUT_DIR, SITES_TXT basename under SCRIPT_DIR, OUT_DIR)
RUNS = [
    (
        '/shared/ha01994/iptm_filtered_neg_notdone_0_output',
        'sites_vdjdbiedb_neg_0.txt',
        'pae_matrix_vdjdbiedb_neg_0',
    ),
    (
        '/shared/ha01994/iptm_filtered_neg_notdone_1_output',
        'sites_vdjdbiedb_neg_1.txt',
        'pae_matrix_vdjdbiedb_neg_1',
    ),
    (
        '/shared/ha01994/iptm_filtered_neg_notdone_2_output',
        'sites_vdjdbiedb_neg_2.txt',
        'pae_matrix_vdjdbiedb_neg_2',
    ),
]


# ── 1. Chain offset computation function ───────────────────────────────────────
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


# ── 2. Parse sites file ───────────────────────────────────────────────────────
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


# ── 3. Load PAE matrix ───────────────────────────────────────────────────────
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


# ── 4. Extract interface PAE submatrix ─────────────────────────
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


# ── 5. CIF path resolver ─────────────────────────────────────────────────────
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


# ── Main ─────────────────────────────────────────────────────────────────────
def run_one(af_output_dir, sites_txt, out_dir, best_csv_path):
    best_csv_path = os.fspath(best_csv_path)
    if not os.path.isfile(best_csv_path):
        print(f"[SKIP] best CSV not found: {best_csv_path}")
        return

    sites_path = SCRIPT_DIR / sites_txt
    if not sites_path.is_file():
        print(f"[SKIP] sites file not found: {sites_path}")
        return

    os.makedirs(out_dir, exist_ok=True)

    df_best = pd.read_csv(best_csv_path)
    sites = parse_sites(str(sites_path))

    success, skipped, errors = 0, 0, []

    for _, row in df_best.iterrows():
        pdb_id = str(row['pdb_id'])
        model_number = int(row['model_number'])

        if pdb_id not in sites:
            print(f"[SKIP] {pdb_id}: missing sites info")
            skipped += 1
            continue

        try:
            cif_path = get_cif_path(pdb_id, model_number, af_output_dir)
            offsets, chain_counts = get_chain_offsets(cif_path)

            pae = load_pae(pdb_id, model_number, af_output_dir)

            total_res = sum(chain_counts.values())
            if pae.shape != (total_res, total_res):
                raise ValueError(
                    f"PAE shape {pae.shape} != expected ({total_res},{total_res})"
                )

            submatrix, labels = extract_interface_pae(
                pae, offsets, sites[pdb_id]
            )

            out_path = os.path.join(out_dir, f'{pdb_id}_interface_pae.npy')
            np.save(out_path, submatrix)

            label_path = os.path.join(out_dir, f'{pdb_id}_interface_labels.txt')
            with open(label_path, 'w') as f:
                f.write('\n'.join(labels))

            print(f"[OK] {pdb_id} model={model_number} | "
                  f"interface={submatrix.shape[0]} residues | "
                  f"PAE mean={submatrix.mean():.3f}")
            success += 1

        except Exception as e:
            print(f"[ERR] {pdb_id}: {e}")
            errors.append((pdb_id, str(e)))

    print(f"\n[{out_dir}] done: {success}succeeded | {skipped}skipped | "
          f"{len(errors)}errors")
    if errors:
        print("Error list:")
        for pid, msg in errors:
            print(f"  {pid}: {msg}")


def main():
    for run_idx, (af_output_dir, sites_txt, out_dir) in enumerate(RUNS):
        best_csv = SCRIPT_DIR / f'results_model_quality_metrics_{run_idx}_best.csv'
        print(f"\n{'=' * 60}\n{out_dir}\nAF: {af_output_dir}\nsites: {sites_txt}\n"
              f"best CSV: {best_csv.name}\n"
              f"{'=' * 60}")
        run_one(af_output_dir, sites_txt, out_dir, best_csv)


if __name__ == '__main__':
    main()
