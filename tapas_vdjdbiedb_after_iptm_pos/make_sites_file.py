"""
make_sites_file_v2.py

results_model_quality_metrics_best.csv를 읽어서
각 pdb_id의 best model_number에 해당하는 CIF 파일을 열고
TCR CDR + peptide interface residues를 추출합니다.

디렉토리 구조:
  <base_dir>/
    <pdb_id>/
      seed-1_sample-<model_number>/
        <pdb_id>_model.cif   (또는 여러 후보 패턴)

Usage:
  python make_sites_file_v2.py \
      --base_dir /path/to/af3_structpred \
      --csv results_model_quality_metrics_best.csv \
      --output sites_all.txt \
      --cutoff 8.0 \
      --n_jobs 16
"""

import os
import csv
import glob
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed


# ─────────────────────────────────────────────
# CIF parser
# ─────────────────────────────────────────────

def parse_cif_atoms(cif_path):
    atoms = []
    headers = []
    in_atom_block = False

    with open(cif_path) as f:
        for line in f:
            line = line.rstrip()
            if line == 'loop_':
                in_atom_block = False
                headers = []
                continue
            if line.startswith('_atom_site.'):
                in_atom_block = True
                headers.append(line.strip())
                continue
            if not in_atom_block or not headers:
                continue
            if line.startswith('_') or line.startswith('#') or not line.strip():
                in_atom_block = False
                headers = []
                continue

            vals = line.split()
            if len(vals) < len(headers):
                continue

            def get(name):
                try:
                    i = headers.index(name)
                    return vals[i]
                except ValueError:
                    return None

            atom_name = get('_atom_site.label_atom_id')
            if atom_name not in ('CA', 'CB', 'N', 'O', 'C'):
                continue

            try:
                atoms.append({
                    'chain':   get('_atom_site.auth_asym_id'),
                    'resnum':  int(get('_atom_site.auth_seq_id')),
                    'resname': get('_atom_site.label_comp_id'),
                    'atom':    atom_name,
                    'x':       float(get('_atom_site.Cartn_x')),
                    'y':       float(get('_atom_site.Cartn_y')),
                    'z':       float(get('_atom_site.Cartn_z')),
                })
            except (TypeError, ValueError):
                continue

    return atoms


# ─────────────────────────────────────────────
# Interface detection
# ─────────────────────────────────────────────

def get_interface_residues(atoms, query_chains, neighbor_chains, cutoff=8.0):
    cutoff2 = cutoff ** 2
    nb_atoms = [(a['x'], a['y'], a['z'])
                for a in atoms if a['chain'] in neighbor_chains]

    interface = defaultdict(set)
    for a in atoms:
        if a['chain'] not in query_chains:
            continue
        ax, ay, az = a['x'], a['y'], a['z']
        for (nx, ny, nz) in nb_atoms:
            dx = ax - nx; dy = ay - ny; dz = az - nz
            if dx*dx + dy*dy + dz*dz <= cutoff2:
                interface[a['chain']].add(a['resnum'])
                break

    return {c: sorted(resnums) for c, resnums in interface.items()}


# ─────────────────────────────────────────────
# CIF 파일 경로 탐색
# ─────────────────────────────────────────────

def find_cif(base_dir, pdb_id, model_number):
    """
    여러 가능한 경로 패턴을 시도해서 CIF 파일 반환.
    못 찾으면 None.
    """
    sample_dir = os.path.join(base_dir, pdb_id, f"seed-1_sample-{model_number}")

    candidates = [
        os.path.join(sample_dir, f"{pdb_id}_model.cif"),
        os.path.join(sample_dir, f"v0_model.cif"),          # 혹시 고정명
        os.path.join(sample_dir, "model.cif"),
    ]
    # glob으로 *.cif도 시도
    glob_hits = glob.glob(os.path.join(sample_dir, "*.cif"))

    for path in candidates:
        if os.path.exists(path):
            return path

    if glob_hits:
        return glob_hits[0]   # 첫 번째 CIF

    return None


# ─────────────────────────────────────────────
# Per-sample worker
# ─────────────────────────────────────────────

def process_one(args_tuple):
    base_dir, pdb_id, model_number, cutoff, tcr_chains, peptide_chain, mhc_chains = args_tuple

    cif_path = find_cif(base_dir, pdb_id, model_number)
    if cif_path is None:
        return [], f"NOT_FOUND {pdb_id} model={model_number}"

    try:
        atoms = parse_cif_atoms(cif_path)
    except Exception as e:
        return [], f"PARSE_ERROR {pdb_id}: {e}"

    if not atoms:
        return [], f"NO_ATOMS {pdb_id} ({cif_path})"

    present = set(a['chain'] for a in atoms)
    lines = []

    # peptide interface (contacts with TCR + MHC)
    pep_nb = (tcr_chains | mhc_chains) & present
    if peptide_chain in present and pep_nb:
        iface = get_interface_residues(atoms, {peptide_chain}, pep_nb, cutoff)
        for chain, resnums in iface.items():
            if resnums:
                lines.append(f"{pdb_id} {chain} {' '.join(map(str, resnums))}")

    # TCR interface (contacts with peptide)
    tcr_present = tcr_chains & present
    if tcr_present and peptide_chain in present:
        iface = get_interface_residues(atoms, tcr_present, {peptide_chain}, cutoff)
        for chain, resnums in iface.items():
            if resnums:
                lines.append(f"{pdb_id} {chain} {' '.join(map(str, resnums))}")

    return lines, None


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir',      required=True,
                        help='af3_structpred 루트 디렉토리')
    parser.add_argument('--csv',           required=True,
                        help='results_model_quality_metrics_best.csv 경로')
    parser.add_argument('--output',        default='sites_all.txt')
    parser.add_argument('--cutoff',        type=float, default=8.0)
    parser.add_argument('--tcr_chains',    nargs='+', default=['A', 'B'],
                        help='TCR chain IDs (A=TRB, B=TRA in this dataset)')
    parser.add_argument('--peptide_chain', default='E')
    parser.add_argument('--mhc_chains',    nargs='+', default=['C'])
    parser.add_argument('--n_jobs',        type=int, default=8)
    args = parser.parse_args()

    # CSV 읽기
    records = []
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append((row['pdb_id'], int(row['model_number'])))

    print(f"Loaded {len(records)} entries from CSV")

    tcr_chains    = set(args.tcr_chains)
    mhc_chains    = set(args.mhc_chains)
    peptide_chain = args.peptide_chain

    job_args = [
        (args.base_dir, pdb_id, model_number,
         args.cutoff, tcr_chains, peptide_chain, mhc_chains)
        for pdb_id, model_number in records
    ]

    all_lines = []
    errors    = []
    done      = 0
    total     = len(job_args)

    with ProcessPoolExecutor(max_workers=args.n_jobs) as exe:
        futures = {exe.submit(process_one, a): a for a in job_args}
        for fut in as_completed(futures):
            lines, err = fut.result()
            all_lines.extend(lines)
            if err:
                errors.append(err)
            done += 1
            if done % 1000 == 0 or done == total:
                print(f"  {done}/{total} processed | "
                      f"{len(all_lines)} site-lines | "
                      f"{len(errors)} errors")

    with open(args.output, 'w') as f:
        f.write('\n'.join(all_lines) + '\n')

    print(f"\nDone. Written {len(all_lines)} lines to {args.output}")

    if errors:
        err_path = args.output.replace('.txt', '_errors.txt')
        with open(err_path, 'w') as f:
            f.write('\n'.join(errors) + '\n')
        print(f"{len(errors)} errors saved to {err_path}")
        print("First 5 errors:")
        for e in errors[:5]:
            print(f"  {e}")


if __name__ == '__main__':
    main()