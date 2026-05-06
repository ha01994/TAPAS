"""
AF3 output에서 best model의 PAE matrix 중 interface residue 부분만 추출
- best model: results_model_quality_metrics_best.csv의 model_number
- interface residues: compute_hermes/sites_af3_vdjdb.txt (chain E, A, B)
- chain 순서 (v0 기준): E(9) → A(116) → B(111) → C(276) → D(100)
  → PAE matrix offset: E=0, A=9, B=125, C=236, D=512

출력: per pdb_id, interface residue 간 PAE submatrix를 numpy .npy로 저장
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
AF_OUTPUT_DIR  = '/shared/ha01994/alphafast_vdjdb_iedb/vdjdb_iedb_alphafast_output/'
BEST_CSV       = 'results_vdjdbiedb_iptm_filtered_best.csv'
SITES_TXT      = 'sites_vdjdbiedb.txt'
OUT_DIR        = 'pae_matrix_vdjdbiedb'

os.makedirs(OUT_DIR, exist_ok=True)


# ── 1. chain offset 계산 함수 ──────────────────────────────────────────────────
def get_chain_offsets(cif_path):
    """
    CIF 파일에서 chain 순서대로 residue 수를 세어 offset dict 반환.
    chain 순서: 등장 순서 기준 (AF3는 입력 순서 유지)
    반환: {'E': 0, 'A': 9, 'B': 125, 'C': 236, 'D': 512} 형태
    """
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


# ── 2. sites 파일 파싱 ─────────────────────────────────────────────────────────
def parse_sites(sites_path):
    """
    반환: {pdb_id: {'E': [1,2,...], 'A': [28,29,...], 'B': [30,92,...]}}
    residue 번호는 1-based (CIF label_seq_id 기준)
    """
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


# ── 3. PAE matrix 로드 ─────────────────────────────────────────────────────────
def load_pae(pdb_id, model_number, af_output_dir):
    """
    {af_output_dir}/{pdb_id}/seed-1_sample-{model_number}/
      {pdb_id}_seed-1_sample-{model_number}_confidences.json
    에서 pae 로드 → np.array (N x N)
    """
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


# ── 4. interface PAE submatrix 추출 ───────────────────────────────────────────
def extract_interface_pae(pae, offsets, site_chains):
    """
    site_chains: {'E': [1,2,...], 'A': [28,...], 'B': [30,...]}
    offsets:     {'E': 0, 'A': 9, 'B': 125, ...}

    residue 번호는 1-based → 0-based index = offset + (resnum - 1)

    반환:
      - submatrix: (n_interface x n_interface) np.array
      - labels: 각 행/열에 해당하는 'E1', 'A28', 'B30' 형태 레이블 리스트
    """
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


# ── 5. CIF 경로 결정 함수 ──────────────────────────────────────────────────────
def get_cif_path(pdb_id, model_number, af_output_dir):
    """
    sample별 CIF 우선, 없으면 top-level CIF 사용
    """
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


# ── 메인 ───────────────────────────────────────────────────────────────────────
def main():
    # 로드
    df_best = pd.read_csv(BEST_CSV)
    sites   = parse_sites(SITES_TXT)

    success, skipped, errors = 0, 0, []

    for _, row in df_best.iterrows():
        pdb_id       = str(row['pdb_id'])
        model_number = int(row['model_number'])

        if pdb_id not in sites:
            print(f"[SKIP] {pdb_id}: sites 정보 없음")
            skipped += 1
            continue

        try:
            # CIF에서 chain offset 계산
            cif_path = get_cif_path(pdb_id, model_number, AF_OUTPUT_DIR)
            offsets, chain_counts = get_chain_offsets(cif_path)

            # PAE 로드
            pae = load_pae(pdb_id, model_number, AF_OUTPUT_DIR)

            # 전체 residue 수 일치 확인
            total_res = sum(chain_counts.values())
            if pae.shape != (total_res, total_res):
                raise ValueError(
                    f"PAE shape {pae.shape} != expected ({total_res},{total_res})"
                )

            # interface submatrix 추출
            submatrix, labels = extract_interface_pae(
                pae, offsets, sites[pdb_id]
            )

            # 저장
            out_path = os.path.join(OUT_DIR, f'{pdb_id}_interface_pae.npy')
            np.save(out_path, submatrix)

            # 레이블도 저장
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

    print(f"\n완료: {success}개 성공 | {skipped}개 스킵 | {len(errors)}개 오류")
    if errors:
        print("오류 목록:")
        for pid, msg in errors:
            print(f"  {pid}: {msg}")


if __name__ == '__main__':
    main()
