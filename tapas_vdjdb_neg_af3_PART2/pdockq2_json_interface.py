from Bio.PDB import PDBParser
import numpy as np
import argparse
import json
import pandas as pd
from Bio.PDB import PDBParser, MMCIFParser  # MMCIFParser for .cif

parser = argparse.ArgumentParser(description='Calculate Interface pDockQ')
parser.add_argument('-json', type=str, required=True, help='Input json file.')
parser.add_argument('-pdb', type=str, required=True, help='Input pdb file.')
parser.add_argument("-dist", type=int, default=8, help="max distance")
parser.add_argument("-r", nargs='+', required=True, help="Receptor chains (e.g. D E)")
parser.add_argument("-l", nargs='+', required=True, help="Ligand chains (e.g. A B C)")

def sigmoid(x, L, x0, k, b):
    return L / (1 + np.exp(-k * (x - x0))) + b

def calc_pmidockq(ifpae_norm, ifplddt):
    # Coefficients from the original paper
    fitpopt = [1.31034849e+00, 8.47326239e+01, 7.47157696e-02, 5.01886443e-03]
    prot = ifpae_norm * ifplddt
    return sigmoid(prot, *fitpopt)

def retrieve_stats(structure, paeMat, ch_of_interest, target_chains, max_dist):
    # 1. Calculate Average Interface plDDT (IF_plDDT)
    ifplddt = []
    has_contact = False
    
    atoms_interest = [a for r in structure[0][ch_of_interest] for a in r if a.name in ['CA', 'CB']]
    atoms_target = []
    for tc in target_chains:
        atoms_target.extend([a for r in structure[0][tc] for a in r if a.name in ['CA', 'CB']])
    
    for a1 in atoms_interest:
        for a2 in atoms_target:
            diff = a1 - a2
            if diff <= max_dist:
                ifplddt.append(a1.get_bfactor())
                has_contact = True

    if not ifplddt:
        avg_ifplddt = 0
    else:
        avg_ifplddt = np.mean(ifplddt)
        
    # 2. Calculate Average Interface PAE (IF_PAE)
    chain_objs = [c for c in structure[0]]
    chain_ids = [c.id for c in chain_objs]
    seqlens = [len(c) for c in chain_objs]
    
    try:
        idx_interest = chain_ids.index(ch_of_interest)
    except ValueError:
        return 0, 0
        
    start_interest = sum(seqlens[:idx_interest])
    end_interest = start_interest + seqlens[idx_interest]
    
    ifpae_values = []
    d = 10.0
    
    if has_contact:
        for tc in target_chains:
            try:
                idx_target = chain_ids.index(tc)
            except ValueError:
                continue
                
            start_target = sum(seqlens[:idx_target])
            end_target = start_target + seqlens[idx_target]
            
            sub_pae = paeMat[start_interest:end_interest, start_target:end_target]
            
            for r1_i, res1 in enumerate(structure[0][ch_of_interest]):
                a1 = res1['CA'] if 'CA' in res1 else (res1['CB'] if 'CB' in res1 else None)
                if not a1: continue
                
                for r2_i, res2 in enumerate(structure[0][tc]):
                    a2 = res2['CA'] if 'CA' in res2 else (res2['CB'] if 'CB' in res2 else None)
                    if not a2: continue
                    
                    if (a1 - a2) <= max_dist:
                        # Grab PAE from matrix
                        pae_val = sub_pae[r1_i, r2_i]
                        ifpae_values.append(pae_val)

    if not ifpae_values:
        avg_ifpae_norm = 0
    else:
        # Normalize: 1 / (1 + (PAE/10)^2)
        avg_ifpae_norm = np.mean(1 / (1 + (np.array(ifpae_values) / d) ** 2))

    return avg_ifplddt, avg_ifpae_norm

def main():
    args = parser.parse_args()
    
    # Pick PDB vs mmCIF parser from the file extension
    if args.pdb.endswith('.cif'):
        parser_struct = MMCIFParser(QUIET=True)
    else:
        parser_struct = PDBParser(QUIET=True)
        
    structure = parser_struct.get_structure('struct', args.pdb)
    # -------------------------------------------------------------------------
    
    with open(args.json) as f:
        data = json.load(f)
    paeMat = np.array(data['pae'])

    # Print output in a format easy to parse: "Chain iPAE pDockQ"
    print("Chain iPAE pDockQ")
    
    # Receptor vs Ligands
    for r_chain in args.r:
        if_plddt, if_pae = retrieve_stats(structure, paeMat, r_chain, args.l, args.dist)
        score = calc_pmidockq(if_pae, if_plddt)
        print(f"{r_chain} {if_pae:.6f} {score:.6f}")
        
    # Ligand vs Receptors
    for l_chain in args.l:
        if_plddt, if_pae = retrieve_stats(structure, paeMat, l_chain, args.r, args.dist)
        score = calc_pmidockq(if_pae, if_plddt)
        print(f"{l_chain} {if_pae:.6f} {score:.6f}")
        
if __name__ == '__main__':
    main()