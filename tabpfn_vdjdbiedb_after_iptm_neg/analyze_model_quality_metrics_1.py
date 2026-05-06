import os, sys, json
import pandas as pd
import subprocess
import re
from analyze_crystal_vs_model import *


def calculate_global_plddt(json_file_path):
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
            
        atom_plddts = data.get('atom_plddts', [])
        if not atom_plddts:
            print("No data found in 'atom_plddts'.")
            return None
        
        mean_plddt = sum(atom_plddts) / len(atom_plddts)
        return mean_plddt

    except FileNotFoundError:
        print(f"Error: The file {json_file_path} was not found.")
        return None
    
    except json.JSONDecodeError:
        print("Error: Failed to decode the JSON file. Please check its format.")
        return None
    
    
    
def extract_b_factors(cdr_atoms, chain):
    b_factors = []
    for atomname, resid, resname, chainid in cdr_atoms:
        if chainid == chain.id:
            try:
                residue = chain[resid]  # Access the residue using its ID
                if atomname in residue:  # Check if the atom exists in the residue
                    atom = residue[atomname]
                    b_factors.append(atom.get_bfactor())
                else:
                    print(f"Atom {atomname} not found in residue {resid} ({resname}) of chain {chain.id}")
            except KeyError:
                print(f"Residue {resid} ({resname}) not found in chain {chain.id}")
    return b_factors



def cdr_plddts(model_file, alpha_chain, beta_chain):
        
    model_sequences, model_dict = extract_sequences(model_file)
        
    residues_A = extract_residues_and_resids(model_file, alpha_chain)
    residues_B = extract_residues_and_resids(model_file, beta_chain)
    
    anarci_A = run_anarci(model_sequences[alpha_chain])
    anarci_B = run_anarci(model_sequences[beta_chain])
    #print(anarci_A)
    #print(anarci_B)
    
    parsed_A = parse_anarci_output(anarci_A)
    parsed_B = parse_anarci_output(anarci_B)
    
    map_A = map_imgt_to_original(parsed_A, residues_A)
    map_B = map_imgt_to_original(parsed_B, residues_B)
    
    # Parse CDR regions from the maps
    cdr3_A = parse_CDR3(map_A)
    cdr3_B = parse_CDR3(map_B)
    cdr2_A = parse_CDR2(map_A)
    cdr2_B = parse_CDR2(map_B)
    cdr1_A = parse_CDR1(map_A)
    cdr1_B = parse_CDR1(map_B)
    
    # Extract atom information for each CDR
    cdr3_atoms_A = extract_atoms_for_cdr(cdr3_A, model_file, alpha_chain)
    cdr3_atoms_B = extract_atoms_for_cdr(cdr3_B, model_file, beta_chain)
    cdr2_atoms_A = extract_atoms_for_cdr(cdr2_A, model_file, alpha_chain)
    cdr2_atoms_B = extract_atoms_for_cdr(cdr2_B, model_file, beta_chain)
    cdr1_atoms_A = extract_atoms_for_cdr(cdr1_A, model_file, alpha_chain)
    cdr1_atoms_B = extract_atoms_for_cdr(cdr1_B, model_file, beta_chain)
    
    # Parse the structure based on file type (PDB or MMCIF)
    if model_file.endswith(".pdb"):
        parser = PDB.PDBParser(QUIET=True)
    else:
        parser = PDB.MMCIFParser(QUIET=True)
        
    structure = parser.get_structure("Model", model_file)
    chain_A = structure[0][alpha_chain]
    chain_B = structure[0][beta_chain]
        
    # Extract B-factors for each CDR region separately
    b_factors_cdr1_A = extract_b_factors(cdr1_atoms_A, chain_A)
    b_factors_cdr2_A = extract_b_factors(cdr2_atoms_A, chain_A)
    b_factors_cdr3_A = extract_b_factors(cdr3_atoms_A, chain_A)
    
    b_factors_cdr1_B = extract_b_factors(cdr1_atoms_B, chain_B)
    b_factors_cdr2_B = extract_b_factors(cdr2_atoms_B, chain_B)
    b_factors_cdr3_B = extract_b_factors(cdr3_atoms_B, chain_B)

    # Mean 
    mean_cdr1_A = np.mean(b_factors_cdr1_A)
    mean_cdr2_A = np.mean(b_factors_cdr2_A)
    mean_cdr3_A = np.mean(b_factors_cdr3_A)
    mean_cdr1_B = np.mean(b_factors_cdr1_B)
    mean_cdr2_B = np.mean(b_factors_cdr2_B)
    mean_cdr3_B = np.mean(b_factors_cdr3_B)
    
    # Return the B-factors for each CDR region separately
    return mean_cdr1_A, mean_cdr2_A, mean_cdr3_A, mean_cdr1_B, mean_cdr2_B, mean_cdr3_B



def calculate_iptms(json_file_path, length=5):
    """
    Calculates the mean of `chain_iptm` and the mean of interface TCR-pMHC iPTMs
    using fixed chain mappings.
    """
    try:
        # Load the JSON data from the file
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        
        # Calculate the mean of chain_iptm
        chain_iptm = data.get('chain_iptm', [])
        if not chain_iptm:
            print("No data found in 'chain_iptm'.")
            chain_iptm_mean = None
        else:
            chain_iptm_mean = sum(chain_iptm) / len(chain_iptm)
        
        # Calculate the mean for interface TCR-pMHC
        chain_pair_iptm = data.get('chain_pair_iptm', [])
        if not chain_pair_iptm:
            print("No data found in 'chain_pair_iptm'.")
            tcr_pmch_mean = None
        else:
            # Fixed indices for TCR-pMHC interactions
            # A (MHC) = 0, B (B2M) = 1, C (peptide) = 2, D (TCRa) = 3, E (TCRb) = 4
            if length == 5:
                tcr_pmch_pairs = [
                    chain_pair_iptm[0][3],  # MHC-TCRa 
                    chain_pair_iptm[0][4],  # MHC-TCRb 
                    chain_pair_iptm[2][3],  # pep-TCRa
                    chain_pair_iptm[2][4]]  # pep-TCRb 
                tcr_pmch_iptm = sum(tcr_pmch_pairs) / len(tcr_pmch_pairs)
            elif length == 4:
                # A (MHC) = 0, C (peptide) = 1, D (TCRa) = 2, E (TCRb) = 3
                tcr_pmch_pairs = [
                    chain_pair_iptm[0][2],  # MHC-TCRa 
                    chain_pair_iptm[0][3],  # MHC-TCRb 
                    chain_pair_iptm[1][2],  # pep-TCRa
                    chain_pair_iptm[1][3]]  # pep-TCRb 
                tcr_pmch_iptm = sum(tcr_pmch_pairs) / len(tcr_pmch_pairs)
        return chain_iptm_mean, tcr_pmch_iptm
    
    except FileNotFoundError:
        print(f"File not found: {json_file_path}")
        return None
    except KeyError as e:
        print(f"Missing key in JSON data: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    


def calculate_pdockq (model_file):
    command=f"python ./scripts_py/pdockq.py --pdbfile {model_file}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
    # Output is displayed as pDockQ = 0.609 for ./pre/merged_models_AB/1ao7_0_merged.pdb This corresponds to a PPV of at least 0.9400192
    # Capture pDockq
    pdockq = float(result.stdout.split('=')[1].split(' ')[1])
    return result.stdout, pdockq



import numpy as np
from scipy.spatial import distance_matrix
from Bio import PDB

def calculate_pdockq_direct(model_file, receptor_chains, ligand_chains, distance_threshold=8.0):
    """
    원본 CIF/PDB 파일을 읽어 두 체인 그룹 간의 pDockQ를 외부 스크립트 없이 직접 계산합니다.
    """
    # 파서 설정 (cif와 pdb 모두 호환)
    if model_file.endswith(".pdb"):
        parser = PDB.PDBParser(QUIET=True)
    else:
        parser = PDB.MMCIFParser(QUIET=True)
        
    structure = parser.get_structure("Model", model_file)
    model = structure[0]

    def get_coords_and_plddts(chain_ids):
        coords = []
        plddts = []
        for chain_id in chain_ids:
            if chain_id not in model:
                continue
            for res in model[chain_id]:
                # 물 분자나 헤테로 원자가 아닌 표준 아미노산만 필터링
                if res.id[0] != ' ':
                    continue
                # Glycine은 CA, 나머지는 CB 원자 기준 (FoldDock pDockQ 원리)
                if 'CB' in res:
                    atom = res['CB']
                elif 'CA' in res:
                    atom = res['CA']
                else:
                    continue
                coords.append(atom.coord)
                # AlphaFold는 B-factor 열에 pLDDT를 저장합니다.
                plddts.append(atom.get_bfactor()) 
        return np.array(coords), np.array(plddts)

    # 1. 두 그룹의 원자 좌표와 pLDDT 추출
    rec_coords, rec_plddts = get_coords_and_plddts(receptor_chains)
    lig_coords, lig_plddts = get_coords_and_plddts(ligand_chains)

    if len(rec_coords) == 0 or len(lig_coords) == 0:
        return 0.0

    # 2. 원자간 거리 계산 (Distance Matrix)
    dists = distance_matrix(rec_coords, lig_coords)

    # 3. Interface Contacts (8.0 Å 이하) 찾기
    contact_indices = np.where(dists < distance_threshold)
    num_contacts = len(contact_indices[0])

    if num_contacts == 0:
        return 0.0

    # 4. Interface Residues의 평균 pLDDT 계산
    rec_contact_idx = np.unique(contact_indices[0])
    lig_contact_idx = np.unique(contact_indices[1])

    if_plddts = np.concatenate([rec_plddts[rec_contact_idx], lig_plddts[lig_contact_idx]])
    avg_if_plddt = np.mean(if_plddts)

    # 5. pDockQ 공식 계산 (Bryant et al., 2022)
    x = avg_if_plddt * np.log10(num_contacts)
    L = 0.724
    x0 = 152.611
    k = 0.052
    b = 0.018
    pdockq = L / (1 + np.exp(-k * (x - x0))) + b

    return pdockq



# 코드 변경
# (1) ORIGINAL un-merged PDB model을 사용하도록
# (2) TCR-pMHC interface 에 대해서만 계산하도록

def calculate_pdockq2_json(model_file, json_file, receptor_chains, ligand_chains):
    import subprocess

    r_args = " ".join(list(receptor_chains))
    l_args = " ".join(list(ligand_chains))    
    command = f"python ./pdockq2_json_interface.py -json {json_file} -pdb {model_file} -r {r_args} -l {l_args}"
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        output_lines = result.stdout.strip().split('\n')
        print(output_lines)
        
        pdockq_scores = {}
        ipae_scores = {}

        # Parse output: "Chain iPAE pDockQ"
        for line in output_lines:
            parts = line.split()
            # We expect 3 parts: Chain, iPAE, pDockQ
            if len(parts) == 3 and parts[0] != "Chain": 
                try:
                    chain = parts[0]
                    ipae_val = float(parts[1])
                    pdockq_val = float(parts[2])
                    
                    ipae_scores[chain] = ipae_val
                    pdockq_scores[chain] = pdockq_val
                except ValueError:
                    continue
        
        # 1. pMHC score (Ligand)
        ligand_pdockq = [pdockq_scores[c] for c in ligand_chains if c in pdockq_scores]
        ligand_ipae = [ipae_scores[c] for c in ligand_chains if c in ipae_scores]
        
        pdockq2_pmhc = sum(ligand_pdockq)/len(ligand_pdockq) if ligand_pdockq else 0
        avgipae_pmhc = sum(ligand_ipae)/len(ligand_ipae) if ligand_ipae else 0
        
        # 2. TCR score (Receptor)
        rec_pdockq = [pdockq_scores[c] for c in receptor_chains if c in pdockq_scores]
        rec_ipae = [ipae_scores[c] for c in receptor_chains if c in ipae_scores]
        
        avg_pdockq2_tcr = sum(rec_pdockq)/len(rec_pdockq) if rec_pdockq else 0
        avg_ipae_tcr = sum(rec_ipae)/len(rec_ipae) if rec_ipae else 0

        # Return real iPAE values now
        return result.stdout, avgipae_pmhc, avg_ipae_tcr, pdockq2_pmhc, avg_pdockq2_tcr

    except Exception as e:
        print(f"Error in calculate_pdockq2_json: {e}")
        return "", 0.0, 0.0, 0.0, 0.0




def safe_round(value, precision=8):    
    if value is not None:
        return round(value, precision)
    return None



            
# TCRVDB, IMMREP, VDJDB Chain Naming
HLA_CHAIN = 'C'
B2M_CHAIN = 'D'
PEP_CHAIN = 'E'
TRA_CHAIN = 'B'
TRB_CHAIN = 'A'



def main(model_folder, output_folder): 
    os.makedirs(f'{output_folder}', exist_ok=True)
    
    foldername = '1'
    csv_path = f"{output_folder}/results_model_quality_metrics_{foldername}.csv"
    pdb_ids = [x.split('/')[-1] for x in glob.glob(f'/shared/ha01994/iptm_filtered_neg_notdone_{foldername}_output/*')]    
    print('len(pdb_ids)', len(pdb_ids))
    pdb_ids.sort()
    
    already_done = []
    if os.path.exists(csv_path):
        with open(csv_path,'r') as f:
            r = csv.reader(f)
            next(r)
            for line in r:
                already_done.append(line[0]+'_model_'+line[1])
    
    for pdb_id in pdb_ids:
        print(f'========================{pdb_id}=======================')

        for model_number in range(5):
            model_number = str(model_number)
            
            # 중복 체크: 이미 처리했다면 건너뛰기
            if f"{pdb_id}_model_{model_number}" in already_done:
                print(f"Skipping: {pdb_id} model {model_number} (already processed)")
                continue

            pdb_model = f'/shared/ha01994/iptm_filtered_neg_notdone_{foldername}_output/{pdb_id}/seed-1_sample-{model_number}/{pdb_id}_seed-1_sample-{model_number}_model.cif'            
            summary_json = f'/shared/ha01994/iptm_filtered_neg_notdone_{foldername}_output/{pdb_id}/seed-1_sample-{model_number}/{pdb_id}_seed-1_sample-{model_number}_summary_confidences.json'            
            all_data_json = f'/shared/ha01994/iptm_filtered_neg_notdone_{foldername}_output/{pdb_id}/seed-1_sample-{model_number}/{pdb_id}_seed-1_sample-{model_number}_confidences.json'
            
            print('calculate global plddt')
            mean = calculate_global_plddt(all_data_json)

            print('calculate iptm')
            b_factors_cdr1_A, b_factors_cdr2_A, \
            b_factors_cdr3_A, b_factors_cdr1_B, \
            b_factors_cdr2_B, b_factors_cdr3_B = cdr_plddts(
                pdb_model, TRA_CHAIN, TRB_CHAIN 
            )
            iptm_mean, iptm_tcrpmhc = calculate_iptms(summary_json)

            print('calculate pdockq directly from original cif')
            pmhc_chains_list = [HLA_CHAIN, B2M_CHAIN, PEP_CHAIN]
            tcr_chains_list = [TRA_CHAIN, TRB_CHAIN]                
            # 새로운 함수로 원본 pdb_model (cif 파일)을 바로 넘깁니다
            pdockq_AB = calculate_pdockq_direct(pdb_model, tcr_chains_list, pmhc_chains_list)

            print('calculate pdockq2 json')                
            pmhc_chains = f"{HLA_CHAIN}{B2M_CHAIN}{PEP_CHAIN}"
            tcr_chains = f"{TRA_CHAIN}{TRB_CHAIN}"

            # Use the ORIGINAL un-merged PDB model
            _, avgipae_pmhc, avgipae_tcr, pdockq2_pmhc, pdockq2_tcr = calculate_pdockq2_json(
                pdb_model,
                all_data_json, 
                receptor_chains=tcr_chains, 
                ligand_chains=pmhc_chains
            )

            row = {
                "pdb_id": pdb_id,
                "model_number": model_number,
                "global_plddt": safe_round(mean),
                "cdr1_A": safe_round(b_factors_cdr1_A),
                "cdr2_A": safe_round(b_factors_cdr2_A),
                "cdr3_A": safe_round(b_factors_cdr3_A),
                "cdr1_B": safe_round(b_factors_cdr1_B),
                "cdr2_B": safe_round(b_factors_cdr2_B),
                "cdr3_B": safe_round(b_factors_cdr3_B),
                "iptm_mean": safe_round(iptm_mean),
                "iptm_tcrpmhc": safe_round(iptm_tcrpmhc),
                "pdockq": safe_round(pdockq_AB),
                "avgipae_pmhc": safe_round(avgipae_pmhc),
                "avgipae_tcr": safe_round(avgipae_tcr),
                "pdockq2_pmhc": safe_round(pdockq2_pmhc),
                "pdockq2_tcr": safe_round(pdockq2_tcr),
            }
            print(row)
            row_df = pd.DataFrame([row])
            row_df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)




if __name__ == '__main__':    
    model_folder = ''
    output_folder = '.'
    
    main(model_folder, output_folder)



