import json
import csv
import os
import copy

# ==========================================
# 1. 파일 조립 함수 (이전과 동일)
# ==========================================
def create_mismatched_json(tcr_source_path, pmhc_source_path, output_path):
    """
    TCR과 pMHC 데이터를 가져와 합치고 PairedMSA를 제거합니다.
    """
    try:
        with open(tcr_source_path, 'r') as f:
            tcr_data = json.load(f)
        with open(pmhc_source_path, 'r') as f:
            pmhc_data = json.load(f)

        new_data = copy.deepcopy(tcr_data)
        new_data['name'] = os.path.splitext(os.path.basename(output_path))[0]
        new_data['sequences'] = []

        # TCR (A, B) / pMHC (C, D, E) 추출
        tcr_seqs = tcr_data['sequences'][0:2]
        pmhc_seqs = pmhc_data['sequences'][2:5]

        combined_seqs = tcr_seqs + pmhc_seqs
        chain_ids = ['A', 'B', 'C', 'D', 'E']

        for i, seq in enumerate(combined_seqs):
            mol_type = list(seq.keys())[0]
            seq[mol_type]['id'] = chain_ids[i]
            # [중요] False Signal 방지를 위해 PairedMSA 제거
            seq[mol_type]['pairedMsa'] = ""

        new_data['sequences'] = combined_seqs

        with open(output_path, 'w') as f:
            json.dump(new_data, f, indent=2)
            
        return True

    except Exception as e:
        print(f"Error creating {output_path}: {e}")
        return False

# ==========================================
# 2. 핵심: 여러 폴더에서 파일 위치 찾기
# ==========================================
def build_file_path_map(folder_list):
    """
    여러 폴더를 순회하며 { 'v0': '/full/path/to/v0_data.json', ... } 형태의 맵을 만듭니다.
    """
    file_path_map = {}
    print("📂 폴더 스캔 중...")
    
    for folder in folder_list:
        if not os.path.exists(folder):
            print(f"  ⚠️ 경고: 폴더가 존재하지 않습니다 -> {folder}")
            continue
            
        # 해당 폴더의 모든 파일 확인
        for filename in os.listdir(folder):
            if filename.endswith("_data.json"):
                # 파일명에서 ID 추출 (예: v0_data.json -> v0)
                file_id = filename.split("_data.json")[0]
                full_path = os.path.join(folder, filename)
                
                # 맵에 저장
                file_path_map[file_id] = full_path
                
    print(f"  -> 총 {len(file_path_map)}개의 원본 파일을 찾았습니다.")
    return file_path_map

# ==========================================
# 3. 메인 실행 로직
# ==========================================
def process_negatives_multi_folder(positive_csv, negative_csv, source_folders, output_dir):
    
    # [Step 0] 모든 폴더를 뒤져서 파일 위치 지도 만들기
    id_to_path_map = build_file_path_map(source_folders)
    
    # [Step 1] TCR/pMHC가 어떤 ID(파일)에 있는지 맵핑
    # (TCR 서열 이름 -> 원본 파일 경로)
    tcr_to_path = {}
    pmhc_to_path = {}
    
    print("\nStep 1: 데이터 내용 매핑 중...")
    with open(positive_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_id = row['id'].strip()   # v0
            pmhc = row['pmhc'].strip()
            tcr = row['tcr'].strip()
            
            # 미리 찾아둔 경로가 있는지 확인
            if file_id in id_to_path_map:
                full_path = id_to_path_map[file_id]
                tcr_to_path[tcr] = full_path
                pmhc_to_path[pmhc] = full_path
            else:
                # 가끔 csv에는 있는데 실제 json 파일이 없는 경우
                print(f"Warning: {file_id}에 해당하는 json 파일을 찾지 못했습니다.")
                pass

    print(f" -> 매핑 완료: TCR {len(tcr_to_path)}개, pMHC {len(pmhc_to_path)}개")

    # [Step 2] Negative 조합 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("\nStep 2: Negative 샘플 생성 시작...")
    
    success_count = 0
    fail_count = 0
    
    with open(negative_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            print(row['id'])
            new_id = row['id'].strip()
            target_pmhc = row['pmhc'].strip()
            target_tcr = row['tcr'].strip()
            
            # 경로 찾기
            tcr_path = tcr_to_path.get(target_tcr)
            pmhc_path = pmhc_to_path.get(target_pmhc)
            
            if not tcr_path or not pmhc_path:
                print(f"Skipping {new_id}: 소스 파일 없음")
                fail_count += 1
                continue
                
            output_path = os.path.join(output_dir, f"{new_id}.json")
            
            if create_mismatched_json(tcr_path, pmhc_path, output_path):
                success_count += 1
            else:
                fail_count += 1

    print(f"\n✅ 작업 완료!")
    print(f"   성공: {success_count}개")
    print(f"   실패: {fail_count}개 (소스 파일 누락 등)")
    print(f"   저장 위치: {output_dir}")

# ==========================================
# 실행 설정
# ==========================================

# 1. 4개의 폴더 경로를 리스트에 담아주세요.
input_folders = [
    "/shared/ha01994/alphafast_vdjdb_iedb/vdjdb_iedb_alphafast_0_output",
    "/shared/ha01994/alphafast_vdjdb_iedb/vdjdb_iedb_alphafast_1_output",
    "/shared/ha01994/alphafast_vdjdb_iedb/vdjdb_iedb_alphafast_2_output",    
]

# 2. 나머지 파일 경로 설정
parsed_csv = "parsed_data_freq10_cap100.csv"
negatives_csv = "negatives.csv"
output_dir = "vdjdb_iedb_neg_json"

import os
os.system('rm -rf vdjdb_iedb_neg_json')
os.system('mkdir vdjdb_iedb_neg_json')

# 3. 실행
process_negatives_multi_folder(parsed_csv, negatives_csv, input_folders, output_dir)

