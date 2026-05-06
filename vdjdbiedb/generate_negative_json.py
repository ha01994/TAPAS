"""
Positive run의 *_data.json 에서 unpaired MSA만 모아 negative용 입력 JSON을 만든다.
- 하위 폴더(예: vdjdb_full_0/vdjdb_full_0_data.json)까지 재귀 탐색
- 동일 sample id가 여러 output 폴더에 있으면 리스트 앞쪽(먼저 스캔한) 경로만 사용
- 상위 메타(modelSeeds 등)는 TCR 소스 우선, 없으면 pMHC 소스, 그다음 기본값
- 각 protein 블록에 pairedMsa 는 빈 문자열 "" 로 둠 (키는 유지)
- 성공/스킵/쓰기 실패마다 어떤 positive *_data.json 을 썼는지 CSV 로그
"""
import copy
import csv
import json
import os
from typing import Optional

# AlphaFold3 입력 JSON에서 sequences 외에 유지할 수 있는 상위 키
_TOP_LEVEL_META_KEYS = (
    "modelSeeds",
    "dialect",
    "version",
    "bondedAtomPairs",
    "userCCD",
)


def _merge_top_level_metadata(tcr_data: dict, pmhc_data: dict) -> dict:
    """TCR 소스 우선, 부족하면 pMHC 소스로 채운 뒤 필수 기본값 보강."""
    out = {}
    for k in _TOP_LEVEL_META_KEYS:
        if k in tcr_data:
            out[k] = copy.deepcopy(tcr_data[k])
        elif k in pmhc_data:
            out[k] = copy.deepcopy(pmhc_data[k])
    if "modelSeeds" not in out:
        out["modelSeeds"] = [1]
    if "dialect" not in out:
        out["dialect"] = "alphafold3"
    if "version" not in out:
        out["version"] = 3
    return out


def create_mismatched_json(tcr_source_path, pmhc_source_path, output_path):
    """
    TCR 소스의 A,B 체인 + pMHC 소스의 C,D,E 체인을 합친다.
    unpairedMsa 유지. protein 이면 pairedMsa 는 항상 "".
    """
    try:
        with open(tcr_source_path, "r", encoding="utf-8") as f:
            tcr_data = json.load(f)
        with open(pmhc_source_path, "r", encoding="utf-8") as f:
            pmhc_data = json.load(f)

        name = os.path.splitext(os.path.basename(output_path))[0]
        meta = _merge_top_level_metadata(tcr_data, pmhc_data)

        # TCR (A, B) / pMHC (C, D, E)
        tcr_seqs = copy.deepcopy(tcr_data["sequences"][0:2])
        pmhc_seqs = copy.deepcopy(pmhc_data["sequences"][2:5])
        combined_seqs = tcr_seqs + pmhc_seqs
        chain_ids = ["A", "B", "C", "D", "E"]

        for i, seq in enumerate(combined_seqs):
            mol_type = next(iter(seq.keys()))
            inner = seq[mol_type]
            inner["id"] = chain_ids[i]
            if mol_type == "protein":
                inner["pairedMsa"] = ""

        new_data = {"name": name, "sequences": combined_seqs, **meta}

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)

        return True

    except Exception as e:
        print(f"Error creating {output_path}: {e}")
        return False


def build_file_path_map(folder_list):
    """
    각 output 루트 아래 재귀적으로 *_data.json 을 찾는다.
    동일 file_id(파일명 기준)가 여러 번 나오면 먼저 등록된 경로만 유지한다.
    """
    file_path_map = {}
    duplicate_count = 0
    print("📂 폴더 스캔 중 (재귀)...")

    for folder in folder_list:
        if not os.path.isdir(folder):
            print(f"  ⚠️ 경고: 폴더가 없습니다 -> {folder}")
            continue

        for dirpath, _dirnames, filenames in os.walk(folder):
            for filename in filenames:
                if not filename.endswith("_data.json"):
                    continue
                file_id = filename[: -len("_data.json")]
                full_path = os.path.join(dirpath, filename)
                if file_id in file_path_map:
                    duplicate_count += 1
                    continue
                file_path_map[file_id] = full_path

    print(f"  -> 고유 sample id {len(file_path_map)}개, 중복 스킵 {duplicate_count}회")
    return file_path_map


def _positive_sample_id_from_json_path(path: str) -> str:
    """vdjdb_full_0_data.json -> vdjdb_full_0"""
    base = os.path.basename(path)
    suf = "_data.json"
    if base.endswith(suf):
        return base[: -len(suf)]
    return base


def _default_source_log_path(output_dir: str) -> str:
    """예: .../vdjdb_iedb_neg_json -> .../vdjdb_iedb_neg_json_msa_sources.csv"""
    parent = os.path.dirname(os.path.abspath(output_dir)) or "."
    base = os.path.basename(os.path.abspath(output_dir))
    return os.path.join(parent, f"{base}_msa_sources.csv")


def _iter_positive_rows(path):
    """헤더가 있으면 DictReader, 없으면 id,pmhc,tcr,label 고정 컬럼."""
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        if not first_line:
            return
        f.seek(0)
        first_stripped = first_line.strip()
        if first_stripped.lower().startswith("id,") and "pmhc" in first_stripped:
            reader = csv.DictReader(f)
            for row in reader:
                yield row
        else:
            reader = csv.DictReader(f, fieldnames=["id", "pmhc", "tcr", "label"])
            for row in reader:
                yield row


def process_negatives_multi_folder(
    positive_csv,
    negative_csv,
    source_folders,
    output_dir,
    source_log_path: Optional[str] = None,
):
    """
    source_log_path: 각 negative가 어떤 positive *_data.json 에서 TCR(A,B) / pMHC(C,D,E)
    MSA를 가져왔는지 기록하는 CSV. None이면 output_dir 옆에 ``{basename}_msa_sources.csv`` 생성.
    """
    id_to_path_map = build_file_path_map(source_folders)

    tcr_to_path = {}
    pmhc_to_path = {}

    print("\nStep 1: positive CSV → TCR / pMHC 소스 경로 매핑...")
    missing_json = 0
    for row in _iter_positive_rows(positive_csv):
        file_id = row["id"].strip()
        pmhc = row["pmhc"].strip()
        tcr = row["tcr"].strip()

        if file_id in id_to_path_map:
            full_path = id_to_path_map[file_id]
            tcr_to_path[tcr] = full_path
            pmhc_to_path[pmhc] = full_path
        else:
            missing_json += 1

    if missing_json:
        print(f"  Warning: JSON 없는 positive 행 ~{missing_json}건 (첫 샘플만 경고 생략 가능)")
    print(f"  -> 매핑: TCR {len(tcr_to_path)}개, pMHC {len(pmhc_to_path)}개")

    os.makedirs(output_dir, exist_ok=True)

    log_path = source_log_path or _default_source_log_path(output_dir)
    log_fields = [
        "negative_id",
        "target_tcr",
        "target_pmhc",
        "tcr_positive_sample_id",
        "pmhc_positive_sample_id",
        "tcr_source_data_json",
        "pmhc_source_data_json",
        "output_json",
        "status",
    ]

    print("\nStep 2: negative JSON 생성...")
    print(f"  소스 추적 로그: {log_path}")
    success_count = 0
    fail_count = 0
    skip_missing = 0

    with open(negative_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        with open(log_path, "w", encoding="utf-8", newline="") as logf:
            w = csv.DictWriter(logf, fieldnames=log_fields)
            w.writeheader()

            for row in reader:
                new_id = row["id"].strip()
                target_pmhc = row["pmhc"].strip()
                target_tcr = row["tcr"].strip()

                tcr_path = tcr_to_path.get(target_tcr)
                pmhc_path = pmhc_to_path.get(target_pmhc)
                out_json = os.path.join(output_dir, f"{new_id}.json")

                if not tcr_path or not pmhc_path:
                    print(f"  Skip (소스 없음): {new_id}  tcr={target_tcr!r} pmhc={target_pmhc!r}")
                    skip_missing += 1
                    fail_count += 1
                    w.writerow(
                        {
                            "negative_id": new_id,
                            "target_tcr": target_tcr,
                            "target_pmhc": target_pmhc,
                            "tcr_positive_sample_id": _positive_sample_id_from_json_path(tcr_path or ""),
                            "pmhc_positive_sample_id": _positive_sample_id_from_json_path(pmhc_path or ""),
                            "tcr_source_data_json": tcr_path or "",
                            "pmhc_source_data_json": pmhc_path or "",
                            "output_json": out_json,
                            "status": "skip_missing_source",
                        }
                    )
                    continue

                ok = create_mismatched_json(tcr_path, pmhc_path, out_json)
                if ok:
                    success_count += 1
                    st = "ok"
                else:
                    fail_count += 1
                    st = "write_error"

                w.writerow(
                    {
                        "negative_id": new_id,
                        "target_tcr": target_tcr,
                        "target_pmhc": target_pmhc,
                        "tcr_positive_sample_id": _positive_sample_id_from_json_path(tcr_path),
                        "pmhc_positive_sample_id": _positive_sample_id_from_json_path(pmhc_path),
                        "tcr_source_data_json": os.path.abspath(tcr_path),
                        "pmhc_source_data_json": os.path.abspath(pmhc_path),
                        "output_json": os.path.abspath(out_json),
                        "status": st,
                    }
                )

    print(f"\n완료: 성공 {success_count}, 실패(에러+스킵) {fail_count} (그중 소스 미매칭 {skip_missing})")
    print(f"저장: {output_dir}")
    print(f"로그: {log_path}")


# ---------------------------------------------------------------------------
# 실행 설정 (경로는 환경에 맞게 수정)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    input_folders = [
        "/shared/ha01994/alphafast_vdjdb_iedb/vdjdb_iedb_alphafast_output",
    ]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parsed_csv = os.path.join(script_dir, "../tabpfn_vdjdbiedb_fast/iptm_filtered_vdjdbiedb.csv")
    negatives_csv = os.path.join(script_dir, "../tabpfn_vdjdbiedb_fast/negatives_dataset_iptm_filtered_todo.csv")
    output_dir = os.path.join(script_dir, "iptm_filtered_neg_notdone")

    source_log_path = os.path.join(script_dir, "_generate_negative_json.log")

    process_negatives_multi_folder(parsed_csv, negatives_csv, input_folders, output_dir, source_log_path=source_log_path)
