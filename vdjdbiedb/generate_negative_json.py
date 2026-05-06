import copy
import csv
import json
import os
from typing import Optional

# Top-level keys to keep besides 'sequences' in AlphaFold3 input JSON
_TOP_LEVEL_META_KEYS = (
    "modelSeeds",
    "dialect",
    "version",
    "bondedAtomPairs",
    "userCCD",
)


def _merge_top_level_metadata(tcr_data: dict, pmhc_data: dict) -> dict:
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
    file_path_map = {}
    duplicate_count = 0
    print("📂 Scanning folders (recursive)...")

    for folder in folder_list:
        if not os.path.isdir(folder):
            print(f"  Warning: folder does not exist -> {folder}")
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

    print(f"  -> unique sample id {len(file_path_map)}items, duplicates skipped {duplicate_count} times")
    return file_path_map


def _positive_sample_id_from_json_path(path: str) -> str:
    base = os.path.basename(path)
    suf = "_data.json"
    if base.endswith(suf):
        return base[: -len(suf)]
    return base


def _default_source_log_path(output_dir: str) -> str:
    parent = os.path.dirname(os.path.abspath(output_dir)) or "."
    base = os.path.basename(os.path.abspath(output_dir))
    return os.path.join(parent, f"{base}_msa_sources.csv")


def _iter_positive_rows(path):
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
    id_to_path_map = build_file_path_map(source_folders)

    tcr_to_path = {}
    pmhc_to_path = {}

    print("\nStep 1: positive CSV → mapping TCR / pMHC source paths...")
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
        print(f"  Warning: positive row without JSON ~{missing_json} cases (can omit warning after the first sample)")
    print(f"  -> mapping: TCR {len(tcr_to_path)}items, pMHC {len(pmhc_to_path)}items")

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

    print("\nStep 2: generating negative JSON...")
    print(f"  source trace log: {log_path}")
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
                    print(f"  Skip (no source): {new_id}  tcr={target_tcr!r} pmhc={target_pmhc!r}")
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

    print(f"\nDone: success {success_count}, failed (errors+skips) {fail_count} (among them, source-unmatched {skip_missing})")
    print(f"saved: {output_dir}")
    print(f"log: {log_path}")


# ---------------------------------------------------------------------------
# Run configuration (adjust paths for your environment)
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
