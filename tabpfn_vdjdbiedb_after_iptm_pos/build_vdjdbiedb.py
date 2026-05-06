#!/usr/bin/env python3
"""
Merge dataset_rs + dataset_ss CSVs, join TCR CDR fields from dic_full_vavb.csv,
and write vdjdbiedb.csv in the same column layout as vdjdb123.csv:
  id,peptide,A1,A2,A3,B1,B2,B3,binder
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from typing import Dict, List, Tuple


def parse_dic_blob(blob: str) -> Tuple[str, str, str, str, str, str]:
    """
    Second column of dic_full_vavb.csv:
      {alpha_aa}_{beta_aa}_{A1}_{A2}_{A3}_{B1}_{B2}_{B3}_{TRAV...}_{TRAJ...}_{TRBV...}_{TRBJ...}
    The last 10 underscore tokens are A1..B3 plus four gene names.
    Everything before that is alpha (first segment) and beta (remaining joined).
    """
    parts = blob.split("_")
    if len(parts) < 12:
        raise ValueError(f"expected >=12 underscore segments, got {len(parts)}")
    tail = parts[-10:]
    a1, a2, a3, b1, b2, b3 = tail[0], tail[1], tail[2], tail[3], tail[4], tail[5]
    head = parts[:-10]
    if len(head) < 2:
        raise ValueError("alpha/beta segment missing")
    return a1, a2, a3, b1, b2, b3


def load_dic(path: str) -> Dict[str, Tuple[str, str, str, str, str, str]]:
    out: Dict[str, Tuple[str, str, str, str, str, str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            tcr_id = row[0].strip()
            blob = row[1]
            out[tcr_id] = parse_dic_blob(blob)
    return out


def iter_dataset_rows(paths: List[str]):
    for p in sorted(paths):
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue
            need = {"id", "pmhc", "tcr", "label"}
            if not need.issubset(set(reader.fieldnames)):
                raise SystemExit(f"{p}: missing columns {need - set(reader.fieldnames)}")
            for row in reader:
                yield row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dic",
        default="dic_full_vavb.csv",
        help="TCR dictionary CSV (tcr_id, concatenated record)",
    )
    ap.add_argument(
        "--out",
        default="vdjdbiedb.csv",
        help="Output path (vdjdb123-compatible columns)",
    )
    ap.add_argument(
        "--root",
        default=".",
        help="Project root containing dataset_rs/ and dataset_ss/",
    )
    args = ap.parse_args()
    root = args.root

    dic_path = os.path.join(root, args.dic) if not os.path.isabs(args.dic) else args.dic
    out_path = os.path.join(root, args.out) if not os.path.isabs(args.out) else args.out

    tcr_to_cdr = load_dic(dic_path)

    rs_glob = glob.glob(os.path.join(root, "dataset_rs", "*.csv"))
    ss_glob = glob.glob(os.path.join(root, "dataset_ss", "*.csv"))
    paths = rs_glob + ss_glob
    if not paths:
        print("No CSV files under dataset_rs/ or dataset_ss/", file=sys.stderr)
        sys.exit(1)

    seen: set[str] = set()
    rows_out: List[dict] = []
    missing_tcr: Dict[str, int] = {}
    dup_ids = 0

    for row in iter_dataset_rows(paths):
        rid = row["id"].strip()
        if rid in seen:
            dup_ids += 1
            continue
        seen.add(rid)

        pmhc = row["pmhc"].strip()
        if "_" not in pmhc:
            peptide = pmhc
        else:
            peptide, _mhc = pmhc.split("_", 1)

        tcr_id = row["tcr"].strip()
        cdrs = tcr_to_cdr.get(tcr_id)
        if cdrs is None:
            missing_tcr[tcr_id] = missing_tcr.get(tcr_id, 0) + 1
            continue

        a1, a2, a3, b1, b2, b3 = cdrs
        label = row["label"].strip()
        rows_out.append(
            {
                "id": rid,
                "peptide": peptide,
                "A1": a1,
                "A2": a2,
                "A3": a3,
                "B1": b1,
                "B2": b2,
                "B3": b3,
                "binder": label,
            }
        )

    fieldnames = ["id", "peptide", "A1", "A2", "A3", "B1", "B2", "B3", "binder"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows to {out_path}")
    print(f"Unique input ids: {len(seen)}, skipped duplicate ids: {dup_ids}")
    if missing_tcr:
        n = sum(missing_tcr.values())
        print(f"Skipped {n} rows with unknown tcr_id ({len(missing_tcr)} distinct tcrs)", file=sys.stderr)


if __name__ == "__main__":
    main()
