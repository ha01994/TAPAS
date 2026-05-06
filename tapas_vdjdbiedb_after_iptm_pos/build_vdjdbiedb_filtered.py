#!/usr/bin/env python3
"""
Build vdjdbiedb_filtered.csv (vdjdbiedb.csv layout) from:
  - iptm_filtered_vdjdbiedb.csv
  - negatives_dataset_iptm_filtered.csv
using TCR CDR fields from dic_full_vavb.csv (same parsing as build_vdjdbiedb.py).

Columns: id,peptide,A1,A2,A3,B1,B2,B3,binder
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

from build_vdjdbiedb import load_dic


def iter_rows(paths: list[str]):
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            need = {"id", "pmhc", "tcr", "label"}
            if reader.fieldnames is None or not need.issubset(set(reader.fieldnames)):
                raise SystemExit(f"{path}: need columns {need}")
            for row in reader:
                yield row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--positives", default="iptm_filtered_vdjdbiedb.csv")
    ap.add_argument("--negatives", default="negatives_dataset_iptm_filtered.csv")
    ap.add_argument("--dic", default="dic_full_vavb.csv")
    ap.add_argument("-o", "--out", default="vdjdbiedb_filtered.csv")
    args = ap.parse_args()

    for p in (args.positives, args.negatives, args.dic):
        if not os.path.isfile(p):
            print(f"Not found: {p}", file=sys.stderr)
            sys.exit(1)

    tcr_to_cdr = load_dic(args.dic)
    fieldnames = ["id", "peptide", "A1", "A2", "A3", "B1", "B2", "B3", "binder"]

    out_rows: list[dict[str, str]] = []
    missing_tcr: dict[str, int] = {}

    for row in iter_rows([args.positives, args.negatives]):
        rid = row["id"].strip()
        pmhc = row["pmhc"].strip()
        tcr_id = row["tcr"].strip()
        label = row["label"].strip()
        if not rid or not pmhc or not tcr_id:
            continue
        peptide = pmhc.split("_", 1)[0] if "_" in pmhc else pmhc
        cdrs = tcr_to_cdr.get(tcr_id)
        if cdrs is None:
            missing_tcr[tcr_id] = missing_tcr.get(tcr_id, 0) + 1
            continue
        a1, a2, a3, b1, b2, b3 = cdrs
        out_rows.append(
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

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    n_miss = sum(missing_tcr.values())
    print(f"Wrote {len(out_rows)} rows to {args.out}")
    if n_miss:
        print(
            f"Skipped {n_miss} rows ({len(missing_tcr)} distinct tcr) not in dic",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
