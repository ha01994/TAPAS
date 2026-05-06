#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results",
        default="results_model_quality_metrics_vdjdbiedb_best.csv",
    )
    ap.add_argument(
        "--iptm-filtered",
        default="iptm_filtered_vdjdbiedb.csv",
        help="CSV with id column (matches results pdb_id)",
    )
    ap.add_argument(
        "-o",
        "--out",
        default="results_vdjdbiedb_iptm_filtered_best.csv",
    )
    args = ap.parse_args()

    for p in (args.results, args.iptm_filtered):
        if not os.path.isfile(p):
            print(f"Not found: {p}", file=sys.stderr)
            sys.exit(1)

    keep: set[str] = set()
    with open(args.iptm_filtered, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rid = row.get("id", "").strip()
            if rid:
                keep.add(rid)

    with open(args.results, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "pdb_id" not in reader.fieldnames:
            sys.exit("results CSV must have pdb_id")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    out_rows = [r for r in rows if r.get("pdb_id", "").strip() in keep]
    pdb_out = {r["pdb_id"].strip() for r in out_rows}
    iptm_without_results = len(keep - pdb_out)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print(
        f"Wrote {len(out_rows)} rows to {args.out} "
        f"(iptm_filtered ids: {len(keep)}; results rows before: {len(rows)}; "
        f"iptm ids with no matching pdb_id in results: {iptm_without_results})"
    )


if __name__ == "__main__":
    main()
