#!/usr/bin/env python3
"""Write ids (pdb_id) with iptm_tcrpmhc >= threshold to a one-column CSV."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "csv",
        nargs="?",
        default=None,
        help="Input metrics CSV (default: first results_model_quality_metrics*best*.csv)",
    )
    ap.add_argument(
        "-o",
        "--out",
        default="iptm_tcrpmhc_ge_0.7_ids.csv",
        help="Output CSV path (single column: id)",
    )
    ap.add_argument(
        "--min-iptm-tcrpmhc",
        type=float,
        default=0.7,
        help="Keep rows with iptm_tcrpmhc >= this value (default: 0.7)",
    )
    args = ap.parse_args()

    in_path = args.csv
    if not in_path:
        matches = sorted(glob.glob("results_model_quality_metrics*best*.csv"))
        if not matches:
            print("No results_model_quality_metrics*best*.csv in cwd.", file=sys.stderr)
            sys.exit(1)
        in_path = matches[0]
        if len(matches) > 1:
            print(f"Using {in_path} ({len(matches)} matches; pass path explicitly if needed)")

    if not os.path.isfile(in_path):
        print(f"Not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    ids: list[str] = []
    with open(in_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            sys.exit("Empty CSV")
        id_key = "pdb_id" if "pdb_id" in reader.fieldnames else "id"
        if id_key not in reader.fieldnames or "iptm_tcrpmhc" not in reader.fieldnames:
            sys.exit(f"Need pdb_id (or id) and iptm_tcrpmhc; got {reader.fieldnames}")
        for row in reader:
            raw = row.get("iptm_tcrpmhc", "").strip()
            if not raw:
                continue
            try:
                v = float(raw)
            except ValueError:
                continue
            if v >= args.min_iptm_tcrpmhc:
                ids.append(row[id_key].strip())

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id"])
        for i in ids:
            w.writerow([i])

    print(f"Wrote {len(ids)} ids to {args.out} (from {in_path}, iptm_tcrpmhc >= {args.min_iptm_tcrpmhc})")


if __name__ == "__main__":
    main()
