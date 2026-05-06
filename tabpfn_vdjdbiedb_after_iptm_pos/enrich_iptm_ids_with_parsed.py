#!/usr/bin/env python3
"""Join iptm id list with parsed_data (no header): id,pmhc,tcr,label -> enriched CSV."""

from __future__ import annotations

import argparse
import csv
import os
import sys


def load_parsed(path: str) -> dict[str, tuple[str, str, str]]:
    out: dict[str, tuple[str, str, str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            rid, pmhc, tcr, label = parts[0], parts[1], parts[2], parts[3]
            out[rid] = (pmhc, tcr, label)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ids",
        default="iptm_tcrpmhc_ge_0.7_ids.csv",
        help="CSV with header id (one column)",
    )
    ap.add_argument(
        "--parsed",
        default="parsed_data_freq10_cap100.csv",
        help="Parsed data: id,pmhc,tcr,label per line (no header)",
    )
    ap.add_argument(
        "-o",
        "--out",
        default=None,
        help="Output path (default: overwrite --ids)",
    )
    args = ap.parse_args()
    out_path = args.out or args.ids

    if not os.path.isfile(args.ids):
        print(f"Not found: {args.ids}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.parsed):
        print(f"Not found: {args.parsed}", file=sys.stderr)
        sys.exit(1)

    parsed = load_parsed(args.parsed)

    order: list[str] = []
    with open(args.ids, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "id" not in reader.fieldnames:
            sys.exit("ids file must have header with column id")
        for row in reader:
            rid = row["id"].strip()
            if rid:
                order.append(rid)

    missing = [rid for rid in order if rid not in parsed]
    if missing:
        print(f"Warning: {len(missing)} ids not in parsed (e.g. {missing[:3]})", file=sys.stderr)

    fieldnames = ["id", "pmhc", "tcr", "label"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rid in order:
            if rid not in parsed:
                w.writerow({"id": rid, "pmhc": "", "tcr": "", "label": ""})
                continue
            pmhc, tcr, label = parsed[rid]
            w.writerow({"id": rid, "pmhc": pmhc, "tcr": tcr, "label": label})

    print(f"Wrote {len(order)} rows to {out_path}")


if __name__ == "__main__":
    main()
