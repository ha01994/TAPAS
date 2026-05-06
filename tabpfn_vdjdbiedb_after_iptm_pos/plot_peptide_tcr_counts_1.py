#!/usr/bin/env python3
"""
From iptm_tcrpmhc_ge_0.7_ids.csv (columns id, pmhc, tcr, label):
  - peptide = pmhc split on first '_'
  - peptide -> number of distinct TCRs (set of tcr per peptide)
  - save mapping as JSON and draw a sorted vertical bar chart (peptide on x-axis).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict


def load_peptide_tcr_counts(path: str) -> dict[str, int]:
    tcr_by_peptide: dict[str, set[str]] = defaultdict(set)
    z = 0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("empty csv")
        need = {"pmhc", "tcr"}
        if not need.issubset(set(reader.fieldnames)):
            raise ValueError(f"need columns {need}, got {reader.fieldnames}")
        for row in reader:
            z+=1
            pmhc = row["pmhc"].strip()
            tcr = row["tcr"].strip()
            if not pmhc or not tcr:
                continue
            peptide = pmhc.split("_", 1)[0]
            tcr_by_peptide[peptide].add(tcr)
    return {p: len(tcrs) for p, tcrs in sorted(tcr_by_peptide.items())},z


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "csv",
        nargs="?",
        default="iptm_tcrpmhc_ge_0.7_ids.csv",
        help="Enriched iptm id list (needs pmhc, tcr)",
    )
    ap.add_argument(
        "-o",
        "--plot",
        default="peptide_tcr_counts_iptm_ge07.png",
        help="Output bar plot path",
    )
    ap.add_argument(
        "--json-out",
        default = "peptide_tcr_counts_iptm_ge07.json",
        help="Save peptide -> distinct tcr count as JSON (empty string to skip)",
    )
    ap.add_argument(
        "--fig-width",
        type=float,
        default=None,
        help="Figure width in inches (default: max(18, 0.14 * n_peptides))",
    )
    ap.add_argument(
        "--fig-height",
        type=float,
        default=7.0,
        help="Figure height in inches (default: 7)",
    )
    ap.add_argument(
        "--label-rotation",
        type=float,
        default=90.0,
        help="X tick label rotation in degrees (default: 90)",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        print(f"Not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    counts, z = load_peptide_tcr_counts(args.csv)
    if not counts:
        print("No rows with pmhc/tcr.", file=sys.stderr)
        sys.exit(1)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(counts, f, indent=2, ensure_ascii=False)
        print(f"Wrote dictionary ({len(counts)} peptides) to {args.json_out}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    # sort by count descending for plot
    items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    peptides = [x[0] for x in items]
    values = [x[1] for x in items]

    n = len(peptides)
    #fig_w = args.fig_width if args.fig_width is not None else max(18.0, 0.14 * n)
    #fig, ax = plt.subplots(figsize=(fig_w, args.fig_height))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(n)
    ax.bar(x, values, color="steelblue", alpha=0.9, edgecolor="white", linewidth=0.3)
    ax.set_xticks(list(x))
    tick_ha = "center" if abs(args.label_rotation - 90.0) < 1e-6 else "right"
    ax.set_xticklabels(
        peptides, fontsize=7, rotation=args.label_rotation, ha=tick_ha
    )
    print(z)    
    ax.set_xlabel("Peptide")
    ax.set_ylabel("TCR count")
    ax.set_title(f"Per-peptide TCR count after ipTM filtering (n={n} peptides, total={z})")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(args.plot, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved bar plot to {args.plot}")


if __name__ == "__main__":
    main()
