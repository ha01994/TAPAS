#!/usr/bin/env python3
"""
2×1 combined bar charts:
  - Top: after ipTM filtering (iptm_ge0.7 list), same logic as plot_peptide_tcr_counts_1.py
  - Bottom: before filtering (full parsed table), same logic as plot_peptide_tcr_counts_2.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt


def load_peptide_tcr_counts(path: str) -> tuple[dict[str, int], int]:
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
            z += 1
            pmhc = row["pmhc"].strip()
            tcr = row["tcr"].strip()
            if not pmhc or not tcr:
                continue
            peptide = pmhc.split("_", 1)[0]
            tcr_by_peptide[peptide].add(tcr)
    return {p: len(tcrs) for p, tcrs in sorted(tcr_by_peptide.items())}, z


def draw_panel(
    ax,
    counts: dict[str, int],
    total_rows: int,
    title: str,
    label_rotation: float,
) -> None:
    items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    peptides = [x[0] for x in items]
    values = [x[1] for x in items]
    n = len(peptides)
    x = range(n)
    ax.bar(x, values, color="steelblue", alpha=0.9, edgecolor="white", linewidth=0.3)
    ax.set_xticks(list(x))
    tick_ha = "center" if abs(label_rotation - 90.0) < 1e-6 else "right"
    ax.set_xticklabels(peptides, fontsize=7, rotation=label_rotation, ha=tick_ha)
    ax.set_xlabel("Peptide")
    ax.set_ylabel("TCR count")
    ax.set_title(f"{title} (n={n} peptides, total rows={total_rows})")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)


def add_panel_letter(ax, letter: str, fontsize: float) -> None:
    ax.text(
        -0.05,
        1.07,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv-after",
        default="iptm_tcrpmhc_ge_0.7_ids.csv",
        help="CSV after ipTM filter (needs pmhc, tcr); top panel",
    )
    ap.add_argument(
        "--csv-before",
        default="parsed_data_freq10_cap100.csv",
        help="CSV before ipTM filter (needs pmhc, tcr); bottom panel",
    )
    ap.add_argument(
        "-o",
        "--plot",
        default="peptide_tcr_counts_combined.png",
        help="Output combined figure path",
    )
    ap.add_argument(
        "--json-out-after",
        default="peptide_tcr_counts_iptm_ge07.json",
        help="JSON for after-filter counts (empty string to skip)",
    )
    ap.add_argument(
        "--json-out-before",
        default="peptide_tcr_counts_parsed_data_freq10_cap100.json",
        help="JSON for before-filter counts (empty string to skip)",
    )
    ap.add_argument(
        "--fig-width",
        type=float,
        default=10.0,
        help="Figure width in inches",
    )
    ap.add_argument(
        "--fig-height",
        type=float,
        default=11.0,
        help="Figure height in inches (2 rows; default 11)",
    )
    ap.add_argument(
        "--panel-letter-size",
        type=float,
        default=11.0,
        help="Font size for (a)/(b) (default 11)",
    )
    ap.add_argument(
        "--label-rotation",
        type=float,
        default=90.0,
        help="X tick label rotation in degrees",
    )
    args = ap.parse_args()

    for label, path in (
        ("--csv-after", args.csv_after),
        ("--csv-before", args.csv_before),
    ):
        if not os.path.isfile(path):
            print(f"Not found {label}: {path}", file=sys.stderr)
            sys.exit(1)

    counts_after, z_after = load_peptide_tcr_counts(args.csv_after)
    counts_before, z_before = load_peptide_tcr_counts(args.csv_before)
    if not counts_after or not counts_before:
        print("One or both CSVs produced no peptide counts.", file=sys.stderr)
        sys.exit(1)

    if args.json_out_after:
        with open(args.json_out_after, "w", encoding="utf-8") as f:
            json.dump(counts_after, f, indent=2, ensure_ascii=False)
        print(f"Wrote {args.json_out_after} ({len(counts_after)} peptides)")
    if args.json_out_before:
        with open(args.json_out_before, "w", encoding="utf-8") as f:
            json.dump(counts_before, f, indent=2, ensure_ascii=False)
        print(f"Wrote {args.json_out_before} ({len(counts_before)} peptides)")

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(args.fig_width, args.fig_height),
        layout="constrained",
    )
    draw_panel(
        ax_top,
        counts_after,
        z_after,
        "After ipTM filtering",
        args.label_rotation,
    )
    draw_panel(
        ax_bottom,
        counts_before,
        z_before,
        "Before ipTM filtering",
        args.label_rotation,
    )
    add_panel_letter(ax_top, "a", args.panel_letter_size)
    add_panel_letter(ax_bottom, "b", args.panel_letter_size)

    fig.savefig(args.plot, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined plot to {args.plot}")


if __name__ == "__main__":
    main()
