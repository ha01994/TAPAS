#!/usr/bin/env python3
"""Histogram of iptm_tcrpmhc with vertical line at 0.7."""
from __future__ import annotations
import argparse
import csv
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot iptm_tcrpmhc distribution.")
    ap.add_argument(
        "csv",
        nargs="?",
        default="results_model_quality_metrics_vdjdbiedb_best.csv",
        help="Input CSV (default: results_model_quality_metrics_vdjdbiedb_best.csv)",
    )
    ap.add_argument(
        "-o",
        "--out",
        default="iptm_tcrpmhc_histogram.png",
        help="Output image path",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Vertical line x position (default: 0.7)",
    )
    ap.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Histogram bin count",
    )
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("matplotlib is required: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    path = args.csv
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    values: list[float] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "iptm_tcrpmhc" not in reader.fieldnames:
            print("Column iptm_tcrpmhc missing.", file=sys.stderr)
            sys.exit(1)
        for row in reader:
            raw = row.get("iptm_tcrpmhc", "").strip()
            if not raw:
                continue
            try:
                values.append(float(raw))
            except ValueError:
                continue

    if not values:
        print("No numeric iptm_tcrpmhc values.", file=sys.stderr)
        sys.exit(1)

    thr = args.threshold
    below = [v for v in values if v < thr]
    above = [v for v in values if v >= thr]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # teal / lightgray (추천)
    ax.hist(below, bins=args.bins, color="#95A5A6", edgecolor="white",
            linewidth=0.4, alpha=0.9, label=f"ipTM < {thr:g} (N={len(below)})")
    ax.hist(above, bins=args.bins, color="#1A9E8F", edgecolor="white",
            linewidth=0.4, alpha=0.9, label=f"ipTM ≥ {thr:g} (N={len(above)})")

    ax.axvline(thr, color="firebrick", linestyle="--",
               linewidth=1.2, zorder=5)

    ymax = ax.get_ylim()[1]
    ax.text(thr + 0.005, ymax * 0.97,
            f"threshold = {thr:g}",
            color="firebrick", fontsize=9, va="top", ha="left")

    ax.set_xlabel("ipTM", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("VDJDBIEDB ipTM distribution (positive data, N=4743)", fontsize=12, pad=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax.legend(frameon=False, fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(
        f"Saved {args.out} "
        f"({len(values)} values, mean={sum(values)/len(values):.4f}, "
        f"above_thr={len(above)} ({100*len(above)/len(values):.1f}%))"
    )


if __name__ == "__main__":
    main()
