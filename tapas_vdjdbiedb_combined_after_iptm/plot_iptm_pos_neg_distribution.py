#!/usr/bin/env python3
"""Overlay histogram of iptm_tcrpmhc for pos vs neg CSVs (VDJDBIEDB filtered best)."""
from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot iptm_tcrpmhc distribution: positive vs negative samples."
    )
    ap.add_argument(
        "--pos",
        default="results_vdjdbiedb_iptm_filtered_pos_best.csv",
        help="Positive-class metrics CSV",
    )
    ap.add_argument(
        "--neg",
        default="results_vdjdbiedb_iptm_filtered_neg_best.csv",
        help="Negative-class metrics CSV",
    )
    ap.add_argument(
        "-o",
        "--out",
        default="iptm_tcrpmhc_pos_neg_distribution.png",
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
    ap.add_argument(
        "--density",
        action="store_true",
        help="Normalize each histogram to area 1 (useful when pos/neg counts differ)",
    )
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import numpy as np
        import pandas as pd
    except ImportError as e:
        print(f"Missing dependency: {e}", file=sys.stderr)
        print("Install: pip install matplotlib pandas numpy", file=sys.stderr)
        sys.exit(1)

    col = "iptm_tcrpmhc"
    for path, label in [(args.pos, "pos"), (args.neg, "neg")]:
        if not os.path.isfile(path):
            print(f"File not found ({label}): {path}", file=sys.stderr)
            sys.exit(1)

    df_pos = pd.read_csv(args.pos)
    df_neg = pd.read_csv(args.neg)
    if col not in df_pos.columns or col not in df_neg.columns:
        print(f"Column {col!r} missing in one or both CSVs.", file=sys.stderr)
        sys.exit(1)

    v_pos = pd.to_numeric(df_pos[col], errors="coerce").dropna().to_numpy(dtype=float)
    v_neg = pd.to_numeric(df_neg[col], errors="coerce").dropna().to_numpy(dtype=float)
    if len(v_pos) == 0 or len(v_neg) == 0:
        print("No numeric iptm_tcrpmhc values in pos or neg.", file=sys.stderr)
        sys.exit(1)

    lo = float(min(v_pos.min(), v_neg.min()))
    hi = float(max(v_pos.max(), v_neg.max()))
    pad = (hi - lo) * 0.02 + 1e-6
    bins = np.linspace(lo - pad, hi + pad, args.bins + 1)

    # Neg first (often wider spread), pos on top — blue / red palette
    color_neg = "#C84C4C"  # warm red
    color_pos = "#3A7CA5"  # steel blue
    edge = "white"
    lw = 0.4
    alpha = 0.82

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    weights_kw = {}
    if args.density:
        weights_kw = {"density": True}

    ax.hist(
        v_neg,
        bins=bins,
        color=color_neg,
        edgecolor=edge,
        linewidth=lw,
        alpha=alpha,
        label=f"Negative (n={len(v_neg):,})",
        zorder=1,
        **weights_kw,
    )
    ax.hist(
        v_pos,
        bins=bins,
        color=color_pos,
        edgecolor=edge,
        linewidth=lw,
        alpha=alpha,
        label=f"Positive (n={len(v_pos):,})",
        zorder=2,
        **weights_kw,
    )

    thr = args.threshold
    ax.axvline(
        thr,
        color="firebrick",
        linestyle="--",
        linewidth=1.2,
        zorder=5,
    )

    ymax = ax.get_ylim()[1]
    ax.text(
        thr + (hi - lo) * 0.01,
        ymax * 0.97,
        f"threshold = {thr:g}",
        color="firebrick",
        fontsize=9,
        va="top",
        ha="left",
    )

    ax.set_xlabel("ipTM", fontsize=11)
    ylab = "Density" if args.density else "Count"
    ax.set_ylabel(ylab, fontsize=11)
    ax.set_title("VDJDBIEDB ipTM distribution (pos/neg) after filtering", fontsize=12, pad=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    if not args.density:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax.legend(frameon=False, fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    m_pos, m_neg = float(v_pos.mean()), float(v_neg.mean())
    print(
        f"Saved {args.out}\n"
        f"  Positive:  n={len(v_pos):,}, mean={m_pos:.4f}\n"
        f"  Negative:  n={len(v_neg):,}, mean={m_neg:.4f}"
    )


if __name__ == "__main__":
    main()
