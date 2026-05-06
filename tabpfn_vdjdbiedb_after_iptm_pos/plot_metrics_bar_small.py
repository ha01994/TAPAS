#!/usr/bin/env python3
"""Very small bar plot for a few model quality metrics (0–1 scale)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_METRICS: dict[str, float] = {
    "pLDDT": 0.893,
    "ipTM": 0.829,
    "iPAE": 0.783,
    "pDockQ": 0.687,
    "pDockQ2": 0.782,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-o",
        "--out",
        default="metrics_bar_small.png",
        help="Output image path",
    )
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    names = list(DEFAULT_METRICS.keys())[::-1]
    values = [DEFAULT_METRICS[k] for k in names]
    y = np.arange(len(names))

    # Small figure: wide enough for labels, short height
    fig, ax = plt.subplots(figsize=(5.2,2.8), layout="constrained")
    ax.barh(y, values, height=0.55, color="#4682B4", edgecolor="none")
    ax.set_yticks(y, names)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="x", length=2, pad=1)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.4)

    # Value labels at bar end (tiny)
    for yi, v in zip(y, values):
        ax.text(
            min(v + 0.02, 0.98),
            yi,
            f"{v:.3f}",
            va="center",
            ha="left",            
        )

    fig.savefig(Path(args.out), dpi=args.dpi, facecolor="white")
    plt.close(fig)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
