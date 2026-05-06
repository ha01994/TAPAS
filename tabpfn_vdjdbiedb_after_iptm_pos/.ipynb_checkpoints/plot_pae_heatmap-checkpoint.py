#!/usr/bin/env python3
"""Plot PAE matrix from AlphaFold3-style confidences JSON.

Values are linearly scaled to [0, 1] using vmin–vmax (default 0 Å and 99th
percentile) before coloring; the colorbar tick labels show the corresponding Å.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_pae(path: str) -> tuple[np.ndarray, np.ndarray | None]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    pae = np.asarray(data["pae"], dtype=np.float64)
    chain_ids = None
    if "token_chain_ids" in data:
        chain_ids = np.asarray(data["token_chain_ids"])
    return pae, chain_ids


def chain_boundaries(chain_ids: np.ndarray) -> list[tuple[float, str]]:
    """Return x positions (between tokens) and label for the chain starting after the line."""
    if chain_ids.size == 0:
        return []
    boundaries: list[tuple[float, str]] = []
    for i in range(1, len(chain_ids)):
        if chain_ids[i] != chain_ids[i - 1]:
            boundaries.append((i - 0.5, str(chain_ids[i])))
    return boundaries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "json_path",
        nargs="?",
        default="vdjdb_full_17129_confidences.json",
        help="Path to *_confidences.json (default: vdjdb_full_17129_confidences.json)",
    )
    ap.add_argument(
        "-o",
        "--out",
        default=None,
        help="Output image path (default: <json_stem>_pae_heatmap.png)",
    )
    ap.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Normalization upper bound (Å); maps to 1 on the colormap. Default: 99th percentile",
    )
    ap.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Normalization lower bound (Å); maps to 0 on the colormap. Default: 0",
    )
    ap.add_argument(
        "--no-chain-lines",
        action="store_true",
        help="Do not draw chain boundaries from token_chain_ids",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI (default 150)",
    )
    ap.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(10.0, 9.0),
        metavar=("W", "H"),
        help="Figure size in inches (default 10 9)",
    )
    args = ap.parse_args()

    path = Path(args.json_path)
    if not path.is_file():
        print(f"Not found: {path}", file=sys.stderr)
        sys.exit(1)

    pae, chain_ids = load_pae(str(path))
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        print(f"Expected square PAE matrix, got shape {pae.shape}", file=sys.stderr)
        sys.exit(1)

    vmin = 0.0 if args.vmin is None else args.vmin
    vmax = float(np.percentile(pae, 99)) if args.vmax is None else args.vmax
    if vmax <= vmin:
        vmax = vmin + 1e-6

    # Linear min–max to [0, 1] for display (clip so out-of-range values saturate)
    pae_norm = np.clip((pae - vmin) / (vmax - vmin), 0.0, 1.0)

    out = args.out or str(path.with_name(path.stem + "_pae_heatmap.png"))

    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # Low PAE (0) -> saturated green, high PAE -> white
    cmap = LinearSegmentedColormap.from_list(
        "pae_green_white",
        #["#3CB371", "#ffffff"],
        ["#FF6347", "#ffffff"],
        N=256,
    )

    fig, ax = plt.subplots(figsize=(args.figsize[0], args.figsize[1]))
    im = ax.imshow(
        pae_norm,
        origin="upper",
        aspect="equal",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax.axis("off")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("PAE (Å)")
    # Colorbar 0–1 ↔ same linear scale as vmin–vmax (Å)
    tick_t = np.linspace(0.0, 1.0, 5)
    cbar.set_ticks(tick_t)
    cbar.set_ticklabels([f"{vmin + t * (vmax - vmin):.1f}" for t in tick_t])

    if chain_ids is not None and not args.no_chain_lines:
        for x, _ in chain_boundaries(chain_ids):
            ax.axvline(x, color="k", linewidth=0.6, alpha=0.45)
            ax.axhline(x, color="k", linewidth=0.6, alpha=0.45)

    fig.tight_layout()
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(
        f"Wrote {out}  (normalized to [0,1] from {vmin:.2f}–{vmax:.2f} Å)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
