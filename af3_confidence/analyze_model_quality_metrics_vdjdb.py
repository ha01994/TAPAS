#!/usr/bin/env python3
"""Analyze VDJDB AF3 confidence metrics and select median/ranking-best samples."""

from __future__ import annotations

import argparse
from pathlib import Path

from analyze_model_quality_metrics_common import add_common_args, parse_roots, run_analysis


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OUT_DIR = SCRIPT_DIR / "vdjdb"
DEFAULT_STRUCTURE_ROOTS = [
    REPO_ROOT / "af3_outputs" / "vdjdb_positive",
    REPO_ROOT / "af3_outputs" / "vdjdb_negative_part1",
    REPO_ROOT / "af3_outputs" / "vdjdb_negative_part2",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, DEFAULT_OUT_DIR, DEFAULT_STRUCTURE_ROOTS)
    args = parser.parse_args()
    run_analysis(
        dataset="vdjdb",
        structure_roots=parse_roots(args),
        out_dir=args.out_dir,
        limit=args.limit,
        selection_metric=args.selection_metric,
    )


if __name__ == "__main__":
    main()
