#!/usr/bin/env python3
"""Extract best-AF3-ranking confidence features for the active ePytope viral set."""

from __future__ import annotations

import argparse
import csv
import warnings
from collections import Counter
from pathlib import Path

import analyze_model_quality_metrics_common as common


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
VIRAL_DIR = REPO_ROOT / "tapas" / "tabpfn_epytope_af3"
DEFAULT_OUT_DIR = SCRIPT_DIR / "epytope_tcr_viral"
DEFAULT_STRUCTURE_ROOTS = [REPO_ROOT / "af3_outputs" / "epytope_tcr_viral"]
DEFAULT_MANIFEST = VIRAL_DIR / "data" / "manifest.csv"
OUTPUT_PAIR_SUFFIX = "_structure"

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")


def load_active_pair_ids(manifest_path: Path) -> list[str]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "job_name" not in reader.fieldnames:
            raise ValueError(f"{manifest_path} is missing the job_name column")
        pair_ids = []
        for row in reader:
            job_name = str(row.get("job_name", "")).strip()
            if not job_name:
                continue
            pair_ids.append(
                job_name
                if job_name.endswith(OUTPUT_PAIR_SUFFIX)
                else f"{job_name}{OUTPUT_PAIR_SUFFIX}"
            )
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError(f"{manifest_path} contains duplicated job_name values")
    return sorted(pair_ids)


def find_pair_dirs(
    structure_roots: list[Path], pair_ids: list[str]
) -> tuple[dict[str, Path], int]:
    pair_dirs: dict[str, Path] = {}
    for pair_id in pair_ids:
        for root in structure_roots:
            candidate = root / pair_id
            if candidate.is_dir():
                pair_dirs[pair_id] = candidate
                break
    return pair_dirs, len(pair_ids) - len(pair_dirs)


def matching_rows(
    path: Path,
    active_pairs: set[str],
    best_selection: dict[str, int],
) -> dict[str, dict]:
    rows_by_pair, _ = common.read_existing_sample_rows(path)
    matched: dict[str, dict] = {}
    for pair_id, model_rows in rows_by_pair.items():
        selected_model = best_selection.get(pair_id)
        if pair_id not in active_pairs or selected_model is None:
            continue
        row = model_rows.get(selected_model)
        if row is not None:
            matched[pair_id] = row
    return matched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--structure-root", type=Path, action="append", default=None
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    structure_roots = args.structure_root or DEFAULT_STRUCTURE_ROOTS
    pair_ids = load_active_pair_ids(args.manifest)
    if args.limit is not None:
        pair_ids = pair_ids[: args.limit]
    active_pairs = set(pair_ids)
    pair_dirs, missing_pair_dirs = find_pair_dirs(structure_roots, pair_ids)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.out_dir / "model_quality_metrics_best_af3_ranking_score.csv"
    legacy_all_csv = args.out_dir / "model_quality_metrics_all_samples.csv"
    summary_path = args.out_dir / "model_quality_metrics_summary.txt"
    common.ensure_csv_schema(output_csv)

    print("Dataset: epytope_tcr_viral", flush=True)
    print(f"Active manifest: {args.manifest}", flush=True)
    print(f"Active pairs: {len(pair_ids)}", flush=True)
    print(f"Matched AF3 pair dirs: {len(pair_dirs)}", flush=True)
    print("Mode: best_af3_ranking_only", flush=True)
    print("Selecting best AF3 ranking-score sample for each active pair...", flush=True)
    best_selection, _, ranking_stats = common.load_af3_ranking_selections(pair_dirs)
    print(f"Best ranking selections: {len(best_selection)}", flush=True)

    # Resume from the best-only output and reuse matching rows from the earlier
    # all-sample partial run without modifying that preserved CSV.
    completed = matching_rows(legacy_all_csv, active_pairs, best_selection)
    completed.update(matching_rows(output_csv, active_pairs, best_selection))
    common.write_rows(
        output_csv,
        [completed[pair_id] for pair_id in sorted(completed)],
    )

    stats: Counter[str] = Counter()
    stats["missing_pair_dirs"] = missing_pair_dirs
    stats["reused_best_rows"] = len(completed)
    written = 0

    print(f"Resume/reused best rows: {len(completed)}", flush=True)

    for index, pair_id in enumerate(pair_ids, start=1):
        if pair_id in completed:
            stats["skipped_existing_best"] += 1
        else:
            pair_dir = pair_dirs.get(pair_id)
            selected_model = best_selection.get(pair_id)
            if pair_dir is None:
                stats["missing_pair_dir"] += 1
            elif selected_model is None:
                stats["missing_ranking_selection"] += 1
            else:
                sample_dirs = {
                    common.sample_index(path): path
                    for path in common.find_sample_dirs(pair_dir)
                }
                sample_dir = sample_dirs.get(selected_model)
                if sample_dir is None:
                    stats["missing_selected_sample_dir"] += 1
                else:
                    row = common.metrics_for_sample(pair_id, sample_dir)
                    if row is None:
                        stats["missing_selected_sample_files"] += 1
                    else:
                        common.append_rows(output_csv, [row])
                        completed[pair_id] = row
                        written += 1

        if index % 100 == 0 or index == len(pair_ids):
            print(
                f"Processed {index}/{len(pair_ids)}; "
                f"completed_best={len(completed)}; written={written}; "
                f"skipped_existing={stats['skipped_existing_best']}",
                flush=True,
            )

    final_rows = [completed[pair_id] for pair_id in sorted(completed)]
    common.write_rows(output_csv, final_rows)
    summary_lines = [
        "dataset: epytope_tcr_viral",
        "mode: best_af3_ranking_only",
        f"manifest: {args.manifest}",
        f"active_pairs: {len(pair_ids)}",
        f"matched_pair_dirs: {len(pair_dirs)}",
        f"best_output_rows: {len(final_rows)}",
        f"new_best_rows: {written}",
        f"output_csv: {output_csv}",
        f"preserved_legacy_all_samples_csv: {legacy_all_csv}",
    ]
    for key, value in sorted(ranking_stats.items()):
        summary_lines.append(f"{key}: {value}")
    for key, value in sorted(stats.items()):
        summary_lines.append(f"{key}: {value}")
    summary_path.write_text("\n".join(summary_lines) + "\n")

    print(f"Saved best AF3 ranking-score samples: {output_csv} ({len(final_rows)} rows)")
    print(f"Saved summary: {summary_path}")
    if len(final_rows) != len(pair_ids):
        raise SystemExit(
            f"Incomplete best-only extraction: expected {len(pair_ids)}, got {len(final_rows)}"
        )


if __name__ == "__main__":
    main()
