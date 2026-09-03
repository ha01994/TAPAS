#!/usr/bin/env python3
"""Extract best-AF3-ranking geometry features for the active ePytope viral set."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import extract_af3_geometry_features_immrep25 as extractor


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
VIRAL_DIR = REPO_ROOT / "tapas" / "tabpfn_epytope_af3"
DEFAULT_OUTPUT_DIRS = [REPO_ROOT / "af3_outputs" / "epytope_tcr_viral"]
DEFAULT_MANIFEST = VIRAL_DIR / "data" / "manifest.csv"
DEFAULT_TCR_LOOKUP = VIRAL_DIR / "data" / "tcr_sequences.csv"
DEFAULT_OUT_DIR = SCRIPT_DIR / "epytope_tcr_viral"
OUTPUT_PAIR_SUFFIX = "_structure"
DATASET = "epytope_tcr_viral"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require_columns(path: Path, rows: list[dict[str, str]], required: set[str]) -> None:
    columns = set(rows[0]) if rows else set()
    missing = required - columns
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")


def load_requested_pairs(
    manifest_path: Path, tcr_lookup_path: Path
) -> tuple[list[dict[str, str]], dict[str, int]]:
    manifest = read_csv(manifest_path)
    tcr_rows = read_csv(tcr_lookup_path)
    require_columns(
        manifest_path,
        manifest,
        {"job_name", "tcr_index", "label", "cdr3_alpha", "cdr3_beta"},
    )
    require_columns(
        tcr_lookup_path,
        tcr_rows,
        {"tcr_index", "tra_sequence", "trb_sequence"},
    )

    tcr_lookup = {row["tcr_index"].strip(): row for row in tcr_rows}
    stats: Counter[str] = Counter(
        manifest_rows=len(manifest),
        tcr_lookup_rows=len(tcr_lookup),
    )
    pairs: list[dict[str, str]] = []
    seen_pair_ids: set[str] = set()
    for row in manifest:
        tcr_index = row["tcr_index"].strip()
        tcr = tcr_lookup.get(tcr_index)
        if tcr is None:
            stats["missing_tcr_metadata"] += 1
            continue
        label = row["label"].strip()
        if label not in {"0", "1"}:
            stats["invalid_label"] += 1
            continue
        job_name = row["job_name"].strip()
        pair_id = (
            job_name
            if job_name.endswith(OUTPUT_PAIR_SUFFIX)
            else f"{job_name}{OUTPUT_PAIR_SUFFIX}"
        )
        if not job_name or pair_id in seen_pair_ids:
            stats["missing_or_duplicate_pair_id"] += 1
            continue
        seen_pair_ids.add(pair_id)
        cdr3a = row["cdr3_alpha"].strip()
        cdr3b = row["cdr3_beta"].strip()
        pairs.append(
            {
                "pair_id": pair_id,
                "dataset": DATASET,
                "label": label,
                "tra_seq": tcr["tra_sequence"].strip(),
                "trb_seq": tcr["trb_sequence"].strip(),
                "cdr3a_len": str(len(cdr3a)),
                "cdr3b_len": str(len(cdr3b)),
                "cdr3a_seq": cdr3a,
                "cdr3b_seq": cdr3b,
            }
        )
    stats["mapped_pairs"] = len(pairs)
    stats["positive_pairs"] = sum(pair["label"] == "1" for pair in pairs)
    stats["negative_pairs"] = sum(pair["label"] == "0" for pair in pairs)
    return pairs, dict(stats)


def matching_sample_rows(
    path: Path,
    sample_fieldnames: list[str],
    active_pairs: set[str],
    best_selection: dict[str, int],
) -> dict[str, dict[str, object]]:
    rows, _, _ = extractor.read_existing_sample_rows(path, sample_fieldnames)
    matched: dict[str, dict[str, object]] = {}
    for row in rows:
        pair_id = str(row["pair_id"])
        model_number = row.get("model_number")
        if (
            pair_id in active_pairs
            and isinstance(model_number, int)
            and best_selection.get(pair_id) == model_number
        ):
            matched[pair_id] = row
    return matched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, action="append", default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tcr-sequences", type=Path, default=DEFAULT_TCR_LOOKUP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output_dirs = args.output_dir or DEFAULT_OUTPUT_DIRS
    pairs, input_stats = load_requested_pairs(args.manifest, args.tcr_sequences)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    pair_by_id = {pair["pair_id"]: pair for pair in pairs}
    pair_ids = set(pair_by_id)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = extractor.REDUCED_FEATURE_COLUMNS
    sample_fieldnames = extractor.sample_feature_columns(fieldnames)
    progress_csv = args.out_dir / "geometry_features_best_af3_ranking_score_samples.csv"
    legacy_all_csv = args.out_dir / "geometry_features_all_samples.csv"
    output_csv = args.out_dir / "geometry_features_best_af3_ranking_score.csv"
    summary_path = args.out_dir / "extraction_summary.txt"

    best_selection, _, ranking_stats = extractor.load_af3_ranking_selections(
        output_dirs, pair_ids
    )
    completed = matching_sample_rows(
        legacy_all_csv, sample_fieldnames, pair_ids, best_selection
    )
    completed.update(
        matching_sample_rows(progress_csv, sample_fieldnames, pair_ids, best_selection)
    )
    extractor.write_rows(
        progress_csv,
        [completed[pair_id] for pair_id in sorted(completed)],
        sample_fieldnames,
    )

    incomplete_pair_ids = pair_ids - set(completed)
    sample_index = extractor.build_sample_file_index(output_dirs, incomplete_pair_ids)
    stats: Counter[str] = Counter()
    stats["reused_best_rows"] = len(completed)
    written = 0

    print(f"Active manifest: {args.manifest}", flush=True)
    print(f"Active pairs: {len(pairs)}", flush=True)
    print("Mode: best_af3_ranking_only", flush=True)
    print(f"Resume/reused best rows: {len(completed)}", flush=True)

    for index, pair_id in enumerate(sorted(incomplete_pair_ids), start=1):
        selected_model = best_selection.get(pair_id)
        sample = sample_index.get(pair_id, {}).get(selected_model)
        if selected_model is None:
            stats["missing_ranking_selection"] += 1
        elif sample is None:
            stats["missing_selected_sample"] += 1
        else:
            model_path, conf_path = sample
            row = extractor.geometry_for_job(
                pair_by_id[pair_id],
                "baseline_default",
                model_path,
                conf_path,
            )
            if row is None:
                stats["failed_selected_geometry"] += 1
            else:
                row = extractor.reduce_feature_row(row)
                row["model_number"] = selected_model
                extractor.append_rows(progress_csv, [row], sample_fieldnames)
                completed[pair_id] = row
                written += 1

        if index % 100 == 0 or index == len(incomplete_pair_ids):
            print(
                f"Processed {index}/{len(incomplete_pair_ids)} incomplete pairs; "
                f"completed_best={len(completed)}; written={written}",
                flush=True,
            )

    sample_rows = [completed[pair_id] for pair_id in sorted(completed)]
    selected_rows, selection_stats = extractor.select_af3_ranking_geometry_rows(
        sample_rows,
        best_selection,
        fieldnames,
        "best",
    )
    extractor.write_rows(output_csv, selected_rows, fieldnames)

    summary_lines = [
        "dataset: epytope_tcr_viral",
        "mode: best_af3_ranking_only",
        f"manifest: {args.manifest}",
        f"active_pairs: {len(pairs)}",
        f"best_output_rows: {len(selected_rows)}",
        f"new_best_rows: {written}",
        f"progress_csv: {progress_csv}",
        f"output_csv: {output_csv}",
        f"preserved_legacy_all_samples_csv: {legacy_all_csv}",
    ]
    for source in (input_stats, ranking_stats, selection_stats, stats):
        for key, value in sorted(source.items()):
            summary_lines.append(f"{key}: {value}")
    summary_path.write_text("\n".join(summary_lines) + "\n")

    print(f"Saved best AF3 ranking-score geometry: {output_csv} ({len(selected_rows)} rows)")
    print(f"Saved resume rows: {progress_csv}")
    print(f"Saved summary: {summary_path}")
    if len(selected_rows) != len(pairs):
        raise SystemExit(
            f"Incomplete best-only extraction: expected {len(pairs)}, got {len(selected_rows)}"
        )


if __name__ == "__main__":
    main()
