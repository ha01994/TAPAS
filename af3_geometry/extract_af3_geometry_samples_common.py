#!/usr/bin/env python3
"""Shared helpers for AF3 geometry extraction over all sample directories."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


META_COLUMNS = {"pair_id", "dataset", "label", "condition"}


def sample_feature_columns(reduced_columns: list[str]) -> list[str]:
    return reduced_columns[:4] + ["model_number"] + reduced_columns[4:]


def sample_index(path: Path) -> int:
    match = re.search(r"sample-(\d+)", path.name)
    if match:
        return int(match.group(1))
    return 10**9


def find_sample_dirs(pair_dir: Path) -> list[Path]:
    if not pair_dir.is_dir():
        return []
    return sorted(
        [
            path
            for path in pair_dir.iterdir()
            if path.is_dir() and re.search(r"sample-\d+", path.name)
        ],
        key=lambda path: (sample_index(path), path.name),
    )


def choose_file(sample_dir: Path, pair_id: str, suffix: str) -> Path | None:
    candidates = [
        sample_dir / f"{pair_id}_{suffix}",
        sample_dir / suffix,
    ]
    for path in candidates:
        if path.exists():
            return path

    matches = sorted(sample_dir.glob(f"*_{suffix}"))
    if suffix == "confidences.json":
        matches = [path for path in matches if "summary_confidences" not in path.name]
    if matches:
        return matches[0]

    matches = sorted(sample_dir.glob(f"*{suffix}"))
    if suffix == "confidences.json":
        matches = [path for path in matches if "summary_confidences" not in path.name]
    if matches:
        return matches[0]

    return None


def find_sample_files(sample_dir: Path, pair_id: str) -> tuple[Path | None, Path | None]:
    model_path = choose_file(sample_dir, pair_id, "model.cif")
    conf_path = choose_file(sample_dir, pair_id, "confidences.json")
    return model_path, conf_path


def build_sample_file_index(
    output_dirs: list[Path],
    pair_ids: set[str] | None = None,
) -> dict[str, dict[int, tuple[Path, Path | None]]]:
    sample_files: dict[str, dict[int, tuple[Path, Path | None]]] = {}
    if pair_ids is not None:
        sorted_pair_ids = sorted(pair_ids)
        total = len(sorted_pair_ids)
        print(f"Indexing AF3 samples for {total} incomplete pairs...", flush=True)
        for pair_index, pair_id in enumerate(sorted_pair_ids, start=1):
            pair_dir = None
            for output_dir in output_dirs:
                candidate = output_dir / pair_id
                if candidate.is_dir():
                    pair_dir = candidate
                    break
            if pair_dir is not None:
                for sample_dir in find_sample_dirs(pair_dir):
                    model_path, conf_path = find_sample_files(sample_dir, pair_id)
                    if model_path is None:
                        continue
                    model_number = sample_index(sample_dir)
                    sample_files.setdefault(pair_id, {})[model_number] = (
                        model_path,
                        conf_path if conf_path and conf_path.exists() else None,
                    )
            if pair_index % 1000 == 0 or pair_index == total:
                print(
                    f"Indexed incomplete AF3 pairs: {pair_index}/{total} "
                    f"(matched: {len(sample_files)})",
                    flush=True,
                )
        return sample_files

    for output_dir in output_dirs:
        if not output_dir.exists():
            continue
        for pair_dir in output_dir.iterdir():
            if not pair_dir.is_dir():
                continue
            pair_id = pair_dir.name
            for sample_dir in find_sample_dirs(pair_dir):
                model_path, conf_path = find_sample_files(sample_dir, pair_id)
                if model_path is None:
                    continue
                model_number = sample_index(sample_dir)
                sample_files.setdefault(pair_id, {}).setdefault(
                    model_number,
                    (model_path, conf_path if conf_path and conf_path.exists() else None),
                )
    return sample_files


def format_csv_value(value: object) -> object:
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return "nan"
        return f"{float(value):.6f}"
    return value


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def load_af3_ranking_selections(
    output_dirs: list[Path],
    pair_ids: set[str],
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Map each pair to its highest and median AF3 ranking_score samples."""
    best_selected: dict[str, int] = {}
    median_selected: dict[str, int] = {}
    stats = {
        "af3_ranking_pair_dirs": 0,
        "best_af3_ranking_selection_rows": 0,
        "median_af3_ranking_selection_rows": 0,
        "af3_ranking_missing_pair_dirs": 0,
        "af3_ranking_missing_scores": 0,
    }

    sorted_pair_ids = sorted(pair_ids)
    total = len(sorted_pair_ids)
    print(f"Selecting AF3 ranking-score models for {total} existing pairs...", flush=True)
    for pair_index, pair_id in enumerate(sorted_pair_ids, start=1):
        pair_dir = None
        for output_dir in output_dirs:
            candidate = output_dir / pair_id
            if candidate.is_dir():
                pair_dir = candidate
                break
        if pair_dir is None:
            stats["af3_ranking_missing_pair_dirs"] += 1
            if pair_index % 1000 == 0 or pair_index == total:
                print(
                    f"Selected AF3 ranking-score models: {pair_index}/{total} "
                    f"(matched: {len(best_selected)})",
                    flush=True,
                )
            continue

        stats["af3_ranking_pair_dirs"] += 1
        scores: dict[int, float] = {}
        ranking_paths = sorted(pair_dir.glob("*_ranking_scores.csv"))
        ranking_paths.extend(sorted(pair_dir.glob("ranking_scores.csv")))
        for ranking_path in ranking_paths:
            with ranking_path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    model_number = safe_int(row.get("sample"))
                    score = safe_float(row.get("ranking_score"))
                    if model_number is None or math.isnan(score):
                        continue
                    scores[model_number] = max(scores.get(model_number, -math.inf), score)

        if not scores:
            for sample_dir in find_sample_dirs(pair_dir):
                model_number = sample_index(sample_dir)
                summary_paths = sorted(sample_dir.glob("*_summary_confidences.json"))
                summary_paths.extend(sorted(sample_dir.glob("summary_confidences.json")))
                for summary_path in summary_paths:
                    try:
                        summary = json.loads(summary_path.read_text())
                    except (OSError, json.JSONDecodeError):
                        continue
                    score = safe_float(summary.get("ranking_score"))
                    if math.isnan(score):
                        continue
                    scores[model_number] = max(scores.get(model_number, -math.inf), score)

        if not scores:
            stats["af3_ranking_missing_scores"] += 1
            if pair_index % 1000 == 0 or pair_index == total:
                print(
                    f"Selected AF3 ranking-score models: {pair_index}/{total} "
                    f"(matched: {len(best_selected)})",
                    flush=True,
                )
            continue

        best_model = min(scores, key=lambda model: (-scores[model], model))
        median_score = float(np.median(list(scores.values())))
        median_model = min(
            scores,
            key=lambda model: (abs(scores[model] - median_score), model),
        )
        best_selected[pair_id] = best_model
        median_selected[pair_id] = median_model
        if pair_index % 1000 == 0 or pair_index == total:
            print(
                f"Selected AF3 ranking-score models: {pair_index}/{total} "
                f"(matched: {len(best_selected)})",
                flush=True,
            )

    stats["best_af3_ranking_selection_rows"] = len(best_selected)
    stats["median_af3_ranking_selection_rows"] = len(median_selected)
    return best_selected, median_selected, stats


def move_schema_mismatch(path: Path, fieldnames: list[str]) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
    if header == fieldnames:
        return False

    backup = path.with_suffix(path.suffix + ".old_schema")
    counter = 1
    while backup.exists():
        backup = path.with_suffix(path.suffix + f".old_schema{counter}")
        counter += 1
    path.rename(backup)
    print(f"Existing CSV schema differs; moved old file to {backup}", flush=True)
    return True


def parse_feature_row(row: dict[str, str], fieldnames: list[str]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for key in fieldnames:
        value = row.get(key, "")
        if key in META_COLUMNS:
            parsed[key] = value
        elif key == "model_number":
            parsed[key] = safe_int(value)
        else:
            parsed[key] = safe_float(value)
    return parsed


def read_existing_sample_rows(
    path: Path,
    fieldnames: list[str],
) -> tuple[list[dict[str, object]], set[tuple[str, int]], bool]:
    if move_schema_mismatch(path, fieldnames):
        return [], set(), False
    if not path.exists() or path.stat().st_size == 0:
        return [], set(), False

    rows: list[dict[str, object]] = []
    processed: set[tuple[str, int]] = set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = parse_feature_row(row, fieldnames)
            model_number = parsed.get("model_number")
            if not isinstance(model_number, int):
                continue
            pair_id = str(parsed["pair_id"])
            rows.append(parsed)
            processed.add((pair_id, model_number))
    return rows, processed, True


def append_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    append = path.exists() and path.stat().st_size > 0
    with path.open("a" if append else "w", newline="", buffering=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not append:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: format_csv_value(row.get(key, "")) for key in fieldnames})
        handle.flush()


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_csv_value(row.get(key, "")) for key in fieldnames})


def load_median_selection(path: Path) -> dict[str, int]:
    if not path.exists():
        raise SystemExit(f"Missing confidence median file: {path}")

    selected: dict[str, int] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pair_id = row.get("pdb_id") or row.get("pair_id")
            model_number = safe_int(row.get("model_number"))
            if pair_id and model_number is not None:
                selected[str(pair_id)] = model_number
    return selected


def select_median_geometry_rows(
    sample_rows: list[dict[str, object]],
    median_selection: dict[str, int],
    reduced_fieldnames: list[str],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    selected_rows: list[dict[str, object]] = []
    stats = {
        "median_selection_rows": len(median_selection),
        "median_geometry_rows": 0,
        "median_missing_geometry": 0,
    }

    rows_by_key: dict[tuple[str, int, str], dict[str, object]] = {}
    conditions_by_pair: dict[str, set[str]] = {}
    for row in sample_rows:
        model_number = row.get("model_number")
        if not isinstance(model_number, int):
            continue
        pair_id = str(row["pair_id"])
        condition = str(row["condition"])
        rows_by_key[(pair_id, model_number, condition)] = row
        conditions_by_pair.setdefault(pair_id, set()).add(condition)

    for pair_id, model_number in sorted(median_selection.items()):
        conditions = sorted(conditions_by_pair.get(pair_id, []))
        if not conditions:
            stats["median_missing_geometry"] += 1
            continue
        matched = False
        for condition in conditions:
            row = rows_by_key.get((pair_id, model_number, condition))
            if row is None:
                continue
            selected_rows.append({key: row.get(key, "") for key in reduced_fieldnames})
            matched = True
        if matched:
            stats["median_geometry_rows"] += 1
        else:
            stats["median_missing_geometry"] += 1

    return selected_rows, stats


def select_af3_ranking_geometry_rows(
    sample_rows: list[dict[str, object]],
    ranking_selection: dict[str, int],
    reduced_fieldnames: list[str],
    selection_name: str,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Select geometry rows matching an AF3 ranking_score selection."""
    selected_rows: list[dict[str, object]] = []
    stats = {
        f"{selection_name}_af3_ranking_selection_rows": len(ranking_selection),
        f"{selection_name}_af3_ranking_geometry_rows": 0,
        f"{selection_name}_af3_ranking_missing_geometry": 0,
    }

    rows_by_key: dict[tuple[str, int, str], dict[str, object]] = {}
    conditions_by_pair: dict[str, set[str]] = {}
    for row in sample_rows:
        model_number = row.get("model_number")
        if not isinstance(model_number, int):
            continue
        pair_id = str(row["pair_id"])
        condition = str(row["condition"])
        rows_by_key[(pair_id, model_number, condition)] = row
        conditions_by_pair.setdefault(pair_id, set()).add(condition)

    for pair_id, model_number in sorted(ranking_selection.items()):
        conditions = sorted(conditions_by_pair.get(pair_id, []))
        matched = False
        for condition in conditions:
            row = rows_by_key.get((pair_id, model_number, condition))
            if row is None:
                continue
            selected_rows.append({key: row.get(key, "") for key in reduced_fieldnames})
            matched = True
        if matched:
            stats[f"{selection_name}_af3_ranking_geometry_rows"] += 1
        else:
            stats[f"{selection_name}_af3_ranking_missing_geometry"] += 1

    return selected_rows, stats


def write_canonical_feature_outputs(
    out_dir: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> tuple[dict[str, int], dict[str, int]]:
    write_rows(out_dir / "geometry_features.csv", rows, fieldnames)

    label_rows = {"positive": [], "negative": []}
    for row in rows:
        label_name = {"1": "positive", "0": "negative"}.get(str(row.get("label")))
        if label_name:
            label_rows[label_name].append(row)

    for label_name, label_subset in label_rows.items():
        write_rows(out_dir / f"{label_name}_geometry_features.csv", label_subset, fieldnames)

    dataset_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    for row in rows:
        dataset = str(row.get("dataset") or "pairs")
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        label_name = {"1": "positive", "0": "negative"}.get(str(row.get("label")))
        if label_name:
            label_counts[label_name] = label_counts.get(label_name, 0) + 1
    return dataset_counts, label_counts
