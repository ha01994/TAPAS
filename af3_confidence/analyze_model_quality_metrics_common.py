#!/usr/bin/env python3
"""Common AF3 multi-sample confidence extraction and sample selection."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any
import importlib.util

import numpy as np


CHAIN_MHC = "C"
CHAIN_B2M = "D"
CHAIN_PEP = "E"
CHAIN_TRA = "B"
CHAIN_TRB = "A"
TCR_CHAINS = (CHAIN_TRA, CHAIN_TRB)
PMHC_CHAINS = (CHAIN_MHC, CHAIN_B2M, CHAIN_PEP)
FIELDNAMES = [
    "pdb_id",
    "model_number",
    "global_plddt",
    "cdr1_A",
    "cdr2_A",
    "cdr3_A",
    "cdr1_B",
    "cdr2_B",
    "cdr3_B",
    "iptm_mean",
    "iptm_tcrpmhc",
    "pdockq",
    "avgipae_pmhc",
    "avgipae_tcr",
    "pdockq2_pmhc",
    "pdockq2_tcr",
]

REFERENCE_METRICS_DIR = Path(__file__).resolve().parent
REFERENCE_METRICS_PATH = REFERENCE_METRICS_DIR / "analyze_model_quality_metrics_reference.py"
REFERENCE_PDOCKQ2_PATH = REFERENCE_METRICS_DIR / "pdockq2_json_interface.py"
_REFERENCE_METRICS = None


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_value(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return "nan"
        return f"{float(value):.8f}"
    return value


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
    if matches:
        return matches[0]

    matches = sorted(sample_dir.glob(f"*{suffix}"))
    if matches:
        return matches[0]

    return None


def find_sample_files(sample_dir: Path, pair_id: str) -> tuple[Path | None, Path | None, Path | None]:
    model = choose_file(sample_dir, pair_id, "model.cif")
    summary = choose_file(sample_dir, pair_id, "summary_confidences.json")
    confidences = choose_file(sample_dir, pair_id, "confidences.json")
    return model, summary, confidences


def read_pair_ids(pair_root: Path) -> list[str]:
    if not pair_root.exists():
        return []
    return sorted(path.name for path in pair_root.iterdir() if path.is_dir())


def read_completed_ids(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["pdb_id"] for row in reader if row.get("pdb_id")}


def read_existing_sample_rows(path: Path) -> tuple[dict[str, dict[int, dict[str, Any]]], set[tuple[str, int]]]:
    rows_by_pair: dict[str, dict[int, dict[str, Any]]] = {}
    completed_samples: set[tuple[str, int]] = set()
    if not path.exists() or path.stat().st_size == 0:
        return rows_by_pair, completed_samples

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pair_id = row.get("pdb_id")
            model_number = row.get("model_number")
            if not pair_id or model_number is None:
                continue
            try:
                model_idx = int(float(model_number))
            except ValueError:
                continue
            completed_samples.add((pair_id, model_idx))
            rows_by_pair.setdefault(pair_id, {})[model_idx] = row
    return rows_by_pair, completed_samples


def ensure_csv_schema(path: Path, fieldnames: list[str] = FIELDNAMES) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
    if header == fieldnames:
        return
    backup = path.with_suffix(path.suffix + ".old_schema")
    counter = 1
    while backup.exists():
        backup = path.with_suffix(path.suffix + f".old_schema{counter}")
        counter += 1
    path.rename(backup)
    print(f"Existing CSV schema differs; moved old file to {backup}", flush=True)


def append_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] = FIELDNAMES) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    append = path.exists() and path.stat().st_size > 0
    with path.open("a" if append else "w", newline="", buffering=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not append:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key, "")) for key in fieldnames})


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] = FIELDNAMES) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key, "")) for key in fieldnames})


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def get_reference_metrics_module():
    global _REFERENCE_METRICS
    if _REFERENCE_METRICS is not None:
        return _REFERENCE_METRICS
    if not REFERENCE_METRICS_PATH.exists():
        raise FileNotFoundError(f"Missing reference metrics script: {REFERENCE_METRICS_PATH}")
    sys.path.insert(0, str(REFERENCE_METRICS_DIR))
    spec = importlib.util.spec_from_file_location(
        "_af3_reference_model_quality_metrics",
        REFERENCE_METRICS_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import reference metrics script: {REFERENCE_METRICS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _REFERENCE_METRICS = module
    return module


def safe_round(value: Any, precision: int = 8) -> float | None:
    if value is None:
        return None
    try:
        if math.isnan(float(value)):
            return math.nan
    except (TypeError, ValueError):
        return None
    return round(float(value), precision)


def calculate_global_plddt(conf_data: dict[str, Any]) -> float:
    atom_plddts = conf_data.get("atom_plddts", [])
    if atom_plddts:
        vals = [safe_float(value) for value in atom_plddts]
        vals = [value for value in vals if not math.isnan(value)]
        if vals:
            return float(np.mean(vals))
    return math.nan


def calculate_iptms(summary_data: dict[str, Any]) -> tuple[float, float]:
    chain_iptm = summary_data.get("chain_iptm", [])
    iptm_mean = float(np.mean(chain_iptm)) if chain_iptm else math.nan

    chain_pair_iptm = summary_data.get("chain_pair_iptm", [])
    tcr_pmhc = math.nan
    if chain_pair_iptm:
        try:
            if len(chain_pair_iptm) >= 5:
                vals = [
                    chain_pair_iptm[0][3],
                    chain_pair_iptm[0][4],
                    chain_pair_iptm[2][3],
                    chain_pair_iptm[2][4],
                ]
            else:
                vals = [
                    chain_pair_iptm[0][2],
                    chain_pair_iptm[0][3],
                    chain_pair_iptm[1][2],
                    chain_pair_iptm[1][3],
                ]
            vals = [safe_float(value) for value in vals]
            vals = [value for value in vals if not math.isnan(value)]
            if vals:
                tcr_pmhc = float(np.mean(vals))
        except (IndexError, TypeError):
            tcr_pmhc = math.nan
    return iptm_mean, tcr_pmhc


def parse_mmcif_atoms(path: Path) -> list[dict[str, Any]]:
    atom_cols: list[str] = []
    atoms: list[dict[str, Any]] = []
    in_atom_loop = False

    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "loop_":
            in_atom_loop = False
            atom_cols = []
            continue
        if line.startswith("_atom_site."):
            in_atom_loop = True
            atom_cols.append(line.split(".", 1)[1])
            continue
        if in_atom_loop and (line.startswith("ATOM ") or line.startswith("HETATM ")):
            parts = line.split()
            if len(parts) < len(atom_cols):
                continue
            row = dict(zip(atom_cols, parts))
            try:
                atoms.append(
                    {
                        "chain": row["label_asym_id"],
                        "seq_id": int(float(row["label_seq_id"])),
                        "atom": row["label_atom_id"],
                        "coord": np.array(
                            [
                                float(row["Cartn_x"]),
                                float(row["Cartn_y"]),
                                float(row["Cartn_z"]),
                            ],
                            dtype=float,
                        ),
                        "plddt": float(row.get("B_iso_or_equiv", "nan")),
                    }
                )
            except (KeyError, ValueError):
                continue
        elif in_atom_loop and atom_cols and line.startswith("#"):
            in_atom_loop = False

    return atoms


def chain_plddt_proxy(atoms: list[dict[str, Any]], chain: str) -> float:
    vals = [
        float(atom["plddt"])
        for atom in atoms
        if atom["chain"] == chain and atom["atom"] in {"CA", "CB"}
    ]
    return float(np.mean(vals)) if vals else math.nan


def representative_atoms(atoms: list[dict[str, Any]], chains: tuple[str, ...]) -> list[dict[str, Any]]:
    out = []
    for atom in atoms:
        if atom["chain"] not in chains:
            continue
        if atom["atom"] not in {"CA", "CB"}:
            continue
        out.append(atom)
    return out


def pdockq_direct(atoms: list[dict[str, Any]], receptor_chains: tuple[str, ...], ligand_chains: tuple[str, ...]) -> float:
    rec = representative_atoms(atoms, receptor_chains)
    lig = representative_atoms(atoms, ligand_chains)
    if not rec or not lig:
        return 0.0

    rec_xyz = np.stack([atom["coord"] for atom in rec])
    lig_xyz = np.stack([atom["coord"] for atom in lig])
    dists = np.linalg.norm(rec_xyz[:, None, :] - lig_xyz[None, :, :], axis=2)
    rec_idx, lig_idx = np.where(dists < 8.0)
    if len(rec_idx) == 0:
        return 0.0

    plddts = [rec[i]["plddt"] for i in np.unique(rec_idx)] + [lig[i]["plddt"] for i in np.unique(lig_idx)]
    avg_if_plddt = float(np.mean(plddts)) if plddts else 0.0
    x = avg_if_plddt * np.log10(len(rec_idx))
    return float(0.724 / (1.0 + np.exp(-0.052 * (x - 152.611))) + 0.018)


def pae_matrix(conf_data: dict[str, Any]) -> np.ndarray | None:
    pae = conf_data.get("pae")
    if pae is None:
        return None
    return np.array(pae, dtype=float)


def residue_order(atoms: list[dict[str, Any]]) -> tuple[list[str], dict[str, tuple[int, int]]]:
    chains = []
    residues_by_chain: dict[str, list[int]] = {}
    for atom in atoms:
        chain = str(atom["chain"])
        if chain not in residues_by_chain:
            residues_by_chain[chain] = []
            chains.append(chain)
        seq_id = int(atom["seq_id"])
        if seq_id not in residues_by_chain[chain]:
            residues_by_chain[chain].append(seq_id)

    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for chain in chains:
        n = len(residues_by_chain[chain])
        offsets[chain] = (cursor, cursor + n)
        cursor += n
    return chains, offsets


def pmidockq_from_values(ifpae_norm: float, ifplddt: float) -> float:
    return float(1.31034849 / (1.0 + np.exp(-0.0747157696 * (ifpae_norm * ifplddt - 84.7326239))) + 0.00501886443)


def retrieve_interface_stats(
    atoms: list[dict[str, Any]],
    pae: np.ndarray | None,
    chain_of_interest: str,
    target_chains: tuple[str, ...],
) -> tuple[float, float]:
    interest = representative_atoms(atoms, (chain_of_interest,))
    target = representative_atoms(atoms, target_chains)
    if not interest or not target:
        return 0.0, 0.0

    target_xyz = np.stack([atom["coord"] for atom in target])
    ifplddt = []
    contact_res_pairs: list[tuple[str, int, str, int]] = []
    for atom in interest:
        dists = np.linalg.norm(target_xyz - atom["coord"], axis=1)
        for idx in np.where(dists <= 8.0)[0]:
            ifplddt.append(float(atom["plddt"]))
            contact_res_pairs.append(
                (
                    str(atom["chain"]),
                    int(atom["seq_id"]),
                    str(target[idx]["chain"]),
                    int(target[idx]["seq_id"]),
                )
            )

    avg_ifplddt = float(np.mean(ifplddt)) if ifplddt else 0.0
    if pae is None or not contact_res_pairs:
        return avg_ifplddt, 0.0

    _, offsets = residue_order(atoms)
    values = []
    seen = set()
    for c1, r1, c2, r2 in contact_res_pairs:
        key = (c1, r1, c2, r2)
        if key in seen:
            continue
        seen.add(key)
        if c1 not in offsets or c2 not in offsets:
            continue
        i = offsets[c1][0] + max(0, r1 - 1)
        j = offsets[c2][0] + max(0, r2 - 1)
        if i < pae.shape[0] and j < pae.shape[1]:
            values.append(float(pae[i, j]))

    if not values:
        return avg_ifplddt, 0.0
    avg_ifpae_norm = float(np.mean(1.0 / (1.0 + (np.array(values) / 10.0) ** 2)))
    return avg_ifplddt, avg_ifpae_norm


def pdockq2_metrics(atoms: list[dict[str, Any]], conf_data: dict[str, Any]) -> tuple[float, float, float, float]:
    pae = pae_matrix(conf_data)
    chain_scores = {}
    for chain in TCR_CHAINS:
        ifplddt, ifpae_norm = retrieve_interface_stats(atoms, pae, chain, PMHC_CHAINS)
        chain_scores[chain] = (ifpae_norm, pmidockq_from_values(ifpae_norm, ifplddt))
    for chain in PMHC_CHAINS:
        ifplddt, ifpae_norm = retrieve_interface_stats(atoms, pae, chain, TCR_CHAINS)
        chain_scores[chain] = (ifpae_norm, pmidockq_from_values(ifpae_norm, ifplddt))

    pmhc_ipae = [chain_scores[c][0] for c in PMHC_CHAINS if c in chain_scores]
    tcr_ipae = [chain_scores[c][0] for c in TCR_CHAINS if c in chain_scores]
    pmhc_pdockq2 = [chain_scores[c][1] for c in PMHC_CHAINS if c in chain_scores]
    tcr_pdockq2 = [chain_scores[c][1] for c in TCR_CHAINS if c in chain_scores]
    return (
        float(np.mean(pmhc_ipae)) if pmhc_ipae else 0.0,
        float(np.mean(tcr_ipae)) if tcr_ipae else 0.0,
        float(np.mean(pmhc_pdockq2)) if pmhc_pdockq2 else 0.0,
        float(np.mean(tcr_pdockq2)) if tcr_pdockq2 else 0.0,
    )


def calculate_pdockq2_json_exact(
    model_file: Path,
    json_file: Path,
    receptor_chains: str,
    ligand_chains: str,
) -> tuple[str, float, float, float, float]:
    if not REFERENCE_PDOCKQ2_PATH.exists():
        raise FileNotFoundError(f"Missing reference pDockQ2 script: {REFERENCE_PDOCKQ2_PATH}")

    command = [
        sys.executable,
        str(REFERENCE_PDOCKQ2_PATH),
        "-json",
        str(json_file),
        "-pdb",
        str(model_file),
        "-r",
        *list(receptor_chains),
        "-l",
        *list(ligand_chains),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output_lines = result.stdout.strip().split("\n")

        pdockq_scores = {}
        ipae_scores = {}
        for line in output_lines:
            parts = line.split()
            if len(parts) == 3 and parts[0] != "Chain":
                try:
                    chain = parts[0]
                    ipae_scores[chain] = float(parts[1])
                    pdockq_scores[chain] = float(parts[2])
                except ValueError:
                    continue

        ligand_pdockq = [pdockq_scores[c] for c in ligand_chains if c in pdockq_scores]
        ligand_ipae = [ipae_scores[c] for c in ligand_chains if c in ipae_scores]
        pdockq2_pmhc = sum(ligand_pdockq) / len(ligand_pdockq) if ligand_pdockq else 0.0
        avgipae_pmhc = sum(ligand_ipae) / len(ligand_ipae) if ligand_ipae else 0.0

        rec_pdockq = [pdockq_scores[c] for c in receptor_chains if c in pdockq_scores]
        rec_ipae = [ipae_scores[c] for c in receptor_chains if c in ipae_scores]
        avg_pdockq2_tcr = sum(rec_pdockq) / len(rec_pdockq) if rec_pdockq else 0.0
        avg_ipae_tcr = sum(rec_ipae) / len(rec_ipae) if rec_ipae else 0.0
        return result.stdout, avgipae_pmhc, avg_ipae_tcr, pdockq2_pmhc, avg_pdockq2_tcr
    except Exception as exc:
        print(f"Error in calculate_pdockq2_json: {exc}", flush=True)
        return "", 0.0, 0.0, 0.0, 0.0


def metrics_for_sample(pair_id: str, sample_dir: Path) -> dict[str, Any] | None:
    model_path, summary_path, conf_path = find_sample_files(sample_dir, pair_id)
    if model_path is None or summary_path is None or conf_path is None:
        return None

    reference = get_reference_metrics_module()
    model_file = str(model_path)
    summary_file = str(summary_path)
    conf_file = str(conf_path)

    global_plddt = reference.calculate_global_plddt(conf_file)
    (
        cdr1_A,
        cdr2_A,
        cdr3_A,
        cdr1_B,
        cdr2_B,
        cdr3_B,
    ) = reference.cdr_plddts(model_file, CHAIN_TRA, CHAIN_TRB)
    iptm_mean, iptm_tcrpmhc = reference.calculate_iptms(summary_file)
    pdockq = reference.calculate_pdockq_direct(
        model_file,
        [CHAIN_TRA, CHAIN_TRB],
        [CHAIN_MHC, CHAIN_B2M, CHAIN_PEP],
    )
    _, avgipae_pmhc, avgipae_tcr, pdockq2_pmhc, pdockq2_tcr = calculate_pdockq2_json_exact(
        model_path,
        conf_path,
        receptor_chains=f"{CHAIN_TRA}{CHAIN_TRB}",
        ligand_chains=f"{CHAIN_MHC}{CHAIN_B2M}{CHAIN_PEP}",
    )

    return {
        "pdb_id": pair_id,
        "model_number": sample_index(sample_dir),
        "global_plddt": safe_round(global_plddt),
        "cdr1_A": safe_round(cdr1_A),
        "cdr2_A": safe_round(cdr2_A),
        "cdr3_A": safe_round(cdr3_A),
        "cdr1_B": safe_round(cdr1_B),
        "cdr2_B": safe_round(cdr2_B),
        "cdr3_B": safe_round(cdr3_B),
        "iptm_mean": safe_round(iptm_mean),
        "iptm_tcrpmhc": safe_round(iptm_tcrpmhc),
        "pdockq": safe_round(pdockq),
        "avgipae_pmhc": safe_round(avgipae_pmhc),
        "avgipae_tcr": safe_round(avgipae_tcr),
        "pdockq2_pmhc": safe_round(pdockq2_pmhc),
        "pdockq2_tcr": safe_round(pdockq2_tcr),
    }


def select_median_sample(rows: list[dict[str, Any]], metric: str = "iptm_tcrpmhc") -> dict[str, Any] | None:
    valid = [row for row in rows if not math.isnan(safe_float(row.get(metric)))]
    if not valid:
        return None
    values = np.array([safe_float(row[metric]) for row in valid], dtype=float)
    median_value = float(np.median(values))
    ranked = sorted(
        valid,
        key=lambda row: (
            abs(safe_float(row[metric]) - median_value),
            int(float(row.get("model_number", 10**9))),
        ),
    )
    selected = dict(ranked[0])
    return selected


def build_median_rows(
    rows_by_pair: dict[str, dict[int, dict[str, Any]]],
    selection_metric: str,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for pair_id in sorted(rows_by_pair):
        sample_rows = list(rows_by_pair[pair_id].values())
        if not sample_rows:
            stats["median_no_sample_rows"] += 1
            continue
        median_row = select_median_sample(sample_rows, selection_metric)
        if median_row is None:
            stats["median_no_valid_selection_metric"] += 1
            continue
        rows.append(median_row)
        stats[f"median_samples_{len(sample_rows)}"] += 1
    return rows, stats


def load_af3_ranking_selections(
    pair_dirs: dict[str, Path],
) -> tuple[dict[str, int], dict[str, int], Counter[str]]:
    """Map each pair to its highest and median AF3 ranking_score samples."""
    best_selected: dict[str, int] = {}
    median_selected: dict[str, int] = {}
    stats: Counter[str] = Counter()

    for pair_id, pair_dir in pair_dirs.items():
        sample_dirs = find_sample_dirs(pair_dir)
        if not sample_dirs:
            continue

        scores: dict[int, float] = {}
        ranking_paths = sorted(pair_dir.glob("*_ranking_scores.csv"))
        ranking_paths.extend(sorted(pair_dir.glob("ranking_scores.csv")))
        for ranking_path in ranking_paths:
            with ranking_path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    try:
                        model_number = int(float(row.get("sample", "")))
                    except (TypeError, ValueError):
                        continue
                    score = safe_float(row.get("ranking_score"))
                    if math.isnan(score):
                        continue
                    scores[model_number] = max(scores.get(model_number, -math.inf), score)

        if not scores:
            for sample_dir in sample_dirs:
                _, summary_path, _ = find_sample_files(sample_dir, pair_id)
                summary = load_json(summary_path)
                score = safe_float(summary.get("ranking_score"))
                if math.isnan(score):
                    continue
                model_number = sample_index(sample_dir)
                scores[model_number] = max(scores.get(model_number, -math.inf), score)

        if not scores:
            stats["af3_ranking_missing_scores"] += 1
            continue

        best_model = min(scores, key=lambda model: (-scores[model], model))
        median_score = float(np.median(list(scores.values())))
        median_model = min(
            scores,
            key=lambda model: (abs(scores[model] - median_score), model),
        )
        best_selected[pair_id] = best_model
        median_selected[pair_id] = median_model

    stats["best_af3_ranking_selection_rows"] = len(best_selected)
    stats["median_af3_ranking_selection_rows"] = len(median_selected)
    return best_selected, median_selected, stats


def build_af3_ranking_rows(
    rows_by_pair: dict[str, dict[int, dict[str, Any]]],
    ranking_selection: dict[str, int],
    selection_name: str,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for pair_id, model_number in sorted(ranking_selection.items()):
        row = rows_by_pair.get(pair_id, {}).get(model_number)
        if row is None:
            stats[f"{selection_name}_af3_ranking_missing_metric_row"] += 1
            continue
        rows.append(dict(row))
    return rows, stats


def run_analysis(
    dataset: str,
    structure_roots: list[Path],
    out_dir: Path,
    limit: int | None = None,
    selection_metric: str = "iptm_tcrpmhc",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_csv = out_dir / "model_quality_metrics_all_samples.csv"
    median_csv = out_dir / f"model_quality_metrics_median_{selection_metric}.csv"
    best_af3_ranking_csv = out_dir / "model_quality_metrics_best_af3_ranking_score.csv"
    median_af3_ranking_csv = out_dir / "model_quality_metrics_median_af3_ranking_score.csv"
    summary_path = out_dir / "model_quality_metrics_summary.txt"

    ensure_csv_schema(all_csv)
    existing_samples, completed_samples = read_existing_sample_rows(all_csv)
    stats: Counter[str] = Counter()
    written_all = 0

    pair_dirs: dict[str, Path] = {}
    for root in structure_roots:
        if not root.exists():
            stats[f"missing_root:{root}"] += 1
            continue
        for pair_id in read_pair_ids(root):
            pair_dirs.setdefault(pair_id, root / pair_id)

    pair_ids = sorted(pair_dirs)
    if limit is not None:
        pair_ids = pair_ids[:limit]

    print(f"Dataset: {dataset}", flush=True)
    print("Structure roots: " + ", ".join(str(path) for path in structure_roots), flush=True)
    print(f"Out dir: {out_dir}", flush=True)
    print(f"Indexed pair dirs: {len(pair_dirs)}; processing: {len(pair_ids)}", flush=True)
    print(f"Resume completed sample rows: {len(completed_samples)}", flush=True)

    for idx, pair_id in enumerate(pair_ids, start=1):
        sample_rows = list(existing_samples.get(pair_id, {}).values())
        sample_dirs = find_sample_dirs(pair_dirs[pair_id])
        if not sample_dirs:
            stats["missing_sample_dirs"] += 1
        else:
            for sample_dir in sample_dirs:
                model_number = sample_index(sample_dir)
                if (pair_id, model_number) in completed_samples:
                    stats["skipped_existing_sample"] += 1
                    continue
                row = metrics_for_sample(pair_id, sample_dir)
                if row is None:
                    stats["missing_sample_files"] += 1
                    continue
                append_rows(all_csv, [row])
                completed_samples.add((pair_id, model_number))
                existing_samples.setdefault(pair_id, {})[model_number] = row
                sample_rows.append(row)
                written_all += 1

        if sample_rows:
            stats[f"samples_{len(sample_rows)}"] += 1
        else:
            stats["no_valid_samples"] += 1

        if idx % 100 == 0 or idx == len(pair_ids):
            print(
                f"Processed {idx}/{len(pair_ids)}; "
                f"written_all={written_all}; "
                f"skipped_existing={stats['skipped_existing_sample']}; "
                f"missing_sample_dirs={stats['missing_sample_dirs']}; "
                f"missing_sample_files={stats['missing_sample_files']}",
                flush=True,
            )

    refreshed_samples, _ = read_existing_sample_rows(all_csv)
    median_rows, median_stats = build_median_rows(refreshed_samples, selection_metric)
    write_rows(median_csv, median_rows)
    stats.update(median_stats)

    best_ranking_selection, median_ranking_selection, ranking_selection_stats = (
        load_af3_ranking_selections(pair_dirs)
    )
    best_af3_ranking_rows, best_ranking_row_stats = build_af3_ranking_rows(
        refreshed_samples,
        best_ranking_selection,
        "best",
    )
    if not best_af3_ranking_rows:
        raise SystemExit("No AF3 ranking-score-selected confidence rows matched all-sample metrics.")
    write_rows(best_af3_ranking_csv, best_af3_ranking_rows)

    median_af3_ranking_rows, median_ranking_row_stats = build_af3_ranking_rows(
        refreshed_samples,
        median_ranking_selection,
        "median",
    )
    if not median_af3_ranking_rows:
        raise SystemExit("No median AF3 ranking-score confidence rows matched all-sample metrics.")
    write_rows(median_af3_ranking_csv, median_af3_ranking_rows)
    stats.update(ranking_selection_stats)
    stats.update(best_ranking_row_stats)
    stats.update(median_ranking_row_stats)

    lines = [
        f"dataset: {dataset}",
        f"structure_roots: {', '.join(str(path) for path in structure_roots)}",
        f"indexed_pair_dirs: {len(pair_dirs)}",
        f"processed_pair_ids: {len(pair_ids)}",
        f"median_rows: {len(median_rows)}",
        f"best_af3_ranking_rows: {len(best_af3_ranking_rows)}",
        f"median_af3_ranking_rows: {len(median_af3_ranking_rows)}",
        f"new_all_sample_rows: {written_all}",
        f"selection_metric: {selection_metric}",
        f"all_samples_csv: {all_csv}",
        f"median_csv: {median_csv}",
        f"best_af3_ranking_csv: {best_af3_ranking_csv}",
        f"median_af3_ranking_csv: {median_af3_ranking_csv}",
    ]
    for key, value in sorted(stats.items()):
        lines.append(f"{key}: {value}")
    summary_path.write_text("\n".join(lines) + "\n")

    print(f"Saved all samples: {all_csv}", flush=True)
    print(f"Saved median samples: {median_csv} ({len(median_rows)} rows)", flush=True)
    print(
        f"Saved best AF3 ranking-score samples: {best_af3_ranking_csv} "
        f"({len(best_af3_ranking_rows)} rows)",
        flush=True,
    )
    print(
        f"Saved median AF3 ranking-score samples: {median_af3_ranking_csv} "
        f"({len(median_af3_ranking_rows)} rows)",
        flush=True,
    )
    print(f"Saved summary: {summary_path}", flush=True)


def add_common_args(parser: argparse.ArgumentParser, default_out_dir: Path, default_roots: list[Path]) -> None:
    parser.add_argument("--out-dir", type=Path, default=default_out_dir)
    parser.add_argument("--structure-root", type=Path, action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--selection-metric", default="iptm_tcrpmhc")
    parser.set_defaults(default_roots=default_roots)


def parse_roots(args: argparse.Namespace) -> list[Path]:
    return args.structure_root if args.structure_root is not None else args.default_roots
