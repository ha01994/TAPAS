#!/usr/bin/env python3
"""Extract ImmRep25 TCR-pMHC geometry features from AF3 predicted structures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

from extract_af3_geometry_samples_common import (
    append_rows,
    build_sample_file_index,
    load_af3_ranking_selections,
    load_median_selection,
    read_existing_sample_rows,
    sample_feature_columns,
    select_af3_ranking_geometry_rows,
    select_median_geometry_rows,
    write_canonical_feature_outputs,
    write_rows,
)


TRB = "A"
TRA = "B"
MHC = "C"
PEP = "E"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
IMMREP25_DIR = REPO_ROOT / "tabpfn_developed" / "tabpfn_immrep25_af3"
DEFAULT_CONDITIONS = ["baseline_default"]
DEFAULT_OUTPUT_DIRS = [
    REPO_ROOT / "af3_outputs" / "immrep25",
]
DEFAULT_PAIR = IMMREP25_DIR / "immrep25_pairs.csv"
DEFAULT_TCR_LOOKUP = IMMREP25_DIR / "immrep25_tcrs.csv"
DEFAULT_MHC_LOOKUP = IMMREP25_DIR / "mhc_i_protein_seq.csv"
DEFAULT_OUT_DIR = SCRIPT_DIR / "immrep25"
DEFAULT_CONFIDENCE_MEDIAN = (
    REPO_ROOT / "af3_confidence" / "immrep25"
    / "model_quality_metrics_median_iptm_tcrpmhc.csv"
)
B2M_SEQ = "MIQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEKDEYACRVNHVTLSQPKIVKWDRDM"
CONTACT_CUTOFFS = [5.0]
PAE_CONFIDENT_CUTOFF = 10.0
PLDDT_CONFIDENT_CUTOFF = 70.0

REDUCED_FEATURE_COLUMNS = [
    "pair_id", "dataset", "label", "condition",
    "cdr3a_inferred_len", "cdr3b_inferred_len",
    "cdr3a_pep_min_dist", "cdr3b_pep_min_dist", "cdr3_all_pep_min_dist",
    "cdr3a_pep_residue_contacts_5a",
    "cdr3b_pep_residue_contacts_5a",
    "cdr3_all_pep_residue_contacts_5a",
    "cdr3_all_pep_group1_contact_fraction_5a",
    "cdr3_all_pep_group2_contact_fraction_5a",
    "cdr3a_pep_confident_residue_contacts_5a",
    "cdr3b_pep_confident_residue_contacts_5a",
    "cdr3_all_pep_confident_residue_contacts_5a",
    "cdr3_all_pep_centroid_dist",
    "tcr_pep_centroid_dist",
    "tcr_pep_residue_contacts_5a",
    "tcr_pep_group1_contact_fraction_5a",
    "tcr_pep_group2_contact_fraction_5a",
    "trb_pep_residue_contacts_5a",
    "tra_pep_residue_contacts_5a",
    "trb_fraction_of_tcr_pep_contacts_5a",
    "cdr3b_minus_cdr3a_pep_min_dist",
    "cdr3b_minus_cdr3a_pep_contacts_5a",
    "cdr3_all_mhc_residue_contacts_5a",
    "tcr_mhc_residue_contacts_5a",
    "trb_mhc_residue_contacts_5a",
    "tcr_over_peptide_angle_proxy",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_pair_row(row: dict[str, str]) -> dict[str, str]:
    normalized = dict(row)
    if "pair_id" not in normalized and "id" in normalized:
        normalized["pair_id"] = normalized["id"]
    return normalized


def load_tcr_lookup(path: Path) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames and {"tcr_id", "tcr_seq"} <= set(reader.fieldnames):
            for row in reader:
                tcr_id = clean(row.get("tcr_id", ""))
                parts = clean(row.get("tcr_seq", "")).split("_")
                if len(parts) < 4:
                    continue
                lookup[tcr_id] = {
                    "tra_seq": parts[0],
                    "trb_seq": parts[1],
                    "cdr3a_len": str(len(parts[2])),
                    "cdr3b_len": str(len(parts[3])),
                    "cdr3a_seq": parts[2],
                    "cdr3b_seq": parts[3],
                }
            return lookup

    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 2:
                continue
            tcr_id = clean(row[0])
            parts = row[1].strip().split("_")
            if len(parts) < 8:
                continue
            lookup[tcr_id] = {
                "tra_seq": parts[0],
                "trb_seq": parts[1],
                "cdr3a_len": str(len(parts[4])),
                "cdr3b_len": str(len(parts[7])),
                "trav": parts[8] if len(parts) > 8 else "",
                "trbv": parts[10] if len(parts) > 10 else "",
            }
    return lookup


def load_mhc_lookup(path: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) >= 2:
                allele = clean(row[0])
                seq = clean(row[1])
                lookup[allele] = seq
                if allele.startswith("HLA-"):
                    lookup[allele[4:]] = seq
    return lookup


def parse_pmhc(value: str) -> tuple[str, str] | tuple[None, None]:
    parts = clean(value).split("_", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None, None
    return parts[0], parts[1]


def build_pairs_from_manifest(
    pair_path: Path,
    tcr_lookup: dict[str, dict[str, str]],
    mhc_lookup: dict[str, str],
) -> tuple[list[dict[str, str]], Counter]:
    pairs: list[dict[str, str]] = []
    stats = Counter()
    for raw in read_csv(pair_path):
        raw = normalize_pair_row(raw)
        pair_id = clean(raw.get("pair_id", ""))
        tcr_id = clean(raw.get("tcr_id", raw.get("tcr", "")))
        peptide_seq, mhc_allele = parse_pmhc(raw.get("pmhc", ""))
        if peptide_seq is None or mhc_allele is None:
            stats["invalid_pmhc"] += 1
            continue

        tcr_meta = tcr_lookup.get(tcr_id)
        if tcr_meta is None:
            stats["missing_tcr_metadata"] += 1
            continue

        mhc_seq = mhc_lookup.get(mhc_allele)
        if mhc_seq is None:
            stats["missing_mhc_metadata"] += 1
            continue

        pair = dict(tcr_meta)
        pair["pair_id"] = pair_id
        pair["label"] = clean(raw.get("label", pair.get("label", "")))
        pair["dataset"] = "immrep25"
        pair["peptide_seq"] = peptide_seq
        pair["mhc_seq"] = mhc_seq
        pair["b2m_seq"] = B2M_SEQ
        pair["mhc_allele"] = mhc_allele
        pair["peptide_length"] = str(len(peptide_seq))
        pair["pmhc"] = clean(raw.get("pmhc", ""))
        pair["tcr"] = tcr_id
        pairs.append(pair)
    return pairs, stats


def load_requested_pairs(args: argparse.Namespace) -> tuple[list[dict[str, str]], dict[str, int]]:
    stats = Counter()
    if args.pairs is None:
        raise SystemExit("Provide --pairs.")

    tcr_lookup = load_tcr_lookup(args.tcr_lookup)
    mhc_lookup = load_mhc_lookup(args.mhc_lookup)
    stats["tcr_lookup_rows"] = len(tcr_lookup)
    stats["mhc_lookup_rows"] = len(mhc_lookup)
    stats["pairs_rows"] = len(read_csv(args.pairs))
    pairs, pair_stats = build_pairs_from_manifest(args.pairs, tcr_lookup, mhc_lookup)
    stats.update(pair_stats)
    return pairs, dict(stats)


def clean(value: str) -> str:
    return (value or "").strip()


def safe_float(value: str | float | int | None, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: str | int | None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def infer_cdr3_residue_ids(sequence: str, cdr3_len: int | None) -> list[int]:
    """Infer 1-based CDR3 residue ids from a variable-domain sequence.

    VDJ-style CDR3 strings usually include the leading Cys and the conserved
    Phe/Trp just before the FGXG/WGXG motif. The pair manifest has lengths but
    not CDR3 sequences, so use the final FGXG/WGXG-like motif as the anchor.
    """
    sequence = clean(sequence)
    if not sequence or not cdr3_len or cdr3_len <= 0 or cdr3_len > len(sequence):
        return []

    motif_matches = list(re.finditer(r"[FW]G.G", sequence))
    motif_matches = [m for m in motif_matches if m.start() > len(sequence) * 0.55]
    if motif_matches:
        conserved_fw_index = motif_matches[-1].start()
        end_exclusive = conserved_fw_index + 1
        start = end_exclusive - cdr3_len
        if 0 <= start < end_exclusive <= len(sequence):
            return list(range(start + 1, end_exclusive + 1))

    cysteines = [i for i, aa in enumerate(sequence) if aa == "C" and i > len(sequence) * 0.45]
    for start in reversed(cysteines):
        end_exclusive = start + cdr3_len
        if end_exclusive <= len(sequence):
            return list(range(start + 1, end_exclusive + 1))

    start = max(0, len(sequence) - cdr3_len)
    return list(range(start + 1, start + cdr3_len + 1))


def parse_mmcif_atoms(path: Path) -> list[dict[str, object]]:
    atom_cols: list[str] = []
    atoms: list[dict[str, object]] = []
    in_atom_loop = False

    for raw_line in path.read_text().splitlines():
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
            parts = shlex.split(line)
            if len(parts) < len(atom_cols):
                continue
            row = dict(zip(atom_cols, parts))
            try:
                atoms.append(
                    {
                        "chain": row["label_asym_id"],
                        "seq_id": int(row["label_seq_id"]),
                        "atom": row["label_atom_id"],
                        "element": row["type_symbol"],
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


def atoms_for(
    atoms: list[dict[str, object]],
    chains: set[str],
    residue_ids_by_chain: dict[str, set[int]] | None = None,
    atom_name: str | None = None,
) -> list[dict[str, object]]:
    selected = []
    for atom in atoms:
        chain = str(atom["chain"])
        if chain not in chains:
            continue
        if atom_name and atom["atom"] != atom_name:
            continue
        if residue_ids_by_chain is not None:
            allowed = residue_ids_by_chain.get(chain, set())
            if int(atom["seq_id"]) not in allowed:
                continue
        selected.append(atom)
    return selected


def coords(group: list[dict[str, object]]) -> np.ndarray:
    if not group:
        return np.zeros((0, 3), dtype=float)
    return np.stack([atom["coord"] for atom in group]).astype(float)


def centroid(group: list[dict[str, object]]) -> np.ndarray | None:
    xyz = coords(group)
    if len(xyz) == 0:
        return None
    return xyz.mean(axis=0)


def centroid_distance(group1: list[dict[str, object]], group2: list[dict[str, object]]) -> float:
    c1 = centroid(group1)
    c2 = centroid(group2)
    if c1 is None or c2 is None:
        return math.nan
    return float(np.linalg.norm(c1 - c2))


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return math.nan
    cosine = float(np.dot(v1, v2) / (norm1 * norm2))
    cosine = max(-1.0, min(1.0, cosine))
    return float(np.degrees(np.arccos(cosine)))


def centroid_angle(
    group1: list[dict[str, object]],
    vertex_group: list[dict[str, object]],
    group2: list[dict[str, object]],
) -> float:
    c1 = centroid(group1)
    cv = centroid(vertex_group)
    c2 = centroid(group2)
    if c1 is None or cv is None or c2 is None:
        return math.nan
    return angle_between(cv - c1, cv - c2)


def residue_plddt_map(atoms: list[dict[str, object]]) -> dict[tuple[str, int], float]:
    values: dict[tuple[str, int], list[float]] = defaultdict(list)
    for atom in atoms:
        values[(str(atom["chain"]), int(atom["seq_id"]))].append(float(atom["plddt"]))
    return {key: float(np.nanmean(vals)) for key, vals in values.items()}


def load_pae(conf_path: Path | None) -> tuple[dict[tuple[str, int], int], np.ndarray] | tuple[None, None]:
    if conf_path is None or not conf_path.exists():
        return None, None
    data = json.loads(conf_path.read_text())
    if not {"token_chain_ids", "token_res_ids", "pae"} <= set(data):
        return None, None
    token_index: dict[tuple[str, int], int] = {}
    for i, (chain, res_id) in enumerate(zip(data["token_chain_ids"], data["token_res_ids"])):
        token_index.setdefault((str(chain), int(res_id)), i)
    return token_index, np.array(data["pae"], dtype=float)


def pair_pae(
    token_index: dict[tuple[str, int], int] | None,
    pae: np.ndarray | None,
    res1: tuple[str, int],
    res2: tuple[str, int],
) -> float:
    if token_index is None or pae is None:
        return math.nan
    i = token_index.get(res1)
    j = token_index.get(res2)
    if i is None or j is None:
        return math.nan
    return float((pae[i, j] + pae[j, i]) / 2.0)


def contact_features(
    prefix: str,
    group1: list[dict[str, object]],
    group2: list[dict[str, object]],
    residue_plddt: dict[tuple[str, int], float],
    token_index: dict[tuple[str, int], int] | None,
    pae: np.ndarray | None,
) -> dict[str, float]:
    features: dict[str, float] = {}
    xyz1 = coords(group1)
    xyz2 = coords(group2)
    residues1 = {(str(a["chain"]), int(a["seq_id"])) for a in group1}
    residues2 = {(str(a["chain"]), int(a["seq_id"])) for a in group2}

    if len(xyz1) == 0 or len(xyz2) == 0:
        features[f"{prefix}_min_dist"] = math.nan
        for cutoff in CONTACT_CUTOFFS:
            label = f"{int(cutoff)}a"
            features[f"{prefix}_atom_contacts_{label}"] = 0
            features[f"{prefix}_residue_contacts_{label}"] = 0
        features[f"{prefix}_group1_contact_fraction_5a"] = math.nan
        features[f"{prefix}_group2_contact_fraction_5a"] = math.nan
        features[f"{prefix}_confident_residue_contacts_5a"] = 0
        return features

    tree = cKDTree(xyz2)
    dists, _ = tree.query(xyz1, k=1)
    features[f"{prefix}_min_dist"] = float(np.min(dists))

    for cutoff in CONTACT_CUTOFFS:
        label = f"{int(cutoff)}a"
        pairs = tree.query_ball_point(xyz1, r=cutoff)
        atom_contacts = sum(len(indices) for indices in pairs)
        residue_pairs = set()
        for i, js in enumerate(pairs):
            if not js:
                continue
            res1 = (str(group1[i]["chain"]), int(group1[i]["seq_id"]))
            for j in js:
                res2 = (str(group2[j]["chain"]), int(group2[j]["seq_id"]))
                residue_pairs.add((res1, res2))
        features[f"{prefix}_atom_contacts_{label}"] = atom_contacts
        features[f"{prefix}_residue_contacts_{label}"] = len(residue_pairs)

        if cutoff == 5.0:
            contacted1 = {res1 for res1, _ in residue_pairs}
            contacted2 = {res2 for _, res2 in residue_pairs}
            features[f"{prefix}_group1_contact_fraction_5a"] = (
                len(contacted1) / len(residues1) if residues1 else math.nan
            )
            features[f"{prefix}_group2_contact_fraction_5a"] = (
                len(contacted2) / len(residues2) if residues2 else math.nan
            )
            confident = 0
            for res1, res2 in residue_pairs:
                plddt1 = residue_plddt.get(res1, math.nan)
                plddt2 = residue_plddt.get(res2, math.nan)
                avg_plddt = np.nanmean([plddt1, plddt2])
                pae_val = pair_pae(token_index, pae, res1, res2)
                if (
                    not math.isnan(avg_plddt)
                    and avg_plddt >= PLDDT_CONFIDENT_CUTOFF
                    and not math.isnan(pae_val)
                    and pae_val <= PAE_CONFIDENT_CUTOFF
                ):
                    confident += 1
            features[f"{prefix}_confident_residue_contacts_5a"] = confident

    return features


def ca_min_distance(group1: list[dict[str, object]], group2: list[dict[str, object]]) -> float:
    ca1 = [atom for atom in group1 if atom["atom"] == "CA"]
    ca2 = [atom for atom in group2 if atom["atom"] == "CA"]
    if not ca1 or not ca2:
        return math.nan
    xyz1 = coords(ca1)
    xyz2 = coords(ca2)
    dists, _ = cKDTree(xyz2).query(xyz1, k=1)
    return float(np.min(dists))


def find_job_files(
    pair_id: str,
    condition: str,
    output_dirs: list[Path],
) -> tuple[Path, Path | None] | None:
    """Find AF3 model and optional confidence files for raw or template layouts."""
    candidates = []
    for output_dir in output_dirs:
        raw_job = output_dir / pair_id
        candidates.append(
            (
                raw_job / f"{pair_id}_model.cif",
                raw_job / f"{pair_id}_confidences.json",
            )
        )

        template_name = f"{pair_id}_{condition}_structure"
        template_job = output_dir / template_name
        candidates.append(
            (
                template_job / f"{template_name}_model.cif",
                template_job / f"{template_name}_confidences.json",
            )
        )

        condition_name = f"{pair_id}_{condition}"
        condition_job = output_dir / condition_name
        candidates.append(
            (
                condition_job / f"{condition_name}_model.cif",
                condition_job / f"{condition_name}_confidences.json",
            )
        )

    for model_path, conf_path in candidates:
        if model_path.exists():
            return model_path, conf_path if conf_path.exists() else None
    return None


def build_job_file_index(
    output_dirs: list[Path],
    conditions: list[str],
) -> dict[tuple[str, str], tuple[Path, Path | None]]:
    job_files: dict[tuple[str, str], tuple[Path, Path | None]] = {}
    for output_dir in output_dirs:
        if not output_dir.exists():
            continue
        for job_dir in output_dir.iterdir():
            if not job_dir.is_dir():
                continue

            job_id = job_dir.name
            raw_model = job_dir / f"{job_id}_model.cif"
            raw_conf = job_dir / f"{job_id}_confidences.json"
            if raw_model.exists():
                for condition in conditions:
                    job_files.setdefault(
                        (job_id, condition),
                        (raw_model, raw_conf if raw_conf.exists() else None),
                    )
                continue

            for condition in conditions:
                for suffix in (f"_{condition}_structure", f"_{condition}"):
                    if not job_id.endswith(suffix):
                        continue
                    pair_id = job_id[: -len(suffix)]
                    model = job_dir / f"{job_id}_model.cif"
                    conf = job_dir / f"{job_id}_confidences.json"
                    if model.exists():
                        job_files.setdefault(
                            (pair_id, condition),
                            (model, conf if conf.exists() else None),
                        )
    return job_files


def geometry_for_job(
    pair: dict[str, str],
    condition: str,
    model_path: Path,
    conf_path: Path | None,
) -> dict[str, object] | None:
    if not model_path.exists():
        return None

    atoms = parse_mmcif_atoms(model_path)
    residue_plddt = residue_plddt_map(atoms)
    token_index, pae = load_pae(conf_path)

    cdr3a_ids = set(infer_cdr3_residue_ids(pair["tra_seq"], safe_int(pair.get("cdr3a_len"))))
    cdr3b_ids = set(infer_cdr3_residue_ids(pair["trb_seq"], safe_int(pair.get("cdr3b_len"))))
    cdr_ids = {TRA: cdr3a_ids, TRB: cdr3b_ids}

    groups = {
        "tra": atoms_for(atoms, {TRA}),
        "trb": atoms_for(atoms, {TRB}),
        "tcr": atoms_for(atoms, {TRA, TRB}),
        "mhc": atoms_for(atoms, {MHC}),
        "pep": atoms_for(atoms, {PEP}),
        "cdr3a": atoms_for(atoms, {TRA}, cdr_ids),
        "cdr3b": atoms_for(atoms, {TRB}, cdr_ids),
        "cdr3_all": atoms_for(atoms, {TRA, TRB}, cdr_ids),
    }

    row: dict[str, object] = {
        "pair_id": pair["pair_id"],
        "dataset": pair.get("dataset", ""),
        "label": pair["label"],
        "condition": condition,
        "cdr3a_inferred_len": len(cdr3a_ids),
        "cdr3b_inferred_len": len(cdr3b_ids),
    }

    for name in ["cdr3a", "cdr3b", "cdr3_all", "tcr", "trb", "tra"]:
        row.update(
            contact_features(
                f"{name}_pep",
                groups[name],
                groups["pep"],
                residue_plddt,
                token_index,
                pae,
            )
        )
        row[f"{name}_pep_ca_min_dist"] = ca_min_distance(groups[name], groups["pep"])
        row[f"{name}_pep_centroid_dist"] = centroid_distance(groups[name], groups["pep"])

    for name in ["cdr3a", "cdr3b", "cdr3_all", "tcr", "trb", "tra"]:
        row.update(
            contact_features(
                f"{name}_mhc",
                groups[name],
                groups["mhc"],
                residue_plddt,
                token_index,
                pae,
            )
        )
        row[f"{name}_mhc_ca_min_dist"] = ca_min_distance(groups[name], groups["mhc"])
        row[f"{name}_mhc_centroid_dist"] = centroid_distance(groups[name], groups["mhc"])

    trb_contacts = safe_float(row.get("trb_pep_residue_contacts_5a"), 0.0)
    tra_contacts = safe_float(row.get("tra_pep_residue_contacts_5a"), 0.0)
    row["trb_fraction_of_tcr_pep_contacts_5a"] = (
        trb_contacts / (trb_contacts + tra_contacts)
        if (trb_contacts + tra_contacts) > 0
        else math.nan
    )
    row["cdr3b_minus_cdr3a_pep_min_dist"] = (
        safe_float(row.get("cdr3b_pep_min_dist"))
        - safe_float(row.get("cdr3a_pep_min_dist"))
    )
    row["cdr3b_minus_cdr3a_pep_contacts_5a"] = (
        safe_float(row.get("cdr3b_pep_residue_contacts_5a"), 0.0)
        - safe_float(row.get("cdr3a_pep_residue_contacts_5a"), 0.0)
    )
    row["tcr_over_peptide_angle_proxy"] = centroid_angle(
        groups["tcr"],
        groups["pep"],
        groups["mhc"],
    )
    return row


def reduce_feature_row(row: dict[str, object]) -> dict[str, object]:
    missing = [key for key in REDUCED_FEATURE_COLUMNS if key not in row]
    if missing:
        raise KeyError(f"Missing reduced feature columns: {', '.join(missing)}")
    return {key: row[key] for key in REDUCED_FEATURE_COLUMNS}


def format_csv_value(value: object) -> object:
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return "nan"
        return f"{float(value):.6f}"
    return value


def write_feature_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_csv_value(row.get(key, "")) for key in fieldnames})


def remove_stale_outputs(out_dir: Path) -> None:
    stale_paths = [
        out_dir / "geometry_features.csv",
        out_dir / "pairs_geometry_features.csv",
        out_dir / "positive_geometry_features.csv",
        out_dir / "negative_geometry_features.csv",
        out_dir / "geometry_feature_auc.txt",
        out_dir / "extraction_summary.txt",
    ]
    for path in stale_paths:
        if path.exists():
            path.unlink()


def load_existing_outputs(
    out_dir: Path,
    fieldnames: list[str],
) -> tuple[list[dict[str, object]], set[tuple[str, str]], Counter[str], Counter[str], bool]:
    path = out_dir / "geometry_features.csv"
    if not path.exists():
        remove_stale_outputs(out_dir)
        return [], set(), Counter(), Counter(), False

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fieldnames:
            remove_stale_outputs(out_dir)
            return [], set(), Counter(), Counter(), False

        rows: list[dict[str, object]] = []
        processed: set[tuple[str, str]] = set()
        dataset_counts: Counter[str] = Counter()
        label_counts: Counter[str] = Counter()
        for row in reader:
            parsed: dict[str, object] = {}
            for key in fieldnames:
                value = row.get(key, "")
                if key in {"pair_id", "dataset", "label", "condition"}:
                    parsed[key] = value
                else:
                    parsed[key] = safe_float(value)
            rows.append(parsed)
            processed.add((str(parsed["pair_id"]), str(parsed["condition"])))
            dataset_counts[str(parsed.get("dataset") or "pairs")] += 1
            label_name = {"1": "positive", "0": "negative"}.get(str(parsed.get("label")))
            if label_name:
                label_counts[label_name] += 1
    return rows, processed, dataset_counts, label_counts, True


class StreamingFeatureCsvs:
    def __init__(self, out_dir: Path, fieldnames: list[str]) -> None:
        self.out_dir = out_dir
        self.fieldnames = fieldnames
        self.stack = ExitStack()
        self.writers: dict[Path, csv.DictWriter] = {}
        self.handles: dict[Path, object] = {}

    def __enter__(self) -> "StreamingFeatureCsvs":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stack.close()

    def writer_for(self, path: Path) -> csv.DictWriter:
        if path not in self.writers:
            append = path.exists() and path.stat().st_size > 0
            handle = self.stack.enter_context(path.open("a" if append else "w", newline="", buffering=1))
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            if not append:
                writer.writeheader()
            self.writers[path] = writer
            self.handles[path] = handle
        return self.writers[path]

    def write(self, row: dict[str, object]) -> None:
        paths = {
            self.out_dir / "geometry_features.csv",
        }
        label_name = {"1": "positive", "0": "negative"}.get(str(row.get("label")))
        if label_name:
            paths.add(self.out_dir / f"{label_name}_geometry_features.csv")

        formatted = {key: format_csv_value(row.get(key, "")) for key in self.fieldnames}
        for path in paths:
            self.writer_for(path).writerow(formatted)
            self.handles[path].flush()


def auc_table(rows: list[dict[str, object]], out_path: Path) -> None:
    by_condition: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition"])].append(row)

    numeric_keys = [
        key
        for key in rows[0]
        if key not in {"pair_id", "dataset", "label", "condition"}
        and all(isinstance(row.get(key), (int, float)) for row in rows)
    ]
    lines = []
    for condition, cond_rows in sorted(by_condition.items()):
        labels = np.array([int(row["label"]) for row in cond_rows])
        lines.append(f"[{condition}] n={len(cond_rows)}")
        scored = []
        for key in numeric_keys:
            values = np.array([float(row[key]) for row in cond_rows], dtype=float)
            mask = ~np.isnan(values)
            if mask.sum() < 4 or len(set(labels[mask])) < 2:
                continue
            auc_pos = roc_auc_score(labels[mask], values[mask])
            auc_val = auc_pos if auc_pos >= 0.5 else 1.0 - auc_pos
            direction = "+" if auc_pos >= 0.5 else "-"
            scored.append((auc_val, direction, key, auc_pos))
        for auc_val, direction, key, auc_pos in sorted(scored, reverse=True)[:30]:
            lines.append(f"  {key:<48} AUC={auc_pos:.3f} best={auc_val:.3f} dir={direction}")
        lines.append("")
    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        action="append",
        default=None,
        help="Directory containing AF3 output job folders. Can be passed multiple times.",
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=DEFAULT_PAIR,
        help="ImmRep25 pair manifest with id/pmhc/tcr_id/label columns.",
    )
    parser.add_argument(
        "--tcr-lookup",
        type=Path,
        default=DEFAULT_TCR_LOOKUP,
        help="CSV mapping tcr_id to TRA/TRB variable-domain sequences and CDR3 fields.",
    )
    parser.add_argument(
        "--mhc-lookup",
        type=Path,
        default=DEFAULT_MHC_LOOKUP,
        help="CSV mapping HLA allele to MHC protein sequence.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )
    parser.add_argument(
        "--confidence-median",
        type=Path,
        default=DEFAULT_CONFIDENCE_MEDIAN,
        help="CSV from af3_confidence containing the selected median sample per pair.",
    )
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_DIRS
    print(f"Using pairs: {args.pairs}", flush=True)
    print(f"Using TCR lookup: {args.tcr_lookup}", flush=True)
    print(f"Using MHC lookup: {args.mhc_lookup}", flush=True)
    print(f"Using output directory: {args.out_dir}", flush=True)
    print(f"Using confidence median: {args.confidence_median}", flush=True)

    pairs, input_stats = load_requested_pairs(args)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = REDUCED_FEATURE_COLUMNS
    sample_fieldnames = sample_feature_columns(fieldnames)
    all_samples_csv = args.out_dir / "geometry_features_all_samples.csv"
    rows, processed_keys, _ = read_existing_sample_rows(
        all_samples_csv,
        sample_fieldnames,
    )
    existing_models_by_pair: dict[str, set[int]] = defaultdict(set)
    for pair_id, model_number in processed_keys:
        existing_models_by_pair[pair_id].add(model_number)
    pairs_to_process = [
        pair
        for pair in pairs
        if len(existing_models_by_pair.get(pair["pair_id"], set())) < 5
    ]
    if pairs_to_process:
        sample_file_index = build_sample_file_index(
            args.output_dir,
            {pair["pair_id"] for pair in pairs_to_process},
        )
        print(
            f"Existing all-sample geometry is incomplete for {len(pairs_to_process)} pairs; "
            f"indexed {len(sample_file_index)} matching AF3 pair directories.",
            flush=True,
        )
    else:
        sample_file_index = {}
        print(
            f"Reusing complete all-sample geometry for {len(pairs)} pairs; "
            "skipping AF3 model-file indexing.",
            flush=True,
        )

    completed_pair_ids = {
        pair["pair_id"]
        for pair in pairs
        if len(existing_models_by_pair.get(pair["pair_id"], set())) >= 5
    }
    missing_sample_dirs = 0
    missing_sample_files = 0
    skipped_existing = 0
    written_all = 0
    for pair_index, pair in enumerate(pairs_to_process, start=1):
        pair_had_model = False
        samples = sample_file_index.get(pair["pair_id"], {})
        if not samples:
            missing_sample_dirs += 1
        for condition in conditions:
            for model_number, (model_path, conf_path) in sorted(samples.items()):
                key = (pair["pair_id"], model_number)
                if key in processed_keys:
                    skipped_existing += 1
                    pair_had_model = True
                    continue
                pair_had_model = True
                row = geometry_for_job(pair, condition, model_path, conf_path)
                if row is None:
                    missing_sample_files += 1
                    continue
                row = reduce_feature_row(row)
                row["model_number"] = model_number
                append_rows(all_samples_csv, [row], sample_fieldnames)
                rows.append(row)
                processed_keys.add(key)
                written_all += 1
        if pair_had_model:
            completed_pair_ids.add(pair["pair_id"])
        if pair_index % 100 == 0 or pair_index == len(pairs_to_process):
            print(
                f"Processed {pair_index}/{len(pairs_to_process)} incomplete input rows; "
                f"found {len(completed_pair_ids)} pairs with sample CIFs; "
                f"all_sample_rows={len(rows)} written_all={written_all} "
                f"skipped_existing={skipped_existing}.",
                flush=True,
            )

    if not rows:
        raise SystemExit("No sample-level geometry features extracted; check output directory.")

    rows, _, _ = read_existing_sample_rows(all_samples_csv, sample_fieldnames)
    median_selection = load_median_selection(args.confidence_median)
    median_rows, median_stats = select_median_geometry_rows(
        rows,
        median_selection,
        fieldnames,
    )
    if not median_rows:
        raise SystemExit("No median-selected geometry features matched confidence selections.")
    dataset_counts, label_counts = write_canonical_feature_outputs(
        args.out_dir,
        median_rows,
        fieldnames,
    )

    ranking_selection, median_ranking_selection, ranking_load_stats = load_af3_ranking_selections(
        args.output_dir,
        {str(row["pair_id"]) for row in rows},
    )
    ranking_rows, ranking_geometry_stats = select_af3_ranking_geometry_rows(
        rows,
        ranking_selection,
        fieldnames,
        "best",
    )
    if not ranking_rows:
        raise SystemExit("No AF3 ranking-score-selected geometry features matched sample rows.")
    ranking_out_csv = args.out_dir / "geometry_features_best_af3_ranking_score.csv"
    write_rows(ranking_out_csv, ranking_rows, fieldnames)

    median_ranking_rows, median_ranking_geometry_stats = select_af3_ranking_geometry_rows(
        rows,
        median_ranking_selection,
        fieldnames,
        "median",
    )
    if not median_ranking_rows:
        raise SystemExit("No median AF3 ranking-score geometry features matched sample rows.")
    median_ranking_out_csv = args.out_dir / "geometry_features_median_af3_ranking_score.csv"
    write_rows(median_ranking_out_csv, median_ranking_rows, fieldnames)

    auc_path = args.out_dir / "geometry_feature_auc.txt"
    auc_table(median_rows, auc_path)
    out_csv = args.out_dir / "geometry_features.csv"
    summary_path = args.out_dir / "extraction_summary.txt"
    summary_lines = [
        f"Output dirs: {', '.join(str(path) for path in args.output_dir)}",
        f"Conditions: {', '.join(conditions)}",
        f"Confidence median: {args.confidence_median}",
        f"Input rows after metadata mapping: {len(pairs)}",
        f"Completed pairs considered: {len(completed_pair_ids)}",
        f"All-sample rows: {len(rows)}",
        f"New all-sample rows: {written_all}",
        f"Median-selected rows: {len(median_rows)}",
        f"Best AF3 ranking-score rows: {len(ranking_rows)}",
        f"Best AF3 ranking-score CSV: {ranking_out_csv}",
        f"Median AF3 ranking-score rows: {len(median_ranking_rows)}",
        f"Median AF3 ranking-score CSV: {median_ranking_out_csv}",
        f"Skipped existing rows: {skipped_existing}",
        f"Missing sample dirs: {missing_sample_dirs}",
        f"Missing sample files: {missing_sample_files}",
        f"All-samples CSV: {all_samples_csv}",
    ]
    for key, value in sorted(median_stats.items()):
        summary_lines.append(f"{key}: {value}")
    for key, value in sorted(ranking_load_stats.items()):
        summary_lines.append(f"{key}: {value}")
    for key, value in sorted(ranking_geometry_stats.items()):
        summary_lines.append(f"{key}: {value}")
    for key, value in sorted(median_ranking_geometry_stats.items()):
        summary_lines.append(f"{key}: {value}")
    for key, value in sorted(input_stats.items()):
        summary_lines.append(f"{key}: {value}")
    for dataset, count in sorted(dataset_counts.items()):
        summary_lines.append(f"{dataset}_extracted_rows: {count}")
    for label_name, count in sorted(label_counts.items()):
        summary_lines.append(f"{label_name}_extracted_rows: {count}")
    summary_path.write_text("\n".join(summary_lines) + "\n")

    print(f"Input rows after metadata mapping: {len(pairs)}")
    print(f"Completed pairs considered: {len(completed_pair_ids)}")
    print(f"All-sample rows: {len(rows)}  new rows: {written_all}")
    print(f"Median-selected rows: {len(median_rows)}")
    print(f"Best AF3 ranking-score rows: {len(ranking_rows)}")
    print(f"Median AF3 ranking-score rows: {len(median_ranking_rows)}")
    print(f"Missing sample dirs: {missing_sample_dirs}  missing sample files: {missing_sample_files}")
    print(f"Saved all samples: {all_samples_csv}")
    print(f"Saved: {out_csv}")
    print(f"Saved: {ranking_out_csv}")
    print(f"Saved: {median_ranking_out_csv}")
    print(f"Saved: {auc_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
