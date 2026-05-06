#!/usr/bin/env python3
"""Subset AF3 TCR–pMHC CIF: TCR V (approx.) + MHC α1α2 + peptide (full chain by default)."""

from __future__ import annotations

import argparse
import sys

from Bio.PDB import MMCIFParser, PDBIO, Select


class SubsetSelect(Select):
    def __init__(
        self,
        chain_ranges: dict[str, tuple[int | None, int | None]],
    ) -> None:
        # chain_id -> (inclusive start resseq, inclusive end resseq); None = no limit
        self.ranges = {k: v for k, v in chain_ranges.items()}

    def accept_chain(self, chain) -> int:
        return 1 if chain.id in self.ranges else 0

    def accept_residue(self, residue) -> int:
        if residue.id[0] != " ":
            return 0
        chain_id = residue.parent.id
        if chain_id not in self.ranges:
            return 0
        lo, hi = self.ranges[chain_id]
        seq = residue.id[1]
        if lo is not None and seq < lo:
            return 0
        if hi is not None and seq > hi:
            return 0
        return 1


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Extract TCR variable (approximate IMGT V span), MHC class I α1α2 "
            "(1–mhc_top_last; default 180), and the peptide chain (full length by default)."
        )
    )
    ap.add_argument("incif", help="Input mmCIF (e.g. AF3 model)")
    ap.add_argument(
        "-o",
        "--out",
        default=None,
        help="Output PDB path (default: <stem>_tcrV_mhc_a1a2_pep.pdb)",
    )
    ap.add_argument(
        "--mhc-chain",
        default="C",
        help="MHC heavy chain ID (default C for this project)",
    )
    ap.add_argument(
        "--tra-chain",
        default="B",
        help="TCR alpha chain ID (default B)",
    )
    ap.add_argument(
        "--trb-chain",
        default="A",
        help="TCR beta chain ID (default A)",
    )
    ap.add_argument(
        "--mhc-top-last",
        type=int,
        default=180,
        help="Last residue of MHC 'upper' α1+α2 region (default 180)",
    )
    ap.add_argument(
        "--tra-v-last",
        type=int,
        default=112,
        help="Approx last residue of TRA V domain (IMGT ballpark; default 112)",
    )
    ap.add_argument(
        "--trb-v-last",
        type=int,
        default=114,
        help="Approx last residue of TRB V domain (IMGT ballpark; default 114)",
    )
    ap.add_argument(
        "--peptide-chain",
        default="E",
        help="Peptide chain ID (default E). Use empty string to omit peptide.",
    )
    args = ap.parse_args()

    out = args.out or args.incif.replace(".cif", "").replace(".mmcif", "") + "_tcrV_mhc_a1a2_pep.pdb"

    chain_ranges: dict[str, tuple[int | None, int | None]] = {
        args.trb_chain: (1, args.trb_v_last),
        args.tra_chain: (1, args.tra_v_last),
        args.mhc_chain: (1, args.mhc_top_last),
    }
    if args.peptide_chain:
        chain_ranges[args.peptide_chain] = (1, None)

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("m", args.incif)

    io = PDBIO()
    io.set_structure(structure)
    io.save(out, SubsetSelect(chain_ranges))
    pep_note = (
        f", {args.peptide_chain} (peptide full)"
        if args.peptide_chain
        else ""
    )
    print(
        f"Wrote {out}\n"
        f"  chains {args.trb_chain} (TRB V 1–{args.trb_v_last}), "
        f"{args.tra_chain} (TRA V 1–{args.tra_v_last}), "
        f"{args.mhc_chain} (MHC α1α2 1–{args.mhc_top_last}){pep_note}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
