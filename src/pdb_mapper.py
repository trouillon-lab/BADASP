from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Align import PairwiseAligner
from Bio.PDB import MMCIFParser, PDBList, PDBParser, Polypeptide
from matplotlib.colorbar import ColorbarBase


class PDBMapper:
    """Map BADASP alignment positions to structure residue numbering."""

    def __init__(
        self,
        pdb_id: str,
        pdb_file: Optional[str] = None,
        cache_dir: str = "data/raw",
    ) -> None:
        self.pdb_id = pdb_id.lower()
        self.pdb_file = Path(pdb_file) if pdb_file else None
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_protein_chain_id: Optional[str] = None

    def download_pdb(self) -> str:
        """Download a PDB file using Biopython or reuse a cached local copy."""
        if self.pdb_file:
            return str(self.pdb_file)

        canonical_pdb = self.cache_dir / f"{self.pdb_id}.pdb"
        if canonical_pdb.exists() and canonical_pdb.stat().st_size > 0:
            self.pdb_file = canonical_pdb
            return str(self.pdb_file)

        pdbl = PDBList(verbose=False)
        fetched = pdbl.retrieve_pdb_file(
            self.pdb_id,
            pdir=str(self.cache_dir),
            file_format="pdb",
            overwrite=False,
        )
        fetched_path = Path(fetched)

        # Normalize to a predictable cache filename for stable reuse.
        if fetched_path != canonical_pdb:
            canonical_pdb.write_text(fetched_path.read_text())

        self.pdb_file = canonical_pdb
        return str(self.pdb_file)

    def _extract_representative_msa_sequence(self, alignment_path: Path) -> str:
        records = list(SeqIO.parse(str(alignment_path), "fasta"))
        if not records:
            raise ValueError(f"No sequences found in alignment: {alignment_path}")

        # Use the sequence with the fewest gaps as representative.
        representative = min(records, key=lambda r: str(r.seq).count("-"))
        return str(representative.seq)

    def _extract_pdb_sequence_and_residue_numbers(self, pdb_path: Path) -> List[Tuple[str, str, List[int]]]:
        """Return list of (chain_id, sequence, residue_numbers) for protein chains in structure."""
        if pdb_path.suffix.lower() == ".cif":
            structure = MMCIFParser(QUIET=True).get_structure(self.pdb_id, str(pdb_path))
        else:
            structure = PDBParser(QUIET=True).get_structure(self.pdb_id, str(pdb_path))

        chains_info: List[Tuple[str, str, List[int]]] = []
        for model in structure:
            for chain in model:
                chain_residues: List[Tuple[str, int]] = []
                for residue in chain:
                    if not Polypeptide.is_aa(residue, standard=True):
                        continue
                    resseq = int(residue.id[1])
                    aa = Polypeptide.protein_letters_3to1.get(residue.resname.upper(), None)
                    if aa is None:
                        continue
                    chain_residues.append((aa, resseq))
                if chain_residues:
                    seq = "".join(aa for aa, _ in chain_residues)
                    numbers = [n for _, n in chain_residues]
                    chains_info.append((str(chain.id).strip() or "A", seq, numbers))
            if chains_info:
                break

        if not chains_info:
            raise ValueError(f"No protein residues found in structure: {pdb_path}")

        return chains_info

    def map_alignment_to_structure(self, alignment_path: Path) -> Dict[int, int]:
        """Map alignment column index (1-based) to PDB residue number."""
        pdb_path = Path(self.download_pdb())
        msa_seq_gapped = self._extract_representative_msa_sequence(alignment_path)
        chains_info = self._extract_pdb_sequence_and_residue_numbers(pdb_path)
        # store last seen protein chains
        self._last_protein_chain_id = chains_info[0][0] if chains_info else None

        ungapped_to_msa_col: List[int] = []
        msa_ungapped_chars: List[str] = []
        for idx, aa in enumerate(msa_seq_gapped, start=1):
            if aa != "-":
                msa_ungapped_chars.append(aa)
                ungapped_to_msa_col.append(idx)
        msa_seq_ungapped = "".join(msa_ungapped_chars)

        if not msa_seq_ungapped or not chains_info:
            return {}

        aligner = PairwiseAligner()
        aligner.mode = "global"
        aligner.match_score = 2.0
        aligner.mismatch_score = -1.0
        aligner.open_gap_score = -5.0
        aligner.extend_gap_score = -0.5

        mapping: Dict[int, List[Tuple[str, int]]] = {}

        # Align to each protein chain and record chain-specific residue mappings
        for chain_id, chain_seq, chain_numbers in chains_info:
            alignment = aligner.align(msa_seq_ungapped, chain_seq)[0]
            msa_blocks, pdb_blocks = alignment.aligned
            for (msa_start, msa_end), (pdb_start, pdb_end) in zip(msa_blocks, pdb_blocks):
                block_len = min(msa_end - msa_start, pdb_end - pdb_start)
                for offset in range(block_len):
                    msa_ungapped_index = msa_start + offset
                    pdb_index = pdb_start + offset
                    if msa_ungapped_index >= len(ungapped_to_msa_col):
                        continue
                    if pdb_index >= len(chain_numbers):
                        continue
                    msa_col = ungapped_to_msa_col[msa_ungapped_index]
                    resnum = int(chain_numbers[pdb_index])
                    mapping.setdefault(msa_col, []).append((chain_id, resnum))

        return mapping

    def _top_switch_rows_from_csv(self, csv_path: Path, top_n: int) -> List[Tuple[int, float]]:
        """Return top positions with switch counts (or fallback score) for gradient coloring."""
        if not csv_path.exists():
            return []

        df = pd.read_csv(csv_path)
        if "position" not in df.columns:
            return []

        value_col = "switch_count" if "switch_count" in df.columns else "max_score"
        if value_col is None or value_col not in df.columns:
            return []

        if "switch_count" in df.columns and "max_score" in df.columns:
            df = df.sort_values(["switch_count", "max_score"], ascending=[False, False])
        else:
            df = df.sort_values([value_col], ascending=[False])

        top_df = df[["position", value_col]].head(top_n)
        return [(int(pos), float(val)) for pos, val in top_df.itertuples(index=False, name=None)]

    def _all_switch_rows_from_csv(
        self,
        csv_path: Path,
        event_type_filter: Optional[str] = None,
        mdo_only: bool = False,
    ) -> Tuple[List[Tuple[int, float]], str]:
        """Return every mapped switch position with switch_count > 0 for a level."""
        if not csv_path.exists():
            return [], f"missing csv: {csv_path}"

        try:
            df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            return [], f"empty csv: {csv_path}"
        if "position" not in df.columns:
            return [], f"missing position column: {csv_path}"

        if event_type_filter:
            event_col = "Event_Type" if "Event_Type" in df.columns else "event_type" if "event_type" in df.columns else None
            if event_col is None:
                return [], f"event filter requested but no Event_Type column: {csv_path}"
            df = df[df[event_col].astype(str).str.lower() == str(event_type_filter).strip().lower()].copy()

        if mdo_only and "MDO_Node" not in df.columns and "MDO_Label" not in df.columns:
            return [], f"mdo-only filter requested but no MDO_Node/MDO_Label column: {csv_path}"

        if "switch_count" in df.columns:
            switch_series = pd.to_numeric(df["switch_count"], errors="coerce").fillna(0.0)
            df = df[switch_series > 0].copy()
            if "max_score" in df.columns:
                df = df.sort_values(["switch_count", "max_score"], ascending=[False, False])
            else:
                df = df.sort_values(["switch_count"], ascending=[False])
            if df.empty:
                return [], f"no switch_count > 0 rows in {csv_path}"
            return [(int(pos), float(val)) for pos, val in df[["position", "switch_count"]].itertuples(index=False, name=None)], ""

        if "max_score" in df.columns:
            df = df[df["max_score"].notna()].copy()
            df = df.sort_values(["max_score"], ascending=[False])
            if df.empty:
                return [], f"no finite max_score rows in {csv_path}"
            return [(int(pos), float(val)) for pos, val in df[["position", "max_score"]].itertuples(index=False, name=None)], ""

        if "score" in df.columns:
            numeric_scores = pd.to_numeric(df["score"], errors="coerce")
            numeric_scores = numeric_scores[np.isfinite(numeric_scores)]
            threshold = float(np.percentile(numeric_scores.to_numpy(dtype=float), 95)) if not numeric_scores.empty else 0.0
            switched = df[pd.to_numeric(df["score"], errors="coerce") >= threshold].copy()
            if switched.empty:
                return [], f"no switched rows above threshold in {csv_path}"
            switch_df = switched.groupby("position", as_index=False).size().rename(columns={"size": "switch_count"})
            switch_df = switch_df.sort_values(["switch_count", "position"], ascending=[False, True])
            return [(int(pos), float(val)) for pos, val in switch_df[["position", "switch_count"]].itertuples(index=False, name=None)], ""

        return [], f"no switch_count/max_score column: {csv_path}"

    @staticmethod
    def _switch_count_bounds(rows: Sequence[Tuple[int, float]]) -> Tuple[int, int]:
        counts = [int(value) for _, value in rows if float(value) > 0]
        if not counts:
            return 0, 0
        return min(counts), max(counts)

    @staticmethod
    def _hex_to_rgb(color_hex: str) -> Tuple[int, int, int]:
        color = color_hex.lstrip("#")
        return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)

    @staticmethod
    def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
        return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"

    def _interpolate_hex(self, low_hex: str, high_hex: str, fraction: float) -> str:
        low = self._hex_to_rgb(low_hex)
        high = self._hex_to_rgb(high_hex)
        frac = max(0.0, min(1.0, fraction))
        rgb = (
            int(round(low[0] + (high[0] - low[0]) * frac)),
            int(round(low[1] + (high[1] - low[1]) * frac)),
            int(round(low[2] + (high[2] - low[2]) * frac)),
        )
        return self._rgb_to_hex(rgb)

    def _residue_color_pairs(
        self,
        top_rows: Sequence[Tuple[int, float]],
        mapping: Dict[int, List[Tuple[str, int]]],
        low_hex: str,
        high_hex: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> List[Tuple[str, int, str, int, float]]:
        """Return list of (chain_id, residue_num, color_hex, msa_col, value)."""
        scored_residues: Dict[Tuple[str, int], Tuple[int, float]] = {}
        for msa_pos, value in top_rows:
            if msa_pos not in mapping:
                continue
            entries = mapping[msa_pos]
            # support legacy single-int mapping
            if isinstance(entries, int):
                entries = [(None, int(entries))]
            for chain_id, residue in entries:
                key = (chain_id, residue)
                existing = scored_residues.get(key)
                if existing is None or float(value) > float(existing[1]):
                    scored_residues[key] = (int(msa_pos), float(value))

        if not scored_residues:
            return []

        values = [float(item[1]) for item in scored_residues.values()]
        min_val = float(min_value) if min_value is not None else min(values)
        max_val = float(max_value) if max_value is not None else max(values)
        scale = max(max_val - min_val, 1e-9)

        residue_colors: List[Tuple[str, int, str, int, float]] = []
        for (chain_id, residue), (msa_col, value) in sorted(scored_residues.items()):
            fraction = (value - min_val) / scale
            color_hex = self._interpolate_hex(low_hex, high_hex, fraction)
            residue_colors.append((chain_id, int(residue), color_hex, int(msa_col), float(value)))
        return residue_colors

    @staticmethod
    def _format_residue_selector(residues: Sequence[int]) -> str:
        return "+".join(str(r) for r in sorted(set(residues)))

    def generate_pymol_script(
        self,
        alignment_path: Path,
        sdp_csv_groups: Path,
        sdp_csv_families: Path,
        sdp_csv_subfamilies: Path,
        output_pml: Path,
        top_n: int = 10,
    ) -> Path:
        """Generate a PyMOL script highlighting mapped SDP residues by hierarchy."""
        output_pml.parent.mkdir(parents=True, exist_ok=True)
        pdb_path = Path(self.download_pdb())
        mapping = self.map_alignment_to_structure(alignment_path)

        group_msa = [pos for pos, _ in self._top_switch_rows_from_csv(sdp_csv_groups, top_n=top_n)]
        family_msa = [pos for pos, _ in self._top_switch_rows_from_csv(sdp_csv_families, top_n=top_n)]
        subfamily_msa = [pos for pos, _ in self._top_switch_rows_from_csv(sdp_csv_subfamilies, top_n=top_n)]

        # Flatten mapping to residue numbers across chains, restrict to protein polymer in PyMOL selection
        def _flatten_resnums(msa_positions: List[int]) -> List[int]:
            resnums: List[int] = []
            for p in msa_positions:
                if p in mapping:
                    for chain_id, res in mapping[p]:
                        resnums.append(int(res))
            return resnums

        group_res = _flatten_resnums(group_msa)
        family_res = _flatten_resnums(family_msa)
        subfamily_res = _flatten_resnums(subfamily_msa)

        lines: List[str] = [
            "# BADASP hierarchical SDP highlighting",
            f"# source_pdb: {pdb_path}",
            "# groups: red, families: blue, subfamilies: green",
            f"load {pdb_path}",
            "hide everything",
            "show cartoon",
            "color white, polymer.protein",
        ]

        if group_res:
            selector = self._format_residue_selector(group_res)
            lines.append(f"select group_sdps, polymer.protein and resi {selector}")
            lines.append("color red, group_sdps")
        else:
            lines.append("# group_sdps: no mapped residues")

        if family_res:
            selector = self._format_residue_selector(family_res)
            lines.append(f"select family_sdps, polymer.protein and resi {selector}")
            lines.append("color blue, family_sdps")
        else:
            lines.append("# family_sdps: no mapped residues")

        if subfamily_res:
            selector = self._format_residue_selector(subfamily_res)
            lines.append(f"select subfamily_sdps, polymer.protein and resi {selector}")
            lines.append("color green, subfamily_sdps")
        else:
            lines.append("# subfamily_sdps: no mapped residues")

        output_pml.write_text("\n".join(lines) + "\n")
        return output_pml

    def _build_chimerax_script(
        self,
        pdb_path: Path,
        residue_pairs: Sequence[Tuple[str, int, str, int, float]],
        output_path: Path,
        level_label: str,
        min_switch_count: int,
        max_switch_count: int,
        low_hex: str,
        high_hex: str,
        no_switch_reason: str = "",
    ) -> Path:
        """Write a publication-quality ChimeraX script for one hierarchy level."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        residues = [f"{residue}" if chain is None else f"{chain}:{residue}" for chain, residue, _, _, _ in residue_pairs]

        lines = [
            f"# BADASP Phase 6 structural mapping: {level_label}",
            f"open {pdb_path.resolve()}",
            "set bgColor white",
            "lighting soft",
            "lighting shadows false",
            "lighting depthCue false",
            "graphics silhouettes true color black width 4",
            "material dull",
            "show cartoon",
            "hide atoms",
            "color protein gainsboro",
        ]

        if residues:
            selector = ",".join(str(r) for r in residues)
            for chain_id, residue, color_hex, alignment_col, switch_value in residue_pairs:
                lines.append(f"# Mapped from alignment col {alignment_col} (switch_count={int(round(switch_value))})")
                if chain_id is None:
                    residue_selector = f":{residue}"
                else:
                    residue_selector = f"/{chain_id}:{residue}"
                lines.append(f"color {residue_selector} {color_hex}")

            lines.append(f"# {level_label.lower()}_residues: {selector}")
        else:
            lines.append(f"# {level_label.lower()}_residues: none")
            if no_switch_reason:
                lines.append(f"# reason: {no_switch_reason}")

        output_path.write_text("\n".join(lines) + "\n")
        return output_path

    def _write_switch_legend_png(
        self,
        output_path: Path,
        low_hex: str,
        high_hex: str,
        min_switch_count: int,
        max_switch_count: int,
        level_label: str,
    ) -> Path:
        """Write a standalone high-resolution colorbar legend PNG for one level."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        vmin = int(min_switch_count)
        vmax = int(max_switch_count)
        if vmax < vmin:
            vmin, vmax = 0, 0
        if vmax == vmin:
            # ColorbarBase requires a non-zero normalization range.
            vmax = vmin + 1

        cmap = mcolors.LinearSegmentedColormap.from_list(
            f"{level_label.lower()}_switch_cmap",
            [low_hex, high_hex],
        )
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        fig, ax = plt.subplots(figsize=(5.5, 1.2), dpi=300)
        cbar = ColorbarBase(ax, cmap=cmap, norm=norm, orientation="horizontal")
        cbar.set_label("Number of Switches")
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels([str(int(min_switch_count)), str(int(max_switch_count))])
        ax.set_title(f"{level_label} Switches")
        fig.tight_layout()
        fig.savefig(output_path, format="png", dpi=300)
        plt.close(fig)
        return output_path

    def generate_chimerax_scripts(
        self,
        alignment_path: Path,
        sdp_csv_duplications: Optional[Path] = None,
        sdp_csv_groups: Optional[Path] = None,
        sdp_csv_families: Optional[Path] = None,
        sdp_csv_subfamilies: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        scores_dir: Optional[Path] = None,
    ) -> Dict[str, Path]:
        """Generate ChimeraX scripts for any available BADASP hierarchy outputs."""
        if output_dir is None:
            raise ValueError("output_dir is required")
        output_dir.mkdir(parents=True, exist_ok=True)
        pdb_path = Path(self.download_pdb())
        mapping = self.map_alignment_to_structure(alignment_path)

        level_specs = {}
        if sdp_csv_duplications:
            level_specs["duplications"] = (sdp_csv_duplications, "highlight_sdps_duplications.cxc")
        if sdp_csv_groups:
            level_specs["groups"] = (sdp_csv_groups, "highlight_sdps_groups.cxc")
        if sdp_csv_families:
            level_specs["families"] = (sdp_csv_families, "highlight_sdps_families.cxc")
        if sdp_csv_subfamilies:
            level_specs["subfamilies"] = (sdp_csv_subfamilies, "highlight_sdps_subfamilies.cxc")

        if scores_dir and scores_dir.exists():
            for layer_csv in scores_dir.glob("badasp_sdps_layer*.csv"):
                # layer_csv.stem could be 'badasp_sdps_layer01_duplications'
                # Extract everything after 'badasp_sdps_'
                layer_name = layer_csv.stem.replace("badasp_sdps_", "")
                level_specs[layer_name] = (layer_csv, f"highlight_sdps_{layer_name}.cxc")

        level_outputs: Dict[str, Path] = {}
        for level, (csv_path, filename) in level_specs.items():
            if csv_path is None:
                continue
            rows, no_switch_reason = self._all_switch_rows_from_csv(csv_path)
            _, max_switch_count = self._switch_count_bounds(rows)
            # Intensity-based palette: 0 is represented by the gray base structure,
            # and switched residues transition from very light red to dark red.
            min_switch_count = 0
            low_hex = "#FBE6E6"
            high_hex = "#7A0000"
            pairs = self._residue_color_pairs(
                rows,
                mapping,
                low_hex=low_hex,
                high_hex=high_hex,
                min_value=min_switch_count,
                max_value=max_switch_count,
            )
            if rows and not pairs:
                top_alignment_col, top_switch = max(rows, key=lambda item: float(item[1]))
                no_switch_reason = (
                    f"{len(rows)} switched alignment columns found, but none mapped to PDB residues "
                    f"(top alignment col {int(top_alignment_col)}, switch_count={int(round(top_switch))})"
                )
            output_path = output_dir / filename
            self._build_chimerax_script(
                pdb_path,
                pairs,
                output_path,
                level.capitalize(),
                min_switch_count,
                max_switch_count,
                low_hex,
                high_hex,
                    no_switch_reason=no_switch_reason,
            )
            self._write_switch_legend_png(
                output_path=output_dir / f"legend_{level}.png",
                low_hex=low_hex,
                high_hex=high_hex,
                min_switch_count=min_switch_count,
                max_switch_count=max_switch_count,
                level_label=level.capitalize(),
            )
            level_outputs[level] = output_path
        return level_outputs

    def generate_chimerax_script(
        self,
        alignment_path: Path,
        sdp_csv_duplications: Optional[Path],
        sdp_csv_groups: Path,
        sdp_csv_families: Path,
        sdp_csv_subfamilies: Path,
        output_cxc: Path,
        top_n: int = 10,
    ) -> Path:
        """Backward-compatible wrapper that writes the family-level script path."""
        output_dir = output_cxc.parent
        outputs = self.generate_chimerax_scripts(
            alignment_path=alignment_path,
            sdp_csv_duplications=sdp_csv_duplications,
            sdp_csv_groups=sdp_csv_groups,
            sdp_csv_families=sdp_csv_families,
            sdp_csv_subfamilies=sdp_csv_subfamilies,
            output_dir=output_dir,
        )
        return outputs.get("duplications", outputs.get("families", next(iter(outputs.values()))))

    def generate_single_chimerax_script(
        self,
        alignment_path: Path,
        sdp_csv: Path,
        output_cxc: Path,
        level_label: str = "Duplications",
        event_type_filter: Optional[str] = None,
        mdo_only: bool = False,
    ) -> Path:
        """Generate one ChimeraX script from an explicit SDP CSV file."""
        pdb_path = Path(self.download_pdb())
        mapping = self.map_alignment_to_structure(alignment_path)
        rows, no_switch_reason = self._all_switch_rows_from_csv(
            sdp_csv,
            event_type_filter=event_type_filter,
            mdo_only=mdo_only,
        )
        _, max_switch_count = self._switch_count_bounds(rows)
        min_switch_count = 0
        low_hex = "#FBE6E6"
        high_hex = "#7A0000"
        pairs = self._residue_color_pairs(
            rows,
            mapping,
            low_hex=low_hex,
            high_hex=high_hex,
            min_value=min_switch_count,
            max_value=max_switch_count,
        )
        if rows and not pairs:
            top_alignment_col, top_switch = max(rows, key=lambda item: float(item[1]))
            no_switch_reason = (
                f"{len(rows)} switched alignment columns found, but none mapped to PDB residues "
                f"(top alignment col {int(top_alignment_col)}, switch_count={int(round(top_switch))})"
            )

        self._build_chimerax_script(
            pdb_path,
            pairs,
            output_cxc,
            level_label,
            min_switch_count,
            max_switch_count,
            low_hex,
            high_hex,
            no_switch_reason=no_switch_reason,
        )
        return output_cxc

    def generate_physicochemical_chimerax_script(
        self,
        alignment_path: Path,
        physicochemical_csv: Path,
        output_cxc: Path,
        volume_threshold: float = 45.0,
    ) -> Path:
        """Generate a ChimeraX script coloring residues by biochemical shift class."""
        output_cxc.parent.mkdir(parents=True, exist_ok=True)
        pdb_path = Path(self.download_pdb())
        mapping = self.map_alignment_to_structure(alignment_path)

        if not physicochemical_csv.exists():
            output_cxc.write_text(
                "\n".join(
                    [
                        "# BADASP Phase 7 physicochemical structural mapping",
                        f"open {pdb_path.resolve()}",
                        "set bgColor white",
                        "lighting soft",
                        "show cartoon",
                        "hide atoms",
                        "color protein gainsboro",
                        "# no physicochemical shifts found",
                    ]
                )
                + "\n"
            )
            return output_cxc

        df = pd.read_csv(physicochemical_csv)

        def _is_shift(change: str) -> bool:
            parts = str(change).split("->")
            return len(parts) == 2 and parts[0] != parts[1]

        # Color rules:
        # charge shift = red, hydrophobicity shift = green, size shift = blue, multiple = purple.
        color_map = {
            "charge_shift": "#D62728",
            "hydrophobicity_shift": "#2CA02C",
            "size_shift": "#1F77B4",
            "multiple_complex": "#9467BD",
        }

        residue_color: Dict[Tuple[str, int], str] = {}
        for _, row in df.iterrows():
            pos = int(row.get("position", -1))
            if pos not in mapping:
                continue
            charge_shift = _is_shift(str(row.get("charge_change", "")))
            hydro_shift = _is_shift(str(row.get("hydrophobicity_change", "")))
            volume_delta = float(row.get("volume_change", 0.0)) if pd.notna(row.get("volume_change", np.nan)) else 0.0
            size_shift = abs(volume_delta) >= float(volume_threshold)
            n_shifts = int(charge_shift) + int(hydro_shift) + int(size_shift)

            if n_shifts >= 2:
                category = "multiple_complex"
            elif charge_shift:
                category = "charge_shift"
            elif hydro_shift:
                category = "hydrophobicity_shift"
            elif size_shift:
                category = "size_shift"
            else:
                continue
            # assign color to each chain/residue mapping for this alignment position
            entries = mapping[pos]
            if isinstance(entries, int):
                entries = [(None, int(entries))]
            for chain_id, residue in entries:
                residue_color[(chain_id, int(residue))] = color_map[category]

        lines = [
            "# BADASP Phase 7 physicochemical structural mapping",
            f"open {pdb_path.resolve()}",
            "set bgColor white",
            "lighting soft",
            "lighting shadows false",
            "lighting depthCue false",
            "graphics silhouettes true color black width 4",
            "material dull",
            "show cartoon",
            "hide atoms",
            "color protein gainsboro",
        ]

        if residue_color:
            for (chain_id, residue), color in sorted(residue_color.items()):
                if chain_id is None:
                    lines.append(f"color :{residue} {color}")
                else:
                    lines.append(f"color /{chain_id}:{residue} {color}")
            selector_items = []
            for (chain, res) in sorted(residue_color):
                if chain is None:
                    selector_items.append(f":{res}")
                else:
                    selector_items.append(f"/{chain}:{res}")
            selector = ",".join(selector_items)
            lines.extend(
                [
                    f"show {selector} atoms",
                    f"style {selector} stick",
                    "size stickRadius 0.28",
                    "size atomRadius 1.05",
                ]
            )
        else:
            lines.append("# no mapped physicochemical shifts passed filters")

        lines.extend(
            [
                "# legend",
                "# charge_shift: #D62728",
                "# hydrophobicity_shift: #2CA02C",
                "# size_shift: #1F77B4",
                "# multiple_complex: #9467BD",
            ]
        )

        output_cxc.write_text("\n".join(lines) + "\n")
        return output_cxc


def _resolve_sdp_csv(base_dir: Path, level: str) -> Path:
    preferred = base_dir / f"badasp_sdps_{level}.csv"
    if preferred.exists():
        return preferred
    return base_dir / f"badasp_scores_{level}.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 6 structural mapping for BADASP SDPs")
    parser.add_argument("--pdb-id", default="2cg4", help="Target PDB identifier")
    parser.add_argument("--pdb-file", default=None, help="Optional local PDB/CIF path")
    parser.add_argument(
        "--alignment",
        default="data/interim/IPR019888_trimmed.aln",
        help="Trimmed alignment FASTA path",
    )
    parser.add_argument(
        "--scores-dir",
        default="results/badasp_scoring",
        help="Directory containing BADASP score/SDP CSVs",
    )
    parser.add_argument(
        "--output-cxc",
        default="results/structural_mapping/highlight_sdps.cxc",
        help="Output ChimeraX script path",
    )
    parser.add_argument(
        "--sdp-csv",
        default=None,
        help="Optional explicit SDP CSV to map into a single output CXC script",
    )
    parser.add_argument(
        "--event-type-filter",
        default=None,
        choices=["Duplication", "Speciation", "Unknown"],
        help="Optional Event_Type filter for --sdp-csv mode",
    )
    parser.add_argument(
        "--mdo-only",
        action="store_true",
        help="When set in --sdp-csv mode, require MDO-tagged rows",
    )
    parser.add_argument("--top-n", type=int, default=10, help="Top N SDPs per level")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    mapper = PDBMapper(pdb_id=args.pdb_id, pdb_file=args.pdb_file)
    scores_dir = Path(args.scores_dir)
    alignment = Path(args.alignment)
    output_cxc = Path(args.output_cxc)

    if args.sdp_csv:
        sdp_csv = Path(args.sdp_csv)
        mapper.generate_single_chimerax_script(
            alignment_path=alignment,
            sdp_csv=sdp_csv,
            output_cxc=output_cxc,
            level_label="Duplications",
            event_type_filter=args.event_type_filter,
            mdo_only=bool(args.mdo_only),
        )
        print(f"Generated ChimeraX script: {output_cxc}")
        return

    duplications_csv = _resolve_sdp_csv(scores_dir, "duplications")
    groups_csv = _resolve_sdp_csv(scores_dir, "groups")
    families_csv = _resolve_sdp_csv(scores_dir, "families")
    subfamilies_csv = _resolve_sdp_csv(scores_dir, "subfamilies")

    outputs = mapper.generate_chimerax_scripts(
        alignment_path=alignment,
        sdp_csv_duplications=duplications_csv if duplications_csv.exists() else None,
        sdp_csv_groups=groups_csv if groups_csv.exists() else None,
        sdp_csv_families=families_csv if families_csv.exists() else None,
        sdp_csv_subfamilies=subfamilies_csv if subfamilies_csv.exists() else None,
        output_dir=output_cxc.parent,
        scores_dir=scores_dir,
    )
    combined = output_cxc
    if combined.exists() and combined not in set(outputs.values()):
        combined.unlink()
    print("Generated ChimeraX scripts: " + ", ".join(str(path) for path in outputs.values()))


if __name__ == "__main__":
    main()
