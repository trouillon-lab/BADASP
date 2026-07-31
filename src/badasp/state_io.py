"""Array-backed readers for IQ-TREE ancestral-state output and FASTA alignments.

Intended to replace the per-node ``pandas.DataFrame`` representation used by
``badasp.scoring.load_state_file``, which builds one DataFrame per internal node
(~20k for IPR019888) and runs a boolean mask over one of them on every posterior
lookup. That cost is tolerable for a single scoring run but not when scoring many
simulated replicates for null calibration.

Column order of the ``p_*`` fields in an IQ-TREE ``.state`` file is
``p_A p_R p_N p_D p_C p_Q p_E p_G p_H p_I p_L p_K p_M p_F p_P p_S p_T p_W p_Y p_V``,
i.e. :data:`AA_ORDER`. This is asserted at parse time rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
GAP_CHARS = frozenset({"-", "."})

_STATE_COLUMNS = ["Node", "Site", "State"] + [f"p_{aa}" for aa in AA_ORDER]


class StateFileError(ValueError):
    """Raised when a ``.state`` file does not have the expected structure."""


@dataclass
class StateArray:
    """Ancestral states and posteriors for a set of nodes, as dense arrays.

    Attributes:
        node_index: node name -> row index into ``states`` / ``probs``.
        states: ``(n_nodes, n_sites)`` uint8 of ASCII codes for the ML state.
            Zero means "no row was present in the file for this (node, site)".
        probs: ``(n_nodes, n_sites, 20)`` float64 posteriors, ordered by
            :data:`AA_ORDER`.
        n_sites: number of alignment sites represented (1-based site ``s`` lives
            at index ``s - 1``).
    """

    node_index: Dict[str, int]
    states: np.ndarray
    probs: np.ndarray
    n_sites: int

    def __contains__(self, node: str) -> bool:
        return node in self.node_index

    def __len__(self) -> int:
        return len(self.node_index)

    @property
    def nodes(self) -> List[str]:
        return list(self.node_index)

    def ancestral_sequence(self, node: str) -> Optional[str]:
        """ML ancestral sequence for ``node``, or ``None`` if absent.

        Mirrors ``scoring._reconstruct_ancestral_sequence_from_state``: sites are
        ordered ascending and a site with no recorded state becomes ``"X"``.
        """
        row = self.node_index.get(node)
        if row is None:
            return None
        codes = self.states[row]
        return "".join(chr(c) if c else "X" for c in codes)

    def posterior(self, node: str, site: int, aa: str) -> float:
        """Posterior probability of ``aa`` at 1-based ``site`` for ``node``.

        Returns 0.0 for a residue outside :data:`AA_ORDER` or a site outside the
        represented range, matching ``scoring.extract_posterior_probability``.
        Unlike that function, a *missing node* raises: silently returning 0.0
        makes ``p_ac`` collapse to 1.0 at ``AC == -1`` and awards the maximum
        possible switch bonus, which must not happen unnoticed across many
        unattended replicates.
        """
        row = self.node_index.get(node)
        if row is None:
            raise KeyError(
                f"node {node!r} has no ancestral-state record; refusing to "
                "default its posterior to 0.0"
            )
        if not 1 <= site <= self.n_sites:
            return 0.0
        idx = AA_ORDER.find(aa.upper())
        if idx < 0:
            return 0.0
        return float(self.probs[row, site - 1, idx])

    def posterior_vector(self, node: str, site: int) -> np.ndarray:
        """Full 20-vector of posteriors for ``node`` at 1-based ``site``."""
        row = self.node_index.get(node)
        if row is None:
            raise KeyError(f"node {node!r} has no ancestral-state record")
        return self.probs[row, site - 1]

    def require_nodes(self, nodes: Iterable[str]) -> None:
        """Raise if any of ``nodes`` lacks a record.

        Used as an up-front guard so a coverage problem surfaces before scoring
        rather than as silently inflated scores.
        """
        missing = sorted({n for n in nodes if n not in self.node_index})
        if missing:
            shown = ", ".join(missing[:10])
            more = f" (and {len(missing) - 10} more)" if len(missing) > 10 else ""
            raise StateFileError(
                f"{len(missing)} node(s) absent from the .state file: {shown}{more}"
            )


def load_state_array(
    state_path: Path,
    nodes: Optional[Iterable[str]] = None,
    n_sites: Optional[int] = None,
    chunk_rows: int = 1_000_000,
) -> StateArray:
    """Parse an IQ-TREE ``.state`` file into a :class:`StateArray`.

    Args:
        state_path: path to the ``.state`` file.
        nodes: if given, only these nodes are retained. For the null-calibration
            pipeline this is the scored-node subset (~3.5k of ~20k), which keeps
            the resident arrays small.
        n_sites: number of alignment sites. If omitted it is taken as the maximum
            ``Site`` value observed, which requires holding the parse result for
            all retained rows.
        chunk_rows: rows per read chunk; bounds peak memory during parsing.

    Raises:
        StateFileError: if the posterior columns are not in :data:`AA_ORDER`.
    """
    state_path = Path(state_path)
    wanted = set(nodes) if nodes is not None else None

    reader = pd.read_csv(
        state_path,
        sep="\t",
        comment="#",
        chunksize=chunk_rows,
        dtype={"Node": str, "Site": np.int32, "State": str},
    )

    node_index: Dict[str, int] = {}
    chunks: List[pd.DataFrame] = []
    max_site = 0
    checked_columns = False

    for chunk in reader:
        if not checked_columns:
            if list(chunk.columns) != _STATE_COLUMNS:
                raise StateFileError(
                    f"unexpected columns in {state_path}: expected "
                    f"{_STATE_COLUMNS} but found {list(chunk.columns)}"
                )
            checked_columns = True
        if wanted is not None:
            chunk = chunk[chunk["Node"].isin(wanted)]
            if chunk.empty:
                continue
        max_site = max(max_site, int(chunk["Site"].max()))
        for name in chunk["Node"].unique():
            if name not in node_index:
                node_index[name] = len(node_index)
        chunks.append(chunk)

    if not checked_columns:
        raise StateFileError(f"no data rows found in {state_path}")

    total_sites = int(n_sites) if n_sites is not None else max_site
    if total_sites <= 0:
        raise StateFileError(f"no sites found in {state_path}")

    n_nodes = len(node_index)
    states = np.zeros((n_nodes, total_sites), dtype=np.uint8)
    probs = np.zeros((n_nodes, total_sites, len(AA_ORDER)), dtype=np.float64)
    prob_cols = [f"p_{aa}" for aa in AA_ORDER]

    for chunk in chunks:
        rows = chunk["Node"].map(node_index).to_numpy(dtype=np.int64)
        sites = chunk["Site"].to_numpy(dtype=np.int64) - 1
        keep = (sites >= 0) & (sites < total_sites)
        if not keep.all():
            rows, sites, chunk = rows[keep], sites[keep], chunk[keep]
        # ``State`` may carry a multi-character token; the first character is the
        # residue, matching scoring._reconstruct_ancestral_sequence_from_state.
        # An empty/NaN state becomes "X", as that helper also does.
        first_char = (
            chunk["State"].fillna("").astype(str).str.slice(0, 1).replace("", "X")
        )
        codes = np.frombuffer(
            "".join(first_char).encode("ascii", "replace"), dtype=np.uint8
        )
        states[rows, sites] = codes
        probs[rows, sites, :] = chunk[prob_cols].to_numpy(dtype=np.float64)

    return StateArray(
        node_index=node_index, states=states, probs=probs, n_sites=total_sites
    )


@dataclass
class AlignmentMatrix:
    """A FASTA alignment as a dense ASCII matrix.

    ``scoring`` previously rebuilt a Python list of full sequence strings for
    every (clade, position) pair, so a 10k-leaf clade cost ~10k string lookups
    per alignment column. Holding the alignment as ``(n_seqs, n_sites)`` uint8
    lets a column be sliced for an arbitrary set of rows instead.
    """

    seq_index: Dict[str, int]
    matrix: np.ndarray
    n_sites: int

    def __len__(self) -> int:
        return len(self.seq_index)

    def row_indices(self, names: Sequence[str]) -> np.ndarray:
        """Row indices for ``names``, silently dropping names not present.

        Dropping matches the previous ``if leaf in seq_dict`` filter.
        """
        idx = [self.seq_index[n] for n in names if n in self.seq_index]
        return np.asarray(idx, dtype=np.int64)

    def column_counts(self, rows: np.ndarray, position: int) -> np.ndarray:
        """ASCII histogram (length 256) of column ``position`` over ``rows``.

        ``position`` is 0-based, matching ``calculate_recent_conservation``.
        """
        if rows.size == 0 or not 0 <= position < self.n_sites:
            return np.zeros(256, dtype=np.int64)
        return np.bincount(self.matrix[rows, position], minlength=256)


def load_alignment_matrix(alignment_path: Path) -> AlignmentMatrix:
    """Read a FASTA alignment into an :class:`AlignmentMatrix`.

    Parsed directly rather than via Bio.AlignIO to avoid materialising per-record
    objects for 21k sequences.
    """
    alignment_path = Path(alignment_path)
    names: List[str] = []
    seqs: List[str] = []
    current: List[str] = []

    with alignment_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n\r")
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    seqs.append("".join(current))
                    current = []
                names.append(line[1:].split()[0] if line[1:].strip() else "")
            else:
                current.append(line.strip())
    if current:
        seqs.append("".join(current))

    if len(names) != len(seqs):
        raise ValueError(
            f"{alignment_path}: {len(names)} headers but {len(seqs)} sequences"
        )
    if not seqs:
        raise ValueError(f"{alignment_path}: no sequences found")

    lengths = {len(s) for s in seqs}
    if len(lengths) != 1:
        raise ValueError(
            f"{alignment_path}: sequences are not aligned "
            f"(lengths {sorted(lengths)[:5]}...)"
        )
    n_sites = lengths.pop()

    matrix = np.frombuffer(
        "".join(seqs).encode("ascii", "replace"), dtype=np.uint8
    ).reshape(len(seqs), n_sites)

    seq_index = {name: i for i, name in enumerate(names)}
    if len(seq_index) != len(names):
        raise ValueError(f"{alignment_path}: duplicate sequence identifiers")

    return AlignmentMatrix(seq_index=seq_index, matrix=matrix, n_sites=n_sites)
