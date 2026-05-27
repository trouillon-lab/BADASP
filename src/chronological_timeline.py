import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import Phylo, SeqIO
from Bio.Phylo.BaseTree import Clade, Tree
from ete3 import NCBITaxa

# Clean environmental/metagenomic taxids to ensure high-confidence split calculations
_GARBAGE_TAXON_PATTERN = re.compile(
    r"metagenome|"
    r"environmental\s+sample|"
    r"unidentified|"
    r"mixed\s+culture|"
    r"enrichment\s+culture|"
    r"uncultured\s+bacterium",
    re.IGNORECASE
)

def _is_garbage_lineage(desc: str) -> bool:
    return bool(_GARBAGE_TAXON_PATTERN.search(desc))

def load_calibrations(config_path: Path) -> Dict[int, float]:
    """Load calibration ages from JSON config."""
    if not config_path.exists():
        # Fallback defaults if config missing
        return {
            131567: 3800.0,   # Cellular organisms
            2157: 3500.0,     # Archaea
            2: 3200.0,        # Bacteria
            68336: 3000.0,    # Gracilicutes
            1783272: 3000.0,  # Terrabacteria
            1224: 2200.0,     # Proteobacteria
            1239: 3000.0,     # Firmicutes
            28890: 2500.0     # Euryarchaeota
        }
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
        return {int(k): float(v) for k, v in data.items()}

def parse_accession_taxids(fasta_path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Parse accession to TaxID map and build TaxID descriptions to identify garbage taxids."""
    acc_to_taxid: Dict[str, int] = {}
    taxid_to_desc: Dict[int, str] = {}
    
    print(f"Loading sequence metadata from {fasta_path}...")
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        parts = record.id.split("|")
        acc = parts[1] if len(parts) >= 3 else record.id
        
        # Extract OX= (TaxID)
        ox_match = re.search(r"\bOX=(\d+)\b", record.description)
        if ox_match:
            taxid = int(ox_match.group(1))
            acc_to_taxid[acc] = taxid
            taxid_to_desc[taxid] = record.description
            
    print(f"  -> Successfully loaded {len(acc_to_taxid)} accession mappings.")
    return acc_to_taxid, taxid_to_desc

class ChronoCalibrator:
    def __init__(self, tree_path: Path, acc_to_taxid: Dict[str, int], taxid_to_desc: Dict[int, str], calibration_config: Path):
        self.tree = Phylo.read(str(tree_path), "newick")
        self.acc_to_taxid = acc_to_taxid
        self.taxid_to_desc = taxid_to_desc
        self.calibrations = load_calibrations(calibration_config)
        self.ncbi = NCBITaxa()
        self.epsilon = 0.1  # enforce strictly T_parent > T_child + epsilon Mya
        
        # Precompute lineage hierarchy for descendants
        self._precompute_taxa()

    def _precompute_taxa(self):
        """Map every leaf to its NCBI taxonomic lineage."""
        self.leaf_taxids: Dict[str, Optional[int]] = {}
        for leaf in self.tree.get_terminals():
            parts = leaf.name.split("|")
            acc = parts[1] if len(parts) >= 3 else leaf.name
            self.leaf_taxids[leaf.name] = self.acc_to_taxid.get(acc)
            
        print("Precomputing leaf taxonomic lineages...")
        unique_taxids = {t for t in self.leaf_taxids.values() if t is not None}
        self.taxid_lineages: Dict[int, List[int]] = {}
        
        # ETE3 get_lineage fetches taxids from root (1) down
        for taxid in unique_taxids:
            # Check if this taxid description is garbage (uncultured / environmental)
            desc = self.taxid_to_desc.get(taxid, "")
            if _is_garbage_lineage(desc):
                continue
            try:
                self.taxid_lineages[taxid] = self.ncbi.get_lineage(taxid)
            except Exception:
                pass
                
        print(f"  -> Loaded {len(self.taxid_lineages)} valid, high-confidence lineages.")

    def find_split_nodes(self) -> Dict[int, Clade]:
        """Locate the best representative gene tree node for each configured calibration split."""
        split_candidates: Dict[int, Tuple[Clade, float]] = {}  # TaxID -> (Node, Split Score)
        
        # Precompute descendant leaves for internal nodes recursively to optimize performance
        descendants_cache: Dict[Clade, List[str]] = {}
        def _get_descendants(node: Clade) -> List[str]:
            if node in descendants_cache:
                return descendants_cache[node]
            if node.is_terminal():
                descendants_cache[node] = [node.name]
                return descendants_cache[node]
            leaves = []
            for child in node.clades:
                leaves.extend(_get_descendants(child))
            descendants_cache[node] = leaves
            return leaves

        _get_descendants(self.tree.root)
        
        # Define high-confidence split criteria based on children lineage proportions
        target_splits = {
            131567: (2157, 2),        # Cellular: Archaea (2157) vs Bacteria (2)
            2: (1783272, 68336),     # Bacteria: Terrabacteria (1783272) vs Gracilicutes (68336)
            2157: (28890, 115781)    # Archaea: Euryarchaeota (28890) vs Proteoarchaeota (115781)
        }

        print("Evaluating domain and phylum taxonomic splits across internal nodes...")
        for node in self.tree.get_nonterminals(order="preorder"):
            if len(node.clades) != 2:
                continue
            
            c1, c2 = node.clades[0], node.clades[1]
            c1_leaves = _get_descendants(c1)
            c2_leaves = _get_descendants(c2)
            
            # Map leaves to valid high-confidence taxids
            c1_taxids = [self.leaf_taxids.get(l) for l in c1_leaves]
            c2_taxids = [self.leaf_taxids.get(l) for l in c2_leaves]
            
            c1_valid = [t for t in c1_taxids if t in self.taxid_lineages]
            c2_valid = [t for t in c2_taxids if t in self.taxid_lineages]
            
            if not c1_valid or not c2_valid:
                continue
                
            # Helper to calculate lineage coverage fraction
            def get_coverage(taxids: List[int], target_taxid: int) -> float:
                count = sum(1 for t in taxids if target_taxid in self.taxid_lineages[t])
                return count / len(taxids)

            # Evaluate each target split
            for split_taxid, (t1, t2) in target_splits.items():
                if split_taxid not in self.calibrations:
                    continue
                # Score = coverage of t1 in C1 * coverage of t2 in C2 + coverage of t2 in C1 * coverage of t1 in C2
                cov11 = get_coverage(c1_valid, t1)
                cov12 = get_coverage(c1_valid, t2)
                cov21 = get_coverage(c2_valid, t1)
                cov22 = get_coverage(c2_valid, t2)
                
                score = (cov11 * cov22) + (cov12 * cov21)
                if score > 0.3:  # Only consider high confidence splits
                    current_best = split_candidates.get(split_taxid)
                    if current_best is None or score > current_best[1]:
                        split_candidates[split_taxid] = (node, score)

        calibrated_nodes: Dict[int, Clade] = {}
        for taxid, (node, score) in split_candidates.items():
            print(f"  -> Calibrated Split Node for TaxID {taxid} ({self.ncbi.get_taxid_translator([taxid]).get(taxid, 'Unknown')}): split score = {score:.4f}, node name = {node.name}")
            calibrated_nodes[taxid] = node
            
        return calibrated_nodes

    def calibrate_chronogram(self, split_nodes: Dict[int, Clade]) -> Dict[str, float]:
        """Perform Relative-Path Interpolation to compute strictly monotonic Mya dates for all nodes."""
        if not self.tree.root.name:
            self.tree.root.name = "Root"
            
        node_depths = self.tree.depths()
        
        # Max age constraint: root is 3800 Mya, tips are 0.0 Mya
        root_age = self.calibrations.get(131567, 3800.0)
        
        node_ages: Dict[Clade, float] = {}
        
        # 1. Assign known anchor ages
        node_ages[self.tree.root] = root_age
        for taxid, node in split_nodes.items():
            node_ages[node] = self.calibrations.get(taxid, 3000.0)
            
        # Set all leaf tip ages to 0.0
        for leaf in self.tree.get_terminals():
            node_ages[leaf] = 0.0
            
        # 2. Linear relative rate path scaling recursive function
        def interpolate(node: Clade, upper_anchor: Clade, upper_age: float, lower_anchor: Clade, lower_age: float):
            """Linearly scale internal node ages between an upper and lower calibrated anchor."""
            if node in node_ages and node != upper_anchor and node != lower_anchor:
                return
            
            d_upper = node_depths[upper_anchor]
            d_lower = node_depths[lower_anchor]
            d_node = node_depths[node]
            
            # Linear scaling
            if d_lower > d_upper:
                frac = (d_node - d_upper) / (d_lower - d_upper)
                age = upper_age - (upper_age - lower_age) * frac
                node_ages[node] = age
            else:
                node_ages[node] = lower_age

        # Trace and find next calibrated anchors down the lineage
        def traverse_and_interpolate(current_node: Clade, last_anchor: Clade, last_age: float):
            if current_node.is_terminal():
                return
                
            # If current_node is calibrated itself, it becomes the new anchor
            if current_node in node_ages and current_node != self.tree.root:
                anchor = current_node
                age = node_ages[current_node]
            else:
                anchor = last_anchor
                age = last_age
                
            # For each child, find its next descendant anchor (or leaf tip)
            for child in current_node.clades:
                # Find the closest calibrated descendant or tip
                desc_anchor = None
                desc_age = 0.0
                
                # Helper BFS to find closest calibrated descendant node in child's subtree
                queue = [child]
                while queue:
                    curr = queue.pop(0)
                    if curr in node_ages and curr != self.tree.root and curr not in split_nodes.values():
                        # Tip or already set node
                        desc_anchor = curr
                        desc_age = node_ages[curr]
                        break
                    if curr in split_nodes.values():
                        # Calibrated split node
                        desc_anchor = curr
                        desc_age = node_ages[curr]
                        break
                    if not curr.is_terminal():
                        queue.extend(curr.clades)
                        
                if desc_anchor is None:
                    desc_anchor = child
                    desc_age = 0.0
                    
                # Interpolate child
                interpolate(child, anchor, age, desc_anchor, desc_age)
                traverse_and_interpolate(child, anchor, age)

        traverse_and_interpolate(self.tree.root, self.tree.root, root_age)
        
        # 3. Post-process to guarantee strict monotonicity: T_parent >= T_child + epsilon
        # Postorder traversal to propagate from leaves up
        clades_postorder = list(self.tree.find_clades(order="postorder"))
        for node in clades_postorder:
            if node.is_terminal():
                continue
            min_parent_age = max(node_ages.get(child, 0.0) for child in node.clades) + self.epsilon
            if node_ages[node] < min_parent_age:
                node_ages[node] = min_parent_age
                
        # Preorder traversal to propagate constraints from root down
        clades_preorder = list(self.tree.find_clades(order="preorder"))
        for node in clades_preorder:
            if node == self.tree.root:
                continue
            # Find parent node
            parent = None
            for p in clades_preorder:
                if node in p.clades:
                    parent = p
                    break
            if parent is not None:
                max_child_age = node_ages[parent] - self.epsilon
                if node_ages[node] > max_child_age:
                    node_ages[node] = max_child_age
                    
        # Return name-keyed ages
        return {node.name: float(age) for node, age in node_ages.items() if node.name}

def plot_chronological_switch_timeline(
    duplications_path: Path,
    speciations_path: Path,
    node_ages: Dict[str, float],
    output_svg: Path
) -> None:
    """Plot high-quality chronological switch timelines showing evolutionary switch density over time."""
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    
    def load_event_ages(path: Path) -> List[float]:
        if not path.exists():
            return []
        df = pd.read_csv(path)
        if df.empty or "score" not in df.columns:
            return []
            
        node_column = None
        for candidate in ("duplication_node", "lca_node_name"):
            if candidate in df.columns:
                node_column = candidate
                break
        if not node_column:
            return []
            
        ages = []
        
        for layer_idx, layer_df in df.groupby("layer_index"):
            n_comps = layer_df["pair"].nunique()
            if n_comps == 0:
                continue
                
            scores = pd.to_numeric(layer_df["score"], errors="coerce").dropna()
            if scores.empty:
                continue
                
            threshold = np.percentile(scores, 95)
            switched_df = layer_df[layer_df["score"] >= threshold]
            
            for idx, row in switched_df.iterrows():
                node = str(row[node_column])
                if node in node_ages:
                    ages.append(node_ages[node])
                    
        return ages

    dup_ages = load_event_ages(duplications_path)
    spec_ages = load_event_ages(speciations_path)
    
    print(f"Plotting Chronological Switch Timeline:")
    print(f"  -> Mapped {len(dup_ages)} high-confidence Duplications.")
    print(f"  -> Mapped {len(spec_ages)} high-confidence Speciations.")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    
    if dup_ages:
        sns.kdeplot(
            x=dup_ages,
            color="#B24A2A",
            fill=True,
            alpha=0.25,
            linewidth=2.0,
            label="Duplications",
            ax=ax,
            bw_adjust=0.7
        )
    if spec_ages:
        sns.kdeplot(
            x=spec_ages,
            color="#2A6FB2",
            fill=True,
            alpha=0.25,
            linewidth=2.0,
            label="Speciations",
            ax=ax,
            bw_adjust=0.7
        )
        
    milestones = [
        (3800, "LUCA / Origin of Cellular Life", "#444444"),
        (2400, "Great Oxidation Event (GOE)", "#55B22A"),
        (1500, "Origin of Eukaryotes", "#7A2AB2"),
        (540, "Cambrian Explosion", "#B28E2A")
    ]
    
    y_limits = ax.get_ylim()
    y_max = y_limits[1] if len(y_limits) > 1 and y_limits[1] > 0 else 0.01
    
    for age, label, color in milestones:
        ax.axvline(x=age, color=color, linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(
            age - 50,
            y_max * 0.85,
            label,
            rotation=90,
            ha="right",
            va="top",
            fontsize=9,
            color=color,
            fontweight="semibold"
        )
        
    ax.set_xlim(4000, 0)
    ax.set_title("Absolute Geological Timeline of Functional Specificity Divergence", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Absolute Geological Time (Millions of Years Ago, Mya)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Density of Switch Events", fontsize=11, fontweight="semibold")
    ax.legend(frameon=True, loc="upper right", facecolor="white", edgecolor="none")
    ax.set_xticks(np.arange(0, 4100, 500))
    
    fig.tight_layout()
    fig.savefig(output_svg, format="svg")
    fig.savefig(output_svg.with_suffix(".png"), format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  -> Saved chronological switch timeline plot to: {output_svg}")

def run_chronology_pipeline(
    tree_path: Path,
    fasta_path: Path,
    calibration_config: Path,
    duplications_path: Path,
    speciations_path: Path,
    output_svg: Path
) -> Dict[str, float]:
    """Execute the complete time-calibrated chronogram pipeline."""
    acc_to_taxid, taxid_to_desc = parse_accession_taxids(fasta_path)
    calibrator = ChronoCalibrator(
        tree_path=tree_path,
        acc_to_taxid=acc_to_taxid,
        taxid_to_desc=taxid_to_desc,
        calibration_config=calibration_config
    )
    split_nodes = calibrator.find_split_nodes()
    node_ages = calibrator.calibrate_chronogram(split_nodes)
    
    plot_chronological_switch_timeline(
        duplications_path=duplications_path,
        speciations_path=speciations_path,
        node_ages=node_ages,
        output_svg=output_svg
    )
    return node_ages

if __name__ == "__main__":
    # Self-run default inputs
    run_chronology_pipeline(
        tree_path=Path("data/interim/asr_run.treefile"),
        fasta_path=Path("data/interim/IPR019888_length_filtered.fasta"),
        calibration_config=Path("data/time_calibration.json"),
        duplications_path=Path("results/badasp_scoring/raw_pairwise_duplications.csv"),
        speciations_path=Path("results/badasp_scoring/raw_pairwise_speciations.csv"),
        output_svg=Path("results/evolutionary_analysis/chronological_switch_timeline.svg")
    )
