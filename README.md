# BADASP Pipeline (IPR019888)

## Purpose
This repository implements a reproducible BADASP-inspired computational pipeline focused on the IPR019888 transcription factor family. The workflow performs sequence ingestion, quality-controlled alignment/phylogeny generation, and duplication-directed BADASP scoring to support downstream specificity-determining position analysis.

## Current Status
- **Phase 1 (Architecture & Data Ingestion)**: ✓ Complete
- **Phase 2 (Alignment & Phylogeny)**: ✓ Complete — CD-HIT (default 0.80), FAMSA/MAFFT, trimAl, native OpenMP FastTreeMP
- **Phase 3 (Topological Subfamily Clustering)**: ✓ Complete — archival support only; downstream scoring now uses duplication-directed clade pairs
- **Phase 4 (Ancestral Sequence Reconstruction)**: ✓ Complete — single-pass global IQ-TREE2 ASR with hierarchical LCA extraction
- **Phase 5 (Restricted BADASP Scoring)**: ✓ Complete — Dual-Track (Duplications & Speciations) scoring, LDO/MDO asymmetric branch tagging, multi-layer evaluation
- **Phase 6 (Structural Mapping)**: ✓ Complete — PyMOL/ChimeraX script generation, sequence-to-structure alignment mapping
- **Phase 7 (Evolutionary & Physicochemical Analysis)**: ✓ Complete — Evolutionary timeline, structural clustering, co-evolution networks, multi-level synthesis
- **Phase 7b (Advanced Synthesis)**: ✓ Complete — Architectural domain mapping, community extraction, taxonomic distribution
- **Dendrogram Visualizations**: ✓ Complete & Refined — Orientation standardization, style cleanup (endpoint removal), architecture normalization
- All development uses TDD and the root virtual environment (`venv/`). Full test suite: **89/89 passing**.
- **Reconciliation Audit Note**: The Phase 9 reconciliation output is under active audit because taxon resolution for some inputs previously defaulted to near-all duplications; the reconciliation code now prefers the header-rich clustered FASTA and is being revalidated against tiny-tree and real-data checks.

## Methodology Summary

### Phase 1-2: Sequence Ingestion & Alignment
1. Fetch IPR019888 sequences from UniProt/InterPro (117k raw sequences).
2. Filter sequences by domain length (130-200 AA; 110k retained).
3. Perform CD-HIT representative clustering (5.9k clusters at 80% identity default for the tuned pipeline; benchmarked across 0.65-0.80).
4. Build MSA with FAMSA by default (MAFFT remains available); trim columns with trimAl (`-gt 0.2`; 165 columns).
5. Build ML phylogeny with native OpenMP FastTreeMP on Apple Silicon, with automatic fallback to FastTree if needed.

### Phase 3: Topological Subfamily Clustering
6. Root tree with the canonical MAD Python implementation (`venv/bin/mad.py`); if unavailable, fall back to midpoint rooting.
7. Perform high-efficiency O(N) tree slicing via `--multi-layer <int>` (bypassing legacy memory-intensive SciPy linkage).
8. Cut the tree evenly into N discrete topological layers.
9. Retain clades ≥5 sequences per layer.
10. Identify and extract clade Last Common Ancestors (LCAs).

### Phase 4-5: Ancestral Reconstruction & Dual-Track Scoring
11. Run IQ-TREE2 ASR once on the full alignment/tree (`-asr -T AUTO`) to infer ancestral amino acid sequences.
12. Map dynamic layer LCA nodes onto the ASR tree to extract the corresponding ancestral sequences from the global reconstruction.
13. Compute Dual-Track BADASP scores across both "Duplication" and "Speciation" events, analyzing left-vs-right clades at each threshold layer.
14. Assign LDO (Least Diverged Ortholog) and MDO (Most Diverged Ortholog) asymmetric branches utilizing direct ASR-resolved node distances.
15. Score formula: `RC - (AC * p(AC))` where RC=conservation, AC=ancestral call, p(AC)=posterior probability.
16. Calculate 95th percentile threshold on raw pairwise scores across tracks; identify Specificity Determining Positions (SDPs).

### Reconciliation Refinement: Cluster-Expanded Fuzzy Logic
The reconciliation stage expands each tree leaf (CD-HIT representative at 80% identity) to the full species set in its `.clstr` cluster before classifying events.

Why this is required:
1. Representative bias: a single metagenome/environmental representative can hide many real species inside the cluster.
2. Dense paralog families: strict binary overlap can overcall duplication in the presence of minor horizontal transfer signal.

Current reconciliation policy:
1. Cluster expansion: each leaf is assigned the union of species/taxids from all members of its CD-HIT cluster.
2. Garbage filtering: taxa labeled metagenome/environmental/uncultured are excluded from species-set construction.
3. Fuzzy classification: an internal node is treated as Speciation when overlap between left/right species sets is <=2 species or <5% of their union; otherwise Duplication.

This preserves biological signal while preventing false duplication inflation from metadata artifacts.

### Architecture Evolution: Multi-Threshold Dual-Track Scoring
Phase 5 utilizes a Multi-Threshold Dual-Track system. Rather than restricting analysis to static Group/Family/Subfamily tiers, the pipeline generates evenly distributed threshold layers across the dendrogram depth using an optimized O(N) fast-traversal slicing method (`tree_cluster.py --multi-layer 10`).

How scoring now works:
1. Ingest reconciliation logic to identify Duplication and Speciation internal nodes.
2. For each defined layer (`layer01` to `layer10`), assess clade pairs. Keep nodes whose left and right descendant clades each contain at least 5 sequences.
3. Compute distances from the immediate parent node to the ASR-resolved LCA node for both branches to establish the Least Diverged Ortholog (LDO) and Most Diverged Ortholog (MDO) tags.
4. Separate the raw pairwise metrics and threshold-passing scores into three tracks per layer: `Combined`, `Duplications_Only`, and `Speciations_Only`.
5. Downstream processing (Phase 6 mapping, Phase 7 timelines) ingests these discrete tracks dynamically to decouple event types without muddying the phylogenetic signals.

### Phase 6-7: Structural & Evolutionary Analysis (Complete)
16. Map trimmed alignment columns to PDB residue numbers; generate PyMOL/ChimeraX scripts for SDP visualization.
17. Analyze SDP evolution: phylogenetic depth timeline, 3D spatial clustering, co-evolution networks, physicochemical trajectories.
18. Perform multilevel (Groups/Families/Subfamilies) architectural domain mapping, community extraction from coevolution matrices, and taxonomic SDP distribution analysis.
19. Generate publication-ready visualizations with architectural switch distributions, compact count-based boxplots, and hierarchical dendrograms with refined styling.

## Repository Structure
- `src/`: pipeline modules
  - `data_fetcher.py`: InterPro/UniProt sequence ingestion
  - `sequence_cluster.py`: length filtering + CD-HIT clustering
  - `msa_builder.py`: MAFFT alignment + trimAl trimming
  - `tree_builder.py`: FastTree tree construction
  - `tree_cluster.py`: topological clade clustering + LCA reporting
  - `badasp_core.py`: duplication-directed BADASP scoring + SDP identification
  - `asr_runner.py`: IQ-TREE2 ancestral sequence reconstruction
  - `pdb_mapper.py`: sequence-to-structure alignment + PyMOL/ChimeraX script generation
  - `evolutionary_analysis.py`: evolutionary timeline, structural clustering, coevolution, physicochemical analysis, multilevel synthesis
  - `visualization.py`: QC and clustering visual outputs including dendrogram rendering
- `tests/`: pytest suite for all core modules
- `data/raw/`: source sequence inputs (gitignored)
- `data/interim/`: intermediate artifacts (gitignored)
- `data/processed/`: processed artifacts (gitignored)
- `results/`: vector graphics and tabular outputs organized by analysis
  - `results/sequence_filtering/`
  - `results/alignment_qc/`
  - `results/topological_clustering/`
  - `results/badasp_scoring/`
  - `results/structural_mapping/`
  - `results/evolutionary_analysis/`

## Results Organization Policy
Results are grouped by analysis purpose and never by phase number:
- `results/sequence_filtering/`: sequence-length QC outputs
- `results/alignment_qc/`: MSA quality outputs
- `results/topological_clustering/`: tree-clade assignments, LCA summaries, and dendrograms (rotated, color-refined, architecture-normalized)
- `results/badasp_scoring/`: duplication-directed BADASP scores, switch distributions, and SDP tables
  - `raw_pairwise_duplications.csv`: pooled left-vs-right clade pair scores for high-confidence duplication nodes
  - `badasp_scores_duplications.csv`: position-level pooled score table
  - `badasp_sdps_duplications.csv`: final SDP calls after pooled 95th-percentile thresholding
- `results/structural_mapping/`: ChimeraX/PyMOL visualization scripts, PDB mappings, and legends
- `results/evolutionary_analysis/`: phylogenetic timelines, structural clustering heatmaps, coevolution matrices, physicochemical shifts, architectural domain distributions, compact count-based boxplots, taxonomic SDP mapping, and multilevel synthesized outputs

Generated CSV outputs under `results/` are treated as local analysis artifacts and are not tracked in git; tree files and SVG figures remain available for committed outputs when needed.

## Reproducibility Notes
- Use root virtual environment commands, for example: `./venv/bin/python -m pytest -q`.
- Generate vector figures as SVG by default.
- `_archive_v1/` is excluded from active development and execution.
- The pipeline intentionally uses a single MAD execution path through `venv/bin/mad.py` (no separate binary-mode integration in pipeline code).
- The tree-building stage now prefers a natively compiled `venv/bin/FastTreeMP` built from source with OpenMP for multicore Apple Silicon execution; single-threaded FastTree is only a fallback.
- IQ-TREE2 benchmark outputs are written to `results/iqtree_scaling.csv` and `results/iqtree_scaling_plot.svg`; the benchmark samples 500/1000/2000/4000-sequence subsets from the full alignment/tree.
- IQ-TREE2 extrapolation plotting now marks the 24,608-sequence 0.80 threshold and saves the result to `results/iqtree_scaling_plot_extrapolated.svg`.
