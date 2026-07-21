configfile: "config/snakemake.yaml"

RAW_FASTA = config["paths"]["raw_fasta"]
CLUSTERED_FASTA = config["paths"]["clustered_fasta"]
CLUSTERED_CLSTR = config["paths"]["clustered_clstr"]
ALIGNED_FASTA = config["paths"]["aligned_fasta"]
TRIMMED_FASTA = config["paths"]["trimmed_fasta"]
IQTREE_PREFIX = config["paths"]["iqtree_prefix"]
IQTREE_TREEFILE = f"{IQTREE_PREFIX}.contree"
IQTREE_BOOTTREES = config["paths"]["iqtree_bootstrap_trees"]
IQTREE_ROOTED_TREE = config["paths"]["iqtree_rooted_tree"]
SPECIES_TREE_SOURCE = config["paths"]["species_tree_source"]
SPECIES_TREE_NORMALIZED = config["paths"]["species_tree_normalized"]
ALERAX_MAPPING = config["paths"]["alerax_mapping"]
ALERAX_FAMILIES = config["paths"]["alerax_families"]
ALERAX_OUTPUT_DIR = config["paths"]["alerax_output_dir"]

BADASP_MIN_OCCUPANCY = config["parameters"].get("badasp_min_occupancy", 0.8)
OCC_DIR = f"results/badasp_scoring/threshold_comparison/occupancy_{int(float(BADASP_MIN_OCCUPANCY)*100)}"
EV_DEC_DIR = f"results/badasp_scoring/event_decoupling/occupancy_{int(float(BADASP_MIN_OCCUPANCY)*100)}"


rule all:
    input:
        CLUSTERED_FASTA,
        CLUSTERED_CLSTR,
        ALIGNED_FASTA,
        TRIMMED_FASTA,
        IQTREE_TREEFILE,
        IQTREE_BOOTTREES,
        IQTREE_ROOTED_TREE,
        SPECIES_TREE_NORMALIZED,
        ALERAX_MAPPING,
        ALERAX_FAMILIES,
        f"{ALERAX_OUTPUT_DIR}/reconciliations/{config['project']['family_name']}.nwk",
        "results/badasp_scoring/raw_node_scores.csv",
        "results/badasp_scoring/plots/tree_score_mapping.svg",
        f"{ALERAX_OUTPUT_DIR}/plots/event_proportions_comparison.svg",
        f"{ALERAX_OUTPUT_DIR}/plots/event_proportions_report.md",
        "results/badasp_scoring/plots/tree_losses_mapping.svg",
        "results/badasp_scoring/analyses/branch_loss_counts.csv",
        f"{OCC_DIR}/threshold_comparison_stats.csv",
        f"{OCC_DIR}/positional_switches_comparison.csv",
        f"{OCC_DIR}/switch_threshold_comparison.svg",
        f"{OCC_DIR}/hard_thresholds_domain_counts.csv",
        f"{OCC_DIR}/hard_thresholds_details.svg",
        f"{OCC_DIR}/percentile_thresholds_domain_counts.csv",
        f"{OCC_DIR}/percentile_thresholds_details.svg",
        f"{OCC_DIR}/sequence_bins_stats.csv",
        f"{OCC_DIR}/sequence_bins_score_distribution.svg",
        f"{OCC_DIR}/sequence_bins_violin_distribution.svg",
        f"{OCC_DIR}/hard_threshold_1.7_significance.svg",
        f"{EV_DEC_DIR}/decoupled_event_switches_comparison.svg",
        f"{EV_DEC_DIR}/decoupled_event_switches_hard1.7_comparison.svg",


rule cdhit_cluster:
    input:
        RAW_FASTA,
    output:
        fasta=CLUSTERED_FASTA,
        clstr=CLUSTERED_CLSTR,
    params:
        identity=config["parameters"]["cdhit_identity"],
        word_size=config["parameters"]["cdhit_word_size"],
    shell:
        """
        set -euo pipefail
        cd-hit -i {input} -o {output.fasta} -c {params.identity} -n {params.word_size}
        """


rule famsa_align:
    input:
        CLUSTERED_FASTA,
    output:
        ALIGNED_FASTA,
    shell:
        """
        set -euo pipefail
        famsa {input} {output}
        """


rule trimal_trim:
    input:
        ALIGNED_FASTA,
    output:
        TRIMMED_FASTA,
    params:
        gap_threshold=config["parameters"]["trimal_gap_threshold"],
    shell:
        """
        set -euo pipefail
        trimal -in {input} -out {output} -gt {params.gap_threshold}
        """


rule iqtree_bootstrap_tree_distribution:
    input:
        TRIMMED_FASTA,
    output:
        treefile=IQTREE_TREEFILE,
        boottrees=IQTREE_BOOTTREES,
    params:
        prefix=IQTREE_PREFIX,
        model=config["parameters"]["iqtree_model"],
        bootstrap=config["parameters"]["iqtree_bootstrap"],
        binary=config["tools"]["iqtree"],
    shell:
        """
        set -euo pipefail
        rm -f {params.prefix}.ckp.gz {params.prefix}.log
        {params.binary} -s {input} -m {params.model} -T AUTO --wbt -B {params.bootstrap} --prefix {params.prefix}
        """


rule mad_root_gene_tree:
    input:
        IQTREE_TREEFILE,
    output:
        IQTREE_ROOTED_TREE,
    params:
        python=config["tools"]["python"],
        rooting_method=config["parameters"]["gene_tree_rooting_method"],
    shell:
        """
        set -euo pipefail
        {params.python} -m src.badasp.alerax_inputs root-gene-tree \
          --boot-trees {input} \
          --output {output} \
          --rooting-method {params.rooting_method}
        """


rule normalize_species_tree:
    input:
        source=SPECIES_TREE_SOURCE,
        clustered=CLUSTERED_FASTA,
    output:
        SPECIES_TREE_NORMALIZED,
    params:
        python=config["tools"]["python"],
    shell:
        """
        set -euo pipefail
        {params.python} -m src.badasp.alerax_inputs species-tree \
          --source-tree {input.source} \
          --raw-fasta {input.clustered} \
          --output {output}
        """


rule build_alerax_mapping:
    input:
        clustered=CLUSTERED_FASTA,
        raw=RAW_FASTA,
        species_tree=SPECIES_TREE_NORMALIZED,
    output:
        ALERAX_MAPPING,
    params:
        python=config["tools"]["python"],
    shell:
        """
        set -euo pipefail
        {params.python} -m src.badasp.alerax_inputs mapping \
          --clustered-fasta {input.clustered} \
          --raw-fasta {input.raw} \
          --species-tree {input.species_tree} \
          --output {output}
        """


rule build_alerax_families:
    input:
        boottrees=IQTREE_BOOTTREES,
        mapping=ALERAX_MAPPING,
        rooted_tree=IQTREE_ROOTED_TREE,
    output:
        ALERAX_FAMILIES,
    params:
        python=config["tools"]["python"],
        family_name=config["project"]["family_name"],
    shell:
        """
        set -euo pipefail
        {params.python} -m src.badasp.alerax_inputs families \
          --boot-trees {input.boottrees} \
          --resolved-gene-tree {input.rooted_tree} \
          --mapping {input.mapping} \
          --family-name {params.family_name} \
          --output {output}
        """


rule alerax_reconcile:
    input:
        families=ALERAX_FAMILIES,
        species_tree=SPECIES_TREE_NORMALIZED,
    output:
        reconciled_tree=f"{ALERAX_OUTPUT_DIR}/reconciliations/{config['project']['family_name']}.nwk",
    threads: config.get("parameters", {}).get("alerax_threads", 1)
    params:
        binary=config["tools"]["alerax"],
        prefix=ALERAX_OUTPUT_DIR,
        consensus=f"{ALERAX_OUTPUT_DIR}/reconciliations/summaries/{config['project']['family_name']}_consensus_50.newick",
        family_name=config["project"]["family_name"],
    shell:
        """
        set -euo pipefail
        # Run AleRax DTL reconciliation.
        # Note: Large runs (21k genes / 8.8k species) require massive memory (~500GB) and wall-time (~110h).
        # Run on a single core (no MPI) as AleRax only uses MPI parallelization across multiple families.
        # Memory-savings is omitted here for high-resource cluster nodes, but can be added back if needed.
        if [ {threads} -gt 1 ]; then
            mpirun -n {threads} {params.binary} -f {input.families} -s {input.species_tree} --rec-model UndatedDTL -p {params.prefix} --prune-species-tree
        else
            {params.binary} -f {input.families} -s {input.species_tree} --rec-model UndatedDTL -p {params.prefix} --prune-species-tree
        fi

        # Post-process: ensure the expected reconciled tree exists.
        # For tree distributions (ufboot), AleRax saves the consensus tree under summaries/
        if [ ! -f "{output.reconciled_tree}" ]; then
            if [ -f "{params.consensus}" ]; then
                cp "{params.consensus}" "{output.reconciled_tree}"
            else
                # Fallback: search for any generated reconciled tree file
                FOUND_TREE=$(find "{params.prefix}/reconciliations" -name "*{params.family_name}*.newick" -o -name "*{params.family_name}*.nwk" | head -n 1)
                if [ -n "$FOUND_TREE" ]; then
                    cp "$FOUND_TREE" "{output.reconciled_tree}"
                else
                    echo "Error: No reconciled tree found!" >&2
                    exit 1
                fi
            fi
        fi
        """



rule iqtree_asr_reconciled:
    input:
        alignment=TRIMMED_FASTA,
        reconciled_tree=f"{ALERAX_OUTPUT_DIR}/reconciliations/{config['project']['family_name']}.nwk",
    output:
        state=f"data/interim/iqtree_asr/{config['project']['family_name']}.state",
        treefile=f"data/interim/iqtree_asr/{config['project']['family_name']}.treefile",
    params:
        prefix=f"data/interim/iqtree_asr/{config['project']['family_name']}",
        model=config["parameters"]["iqtree_model"],
        binary=config["tools"]["iqtree"],
    shell:
        """
        set -euo pipefail
        {params.binary} -s {input.alignment} -m {params.model} -te {input.reconciled_tree} -asr --prefix {params.prefix}
        """

rule root_and_map_asr_tree:
    input:
        treefile=f"data/interim/iqtree_asr/{config['project']['family_name']}.treefile",
    output:
        rooted_tree=f"data/interim/iqtree_asr/{config['project']['family_name']}_rooted.tree",
    params:
        python=config["tools"]["python"],
    shell:
        """
        set -euo pipefail
        {params.python} scripts/root_and_map_tree.py \
          --unrooted-tree {input.treefile} \
          --output-tree {output.rooted_tree}
        """

rule badasp_node_scoring:
    input:
        alerax_tree=f"{ALERAX_OUTPUT_DIR}/reconciliations/{config['project']['family_name']}.nwk",
        asr_tree=f"data/interim/iqtree_asr/{config['project']['family_name']}_rooted.tree",
        state=f"data/interim/iqtree_asr/{config['project']['family_name']}.state",
        alignment=TRIMMED_FASTA,
    output:
        csv="results/badasp_scoring/raw_node_scores.csv",
    params:
        python=config["tools"]["python"],
        min_clade=5,
    shell:
        """
        set -euo pipefail
        {params.python} src/badasp/scoring.py \
          --alerax-tree {input.alerax_tree} \
          --asr-tree {input.asr_tree} \
          --state {input.state} \
          --alignment {input.alignment} \
          --output {output.csv} \
          --min-clade {params.min_clade}
        """

rule plot_node_scores:
    input:
        scores="results/badasp_scoring/raw_node_scores.csv",
        tree=f"data/interim/iqtree_asr/{config['project']['family_name']}_rooted.tree",
        alignment=TRIMMED_FASTA,
    output:
        "results/badasp_scoring/plots/tree_score_mapping.svg",
    params:
        python=config["tools"]["python"],
        outdir="results/badasp_scoring/plots",
    shell:
        """
        set -euo pipefail
        {params.python} src/badasp/plot_node_scores.py \
          --scores {input.scores} \
          --tree {input.tree} \
          --alignment {input.alignment} \
          --outdir {params.outdir}
        """

rule compare_event_stats:
    input:
        scores="results/badasp_scoring/raw_node_scores.csv",
        reconciled_tree=f"{ALERAX_OUTPUT_DIR}/reconciliations/{config['project']['family_name']}.nwk",
    output:
        comparison_svg=f"{ALERAX_OUTPUT_DIR}/plots/event_proportions_comparison.svg",
        report_md=f"{ALERAX_OUTPUT_DIR}/plots/event_proportions_report.md",
    params:
        python=config["tools"]["python"],
        family=config["project"]["family_name"],
        reconc_dir=f"{ALERAX_OUTPUT_DIR}/reconciliations",
        outdir=f"{ALERAX_OUTPUT_DIR}/plots",
    shell:
        """
        set -euo pipefail
        {params.python} scripts/compare_event_stats.py \
          --family {params.family} \
          --reconc-dir {params.reconc_dir} \
          --scores {input.scores} \
          --outdir {params.outdir}
        """

rule plot_tree_losses:
    input:
        reconciled_tree=f"{ALERAX_OUTPUT_DIR}/reconciliations/{config['project']['family_name']}.nwk",
        tree=f"data/interim/iqtree_asr/{config['project']['family_name']}_rooted.tree",
    output:
        plot="results/badasp_scoring/plots/tree_losses_mapping.svg",
        csv="results/badasp_scoring/analyses/branch_loss_counts.csv",
    params:
        python=config["tools"]["python"],
        xml_dir=f"{ALERAX_OUTPUT_DIR}/reconciliations/all",
        outdir="results/badasp_scoring/plots",
    shell:
        """
        set -euo pipefail
        {params.python} src/badasp/plot_tree_losses.py \
          --tree {input.tree} \
          --xml-dir {params.xml_dir} \
          --outdir {params.outdir}
        """


rule compare_thresholds:
    input:
        scores="results/badasp_scoring/raw_node_scores.csv",
        alignment=TRIMMED_FASTA,
    output:
        stats=f"{OCC_DIR}/threshold_comparison_stats.csv",
        positional=f"{OCC_DIR}/positional_switches_comparison.csv",
        plot_svg=f"{OCC_DIR}/switch_threshold_comparison.svg",
        plot_png=f"{OCC_DIR}/switch_threshold_comparison.png",
    params:
        python=config["tools"]["python"],
        min_occupancy=BADASP_MIN_OCCUPANCY,
    shell:
        """
        set -euo pipefail
        {params.python} scripts/compare_thresholds.py \
          --scores {input.scores} \
          --alignment {input.alignment} \
          --min-occupancy {params.min_occupancy}
        """


rule plot_hard_threshold_details:
    input:
        positional=f"{OCC_DIR}/positional_switches_comparison.csv",
    output:
        csv=f"{OCC_DIR}/hard_thresholds_domain_counts.csv",
        plot_svg=f"{OCC_DIR}/hard_thresholds_details.svg",
        plot_png=f"{OCC_DIR}/hard_thresholds_details.png",
        pct_csv=f"{OCC_DIR}/percentile_thresholds_domain_counts.csv",
        pct_plot_svg=f"{OCC_DIR}/percentile_thresholds_details.svg",
        pct_plot_png=f"{OCC_DIR}/percentile_thresholds_details.png",
    params:
        python=config["tools"]["python"],
        min_occupancy=BADASP_MIN_OCCUPANCY,
    shell:
        """
        set -euo pipefail
        {params.python} scripts/plot_hard_threshold_details.py \
          --min-occupancy {params.min_occupancy} \
          --positional {input.positional}
        """


rule plot_sequence_bins_distribution:
    input:
        scores="results/badasp_scoring/raw_node_scores.csv",
        alignment=TRIMMED_FASTA,
    output:
        stats=f"{OCC_DIR}/sequence_bins_stats.csv",
        plot_svg=f"{OCC_DIR}/sequence_bins_score_distribution.svg",
        plot_png=f"{OCC_DIR}/sequence_bins_score_distribution.png",
        violin_svg=f"{OCC_DIR}/sequence_bins_violin_distribution.svg",
        violin_png=f"{OCC_DIR}/sequence_bins_violin_distribution.png",
    params:
        python=config["tools"]["python"],
        min_occupancy=BADASP_MIN_OCCUPANCY,
    shell:
        """
        set -euo pipefail
        {params.python} scripts/plot_sequence_bins_distribution.py \
          --scores {input.scores} \
          --alignment {input.alignment} \
          --min-occupancy {params.min_occupancy}
        """


rule plot_hard_threshold_significance:
    input:
        scores="results/badasp_scoring/raw_node_scores.csv",
        alignment=TRIMMED_FASTA,
        positional=f"{OCC_DIR}/positional_switches_comparison.csv",
    output:
        plot_svg=f"{OCC_DIR}/hard_threshold_1.7_significance.svg",
        plot_png=f"{OCC_DIR}/hard_threshold_1.7_significance.png",
    params:
        python=config["tools"]["python"],
        min_occupancy=BADASP_MIN_OCCUPANCY,
    shell:
        """
        set -euo pipefail
        {params.python} scripts/plot_hard_threshold_significance.py \
          --scores {input.scores} \
          --alignment {input.alignment} \
          --positional {input.positional} \
          --min-occupancy {params.min_occupancy}
        """


rule plot_decoupled_event_switches:
    input:
        scores="results/badasp_scoring/raw_node_scores.csv",
        alignment=TRIMMED_FASTA,
    output:
        stats=f"{EV_DEC_DIR}/event_decoupled_stats.csv",
        pos_csv=f"{EV_DEC_DIR}/event_positional_switches.csv",
        domain_csv=f"{EV_DEC_DIR}/event_domain_densities.csv",
        plot_svg=f"{EV_DEC_DIR}/decoupled_event_switches_comparison.svg",
        plot_png=f"{EV_DEC_DIR}/decoupled_event_switches_comparison.png",
        plot_h17_svg=f"{EV_DEC_DIR}/decoupled_event_switches_hard1.7_comparison.svg",
        plot_h17_png=f"{EV_DEC_DIR}/decoupled_event_switches_hard1.7_comparison.png",
    params:
        python=config["tools"]["python"],
        min_occupancy=BADASP_MIN_OCCUPANCY,
    shell:
        """
        set -euo pipefail
        {params.python} scripts/plot_decoupled_event_switches.py \
          --scores {input.scores} \
          --alignment {input.alignment} \
          --min-occupancy {params.min_occupancy}
        """