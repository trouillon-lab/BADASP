configfile: "config/snakemake.yaml"

RAW_FASTA = config["paths"]["raw_fasta"]
CLUSTERED_FASTA = config["paths"]["clustered_fasta"]
CLUSTERED_CLSTR = config["paths"]["clustered_clstr"]
ALIGNED_FASTA = config["paths"]["aligned_fasta"]
TRIMMED_FASTA = config["paths"]["trimmed_fasta"]
IQTREE_PREFIX = config["paths"]["iqtree_prefix"]
IQTREE_TREEFILE = f"{IQTREE_PREFIX}.treefile"
IQTREE_BOOTTREES = f"{IQTREE_PREFIX}.boottrees"
IQTREE_ROOTED_TREE = config["paths"]["iqtree_rooted_tree"]
SPECIES_TREE_SOURCE = config["paths"]["species_tree_source"]
SPECIES_TREE_NORMALIZED = config["paths"]["species_tree_normalized"]
ALERAX_MAPPING = config["paths"]["alerax_mapping"]
ALERAX_FAMILIES = config["paths"]["alerax_families"]
ALERAX_OUTPUT_DIR = config["paths"]["alerax_output_dir"]


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
        ALERAX_OUTPUT_DIR,


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
        {params.binary} -s {input} -m {params.model} -T AUTO --wbt -B {params.bootstrap} --prefix {params.prefix}
        """


rule mad_root_gene_tree:
    input:
        IQTREE_BOOTTREES,
    output:
        IQTREE_ROOTED_TREE,
    params:
        python=config["tools"]["python"],
        rooting_method=config["parameters"]["gene_tree_rooting_method"],
    shell:
        """
        set -euo pipefail
        {params.python} -m src.badasp_next.alerax_inputs root-gene-tree \
          --boot-trees {input} \
          --output {output} \
          --rooting-method {params.rooting_method}
        """


rule normalize_species_tree:
    input:
        source=SPECIES_TREE_SOURCE,
        raw=RAW_FASTA,
    output:
        SPECIES_TREE_NORMALIZED,
    params:
        python=config["tools"]["python"],
    shell:
        """
        set -euo pipefail
        {params.python} -m src.badasp_next.alerax_inputs species-tree \
          --source-tree {input.source} \
          --raw-fasta {input.raw} \
          --output {output}
        """


rule build_alerax_mapping:
    input:
        clustered=CLUSTERED_FASTA,
        raw=RAW_FASTA,
    output:
        ALERAX_MAPPING,
    params:
        python=config["tools"]["python"],
    shell:
        """
        set -euo pipefail
        {params.python} -m src.badasp_next.alerax_inputs mapping \
          --clustered-fasta {input.clustered} \
          --raw-fasta {input.raw} \
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
        {params.python} -m src.badasp_next.alerax_inputs families \
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
        directory(ALERAX_OUTPUT_DIR),
    params:
        binary=config["tools"]["alerax"],
        prefix=ALERAX_OUTPUT_DIR,
    shell:
        """
        set -euo pipefail
        {params.binary} -f {input.families} -s {input.species_tree} --rec-model UndatedDTL -p {params.prefix}
        """