#!/bin/bash
# Print md5 + size for every input the null-calibration jobs read.
# Run on BOTH machines and diff the output. An input that differs
# silently changes results rather than raising an error -- the
# AleRax samples file in particular decides which nodes are scored
# at all, and a stale copy cut the scored node pairs to 59% without
# any error until the post-ASR key check.
#
# Usage: verify_euler_inputs.sh [project root]
# Defaults to /cluster/project/beltrao/lucla/repos/badasp (Euler); pass your local repo
# root when running it on the laptop. md5sum and md5 are both handled.
set -u
ROOT="${1:-/cluster/project/beltrao/lucla/repos/badasp}"
if [ ! -d "$ROOT" ]; then
  echo "No such project root: $ROOT" >&2
  echo "Usage: $0 [project root]" >&2
  exit 2
fi
cd "$ROOT" || exit 2
echo "# project root: $ROOT"
for f in \
  data/interim/IPR019888_trimmed.aln \
  data/interim/iqtree_asr/IPR019888.treefile \
  data/interim/iqtree_asr/IPR019888_rooted.tree \
  results/reconciliation/alerax/IPR019888/reconciliations/IPR019888.nwk \
  results/reconciliation/alerax/IPR019888/reconciliations/all/IPR019888_samples.newick \
  results/badasp_scoring/raw_node_scores.csv \
; do
  if [ -f "$f" ]; then
    printf "%-78s %s %s\n" "$f" "$( (md5sum "$f" 2>/dev/null || md5 -q "$f") | cut -c1-32 )" "$(wc -c <"$f" | tr -d " ")"
  else
    printf "%-78s %s\n" "$f" "MISSING"
  fi
done
