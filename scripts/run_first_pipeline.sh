#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${repo_root}/venv/bin/python"
snakemake_bin="${python_bin} -m snakemake"
snakefile="${repo_root}/Snakefile"

if ! command -v cd-hit >/dev/null 2>&1; then
  echo "Missing prerequisite: cd-hit" >&2
  exit 1
fi
if ! command -v famsa >/dev/null 2>&1; then
  echo "Missing prerequisite: famsa" >&2
  exit 1
fi
if ! command -v trimal >/dev/null 2>&1; then
  echo "Missing prerequisite: trimal" >&2
  exit 1
fi
if ! command -v iqtree2 >/dev/null 2>&1; then
  echo "Missing prerequisite: iqtree2" >&2
  exit 1
fi

has_alerax=0
if command -v alerax >/dev/null 2>&1; then
  has_alerax=1
fi

cd "${repo_root}"

if [[ "${has_alerax}" -eq 1 ]]; then
  exec ${snakemake_bin} --snakefile "${snakefile}" --cores 1 --rerun-incomplete "$@"
fi

echo "alerax not found on PATH; running the pipeline through AleRax input generation only." >&2
exec ${snakemake_bin} --snakefile "${snakefile}" --cores 1 --rerun-incomplete --until build_alerax_families "$@"