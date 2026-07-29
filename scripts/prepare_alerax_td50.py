#!/usr/bin/env python3
"""
prepare_alerax_td50.py

Generate AleRax model parametrization files with T:D = 50:1, using the
converged T and L rates from existing runs. D is set to T / 50 for
both the initial run and the rec_check run.

Outputs are written to data/alerax_td50/ and are referenced by the
corresponding Euler sbatch scripts.

Approach follows Bremer et al. (2022) and Tria & Martin (2021), which
show the default 1:1 T:D prior in AleRax systematically inflates
duplications; realistic T:D for prokaryotes is 50–100:1.
"""

import argparse
from pathlib import Path


TD_RATIO = 50.0


def make_td50_params(src: Path, dst: Path) -> None:
    lines = src.read_text().splitlines()
    header = lines[0]  # "node D L T"

    out_lines = [header]
    for line in lines[1:]:
        parts = line.split()
        if len(parts) != 4:
            out_lines.append(line)
            continue
        node, d_str, l_str, t_str = parts
        t_val = float(t_str)
        l_val = float(l_str)
        d_new = t_val / TD_RATIO
        out_lines.append(f"{node} {d_new:.6g} {l_val:.6g} {t_val:.6g}")

    dst.write_text("\n".join(out_lines) + "\n")
    print(f"  Written {len(out_lines) - 1} rate rows to {dst}")

    # Print summary
    first = lines[1].split()
    t_old = float(first[3])
    d_old = float(first[1])
    d_new = t_old / TD_RATIO
    print(f"  D:  {d_old:.6g} → {d_new:.6g} (÷{TD_RATIO:.0f})")
    print(f"  T:  {t_old:.6g} (unchanged)")
    print(f"  L:  {float(first[2]):.6g} (unchanged)")
    print(f"  T:D ratio: {t_old / d_new:.1f}:1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create T:D=50:1 AleRax parametrization files")
    parser.add_argument(
        "--init-model-params",
        type=Path,
        default=Path("results/reconciliation/alerax/IPR019888/model_parameters/model_parameters.txt"),
    )
    parser.add_argument(
        "--rec-model-params",
        type=Path,
        default=Path("results/rec_check/output_rec_check/model_parameters/model_parameters.txt"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/alerax_td50"),
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print("=== Initial run ===")
    make_td50_params(
        args.init_model_params,
        args.outdir / "model_params_initial_td50.txt",
    )

    print("\n=== rec_check run ===")
    make_td50_params(
        args.rec_model_params,
        args.outdir / "model_params_reccheck_td50.txt",
    )

    print("\nDone. Use these files with:")
    print("  --model-parametrization data/alerax_td50/model_params_initial_td50.txt --fix-rates")
    print("  --model-parametrization data/alerax_td50/model_params_reccheck_td50.txt --fix-rates")


if __name__ == "__main__":
    main()
