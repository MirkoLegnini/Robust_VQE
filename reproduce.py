#!/usr/bin/env python
"""
reproduce.py
============

Reproduces the numerical results of Sec. VII (Figs. 2-7, Tables I-II and the
numbers quoted in the text). See README.md.

    python reproduce.py figures                  # figures and tables from the reference results (seconds)
    python reproduce.py quick                    # smoke test of the whole pipeline (~15 min, 2 cores)
    python reproduce.py all                      # every experiment, then figures and tables (~75 min, 2 cores)
    python reproduce.py compare runs reference   # a fresh run against the reference results

    python reproduce.py run continuation TFIM:coherent QAOA     # selected jobs only
    python reproduce.py run convergence convergence:TFIM:bitflip
    python reproduce.py run flat
"""

import os

# One thread per process (as in the reference runs); must be set before numpy/jax load.
# Spawned worker processes inherit these variables.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import argparse
import sys


def _outputs(results_dir, outdir, min_points=None):
    from robustvqe import figures, tables
    if min_points is not None:
        figures.MIN_POINTS = min_points
    figures.make_all(results_dir, os.path.join(outdir, "figures"))
    tables.make_all(results_dir, os.path.join(outdir, "tables"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_all = sub.add_parser("all", help="run every experiment, then figures and tables")
    p_quick = sub.add_parser("quick", help="smoke test with reduced settings (not the paper's results)")
    p_run = sub.add_parser("run", help="run selected parts or jobs")
    p_run.add_argument("part", choices=["flat", "continuation", "convergence"])
    p_run.add_argument("select", nargs="*", help="jobs, e.g. TFIM:coherent, QAOA, convergence:TFIM:bitflip")
    for p, default in ((p_all, "runs"), (p_quick, "quick"), (p_run, "runs")):
        p.add_argument("--out", default=default, help=f"output folder (default: {default})")
        p.add_argument("--workers", type=int, default=2, help="parallel processes (default: 2)")
    p_fig = sub.add_parser("figures", help="figures and tables from stored results")
    p_fig.add_argument("--results", default="reference", help="results folder (default: reference)")
    p_fig.add_argument("--out", default="output", help="output folder (default: output)")
    p_cmp = sub.add_parser("compare", help="compare a fresh run with the reference results")
    p_cmp.add_argument("new")
    p_cmp.add_argument("reference", nargs="?", default="reference")
    p_cmp.add_argument("--gtol", type=float, default=0.05,
                       help="tolerance on the geometric means (default: 0.05)")
    args = ap.parse_args(argv)

    from robustvqe import config, runner
    if args.cmd == "all":
        runner.run(config.full(), args.out, workers=args.workers)
        _outputs(args.out, args.out)
    elif args.cmd == "quick":
        runner.run(config.quick(), args.out, workers=args.workers)
        _outputs(args.out, args.out, min_points=1)
        print(f"\nsmoke test passed: results, figures and tables in {args.out}/ "
              "(reduced settings: not the paper's numbers)")
    elif args.cmd == "run":
        runner.run(config.full(), args.out, parts=(args.part,), select=args.select,
                   workers=args.workers)
    elif args.cmd == "figures":
        _outputs(args.results, args.out)
    elif args.cmd == "compare":
        from robustvqe.compare import compare
        return 0 if compare(args.new, args.reference, gtol=args.gtol) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
