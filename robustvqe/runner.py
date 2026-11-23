"""
robustvqe/runner.py
===================

Plans and runs the jobs: the flat minimum (in the calling process; it fixes
the field h_c of the same-field TFIM curve), then the continuation curves and
the convergence runs, one process per job, longest first.
"""

import copy
import json
import multiprocessing as mp
import os
import time

from . import config, continuation, convergence, flat_minimum

# approximate run time per job on one core (minutes), used to start the longest first
MINUTES = {("convergence", "TFIM", "coherent"): 22, ("convergence", "TFIM", "bitflip"): 19,
           ("continuation", "TFIM", "coherent"): 18, ("continuation", "TFIM_hc", "coherent"): 17,
           ("continuation", "TFIM", "depolarizing"): 14, ("continuation", "TFIM", "bitflip"): 12,
           ("continuation", "VQC", "coherent"): 11, ("continuation", "VQC", "bitflip"): 10,
           ("continuation", "QAOA", "bitflip"): 5, ("convergence", "QAOA", "coherent"): 4}


def continuation_jobs(S, outdir, h_c):
    """(gidx, spec, kind, settings, outdir) of every curve; the same-field TFIM curve
    reuses the seeds of the TFIM curve (gidx 0)."""
    jobs = [(g, spec, kind, S, outdir) for g, spec in enumerate(config.PROBLEMS)
            for kind in spec["noises"]]
    spec = copy.deepcopy(config.PROBLEMS[0])
    spec["name"] = config.SAME_FIELD_NAME
    spec["problem_kwargs"]["h"] = h_c
    spec["noises"] = ["coherent"]
    jobs.append((0, spec, "coherent", S, outdir))
    return jobs


def dispatch(job):
    what, args = job
    if what == "continuation":
        return continuation.run_one(args)
    return convergence.run_one(args)


def _label(job):
    what, args = job
    if what == "continuation":
        return f"{args[1]['name']}:{args[2]}"
    return f"convergence:{args[2]['cases'][args[0]]['group']}:{args[1]}"


def _minutes(job):
    what, args = job
    if what == "continuation":
        return MINUTES.get((what, args[1]["name"], args[2]), 2)
    return MINUTES.get((what, args[2]["cases"][args[0]]["group"], args[1]), 3)


def run(settings, outdir, parts=("flat", "continuation", "convergence"), select=(), workers=2):
    """Run the selected parts; `select` optionally restricts the continuation and
    convergence jobs to labels such as TFIM:coherent, QAOA, convergence:TFIM:bitflip."""
    t0 = time.time()
    os.makedirs(outdir, exist_ok=True)
    flat_path = os.path.join(outdir, "flat_minimum.json")
    if "flat" in parts:
        flat_minimum.run(settings["flat"], outdir)
    jobs = []
    if "continuation" in parts:
        if not os.path.exists(flat_path):
            raise SystemExit(f"{flat_path} missing: the flat minimum fixes h_c; run it first")
        with open(flat_path) as fh:
            h_c = json.load(fh)["h_c"]
        jobs += [("continuation", j) for j in continuation_jobs(settings["continuation"], outdir, h_c)]
    if "convergence" in parts:
        jobs += [("convergence", (ci, kind, settings["convergence"], outdir))
                 for ci, kind in convergence.jobs()]
    if select:
        jobs = [j for j in jobs if any(_label(j) == s or _label(j).startswith(s + ":") for s in select)]
    jobs.sort(key=_minutes, reverse=True)
    print(f"{len(jobs)} jobs on {workers} worker(s): " + ", ".join(_label(j) for j in jobs), flush=True)
    if workers > 1 and len(jobs) > 1:
        with mp.get_context("spawn").Pool(min(workers, len(jobs)), maxtasksperchild=1) as pool:
            for tag in pool.imap_unordered(dispatch, jobs, chunksize=1):
                print(f"*** {tag} finished after {time.time() - t0:.0f}s", flush=True)
    else:
        for j in jobs:
            dispatch(j)
    print(f"all jobs done in {time.time() - t0:.0f}s", flush=True)
