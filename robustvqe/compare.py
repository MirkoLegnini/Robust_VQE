"""
robustvqe/compare.py
====================

Compares a fresh run with the reference results.

A curve agrees if (i) the same points are accepted, (ii) the minimum values
agree to 1e-9, and (iii) the plotted quantity -- the geometric mean of the
distance at every epsilon -- agrees to `gtol` (default 5%; the 95% confidence
bands in the figures are 20-50% wide). Distances at the level of round-off
(below 1e-12, e.g. under global depolarization) count as equal. The largest
per-point difference is printed for information: it is round-off for most
curves, but under bit flips the SU(N) minimisers lie in a nearly flat valley,
whose end point can move by a few percent with the floating-point details
(other hardware or BLAS builds).
"""

import glob
import json
import os

import numpy as np

ROUND_OFF = 1e-12


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _gmean(values):
    v = np.asarray([x for x in values if x is not None and x > 0], dtype=float)
    return float(np.exp(np.mean(np.log(v)))) if len(v) else float("nan")


def _gmean_diff(pairs):
    """Largest relative difference of the geometric means (new, reference) over
    epsilon; pairs at round-off on both sides count as equal."""
    out = 0.0
    for ga, gb in pairs:
        if ga <= ROUND_OFF and gb <= ROUND_OFF:
            continue
        out = max(out, abs(ga / gb - 1))
    return out


def _point_diff(pairs):
    out = 0.0
    for a, b in pairs:
        if a <= ROUND_OFF and b <= ROUND_OFF:
            continue
        out = max(out, abs(a - b) / b)
    return out


def compare(new_dir, ref_dir, gtol=0.05):
    """Prints one line per result file; returns True if everything agrees."""
    lines, good = [], True

    def line(name, ok, text):
        nonlocal good
        good &= ok
        lines.append(f"{'OK  ' if ok else 'DIFF'} {name:34s} {text}")

    for ref_path in sorted(glob.glob(os.path.join(ref_dir, "results_*.json"))):
        name = os.path.basename(ref_path)
        new_path = os.path.join(new_dir, name)
        if not os.path.exists(new_path):
            line(name, False, "missing in the new run")
            continue
        a, b = _load(new_path), _load(ref_path)
        ra = {(r["seed"], r["i"]): r for r in a["records"]}
        rb = {(r["seed"], r["i"]): r for r in b["records"]}
        same = ra.keys() == rb.keys() and all(ra[k]["ok"] == rb[k]["ok"] for k in rb)
        both = [k for k in rb if k in ra and ra[k]["ok"] and rb[k]["ok"]]
        fmax = max((abs(ra[k]["f"] - rb[k]["f"]) for k in both), default=0.0)
        gdiff = _gmean_diff([(_gmean(ra[k]["d"] for k in both if k[1] == i),
                              _gmean(rb[k]["d"] for k in both if k[1] == i))
                             for i in range(len(b["eps"]))])
        pdiff = _point_diff([(ra[k]["d"], rb[k]["d"]) for k in both])
        line(name, same and fmax <= 1e-9 and gdiff <= gtol,
             f"same accepted points: {same}; geometric means differ by <= {gdiff:.1e} "
             f"(single points <= {pdiff:.1e}); minimum values by <= {fmax:.1e}")

    for ref_path in sorted(glob.glob(os.path.join(ref_dir, "convergence_*.json"))):
        name = os.path.basename(ref_path)
        new_path = os.path.join(new_dir, name)
        if not os.path.exists(new_path):
            line(name, False, "missing in the new run")
            continue
        a, b = _load(new_path), _load(ref_path)
        ra = {(r["eps"], r["run"]): r for r in a["records"]}
        rb = {(r["eps"], r["run"]): r for r in b["records"]}
        same = ra.keys() == rb.keys() and all(
            ra[k]["cert_ok"] == rb[k]["cert_ok"] and ra[k].get("global") == rb[k].get("global") for k in rb)
        fmax = max(abs(ra[k]["f"] - rb[k]["f"]) for k in rb if k in ra)
        gdiff = _gmean_diff([(_gmean(ra[k].get("dist") for k in rb if k[0] == e),
                              _gmean(rb[k].get("dist") for k in rb if k[0] == e))
                             for e in b["eps"] if e > 0 and "dist" in b["records"][0]])
        za, zb = np.load(new_path[:-5] + ".npz"), np.load(ref_path[:-5] + ".npz")
        tmax = float(np.nanmax(np.abs(za["traces"] - zb["traces"])))
        line(name, same and fmax <= 1e-9 and gdiff <= gtol,
             f"same outcomes: {same}; limit values differ by <= {fmax:.1e}, distances (geometric "
             f"means) by <= {gdiff:.1e}, GD trajectories by <= {tmax:.1e}")

    ref_path = os.path.join(ref_dir, "flat_minimum.json")
    if os.path.exists(ref_path):
        new_path = os.path.join(new_dir, "flat_minimum.json")
        if not os.path.exists(new_path):
            line("flat_minimum.json", False, "missing in the new run")
        else:
            a, b = _load(new_path), _load(ref_path)
            dh = abs(a["h_c"] - b["h_c"])
            gdiff = _gmean_diff(zip(a["dist"], b["dist"]))
            pdiff = _point_diff(zip(np.ravel(a["per_draw_dist"]), np.ravel(b["per_draw_dist"])))
            line("flat_minimum.json", dh <= 1e-10 and gdiff <= gtol,
                 f"h_c differs by {dh:.1e}; geometric means by <= {gdiff:.1e} (single points <= {pdiff:.1e})")
    print("\n".join(lines))
    print("all results agree" if good else "some results differ (see DIFF lines)")
    return good
