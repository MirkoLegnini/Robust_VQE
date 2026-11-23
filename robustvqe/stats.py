"""
robustvqe/stats.py
==================

Loading the stored results and the statistics shown in the figures and tables:
per perturbation level, the geometric mean of the distance over the accepted
(certified) perturbed minimisers and the 95% confidence interval of the
geometric mean (Student t on the logarithms).
"""

import glob
import json
import os

import numpy as np
from scipy.stats import t as student_t


def load(results_dir):
    """Continuation results, one dict per problem:
    res[name] = dict(spec, info, eps, problem, nominal={kind: [...]},
    records={kind: [...]}, counts={kind: [...]})."""
    out = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "results_*.json"))):
        with open(path) as fh:
            r = json.load(fh)
        g = out.setdefault(r["group"], dict(group=r["group"], spec=r["spec"], info=r["info"],
                                            eps=r["eps"], problem=r["problem"], nominal={},
                                            records={}, counts={}))
        g["records"][r["kind"]] = r["records"]
        g["counts"][r["kind"]] = r.get("counts")
        g["nominal"][r["kind"]] = r["nominal"]
    return out


def load_convergence(results_dir):
    """Convergence results: res[(group, kind)] = dict(json..., traces, trace_every).
    Every record gets f_ref, the minimum value its gap is measured against: the
    exact l~* where known (SU(N) ansatz), else the lowest value over all runs of
    the same noise model and eps (QAOA)."""
    res = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "convergence_*.json"))):
        with open(path) as fh:
            r = json.load(fh)
        z = np.load(path[:-5] + ".npz")
        r["traces"], r["trace_every"] = z["traces"], int(z["trace_every"])
        for e in r["eps"]:
            recs = [x for x in r["records"] if x["eps"] == e]
            best = min(x["f"] for x in recs)
            for x in recs:
                x["f_ref"] = x["f_known"] if x["f_known"] is not None else best
        res[(r["group"], r["kind"])] = r
    return res


def load_flat(results_dir):
    path = os.path.join(results_dir, "flat_minimum.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def geo_stats(values, level=0.95):
    """Geometric mean and confidence interval of the geometric mean."""
    v = np.asarray(values, dtype=float)
    v = v[v > 0]
    n = len(v)
    if n == 0:
        return dict(n=0, gmean=np.nan, lo=np.nan, hi=np.nan)
    lg = np.log(v)
    m = lg.mean()
    half = student_t.ppf(0.5 + level / 2, n - 1) * lg.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return dict(n=n, gmean=float(np.exp(m)), lo=float(np.exp(m - half)), hi=float(np.exp(m + half)))


def curve(result, kind, min_points=1, value=None):
    """Per-epsilon statistics over the ACCEPTED points of one curve of
    value(record) (default: the distance d)."""
    value = value or (lambda r: r["d"])
    recs = result["records"][kind]
    rows = []
    for i, e in enumerate(result["eps"]):
        st = geo_stats([value(r) for r in recs if r["i"] == i and r["ok"]])
        st.update(eps=float(e), total=sum(1 for r in recs if r["i"] == i))
        rows.append(st)
    return [r for r in rows if r["n"] >= min_points]


def per_draw_curve(eps, per_draw):
    """The same statistics for an array of distances (draws x eps)."""
    d = np.asarray(per_draw)
    return [dict(geo_stats(d[:, i]), eps=float(e)) for i, e in enumerate(eps)]


def loglog_slope(eps, values):
    """Least-squares slope of log(values) against log(eps)."""
    return float(np.polyfit(np.log(eps), np.log(values), 1)[0])
