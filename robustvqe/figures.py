"""
robustvqe/figures.py
====================

The figures of Sec. VII from stored results (nothing is recomputed). Each
distance figure shows, per perturbation level, the geometric mean of the
distance over the accepted perturbed minimisers (markers and line) and the
95% confidence interval of the geometric mean (band); a level is drawn only
if it has at least MIN_POINTS accepted minimisers.

  Fig. 2  coherent_distance.pdf               TFIM, QAOA, VQC; coherent errors
  Fig. 3  bitflip_distance.pdf                TFIM, QAOA, VQC; bit flips
  Fig. 4  flat_vs_linear.pdf                  SU(N) ansatz and HVA, TFIM at h = h_c
  Fig. 5  convergence_traj_TFIM_bitflip.pdf   gradient descent, TFIM, no noise and bit flips
  Fig. 6  control_distance.pdf                QAOA, coherent control errors
  Fig. 7  depolarizing_distance.pdf           TFIM, global depolarization
"""

import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator

from . import config, stats

MIN_POINTS = 10
MARKER_SIZE = {"o": 10, "s": 8, "^": 7, "D": 7}
PROBLEM_STYLE = [("TFIM", "steelblue", "o"), ("QAOA", "darkorange", "s"),
                 ("VQC", "forestgreen", "^")]
EPS = r"$\epsilon$"
DIST = r"$\mathrm{dist}(\tilde\theta^*_\epsilon,\Theta^*)$"

# Fig. 5: colour per eps; trajectories are drawn until their gap reaches round-off
CONVERGED = 1e-12
EPS_COL = {0.0: "black", 1e-3: "steelblue", 1e-2: "darkorange", 1e-1: "forestgreen",
           10 ** -0.5: "firebrick"}
EPS_TXT = {0.0: "0", 1e-3: r"$10^{-3}$", 1e-2: r"$10^{-2}$", 1e-1: r"$10^{-1}$",
           10 ** -0.5: r"$10^{-1/2}$"}

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 20,
    "axes.titlesize": 22,
    "axes.labelsize": 22,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 15,
})


def _save(fig, outdir, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outdir}/{name}.pdf")


def _style_log_axis(ax):
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=9))
    ax.grid(which="major", alpha=0.4)
    ax.grid(which="minor", alpha=0.12)


def _plot_rows(ax, rows, col, mk, ls, label=None):
    """Geometric mean (markers and line) and its 95% confidence band."""
    e = np.array([r["eps"] for r in rows])
    ax.fill_between(e, [r["lo"] for r in rows], [r["hi"] for r in rows], color=col, alpha=0.25,
                    linewidth=0)
    ax.loglog(e, [r["gmean"] for r in rows], linestyle=ls, marker=mk, ms=MARKER_SIZE[mk], lw=2,
              color=col, markerfacecolor="white", markeredgewidth=1.6, label=label)


def distance(res, kind, outdir, name):
    """Figs. 2 and 3."""
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for g, col, mk in PROBLEM_STYLE:
        _plot_rows(ax, stats.curve(res[g], kind, min_points=MIN_POINTS), col, mk, "-", label=g)
    ax.set_xlabel(EPS)
    ax.set_ylabel(DIST)
    _style_log_axis(ax)
    ax.legend(loc="upper left", framealpha=0.9)
    _save(fig, outdir, name)


def flat_vs_linear(res, flat, outdir):
    """Fig. 4: the SU(N) ansatz (linear) and the HVA at its flat minimum
    (approximately eps^(1/3)), both for the TFIM at h = h_c, coherent errors."""
    sun = res[config.SAME_FIELD_NAME]
    assert abs(sun["spec"]["problem_kwargs"]["h"] - flat["h_c"]) < 1e-12, "rerun TFIM_hc at h_c"
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    _plot_rows(ax, stats.curve(sun, "coherent", min_points=MIN_POINTS), "steelblue", "o", "-",
               label=r"SU($N$) ansatz")
    _plot_rows(ax, stats.per_draw_curve(flat["eps"], flat["per_draw_dist"]), "darkorange", "s", "-",
               label=r"HVA")
    ax.set_xlabel(EPS)
    ax.set_ylabel(DIST)
    ax.set_ylim(3e-5, 1)
    _style_log_axis(ax)
    ax.legend(loc="lower right", framealpha=0.9)
    _save(fig, outdir, "flat_vs_linear")


def _eps_key(e):
    return min(EPS_COL, key=lambda k: abs(k - e))


def trajectories(conv, group, kind, outdir):
    """Fig. 5: l~(theta^k) - l~* along every gradient-descent run, coloured by
    eps (the noise-free runs in black, drawn on top)."""
    parts = [conv[(group, k)] for k in ("none", kind)]
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for r in parts[::-1]:
        k = np.arange(r["traces"].shape[-1]) * r["trace_every"] + 1
        for ie, e in enumerate(r["eps"]):
            recs = {x["run"]: x for x in r["records"] if x["eps"] == e}
            for s, tr in enumerate(r["traces"][ie]):
                if s in recs and np.all(np.isfinite(tr)):
                    gap = tr - recs[s]["f_ref"]
                    stop = int(np.argmax(gap <= CONVERGED)) + 1 if np.any(gap <= CONVERGED) else len(gap)
                    ax.loglog(k[:stop], np.maximum(gap[:stop], CONVERGED), "-",
                              color=EPS_COL[_eps_key(e)], lw=0.9, alpha=0.6)
    handles = [Line2D([], [], color=EPS_COL[_eps_key(e)], lw=2,
                      label=r"$\epsilon=$" + EPS_TXT[_eps_key(e)])
               for r in parts for e in r["eps"]]
    ax.set_ylim(CONVERGED / 3, None)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\tilde\ell(\theta^k)-\tilde\ell^*$")
    ax.grid(which="major", alpha=0.4)
    ax.legend(handles=handles, loc="lower left", framealpha=0.9, fontsize=13)
    _save(fig, outdir, f"convergence_traj_{group}_{kind}")


def control(res, outdir):
    """Fig. 6: perturbed minimisers under coherent control errors (QAOA), the
    closed form (28) and the bound (29)."""
    q = res["QAOA"]
    eps = np.array(q["eps"])
    recs = [r for r in q["records"]["control"] if r["ok"]]
    st = {k: [stats.geo_stats([r[k] for r in recs if r["i"] == i]) for i in range(len(eps))]
          for k in ("d", "d_closed", "bound")}
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for key, col, ls, mk, lab in (
            ("bound", "black", "--", None, r"$\|\eta\|_\infty\|\tilde\theta^*_\eta\|$"),
            ("d", "firebrick", "-", "D", "simulation")):
        ax.fill_between(eps, [s["lo"] for s in st[key]], [s["hi"] for s in st[key]], color=col,
                        alpha=0.15 if key == "bound" else 0.25, linewidth=0)
        ax.loglog(eps, [s["gmean"] for s in st[key]], linestyle=ls, color=col, lw=2, marker=mk,
                  ms=8, markerfacecolor="white", markeredgewidth=1.6, label=lab)
    ax.loglog(eps, [s["gmean"] for s in st["d_closed"]], "-", color="black", lw=1,
              label=r"$\theta^*_j/(1+\eta_j)$")
    ax.set_xlabel(r"$\|\eta\|_\infty$")
    ax.set_ylabel(r"$\|\tilde\theta^*_\eta-\theta^*\|$")
    _style_log_axis(ax)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=13)
    _save(fig, outdir, "control_distance")


def depolarizing(res, outdir):
    """Fig. 7: global depolarization does not move the minimisers (Lemma 5):
    the distances stay at the level of round-off."""
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    _plot_rows(ax, stats.curve(res["TFIM"], "depolarizing", min_points=MIN_POINTS), "firebrick",
               "D", "--", label="TFIM")
    ax.axhline(np.finfo(float).eps, color="gray", lw=1.2, ls=":", label="machine precision")
    ax.set_ylim(1e-17, 1e-11)
    ax.set_xlabel(EPS)
    ax.set_ylabel(DIST)
    _style_log_axis(ax)
    ax.legend(loc="upper left", framealpha=0.9)
    _save(fig, outdir, "depolarizing_distance")


def make_all(results_dir, outdir):
    os.makedirs(outdir, exist_ok=True)
    res = stats.load(results_dir)
    conv = stats.load_convergence(results_dir)
    flat = stats.load_flat(results_dir)
    distance(res, "coherent", outdir, "coherent_distance")           # Fig. 2
    distance(res, "bitflip", outdir, "bitflip_distance")             # Fig. 3
    flat_vs_linear(res, flat, outdir)                                # Fig. 4
    trajectories(conv, "TFIM", "bitflip", outdir)                    # Fig. 5
    control(res, outdir)                                             # Fig. 6
    depolarizing(res, outdir)                                        # Fig. 7
