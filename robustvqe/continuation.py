"""
robustvqe/continuation.py
=========================

Perturbed optimal parameters versus the perturbation level (Sec. VII.A and
VII.C; Figs. 2, 3, 6, 7, the SU(N) curve of Fig. 4 and Table I).

One JOB = one curve: a problem (config.PROBLEMS) under one noise model. Per
initialisation theta^0 (seed):

 1. noise-free minimiser theta* from theta^0 (gradient descent (6), then
    saddle-free Newton and Levenberg-Marquardt polish), certified by first-
    and second-order conditions; for the SU(N) ansatz, which reaches the
    ground state, also l(theta*) = E0.

 2. the branch of perturbed minimisers through theta*, followed over the
    epsilon grid by numerical continuation (optimizer.track: tangent predictor
    from the Hessian, LM corrector, adaptive step). A branch that cannot be
    followed -- a fold, where the minimum merges with a saddle -- ends there,
    and its remaining points are rejected.
      * ascending from theta* (epsilon = 0): the default;
      * descending from the largest epsilon: bit flips on a DEGENERATE optimum
        (SU(N) ansatz). Bit flips break the degeneracy of Theta*, so the
        perturbed minimiser lies O(1) away from theta* ALONG Theta*, while its
        distance to the SET Theta* is small. An L-BFGS descent from theta* at
        the largest epsilon makes this trip; the branch is then followed down.
    Every perturbed minimiser is certified (gradient, Hessian, and the exactly
    known minimum value where one exists).

 3. its distance to the noise-free optimal set:
      * degenerate optimum (SU(N) ansatz, Theta* a manifold): projection onto
        Theta* (optimizer.project); the projected point must be a certified
        noise-free minimiser with the same cost as theta*;
      * isolated optimum (QAOA): |theta~* - theta*| along the branch;
      * coherent control errors: |theta~* - theta*|, with the closed form
        theta~*_j = theta*_j / (1 + eta_j) of (28) checked and the bound (29)
        recorded.

A point is "accepted" only if all certificates pass. Initialisations are added
until every epsilon has `target` accepted points (at least `min_seeds`, at
most `max_seeds` initialisations).
"""

import json
import os
import time

import numpy as np

from . import circuits, hamiltonians
from . import noise as nzm
from .optimizer import certify, minimize_from_random, project, refine, track

TOL_E = 1e-9      # cost tolerance of the known-minimum checks


def known_min(kind, eps, nz, problem, info, f_star, reachable):
    """Exact minimum value of the noisy cost, if known; else None."""
    O, n = problem["matrix"], info["n_qubits"]
    N = 2 ** n
    e_ref = problem["e0"] if reachable else f_star
    if kind in nzm.CONTROLS:
        return f_star                      # l(D theta): same minimum values
    if kind == nzm.DEPOL:
        p_tot = eps / (1.0 + eps)          # Lemmas 3-5
        return (1 - p_tot) * e_ref + p_tot * float(np.real(np.trace(O))) / N
    if not reachable:
        return None
    if kind == nzm.COHERENT:
        return problem["e0"]               # the noisy SU(N) ansatz still reaches E0
    if kind == nzm.BITFLIP:
        # 2-block SU(N): the first block can prepare a state that the bit flips of
        # the first layer leave invariant (|+>^n), so the minimum is lambda_min(C(O)),
        # C the bit-flip channel of the last layer in the Heisenberg picture.
        C = nzm.pauli_adjoint(O, n, [(float(nz["p"]), 0.0, 0.0)] * n)
        return float(np.linalg.eigvalsh(0.5 * (C + C.conj().T))[0])
    return None


def follow_branch(kind, ki, seed, xs, cert0, c0, cn, problem, info, eps_grid, reachable):
    """Records (one per epsilon, grid order) of the branch through theta* = xs."""
    p, L, n = info["n_params"], info["n_layers"], info["n_qubits"]
    E = len(eps_grid)
    f_star, degenerate = cert0["f"], cert0["n_flat"] > 0
    draw = nzm.draw(kind, np.random.default_rng([seed, ki + 1]), L, n, p)

    def nz_at(e):
        return nzm.params(kind, draw, e, L, n)

    descending = kind == nzm.BITFLIP and degenerate
    order = list(range(E))[::-1] if descending else list(range(E))
    recs = [None] * E
    x_prev, e_prev, eig_prev, alive = xs, 0.0, None, True
    foot_prev = None                        # nearest point of Theta* at the previous eps
    for step, i in enumerate(order):
        eps = float(eps_grid[i])
        nz = nz_at(eps)
        rec = dict(seed=seed, eps=eps, i=i, ok=False, tracked=False, d=None)
        recs[i] = rec
        if not alive:                               # branch ended at a previous eps
            continue
        if descending and step == 0:
            # the trip along Theta*: L-BFGS descent from theta*, then LM
            xt, cc, it = refine(cn, xs, nz, pre_lbfgs=True)
            tracked, eig_prev = cc["ok"], None
        else:
            xt, cc, it = track(cn, nz_at, x_prev, e_prev, eps, eig=eig_prev)
            tracked, eig_prev = it.pop("ok"), it.pop("eig")
        kmin = known_min(kind, eps, nz, problem, info, f_star, reachable)
        known_ok = kmin is None or abs(cc["f"] - kmin) <= TOL_E
        rec.update(tracked=tracked, noisy_ok=cc["ok"], known_ok=known_ok, f=cc["f"],
                   f_known=kmin, g=cc["g"], lmin=cc["lmin"], lmax=cc["lmax"],
                   n_flat=cc["n_flat"], dist_to_start=float(np.linalg.norm(xt - xs)), **it)
        if not tracked:                             # branch lost here: nothing to measure
            alive = False
            continue
        if kind in nzm.CONTROLS:
            eta = eps * draw["u"]
            d = float(np.linalg.norm(xt - xs))
            cf = float(np.linalg.norm(xt - xs / (1 + eta)))            # closed form (28)
            rec.update(d=d, closed_form_err=cf,
                       d_closed=float(np.linalg.norm(xs / (1 + eta) - xs)),
                       bound=float(np.max(np.abs(eta)) * np.linalg.norm(xt)))   # bound (29)
            ok = cc["ok"] and known_ok and cf <= 1e-7 * max(1.0, np.linalg.norm(xs))
        elif degenerate:
            starts = [("theta*", xs)] + ([("previous", foot_prev)] if foot_prev is not None else [])
            d, xr, cr = project(c0, xt, {}, f_star, starts=starts)
            rec.update(d=d, proj_ok=cr["ok"], same=cr["same"], floor=cr["floor"],
                       ortho=cr["ortho"])
            if cr["ok"] and cr["same"]:
                foot_prev = xr
            ok = cc["ok"] and known_ok and cr["ok"] and cr["same"]
        else:                                       # isolated minimum: |theta~* - theta*|
            d = float(np.linalg.norm(xt - xs))
            rec.update(d=d)
            ok = cc["ok"] and known_ok
        rec["ok"] = bool(ok)
        x_prev, e_prev = xt, eps                    # continue along the branch
    return recs


def run_job(gidx, spec, kind, S, outdir):
    """One curve: problem `spec` (index gidx in config.PROBLEMS, which fixes the
    seeds) under noise `kind`. Writes <outdir>/results_<name>_<kind>.json."""
    t_start = time.time()
    name, tag = spec["name"], f"{spec['name']}_{kind}"
    os.makedirs(outdir, exist_ok=True)
    logf = open(os.path.join(outdir, f"log_{tag}.txt"), "w")

    def log(msg):
        print(f"[{tag}] {msg}", flush=True)
        logf.write(msg + "\n")
        logf.flush()

    problem = hamiltonians.build(spec["problem"], **spec.get("problem_kwargs", {}))
    reachable = spec["ansatz"] == "su"
    log(f"problem: {problem['name']}  E0={problem['e0']:.6f}  degeneracy="
        f"{problem['degeneracy']}  gap={problem['gap']:.4f}")

    t0 = time.time()
    q, info = circuits.build(spec, problem, nzm.NONE)
    c0 = circuits.Compiled(q, info, nzm.NONE)
    c0.warmup()
    q, _ = circuits.build(spec, problem, kind)
    cn = circuits.Compiled(q, info, kind)
    cn.warmup()
    p, L, n = info["n_params"], info["n_layers"], info["n_qubits"]
    log(f"ansatz {spec['ansatz']}: n={n}, L={L}, p={p}; compiled in {time.time() - t0:.0f}s")

    ki = spec["noises"].index(kind)                 # fixes the error draw of each seed
    eps_grid = np.asarray(S["eps"], dtype=float)
    counts = np.zeros(len(eps_grid), dtype=int)
    out = dict(group=name, kind=kind, spec=dict(spec), info=info, eps=eps_grid.tolist(),
               problem={k: v for k, v in problem.items()
                        if k not in ("matrix", "cost_hamiltonian")},
               nominal=[], records=[], settings=dict(S))

    def save():
        out["counts"] = counts.tolist()
        out["secs"] = time.time() - t_start
        with open(os.path.join(outdir, f"results_{tag}.json"), "w") as fh:
            json.dump(out, fh)

    for s in range(S["max_seeds"]):
        if s >= S["min_seeds"] and (counts >= S["target"]).all():
            break
        seed = S["seed"] + 1000 * gidx + s
        x0 = np.random.default_rng(seed).uniform(-np.pi, np.pi, p)
        t1 = time.time()
        xs, oi = minimize_from_random(c0, x0, {}, spec["gd_step"], spec["gd_iters"])
        c = certify(c0, xs, {})
        if reachable and c["f"] - problem["e0"] > TOL_E:
            c["ok"] = False
        out["nominal"].append(dict(seed=seed, ok=c["ok"], f=c["f"], g=c["g"], lmin=c["lmin"],
                                   lmax=c["lmax"], n_flat=c["n_flat"], **oi))
        msg = (f"seed {seed}: f-E0={c['f'] - problem['e0']:.2e} |g|={c['g']:.0e} "
               f"lmin={c['lmin']:.1e} flat={c['n_flat']}")
        if not c["ok"]:
            log(msg + "  -> noise-free minimum NOT certified, seed skipped")
            continue
        t2 = time.time()
        recs = follow_branch(kind, ki, seed, xs, c, c0, cn, problem, info, eps_grid, reachable)
        out["records"] += recs
        counts += np.array([r["ok"] for r in recs], dtype=int)
        line = " ".join("x" if r["d"] is None else f"{r['d'] / r['eps']:.3g}" +
                        ("" if r["ok"] else "!") for r in recs)
        log(f"{msg} | d/eps: {line}  [{t2 - t1:.0f}+{time.time() - t2:.0f}s]")
        save()

    save()
    log(f"done in {out['secs']:.0f}s; accepted per eps: {counts.tolist()}")
    logf.close()
    return tag


def run_one(job):
    """Pool entry point: job = (gidx, spec, kind, settings, outdir)."""
    return run_job(*job)
