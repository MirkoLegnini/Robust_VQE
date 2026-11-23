"""
robustvqe/convergence.py
========================

Robust convergence (Theorem 3, Sec. VII.B; Fig. 5 and Table II): gradient
descent (6) on the PERTURBED cost of the TFIM VQE (locally surjective SU(N)
ansatz) converges to a global minimiser of the perturbed cost, which lies
within O(eps) of the noise-free optimal set Theta*. QAOA, which is not
locally surjective, is the contrast: gradient descent ends in different local
minima, with and without noise.

For every case of config.CONVERGENCE["cases"], noise model and eps (eps = 0 is
the noise-free problem), and each of `runs` initial points theta^0 drawn
uniformly from [-pi, pi)^p (the SAME points for every noise model and eps; one
error direction per problem and noise model, the same for all runs and eps):

 1. K iterations of gradient descent (6) with a fixed step gamma on the
    perturbed cost; l~(theta^k) is recorded (every `trace_every` iterations),
    with the number of iterations at which it increased.
 2. The limit point: the descent is continued with L-BFGS from theta^K (under
    bit flips the last part of the descent runs along directions whose
    curvature is of order eps), polished with LM and certified as a local
    minimiser (gradient, Hessian).
 3. Global or not: its cost against the exact minimum value l~* of the
    perturbed cost (SU(N) ansatz: E0 without noise and for coherent errors,
    lambda_min(C(O)) for bit flips). QAOA: no exact minimum value is known.
 4. SU(N) ansatz: dist(theta^K, Theta*) after gradient descent and
    dist(theta^infty, Theta*) at the limit.
"""

import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import circuits, config, hamiltonians
from . import noise as nzm
from .continuation import known_min
from .optimizer import lbfgs, project, refine


def gd_scan(qnode, gamma, iters):
    """Jitted gradient descent (6): returns theta^K and l~(theta^k), k < K."""
    vg = jax.value_and_grad(qnode, argnums=0)

    def run(x0, nz):
        def body(x, _):
            v, g = vg(x, nz)
            return x - gamma * g, v
        return jax.lax.scan(body, x0, None, length=iters)

    return jax.jit(run)


def run_job(case, kind, S, outdir):
    """All runs of one problem under one noise model ("none": eps = 0)."""
    t_start = time.time()
    name = case["group"]
    gidx = [g["name"] for g in config.PROBLEMS].index(name)
    spec = config.PROBLEMS[gidx]
    tag = f"{name}_{kind}"
    os.makedirs(outdir, exist_ok=True)
    problem = hamiltonians.build(spec["problem"], **spec.get("problem_kwargs", {}))
    reachable = spec["ansatz"] == "su"
    q, info = circuits.build(spec, problem, kind)
    cn = circuits.Compiled(q, info, kind)
    q0, _ = circuits.build(spec, problem, nzm.NONE)
    c0 = circuits.Compiled(q0, info, nzm.NONE)
    K, runs, every = S["K"], S["runs"], S["trace_every"]
    gd = gd_scan(q, case["gamma"], K)
    p, L, n = info["n_params"], info["n_layers"], info["n_qubits"]
    e0 = problem["e0"]
    eps_list = [0.0] if kind == nzm.NONE else [e for e in S["eps"] if e > 0]
    ki = 0 if kind == nzm.NONE else case["noises"].index(kind) + 1
    out = dict(group=name, kind=kind, eps=eps_list, runs=runs, K=K, gamma=case["gamma"],
               e0=e0, tol_e=S["tol_e"], info=info, records=[],
               problem={k: v for k, v in problem.items() if k not in ("matrix", "cost_hamiltonian")})
    traces = np.full((len(eps_list), runs, K // every), np.nan)
    logf = open(os.path.join(outdir, f"log_convergence_{tag}.txt"), "w")

    def log(msg):
        print(f"[convergence {tag}] {msg}", flush=True)
        logf.write(msg + "\n")
        logf.flush()

    # ONE error direction per noise model: every run minimises the same perturbed cost
    draw = None if kind == nzm.NONE else nzm.draw(
        kind, np.random.default_rng([S["seed"], gidx, ki]), L, n, p)
    log(f"{problem['name']}: p={p}, gamma={case['gamma']}, K={K}, {runs} runs per eps")
    for ie, eps in enumerate(eps_list):
        for s in range(runs):
            t0 = time.time()
            x0 = np.random.default_rng([S["seed"], gidx, s]).uniform(-np.pi, np.pi, p)
            nz = {} if kind == nzm.NONE else nzm.params(kind, draw, eps, L, n)
            # 1. gradient descent (6)
            xK, vals = gd(jnp.asarray(x0), nz)
            xK, vals = np.asarray(xK), np.asarray(vals)
            increases = int(np.sum(np.diff(vals) > 1e-12 * (1.0 + np.abs(vals[:-1]))))
            traces[ie, s] = vals[::every]
            f_known = None
            if reachable:
                f_known = e0 if kind == nzm.NONE else known_min(kind, eps, nz, problem, info,
                                                                e0, reachable)
            rec = dict(group=name, kind=kind, eps=eps, run=s, f_gd=float(cn.f(xK, nz)),
                       g_gd=float(np.linalg.norm(cn.g(xK, nz))), increases=increases,
                       f_known=f_known)
            if reachable:
                rec["dist_gd"] = project(c0, xK, {}, e0)[0]
            # 2. continue the descent, polish, certify
            xl, n_bfgs = lbfgs(cn, xK, nz)
            xl, cc, it = refine(cn, xl, nz)
            rec.update(lbfgs_iters=n_bfgs, lm_iters=it["lm_iters"], sf_iters=it["sf_iters"],
                       f=cc["f"], g=cc["g"], lmin=cc["lmin"], lmax=cc["lmax"],
                       n_flat=cc["n_flat"], cert_ok=cc["ok"], move=float(np.linalg.norm(xl - xK)))
            # 3. global minimiser?
            if f_known is not None:
                rec["gap"] = cc["f"] - f_known
                rec["gap_gd"] = rec["f_gd"] - f_known
                rec["global"] = bool(cc["ok"] and abs(rec["gap"]) <= S["tol_e"])
            # 4. distance of the limit to the noise-free optimal set
            if reachable:
                d, _, cr = project(c0, xl, {}, e0)
                rec.update(dist=d, proj_ok=bool(cr["ok"] and cr["same"]))
            out["records"].append(rec)
            msg = (f"eps={eps:.2e} run {s}: GD {rec['f_gd']:.10f} (|g| {rec['g_gd']:.1e}, "
                   f"{increases} increases) -> limit {cc['f']:.10f} certified={cc['ok']}")
            if f_known is not None:
                msg += (f" gap={rec['gap']:+.1e} global={rec['global']} dist_gd={rec['dist_gd']:.2e}"
                        f" dist={rec['dist']:.2e}")
            log(msg + f" [{time.time() - t0:.0f}s]")
        with open(os.path.join(outdir, f"convergence_{tag}.json"), "w") as fh:
            json.dump(out, fh)
        np.savez_compressed(os.path.join(outdir, f"convergence_{tag}.npz"),
                            eps=np.array(eps_list), trace_every=every, traces=traces)
    log(f"done in {time.time() - t_start:.0f}s")
    logf.close()
    return f"convergence {tag}"


def jobs():
    """(case index, noise kind) of every job; the longest first."""
    js = [(ci, kind) for ci, c in enumerate(config.CONVERGENCE["cases"])
          for kind in [nzm.NONE] + c["noises"]]
    js.sort(key=lambda j: (config.CONVERGENCE["cases"][j[0]]["group"] != "TFIM", j[1] != "bitflip"))
    return js


def run_one(job):
    """Pool entry point: job = (case index, kind, settings, outdir)."""
    ci, kind, S, outdir = job
    return run_job(S["cases"][ci], kind, S, outdir)
