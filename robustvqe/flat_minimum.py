"""
robustvqe/flat_minimum.py
=========================

The flat minimum of Sec. VII.A (Fig. 4, HVA row of Table I): the open TFIM
chain H = -J sum_i Z_i Z_{i+1} - h sum_i X_i on n = 4 qubits with a
Hamiltonian variational ansatz of one layer, exp(-i beta sum_i X_i)
exp(-i gamma sum_i Z_i Z_{i+1}), applied to the product state
RY(theta_1)^{(x)n} |0>, parameters x = (theta_1, gamma, beta).

 1. The flat point. The parity prod_i X_i maps theta_1 to pi - theta_1 and
    commutes with H and both layers, so the cost is even in the tilt
    t = theta_1 - pi/2. With (gamma, beta) minimised at t = 0, the curvature
    along t vanishes at one field h_c (root found by bisection); there the
    growth along t is quartic, l - l* = a t^4 + ..., with a > 0 obtained by
    eliminating (gamma, beta) exactly (a = l_tttt/24 - v^T H_ee^{-1} v / 8,
    v = d_e l_tt), while the (gamma, beta) block of the Hessian stays positive
    definite: the minimum x* remains isolated. Global minimality is checked
    from random starts.
 2. Coherent errors (the model of the other curves: after each of the two
    layers, on every qubit, a rotation by eps * u about a random Pauli axis)
    for `draws` error draws: the perturbed minimiser is followed along the eps
    grid (certified: gradient <= 1e-10, Hessian positive definite) and its
    distance |x~* - x*| recorded.

Noise is simulated on the 16 x 16 density matrix in JAX; no PennyLane.
"""

import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq, minimize

from . import noise as nzm

TOL_G = 1e-10                   # gradient tolerance of a certified minimiser
MAX_STEP = 0.3                  # largest Newton step (rad): no jumps to periodic images
LAYERS = 2                      # noise locations: after the RY layer and after the HVA layer

N = 4                           # qubits (config.FLAT["n"]; the operators below are built for it)
J = 1.0

I2 = np.eye(2, dtype=complex)
PAULIS = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=complex)


def _kron(mats):
    out = mats[0]
    for m in mats[1:]:
        out = jnp.kron(out, m)
    return out


def _embed(P, q):
    return np.asarray(_kron([P if k == q else I2 for k in range(N)]))


ZZ_DIAG = jnp.asarray(np.real(np.diag(sum(_embed(PAULIS[2], q) @ _embed(PAULIS[2], q + 1)
                                          for q in range(N - 1)))))
SUM_X = jnp.asarray(sum(_embed(PAULIS[0], q) for q in range(N)))


def hamiltonian(h):
    return -J * jnp.diag(ZZ_DIAG) - h * SUM_X


def _ry(t):
    c, s = jnp.cos(t / 2), jnp.sin(t / 2)
    return jnp.array([[c, -s], [s, c]], dtype=complex)


def layer_unitaries(x):
    c, s = jnp.cos(x[2]), jnp.sin(x[2])                      # exp(-i beta X) per qubit
    mixer = _kron([jnp.array([[c, -1j * s], [-1j * s, c]])] * N)
    return [_kron([_ry(x[0])] * N), mixer @ jnp.diag(jnp.exp(-1j * x[1] * ZZ_DIAG))]


def output_state(x, nz):
    """Density matrix after the circuit; after layer l, qubit q is rotated by
    a[l, q] about the axis with one-hot weights w[l, q]."""
    rho = jnp.zeros((2 ** N, 2 ** N), dtype=complex).at[0, 0].set(1.0)
    for l, U in enumerate(layer_unitaries(x)):
        rho = U @ rho @ jnp.conj(U).T
        E = _kron([jnp.cos(nz["a"][l, q] / 2) * I2
                   - 1j * jnp.sin(nz["a"][l, q] / 2) * jnp.tensordot(nz["w"][l, q], PAULIS, axes=1)
                   for q in range(N)])
        rho = E @ rho @ jnp.conj(E).T
    return rho


def cost(x, h, nz):
    return jnp.real(jnp.trace(hamiltonian(h) @ output_state(x, nz)))


F = jax.jit(cost)
G = jax.jit(jax.grad(cost))
H = jax.jit(jax.hessian(cost))


def coherent(d, eps):
    """Noise parameters of the coherent error draw d at level eps."""
    return dict(w=jnp.asarray(np.eye(3)[d["axes"]]), a=jnp.asarray(eps * d["u"]))


def no_noise():
    return dict(w=jnp.zeros((LAYERS, N, 3)), a=jnp.zeros((LAYERS, N)))


def error_draw(seed, s):
    """Error draw s: the coherent model of the other curves (noise.draw)."""
    return nzm.draw(nzm.COHERENT, np.random.default_rng([seed, s]), LAYERS, N, 0)


# ----------------------------------------------------------------------------
# Certified minimisers
# ----------------------------------------------------------------------------

def newton(x0, h, nz, iters=400):
    """Damped Newton iteration (eigenvalues floored in modulus, step at most
    MAX_STEP, backtracking far from the solution). Returns x, |gradient|,
    smallest Hessian eigenvalue."""
    x = np.array(x0, dtype=float)
    for _ in range(iters):
        g = np.asarray(G(x, h, nz))
        gn = np.linalg.norm(g)
        if gn < 1e-14:
            break
        lam, Q = np.linalg.eigh(np.asarray(H(x, h, nz)))
        step = -Q @ ((Q.T @ g) / np.maximum(np.abs(lam), 1e-9))
        step *= min(1.0, MAX_STEP / np.linalg.norm(step))   # stay with the nearby minimiser:
        if gn < 1e-7 and lam[0] > 0:                        # the cost is periodic in every angle
            x = x + step
            continue
        t, f0 = 1.0, float(F(x, h, nz))
        while t > 1e-14 and float(F(x + t * step, h, nz)) > f0:
            t /= 2
        x = x + t * step
    g = np.linalg.norm(np.asarray(G(x, h, nz)))
    return x, float(g), float(np.linalg.eigvalsh(np.asarray(H(x, h, nz)))[0])


def minimiser(x_star, h, nz, eps, prev=None):
    """The certified local minimiser of lowest cost reached from x_star, from
    tilts of every size that can occur (eps, eps^1/2, eps^1/3, eps^1/5) and
    from the previous solution."""
    tilt = np.eye(3)[0]
    scales = [s * k * eps ** e for e in (1.0, 0.5, 1 / 3, 0.2) for k in (0.3, 1.0)
              for s in (1.0, -1.0)]
    starts = [x_star] + [x_star + k * tilt for k in scales] + ([prev] if prev is not None else [])
    best = None
    for x0 in starts:
        x, g, lmin = newton(x0, h, nz)
        if g <= TOL_G and lmin > 0:
            fx = float(F(x, h, nz))
            if best is None or fx < best[1] - 1e-13:
                best = (x, fx)
    if best is None:
        raise RuntimeError(f"no certified minimiser (h={h}, eps={eps})")
    assert np.abs(best[0] - x_star).max() < 1.5, "minimiser on another periodic image"
    return best[0]


# ----------------------------------------------------------------------------
# The flat point
# ----------------------------------------------------------------------------

def symmetric_point(h, y0=(0.23, -3 * np.pi / 8)):
    """Minimiser over (gamma, beta) at theta_1 = pi/2 (Newton on that block)."""
    x, nz = np.array([np.pi / 2, *y0]), no_noise()
    for _ in range(100):
        g = np.asarray(G(x, h, nz))[1:]
        if np.linalg.norm(g) < 1e-14:
            break
        x[1:] -= np.linalg.solve(np.asarray(H(x, h, nz))[1:, 1:], g)
    return x


def _tilt_cost(t, e, h):
    """Noise-free cost at tilt t and (gamma, beta) = e."""
    return cost(jnp.concatenate([jnp.array([np.pi / 2]) + t, e]), h, no_noise())


_TILT_2 = jax.grad(jax.grad(_tilt_cost))
TILT_4 = jax.jit(jax.grad(jax.grad(_TILT_2)))             # l_tttt
TILT_2_EVEN = jax.jit(jax.grad(_TILT_2, argnums=1))       # d_e l_tt


def tilt_curvature(h):
    """Curvature along the tilt at the symmetric minimiser (the mixed block
    vanishes there by the parity)."""
    return float(H(symmetric_point(h), h, no_noise())[0, 0])


def landau_coefficients(h):
    """x*, mu and a of  l(t) - l* = mu t^2 / 2 + a t^4 + ...  along the tilt,
    with (gamma, beta) eliminated, and the Hessian at x*."""
    xs = symmetric_point(h)
    e = jnp.asarray(xs[1:])
    v = np.asarray(TILT_2_EVEN(0.0, e, h))
    Hm = np.asarray(H(xs, h, no_noise()))
    a = float(TILT_4(0.0, e, h)) / 24.0 - v @ np.linalg.solve(Hm[1:, 1:], v) / 8.0
    return xs, float(Hm[0, 0]), float(a), Hm


def slope(eps, values, fit_points):
    """Least-squares slope of log(values) against log(eps) on the smallest levels."""
    e, v = np.asarray(eps)[:fit_points], np.asarray(values)[:fit_points]
    return float(np.polyfit(np.log(e), np.log(v), 1)[0])


def run(S, outdir):
    """Flat point and distances under coherent errors; writes <outdir>/flat_minimum.json."""
    assert S["n"] == N and S["J"] == J
    t0 = time.time()
    os.makedirs(outdir, exist_ok=True)
    hc = brentq(tilt_curvature, *S["hc_bracket"], xtol=1e-13)
    xs, mu, a, Hm = landau_coefficients(hc)
    nz0 = no_noise()
    rng, best = np.random.default_rng([S["seed"], 999]), np.inf
    for _ in range(S["random_starts"]):              # is the flat point the global minimum?
        r = minimize(lambda x: float(F(x, hc, nz0)), rng.uniform(-np.pi, np.pi, 3),
                     jac=lambda x: np.asarray(G(x, hc, nz0)), method="BFGS",
                     options=dict(gtol=1e-11, maxiter=3000))
        best = min(best, r.fun)
    print(f"[flat] h_c = {hc:.10f}, x* = {np.round(xs, 6).tolist()}, a = {a:.5f}, "
          f"(gamma, beta) block eigenvalues {np.round(np.linalg.eigvalsh(Hm[1:, 1:]), 3).tolist()}",
          flush=True)
    per_draw = []
    for s in range(S["draws"]):
        d, prev, row = error_draw(S["seed"], s), None, []
        for e in S["eps"]:
            prev = minimiser(xs, hc, coherent(d, e), e, prev)
            row.append(float(np.linalg.norm(prev - xs)))
        per_draw.append(row)
        print(f"[flat] draw {s}: |x~* - x*| / eps^(1/3) = "
              + " ".join(f"{v / e ** (1 / 3):.3g}" for v, e in zip(row, S["eps"])), flush=True)
    gm = np.exp(np.mean(np.log(per_draw), axis=0))
    out = dict(settings=dict(S), h_c=hc, x_star=xs.tolist(), mu=mu, a=a,
               mixed_block=float(np.abs(Hm[0, 1:]).max()),
               even_eigenvalues=np.linalg.eigvalsh(Hm[1:, 1:]).tolist(),
               value=float(F(xs, hc, nz0)), best_random_start=float(best),
               eps=list(S["eps"]), per_draw_dist=per_draw, dist=gm.tolist(),
               slope=slope(S["eps"], gm, S["fit_points"]),
               slopes_per_draw=[slope(S["eps"], r, S["fit_points"]) for r in per_draw],
               secs=time.time() - t0)
    with open(os.path.join(outdir, "flat_minimum.json"), "w") as fh:
        json.dump(out, fh)
    print(f"[flat] done in {out['secs']:.0f}s; slope of the geometric mean {out['slope']:.3f}",
          flush=True)
    return out
