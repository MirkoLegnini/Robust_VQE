"""
robustvqe/optimizer.py
======================

Locating, following and certifying minima. `c` is a circuits.Compiled object
(cost c.f, gradient c.g, Hessian c.H, all functions of (theta, noise_params)).

  gd          fixed-step gradient descent, Eq. (6) of the paper (nominal
              training from a random initial point)
  sfnewton    saddle-free Newton (|Hessian| eigenvalues, Armijo descent on the
              cost, trust region): leaves saddles and travels to a minimum
  lm          Levenberg-Marquardt on grad l = 0 with mu = |grad|^2
              (Yamashita-Fukushima). Converges quadratically to minima that
              form manifolds WITHOUT drifting along them.
  lbfgs       L-BFGS descent; used once per seed for the bit-flip "trip"
              along a degenerate optimal manifold (see continuation.py)
  certify     gradient norm, smallest Hessian eigenvalue, number of flat
              directions
  track       epsilon-continuation of a minimiser (tangent predictor from the
              implicit function theorem, LM corrector, adaptive step):
              follows the branch of minimisers through theta* and reports when
              it cannot be followed (e.g. a fold, where the minimum ceases to
              exist)
  project     distance of a point to the noise-free optimal SET: LM on the
              noise-free cost lands on the set, and the landing point is then
              moved along the set to the foot of the perpendicular (nearest
              point), so that the distance is the true (local) set distance.

Tolerances (defaults): a point is certified if |grad| <= 1e-9 and
lambda_min(Hessian) >= -1e-9 * |lambda|_max.
"""

import numpy as np

TOL_G = 1e-9      # gradient-norm tolerance of a certified critical point
NEG_REL = 1e-9    # admissible relative negative curvature (round-off)
TAU = 1e-8        # relative eigenvalue cut-off: |lambda| < TAU*|lambda|_max is "flat"


def _eigh(c, x, nz):
    H = np.asarray(c.H(x, nz))
    return np.linalg.eigh(0.5 * (H + H.T))


def gd(c, x, nz, step, iters):
    """Fixed-step gradient descent; halves the step only if the cost rises."""
    x = np.asarray(x, dtype=float)
    f_ref = float(c.f(x, nz))
    for k in range(iters):
        x = x - step * np.asarray(c.g(x, nz))
        if (k + 1) % 200 == 0:
            f_now = float(c.f(x, nz))
            if not np.isfinite(f_now) or f_now > f_ref + 1e-12:
                step *= 0.5
            f_ref = f_now
    return x


def sfnewton(c, x, nz, max_iter=400, trust=0.5, tol_g=1e-8, kick_rel=1e-3):
    """Saddle-free Newton descent on the cost: steps -|H|^{-1} g (flat
    directions dropped), trust region, Armijo backtracking; a step along the
    most negative curvature direction leaves strict saddles."""
    x = np.asarray(x, dtype=float)
    fx = float(c.f(x, nz))
    it = 0
    for it in range(max_iter):
        gx = np.asarray(c.g(x, nz))
        lam, Q = _eigh(c, x, nz)
        lmax = max(np.abs(lam).max(), 1e-300)
        strong_neg = lam[0] < -kick_rel * lmax
        if np.linalg.norm(gx) < tol_g and not strong_neg:
            break
        coef = Q.T @ gx
        keep = np.abs(lam) > 1e-9 * lmax
        d = -Q[:, keep] @ (coef[keep] / np.abs(lam[keep]))
        if strong_neg:                       # leave a genuine saddle
            s = -np.sign(coef[0]) if coef[0] != 0 else 1.0
            d = d + s * min(trust, 0.3) * Q[:, 0]
        nd = np.linalg.norm(d)
        if nd > trust:
            d *= trust / nd
        slope, t = float(gx @ d), 1.0
        while True:
            xn = x + t * d
            fn = float(c.f(xn, nz))
            if fn <= fx + 1e-4 * t * min(slope, 0.0) + 1e-14 * abs(fx):
                break
            t *= 0.5
            if t < 1e-8:
                break
        if t < 1e-8:
            break
        x, fx = xn, fn
    return x, it


def lm(c, x, nz, tol_g=TOL_G * 1e-2, max_iter=100, min_steps=0, tau=TAU):
    """Levenberg-Marquardt on grad l = 0, Armijo on phi = |grad|^2 / 2.
    mu = |g|^2, floored at (tau * |lambda|_max)^2 so that eigenvalues at the
    numerical-noise level (the exactly flat directions) are never inverted."""
    x = np.asarray(x, dtype=float)
    gx = np.asarray(c.g(x, nz))
    phi = 0.5 * float(gx @ gx)
    it = 0
    for it in range(max_iter):
        if np.sqrt(2 * phi) <= tol_g and it >= min_steps:
            break
        lam, Q = _eigh(c, x, nz)
        coef = Q.T @ gx
        mu = max(min(2 * phi, 1.0), (tau * np.abs(lam).max()) ** 2)
        d = -Q @ (lam * coef / (lam ** 2 + mu))
        t = 1.0
        while True:
            xn = x + t * d
            gn = np.asarray(c.g(xn, nz))
            phin = 0.5 * float(gn @ gn)
            if phin <= (1 - 1e-4 * t) * phi or t < 1e-8:
                break
            t *= 0.5
        if phin > phi:                    # no progress possible
            break
        x, gx, phi = xn, gn, phin
    return x, it


def certify(c, x, nz, tol_g=TOL_G, neg_rel=NEG_REL, eig=False):
    """Certificate of a local minimiser. With eig=True also returns the
    Hessian eigendecomposition (lam, Q), reused by `track`."""
    gx = np.asarray(c.g(x, nz))
    lam, Q = _eigh(c, x, nz)
    lmax = float(np.abs(lam).max())
    gn = float(np.linalg.norm(gx))
    cert = dict(f=float(c.f(x, nz)), g=gn, lmin=float(lam[0]), lmax=lmax,
                n_flat=int(np.sum(np.abs(lam) < 1e-6 * lmax)),
                ok=bool(gn <= tol_g and lam[0] >= -neg_rel * lmax))
    return (cert, lam, Q) if eig else cert


def lbfgs(c, x, nz, maxiter=20000):
    """Gradient-only quasi-Newton descent (scipy L-BFGS-B). Used to travel
    along slowly curved directions, e.g. the bit-flip 'trip' along Theta*."""
    from scipy.optimize import minimize

    def fun(y):
        return float(c.f(y, nz)), np.asarray(c.g(y, nz), dtype=float)

    r = minimize(fun, np.asarray(x, dtype=float), jac=True, method="L-BFGS-B",
                 options=dict(maxiter=maxiter, gtol=1e-10, ftol=1e-15, maxcor=30))
    return np.asarray(r.x), int(r.nit)


def refine(c, x, nz, pre_lbfgs=False):
    """Nearest minimiser from x: (optional L-BFGS descent, then) LM, which
    does not drift along flat directions; if that does not certify,
    saddle-free Newton descent, then LM."""
    n_pre = 0
    if pre_lbfgs:
        x, n_pre = lbfgs(c, x, nz)
    xr, n_lm = lm(c, x, nz)
    cert = certify(c, xr, nz)
    used_sf = 0
    if not cert["ok"]:
        xs, used_sf = sfnewton(c, x, nz)
        xr, n2 = lm(c, xs, nz)
        n_lm += n2
        cert = certify(c, xr, nz)
    return xr, cert, dict(lm_iters=n_lm, sf_iters=used_sf, lbfgs_iters=n_pre)


def tangent(c, x, nz_at, e, lam, Q, de=1e-6, tau=TAU):
    """Implicit-function-theorem tangent of the branch of minimisers,
    dx/deps = -H^+ d(grad)/deps, at a minimiser x of the cost at eps = e.
    d(grad)/deps by a forward difference in eps; H^+ drops flat directions,
    so the predicted motion is orthogonal to a degenerate optimal set."""
    dg = (np.asarray(c.g(x, nz_at(e + de))) - np.asarray(c.g(x, nz_at(e)))) / de
    keep = lam > tau * np.abs(lam).max()
    coef = Q.T @ dg
    return -Q[:, keep] @ (coef[keep] / lam[keep])


def track(c, nz_at, x, e0, e1, eig=None, max_halvings=8, max_fails=16, lm_iter=8,
          jump_rel=0.5):
    """
    Natural-parameter continuation of a certified minimiser x of the cost at
    eps = e0 to eps = e1 (either direction): tangent predictor, LM corrector.
    A step is accepted if the corrector returns a certified minimiser close to
    the prediction (|x_corr - x_pred| <= jump_rel * |x_pred - x|, i.e. no
    switch to another branch); otherwise the step is halved. If the step
    falls below |e1 - e0| / 2^max_halvings, or after max_fails failed steps,
    the branch is declared lost (info['ok'] = False): typically a fold, where
    the minimum merges with a saddle (lambda_min -> 0) and ceases to exist.

    Returns (x, cert, info); info['eig'] holds the Hessian eigendecomposition
    at the returned point (to be passed back as `eig` for the next call).
    """
    x = np.asarray(x, dtype=float)
    if eig is None:
        _, lam, Q = certify(c, x, nz_at(e0), eig=True)
    else:
        lam, Q = eig
    e, h = e0, e1 - e0
    h_min = abs(e1 - e0) / 2 ** max_halvings
    steps = halvings = lm_tot = 0
    cert = None
    while True:
        last = abs(e1 - e) <= abs(h) * (1 + 1e-12)
        e_try = e1 if last else e + h
        t = tangent(c, x, nz_at, e, lam, Q)
        x_pred = x + (e_try - e) * t
        nz = nz_at(e_try)
        xr, it = lm(c, x_pred, nz, max_iter=lm_iter, min_steps=1)   # always polish once
        lm_tot += it
        cr, lam_r, Q_r = certify(c, xr, nz, eig=True)
        move = np.linalg.norm(x_pred - x)
        jump = np.linalg.norm(xr - x_pred)
        if cr["ok"] and jump <= jump_rel * move + 1e-9 * (1.0 + np.linalg.norm(x)):
            x, e, lam, Q, cert = xr, e_try, lam_r, Q_r, cr
            steps += 1
            if last:
                return x, cert, dict(ok=True, steps=steps, halvings=halvings,
                                     lm_iters=lm_tot, eig=(lam, Q))
            h = min(1.5 * abs(h), abs(e1 - e)) * np.sign(e1 - e0)   # grow again
        else:
            h *= 0.5
            halvings += 1
            if abs(h) < h_min or halvings >= max_fails:
                return xr, cr, dict(ok=False, steps=steps, halvings=halvings,
                                    lm_iters=lm_tot, eig=None, eps_reached=e,
                                    lmin_reached=float(lam[0]))


def newton_norm(c, x, nz, tau=TAU):
    """|H^+ g| at x: first-order estimate of the distance to the nearest
    critical set (used when x already lies on it to machine precision)."""
    gx = np.asarray(c.g(x, nz))
    lam, Q = _eigh(c, x, nz)
    keep = np.abs(lam) > tau * np.abs(lam).max()
    coef = Q.T @ gx
    return float(np.linalg.norm(coef[keep] / lam[keep]))


def _retract(c0, y, nz0, lam, Q, iters=10, tau=TAU):
    """Back onto the critical set from a nearby point y: chord Newton steps
    with a frozen Hessian (eigendecomposition lam, Q of a nearby point; its
    flat directions are dropped), then LM if that did not converge."""
    keep = np.abs(lam) > tau * np.abs(lam).max()
    Qk, lk = Q[:, keep], lam[keep]
    for _ in range(iters):
        gy = np.asarray(c0.g(y, nz0))
        if np.linalg.norm(gy) <= TOL_G * 1e-2:
            return y
        y = y - Qk @ ((Qk.T @ gy) / lk)
    return lm(c0, y, nz0)[0]


def project(c0, x, nz0, f_star, starts=(), tol_f=1e-9, ortho_tol=2e-2, max_ortho=25):
    """
    Distance from x to the noise-free optimal set Theta* (a manifold).

    1. Starting point on Theta*: LM on the noise-free cost from x lands on a
       point of Theta* (a certified minimiser with cost f_star; otherwise a
       descent-based retry is made). Further candidates on Theta* can be
       passed in `starts` (e.g. theta* itself and the nearest point found at
       the previous epsilon of the branch); the closest certified candidate
       is used. Since theta* is always a candidate, the result never exceeds
       |x - theta*|.
    2. LM steps are NOT exactly normal to Theta*: off the manifold, the Hessian
       directions along Theta* have eigenvalues of order |x - x_r|, and they
       are inverted. So x_r is then moved along Theta* to the foot of the
       perpendicular: tangent step t = P_T (x - x_r), with T the null space of
       the Hessian at x_r (the tangent space of Theta*), followed by a
       retraction onto Theta*, until |P_T (x - x_r)| <= ortho_tol |x - x_r|.
       Since x - x_r is then orthogonal to Theta* up to ortho_tol, the
       distance is exact up to a relative error ~ ortho_tol^2 / 2 (2e-4).
       A step is only kept if the result is a certified point of Theta*
       closer to x (steps are over-relaxed by the observed contraction rate,
       or shortened if they fail). Far from Theta* (the largest epsilon) the
       iteration can stall; it is stopped after max_ortho steps, and the
       residual is reported: the distance is then an upper bound.

    If x lies on Theta* to machine precision, the first-order estimate |H^+ g|
    is returned instead of an exact zero (cert['floor'] = True).
    Returns (distance, point, certificate); the certificate also holds the
    distance to the LM landing point ('d_lm'), which candidate was used
    ('start'), the final tangential residual ('ortho') and the number of
    tangent steps ('ortho_steps').
    """
    x = np.asarray(x, dtype=float)
    xr, it = lm(c0, x, nz0, min_steps=1)
    cert, lam, Q = certify(c0, xr, nz0, eig=True)
    if not (cert["ok"] and abs(cert["f"] - f_star) <= tol_f):
        xs, _ = sfnewton(c0, x, nz0)
        xr, _ = lm(c0, xs, nz0, min_steps=1)
        cert, lam, Q = certify(c0, xr, nz0, eig=True)
    d_lm = float(np.linalg.norm(x - xr))
    best = (xr, cert, lam, Q, "lm") if (cert["ok"] and abs(cert["f"] - f_star) <= tol_f) else None
    for name, s in starts:                                # closest certified candidate
        s = np.asarray(s, dtype=float)
        if best is None or np.linalg.norm(x - s) < np.linalg.norm(x - best[0]):
            cs, lam_s, Q_s = certify(c0, s, nz0, eig=True)
            if cs["ok"] and abs(cs["f"] - f_star) <= tol_f:
                best = (s, cs, lam_s, Q_s, name)
    if best is None:                                      # nothing on Theta*: reject
        cert.update(same=False, d_lm=d_lm, start="none", ortho=float("nan"), ortho_steps=0,
                    floor=False)
        return float(np.linalg.norm(x - xr)), xr, cert
    xr, cert, lam, Q, start = best
    ortho, steps = float("nan"), 0
    if np.linalg.norm(x - xr) > 1e-12 * (1.0 + np.linalg.norm(x)):
        omega, prev = 1.0, None
        for steps in range(max_ortho + 1):
            v = x - xr
            Tb = Q[:, np.abs(lam) < 1e-6 * np.abs(lam).max()]     # basis of T_{x_r} Theta*
            t = Tb @ (Tb.T @ v)
            ortho = float(np.linalg.norm(t) / np.linalg.norm(v))
            if ortho <= ortho_tol or steps == max_ortho:
                break
            rate = ortho / prev if prev else 1.0                    # observed contraction
            omega = min(1.0 / (1.0 - rate), 4.0) if 0.0 < rate < 0.95 else 1.0
            prev = ortho
            good = False
            for w in ((omega,) if omega > 1.0 else ()) + (1.0, 0.5, 0.25):
                xn = _retract(c0, xr + w * t, nz0, lam, Q)
                cn, lam_n, Q_n = certify(c0, xn, nz0, eig=True)
                good = cn["ok"] and abs(cn["f"] - f_star) <= tol_f and \
                    np.linalg.norm(x - xn) < np.linalg.norm(v)
                if good:
                    break
            if not good:
                break                                               # keep the last good point
            xr, cert, lam, Q = xn, cn, lam_n, Q_n
    cert.update(same=True, d_lm=d_lm, start=start, ortho=ortho, ortho_steps=steps, floor=False)
    d = float(np.linalg.norm(x - xr))
    if d == 0.0:
        d = newton_norm(c0, x, nz0)
        cert["floor"] = True
    return d, xr, cert


def minimize_from_random(c, x0, nz, gd_step, gd_iters):
    """Nominal training: GD (Eq. 6) -> saddle-free Newton -> LM polish."""
    x = gd(c, x0, nz, gd_step, gd_iters)
    x, n_sf = sfnewton(c, x, nz)
    x, n_lm = lm(c, x, nz)
    return x, dict(sf_iters=n_sf, lm_iters=n_lm)
