"""
robustvqe/config.py
===================

Every setting of the numerical experiments of Sec. VII. The values below
reproduce the paper; `quick()` returns a reduced copy for a smoke test.

Seeds: the random initial points and error draws are fixed by the seeds below
and by the POSITION of each noise model in its `noises` list (the position
selects the random stream of the per-initialisation error draw), so do not
reorder the lists.
"""

import copy

# ---------------------------------------------------------------------------
# Sec. VII.A and VII.C: perturbed optimal parameters (continuation.py)
# ---------------------------------------------------------------------------

CONTINUATION = dict(
    seed=62213,                      # problem g, initialisation s -> seed + 1000 g + s
    eps=[1e-4 * 10 ** (3.5 * k / 9) for k in range(10)],   # 1e-4 ... 10^-1/2
    target=12,                       # certified perturbed minimisers wanted at EVERY eps
    min_seeds=12,                    # initialisations always run per curve
    max_seeds=64,                    # hard cap (QAOA needs more: branches can end at a fold)
)

PROBLEMS = [
    # (i) transverse-field Ising chain, locally surjective SU(N) ansatz [19]
    dict(name="TFIM", problem="tfim", problem_kwargs=dict(n=4, J=1.0, h=1.0),
         ansatz="su", n=4, gd_step=5e-3, gd_iters=3000,
         noises=["coherent", "bitflip", "depolarizing"]),
    # (ii) QAOA for minimum vertex cover on a fixed Erdos-Renyi graph, depth 6;
    #      coherent control errors with random relative errors ("control") and
    #      with a global miscalibration eta_j = eps for all j ("control_global")
    dict(name="QAOA", problem="mvc", problem_kwargs=dict(n=5, p_edge=0.4, graph_seed=3),
         ansatz="qaoa", n=5, depth=6, gd_step=1e-2, gd_iters=2000,
         noises=["coherent", "bitflip", "control", "control_global"]),
    # (iii) variational compiling of the QFT acting on |0...01>, SU(N) ansatz
    dict(name="VQC", problem="qft", problem_kwargs=dict(n=4, x=1),
         ansatz="su", n=4, gd_step=5e-3, gd_iters=3000,
         noises=["coherent", "bitflip"]),
]

# Fig. 4 compares the HVA with the SU(N) ansatz on the SAME Hamiltonian: the
# TFIM curve under coherent errors is rerun at the field h_c of the flat HVA
# minimum (same seeds and error draws as the TFIM curve; h_c is read from the
# output of flat_minimum.py).
SAME_FIELD_NAME = "TFIM_hc"

# ---------------------------------------------------------------------------
# Sec. VII.B: gradient descent on the perturbed cost (convergence.py)
# ---------------------------------------------------------------------------

CONVERGENCE = dict(
    seed=73213,                      # initial points and error directions
    eps=[0.0, 1e-3, 1e-2, 1e-1, 10 ** -0.5],
    runs=12,                         # initial points per noise model and eps
    K=8000,                          # gradient-descent iterations
    trace_every=10,                  # store every 10th value of l~(theta^k)
    tol_e=1e-9,                      # a limit is a global minimiser if l~ - l~* <= tol_e
    cases=[
        # step sizes below 2 / lambda_max of the Hessian at the minima, so that the
        # cost decreases monotonically (Eq. (6))
        dict(group="TFIM", gamma=4e-3, noises=["coherent", "bitflip"]),
        dict(group="QAOA", gamma=3e-3, noises=["coherent"]),
    ],
)

# ---------------------------------------------------------------------------
# Sec. VII.A, Fig. 4: flat minimum of the HVA (flat_minimum.py)
# ---------------------------------------------------------------------------

FLAT = dict(
    n=4, J=1.0,
    seed=20261112,                   # error draw s uses [seed, s]
    draws=12,                        # coherent error draws
    eps=CONTINUATION["eps"],
    hc_bracket=(0.8, 1.0),           # h_c: root of the curvature along the tilt
    random_starts=200,               # global-minimality check of the flat point
    fit_points=6,                    # slope: least squares on the 6 smallest eps
)


def quick():
    """Reduced settings for a smoke test (a few minutes on 2 cores): fewer
    perturbation levels, initialisations, runs and iterations. The results are
    NOT those of the paper."""
    q = dict(continuation=copy.deepcopy(CONTINUATION), convergence=copy.deepcopy(CONVERGENCE),
             flat=copy.deepcopy(FLAT))
    q["continuation"].update(eps=q["continuation"]["eps"][::4], target=2, min_seeds=2, max_seeds=4)
    q["convergence"].update(eps=[0.0, 1e-2], runs=2, K=400)
    q["flat"].update(draws=2, eps=q["flat"]["eps"][::4], random_starts=8)
    return q


def full():
    return dict(continuation=copy.deepcopy(CONTINUATION), convergence=copy.deepcopy(CONVERGENCE),
                flat=copy.deepcopy(FLAT))
