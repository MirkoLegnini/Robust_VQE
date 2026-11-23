# Noise resilience and robust convergence of the VQE: numerical experiments

Code for Sec. VII of *Noise Resilience and Robust Convergence Guarantees for the
Variational Quantum Eigensolver* (M. Legnini, J. Berberich). It reproduces every
figure, table and number of that section, and nothing else.

| Paper | Output file | Experiment |
|---|---|---|
| Fig. 2 | `figures/coherent_distance.pdf` | continuation: TFIM, QAOA, VQC under coherent errors |
| Fig. 3 | `figures/bitflip_distance.pdf` | continuation: TFIM, QAOA, VQC under bit flips |
| Fig. 4 | `figures/flat_vs_linear.pdf` | flat minimum of the HVA, and the TFIM continuation at h = h_c |
| Fig. 5 | `figures/convergence_traj_TFIM_bitflip.pdf` | convergence: TFIM without noise and under bit flips |
| Fig. 6 | `figures/control_distance.pdf` | continuation: QAOA under coherent control errors |
| Fig. 7 | `figures/depolarizing_distance.pdf` | continuation: TFIM under global depolarization |
| Table I | `tables/table_runs.tex` | continuation (Figs. 2-4) |
| Table II | `tables/table_convergence.tex` | convergence |
| numbers in the text of Sec. VII | `tables/paper_numbers.txt` | all |

Fig. 1 is a schematic and is not generated here.

## Installation

Tested with Python 3.13 on Linux x86_64 (CPU only). The lock file pins every
package, including dependencies, by version and hash:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install --require-hashes -r requirements.lock
```

`requirements.in` lists the direct dependencies (JAX 0.6.2, PennyLane 0.42.3,
NumPy, SciPy, Matplotlib, NetworkX) from which the lock file was compiled with
`pip-compile --generate-hashes`.

## Usage

```bash
python reproduce.py figures     # Figs. 2-7, Tables I-II and the numbers, from reference/ (seconds)
python reproduce.py quick       # smoke test of the whole pipeline with reduced settings (~15 min)
python reproduce.py all         # all experiments from scratch, then figures and tables (~75 min)
python reproduce.py compare runs reference    # check a fresh run against the reference results
```

* `figures` reads the reference results shipped in `reference/` and writes
  `output/figures/` and `output/tables/`. Use `--results runs` to plot your own run.
* `quick` runs every experiment with fewer initializations, perturbation levels
  and iterations (`config.quick()`), then builds the figures and tables in
  `quick/`. It only checks that the installation works: its numbers are not those
  of the paper.
* `all` writes the results to `runs/` and the figures and tables to
  `runs/figures/` and `runs/tables/`. `compare` then checks the new results
  against `reference/` (see Reproducibility below).
* Parts and single jobs can be run separately; the flat minimum comes first
  because it fixes the field h_c of the TFIM curve in Fig. 4:

  ```bash
  python reproduce.py run flat
  python reproduce.py run continuation TFIM:coherent QAOA     # one curve, or all curves of a problem
  python reproduce.py run convergence convergence:TFIM:bitflip
  ```

All commands that run experiments accept `--workers N` (parallel processes, default 2)
and `--out DIR`.

### Run time

Measured on 2 cores (`--workers 2`), single-threaded processes:

| Part | Jobs (minutes on one core each) | Wall time |
|---|---|---|
| `run flat` | HVA flat minimum: 1 | 1 min |
| `run continuation` | TFIM: 18 (coherent), 12 (bit flips), 14 (depolarization), 17 (coherent, h = h_c); VQC: 11 (coherent), 10 (bit flips); QAOA: 2 (coherent), 5 (bit flips), 1 + 1 (control errors) | 46 min |
| `run convergence` | TFIM: 2 (no noise), 22 (coherent), 19 (bit flips); QAOA: 1 (no noise), 4 (coherent) | 24 min |
| `all` | | about 75 min |

Each process needs up to about 1.6 GB of memory.

## What is computed

All settings, seeds included, are in `robustvqe/config.py`.

**Continuation** (`robustvqe/continuation.py`; Figs. 2, 3, 4, 6, 7, Table I).
One curve is one problem under one noise model. For each initialization
θ⁰ ~ U[-π, π)^p, a noise-free minimizer θ* is computed by gradient descent (6),
refined (saddle-free Newton, Levenberg-Marquardt) and certified by first- and
second-order conditions. The minimizer θ̃*_ε of the perturbed cost is then
followed from θ* by numerical continuation in ε (tangent predictor from the
Hessian, Levenberg-Marquardt corrector), and certified in the same way. Its
distance to Θ* is measured by projecting onto Θ* for the SU(N) ansatz (whose
optima form a manifold), and as |θ̃*_ε − θ*| for QAOA (isolated minima) and for
control errors. A branch that can no longer be followed is ended, and
initializations are added until every ε has at least 12 certified minimizers.
The figures show the geometric mean of the distance and its 95% confidence
interval.

* Problems: (i) TFIM VQE, n = 4, J = h = 1, SU(N) ansatz with two blocks
  (p = 510); (ii) QAOA for minimum vertex cover on a fixed Erdős-Rényi graph
  (5 vertices, 6 edges), depth 6 (p = 12); (iii) VQC of the QFT acting on |1⟩,
  SU(N) ansatz (p = 510).
* Noise, after every layer (`robustvqe/noise.py`): coherent rotations about
  random Pauli axes with largest angle ε; bit flips with probability
  p = 1 − (1+ε)^(−1/(nL)) per qubit; global depolarization with
  p = 1 − (1+ε)^(−1/L); coherent control errors θ_j → (1+η_j)θ_j with ‖η‖∞ = ε
  (random η, and η_j = ε for all j).

**Flat minimum** (`robustvqe/flat_minimum.py`; Fig. 4, Table I). One HVA layer
on the product state R_Y(θ₁)^⊗n|0⟩ for the TFIM. The field h_c where the
curvature along θ₁ vanishes is found by bisection; the quartic coefficient, the
positive definite (γ, β) block of the Hessian and global minimality (200 random
starts) are checked. The perturbed minimizer is followed under the coherent
errors above for 12 error draws. The SU(N) curve of Fig. 4 is the TFIM
continuation rerun at h = h_c, with the same seeds and error draws.

**Convergence** (`robustvqe/convergence.py`; Fig. 5, Table II). Gradient
descent (6) with a fixed step on the perturbed cost from 12 initial points per
ε, for K = 8000 iterations, then continued with L-BFGS and certified. For the
TFIM the limit is compared with the exactly known minimum value of the perturbed
cost; for QAOA the number of distinct local minima is reported.

## Reproducibility

The runs are deterministic: all random numbers come from fixed seeds, and every
process is single-threaded (set in `reproduce.py`). The results in `reference/`
were produced with `python reproduce.py run ...` (flat minimum, continuation,
convergence) in the environment of `requirements.lock` on Linux x86_64, and
agree with the results behind the paper's figures: the same accepted points and
minimum values; distances bitwise identical for QAOA, the depolarized and the
same-field TFIM curves and the flat minimum; plotted geometric means equal to
1e-6 under coherent errors and within 1.4% under bit flips; gradient-descent
trajectories bitwise identical, with limits that agree to 1e-13.

On other hardware or with other BLAS builds the results can differ at the level
of round-off, with one visible exception: under bit flips, the perturbed
minimizers of the SU(N) ansatz lie in a nearly flat valley, and the point where
the descent along it ends can move by a few percent. `python reproduce.py
compare runs reference` therefore checks what the figures and tables show:
the same accepted points, minimum values to 1e-9, and geometric means of the
distances to 5% (the 95% confidence bands are 20-50% wide). It prints the
largest single-point differences for information.

## Layout

```
reproduce.py              command-line entry point
requirements.lock         pinned environment (with hashes)
requirements.in           direct dependencies
robustvqe/
    config.py             all settings and seeds
    hamiltonians.py       TFIM, minimum vertex cover, QFT compilation
    circuits.py           SU(N) and QAOA ansaetze with noise (PennyLane, JAX)
    noise.py              noise models
    optimizer.py          minimization, certification, continuation, projection onto Theta*
    continuation.py       perturbed optimal parameters (Sec. VII.A, VII.C)
    flat_minimum.py       HVA example (Fig. 4)
    convergence.py        gradient descent on the perturbed cost (Sec. VII.B)
    runner.py             job scheduling
    stats.py              geometric means and confidence intervals
    figures.py            Figs. 2-7
    tables.py             Tables I-II and the numbers of Sec. VII
    compare.py            comparison of two result folders
reference/                results of the paper (JSON; NPZ for the gradient-descent traces)
```
