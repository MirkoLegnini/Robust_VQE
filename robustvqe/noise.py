"""
robustvqe/noise.py
==================

Noise models of Sec. VII, written so that the noise realisation is an explicit
INPUT of the cost function (no random numbers are drawn while a circuit is
traced). Cost, gradient and Hessian then refer to the same noisy problem, and
one compiled function serves every initialisation and every epsilon.

Perturbation level epsilon, per model:

  coherent        after every layer, on every qubit, a rotation R_sigma(a) with
                  a random axis sigma in {X, Y, Z} and a = eps * u, u in [-1, 1]
                  drawn once per initialisation and normalised so that
                  max|u| = 1: eps is the largest single-qubit over-rotation
                  angle (= ||eta||_inf with H_e = sigma/2, Theorem 4).
  bitflip         after every layer, a bit flip on every qubit with probability
                  p = 1 - (1 + eps)^(-1/(n L)): the probability of no error at
                  all is 1/(1 + eps), so by Lemmas 3-4 the perturbation level is
                  exactly eps.
  depolarizing    global depolarising channel (1 - p) rho + p I/N after every
                  layer, p = 1 - (1 + eps)^(-1/L), simulated with its 4^n Pauli
                  Kraus operators (a channel simulation, not the closed form of
                  Lemma 5). Perturbation level eps.
  control         coherent control error (Sec. VI.A): every rotation angle
                  theta_j -> (1 + eta_j) theta_j, eta = eps * u, u in [-1, 1]
                  drawn once per initialisation with max|u| = 1: ||eta||_inf = eps.
  control_global  the same with eta_j = eps for ALL j (global miscalibration of
                  the pulse amplitude), for which (29) holds with equality.
"""

import itertools

import numpy as np
import jax.numpy as jnp
import pennylane as qml

COHERENT, BITFLIP, DEPOL, NONE = "coherent", "bitflip", "depolarizing", "none"
CONTROL, CONTROL_GLOBAL = "control", "control_global"
CONTROLS = (CONTROL, CONTROL_GLOBAL)

MIXED = {BITFLIP, DEPOL}                 # models that need a density-matrix simulator


def device_name(kind):
    return "default.mixed" if kind in MIXED else "default.qubit"


# ---------------------------------------------------------------------------
# Per-initialisation draws (direction of the error; epsilon only rescales it)
# ---------------------------------------------------------------------------

def draw(kind, rng, n_layers, n_qubits, n_params):
    if kind == COHERENT:
        axes = rng.integers(0, 3, size=(n_layers, n_qubits))
        u = rng.uniform(-1.0, 1.0, size=(n_layers, n_qubits))
        return dict(axes=axes, u=u / np.abs(u).max())
    if kind == CONTROL:
        u = rng.uniform(-1.0, 1.0, size=n_params)
        return dict(u=u / np.abs(u).max())
    if kind == CONTROL_GLOBAL:
        return dict(u=np.ones(n_params))
    return {}


def params(kind, d, eps, n_layers, n_qubits):
    """Traced noise parameters for the circuit, at perturbation level eps."""
    if kind == COHERENT:
        a = eps * d["u"]
        ax, ay, az = (np.where(d["axes"] == k, a, 0.0) for k in range(3))
        return dict(ax=jnp.asarray(ax), ay=jnp.asarray(ay), az=jnp.asarray(az))
    if kind == BITFLIP:
        return dict(p=jnp.asarray(1.0 - (1.0 + eps) ** (-1.0 / (n_qubits * n_layers))))
    if kind == DEPOL:
        return dict(p=jnp.asarray(1.0 - (1.0 + eps) ** (-1.0 / n_layers)))
    if kind in CONTROLS:
        return dict(eta=jnp.asarray(eps * d["u"]))
    return {}


def zero_params(kind, n_layers, n_qubits, n_params):
    """Parameters that switch the model off (used for compilation)."""
    if kind == COHERENT:
        z = jnp.zeros((n_layers, n_qubits))
        return dict(ax=z, ay=z, az=z)
    if kind in (BITFLIP, DEPOL):
        return dict(p=jnp.asarray(0.0))
    if kind in CONTROLS:
        return dict(eta=jnp.zeros(n_params))
    return {}


# ---------------------------------------------------------------------------
# In-circuit application
# ---------------------------------------------------------------------------

PAULIS = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]),
          np.diag([1.0, -1.0])]                      # I, X, Y, Z


def _pauli_strings(n):
    out = []
    for idx in itertools.product(range(4), repeat=n):
        M = np.array([[1.0 + 0j]])
        for k in idx:
            M = np.kron(M, PAULIS[k])
        out.append(M)
    return out  # out[0] is the identity


class Layer:
    """Callable that inserts the noise of one layer inside a QNode."""

    def __init__(self, kind, wires):
        self.kind, self.wires = kind, list(wires)
        if kind == DEPOL:
            self.paulis = _pauli_strings(len(self.wires))

    def __call__(self, j, nz):
        if self.kind == COHERENT:
            for k, w in enumerate(self.wires):
                qml.RX(nz["ax"][j, k], wires=w)
                qml.RY(nz["ay"][j, k], wires=w)
                qml.RZ(nz["az"][j, k], wires=w)
        elif self.kind == BITFLIP:
            for w in self.wires:
                qml.BitFlip(nz["p"], wires=w)
        elif self.kind == DEPOL:
            p, N2 = nz["p"], len(self.paulis)
            K = [jnp.sqrt(1.0 - p + p / N2) * self.paulis[0]]
            K += [jnp.sqrt(p / N2) * M for M in self.paulis[1:]]
            qml.QubitChannel(K, wires=self.wires)


def apply_control(kind, x, nz):
    """Control errors rescale the trainable angles; identity otherwise."""
    return x * (1.0 + nz["eta"]) if kind in CONTROLS else x


# ---------------------------------------------------------------------------
# Exactly known minimum values (used to certify GLOBAL minimality)
# ---------------------------------------------------------------------------

def pauli_adjoint(O, n, probs):
    """Heisenberg-picture action C(O) of a product of single-qubit Pauli
    channels (self-adjoint): qubit w is flipped by X, Y, Z with probabilities
    probs[w] = (px, py, pz). A bit flip with probability p is (p, 0, 0)."""
    I2 = PAULIS[0]
    B = np.array(O, dtype=complex)
    for w in range(n):
        q = np.asarray(probs[w], dtype=float)
        C = (1.0 - q.sum()) * B
        for P, pk in zip(PAULIS[1:], q):
            if pk:
                Pw = np.array([[1.0]])
                for k in range(n):
                    Pw = np.kron(Pw, P if k == w else I2)
                C = C + pk * Pw @ B @ Pw
        B = C
    return B
