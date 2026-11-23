"""
robustvqe/circuits.py
=====================

Ansaetze and their noisy cost functions  cost(x, nz) -> float.

  "su"    two SpecialUnitary blocks, the second reparameterised through
          2 R arctan(phi)/pi (bounded generator), the locally surjective ansatz
          of Wiedmann et al. [19]. L = 2 layers, p = 2(4^n - 1) parameters.
  "qaoa"  |+>^n followed by `depth` cost/mixer layers (transverse-field
          mixer), p = 2 * depth. Every layer is a single rotation
          exp(-i theta_j H_j), the form (3) assumed in Sec. VI.A.

Noise is inserted after every layer (noise.Layer); the coherent control error
rescales the angles (noise.apply_control). The observable is measured as a
dense qml.Hermitian.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pennylane as qml

from . import noise as nzm


def su_cost(n, O, kind):
    wires = list(range(n))
    dim = 4 ** n - 1
    R = np.sqrt(2 * n) * np.pi
    layer = nzm.Layer(kind, wires)
    obs = qml.Hermitian(np.asarray(O), wires=wires)
    dev = qml.device(nzm.device_name(kind), wires=n)

    def circuit(x, nz):
        x = nzm.apply_control(kind, x, nz)
        qml.SpecialUnitary(x[:dim], wires=wires)
        layer(0, nz)
        qml.SpecialUnitary(2 * R * jnp.arctan(x[dim:]) / jnp.pi, wires=wires)
        layer(1, nz)
        return qml.expval(obs)

    qnode = qml.QNode(circuit, dev, interface="jax", diff_method="backprop")
    return qnode, dict(n_params=2 * dim, n_layers=2, n_qubits=n)


def qaoa_cost(n, depth, O, cost_hamiltonian, kind):
    from pennylane import qaoa

    wires = list(range(n))
    mixer = qml.Hamiltonian([1.0] * n, [qml.PauliX(w) for w in wires])
    layer = nzm.Layer(kind, wires)
    obs = qml.Hermitian(np.asarray(O), wires=wires)
    dev = qml.device(nzm.device_name(kind), wires=n)

    def circuit(x, nz):
        x = nzm.apply_control(kind, x, nz)
        gamma, beta = x[:depth], x[depth:]
        for w in wires:
            qml.Hadamard(wires=w)
        for j in range(depth):
            qaoa.cost_layer(gamma[j], cost_hamiltonian)
            qaoa.mixer_layer(beta[j], mixer)
            layer(j, nz)
        return qml.expval(obs)

    qnode = qml.QNode(circuit, dev, interface="jax", diff_method="backprop")
    return qnode, dict(n_params=2 * depth, n_layers=depth, n_qubits=n)


def build(spec, problem, kind):
    """spec: problem dict of config.PROBLEMS; problem: output of hamiltonians.build."""
    if spec["ansatz"] == "su":
        return su_cost(spec["n"], problem["matrix"], kind)
    if spec["ansatz"] == "qaoa":
        return qaoa_cost(spec["n"], spec["depth"], problem["matrix"],
                         problem["cost_hamiltonian"], kind)
    raise ValueError(f"unknown ansatz {spec['ansatz']!r}")


class Compiled:
    """Jitted cost, gradient and Hessian of one noisy problem."""

    def __init__(self, qnode, info, kind):
        self.kind, self.info = kind, info
        self.f = jax.jit(qnode)
        self.g = jax.jit(jax.grad(qnode, argnums=0))
        self.H = jax.jit(jax.hessian(qnode, argnums=0))

    def warmup(self):
        x = jnp.zeros(self.info["n_params"])
        nz = nzm.zero_params(self.kind, self.info["n_layers"], self.info["n_qubits"],
                             self.info["n_params"])
        self.f(x, nz).block_until_ready()
        self.g(x, nz).block_until_ready()
        self.H(x, nz).block_until_ready()
