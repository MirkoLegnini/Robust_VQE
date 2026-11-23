"""
robustvqe/hamiltonians.py
=========================

Observables of the three problems used in the paper's numerics. Every builder
returns a dict with the dense observable matrix (measured as qml.Hermitian, so
no sum-of-terms code path is involved on default.mixed) plus spectral data.

    tfim(n, J, h)        transverse-field Ising chain, VQE target
    mvc(n, p_edge, seed) minimum vertex cover cost Hamiltonian, QAOA target
    qft_state(n, x)      VQC cost 1 - |<x|U_QFT^dag U(theta)|0>|^2
"""

import numpy as np
import pennylane as qml


def _spectrum(M):
    ev = np.sort(np.linalg.eigvalsh(M))
    deg = int(np.sum(np.abs(ev - ev[0]) < 1e-9))
    gap = float(ev[deg] - ev[0]) if deg < len(ev) else 0.0
    return dict(e0=float(ev[0]), degeneracy=deg, gap=gap)


def tfim(n, J=1.0, h=1.0):
    """
    Open transverse-field Ising chain  H = -J sum_i Z_i Z_{i+1} - h sum_i X_i.
    At finite n the ground state is unique (gap > 0 for h > 0).

    Reference: S. Sachdev, Quantum Phase Transitions, 2nd ed. (2011), Ch. 4.
    """
    ops = [qml.PauliZ(i) @ qml.PauliZ(i + 1) for i in range(n - 1)]
    ops += [qml.PauliX(i) for i in range(n)]
    coeffs = [-J] * (n - 1) + [-h] * n
    M = np.asarray(qml.matrix(qml.Hamiltonian(coeffs, ops), wire_order=range(n)))
    M = np.real_if_close(0.5 * (M + M.conj().T))
    return dict(name=f"TFIM(n={n}, h/J={h / J:g})", matrix=M, **_spectrum(M))


def mvc(n, p_edge=0.4, graph_seed=3):
    """
    Minimum vertex cover cost Hamiltonian (unconstrained form) on an
    Erdos-Renyi graph G(n, p_edge) drawn with a FIXED seed, so that every run
    uses the same instance. The seed is advanced until the graph is connected.

    Returns the matrix (for measurement) and the PennyLane Hamiltonian (for
    the QAOA cost layer).
    """
    import networkx as nx
    from pennylane import qaoa

    seed = graph_seed
    while True:
        G = nx.erdos_renyi_graph(n, p_edge, seed=seed)
        if nx.is_connected(G):
            break
        seed += 1
    H_cost, _ = qaoa.min_vertex_cover(G, constrained=False)
    M = np.real(np.asarray(qml.matrix(H_cost, wire_order=range(n))))
    return dict(name=f"MVC on G({n},{p_edge}), seed {seed}", matrix=M,
                cost_hamiltonian=H_cost, graph_seed=seed,
                edges=sorted(G.edges()), **_spectrum(M))


def qft_state(n, x=1):
    """
    VQC cost for the quantum Fourier transform acting on the basis state |x>:
        l(theta) = 1 - |<x| U_QFT^dag U(theta) |0>|^2,
    i.e. a VQE with observable I - U_QFT |x><x| U_QFT^dag (minimum 0), whose
    ground state U_QFT|x> = 2^(-n/2) sum_k exp(2 pi i x k / 2^n) |k> is a
    product state.
    x = 0 would give |+>^n, a stabilizer state that is invariant under bit
    flips: the bit-flip-perturbed minimisers would then lie EXACTLY on Theta*
    (zero displacement; optimal-parameter resilience, Sharma et al. 2020).
    For odd x and n >= 3, U_QFT|x> is not a stabilizer state.
    """
    N = 2 ** n
    W = np.asarray(qml.matrix(qml.QFT(wires=range(n))))
    t = W[:, x]                                   # U_QFT |x>
    M = np.eye(N) - np.outer(t, t.conj())
    M = 0.5 * (M + M.conj().T)
    return dict(name=f"QFT state compilation (n={n}, input |{x}>)", matrix=M, **_spectrum(M))


BUILDERS = {"tfim": tfim, "mvc": mvc, "qft": qft_state}


def build(problem, **kwargs):
    if problem not in BUILDERS:
        raise ValueError(f"unknown problem {problem!r}; have {list(BUILDERS)}")
    return BUILDERS[problem](**kwargs)
