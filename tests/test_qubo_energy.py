from __future__ import annotations
import itertools
import numpy as np

from qehh.quantum.qubo_builder import qubo_energy
from qehh.quantum.qaoa_runner import qubo_to_ising, ising_energy, bitstring_to_spin


def brute_qubo(Q: np.ndarray) -> tuple[np.ndarray, float]:
    n = Q.shape[0]
    best_x = None
    best_e = float("inf")
    for bits in itertools.product([0, 1], repeat=n):
        x = np.array(bits, dtype=int)
        e = qubo_energy(Q, x)
        if e < best_e:
            best_e = e
            best_x = x
    assert best_x is not None
    return best_x, float(best_e)


def test_qubo_to_ising_consistency_up_to_constant():
    rng = np.random.default_rng(0)
    n = 6
    # Random symmetric Q
    A = rng.normal(size=(n, n))
    Q = 0.5 * (A + A.T)

    model = qubo_to_ising(Q)

    # Compare energies across all bitstrings: E_qubo(x) - E_ising(z) should be constant shift
    diffs = []
    for bits in itertools.product([0, 1], repeat=n):
        x = np.array(bits, dtype=int)
        z = bitstring_to_spin(x)
        eq = float(x @ Q @ x)
        ei = ising_energy(model, z)
        diffs.append(eq - ei)

    diffs = np.array(diffs, dtype=float)
    assert np.max(diffs) - np.min(diffs) < 1e-8