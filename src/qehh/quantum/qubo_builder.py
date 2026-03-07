from __future__ import annotations
import numpy as np

from .subproblem import BalanceSubproblem


def build_balance_qubo(sub: BalanceSubproblem) -> np.ndarray:
    """
    Build a QUBO matrix Q (k x k) for minimizing squared imbalance after moving chosen jobs.

    Decision x_i in {0,1}:
      x_i = 1 means move job i from heavy to light.

    Let Δ = load_H - load_L.
    After moves, imbalance proxy becomes:
      Δ' = Δ - 2 * sum(p_i x_i)
    Minimize:
      (Δ')^2
    """
    p = sub.p
    k = p.shape[0]

    delta = sub.load_H - sub.load_L

    # Energy(x) = (delta - 2 p^T x)^2
    #          = delta^2 - 4 delta p^T x + 4 x^T (p p^T) x
    # Drop constant delta^2 for optimization; encode remaining as QUBO.
    Q = np.zeros((k, k), dtype=float)

    # Quadratic part: 4 p_i p_j x_i x_j
    # We store full matrix; objective uses x^T Q x.
    Q += 4.0 * np.outer(p, p)

    # Linear part: -4 delta p_i x_i can be encoded on diagonal as addition
    # because x_i^2 = x_i for binary vars
    for i in range(k):
        Q[i, i] += -4.0 * delta * p[i]

    return Q


def qubo_energy(Q: np.ndarray, x: np.ndarray) -> float:
    """
    Compute x^T Q x for binary vector x.
    """
    x = x.astype(float)
    return float(x @ Q @ x)