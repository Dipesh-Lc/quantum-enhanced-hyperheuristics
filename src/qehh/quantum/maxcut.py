from __future__ import annotations
import numpy as np


def maxcut_qubo_from_adj(W: np.ndarray) -> np.ndarray:
    """
    MaxCut on weighted undirected graph with adjacency/weight matrix W (n x n, symmetric, diag 0).
    We convert to a QUBO for minimizing -CutValue.

    Using x_i in {0,1} indicates partition membership.
    Cut weight = sum_{i<j} W_ij [x_i != x_j] = sum_{i<j} W_ij (x_i + x_j - 2 x_i x_j)
    We want maximize cut => minimize negative cut.
    Negative cut objective (drop constant): 
        E(x) = sum_{i<j} W_ij (2 x_i x_j - x_i - x_j)
    Encode as x^T Q x with symmetric Q.

    Returns Q such that minimize x^T Q x corresponds to max cut.
    """
    W = np.array(W, dtype=float)
    n = W.shape[0]
    Q = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            w = float(W[i, j])
            if abs(w) < 1e-12:
                continue
            # term: 2 w x_i x_j
            Q[i, j] += w
            Q[j, i] += w
            # term: -w x_i  and -w x_j  -> put on diagonal
            Q[i, i] += -w
            Q[j, j] += -w

    return Q


def random_weighted_graph(n: int, p: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                W[i, j] = W[j, i] = float(rng.uniform(0.5, 2.0))
    return W