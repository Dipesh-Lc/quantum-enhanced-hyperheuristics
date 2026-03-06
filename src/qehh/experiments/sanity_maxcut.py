from __future__ import annotations
import argparse
import numpy as np

from qehh.quantum.maxcut import random_weighted_graph, maxcut_qubo_from_adj
from qehh.quantum.qaoa_runner import solve_qubo_qaoa


def cut_value(W: np.ndarray, x: np.ndarray) -> float:
    n = W.shape[0]
    val = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if x[i] != x[j]:
                val += float(W[i, j])
    return val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--p_edge", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--samples", type=int, default=64)
    args = ap.parse_args()

    W = random_weighted_graph(args.n, args.p_edge, seed=args.seed)
    Q = maxcut_qubo_from_adj(W)

    sol = solve_qubo_qaoa(Q, reps=args.reps, seed=args.seed, n_param_samples=args.samples)
    x = sol["bitstring"].astype(int)
    cv = cut_value(W, x)

    print("QAOA solution x:", x.tolist())
    print("Cut value:", cv)
    print("QUBO energy:", sol["qubo_energy"])
    print("Params reps:", sol["reps"], "param_samples:", sol["n_param_samples"])


if __name__ == "__main__":
    main()