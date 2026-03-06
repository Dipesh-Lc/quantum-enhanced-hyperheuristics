from __future__ import annotations
import argparse
import json
from pathlib import Path
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--p_edge", type=float, default=0.3)
    ap.add_argument("--outdir", type=str, default="results/maxcut_sweep")
    args = ap.parse_args()

    W = random_weighted_graph(args.n, args.p_edge, seed=args.seed)
    Q = maxcut_qubo_from_adj(W)

    reps_list = [1, 2]
    samples_list = [8, 16, 32, 64, 128]

    rows = []
    for reps in reps_list:
        for ns in samples_list:
            sol = solve_qubo_qaoa(Q, reps=reps, seed=args.seed, n_param_samples=ns)
            x = sol["bitstring"].astype(int)
            rows.append({
                "reps": reps,
                "n_param_samples": ns,
                "cut_value": float(cut_value(W, x)),
                "qubo_energy": float(sol["qubo_energy"]),
            })

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"sweep_seed{args.seed}.json"
    outpath.write_text(json.dumps({"config": vars(args), "rows": rows}, indent=2))
    print("Wrote:", outpath)


if __name__ == "__main__":
    main()