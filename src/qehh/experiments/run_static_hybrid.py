from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np

from qehh.core.instances import generate_jobs, random_assignment
from qehh.core.rng import make_rng
from qehh.core.metrics import compute_metrics
from qehh.operators.classical import op_two_smallest_from_max
from qehh.quantum.qaoa_operator import op_qaoa_balance_move


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_jobs", type=int, default=200)
    ap.add_argument("--m", type=int, default=10)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--qaoa_steps", type=int, default=10)
    ap.add_argument("--dist", type=str, default="lognormal")
    ap.add_argument("--outdir", type=str, default="results/static_hybrid")
    args = ap.parse_args()

    rng = make_rng(args.seed)
    jobs = generate_jobs(args.n_jobs, dist=args.dist, seed=args.seed)
    s = random_assignment(jobs, args.m, rng)

    cmax_hist = [float(s.makespan)]
    t_total = 0.0

    for t in range(args.steps):
        t0 = time.perf_counter()
        if t < args.qaoa_steps:
            s2 = op_qaoa_balance_move(s, rng, k=10, reps=1, n_param_samples=16, seed_offset=1337)
        else:
            s2 = op_two_smallest_from_max(s, rng)
        t_total += time.perf_counter() - t0
        s = s2
        cmax_hist.append(float(s.makespan))

    out = {
        "config": vars(args),
        "initial": compute_metrics(random_assignment(jobs, args.m, make_rng(args.seed))).__dict__,
        "final_makespan": float(s.makespan),
        "cmax_hist": cmax_hist,
        "runtime_sec": float(t_total),
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"static_hybrid_seed{args.seed}.json"
    outpath.write_text(json.dumps(out, indent=2))
    print("Wrote:", outpath)


if __name__ == "__main__":
    main()