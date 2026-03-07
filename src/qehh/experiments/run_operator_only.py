from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np

from qehh.core.instances import generate_jobs, random_assignment
from qehh.core.rng import make_rng
from qehh.core.metrics import compute_metrics
from qehh.operators.registry import default_operators
from qehh.quantum.qaoa_operator import make_qaoa_operator


def run_episode(op, schedule, rng, steps: int) -> dict:
    cmax_hist = []
    t_total = 0.0

    s = schedule
    for _ in range(steps):
        t0 = time.perf_counter()
        s2 = op.fn(s, rng)
        t_total += time.perf_counter() - t0
        s = s2
        cmax_hist.append(float(s.makespan))

    return {
        "final_makespan": float(s.makespan),
        "cmax_hist": cmax_hist,
        "runtime_sec": float(t_total),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_jobs", type=int, default=200)
    ap.add_argument("--m", type=int, default=10)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--dist", type=str, default="lognormal")
    ap.add_argument("--outdir", type=str, default="results/operator_only")
    args = ap.parse_args()

    rng = make_rng(args.seed)
    jobs = generate_jobs(args.n_jobs, dist=args.dist, seed=args.seed)
    sched0 = random_assignment(jobs, args.m, rng)

    ops = default_operators()
    ops.append(make_qaoa_operator(k=10, reps=1, n_param_samples=16, seed_offset=1337))

    results = {
        "config": vars(args),
        "initial": compute_metrics(sched0).__dict__,
        "runs": [],
    }

    for op in ops:
        ep_rng = make_rng(args.seed + 1000)  # ensure fair comparison per operator
        run = run_episode(op, sched0, ep_rng, steps=args.steps)
        results["runs"].append({"operator": op.name, **run})

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"operator_only_seed{args.seed}.json"
    outpath.write_text(json.dumps(results, indent=2))
    print(f"Wrote: {outpath}")


if __name__ == "__main__":
    main()