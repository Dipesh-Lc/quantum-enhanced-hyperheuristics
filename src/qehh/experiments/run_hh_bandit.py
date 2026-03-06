from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

from qehh.core.instances import generate_jobs, random_assignment
from qehh.core.rng import make_rng
from qehh.core.metrics import compute_metrics
from qehh.operators.operator_types import Operator
from qehh.operators.registry import default_operators
from qehh.quantum.qaoa_operator import make_qaoa_operator


@dataclass
class BanditStats:
    pulls: int = 0
    total_reward: float = 0.0  # reward = improvement in makespan (positive is good)
    total_time: float = 0.0    # runtime spent in operator
    best_reward: float = -1e18


def ucb1_score(stats: BanditStats, t: int, c: float = 1.5) -> float:
    if stats.pulls == 0:
        return float("inf")
    avg = stats.total_reward / stats.pulls
    bonus = c * np.sqrt(np.log(max(t, 2)) / stats.pulls)
    return avg + bonus


def run_bandit_hh(
    ops: List[Operator],
    schedule,
    rng: np.random.Generator,
    *,
    steps: int = 200,
    policy: str = "ucb1",   # "ucb1" or "eps_greedy"
    eps: float = 0.1,
    ucb_c: float = 1.5,
) -> Dict[str, Any]:
    stats = {op.name: BanditStats() for op in ops}
    cmax_hist = [float(schedule.makespan)]
    chosen_hist: List[str] = []
    time_hist: List[float] = []
    reward_hist: List[float] = []
    proposal_reward_hist: List[float] = []

    s = schedule

    for t in range(1, steps + 1):
        # choose operator
        if policy == "eps_greedy":
            if rng.random() < eps:
                idx = int(rng.integers(0, len(ops)))
            else:
                # exploit: highest average reward
                avg_rewards = []
                for op in ops:
                    st = stats[op.name]
                    avg_rewards.append((st.total_reward / st.pulls) if st.pulls > 0 else float("inf"))
                idx = int(np.argmax(avg_rewards))
        elif policy == "ucb1":
            scores = [ucb1_score(stats[op.name], t, c=ucb_c) for op in ops]
            idx = int(np.argmax(scores))
        else:
            raise ValueError(f"Unknown policy: {policy}")

        op = ops[idx]
        before = float(s.makespan)

        t0 = time.perf_counter()
        s2 = op.fn(s, rng)
        dt = time.perf_counter() - t0

        after = float(s2.makespan)
        proposal_reward = before - after

        if after <= before: 
            s = s2  # accept-if-better

        # reward should reflect the state transition that actually occurred
        reward = before - float(s.makespan) # >=0 always under accept-if-better

        # # update stats based on accepted reward
        st = stats[op.name]
        st.pulls += 1
        st.total_reward += float(reward)
        st.total_time += float(dt)
        st.best_reward = max(st.best_reward, float(reward)) 

        cmax_hist.append(float(s.makespan))
        chosen_hist.append(op.name)
        time_hist.append(float(dt))
        reward_hist.append(float(reward))
        proposal_reward_hist.append(float(proposal_reward))

    # pack stats
    out_stats = {}
    for name, st in stats.items():
        out_stats[name] = {
            "pulls": st.pulls,
            "avg_reward": (st.total_reward / st.pulls) if st.pulls > 0 else 0.0,
            "best_reward": st.best_reward,
            "total_time_sec": st.total_time,
            "avg_time_ms": (1000.0 * st.total_time / st.pulls) if st.pulls > 0 else 0.0,
        }

    return {
        "final_makespan": float(s.makespan),
        "cmax_hist": cmax_hist,
        "chosen_hist": chosen_hist,
        "reward_hist": reward_hist,
        "proposal_reward_hist": proposal_reward_hist,
        "time_hist": time_hist,
        "stats": out_stats,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_jobs", type=int, default=200)
    ap.add_argument("--m", type=int, default=10)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--dist", type=str, default="lognormal")
    ap.add_argument("--policy", type=str, default="ucb1")      # ucb1|eps_greedy
    ap.add_argument("--eps", type=float, default=0.1)          # for eps_greedy
    ap.add_argument("--ucb_c", type=float, default=1.5)        # for ucb1
    ap.add_argument("--outdir", type=str, default="results/bandit_hh")
    args = ap.parse_args()

    rng = make_rng(args.seed)
    jobs = generate_jobs(args.n_jobs, dist=args.dist, seed=args.seed)
    sched0 = random_assignment(jobs, args.m, rng)

    ops = default_operators()

    # Use your tuned fast QAOA settings:
    ops.append(make_qaoa_operator(k=10, reps=1, n_param_samples=16, seed_offset=1337))

    res = {
        "config": vars(args),
        "initial": compute_metrics(sched0).__dict__,
        "bandit": run_bandit_hh(
            ops,
            sched0,
            make_rng(args.seed + 999),  # separate rng for action selection
            steps=args.steps,
            policy=args.policy,
            eps=args.eps,
            ucb_c=args.ucb_c,
        ),
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"bandit_{args.policy}_seed{args.seed}.json"
    outpath.write_text(json.dumps(res, indent=2))
    print(f"Wrote: {outpath}")


if __name__ == "__main__":
    main()