from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from qehh.core.schedule import Schedule


@dataclass(frozen=True)
class BalanceSubproblem:
    heavy_idx: int
    light_idx: int
    # positions of selected jobs in the heavy machine list
    heavy_job_positions: List[int]
    # processing times of those selected jobs in the same order
    p: np.ndarray
    load_H: float
    load_L: float


def extract_balance_subproblem(
    schedule: Schedule,
    rng: np.random.Generator,
    *,
    k: int = 12,
    pick: str = "largest",  # "largest" or "random"
) -> Optional[BalanceSubproblem]:
    machines = schedule.machines
    m = len(machines)
    if m < 2:
        return None

    loads = schedule.loads
    heavy_idx = int(np.argmax(loads))
    light_idx = int(np.argmin(loads))

    if heavy_idx == light_idx:
        return None

    heavy_jobs = machines[heavy_idx]
    if len(heavy_jobs) < 1:
        return None

    k_eff = min(k, len(heavy_jobs))
    if k_eff < 2:
        return None

    if pick == "largest":
        # choose top-k by processing time (stable, good for balancing)
        idx_sorted = sorted(range(len(heavy_jobs)), key=lambda t: heavy_jobs[t], reverse=True)
        pos = idx_sorted[:k_eff]
    elif pick == "random":
        pos = rng.choice(len(heavy_jobs), size=k_eff, replace=False).tolist()
    else:
        raise ValueError(f"Unknown pick strategy: {pick}")

    p = np.array([float(heavy_jobs[i]) for i in pos], dtype=float)
    return BalanceSubproblem(
        heavy_idx=heavy_idx,
        light_idx=light_idx,
        heavy_job_positions=list(pos),
        p=p,
        load_H=float(loads[heavy_idx]),
        load_L=float(loads[light_idx]),
    )