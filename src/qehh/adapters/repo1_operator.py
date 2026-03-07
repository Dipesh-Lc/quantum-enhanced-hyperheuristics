
#Integration adapter for RL Hyper-Heuristic for Identical Parallel Machine Scheduling 
# (Repo 1: https://github.com/Dipesh-Lc/rl-hyperheuristic )

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

from qehh.quantum.qaoa_operator import op_qaoa_balance_move
from qehh.core.schedule import Schedule as Repo2Schedule

def make_repo1_qaoa_operator(
    name: str = "qaoa_balance_move",
    *,
    k: int = 12,
    reps: int = 1,
    n_param_samples: int = 64,
    seed_offset: int = 0,
):
    """
    Returns Repo 1's Operator dataclass instance:
      Operator(name: str, fn: Callable[[rl_hh.vendor.identical_scheduling.Schedule, np.random.Generator], Schedule])

    Usage in Repo 1:
      from qehh.adapters.repo1_operator import make_repo1_qaoa_operator
      ops = default_operators() + [make_repo1_qaoa_operator(...)]
    """
    from rl_hh.heuristics.operators import Operator as Repo1Operator  # type: ignore
    from rl_hh.vendor.identical_scheduling import Schedule as Repo1Schedule  # type: ignore

    def _fn(s: "Repo1Schedule", r: np.random.Generator) -> "Repo1Schedule":
        # Convert repo1 Schedule -> repo2 Schedule (same structure)
        s2 = Repo2Schedule(machines=[list(ms) for ms in s.machines])
        out2 = op_qaoa_balance_move(
            s2,
            r,
            k=k,
            reps=reps,
            n_param_samples=n_param_samples,
            seed_offset=seed_offset,
        )
        # Convert back repo2 -> repo1 Schedule
        return Repo1Schedule(machines=[list(ms) for ms in out2.machines])

    return Repo1Operator(name=name, fn=_fn)