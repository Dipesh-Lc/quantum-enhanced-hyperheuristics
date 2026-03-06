# src/qehh/quantum/qaoa_operator.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from qehh.core.schedule import Schedule
from qehh.operators.operator_types import Operator
from .subproblem import extract_balance_subproblem
from .qubo_builder import build_balance_qubo
from .qaoa_runner import solve_qubo_qaoa
from .decode import decode_balance_move, apply_balance_move


@dataclass(frozen=True)
class QAOAConfig:
    k: int = 10
    reps: int = 1
    n_param_samples: int = 16
    seed_offset: int = 0  # mix into seed for reproducibility across calls


def op_qaoa_balance_move(
    schedule: Schedule,
    rng: np.random.Generator,
    *,
    k: int = 10,
    reps: int = 1,
    n_param_samples: int = 16,
    seed_offset: int = 0,
) -> Schedule:
    """
    Low-level operator: extract (heavy, light, k jobs) balance subproblem,
    solve it with shallow QAOA, decode bitstring -> move, apply move.
    """
    sub = extract_balance_subproblem(schedule, rng, k=k, pick="largest")
    if sub is None:
        return schedule

    Q = build_balance_qubo(sub)

    # Seed QAOA deterministically from RNG + seed_offset without consuming too much randomness.
    # We'll draw one uint32 from rng and combine.
    base = int(rng.integers(0, 2**31 - 1))
    seed = (base + int(seed_offset)) % (2**31 - 1)

    sol = solve_qubo_qaoa(Q, reps=reps, seed=seed, n_param_samples=n_param_samples)
    bits = sol["bitstring"]

    move = decode_balance_move(sub, np.array(bits, dtype=int))
    new_schedule = apply_balance_move(schedule, move)
    return new_schedule


def make_qaoa_operator(
    name: str = "qaoa_balance_move",
    *,
    k: int = 10,
    reps: int = 1,
    n_param_samples: int = 16,
    seed_offset: int = 0,
) -> Operator:
    """
    Repo2-native Operator factory.
    """
    def _fn(s: Schedule, r: np.random.Generator) -> Schedule:
        return op_qaoa_balance_move(
            s,
            r,
            k=k,
            reps=reps,
            n_param_samples=n_param_samples,
            seed_offset=seed_offset,
        )

    return Operator(name=name, fn=_fn)