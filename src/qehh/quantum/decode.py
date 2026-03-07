from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np

from qehh.core.schedule import Schedule
from .subproblem import BalanceSubproblem


@dataclass(frozen=True)
class BalanceMove:
    heavy_idx: int
    light_idx: int
    # positions (indices) in heavy machine list to move
    heavy_positions_to_move: List[int]


def decode_balance_move(sub: BalanceSubproblem, bitstring: np.ndarray) -> BalanceMove:
    """
    bitstring[i] = 1 => move job at sub.heavy_job_positions[i] from heavy to light
    """
    bitstring = np.array(bitstring, dtype=int).ravel()
    if bitstring.shape[0] != len(sub.heavy_job_positions):
        raise ValueError("bitstring length mismatch")

    positions = []
    for i, b in enumerate(bitstring.tolist()):
        if int(b) == 1:
            positions.append(int(sub.heavy_job_positions[i]))

    return BalanceMove(
        heavy_idx=int(sub.heavy_idx),
        light_idx=int(sub.light_idx),
        heavy_positions_to_move=positions,
    )


def apply_balance_move(schedule: Schedule, move: BalanceMove) -> Schedule:
    machines = [list(ms) for ms in schedule.machines]
    H = move.heavy_idx
    L = move.light_idx

    if H < 0 or H >= len(machines) or L < 0 or L >= len(machines):
        return schedule

    if H == L or len(machines[H]) == 0:
        return schedule

    # remove from heavy in descending order to keep indices valid
    to_move = sorted(move.heavy_positions_to_move, reverse=True)
    moved = []
    for pos in to_move:
        if 0 <= pos < len(machines[H]):
            moved.append(machines[H].pop(pos))

    for job in moved:
        machines[L].append(float(job))

    return Schedule(machines=machines)