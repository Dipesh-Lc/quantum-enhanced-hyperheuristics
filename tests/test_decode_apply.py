# tests/test_decode_apply.py
from __future__ import annotations
import numpy as np

from qehh.core.schedule import Schedule
from qehh.quantum.subproblem import BalanceSubproblem
from qehh.quantum.decode import decode_balance_move, apply_balance_move


def test_decode_apply_move_preserves_jobs():
    s = Schedule(machines=[[10.0, 3.0, 2.0], [1.0], [4.0]])
    # heavy is machine 0, light is machine 1 (loads 15 vs 1)
    sub = BalanceSubproblem(
        heavy_idx=0,
        light_idx=1,
        heavy_job_positions=[0, 2],  # jobs 10.0 and 2.0
        p=np.array([10.0, 2.0], dtype=float),
        load_H=15.0,
        load_L=1.0,
    )

    bits = np.array([1, 0], dtype=int)  # move only job at pos 0 (10.0)
    move = decode_balance_move(sub, bits)
    s2 = apply_balance_move(s, move)

    # Check multiset equality
    flat1 = sorted([p for ms in s.machines for p in ms])
    flat2 = sorted([p for ms in s2.machines for p in ms])
    assert flat1 == flat2

    # moved job should now be on machine 1
    assert 10.0 in s2.machines[1]