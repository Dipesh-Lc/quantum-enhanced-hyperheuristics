# tests/test_qaoa_operator_smoke.py
from __future__ import annotations
import numpy as np

from qehh.core.instances import generate_jobs, random_assignment
from qehh.core.rng import make_rng
from qehh.quantum.qaoa_operator import op_qaoa_balance_move


def flatten(machines):
    out = []
    for ms in machines:
        out.extend(ms)
    return sorted(out)


def test_qaoa_operator_returns_valid_schedule():
    rng = make_rng(0)
    jobs = generate_jobs(80, dist="lognormal", seed=0)
    s0 = random_assignment(jobs, m=6, rng=rng)

    base = flatten(s0.machines)
    s1 = op_qaoa_balance_move(s0, make_rng(1), k=10, reps=1, n_param_samples=16, seed_offset=0)

    assert flatten(s1.machines) == base
    assert len(s1.machines) == len(s0.machines)
    assert s1.makespan >= 0.0