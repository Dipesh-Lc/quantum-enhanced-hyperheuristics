from __future__ import annotations
import numpy as np

from qehh.core.schedule import Schedule
from qehh.core.instances import generate_jobs, random_assignment
from qehh.core.rng import make_rng
from qehh.operators.classical import op_move_max_to_min, op_swap_two_jobs, op_two_smallest_from_max


def flatten(s: Schedule) -> list[float]:
    out = []
    for ms in s.machines:
        out.extend(ms)
    return sorted(out)


def test_schedule_makespan_loads_basic():
    s = Schedule(machines=[[1.0, 2.0], [3.0]])
    assert s.loads == [3.0, 3.0]
    assert s.makespan == 3.0


def test_classical_ops_preserve_multiset():
    rng = make_rng(0)
    jobs = generate_jobs(50, dist="uniform", seed=0)
    s0 = random_assignment(jobs, m=5, rng=rng)
    base = flatten(s0)

    for op in [op_move_max_to_min, op_swap_two_jobs, op_two_smallest_from_max]:
        r = make_rng(1)
        s1 = op(s0, r)
        assert flatten(s1) == base
        assert s1.n_jobs() == s0.n_jobs()
        assert s1.n_machines() == s0.n_machines()