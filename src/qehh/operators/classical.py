from __future__ import annotations
from typing import List
import numpy as np

from qehh.core.schedule import Schedule


def _clone_machines(machines: List[List[float]]) -> List[List[float]]:
    return [list(ms) for ms in machines]


def op_swap_two_jobs(schedule: Schedule, rng: np.random.Generator) -> Schedule:
    machines = _clone_machines(schedule.machines)
    m = len(machines)
    if m < 1:
        return schedule

    nonempty = [i for i in range(m) if len(machines[i]) > 0]
    if len(nonempty) < 1:
        return schedule

    i = int(rng.choice(nonempty))
    j = int(rng.choice(nonempty)) if len(nonempty) > 1 else i

    ai = int(rng.integers(0, len(machines[i])))
    aj = int(rng.integers(0, len(machines[j])))

    machines[i][ai], machines[j][aj] = machines[j][aj], machines[i][ai]
    return Schedule(machines=machines)


def op_move_max_to_min(schedule: Schedule, rng: np.random.Generator) -> Schedule:
    machines = _clone_machines(schedule.machines)
    if not machines:
        return schedule

    loads = [sum(ms) for ms in machines]
    i_max = int(np.argmax(loads))
    i_min = int(np.argmin(loads))

    if i_max == i_min or len(machines[i_max]) == 0:
        return schedule

    k = int(rng.integers(0, len(machines[i_max])))
    job = machines[i_max].pop(k)
    machines[i_min].append(job)
    return Schedule(machines=machines)


def op_two_smallest_from_max(schedule: Schedule, rng: np.random.Generator) -> Schedule:
    machines = _clone_machines(schedule.machines)
    if not machines:
        return schedule

    loads = [sum(ms) for ms in machines]
    i_max = int(np.argmax(loads))
    if len(machines[i_max]) == 0:
        return schedule

    jobs = machines[i_max]
    idx_sorted = sorted(range(len(jobs)), key=lambda t: jobs[t])
    take = idx_sorted[: min(2, len(idx_sorted))]

    moved = []
    for idx in sorted(take, reverse=True):
        moved.append(machines[i_max].pop(idx))

    loads2 = [sum(ms) for ms in machines]
    i_min = int(np.argmin(loads2))

    for job in moved:
        machines[i_min].append(job)

    return Schedule(machines=machines)