from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np

from .schedule import Schedule


@dataclass(frozen=True)
class ScheduleMetrics:
    makespan: float
    loads: List[float]
    imbalance: float  # max(loads) - min(loads)


def compute_metrics(schedule: Schedule) -> ScheduleMetrics:
    loads = schedule.loads
    if not loads:
        return ScheduleMetrics(makespan=0.0, loads=[], imbalance=0.0)
    return ScheduleMetrics(
        makespan=schedule.makespan,
        loads=loads,
        imbalance=float(np.max(loads) - np.min(loads)),
    )