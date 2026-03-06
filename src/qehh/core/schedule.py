# src/qehh/core/schedule.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List


def _clone_machines(machines: List[List[float]]) -> List[List[float]]:
    return [list(ms) for ms in machines]


@dataclass(frozen=True)
class Schedule:
    """
    Minimal, independent schedule representation for identical machines.
    Matches Repo 1 structure: machines: List[List[float]].
    """
    machines: List[List[float]]

    def clone(self) -> "Schedule":
        return Schedule(machines=_clone_machines(self.machines))

    @property
    def loads(self) -> List[float]:
        return [sum(ms) for ms in self.machines]

    @property
    def makespan(self) -> float:
        return max(self.loads) if self.machines else 0.0

    def n_machines(self) -> int:
        return len(self.machines)

    def n_jobs(self) -> int:
        return sum(len(ms) for ms in self.machines)