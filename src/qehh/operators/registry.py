from __future__ import annotations
from typing import List

from .operator_types import Operator
from .classical import op_move_max_to_min, op_swap_two_jobs, op_two_smallest_from_max


def default_operators() -> List[Operator]:
    """
    Minimal baseline set for Repo 2 experiments.
    """
    return [
        Operator("move_max_to_min", op_move_max_to_min),
        Operator("swap_two_jobs", op_swap_two_jobs),
        Operator("two_smallest_from_max", op_two_smallest_from_max),
    ]