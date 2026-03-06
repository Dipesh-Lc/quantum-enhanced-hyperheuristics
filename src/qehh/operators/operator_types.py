# src/qehh/operators/operator_types.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np

from qehh.core.schedule import Schedule


@dataclass(frozen=True)
class Operator:
    name: str
    fn: Callable[[Schedule, np.random.Generator], Schedule]