# src/qehh/core/rng.py
from __future__ import annotations
import numpy as np


def make_rng(seed: int | None) -> np.random.Generator:
    """
    Create a numpy Generator with a fixed seed for reproducibility.
    If seed is None, uses entropy (not reproducible).
    """
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(seed))