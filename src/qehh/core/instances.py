from __future__ import annotations
from typing import List, Literal
import numpy as np

from .schedule import Schedule


Dist = Literal["uniform", "lognormal", "pareto"]


def generate_jobs(
    n: int,
    dist: Dist = "uniform",
    *,
    seed: int | None = 0,
    low: float = 1.0,
    high: float = 100.0,
) -> List[float]:
    """
    Generate processing times.

    - uniform: U(low, high)
    - lognormal: scaled lognormal for heavy-tail-ish behavior
    - pareto: classic heavy tail (shifted/scaled)
    """
    rng = np.random.default_rng(seed)

    if n <= 0:
        return []

    if dist == "uniform":
        jobs = rng.uniform(low, high, size=n)

    elif dist == "lognormal":
        # median ~ exp(mu), heavy tail controlled by sigma
        mu, sigma = 2.0, 1.0
        raw = rng.lognormal(mean=mu, sigma=sigma, size=n)
        # scale to roughly [low, high]
        raw = raw / (np.max(raw) + 1e-12)
        jobs = low + (high - low) * raw

    elif dist == "pareto":
        # Pareto(alpha) => heavy tail; shift to avoid extreme blow-ups
        alpha = 2.0
        raw = rng.pareto(alpha, size=n) + 1.0
        raw = raw / (np.max(raw) + 1e-12)
        jobs = low + (high - low) * raw

    else:
        raise ValueError(f"Unknown dist: {dist}")

    return [float(x) for x in jobs]


def random_assignment(jobs: List[float], m: int, rng: np.random.Generator) -> Schedule:
    """
    Randomly assign jobs to machines.
    """
    if m <= 0:
        raise ValueError("m must be >= 1")
    machines: List[List[float]] = [[] for _ in range(m)]
    for p in jobs:
        i = int(rng.integers(0, m))
        machines[i].append(float(p))
    return Schedule(machines=machines)