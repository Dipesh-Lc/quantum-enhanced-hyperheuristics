# Quantum‑Enhanced Hyperheuristics (QEHh)

**Hybrid Classical-Quantum Metaheuristic Framework for Scheduling
Optimization**

This repository implements a **quantum‑enhanced hyperheuristic
framework** where a shallow **QAOA‑based quantum operator** is
integrated with classical heuristics to solve **identical machine
scheduling** problems.

The goal is **not to replace classical heuristics**, but to **augment
them with quantum search primitives** and evaluate how they interact
with **adaptive hyperheuristic controllers**.

---

# Overview

This project demonstrates a **hybrid classical-quantum metaheuristic
pipeline**:

1.  Classical scheduling heuristics
2.  Quantum subproblem formulation
3.  QAOA solution via quantum simulation
4.  Hybrid operator integration
5.  Hyperheuristic control (bandit)
6.  Empirical evaluation across seeds
7.  Integration adapter for external RL controller (Repo 1: https://github.com/Dipesh-Lc/rl-hyperheuristic )

The repository is intentionally **independent of the RL system (Repo
1)** but includes an adapter allowing seamless integration.

---

# Key Idea

Large scheduling problems are difficult to solve globally. Instead we:

1.  Extract a **small local subproblem**
2.  Encode it as a **QUBO**
3.  Solve using **shallow QAOA**
4.  Convert the result back into a scheduling move

This produces a **quantum‑inspired local search operator**.

---

# Mathematical Foundations

## Scheduling Objective

We consider identical machine scheduling with objective:

$$
\min C_{\max}
$$

where

$$
C_{\max} = \max_{i} \sum_{j \in M_i} p_j
$$

- $p_j$ : processing time of job $j$
- $M_i$ : set of jobs assigned to machine $i$

The goal is to **minimize makespan**.

---

# Local Quantum Subproblem

At each step we identify:

- the **most loaded machine**
- the **least loaded machine**

Define load imbalance:

$$
\Delta = L_{max} - L_{min}
$$

Select a subset of jobs

$$
S = \{p_1, p_2, ..., p_k\}
$$

from the overloaded machine.

Binary variable:

$$
x_i =
\begin{cases}
1 & \text{move job } i \\
0 & \text{keep job } i
\end{cases}
$$

New imbalance after move:

$$
\Delta' = \Delta - 2\sum_i p_i x_i
$$

We minimize squared imbalance:

$$
\min (\Delta - 2\sum_i p_i x_i)^2
$$

which expands to a **QUBO**:

$$
E(x) = x^T Q x + c
$$

This QUBO is solved with **QAOA**.

---

# QAOA Formulation

Convert QUBO to Ising Hamiltonian:

$$
x_i = \frac{1 - z_i}{2}
$$

giving

$$
H_C = \sum_{i,j} Q_{ij} Z_i Z_j + \sum_i h_i Z_i
$$

QAOA ansatz:

$$
|\psi(\beta,\gamma)\rangle =
\prod_{l=1}^{p}
e^{-i\beta_l H_M}
e^{-i\gamma_l H_C}
|+\rangle^{\otimes n}
$$

where

Mixer Hamiltonian:

$$
H_M = \sum_i X_i
$$

Parameters $(\beta,\gamma)$ are optimized via random sampling.

---

# Quantum Operator

The QAOA operator works as:

1.  Extract job subset
2.  Build QUBO
3.  Run shallow QAOA
4.  Decode best bitstring
5.  Apply job moves to schedule

This becomes a **low‑level heuristic** inside the hyperheuristic framework.

---

# Classical Operators

Implemented baseline heuristics:

- move_max_to_min
- two_smallest_from_max
- swap_two_jobs

These provide the classical baseline.

---

# Hyperheuristic Controller

Two adaptive controllers are evaluated.

## Static Hybrid

First steps use QAOA, then classical heuristics.

## Multi‑Armed Bandit

Operators treated as arms:

$$
a \in \{H_1,H_2,H_3,Q\}
$$

Selection uses **UCB1**:

$$
a_t = \arg\max
\left(
\bar r_a + c\sqrt{\frac{\ln t}{n_a}}
\right)
$$

where

- $\bar r_a$ = average reward
- $n_a$ = number of pulls

Reward:

$$
r = C_{before} - C_{after}
$$

---

# Repository Structure

    src/qehh
     ├── adapters
     ├── core
     ├── experiments
     ├── operators
     └── quantum

### Core

Scheduling primitives.

### Operators

Classical heuristics and operator registry.

### Quantum

QAOA pipeline:

- QUBO builder
- Ising conversion
- circuit execution
- decoding

### Experiments

Reproducible experiment scripts.

---

# Installation

## Conda

    conda env create -f environment.yml
    conda activate qehh
    pip install -e .

---

# Running Experiments

## Operator‑only comparison

    python -m qehh.experiments.run_operator_only --seed 0

Generate plots

    python -m qehh.experiments.plot_results

---

# Static Hybrid

    python -m qehh.experiments.run_static_hybrid --seed 0

---

# Bandit Hyperheuristic

    python -m qehh.experiments.run_hh_bandit --seed 0 --policy ucb1

---

# Aggregate Results

    python -m qehh.experiments.aggregate_operator_only
    python -m qehh.experiments.aggregate_static_hybrid
    python -m qehh.experiments.compare_methods

---

# Results Summary

Average across 10 seeds:

| Method                | Final Makespan | Steps to 1% | Runtime |
|-----------------------|----------------|-------------|---------|
| move_max_to_min       | 281.48         | 22.6        | 0.002s  |
| two_smallest_from_max | 271.05         | 22.3        | 0.002s  |
| QAOA operator         | 268.05         | **6.7**     | 80.7s   |
| Static Hybrid         | 266.43         | 8.9         | 3.65s   |
| Bandit HH             | **265.96**     | 17.3        | ~4s     |

Key findings:

- **QAOA achieves fastest convergence**
- Classical heuristics are efficient for refinement
- Hybrid control provides best trade‑off


The full raw experiment outputs and aggregated statistics used to produce these tables are available in the `results/` directory of this repository.

---

# MaxCut QAOA Demonstration

Sanity check implementation:

    python -m qehh.experiments.sanity_maxcut

Parameter sweep:

    python -m qehh.experiments.maxcut_param_sweep

---

# Integration with RL Hyperheuristic (Repo 1) -- *Future Work*

This repository provides an adapter:

    qehh.adapters.repo1_operator

which exposes the QAOA operator as a standard operator for the RL
controller.

Add to Repo 1 operator list:

    make_repo1_qaoa_operator(k=10, reps=1, n_param_samples=16)

Then retrain RL.

---

# Testing

    pytest

Includes tests for:

- QUBO energy correctness
- schedule invariants
- decode logic
- QAOA operator smoke test

---

# Key Insight

Quantum heuristics are **not universally superior**.

Instead they are valuable for **specific search states**, particularly:

- large imbalance
- early exploration phases

Hybrid metaheuristics can therefore leverage quantum operators
selectively.

---

# Future Work

- RL‑driven operator selection
- Larger scheduling benchmarks
- Hardware execution on real quantum devices
- Parameter‑learning QAOA

---

# References

The mathematical models and algorithms used in this project build upon established research in scheduling theory, hyper-heuristics, combinatorial optimization, and quantum optimization.

### Scheduling Theory

1. Pinedo, M. L.  
   *Scheduling: Theory, Algorithms, and Systems (5th Edition).*  
   Springer, 2016.  
   https://doi.org/10.1007/978-3-319-26580-3

2. Graham, R. L., Lawler, E. L., Lenstra, J. K., & Rinnooy Kan, A. H. G.  
   *Optimization and Approximation in Deterministic Sequencing and Scheduling.*  
   Annals of Discrete Mathematics, 1979.

3. Brucker, P.  
   *Scheduling Algorithms (5th Edition).*  
   Springer, 2007.

### Hyper-Heuristics

4. Burke, E. K., Hyde, M., Kendall, G., Ochoa, G., Özcan, E., & Woodward, J.  
   *A Classification of Hyper-Heuristic Approaches.*  
   In Handbook of Metaheuristics, Springer, 2010.

5. Cowling, P., Kendall, G., & Soubeiga, E.  
   *A Hyperheuristic Approach to Scheduling a Sales Summit.*  
   In Practice and Theory of Automated Timetabling, 2001.

### QUBO and Ising Formulations

6. Lucas, A.  
   *Ising formulations of many NP problems.*  
   Frontiers in Physics, 2014.  
   https://doi.org/10.3389/fphy.2014.00005

7. Glover, F., Kochenberger, G., & Du, Y.  
   *A Tutorial on Formulating and Using QUBO Models.*  
   arXiv:1811.11538

### Quantum Approximate Optimization Algorithm (QAOA)

8. Farhi, E., Goldstone, J., & Gutmann, S.  
   *A Quantum Approximate Optimization Algorithm.*  
   arXiv:1411.4028

9. Hadfield, S. et al.  
   *From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz.*  
   Algorithms, 2019.

### Hybrid Quantum–Classical Optimization

10. Preskill, J.  
    *Quantum Computing in the NISQ era and beyond.*  
    Quantum, 2018.  
    https://doi.org/10.22331/q-2018-08-06-79

11. Cerezo, M. et al.  
    *Variational Quantum Algorithms.*  
    Nature Reviews Physics, 2021.

### Multi-Armed Bandit Algorithms

12. Auer, P., Cesa-Bianchi, N., & Fischer, P.  
    *Finite-time Analysis of the Multiarmed Bandit Problem.*  
    Machine Learning, 2002.

13. Sutton, R. S., & Barto, A. G.  
    *Reinforcement Learning: An Introduction (2nd Edition).*  
    MIT Press, 2018.

