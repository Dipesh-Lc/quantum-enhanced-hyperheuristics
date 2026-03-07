from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


@dataclass(frozen=True)
class IsingModel:
    """
    E(z) = offset + sum_i h_i z_i + sum_{i<j} J_ij z_i z_j, where z_i ∈ {+1,-1}.
    """
    h: np.ndarray           # (n,)
    J: np.ndarray           # (n,n) symmetric with zeros on diagonal
    offset: float = 0.0


def qubo_to_ising(Q: np.ndarray) -> IsingModel:
    """
    Map QUBO energy x^T Q x (x in {0,1}^n) to an Ising model over spins z in {+1,-1}^n.
    Use mapping: x = (1 - z)/2  => z = 1 - 2x.

    Returns Ising parameters (h, J, offset) such that:
      E_ising(z) = offset + h·z + sum_{i<j} J_ij z_i z_j
    is equal to E_qubo(x) up to an exact constant shift.
    """
    Q = np.array(Q, dtype=float)
    n = Q.shape[0]
    if Q.shape != (n, n):
        raise ValueError("Q must be square")
    # Symmetrize (just in case)
    Qs = 0.5 * (Q + Q.T)

    # Expand E = sum_{i,j} Q_ij x_i x_j
    # Substitute x_i = (1 - z_i)/2.
    # For i==j: x_i x_i = x_i.
    #
    # Derive h, J, offset by algebra:
    # x_i x_j = (1 - z_i - z_j + z_i z_j)/4 for i!=j
    # x_i      = (1 - z_i)/2 for i==j terms

    h = np.zeros(n, dtype=float)
    J = np.zeros((n, n), dtype=float)
    offset = 0.0

    # Diagonal terms: Q_ii x_i
    for i in range(n):
        q = Qs[i, i]
        # q * (1 - z_i)/2 = q/2 - (q/2) z_i
        offset += q / 2.0
        h[i] += -q / 2.0

    # Off-diagonal terms: for i<j, contribution 2*Q_ij x_i x_j if using full x^T Q x with symmetric Q
    # But since we used x^T Q x with full matrix, the sum includes both (i,j) and (j,i).
    # After symmetrization, x^T Q x = sum_i Q_ii x_i + 2*sum_{i<j} Q_ij x_i x_j.
    for i in range(n):
        for j in range(i + 1, n):
            q = Qs[i, j]
            # contribution: 2*q * x_i x_j
            # 2q * (1 - z_i - z_j + z_i z_j)/4 = q/2 - (q/2)z_i - (q/2)z_j + (q/2) z_i z_j
            offset += q / 2.0
            h[i] += -q / 2.0
            h[j] += -q / 2.0
            J[i, j] += q / 2.0
            J[j, i] += q / 2.0

    return IsingModel(h=h, J=J, offset=float(offset))


def ising_energy(model: IsingModel, z: np.ndarray) -> float:
    """
    Compute E(z) for z in {+1,-1}^n.
    """
    z = z.astype(float)
    e = model.offset + float(model.h @ z)
    # sum_{i<j} J_ij z_i z_j
    n = z.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            e += float(model.J[i, j] * z[i] * z[j])
    return float(e)


def bitstring_to_spin(bits: np.ndarray) -> np.ndarray:
    """
    bits in {0,1} -> z in {+1,-1} with mapping x=(1-z)/2 => z=1-2x
    """
    bits = bits.astype(int)
    return 1.0 - 2.0 * bits


def build_qaoa_circuit(model: IsingModel, reps: int, gammas: np.ndarray, betas: np.ndarray) -> QuantumCircuit:
    """
    Build QAOA circuit for Ising Hamiltonian H_C = sum h_i Z_i + sum J_ij Z_i Z_j
    using phases:
      U_C(gamma) = exp(-i gamma H_C)
      U_B(beta)  = exp(-i beta sum X_i)
    """
    n = model.h.shape[0]
    qc = QuantumCircuit(n)

    # Start in |+>^n
    qc.h(range(n))

    for layer in range(reps):
        gamma = float(gammas[layer])
        beta = float(betas[layer])

        # Cost unitary from h_i Z_i terms: exp(-i gamma h_i Z) -> RZ(2*gamma*h_i)
        for i in range(n):
            hi = float(model.h[i])
            if abs(hi) > 1e-12:
                qc.rz(2.0 * gamma * hi, i)

        # Cost unitary from J_ij Z_i Z_j terms: exp(-i gamma J_ij Z_i Z_j) -> RZZ(2*gamma*J_ij)
        for i in range(n):
            for j in range(i + 1, n):
                Jij = float(model.J[i, j])
                if abs(Jij) > 1e-12:
                    qc.rzz(2.0 * gamma * Jij, i, j)

        # Mixer unitary: exp(-i beta X) -> RX(2*beta)
        for i in range(n):
            qc.rx(2.0 * beta, i)

    return qc


def statevector_expectation(model: IsingModel, sv: Statevector) -> float:
    """
    Compute expectation <H_C> from statevector by summing over computational basis probabilities.
    Deterministic and reproducible. Complexity O(2^n).
    """
    n = model.h.shape[0]
    probs = np.abs(np.array(sv.data)) ** 2
    exp_val = 0.0
    for idx, p in enumerate(probs):
        if p <= 0.0:
            continue
        bits = np.array([(idx >> (n - 1 - b)) & 1 for b in range(n)], dtype=int)
        z = bitstring_to_spin(bits)
        exp_val += float(p) * ising_energy(model, z)
    return float(exp_val)


def best_bitstring_from_statevector(model: IsingModel, sv: Statevector) -> Tuple[np.ndarray, float]:
    """
    Return the bitstring with minimal Ising energy among basis states, weighted by nonzero amplitude
    (i.e., among states with any probability > 0, but in practice, search all).
    """
    n = model.h.shape[0]
    probs = np.abs(np.array(sv.data)) ** 2

    best_bits = None
    best_e = float("inf")

    for idx, p in enumerate(probs):
        if p <= 0.0:
            continue
        bits = np.array([(idx >> (n - 1 - b)) & 1 for b in range(n)], dtype=int)
        z = bitstring_to_spin(bits)
        e = ising_energy(model, z)
        if e < best_e:
            best_e = e
            best_bits = bits

    if best_bits is None:
        best_bits = np.zeros(n, dtype=int)
        best_e = ising_energy(model, bitstring_to_spin(best_bits))

    return best_bits, float(best_e)


def solve_qubo_qaoa(
    Q: np.ndarray,
    *,
    reps: int = 1,
    shots: int = 0,  
    seed: int = 0,
    n_param_samples: int = 16,
) -> Dict[str, object]:
    """
    Solve a QUBO approximately with shallow QAOA via statevector simulation and seeded random search.

    - reps: p (1 or 2 recommended)
    - n_param_samples: number of random (gamma,beta) sets to try (seeded)
    Returns: dict with best bitstring and energies.
    """
    if reps < 1:
        raise ValueError("reps must be >= 1")

    Q = np.array(Q, dtype=float)
    n = Q.shape[0]
    if n == 0:
        return {"bitstring": np.array([], dtype=int), "ising_energy": 0.0, "qubo_energy": 0.0}

    model = qubo_to_ising(Q)
    rng = np.random.default_rng(int(seed))

    # Random-search parameter ranges: common practice gamma in [0, pi], beta in [0, pi/2]
    # Works reasonably for p=1..2.
    best_exp = float("inf")
    best_gammas = None
    best_betas = None
    best_sv = None

    for _ in range(int(n_param_samples)):
        gammas = rng.uniform(0.0, np.pi, size=reps)
        betas = rng.uniform(0.0, np.pi / 2.0, size=reps)

        qc = build_qaoa_circuit(model, reps, gammas, betas)
        sv = Statevector.from_instruction(qc)

        exp_val = statevector_expectation(model, sv)
        if exp_val < best_exp:
            best_exp = exp_val
            best_gammas = gammas
            best_betas = betas
            best_sv = sv

    assert best_sv is not None and best_gammas is not None and best_betas is not None

    best_bits, best_ising_e = best_bitstring_from_statevector(model, best_sv)

    # compute QUBO energy directly on bits x
    x = best_bits.astype(int)
    qubo_e = float(x @ Q @ x)

    return {
        "bitstring": best_bits,
        "ising_energy": float(best_ising_e),
        "qubo_energy": float(qubo_e),
        "exp_value": float(best_exp),
        "gammas": best_gammas,
        "betas": best_betas,
        "reps": int(reps),
        "n": int(n),
        "seed": int(seed),
        "n_param_samples": int(n_param_samples),
    }