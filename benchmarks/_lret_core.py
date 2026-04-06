"""
Correct LRET algorithm implementation in numpy, faithful to:
  Chen, Farquhar, Parrish. "Low-rank density-matrix evolution for noisy quantum circuits."
  npj Quantum Information 7, 61 (2021).

This module provides the core LRET primitives:
  - 1-qubit and 2-qubit gate application on the low-rank factor L
  - Kraus channel noise application with proper rank expansion
  - Gram-matrix eigenvalue truncation
  - Per-qubit iterative Kraus decomposition (Section III.B of the paper)
  - Random dense circuit generation
  - Distortion metric computation
"""

import numpy as np
from numpy.linalg import norm, svd, eigvalsh


# ──────────────────────────────────────────────────────────────
# Standard gates
# ──────────────────────────────────────────────────────────────

GATES_1Q = {
    'H': np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
    'S': np.array([[1, 0], [0, 1j]], dtype=complex),
    'T': np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
    'I': np.eye(2, dtype=complex),
}

def _rx(theta):
    return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                     [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=complex)

def _ry(theta):
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                     [np.sin(theta/2), np.cos(theta/2)]], dtype=complex)

def _rz(theta):
    return np.array([[np.exp(-1j*theta/2), 0],
                     [0, np.exp(1j*theta/2)]], dtype=complex)

CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
CZ   = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]], dtype=complex)
SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)


# ──────────────────────────────────────────────────────────────
# Gate application on low-rank factor L (dim x rank)
# ──────────────────────────────────────────────────────────────

def apply_1q_gate(L, gate, qubit, n_qubits):
    """Apply 2x2 gate to qubit q of L (dim x rank) via tensor-index contraction.
    Complexity: O(2^n * rank).
    """
    rank = L.shape[1]
    L3 = L.reshape([2] * n_qubits + [rank])
    L3 = np.tensordot(gate, L3, axes=[[1], [qubit]])
    L3 = np.moveaxis(L3, 0, qubit)
    return L3.reshape(-1, rank)


def apply_2q_gate(L, gate4x4, q1, q2, n_qubits):
    """Apply 4x4 gate to qubits (q1, q2) of L (dim x rank).

    Gate convention: row/col index = (q1_bit << 1) | q2_bit.
    Reference: gates_and_noise.cpp:267-313
    Complexity: O(4 * 2^n * rank).
    """
    dim = L.shape[0]
    rank = L.shape[1]
    result = L.copy()

    step_q1 = 1 << q1
    step_q2 = 1 << q2

    for base in range(dim):
        if (base & step_q1) != 0 or (base & step_q2) != 0:
            continue
        idx = [
            base,                        # q1=0, q2=0
            base | step_q2,              # q1=0, q2=1
            base | step_q1,              # q1=1, q2=0
            base | step_q1 | step_q2,    # q1=1, q2=1
        ]
        v = L[idx, :]  # (4, rank)
        result[idx, :] = gate4x4 @ v

    return result


# ──────────────────────────────────────────────────────────────
# Kraus noise application with rank expansion
# Reference: gates_and_noise.cpp:719-772
# ──────────────────────────────────────────────────────────────

def depolarizing_kraus(p):
    """Return 4 Kraus operators for the depolarizing channel.
    rho -> (1-p) rho + (p/3)(X rho X + Y rho Y + Z rho Z)
    """
    c0 = np.sqrt(1.0 - p)
    c1 = np.sqrt(p / 3.0)
    K0 = c0 * GATES_1Q['I']
    K1 = c1 * GATES_1Q['X']
    K2 = c1 * GATES_1Q['Y']
    K3 = c1 * GATES_1Q['Z']
    return [K0, K1, K2, K3]


def apply_kraus_1q(L, kraus_ops, qubit, n_qubits):
    """Apply a single-qubit Kraus channel to L with proper rank expansion.

    L_new = [K_0 L | K_1 L | ... | K_{A-1} L]  (column concatenation)
    New rank = old_rank * len(kraus_ops).

    Reference: gates_and_noise.cpp:747-769
    """
    rank = L.shape[1]
    dim = L.shape[0]
    num_kraus = len(kraus_ops)

    L_new = np.empty((dim, rank * num_kraus), dtype=complex)
    for k, K in enumerate(kraus_ops):
        L_k = apply_1q_gate(L, K, qubit, n_qubits)
        L_new[:, k * rank:(k + 1) * rank] = L_k

    return L_new


# ──────────────────────────────────────────────────────────────
# Eigenvalue truncation via Gram matrix
# Reference: simulator.cpp:56-130
# ──────────────────────────────────────────────────────────────

def truncate_eigenvalue(L, epsilon, max_rank=0):
    """Truncate L by discarding small eigenvalue components of rho = L L†.

    Algorithm (paper Section "Eigenvalue truncation"):
      1. G = L† L  (rank x rank Gram matrix)
      2. Eigendecompose G → eigenvalues, eigenvectors
      3. Keep eigenvalues > epsilon * total_trace
      4. L_new = L * V_kept, renormalize to Tr[rho] = 1

    Complexity: O(rank^2 * 2^n) for Gram, O(rank^3) for eigen.
    """
    if L.shape[1] <= 1:
        return L

    # Gram matrix (rank x rank)
    G = L.conj().T @ L

    # Eigendecomposition (returns ascending eigenvalues)
    eigenvalues = np.linalg.eigvalsh(G)
    # Full eigen for eigenvectors
    eigenvalues_full, eigenvectors = np.linalg.eigh(G)

    total_trace = eigenvalues_full.sum().real
    if total_trace < 1e-15:
        return L[:, :1]

    threshold_val = epsilon * total_trace

    # Keep eigenvalues above threshold
    keep_mask = eigenvalues_full.real > threshold_val
    if not np.any(keep_mask):
        # Keep at least the largest eigenvalue
        keep_mask[-1] = True

    kept_indices = np.where(keep_mask)[0]

    # Apply max_rank limit
    if max_rank > 0 and len(kept_indices) > max_rank:
        # Sort by eigenvalue descending, keep top max_rank
        sorted_idx = kept_indices[np.argsort(-eigenvalues_full[kept_indices].real)]
        kept_indices = sorted_idx[:max_rank]

    new_rank = len(kept_indices)
    if new_rank >= L.shape[1]:
        return L  # No truncation needed

    V_kept = eigenvectors[:, kept_indices]
    L_new = L @ V_kept

    # Renormalize to Tr[rho] = 1
    fro = norm(L_new, 'fro')
    if fro > 1e-15:
        L_new /= fro

    return L_new


# ──────────────────────────────────────────────────────────────
# Per-qubit iterative Kraus decomposition (Paper Section III.B)
# Prevents rank explosion by truncating after each qubit's noise
# ──────────────────────────────────────────────────────────────

def apply_noise_layer_iterative(L, noise_prob, n_qubits, epsilon, max_rank=0):
    """Apply depolarizing noise to ALL qubits in one layer, with
    per-qubit iterative truncation.

    Paper Algorithm 1: for beta = 1 to B:
        L = Concatenate_alpha(sqrt(p_alpha) * K_{beta,alpha} * L)
        L = EigenvalueTruncation_epsilon(L)

    Here B = n_qubits (one Kraus group per qubit).
    """
    kraus_ops = depolarizing_kraus(noise_prob)

    for q in range(n_qubits):
        L = apply_kraus_1q(L, kraus_ops, q, n_qubits)
        L = truncate_eigenvalue(L, epsilon, max_rank=max_rank)

    return L


# ──────────────────────────────────────────────────────────────
# Random dense circuit generation (Paper Section "Randomized benchmarking")
# ──────────────────────────────────────────────────────────────

def build_random_dense_circuit(n_qubits, depth, rng=None):
    """Generate a random dense circuit as described in the paper.

    Each layer:
      - One random 1-qubit gate per qubit (from {H,X,Y,Z,S,T,RX,RY,RZ})
      - Adjacent-pair 2-qubit gates (CNOT or CZ, alternating parity)

    Returns a list of layers, each layer = list of (gate, qubits) tuples.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    gate_names_1q = ['H', 'X', 'Y', 'Z', 'S', 'T', 'RX', 'RY', 'RZ']
    gate_names_2q = ['CNOT', 'CZ']
    gates_2q = {'CNOT': CNOT, 'CZ': CZ}

    circuit = []
    for d in range(depth):
        layer = []
        # 1-qubit gates on every qubit
        for q in range(n_qubits):
            name = rng.choice(gate_names_1q)
            if name in GATES_1Q:
                gate = GATES_1Q[name]
            elif name == 'RX':
                gate = _rx(rng.uniform(0, 2 * np.pi))
            elif name == 'RY':
                gate = _ry(rng.uniform(0, 2 * np.pi))
            elif name == 'RZ':
                gate = _rz(rng.uniform(0, 2 * np.pi))
            layer.append(('1q', gate, q))

        # 2-qubit gates on adjacent pairs (alternating even/odd layers)
        start = d % 2
        for i in range(start, n_qubits - 1, 2):
            name = rng.choice(gate_names_2q)
            layer.append(('2q', gates_2q[name], i, i + 1))

        circuit.append(layer)
    return circuit


def build_cirq_circuit_from_layers(circuit_layers, n_qubits, noise_prob):
    """Convert our circuit layer format into a Cirq circuit with depolarizing noise."""
    import cirq

    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()

    for layer in circuit_layers:
        ops = []
        for instr in layer:
            if instr[0] == '1q':
                _, gate, q = instr
                ops.append(cirq.MatrixGate(gate).on(qubits[q]))
            elif instr[0] == '2q':
                _, gate, q1, q2 = instr
                ops.append(cirq.MatrixGate(gate).on(qubits[q1], qubits[q2]))
        circuit.append(ops)

        # Depolarizing noise on every qubit after each layer
        if noise_prob > 0:
            circuit.append([cirq.depolarize(noise_prob).on(q) for q in qubits])

    return circuit


# ──────────────────────────────────────────────────────────────
# Full LRET simulation (Algorithm 1 from the paper)
# ──────────────────────────────────────────────────────────────

def run_lret_simulation(circuit_layers, n_qubits, noise_prob, epsilon=1e-4, max_rank=0):
    """Run the full LRET algorithm on a circuit.

    Algorithm 1 (paper):
      L(0) = [1, 0, ..., 0]^T
      for d = 1 to D:
          L(d) = G(d) L(d-1)                    // gate application
          for beta = 1 to B:                     // iterative Kraus per qubit
              L(d) = Concat(sqrt(p_a) K_{b,a} L)
              L(d) = Truncation_eps(L(d))

    Returns: L_final, timing_ms, max_rank_seen
    """
    import time

    dim = 2 ** n_qubits
    L = np.zeros((dim, 1), dtype=complex)
    L[0, 0] = 1.0  # |0...0> state

    max_rank_seen = 1

    t0 = time.perf_counter()

    for layer in circuit_layers:
        # Apply gates
        for instr in layer:
            if instr[0] == '1q':
                _, gate, q = instr
                L = apply_1q_gate(L, gate, q, n_qubits)
            elif instr[0] == '2q':
                _, gate, q1, q2 = instr
                L = apply_2q_gate(L, gate, q1, q2, n_qubits)

        # Apply noise (iterative per-qubit Kraus + truncation)
        if noise_prob > 0:
            L = apply_noise_layer_iterative(L, noise_prob, n_qubits, epsilon,
                                            max_rank=max_rank)
            if L.shape[1] > max_rank_seen:
                max_rank_seen = L.shape[1]

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return L, elapsed_ms, max_rank_seen


# ──────────────────────────────────────────────────────────────
# Error metrics (Paper Section "Implementation and benchmarking")
# ──────────────────────────────────────────────────────────────

def reconstruct_density_matrix(L):
    """Compute rho = L L† from low-rank factor."""
    return L @ L.conj().T


def trace_distance(rho_a, rho_b):
    """T(A, B) = 0.5 * Tr|A - B| = 0.5 * sum of singular values of (A - B)."""
    diff = rho_a - rho_b
    s = np.linalg.svd(diff, compute_uv=False)
    return 0.5 * np.sum(s)


def compute_distortion(L_lret, rho_exact, rho_noiseless):
    """Paper's distortion metric (Eq. 10):
    Distortion = T(rho_LRET, rho_exact) / T(rho_exact, rho_noiseless)

    Only feasible for small qubit counts (N <= ~12) since it requires
    full density matrices.
    """
    rho_lret = reconstruct_density_matrix(L_lret)
    numerator = trace_distance(rho_lret, rho_exact)
    denominator = trace_distance(rho_exact, rho_noiseless)
    if denominator < 1e-15:
        return 0.0  # No noise effect
    return float(numerator / denominator)


def compute_probability_distribution(L):
    """Prob(x) = sum_v |L_{x,v}|^2  (paper Eq. 9)."""
    return np.sum(np.abs(L) ** 2, axis=1)


def probability_tvd(prob_a, prob_b):
    """Total variation distance between probability distributions."""
    return 0.5 * np.sum(np.abs(prob_a - prob_b))
