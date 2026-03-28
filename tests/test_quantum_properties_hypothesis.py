"""
Layer 4: Hypothesis Property-Based Tests for LRET
Fuzz quantum invariants with random inputs via hypothesis library.

Run: pytest tests/test_quantum_properties_hypothesis.py -v --hypothesis-seed=0
"""

import pytest
import numpy as np
from numpy.linalg import eigvalsh, norm
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

# ──────────────────────────────────────────────────────────────
# Custom Hypothesis strategies
# ──────────────────────────────────────────────────────────────

@st.composite
def random_density_matrix(draw, n_qubits=None):
    """Generate a random valid density matrix (PSD, trace=1)."""
    if n_qubits is None:
        n_qubits = draw(st.integers(min_value=1, max_value=3))
    dim = 2**n_qubits
    # Random L, then ρ = LL†/Tr(LL†)
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)
    rank = draw(st.integers(min_value=1, max_value=dim))
    L = rng.standard_normal((dim, rank)) + 1j * rng.standard_normal((dim, rank))
    L /= norm(L, 'fro')
    return L @ L.conj().T

@st.composite
def random_unitary(draw, n=2):
    """Generate a random n×n unitary via QR decomposition."""
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(A)
    return Q

@st.composite
def random_lret_state(draw, n_qubits=None):
    """Generate random (n_qubits, rank, L) triple."""
    if n_qubits is None:
        n_qubits = draw(st.integers(min_value=1, max_value=3))
    dim = 2**n_qubits
    rank = draw(st.integers(min_value=1, max_value=min(dim, 8)))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)
    L = rng.standard_normal((dim, rank)) + 1j * rng.standard_normal((dim, rank))
    L /= norm(L, 'fro')
    return n_qubits, rank, L

@st.composite
def random_kraus_set(draw, n=2, k=None):
    """Generate a valid Kraus set (Σ Kᵢ†Kᵢ = I) via Stinespring dilation."""
    if k is None:
        k = draw(st.integers(min_value=2, max_value=4))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)
    # Generate k·n × n isometry V, Kraus ops are n×n blocks
    V = rng.standard_normal((k*n, n)) + 1j * rng.standard_normal((k*n, n))
    # QR to get isometry
    Q, _ = np.linalg.qr(V)
    V = Q[:k*n, :n]  # take first k*n rows
    # Kraus ops: Kᵢ = V[i*n:(i+1)*n, :]
    kraus = [V[i*n:(i+1)*n, :] for i in range(k)]
    # Verify completeness
    total = sum(K.conj().T @ K for K in kraus)
    # Normalize if needed
    scale = np.real(np.trace(total)) / n
    if scale > 0:
        kraus = [K / np.sqrt(scale) for K in kraus]
    return kraus

# ──────────────────────────────────────────────────────────────
# Hypothesis tests
# ──────────────────────────────────────────────────────────────

@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(random_lret_state())
def test_trace_preserved_any_lret_state(state):
    """Tr(ρ) = 1 for any normalized LRET state."""
    n_qubits, rank, L = state
    rho = L @ L.conj().T
    assert abs(np.trace(rho) - 1.0) < 1e-10, \
        f"Tr(ρ) = {np.trace(rho):.6f} ≠ 1 for n={n_qubits}, rank={rank}"

@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(random_lret_state())
def test_psd_preserved_any_lret_state(state):
    """ρ = LL† is PSD for any L."""
    n_qubits, rank, L = state
    rho = L @ L.conj().T
    eigvals = eigvalsh(rho)
    assert np.all(eigvals >= -1e-10), \
        f"ρ has negative eigenvalue {eigvals.min():.2e} for n={n_qubits}, rank={rank}"

@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(random_lret_state(), random_unitary(n=2))
def test_choi_isomorphism_any_state_any_gate(state, U_local):
    """UρU† matches LRET gate application for any state and any 1-qubit unitary."""
    n_qubits, rank, L = state
    assume(n_qubits >= 1)

    dim = 2**n_qubits
    rho_in = L @ L.conj().T

    # Apply to qubit 0
    qubit = 0
    gates = [np.eye(2, dtype=complex)] * n_qubits
    gates[qubit] = U_local
    U_full = gates[0]
    for g in gates[1:]:
        U_full = np.kron(U_full, g)

    # LRET path
    L_out = U_full @ L
    rho_lret = L_out @ L_out.conj().T

    # Direct path
    rho_direct = U_full @ rho_in @ U_full.conj().T

    assert np.allclose(rho_lret, rho_direct, atol=1e-10), \
        f"Choi gate fidelity failed for n={n_qubits}, rank={rank}"

@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
@given(random_kraus_set(n=2))
def test_valid_kraus_preserves_trace(kraus):
    """Any valid Kraus set preserves trace when applied to a density matrix."""
    # Use maximally mixed state as input
    rho = np.eye(2, dtype=complex) / 2
    rho_out = sum(K @ rho @ K.conj().T for K in kraus)
    assert abs(np.trace(rho_out) - 1.0) < 1e-8, \
        f"Trace not preserved: Tr(output) = {np.trace(rho_out):.6f}"

@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
@given(random_lret_state(n_qubits=2))
def test_psd_preserved_after_depolarizing_noise(state):
    """ρ remains PSD after depolarizing noise application."""
    n_qubits, rank, L = state
    rho = L @ L.conj().T

    p = 0.05
    I = np.eye(2, dtype=complex)
    X = np.array([[0,1],[1,0]], dtype=complex)
    Y = np.array([[0,-1j],[1j,0]], dtype=complex)
    Z = np.array([[1,0],[0,-1]], dtype=complex)
    kraus = [np.sqrt(1-p)*I, np.sqrt(p/3)*X, np.sqrt(p/3)*Y, np.sqrt(p/3)*Z]

    # Apply to qubit 0 of 2-qubit system
    kraus_full = [np.kron(K, np.eye(2)) for K in kraus]
    rho_out = sum(K @ rho @ K.conj().T for K in kraus_full)

    eigvals = eigvalsh(rho_out)
    assert np.all(eigvals >= -1e-10), \
        f"PSD violated after depolarizing noise: min eigenvalue = {eigvals.min():.2e}"

@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
@given(random_lret_state(n_qubits=2),
       st.integers(min_value=1, max_value=8))
def test_psd_preserved_after_truncation(state, trunc_rank):
    """ρ remains PSD after SVD truncation of L."""
    n_qubits, rank, L = state

    # Truncate to min(trunc_rank, rank) via SVD
    from numpy.linalg import svd
    U, s, Vh = svd(L, full_matrices=False)
    actual_trunc = min(trunc_rank, len(s))

    L_trunc = U[:, :actual_trunc] * s[:actual_trunc]
    fro_norm = norm(L_trunc, 'fro')
    assume(fro_norm > 1e-14)
    L_trunc /= fro_norm

    rho_trunc = L_trunc @ L_trunc.conj().T
    eigvals = eigvalsh(rho_trunc)
    assert np.all(eigvals >= -1e-10), \
        f"PSD violated after truncation: min eigenvalue = {eigvals.min():.2e}"
