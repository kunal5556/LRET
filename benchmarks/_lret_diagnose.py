"""
LRET-vs-Cirq minimal reproducer.

Walks from N=2/depth=1/no-noise → progressively more complex circuits,
asserting at each step that rho_LRET ≈ rho_Cirq. The smallest failing
case localises the bug.

Usage:
    python benchmarks/_lret_diagnose.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from _lret_core import (
    GATES_1Q, CNOT, CZ,
    apply_1q_gate, apply_2q_gate,
    build_random_dense_circuit, build_cirq_circuit_from_layers,
    run_lret_simulation, reconstruct_density_matrix,
    apply_kraus_1q, depolarizing_kraus,
)

import cirq


def lret_run_layers(circuit_layers, n_qubits, noise_prob=0.0, epsilon=1e-12):
    """Wrapper that returns ρ_LRET via the standard run_lret_simulation."""
    L, _, _ = run_lret_simulation(circuit_layers, n_qubits, noise_prob, epsilon=epsilon)
    return reconstruct_density_matrix(L)


def cirq_run_layers(circuit_layers, n_qubits, noise_prob=0.0):
    """Build matching Cirq circuit and return its density matrix."""
    c = build_cirq_circuit_from_layers(circuit_layers, n_qubits, noise_prob)
    sim = cirq.DensityMatrixSimulator()
    qubits = cirq.LineQubit.range(n_qubits)
    return sim.simulate(c, qubit_order=qubits).final_density_matrix


def report(name, rho_a, rho_b):
    diff = np.linalg.norm(rho_a - rho_b)
    # Cirq's DensityMatrixSimulator uses float32 internally → tolerance ~1e-5
    tol = 1e-5
    ok = diff < tol
    tag = "OK " if ok else "FAIL"
    print(f"  [{tag}] {name:60s}  ||rho_LRET - rho_Cirq|| = {diff:.3e}")
    if not ok:
        # Show probability vectors for inspection
        pa = np.real(np.diag(rho_a))
        pb = np.real(np.diag(rho_b))
        print(f"        diag(LRET) = {np.round(pa, 4)}")
        print(f"        diag(Cirq) = {np.round(pb, 4)}")
    return ok


def case_single_x_q0_n2():
    """Apply X to qubit 0 of |00⟩. Expected: |10⟩.
    flat index 2 in MSB convention, flat index 1 in LSB.
    """
    n = 2
    layers = [[('1q', GATES_1Q['X'], 0)]]
    rho_l = lret_run_layers(layers, n)
    rho_c = cirq_run_layers(layers, n)
    return report("1q X on q0 (n=2)", rho_l, rho_c)


def case_single_x_q1_n2():
    n = 2
    layers = [[('1q', GATES_1Q['X'], 1)]]
    rho_l = lret_run_layers(layers, n)
    rho_c = cirq_run_layers(layers, n)
    return report("1q X on q1 (n=2)", rho_l, rho_c)


def case_cnot_n2():
    """H on q0 then CNOT(q0,q1) → Bell state."""
    n = 2
    layers = [[('1q', GATES_1Q['H'], 0), ('2q', CNOT, 0, 1)]]
    rho_l = lret_run_layers(layers, n)
    rho_c = cirq_run_layers(layers, n)
    return report("H(q0) + CNOT(q0,q1) (Bell, n=2)", rho_l, rho_c)


def case_cnot_target_first_n3():
    """X(q1) then CNOT(q1,q2) on n=3."""
    n = 3
    layers = [[('1q', GATES_1Q['X'], 1), ('2q', CNOT, 1, 2)]]
    rho_l = lret_run_layers(layers, n)
    rho_c = cirq_run_layers(layers, n)
    return report("X(q1) + CNOT(q1,q2) (n=3)", rho_l, rho_c)


def case_cz_n2():
    n = 2
    layers = [[('1q', GATES_1Q['H'], 0), ('1q', GATES_1Q['H'], 1),
               ('2q', CZ, 0, 1)]]
    rho_l = lret_run_layers(layers, n)
    rho_c = cirq_run_layers(layers, n)
    return report("H(q0) H(q1) CZ(q0,q1) (n=2)", rho_l, rho_c)


def case_random_no_noise(n, depth, seed=7):
    rng = np.random.default_rng(seed)
    layers = build_random_dense_circuit(n, depth, rng)
    rho_l = lret_run_layers(layers, n, noise_prob=0.0)
    rho_c = cirq_run_layers(layers, n, noise_prob=0.0)
    return report(f"random circuit n={n} depth={depth} no-noise", rho_l, rho_c)


def case_random_with_noise(n, depth, p=0.001, seed=7):
    rng = np.random.default_rng(seed)
    layers = build_random_dense_circuit(n, depth, rng)
    rho_l = lret_run_layers(layers, n, noise_prob=p, epsilon=1e-12)
    rho_c = cirq_run_layers(layers, n, noise_prob=p)
    return report(f"random circuit n={n} depth={depth} p={p}", rho_l, rho_c)


def main():
    print("=" * 78)
    print("LRET vs Cirq diagnostic walk")
    print("=" * 78)

    cases = [
        case_single_x_q0_n2,
        case_single_x_q1_n2,
        case_cnot_n2,
        case_cnot_target_first_n3,
        case_cz_n2,
        lambda: case_random_no_noise(2, 1),
        lambda: case_random_no_noise(2, 3),
        lambda: case_random_no_noise(3, 3),
        lambda: case_random_no_noise(4, 5),
        lambda: case_random_no_noise(6, 8),
        lambda: case_random_with_noise(2, 3, p=0.001),
        lambda: case_random_with_noise(4, 5, p=0.001),
        lambda: case_random_with_noise(6, 8, p=0.001),
    ]

    n_pass = 0
    n_fail = 0
    for c in cases:
        try:
            ok = c()
            n_pass += int(ok)
            n_fail += int(not ok)
        except Exception as exc:
            print(f"  [ERR ] {c.__name__}: {exc}")
            n_fail += 1

    print("-" * 78)
    print(f"  PASS: {n_pass}    FAIL: {n_fail}")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == '__main__':
    main()
