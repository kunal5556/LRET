"""Quick benchmark test - immediate output."""
import sys
sys.path.insert(0, "d:/LRET/python")

import time
import numpy as np
from qlret import simulate_json
import cirq

print("Starting quick benchmark...", flush=True)

for n_qubits in [7, 8, 9]:
    print(f"\n=== {n_qubits} qubits, depth 15 ===", flush=True)
    
    # Build LRET circuit
    ops = [{"name": "H", "wires": [i]} for i in range(n_qubits)]
    for d in range(15):
        for i in range(0, n_qubits - 1, 2):
            ops.append({"name": "CNOT", "wires": [i, i+1]})
        if d % 2 == 1:
            for i in range(1, n_qubits - 1, 2):
                ops.append({"name": "CNOT", "wires": [i, i+1]})
    
    lret_circuit = {
        "circuit": {"num_qubits": n_qubits, "operations": ops},
        "config": {"epsilon": 1e-4, "initial_rank": 1},
    }
    
    # LRET timing
    print("Running LRET...", flush=True)
    start = time.perf_counter()
    lret_result = simulate_json(lret_circuit, export_state=True)
    lret_time = (time.perf_counter() - start) * 1000
    print(f"  LRET: {lret_time:.2f}ms, rank={lret_result.get('final_rank')}", flush=True)
    
    # Build Cirq circuit
    qubits = cirq.LineQubit.range(n_qubits)
    cirq_ops = [cirq.H(qubits[i]) for i in range(n_qubits)]
    for d in range(15):
        for i in range(0, n_qubits - 1, 2):
            cirq_ops.append(cirq.CNOT(qubits[i], qubits[i+1]))
        if d % 2 == 1:
            for i in range(1, n_qubits - 1, 2):
                cirq_ops.append(cirq.CNOT(qubits[i], qubits[i+1]))
    
    cirq_circuit = cirq.Circuit(cirq_ops)
    cirq_sim = cirq.DensityMatrixSimulator()
    
    # Cirq timing
    print("Running Cirq...", flush=True)
    start = time.perf_counter()
    cirq_result = cirq_sim.simulate(cirq_circuit)
    cirq_time = (time.perf_counter() - start) * 1000
    print(f"  Cirq: {cirq_time:.2f}ms", flush=True)
    
    speedup = cirq_time / lret_time
    print(f"  Speedup: {speedup:.1f}x", flush=True)
    
    # State comparison
    if 'state' in lret_result:
        state = lret_result['state']
        L_real = np.array(state['L_real']).reshape(-1, state.get('cols', 1))
        L_imag = np.array(state['L_imag']).reshape(-1, state.get('cols', 1))
        L = L_real + 1j * L_imag
        rho_lret = L @ L.conj().T
        rho_cirq = cirq_result.final_density_matrix
        
        fidelity = np.abs(np.trace(rho_lret @ rho_cirq)).real
        max_diff = np.max(np.abs(rho_lret - rho_cirq))
        print(f"  Fidelity: {fidelity:.6f}, Max diff: {max_diff:.2e}", flush=True)

print("\nDone!", flush=True)
