"""
Complete benchmark with file logging - 7-9 qubits, depth 15
Saves all results to files as it runs
"""
import sys
sys.path.insert(0, "d:/LRET/python")

import time
import numpy as np
import json
from pathlib import Path
import cirq

# Import LRET native module directly (avoid slow PennyLane imports)
sys.path.insert(0, "d:/LRET/python")
import qlret._qlret_native as lret_native

def simulate_json(circuit, export_state=False):
    """Simulate using native bindings."""
    json_str = json.dumps(circuit)
    result_str = lret_native.run_circuit_json(json_str, export_state)
    return json.loads(result_str)

# Output
OUTPUT_DIR = Path("d:/LRET/cirq_comparison/benchmark_results")
OUTPUT_DIR.mkdir(exist_ok=True)

log_file = OUTPUT_DIR / "benchmark_log.txt"

def log(msg):
    """Write to log file and print."""
    print(msg)
    with open(log_file, 'a') as f:
        f.write(msg + '\n')

log("="*80)
log("COMPREHENSIVE BENCHMARK: LRET vs Cirq")
log("="*80)

results = []

for n_qubits in [7, 8, 9]:
    log(f"\n=== Testing {n_qubits} qubits, depth=15 ===")
    
    # Build LRET circuit
    ops = [{"name": "H", "wires": [i]} for i in range(n_qubits)]
    for d in range(15):
        for i in range(0, n_qubits - 1, 2):
            ops.append({"name": "CNOT", "wires": [i, i+1]})
        if d % 2 == 1:
            for i in range(1, n_qubits - 1, 2):
                ops.append({"name": "CNOT", "wires": [i, i+1]})
    
    lret_circuit = {"circuit": {"num_qubits": n_qubits, "operations": ops}, "config": {"epsilon": 1e-4, "initial_rank": 1}}
    
    # LRET
    log("  Running LRET...")
    lret_times = []
    for trial in range(3):
        start = time.perf_counter()
        lret_result = simulate_json(lret_circuit, export_state=(trial==0))
        lret_times.append((time.perf_counter() - start) * 1000)
    
    lret_mean = np.mean(lret_times)
    lret_std = np.std(lret_times)
    rank = lret_result.get('final_rank', 1)
    log(f"    LRET: {lret_mean:.2f}±{lret_std:.2f}ms, rank={rank}")
    
    # Cirq
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
    
    log("  Running Cirq...")
    cirq_times = []
    for trial in range(3):
        start = time.perf_counter()
        cirq_result = cirq_sim.simulate(cirq_circuit)
        cirq_times.append((time.perf_counter() - start) * 1000)
    
    cirq_mean = np.mean(cirq_times)
    cirq_std = np.std(cirq_times)
    log(f"    Cirq: {cirq_mean:.2f}±{cirq_std:.2f}ms")
    
    speedup = cirq_mean / lret_mean
    log(f"    Speedup: {speedup:.1f}x")
    
    # Fidelity
    if 'state' in lret_result:
        state = lret_result['state']
        L_real = np.array(state['L_real']).reshape(-1, state.get('cols', 1))
        L_imag = np.array(state['L_imag']).reshape(-1, state.get('cols', 1))
        L = L_real + 1j * L_imag
        rho_lret = L @ L.conj().T
        rho_cirq = cirq_result.final_density_matrix
        
        fidelity = np.abs(np.trace(rho_lret @ rho_cirq)).real
        max_diff = np.max(np.abs(rho_lret - rho_cirq))
        log(f"    Fidelity: {fidelity:.6f}, Max diff: {max_diff:.2e}")
    else:
        fidelity = None
        max_diff = None
        log(f"    Fidelity: N/A")
    
    results.append({
        'n_qubits': n_qubits,
        'lret_mean': lret_mean,
        'lret_std': lret_std,
        'cirq_mean': cirq_mean,
        'cirq_std': cirq_std,
        'speedup': speedup,
        'rank': rank,
        'fidelity': fidelity,
        'max_diff': max_diff,
    })

# Save results
json_path = OUTPUT_DIR / "results.json"
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
log(f"\nSaved to: {json_path}")

#Generate plots
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    qubits = [r['n_qubits'] for r in results]
    lret_times = [r['lret_mean'] for r in results]
    cirq_times = [r['cirq_mean'] for r in results]
    speedups = [r['speedup'] for r in results]
    
    # Time comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(qubits, lret_times, 'o-', label='LRET', linewidth=2, markersize=8)
    ax1.plot(qubits, cirq_times, 's-', label='Cirq', linewidth=2, markersize=8)
    ax1.set_xlabel('Qubits', fontsize=12)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title('Execution Time: LRET vs Cirq', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    ax2.plot(qubits, speedups, 'o-', color='green', linewidth=2, markersize=8)
    ax2.set_xlabel('Qubits', fontsize=12)
    ax2.set_ylabel('Speedup (x)', fontsize=12)
    ax2.set_title('LRET Speedup over Cirq', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "benchmark_plot.png"
    plt.savefig(plot_path, dpi=150)
    log(f"Saved plot: {plot_path}")
    plt.close()
    
except Exception as e:
    log(f"Plot generation failed: {e}")

log("\n" + "="*80)
log("BENCHMARK COMPLETE!")
log("="*80)
