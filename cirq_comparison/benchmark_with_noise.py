"""
Comprehensive Benchmark with Noise: LRET vs Cirq
- Qubits: 7-9
- Depth: 15
- Noise: 0.1% depolarizing per gate
- With state export and plots
"""
import sys
sys.path.insert(0, "d:/LRET/python")

import time
import numpy as np
import json
from pathlib import Path
import cirq

# Import LRET native module directly (avoid slow PennyLane imports)
import qlret._qlret_native as lret_native

def simulate_json(circuit, export_state=False):
    """Simulate using native bindings."""
    json_str = json.dumps(circuit)
    result_str = lret_native.run_circuit_json(json_str, export_state)
    return json.loads(result_str)

# For plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping plots")

# Output directory
OUTPUT_DIR = Path("d:/LRET/cirq_comparison/benchmark_results_with_noise")
OUTPUT_DIR.mkdir(exist_ok=True)

log_file = OUTPUT_DIR / "benchmark_log.txt"

def log(msg):
    """Write to log file and print."""
    print(msg)
    with open(log_file, 'a') as f:
        f.write(msg + '\n')

log("="*80)
log("COMPREHENSIVE BENCHMARK WITH NOISE: LRET vs Cirq")
log("Configuration: 7-9 qubits, depth=15, noise=0.1% per gate")
log("="*80)

# Configuration
QUBITS = [7, 8, 9]
DEPTH = 15
NOISE_PROB = 0.001  # 0.1%
N_TRIALS = 3

def build_circuit_lret(n_qubits, depth, noise_prob):
    """Build circuit for LRET with noise."""
    ops = []
    
    # Initial H layer
    for i in range(n_qubits):
        ops.append({"name": "H", "wires": [i]})
        if noise_prob > 0:
            ops.append({"name": "DEPOLARIZE", "wires": [i], "params": [noise_prob]})
    
    # CNOT layers
    for d in range(depth):
        # Even layer
        for i in range(0, n_qubits - 1, 2):
            ops.append({"name": "CNOT", "wires": [i, i+1]})
            if noise_prob > 0:
                ops.append({"name": "DEPOLARIZE", "wires": [i], "params": [noise_prob]})
                ops.append({"name": "DEPOLARIZE", "wires": [i+1], "params": [noise_prob]})
        
        # Odd layer
        if d % 2 == 1:
            for i in range(1, n_qubits - 1, 2):
                ops.append({"name": "CNOT", "wires": [i, i+1]})
                if noise_prob > 0:
                    ops.append({"name": "DEPOLARIZE", "wires": [i], "params": [noise_prob]})
                    ops.append({"name": "DEPOLARIZE", "wires": [i+1], "params": [noise_prob]})
    
    return {"circuit": {"num_qubits": n_qubits, "operations": ops}, "config": {"epsilon": 1e-4, "initial_rank": 1}}

def build_circuit_cirq(n_qubits, depth, noise_prob):
    """Build equivalent circuit for Cirq with noise."""
    qubits = cirq.LineQubit.range(n_qubits)
    ops = []
    
    # Initial H layer
    for i in range(n_qubits):
        ops.append(cirq.H(qubits[i]))
        if noise_prob > 0:
            ops.append(cirq.depolarize(noise_prob).on(qubits[i]))
    
    # CNOT layers
    for d in range(depth):
        # Even layer
        for i in range(0, n_qubits - 1, 2):
            ops.append(cirq.CNOT(qubits[i], qubits[i+1]))
            if noise_prob > 0:
                ops.append(cirq.depolarize(noise_prob).on(qubits[i]))
                ops.append(cirq.depolarize(noise_prob).on(qubits[i+1]))
        
        # Odd layer
        if d % 2 == 1:
            for i in range(1, n_qubits - 1, 2):
                ops.append(cirq.CNOT(qubits[i], qubits[i+1]))
                if noise_prob > 0:
                    ops.append(cirq.depolarize(noise_prob).on(qubits[i]))
                    ops.append(cirq.depolarize(noise_prob).on(qubits[i+1]))
    
    return cirq.Circuit(ops)

results = []

for n_qubits in QUBITS:
    log(f"\n=== Testing {n_qubits} qubits, depth={DEPTH}, noise={NOISE_PROB*100:.1f}% ===")
    
    lret_circuit = build_circuit_lret(n_qubits, DEPTH, NOISE_PROB)
    cirq_circuit = build_circuit_cirq(n_qubits, DEPTH, NOISE_PROB)
    cirq_sim = cirq.DensityMatrixSimulator()
    
    # LRET
    log("  Running LRET...")
    lret_times = []
    for trial in range(N_TRIALS):
        start = time.perf_counter()
        lret_result = simulate_json(lret_circuit, export_state=(trial==0))
        lret_times.append((time.perf_counter() - start) * 1000)
    
    lret_mean = np.mean(lret_times)
    lret_std = np.std(lret_times)
    rank = lret_result.get('final_rank', 1)
    status = lret_result.get('status', 'unknown')
    log(f"    LRET: {lret_mean:.2f}±{lret_std:.2f}ms, rank={rank}, status={status}")
    
    # Cirq
    log("  Running Cirq...")
    cirq_times = []
    for trial in range(N_TRIALS):
        start = time.perf_counter()
        cirq_result = cirq_sim.simulate(cirq_circuit)
        cirq_times.append((time.perf_counter() - start) * 1000)
    
    cirq_mean = np.mean(cirq_times)
    cirq_std = np.std(cirq_times)
    log(f"    Cirq: {cirq_mean:.2f}±{cirq_std:.2f}ms")
    
    speedup = cirq_mean / lret_mean
    log(f"    Speedup: {speedup:.1f}x")
    
    results.append({
        'n_qubits': n_qubits,
        'lret_mean': lret_mean,
        'lret_std': lret_std,
        'cirq_mean': cirq_mean,
        'cirq_std': cirq_std,
        'speedup': speedup,
        'rank': rank,
        'status': status,
    })

# Save results
json_path = OUTPUT_DIR / "results.json"
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
log(f"\nSaved to: {json_path}")

# Generate plots
if HAS_MATPLOTLIB and len(results) > 0:
    log("\nGenerating plots...")
    
    qubits = [r['n_qubits'] for r in results]
    lret_times = [r['lret_mean'] for r in results]
    cirq_times = [r['cirq_mean'] for r in results]
    speedups = [r['speedup'] for r in results]
    ranks = [r['rank'] for r in results]
    
    # Time comparison + Speedup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(qubits, lret_times, 'o-', label='LRET', linewidth=2, markersize=8)
    ax1.plot(qubits, cirq_times, 's-', label='Cirq', linewidth=2, markersize=8)
    ax1.set_xlabel('Qubits', fontsize=12)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title(f'Execution Time (depth={DEPTH}, noise={NOISE_PROB*100:.1f}%)', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    ax2.plot(qubits, speedups, 'o-', color='green', linewidth=2, markersize=8)
    ax2.set_xlabel('Qubits', fontsize=12)
    ax2.set_ylabel('Speedup (x)', fontsize=12)
    ax2.set_title('LRET Speedup over Cirq', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "benchmark_plot.png"
    plt.savefig(plot_path, dpi=150)
    log(f"  Saved: {plot_path}")
    plt.close()
    
    # Rank evolution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(qubits, ranks, 'o-', color='purple', linewidth=2, markersize=8)
    ax.set_xlabel('Qubits', fontsize=12)
    ax.set_ylabel('Final Rank', fontsize=12)
    ax.set_title(f'LRET Rank Evolution (depth={DEPTH}, noise={NOISE_PROB*100:.1f}%)', fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    rank_plot_path = OUTPUT_DIR / "rank_plot.png"
    plt.savefig(rank_plot_path, dpi=150)
    log(f"  Saved: {rank_plot_path}")
    plt.close()

log("\n" + "="*80)
log("Summary:")
for r in results:
    log(f"  {r['n_qubits']}q: LRET={r['lret_mean']:.2f}ms, Cirq={r['cirq_mean']:.2f}ms, Speedup={r['speedup']:.1f}x, Rank={r['rank']}")

avg_speedup = np.mean([r['speedup'] for r in results])
log(f"\nAverage speedup: {avg_speedup:.1f}x")
log("="*80)
log("BENCHMARK COMPLETE!")
log("="*80)
