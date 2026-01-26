"""
Comprehensive Benchmark: LRET vs Cirq (NO NOISE VERSION)
- Qubits: 7-11
- Depth: 20 (increased to compensate for no noise)
- Pure state circuits to see rank=1 behavior
- Complete analysis with plots
"""
import sys
sys.path.insert(0, "d:/LRET/python")

import time
import numpy as np
import json
from pathlib import Path
from qlret import simulate_json
import cirq

# For plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping plots")

print("="*80)
print("COMPREHENSIVE BENCHMARK: LRET vs Cirq (Pure States)")
print("Configuration: 7-9 qubits, depth=15, no noise")
print("="*80)

# Output directory
OUTPUT_DIR = Path("d:/LRET/cirq_comparison/benchmark_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Benchmark configuration
QUBITS = list(range(7, 10))  # 7, 8, 9 only (Cirq is too slow for 10+)
DEPTH = 15
N_TRIALS = 3

def build_circuit_lret(n_qubits, depth):
    """Build circuit for LRET."""
    ops = [{"name": "H", "wires": [i]} for i in range(n_qubits)]
    
    for d in range(depth):
        for i in range(0, n_qubits - 1, 2):
            ops.append({"name": "CNOT", "wires": [i, i+1]})
        if d % 2 == 1:
            for i in range(1, n_qubits - 1, 2):
                ops.append({"name": "CNOT", "wires": [i, i+1]})
    
    return {
        "circuit": {"num_qubits": n_qubits, "operations": ops},
        "config": {"epsilon": 1e-4, "initial_rank": 1},
    }

def build_circuit_cirq(n_qubits, depth):
    """Build equivalent circuit for Cirq."""
    qubits = cirq.LineQubit.range(n_qubits)
    ops = [cirq.H(qubits[i]) for i in range(n_qubits)]
    
    for d in range(depth):
        for i in range(0, n_qubits - 1, 2):
            ops.append(cirq.CNOT(qubits[i], qubits[i+1]))
        if d % 2 == 1:
            for i in range(1, n_qubits - 1, 2):
                ops.append(cirq.CNOT(qubits[i], qubits[i+1]))
    
    return cirq.Circuit(ops)

def compute_fidelity(rho1, rho2):
    """Compute fidelity between two density matrices."""
    return np.abs(np.trace(rho1 @ rho2)).real

def run_benchmark_single(n_qubits, depth):
    """Run single benchmark configuration."""
    print(f"\n--- Testing {n_qubits} qubits, depth={depth} ---")
    
    lret_circuit = build_circuit_lret(n_qubits, depth)
    cirq_circuit = build_circuit_cirq(n_qubits, depth)
    cirq_sim = cirq.DensityMatrixSimulator()
    
    # LRET timing
    lret_times = []
    lret_internal = []
    lret_rank = None
    lret_state = None
    
    for trial in range(N_TRIALS):
        start = time.perf_counter()
        result = simulate_json(lret_circuit, export_state=(trial == 0), use_native=True)
        elapsed = (time.perf_counter() - start) * 1000
        lret_times.append(elapsed)
        lret_internal.append(result.get('execution_time_ms', 0))
        
        if trial == 0:
            lret_rank = result.get('final_rank', 1)
            if result.get('status') == 'success' and 'state' in result:
                state = result['state']
                try:
                    L_real = np.array(state['L_real']).reshape(-1, state.get('cols', 1))
                    L_imag = np.array(state['L_imag']).reshape(-1, state.get('cols', 1))
                    L = L_real + 1j * L_imag
                    lret_state = L @ L.conj().T
                except Exception as e:
                    print(f"    Warning: Failed to reconstruct LRET state: {e}")
                    lret_state = None
    
    # Cirq timing
    cirq_times = []
    cirq_state = None
    
    for trial in range(N_TRIALS):
        start = time.perf_counter()
        cirq_result = cirq_sim.simulate(cirq_circuit)
        elapsed = (time.perf_counter() - start) * 1000
        cirq_times.append(elapsed)
        
        if trial == 0:
            cirq_state = cirq_result.final_density_matrix
    
    # Compute metrics
    fidelity = compute_fidelity(lret_state, cirq_state) if lret_state is not None else None
    trace_dist = 0.5 * np.sum(np.abs(np.linalg.eigvalsh(lret_state - cirq_state))) if lret_state is not None else None
    max_diff = np.max(np.abs(lret_state - cirq_state)) if lret_state is not None else None
    
    result = {
        'n_qubits': n_qubits,
        'depth': depth,
        'lret_wall_mean': np.mean(lret_times),
        'lret_wall_std': np.std(lret_times),
        'lret_internal_mean': np.mean(lret_internal),
        'cirq_mean': np.mean(cirq_times),
        'cirq_std': np.std(cirq_times),
        'speedup': np.mean(cirq_times) / np.mean(lret_times),
        'final_rank': lret_rank,
        'fidelity': fidelity,
        'trace_distance': trace_dist,
        'max_diff': max_diff,
    }
    
    print(f"  LRET: {result['lret_wall_mean']:.2f}±{result['lret_wall_std']:.2f}ms (internal: {result['lret_internal_mean']:.2f}ms)")
    print(f"  Cirq: {result['cirq_mean']:.2f}±{result['cirq_std']:.2f}ms")
    print(f"  Speedup: {result['speedup']:.1f}x")
    print(f"  Rank: {result['final_rank']}")
    if result['fidelity'] is not None:
        print(f"  Fidelity: {result['fidelity']:.6f}")
        print(f"  Trace distance: {result['trace_distance']:.6e}")
        print(f"  Max |diff|: {result['max_diff']:.6e}")
    else:
        print(f"  Fidelity: N/A (state not available)")
    
    return result

# Run benchmarks
print("\n" + "="*80)
print("RUNNING BENCHMARKS")
print("="*80)

results = []
for n_qubits in QUBITS:
    try:
        result = run_benchmark_single(n_qubits, DEPTH)
        results.append(result)
    except Exception as e:
        print(f"ERROR for {n_qubits} qubits: {e}")
        import traceback
        traceback.print_exc()

# Save results to CSV
csv_path = OUTPUT_DIR / "benchmark_results.csv"
print(f"\nSaving results to {csv_path}")

with open(csv_path, 'w') as f:
    f.write("n_qubits,depth,lret_wall_mean,lret_wall_std,lret_internal_mean,")
    f.write("cirq_mean,cirq_std,speedup,final_rank,fidelity,trace_distance,max_diff\n")
    
    for r in results:
        f.write(f"{r['n_qubits']},{r['depth']},{r['lret_wall_mean']:.6f},")
        f.write(f"{r['lret_wall_std']:.6f},{r['lret_internal_mean']:.6f},{r['cirq_mean']:.6f},")
        f.write(f"{r['cirq_std']:.6f},{r['speedup']:.6f},{r['final_rank']},")
        fid = r['fidelity'] if r['fidelity'] is not None else 0.0
        td = r['trace_distance'] if r['trace_distance'] is not None else 0.0
        md = r['max_diff'] if r['max_diff'] is not None else 0.0
        f.write(f"{fid:.10f},{td:.10e},{md:.10e}\n")

# Save results to JSON
json_path = OUTPUT_DIR / "benchmark_results.json"
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Saved results to {json_path}")

# Generate plots
if HAS_MATPLOTLIB and len(results) > 0:
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    qubits = [r['n_qubits'] for r in results]
    lret_times = [r['lret_wall_mean'] for r in results]
    lret_stds = [r['lret_wall_std'] for r in results]
    cirq_times = [r['cirq_mean'] for r in results]
    cirq_stds = [r['cirq_std'] for r in results]
    speedups = [r['speedup'] for r in results]
    ranks = [r['final_rank'] for r in results]
    fidelities = [r['fidelity'] if r['fidelity'] is not None else 0.0 for r in results]
    trace_dists = [r['trace_distance'] if r['trace_distance'] is not None else 0.0 for r in results]
    
    # Plot 1: Execution time comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(qubits, lret_times, yerr=lret_stds, marker='o', label='LRET (native)', capsize=5, linewidth=2)
    ax.errorbar(qubits, cirq_times, yerr=cirq_stds, marker='s', label='Cirq', capsize=5, linewidth=2)
    ax.set_xlabel('Number of Qubits', fontsize=12)
    ax.set_ylabel('Execution Time (ms)', fontsize=12)
    ax.set_title(f'Execution Time: LRET vs Cirq (depth={DEPTH}, pure states)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plot1_path = OUTPUT_DIR / "plot_execution_time.png"
    plt.savefig(plot1_path, dpi=150)
    print(f"  Saved: {plot1_path}")
    plt.close()
    
    # Plot 2: Speedup
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(qubits, speedups, marker='o', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('Number of Qubits', fontsize=12)
    ax.set_ylabel('Speedup (Cirq time / LRET time)', fontsize=12)
    ax.set_title(f'LRET Speedup over Cirq (depth={DEPTH}, pure states)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='red', linestyle='--', label='No speedup', alpha=0.5)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plot2_path = OUTPUT_DIR / "plot_speedup.png"
    plt.savefig(plot2_path, dpi=150)
    print(f"  Saved: {plot2_path}")
    plt.close()
    
    # Plot 3: Rank (should all be 1 for pure states)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(qubits, ranks, marker='o', linewidth=2, markersize=8, color='purple')
    ax.set_xlabel('Number of Qubits', fontsize=12)
    ax.set_ylabel('Final Rank', fontsize=12)
    ax.set_title(f'LRET Rank (depth={DEPTH}, pure states - should be 1)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='green', linestyle='--', label='Pure state (rank=1)', alpha=0.5)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plot3_path = OUTPUT_DIR / "plot_rank.png"
    plt.savefig(plot3_path, dpi=150)
    print(f"  Saved: {plot3_path}")
    plt.close()
    
    # Plot 4: Fidelity
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(qubits, fidelities, marker='o', linewidth=2, markersize=8, color='blue')
    ax.set_xlabel('Number of Qubits', fontsize=12)
    ax.set_ylabel('Fidelity (LRET vs Cirq)', fontsize=12)
    ax.set_title(f'State Fidelity: LRET vs Cirq (depth={DEPTH})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='green', linestyle='--', label='Perfect fidelity', alpha=0.5)
    ax.axhline(y=0.99, color='orange', linestyle='--', label='99% threshold', alpha=0.5)
    ax.legend(fontsize=11)
    ax.set_ylim([0.98, 1.002])
    plt.tight_layout()
    plot4_path = OUTPUT_DIR / "plot_fidelity.png"
    plt.savefig(plot4_path, dpi=150)
    print(f"  Saved: {plot4_path}")
    plt.close()
    
    # Plot 5: Combined summary (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top-left: Execution time
    axes[0, 0].errorbar(qubits, lret_times, yerr=lret_stds, marker='o', label='LRET', capsize=5)
    axes[0, 0].errorbar(qubits, cirq_times, yerr=cirq_stds, marker='s', label='Cirq', capsize=5)
    axes[0, 0].set_ylabel('Time (ms)')
    axes[0, 0].set_title('Execution Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Top-right: Speedup
    axes[0, 1].plot(qubits, speedups, marker='o', color='green')
    axes[0, 1].set_ylabel('Speedup (x)')
    axes[0, 1].set_title('LRET Speedup')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=1, color='red', linestyle='--', alpha=0.3)
    
    # Bottom-left: Rank
    axes[1, 0].plot(qubits, ranks, marker='o', color='purple')
    axes[1, 0].set_xlabel('Qubits')
    axes[1, 0].set_ylabel('Rank')
    axes[1, 0].set_title('LRET Final Rank')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=1, color='green', linestyle='--', alpha=0.3)
    
    # Bottom-right: Fidelity
    axes[1, 1].plot(qubits, fidelities, marker='o', color='blue')
    axes[1, 1].set_xlabel('Qubits')
    axes[1, 1].set_ylabel('Fidelity')
    axes[1, 1].set_title('State Fidelity')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0.99, color='orange', linestyle='--', alpha=0.3)
    axes[1, 1].set_ylim([0.98, 1.002])
    
    fig.suptitle(f'Comprehensive Benchmark: LRET vs Cirq (depth={DEPTH}, pure states)', 
                 fontsize=16, y=0.995)
    plt.tight_layout()
    plot5_path = OUTPUT_DIR / "plot_summary.png"
    plt.savefig(plot5_path, dpi=150)
    print(f"  Saved: {plot5_path}")
    plt.close()

# Generate summary report
print("\n" + "="*80)
print("SUMMARY REPORT")
print("="*80)

if results:
    print(f"\nConfiguration:")
    print(f"  Qubits: {QUBITS[0]}-{QUBITS[-1]}")
    print(f"  Depth: {DEPTH}")
    print(f"  Circuit type: Pure states (no noise)")
    print(f"  Trials per config: {N_TRIALS}")
    
    print(f"\nResults Summary:")
    print(f"  {'Qubits':<8} {'LRET(ms)':<12} {'Cirq(ms)':<12} {'Speedup':<10} {'Rank':<6} {'Fidelity'}")
    print(f"  {'-'*70}")
    for r in results:
        fid_str = f"{r['fidelity']:.6f}" if r['fidelity'] is not None else "N/A"
        print(f"  {r['n_qubits']:<8} {r['lret_wall_mean']:<12.2f} {r['cirq_mean']:<12.2f} "
              f"{r['speedup']:<10.1f}x {r['final_rank']:<6} {fid_str}")
    
    avg_speedup = np.mean([r['speedup'] for r in results])
    valid_fidelities = [r['fidelity'] for r in results if r['fidelity'] is not None]
    min_fidelity = np.min(valid_fidelities) if valid_fidelities else 0.0
    max_rank = np.max([r['final_rank'] for r in results])
    
    print(f"\nKey Metrics:")
    print(f"  Average speedup: {avg_speedup:.1f}x")
    if valid_fidelities:
        print(f"  Minimum fidelity: {min_fidelity:.6f}")
        print(f"  All fidelities > 0.99: {'YES' if min_fidelity > 0.99 else 'NO'}")
    print(f"  Maximum rank: {max_rank}")
    print(f"  All ranks = 1 (pure states): {'YES' if max_rank == 1 else 'NO'}")

print(f"\n{'='*80}")
print(f"BENCHMARK COMPLETE!")
print(f"Results saved to: {OUTPUT_DIR}")
print(f"{'='*80}")
