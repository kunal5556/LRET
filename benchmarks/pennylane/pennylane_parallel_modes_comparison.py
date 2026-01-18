#!/usr/bin/env python3
"""
LRET Parallelization Mode Comparison Benchmark
===============================================
Compares LRET execution across ALL parallelization modes:
- SEQUENTIAL: Single-threaded baseline
- ROW: Row-wise parallel operations
- COLUMN: Column-wise parallel operations
- HYBRID: Combined row + batch (default)
- BATCH: Gate batching parallelism

This benchmark does NOT compare against default.mixed.
It focuses on comparing LRET's own parallelization strategies.
"""

import time
import sys
import os
import json
import subprocess
import numpy as np
import psutil
import csv
from datetime import datetime

# Check if we're the launcher or the worker
if len(sys.argv) > 1 and sys.argv[1] == "--worker":
    IS_WORKER = True
else:
    IS_WORKER = False

# =============================================================================
# LAUNCHER MODE - Start benchmark and CPU monitor in separate windows
# =============================================================================
if not IS_WORKER:
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    script_name = os.path.splitext(os.path.basename(script_path))[0]
    
    # Create unique results directory for this run
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(script_dir, '..', '..', 'results', f'{script_name}_{run_id}')
    os.makedirs(log_dir, exist_ok=True)
    
    print("=" * 70)
    print("LAUNCHING BENCHMARK WITH CPU MONITORING")
    print("=" * 70)
    print(f"Script: {os.path.basename(script_path)}")
    print(f"Results directory: {log_dir}")
    print("This will open TWO new PowerShell windows:")
    print("  1. Benchmark execution window")
    print("  2. CPU monitoring window")
    print("=" * 70)
    
    # Start benchmark in new window, pass log_dir as argument
    benchmark_cmd = f'cd "{script_dir}"; python "{script_path}" --worker "{log_dir}"'
    subprocess.Popen(
        ["powershell", "-NoExit", "-Command", benchmark_cmd],
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    
    # Wait a moment for benchmark to start
    time.sleep(2)
    
    # Start CPU monitor in new window with log_dir argument
    monitor_path = os.path.join(script_dir, "monitor_cpu.py")
    monitor_cmd = f'cd "{script_dir}"; python "{monitor_path}" "{log_dir}"'
    subprocess.Popen(
        ["powershell", "-NoExit", "-Command", monitor_cmd],
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    
    print("\n✓ Both windows launched. Check the new PowerShell windows.")
    print(f"✓ Results will be saved to: {log_dir}")
    print("  - Close this window or press Ctrl+C to exit.")
    sys.exit(0)

# =============================================================================
# WORKER MODE - Actual benchmark execution
# =============================================================================

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

import pennylane as qml

# Get log directory from command line argument
LOG_DIR = sys.argv[2] if len(sys.argv) > 2 else None
if LOG_DIR is None:
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results', f'{script_name}_{RUN_ID}')
    os.makedirs(LOG_DIR, exist_ok=True)
else:
    RUN_ID = os.path.basename(LOG_DIR).split('_')[-2] + '_' + os.path.basename(LOG_DIR).split('_')[-1]

# =============================================================================
# CONFIGURATION
# =============================================================================
N_QUBITS = 4
N_EPOCHS = 25  # Fewer epochs for comparison
N_SAMPLES = 25
NOISE_RATE = 0.10
LEARNING_RATE = 0.1
RANDOM_SEED = 42

# All parallelization modes to test
PARALLEL_MODES = ["sequential", "row", "column", "hybrid", "batch"]

# =============================================================================
# SETUP
# =============================================================================
print("=" * 80)
print("LRET PARALLELIZATION MODE COMPARISON BENCHMARK")
print("=" * 80)
print(f"Log directory: {LOG_DIR}")
print(f"PennyLane version: {qml.__version__}")
print("")
print("CONFIGURATION:")
print(f"  Qubits:       {N_QUBITS}")
print(f"  Epochs:       {N_EPOCHS}")
print(f"  Batch size:   {N_SAMPLES}")
print(f"  Noise rate:   {NOISE_RATE:.0%}")
print(f"  Learning:     {LEARNING_RATE}")
print(f"  Seed:         {RANDOM_SEED}")
print(f"  CPU Cores:    {os.cpu_count()}")
print(f"\nModes to test: {', '.join(m.upper() for m in PARALLEL_MODES)}")
print("=" * 80)

# Generate training data
np.random.seed(RANDOM_SEED)
X_train = np.random.randn(N_SAMPLES, N_QUBITS).astype(np.float64)
y_train = np.sign(np.sum(X_train[:, :2], axis=1))
init_params = np.random.randn(2, N_QUBITS, 2) * 0.1

# =============================================================================
# CREATE CIRCUITS FOR EACH MODE
# =============================================================================

def make_circuit(dev):
    @qml.qnode(dev)
    def circuit(params, x):
        # Data encoding with noise
        for i in range(N_QUBITS):
            qml.RY(x[i] * np.pi, wires=i)
            qml.DepolarizingChannel(NOISE_RATE, wires=i)
        
        # Variational layers
        for layer in range(2):
            for i in range(N_QUBITS):
                qml.RY(params[layer, i, 0], wires=i)
                qml.RZ(params[layer, i, 1], wires=i)
            for i in range(N_QUBITS - 1):
                qml.CNOT(wires=[i, i + 1])
        
        return qml.expval(qml.PauliZ(0))
    return circuit

# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train(name, circuit, params, verbose=True):
    """Train QNN with numerical gradients."""
    if verbose:
        print(f"\n{'='*70}")
        print(f"TRAINING: {name}")
        print(f"{'='*70}")
    
    # Create CSV file for this training run
    safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    csv_path = os.path.join(LOG_DIR, f"epochs_{safe_name}.csv")
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'loss', 'time_seconds', 'elapsed_seconds', 'eta_seconds'])
    
    start_time = time.time()
    start_mem = psutil.Process().memory_info().rss / (1024**2)
    
    losses = []
    epoch_times = []
    params = params.copy()
    
    for epoch in range(N_EPOCHS):
        epoch_start = time.time()
        epoch_loss = 0.0
        
        for x, target in zip(X_train, y_train):
            # Forward pass
            pred = float(circuit(params, x))
            loss = (pred - target) ** 2
            epoch_loss += loss
            
            # Numerical gradient (finite differences)
            grad = np.zeros_like(params)
            shift = 0.1
            for idx in np.ndindex(params.shape):
                p_plus = params.copy()
                p_plus[idx] += shift
                p_minus = params.copy()
                p_minus[idx] -= shift
                grad[idx] = (circuit(p_plus, x) - circuit(p_minus, x)) / (2 * shift)
            
            # Update parameters
            params = params - LEARNING_RATE * 2 * (pred - target) * grad
        
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / N_SAMPLES
        losses.append(float(avg_loss))
        epoch_times.append(epoch_time)
        
        elapsed = time.time() - start_time
        eta = (elapsed / (epoch + 1)) * (N_EPOCHS - epoch - 1)
        
        # Save to CSV
        csv_writer.writerow([epoch + 1, f"{avg_loss:.6f}", f"{epoch_time:.2f}", f"{elapsed:.2f}", f"{eta:.2f}"])
        csv_file.flush()
        
        # Progress
        if verbose:
            print(f"  Epoch {epoch+1:3d}/{N_EPOCHS}: loss={avg_loss:.6f}, "
                  f"time={epoch_time:.2f}s, ETA={eta/60:.1f}min")
    
    csv_file.close()
    if verbose:
        print(f"  ✓ Epoch data saved to: {csv_path}")
    
    total_time = time.time() - start_time
    end_mem = psutil.Process().memory_info().rss / (1024**2)
    
    if verbose:
        print(f"  COMPLETED: {total_time:.1f}s total, {total_time/N_EPOCHS:.2f}s/epoch")
    
    return {
        "total_time_seconds": total_time,
        "avg_epoch_time": total_time / N_EPOCHS,
        "final_loss": losses[-1],
        "memory_delta_mb": end_mem - start_mem,
        "losses": losses,
        "epoch_times": epoch_times,
        "csv_file": csv_path,
    }

# =============================================================================
# RUN BENCHMARKS FOR ALL MODES
# =============================================================================

print("\n" + "=" * 80)
print("RUNNING BENCHMARKS FOR ALL PARALLELIZATION MODES")
print("=" * 80)

results = {}

for mode in PARALLEL_MODES:
    print(f"\n{'='*80}")
    print(f"TESTING MODE: {mode.upper()}")
    print(f"{'='*80}")
    
    # Create device with specific mode
    dev = qml.device('qlret.mixed', wires=N_QUBITS, epsilon=1e-4,
                     num_threads=0, parallel_mode=mode)
    
    print(f"Device: {dev.name}")
    print(f"Mode: {mode.upper()}, Threads: auto (all cores)")
    
    # Create circuit
    circuit = make_circuit(dev)
    
    # Warmup
    print("\nWarmup...")
    _ = circuit(init_params, X_train[0])
    
    # Train
    result = train(f"LRET ({mode.upper()})", circuit, init_params.copy())
    results[mode] = result

# =============================================================================
# COMPARISON SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("PARALLELIZATION MODE COMPARISON SUMMARY")
print("=" * 80)

# Find baseline (sequential)
sequential_time = results["sequential"]["total_time_seconds"]

print(f"\n{'Mode':<12} {'Total Time':>12} {'Time/Epoch':>12} {'Speedup':>10} {'Final Loss':>12}")
print("-" * 60)

# Sort by time (fastest first)
sorted_modes = sorted(results.keys(), key=lambda m: results[m]["total_time_seconds"])

for mode in sorted_modes:
    r = results[mode]
    speedup = sequential_time / r["total_time_seconds"]
    print(f"{mode.upper():<12} {r['total_time_seconds']:>10.1f}s {r['avg_epoch_time']:>10.2f}s {speedup:>9.2f}x {r['final_loss']:>11.6f}")

# Best mode
best_mode = sorted_modes[0]
best_speedup = sequential_time / results[best_mode]["total_time_seconds"]
print(f"\n✓ FASTEST MODE: {best_mode.upper()} ({best_speedup:.2f}x faster than SEQUENTIAL)")

# CPU utilization analysis
print("\n" + "=" * 80)
print("ANALYSIS: Why Different Modes Have Different Performance")
print("=" * 80)
print("""
- SEQUENTIAL: Single-threaded, no OpenMP overhead, good for small problems
- ROW: Parallelizes across density matrix rows, best for large rank
- COLUMN: Parallelizes across rank columns, best when rank > cores
- HYBRID: Combines row + batch strategies, generally best overall
- BATCH: Groups gates for batch application, reduces OpenMP sync overhead

For {N_QUBITS} qubits (dim={2**N_QUBITS}x{2**N_QUBITS} density matrix):
- Problem is relatively SMALL for OpenMP parallelization
- OpenMP overhead (~10-50μs per sync) can exceed computation time
- Memory access patterns matter more than raw parallelism
""")

# =============================================================================
# SAVE RESULTS
# =============================================================================

output_data = {
    "run_id": RUN_ID,
    "log_dir": LOG_DIR,
    "config": {
        "n_qubits": N_QUBITS,
        "n_epochs": N_EPOCHS,
        "n_samples": N_SAMPLES,
        "noise_rate": NOISE_RATE,
        "cpu_cores": os.cpu_count(),
    },
    "results": {mode: {
        "total_time_seconds": r["total_time_seconds"],
        "avg_epoch_time": r["avg_epoch_time"],
        "final_loss": r["final_loss"],
        "speedup_vs_sequential": sequential_time / r["total_time_seconds"],
        "csv_file": r.get("csv_file", ""),
    } for mode, r in results.items()},
    "best_mode": best_mode,
    "best_speedup": best_speedup,
}

output_file = os.path.join(LOG_DIR, "results.json")
with open(output_file, "w") as f:
    json.dump(output_data, f, indent=2, default=float)

print(f"\nResults saved to: {output_file}")
print(f"All CSV files saved to: {LOG_DIR}")
print("=" * 80)
