#!/usr/bin/env python3
"""
Comprehensive Comparison: ALL LRET Modes + default.mixed
=========================================================
Configuration: 8 qubits, 100 epochs, 100 samples, 12% noise

Tests all parallelization modes:
- LRET: sequential, row, column, hybrid, batch
- Baseline: default.mixed (with OOM handling)

Includes integrated CPU monitoring.
"""

import time
import sys
import os
import json
import subprocess
import platform
import numpy as np
import psutil
from datetime import datetime

# Check if we're the launcher or the worker
if len(sys.argv) > 1 and sys.argv[1] == "--worker":
    IS_WORKER = True
else:
    IS_WORKER = False

# =============================================================================
# LAUNCHER MODE
# =============================================================================
if not IS_WORKER:
    from launcher_utils import launch_in_new_terminal, get_terminal_name, format_command_for_platform
    
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    script_name = os.path.splitext(os.path.basename(script_path))[0]
    
    # Create unique results directory for this run
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(script_dir, '..', '..', 'results', f'{script_name}_{run_id}')
    os.makedirs(log_dir, exist_ok=True)
    
    terminal_name = get_terminal_name()
    
    print("=" * 70)
    print("LAUNCHING BENCHMARK WITH CPU MONITORING")
    print("=" * 70)
    print(f"Platform: {platform.system()}")
    print(f"Script: {os.path.basename(script_path)}")
    print(f"Results directory: {log_dir}")
    print(f"This will open TWO new {terminal_name} windows:")
    print("  1. Benchmark execution window")
    print("  2. CPU monitoring window")
    print("=" * 70)
    
    # Start benchmark in new window with log_dir argument
    benchmark_cmd = format_command_for_platform(script_path, "--worker", log_dir)
    launch_in_new_terminal(benchmark_cmd, "LRET Benchmark")
    
    time.sleep(2)
    
    # Start CPU monitor in new window with log_dir argument
    monitor_path = os.path.join(script_dir, "monitor_cpu.py")
    monitor_cmd = format_command_for_platform(monitor_path, log_dir)
    launch_in_new_terminal(monitor_cmd, "CPU Monitor")
    
    print(f"\n✓ Both windows launched. Check the new {terminal_name} windows.")
    print(f"✓ Results will be saved to: {log_dir}")
    sys.exit(0)

# =============================================================================
# WORKER MODE
# =============================================================================
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
import pennylane as qml
import csv

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
N_QUBITS = 8
N_EPOCHS = 100
N_SAMPLES = 100
NOISE_RATE = 0.12
LEARNING_RATE = 0.1
RANDOM_SEED = 42

LRET_MODES = ["sequential", "row", "column", "hybrid", "batch"]

# =============================================================================
# SETUP
# =============================================================================
print("=" * 80)
print("COMPREHENSIVE COMPARISON: ALL LRET MODES + default.mixed")
print("=" * 80)
print(f"Log directory: {LOG_DIR}")
print(f"PennyLane version: {qml.__version__}")
print(f"Process PID: {os.getpid()}")
print("")
print("CONFIGURATION:")
print(f"  Qubits:       {N_QUBITS}")
print(f"  Epochs:       {N_EPOCHS}")
print(f"  Batch size:   {N_SAMPLES}")
print(f"  Noise rate:   {NOISE_RATE:.0%}")
print(f"  Learning:     {LEARNING_RATE}")
print(f"  Seed:         {RANDOM_SEED}")
print(f"  CPU Cores:    {os.cpu_count()}")
print(f"\n⚠️  WARNING: 8 qubits - default.mixed may run out of memory (OOM)")
print(f"\nModes to test: {', '.join(m.upper() for m in LRET_MODES)} + default.mixed")
print("=" * 80)

np.random.seed(RANDOM_SEED)
X_train = np.random.randn(N_SAMPLES, N_QUBITS).astype(np.float64)
y_train = np.sign(np.sum(X_train[:, :2], axis=1))
init_params = np.random.randn(2, N_QUBITS, 2) * 0.1

# =============================================================================
# CIRCUIT FACTORY
# =============================================================================
def make_circuit(dev):
    @qml.qnode(dev)
    def circuit(params, x):
        for i in range(N_QUBITS):
            qml.RY(x[i] * np.pi, wires=i)
            qml.DepolarizingChannel(NOISE_RATE, wires=i)
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
def train(name, circuit, params, max_epochs=None):
    if max_epochs is None:
        max_epochs = N_EPOCHS
    
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
    
    try:
        for epoch in range(max_epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            
            for x, target in zip(X_train, y_train):
                pred = float(circuit(params, x))
                loss = (pred - target) ** 2
                epoch_loss += loss
                
                grad = np.zeros_like(params)
                shift = 0.1
                for idx in np.ndindex(params.shape):
                    p_plus = params.copy()
                    p_plus[idx] += shift
                    p_minus = params.copy()
                    p_minus[idx] -= shift
                    grad[idx] = (circuit(p_plus, x) - circuit(p_minus, x)) / (2 * shift)
                
                params = params - LEARNING_RATE * 2 * (pred - target) * grad
            
            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / N_SAMPLES
            losses.append(float(avg_loss))
            epoch_times.append(epoch_time)
            
            elapsed = time.time() - start_time
            eta = (elapsed / (epoch + 1)) * (max_epochs - epoch - 1)
            
            # Save to CSV
            csv_writer.writerow([epoch + 1, f"{avg_loss:.6f}", f"{epoch_time:.2f}", f"{elapsed:.2f}", f"{eta:.2f}"])
            csv_file.flush()
            
            print(f"  Epoch {epoch+1:3d}/{max_epochs}: loss={avg_loss:.6f}, "
                  f"time={epoch_time:.2f}s, ETA={eta/60:.1f}min")
        
        csv_file.close()
        print(f"  ✓ Epoch data saved to: {csv_path}")
        
        total_time = time.time() - start_time
        end_mem = psutil.Process().memory_info().rss / (1024**2)
        print(f"  COMPLETED: {total_time:.1f}s total, {total_time/len(losses):.2f}s/epoch")
        
        return {
            "status": "completed",
            "total_time_seconds": total_time,
            "avg_epoch_time": total_time / len(losses),
            "final_loss": losses[-1],
            "memory_delta_mb": end_mem - start_mem,
            "epochs_completed": len(losses),
            "csv_file": csv_path,
        }
    
    except MemoryError:
        csv_file.close()
        print(f"  ❌ OUT OF MEMORY after {len(losses)} epochs!")
        return {
            "status": "OOM",
            "total_time_seconds": time.time() - start_time,
            "avg_epoch_time": float('inf'),
            "final_loss": losses[-1] if losses else float('inf'),
            "memory_delta_mb": float('inf'),
            "epochs_completed": len(losses),
            "csv_file": csv_path,
        }

# =============================================================================
# RUN ALL BENCHMARKS
# =============================================================================
print("\n" + "=" * 80)
print("RUNNING BENCHMARKS")
print("=" * 80)

results = {}

# Test all LRET modes
for mode in LRET_MODES:
    print(f"\n>>> Testing LRET ({mode.upper()}) <<<")
    dev = qml.device('qlret.mixed', wires=N_QUBITS, epsilon=1e-4,
                     num_threads=0, parallel_mode=mode)
    circuit = make_circuit(dev)
    _ = circuit(init_params, X_train[0])
    results[f"lret_{mode}"] = train(f"LRET ({mode.upper()})", circuit, init_params.copy())

# Test default.mixed (may OOM)
print(f"\n>>> Testing default.mixed (may OOM at 8 qubits) <<<")
try:
    dev_baseline = qml.device('default.mixed', wires=N_QUBITS)
    circuit_baseline = make_circuit(dev_baseline)
    _ = circuit_baseline(init_params, X_train[0])
    # Limit to 10 epochs for baseline at 8 qubits (very slow)
    results["default_mixed"] = train("default.mixed", circuit_baseline, init_params.copy(), max_epochs=10)
    if results["default_mixed"]["status"] == "completed":
        # Extrapolate to full epochs
        results["default_mixed"]["extrapolated_total_time"] = (
            results["default_mixed"]["avg_epoch_time"] * N_EPOCHS
        )
        print(f"  (Extrapolated full {N_EPOCHS} epochs: {results['default_mixed']['extrapolated_total_time']:.1f}s)")
except MemoryError:
    print("  ❌ default.mixed OUT OF MEMORY during setup!")
    results["default_mixed"] = {"status": "OOM", "total_time_seconds": float('inf')}

# =============================================================================
# RESULTS SUMMARY
# =============================================================================
# Use best LRET time as reference if baseline OOM
if results["default_mixed"]["status"] == "OOM":
    baseline_time = float('inf')
else:
    baseline_time = results["default_mixed"].get("extrapolated_total_time", 
                    results["default_mixed"]["total_time_seconds"])

print("\n" + "=" * 80)
print("COMPREHENSIVE COMPARISON RESULTS")
print("=" * 80)
print(f"\n{'Device':<20} {'Status':<10} {'Total Time':>12} {'Time/Epoch':>12} {'vs Baseline':>12}")
print("-" * 72)

sorted_results = sorted(results.items(), 
                       key=lambda x: x[1]["total_time_seconds"] if x[1]["status"] != "OOM" else float('inf'))

for name, r in sorted_results:
    status = r["status"] if "status" in r else "completed"
    if status == "OOM":
        print(f"{name.upper():<20} {'OOM':<10} {'N/A':>12} {'N/A':>12} {'N/A':>12}")
    else:
        speedup = baseline_time / r["total_time_seconds"] if baseline_time != float('inf') else float('inf')
        speedup_str = f"{speedup:.2f}x" if speedup != float('inf') else "∞"
        print(f"{name.upper():<20} {'OK':<10} {r['total_time_seconds']:>10.1f}s "
              f"{r['avg_epoch_time']:>10.2f}s {speedup_str:>12}")

# Find best LRET mode
lret_results = [(n, r) for n, r in sorted_results if n.startswith("lret_") and r.get("status") != "OOM"]
if lret_results:
    best_name, best_result = lret_results[0]
    print(f"\n✓ FASTEST LRET MODE: {best_name.upper()}")
    if results["default_mixed"]["status"] == "OOM":
        print(f"✓ default.mixed: OUT OF MEMORY (LRET wins by default!)")

# =============================================================================
# SAVE RESULTS
# =============================================================================
output_data = {
    "run_id": RUN_ID,
    "log_dir": LOG_DIR,
    "config": {
        "n_qubits": N_QUBITS, "n_epochs": N_EPOCHS, "n_samples": N_SAMPLES,
        "noise_rate": NOISE_RATE, "cpu_cores": os.cpu_count(),
    },
    "results": {name: {k: v for k, v in r.items() if k != "losses" and k != "epoch_times"} 
                for name, r in results.items()},
}

output_file = os.path.join(LOG_DIR, "results.json")
with open(output_file, "w") as f:
    json.dump(output_data, f, indent=2, default=lambda x: str(x) if x == float('inf') else x)

print(f"\nResults saved to: {output_file}")
print(f"All CSV files saved to: {LOG_DIR}")
print("=" * 80)
