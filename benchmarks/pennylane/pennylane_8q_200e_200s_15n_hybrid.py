#!/usr/bin/env python3
"""
LRET vs default.mixed Training Benchmark - HYBRID Parallelization (8 Qubits Heavy)
===================================================================================
Configuration:
  - Qubits: 8
  - Epochs: 200
  - Batch size: 200
  - Noise: 15% DepolarizingChannel
  - Parallelization: HYBRID mode (recommended)

This benchmark runs actual QNN training with gradient computation.
Expected runtime: LRET ~30-60 minutes, default.mixed likely OOM

WARNING: Heavy benchmark! 8-qubit density matrices with large batch size.
default.mixed will almost certainly run out of memory.
Now includes integrated CPU monitoring - launches in separate windows.
"""

import time
import sys
import os
import json
import subprocess
import platform

# =============================================================================
# LAUNCHER MODE
# =============================================================================
if len(sys.argv) <= 1 or sys.argv[1] != "--worker":
    from datetime import datetime
    from launcher_utils import launch_in_new_terminal, get_terminal_name, format_command_for_platform
    
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    script_name = os.path.splitext(os.path.basename(script_path))[0]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(script_dir, '..', '..', 'results', f'{script_name}_{run_id}')
    os.makedirs(log_dir, exist_ok=True)
    
    terminal_name = get_terminal_name()
    
    print("=" * 70)
    print("LAUNCHING BENCHMARK WITH CPU MONITORING")
    print("=" * 70)
    print(f"Platform: {platform.system()}")
    print(f"Script: {os.path.basename(script_path)}")
    print("Mode: HYBRID parallelization (8 qubits HEAVY)")
    print(f"Results directory: {log_dir}")
    print("⚠️  WARNING: HEAVY benchmark - default.mixed WILL likely OOM!")
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
    sys.exit(0)

# =============================================================================
# WORKER MODE
# =============================================================================
import csv
import numpy as np
import psutil
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

import pennylane as qml

# =============================================================================
# CONFIGURATION
# =============================================================================
N_QUBITS = 8
N_EPOCHS = 200
N_SAMPLES = 200
NOISE_RATE = 0.15
LEARNING_RATE = 0.1
RANDOM_SEED = 42

# Parallelization settings for LRET
PARALLEL_MODE = "hybrid"
NUM_THREADS = 0  # 0 = auto (all cores)

# =============================================================================
# SETUP
# =============================================================================
LOG_DIR = sys.argv[2] if len(sys.argv) > 2 else None
if LOG_DIR is None:
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results', f'{script_name}_{RUN_ID}')
    os.makedirs(LOG_DIR, exist_ok=True)
else:
    RUN_ID = os.path.basename(LOG_DIR).split('_')[-2] + '_' + os.path.basename(LOG_DIR).split('_')[-1]

print("=" * 70)
print("LRET vs default.mixed TRAINING BENCHMARK (8 QUBITS HEAVY)")
print("=" * 70)
print(f"Parallelization Mode: {PARALLEL_MODE.upper()}")
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
print(f"  Threads:      {NUM_THREADS} (0=auto)")
print("=" * 70)

# Generate training data
np.random.seed(RANDOM_SEED)
X_train = np.random.randn(N_SAMPLES, N_QUBITS).astype(np.float64)
y_train = np.sign(np.sum(X_train[:, :2], axis=1))
init_params = np.random.randn(2, N_QUBITS, 2) * 0.1
n_params = init_params.size

print(f"\nCircuit complexity:")
print(f"  Parameters:   {n_params}")
print(f"  Hilbert dim:  2^{N_QUBITS} = {2**N_QUBITS}")
print(f"  Circuit evals per epoch: {N_SAMPLES} × (1 + 2×{n_params}) = {N_SAMPLES * (1 + 2*n_params)}")

# =============================================================================
# CREATE DEVICES
# =============================================================================
print("\nCreating devices...")

dev_lret = qml.device('qlret.mixed', wires=N_QUBITS, epsilon=1e-4,
                      num_threads=NUM_THREADS, parallel_mode=PARALLEL_MODE)
print(f"  LRET: {dev_lret.name} (mode={PARALLEL_MODE})")

try:
    dev_baseline = qml.device('default.mixed', wires=N_QUBITS)
    print(f"  Baseline: {dev_baseline.name}")
    run_baseline = True
except Exception as e:
    print(f"  Baseline: FAILED - {e}")
    dev_baseline = None
    run_baseline = False

# =============================================================================
# DEFINE CIRCUITS
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

circuit_lret = make_circuit(dev_lret)
circuit_baseline = make_circuit(dev_baseline) if run_baseline else None

# =============================================================================
# TRAINING FUNCTION
# =============================================================================
def train(name, circuit, params, max_epochs=N_EPOCHS):
    """Train QNN with numerical gradients."""
    print(f"\n{'='*70}")
    print(f"TRAINING: {name}")
    print(f"{'='*70}")
    
    # Setup CSV logging
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
            
            # Progress
            elapsed = time.time() - start_time
            eta = (elapsed / (epoch + 1)) * (max_epochs - epoch - 1)
            csv_writer.writerow([epoch + 1, f"{avg_loss:.6f}", f"{epoch_time:.2f}", f"{elapsed:.2f}", f"{eta:.2f}"])
            csv_file.flush()
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{max_epochs}: loss={avg_loss:.6f}, "
                      f"time={epoch_time:.2f}s, ETA={eta/60:.1f}min")
        
        status = "completed"
    except MemoryError:
        print(f"  *** OUT OF MEMORY at epoch {len(losses)+1} ***")
        status = "oom"
    except Exception as e:
        print(f"  *** ERROR at epoch {len(losses)+1}: {e} ***")
        status = "error"
    
    csv_file.close()
    print(f"  ✓ Epoch data saved to: {csv_path}")
    
    total_time = time.time() - start_time
    end_mem = psutil.Process().memory_info().rss / (1024**2)
    
    print(f"  {status.upper()}: {total_time:.1f}s total")
    
    return {
        "status": status,
        "epochs_completed": len(losses),
        "total_time_seconds": total_time,
        "avg_epoch_time": total_time / max(len(losses), 1),
        "final_loss": losses[-1] if losses else None,
        "memory_delta_mb": end_mem - start_mem,
        "losses": losses,
        "epoch_times": epoch_times,
        "csv_file": csv_path,
    }

# =============================================================================
# RUN BENCHMARKS
# =============================================================================
print("\n" + "=" * 70)
print("RUNNING BENCHMARKS")
print("=" * 70)

# Warmup
print("\nWarmup runs...")
_ = circuit_lret(init_params, X_train[0])
if run_baseline:
    try:
        _ = circuit_baseline(init_params, X_train[0])
    except Exception as e:
        print(f"  Baseline warmup failed: {e}")
        run_baseline = False

# Train LRET
lret_result = train(f"LRET ({PARALLEL_MODE.upper()})", circuit_lret, init_params.copy())

# Train baseline (if available)
if run_baseline:
    baseline_result = train("default.mixed", circuit_baseline, init_params.copy())
else:
    baseline_result = {
        "status": "skipped",
        "epochs_completed": 0,
        "total_time_seconds": 0,
        "avg_epoch_time": 0,
        "final_loss": None,
        "memory_delta_mb": 0,
    }

# =============================================================================
# RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

print(f"\nLRET ({PARALLEL_MODE.upper()} mode):")
print(f"  Status:        {lret_result['status']}")
print(f"  Epochs:        {lret_result['epochs_completed']}/{N_EPOCHS}")
print(f"  Total time:    {lret_result['total_time_seconds']:.1f} seconds")
print(f"  Time/epoch:    {lret_result['avg_epoch_time']:.2f} seconds")
if lret_result['final_loss']:
    print(f"  Final loss:    {lret_result['final_loss']:.6f}")
print(f"  Memory delta:  {lret_result['memory_delta_mb']:.1f} MB")

print(f"\ndefault.mixed:")
print(f"  Status:        {baseline_result['status']}")
if baseline_result['status'] != 'skipped':
    print(f"  Epochs:        {baseline_result['epochs_completed']}/{N_EPOCHS}")
    print(f"  Total time:    {baseline_result['total_time_seconds']:.1f} seconds")
    print(f"  Time/epoch:    {baseline_result['avg_epoch_time']:.2f} seconds")
    if baseline_result['final_loss']:
        print(f"  Final loss:    {baseline_result['final_loss']:.6f}")
    print(f"  Memory delta:  {baseline_result['memory_delta_mb']:.1f} MB")

if baseline_result['total_time_seconds'] > 0 and lret_result['total_time_seconds'] > 0:
    speedup = baseline_result["total_time_seconds"] / lret_result["total_time_seconds"]
    print(f"\nComparison:")
    print(f"  Speedup:       {speedup:.2f}x {'(LRET faster)' if speedup > 1 else '(baseline faster)'}")
    if lret_result['final_loss'] and baseline_result['final_loss']:
        loss_diff = abs(lret_result["final_loss"] - baseline_result["final_loss"])
        print(f"  Loss diff:     {loss_diff:.6f}")
else:
    speedup = float('inf') if baseline_result['status'] in ('oom', 'error', 'skipped') else 0
    print(f"\nComparison:")
    print(f"  LRET advantage: ∞ (baseline {'failed' if baseline_result['status'] != 'skipped' else 'skipped'})")

# Save results
results = {
    "run_id": RUN_ID,
    "log_dir": LOG_DIR,
    "config": {
        "n_qubits": N_QUBITS,
        "n_epochs": N_EPOCHS,
        "n_samples": N_SAMPLES,
        "noise_rate": NOISE_RATE,
        "parallel_mode": PARALLEL_MODE,
        "num_threads": NUM_THREADS,
    },
    "lret": lret_result,
    "baseline": baseline_result,
    "comparison": {
        "speedup": speedup if speedup != float('inf') else "infinity",
    }
}

output_file = os.path.join(LOG_DIR, "results.json")
with open(output_file, "w") as f:
    json.dump(results, f, indent=2, default=float)

print(f"\nResults saved to: {output_file}")
print(f"CSV files saved to: {LOG_DIR}")
print("=" * 70)
