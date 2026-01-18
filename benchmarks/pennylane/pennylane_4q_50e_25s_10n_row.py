#!/usr/bin/env python3
"""
LRET vs default.mixed Training Benchmark - ROW Parallelization
===============================================================
Configuration:
  - Qubits: 4
  - Epochs: 50
  - Batch size: 25
  - Noise: 10% DepolarizingChannel
  - Parallelization: ROW mode

This benchmark runs actual QNN training with gradient computation.
Expected runtime: LRET ~1-2 minutes, default.mixed ~3-5 minutes

Now includes integrated CPU monitoring - launches in separate windows.
"""

import time
import sys
import os
import json
import subprocess

# =============================================================================
# LAUNCHER MODE - Start benchmark and CPU monitor in separate windows
# =============================================================================
if len(sys.argv) <= 1 or sys.argv[1] != "--worker":
    from datetime import datetime
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    script_name = os.path.splitext(os.path.basename(script_path))[0]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(script_dir, '..', '..', 'results', f'{script_name}_{run_id}')
    os.makedirs(log_dir, exist_ok=True)
    
    print("=" * 70)
    print("LAUNCHING BENCHMARK WITH CPU MONITORING")
    print("=" * 70)
    print(f"Script: {os.path.basename(script_path)}")
    print("Mode: ROW parallelization")
    print(f"Results directory: {log_dir}")
    print("This will open TWO new PowerShell windows:")
    print("  1. Benchmark execution window")
    print("  2. CPU monitoring window")
    print("=" * 70)
    
    benchmark_cmd = f'cd "{script_dir}"; python "{script_path}" --worker "{log_dir}"'
    subprocess.Popen(
        ["powershell", "-NoExit", "-Command", benchmark_cmd],
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    
    time.sleep(2)
    
    monitor_path = os.path.join(script_dir, "monitor_cpu.py")
    monitor_cmd = f'cd "{script_dir}"; python "{monitor_path}" "{log_dir}"'
    subprocess.Popen(
        ["powershell", "-NoExit", "-Command", monitor_cmd],
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    
    print("\n✓ Both windows launched. Check the new PowerShell windows.")
    sys.exit(0)

# =============================================================================
# WORKER MODE - Actual benchmark execution
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
N_QUBITS = 4
N_EPOCHS = 50
N_SAMPLES = 25
NOISE_RATE = 0.10
LEARNING_RATE = 0.1
RANDOM_SEED = 42

# Parallelization settings for LRET
PARALLEL_MODE = "row"
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
print("LRET vs default.mixed TRAINING BENCHMARK")
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

# =============================================================================
# CREATE DEVICES
# =============================================================================
print("\nCreating devices...")

dev_lret = qml.device('qlret.mixed', wires=N_QUBITS, epsilon=1e-4,
                      num_threads=NUM_THREADS, parallel_mode=PARALLEL_MODE)
dev_baseline = qml.device('default.mixed', wires=N_QUBITS)

print(f"  LRET: {dev_lret.name} (mode={PARALLEL_MODE})")
print(f"  Baseline: {dev_baseline.name}")

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
circuit_baseline = make_circuit(dev_baseline)

# =============================================================================
# TRAINING FUNCTION
# =============================================================================
def train(name, circuit, params):
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
        
        # Progress
        elapsed = time.time() - start_time
        eta = (elapsed / (epoch + 1)) * (N_EPOCHS - epoch - 1)
        csv_writer.writerow([epoch + 1, f"{avg_loss:.6f}", f"{epoch_time:.2f}", f"{elapsed:.2f}", f"{eta:.2f}"])
        csv_file.flush()
        print(f"  Epoch {epoch+1:3d}/{N_EPOCHS}: loss={avg_loss:.6f}, "
              f"time={epoch_time:.2f}s, ETA={eta/60:.1f}min")
    
    csv_file.close()
    print(f"  ✓ Epoch data saved to: {csv_path}")
    
    total_time = time.time() - start_time
    end_mem = psutil.Process().memory_info().rss / (1024**2)
    
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
# RUN BENCHMARKS
# =============================================================================
print("\n" + "=" * 70)
print("RUNNING BENCHMARKS")
print("=" * 70)

# Warmup
print("\nWarmup runs...")
_ = circuit_lret(init_params, X_train[0])
_ = circuit_baseline(init_params, X_train[0])

# Train LRET
lret_result = train(f"LRET ({PARALLEL_MODE.upper()})", circuit_lret, init_params.copy())

# Train baseline
baseline_result = train("default.mixed", circuit_baseline, init_params.copy())

# =============================================================================
# RESULTS
# =============================================================================
speedup = baseline_result["total_time_seconds"] / lret_result["total_time_seconds"]
loss_diff = abs(lret_result["final_loss"] - baseline_result["final_loss"])

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"\nLRET ({PARALLEL_MODE.upper()} mode):")
print(f"  Total time:    {lret_result['total_time_seconds']:.1f} seconds")
print(f"  Time/epoch:    {lret_result['avg_epoch_time']:.2f} seconds")
print(f"  Final loss:    {lret_result['final_loss']:.6f}")
print(f"  Memory delta:  {lret_result['memory_delta_mb']:.1f} MB")

print(f"\ndefault.mixed:")
print(f"  Total time:    {baseline_result['total_time_seconds']:.1f} seconds")
print(f"  Time/epoch:    {baseline_result['avg_epoch_time']:.2f} seconds")
print(f"  Final loss:    {baseline_result['final_loss']:.6f}")
print(f"  Memory delta:  {baseline_result['memory_delta_mb']:.1f} MB")

print(f"\nComparison:")
print(f"  Speedup:       {speedup:.2f}x {'(LRET faster)' if speedup > 1 else '(baseline faster)'}")
print(f"  Loss diff:     {loss_diff:.6f}")
print(f"  Results match: {'YES ✓' if loss_diff < 0.01 else 'NO ✗'}")

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
        "speedup": speedup,
        "loss_diff": loss_diff,
    }
}

output_file = os.path.join(LOG_DIR, "results.json")
with open(output_file, "w") as f:
    json.dump(results, f, indent=2, default=float)

print(f"\nResults saved to: {output_file}")
print(f"CSV files saved to: {LOG_DIR}")
print("=" * 70)
