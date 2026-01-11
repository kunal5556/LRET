#!/usr/bin/env python3
"""
LRET vs default.mixed Benchmark (Fast Version)
===============================================
Configuration:
  - Qubits: 4
  - Batch size: 25
  - Epochs: 100
  - Noise: 10% DepolarizingChannel

Estimated time: LRET ~2.7 hours, default.mixed ~26 hours
Run: python benchmarks/benchmark_4q_25s_100e_10n.py
"""

import time
import sys
import os
import json
import numpy as np
import pennylane as qml
import psutil
import traceback
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================
N_QUBITS = 4
N_EPOCHS = 100
N_SAMPLES = 25       # batch size
NOISE_RATE = 0.10    # 10% depolarizing noise
LEARNING_RATE = 0.1
RANDOM_SEED = 42

# =============================================================================
# SETUP LOGGING
# =============================================================================
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = f"D:/LRET/results/benchmark_{RUN_ID}"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = f"{LOG_DIR}/benchmark.log"
PROGRESS_FILE = f"{LOG_DIR}/progress.log"
RESULTS_FILE = f"{LOG_DIR}/results.json"
LRET_EPOCHS_FILE = f"{LOG_DIR}/lret_epochs.csv"
BASELINE_EPOCHS_FILE = f"{LOG_DIR}/baseline_epochs.csv"

def log(msg, progress=False):
    """Log with timestamp"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    if progress:
        with open(PROGRESS_FILE, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

def save_results(data):
    """Save results to JSON"""
    with open(RESULTS_FILE, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def save_epoch_data(filename, epoch, loss, time_s):
    """Append epoch data to CSV"""
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write("epoch,loss,time_seconds\n")
    with open(filename, 'a') as f:
        f.write(f"{epoch},{loss:.6f},{time_s:.2f}\n")

# =============================================================================
# MAIN BENCHMARK
# =============================================================================
def main():
    log("=" * 70)
    log("LRET vs default.mixed BENCHMARK (Fast Version)")
    log("=" * 70)
    log(f"Run ID: {RUN_ID}")
    log(f"Log directory: {LOG_DIR}")
    log(f"PennyLane version: {qml.__version__}")
    log(f"Python version: {sys.version.split()[0]}")
    log("")
    log("CONFIGURATION:")
    log(f"  Qubits:     {N_QUBITS}")
    log(f"  Epochs:     {N_EPOCHS}")
    log(f"  Batch size: {N_SAMPLES}")
    log(f"  Noise rate: {NOISE_RATE:.0%}")
    log(f"  Learning:   {LEARNING_RATE}")
    log(f"  Seed:       {RANDOM_SEED}")
    log("")
    
    # Initialize results
    results = {
        "run_id": RUN_ID,
        "config": {
            "n_qubits": N_QUBITS,
            "n_epochs": N_EPOCHS,
            "n_samples": N_SAMPLES,
            "noise_rate": NOISE_RATE,
            "learning_rate": LEARNING_RATE,
            "seed": RANDOM_SEED,
        },
        "pennylane_version": qml.__version__,
        "status": "initializing",
        "lret": {},
        "baseline": {},
    }
    save_results(results)
    
    # Generate training data (same for both devices)
    np.random.seed(RANDOM_SEED)
    X_train = np.random.randn(N_SAMPLES, N_QUBITS).astype(np.float64)
    y_train = np.sign(np.sum(X_train[:, :2], axis=1))  # Binary classification
    init_params = np.random.randn(2, N_QUBITS, 2) * 0.1
    
    n_params = init_params.size
    log(f"Training data: {N_SAMPLES} samples, {n_params} parameters")
    log("")
    
    # =========================================================================
    # CREATE DEVICES
    # =========================================================================
    log("Creating devices...")
    
    try:
        dev_lret = qml.device('qlret.mixed', wires=N_QUBITS, epsilon=1e-4)
        log(f"  LRET: {dev_lret.name} [OK]")
    except Exception as e:
        log(f"  LRET: FAILED - {e}")
        dev_lret = None
    
    try:
        dev_baseline = qml.device('default.mixed', wires=N_QUBITS)
        log(f"  default.mixed: {dev_baseline.name} [OK]")
    except Exception as e:
        log(f"  default.mixed: FAILED - {e}")
        dev_baseline = None
    
    log("")
    
    # =========================================================================
    # CIRCUIT DEFINITION
    # =========================================================================
    def make_circuit(dev):
        @qml.qnode(dev)
        def circuit(params, x):
            # Data encoding with noise
            for i in range(N_QUBITS):
                qml.RY(x[i] * np.pi, wires=i)
                qml.DepolarizingChannel(NOISE_RATE, wires=i)
            
            # Variational layers (2 layers)
            for layer in range(2):
                for i in range(N_QUBITS):
                    qml.RY(params[layer, i, 0], wires=i)
                    qml.RZ(params[layer, i, 1], wires=i)
                for i in range(N_QUBITS - 1):
                    qml.CNOT(wires=[i, i + 1])
            
            return qml.expval(qml.PauliZ(0))
        return circuit
    
    # =========================================================================
    # TRAINING FUNCTION
    # =========================================================================
    def train(name, circuit, params, csv_file):
        log(f"{'='*70}", progress=True)
        log(f"TRAINING: {name}", progress=True)
        log(f"{'='*70}", progress=True)
        
        start_time = time.time()
        start_mem = psutil.Process().memory_info().rss / (1024**2)
        
        losses = []
        epoch_times = []
        
        for epoch in range(N_EPOCHS):
            epoch_start = time.time()
            epoch_loss = 0.0
            
            for x, target in zip(X_train, y_train):
                # Forward pass
                pred = float(circuit(params, x))
                loss = (pred - target) ** 2
                epoch_loss += loss
                
                # Numerical gradient
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
            
            # Save epoch data
            save_epoch_data(csv_file, epoch + 1, avg_loss, epoch_time)
            
            # Log progress
            elapsed = time.time() - start_time
            eta = (elapsed / (epoch + 1)) * (N_EPOCHS - epoch - 1)
            log(f"  [{name}] Epoch {epoch+1:3d}/{N_EPOCHS}: loss={avg_loss:.6f}, "
                f"time={epoch_time:.1f}s, elapsed={elapsed/60:.1f}min, ETA={eta/60:.1f}min", 
                progress=True)
            
            # Update results file
            results[name.lower().replace('.', '_')] = {
                "status": "running",
                "current_epoch": epoch + 1,
                "current_loss": float(avg_loss),
                "elapsed_seconds": elapsed,
            }
            save_results(results)
        
        # Final statistics
        total_time = time.time() - start_time
        end_mem = psutil.Process().memory_info().rss / (1024**2)
        
        result = {
            "status": "completed",
            "total_time_seconds": total_time,
            "avg_epoch_time": total_time / N_EPOCHS,
            "memory_start_mb": start_mem,
            "memory_end_mb": end_mem,
            "memory_delta_mb": end_mem - start_mem,
            "final_loss": losses[-1],
            "losses": losses,
            "epoch_times": epoch_times,
        }
        
        log(f"  [{name}] COMPLETED: {total_time:.1f}s total, "
            f"{total_time/N_EPOCHS:.1f}s/epoch, final_loss={losses[-1]:.6f}", progress=True)
        log("", progress=True)
        
        return result, params
    
    # =========================================================================
    # RUN LRET BENCHMARK
    # =========================================================================
    if dev_lret:
        results["status"] = "running_lret"
        save_results(results)
        
        try:
            circuit_lret = make_circuit(dev_lret)
            lret_result, lret_params = train("LRET", circuit_lret, init_params.copy(), LRET_EPOCHS_FILE)
            results["lret"] = lret_result
        except Exception as e:
            log(f"LRET FAILED: {e}")
            log(traceback.format_exc())
            results["lret"] = {"status": "failed", "error": str(e)}
        
        save_results(results)
    
    # =========================================================================
    # RUN BASELINE BENCHMARK
    # =========================================================================
    if dev_baseline:
        results["status"] = "running_baseline"
        save_results(results)
        
        try:
            circuit_baseline = make_circuit(dev_baseline)
            baseline_result, baseline_params = train("default.mixed", circuit_baseline, init_params.copy(), BASELINE_EPOCHS_FILE)
            results["baseline"] = baseline_result
        except Exception as e:
            log(f"default.mixed FAILED: {e}")
            log(traceback.format_exc())
            results["baseline"] = {"status": "failed", "error": str(e)}
        
        save_results(results)
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    log("=" * 70)
    log("BENCHMARK COMPLETE")
    log("=" * 70)
    
    results["status"] = "completed"
    
    if results["lret"].get("status") == "completed" and results["baseline"].get("status") == "completed":
        lret_time = results["lret"]["total_time_seconds"]
        baseline_time = results["baseline"]["total_time_seconds"]
        speedup = baseline_time / lret_time
        
        lret_loss = results["lret"]["final_loss"]
        baseline_loss = results["baseline"]["final_loss"]
        loss_diff = abs(lret_loss - baseline_loss)
        
        log("")
        log(f"{'METRIC':<30} {'LRET':>15} {'default.mixed':>15} {'RATIO':>12}")
        log("-" * 75)
        log(f"{'Total time (seconds)':<30} {lret_time:>15.1f} {baseline_time:>15.1f} {speedup:>11.2f}x")
        log(f"{'Avg time per epoch (s)':<30} {lret_time/N_EPOCHS:>15.1f} {baseline_time/N_EPOCHS:>15.1f}")
        log(f"{'Final loss':<30} {lret_loss:>15.6f} {baseline_loss:>15.6f}")
        log(f"{'Loss difference':<30} {loss_diff:>15.6f}")
        log("")
        
        if speedup > 1:
            log(f"[RESULT] LRET is {speedup:.2f}x FASTER than default.mixed")
        else:
            log(f"[RESULT] LRET is {1/speedup:.2f}x SLOWER than default.mixed")
        
        if loss_diff < 0.01:
            log(f"[RESULT] Results MATCH (loss difference < 0.01)")
        else:
            log(f"[RESULT] Results DIFFER (loss difference = {loss_diff:.4f})")
        
        results["summary"] = {
            "speedup": speedup,
            "loss_difference": loss_diff,
            "lret_faster": speedup > 1,
            "results_match": loss_diff < 0.01,
        }
    else:
        log("One or both benchmarks failed - see logs for details")
    
    save_results(results)
    
    log("")
    log("=" * 70)
    log("OUTPUT FILES:")
    log(f"  Main log:       {LOG_FILE}")
    log(f"  Progress log:   {PROGRESS_FILE}")
    log(f"  Results JSON:   {RESULTS_FILE}")
    log(f"  LRET epochs:    {LRET_EPOCHS_FILE}")
    log(f"  Baseline epochs:{BASELINE_EPOCHS_FILE}")
    log("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL ERROR: {e}")
        log(traceback.format_exc())
        sys.exit(1)
