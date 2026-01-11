#!/usr/bin/env python3
"""
Fair Noisy Benchmark v2: LRET vs default.mixed
With detailed progress logging and timestamps

Run with: python benchmarks/fair_noisy_benchmark_v2.py
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

# Force unbuffered output for real-time logging
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

# ============================================================================
# Configuration
# ============================================================================
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = f"D:/LRET/results/run_{RUN_ID}"
os.makedirs(LOG_DIR, exist_ok=True)

# Benchmark parameters
N_QUBITS = 8
N_EPOCHS = 50
N_SAMPLES = 30
NOISE_RATE = 0.05  # 5% depolarizing noise
LEARNING_RATE = 0.1

# Log files
MAIN_LOG = f"{LOG_DIR}/benchmark_main.log"
PROGRESS_LOG = f"{LOG_DIR}/progress.log"
RESULTS_JSON = f"{LOG_DIR}/results.json"

def log(msg, also_progress=False):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(MAIN_LOG, 'a') as f:
        f.write(line + '\n')
    if also_progress:
        with open(PROGRESS_LOG, 'a') as f:
            f.write(line + '\n')

def save_progress(data):
    """Save current progress to JSON"""
    with open(RESULTS_JSON, 'w') as f:
        json.dump(data, f, indent=2, default=str)

# ============================================================================
# Main Benchmark
# ============================================================================
def main():
    log("=" * 70)
    log("Fair Noisy QNN Benchmark v2: LRET vs default.mixed")
    log("=" * 70)
    log(f"Run ID: {RUN_ID}")
    log(f"Log directory: {LOG_DIR}")
    log(f"PennyLane version: {qml.__version__}")
    log("")
    log(f"Configuration:")
    log(f"  Qubits: {N_QUBITS}")
    log(f"  Epochs: {N_EPOCHS}")
    log(f"  Samples: {N_SAMPLES}")
    log(f"  Noise rate: {NOISE_RATE:.1%}")
    log("")
    
    progress = {
        "run_id": RUN_ID,
        "config": {
            "n_qubits": N_QUBITS,
            "n_epochs": N_EPOCHS,
            "n_samples": N_SAMPLES,
            "noise_rate": NOISE_RATE,
        },
        "status": "initializing",
        "lret": {},
        "baseline": {},
    }
    save_progress(progress)
    
    # Initialize parameters (same for both devices)
    np.random.seed(42)
    X_train = np.random.randn(N_SAMPLES, N_QUBITS).astype(np.float64)
    y_train = np.sign(np.sum(X_train[:, :2], axis=1))
    init_params = np.random.randn(2, N_QUBITS, 2) * 0.1
    
    # ========================================================================
    # Create devices
    # ========================================================================
    log("Creating devices...")
    
    try:
        dev_lret = qml.device('qlret.mixed', wires=N_QUBITS, epsilon=1e-4)
        log(f"  LRET device: {dev_lret.name} [OK]")
    except Exception as e:
        log(f"  LRET device FAILED: {e}")
        dev_lret = None
    
    try:
        dev_baseline = qml.device('default.mixed', wires=N_QUBITS)
        log(f"  Baseline device: {dev_baseline.name} [OK]")
    except Exception as e:
        log(f"  Baseline device FAILED: {e}")
        dev_baseline = None
    
    log("")
    
    # ========================================================================
    # Define noisy QNN circuit
    # ========================================================================
    def create_noisy_qnn(device):
        @qml.qnode(device)
        def circuit(params, x):
            # Data encoding with noise
            for i in range(N_QUBITS):
                qml.RY(x[i % len(x)] * np.pi, wires=i)
                qml.DepolarizingChannel(NOISE_RATE, wires=i)
            
            # Variational layers with noise
            for layer in range(2):
                for i in range(N_QUBITS):
                    qml.RY(params[layer, i, 0], wires=i)
                    qml.RZ(params[layer, i, 1], wires=i)
                    qml.DepolarizingChannel(NOISE_RATE, wires=i)
                
                for i in range(N_QUBITS - 1):
                    qml.CNOT(wires=[i, i + 1])
                    qml.DepolarizingChannel(NOISE_RATE, wires=i)
                    qml.DepolarizingChannel(NOISE_RATE, wires=i + 1)
            
            return qml.expval(qml.PauliZ(0))
        
        return circuit
    
    # ========================================================================
    # Training function with detailed logging
    # ========================================================================
    def train_qnn(circuit, name, params):
        log(f"Starting {name} training...", also_progress=True)
        
        start_time = time.time()
        start_mem = psutil.Process().memory_info().rss / (1024**2)
        
        losses = []
        epoch_times = []
        
        for epoch in range(N_EPOCHS):
            epoch_start = time.time()
            epoch_loss = 0.0
            
            for sample_idx, (x, y) in enumerate(zip(X_train, y_train)):
                # Forward pass
                try:
                    pred = circuit(params, x)
                    loss = float((pred - y) ** 2)
                    epoch_loss += loss
                except Exception as e:
                    log(f"  ERROR in {name} epoch {epoch+1} sample {sample_idx}: {e}")
                    raise
                
                # Numerical gradient
                grad = np.zeros_like(params)
                shift = 0.1
                for idx in np.ndindex(params.shape):
                    params_plus = params.copy()
                    params_plus[idx] += shift
                    params_minus = params.copy()
                    params_minus[idx] -= shift
                    grad[idx] = (circuit(params_plus, x) - circuit(params_minus, x)) / (2 * shift)
                
                # Update parameters
                params = params - LEARNING_RATE * 2 * (pred - y) * grad
            
            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / N_SAMPLES
            losses.append(float(avg_loss))
            epoch_times.append(epoch_time)
            
            # Log every epoch for tracking
            log(f"  [{name}] Epoch {epoch+1:3d}/{N_EPOCHS}: Loss={avg_loss:.6f}, Time={epoch_time:.1f}s", also_progress=True)
            
            # Update progress file
            progress[name.lower().replace('.', '_').replace(' ', '_')] = {
                "status": "running",
                "current_epoch": epoch + 1,
                "current_loss": float(avg_loss),
                "elapsed_time": time.time() - start_time,
            }
            save_progress(progress)
        
        total_time = time.time() - start_time
        end_mem = psutil.Process().memory_info().rss / (1024**2)
        
        result = {
            "status": "completed",
            "total_time_seconds": total_time,
            "time_per_epoch": total_time / N_EPOCHS,
            "memory_start_mb": start_mem,
            "memory_end_mb": end_mem,
            "memory_delta_mb": end_mem - start_mem,
            "final_loss": losses[-1],
            "all_losses": losses,
            "epoch_times": epoch_times,
        }
        
        log(f"  [{name}] COMPLETED in {total_time:.1f}s (avg {total_time/N_EPOCHS:.1f}s/epoch)", also_progress=True)
        return result, params
    
    # ========================================================================
    # Run LRET Benchmark
    # ========================================================================
    if dev_lret:
        log("-" * 70)
        progress["status"] = "running_lret"
        save_progress(progress)
        
        try:
            circuit_lret = create_noisy_qnn(dev_lret)
            params_lret = init_params.copy()
            results_lret, final_params_lret = train_qnn(circuit_lret, "LRET", params_lret)
            progress["lret"] = results_lret
        except Exception as e:
            log(f"LRET FAILED: {e}")
            log(traceback.format_exc())
            progress["lret"] = {"status": "failed", "error": str(e)}
        
        save_progress(progress)
        log("")
    
    # ========================================================================
    # Run Baseline Benchmark
    # ========================================================================
    if dev_baseline:
        log("-" * 70)
        progress["status"] = "running_baseline"
        save_progress(progress)
        
        try:
            circuit_baseline = create_noisy_qnn(dev_baseline)
            params_baseline = init_params.copy()
            results_baseline, final_params_baseline = train_qnn(circuit_baseline, "default.mixed", params_baseline)
            progress["baseline"] = results_baseline
        except Exception as e:
            log(f"Baseline FAILED: {e}")
            log(traceback.format_exc())
            progress["baseline"] = {"status": "failed", "error": str(e)}
        
        save_progress(progress)
        log("")
    
    # ========================================================================
    # Results Summary
    # ========================================================================
    log("=" * 70)
    log("RESULTS SUMMARY")
    log("=" * 70)
    
    progress["status"] = "completed"
    
    if progress["lret"].get("status") == "completed" and progress["baseline"].get("status") == "completed":
        lret_time = progress["lret"]["total_time_seconds"]
        baseline_time = progress["baseline"]["total_time_seconds"]
        speedup = baseline_time / lret_time
        
        lret_loss = progress["lret"]["final_loss"]
        baseline_loss = progress["baseline"]["final_loss"]
        loss_diff = abs(lret_loss - baseline_loss)
        
        log(f"")
        log(f"{'Metric':<25} {'LRET':>15} {'default.mixed':>15} {'Ratio':>12}")
        log("-" * 70)
        log(f"{'Total time (s)':<25} {lret_time:>15.1f} {baseline_time:>15.1f} {speedup:>11.2f}x")
        log(f"{'Time per epoch (s)':<25} {lret_time/N_EPOCHS:>15.1f} {baseline_time/N_EPOCHS:>15.1f}")
        log(f"{'Final loss':<25} {lret_loss:>15.6f} {baseline_loss:>15.6f}")
        log(f"{'Loss difference':<25} {loss_diff:>15.6f}")
        log("")
        
        if speedup > 1:
            log(f"[+] LRET is {speedup:.2f}x FASTER than default.mixed")
        else:
            log(f"[-] LRET is {1/speedup:.2f}x SLOWER than default.mixed")
        
        if loss_diff < 0.1:
            log(f"[+] Results MATCH (loss diff < 0.1)")
        else:
            log(f"[!] Results DIFFER (loss diff = {loss_diff:.4f})")
        
        progress["summary"] = {
            "speedup": speedup,
            "loss_difference": loss_diff,
            "results_match": loss_diff < 0.1,
        }
    else:
        log("One or both benchmarks failed - see logs for details")
    
    save_progress(progress)
    log("")
    log("=" * 70)
    log(f"Results saved to: {RESULTS_JSON}")
    log(f"Full log saved to: {MAIN_LOG}")
    log("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL ERROR: {e}")
        log(traceback.format_exc())
        sys.exit(1)
