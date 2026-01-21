#!/usr/bin/env python3
"""
LRET vs default.mixed Benchmark - Heavy Test
=============================================
Configuration:
  - Qubits: 8
  - Batch size: 200
  - Epochs: 200
  - Noise: 15% DepolarizingChannel

Estimated time: LRET ~6-10 hours, default.mixed ~60-100 hours (likely OOM!)
Run: python benchmarks/pennylane/8q_200e_200s_15n.py

WARNING: This is a heavy benchmark demonstrating LRET's scalability.
         default.mixed will likely run out of memory with 8 qubits.

Now includes integrated CPU monitoring - launches in separate windows.
"""

import time
import sys
import os
import json
import subprocess
import platform

# =============================================================================
# LAUNCHER MODE - Start benchmark and CPU monitor in separate windows
# =============================================================================
if len(sys.argv) <= 1 or sys.argv[1] != "--worker":
    from datetime import datetime
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
    print("⚠️  WARNING: 8-qubit benchmark - default.mixed may OOM!")
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
# WORKER MODE - Actual benchmark execution
# =============================================================================
import csv
import numpy as np
import pennylane as qml
import psutil
import traceback
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================
N_QUBITS = 8
N_EPOCHS = 200
N_SAMPLES = 200       # batch size
NOISE_RATE = 0.15    # 15% depolarizing noise
LEARNING_RATE = 0.1
RANDOM_SEED = 42

# =============================================================================
# SETUP LOGGING
# =============================================================================
# Get log directory from command line argument (passed from launcher)
LOG_DIR = sys.argv[2] if len(sys.argv) > 2 else None
if LOG_DIR is None:
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results', f'{script_name}_{RUN_ID}')
    os.makedirs(LOG_DIR, exist_ok=True)
else:
    RUN_ID = os.path.basename(LOG_DIR).split('_')[-2] + '_' + os.path.basename(LOG_DIR).split('_')[-1]

LOG_FILE = os.path.join(LOG_DIR, "benchmark.log")
PROGRESS_FILE = os.path.join(LOG_DIR, "progress.log")
RESULTS_FILE = os.path.join(LOG_DIR, "results.json")

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

# =============================================================================
# MAIN BENCHMARK
# =============================================================================
def main():
    log("=" * 70)
    log("LRET vs default.mixed BENCHMARK - Heavy Test (8q 200e)")
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
    log("⚠️  HEAVY BENCHMARK WARNING ⚠️")
    log("This benchmark pushes both devices to their limits:")
    log("  - LRET should complete in 6-10 hours")
    log("  - default.mixed will likely fail with OOM (Out of Memory)")
    log("  - This demonstrates LRET's scalability advantage")
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
    def train(name, circuit, params, mode_name):
        log(f"{'='*70}", progress=True)
        log(f"TRAINING: {name}", progress=True)
        log(f"{'='*70}", progress=True)
        
        # Create CSV file for this training run
        safe_name = mode_name.lower().replace(' ', '_').replace('.', '_')
        csv_path = os.path.join(LOG_DIR, f"epochs_{safe_name}.csv")
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['epoch', 'loss', 'time_seconds', 'elapsed_seconds', 'eta_seconds'])
        
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
            
            # Save epoch data to CSV
            elapsed = time.time() - start_time
            eta = (elapsed / (epoch + 1)) * (N_EPOCHS - epoch - 1)
            csv_writer.writerow([epoch + 1, f"{avg_loss:.6f}", f"{epoch_time:.2f}", f"{elapsed:.2f}", f"{eta:.2f}"])
            csv_file.flush()
            
            # Log progress
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
        
        csv_file.close()
        log(f"  ✓ Epoch data saved to: {csv_path}", progress=True)
        
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
            "csv_file": csv_path,
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
            lret_result, lret_params = train("LRET", circuit_lret, init_params.copy(), "lret")
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
            baseline_result, baseline_params = train("default.mixed", circuit_baseline, init_params.copy(), "default_mixed")
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
    elif results["lret"].get("status") == "completed" and results["baseline"].get("status") == "failed":
        log("")
        log("🎯 RESULT: LRET completed successfully while default.mixed failed!")
        log("   This demonstrates LRET's superior scalability for 8-qubit systems.")
        log("")
        results["summary"] = {
            "lret_completed": True,
            "baseline_failed": True,
            "demonstrates_scalability": True,
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
    log(f"  Results dir:    {LOG_DIR}")
    log("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL ERROR: {e}")
        log(traceback.format_exc())
        sys.exit(1)
