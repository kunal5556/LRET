#!/usr/bin/env python3
"""
Simple Fair Noisy Benchmark: LRET vs default.mixed
Reduced complexity for faster execution with clear progress tracking.
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

# Configuration
N_QUBITS = 8
N_EPOCHS = 10        # Reduced from 50
N_SAMPLES = 10       # Reduced from 30  
NOISE_RATE = 0.05
LEARNING_RATE = 0.1

# Setup logging
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = f"D:/LRET/results/run_{RUN_ID}"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = f"{LOG_DIR}/benchmark.log"
JSON_FILE = f"{LOG_DIR}/results.json"

def log(msg):
    """Log with timestamp to file and stdout"""
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def save_json(data):
    with open(JSON_FILE, 'w') as f:
        json.dump(data, f, indent=2, default=str)

# ============================================================================
log("=" * 60)
log("Fair Noisy QNN Benchmark (Simple)")
log("=" * 60)
log(f"Run ID: {RUN_ID}")
log(f"PennyLane: {qml.__version__}")
log(f"Config: {N_QUBITS}q, {N_EPOCHS} epochs, {N_SAMPLES} samples, {NOISE_RATE:.0%} noise")
log("")

results = {"run_id": RUN_ID, "config": {"qubits": N_QUBITS, "epochs": N_EPOCHS, "samples": N_SAMPLES}}

# Generate data
np.random.seed(42)
X = np.random.randn(N_SAMPLES, N_QUBITS)
y = np.sign(np.sum(X[:, :2], axis=1))
init_params = np.random.randn(2, N_QUBITS, 2) * 0.1

# Circuit with noise
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

# Training function
def train(name, circuit, params):
    log(f"--- {name} Training ---")
    start = time.time()
    losses = []
    
    for epoch in range(N_EPOCHS):
        epoch_start = time.time()
        epoch_loss = 0
        
        for x, target in zip(X, y):
            pred = float(circuit(params, x))
            epoch_loss += (pred - target) ** 2
            
            # Simple gradient update (no parameter shift for speed)
            grad = np.zeros_like(params)
            for idx in np.ndindex(params.shape):
                p_plus = params.copy()
                p_plus[idx] += 0.1
                p_minus = params.copy()
                p_minus[idx] -= 0.1
                grad[idx] = (circuit(p_plus, x) - circuit(p_minus, x)) / 0.2
            params = params - LEARNING_RATE * 2 * (pred - target) * grad
        
        avg_loss = epoch_loss / N_SAMPLES
        losses.append(float(avg_loss))
        epoch_time = time.time() - epoch_start
        log(f"  Epoch {epoch+1:2d}/{N_EPOCHS}: loss={avg_loss:.4f}, time={epoch_time:.1f}s")
    
    total = time.time() - start
    log(f"  DONE: {total:.1f}s total, final loss={losses[-1]:.4f}")
    return {"time": total, "final_loss": losses[-1], "losses": losses}

# ============================================================================
# Run LRET
# ============================================================================
try:
    log("Creating LRET device...")
    dev_lret = qml.device('qlret.mixed', wires=N_QUBITS, epsilon=1e-4)
    circuit_lret = make_circuit(dev_lret)
    results["lret"] = train("LRET", circuit_lret, init_params.copy())
except Exception as e:
    log(f"LRET FAILED: {e}")
    log(traceback.format_exc())
    results["lret"] = {"error": str(e)}

save_json(results)
log("")

# ============================================================================
# Run default.mixed
# ============================================================================
try:
    log("Creating default.mixed device...")
    dev_baseline = qml.device('default.mixed', wires=N_QUBITS)
    circuit_baseline = make_circuit(dev_baseline)
    results["baseline"] = train("default.mixed", circuit_baseline, init_params.copy())
except Exception as e:
    log(f"default.mixed FAILED: {e}")
    log(traceback.format_exc())
    results["baseline"] = {"error": str(e)}

save_json(results)

# ============================================================================
# Summary
# ============================================================================
log("")
log("=" * 60)
log("RESULTS SUMMARY")
log("=" * 60)

if "time" in results.get("lret", {}) and "time" in results.get("baseline", {}):
    lret_time = results["lret"]["time"]
    base_time = results["baseline"]["time"]
    speedup = base_time / lret_time
    
    log(f"LRET time:         {lret_time:8.1f}s")
    log(f"default.mixed time:{base_time:8.1f}s")
    log(f"Speedup:           {speedup:8.2f}x {'(LRET faster)' if speedup > 1 else '(baseline faster)'}")
    log(f"LRET loss:         {results['lret']['final_loss']:.6f}")
    log(f"Baseline loss:     {results['baseline']['final_loss']:.6f}")
    
    results["summary"] = {"speedup": speedup, "lret_faster": speedup > 1}
else:
    log("One or both benchmarks failed")

save_json(results)
log("")
log(f"Results: {JSON_FILE}")
log(f"Log: {LOG_FILE}")
log("=" * 60)
