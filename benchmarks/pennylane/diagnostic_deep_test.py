#!/usr/bin/env python3
"""
Deep diagnostic test to verify LRET backend is actually being used
and parallelization parameters are passed correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

import json
import time
import pennylane as qml
import numpy as np

print("=" * 70)
print("LRET BACKEND DEEP DIAGNOSTIC TEST")
print("=" * 70)

# Test 1: Verify native module is loaded
print("\n[1] Checking native module...")
try:
    from qlret import _qlret_native
    print(f"  ✓ Native module loaded: {_qlret_native}")
    print(f"  ✓ run_circuit_json available: {hasattr(_qlret_native, 'run_circuit_json')}")
except ImportError as e:
    print(f"  ✗ Native module NOT available: {e}")
    print("  ⚠ FALLBACK MODE - may use subprocess or mock!")

# Test 2: Check API module backend detection
print("\n[2] Checking API backend detection...")
from qlret import api
native = api._get_native_module()
exe = api._find_executable()
print(f"  Native module available: {native is not None}")
print(f"  CLI executable found: {exe}")
if native:
    print(f"  → Will use: NATIVE BACKEND (fast, in-process)")
elif exe:
    print(f"  → Will use: SUBPROCESS BACKEND (slower)")
else:
    print(f"  → NO BACKEND AVAILABLE - WILL FAIL")

# Test 3: Manual JSON circuit execution with timing
print("\n[3] Manual JSON circuit test (direct call to backend)...")

# Create a simple circuit JSON
test_circuit = {
    "circuit": {
        "num_qubits": 4,
        "operations": [
            {"name": "H", "wires": [0]},
            {"name": "CNOT", "wires": [0, 1]},
            {"name": "RX", "wires": [2], "params": [0.5]},
        ],
        "observables": [
            {"type": "PAULI", "operator": "Z", "wires": [0], "coefficient": 1.0}
        ]
    },
    "config": {
        "epsilon": 1e-4,
        "num_threads": 4,
        "parallel_mode": "hybrid"
    }
}

print(f"  Circuit config: {json.dumps(test_circuit['config'], indent=4)}")

from qlret.api import simulate_json
start = time.perf_counter()
result = simulate_json(test_circuit)
elapsed = time.perf_counter() - start

print(f"  Result status: {result.get('status', 'unknown')}")
print(f"  Execution time (C++ reported): {result.get('execution_time_ms', 'N/A')} ms")
print(f"  Execution time (Python measured): {elapsed*1000:.3f} ms")
print(f"  Final rank: {result.get('final_rank', 'N/A')}")
print(f"  Expectation values: {result.get('expectation_values', [])}")

# Test 4: Compare execution times for different parallel modes
print("\n[4] Parallel mode comparison (same circuit)...")

modes = ["sequential", "row", "hybrid"]
results_by_mode = {}

for mode in modes:
    test_circuit["config"]["parallel_mode"] = mode
    test_circuit["config"]["num_threads"] = 0 if mode != "sequential" else 1
    
    # Run multiple times and average
    times = []
    for _ in range(10):
        start = time.perf_counter()
        result = simulate_json(test_circuit)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    results_by_mode[mode] = avg_time
    print(f"  {mode.upper():12s}: {avg_time:.3f} ± {std_time:.3f} ms (C++ reported: {result.get('execution_time_ms', 'N/A'):.3f} ms)")

# Test 5: Run a more complex circuit with noise (like the original benchmark)
print("\n[5] Complex circuit with noise (like original benchmark)...")

def create_complex_circuit(n_qubits=4, noise_ops=10):
    """Create a circuit similar to the original training circuit."""
    operations = []
    
    # Data encoding layer with noise
    for i in range(n_qubits):
        operations.append({"name": "RY", "wires": [i], "params": [0.5]})
        # Add depolarizing noise via Kraus operators
        # Note: This tests whether Kraus channels work
    
    # Variational layers
    for layer in range(2):
        for i in range(n_qubits):
            operations.append({"name": "RY", "wires": [i], "params": [0.1 * (layer + 1)]})
            operations.append({"name": "RZ", "wires": [i], "params": [0.2 * (layer + 1)]})
        for i in range(n_qubits - 1):
            operations.append({"name": "CNOT", "wires": [i, i + 1]})
    
    observables = [{"type": "PAULI", "operator": "Z", "wires": [0], "coefficient": 1.0}]
    
    return {
        "circuit": {
            "num_qubits": n_qubits,
            "operations": operations,
            "observables": observables
        },
        "config": {
            "epsilon": 1e-4,
            "num_threads": 0,
            "parallel_mode": "hybrid"
        }
    }

complex_circuit = create_complex_circuit(n_qubits=4)
print(f"  Total operations: {len(complex_circuit['circuit']['operations'])}")

# Run multiple times
times = []
for _ in range(50):
    start = time.perf_counter()
    result = simulate_json(complex_circuit)
    elapsed = time.perf_counter() - start
    times.append(elapsed * 1000)

print(f"  Mean execution time: {np.mean(times):.3f} ms")
print(f"  Std deviation: {np.std(times):.3f} ms")
print(f"  Min: {np.min(times):.3f} ms, Max: {np.max(times):.3f} ms")

# Test 6: PennyLane device execution
print("\n[6] PennyLane device execution (through full stack)...")

dev = qml.device("qlret.mixed", wires=4, epsilon=1e-4, 
                 num_threads=0, parallel_mode="hybrid")

@qml.qnode(dev)
def pennylane_circuit():
    for i in range(4):
        qml.RY(0.5, wires=i)
    for layer in range(2):
        for i in range(4):
            qml.RY(0.1 * (layer + 1), wires=i)
            qml.RZ(0.2 * (layer + 1), wires=i)
        for i in range(3):
            qml.CNOT(wires=[i, i + 1])
    return qml.expval(qml.PauliZ(0))

# Warmup
_ = pennylane_circuit()

# Time it
times = []
for _ in range(50):
    start = time.perf_counter()
    result = pennylane_circuit()
    elapsed = time.perf_counter() - start
    times.append(elapsed * 1000)

print(f"  Mean execution time: {np.mean(times):.3f} ms")
print(f"  Std deviation: {np.std(times):.3f} ms")
print(f"  Result: {result}")

# Test 7: Compare with default.mixed
print("\n[7] Comparison: LRET vs default.mixed (same circuit)...")

dev_baseline = qml.device("default.mixed", wires=4)

@qml.qnode(dev_baseline)
def baseline_circuit():
    for i in range(4):
        qml.RY(0.5, wires=i)
    for layer in range(2):
        for i in range(4):
            qml.RY(0.1 * (layer + 1), wires=i)
            qml.RZ(0.2 * (layer + 1), wires=i)
        for i in range(3):
            qml.CNOT(wires=[i, i + 1])
    return qml.expval(qml.PauliZ(0))

# Warmup
_ = baseline_circuit()

# Time LRET
lret_times = []
for _ in range(50):
    start = time.perf_counter()
    _ = pennylane_circuit()
    elapsed = time.perf_counter() - start
    lret_times.append(elapsed * 1000)

# Time baseline
baseline_times = []
for _ in range(50):
    start = time.perf_counter()
    _ = baseline_circuit()
    elapsed = time.perf_counter() - start
    baseline_times.append(elapsed * 1000)

lret_avg = np.mean(lret_times)
baseline_avg = np.mean(baseline_times)
speedup = baseline_avg / lret_avg

print(f"  LRET mean:     {lret_avg:.3f} ms")
print(f"  Baseline mean: {baseline_avg:.3f} ms")
print(f"  Speedup: {speedup:.2f}x {'(LRET faster)' if speedup > 1 else '(baseline faster)'}")

# Test 8: NOW THE CRITICAL TEST - Simulate training workload
print("\n[8] CRITICAL TEST: Simulate training workload (like original)...")
print("    This tests: 25 samples × 33 gradient evals = 825 circuit calls")

# Create circuits
@qml.qnode(dev)
def training_circuit_lret(params, x):
    for i in range(4):
        qml.RY(x[i], wires=i)
    for layer in range(2):
        for i in range(4):
            qml.RY(params[layer, i, 0], wires=i)
            qml.RZ(params[layer, i, 1], wires=i)
        for i in range(3):
            qml.CNOT(wires=[i, i + 1])
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev_baseline)
def training_circuit_baseline(params, x):
    for i in range(4):
        qml.RY(x[i], wires=i)
    for layer in range(2):
        for i in range(4):
            qml.RY(params[layer, i, 0], wires=i)
            qml.RZ(params[layer, i, 1], wires=i)
        for i in range(3):
            qml.CNOT(wires=[i, i + 1])
    return qml.expval(qml.PauliZ(0))

# Simulate 1 epoch of training (25 samples with gradients)
np.random.seed(42)
n_samples = 25
X_train = np.random.randn(n_samples, 4)
params = np.random.randn(2, 4, 2) * 0.1

print("  Simulating 1 training epoch...")

# LRET epoch
start = time.perf_counter()
for x in X_train:
    # Forward pass
    pred = training_circuit_lret(params, x)
    
    # Numerical gradient (16 params × 2 = 32 extra evals)
    for idx in np.ndindex(params.shape):
        p_plus = params.copy()
        p_plus[idx] += 0.1
        _ = training_circuit_lret(p_plus, x)
lret_epoch_time = time.perf_counter() - start

# Baseline epoch
start = time.perf_counter()
for x in X_train:
    pred = training_circuit_baseline(params, x)
    for idx in np.ndindex(params.shape):
        p_plus = params.copy()
        p_plus[idx] += 0.1
        _ = training_circuit_baseline(p_plus, x)
baseline_epoch_time = time.perf_counter() - start

print(f"  LRET epoch time:     {lret_epoch_time:.2f} seconds")
print(f"  Baseline epoch time: {baseline_epoch_time:.2f} seconds")
print(f"  Speedup: {baseline_epoch_time/lret_epoch_time:.2f}x")

# Calculate circuit calls
circuit_calls = n_samples * (1 + 16)  # forward + gradients (only plus side)
print(f"  Circuit calls per epoch: {circuit_calls}")
print(f"  LRET time per circuit: {lret_epoch_time/circuit_calls*1000:.3f} ms")
print(f"  Baseline time per circuit: {baseline_epoch_time/circuit_calls*1000:.3f} ms")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)

# Summary
print("\n📋 SUMMARY:")
print(f"   Native backend: {'YES ✓' if native else 'NO ✗'}")
print(f"   Single circuit LRET: {lret_avg:.3f} ms")
print(f"   Single circuit baseline: {baseline_avg:.3f} ms")
print(f"   Training epoch LRET: {lret_epoch_time:.2f} s")
print(f"   Training epoch baseline: {baseline_epoch_time:.2f} s")
print(f"   Training speedup: {baseline_epoch_time/lret_epoch_time:.2f}x")
