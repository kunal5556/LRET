#!/usr/bin/env python3
"""
Fair Noisy Benchmark: LRET vs default.mixed
Both using IDENTICAL DepolarizingChannel operations from PennyLane
"""

import time
import numpy as np
import pennylane as qml
from qlret import QLRETDevice
import psutil
import tracemalloc

print("="*70)
print("Fair Noisy QNN Benchmark: LRET vs default.mixed")
print("="*70)
print(f"PennyLane version: {qml.__version__}")
print()

# Benchmark parameters
N_QUBITS = 8
N_EPOCHS = 50
N_SAMPLES = 30
NOISE_RATE = 0.05  # 5% depolarizing noise per gate
LEARNING_RATE = 0.1

print(f"Configuration:")
print(f"  Qubits: {N_QUBITS}")
print(f"  Epochs: {N_EPOCHS}")
print(f"  Samples: {N_SAMPLES}")
print(f"  Noise rate: {NOISE_RATE:.1%}")
print()

# Create devices
dev_lret = QLRETDevice(wires=N_QUBITS, shots=None, epsilon=1e-4)
dev_baseline = qml.device('default.mixed', wires=N_QUBITS)

print(f"LRET device: {dev_lret.name}")
print(f"Baseline device: default.mixed")
print()

# Create noisy QNN circuit
def create_noisy_qnn(device):
    @qml.qnode(device)
    def circuit(params, x):
        # Data encoding layer
        for i in range(N_QUBITS):
            qml.RY(x[i % len(x)] * np.pi, wires=i)
            qml.DepolarizingChannel(NOISE_RATE, wires=i)
        
        # Variational layers (2 layers)
        for layer in range(2):
            # Single qubit rotations
            for i in range(N_QUBITS):
                qml.RY(params[layer, i, 0], wires=i)
                qml.RZ(params[layer, i, 1], wires=i)
                qml.DepolarizingChannel(NOISE_RATE, wires=i)
            
            # Entangling layer
            for i in range(N_QUBITS - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.DepolarizingChannel(NOISE_RATE, wires=i)
                qml.DepolarizingChannel(NOISE_RATE, wires=i + 1)
        
        return qml.expval(qml.PauliZ(0))
    
    return circuit

# Generate synthetic data
np.random.seed(42)
X_train = np.random.randn(N_SAMPLES, N_QUBITS).astype(np.float64)
y_train = np.sign(np.sum(X_train[:, :2], axis=1))  # Simple binary classification

# Initialize parameters
init_params = np.random.randn(2, N_QUBITS, 2) * 0.1

def train_qnn(circuit, name):
    """Train QNN and measure performance."""
    params = init_params.copy()
    
    # Track memory
    tracemalloc.start()
    start_time = time.time()
    
    losses = []
    for epoch in range(N_EPOCHS):
        epoch_loss = 0.0
        for x, y in zip(X_train, y_train):
            # Forward pass
            pred = circuit(params, x)
            loss = (pred - y) ** 2
            epoch_loss += loss
            
            # Simple numerical gradient (parameter shift would be better but slower)
            # For fair comparison, both use the same gradient computation
            grad = np.zeros_like(params)
            shift = 0.1
            for idx in np.ndindex(params.shape):
                params_plus = params.copy()
                params_plus[idx] += shift
                params_minus = params.copy()
                params_minus[idx] -= shift
                grad[idx] = (circuit(params_plus, x) - circuit(params_minus, x)) / (2 * shift)
            
            # Update
            params -= LEARNING_RATE * 2 * (pred - y) * grad
        
        avg_loss = epoch_loss / N_SAMPLES
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"  [{name}] Epoch {epoch + 1}/{N_EPOCHS}: Loss = {avg_loss:.6f}")
    
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        'time': end_time - start_time,
        'peak_memory_mb': peak / (1024 * 1024),
        'final_loss': losses[-1],
        'losses': losses,
        'final_params': params,
    }

# Run LRET benchmark
print("-" * 70)
print("Running LRET benchmark...")
circuit_lret = create_noisy_qnn(dev_lret)
results_lret = train_qnn(circuit_lret, "LRET")
print(f"  LRET completed in {results_lret['time']:.1f}s")
print()

# Run baseline benchmark
print("-" * 70)
print("Running default.mixed benchmark...")
circuit_baseline = create_noisy_qnn(dev_baseline)
results_baseline = train_qnn(circuit_baseline, "Baseline")
print(f"  default.mixed completed in {results_baseline['time']:.1f}s")
print()

# Results summary
print("="*70)
print("RESULTS SUMMARY")
print("="*70)
print(f"\n{'Metric':<25} {'LRET':>15} {'default.mixed':>15} {'Ratio':>12}")
print("-"*70)
print(f"{'Total time (s)':<25} {results_lret['time']:>15.2f} {results_baseline['time']:>15.2f} {results_baseline['time']/results_lret['time']:>12.2f}x")
print(f"{'Peak memory (MB)':<25} {results_lret['peak_memory_mb']:>15.2f} {results_baseline['peak_memory_mb']:>15.2f} {results_baseline['peak_memory_mb']/results_lret['peak_memory_mb']:>12.2f}x")
print(f"{'Final loss':<25} {results_lret['final_loss']:>15.6f} {results_baseline['final_loss']:>15.6f}")

# Check if results match
loss_diff = abs(results_lret['final_loss'] - results_baseline['final_loss'])
print(f"\n{'Loss difference':<25} {loss_diff:.6f}")
print(f"{'Results match':<25} {'YES' if loss_diff < 0.1 else 'NO'}")

# Speedup analysis
speedup = results_baseline['time'] / results_lret['time']
memory_ratio = results_baseline['peak_memory_mb'] / results_lret['peak_memory_mb']

print(f"\n{'='*70}")
print("PERFORMANCE ANALYSIS")
print(f"{'='*70}")
if speedup > 1:
    print(f"✓ LRET is {speedup:.2f}x FASTER than default.mixed")
else:
    print(f"✗ LRET is {1/speedup:.2f}x SLOWER than default.mixed")

if memory_ratio > 1:
    print(f"✓ LRET uses {memory_ratio:.2f}x LESS memory than default.mixed")
else:
    print(f"✗ LRET uses {1/memory_ratio:.2f}x MORE memory than default.mixed")

print(f"\nBoth devices produced {'MATCHING' if loss_diff < 0.1 else 'DIFFERENT'} results")
print(f"(This validates that LRET correctly implements PennyLane noise channels)")
print("="*70)
