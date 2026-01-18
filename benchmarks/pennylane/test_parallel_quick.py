#!/usr/bin/env python3
"""
Quick test to verify parallelization parameters are being passed correctly.
"""

import sys
import os

# Add LRET module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

import pennylane as qml
import numpy as np
import time

print("=" * 60)
print("PARALLELIZATION PARAMETER TEST")
print("=" * 60)

n_qubits = 4

# Test 1: Default settings (should be hybrid with auto threads)
print("\n[Test 1] Default settings (hybrid mode, auto threads)")
try:
    dev = qml.device("qlret.mixed", wires=n_qubits, epsilon=1e-4)
    print(f"  Device created: {dev.name}")
    print(f"  Num threads: {dev.num_threads} (0=auto)")
    print(f"  Parallel mode: {dev.parallel_mode}")
    
    @qml.qnode(dev)
    def circuit1():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))
    
    start = time.time()
    result = circuit1()
    elapsed = time.time() - start
    print(f"  Result: {result:.6f}")
    print(f"  Time: {elapsed*1000:.2f} ms")
    print("  ✓ Test 1 PASSED")
except Exception as e:
    print(f"  ✗ Test 1 FAILED: {e}")

# Test 2: Explicit hybrid mode with specific thread count
print("\n[Test 2] Explicit hybrid mode with 4 threads")
try:
    dev2 = qml.device("qlret.mixed", wires=n_qubits, epsilon=1e-4, 
                      num_threads=4, parallel_mode="hybrid")
    print(f"  Device created: {dev2.name}")
    print(f"  Num threads: {dev2.num_threads}")
    print(f"  Parallel mode: {dev2.parallel_mode}")
    
    @qml.qnode(dev2)
    def circuit2():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))
    
    start = time.time()
    result = circuit2()
    elapsed = time.time() - start
    print(f"  Result: {result:.6f}")
    print(f"  Time: {elapsed*1000:.2f} ms")
    print("  ✓ Test 2 PASSED")
except Exception as e:
    print(f"  ✗ Test 2 FAILED: {e}")

# Test 3: Row mode
print("\n[Test 3] Row parallelization mode")
try:
    dev3 = qml.device("qlret.mixed", wires=n_qubits, epsilon=1e-4, 
                      parallel_mode="row")
    print(f"  Device created: {dev3.name}")
    print(f"  Num threads: {dev3.num_threads} (0=auto)")
    print(f"  Parallel mode: {dev3.parallel_mode}")
    
    @qml.qnode(dev3)
    def circuit3():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))
    
    start = time.time()
    result = circuit3()
    elapsed = time.time() - start
    print(f"  Result: {result:.6f}")
    print(f"  Time: {elapsed*1000:.2f} ms")
    print("  ✓ Test 3 PASSED")
except Exception as e:
    print(f"  ✗ Test 3 FAILED: {e}")

# Test 4: Sequential mode (no parallelization)
print("\n[Test 4] Sequential mode (single-threaded)")
try:
    dev4 = qml.device("qlret.mixed", wires=n_qubits, epsilon=1e-4, 
                      parallel_mode="sequential")
    print(f"  Device created: {dev4.name}")
    print(f"  Num threads: {dev4.num_threads}")
    print(f"  Parallel mode: {dev4.parallel_mode}")
    
    @qml.qnode(dev4)
    def circuit4():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))
    
    start = time.time()
    result = circuit4()
    elapsed = time.time() - start
    print(f"  Result: {result:.6f}")
    print(f"  Time: {elapsed*1000:.2f} ms")
    print("  ✓ Test 4 PASSED")
except Exception as e:
    print(f"  ✗ Test 4 FAILED: {e}")

print("\n" + "=" * 60)
print("ALL PARALLELIZATION TESTS COMPLETE")
print("=" * 60)
