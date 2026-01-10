#!/usr/bin/env python3
"""Test Kraus operator support for noise channels in LRET."""

import pennylane as qml
import numpy as np
from qlret import QLRETDevice

print("="*60)
print("Testing LRET Kraus Operator Support for Noise Channels")
print("="*60)

# Test LRET device with DepolarizingChannel
dev = QLRETDevice(wires=2, shots=None)

@qml.qnode(dev)
def circuit_lret():
    qml.Hadamard(wires=0)
    qml.DepolarizingChannel(0.1, wires=0)
    qml.CNOT(wires=[0, 1])
    qml.DepolarizingChannel(0.05, wires=1)
    return qml.expval(qml.PauliZ(0))

# Test default.mixed device for comparison  
dev_baseline = qml.device('default.mixed', wires=2)

@qml.qnode(dev_baseline)
def circuit_baseline():
    qml.Hadamard(wires=0)
    qml.DepolarizingChannel(0.1, wires=0)
    qml.CNOT(wires=[0, 1])
    qml.DepolarizingChannel(0.05, wires=1)
    return qml.expval(qml.PauliZ(0))

print("\n1. Testing DepolarizingChannel support...")
try:
    result_lret = circuit_lret()
    print(f"   LRET result: {result_lret:.6f}")
except Exception as e:
    print(f"   LRET ERROR: {e}")
    result_lret = None

result_baseline = circuit_baseline()
print(f"   default.mixed result: {result_baseline:.6f}")

if result_lret is not None:
    diff = abs(result_lret - result_baseline)
    print(f"   Difference: {diff:.6f}")
    print(f"   Match: {'YES (< 0.01)' if diff < 0.01 else 'NO'}")

# Test 2: AmplitudeDamping
print("\n2. Testing AmplitudeDampingChannel support...")

@qml.qnode(dev)
def circuit_lret_ad():
    qml.Hadamard(wires=0)
    qml.AmplitudeDamping(0.2, wires=0)
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev_baseline)
def circuit_baseline_ad():
    qml.Hadamard(wires=0)
    qml.AmplitudeDamping(0.2, wires=0)
    return qml.expval(qml.PauliZ(0))

try:
    result_lret_ad = circuit_lret_ad()
    print(f"   LRET result: {result_lret_ad:.6f}")
except Exception as e:
    print(f"   LRET ERROR: {e}")
    result_lret_ad = None

result_baseline_ad = circuit_baseline_ad()
print(f"   default.mixed result: {result_baseline_ad:.6f}")

if result_lret_ad is not None:
    diff = abs(result_lret_ad - result_baseline_ad)
    print(f"   Difference: {diff:.6f}")
    print(f"   Match: {'YES (< 0.01)' if diff < 0.01 else 'NO'}")

# Test 3: PhaseDamping
print("\n3. Testing PhaseDampingChannel support...")

@qml.qnode(dev)
def circuit_lret_pd():
    qml.Hadamard(wires=0)
    qml.PhaseDamping(0.15, wires=0)
    return qml.expval(qml.PauliX(0))

@qml.qnode(dev_baseline)
def circuit_baseline_pd():
    qml.Hadamard(wires=0)
    qml.PhaseDamping(0.15, wires=0)
    return qml.expval(qml.PauliX(0))

try:
    result_lret_pd = circuit_lret_pd()
    print(f"   LRET result: {result_lret_pd:.6f}")
except Exception as e:
    print(f"   LRET ERROR: {e}")
    result_lret_pd = None

result_baseline_pd = circuit_baseline_pd()
print(f"   default.mixed result: {result_baseline_pd:.6f}")

if result_lret_pd is not None:
    diff = abs(result_lret_pd - result_baseline_pd)
    print(f"   Difference: {diff:.6f}")
    print(f"   Match: {'YES (< 0.01)' if diff < 0.01 else 'NO'}")

print("\n" + "="*60)
print("Test complete!")
print("="*60)
