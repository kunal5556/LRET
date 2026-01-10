#!/usr/bin/env python3
"""Debug gradient computation."""

import pennylane as qml
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from qlret import QLRETDevice, set_use_fallback
set_use_fallback(True)

dev = QLRETDevice(wires=2, shots=None, epsilon=1e-4)

# Test 1: Try with finite differences
print("Test 1: Using finite difference gradient...")
@qml.qnode(dev, diff_method="finite-diff")
def grad_circuit_fd(theta):
    qml.RY(theta, wires=0)
    return qml.expval(qml.PauliZ(0))

theta_val = np.pi/4
result = grad_circuit_fd(theta_val)
print(f'Result: {result}')

try:
    grad_fn = qml.grad(grad_circuit_fd)
    grad = grad_fn(theta_val)
    print(f'Grad: {grad}')
    print(f'Expected: {-np.sin(np.pi/4)}')
except Exception as e:
    print(f'Error: {e}')

# Test 2: Manual parameter-shift
print("\nTest 2: Manual parameter-shift gradient...")
@qml.qnode(dev)
def basic_circuit(theta):
    qml.RY(theta, wires=0)
    return qml.expval(qml.PauliZ(0))

shift = np.pi / 2
f_plus = basic_circuit(theta_val + shift)
f_minus = basic_circuit(theta_val - shift)
manual_grad = (f_plus - f_minus) / 2
print(f'f(θ+π/2) = {f_plus}')
print(f'f(θ-π/2) = {f_minus}')
print(f'Manual grad: {manual_grad}')
print(f'Expected: {-np.sin(theta_val)}')
