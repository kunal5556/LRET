#!/usr/bin/env python3
"""Test script to verify QLRET PennyLane device fixes.

This tests:
1. Pure-Python fallback simulator
2. PennyLane device integration
3. Gradient computation
4. Multiple measurement types
"""

import sys
import numpy as np

print("=" * 60)
print("QLRET PennyLane Device Verification Test")
print("=" * 60)

# Test 1: Import check
print("\n[1] Testing imports...")
try:
    import pennylane as qml
    print(f"    ✓ PennyLane version: {qml.__version__}")
except ImportError as e:
    print(f"    ✗ PennyLane import failed: {e}")
    sys.exit(1)

try:
    from qlret import QLRETDevice, simulate_json, set_use_fallback
    print("    ✓ QLRET package imported successfully")
except ImportError as e:
    print(f"    ✗ QLRET import failed: {e}")
    sys.exit(1)

# Test 2: Fallback simulator
print("\n[2] Testing fallback simulator...")
set_use_fallback(True)  # Force fallback mode

test_circuit = {
    "circuit": {
        "num_qubits": 2,
        "operations": [
            {"name": "H", "wires": [0]},
            {"name": "CNOT", "wires": [0, 1]},
        ],
        "observables": [
            {"type": "PAULI", "operator": "Z", "wires": [0], "coefficient": 1.0},
        ],
    },
    "config": {"epsilon": 1e-4},
}

try:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = simulate_json(test_circuit)
    print(f"    ✓ Fallback simulator executed successfully")
    print(f"      Status: {result.get('status')}")
    print(f"      Backend: {result.get('backend')}")
    print(f"      Expectation values: {result.get('expectation_values')}")
except Exception as e:
    print(f"    ✗ Fallback simulator failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: PennyLane device instantiation
print("\n[3] Testing QLRETDevice instantiation...")
try:
    dev = QLRETDevice(wires=4, shots=None, epsilon=1e-4)
    print(f"    ✓ Device created: {dev.name}")
    print(f"      Wires: {dev.num_wires}")
    print(f"      Shots: {dev.shots}")
except Exception as e:
    print(f"    ✗ Device creation failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Simple circuit execution
print("\n[4] Testing simple circuit execution...")
try:
    dev = QLRETDevice(wires=2, shots=None, epsilon=1e-4)
    
    @qml.qnode(dev)
    def simple_circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = simple_circuit()
    print(f"    ✓ Circuit executed successfully")
    print(f"      Result: {result}")
    # Bell state should give Z expectation ~0
    expected = 0.0
    if abs(result - expected) < 0.1:
        print(f"      ✓ Result correct (expected ~{expected})")
    else:
        print(f"      ⚠ Result unexpected (expected ~{expected})")
except Exception as e:
    print(f"    ✗ Circuit execution failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Parametric circuit
print("\n[5] Testing parametric circuit...")
try:
    dev = QLRETDevice(wires=2, shots=None, epsilon=1e-4)
    
    @qml.qnode(dev)
    def param_circuit(theta):
        qml.RY(theta, wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = []
        for theta in [0.0, np.pi/4, np.pi/2, np.pi]:
            results.append(param_circuit(theta))
    
    print(f"    ✓ Parametric circuit executed successfully")
    print(f"      θ=0:    Z = {results[0]:.4f} (expected 1.0)")
    print(f"      θ=π/4:  Z = {results[1]:.4f} (expected ~0.707)")
    print(f"      θ=π/2:  Z = {results[2]:.4f} (expected 0.0)")
    print(f"      θ=π:    Z = {results[3]:.4f} (expected -1.0)")
except Exception as e:
    print(f"    ✗ Parametric circuit failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Gradient computation (manual parameter-shift)
print("\n[6] Testing gradient computation (parameter-shift)...")
try:
    dev = QLRETDevice(wires=2, shots=None, epsilon=1e-4)
    
    @qml.qnode(dev)
    def grad_circuit(theta):
        qml.RY(theta, wires=0)
        return qml.expval(qml.PauliZ(0))
    
    theta_val = np.pi/4
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Manual parameter-shift gradient
        shift = np.pi / 2
        f_plus = grad_circuit(theta_val + shift)
        f_minus = grad_circuit(theta_val - shift)
        grad = (f_plus - f_minus) / 2
        result = grad_circuit(theta_val)
    
    print(f"    ✓ Gradient computed successfully")
    print(f"      θ = π/4")
    print(f"      ⟨Z⟩ = {result:.4f}")
    print(f"      d⟨Z⟩/dθ = {grad:.4f}")
    # For RY(θ)|0⟩, ⟨Z⟩ = cos(θ), d⟨Z⟩/dθ = -sin(θ)
    expected_grad = -np.sin(np.pi/4)
    print(f"      Expected gradient: {expected_grad:.4f}")
    if abs(grad - expected_grad) < 0.01:
        print(f"      ✓ Gradient correct!")
    else:
        print(f"      ⚠ Gradient differs from expected")
except Exception as e:
    print(f"    ✗ Gradient computation failed: {e}")
    import traceback
    traceback.print_exc()

# Test 7: Multiple observables
print("\n[7] Testing multiple observables...")
try:
    dev = QLRETDevice(wires=2, shots=None, epsilon=1e-4)
    
    @qml.qnode(dev)
    def multi_obs_circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z0, z1 = multi_obs_circuit()
    
    print(f"    ✓ Multiple observables executed")
    print(f"      ⟨Z₀⟩ = {z0:.4f}")
    print(f"      ⟨Z₁⟩ = {z1:.4f}")
except Exception as e:
    print(f"    ✗ Multiple observables failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)
print("\nNotes:")
print("- Using pure-Python fallback simulator (not low-rank)")
print("- Build native C++ backend for production performance")
print("- Run: cd build && cmake .. && make")
