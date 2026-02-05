#!/usr/bin/env python3
"""
Debug script to trace rank evolution in baseline vs optimized.
This creates progressively more complex circuits to find the divergence point.
"""

import json
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(r"D:\LRET\validation")
BASELINE_EXE = BASE_DIR / "baseline" / "quantum_sim.exe"
OPTIMIZED_EXE = BASE_DIR / "optimized" / "quantum_sim.exe"
TEST_DIR = BASE_DIR / "test_circuits"


def depolarizing_kraus(p=0.01):
    """Generate depolarizing channel Kraus operators for p=probability."""
    import math
    # K0 = sqrt(1 - 3p/4) * I
    # K1 = sqrt(p/4) * X
    # K2 = sqrt(p/4) * Y  
    # K3 = sqrt(p/4) * Z
    
    c0 = math.sqrt(1 - 3*p/4)
    c1 = math.sqrt(p/4)
    
    K0 = {"real": [[c0, 0.0], [0.0, c0]], "imag": [[0.0, 0.0], [0.0, 0.0]]}
    K1 = {"real": [[0.0, c1], [c1, 0.0]], "imag": [[0.0, 0.0], [0.0, 0.0]]}
    K2 = {"real": [[0.0, 0.0], [0.0, 0.0]], "imag": [[0.0, -c1], [c1, 0.0]]}
    K3 = {"real": [[c1, 0.0], [0.0, -c1]], "imag": [[0.0, 0.0], [0.0, 0.0]]}
    
    return [K0, K1, K2, K3]


def run_simulator(exe_path, circuit_json_path):
    """Run simulator and return final_rank."""
    cmd = [
        str(exe_path),
        "--input-json", str(circuit_json_path),
        "--allow-swap",
        "--non-interactive"
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120
    )
    
    try:
        output = result.stdout
        data = json.loads(output)
        return data.get("final_rank", -1)
    except:
        print(f"Failed to parse output: {result.stdout[:200]}")
        print(f"Stderr: {result.stderr[:200]}")
        return -1


def create_circuit(num_qubits, noise_ops, gates):
    """Create a circuit with specified noise and gates."""
    operations = []
    
    for op in gates:
        if op["type"] == "gate":
            operations.append({"name": op["name"], "wires": op["wires"]})
        elif op["type"] == "noise":
            kraus = depolarizing_kraus(op.get("p", 0.01))
            operations.append({
                "name": "KRAUS",
                "wires": [op["qubit"]],
                "kraus_operators": kraus
            })
    
    return {"circuit": {"num_qubits": num_qubits, "operations": operations}}


def test_progression():
    """Test progressively complex circuits to find divergence."""
    
    print("=" * 70)
    print("RANK DIVERGENCE INVESTIGATION")
    print("=" * 70)
    print()
    
    test_cases = [
        # Test 1: Single H gate, single noise
        {
            "name": "1 noise op",
            "qubits": 3,
            "ops": [
                {"type": "gate", "name": "H", "wires": [0]},
                {"type": "noise", "qubit": 0, "p": 0.01},
            ]
        },
        # Test 2: Two noise ops
        {
            "name": "2 noise ops",
            "qubits": 3,
            "ops": [
                {"type": "gate", "name": "H", "wires": [0]},
                {"type": "noise", "qubit": 0, "p": 0.01},
                {"type": "gate", "name": "CNOT", "wires": [0, 1]},
                {"type": "noise", "qubit": 1, "p": 0.01},
            ]
        },
        # Test 3: Three noise ops
        {
            "name": "3 noise ops",
            "qubits": 4,
            "ops": [
                {"type": "gate", "name": "H", "wires": [0]},
                {"type": "noise", "qubit": 0, "p": 0.01},
                {"type": "gate", "name": "CNOT", "wires": [0, 1]},
                {"type": "noise", "qubit": 1, "p": 0.01},
                {"type": "gate", "name": "CNOT", "wires": [1, 2]},
                {"type": "noise", "qubit": 2, "p": 0.01},
            ]
        },
        # Test 4: Four noise ops
        {
            "name": "4 noise ops",
            "qubits": 4,
            "ops": [
                {"type": "gate", "name": "H", "wires": [0]},
                {"type": "noise", "qubit": 0, "p": 0.01},
                {"type": "gate", "name": "CNOT", "wires": [0, 1]},
                {"type": "noise", "qubit": 1, "p": 0.01},
                {"type": "gate", "name": "CNOT", "wires": [1, 2]},
                {"type": "noise", "qubit": 2, "p": 0.01},
                {"type": "gate", "name": "CNOT", "wires": [2, 3]},
                {"type": "noise", "qubit": 3, "p": 0.01},
            ]
        },
        # Test 5: Five noise ops
        {
            "name": "5 noise ops",
            "qubits": 4,
            "ops": [
                {"type": "gate", "name": "H", "wires": [0]},
                {"type": "noise", "qubit": 0, "p": 0.01},
                {"type": "gate", "name": "CNOT", "wires": [0, 1]},
                {"type": "noise", "qubit": 1, "p": 0.01},
                {"type": "gate", "name": "CNOT", "wires": [1, 2]},
                {"type": "noise", "qubit": 2, "p": 0.01},
                {"type": "gate", "name": "CNOT", "wires": [2, 3]},
                {"type": "noise", "qubit": 3, "p": 0.01},
                {"type": "gate", "name": "H", "wires": [1]},
                {"type": "noise", "qubit": 1, "p": 0.01},
            ]
        },
        # Test 6: Six noise ops
        {
            "name": "6 noise ops",
            "qubits": 5,
            "ops": [
                {"type": "gate", "name": "H", "wires": [0]},
                {"type": "noise", "qubit": 0, "p": 0.01},
                {"type": "gate", "name": "CNOT", "wires": [0, 1]},
                {"type": "noise", "qubit": 1, "p": 0.01},
                {"type": "gate", "name": "CNOT", "wires": [1, 2]},
                {"type": "noise", "qubit": 2, "p": 0.01},
                {"type": "gate", "name": "CNOT", "wires": [2, 3]},
                {"type": "noise", "qubit": 3, "p": 0.01},
                {"type": "gate", "name": "CNOT", "wires": [3, 4]},
                {"type": "noise", "qubit": 4, "p": 0.01},
                {"type": "gate", "name": "H", "wires": [2]},
                {"type": "noise", "qubit": 2, "p": 0.01},
            ]
        },
    ]
    
    print(f"{'Test Case':<20} {'Baseline':<12} {'Optimized':<12} {'Match':<8}")
    print("-" * 55)
    
    first_divergence = None
    
    for i, test in enumerate(test_cases):
        # Create circuit
        circuit = create_circuit(test["qubits"], 0, test["ops"])
        
        # Save to file
        circuit_path = TEST_DIR / f"debug_trace_{i+1}.json"
        with open(circuit_path, "w") as f:
            json.dump(circuit, f, indent=2)
        
        # Run both simulators
        baseline_rank = run_simulator(BASELINE_EXE, circuit_path)
        optimized_rank = run_simulator(OPTIMIZED_EXE, circuit_path)
        
        match = "✓" if baseline_rank == optimized_rank else "✗"
        
        print(f"{test['name']:<20} {baseline_rank:<12} {optimized_rank:<12} {match:<8}")
        
        if baseline_rank != optimized_rank and first_divergence is None:
            first_divergence = test
            print(f"\n>>> FIRST DIVERGENCE at: {test['name']}")
            print(f"    Circuit saved to: {circuit_path}")
    
    print()
    
    if first_divergence:
        print("=" * 70)
        print("DIVERGENCE DETECTED!")
        print("=" * 70)
    else:
        print("All tests MATCH - no divergence found with these circuits")
    
    return first_divergence


def main():
    print("Testing rank evolution between baseline and optimized...")
    print()
    
    # First verify both executables exist
    if not BASELINE_EXE.exists():
        print(f"ERROR: Baseline not found: {BASELINE_EXE}")
        sys.exit(1)
    if not OPTIMIZED_EXE.exists():
        print(f"ERROR: Optimized not found: {OPTIMIZED_EXE}")
        sys.exit(1)
    
    TEST_DIR.mkdir(exist_ok=True)
    
    # Run progression tests
    divergence = test_progression()
    
    if divergence:
        print(f"\nNext step: Analyze the divergent circuit in detail")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
