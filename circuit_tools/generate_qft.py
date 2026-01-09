#!/usr/bin/env python3
"""
Generate Quantum Fourier Transform Circuit for LRET
Usage: python3 generate_qft.py <num_qubits> <output_file>
"""
import json
import numpy as np
import sys

def qft_circuit(num_qubits):
    """Generate a Quantum Fourier Transform circuit"""
    operations = []
    
    for target_qubit in range(num_qubits):
        # Hadamard gate
        operations.append({
            "name": "H",
            "wires": [target_qubit]
        })
        
        # Controlled rotations
        for control_qubit in range(target_qubit + 1, num_qubits):
            angle = np.pi / (2 ** (control_qubit - target_qubit))
            # CPhase gate (equivalent to controlled-RZ)
            operations.append({
                "name": "CPHASE",
                "wires": [control_qubit, target_qubit],
                "params": [float(angle)]
            })
    
    # Swap qubits to reverse order
    for i in range(num_qubits // 2):
        operations.append({
            "name": "SWAP",
            "wires": [i, num_qubits - 1 - i]
        })
    
    circuit = {
        "circuit": {
            "num_qubits": num_qubits,
            "operations": operations,
            "observables": [
                {"type": "PAULI", "operator": "Z", "wires": [q], "coefficient": 1.0}
                for q in range(num_qubits)
            ]
        },
        "config": {
            "epsilon": 1e-4,
            "initial_rank": 2,
            "shots": 1000
        }
    }
    
    return circuit

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 generate_qft.py <num_qubits> [output_file]")
        print("\nExample: python3 generate_qft.py 8 circuits/qft_8q.json")
        sys.exit(1)
    
    num_qubits = int(sys.argv[1])
    output_file = sys.argv[2] if len(sys.argv) > 2 else f"qft_{num_qubits}q.json"
    
    circuit = qft_circuit(num_qubits)
    
    with open(output_file, 'w') as f:
        json.dump(circuit, f, indent=2)
    
    num_gates = len(circuit['circuit']['operations'])
    print(f"✅ Generated {num_qubits}-qubit QFT circuit")
    print(f"   Gates: {num_gates}")
    print(f"   File: {output_file}")
