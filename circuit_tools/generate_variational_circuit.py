#!/usr/bin/env python3
"""
Generate Variational Quantum Circuit for LRET
Usage: python3 generate_variational_circuit.py <num_qubits> <num_layers> <output_file>
"""
import json
import numpy as np
import sys

def variational_circuit(num_qubits, num_layers, params=None, seed=42):
    """Generate a variational quantum circuit"""
    np.random.seed(seed)
    
    if params is None:
        params = np.random.random((num_layers, num_qubits, 3)) * 2 * np.pi
    
    operations = []
    
    for layer in range(num_layers):
        # Rotation layer
        for q in range(num_qubits):
            operations.append({
                "name": "RX",
                "wires": [q],
                "params": [float(params[layer, q, 0])]
            })
            operations.append({
                "name": "RY",
                "wires": [q],
                "params": [float(params[layer, q, 1])]
            })
            operations.append({
                "name": "RZ",
                "wires": [q],
                "params": [float(params[layer, q, 2])]
            })
        
        # Entangling layer
        for q in range(num_qubits - 1):
            operations.append({
                "name": "CNOT",
                "wires": [q, q + 1]
            })
        # Circular entanglement
        if num_qubits > 2:
            operations.append({
                "name": "CNOT",
                "wires": [num_qubits - 1, 0]
            })
    
    circuit = {
        "circuit": {
            "num_qubits": num_qubits,
            "operations": operations,
            "observables": [
                {"type": "PAULI", "operator": "Z", "wires": [0], "coefficient": 1.0}
            ]
        },
        "config": {
            "epsilon": 1e-4,
            "initial_rank": 1,
            "shots": 1000
        }
    }
    
    return circuit

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 generate_variational_circuit.py <num_qubits> [num_layers] [output_file]")
        print("\nExample: python3 generate_variational_circuit.py 6 4 circuits/vqe_6q.json")
        sys.exit(1)
    
    num_qubits = int(sys.argv[1])
    num_layers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    output_file = sys.argv[3] if len(sys.argv) > 3 else f"variational_{num_qubits}q_{num_layers}l.json"
    
    circuit = variational_circuit(num_qubits, num_layers)
    
    with open(output_file, 'w') as f:
        json.dump(circuit, f, indent=2)
    
    num_gates = len(circuit['circuit']['operations'])
    print(f"✅ Generated {num_qubits}-qubit, {num_layers}-layer variational circuit")
    print(f"   Gates: {num_gates}")
    print(f"   File: {output_file}")
