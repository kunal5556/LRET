#!/usr/bin/env python3
"""
LRET High-Rank Circuit Generator - Phase D
Generates circuits with noise channels that grow density matrix rank

The key insight is that noise channels (depolarizing, amplitude damping) 
expand the rank of the density matrix L. When rank exceeds 32, the row-parallelism
optimization kicks in, providing larger speedups.

Noise Format: LRET accepts KRAUS operations with explicit Kraus matrices:
{
    "name": "KRAUS",
    "wires": [0],
    "kraus_operators": [
        {"real": [[...]], "imag": [[...]]},
        ...
    ]
}

Usage:
    python generate_noisy_circuits.py --min-qubits 6 --max-qubits 12
"""

import json
import os
import random
import math
import numpy as np
import argparse
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime

# =============================================================================
# Noise Kraus Operators
# =============================================================================

def depolarizing_kraus(p: float) -> List[np.ndarray]:
    """
    Depolarizing channel Kraus operators.
    E(rho) = (1-p)*rho + (p/3)*(X*rho*X + Y*rho*Y + Z*rho*Z)
    
    Has 4 Kraus operators -> grows rank by 4x
    """
    # Identity, X, Y, Z
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Kraus operators
    K0 = np.sqrt(1 - p) * I
    K1 = np.sqrt(p / 3) * X
    K2 = np.sqrt(p / 3) * Y
    K3 = np.sqrt(p / 3) * Z
    
    return [K0, K1, K2, K3]


def amplitude_damping_kraus(gamma: float) -> List[np.ndarray]:
    """
    Amplitude damping channel Kraus operators.
    Models energy decay (T1).
    
    Has 2 Kraus operators -> grows rank by 2x
    """
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    
    return [K0, K1]


def phase_damping_kraus(lambd: float) -> List[np.ndarray]:
    """
    Phase damping channel Kraus operators.
    Models dephasing (T2).
    
    Has 2 Kraus operators -> grows rank by 2x
    """
    K0 = np.array([[1, 0], [0, np.sqrt(1 - lambd)]], dtype=complex)
    K1 = np.array([[0, 0], [0, np.sqrt(lambd)]], dtype=complex)
    
    return [K0, K1]


def bit_flip_kraus(p: float) -> List[np.ndarray]:
    """
    Bit flip channel Kraus operators.
    
    Has 2 Kraus operators -> grows rank by 2x
    """
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    
    K0 = np.sqrt(1 - p) * I
    K1 = np.sqrt(p) * X
    
    return [K0, K1]


def phase_flip_kraus(p: float) -> List[np.ndarray]:
    """
    Phase flip channel Kraus operators.
    
    Has 2 Kraus operators -> grows rank by 2x
    """
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    K0 = np.sqrt(1 - p) * I
    K1 = np.sqrt(p) * Z
    
    return [K0, K1]


def kraus_to_json(kraus_ops: List[np.ndarray]) -> List[Dict]:
    """Convert Kraus operators to JSON format."""
    result = []
    for K in kraus_ops:
        result.append({
            "real": K.real.tolist(),
            "imag": K.imag.tolist()
        })
    return result


def noise_op_json(qubit: int, noise_type: str, param: float) -> Dict:
    """Create a noise operation in JSON format."""
    if noise_type == "depolarizing":
        kraus = depolarizing_kraus(param)
    elif noise_type == "amplitude_damping":
        kraus = amplitude_damping_kraus(param)
    elif noise_type == "phase_damping":
        kraus = phase_damping_kraus(param)
    elif noise_type == "bit_flip":
        kraus = bit_flip_kraus(param)
    elif noise_type == "phase_flip":
        kraus = phase_flip_kraus(param)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return {
        "name": "KRAUS",
        "wires": [qubit],
        "kraus_operators": kraus_to_json(kraus)
    }


# =============================================================================
# Gate Helpers
# =============================================================================

def gate_op(name: str, wires: List[int], params: List[float] = None) -> Dict:
    """Create gate operation JSON."""
    d = {"name": name, "wires": wires}
    if params:
        d["params"] = params
    return d


# =============================================================================
# High-Rank Circuit Generators
# =============================================================================

class NoisyCircuitGenerator:
    """Generate circuits with noise that grows rank."""
    
    def __init__(self, seed: int = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_noisy_random(self, n_qubits: int, depth: int,
                               noise_type: str = "depolarizing",
                               noise_rate: float = 0.01) -> Tuple[Dict, Dict]:
        """
        Generate random circuit with noise after each gate.
        
        Expected rank growth:
        - Depolarizing (4 Kraus): rank grows by ~4x per noise op
        - Amplitude/Phase (2 Kraus): rank grows by ~2x per noise op
        
        After d noise operations: rank ~ initial_rank * growth_factor^d
        With truncation, rank is bounded by epsilon threshold.
        """
        ops = []
        
        for layer in range(depth):
            # Single qubit gates
            for q in range(n_qubits):
                if random.random() < 0.7:
                    gate = random.choice(["H", "X", "Y", "Z", "S", "T"])
                    ops.append(gate_op(gate, [q]))
                    # Add noise after gate
                    ops.append(noise_op_json(q, noise_type, noise_rate))
                else:
                    gate = random.choice(["RX", "RY", "RZ"])
                    angle = random.uniform(0, 2 * math.pi)
                    ops.append(gate_op(gate, [q], [angle]))
                    ops.append(noise_op_json(q, noise_type, noise_rate))
            
            # Two-qubit layer
            for q in range(0, n_qubits - 1, 2):
                if random.random() < 0.3:
                    ops.append(gate_op("CNOT", [q, q + 1]))
                    # Noise on both qubits
                    ops.append(noise_op_json(q, noise_type, noise_rate))
                    ops.append(noise_op_json(q + 1, noise_type, noise_rate))
        
        circuit = {
            "num_qubits": n_qubits,
            "operations": ops
        }
        
        # Count noise operations
        noise_count = sum(1 for op in ops if op["name"] == "KRAUS")
        
        metadata = {
            "type": "noisy",
            "subtype": f"random_{noise_type}",
            "n_qubits": n_qubits,
            "depth": depth,
            "noise_type": noise_type,
            "noise_rate": noise_rate,
            "noise_count": noise_count,
            "n_operations": len(ops),
            # Theoretical rank growth (before truncation)
            "expected_rank_factor": 4 if noise_type == "depolarizing" else 2
        }
        
        return circuit, metadata
    
    def generate_ghz_noisy(self, n_qubits: int, 
                            noise_type: str = "depolarizing",
                            noise_rate: float = 0.01) -> Tuple[Dict, Dict]:
        """
        GHZ state with noise - creates highly entangled + noisy state.
        """
        ops = []
        
        # Create GHZ state with noise
        ops.append(gate_op("H", [0]))
        ops.append(noise_op_json(0, noise_type, noise_rate))
        
        for q in range(n_qubits - 1):
            ops.append(gate_op("CNOT", [q, q + 1]))
            ops.append(noise_op_json(q, noise_type, noise_rate))
            ops.append(noise_op_json(q + 1, noise_type, noise_rate))
        
        circuit = {
            "num_qubits": n_qubits,
            "operations": ops
        }
        
        noise_count = sum(1 for op in ops if op["name"] == "KRAUS")
        
        metadata = {
            "type": "noisy",
            "subtype": f"ghz_{noise_type}",
            "n_qubits": n_qubits,
            "noise_type": noise_type,
            "noise_rate": noise_rate,
            "noise_count": noise_count,
            "n_operations": len(ops)
        }
        
        return circuit, metadata
    
    def generate_vqe_noisy(self, n_qubits: int, layers: int = 2,
                           noise_type: str = "depolarizing",
                           noise_rate: float = 0.01) -> Tuple[Dict, Dict]:
        """
        VQE ansatz with realistic noise - practical application circuit.
        """
        ops = []
        
        for layer in range(layers):
            # Rotation layer with noise
            for q in range(n_qubits):
                ops.append(gate_op("RY", [q], [random.uniform(0, 2*math.pi)]))
                ops.append(noise_op_json(q, noise_type, noise_rate))
                ops.append(gate_op("RZ", [q], [random.uniform(0, 2*math.pi)]))
                ops.append(noise_op_json(q, noise_type, noise_rate))
            
            # Entangling layer with noise
            for q in range(n_qubits - 1):
                ops.append(gate_op("CNOT", [q, q + 1]))
                ops.append(noise_op_json(q, noise_type, noise_rate))
                ops.append(noise_op_json(q + 1, noise_type, noise_rate))
        
        # Final rotation
        for q in range(n_qubits):
            ops.append(gate_op("RY", [q], [random.uniform(0, 2*math.pi)]))
            ops.append(noise_op_json(q, noise_type, noise_rate))
        
        circuit = {
            "num_qubits": n_qubits,
            "operations": ops
        }
        
        noise_count = sum(1 for op in ops if op["name"] == "KRAUS")
        
        metadata = {
            "type": "noisy",
            "subtype": f"vqe_{noise_type}",
            "n_qubits": n_qubits,
            "layers": layers,
            "noise_type": noise_type,
            "noise_rate": noise_rate,
            "noise_count": noise_count,
            "n_operations": len(ops)
        }
        
        return circuit, metadata
    
    def generate_noise_stress(self, n_qubits: int, 
                               noise_type: str = "depolarizing",
                               noise_rate: float = 0.05) -> Tuple[Dict, Dict]:
        """
        Circuit specifically designed to maximize rank growth.
        Heavy noise after every operation.
        """
        ops = []
        
        # Initial superposition with noise
        for q in range(n_qubits):
            ops.append(gate_op("H", [q]))
            ops.append(noise_op_json(q, noise_type, noise_rate))
        
        # Dense entanglement + noise layers
        for layer in range(3):
            # CNOT ladder
            for q in range(n_qubits - 1):
                ops.append(gate_op("CNOT", [q, q + 1]))
                for qn in range(n_qubits):
                    ops.append(noise_op_json(qn, noise_type, noise_rate))
            
            # Reverse ladder
            for q in range(n_qubits - 2, -1, -1):
                ops.append(gate_op("CNOT", [q + 1, q]))
                for qn in range(n_qubits):
                    ops.append(noise_op_json(qn, noise_type, noise_rate))
        
        circuit = {
            "num_qubits": n_qubits,
            "operations": ops
        }
        
        noise_count = sum(1 for op in ops if op["name"] == "KRAUS")
        
        metadata = {
            "type": "noisy",
            "subtype": f"stress_{noise_type}",
            "n_qubits": n_qubits,
            "noise_type": noise_type,
            "noise_rate": noise_rate,
            "noise_count": noise_count,
            "n_operations": len(ops)
        }
        
        return circuit, metadata
    
    def generate_mixed_noise(self, n_qubits: int, depth: int = 10) -> Tuple[Dict, Dict]:
        """
        Circuit with mixed noise types - realistic scenario.
        """
        ops = []
        
        noise_types = ["depolarizing", "amplitude_damping", "phase_damping"]
        noise_rates = {"depolarizing": 0.01, "amplitude_damping": 0.02, "phase_damping": 0.03}
        
        for layer in range(depth):
            for q in range(n_qubits):
                # Random gate
                if random.random() < 0.5:
                    ops.append(gate_op("H", [q]))
                else:
                    ops.append(gate_op("RY", [q], [random.uniform(0, 2*math.pi)]))
                
                # Random noise type
                nt = random.choice(noise_types)
                ops.append(noise_op_json(q, nt, noise_rates[nt]))
            
            # Entangling
            if layer % 2 == 0:
                for q in range(0, n_qubits - 1, 2):
                    ops.append(gate_op("CNOT", [q, q + 1]))
                    ops.append(noise_op_json(q, "depolarizing", 0.01))
        
        circuit = {
            "num_qubits": n_qubits,
            "operations": ops
        }
        
        noise_count = sum(1 for op in ops if op["name"] == "KRAUS")
        
        metadata = {
            "type": "noisy",
            "subtype": "mixed_noise",
            "n_qubits": n_qubits,
            "depth": depth,
            "noise_count": noise_count,
            "n_operations": len(ops)
        }
        
        return circuit, metadata


# =============================================================================
# Suite Generator
# =============================================================================

class NoisySuiteGenerator:
    """Generate complete noisy circuit test suite."""
    
    def __init__(self, output_dir: str, seed: int = 42):
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.manifest = []
        self.generator = NoisyCircuitGenerator(seed)
    
    def generate_suite(self, min_qubits: int = 6, max_qubits: int = 12,
                       instances: int = 2) -> int:
        """Generate comprehensive noisy circuit suite."""
        print("=" * 70)
        print("LRET High-Rank Circuit Generator - Phase D")
        print("=" * 70)
        print(f"Output: {self.output_dir}")
        print(f"Qubit range: {min_qubits}-{max_qubits}")
        print(f"Seed: {self.seed}")
        print()
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        circuits = []
        
        noise_types = ["depolarizing", "amplitude_damping", "phase_damping"]
        noise_rates = [0.01, 0.02, 0.05]
        
        # 1. Noisy random circuits
        print("Generating noisy random circuits...")
        for n in range(min_qubits, max_qubits + 1, 2):
            for nt in noise_types:
                for nr in noise_rates:
                    for i in range(instances):
                        random.seed(self.seed + n * 1000 + i)
                        np.random.seed(self.seed + n * 1000 + i)
                        circuit, meta = self.generator.generate_noisy_random(n, 10, nt, nr)
                        meta["instance"] = i
                        circuits.append((circuit, meta))
        
        # 2. Noisy GHZ states
        print("Generating noisy GHZ circuits...")
        for n in range(min_qubits, max_qubits + 1, 2):
            for nt in noise_types:
                for nr in [0.01, 0.05]:
                    circuit, meta = self.generator.generate_ghz_noisy(n, nt, nr)
                    circuits.append((circuit, meta))
        
        # 3. Noisy VQE
        print("Generating noisy VQE circuits...")
        for n in range(min_qubits, max_qubits + 1, 2):
            for layers in [2, 3]:
                for nr in [0.01, 0.03]:
                    random.seed(self.seed + n * 2000 + layers)
                    np.random.seed(self.seed + n * 2000 + layers)
                    circuit, meta = self.generator.generate_vqe_noisy(n, layers, "depolarizing", nr)
                    circuits.append((circuit, meta))
        
        # 4. Noise stress tests
        print("Generating noise stress circuits...")
        for n in range(min_qubits, min(max_qubits, 10) + 1, 2):  # Smaller for stress
            for nt in ["depolarizing", "amplitude_damping"]:
                for nr in [0.02, 0.05]:
                    circuit, meta = self.generator.generate_noise_stress(n, nt, nr)
                    circuits.append((circuit, meta))
        
        # 5. Mixed noise
        print("Generating mixed noise circuits...")
        for n in range(min_qubits, max_qubits + 1, 2):
            for depth in [10, 20]:
                random.seed(self.seed + n * 3000 + depth)
                np.random.seed(self.seed + n * 3000 + depth)
                circuit, meta = self.generator.generate_mixed_noise(n, depth)
                circuits.append((circuit, meta))
        
        # Save all circuits
        print()
        print("Saving circuits...")
        self._save_all(circuits)
        
        # Summary
        print()
        print("=" * 70)
        print(f"COMPLETE: Generated {len(circuits)} noisy circuits")
        print()
        
        # Statistics
        by_qubits = {}
        by_type = {}
        for _, meta in circuits:
            n = meta["n_qubits"]
            subtype = meta["subtype"]
            by_qubits[n] = by_qubits.get(n, 0) + 1
            by_type[subtype] = by_type.get(subtype, 0) + 1
        
        print("By qubit count:")
        for n in sorted(by_qubits.keys()):
            print(f"  {n} qubits: {by_qubits[n]} circuits")
        
        print()
        print("By circuit type:")
        for t in sorted(by_type.keys()):
            print(f"  {t}: {by_type[t]} circuits")
        
        print("=" * 70)
        return len(circuits)
    
    def _save_all(self, circuits: List[Tuple[Dict, Dict]]):
        """Save all circuits and manifest."""
        for i, (circuit, metadata) in enumerate(circuits):
            subtype = metadata["subtype"]
            n_qubits = metadata["n_qubits"]
            
            # Generate unique filename
            instance = metadata.get("instance", 0)
            nr = metadata.get("noise_rate", 0)
            nr_str = f"_p{int(nr*100):02d}" if nr else ""
            
            filename = f"{subtype}_{n_qubits}q{nr_str}_{i:04d}.json"
            filepath = self.output_dir / filename
            
            # Config with lower epsilon to allow rank growth
            config = {
                "epsilon": 1e-3,  # Lower threshold to allow rank to grow
                "initial_rank": 1,
                "export_state": n_qubits <= 10
            }
            
            # Save
            data = {
                "circuit": circuit,
                "config": config,
                "metadata": metadata
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Add to manifest
            self.manifest.append({
                "file": filename,
                "category": "noisy",
                "subtype": subtype,
                "n_qubits": n_qubits,
                "n_operations": metadata["n_operations"],
                "noise_count": metadata.get("noise_count", 0),
                "noise_type": metadata.get("noise_type", "mixed"),
                "noise_rate": metadata.get("noise_rate", 0),
                "metadata": metadata
            })
        
        # Save manifest
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)
        print(f"  Saved manifest: {manifest_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LRET Noisy Circuit Generator")
    parser.add_argument("--output", "-o", default="test_circuits/noisy",
                        help="Output directory")
    parser.add_argument("--min-qubits", type=int, default=6,
                        help="Minimum qubit count")
    parser.add_argument("--max-qubits", type=int, default=12,
                        help="Maximum qubit count")
    parser.add_argument("--instances", "-n", type=int, default=2,
                        help="Instances per configuration")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    generator = NoisySuiteGenerator(args.output, args.seed)
    generator.generate_suite(
        min_qubits=args.min_qubits,
        max_qubits=args.max_qubits,
        instances=args.instances
    )


if __name__ == "__main__":
    main()
