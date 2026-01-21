"""
Corrected Circuit Generator for LRET vs Cirq Comparison

This version uses the CORRECT LRET JSON schema:
- "name" instead of "gate"
- "wires" instead of "targets"/"control"

Author: Corrected after diagnostic analysis
Date: January 22, 2026
"""

import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List

class CorrectedCircuitGenerator:
    """Generate circuits in the CORRECT LRET JSON format."""
    
    def __init__(self, output_dir: str = "circuits_corrected"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.rng = np.random.default_rng(42)
    
    def generate_all_circuits(
        self,
        max_qubits: int = 10,
        noise_levels: List[float] = [0.0, 0.001, 0.01],
    ) -> List[Dict[str, Any]]:
        """Generate all test circuits with correct schema."""
        circuits = []
        
        # Bell states (2, 4, 6, 8, 10 qubits)
        for n in [2, 4, 6, 8, 10]:
            if n <= max_qubits:
                circuits.extend(self._generate_bell_circuits(n, noise_levels))
        
        # GHZ states (3, 4, 5, 6, 8, 10 qubits)
        for n in [3, 4, 5, 6, 8, 10]:
            if n <= max_qubits:
                circuits.extend(self._generate_ghz_circuits(n, noise_levels))
        
        # QFT circuits
        for n in [3, 4, 5, 6, 7, 8]:
            if n <= max_qubits:
                circuits.extend(self._generate_qft_circuits(n, noise_levels))
        
        # Random circuits
        for n in [4, 6, 8]:
            if n <= max_qubits:
                for depth in [10, 20, 50]:
                    circuits.extend(
                        self._generate_random_circuits(n, depth, noise_levels)
                    )
        
        print(f"✓ Generated {len(circuits)} circuits in {self.output_dir}")
        return circuits
    
    def _generate_bell_circuits(
        self, num_qubits: int, noise_levels: List[float]
    ) -> List[Dict[str, Any]]:
        """Generate Bell state circuits."""
        circuits = []
        
        for noise in noise_levels:
            ops = []
            # Create Bell pairs
            for i in range(0, num_qubits, 2):
                ops.append({"name": "H", "wires": [i]})
                ops.append({"name": "CNOT", "wires": [i, i + 1]})
            
            circuit = {
                "circuit": {
                    "num_qubits": num_qubits,
                    "operations": ops,
                },
                "config": {
                    "epsilon": 1e-4,
                    "initial_rank": 1,
                },
            }
            
            if noise > 0:
                circuit["config"]["noise"] = {
                    "type": "depolarizing",
                    "parameter": noise,
                }
            
            noise_str = f"_noise{noise:.4f}" if noise > 0 else ""
            filename = f"bell_{num_qubits}q{noise_str}.json"
            
            with open(self.output_dir / filename, "w") as f:
                json.dump(circuit, f, indent=2)
            
            circuits.append({
                "name": f"Bell {num_qubits}q" + (f" noise={noise}" if noise > 0 else ""),
                "path": str(self.output_dir / filename),
                "qubits": num_qubits,
                "depth": num_qubits,
                "noise": noise,
                "category": "low-rank",
            })
        
        return circuits
    
    def _generate_ghz_circuits(
        self, num_qubits: int, noise_levels: List[float]
    ) -> List[Dict[str, Any]]:
        """Generate GHZ state circuits."""
        circuits = []
        
        for noise in noise_levels:
            ops = [{"name": "H", "wires": [0]}]
            for i in range(num_qubits - 1):
                ops.append({"name": "CNOT", "wires": [i, i + 1]})
            
            circuit = {
                "circuit": {
                    "num_qubits": num_qubits,
                    "operations": ops,
                },
                "config": {
                    "epsilon": 1e-4,
                    "initial_rank": 1,
                },
            }
            
            if noise > 0:
                circuit["config"]["noise"] = {
                    "type": "depolarizing",
                    "parameter": noise,
                }
            
            noise_str = f"_noise{noise:.4f}" if noise > 0 else ""
            filename = f"ghz_{num_qubits}q{noise_str}.json"
            
            with open(self.output_dir / filename, "w") as f:
                json.dump(circuit, f, indent=2)
            
            circuits.append({
                "name": f"GHZ {num_qubits}q" + (f" noise={noise}" if noise > 0 else ""),
                "path": str(self.output_dir / filename),
                "qubits": num_qubits,
                "depth": num_qubits,
                "noise": noise,
                "category": "low-rank",
            })
        
        return circuits
    
    def _generate_qft_circuits(
        self, num_qubits: int, noise_levels: List[float]
    ) -> List[Dict[str, Any]]:
        """Generate QFT circuits."""
        circuits = []
        
        for noise in noise_levels:
            ops = []
            
            # Standard QFT implementation
            for i in range(num_qubits):
                ops.append({"name": "H", "wires": [i]})
                for j in range(i + 1, num_qubits):
                    k = j - i + 1
                    angle = np.pi / (2 ** (k - 1))
                    # Controlled phase rotation (use RZ on target after CNOT pattern)
                    ops.append({"name": "CRZ", "wires": [j, i], "params": [angle]})
            
            # Final swaps
            for i in range(num_qubits // 2):
                ops.append({"name": "SWAP", "wires": [i, num_qubits - 1 - i]})
            
            circuit = {
                "circuit": {
                    "num_qubits": num_qubits,
                    "operations": ops,
                },
                "config": {
                    "epsilon": 1e-4,
                    "initial_rank": 1,
                },
            }
            
            if noise > 0:
                circuit["config"]["noise"] = {
                    "type": "depolarizing",
                    "parameter": noise,
                }
            
            noise_str = f"_noise{noise:.4f}" if noise > 0 else ""
            filename = f"qft_{num_qubits}q{noise_str}.json"
            
            with open(self.output_dir / filename, "w") as f:
                json.dump(circuit, f, indent=2)
            
            circuits.append({
                "name": f"QFT {num_qubits}q" + (f" noise={noise}" if noise > 0 else ""),
                "path": str(self.output_dir / filename),
                "qubits": num_qubits,
                "depth": num_qubits * (num_qubits + 1) // 2,
                "noise": noise,
                "category": "moderate",
            })
        
        return circuits
    
    def _generate_random_circuits(
        self, num_qubits: int, depth: int, noise_levels: List[float]
    ) -> List[Dict[str, Any]]:
        """Generate random circuits."""
        circuits = []
        
        single_gates = ["H", "X", "Y", "Z", "S", "T"]
        param_gates = ["RX", "RY", "RZ"]
        
        for noise in noise_levels:
            ops = []
            
            for _ in range(depth):
                # Single qubit gates on each qubit
                for q in range(num_qubits):
                    if self.rng.random() < 0.7:
                        if self.rng.random() < 0.5:
                            gate = self.rng.choice(single_gates)
                            ops.append({"name": gate, "wires": [q]})
                        else:
                            gate = self.rng.choice(param_gates)
                            angle = float(self.rng.uniform(0, 2 * np.pi))
                            ops.append({"name": gate, "wires": [q], "params": [angle]})
                
                # Two-qubit gates
                for q in range(0, num_qubits - 1, 2):
                    if self.rng.random() < 0.5:
                        ops.append({"name": "CNOT", "wires": [q, q + 1]})
            
            circuit = {
                "circuit": {
                    "num_qubits": num_qubits,
                    "operations": ops,
                },
                "config": {
                    "epsilon": 1e-4,
                    "initial_rank": 1,
                },
            }
            
            if noise > 0:
                circuit["config"]["noise"] = {
                    "type": "depolarizing",
                    "parameter": noise,
                }
            
            noise_str = f"_noise{noise:.4f}" if noise > 0 else ""
            filename = f"random_{num_qubits}q_d{depth}{noise_str}.json"
            
            with open(self.output_dir / filename, "w") as f:
                json.dump(circuit, f, indent=2)
            
            circuits.append({
                "name": f"Random {num_qubits}q d{depth}" + (f" noise={noise}" if noise > 0 else ""),
                "path": str(self.output_dir / filename),
                "qubits": num_qubits,
                "depth": depth,
                "noise": noise,
                "category": "high-rank",
            })
        
        return circuits


if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "d:/LRET/cirq_comparison/circuits_corrected"
    
    generator = CorrectedCircuitGenerator(output_dir)
    circuits = generator.generate_all_circuits(max_qubits=10, noise_levels=[0.0])
    
    print(f"\nGenerated circuits:")
    for c in circuits[:10]:
        print(f"  - {c['name']} ({c['qubits']}q, depth={c['depth']})")
    if len(circuits) > 10:
        print(f"  ... and {len(circuits) - 10} more")
