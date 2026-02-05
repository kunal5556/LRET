#!/usr/bin/env python3
"""
LRET Circuit Generator - Phase B.1
Generates diverse quantum circuits for benchmarking

Usage:
    python generate_circuits.py --all              # Generate all circuit types
    python generate_circuits.py --type random      # Generate only random circuits
    python generate_circuits.py --count 50         # Generate 50 circuits per category
    
Output: JSON files in validation/test_circuits/generated/
"""

import json
import os
import random
import math
import argparse
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

# =============================================================================
# Circuit Data Structures
# =============================================================================

@dataclass
class Operation:
    name: str
    wires: List[int]
    params: List[float] = None
    
    def to_dict(self) -> Dict:
        d = {"name": self.name, "wires": self.wires}
        if self.params:
            d["params"] = self.params
        return d

@dataclass 
class Circuit:
    num_qubits: int
    operations: List[Operation]
    
    def to_dict(self) -> Dict:
        return {
            "num_qubits": self.num_qubits,
            "operations": [op.to_dict() for op in self.operations]
        }

@dataclass
class CircuitConfig:
    epsilon: float = 1e-4
    initial_rank: int = 1
    export_state: bool = True
    
    def to_dict(self) -> Dict:
        return asdict(self)

# =============================================================================
# Gate Definitions
# =============================================================================

SINGLE_QUBIT_GATES = ["H", "X", "Y", "Z", "S", "T"]
SINGLE_QUBIT_PARAM_GATES = ["RX", "RY", "RZ"]  # Removed PHASE - not supported
TWO_QUBIT_GATES = ["CNOT", "CZ"]  # Removed SWAP - not always supported
NOISE_GATES = ["DEPOLARIZE", "AMPLITUDE_DAMP", "PHASE_DAMP"]

# =============================================================================
# Circuit Generators
# =============================================================================

class CircuitGenerator:
    """Base class for circuit generation."""
    
    def __init__(self, seed: int = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
    
    def save_circuit(self, circuit: Circuit, config: CircuitConfig, 
                     filepath: str, metadata: Dict = None):
        """Save circuit to JSON file."""
        data = {
            "circuit": circuit.to_dict(),
            "config": config.to_dict()
        }
        if metadata:
            data["metadata"] = metadata
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


class BasicGatesGenerator(CircuitGenerator):
    """Generate basic gate test circuits."""
    
    def generate_single_qubit_tests(self, n_qubits: int = 4) -> List[Tuple[Circuit, Dict]]:
        """Test each single-qubit gate individually."""
        circuits = []
        
        for gate in SINGLE_QUBIT_GATES:
            ops = []
            for q in range(n_qubits):
                ops.append(Operation(gate, [q]))
            circuit = Circuit(n_qubits, ops)
            metadata = {"type": "basic", "subtype": "single_qubit", "gate": gate}
            circuits.append((circuit, metadata))
        
        return circuits
    
    def generate_param_gate_tests(self, n_qubits: int = 4) -> List[Tuple[Circuit, Dict]]:
        """Test parameterized gates with various angles."""
        circuits = []
        angles = [0, math.pi/4, math.pi/2, math.pi, 3*math.pi/2]
        
        for gate in SINGLE_QUBIT_PARAM_GATES:
            for angle in angles:
                ops = []
                for q in range(n_qubits):
                    ops.append(Operation(gate, [q], [angle]))
                circuit = Circuit(n_qubits, ops)
                metadata = {"type": "basic", "subtype": "param_gate", 
                           "gate": gate, "angle": angle}
                circuits.append((circuit, metadata))
        
        return circuits
    
    def generate_two_qubit_tests(self, n_qubits: int = 4) -> List[Tuple[Circuit, Dict]]:
        """Test two-qubit gates."""
        circuits = []
        
        for gate in TWO_QUBIT_GATES:
            ops = []
            for q in range(0, n_qubits - 1, 2):
                ops.append(Operation(gate, [q, q+1]))
            circuit = Circuit(n_qubits, ops)
            metadata = {"type": "basic", "subtype": "two_qubit", "gate": gate}
            circuits.append((circuit, metadata))
        
        return circuits


class EntanglementGenerator(CircuitGenerator):
    """Generate entangled state circuits."""
    
    def generate_bell_states(self, n_qubits: int = 4) -> List[Tuple[Circuit, Dict]]:
        """Generate Bell pair circuits."""
        circuits = []
        
        # Standard Bell state on each pair
        ops = []
        for q in range(0, n_qubits - 1, 2):
            ops.append(Operation("H", [q]))
            ops.append(Operation("CNOT", [q, q+1]))
        circuit = Circuit(n_qubits, ops)
        metadata = {"type": "entanglement", "subtype": "bell", "pairs": n_qubits // 2}
        circuits.append((circuit, metadata))
        
        return circuits
    
    def generate_ghz_states(self, qubit_range: range = range(3, 13)) -> List[Tuple[Circuit, Dict]]:
        """Generate GHZ states of various sizes."""
        circuits = []
        
        for n in qubit_range:
            ops = [Operation("H", [0])]
            for q in range(n - 1):
                ops.append(Operation("CNOT", [q, q+1]))
            circuit = Circuit(n, ops)
            metadata = {"type": "entanglement", "subtype": "ghz", "n_qubits": n}
            circuits.append((circuit, metadata))
        
        return circuits
    
    def generate_w_states(self, qubit_range: range = range(3, 9)) -> List[Tuple[Circuit, Dict]]:
        """Generate W state approximations."""
        circuits = []
        
        for n in qubit_range:
            ops = []
            # W state preparation (approximate)
            angle = 2 * math.asin(1 / math.sqrt(n))
            ops.append(Operation("RY", [0], [angle]))
            for q in range(n - 1):
                angle_q = 2 * math.asin(1 / math.sqrt(n - q))
                ops.append(Operation("CNOT", [q, q+1]))
                ops.append(Operation("RY", [q+1], [angle_q]))
            circuit = Circuit(n, ops)
            metadata = {"type": "entanglement", "subtype": "w_state", "n_qubits": n}
            circuits.append((circuit, metadata))
        
        return circuits


class RandomCircuitGenerator(CircuitGenerator):
    """Generate random circuits for stress testing."""
    
    def generate_random_circuit(self, n_qubits: int, depth: int, 
                                 two_qubit_prob: float = 0.3) -> Tuple[Circuit, Dict]:
        """Generate a random circuit with specified properties."""
        ops = []
        
        for _ in range(depth):
            if random.random() < two_qubit_prob and n_qubits > 1:
                # Two-qubit gate
                gate = random.choice(TWO_QUBIT_GATES)
                q1, q2 = random.sample(range(n_qubits), 2)
                ops.append(Operation(gate, [q1, q2]))
            else:
                # Single-qubit gate
                q = random.randint(0, n_qubits - 1)
                if random.random() < 0.5:
                    gate = random.choice(SINGLE_QUBIT_GATES)
                    ops.append(Operation(gate, [q]))
                else:
                    gate = random.choice(SINGLE_QUBIT_PARAM_GATES)
                    angle = random.uniform(0, 2 * math.pi)
                    ops.append(Operation(gate, [q], [angle]))
        
        circuit = Circuit(n_qubits, ops)
        metadata = {
            "type": "random", 
            "n_qubits": n_qubits, 
            "depth": depth,
            "two_qubit_prob": two_qubit_prob
        }
        return circuit, metadata
    
    def generate_random_suite(self, qubit_range: range = range(4, 13),
                               depths: List[int] = [10, 20, 50, 100],
                               circuits_per_config: int = 3) -> List[Tuple[Circuit, Dict]]:
        """Generate suite of random circuits."""
        circuits = []
        
        for n in qubit_range:
            for d in depths:
                for i in range(circuits_per_config):
                    circuit, metadata = self.generate_random_circuit(n, d)
                    metadata["instance"] = i
                    circuits.append((circuit, metadata))
        
        return circuits


class AlgorithmCircuitGenerator(CircuitGenerator):
    """Generate circuits based on quantum algorithms."""
    
    def generate_qft(self, qubit_range: range = range(3, 11)) -> List[Tuple[Circuit, Dict]]:
        """Generate Quantum Fourier Transform circuits."""
        circuits = []
        
        for n in qubit_range:
            ops = []
            for q in range(n):
                ops.append(Operation("H", [q]))
                for k in range(q + 1, n):
                    angle = math.pi / (2 ** (k - q))
                    ops.append(Operation("PHASE", [k], [angle]))
                    ops.append(Operation("CNOT", [q, k]))
            
            # Swap for bit reversal
            for q in range(n // 2):
                ops.append(Operation("SWAP", [q, n - 1 - q]))
            
            circuit = Circuit(n, ops)
            metadata = {"type": "algorithm", "subtype": "qft", "n_qubits": n}
            circuits.append((circuit, metadata))
        
        return circuits
    
    def generate_vqe_ansatz(self, n_qubits: int = 4, layers: int = 2,
                            count: int = 5) -> List[Tuple[Circuit, Dict]]:
        """Generate VQE-style variational ansätze."""
        circuits = []
        
        for i in range(count):
            ops = []
            
            # Hardware-efficient ansatz
            for layer in range(layers):
                # Single-qubit rotation layer
                for q in range(n_qubits):
                    ops.append(Operation("RY", [q], [random.uniform(0, 2*math.pi)]))
                    ops.append(Operation("RZ", [q], [random.uniform(0, 2*math.pi)]))
                
                # Entangling layer (linear connectivity)
                for q in range(n_qubits - 1):
                    ops.append(Operation("CNOT", [q, q+1]))
            
            circuit = Circuit(n_qubits, ops)
            metadata = {
                "type": "algorithm", 
                "subtype": "vqe_ansatz",
                "n_qubits": n_qubits,
                "layers": layers,
                "instance": i
            }
            circuits.append((circuit, metadata))
        
        return circuits
    
    def generate_qaoa(self, n_qubits: int = 4, p: int = 2,
                      count: int = 5) -> List[Tuple[Circuit, Dict]]:
        """Generate QAOA circuits for MaxCut."""
        circuits = []
        
        for i in range(count):
            ops = []
            
            # Initial superposition
            for q in range(n_qubits):
                ops.append(Operation("H", [q]))
            
            # QAOA layers
            for layer in range(p):
                gamma = random.uniform(0, 2*math.pi)
                beta = random.uniform(0, math.pi)
                
                # Problem unitary (random graph edges)
                for q1 in range(n_qubits):
                    for q2 in range(q1+1, n_qubits):
                        if random.random() < 0.5:  # Random edge
                            ops.append(Operation("CNOT", [q1, q2]))
                            ops.append(Operation("RZ", [q2], [gamma]))
                            ops.append(Operation("CNOT", [q1, q2]))
                
                # Mixer unitary
                for q in range(n_qubits):
                    ops.append(Operation("RX", [q], [beta]))
            
            circuit = Circuit(n_qubits, ops)
            metadata = {
                "type": "algorithm",
                "subtype": "qaoa",
                "n_qubits": n_qubits,
                "p_layers": p,
                "instance": i
            }
            circuits.append((circuit, metadata))
        
        return circuits
    
    def generate_grover(self, n_qubits: int = 4, 
                        iterations: int = None) -> Tuple[Circuit, Dict]:
        """Generate Grover's algorithm circuit."""
        if iterations is None:
            iterations = int(math.pi / 4 * math.sqrt(2 ** n_qubits))
        
        ops = []
        
        # Initial superposition
        for q in range(n_qubits):
            ops.append(Operation("H", [q]))
        
        # Grover iterations
        for _ in range(iterations):
            # Oracle (mark random state)
            target = random.randint(0, 2**n_qubits - 1)
            for q in range(n_qubits):
                if not (target & (1 << q)):
                    ops.append(Operation("X", [q]))
            
            # Multi-controlled Z (simplified as cascade of CNOTs and Toffoli decomposition)
            if n_qubits >= 2:
                ops.append(Operation("H", [n_qubits - 1]))
                for q in range(n_qubits - 1):
                    ops.append(Operation("CNOT", [q, n_qubits - 1]))
                ops.append(Operation("H", [n_qubits - 1]))
            
            for q in range(n_qubits):
                if not (target & (1 << q)):
                    ops.append(Operation("X", [q]))
            
            # Diffusion operator
            for q in range(n_qubits):
                ops.append(Operation("H", [q]))
                ops.append(Operation("X", [q]))
            
            ops.append(Operation("H", [n_qubits - 1]))
            for q in range(n_qubits - 1):
                ops.append(Operation("CNOT", [q, n_qubits - 1]))
            ops.append(Operation("H", [n_qubits - 1]))
            
            for q in range(n_qubits):
                ops.append(Operation("X", [q]))
                ops.append(Operation("H", [q]))
        
        circuit = Circuit(n_qubits, ops)
        metadata = {
            "type": "algorithm",
            "subtype": "grover",
            "n_qubits": n_qubits,
            "iterations": iterations
        }
        return circuit, metadata


class NoiseCircuitGenerator(CircuitGenerator):
    """Generate circuits with noise channels for Phase 4 testing."""
    
    def add_noise_to_circuit(self, circuit: Circuit, noise_type: str,
                              noise_prob: float) -> Circuit:
        """Add noise channels after each gate."""
        new_ops = []
        
        for op in circuit.operations:
            new_ops.append(op)
            # Add noise after each operation
            for wire in op.wires:
                new_ops.append(Operation(noise_type, [wire], [noise_prob]))
        
        return Circuit(circuit.num_qubits, new_ops)
    
    def generate_noisy_ghz(self, n_qubits: int = 4, 
                           noise_probs: List[float] = [0.01, 0.05, 0.1]) -> List[Tuple[Circuit, Dict]]:
        """Generate GHZ circuits with various noise levels."""
        circuits = []
        
        # Base GHZ circuit
        base_ops = [Operation("H", [0])]
        for q in range(n_qubits - 1):
            base_ops.append(Operation("CNOT", [q, q+1]))
        base_circuit = Circuit(n_qubits, base_ops)
        
        for noise_type in NOISE_GATES:
            for prob in noise_probs:
                noisy_circuit = self.add_noise_to_circuit(base_circuit, noise_type, prob)
                metadata = {
                    "type": "noisy",
                    "subtype": "ghz",
                    "noise_type": noise_type,
                    "noise_prob": prob,
                    "n_qubits": n_qubits
                }
                circuits.append((noisy_circuit, metadata))
        
        return circuits
    
    def generate_noisy_random(self, n_qubits: int = 6, depth: int = 20,
                               noise_probs: List[float] = [0.01, 0.05]) -> List[Tuple[Circuit, Dict]]:
        """Generate random circuits with noise."""
        circuits = []
        rand_gen = RandomCircuitGenerator(self.seed)
        
        for noise_type in NOISE_GATES:
            for prob in noise_probs:
                base_circuit, _ = rand_gen.generate_random_circuit(n_qubits, depth)
                noisy_circuit = self.add_noise_to_circuit(base_circuit, noise_type, prob)
                metadata = {
                    "type": "noisy",
                    "subtype": "random",
                    "noise_type": noise_type,
                    "noise_prob": prob,
                    "n_qubits": n_qubits,
                    "depth": depth
                }
                circuits.append((noisy_circuit, metadata))
        
        return circuits


class StressTestGenerator(CircuitGenerator):
    """Generate circuits for stress testing optimization phases."""
    
    def generate_high_depth(self, n_qubits: int = 6,
                             depths: List[int] = [100, 200, 500]) -> List[Tuple[Circuit, Dict]]:
        """Generate high-depth circuits for rank growth testing."""
        circuits = []
        rand_gen = RandomCircuitGenerator(self.seed)
        
        for depth in depths:
            circuit, metadata = rand_gen.generate_random_circuit(n_qubits, depth, 
                                                                   two_qubit_prob=0.4)
            metadata["type"] = "stress"
            metadata["subtype"] = "high_depth"
            circuits.append((circuit, metadata))
        
        return circuits
    
    def generate_high_entanglement(self, qubit_range: range = range(6, 12)) -> List[Tuple[Circuit, Dict]]:
        """Generate highly entangling circuits."""
        circuits = []
        
        for n in qubit_range:
            ops = []
            # Initial layer
            for q in range(n):
                ops.append(Operation("H", [q]))
            
            # Dense entanglement layers
            for layer in range(n):
                for q1 in range(n):
                    for q2 in range(q1 + 1, n):
                        ops.append(Operation("CNOT", [q1, q2]))
                        ops.append(Operation("RZ", [q2], [random.uniform(0, 2*math.pi)]))
            
            circuit = Circuit(n, ops)
            metadata = {
                "type": "stress",
                "subtype": "high_entanglement",
                "n_qubits": n,
                "density": "all_to_all"
            }
            circuits.append((circuit, metadata))
        
        return circuits
    
    def generate_parallel_friendly(self, n_qubits: int = 8,
                                    layer_count: int = 10) -> Tuple[Circuit, Dict]:
        """Generate circuits optimized for parallelization testing."""
        ops = []
        
        for layer in range(layer_count):
            # Even layer: gates on even qubits
            for q in range(0, n_qubits, 2):
                ops.append(Operation("RY", [q], [random.uniform(0, 2*math.pi)]))
            # Odd layer: gates on odd qubits
            for q in range(1, n_qubits, 2):
                ops.append(Operation("RY", [q], [random.uniform(0, 2*math.pi)]))
            # Entangling layer
            for q in range(0, n_qubits - 1, 2):
                ops.append(Operation("CNOT", [q, q+1]))
        
        circuit = Circuit(n_qubits, ops)
        metadata = {
            "type": "stress",
            "subtype": "parallel_friendly",
            "n_qubits": n_qubits,
            "layers": layer_count
        }
        return circuit, metadata


# =============================================================================
# Main Generator Class
# =============================================================================

class FullCircuitSuiteGenerator:
    """Generate complete circuit test suite."""
    
    def __init__(self, output_dir: str, seed: int = 42):
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.manifest = []
        
    def generate_all(self, counts: Dict[str, int] = None):
        """Generate all circuit types."""
        if counts is None:
            counts = {
                "basic": 1,
                "entanglement": 1,
                "random": 3,
                "algorithm": 3,
                "noisy": 1,
                "stress": 1
            }
        
        print("=" * 60)
        print("LRET Circuit Generator - Phase B.1")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}")
        print(f"Seed: {self.seed}")
        print()
        
        total = 0
        
        # Basic gates
        print("Generating basic gate circuits...")
        basic_gen = BasicGatesGenerator(self.seed)
        circuits = basic_gen.generate_single_qubit_tests(6)
        circuits.extend(basic_gen.generate_param_gate_tests(6))
        circuits.extend(basic_gen.generate_two_qubit_tests(6))
        total += self._save_circuits(circuits, "basic")
        
        # Entanglement
        print("Generating entanglement circuits...")
        ent_gen = EntanglementGenerator(self.seed)
        circuits = ent_gen.generate_bell_states(8)
        circuits.extend(ent_gen.generate_ghz_states(range(3, 13)))
        circuits.extend(ent_gen.generate_w_states(range(3, 9)))
        total += self._save_circuits(circuits, "entanglement")
        
        # Random
        print("Generating random circuits...")
        rand_gen = RandomCircuitGenerator(self.seed)
        circuits = rand_gen.generate_random_suite(
            qubit_range=range(4, 12),
            depths=[10, 20, 50],
            circuits_per_config=counts["random"]
        )
        total += self._save_circuits(circuits, "random")
        
        # Algorithms
        print("Generating algorithm circuits...")
        algo_gen = AlgorithmCircuitGenerator(self.seed)
        circuits = algo_gen.generate_qft(range(3, 10))
        circuits.extend(algo_gen.generate_vqe_ansatz(6, 2, counts["algorithm"]))
        circuits.extend(algo_gen.generate_vqe_ansatz(8, 3, counts["algorithm"]))
        circuits.extend(algo_gen.generate_qaoa(6, 2, counts["algorithm"]))
        circuits.extend(algo_gen.generate_qaoa(8, 3, counts["algorithm"]))
        for n in range(3, 7):
            circuits.append(algo_gen.generate_grover(n))
        total += self._save_circuits(circuits, "algorithm")
        
        # Noisy
        print("Generating noisy circuits...")
        noise_gen = NoiseCircuitGenerator(self.seed)
        circuits = noise_gen.generate_noisy_ghz(6, [0.01, 0.05, 0.1])
        circuits.extend(noise_gen.generate_noisy_random(6, 20, [0.01, 0.05]))
        circuits.extend(noise_gen.generate_noisy_random(8, 30, [0.01, 0.05]))
        total += self._save_circuits(circuits, "noisy")
        
        # Stress tests
        print("Generating stress test circuits...")
        stress_gen = StressTestGenerator(self.seed)
        circuits = stress_gen.generate_high_depth(6, [50, 100, 200])
        circuits.extend(stress_gen.generate_high_depth(8, [50, 100]))
        circuits.extend(stress_gen.generate_high_entanglement(range(4, 10)))
        for n in [6, 8, 10]:
            circuits.append(stress_gen.generate_parallel_friendly(n, 20))
        total += self._save_circuits(circuits, "stress")
        
        # Save manifest
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)
        
        print()
        print("=" * 60)
        print(f"COMPLETE: Generated {total} circuits")
        print(f"Manifest: {manifest_path}")
        print("=" * 60)
        
        return total
    
    def _save_circuits(self, circuits: List[Tuple[Circuit, Dict]], 
                       category: str) -> int:
        """Save circuits to files and update manifest."""
        category_dir = self.output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        count = 0
        for i, (circuit, metadata) in enumerate(circuits):
            # Generate filename
            subtype = metadata.get("subtype", "unknown")
            n_qubits = metadata.get("n_qubits", circuit.num_qubits)
            filename = f"{category}_{subtype}_{n_qubits}q_{i:04d}.json"
            filepath = category_dir / filename
            
            # Config based on circuit size
            config = CircuitConfig(
                epsilon=1e-4 if n_qubits < 10 else 1e-3,
                initial_rank=1,
                export_state=n_qubits <= 12
            )
            
            # Save
            data = {
                "circuit": circuit.to_dict(),
                "config": config.to_dict(),
                "metadata": metadata
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Add to manifest
            self.manifest.append({
                "file": str(filepath.relative_to(self.output_dir)),
                "category": category,
                "subtype": subtype,
                "n_qubits": n_qubits,
                "n_operations": len(circuit.operations),
                "metadata": metadata
            })
            count += 1
        
        print(f"  {category}: {count} circuits")
        return count


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LRET Circuit Generator")
    parser.add_argument("--output", "-o", default="test_circuits/generated",
                        help="Output directory for circuits")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--count", "-c", type=int, default=3,
                        help="Circuits per configuration")
    parser.add_argument("--all", action="store_true",
                        help="Generate all circuit types")
    parser.add_argument("--type", "-t", choices=["basic", "entanglement", "random", 
                                                  "algorithm", "noisy", "stress"],
                        help="Generate only specific type")
    
    args = parser.parse_args()
    
    generator = FullCircuitSuiteGenerator(args.output, args.seed)
    
    counts = {
        "basic": 1,
        "entanglement": 1,
        "random": args.count,
        "algorithm": args.count,
        "noisy": 1,
        "stress": 1
    }
    
    generator.generate_all(counts)


if __name__ == "__main__":
    main()
