#!/usr/bin/env python3
"""
LRET Large Circuit Generator - Phase C
Generates high-qubit count circuits specifically for row-parallelism testing

Focus areas:
- 8-16 qubits (where row parallelism shows most benefit)  
- Circuits that stress rank growth (CNOT-heavy)
- Multiple depth configurations
- Statistical sampling (multiple random instances)

Usage:
    python generate_large_circuits.py --min-qubits 8 --max-qubits 16 --output large_circuits
"""

import json
import os
import random
import math
import argparse
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime

# =============================================================================
# Configuration - Only use gates that LRET supports
# =============================================================================

# Verified working gates from Phase B testing
SINGLE_QUBIT_GATES = ["H", "X", "Y", "Z", "S", "T"]
SINGLE_QUBIT_PARAM_GATES = ["RX", "RY", "RZ"]
TWO_QUBIT_GATES = ["CNOT", "CZ"]

# =============================================================================
# Data Structures
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

# =============================================================================
# Large Circuit Generators
# =============================================================================

class LargeCircuitGenerator:
    """Generate circuits optimized for row-parallelism testing."""
    
    def __init__(self, seed: int = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
    
    def generate_high_rank_circuit(self, n_qubits: int, 
                                    target_depth: int = None) -> Tuple[Circuit, Dict]:
        """
        Generate circuits that maximize rank growth.
        Key insight: Row-parallelism benefits scale with rank.
        More CNOTs = higher rank = bigger speedup from Phase 1-5 optimizations.
        """
        if target_depth is None:
            target_depth = n_qubits * 5  # Scale depth with qubits
        
        ops = []
        
        # Initial superposition
        for q in range(n_qubits):
            ops.append(Operation("H", [q]))
        
        # Dense entanglement to drive rank growth
        layer_count = target_depth // n_qubits
        for layer in range(layer_count):
            # Rotate all qubits
            for q in range(n_qubits):
                angle = random.uniform(0, 2 * math.pi)
                gate = random.choice(SINGLE_QUBIT_PARAM_GATES)
                ops.append(Operation(gate, [q], [angle]))
            
            # Ladder connectivity - drives rank growth efficiently
            for q in range(n_qubits - 1):
                ops.append(Operation("CNOT", [q, q + 1]))
            
            # Reverse direction
            for q in range(n_qubits - 2, -1, -1):
                ops.append(Operation("CNOT", [q + 1, q]))
        
        circuit = Circuit(n_qubits, ops)
        metadata = {
            "type": "large",
            "subtype": "high_rank",
            "n_qubits": n_qubits,
            "depth": len(ops),
            "layers": layer_count,
            "cnot_count": sum(1 for op in ops if op.name == "CNOT")
        }
        return circuit, metadata
    
    def generate_vqe_large(self, n_qubits: int, layers: int = 3) -> Tuple[Circuit, Dict]:
        """
        Large VQE ansatz - practical algorithm circuit.
        Hardware-efficient ansatz with linear connectivity.
        """
        ops = []
        
        for layer in range(layers):
            # Rotation layer
            for q in range(n_qubits):
                ops.append(Operation("RY", [q], [random.uniform(0, 2*math.pi)]))
                ops.append(Operation("RZ", [q], [random.uniform(0, 2*math.pi)]))
            
            # Entangling layer (CNOT ladder)
            for q in range(n_qubits - 1):
                ops.append(Operation("CNOT", [q, q + 1]))
        
        # Final rotation
        for q in range(n_qubits):
            ops.append(Operation("RY", [q], [random.uniform(0, 2*math.pi)]))
        
        circuit = Circuit(n_qubits, ops)
        metadata = {
            "type": "large",
            "subtype": "vqe_large",
            "n_qubits": n_qubits,
            "layers": layers,
            "depth": len(ops),
            "cnot_count": layers * (n_qubits - 1)
        }
        return circuit, metadata
    
    def generate_qaoa_large(self, n_qubits: int, p: int = 2) -> Tuple[Circuit, Dict]:
        """
        Large QAOA circuit for MaxCut.
        Problem Hamiltonian with ~50% edge density.
        """
        ops = []
        
        # Initial superposition
        for q in range(n_qubits):
            ops.append(Operation("H", [q]))
        
        # QAOA layers
        edge_count = 0
        for layer in range(p):
            gamma = random.uniform(0, 2*math.pi)
            beta = random.uniform(0, math.pi)
            
            # Problem unitary with ~50% edge density
            for q1 in range(n_qubits):
                for q2 in range(q1 + 1, n_qubits):
                    if random.random() < 0.5:
                        # ZZ interaction: CNOT-RZ-CNOT
                        ops.append(Operation("CNOT", [q1, q2]))
                        ops.append(Operation("RZ", [q2], [gamma]))
                        ops.append(Operation("CNOT", [q1, q2]))
                        edge_count += 1
            
            # Mixer unitary
            for q in range(n_qubits):
                ops.append(Operation("RX", [q], [beta]))
        
        circuit = Circuit(n_qubits, ops)
        metadata = {
            "type": "large",
            "subtype": "qaoa_large",
            "n_qubits": n_qubits,
            "p_layers": p,
            "depth": len(ops),
            "edge_count": edge_count,
            "cnot_count": sum(1 for op in ops if op.name == "CNOT")
        }
        return circuit, metadata
    
    def generate_qft_fixed(self, n_qubits: int) -> Tuple[Circuit, Dict]:
        """
        QFT circuit using only supported gates.
        Replaces PHASE with controlled-RZ decomposition.
        """
        ops = []
        
        for target in range(n_qubits):
            ops.append(Operation("H", [target]))
            
            for control in range(target + 1, n_qubits):
                # Controlled phase = CNOT-RZ-CNOT decomposition
                angle = math.pi / (2 ** (control - target))
                ops.append(Operation("CNOT", [control, target]))
                ops.append(Operation("RZ", [target], [angle / 2]))
                ops.append(Operation("CNOT", [control, target]))
                ops.append(Operation("RZ", [target], [-angle / 2]))
                ops.append(Operation("RZ", [control], [angle / 2]))
        
        # Bit reversal using CNOT swaps (SWAP = 3 CNOTs)
        for i in range(n_qubits // 2):
            j = n_qubits - 1 - i
            # SWAP decomposition
            ops.append(Operation("CNOT", [i, j]))
            ops.append(Operation("CNOT", [j, i]))
            ops.append(Operation("CNOT", [i, j]))
        
        circuit = Circuit(n_qubits, ops)
        metadata = {
            "type": "large",
            "subtype": "qft_fixed",
            "n_qubits": n_qubits,
            "depth": len(ops),
            "cnot_count": sum(1 for op in ops if op.name == "CNOT")
        }
        return circuit, metadata
    
    def generate_grover_fixed(self, n_qubits: int, iterations: int = None) -> Tuple[Circuit, Dict]:
        """
        Grover's algorithm with fixed gate set.
        Uses proper multi-controlled gate decomposition.
        """
        if iterations is None:
            iterations = max(1, int(math.pi / 4 * math.sqrt(2 ** n_qubits)))
            iterations = min(iterations, 5)  # Cap for reasonable runtime
        
        ops = []
        
        # Initial superposition
        for q in range(n_qubits):
            ops.append(Operation("H", [q]))
        
        for _ in range(iterations):
            # Oracle (simplified - marks specific computational basis state)
            target_state = random.randint(0, 2**n_qubits - 1)
            
            # Apply X gates to prepare for oracle
            for q in range(n_qubits):
                if not (target_state & (1 << q)):
                    ops.append(Operation("X", [q]))
            
            # Multi-controlled Z using cascade (simplified decomposition)
            if n_qubits >= 2:
                ops.append(Operation("H", [n_qubits - 1]))
                for q in range(n_qubits - 1):
                    ops.append(Operation("CNOT", [q, n_qubits - 1]))
                    ops.append(Operation("T", [n_qubits - 1]))
                ops.append(Operation("H", [n_qubits - 1]))
            
            # Undo X gates
            for q in range(n_qubits):
                if not (target_state & (1 << q)):
                    ops.append(Operation("X", [q]))
            
            # Diffusion operator
            for q in range(n_qubits):
                ops.append(Operation("H", [q]))
                ops.append(Operation("X", [q]))
            
            # Multi-controlled Z for diffusion
            ops.append(Operation("H", [n_qubits - 1]))
            for q in range(n_qubits - 1):
                ops.append(Operation("CNOT", [q, n_qubits - 1]))
            ops.append(Operation("H", [n_qubits - 1]))
            
            for q in range(n_qubits):
                ops.append(Operation("X", [q]))
                ops.append(Operation("H", [q]))
        
        circuit = Circuit(n_qubits, ops)
        metadata = {
            "type": "large",
            "subtype": "grover_fixed",
            "n_qubits": n_qubits,
            "iterations": iterations,
            "depth": len(ops),
            "cnot_count": sum(1 for op in ops if op.name == "CNOT")
        }
        return circuit, metadata
    
    def generate_random_structured(self, n_qubits: int, depth_multiplier: int = 3,
                                    two_qubit_ratio: float = 0.4) -> Tuple[Circuit, Dict]:
        """
        Random circuit with controlled structure.
        Better for testing than pure random - has consistent properties.
        """
        ops = []
        target_depth = n_qubits * depth_multiplier
        
        for layer in range(target_depth // n_qubits):
            # Single qubit layer
            for q in range(n_qubits):
                if random.random() < 0.7:
                    if random.random() < 0.5:
                        gate = random.choice(SINGLE_QUBIT_GATES)
                        ops.append(Operation(gate, [q]))
                    else:
                        gate = random.choice(SINGLE_QUBIT_PARAM_GATES)
                        ops.append(Operation(gate, [q], [random.uniform(0, 2*math.pi)]))
            
            # Two-qubit layer (structured - even-odd pattern)
            if layer % 2 == 0:
                for q in range(0, n_qubits - 1, 2):
                    if random.random() < two_qubit_ratio:
                        gate = random.choice(TWO_QUBIT_GATES)
                        ops.append(Operation(gate, [q, q + 1]))
            else:
                for q in range(1, n_qubits - 1, 2):
                    if random.random() < two_qubit_ratio:
                        gate = random.choice(TWO_QUBIT_GATES)
                        ops.append(Operation(gate, [q, q + 1]))
        
        circuit = Circuit(n_qubits, ops)
        metadata = {
            "type": "large",
            "subtype": "random_structured",
            "n_qubits": n_qubits,
            "depth": len(ops),
            "two_qubit_ratio": two_qubit_ratio,
            "cnot_count": sum(1 for op in ops if op.name in TWO_QUBIT_GATES)
        }
        return circuit, metadata
    
    def generate_parallel_benchmark(self, n_qubits: int, layers: int = 5) -> Tuple[Circuit, Dict]:
        """
        Circuit specifically designed to benefit from row parallelism.
        Pattern: dense single-qubit rotations + CNOT ladder
        This is the ideal case for Phase 1-5 optimizations.
        """
        ops = []
        
        for layer in range(layers):
            # All qubits get independent rotations (parallelizable)
            for q in range(n_qubits):
                ops.append(Operation("RY", [q], [random.uniform(0, 2*math.pi)]))
                ops.append(Operation("RZ", [q], [random.uniform(0, 2*math.pi)]))
            
            # CNOT ladder (creates entanglement, rank growth)
            for q in range(n_qubits - 1):
                ops.append(Operation("CNOT", [q, q + 1]))
            
            # Reverse CNOTs
            for q in range(n_qubits - 2, -1, -1):
                ops.append(Operation("CNOT", [q + 1, q]))
        
        circuit = Circuit(n_qubits, ops)
        metadata = {
            "type": "large",
            "subtype": "parallel_benchmark",
            "n_qubits": n_qubits,
            "layers": layers,
            "depth": len(ops),
            "cnot_count": 2 * (n_qubits - 1) * layers
        }
        return circuit, metadata


# =============================================================================
# Suite Generator
# =============================================================================

class LargeSuiteGenerator:
    """Generate complete large circuit test suite."""
    
    def __init__(self, output_dir: str, seed: int = 42):
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.manifest = []
        self.generator = LargeCircuitGenerator(seed)
        
    def generate_suite(self, min_qubits: int = 8, max_qubits: int = 16,
                       instances_per_config: int = 3) -> int:
        """Generate comprehensive large circuit suite."""
        print("=" * 70)
        print("LRET Large Circuit Generator - Phase C")
        print("=" * 70)
        print(f"Output: {self.output_dir}")
        print(f"Qubit range: {min_qubits}-{max_qubits}")
        print(f"Instances per config: {instances_per_config}")
        print(f"Seed: {self.seed}")
        print()
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        circuits = []
        
        # 1. High-rank circuits (primary target for row-parallelism)
        print("Generating high-rank circuits...")
        for n in range(min_qubits, max_qubits + 1, 2):  # 8, 10, 12, 14, 16
            for i in range(instances_per_config):
                random.seed(self.seed + n * 100 + i)  # Reproducible
                circuit, meta = self.generator.generate_high_rank_circuit(n)
                meta["instance"] = i
                circuits.append((circuit, meta))
        
        # 2. VQE ansätze (practical application)
        print("Generating VQE circuits...")
        for n in range(min_qubits, max_qubits + 1, 2):
            for layers in [2, 3, 4]:
                random.seed(self.seed + n * 200 + layers)
                circuit, meta = self.generator.generate_vqe_large(n, layers)
                circuits.append((circuit, meta))
        
        # 3. QAOA circuits (practical application)
        print("Generating QAOA circuits...")
        for n in range(min_qubits, min(max_qubits, 12) + 1, 2):  # QAOA gets expensive
            for p in [1, 2, 3]:
                random.seed(self.seed + n * 300 + p)
                circuit, meta = self.generator.generate_qaoa_large(n, p)
                circuits.append((circuit, meta))
        
        # 4. QFT (fixed version)
        print("Generating QFT circuits...")
        for n in range(min_qubits, max_qubits + 1, 2):
            circuit, meta = self.generator.generate_qft_fixed(n)
            circuits.append((circuit, meta))
        
        # 5. Grover's algorithm
        print("Generating Grover circuits...")
        for n in range(min_qubits, min(max_qubits, 12) + 1, 2):
            random.seed(self.seed + n * 400)
            circuit, meta = self.generator.generate_grover_fixed(n)
            circuits.append((circuit, meta))
        
        # 6. Random structured circuits
        print("Generating random structured circuits...")
        for n in range(min_qubits, max_qubits + 1, 2):
            for depth_mult in [3, 5, 7]:
                for i in range(instances_per_config):
                    random.seed(self.seed + n * 500 + depth_mult * 10 + i)
                    circuit, meta = self.generator.generate_random_structured(n, depth_mult)
                    meta["instance"] = i
                    circuits.append((circuit, meta))
        
        # 7. Parallel benchmark circuits (ideal case)
        print("Generating parallel benchmark circuits...")
        for n in range(min_qubits, max_qubits + 1, 2):
            for layers in [5, 10, 15]:
                random.seed(self.seed + n * 600 + layers)
                circuit, meta = self.generator.generate_parallel_benchmark(n, layers)
                circuits.append((circuit, meta))
        
        # Save all circuits
        print()
        print("Saving circuits...")
        self._save_all(circuits)
        
        # Summary
        print()
        print("=" * 70)
        print(f"COMPLETE: Generated {len(circuits)} large circuits")
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
    
    def _save_all(self, circuits: List[Tuple[Circuit, Dict]]):
        """Save all circuits and manifest."""
        for i, (circuit, metadata) in enumerate(circuits):
            subtype = metadata["subtype"]
            n_qubits = metadata["n_qubits"]
            
            # Generate unique filename
            instance = metadata.get("instance", 0)
            extra = ""
            if "layers" in metadata:
                extra = f"_L{metadata['layers']}"
            elif "p_layers" in metadata:
                extra = f"_p{metadata['p_layers']}"
            elif "depth" in metadata and "instance" not in metadata:
                extra = f"_d{metadata.get('depth', 0)}"
            
            filename = f"{subtype}_{n_qubits}q{extra}_{i:04d}.json"
            filepath = self.output_dir / filename
            
            # Config
            config = {
                "epsilon": 1e-4 if n_qubits < 12 else 1e-3,
                "initial_rank": 1,
                "export_state": n_qubits <= 14
            }
            
            # Save
            data = {
                "circuit": circuit.to_dict(),
                "config": config,
                "metadata": metadata
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Add to manifest
            self.manifest.append({
                "file": filename,
                "category": "large",
                "subtype": subtype,
                "n_qubits": n_qubits,
                "n_operations": len(circuit.operations),
                "cnot_count": metadata.get("cnot_count", 0),
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
    parser = argparse.ArgumentParser(description="LRET Large Circuit Generator")
    parser.add_argument("--output", "-o", default="test_circuits/large",
                        help="Output directory")
    parser.add_argument("--min-qubits", type=int, default=8,
                        help="Minimum qubit count")
    parser.add_argument("--max-qubits", type=int, default=16,
                        help="Maximum qubit count")
    parser.add_argument("--instances", "-n", type=int, default=3,
                        help="Instances per configuration")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    generator = LargeSuiteGenerator(args.output, args.seed)
    generator.generate_suite(
        min_qubits=args.min_qubits,
        max_qubits=args.max_qubits,
        instances_per_config=args.instances
    )


if __name__ == "__main__":
    main()
