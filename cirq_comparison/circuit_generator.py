"""Circuit Generator for LRET vs Cirq Comparison.

Generates test circuits in both LRET JSON and Cirq formats for benchmarking.
Includes Bell states, GHZ states, QFT, random circuits, and QAOA-like circuits.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cirq
import numpy as np


class CircuitGenerator:
    """Generate quantum circuits for LRET vs Cirq comparison."""

    def __init__(self, output_dir: str = "circuits"):
        """Initialize circuit generator.

        Args:
            output_dir: Directory to save generated circuits
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_circuits(
        self,
        max_qubits: int = 10,
        noise_levels: List[float] = [0.0, 0.001, 0.01],
    ) -> List[Dict[str, Any]]:
        """Generate all test circuits.

        Args:
            max_qubits: Maximum number of qubits to test
            noise_levels: List of depolarizing noise levels

        Returns:
            List of circuit metadata dictionaries
        """
        circuits = []

        print(f"Generating circuits (max {max_qubits} qubits)...")

        # Bell states (2 to max_qubits, pairs only)
        for n in range(2, min(max_qubits + 1, 11), 2):
            circuits.extend(self._generate_bell_circuits(n, noise_levels))

        # GHZ states
        for n in range(3, max_qubits + 1):
            circuits.extend(self._generate_ghz_circuits(n, noise_levels))

        # QFT circuits
        for n in range(3, min(max_qubits + 1, 13)):
            circuits.extend(self._generate_qft_circuits(n, noise_levels))

        # Random circuits
        for n in [4, 6, 8, 10]:
            if n <= max_qubits:
                for depth in [5, 10, 20]:
                    circuits.extend(
                        self._generate_random_circuits(n, depth, noise_levels)
                    )

        print(f"✓ Generated {len(circuits)} circuits in {self.output_dir}")
        return circuits

    def _generate_bell_circuits(
        self,
        num_qubits: int,
        noise_levels: List[float],
    ) -> List[Dict[str, Any]]:
        """Generate Bell state circuits.

        Args:
            num_qubits: Number of qubits (must be even)
            noise_levels: Noise levels to test

        Returns:
            List of circuit metadata
        """
        circuits = []

        for noise in noise_levels:
            # LRET JSON format
            lret_circuit = {
                "circuit": {
                    "num_qubits": num_qubits,
                    "operations": [],
                },
                "config": {
                    "epsilon": 1e-6,
                    "max_rank": 1000,
                },
            }

            # Create Bell pairs
            for i in range(0, num_qubits, 2):
                lret_circuit["circuit"]["operations"].extend([
                    {"gate": "H", "targets": [i]},
                    {"gate": "CNOT", "control": i, "targets": [i + 1]},
                ])

            # Add noise
            if noise > 0:
                lret_circuit["config"]["noise"] = {
                    "type": "depolarizing",
                    "parameter": noise,
                }

            # Save LRET JSON
            noise_str = f"_noise{noise:.4f}" if noise > 0 else ""
            filename = f"bell_{num_qubits}q{noise_str}"
            json_path = self.output_dir / f"{filename}.json"

            with open(json_path, "w") as f:
                json.dump(lret_circuit, f, indent=2)

            circuits.append({
                "name": filename,
                "type": "bell",
                "num_qubits": num_qubits,
                "depth": 2,
                "noise": noise,
                "path": str(json_path),
            })

        return circuits

    def _generate_ghz_circuits(
        self,
        num_qubits: int,
        noise_levels: List[float],
    ) -> List[Dict[str, Any]]:
        """Generate GHZ state circuits.

        |GHZ⟩ = (|00...0⟩ + |11...1⟩) / √2

        Args:
            num_qubits: Number of qubits
            noise_levels: Noise levels to test

        Returns:
            List of circuit metadata
        """
        circuits = []

        for noise in noise_levels:
            lret_circuit = {
                "circuit": {
                    "num_qubits": num_qubits,
                    "operations": [
                        {"gate": "H", "targets": [0]},
                    ],
                },
                "config": {
                    "epsilon": 1e-6,
                    "max_rank": 1000,
                },
            }

            # CNOT chain
            for i in range(num_qubits - 1):
                lret_circuit["circuit"]["operations"].append(
                    {"gate": "CNOT", "control": i, "targets": [i + 1]}
                )

            if noise > 0:
                lret_circuit["config"]["noise"] = {
                    "type": "depolarizing",
                    "parameter": noise,
                }

            noise_str = f"_noise{noise:.4f}" if noise > 0 else ""
            filename = f"ghz_{num_qubits}q{noise_str}"
            json_path = self.output_dir / f"{filename}.json"

            with open(json_path, "w") as f:
                json.dump(lret_circuit, f, indent=2)

            circuits.append({
                "name": filename,
                "type": "ghz",
                "num_qubits": num_qubits,
                "depth": num_qubits,
                "noise": noise,
                "path": str(json_path),
            })

        return circuits

    def _generate_qft_circuits(
        self,
        num_qubits: int,
        noise_levels: List[float],
    ) -> List[Dict[str, Any]]:
        """Generate Quantum Fourier Transform circuits.

        Args:
            num_qubits: Number of qubits
            noise_levels: Noise levels to test

        Returns:
            List of circuit metadata
        """
        circuits = []

        for noise in noise_levels:
            operations = []

            # QFT implementation
            for i in range(num_qubits):
                operations.append({"gate": "H", "targets": [i]})

                for j in range(i + 1, num_qubits):
                    angle = 2 * math.pi / (2 ** (j - i + 1))
                    # Controlled phase rotation
                    operations.append({
                        "gate": "CNOT",
                        "control": j,
                        "targets": [i],
                    })
                    operations.append({
                        "gate": "RZ",
                        "targets": [i],
                        "parameters": angle,
                    })
                    operations.append({
                        "gate": "CNOT",
                        "control": j,
                        "targets": [i],
                    })

            # Swap qubits for correct ordering
            for i in range(num_qubits // 2):
                operations.append({
                    "gate": "SWAP",
                    "targets": [i, num_qubits - i - 1],
                })

            lret_circuit = {
                "circuit": {
                    "num_qubits": num_qubits,
                    "operations": operations,
                },
                "config": {
                    "epsilon": 1e-6,
                    "max_rank": 1000,
                },
            }

            if noise > 0:
                lret_circuit["config"]["noise"] = {
                    "type": "depolarizing",
                    "parameter": noise,
                }

            noise_str = f"_noise{noise:.4f}" if noise > 0 else ""
            filename = f"qft_{num_qubits}q{noise_str}"
            json_path = self.output_dir / f"{filename}.json"

            with open(json_path, "w") as f:
                json.dump(lret_circuit, f, indent=2)

            depth = len(operations)
            circuits.append({
                "name": filename,
                "type": "qft",
                "num_qubits": num_qubits,
                "depth": depth,
                "noise": noise,
                "path": str(json_path),
            })

        return circuits

    def _generate_random_circuits(
        self,
        num_qubits: int,
        depth: int,
        noise_levels: List[float],
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """Generate random quantum circuits.

        Args:
            num_qubits: Number of qubits
            depth: Circuit depth (number of layers)
            noise_levels: Noise levels to test
            seed: Random seed

        Returns:
            List of circuit metadata
        """
        circuits = []
        rng = np.random.RandomState(seed)

        for noise in noise_levels:
            operations = []

            for layer in range(depth):
                # Random single-qubit gates
                for q in range(num_qubits):
                    gate = rng.choice(["H", "RX", "RY", "RZ"])
                    if gate in ["RX", "RY", "RZ"]:
                        angle = rng.uniform(0, 2 * math.pi)
                        operations.append({
                            "gate": gate,
                            "targets": [q],
                            "parameters": angle,
                        })
                    else:
                        operations.append({
                            "gate": gate,
                            "targets": [q],
                        })

                # Random two-qubit gates
                for q in range(0, num_qubits - 1, 2):
                    if rng.random() > 0.5:
                        operations.append({
                            "gate": "CNOT",
                            "control": q,
                            "targets": [q + 1],
                        })

            lret_circuit = {
                "circuit": {
                    "num_qubits": num_qubits,
                    "operations": operations,
                },
                "config": {
                    "epsilon": 1e-6,
                    "max_rank": 1000,
                },
            }

            if noise > 0:
                lret_circuit["config"]["noise"] = {
                    "type": "depolarizing",
                    "parameter": noise,
                }

            noise_str = f"_noise{noise:.4f}" if noise > 0 else ""
            filename = f"random_{num_qubits}q_d{depth}{noise_str}"
            json_path = self.output_dir / f"{filename}.json"

            with open(json_path, "w") as f:
                json.dump(lret_circuit, f, indent=2)

            circuits.append({
                "name": filename,
                "type": "random",
                "num_qubits": num_qubits,
                "depth": depth,
                "noise": noise,
                "path": str(json_path),
            })

        return circuits

    def generate_circuit_manifest(self, circuits: List[Dict[str, Any]]) -> None:
        """Save circuit manifest for easy loading.

        Args:
            circuits: List of circuit metadata
        """
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(circuits, f, indent=2)
        print(f"✓ Saved manifest to {manifest_path}")


def main():
    """Generate all test circuits."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate test circuits for LRET vs Cirq comparison"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="circuits",
        help="Output directory for circuits",
    )
    parser.add_argument(
        "--max-qubits",
        type=int,
        default=10,
        help="Maximum number of qubits (default: 10)",
    )
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=[0.0, 0.001, 0.01],
        help="Noise levels to test (default: 0.0 0.001 0.01)",
    )

    args = parser.parse_args()

    # Generate circuits
    generator = CircuitGenerator(args.output)
    circuits = generator.generate_all_circuits(
        max_qubits=args.max_qubits,
        noise_levels=args.noise_levels,
    )

    # Save manifest
    generator.generate_circuit_manifest(circuits)

    print(f"\n✓ Generated {len(circuits)} circuits")
    print(f"  Output directory: {args.output}")
    print(f"  Circuit types: Bell, GHZ, QFT, Random")
    print(f"  Qubit range: 2-{args.max_qubits}")
    print(f"  Noise levels: {args.noise_levels}")


if __name__ == "__main__":
    main()
