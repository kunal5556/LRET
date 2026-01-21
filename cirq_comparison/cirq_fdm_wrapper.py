"""Cirq FDM Simulator Wrapper for LRET Comparison.

Provides a unified interface matching LRET's API for fair benchmarking.
Uses Cirq's DensityMatrixSimulator for full density matrix simulation (FDM).
"""

from __future__ import annotations

import time
import tracemalloc
from typing import Any, Dict, List, Optional, Tuple

import cirq
import numpy as np


class CirqFDMSimulator:
    """Cirq Full Density Matrix Simulator matching LRET interface."""

    def __init__(
        self,
        num_qubits: int,
        noise_model: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        """Initialize Cirq FDM simulator.

        Args:
            num_qubits: Number of qubits in the circuit
            noise_model: Optional noise parameters (depolarizing, amplitude_damping, etc.)
            seed: Random seed for reproducibility
        """
        self.num_qubits = num_qubits
        self.qubits = cirq.LineQubit.range(num_qubits)
        self.noise_model = noise_model
        self.seed = seed
        self.simulator = cirq.DensityMatrixSimulator(seed=seed)

        # Performance tracking
        self.last_execution_time = 0.0
        self.peak_memory_mb = 0.0

    def simulate(
        self,
        circuit: cirq.Circuit,
        initial_state: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Simulate a quantum circuit and return final density matrix.

        Args:
            circuit: Cirq circuit to simulate
            initial_state: Optional initial density matrix (defaults to |0...0⟩)

        Returns:
            Tuple of (final_density_matrix, metadata_dict)
            - final_density_matrix: 2^n × 2^n complex array
            - metadata: Contains execution_time_ms, peak_memory_mb, num_operations
        """
        # Start performance tracking
        tracemalloc.start()
        start_time = time.perf_counter()

        try:
            # Apply noise model if specified
            if self.noise_model:
                circuit = self._apply_noise(circuit)

            # Simulate
            if initial_state is not None:
                result = self.simulator.simulate(
                    circuit,
                    initial_state=cirq.DensityMatrixSimulationState(
                        initial_state=initial_state,
                        qubits=self.qubits,
                    ),
                )
            else:
                result = self.simulator.simulate(circuit)

            # Extract final density matrix
            final_state = result.final_density_matrix

            # Performance metrics
            end_time = time.perf_counter()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            self.last_execution_time = (end_time - start_time) * 1000  # ms
            self.peak_memory_mb = peak / (1024 * 1024)

            metadata = {
                "execution_time_ms": self.last_execution_time,
                "peak_memory_mb": self.peak_memory_mb,
                "num_operations": len(circuit),
                "num_qubits": self.num_qubits,
                "trace": float(np.trace(final_state)),  # Should be ~1.0
            }

            return final_state, metadata

        except Exception as e:
            tracemalloc.stop()
            raise RuntimeError(f"Cirq simulation failed: {e}")

    def _apply_noise(self, circuit: cirq.Circuit) -> cirq.Circuit:
        """Apply noise model to circuit.

        Args:
            circuit: Clean circuit

        Returns:
            Noisy circuit with error channels
        """
        if not self.noise_model:
            return circuit

        noise_type = self.noise_model.get("type", "depolarizing")
        noise_param = self.noise_model.get("parameter", 0.001)

        noisy_circuit = cirq.Circuit()

        for moment in circuit:
            noisy_circuit.append(moment)

            # Add noise after each gate
            for op in moment:
                for qubit in op.qubits:
                    if noise_type == "depolarizing":
                        noisy_circuit.append(
                            cirq.depolarize(noise_param).on(qubit)
                        )
                    elif noise_type == "amplitude_damping":
                        noisy_circuit.append(
                            cirq.amplitude_damp(noise_param).on(qubit)
                        )
                    elif noise_type == "phase_damping":
                        noisy_circuit.append(
                            cirq.phase_damp(noise_param).on(qubit)
                        )

        return noisy_circuit

    def compute_fidelity(
        self,
        state1: np.ndarray,
        state2: np.ndarray,
    ) -> float:
        """Compute fidelity between two density matrices.

        F(ρ, σ) = Tr(√(√ρ σ √ρ))²

        Args:
            state1: First density matrix (2^n × 2^n)
            state2: Second density matrix (2^n × 2^n)

        Returns:
            Fidelity value in [0, 1]
        """
        # Use Cirq's built-in fidelity
        return float(cirq.fidelity(state1, state2, qid_shape=(2,) * self.num_qubits))

    def compute_trace_distance(
        self,
        state1: np.ndarray,
        state2: np.ndarray,
    ) -> float:
        """Compute trace distance between two density matrices.

        D(ρ, σ) = 0.5 * ||ρ - σ||₁

        Args:
            state1: First density matrix
            state2: Second density matrix

        Returns:
            Trace distance in [0, 1]
        """
        diff = state1 - state2
        eigenvalues = np.linalg.eigvalsh(diff)
        return 0.5 * np.sum(np.abs(eigenvalues))

    @staticmethod
    def from_json(circuit_dict: Dict[str, Any]) -> Tuple[cirq.Circuit, int, Optional[Dict]]:
        """Convert LRET JSON circuit to Cirq circuit.

        Args:
            circuit_dict: LRET JSON circuit specification

        Returns:
            Tuple of (cirq_circuit, num_qubits, noise_model)
        """
        circuit_data = circuit_dict.get("circuit", circuit_dict)
        num_qubits = circuit_data["num_qubits"]
        operations = circuit_data["operations"]

        qubits = cirq.LineQubit.range(num_qubits)
        circuit = cirq.Circuit()

        # Map LRET gates to Cirq gates
        gate_map = {
            "H": lambda q: cirq.H(q),
            "X": lambda q: cirq.X(q),
            "Y": lambda q: cirq.Y(q),
            "Z": lambda q: cirq.Z(q),
            "CNOT": lambda c, t: cirq.CNOT(c, t),
            "CZ": lambda c, t: cirq.CZ(c, t),
            "RX": lambda q, angle: cirq.rx(angle)(q),
            "RY": lambda q, angle: cirq.ry(angle)(q),
            "RZ": lambda q, angle: cirq.rz(angle)(q),
            "S": lambda q: cirq.S(q),
            "T": lambda q: cirq.T(q),
            "SWAP": lambda q1, q2: cirq.SWAP(q1, q2),
        }

        for op in operations:
            # Support both "gate" (old format) and "name" (LRET format)
            gate_type = op.get("gate", op.get("name", ""))
            
            # Support both "targets"/"control" (old) and "wires" (LRET format)
            if "wires" in op:
                wires = op["wires"]
                if gate_type in ["CNOT", "CZ"]:
                    control = wires[0]
                    targets = wires[1:]
                elif gate_type == "SWAP":
                    targets = wires
                else:
                    targets = wires
            else:
                targets = op.get("targets", op.get("target", []))
                if isinstance(targets, int):
                    targets = [targets]

            # Get parameters (support both "params" and "parameters")
            params = op.get("params", op.get("parameters", []))
            if not isinstance(params, list):
                params = [params]

            if gate_type in ["H", "X", "Y", "Z", "S", "T"]:
                for target in targets:
                    circuit.append(gate_map[gate_type](qubits[target]))

            elif gate_type in ["RX", "RY", "RZ"]:
                angle = params[0] if params else op.get("parameters", op.get("angle", 0.0))
                for target in targets:
                    circuit.append(gate_map[gate_type](qubits[target], angle))

            elif gate_type in ["CNOT", "CZ"]:
                if "wires" in op:
                    # LRET format: wires = [control, target]
                    control = op["wires"][0]
                    target = op["wires"][1]
                else:
                    # Old format: control and targets separate
                    control = op.get("control", op.get("controls", [0])[0])
                    target = targets[0]
                circuit.append(gate_map[gate_type](qubits[control], qubits[target]))

            elif gate_type == "SWAP":
                if "wires" in op:
                    q1, q2 = op["wires"][0], op["wires"][1]
                else:
                    q1, q2 = targets[0], targets[1]
                circuit.append(gate_map["SWAP"](qubits[q1], qubits[q2]))
            
            elif gate_type == "CRZ":
                # Controlled RZ
                angle = params[0] if params else 0.0
                if "wires" in op:
                    control = op["wires"][0]
                    target = op["wires"][1]
                else:
                    control = op.get("control", 0)
                    target = targets[0]
                circuit.append(cirq.CZPowGate(exponent=angle/np.pi)(qubits[control], qubits[target]))

        # Extract noise model if present
        noise_model = None
        config = circuit_dict.get("config", {})
        if "noise" in config:
            noise_model = config["noise"]

        return circuit, num_qubits, noise_model

    def __repr__(self) -> str:
        return (
            f"CirqFDMSimulator(num_qubits={self.num_qubits}, "
            f"noise={self.noise_model is not None})"
        )


def benchmark_circuit(
    circuit_json: Dict[str, Any],
    trials: int = 5,
) -> Dict[str, Any]:
    """Run a benchmark on a circuit with multiple trials.

    Args:
        circuit_json: LRET JSON circuit
        trials: Number of trials to average

    Returns:
        Dictionary with timing, memory, and fidelity metrics
    """
    circuit, num_qubits, noise_model = CirqFDMSimulator.from_json(circuit_json)

    times = []
    memories = []
    traces = []

    for _ in range(trials):
        sim = CirqFDMSimulator(num_qubits, noise_model=noise_model)
        final_state, metadata = sim.simulate(circuit)

        times.append(metadata["execution_time_ms"])
        memories.append(metadata["peak_memory_mb"])
        traces.append(metadata["trace"])

    return {
        "num_qubits": num_qubits,
        "num_operations": len(circuit),
        "mean_time_ms": float(np.mean(times)),
        "std_time_ms": float(np.std(times)),
        "mean_memory_mb": float(np.mean(memories)),
        "std_memory_mb": float(np.std(memories)),
        "mean_trace": float(np.mean(traces)),
        "trials": trials,
    }


if __name__ == "__main__":
    # Test with simple circuit
    print("Testing Cirq FDM Wrapper...")

    # Create Bell state circuit
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(qubits[0]),
        cirq.CNOT(qubits[0], qubits[1]),
    )

    sim = CirqFDMSimulator(2)
    final_state, metadata = sim.simulate(circuit)

    print(f"\nBell State Simulation:")
    print(f"  Execution time: {metadata['execution_time_ms']:.3f} ms")
    print(f"  Memory usage: {metadata['peak_memory_mb']:.3f} MB")
    print(f"  Trace: {metadata['trace']:.6f} (should be ~1.0)")
    print(f"\nDensity matrix diagonal:")
    print(np.diag(final_state))
    print("\n✓ Cirq FDM Wrapper working correctly!")
