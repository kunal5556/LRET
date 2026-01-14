"""Translate Qiskit QuantumCircuit to LRET JSON format.

This module converts Qiskit circuits into the JSON schema expected by
the LRET simulator core (qlret.api.simulate_json).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.circuit.library import (
    HGate, XGate, YGate, ZGate,
    SGate, SdgGate, TGate, TdgGate, SXGate,
    RXGate, RYGate, RZGate,
    PhaseGate, U1Gate, U2Gate, U3Gate, UGate,
    CXGate, CYGate, CZGate,
    SwapGate, iSwapGate,
)
from qiskit.circuit.measure import Measure
from qiskit.circuit.reset import Reset

__all__ = ["CircuitTranslator", "TranslationError"]


class TranslationError(ValueError):
    """Error during Qiskit to LRET translation."""


# Mapping from Qiskit gate names to LRET JSON gate names
GATE_MAP: Dict[str, str] = {
    # Single-qubit Clifford
    "h": "H",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "s": "S",
    "sdg": "SDG",
    "t": "T",
    "tdg": "TDG",
    "sx": "SX",
    # Rotation gates
    "rx": "RX",
    "ry": "RY",
    "rz": "RZ",
    # Phase gates
    "p": "U1",
    "u1": "U1",
    "u2": "U2",
    "u3": "U3",
    "u": "U3",
    # Two-qubit gates
    "cx": "CNOT",
    "cy": "CY",
    "cz": "CZ",
    "swap": "SWAP",
    "iswap": "ISWAP",
}


class CircuitTranslator:
    """Translates Qiskit QuantumCircuit objects to LRET JSON format."""

    def __init__(self, epsilon: float = 1e-4, shots: int = 1024):
        """Initialize translator with simulation parameters.
        
        Args:
            epsilon: SVD truncation threshold for LRET.
            shots: Number of measurement shots.
        """
        self.epsilon = epsilon
        self.shots = shots

    def translate(
        self,
        circuit: QuantumCircuit,
        observables: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Convert a Qiskit circuit to LRET JSON format.
        
        Args:
            circuit: Qiskit QuantumCircuit to translate.
            observables: Optional list of observable specifications for
                expectation value calculations.
        
        Returns:
            Dictionary matching LRET JSON schema.
        
        Raises:
            TranslationError: If circuit contains unsupported operations.
        """
        operations = []
        # Check if circuit has measurements (based on classical bits)
        has_measurement = circuit.num_clbits > 0

        for circuit_instruction in circuit.data:
            # Qiskit 1.2+ uses named attributes instead of tuple unpacking
            instruction = circuit_instruction.operation
            qargs = circuit_instruction.qubits
            cargs = circuit_instruction.clbits
            
            # Check for measurement instructions
            if instruction.name.lower() == "measure":
                has_measurement = True
            op_json = self._translate_instruction(instruction, qargs, cargs)
            if op_json is not None:
                operations.append(op_json)

        # Build the circuit JSON
        circuit_json: Dict[str, Any] = {
            "num_qubits": circuit.num_qubits,
            "operations": operations,
        }

        # Add observables if provided, otherwise default to Z on qubit 0
        if observables:
            circuit_json["observables"] = observables
        else:
            # Default: measure Z expectation on all qubits
            # Note: LRET uses "operator" (singular) for single-qubit PAULI observables
            circuit_json["observables"] = [
                {"type": "PAULI", "operator": "Z", "wires": [i], "coefficient": 1.0}
                for i in range(circuit.num_qubits)
            ]

        # Build config
        config: Dict[str, Any] = {
            "epsilon": self.epsilon,
            "max_rank": 0,  # 0 = unlimited
            "use_noise": False,
        }

        if has_measurement:
            config["shots"] = self.shots

        return {
            "circuit": circuit_json,
            "config": config,
        }

    def _translate_instruction(
        self,
        instruction: Instruction,
        qargs: List,
        cargs: List,
    ) -> Optional[Dict[str, Any]]:
        """Translate a single Qiskit instruction to LRET JSON.
        
        Args:
            instruction: The Qiskit instruction/gate.
            qargs: Qubit arguments.
            cargs: Classical bit arguments.
        
        Returns:
            JSON dict for the operation, or None to skip.
        
        Raises:
            TranslationError: If instruction is not supported.
        """
        name = instruction.name.lower()
        wires = [qarg._index for qarg in qargs]

        # Handle measurement - LRET uses shots in config, not explicit MEASURE ops
        if isinstance(instruction, Measure) or name == "measure":
            # Skip - measurement is handled via shots parameter
            return None

        # Handle reset (prepare |0⟩) - skip for now
        if isinstance(instruction, Reset) or name == "reset":
            # Skip - LRET starts in |0⟩ state by default
            return None

        # Handle barrier (no-op for simulation)
        if name == "barrier":
            return None

        # Handle standard gates
        lret_name = GATE_MAP.get(name)
        if lret_name is None:
            raise TranslationError(
                f"Unsupported gate '{instruction.name}'. "
                f"Supported gates: {list(GATE_MAP.keys())}"
            )

        result: Dict[str, Any] = {
            "name": lret_name,
            "wires": wires,
        }

        # Extract parameters
        if instruction.params:
            result["params"] = [float(p) for p in instruction.params]

        return result

    def translate_batch(
        self,
        circuits: List[QuantumCircuit],
    ) -> List[Dict[str, Any]]:
        """Translate multiple circuits.
        
        Args:
            circuits: List of Qiskit circuits.
        
        Returns:
            List of LRET JSON circuit specifications.
        """
        return [self.translate(circuit) for circuit in circuits]
