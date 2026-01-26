"""Circuit Translator: Cirq → LRET JSON.

Converts Cirq circuits to LRET's JSON input format for simulation.
Handles qubit mapping (LineQubit, GridQubit, NamedQubit), gate translation
including power gates, and measurement extraction.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

try:
    import cirq
except ImportError:
    raise ImportError(
        "Cirq is required for lret_cirq. Install with: pip install cirq>=1.3.0"
    )

__all__ = ["CircuitTranslator", "TranslationError"]


class TranslationError(ValueError):
    """Raised when circuit translation fails."""
    pass


class CircuitTranslator:
    """
    Translates Cirq circuits to LRET JSON format.
    
    Handles:
    - Qubit mapping (LineQubit, GridQubit, NamedQubit → integers)
    - Gate translation (including power gates)
    - Measurement extraction with key tracking
    - Parameter validation
    
    Example:
        >>> translator = CircuitTranslator()
        >>> lret_json = translator.translate(circuit, epsilon=1e-4, shots=1000)
    """
    
    def __init__(self):
        """Initialize translator with gate mapping."""
        self._gate_map = self._build_gate_map()
        self._qubit_map: Dict[cirq.Qid, int] = {}
        self._measurement_keys: List[str] = []
        self._measurement_qubits: Dict[str, List[cirq.Qid]] = {}
    
    def get_qubit_map(self) -> Dict[cirq.Qid, int]:
        """Get the current qubit mapping (copy)."""
        return self._qubit_map.copy()
    
    def get_measurement_keys(self) -> List[str]:
        """Get list of measurement keys in order."""
        return self._measurement_keys.copy()
    
    def get_measurement_qubits(self) -> Dict[str, List[cirq.Qid]]:
        """Get mapping from measurement key to qubits."""
        return {k: list(v) for k, v in self._measurement_qubits.items()}
    
    def translate(
        self,
        circuit: cirq.Circuit,
        epsilon: float = 1e-4,
        shots: int = 1024,
        noise_model: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Translate Cirq circuit to LRET JSON.
        
        Args:
            circuit: Cirq circuit to translate
            epsilon: Truncation threshold for low-rank approximation
            shots: Number of measurement samples
            noise_model: Optional noise configuration dict
            seed: Random seed for reproducibility
            
        Returns:
            LRET JSON dict with circuit and config
            
        Raises:
            TranslationError: If circuit contains unsupported operations
        """
        # Reset state for new translation
        self._qubit_map = {}
        self._measurement_keys = []
        self._measurement_qubits = {}
        
        # Build qubit mapping from all qubits in circuit
        self._qubit_map = self._build_qubit_map(circuit)
        num_qubits = len(self._qubit_map)
        
        if num_qubits == 0:
            raise TranslationError("Circuit has no qubits")
        
        if num_qubits > 28:
            warnings.warn(
                f"Circuit has {num_qubits} qubits. "
                "LRET performance may degrade above 20 qubits.",
                UserWarning
            )
        
        # Translate operations from all moments
        operations = []
        has_measurements = False
        
        for moment in circuit:
            for op in moment:
                try:
                    # Check if it's a measurement before translating
                    if isinstance(op.gate, cirq.MeasurementGate):
                        has_measurements = True
                    
                    lret_ops = self._translate_operation(op)
                    operations.extend(lret_ops)
                except Exception as e:
                    raise TranslationError(
                        f"Failed to translate operation {op}: {e}"
                    ) from e
        
        # Build LRET JSON structure
        circuit_json: Dict[str, Any] = {
            "num_qubits": num_qubits,
            "operations": operations,
        }
        
        # Add default observables (Z on all qubits) for expectation values
        circuit_json["observables"] = [
            {"type": "PAULI", "operator": "Z", "wires": [i], "coefficient": 1.0}
            for i in range(num_qubits)
        ]
        
        # Build config
        config: Dict[str, Any] = {
            "epsilon": epsilon,
            "max_rank": 0,  # 0 = unlimited
            "use_noise": noise_model is not None,
        }
        
        if has_measurements:
            config["shots"] = shots
        
        if seed is not None:
            config["seed"] = seed
        
        lret_json: Dict[str, Any] = {
            "circuit": circuit_json,
            "config": config,
        }
        
        # Add noise model if provided
        if noise_model:
            lret_json["noise"] = noise_model
        
        return lret_json
    
    def _build_qubit_map(self, circuit: cirq.Circuit) -> Dict[cirq.Qid, int]:
        """
        Create mapping from Cirq qubits to integer indices.
        
        Handles all Cirq qubit types:
        - LineQubit(x) → sorted by x
        - GridQubit(row, col) → sorted by (row, col)
        - NamedQubit(name) → sorted by name
        - Mixed types → Cirq's default sorting
        
        Args:
            circuit: Cirq circuit
            
        Returns:
            Dict mapping cirq.Qid to int index (0-based, contiguous)
        """
        qubits = sorted(circuit.all_qubits())
        return {qubit: idx for idx, qubit in enumerate(qubits)}
    
    def _translate_operation(self, op: cirq.Operation) -> List[Dict[str, Any]]:
        """
        Translate a single Cirq operation to LRET format.
        
        Returns a list because some gates may need decomposition.
        
        Args:
            op: Cirq operation
            
        Returns:
            List of LRET operation dicts
            
        Raises:
            TranslationError: If operation not supported
        """
        gate = op.gate
        qubits = [self._qubit_map[q] for q in op.qubits]
        
        # Handle measurements specially
        if isinstance(gate, cirq.MeasurementGate):
            return self._translate_measurement(op)
        
        # Handle identity (no-op)
        if isinstance(gate, cirq.IdentityGate):
            return []  # Skip identity gates
        
        # Handle wait (no-op for simulation)
        if isinstance(gate, cirq.WaitGate):
            return []
        
        # Try direct gate type mapping first
        gate_type = type(gate)
        if gate_type in self._gate_map:
            gate_info = self._gate_map[gate_type]
            return [{"name": gate_info["lret_name"], "wires": qubits}]
        
        # Handle power gates (XPowGate, YPowGate, ZPowGate, etc.)
        if isinstance(gate, cirq.XPowGate):
            return self._translate_power_gate(gate, qubits, "X", "RX")
        if isinstance(gate, cirq.YPowGate):
            return self._translate_power_gate(gate, qubits, "Y", "RY")
        if isinstance(gate, cirq.ZPowGate):
            return self._translate_z_power_gate(gate, qubits)
        if isinstance(gate, cirq.HPowGate):
            return self._translate_h_power_gate(gate, qubits)
        
        # Handle CZPowGate and CXPowGate
        if isinstance(gate, cirq.CZPowGate):
            return self._translate_cz_power_gate(gate, qubits)
        if isinstance(gate, cirq.CXPowGate):
            return self._translate_cx_power_gate(gate, qubits)
        
        # Handle explicit rotation gates
        if isinstance(gate, cirq.Rx):
            angle = float(gate._rads)
            return [{"name": "RX", "wires": qubits, "params": [angle]}]
        if isinstance(gate, cirq.Ry):
            angle = float(gate._rads)
            return [{"name": "RY", "wires": qubits, "params": [angle]}]
        if isinstance(gate, cirq.Rz):
            angle = float(gate._rads)
            return [{"name": "RZ", "wires": qubits, "params": [angle]}]
        
        # Handle PhasedXPowGate
        if isinstance(gate, cirq.PhasedXPowGate):
            return self._translate_phased_x_pow_gate(gate, qubits)
        
        # Handle ISwapPowGate
        if isinstance(gate, cirq.ISwapPowGate):
            return self._translate_iswap_power_gate(gate, qubits)
        
        # Handle SwapPowGate
        if isinstance(gate, cirq.SwapPowGate):
            return self._translate_swap_power_gate(gate, qubits)
        
        # Unsupported gate - provide helpful error
        supported = list(self._gate_map.keys())
        raise TranslationError(
            f"Unsupported gate: {gate} (type: {type(gate).__name__}). "
            f"Supported gate types include: H, X, Y, Z, S, T, CNOT, CZ, SWAP, "
            f"and power gates (XPowGate, YPowGate, ZPowGate, etc.)"
        )
    
    def _translate_measurement(self, op: cirq.Operation) -> List[Dict[str, Any]]:
        """
        Translate measurement operation.
        
        Extracts measurement key and tracks qubits for result conversion.
        Note: LRET handles measurements via shots, but we track keys for result mapping.
        """
        gate = op.gate
        qubits = list(op.qubits)
        qubit_indices = [self._qubit_map[q] for q in qubits]
        
        # Extract measurement key
        key = str(gate.key) if hasattr(gate, 'key') and gate.key else 'result'
        
        # Track measurement for result conversion
        if key not in self._measurement_keys:
            self._measurement_keys.append(key)
            self._measurement_qubits[key] = qubits
        else:
            # Append to existing key
            self._measurement_qubits[key].extend(qubits)
        
        # Return empty - LRET handles measurements via shots parameter
        # We just need to track which qubits are measured
        return []
    
    def _translate_power_gate(
        self,
        gate: Union[cirq.XPowGate, cirq.YPowGate],
        qubits: List[int],
        base_name: str,
        rotation_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Translate power gates (XPowGate, YPowGate).
        
        Strategy:
        - Exponent = 1.0 → discrete gate (X, Y)
        - Exponent = 0.5 → half gate (SX for X)
        - Exponent = -1.0 → same as 1.0 (X^2 = I)
        - Other → rotation gate (RX, RY)
        """
        exponent = float(gate.exponent)
        global_shift = float(gate.global_shift) if hasattr(gate, 'global_shift') else 0.0
        
        # Normalize exponent to [-1, 1] range
        exponent = exponent % 2
        if exponent > 1:
            exponent -= 2
        
        # Handle common cases
        if np.isclose(exponent, 0.0):
            return []  # Identity
        
        if np.isclose(abs(exponent), 1.0):
            return [{"name": base_name, "wires": qubits}]
        
        if base_name == "X" and np.isclose(exponent, 0.5):
            return [{"name": "SX", "wires": qubits}]
        
        if base_name == "X" and np.isclose(exponent, -0.5):
            # SX†
            return [{"name": "RX", "wires": qubits, "params": [-np.pi / 2]}]
        
        # General case: convert to rotation
        # XPowGate(e) = e^{i π e/2} RX(π e) (up to global phase)
        angle = exponent * np.pi
        return [{"name": rotation_name, "wires": qubits, "params": [angle]}]
    
    def _translate_z_power_gate(
        self,
        gate: cirq.ZPowGate,
        qubits: List[int],
    ) -> List[Dict[str, Any]]:
        """
        Translate ZPowGate with special cases for S, T, Z.
        """
        exponent = float(gate.exponent)
        
        # Normalize exponent
        exponent = exponent % 2
        if exponent > 1:
            exponent -= 2
        
        # Common cases
        if np.isclose(exponent, 0.0):
            return []  # Identity
        
        if np.isclose(abs(exponent), 1.0):
            return [{"name": "Z", "wires": qubits}]
        
        if np.isclose(exponent, 0.5):
            return [{"name": "S", "wires": qubits}]
        
        if np.isclose(exponent, -0.5):
            return [{"name": "SDG", "wires": qubits}]
        
        if np.isclose(exponent, 0.25):
            return [{"name": "T", "wires": qubits}]
        
        if np.isclose(exponent, -0.25):
            return [{"name": "TDG", "wires": qubits}]
        
        # General case: RZ rotation
        angle = exponent * np.pi
        return [{"name": "RZ", "wires": qubits, "params": [angle]}]
    
    def _translate_h_power_gate(
        self,
        gate: cirq.HPowGate,
        qubits: List[int],
    ) -> List[Dict[str, Any]]:
        """Translate HPowGate."""
        exponent = float(gate.exponent)
        
        if np.isclose(exponent % 2, 1.0):
            return [{"name": "H", "wires": qubits}]
        
        if np.isclose(exponent % 2, 0.0):
            return []  # Identity
        
        # Decompose H^e for fractional exponents
        # H = (X + Z) / sqrt(2), so H^e needs decomposition
        # For simplicity, decompose as RY and RZ
        angle = exponent * np.pi / 2
        return [
            {"name": "RY", "wires": qubits, "params": [angle]},
            {"name": "RZ", "wires": qubits, "params": [angle]},
        ]
    
    def _translate_cz_power_gate(
        self,
        gate: cirq.CZPowGate,
        qubits: List[int],
    ) -> List[Dict[str, Any]]:
        """Translate CZPowGate."""
        exponent = float(gate.exponent)
        
        if np.isclose(exponent % 2, 1.0):
            return [{"name": "CZ", "wires": qubits}]
        
        if np.isclose(exponent % 2, 0.0):
            return []  # Identity
        
        # For fractional CZ, we need controlled-RZ
        # CZ^e = diag(1, 1, 1, e^{i π e})
        # Decompose as CZ + single-qubit phases
        angle = exponent * np.pi
        return [
            {"name": "CZ", "wires": qubits},
            {"name": "RZ", "wires": [qubits[1]], "params": [angle - np.pi]},
        ] if not np.isclose(exponent, 1.0) else [{"name": "CZ", "wires": qubits}]
    
    def _translate_cx_power_gate(
        self,
        gate: cirq.CXPowGate,
        qubits: List[int],
    ) -> List[Dict[str, Any]]:
        """Translate CXPowGate (CNOT power gate)."""
        exponent = float(gate.exponent)
        
        if np.isclose(exponent % 2, 1.0):
            return [{"name": "CNOT", "wires": qubits}]
        
        if np.isclose(exponent % 2, 0.0):
            return []  # Identity
        
        # Decompose CX^e using RX on target controlled by source
        angle = exponent * np.pi
        # CX^e ≈ some combination - for now, error on non-integer
        raise TranslationError(
            f"Non-integer CXPowGate exponent ({exponent}) not yet supported. "
            "Please decompose manually or use integer exponents."
        )
    
    def _translate_phased_x_pow_gate(
        self,
        gate: cirq.PhasedXPowGate,
        qubits: List[int],
    ) -> List[Dict[str, Any]]:
        """
        Translate PhasedXPowGate.
        
        PhasedXPowGate(p, e) = Z^(-p) X^e Z^p
        """
        exponent = float(gate.exponent)
        phase_exponent = float(gate.phase_exponent)
        
        if np.isclose(exponent, 0.0):
            return []  # Identity
        
        # Decompose as Z^(-p) X^e Z^p
        ops = []
        
        if not np.isclose(phase_exponent, 0.0):
            z_angle = phase_exponent * np.pi
            ops.append({"name": "RZ", "wires": qubits, "params": [z_angle]})
        
        x_angle = exponent * np.pi
        ops.append({"name": "RX", "wires": qubits, "params": [x_angle]})
        
        if not np.isclose(phase_exponent, 0.0):
            z_angle = -phase_exponent * np.pi
            ops.append({"name": "RZ", "wires": qubits, "params": [z_angle]})
        
        return ops
    
    def _translate_iswap_power_gate(
        self,
        gate: cirq.ISwapPowGate,
        qubits: List[int],
    ) -> List[Dict[str, Any]]:
        """Translate ISwapPowGate."""
        exponent = float(gate.exponent)
        
        if np.isclose(exponent, 0.0):
            return []  # Identity
        
        if np.isclose(abs(exponent), 1.0):
            return [{"name": "ISWAP", "wires": qubits}]
        
        # ISWAP^e decomposition is complex; raise error for non-integer
        raise TranslationError(
            f"Non-integer ISwapPowGate exponent ({exponent}) not yet supported. "
            "Please use integer exponents or decompose manually."
        )
    
    def _translate_swap_power_gate(
        self,
        gate: cirq.SwapPowGate,
        qubits: List[int],
    ) -> List[Dict[str, Any]]:
        """Translate SwapPowGate."""
        exponent = float(gate.exponent)
        
        if np.isclose(exponent, 0.0):
            return []  # Identity
        
        if np.isclose(abs(exponent), 1.0):
            return [{"name": "SWAP", "wires": qubits}]
        
        # SWAP^0.5 = √SWAP can be decomposed
        if np.isclose(exponent, 0.5):
            # √SWAP decomposition: CNOT-based
            return [
                {"name": "CNOT", "wires": qubits},
                {"name": "H", "wires": [qubits[1]]},
                {"name": "CNOT", "wires": [qubits[1], qubits[0]]},
                {"name": "H", "wires": [qubits[1]]},
                {"name": "RZ", "wires": [qubits[0]], "params": [np.pi / 4]},
                {"name": "RZ", "wires": [qubits[1]], "params": [np.pi / 4]},
            ]
        
        raise TranslationError(
            f"SwapPowGate exponent ({exponent}) not supported. "
            "Supported: 0, 0.5, 1.0"
        )
    
    def _build_gate_map(self) -> Dict[type, Dict[str, Any]]:
        """
        Build mapping from Cirq gate types to LRET gate info.
        
        Returns:
            Dict mapping gate class to LRET info dict
        """
        gate_map = {}
        
        # We need to handle most gates via power gate logic
        # These are for explicit gate classes that aren't power gates
        
        # SWAP gate (explicit class)
        try:
            gate_map[cirq.SWAP.__class__] = {"lret_name": "SWAP"}
        except:
            pass
        
        # CNOT gate
        try:
            gate_map[cirq.CNOT.__class__] = {"lret_name": "CNOT"}
        except:
            pass
        
        # CZ gate
        try:
            gate_map[cirq.CZ.__class__] = {"lret_name": "CZ"}
        except:
            pass
        
        return gate_map
