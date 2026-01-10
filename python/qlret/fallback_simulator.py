"""Pure-Python fallback quantum simulator for QLRET.

This module provides a basic density matrix simulator that runs when
the native C++ backend is not available. It enables the PennyLane device
to function (albeit more slowly) without requiring compilation.

The simulator supports:
- Basic single and two-qubit gates
- Density matrix evolution
- Expectation value computation
- Sampling from measurement outcomes

This is NOT a low-rank simulator - it uses full density matrices.
For production use, build the native C++ backend.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional
import time


# =============================================================================
# Gate Matrices
# =============================================================================

# Single-qubit gates
GATE_MATRICES = {
    "I": np.array([[1, 0], [0, 1]], dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    "H": np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2),
    "S": np.array([[1, 0], [0, 1j]], dtype=np.complex128),
    "SDG": np.array([[1, 0], [0, -1j]], dtype=np.complex128),
    "T": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128),
    "TDG": np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=np.complex128),
    "SX": np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=np.complex128) / 2,
}

# Two-qubit gates
TWO_QUBIT_GATES = {
    "CNOT": np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=np.complex128),
    "CZ": np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1],
    ], dtype=np.complex128),
    "CY": np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, -1j],
        [0, 0, 1j, 0],
    ], dtype=np.complex128),
    "SWAP": np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.complex128),
    "ISWAP": np.array([
        [1, 0, 0, 0],
        [0, 0, 1j, 0],
        [0, 1j, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.complex128),
}


def rx_gate(theta: float) -> np.ndarray:
    """RX rotation gate."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def ry_gate(theta: float) -> np.ndarray:
    """RY rotation gate."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz_gate(theta: float) -> np.ndarray:
    """RZ rotation gate."""
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=np.complex128)


def u1_gate(lam: float) -> np.ndarray:
    """U1 (phase) gate."""
    return np.array([[1, 0], [0, np.exp(1j * lam)]], dtype=np.complex128)


def u2_gate(phi: float, lam: float) -> np.ndarray:
    """U2 gate."""
    return np.array([
        [1, -np.exp(1j * lam)],
        [np.exp(1j * phi), np.exp(1j * (phi + lam))]
    ], dtype=np.complex128) / np.sqrt(2)


def u3_gate(theta: float, phi: float, lam: float) -> np.ndarray:
    """U3 (general single-qubit) gate."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([
        [c, -np.exp(1j * lam) * s],
        [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]
    ], dtype=np.complex128)


def get_gate_matrix(name: str, params: Optional[List[float]] = None) -> np.ndarray:
    """Get the matrix for a gate by name."""
    # Fixed gates
    if name in GATE_MATRICES:
        return GATE_MATRICES[name]
    if name in TWO_QUBIT_GATES:
        return TWO_QUBIT_GATES[name]
    
    # Parametric gates
    if params is None:
        params = []
    
    if name == "RX":
        return rx_gate(params[0])
    elif name == "RY":
        return ry_gate(params[0])
    elif name == "RZ":
        return rz_gate(params[0])
    elif name == "U1":
        return u1_gate(params[0])
    elif name == "U2":
        return u2_gate(params[0], params[1])
    elif name == "U3":
        return u3_gate(params[0], params[1], params[2])
    else:
        raise ValueError(f"Unknown gate: {name}")


# =============================================================================
# Density Matrix Operations
# =============================================================================

class DensityMatrixSimulator:
    """Simple density matrix quantum simulator."""
    
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits
        # Initialize to |0...0><0...0|
        self.rho = np.zeros((self.dim, self.dim), dtype=np.complex128)
        self.rho[0, 0] = 1.0
    
    def apply_single_qubit_gate(self, gate_matrix: np.ndarray, qubit: int) -> None:
        """Apply a single-qubit gate to the density matrix."""
        # Build the full operator: I ⊗ ... ⊗ G ⊗ ... ⊗ I
        full_op = self._embed_single_qubit_op(gate_matrix, qubit)
        # ρ' = U ρ U†
        self.rho = full_op @ self.rho @ full_op.conj().T
    
    def apply_two_qubit_gate(self, gate_matrix: np.ndarray, qubit1: int, qubit2: int) -> None:
        """Apply a two-qubit gate to the density matrix."""
        # Build the full operator
        full_op = self._embed_two_qubit_op(gate_matrix, qubit1, qubit2)
        # ρ' = U ρ U†
        self.rho = full_op @ self.rho @ full_op.conj().T
    
    def _embed_single_qubit_op(self, gate: np.ndarray, qubit: int) -> np.ndarray:
        """Embed a single-qubit gate into the full Hilbert space."""
        ops = [np.eye(2, dtype=np.complex128)] * self.num_qubits
        ops[qubit] = gate
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    def _embed_two_qubit_op(self, gate: np.ndarray, qubit1: int, qubit2: int) -> np.ndarray:
        """Embed a two-qubit gate into the full Hilbert space."""
        # This is more complex - need to handle non-adjacent qubits
        if abs(qubit1 - qubit2) != 1:
            # Use SWAP to bring qubits adjacent, apply gate, SWAP back
            return self._embed_nonlocal_two_qubit(gate, qubit1, qubit2)
        
        # Adjacent qubits - simpler case
        min_q, max_q = min(qubit1, qubit2), max(qubit1, qubit2)
        
        # If qubit order is reversed, swap the gate
        if qubit1 > qubit2:
            # Swap indices in the gate (for controlled gates, this matters)
            swap_gate = TWO_QUBIT_GATES["SWAP"]
            gate = swap_gate @ gate @ swap_gate
        
        # Build full operator
        ops_before = [np.eye(2, dtype=np.complex128)] * min_q
        ops_after = [np.eye(2, dtype=np.complex128)] * (self.num_qubits - max_q - 1)
        
        result = np.eye(1, dtype=np.complex128)
        for op in ops_before:
            result = np.kron(result, op)
        result = np.kron(result, gate)
        for op in ops_after:
            result = np.kron(result, op)
        
        return result
    
    def _embed_nonlocal_two_qubit(self, gate: np.ndarray, qubit1: int, qubit2: int) -> np.ndarray:
        """Handle two-qubit gates on non-adjacent qubits using SWAPs."""
        # Simple approach: use permutation matrices
        # More efficient: direct index manipulation
        
        # For now, use direct construction via index manipulation
        full_gate = np.zeros((self.dim, self.dim), dtype=np.complex128)
        
        for i in range(self.dim):
            for j in range(self.dim):
                # Extract bits for qubit1 and qubit2 from indices i and j
                i1 = (i >> (self.num_qubits - 1 - qubit1)) & 1
                i2 = (i >> (self.num_qubits - 1 - qubit2)) & 1
                j1 = (j >> (self.num_qubits - 1 - qubit1)) & 1
                j2 = (j >> (self.num_qubits - 1 - qubit2)) & 1
                
                # Check if other bits match
                i_other = i & ~((1 << (self.num_qubits - 1 - qubit1)) | (1 << (self.num_qubits - 1 - qubit2)))
                j_other = j & ~((1 << (self.num_qubits - 1 - qubit1)) | (1 << (self.num_qubits - 1 - qubit2)))
                
                if i_other == j_other:
                    # Get gate element
                    gate_row = i1 * 2 + i2
                    gate_col = j1 * 2 + j2
                    full_gate[i, j] = gate[gate_row, gate_col]
        
        return full_gate
    
    def expectation_value(self, observable: np.ndarray) -> float:
        """Compute expectation value Tr(ρ O)."""
        return np.real(np.trace(self.rho @ observable))
    
    def get_probabilities(self) -> np.ndarray:
        """Get measurement probabilities (diagonal of density matrix)."""
        return np.real(np.diag(self.rho))
    
    def sample(self, num_shots: int) -> List[int]:
        """Sample measurement outcomes."""
        probs = self.get_probabilities()
        # Normalize (handle numerical errors)
        probs = np.abs(probs)
        probs /= probs.sum()
        return list(np.random.choice(self.dim, size=num_shots, p=probs))


# =============================================================================
# Main Simulation Function
# =============================================================================

def simulate_circuit(circuit_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulate a quantum circuit specified in JSON format.
    
    Parameters
    ----------
    circuit_json : dict
        Circuit specification with format:
        {
            "circuit": {
                "num_qubits": int,
                "operations": [{"name": str, "wires": [int], "params": [float]}],
                "observables": [{"type": str, ...}]
            },
            "config": {
                "epsilon": float,
                "shots": int (optional)
            }
        }
    
    Returns
    -------
    dict
        Results with format:
        {
            "status": "success",
            "execution_time_ms": float,
            "final_rank": int,
            "expectation_values": [float],
            "samples": [int] (if shots specified)
        }
    """
    start_time = time.time()
    
    circuit = circuit_json.get("circuit", {})
    config = circuit_json.get("config", {})
    
    num_qubits = circuit.get("num_qubits", 1)
    operations = circuit.get("operations", [])
    observables = circuit.get("observables", [])
    shots = config.get("shots")
    
    # Initialize simulator
    sim = DensityMatrixSimulator(num_qubits)
    
    # Apply operations
    for op in operations:
        name = op.get("name", "")
        wires = op.get("wires", [])
        params = op.get("params", [])
        
        gate_matrix = get_gate_matrix(name, params if params else None)
        
        if len(wires) == 1:
            sim.apply_single_qubit_gate(gate_matrix, wires[0])
        elif len(wires) == 2:
            sim.apply_two_qubit_gate(gate_matrix, wires[0], wires[1])
        else:
            raise ValueError(f"Gate {name} with {len(wires)} wires not supported")
    
    # Compute expectation values
    expectation_values = []
    for obs in observables:
        obs_type = obs.get("type", "PAULI")
        
        if obs_type == "PAULI":
            pauli = obs.get("operator", "Z")
            wire = obs.get("wires", [0])[0]
            coeff = obs.get("coefficient", 1.0)
            
            pauli_matrix = GATE_MATRICES.get(pauli, GATE_MATRICES["Z"])
            full_obs = sim._embed_single_qubit_op(pauli_matrix, wire)
            exp_val = sim.expectation_value(full_obs) * coeff
            expectation_values.append(float(exp_val))
        
        elif obs_type == "TENSOR":
            operators = obs.get("operators", [])
            wires = obs.get("wires", [])
            coeff = obs.get("coefficient", 1.0)
            
            # Build tensor product observable
            full_obs = np.eye(sim.dim, dtype=np.complex128)
            for pauli, wire in zip(operators, wires):
                pauli_matrix = GATE_MATRICES.get(pauli, GATE_MATRICES["I"])
                single_op = sim._embed_single_qubit_op(pauli_matrix, wire)
                full_obs = full_obs @ single_op
            
            exp_val = sim.expectation_value(full_obs) * coeff
            expectation_values.append(float(exp_val))
        
        elif obs_type == "HERMITIAN":
            wires = obs.get("wires", [])
            matrix_real = np.array(obs.get("matrix_real", []))
            matrix_imag = np.array(obs.get("matrix_imag", []))
            coeff = obs.get("coefficient", 1.0)
            
            matrix = matrix_real + 1j * matrix_imag
            
            if len(wires) == 1:
                full_obs = sim._embed_single_qubit_op(matrix, wires[0])
            else:
                # Multi-qubit Hermitian - embed directly
                full_obs = sim._embed_nonlocal_two_qubit(matrix, wires[0], wires[1])
            
            exp_val = sim.expectation_value(full_obs) * coeff
            expectation_values.append(float(exp_val))
    
    # Sample if shots specified
    samples = None
    if shots is not None and shots > 0:
        samples = sim.sample(shots)
    
    execution_time_ms = (time.time() - start_time) * 1000
    
    return {
        "status": "success",
        "execution_time_ms": execution_time_ms,
        "final_rank": sim.dim,  # Full rank for density matrix
        "expectation_values": expectation_values,
        "samples": samples,
        "backend": "python_fallback",
    }
