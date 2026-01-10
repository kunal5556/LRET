"""Standard benchmark circuits for quantum device comparison.

This module provides circuit generators for various benchmark scenarios:
- Random circuits (testing general performance)
- QFT (Quantum Fourier Transform)
- QAOA (Quantum Approximate Optimization)
- VQE (Variational Quantum Eigensolver)
- QNN (Quantum Neural Network)
- Grover's search
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pennylane as qml


class CircuitType(Enum):
    """Types of benchmark circuits."""
    RANDOM = "random"
    QFT = "qft"
    QAOA = "qaoa"
    VQE = "vqe"
    QNN = "qnn"
    GROVER = "grover"
    GHZSTATE = "ghz"
    RANDOM_PARAMS = "random_params"


@dataclass
class CircuitSpec:
    """Specification for a benchmark circuit."""
    circuit_type: CircuitType
    num_qubits: int
    depth: int
    num_params: int = 0
    observable: Optional[str] = "Z0"  # Measurement observable
    
    @property
    def name(self) -> str:
        return f"{self.circuit_type.value}_{self.num_qubits}q_d{self.depth}"


def create_random_circuit(num_qubits: int, depth: int, seed: int = 42):
    """Create a random circuit with RX, RY, RZ, and CNOT gates.
    
    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    depth : int
        Number of layers.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    circuit_fn : callable
        Function that applies the circuit.
    num_params : int
        Number of parameters.
    """
    rng = np.random.default_rng(seed)
    
    # Pre-generate random parameters and gate choices
    params = rng.uniform(0, 2 * np.pi, size=(depth, num_qubits, 3))
    
    def circuit():
        for layer in range(depth):
            # Single qubit rotations
            for q in range(num_qubits):
                qml.RX(params[layer, q, 0], wires=q)
                qml.RY(params[layer, q, 1], wires=q)
                qml.RZ(params[layer, q, 2], wires=q)
            
            # Entangling layer (nearest-neighbor CNOTs)
            for q in range(0, num_qubits - 1, 2):
                qml.CNOT(wires=[q, q + 1])
            for q in range(1, num_qubits - 1, 2):
                qml.CNOT(wires=[q, q + 1])
    
    return circuit, 0  # No trainable params


def create_parametric_random_circuit(num_qubits: int, depth: int):
    """Create a random parametric circuit.
    
    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    depth : int
        Number of layers.
    
    Returns
    -------
    circuit_fn : callable
        Function(params) that applies the circuit.
    num_params : int
        Number of parameters.
    """
    num_params = depth * num_qubits * 3
    
    def circuit(params):
        idx = 0
        for layer in range(depth):
            for q in range(num_qubits):
                qml.RX(params[idx], wires=q)
                qml.RY(params[idx + 1], wires=q)
                qml.RZ(params[idx + 2], wires=q)
                idx += 3
            
            # Entangling layer
            for q in range(0, num_qubits - 1, 2):
                qml.CNOT(wires=[q, q + 1])
            for q in range(1, num_qubits - 1, 2):
                qml.CNOT(wires=[q, q + 1])
    
    return circuit, num_params


def create_qft_circuit(num_qubits: int):
    """Create QFT (Quantum Fourier Transform) circuit.
    
    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    
    Returns
    -------
    circuit_fn : callable
        Function that applies QFT.
    depth : int
        Circuit depth.
    """
    def circuit():
        # Initialize with some state
        for i in range(num_qubits):
            qml.Hadamard(wires=i)
        
        # QFT
        for i in range(num_qubits):
            qml.Hadamard(wires=i)
            for j in range(i + 1, num_qubits):
                angle = np.pi / (2 ** (j - i))
                qml.ControlledPhaseShift(angle, wires=[j, i])
        
        # Swap qubits
        for i in range(num_qubits // 2):
            qml.SWAP(wires=[i, num_qubits - i - 1])
    
    depth = num_qubits + num_qubits * (num_qubits - 1) // 2
    return circuit, depth


def create_qaoa_circuit(num_qubits: int, depth: int):
    """Create QAOA circuit for MaxCut problem.
    
    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    depth : int
        Number of QAOA layers.
    
    Returns
    -------
    circuit_fn : callable
        Function(gamma, beta) that applies QAOA circuit.
    num_params : int
        Number of parameters.
    """
    # Create a simple ring graph for MaxCut
    edges = [(i, (i + 1) % num_qubits) for i in range(num_qubits)]
    
    num_params = 2 * depth  # gamma and beta per layer
    
    def circuit(params):
        gamma = params[:depth]
        beta = params[depth:]
        
        # Initial superposition
        for i in range(num_qubits):
            qml.Hadamard(wires=i)
        
        # QAOA layers
        for layer in range(depth):
            # Cost layer (ZZ interactions)
            for i, j in edges:
                qml.CNOT(wires=[i, j])
                qml.RZ(2 * gamma[layer], wires=j)
                qml.CNOT(wires=[i, j])
            
            # Mixer layer (X rotations)
            for i in range(num_qubits):
                qml.RX(2 * beta[layer], wires=i)
    
    return circuit, num_params


def create_vqe_circuit(num_qubits: int, depth: int):
    """Create VQE ansatz circuit (hardware-efficient).
    
    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    depth : int
        Number of layers.
    
    Returns
    -------
    circuit_fn : callable
        Function(params) that applies VQE ansatz.
    num_params : int
        Number of parameters.
    """
    num_params = depth * num_qubits * 2  # RY and RZ per qubit per layer
    
    def circuit(params):
        idx = 0
        for layer in range(depth):
            # Rotation layer
            for q in range(num_qubits):
                qml.RY(params[idx], wires=q)
                qml.RZ(params[idx + 1], wires=q)
                idx += 2
            
            # Entangling layer (circular CNOT)
            for q in range(num_qubits):
                qml.CNOT(wires=[q, (q + 1) % num_qubits])
    
    return circuit, num_params


def create_qnn_circuit(num_qubits: int, depth: int):
    """Create QNN (Quantum Neural Network) classifier circuit.
    
    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    depth : int
        Number of variational layers.
    
    Returns
    -------
    circuit_fn : callable
        Function(weights, features) that applies QNN.
    num_params : int
        Number of weight parameters.
    """
    num_params = depth * num_qubits * 2
    
    def circuit(weights, features):
        # Feature embedding (angle encoding)
        for i in range(min(len(features), num_qubits)):
            qml.RY(features[i], wires=i)
        
        # Variational layers
        idx = 0
        for layer in range(depth):
            for q in range(num_qubits):
                qml.RY(weights[idx], wires=q)
                qml.RZ(weights[idx + 1], wires=q)
                idx += 2
            
            # Entangling
            for q in range(num_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
    
    return circuit, num_params


def create_ghz_circuit(num_qubits: int):
    """Create GHZ state preparation circuit.
    
    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    
    Returns
    -------
    circuit_fn : callable
        Function that prepares GHZ state.
    depth : int
        Circuit depth.
    """
    def circuit():
        qml.Hadamard(wires=0)
        for i in range(num_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
    
    return circuit, num_qubits


def create_grover_circuit(num_qubits: int, num_iterations: int = None):
    """Create Grover's search circuit.
    
    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    num_iterations : int, optional
        Number of Grover iterations. Defaults to optimal.
    
    Returns
    -------
    circuit_fn : callable
        Function that applies Grover's algorithm.
    depth : int
        Circuit depth.
    """
    N = 2 ** num_qubits
    if num_iterations is None:
        num_iterations = max(1, int(np.pi / 4 * np.sqrt(N)))
    
    # Oracle marks state |111...1>
    def oracle():
        # Multi-controlled Z
        if num_qubits == 1:
            qml.PauliZ(wires=0)
        elif num_qubits == 2:
            qml.CZ(wires=[0, 1])
        else:
            # Decompose multi-controlled Z
            for i in range(num_qubits):
                qml.Hadamard(wires=i)
            qml.MultiControlledX(wires=list(range(num_qubits)))
            for i in range(num_qubits):
                qml.Hadamard(wires=i)
    
    def diffuser():
        for i in range(num_qubits):
            qml.Hadamard(wires=i)
            qml.PauliX(wires=i)
        
        # Multi-controlled Z
        if num_qubits > 1:
            qml.Hadamard(wires=num_qubits - 1)
            qml.MultiControlledX(wires=list(range(num_qubits)))
            qml.Hadamard(wires=num_qubits - 1)
        else:
            qml.PauliZ(wires=0)
        
        for i in range(num_qubits):
            qml.PauliX(wires=i)
            qml.Hadamard(wires=i)
    
    def circuit():
        # Initial superposition
        for i in range(num_qubits):
            qml.Hadamard(wires=i)
        
        # Grover iterations
        for _ in range(num_iterations):
            oracle()
            diffuser()
    
    depth = 1 + num_iterations * (num_qubits * 4 + 2)
    return circuit, depth


def get_circuit_generator(circuit_type: CircuitType) -> Callable:
    """Get circuit generator function for a given type.
    
    Parameters
    ----------
    circuit_type : CircuitType
        Type of circuit to generate.
    
    Returns
    -------
    generator : callable
        Function that creates circuits of the given type.
    """
    generators = {
        CircuitType.RANDOM: create_random_circuit,
        CircuitType.RANDOM_PARAMS: create_parametric_random_circuit,
        CircuitType.QFT: create_qft_circuit,
        CircuitType.QAOA: create_qaoa_circuit,
        CircuitType.VQE: create_vqe_circuit,
        CircuitType.QNN: create_qnn_circuit,
        CircuitType.GROVER: create_grover_circuit,
        CircuitType.GHZSTATE: create_ghz_circuit,
    }
    return generators[circuit_type]


def create_benchmark_qnode(
    device: qml.Device,
    circuit_spec: CircuitSpec,
    seed: int = 42
) -> Tuple[qml.QNode, int, Optional[np.ndarray]]:
    """Create a QNode for benchmarking.
    
    Parameters
    ----------
    device : pennylane.Device
        PennyLane device to use.
    circuit_spec : CircuitSpec
        Circuit specification.
    seed : int
        Random seed.
    
    Returns
    -------
    qnode : pennylane.QNode
        Ready-to-run QNode.
    num_params : int
        Number of parameters (0 if non-parametric).
    init_params : np.ndarray or None
        Initial parameters (None if non-parametric).
    """
    rng = np.random.default_rng(seed)
    num_qubits = circuit_spec.num_qubits
    depth = circuit_spec.depth
    
    if circuit_spec.circuit_type == CircuitType.RANDOM:
        circuit_fn, num_params = create_random_circuit(num_qubits, depth, seed)
        
        @qml.qnode(device)
        def qnode():
            circuit_fn()
            return qml.expval(qml.PauliZ(0))
        
        return qnode, 0, None
    
    elif circuit_spec.circuit_type == CircuitType.RANDOM_PARAMS:
        circuit_fn, num_params = create_parametric_random_circuit(num_qubits, depth)
        init_params = rng.uniform(0, 2 * np.pi, num_params)
        
        @qml.qnode(device)
        def qnode(params):
            circuit_fn(params)
            return qml.expval(qml.PauliZ(0))
        
        return qnode, num_params, init_params
    
    elif circuit_spec.circuit_type == CircuitType.QFT:
        circuit_fn, actual_depth = create_qft_circuit(num_qubits)
        
        @qml.qnode(device)
        def qnode():
            circuit_fn()
            return qml.expval(qml.PauliZ(0))
        
        return qnode, 0, None
    
    elif circuit_spec.circuit_type == CircuitType.QAOA:
        circuit_fn, num_params = create_qaoa_circuit(num_qubits, depth)
        init_params = rng.uniform(0, np.pi, num_params)
        
        @qml.qnode(device)
        def qnode(params):
            circuit_fn(params)
            # Cost Hamiltonian expectation
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        
        return qnode, num_params, init_params
    
    elif circuit_spec.circuit_type == CircuitType.VQE:
        circuit_fn, num_params = create_vqe_circuit(num_qubits, depth)
        init_params = rng.uniform(0, 2 * np.pi, num_params)
        
        @qml.qnode(device)
        def qnode(params):
            circuit_fn(params)
            # Simple Hamiltonian: sum of ZZ
            obs = qml.PauliZ(0)
            for i in range(1, num_qubits):
                obs = obs + qml.PauliZ(i)
            return qml.expval(obs)
        
        return qnode, num_params, init_params
    
    elif circuit_spec.circuit_type == CircuitType.QNN:
        circuit_fn, num_params = create_qnn_circuit(num_qubits, depth)
        init_weights = rng.uniform(0, 2 * np.pi, num_params)
        init_features = rng.uniform(0, np.pi, num_qubits)
        
        @qml.qnode(device)
        def qnode(weights, features):
            circuit_fn(weights, features)
            return qml.expval(qml.PauliZ(0))
        
        # Return combined params
        return qnode, num_params + num_qubits, np.concatenate([init_weights, init_features])
    
    elif circuit_spec.circuit_type == CircuitType.GROVER:
        circuit_fn, actual_depth = create_grover_circuit(num_qubits, depth)
        
        @qml.qnode(device)
        def qnode():
            circuit_fn()
            return qml.probs(wires=range(num_qubits))
        
        return qnode, 0, None
    
    elif circuit_spec.circuit_type == CircuitType.GHZSTATE:
        circuit_fn, actual_depth = create_ghz_circuit(num_qubits)
        
        @qml.qnode(device)
        def qnode():
            circuit_fn()
            return qml.expval(qml.PauliZ(0))
        
        return qnode, 0, None
    
    else:
        raise ValueError(f"Unknown circuit type: {circuit_spec.circuit_type}")


# Pre-defined benchmark suites
QUICK_SUITE = [
    CircuitSpec(CircuitType.RANDOM, num_qubits=4, depth=5),
    CircuitSpec(CircuitType.RANDOM, num_qubits=6, depth=5),
    CircuitSpec(CircuitType.RANDOM, num_qubits=8, depth=5),
    CircuitSpec(CircuitType.QFT, num_qubits=4, depth=0),
    CircuitSpec(CircuitType.QFT, num_qubits=6, depth=0),
    CircuitSpec(CircuitType.QAOA, num_qubits=4, depth=2),
    CircuitSpec(CircuitType.VQE, num_qubits=4, depth=3),
]

STANDARD_SUITE = [
    # Random circuits
    CircuitSpec(CircuitType.RANDOM, num_qubits=4, depth=5),
    CircuitSpec(CircuitType.RANDOM, num_qubits=6, depth=5),
    CircuitSpec(CircuitType.RANDOM, num_qubits=8, depth=5),
    CircuitSpec(CircuitType.RANDOM, num_qubits=10, depth=5),
    CircuitSpec(CircuitType.RANDOM, num_qubits=12, depth=5),
    # QFT
    CircuitSpec(CircuitType.QFT, num_qubits=4, depth=0),
    CircuitSpec(CircuitType.QFT, num_qubits=6, depth=0),
    CircuitSpec(CircuitType.QFT, num_qubits=8, depth=0),
    CircuitSpec(CircuitType.QFT, num_qubits=10, depth=0),
    # QAOA
    CircuitSpec(CircuitType.QAOA, num_qubits=4, depth=3),
    CircuitSpec(CircuitType.QAOA, num_qubits=6, depth=3),
    CircuitSpec(CircuitType.QAOA, num_qubits=8, depth=3),
    # VQE
    CircuitSpec(CircuitType.VQE, num_qubits=4, depth=4),
    CircuitSpec(CircuitType.VQE, num_qubits=6, depth=4),
    CircuitSpec(CircuitType.VQE, num_qubits=8, depth=4),
    # QNN
    CircuitSpec(CircuitType.QNN, num_qubits=4, depth=3),
    CircuitSpec(CircuitType.QNN, num_qubits=6, depth=3),
    CircuitSpec(CircuitType.QNN, num_qubits=8, depth=3),
]

SCALABILITY_SUITE = [
    CircuitSpec(CircuitType.RANDOM, num_qubits=q, depth=5)
    for q in range(4, 20, 2)
]
