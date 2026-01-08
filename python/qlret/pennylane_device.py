"""PennyLane device for QLRET low-rank quantum simulation.

Usage:
    import pennylane as qml
    from qlret import QLRETDevice

    dev = QLRETDevice(wires=4, shots=1000, epsilon=1e-4)

    @qml.qnode(dev)
    def circuit(theta):
        qml.RX(theta, wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    result = circuit(0.5)
    grad = qml.grad(circuit)(0.5)  # parameter-shift gradient
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .api import simulate_json, QLRETError

try:
    import pennylane as qml
    from pennylane import Device
    from pennylane.tape import QuantumTape
    from pennylane.measurements import (
        ExpectationMP,
        SampleMP,
        VarianceMP,
        ProbabilityMP,
    )
    from pennylane.operation import Observable, Tensor
    # Import Prod for newer PennyLane versions (tensor product via @)
    try:
        from pennylane.ops.op_math import Prod
    except ImportError:
        Prod = None  # Not available in older PennyLane
    _HAS_PENNYLANE = True
except ImportError as exc:
    _HAS_PENNYLANE = False
    _PENNYLANE_ERROR = exc
    Device = object  # type: ignore
    QuantumTape = Any  # type: ignore


__all__ = ["QLRETDevice", "QLRETDeviceError"]


class QLRETDeviceError(RuntimeError):
    """Error from QLRET PennyLane device."""


def _require_pennylane() -> None:
    if not _HAS_PENNYLANE:
        raise ImportError(
            "PennyLane is required for QLRETDevice. Install with: pip install pennylane"
        ) from _PENNYLANE_ERROR


# ---------------------------------------------------------------------------
# Operation and Observable Mapping
# ---------------------------------------------------------------------------

# PennyLane operation name -> QLRET JSON name
OP_MAP: Dict[str, str] = {
    "Hadamard": "H",
    "PauliX": "X",
    "PauliY": "Y",
    "PauliZ": "Z",
    "S": "S",
    "T": "T",
    "Adjoint(S)": "SDG",
    "Adjoint(T)": "TDG",
    "SX": "SX",
    "RX": "RX",
    "RY": "RY",
    "RZ": "RZ",
    "PhaseShift": "U1",
    "U1": "U1",
    "U2": "U2",
    "U3": "U3",
    "Rot": "U3",  # Rot(phi, theta, omega) -> U3
    "CNOT": "CNOT",
    "CZ": "CZ",
    "CY": "CY",
    "SWAP": "SWAP",
    "ISWAP": "ISWAP",
}

# PennyLane observable name -> QLRET Pauli symbol
OBS_MAP: Dict[str, str] = {
    "PauliX": "X",
    "PauliY": "Y",
    "PauliZ": "Z",
    "Identity": "I",
}


def _op_to_json(op: Any) -> Dict[str, Any]:
    """Convert a PennyLane operation to JSON dict."""
    name = op.name
    
    # Handle adjoint operations
    if name.startswith("Adjoint("):
        inner = name[8:-1]
        json_name = OP_MAP.get(f"Adjoint({inner})")
        if json_name is None:
            raise QLRETDeviceError(f"Unsupported adjoint operation: {name}")
    else:
        json_name = OP_MAP.get(name)
    
    if json_name is None:
        raise QLRETDeviceError(f"Unsupported operation: {name}")
    
    wires = [int(w) for w in op.wires]
    result: Dict[str, Any] = {"name": json_name, "wires": wires}
    
    # Add parameters if present
    if op.num_params > 0:
        params = [float(p) for p in op.parameters]
        result["params"] = params
    
    return result


def _obs_to_json(obs: Any, coeff: float = 1.0) -> Dict[str, Any]:
    """Convert a PennyLane observable to JSON dict."""
    # Handle Tensor products (e.g., Z @ Z) - both old Tensor and new Prod types
    is_tensor = isinstance(obs, Tensor)
    is_prod = Prod is not None and isinstance(obs, Prod)
    
    if is_tensor or is_prod:
        operators: List[str] = []
        wires: List[int] = []
        # Get operands - use obs.obs for Tensor, obs.operands for Prod
        operands = obs.obs if is_tensor else obs.operands
        for o in operands:
            pauli = OBS_MAP.get(o.name)
            if pauli is None:
                raise QLRETDeviceError(f"Unsupported observable in tensor: {o.name}")
            operators.append(pauli)
            wires.extend([int(w) for w in o.wires])
        return {
            "type": "TENSOR",
            "operators": operators,
            "wires": wires,
            "coefficient": coeff,
        }
    
    # Handle Hamiltonian (Sum type, has multiple terms with different wires)
    # Check for Hamiltonian by looking for 'terms' method (newer) or checking if it's a Sum
    if hasattr(obs, "terms") and callable(obs.terms):
        try:
            coeffs, ops = obs.terms()
            # If there's more than one term with different structure, it's a Hamiltonian
            if len(coeffs) > 1:
                raise QLRETDeviceError(
                    "Hamiltonian observables not yet supported. Use individual terms."
                )
        except Exception:
            pass  # Not a Hamiltonian, continue to single observable handling
    
    # Single Pauli observable
    pauli = OBS_MAP.get(obs.name)
    if pauli is None:
        # Check if it's a Hermitian observable
        if obs.name == "Hermitian":
            matrix = obs.matrix()
            return {
                "type": "HERMITIAN",
                "wires": [int(w) for w in obs.wires],
                "coefficient": coeff,
                "matrix_real": matrix.real.tolist(),
                "matrix_imag": matrix.imag.tolist(),
            }
        raise QLRETDeviceError(f"Unsupported observable: {obs.name}")
    
    return {
        "type": "PAULI",
        "operator": pauli,
        "wires": [int(w) for w in obs.wires],
        "coefficient": coeff,
    }


# ---------------------------------------------------------------------------
# QLRET Device
# ---------------------------------------------------------------------------


class QLRETDevice(Device):
    """PennyLane device using QLRET low-rank density matrix simulation.

    Parameters
    ----------
    wires : int or Iterable
        Number of wires or wire labels.
    shots : int or None
        Number of measurement shots. None for analytic expectation values.
    epsilon : float
        Truncation threshold for low-rank compression (default: 1e-4).
    """

    name = "QLRET Simulator"
    short_name = "qlret"
    pennylane_requires = ">=0.30"
    version = "1.0.0"
    author = "QLRET Team"

    # Supported operations
    operations = set(OP_MAP.keys())
    observables = {"PauliX", "PauliY", "PauliZ", "Identity", "Hermitian"}

    def __init__(
        self,
        wires: Union[int, Sequence[int]],
        shots: Optional[int] = None,
        epsilon: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        _require_pennylane()
        super().__init__(wires=wires, shots=shots)
        self.epsilon = epsilon
        self._kwargs = kwargs

    # num_wires is set by parent Device.__init__, no need to override

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        circuits: Union[QuantumTape, List[QuantumTape], List[Any]],
        execution_config: Any = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Execute quantum circuits and return results.
        
        Supports both modern API (QuantumTape) and legacy API (operations, measurements).
        """
        # Handle legacy API call: execute(operations, measurements)
        if not isinstance(circuits, (list, QuantumTape)) or (
            isinstance(circuits, list) and circuits and not isinstance(circuits[0], QuantumTape)
        ):
            # Legacy: circuits is actually operations, second arg is measurements
            operations = circuits
            measurements = execution_config  # Actually measurements in legacy mode
            
            # Build a tape from operations and measurements
            with qml.tape.QuantumTape() as tape:
                for op in operations:
                    qml.apply(op)
                if measurements:
                    for m in measurements:
                        qml.apply(m)
            
            return self._execute_tape(tape)
        
        # Modern API: circuits is QuantumTape or list of tapes
        is_single = isinstance(circuits, QuantumTape)
        if is_single:
            circuits = [circuits]

        results = []
        for tape in circuits:
            result = self._execute_tape(tape)
            results.append(result)

        return results[0] if is_single else results

    # ------------------------------------------------------------------
    # Legacy API Support (for PennyLane < 0.30 compatibility)
    # ------------------------------------------------------------------

    def apply(self, operations, **kwargs):
        """Apply operations (legacy API - not used in modern PennyLane)."""
        raise NotImplementedError(
            "Legacy apply() method not supported. Use @qml.qnode(device) decorator pattern."
        )

    def expval(self, observable, **kwargs):
        """Return expectation value (legacy API - not used in modern PennyLane)."""
        raise NotImplementedError(
            "Legacy expval() method not supported. Use @qml.qnode(device) with qml.expval()."
        )

    def reset(self):
        """Reset device (legacy API - not needed in modern PennyLane)."""
        pass  # No-op for stateless device

    # ------------------------------------------------------------------
    # Execution (Modern API)
    # ------------------------------------------------------------------

    def _execute_tape(self, tape: QuantumTape) -> np.ndarray:
        """Execute a single quantum tape."""
        # Build JSON circuit
        circuit_json = self._tape_to_json(tape)
        
        # Run simulation
        try:
            result = simulate_json(circuit_json, export_state=False)
        except QLRETError as e:
            raise QLRETDeviceError(f"Simulation failed: {e}") from e

        # Extract results based on measurement types
        return self._process_results(tape, result)

    def _tape_to_json(self, tape: QuantumTape) -> Dict[str, Any]:
        """Convert a PennyLane tape to QLRET JSON format."""
        # Operations
        operations = []
        for op in tape.operations:
            operations.append(_op_to_json(op))

        # Observables from measurements
        observables = []
        for m in tape.measurements:
            if isinstance(m, (ExpectationMP, VarianceMP, SampleMP)):
                obs = m.obs
                if obs is not None:
                    observables.append(_obs_to_json(obs))
            elif isinstance(m, ProbabilityMP):
                # Probability doesn't need an observable
                pass

        # Build config
        config: Dict[str, Any] = {
            "epsilon": self.epsilon,
            "initial_rank": 1,
            "export_state": False,
        }
        
        if self.shots is not None:
            config["shots"] = self.shots

        return {
            "circuit": {
                "num_qubits": self.num_wires,
                "operations": operations,
                "observables": observables,
            },
            "config": config,
        }

    def _process_results(
        self, tape: QuantumTape, result: Dict[str, Any]
    ) -> np.ndarray:
        """Process QLRET results into PennyLane format."""
        expectations = result.get("expectation_values", [])
        samples = result.get("samples")

        outputs = []
        obs_idx = 0

        for m in tape.measurements:
            if isinstance(m, ExpectationMP):
                if obs_idx < len(expectations):
                    outputs.append(expectations[obs_idx])
                    obs_idx += 1
                else:
                    outputs.append(0.0)

            elif isinstance(m, VarianceMP):
                # Variance requires <O^2> - <O>^2
                # For now, return 0 (proper implementation needs second observable)
                outputs.append(0.0)
                obs_idx += 1

            elif isinstance(m, SampleMP):
                if samples is not None:
                    # Convert integer samples to bit arrays
                    n_qubits = self.num_wires
                    sample_array = np.array(samples, dtype=np.int64)
                    # Unpack to bits if needed
                    outputs.append(sample_array)
                else:
                    outputs.append(np.array([]))
                if m.obs is not None:
                    obs_idx += 1

            elif isinstance(m, ProbabilityMP):
                # Compute probabilities from samples or state
                if samples is not None:
                    counts = np.bincount(samples, minlength=2**self.num_wires)
                    probs = counts / len(samples)
                    outputs.append(probs)
                else:
                    outputs.append(np.zeros(2**self.num_wires))

        if len(outputs) == 1:
            return np.asarray(outputs[0])
        return tuple(np.asarray(o) for o in outputs)

    # ------------------------------------------------------------------
    # Gradient Support (Parameter-Shift)
    # ------------------------------------------------------------------

    @classmethod
    def capabilities(cls) -> Dict[str, Any]:
        """Return device capabilities."""
        return {
            "model": "qubit",
            "supports_broadcasting": False,
            "supports_finite_shots": True,
            "supports_tensor_observables": True,
            "returns_probs": True,
            "returns_state": False,
            "supports_reversible_diff": False,
            "supports_inverse_operations": True,
            "supports_analytic_computation": True,
        }

    def supports_derivatives(
        self,
        execution_config: Any = None,
        circuit: Optional[QuantumTape] = None,
    ) -> bool:
        """QLRET supports parameter-shift differentiation."""
        return True

    def compute_derivatives(
        self,
        circuits: Union[QuantumTape, List[QuantumTape]],
        execution_config: Any = None,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Compute gradients using parameter-shift rule.

        For a parameter θ in gate G(θ), the gradient is:
            ∂<O>/∂θ = (1/2) * [<O>|θ+π/2 - <O>|θ-π/2]
        """
        is_single = isinstance(circuits, QuantumTape)
        if is_single:
            circuits = [circuits]

        all_grads = []
        for tape in circuits:
            grads = self._compute_tape_gradients(tape)
            all_grads.append(grads)

        return all_grads[0] if is_single else all_grads

    def _compute_tape_gradients(self, tape: QuantumTape) -> np.ndarray:
        """Compute parameter-shift gradients for a single tape."""
        trainable_params = tape.trainable_params
        n_params = len(trainable_params)
        
        if n_params == 0:
            return np.array([])

        # Get base circuit operations
        ops = list(tape.operations)
        measurements = tape.measurements
        
        # Number of expectation values
        n_outputs = sum(
            1 for m in measurements if isinstance(m, (ExpectationMP, VarianceMP))
        )
        
        gradients = np.zeros((n_outputs, n_params))
        shift = np.pi / 2

        for param_idx, trainable_idx in enumerate(trainable_params):
            # Find which operation and which parameter
            op_idx, local_param_idx = self._find_param_location(tape, trainable_idx)
            
            # Shift up
            shifted_up = self._shift_param(ops, op_idx, local_param_idx, +shift)
            tape_up = QuantumTape(shifted_up, measurements)
            result_up = self._execute_tape(tape_up)
            
            # Shift down
            shifted_down = self._shift_param(ops, op_idx, local_param_idx, -shift)
            tape_down = QuantumTape(shifted_down, measurements)
            result_down = self._execute_tape(tape_down)
            
            # Parameter-shift formula
            if isinstance(result_up, (int, float)):
                result_up = np.array([result_up])
            if isinstance(result_down, (int, float)):
                result_down = np.array([result_down])
            
            grad = 0.5 * (np.asarray(result_up) - np.asarray(result_down))
            gradients[:, param_idx] = grad.flatten()[:n_outputs]

        return gradients

    def _find_param_location(
        self, tape: QuantumTape, trainable_idx: int
    ) -> Tuple[int, int]:
        """Find operation index and local parameter index for a trainable param."""
        param_count = 0
        for op_idx, op in enumerate(tape.operations):
            n_params = op.num_params
            if param_count + n_params > trainable_idx:
                local_idx = trainable_idx - param_count
                return op_idx, local_idx
            param_count += n_params
        raise ValueError(f"Trainable parameter index {trainable_idx} not found")

    def _shift_param(
        self,
        ops: List[Any],
        op_idx: int,
        param_idx: int,
        shift: float,
    ) -> List[Any]:
        """Create a copy of operations with one parameter shifted."""
        new_ops = []
        for i, op in enumerate(ops):
            if i == op_idx:
                # Shift this parameter
                new_params = list(op.parameters)
                new_params[param_idx] = float(new_params[param_idx]) + shift
                new_op = type(op)(*new_params, wires=op.wires)
                new_ops.append(new_op)
            else:
                new_ops.append(op)
        return new_ops


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_device() -> None:
    """Register QLRETDevice with PennyLane."""
    _require_pennylane()
    qml.plugin.register(QLRETDevice)


# Try to register on import
if _HAS_PENNYLANE:
    try:
        register_device()
    except Exception:
        pass  # Registration may fail in some contexts
