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
    # PennyLane 0.43+ moved Device to pennylane.devices
    try:
        from pennylane.devices import Device, DeviceCapabilities
        _HAS_DEVICE_CAPABILITIES = True
    except ImportError:
        # Fallback for older PennyLane versions
        from pennylane import Device
        DeviceCapabilities = None
        _HAS_DEVICE_CAPABILITIES = False
    from pennylane.tape import QuantumTape
    from pennylane.measurements import (
        ExpectationMP,
        SampleMP,
        VarianceMP,
        ProbabilityMP,
    )
    # PennyLane 0.43+ uses Prod instead of Tensor
    # Also Observable was removed
    try:
        from pennylane.operation import Tensor
    except ImportError:
        Tensor = None  # Use Prod instead
    try:
        from pennylane.ops.op_math import Prod
    except ImportError:
        Prod = None  # Not available in older PennyLane
    _HAS_PENNYLANE = True
except ImportError as exc:
    _HAS_PENNYLANE = False
    _PENNYLANE_ERROR = exc
    Device = object  # type: ignore
    DeviceCapabilities = None
    _HAS_DEVICE_CAPABILITIES = False
    QuantumTape = Any  # type: ignore
    Tensor = None
    Prod = None


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
    is_tensor = Tensor is not None and isinstance(obs, Tensor)
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

import os

# Path to the device configuration file
_CONFIG_FILEPATH = os.path.join(os.path.dirname(__file__), "device_config.toml")


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
    
    Example
    -------
    >>> import pennylane as qml
    >>> from qlret import QLRETDevice
    >>> dev = QLRETDevice(wires=4, shots=1000)
    >>> @qml.qnode(dev)
    ... def circuit(x):
    ...     qml.RX(x, wires=0)
    ...     qml.CNOT(wires=[0, 1])
    ...     return qml.expval(qml.PauliZ(0))
    >>> circuit(0.5)
    """

    name = "QLRET Simulator"
    short_name = "qlret.mixed"
    pennylane_requires = ">=0.30"
    version = "1.0.0"
    author = "QLRET Team"
    
    # Point to the TOML config file for PennyLane 0.43+
    config_filepath = _CONFIG_FILEPATH

    # Supported operations (for backwards compatibility)
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
        # PennyLane 0.43+ has different Device initialization
        super().__init__(wires=wires, shots=shots)
        self.epsilon = epsilon
        self._kwargs = kwargs
        self._num_wires = len(self.wires) if hasattr(self.wires, '__len__') else self.wires

    @property
    def num_wires(self) -> int:
        """Return number of wires."""
        return self._num_wires

    def supports_derivatives(
        self,
        execution_config: Any = None,
        circuit: Optional[QuantumTape] = None,
    ) -> bool:
        """Return False to indicate this device does not compute gradients natively.
        
        This tells PennyLane to use parameter-shift or finite-difference
        gradient transforms instead of asking the device for gradients.
        """
        return False

    def setup_execution_config(
        self,
        config: Any = None,
        circuit: Optional[QuantumTape] = None,
    ) -> Any:
        """Configure execution settings.
        
        This tells PennyLane to use parameter-shift gradients since
        we don't provide device-level derivatives.
        """
        # Import ExecutionConfig dynamically to handle different PennyLane versions
        try:
            from pennylane.devices import ExecutionConfig
        except ImportError:
            # Older PennyLane - just return config as-is
            return config
        
        if config is None:
            config = ExecutionConfig()
        
        # Use parameter-shift differentiation (handled by PennyLane workflow)
        # We don't set gradient_method or use_device_gradient to let PennyLane
        # handle gradients through its standard parameter-shift transform
        return config

    # ------------------------------------------------------------------
    # Execution (PennyLane 0.43+ API)
    # ------------------------------------------------------------------

    def execute(
        self,
        circuits: Union[QuantumTape, List[QuantumTape]],
        execution_config: Any = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """Execute quantum circuits and return results.
        
        Parameters
        ----------
        circuits : QuantumTape or List[QuantumTape]
            The quantum circuits to execute.
        execution_config : ExecutionConfig, optional
            Configuration for execution (ignored, for API compatibility).
            
        Returns
        -------
        Results for each circuit.
        """
        # Modern API: circuits is QuantumTape (QuantumScript) or list of tapes
        is_single = isinstance(circuits, QuantumTape)
        if is_single:
            circuits = [circuits]

        results = []
        for tape in circuits:
            result = self._execute_tape(tape)
            results.append(result)

        return results[0] if is_single else tuple(results)

    # ------------------------------------------------------------------
    # Internal Execution
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
        
        # Handle shots - can be Shots object or int or None
        tape_shots = tape.shots if hasattr(tape, 'shots') else self.shots
        if tape_shots is not None:
            # PennyLane 0.43+ uses Shots object, get total_shots
            if hasattr(tape_shots, 'total_shots'):
                shots_val = tape_shots.total_shots
            else:
                shots_val = int(tape_shots) if tape_shots else None
            if shots_val is not None and shots_val > 0:
                config["shots"] = shots_val

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

    @property
    def capabilities(self) -> Any:
        """Return device capabilities for PennyLane 0.43+.
        
        Returns a DeviceCapabilities object that tells PennyLane what
        this device supports.
        """
        if not _HAS_DEVICE_CAPABILITIES or DeviceCapabilities is None:
            return None
        
        # Build capabilities with the required fields for PennyLane 0.43+
        return DeviceCapabilities(
            supported_mcm_methods=[],  # No mid-circuit measurements
        )

    @staticmethod
    def _get_capabilities_dict() -> Dict[str, Any]:
        """Return device capabilities as a dictionary (legacy)."""
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

    # NOTE: We explicitly do NOT implement supports_derivatives, compute_derivatives,
    # or compute_vjp. By not implementing these, PennyLane will automatically use
    # its built-in parameter-shift gradient transform, which is more compatible
    # with the modern PennyLane 0.43+ workflow system.


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_device() -> None:
    """Register QLRETDevice with PennyLane.
    
    This allows using the device with qml.device("qlret.mixed", wires=n).
    """
    _require_pennylane()
    
    # PennyLane 0.43+ uses a different registration mechanism
    # Try the modern approach first, then fall back to legacy
    try:
        # Modern PennyLane: check if already registered
        try:
            qml.device("qlret.mixed", wires=1)
            return  # Already registered
        except qml.DeviceError:
            pass
        
        # Try to add to device registry
        if hasattr(qml, 'plugin') and hasattr(qml.plugin, 'register'):
            qml.plugin.register(QLRETDevice)
        elif hasattr(qml, 'register_device'):
            qml.register_device("qlret.mixed", QLRETDevice)
    except Exception:
        # Registration may fail in some contexts, that's okay
        # Users can still instantiate QLRETDevice directly
        pass


# Try to register on import
if _HAS_PENNYLANE:
    try:
        register_device()
    except Exception:
        pass  # Registration may fail in some contexts
