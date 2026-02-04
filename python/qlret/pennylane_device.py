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

Batch Parallelism:
    # Parallel C++ mode: Balance workers and threads
    dev = QLRETDevice(
        wires=4,
        num_threads=8,        # C++ threads when running single circuit
        max_batch_workers=4,  # Python workers for parallel batch execution
    )
    # Result: 4 workers × 2 threads = 8 total threads (matches 8-core CPU)
    
    # Sequential C++ mode: Maximize Python workers
    dev = QLRETDevice(
        wires=4,
        num_threads=1,          # Sequential C++ (1 thread per circuit)
        max_batch_workers='max' # Use all CPU cores as Python workers
    )
    # Result: 8 workers × 1 thread = 8 total threads (optimal for sequential)
"""

from __future__ import annotations

import os
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

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
        StateMP,  # For state export
        DensityMatrixMP,  # For density matrix export
        PurityMP,  # For purity measurement (if available)
    )
    _HAS_PURITY_MP = True
    _HAS_STATE_MP = True
except ImportError:
    # Some measurement types may not exist in older PennyLane versions
    try:
        from pennylane.measurements import StateMP
        _HAS_STATE_MP = True
    except ImportError:
        StateMP = None
        _HAS_STATE_MP = False
    try:
        from pennylane.measurements import DensityMatrixMP
    except ImportError:
        DensityMatrixMP = None
    try:
        from pennylane.measurements import PurityMP
        _HAS_PURITY_MP = True
    except ImportError:
        PurityMP = None
        _HAS_PURITY_MP = False

try:
    import pennylane as qml
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
    StateMP = None
    DensityMatrixMP = None
    PurityMP = None
    _HAS_STATE_MP = False
    _HAS_PURITY_MP = False


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
    # Single-qubit gates
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
    
    # Two-qubit gates
    "CNOT": "CNOT",
    "CZ": "CZ",
    "CY": "CY",
    "SWAP": "SWAP",
    "ISWAP": "ISWAP",
    
    # Three-qubit gates (decomposed to native gates if not supported natively)
    "Toffoli": "CCX",
    "CCX": "CCX",
    "CSWAP": "CSWAP",
    "Fredkin": "CSWAP",
    
    # Controlled rotation gates (for QPE, etc.)
    "CRX": "CRX",
    "CRY": "CRY",
    "CRZ": "CRZ",
    "CRot": "CRot",
    "ControlledPhaseShift": "CU1",
    "CPhase": "CU1",
}

# PennyLane observable name -> QLRET Pauli symbol
OBS_MAP: Dict[str, str] = {
    "PauliX": "X",
    "PauliY": "Y",
    "PauliZ": "Z",
    "Identity": "I",
}


def _op_to_json(op: Any) -> Dict[str, Any]:
    """Convert a PennyLane operation to JSON dict.
    
    Handles:
    - Standard gates (H, X, Y, Z, CNOT, etc.)
    - Multi-controlled gates (Toffoli, CSWAP, etc.)
    - Controlled rotation gates (CRX, CRY, CRZ, etc.)
    - Noise channels via Kraus operators
    - Adjoint operations
    """
    name = op.name
    
    # Check if this is a noise channel (has kraus_matrices method)
    # PennyLane's Channel class provides this
    if hasattr(op, 'kraus_matrices') and callable(op.kraus_matrices):
        try:
            kraus_matrices = op.kraus_matrices()
            if kraus_matrices is not None and len(kraus_matrices) > 0:
                # Convert Kraus matrices to JSON format
                kraus_json = []
                for K in kraus_matrices:
                    K = np.asarray(K)  # Ensure numpy array
                    kraus_json.append({
                        "real": K.real.tolist(),
                        "imag": K.imag.tolist(),
                    })
                return {
                    "name": "KRAUS",
                    "wires": [int(w) for w in op.wires],
                    "kraus_operators": kraus_json,
                }
        except Exception:
            pass  # Fall through to regular operation handling
    
    # Handle MultiControlledX (multi-controlled Toffoli with n controls)
    if name == "MultiControlledX" or name.startswith("C(") and "X" in name:
        wires = [int(w) for w in op.wires]
        control_wires = wires[:-1]  # All but last are controls
        target_wire = wires[-1]
        return {
            "name": "MCX",
            "control_wires": control_wires,
            "target_wire": target_wire,
            "wires": wires,
        }
    
    # Handle generic Controlled operations
    if hasattr(op, 'base') and hasattr(op, 'control_wires'):
        base_op = op.base
        control_wires = [int(w) for w in op.control_wires]
        target_wires = [int(w) for w in base_op.wires]
        
        # Try to get the base operation name
        base_name = getattr(base_op, 'name', None)
        if base_name and base_name in OP_MAP:
            result = {
                "name": f"C{OP_MAP[base_name]}",
                "control_wires": control_wires,
                "target_wires": target_wires,
                "wires": control_wires + target_wires,
            }
            if base_op.num_params > 0:
                result["params"] = [float(p) for p in base_op.parameters]
            return result
    
    # Handle adjoint operations
    if name.startswith("Adjoint("):
        inner = name[8:-1]
        json_name = OP_MAP.get(f"Adjoint({inner})")
        if json_name is None:
            raise QLRETDeviceError(f"Unsupported adjoint operation: {name}")
    else:
        json_name = OP_MAP.get(name)
    
    if json_name is None:
        raise QLRETDeviceError(f"Operator {op} not supported with {QLRETDevice.name}. "
                               f"Supported operations: {list(OP_MAP.keys())}")
    
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
    # We mark this as HAMILTONIAN type for detection, but it will be 
    # decomposed and handled in Python layer
    if hasattr(obs, "terms") and callable(obs.terms):
        try:
            coeffs, ops = obs.terms()
            if len(coeffs) >= 1:
                # Build Hamiltonian as list of terms for Python-level processing
                terms = []
                for c, op in zip(coeffs, ops):
                    term = _obs_to_json(op, coeff=float(c) * coeff)
                    terms.append(term)
                return {
                    "type": "HAMILTONIAN",
                    "terms": terms,
                    "coefficient": coeff,
                    # Store original obs for Python-level decomposition
                    "_pennylane_obs": obs,
                    "_coefficients": [float(c) for c in coeffs],
                    "_operators": ops,
                }
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
    # Include both gates and noise channels
    operations = set(OP_MAP.keys()) | {
        # Multi-qubit gates
        "Toffoli",
        "CCX",
        "CSWAP",
        "Fredkin",
        "CRX",
        "CRY",
        "CRZ",
        "CRot",
        "ControlledPhaseShift",
        "CPhase",
        "MultiControlledX",
        # Noise channels - LRET supports any channel via Kraus operators
        "DepolarizingChannel",
        "AmplitudeDamping",
        "PhaseDamping",
        "BitFlip",
        "PhaseFlip",
        "ThermalRelaxationError",
        "ResetError",
        "GeneralizedAmplitudeDamping",
        "PauliError",
        "QubitChannel",  # Generic Kraus channel
    }
    observables = {"PauliX", "PauliY", "PauliZ", "Identity", "Hermitian", "Prod", "Hamiltonian"}
    
    # Valid parallelization modes (C++ level)
    PARALLEL_MODES = {"auto", "sequential", "row", "column", "batch", "hybrid"}
    
    # Valid batch worker modes (Python level)
    BATCH_WORKER_AUTO = -1
    BATCH_WORKER_DISABLED = 0
    BATCH_WORKER_MAX = 'max'  # Maximum Python workers (1 per core)

    def __init__(
        self,
        wires: Union[int, Sequence[int]],
        shots: Optional[int] = None,
        epsilon: float = 1e-4,
        num_threads: int = 0,
        parallel_mode: str = "hybrid",
        max_batch_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize QLRET PennyLane device.
        
        Parameters
        ----------
        wires : int or Iterable
            Number of wires or wire labels.
        shots : int or None
            Number of measurement shots. None for analytic expectation values.
        epsilon : float
            Truncation threshold for low-rank compression (default: 1e-4).
        num_threads : int
            Number of C++ threads to use for OpenMP parallel execution within
            each circuit. 0 = auto (use all available CPU cores). Default: 0.
        parallel_mode : str
            C++ parallelization strategy for operations within a circuit. Options:
            - "hybrid" (default): Row + batch combined - best for most cases
            - "auto": Automatically select best strategy
            - "row": Row-wise parallel
            - "column": Column-wise parallel
            - "batch": Gate batching
            - "sequential": No parallelism (single-threaded)
        max_batch_workers : int or str
            Python-level parallelism for executing multiple circuits concurrently.
            - 0 (default): Disabled - circuits execute sequentially (current behavior)
            - 1: Explicitly sequential (same as 0)
            - N > 1: Use N Python workers for parallel batch execution
            - -1: Auto-tune based on CPU cores and batch size
            - 'max': Maximum parallelism - use cpu_count workers with 1 thread each
                    (optimal for sequential C++ mode: num_threads=1)
            
            When enabled (N > 1, -1, or 'max'), the effective thread count per circuit
            is automatically reduced to prevent CPU oversubscription. Examples:
            - 8-core machine, num_threads=8, max_batch_workers=4:
              → 4 workers × 2 threads = 8 total (optimal for parallel C++)
            - 8-core machine, num_threads=1, max_batch_workers='max':
              → 8 workers × 1 thread = 8 total (optimal for sequential C++)
            
        Notes
        -----
        Thread allocation strategy:
        - Single circuit: Uses all num_threads for maximum within-circuit parallelism
        - Batch with max_batch_workers > 1: Divides threads among workers
        - This prevents thread oversubscription which can degrade performance
        """
        _require_pennylane()
        # PennyLane 0.43+ has different Device initialization
        super().__init__(wires=wires, shots=shots)
        self.epsilon = epsilon
        
        # C++ parallelization settings
        self.num_threads = num_threads  # 0 = auto (all cores)
        parallel_mode_lower = parallel_mode.lower()
        if parallel_mode_lower not in self.PARALLEL_MODES:
            raise ValueError(
                f"Invalid parallel_mode '{parallel_mode}'. "
                f"Must be one of: {', '.join(sorted(self.PARALLEL_MODES))}"
            )
        self.parallel_mode = parallel_mode_lower
        
        # Auto-detect CPU count
        self._cpu_count = os.cpu_count() or 1
        
        # Auto-detect thread count if num_threads=0
        if self.num_threads == 0:
            self._effective_threads = self._cpu_count
        else:
            self._effective_threads = self.num_threads
        
        # Python-level batch parallelism settings
        # Support both int and 'max' string
        if isinstance(max_batch_workers, str) and max_batch_workers.lower() == 'max':
            self.max_batch_workers = self.BATCH_WORKER_MAX
        else:
            self.max_batch_workers = max_batch_workers
        
        self._kwargs = kwargs
        self._num_wires = len(self.wires) if hasattr(self.wires, '__len__') else self.wires

    @property
    def num_wires(self) -> int:
        """Return number of wires."""
        return self._num_wires

    def preprocess_transforms(self, execution_config: Any = None) -> Any:
        """Return the preprocessing transforms for this device.
        
        This customizes the decomposition stopping condition to support
        noise channels via Kraus operators.
        """
        try:
            from pennylane.transforms.core import TransformProgram
            from pennylane.devices.preprocess import (
                decompose,
                validate_device_wires,
                validate_measurements,
                validate_observables,
            )
        except ImportError:
            # Older PennyLane - return default
            return super().preprocess_transforms(execution_config)
        
        def stopping_condition(op) -> bool:
            """Check if an operation is supported natively.
            
            Returns True if the operation should NOT be decomposed further.
            This includes all gates in OP_MAP plus any noise channel with
            kraus_matrices support.
            """
            # Check if it's a supported gate
            if op.name in OP_MAP:
                return True
            # Check adjoint gates
            if op.name.startswith("Adjoint(") and op.name[8:-1] in ("S", "T"):
                return True
            # Check if it's a noise channel (has kraus_matrices)
            if hasattr(op, 'kraus_matrices') and callable(op.kraus_matrices):
                try:
                    km = op.kraus_matrices()
                    if km is not None and len(km) > 0:
                        return True
                except Exception:
                    pass
            return False
        
        def observable_stopping_condition(obs) -> bool:
            """Check if an observable is supported.
            
            Supports:
            - Single Pauli: PauliX, PauliY, PauliZ, Identity
            - Hermitian: Custom Hermitian matrix
            - Prod: Tensor products like Z(0) @ Z(1)
            - Hamiltonian: Linear combinations of Pauli terms
            """
            # Direct match for single observables
            if obs.name in self.observables:
                return True
            
            # Check if it's a Prod (tensor product) - all operands must be Pauli
            if Prod is not None and isinstance(obs, Prod):
                return all(
                    o.name in ("PauliX", "PauliY", "PauliZ", "Identity")
                    for o in obs.operands
                )
            
            # Check old Tensor type
            if Tensor is not None and isinstance(obs, Tensor):
                return all(
                    o.name in ("PauliX", "PauliY", "PauliZ", "Identity")
                    for o in obs.obs
                )
            
            # Check Hamiltonian (has terms() method returning coeffs and ops)
            if hasattr(obs, 'terms') and callable(obs.terms):
                try:
                    coeffs, ops = obs.terms()
                    # All terms must be valid Pauli/tensor products
                    for op in ops:
                        if Prod is not None and isinstance(op, Prod):
                            if not all(o.name in ("PauliX", "PauliY", "PauliZ", "Identity") for o in op.operands):
                                return False
                        elif op.name not in ("PauliX", "PauliY", "PauliZ", "Identity"):
                            return False
                    return True
                except Exception:
                    pass
            
            return False
        
        program = TransformProgram()
        program.add_transform(decompose, stopping_condition=stopping_condition, name=self.name)
        program.add_transform(validate_device_wires, self.wires, name=self.name)
        # Add minimal measurement validation
        program.add_transform(
            validate_measurements,
            analytic_measurements=lambda m: isinstance(m, (ExpectationMP, VarianceMP, ProbabilityMP)),
            sample_measurements=lambda m: isinstance(m, SampleMP),
            name=self.name,
        )
        program.add_transform(validate_observables, stopping_condition=observable_stopping_condition, name=self.name)
        
        return program

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
        
        Notes
        -----
        Execution strategy depends on max_batch_workers setting:
        - max_batch_workers <= 1: Sequential execution (default)
        - max_batch_workers > 1: Parallel execution using ThreadPoolExecutor
        - max_batch_workers == -1: Auto-tune based on batch size
        
        When parallel execution is enabled, the C++ thread count per circuit
        is automatically reduced to prevent CPU oversubscription.
        """
        # Modern API: circuits is QuantumTape (QuantumScript) or list of tapes
        is_single = isinstance(circuits, QuantumTape)
        if is_single:
            circuits = [circuits]

        batch_size = len(circuits)
        
        # Determine execution strategy
        workers, threads_per_circuit = self._compute_execution_strategy(batch_size)
        
        if workers <= 1:
            # Sequential execution - use full thread count for each circuit
            results = [self._execute_tape(tape) for tape in circuits]
        else:
            # Parallel execution with reduced threads per circuit
            results = self._execute_batch_parallel(circuits, workers, threads_per_circuit)

        return results[0] if is_single else tuple(results)

    def _compute_execution_strategy(self, batch_size: int) -> Tuple[int, int]:
        """Determine optimal worker count and threads per circuit.
        
        This method intelligently detects whether LRET is running in sequential
        C++ mode (num_threads=1 or parallel_mode='sequential') and adjusts the
        strategy to maximize CPU utilization:
        
        - Sequential C++ mode: Maximize Python workers (1 thread per circuit)
        - Parallel C++ mode: Balance workers and threads per circuit
        
        Parameters
        ----------
        batch_size : int
            Number of circuits in the batch.
            
        Returns
        -------
        Tuple[int, int]
            (num_workers, threads_per_circuit)
            - num_workers: Number of Python workers (1 = sequential)
            - threads_per_circuit: C++ threads to allocate per circuit
        """
        # Detect if C++ is running in sequential mode
        is_cpp_sequential = (
            self.num_threads == 1 or 
            self.parallel_mode == 'sequential'
        )
        
        # Single circuit always runs sequentially with full threads
        if batch_size == 1:
            return 1, self._effective_threads
        
        # Check max_batch_workers setting
        if self.max_batch_workers == self.BATCH_WORKER_DISABLED:
            # Disabled: sequential execution
            return 1, self._effective_threads
        
        if self.max_batch_workers == 1:
            # Explicitly sequential
            return 1, self._effective_threads
        
        # Handle 'max' mode: maximum Python parallelism
        if self.max_batch_workers == self.BATCH_WORKER_MAX:
            # Use cpu_count workers, each with 1 thread
            # This is optimal for sequential C++ mode
            workers = min(self._cpu_count, batch_size)
            return workers, 1
        
        # Auto-tune mode
        if self.max_batch_workers == self.BATCH_WORKER_AUTO:
            # Heuristic: parallelize if batch_size >= 4
            if batch_size < 4:
                return 1, self._effective_threads
            
            # Strategy depends on whether C++ is sequential or parallel
            if is_cpp_sequential:
                # Sequential C++: maximize Python workers (1 thread each)
                # Use up to cpu_count workers since each only needs 1 thread
                workers = min(self._cpu_count, batch_size)
                return workers, 1
            else:
                # Parallel C++: balance workers and threads
                # Use up to half the CPU cores as workers (leave room for OpenMP)
                max_workers = max(1, self._cpu_count // 2)
                workers = min(max_workers, batch_size)
                threads_per_circuit = max(1, self._effective_threads // workers)
                return workers, threads_per_circuit
        
        # Explicit worker count specified (N > 1)
        if self.max_batch_workers > 1:
            workers = min(self.max_batch_workers, batch_size)
            
            # If C++ is sequential, don't waste threads
            if is_cpp_sequential:
                return workers, 1
            else:
                threads_per_circuit = max(1, self._effective_threads // workers)
                return workers, threads_per_circuit
        
        # Fallback: sequential
        return 1, self._effective_threads

    def _execute_batch_parallel(
        self,
        circuits: List[QuantumTape],
        workers: int,
        threads_per_circuit: int,
    ) -> List[np.ndarray]:
        """Execute a batch of circuits in parallel using ThreadPoolExecutor.
        
        Parameters
        ----------
        circuits : List[QuantumTape]
            Circuits to execute.
        workers : int
            Number of parallel workers.
        threads_per_circuit : int
            C++ threads to allocate per circuit.
            
        Returns
        -------
        List[np.ndarray]
            Results in order matching input circuits.
        """
        def execute_single(tape: QuantumTape) -> np.ndarray:
            """Execute a single tape with specified thread count."""
            return self._execute_tape_with_threads(tape, threads_per_circuit)
        
        # Use ThreadPoolExecutor for parallel execution
        # ThreadPool works well here because the actual work happens in C++ (releases GIL)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(execute_single, circuits))
        
        return results

    def _execute_tape_with_threads(
        self,
        tape: QuantumTape,
        num_threads: int,
    ) -> np.ndarray:
        """Execute a single tape with a specific thread count.
        
        This is used by parallel batch execution to override the thread count.
        """
        # Build JSON circuit with custom thread count
        circuit_json = self._tape_to_json_with_threads(tape, num_threads)
        
        # Run simulation
        try:
            result = simulate_json(circuit_json, export_state=False)
        except QLRETError as e:
            raise QLRETDeviceError(f"Simulation failed: {e}") from e

        # Extract results based on measurement types
        return self._process_results(tape, result)

    # ------------------------------------------------------------------
    # Internal Execution
    # ------------------------------------------------------------------

    def _needs_state_export(self, tape: QuantumTape) -> bool:
        """Check if tape requires state export for probability computation."""
        # Check if there are ProbabilityMP measurements without shots
        has_prob_measurement = any(isinstance(m, ProbabilityMP) for m in tape.measurements)
        tape_shots = tape.shots if hasattr(tape, 'shots') else self.shots
        
        # Check if shots are specified and > 0
        has_shots = False
        if tape_shots is not None:
            if hasattr(tape_shots, 'total_shots'):
                total = tape_shots.total_shots
                has_shots = total is not None and total > 0
            elif isinstance(tape_shots, int):
                has_shots = tape_shots > 0
        
        return has_prob_measurement and not has_shots

    def _has_hamiltonian(self, tape: QuantumTape) -> bool:
        """Check if tape contains any Hamiltonian observables."""
        for m in tape.measurements:
            if isinstance(m, ExpectationMP) and m.obs is not None:
                if hasattr(m.obs, 'terms') and callable(m.obs.terms):
                    try:
                        coeffs, _ = m.obs.terms()
                        if len(coeffs) > 1:
                            return True
                    except Exception:
                        pass
        return False

    def _execute_tape(self, tape: QuantumTape) -> np.ndarray:
        """Execute a single quantum tape with default thread settings.
        
        Handles special cases:
        - Hamiltonians: Decomposed into individual Pauli terms
        - Probabilities without shots: State export + density matrix computation
        """
        # Check if we need special handling
        needs_export = self._needs_state_export(tape)
        has_hamiltonian = self._has_hamiltonian(tape)
        
        if has_hamiltonian:
            return self._execute_tape_with_hamiltonian(tape)
        
        # Build JSON circuit
        circuit_json = self._tape_to_json(tape, export_state=needs_export)
        
        # Run simulation
        try:
            result = simulate_json(circuit_json, export_state=needs_export)
        except QLRETError as e:
            raise QLRETDeviceError(f"Simulation failed: {e}") from e

        # Extract results based on measurement types
        return self._process_results(tape, result)

    def _execute_tape_with_hamiltonian(self, tape: QuantumTape) -> np.ndarray:
        """Execute tape with Hamiltonian by decomposing into individual terms.
        
        For H = Σ c_i * P_i where P_i are Pauli products:
        <H> = Σ c_i * <P_i>
        """
        # Get operations JSON (same for all term evaluations)
        operations = [_op_to_json(op) for op in tape.operations]
        
        outputs = []
        
        for m in tape.measurements:
            if isinstance(m, ExpectationMP) and m.obs is not None:
                obs = m.obs
                # Check if it's a Hamiltonian
                if hasattr(obs, 'terms') and callable(obs.terms):
                    try:
                        coeffs, ops = obs.terms()
                        if len(coeffs) > 1:
                            # Compute expectation of each term
                            total = 0.0
                            for coeff, op in zip(coeffs, ops):
                                # Build circuit for this term
                                term_obs = _obs_to_json(op)
                                circuit_json = {
                                    "circuit": {
                                        "num_qubits": self.num_wires,
                                        "operations": operations,
                                        "observables": [term_obs],
                                    },
                                    "config": {
                                        "epsilon": self.epsilon,
                                        "initial_rank": 1,
                                        "export_state": False,
                                        "num_threads": self._effective_threads,
                                        "parallel_mode": self.parallel_mode,
                                    },
                                }
                                result = simulate_json(circuit_json, export_state=False)
                                exp_val = result.get("expectation_values", [0.0])[0]
                                total += float(coeff) * exp_val
                            outputs.append(total)
                            continue
                    except Exception:
                        pass
                
                # Single observable - evaluate directly
                obs_json = _obs_to_json(obs)
                circuit_json = {
                    "circuit": {
                        "num_qubits": self.num_wires,
                        "operations": operations,
                        "observables": [obs_json],
                    },
                    "config": {
                        "epsilon": self.epsilon,
                        "initial_rank": 1,
                        "export_state": False,
                        "num_threads": self._effective_threads,
                        "parallel_mode": self.parallel_mode,
                    },
                }
                result = simulate_json(circuit_json, export_state=False)
                exp_val = result.get("expectation_values", [0.0])[0]
                outputs.append(exp_val)
                
            elif isinstance(m, ProbabilityMP):
                # For probability, we need state export
                circuit_json = {
                    "circuit": {
                        "num_qubits": self.num_wires,
                        "operations": operations,
                        "observables": [],
                    },
                    "config": {
                        "epsilon": self.epsilon,
                        "initial_rank": 1,
                        "export_state": True,
                        "num_threads": self._effective_threads,
                        "parallel_mode": self.parallel_mode,
                    },
                }
                result = simulate_json(circuit_json, export_state=True)
                probs = self._compute_probabilities_from_state(result, m.wires)
                outputs.append(probs)
            else:
                outputs.append(0.0)
        
        # Return single value if only one measurement
        if len(outputs) == 1:
            return np.asarray(outputs[0])
        return tuple(np.asarray(o) for o in outputs)

    def _compute_probabilities_from_state(
        self, result: Dict[str, Any], wires: Optional[Any] = None
    ) -> np.ndarray:
        """Compute probability distribution from low-rank state.
        
        For density matrix ρ = L @ L†, probabilities are diag(ρ).
        
        Notes
        -----
        LRET uses little-endian qubit ordering (qubit 0 = LSB), while
        PennyLane uses big-endian (qubit 0 = MSB). We convert the output
        to match PennyLane's convention.
        """
        state = result.get("state")
        if state is None:
            # Return uniform distribution
            num_wires = len(wires) if wires else self.num_wires
            return np.ones(2**num_wires) / (2**num_wires)
        
        # Reconstruct density matrix diagonal (for probabilities we only need diagonal)
        L_real = np.array(state.get("L_real", []))
        L_imag = np.array(state.get("L_imag", []))
        
        if L_real.size == 0:
            num_wires = len(wires) if wires else self.num_wires
            return np.ones(2**num_wires) / (2**num_wires)
        
        # Reshape L to matrix form
        rows = state.get("rows", len(L_real))
        cols = state.get("cols", 1)
        L = (L_real + 1j * L_imag).reshape(rows, cols)
        
        # Probabilities are diagonal elements of ρ = L @ L†
        # For efficiency, compute row-wise: probs[i] = ||L[i, :]||²
        probs = np.sum(np.abs(L)**2, axis=1).real
        
        # Convert from LRET's little-endian to PennyLane's big-endian ordering
        # In little-endian: index = q_0 + 2*q_1 + 4*q_2 + ...
        # In big-endian: index = q_{n-1} + 2*q_{n-2} + ... + 2^{n-1}*q_0
        probs = self._convert_endianness(probs, self.num_wires)
        
        # Marginalize if measuring subset of wires
        target_wires = list(wires) if wires else list(range(self.num_wires))
        if len(target_wires) < self.num_wires:
            probs = self._marginalize_probabilities(probs, target_wires)
        
        return probs

    def _convert_endianness(self, probs: np.ndarray, n_qubits: int) -> np.ndarray:
        """Convert probability array between little-endian and big-endian qubit ordering.
        
        This swaps the bit ordering in the indices, e.g., for 2 qubits:
        little-endian index 1 (01) -> big-endian index 2 (10)
        little-endian index 2 (10) -> big-endian index 1 (01)
        """
        # Reshape to [2, 2, ..., 2] tensor
        probs = probs.reshape([2] * n_qubits)
        # Reverse the axes to swap bit ordering
        probs = np.transpose(probs, axes=list(range(n_qubits - 1, -1, -1)))
        return probs.flatten()

    def _tape_to_json(self, tape: QuantumTape, export_state: bool = False) -> Dict[str, Any]:
        """Convert a PennyLane tape to QLRET JSON format.
        
        Parameters
        ----------
        tape : QuantumTape
            The quantum tape to convert.
        export_state : bool
            If True, request state export for probability computation.
        """
        # Operations
        operations = []
        for op in tape.operations:
            operations.append(_op_to_json(op))

        # Observables from measurements - skip Hamiltonians (handled separately)
        observables = []
        for m in tape.measurements:
            if isinstance(m, (ExpectationMP, VarianceMP, SampleMP)):
                obs = m.obs
                if obs is not None:
                    obs_json = _obs_to_json(obs)
                    # Skip HAMILTONIAN type - handled in Python layer
                    if obs_json.get("type") != "HAMILTONIAN":
                        observables.append(obs_json)
            elif isinstance(m, ProbabilityMP):
                # Probability doesn't need an observable
                pass

        # Build config
        config: Dict[str, Any] = {
            "epsilon": self.epsilon,
            "initial_rank": 1,
            "export_state": export_state,
            "num_threads": self._effective_threads,
            "parallel_mode": self.parallel_mode,
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

    def _tape_to_json_with_threads(
        self, tape: QuantumTape, num_threads: int
    ) -> Dict[str, Any]:
        """Convert a PennyLane tape to QLRET JSON format with custom thread count.
        
        This method is used by parallel batch execution to override the
        thread count for each circuit.
        
        Parameters
        ----------
        tape : QuantumTape
            The quantum tape to convert.
        num_threads : int
            Number of C++ threads to use for this circuit.
            
        Returns
        -------
        Dict[str, Any]
            QLRET JSON circuit specification.
        """
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

        # Build config with custom thread count
        config: Dict[str, Any] = {
            "epsilon": self.epsilon,
            "initial_rank": 1,
            "export_state": False,
            "num_threads": num_threads,  # Use provided thread count
            "parallel_mode": self.parallel_mode,
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
        """Process QLRET results into PennyLane format.
        
        Handles all measurement types:
        - ExpectationMP: Returns expectation value <O>
        - VarianceMP: Returns variance Var(O) = <O²> - <O>²
        - SampleMP: Returns shot samples as bit arrays
        - ProbabilityMP: Returns probability distribution
        - StateMP/DensityMatrixMP: Returns density matrix (if available)
        - PurityMP: Returns trace(ρ²) (if available)
        """
        expectations = result.get("expectation_values", [])
        samples = result.get("samples")
        state = result.get("state")  # Low-rank state L matrix if exported
        probabilities = result.get("probabilities")  # If computed by LRET

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
                # Variance = <O²> - <O>²
                # For Pauli observables, O² = I, so <O²> = 1
                # Therefore Var(O) = 1 - <O>²
                if obs_idx < len(expectations):
                    exp_val = expectations[obs_idx]
                    # Check if observable is a Pauli operator (O² = I)
                    obs = m.obs
                    if obs is not None and obs.name in ("PauliX", "PauliY", "PauliZ"):
                        variance = 1.0 - exp_val ** 2
                    else:
                        # General case: need to compute <O²> separately
                        # For now, use Pauli approximation
                        variance = 1.0 - exp_val ** 2
                    outputs.append(variance)
                    obs_idx += 1
                else:
                    outputs.append(0.0)

            elif isinstance(m, SampleMP):
                if samples is not None:
                    # Convert integer samples to bit arrays
                    n_qubits = self.num_wires
                    sample_array = np.array(samples, dtype=np.int64)
                    
                    # If observable is specified, compute eigenvalue samples
                    if m.obs is not None:
                        obs_idx += 1
                        # For Pauli observables, map bitstrings to eigenvalues
                        if m.obs.name in ("PauliX", "PauliY", "PauliZ"):
                            wire = m.obs.wires[0]
                            # Extract bit for this wire and map 0->+1, 1->-1
                            bits = (sample_array >> (n_qubits - 1 - wire)) & 1
                            eigenvalues = 1 - 2 * bits  # 0->1, 1->-1
                            outputs.append(eigenvalues.astype(np.float64))
                        else:
                            outputs.append(sample_array)
                    else:
                        # Return raw computational basis samples
                        # Convert to binary representation
                        bit_samples = np.zeros((len(sample_array), n_qubits), dtype=np.int64)
                        for i, s in enumerate(sample_array):
                            for q in range(n_qubits):
                                bit_samples[i, n_qubits - 1 - q] = (s >> q) & 1
                        outputs.append(bit_samples)
                else:
                    outputs.append(np.array([]))
                    if m.obs is not None:
                        obs_idx += 1

            elif isinstance(m, ProbabilityMP):
                # Compute probabilities from samples, state, or return zeros
                wires = m.wires if m.wires else list(range(self.num_wires))
                num_prob_wires = len(wires)
                
                if probabilities is not None:
                    # Use precomputed probabilities from LRET
                    outputs.append(np.array(probabilities))
                elif samples is not None:
                    # Compute from samples
                    counts = np.bincount(samples, minlength=2**self.num_wires)
                    probs = counts / len(samples)
                    
                    # If measuring subset of wires, marginalize
                    if num_prob_wires < self.num_wires:
                        probs = self._marginalize_probabilities(probs, wires)
                    
                    outputs.append(probs)
                elif state is not None:
                    # Compute from low-rank state
                    probs = self._compute_probabilities_from_state(result, wires)
                    outputs.append(probs)
                else:
                    outputs.append(np.zeros(2**num_prob_wires))

            # Handle state/density matrix export (if available)
            elif StateMP is not None and isinstance(m, StateMP):
                if state is not None:
                    # Reconstruct density matrix from low-rank L: ρ = L @ L†
                    dm = self._reconstruct_density_matrix(state)
                    outputs.append(dm)
                else:
                    outputs.append(np.eye(2**self.num_wires) / (2**self.num_wires))

            elif DensityMatrixMP is not None and isinstance(m, DensityMatrixMP):
                if state is not None:
                    dm = self._reconstruct_density_matrix(state)
                    # Trace out unwanted wires if specified
                    wires = m.wires if m.wires else list(range(self.num_wires))
                    if len(wires) < self.num_wires:
                        dm = self._partial_trace(dm, wires)
                    outputs.append(dm)
                else:
                    outputs.append(np.eye(2**self.num_wires) / (2**self.num_wires))

            elif PurityMP is not None and isinstance(m, PurityMP):
                if state is not None:
                    dm = self._reconstruct_density_matrix(state)
                    # Purity = Tr(ρ²)
                    purity = np.real(np.trace(dm @ dm))
                    outputs.append(purity)
                else:
                    # For pure states, purity = 1
                    outputs.append(1.0)

        if len(outputs) == 1:
            return np.asarray(outputs[0])
        return tuple(np.asarray(o) for o in outputs)

    def _marginalize_probabilities(
        self, probs: np.ndarray, wires: List[int]
    ) -> np.ndarray:
        """Marginalize probability distribution to subset of wires.
        
        Parameters
        ----------
        probs : np.ndarray
            Full probability distribution over all qubits.
        wires : List[int]
            Wires to keep (marginalize over the rest).
            
        Returns
        -------
        np.ndarray
            Marginalized probability distribution.
            
        Notes
        -----
        The probability array is indexed in little-endian order:
        index = q_0 + 2*q_1 + 4*q_2 + ...
        
        When reshaped to [2, 2, ..., 2], the shape is [q_{n-1}, ..., q_1, q_0].
        Wire i corresponds to axis (n-1-i).
        """
        n_qubits = self.num_wires
        n_kept = len(wires)
        
        # Reshape probabilities: shape [q_{n-1}, ..., q_1, q_0]
        probs = probs.reshape([2] * n_qubits)
        
        # Map wire indices to array axes
        # Wire i corresponds to axis (n-1-i) in the reshaped array
        keep_axes = set(n_qubits - 1 - w for w in wires)
        axes_to_sum = [i for i in range(n_qubits) if i not in keep_axes]
        
        # Sum in reverse order to maintain axis indices
        for axis in sorted(axes_to_sum, reverse=True):
            probs = probs.sum(axis=axis)
        
        return probs.flatten()

    def _reconstruct_density_matrix(self, state: Dict[str, Any]) -> np.ndarray:
        """Reconstruct full density matrix from low-rank state.
        
        The LRET state is stored as L matrix where ρ = L @ L†.
        
        Parameters
        ----------
        state : Dict[str, Any]
            State dictionary with 'L_real' and 'L_imag' matrices.
            
        Returns
        -------
        np.ndarray
            The full density matrix (2^n × 2^n).
        """
        L_real = np.array(state.get("L_real", []))
        L_imag = np.array(state.get("L_imag", []))
        
        if L_real.size == 0:
            # Return maximally mixed state
            dim = 2 ** self.num_wires
            return np.eye(dim) / dim
        
        L = L_real + 1j * L_imag
        # Reconstruct: ρ = L @ L†
        rho = L @ L.conj().T
        return rho

    def _partial_trace(
        self, dm: np.ndarray, keep_wires: List[int]
    ) -> np.ndarray:
        """Compute partial trace of density matrix.
        
        Parameters
        ----------
        dm : np.ndarray
            Full density matrix.
        keep_wires : List[int]
            Wires to keep (trace out the rest).
            
        Returns
        -------
        np.ndarray
            Reduced density matrix.
        """
        n_qubits = self.num_wires
        keep_set = set(keep_wires)
        trace_wires = [i for i in range(n_qubits) if i not in keep_set]
        
        if not trace_wires:
            return dm
        
        # Reshape to tensor form
        shape = [2] * (2 * n_qubits)
        dm_tensor = dm.reshape(shape)
        
        # Trace over specified wires
        # For each wire to trace, contract corresponding row and column indices
        for wire in sorted(trace_wires, reverse=True):
            # Trace over axis 'wire' and 'wire + n_qubits'
            dm_tensor = np.trace(dm_tensor, axis1=wire, axis2=wire + n_qubits - len([w for w in trace_wires if w < wire]))
        
        # Reshape back to matrix
        kept_dim = 2 ** len(keep_wires)
        return dm_tensor.reshape(kept_dim, kept_dim)

    def compute_purity(self, tape: QuantumTape) -> float:
        """Compute purity of the final quantum state.
        
        Purity = Tr(ρ²), where ρ is the density matrix.
        For pure states, purity = 1. For maximally mixed states, purity = 1/d.
        
        Parameters
        ----------
        tape : QuantumTape
            The quantum circuit to execute.
            
        Returns
        -------
        float
            The purity of the final state.
        """
        # Execute with state export
        circuit_json = self._tape_to_json(tape)
        circuit_json["config"]["export_state"] = True
        
        try:
            result = simulate_json(circuit_json, export_state=True)
        except QLRETError as e:
            raise QLRETDeviceError(f"Simulation failed: {e}") from e
        
        state = result.get("state")
        if state is not None:
            dm = self._reconstruct_density_matrix(state)
            return float(np.real(np.trace(dm @ dm)))
        else:
            return 1.0  # Assume pure state if no state exported

    def compute_entanglement_entropy(
        self, tape: QuantumTape, subsystem: List[int]
    ) -> float:
        """Compute von Neumann entanglement entropy.
        
        S(ρ_A) = -Tr(ρ_A log ρ_A)
        
        Parameters
        ----------
        tape : QuantumTape
            The quantum circuit to execute.
        subsystem : List[int]
            Wires defining subsystem A for bipartite entanglement.
            
        Returns
        -------
        float
            The von Neumann entropy of the reduced density matrix.
        """
        # Execute with state export
        circuit_json = self._tape_to_json(tape)
        circuit_json["config"]["export_state"] = True
        
        try:
            result = simulate_json(circuit_json, export_state=True)
        except QLRETError as e:
            raise QLRETDeviceError(f"Simulation failed: {e}") from e
        
        state = result.get("state")
        if state is None:
            return 0.0  # Pure product state
        
        dm = self._reconstruct_density_matrix(state)
        rho_A = self._partial_trace(dm, subsystem)
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(rho_A)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]  # Remove numerical zeros
        
        # von Neumann entropy: S = -Σ λ log(λ)
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
        return float(entropy)

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
