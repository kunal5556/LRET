"""LRET Simulator for Cirq.

Provides a Cirq-compatible simulator interface backed by LRET's
low-rank quantum simulation engine.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Union
import warnings

try:
    import cirq
except ImportError:
    raise ImportError(
        "Cirq is required for lret_cirq. Install with: pip install cirq>=1.3.0"
    )

from .translators.circuit_translator import CircuitTranslator, TranslationError
from .translators.result_converter import ResultConverter

# Try to import LRET native module
try:
    from qlret.api import simulate_json
    NATIVE_AVAILABLE = True
except ImportError:
    NATIVE_AVAILABLE = False
    warnings.warn(
        "LRET native module not available. "
        "Install with: cd python/qlret && pip install -e .",
        ImportWarning
    )

__all__ = ["LRETSimulator"]


class LRETSimulator(cirq.SimulatesSamples):
    """
    LRET-backed quantum circuit simulator for Cirq.
    
    This simulator uses low-rank approximations to efficiently simulate
    quantum circuits with automatic rank adaptation.
    
    Attributes:
        epsilon (float): SVD truncation threshold (default: 1e-4)
        noise_model (Optional[dict]): LRET noise configuration
        seed (Optional[int]): Random seed for reproducibility
    
    Example:
        >>> import cirq
        >>> from lret_cirq import LRETSimulator
        >>> 
        >>> q0, q1 = cirq.LineQubit.range(2)
        >>> circuit = cirq.Circuit(
        ...     cirq.H(q0),
        ...     cirq.CNOT(q0, q1),
        ...     cirq.measure(q0, q1, key='result')
        ... )
        >>> 
        >>> sim = LRETSimulator(epsilon=1e-4)
        >>> result = sim.run(circuit, repetitions=1000)
        >>> print(result.histogram(key='result'))
        Counter({0: 489, 3: 511})  # ~50/50 split for Bell state
    """
    
    def __init__(
        self,
        epsilon: float = 1e-4,
        noise_model: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize LRET simulator.
        
        Args:
            epsilon: Truncation threshold for low-rank approximation.
                     Smaller values = higher accuracy but slower.
                     Typical range: 1e-3 (fast) to 1e-6 (accurate)
            noise_model: Optional LRET noise configuration dict
            seed: Random seed for measurement sampling
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)
        
        if not NATIVE_AVAILABLE:
            raise ImportError(
                "LRET native module required for simulation. "
                "Build with: cd python/qlret && pip install -e ."
            )
        
        if epsilon <= 0 or epsilon >= 1:
            raise ValueError(f"Epsilon must be in (0, 1), got {epsilon}")
        
        self._epsilon = epsilon
        self._noise_model = noise_model
        self._seed = seed
        
        # Create translator and converter instances
        self._translator = CircuitTranslator()
        self._converter = ResultConverter()
    
    @property
    def epsilon(self) -> float:
        """Get the current truncation threshold."""
        return self._epsilon
    
    @epsilon.setter
    def epsilon(self, value: float):
        """Set the truncation threshold."""
        if value <= 0 or value >= 1:
            raise ValueError(f"Epsilon must be in (0, 1), got {value}")
        self._epsilon = value
    
    @property
    def noise_model(self) -> Optional[Dict[str, Any]]:
        """Get the current noise model."""
        return self._noise_model
    
    @noise_model.setter
    def noise_model(self, value: Optional[Dict[str, Any]]):
        """Set the noise model."""
        self._noise_model = value
    
    def _run(
        self,
        circuit: cirq.AbstractCircuit,
        param_resolver: cirq.ParamResolver,
        repetitions: int,
    ) -> Dict[str, np.ndarray]:
        """
        Run the circuit and return measurement results.
        
        This is the main interface method required by cirq.SimulatesSamples.
        
        Args:
            circuit: The Cirq circuit to simulate
            param_resolver: Parameter values for parameterized circuits
            repetitions: Number of measurement samples to take
        
        Returns:
            Dict mapping measurement keys to sample arrays
            
        Raises:
            TranslationError: If circuit contains unsupported operations
            ValueError: If circuit has unresolved parameters
        """
        # Resolve parameters if provided
        if param_resolver:
            circuit = cirq.resolve_parameters(circuit, param_resolver)
        
        # Check for unresolved parameters
        if cirq.is_parameterized(circuit):
            raise ValueError(
                "Circuit contains unresolved parameters. "
                "Provide a param_resolver or use cirq.resolve_parameters()."
            )
        
        # Translate circuit to LRET JSON
        try:
            lret_json = self._translator.translate(
                circuit=circuit,
                epsilon=self._epsilon,
                shots=repetitions,
                noise_model=self._noise_model,
                seed=self._seed,
            )
        except TranslationError:
            raise
        except Exception as e:
            raise TranslationError(f"Circuit translation failed: {e}") from e
        
        # Execute simulation
        try:
            lret_result = simulate_json(lret_json)
        except Exception as e:
            raise RuntimeError(f"LRET simulation failed: {e}") from e
        
        # Check for simulation errors
        if lret_result.get("status") == "error":
            raise RuntimeError(
                f"LRET simulation error: {lret_result.get('message', 'Unknown error')}"
            )
        
        # Get translation metadata
        qubit_map = self._translator.get_qubit_map()
        measurement_keys = self._translator.get_measurement_keys()
        measurement_qubits = self._translator.get_measurement_qubits()
        
        # Convert result back to Cirq format
        measurements_dict = self._converter.convert(
            lret_result=lret_result,
            qubit_map=qubit_map,
            measurement_keys=measurement_keys,
            measurement_qubits=measurement_qubits,
            repetitions=repetitions,
        )
        
        return measurements_dict
    
    def _create_simulator_trial_result(
        self,
        params: cirq.ParamResolver,
        measurements: Dict[str, np.ndarray],
        final_simulator_state: Any,
    ) -> cirq.ResultDict:
        """
        Create a trial result from measurements.
        
        Args:
            params: Parameter resolver used
            measurements: Dict of measurement results
            final_simulator_state: Final state (unused for sampling)
            
        Returns:
            cirq.ResultDict
        """
        return cirq.ResultDict(
            params=params,
            measurements=measurements,
        )
    
    def __str__(self) -> str:
        """String representation."""
        return f"LRETSimulator(epsilon={self._epsilon})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        noise_str = "Yes" if self._noise_model else "None"
        return (
            f"LRETSimulator("
            f"epsilon={self._epsilon}, "
            f"noise_model={noise_str}, "
            f"seed={self._seed})"
        )
