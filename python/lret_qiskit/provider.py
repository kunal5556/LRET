"""
LRET Provider for Qiskit
========================

Entry point for accessing LRET simulators from Qiskit.
"""

from qiskit.providers.exceptions import QiskitBackendNotFoundError

from .backends.lret_backend import LRETBackend
from .version import __version__


class LRETProvider:
    """Provider for LRET quantum simulators.
    
    This provider gives access to three LRET backend variants with
    different accuracy/speed tradeoffs controlled by the epsilon
    (SVD truncation threshold) parameter.
    
    Example:
        >>> provider = LRETProvider()
        >>> backend = provider.get_backend("lret_simulator")
        >>> job = backend.run(circuit, shots=1024)
    """

    def __init__(self):
        self._backends = self._initialize_backends()

    def _initialize_backends(self):
        """Create available LRET backends."""
        return {
            "lret_simulator": LRETBackend(
                name="lret_simulator",
                description="LRET Low-Rank Quantum Simulator",
                epsilon=1e-4,
                provider=self,
            ),
            "lret_simulator_accurate": LRETBackend(
                name="lret_simulator_accurate",
                description="LRET Simulator (High Accuracy)",
                epsilon=1e-6,
                provider=self,
            ),
            "lret_simulator_fast": LRETBackend(
                name="lret_simulator_fast",
                description="LRET Simulator (Fast Mode)",
                epsilon=1e-3,
                provider=self,
            ),
        }

    def backends(self, name=None, filters=None, **kwargs):
        """Return a list of backends matching the specified filtering.
        
        Args:
            name: Filter by backend name.
            filters: Callable filter function.
            **kwargs: Additional filter arguments (ignored).
        
        Returns:
            List of matching backends.
        """
        backends = list(self._backends.values())

        if name:
            backends = [b for b in backends if b.name == name]

        if filters:
            backends = [b for b in backends if filters(b)]

        return backends

    def get_backend(self, name=None, **kwargs):
        """Return a single backend matching the specified name.
        
        Args:
            name: Backend name. If None, returns "lret_simulator".
            **kwargs: Additional arguments (ignored).
        
        Returns:
            LRETBackend instance.
        
        Raises:
            QiskitBackendNotFoundError: If backend not found.
        """
        if name is None:
            name = "lret_simulator"

        try:
            return self._backends[name]
        except KeyError as exc:
            raise QiskitBackendNotFoundError(
                f"Backend '{name}' not found. Available: {list(self._backends.keys())}"
            ) from exc

    def __str__(self):
        return f"<LRETProvider(version={__version__})>"
    
    def __repr__(self):
        return self.__str__()
