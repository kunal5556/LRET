"""Basic smoke tests for LRET Qiskit backend."""

import pytest
from qiskit import QuantumCircuit

from lret_qiskit import LRETProvider


def test_provider_lists_backends():
    """Provider exposes backends."""
    provider = LRETProvider()
    backends = provider.backends()
    assert any(b.name == "lret_simulator" for b in backends)


def test_get_backend_returns_correct_type():
    """get_backend returns LRETBackend instance."""
    from lret_qiskit.backends import LRETBackend
    provider = LRETProvider()
    backend = provider.get_backend("lret_simulator")
    assert isinstance(backend, LRETBackend)


def test_backend_has_valid_target():
    """Backend has Target with standard gates."""
    provider = LRETProvider()
    backend = provider.get_backend("lret_simulator")
    
    target = backend.target
    assert target is not None
    assert target.num_qubits > 0
    
    # Check some expected gates exist
    op_names = [op.name.lower() for op in target.operations]
    assert "h" in op_names
    assert "cx" in op_names
