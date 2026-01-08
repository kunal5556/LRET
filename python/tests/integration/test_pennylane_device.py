"""Integration tests for PennyLane device."""

from __future__ import annotations

import numpy as np
import pytest

pennylane = pytest.importorskip("pennylane")
qml = pennylane

from qlret import QLRETDevice, QLRETDeviceError


@pytest.fixture(scope="module")
def ensure_backend(backend_available: bool):
    if not backend_available:
        pytest.skip("Neither native module nor quantum_sim executable is available")


@pytest.mark.pennylane
class TestQLRETDeviceBasics:
    """Basic device functionality tests."""

    def test_device_creation(self, ensure_backend):
        dev = QLRETDevice(wires=4, shots=1000)
        assert dev.num_wires == 4
        assert dev.shots == 1000

    def test_device_capabilities(self, ensure_backend):
        caps = QLRETDevice.capabilities()
        assert caps["model"] == "qubit"
        assert caps["supports_tensor_observables"] is True
        assert caps["supports_analytic_computation"] is True

    def test_basic_circuit_execution(self, ensure_backend):
        dev = QLRETDevice(wires=2, shots=None)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        result = circuit()
        assert abs(result) < 0.1  # |+> expectation for Z is ~0


@pytest.mark.pennylane
class TestObservables:
    """Observable measurement tests."""

    def test_single_observable(self, ensure_backend):
        dev = QLRETDevice(wires=1, shots=None)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(wires=0)
            return qml.expval(qml.PauliZ(0))

        result = circuit()
        assert abs(result + 1.0) < 0.1  # Z|1> = -1

    def test_tensor_observables(self, ensure_backend):
        dev = QLRETDevice(wires=2, shots=None)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        result = circuit()
        assert abs(result - 1.0) < 0.1  # Bell state correlation

    def test_hermitian_observable(self, ensure_backend):
        dev = QLRETDevice(wires=1, shots=None)
        obs_matrix = np.array([[0, 1], [1, 0]], dtype=complex)  # Pauli X

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.expval(qml.Hermitian(obs_matrix, wires=0))

        result = circuit()
        assert abs(result - 1.0) < 0.1


@pytest.mark.pennylane
class TestGradients:
    """Gradient computation tests."""

    def test_parameter_shift_single_param(self, ensure_backend):
        dev = QLRETDevice(wires=1, shots=None)
        pnp = qml.numpy  # PennyLane's numpy with requires_grad support

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(theta):
            qml.RX(theta, wires=0)
            return qml.expval(qml.PauliZ(0))

        theta = pnp.array(0.5, requires_grad=True)
        grad = qml.grad(circuit)(theta)
        expected = -np.sin(0.5)
        assert abs(grad - expected) < 0.1

    def test_multi_param_gradients(self, ensure_backend):
        dev = QLRETDevice(wires=2, shots=None)
        pnp = qml.numpy

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        params = pnp.array([0.3, 0.7], requires_grad=True)
        grads = qml.grad(circuit)(params)
        assert len(grads) == 2
        assert all(isinstance(g, (float, np.floating)) for g in grads)


@pytest.mark.pennylane
@pytest.mark.slow
class TestSampling:
    """Shot-based sampling tests."""

    def test_sampling_mode(self, ensure_backend):
        dev = QLRETDevice(wires=2, shots=100)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.sample()

        samples = circuit()
        assert len(samples) == 100
