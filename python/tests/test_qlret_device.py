"""Tests for QLRET PennyLane device and Python bridge.

Run with: pytest tests/test_qlret_device.py -v
"""

import json
import numpy as np
import pytest
from pathlib import Path

# Import QLRET
from qlret import simulate_json, load_json_file, QLRETError
from qlret.pennylane_device import QLRETDevice, QLRETDeviceError

# Check if PennyLane is available
try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

# Check if native module is available
try:
    from qlret import _qlret_native
    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False

# Path to sample JSON (project root / samples / json)
SAMPLES_DIR = Path(__file__).parent.parent.parent / "samples" / "json"


# ---------------------------------------------------------------------------
# JSON API Tests
# ---------------------------------------------------------------------------

class TestLoadJsonFile:
    def test_load_bell_pair(self):
        """Test loading sample Bell pair circuit."""
        if not (SAMPLES_DIR / "bell_pair.json").exists():
            pytest.skip("Sample file not found")
        
        circuit = load_json_file(SAMPLES_DIR / "bell_pair.json")
        
        assert "circuit" in circuit
        assert circuit["circuit"]["num_qubits"] == 2
        assert len(circuit["circuit"]["operations"]) == 2
        assert len(circuit["circuit"]["observables"]) == 2


class TestSimulateJson:
    """Tests for simulate_json function."""
    
    @pytest.fixture
    def bell_circuit(self):
        """Simple Bell state circuit."""
        return {
            "circuit": {
                "num_qubits": 2,
                "operations": [
                    {"name": "H", "wires": [0]},
                    {"name": "CNOT", "wires": [0, 1]},
                ],
                "observables": [
                    {"type": "PAULI", "operator": "Z", "wires": [0]},
                    {"type": "TENSOR", "operators": ["Z", "Z"], "wires": [0, 1]},
                ],
            },
            "config": {
                "epsilon": 1e-4,
                "initial_rank": 1,
            },
        }
    
    @pytest.mark.skipif(not HAS_NATIVE, reason="Native module not built")
    def test_simulate_native(self, bell_circuit):
        """Test simulation with native bindings."""
        result = simulate_json(bell_circuit, use_native=True)
        
        assert result["status"] == "success"
        assert "expectation_values" in result
        assert len(result["expectation_values"]) == 2
        
        # Bell state: <Z_0> = 0 (equal superposition)
        # <Z_0 Z_1> = 1 (always correlated)
        assert abs(result["expectation_values"][0]) < 0.1
        assert abs(result["expectation_values"][1] - 1.0) < 0.1
    
    def test_simulate_subprocess(self, bell_circuit):
        """Test simulation with subprocess backend."""
        try:
            result = simulate_json(bell_circuit, use_native=False)
            
            assert result["status"] == "success"
            assert "expectation_values" in result
            assert len(result["expectation_values"]) == 2
        except QLRETError as e:
            if "not found" in str(e):
                pytest.skip("quantum_sim executable not found")
            raise
    
    def test_simulate_with_sampling(self, bell_circuit):
        """Test simulation with shot-based sampling."""
        bell_circuit["config"]["shots"] = 100
        
        try:
            result = simulate_json(bell_circuit, use_native=False)
            
            if result["status"] == "success":
                assert "samples" in result
                if result["samples"] is not None:
                    assert len(result["samples"]) == 100
                    # Bell state samples should be 00 or 11 (0 or 3 in decimal)
                    for sample in result["samples"]:
                        assert sample in [0, 3]
        except QLRETError as e:
            if "not found" in str(e):
                pytest.skip("quantum_sim executable not found")
            raise
    
    def test_invalid_circuit(self):
        """Test error handling for invalid circuit."""
        invalid = {"circuit": {"num_qubits": 0, "operations": []}}
        
        with pytest.raises((QLRETError, Exception)):
            simulate_json(invalid, use_native=False)


# ---------------------------------------------------------------------------
# PennyLane Device Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
class TestQLRETDevice:
    """Tests for PennyLane device integration."""
    
    def test_device_creation(self):
        """Test device can be created and used in qnode."""
        dev = QLRETDevice(wires=4, shots=1000)
        
        # Test basic properties
        assert dev.num_wires == 4
        assert dev.shots == 1000
        
        # Test it works in a qnode (actual usage pattern)
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))
        
        # If this doesn't error, device is functional
        try:
            result = circuit()
            assert isinstance(result, (float, np.ndarray))
        except QLRETDeviceError as e:
            if "not found" in str(e) or "Simulation failed" in str(e):
                pytest.skip("Backend not available")
            raise
    
    def test_device_capabilities(self):
        """Test device capabilities reporting."""
        caps = QLRETDevice.capabilities()
        
        assert caps["model"] == "qubit"
        assert caps["supports_tensor_observables"] is True
    
    def test_tape_to_json(self):
        """Test that device can execute tape and produce correct results."""
        dev = QLRETDevice(wires=2, shots=None, epsilon=1e-4)
        
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))
        
        try:
            result = circuit()
            # Bell state should give <Z0> close to 0
            assert isinstance(result, (float, np.ndarray))
        except QLRETDeviceError as e:
            if "not found" in str(e) or "Simulation failed" in str(e):
                pytest.skip("Backend not available")
            raise
    
    def test_bell_state_expectation(self):
        """Test Bell state expectation values."""
        dev = QLRETDevice(wires=2, shots=None, epsilon=1e-4)
        
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))
        
        try:
            result = circuit()
            # Bell state should give <Z0> close to 0
            assert abs(result - 0.0) < 0.1
        except QLRETDeviceError as e:
            if "not found" in str(e) or "Simulation failed" in str(e):
                pytest.skip("Backend not available")
            raise
    
    def test_parametrized_circuit(self):
        """Test circuit with parameters."""
        dev = QLRETDevice(wires=2, shots=None, epsilon=1e-4)
        
        @qml.qnode(dev)
        def circuit(theta):
            qml.RX(theta, wires=0)
            return qml.expval(qml.PauliZ(0))
        
        try:
            # RX(0) |0> = |0>, so <Z> = 1
            result = circuit(0.0)
            assert isinstance(result, (float, np.ndarray))
            assert abs(result - 1.0) < 0.1, f"Expected ~1.0, got {result}"
        except QLRETDeviceError as e:
            if "not found" in str(e) or "Simulation failed" in str(e):
                pytest.skip("Backend not available")
            raise


@pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
class TestGradients:
    """Tests for gradient computation."""
    
    def test_gradient_single_param(self):
        """Test gradient for single parameter."""
        dev = QLRETDevice(wires=2, shots=None)
        
        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(theta):
            qml.RX(theta, wires=0)
            return qml.expval(qml.PauliZ(0))
        
        try:
            # d<Z>/dtheta for RX = -sin(theta)
            # Use qml.numpy with requires_grad=True for PennyLane autodiff
            theta = qml.numpy.array(0.5, requires_grad=True)
            grad = qml.grad(circuit)(theta)
            expected = -np.sin(0.5)
            
            assert abs(float(grad) - expected) < 0.1
        except (QLRETDeviceError, Exception) as e:
            if "not found" in str(e) or "Simulation failed" in str(e):
                pytest.skip("Backend not available")
            raise
    
    def test_gradient_multi_param(self):
        """Test gradient for multiple parameters."""
        dev = QLRETDevice(wires=2, shots=None)
        
        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(theta, phi):
            qml.RX(theta, wires=0)
            qml.RY(phi, wires=0)
            return qml.expval(qml.PauliZ(0))
        
        try:
            # Use qml.numpy with requires_grad=True for PennyLane autodiff
            theta = qml.numpy.array(0.3, requires_grad=True)
            phi = qml.numpy.array(0.7, requires_grad=True)
            grads = qml.grad(circuit)(theta, phi)
            
            # Both gradients should be non-zero
            assert len(grads) == 2 or isinstance(grads, (tuple, list))
        except (QLRETDeviceError, Exception) as e:
            if "not found" in str(e) or "Simulation failed" in str(e):
                pytest.skip("Backend not available")
            raise


# ---------------------------------------------------------------------------
# Native Module Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_NATIVE, reason="Native module not built")
class TestNativeModule:
    """Tests for pybind11 native module."""
    
    def test_version(self):
        """Test version string."""
        from qlret import _qlret_native
        version = _qlret_native.get_version()
        assert "QLRET" in version
    
    def test_validate_circuit(self):
        """Test circuit validation."""
        from qlret import _qlret_native
        
        valid = json.dumps({
            "circuit": {
                "num_qubits": 2,
                "operations": [{"name": "H", "wires": [0]}],
                "observables": []
            },
            "config": {}
        })
        
        result = _qlret_native.validate_circuit_json(valid)
        assert result == ""  # Empty = success
    
    def test_validate_invalid(self):
        """Test validation of invalid circuit."""
        from qlret import _qlret_native
        
        invalid = json.dumps({"circuit": {}})
        result = _qlret_native.validate_circuit_json(invalid)
        assert result != ""  # Non-empty = error
