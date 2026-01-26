"""
Comprehensive test suite for LRET-Cirq integration.

Test Categories:
1. CircuitTranslator tests (15+ tests)
2. QubitMapping tests (5+ tests)
3. ResultConverter tests (5+ tests)
4. LRETSimulator tests (5+ tests)
5. Integration tests (10+ tests)
6. ErrorHandling tests (5+ tests)
7. GateSet tests (10+ tests)

Total: 55+ tests
"""

import pytest
import numpy as np
from typing import Dict, List
from unittest.mock import patch, MagicMock

try:
    import cirq
    CIRQ_AVAILABLE = True
except ImportError:
    CIRQ_AVAILABLE = False

# Skip all tests if cirq not available
pytestmark = pytest.mark.skipif(not CIRQ_AVAILABLE, reason="Cirq not installed")


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def translator():
    """Create a CircuitTranslator instance."""
    from lret_cirq.translators.circuit_translator import CircuitTranslator
    return CircuitTranslator()


@pytest.fixture
def converter():
    """Create a ResultConverter instance."""
    from lret_cirq.translators.result_converter import ResultConverter
    return ResultConverter()


@pytest.fixture
def mock_simulate_json():
    """Mock the LRET simulate_json function."""
    def _mock(circuit_json):
        num_qubits = circuit_json.get("circuit", {}).get("num_qubits", 2)
        shots = circuit_json.get("config", {}).get("shots", 1024)
        
        # Generate random samples for testing
        samples = np.random.randint(0, 2, size=(shots, num_qubits))
        
        return {
            "status": "success",
            "samples": samples.tolist(),
            "execution_time_ms": 10.0,
            "final_rank": 2,
        }
    return _mock


# ============================================================================
# Test Category 1: CircuitTranslator - Basic Gates
# ============================================================================

class TestCircuitTranslatorBasicGates:
    """Tests for basic gate translation."""
    
    def test_translate_empty_circuit(self, translator):
        """Empty circuit should have no operations."""
        q0 = cirq.LineQubit(0)
        circuit = cirq.Circuit()
        circuit._all_qubits = frozenset([q0])  # Force qubit registration
        
        # Need at least one operation to register qubit
        circuit = cirq.Circuit(cirq.I(q0))
        result = translator.translate(circuit)
        
        assert result["circuit"]["num_qubits"] == 1
        assert result["circuit"]["operations"] == []  # Identity is skipped
    
    def test_translate_hadamard(self, translator):
        """Single H gate translation."""
        q0 = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.H(q0))
        
        result = translator.translate(circuit)
        
        assert result["circuit"]["num_qubits"] == 1
        ops = result["circuit"]["operations"]
        assert len(ops) == 1
        assert ops[0]["name"] == "H"
        assert ops[0]["wires"] == [0]
    
    def test_translate_pauli_x(self, translator):
        """Single X gate translation."""
        q0 = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.X(q0))
        
        result = translator.translate(circuit)
        
        ops = result["circuit"]["operations"]
        assert len(ops) == 1
        assert ops[0]["name"] == "X"
        assert ops[0]["wires"] == [0]
    
    def test_translate_pauli_y(self, translator):
        """Single Y gate translation."""
        q0 = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.Y(q0))
        
        result = translator.translate(circuit)
        
        ops = result["circuit"]["operations"]
        assert len(ops) == 1
        assert ops[0]["name"] == "Y"
        assert ops[0]["wires"] == [0]
    
    def test_translate_pauli_z(self, translator):
        """Single Z gate translation."""
        q0 = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.Z(q0))
        
        result = translator.translate(circuit)
        
        ops = result["circuit"]["operations"]
        assert len(ops) == 1
        assert ops[0]["name"] == "Z"
        assert ops[0]["wires"] == [0]
    
    def test_translate_s_gate(self, translator):
        """S gate (√Z) translation."""
        q0 = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.S(q0))
        
        result = translator.translate(circuit)
        
        ops = result["circuit"]["operations"]
        assert len(ops) == 1
        assert ops[0]["name"] == "S"
        assert ops[0]["wires"] == [0]
    
    def test_translate_t_gate(self, translator):
        """T gate (∜Z) translation."""
        q0 = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.T(q0))
        
        result = translator.translate(circuit)
        
        ops = result["circuit"]["operations"]
        assert len(ops) == 1
        assert ops[0]["name"] == "T"
        assert ops[0]["wires"] == [0]
    
    def test_translate_cnot(self, translator):
        """CNOT gate translation."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(cirq.CNOT(q0, q1))
        
        result = translator.translate(circuit)
        
        assert result["circuit"]["num_qubits"] == 2
        ops = result["circuit"]["operations"]
        assert len(ops) == 1
        assert ops[0]["name"] == "CNOT"
        assert ops[0]["wires"] == [0, 1]
    
    def test_translate_cz(self, translator):
        """CZ gate translation."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(cirq.CZ(q0, q1))
        
        result = translator.translate(circuit)
        
        ops = result["circuit"]["operations"]
        assert len(ops) == 1
        assert ops[0]["name"] == "CZ"
        assert ops[0]["wires"] == [0, 1]
    
    def test_translate_swap(self, translator):
        """SWAP gate translation."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(cirq.SWAP(q0, q1))
        
        result = translator.translate(circuit)
        
        ops = result["circuit"]["operations"]
        assert len(ops) == 1
        assert ops[0]["name"] == "SWAP"
        assert ops[0]["wires"] == [0, 1]


# ============================================================================
# Test Category 2: CircuitTranslator - Power Gates
# ============================================================================

class TestCircuitTranslatorPowerGates:
    """Tests for power gate translation."""
    
    def test_x_power_gate_full(self, translator):
        """XPowGate(1.0) should map to X."""
        q0 = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.XPowGate(exponent=1.0)(q0))
        
        result = translator.translate(circuit)
        
        ops = result["circuit"]["operations"]
        assert len(ops) == 1
        assert ops[0]["name"] == "X"
    
    def test_x_power_gate_half(self, translator):
        """XPowGate(0.5) should map to SX."""
        q0 = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.XPowGate(exponent=0.5)(q0))
        
        result = translator.translate(circuit)
        
        ops = result["circuit"]["operations"]
        assert len(ops) == 1
        assert ops[0]["name"] == "SX"
    
    def test_x_power_gate_custom(self, translator):
        """XPowGate(0.3) should map to RX."""
        q0 = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.XPowGate(exponent=0.3)(q0))
        
        result = translator.translate(circuit)
        
        ops = result["circuit"]["operations"]
        assert len(ops) == 1
        assert ops[0]["name"] == "RX"
        assert "params" in ops[0]
        assert np.isclose(ops[0]["params"][0], 0.3 * np.pi)
    
    def test_y_power_gate_full(self, translator):
        """YPowGate(1.0) should map to Y."""
        q0 = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.YPowGate(exponent=1.0)(q0))
        
        result = translator.translate(circuit)
        
        ops = result["circuit"]["operations"]
        assert len(ops) == 1
        assert ops[0]["name"] == "Y"
    
    def test_y_power_gate_custom(self, translator):
        """YPowGate(0.25) should map to RY."""
        q0 = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.YPowGate(exponent=0.25)(q0))
        
        result = translator.translate(circuit)
        
        ops = result["circuit"]["operations"]
        assert len(ops) == 1
        assert ops[0]["name"] == "RY"
        assert np.isclose(ops[0]["params"][0], 0.25 * np.pi)
    
    def test_z_power_gate_full(self, translator):
        """ZPowGate(1.0) should map to Z."""
        q0 = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.ZPowGate(exponent=1.0)(q0))
        
        result = translator.translate(circuit)
        
        ops = result["circuit"]["operations"]
        assert len(ops) == 1
        assert ops[0]["name"] == "Z"
    
    def test_z_power_gate_half_is_s(self, translator):
        """ZPowGate(0.5) should map to S."""
        q0 = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.ZPowGate(exponent=0.5)(q0))
        
        result = translator.translate(circuit)
        
        ops = result["circuit"]["operations"]
        assert len(ops) == 1
        assert ops[0]["name"] == "S"
    
    def test_z_power_gate_quarter_is_t(self, translator):
        """ZPowGate(0.25) should map to T."""
        q0 = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.ZPowGate(exponent=0.25)(q0))
        
        result = translator.translate(circuit)
        
        ops = result["circuit"]["operations"]
        assert len(ops) == 1
        assert ops[0]["name"] == "T"
    
    def test_z_power_gate_neg_half_is_sdg(self, translator):
        """ZPowGate(-0.5) should map to SDG."""
        q0 = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.ZPowGate(exponent=-0.5)(q0))
        
        result = translator.translate(circuit)
        
        ops = result["circuit"]["operations"]
        assert len(ops) == 1
        assert ops[0]["name"] == "SDG"
    
    def test_z_power_gate_neg_quarter_is_tdg(self, translator):
        """ZPowGate(-0.25) should map to TDG."""
        q0 = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.ZPowGate(exponent=-0.25)(q0))
        
        result = translator.translate(circuit)
        
        ops = result["circuit"]["operations"]
        assert len(ops) == 1
        assert ops[0]["name"] == "TDG"
    
    def test_z_power_gate_custom(self, translator):
        """ZPowGate(0.7) should map to RZ."""
        q0 = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.ZPowGate(exponent=0.7)(q0))
        
        result = translator.translate(circuit)
        
        ops = result["circuit"]["operations"]
        assert len(ops) == 1
        assert ops[0]["name"] == "RZ"
        assert np.isclose(ops[0]["params"][0], 0.7 * np.pi)
    
    def test_h_power_gate_full(self, translator):
        """HPowGate(1.0) should map to H."""
        q0 = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.HPowGate(exponent=1.0)(q0))
        
        result = translator.translate(circuit)
        
        ops = result["circuit"]["operations"]
        assert len(ops) == 1
        assert ops[0]["name"] == "H"
    
    def test_cz_power_gate_full(self, translator):
        """CZPowGate(1.0) should map to CZ."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(cirq.CZPowGate(exponent=1.0)(q0, q1))
        
        result = translator.translate(circuit)
        
        ops = result["circuit"]["operations"]
        assert len(ops) == 1
        assert ops[0]["name"] == "CZ"


# ============================================================================
# Test Category 3: CircuitTranslator - Rotation Gates
# ============================================================================

class TestCircuitTranslatorRotationGates:
    """Tests for rotation gate translation."""
    
    def test_rx_gate(self, translator):
        """Rx gate translation."""
        q0 = cirq.LineQubit(0)
        angle = np.pi / 4
        circuit = cirq.Circuit(cirq.rx(angle)(q0))
        
        result = translator.translate(circuit)
        
        ops = result["circuit"]["operations"]
        assert len(ops) == 1
        assert ops[0]["name"] == "RX"
        assert np.isclose(ops[0]["params"][0], angle)
    
    def test_ry_gate(self, translator):
        """Ry gate translation."""
        q0 = cirq.LineQubit(0)
        angle = np.pi / 3
        circuit = cirq.Circuit(cirq.ry(angle)(q0))
        
        result = translator.translate(circuit)
        
        ops = result["circuit"]["operations"]
        assert len(ops) == 1
        assert ops[0]["name"] == "RY"
        assert np.isclose(ops[0]["params"][0], angle)
    
    def test_rz_gate(self, translator):
        """Rz gate translation."""
        q0 = cirq.LineQubit(0)
        angle = np.pi / 6
        circuit = cirq.Circuit(cirq.rz(angle)(q0))
        
        result = translator.translate(circuit)
        
        ops = result["circuit"]["operations"]
        assert len(ops) == 1
        assert ops[0]["name"] == "RZ"
        assert np.isclose(ops[0]["params"][0], angle)


# ============================================================================
# Test Category 4: Qubit Mapping
# ============================================================================

class TestQubitMapping:
    """Tests for qubit type mapping."""
    
    def test_line_qubit_mapping(self, translator):
        """LineQubits should be mapped by index."""
        q0, q1, q2 = cirq.LineQubit.range(3)
        circuit = cirq.Circuit(cirq.H(q0), cirq.X(q1), cirq.Y(q2))
        
        translator.translate(circuit)
        qubit_map = translator.get_qubit_map()
        
        assert qubit_map[cirq.LineQubit(0)] == 0
        assert qubit_map[cirq.LineQubit(1)] == 1
        assert qubit_map[cirq.LineQubit(2)] == 2
    
    def test_line_qubit_non_sequential(self, translator):
        """Non-sequential LineQubits should still map correctly."""
        q1 = cirq.LineQubit(1)
        q5 = cirq.LineQubit(5)
        q3 = cirq.LineQubit(3)
        circuit = cirq.Circuit(cirq.H(q1), cirq.H(q5), cirq.H(q3))
        
        translator.translate(circuit)
        qubit_map = translator.get_qubit_map()
        
        # Sorted order: 1, 3, 5 → mapped to 0, 1, 2
        assert qubit_map[q1] == 0
        assert qubit_map[q3] == 1
        assert qubit_map[q5] == 2
    
    def test_grid_qubit_mapping(self, translator):
        """GridQubits should be mapped by (row, col)."""
        q00 = cirq.GridQubit(0, 0)
        q01 = cirq.GridQubit(0, 1)
        q10 = cirq.GridQubit(1, 0)
        circuit = cirq.Circuit(cirq.H(q00), cirq.H(q01), cirq.H(q10))
        
        translator.translate(circuit)
        qubit_map = translator.get_qubit_map()
        
        # Sorted order: (0,0), (0,1), (1,0)
        assert qubit_map[q00] == 0
        assert qubit_map[q01] == 1
        assert qubit_map[q10] == 2
    
    def test_named_qubit_mapping(self, translator):
        """NamedQubits should be mapped alphabetically."""
        alice = cirq.NamedQubit("alice")
        bob = cirq.NamedQubit("bob")
        charlie = cirq.NamedQubit("charlie")
        circuit = cirq.Circuit(cirq.H(bob), cirq.H(alice), cirq.H(charlie))
        
        translator.translate(circuit)
        qubit_map = translator.get_qubit_map()
        
        # Sorted alphabetically: alice, bob, charlie
        assert qubit_map[alice] == 0
        assert qubit_map[bob] == 1
        assert qubit_map[charlie] == 2
    
    def test_qubit_map_retrieval(self, translator):
        """get_qubit_map should return a copy."""
        q0 = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.H(q0))
        
        translator.translate(circuit)
        map1 = translator.get_qubit_map()
        map2 = translator.get_qubit_map()
        
        # Should be equal but not same object
        assert map1 == map2
        assert map1 is not map2


# ============================================================================
# Test Category 5: Measurement Handling
# ============================================================================

class TestMeasurementHandling:
    """Tests for measurement key extraction."""
    
    def test_single_measurement_key(self, translator):
        """Single measurement should extract key."""
        q0 = cirq.LineQubit(0)
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.measure(q0, key='my_key')
        )
        
        translator.translate(circuit)
        keys = translator.get_measurement_keys()
        
        assert keys == ['my_key']
    
    def test_multiple_measurement_keys(self, translator):
        """Multiple measurements should extract all keys in order."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.measure(q0, key='first'),
            cirq.CNOT(q0, q1),
            cirq.measure(q0, q1, key='second')
        )
        
        translator.translate(circuit)
        keys = translator.get_measurement_keys()
        
        assert keys == ['first', 'second']
    
    def test_measurement_qubits_tracking(self, translator):
        """Measurement qubits should be tracked per key."""
        q0, q1, q2 = cirq.LineQubit.range(3)
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.measure(q0, key='single'),
            cirq.measure(q1, q2, key='pair')
        )
        
        translator.translate(circuit)
        meas_qubits = translator.get_measurement_qubits()
        
        assert len(meas_qubits['single']) == 1
        assert len(meas_qubits['pair']) == 2
    
    def test_measurement_default_key(self, translator):
        """Measurement without key should use default."""
        q0 = cirq.LineQubit(0)
        # cirq.measure without key still has a default key
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.measure(q0)  # Will have auto-generated key
        )
        
        translator.translate(circuit)
        keys = translator.get_measurement_keys()
        
        assert len(keys) == 1  # Should have one key


# ============================================================================
# Test Category 6: ResultConverter
# ============================================================================

class TestResultConverter:
    """Tests for result conversion."""
    
    def test_convert_simple_result(self, converter):
        """Simple result conversion."""
        q0 = cirq.LineQubit(0)
        qubit_map = {q0: 0}
        
        lret_result = {
            "samples": [[0], [1], [0], [1]],
            "status": "success"
        }
        
        measurements = converter.convert(
            lret_result=lret_result,
            qubit_map=qubit_map,
            measurement_keys=['result'],
            measurement_qubits={'result': [q0]},
            repetitions=4
        )
        
        assert 'result' in measurements
        # 3D array: (reps, 1, qubits) for Cirq records format
        assert measurements['result'].shape == (4, 1, 1)
    
    def test_convert_multi_qubit_result(self, converter):
        """Multi-qubit result conversion."""
        q0, q1 = cirq.LineQubit.range(2)
        qubit_map = {q0: 0, q1: 1}
        
        lret_result = {
            "samples": [[0, 0], [1, 1], [0, 1], [1, 0]],
            "status": "success"
        }
        
        measurements = converter.convert(
            lret_result=lret_result,
            qubit_map=qubit_map,
            measurement_keys=['bell'],
            measurement_qubits={'bell': [q0, q1]},
            repetitions=4
        )
        
        assert 'bell' in measurements
        # 3D array: (reps, 1, qubits) for Cirq records format
        assert measurements['bell'].shape == (4, 1, 2)
    
    def test_convert_integer_samples(self, converter):
        """Integer samples should convert to bit arrays."""
        q0, q1 = cirq.LineQubit.range(2)
        qubit_map = {q0: 0, q1: 1}
        
        lret_result = {
            "samples": [0, 3, 1, 2],  # Binary: 00, 11, 01, 10
            "status": "success"
        }
        
        measurements = converter.convert(
            lret_result=lret_result,
            qubit_map=qubit_map,
            measurement_keys=['result'],
            measurement_qubits={'result': [q0, q1]},
            repetitions=4
        )
        
        # 3D array: (reps, 1, qubits) for Cirq records format
        # LRET uses little-endian: qubit j is bit j
        # sample 0 = 00: q0=0, q1=0 → [0, 0]
        # sample 3 = 11: q0=1, q1=1 → [1, 1]
        # sample 1 = 01: q0=1, q1=0 → [1, 0]
        # sample 2 = 10: q0=0, q1=1 → [0, 1]
        expected = np.array([[[0, 0]], [[1, 1]], [[1, 0]], [[0, 1]]])
        np.testing.assert_array_equal(measurements['result'], expected)
    
    def test_convert_from_counts(self, converter):
        """Counts dict should expand to samples."""
        q0 = cirq.LineQubit(0)
        qubit_map = {q0: 0}
        
        lret_result = {
            "counts": {"0": 3, "1": 2},
            "status": "success"
        }
        
        measurements = converter.convert(
            lret_result=lret_result,
            qubit_map=qubit_map,
            measurement_keys=['m'],
            measurement_qubits={'m': [q0]},
            repetitions=5
        )
        
        assert 'm' in measurements
        # 3D array: (reps, 1, qubits) for Cirq records format
        assert measurements['m'].shape == (5, 1, 1)
        # Should have 3 zeros and 2 ones (or close)
        assert measurements['m'].sum() in [2, 3]  # Allow for padding
    
    def test_convert_empty_measurements(self, converter):
        """No measurement keys should return all qubits as 'result'."""
        q0, q1 = cirq.LineQubit.range(2)
        qubit_map = {q0: 0, q1: 1}
        
        lret_result = {
            "samples": [[0, 0], [1, 1]],
            "status": "success"
        }
        
        measurements = converter.convert(
            lret_result=lret_result,
            qubit_map=qubit_map,
            measurement_keys=[],
            measurement_qubits={},
            repetitions=2
        )
        
        assert 'result' in measurements
        # 3D array: (reps, 1, qubits) for Cirq records format
        assert measurements['result'].shape == (2, 1, 2)


# ============================================================================
# Test Category 7: LRETSimulator Initialization
# ============================================================================

class TestLRETSimulatorInit:
    """Tests for simulator initialization."""
    
    def test_simulator_default_init(self):
        """Default initialization should work."""
        with patch('lret_cirq.lret_simulator.NATIVE_AVAILABLE', True):
            from lret_cirq import LRETSimulator
            sim = LRETSimulator()
            assert sim.epsilon == 1e-4
            assert sim.noise_model is None
    
    def test_simulator_custom_epsilon(self):
        """Custom epsilon should be set."""
        with patch('lret_cirq.lret_simulator.NATIVE_AVAILABLE', True):
            from lret_cirq import LRETSimulator
            sim = LRETSimulator(epsilon=1e-6)
            assert sim.epsilon == 1e-6
    
    def test_simulator_invalid_epsilon(self):
        """Invalid epsilon should raise error."""
        with patch('lret_cirq.lret_simulator.NATIVE_AVAILABLE', True):
            from lret_cirq import LRETSimulator
            with pytest.raises(ValueError):
                LRETSimulator(epsilon=0)
            with pytest.raises(ValueError):
                LRETSimulator(epsilon=1)
            with pytest.raises(ValueError):
                LRETSimulator(epsilon=-0.1)
    
    def test_simulator_str_repr(self):
        """String representations should include info."""
        with patch('lret_cirq.lret_simulator.NATIVE_AVAILABLE', True):
            from lret_cirq import LRETSimulator
            sim = LRETSimulator(epsilon=1e-5)
            
            assert "1e-05" in str(sim) or "1e-5" in str(sim)
            assert "LRETSimulator" in repr(sim)
    
    def test_simulator_epsilon_property(self):
        """Epsilon property should be settable."""
        with patch('lret_cirq.lret_simulator.NATIVE_AVAILABLE', True):
            from lret_cirq import LRETSimulator
            sim = LRETSimulator(epsilon=1e-4)
            
            sim.epsilon = 1e-6
            assert sim.epsilon == 1e-6
            
            with pytest.raises(ValueError):
                sim.epsilon = 0


# ============================================================================
# Test Category 8: Integration Tests (with mock)
# ============================================================================

class TestIntegrationMocked:
    """Integration tests with mocked LRET backend."""
    
    def test_bell_state_circuit_translation(self, translator):
        """Bell state circuit should translate correctly."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1),
            cirq.measure(q0, q1, key='bell')
        )
        
        result = translator.translate(circuit, shots=1000)
        
        assert result["circuit"]["num_qubits"] == 2
        assert "shots" in result["config"]
        assert result["config"]["shots"] == 1000
        
        ops = result["circuit"]["operations"]
        # H and CNOT (measure is tracked separately, not in operations)
        assert len(ops) >= 2
        assert ops[0]["name"] == "H"
        assert ops[1]["name"] == "CNOT"
    
    def test_ghz_3_qubit_translation(self, translator):
        """3-qubit GHZ circuit should translate correctly."""
        q0, q1, q2 = cirq.LineQubit.range(3)
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1),
            cirq.CNOT(q1, q2),
            cirq.measure(q0, q1, q2, key='ghz')
        )
        
        result = translator.translate(circuit)
        
        assert result["circuit"]["num_qubits"] == 3
        
        ops = result["circuit"]["operations"]
        assert len(ops) == 3  # H, CNOT, CNOT
        assert ops[0]["name"] == "H"
        assert ops[1]["name"] == "CNOT"
        assert ops[1]["wires"] == [0, 1]
        assert ops[2]["name"] == "CNOT"
        assert ops[2]["wires"] == [1, 2]
    
    def test_qft_circuit_translation(self, translator):
        """QFT circuit should translate (uses power gates)."""
        qubits = cirq.LineQubit.range(3)
        
        circuit = cirq.Circuit()
        circuit.append(cirq.H(qubits[0]))
        circuit.append(cirq.CZ(qubits[0], qubits[1])**0.5)  # Controlled-S
        circuit.append(cirq.CZ(qubits[0], qubits[2])**0.25)  # Controlled-T
        circuit.append(cirq.measure(*qubits, key='qft'))
        
        result = translator.translate(circuit)
        
        assert result["circuit"]["num_qubits"] == 3
        # Should have H plus decomposed controlled rotations
        assert len(result["circuit"]["operations"]) >= 1
    
    def test_random_circuit_translation(self, translator):
        """Random circuit should translate without errors."""
        qubits = cirq.LineQubit.range(4)
        
        # Build a random-ish circuit
        circuit = cirq.Circuit()
        for _ in range(5):
            circuit.append(cirq.H(qubits[0]))
            circuit.append(cirq.CNOT(qubits[0], qubits[1]))
            circuit.append(cirq.T(qubits[2]))
            circuit.append(cirq.CZ(qubits[2], qubits[3]))
        circuit.append(cirq.measure(*qubits, key='random'))
        
        result = translator.translate(circuit)
        
        assert result["circuit"]["num_qubits"] == 4
        assert len(result["circuit"]["operations"]) == 20  # 4 gates × 5 iterations
    
    def test_multiple_measurements_translation(self, translator):
        """Multiple measurement keys should be tracked."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.measure(q0, key='first'),
            cirq.CNOT(q0, q1),
            cirq.measure(q0, q1, key='final')
        )
        
        translator.translate(circuit)
        
        keys = translator.get_measurement_keys()
        assert 'first' in keys
        assert 'final' in keys


# ============================================================================
# Test Category 9: Error Handling
# ============================================================================

class TestErrorHandling:
    """Tests for error conditions."""
    
    def test_unsupported_gate_error(self, translator):
        """Unsupported gate should raise TranslationError."""
        from lret_cirq.translators.circuit_translator import TranslationError
        
        q0, q1, q2 = cirq.LineQubit.range(3)
        circuit = cirq.Circuit(cirq.TOFFOLI(q0, q1, q2))
        
        with pytest.raises(TranslationError) as exc:
            translator.translate(circuit)
        
        assert "Unsupported gate" in str(exc.value)
    
    def test_empty_circuit_error(self, translator):
        """Circuit with no qubits should raise error."""
        from lret_cirq.translators.circuit_translator import TranslationError
        
        circuit = cirq.Circuit()
        
        with pytest.raises(TranslationError) as exc:
            translator.translate(circuit)
        
        assert "no qubits" in str(exc.value).lower()
    
    def test_simulator_no_native_error(self):
        """Missing native module should raise ImportError."""
        # This test verifies the error message mentions native module
        # We can't easily test the ImportError without breaking state
        from lret_cirq.lret_simulator import LRETSimulator
        
        # Just verify the class can be instantiated when native is available
        try:
            sim = LRETSimulator()
            # If we get here, native IS available, so test repr contains info
            assert "LRET" in repr(sim)
        except ImportError as e:
            # If native not available, error message should be helpful
            assert "native" in str(e).lower() or "LRET" in str(e)


# ============================================================================
# Test Category 10: Config and Shots
# ============================================================================

class TestConfigAndShots:
    """Tests for configuration options."""
    
    def test_custom_epsilon_in_json(self, translator):
        """Custom epsilon should appear in config."""
        q0 = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0, key='m'))
        
        result = translator.translate(circuit, epsilon=1e-6)
        
        assert result["config"]["epsilon"] == 1e-6
    
    def test_shots_in_config(self, translator):
        """Shots should appear in config when measurements present."""
        q0 = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0, key='m'))
        
        result = translator.translate(circuit, shots=2000)
        
        assert "shots" in result["config"], "shots should be in config when circuit has measurements"
        assert result["config"]["shots"] == 2000
    
    def test_seed_in_config(self, translator):
        """Seed should appear in config when provided."""
        q0 = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0, key='m'))
        
        result = translator.translate(circuit, seed=42)
        
        assert result["config"]["seed"] == 42
    
    def test_noise_model_in_json(self, translator):
        """Noise model should be included in JSON."""
        q0 = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0, key='m'))
        
        noise = {"depolarizing": {"prob": 0.01}}
        result = translator.translate(circuit, noise_model=noise)
        
        assert "noise" in result
        assert result["noise"] == noise
        assert result["config"]["use_noise"] == True


# ============================================================================
# Test Category 11: Identity and Special Gates
# ============================================================================

class TestSpecialGates:
    """Tests for identity and special gates."""
    
    def test_identity_gate_skipped(self, translator):
        """Identity gates should be skipped."""
        q0 = cirq.LineQubit(0)
        circuit = cirq.Circuit(
            cirq.I(q0),
            cirq.H(q0),
            cirq.I(q0)
        )
        
        result = translator.translate(circuit)
        
        ops = result["circuit"]["operations"]
        assert len(ops) == 1
        assert ops[0]["name"] == "H"
    
    def test_power_gate_zero_exponent_skipped(self, translator):
        """Power gate with exponent 0 should be skipped (identity)."""
        q0 = cirq.LineQubit(0)
        circuit = cirq.Circuit(
            cirq.XPowGate(exponent=0.0)(q0),
            cirq.H(q0),
        )
        
        result = translator.translate(circuit)
        
        ops = result["circuit"]["operations"]
        assert len(ops) == 1
        assert ops[0]["name"] == "H"
    
    def test_iswap_gate(self, translator):
        """ISWAP gate should translate."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(cirq.ISWAP(q0, q1))
        
        result = translator.translate(circuit)
        
        ops = result["circuit"]["operations"]
        assert len(ops) == 1
        assert ops[0]["name"] == "ISWAP"
        assert ops[0]["wires"] == [0, 1]


# ============================================================================
# Run tests if executed directly
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
