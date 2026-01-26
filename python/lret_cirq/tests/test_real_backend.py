"""End-to-end integration tests with real LRET backend.

These tests verify the complete pipeline from Cirq circuit to LRET
simulation and back to Cirq results.

Requires: LRET native module built and installed
Run: pytest lret_cirq/tests/test_real_backend.py -v
"""

import numpy as np
import pytest
import sys

try:
    import cirq
except ImportError:
    pytest.skip("Cirq not available", allow_module_level=True)

# Add parent path for imports
sys.path.insert(0, '.')

# Check if LRET native module is available
try:
    from qlret.api import simulate_json
    NATIVE_AVAILABLE = True
except ImportError:
    NATIVE_AVAILABLE = False


# Skip all tests if native module not available
pytestmark = pytest.mark.skipif(
    not NATIVE_AVAILABLE,
    reason="LRET native module not available"
)


class TestRealBackendBellState:
    """Bell state tests with real LRET backend."""
    
    def test_bell_state_histogram(self):
        """Bell state should produce ~50/50 split of |00⟩ and |11⟩."""
        from lret_cirq import LRETSimulator
        
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1),
            cirq.measure(q0, q1, key='result')
        )
        
        sim = LRETSimulator(epsilon=1e-4)
        result = sim.run(circuit, repetitions=1000)
        
        counts = result.histogram(key='result')
        
        # Should only have |00⟩ (0) and |11⟩ (3) outcomes
        assert set(counts.keys()).issubset({0, 3})
        
        # Should be roughly 50/50 (allow 10% tolerance)
        total = sum(counts.values())
        for outcome, count in counts.items():
            ratio = count / total
            assert 0.4 <= ratio <= 0.6, f"Outcome {outcome}: {ratio:.2%} is not ~50%"
    
    def test_bell_state_measurements_array(self):
        """Bell state measurements should be correlated."""
        from lret_cirq import LRETSimulator
        
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1),
            cirq.measure(q0, q1, key='bell')
        )
        
        sim = LRETSimulator(epsilon=1e-4)
        result = sim.run(circuit, repetitions=100)
        
        # Get raw measurements
        measurements = result.measurements['bell']
        
        # Each row should have both qubits equal (|00⟩ or |11⟩)
        for row in measurements:
            assert row[0] == row[1], f"Qubits not correlated: {row}"


class TestRealBackendGHZState:
    """GHZ state tests with real LRET backend."""
    
    def test_ghz_3_qubit(self):
        """3-qubit GHZ state should produce |000⟩ and |111⟩."""
        from lret_cirq import LRETSimulator
        
        qubits = cirq.LineQubit.range(3)
        circuit = cirq.Circuit(
            cirq.H(qubits[0]),
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.CNOT(qubits[1], qubits[2]),
            cirq.measure(*qubits, key='ghz')
        )
        
        sim = LRETSimulator(epsilon=1e-4)
        result = sim.run(circuit, repetitions=1000)
        
        counts = result.histogram(key='ghz')
        
        # Should only have |000⟩ (0) and |111⟩ (7) outcomes
        assert set(counts.keys()).issubset({0, 7})
        
        # Should be roughly 50/50
        total = sum(counts.values())
        for count in counts.values():
            assert 0.4 <= count/total <= 0.6
    
    def test_ghz_4_qubit(self):
        """4-qubit GHZ state should produce |0000⟩ and |1111⟩."""
        from lret_cirq import LRETSimulator
        
        qubits = cirq.LineQubit.range(4)
        circuit = cirq.Circuit(
            cirq.H(qubits[0]),
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.CNOT(qubits[1], qubits[2]),
            cirq.CNOT(qubits[2], qubits[3]),
            cirq.measure(*qubits, key='ghz4')
        )
        
        sim = LRETSimulator(epsilon=1e-4)
        result = sim.run(circuit, repetitions=1000)
        
        counts = result.histogram(key='ghz4')
        
        # Should only have |0000⟩ (0) and |1111⟩ (15) outcomes
        assert set(counts.keys()).issubset({0, 15})


class TestRealBackendQuantumGates:
    """Individual gate tests with real LRET backend."""
    
    def test_hadamard_superposition(self):
        """H gate should create 50/50 superposition."""
        from lret_cirq import LRETSimulator
        
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(
            cirq.H(q),
            cirq.measure(q, key='m')
        )
        
        sim = LRETSimulator(epsilon=1e-4)
        result = sim.run(circuit, repetitions=1000)
        
        counts = result.histogram(key='m')
        
        # Should be roughly 50/50
        total = sum(counts.values())
        assert 0.4 <= counts.get(0, 0)/total <= 0.6
        assert 0.4 <= counts.get(1, 0)/total <= 0.6
    
    def test_pauli_x_flip(self):
        """X gate should flip |0⟩ to |1⟩."""
        from lret_cirq import LRETSimulator
        
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(
            cirq.X(q),
            cirq.measure(q, key='m')
        )
        
        sim = LRETSimulator(epsilon=1e-4)
        result = sim.run(circuit, repetitions=100)
        
        counts = result.histogram(key='m')
        
        # Should always be |1⟩
        assert counts.get(1, 0) == 100
        assert counts.get(0, 0) == 0
    
    def test_double_x_identity(self):
        """Two X gates should return to |0⟩."""
        from lret_cirq import LRETSimulator
        
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(
            cirq.X(q),
            cirq.X(q),
            cirq.measure(q, key='m')
        )
        
        sim = LRETSimulator(epsilon=1e-4)
        result = sim.run(circuit, repetitions=100)
        
        counts = result.histogram(key='m')
        
        # Should always be |0⟩
        assert counts.get(0, 0) == 100
        assert counts.get(1, 0) == 0
    
    def test_z_phase_after_h(self):
        """Z gate after H should change phase but not affect measurement probabilities."""
        from lret_cirq import LRETSimulator
        
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(
            cirq.H(q),
            cirq.Z(q),
            cirq.measure(q, key='m')
        )
        
        sim = LRETSimulator(epsilon=1e-4)
        result = sim.run(circuit, repetitions=1000)
        
        counts = result.histogram(key='m')
        
        # Should still be roughly 50/50 (Z only affects phase)
        total = sum(counts.values())
        assert 0.4 <= counts.get(0, 0)/total <= 0.6
    
    def test_t_gate_phase(self):
        """T gate should apply π/4 phase."""
        from lret_cirq import LRETSimulator
        
        q = cirq.LineQubit(0)
        # H -> T -> H should produce specific interference pattern
        circuit = cirq.Circuit(
            cirq.H(q),
            cirq.T(q),
            cirq.H(q),
            cirq.measure(q, key='m')
        )
        
        sim = LRETSimulator(epsilon=1e-4)
        result = sim.run(circuit, repetitions=1000)
        
        counts = result.histogram(key='m')
        
        # cos²(π/8) ≈ 0.854, sin²(π/8) ≈ 0.146
        total = sum(counts.values())
        p0 = counts.get(0, 0)/total
        assert 0.75 <= p0 <= 0.95, f"P(0) = {p0:.3f}, expected ~0.85"


class TestRealBackendRotationGates:
    """Rotation gate tests with real LRET backend."""
    
    def test_rx_pi_half(self):
        """RX(π/2) should create 50/50 superposition."""
        from lret_cirq import LRETSimulator
        
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(
            cirq.rx(np.pi/2)(q),
            cirq.measure(q, key='m')
        )
        
        sim = LRETSimulator(epsilon=1e-4)
        result = sim.run(circuit, repetitions=1000)
        
        counts = result.histogram(key='m')
        total = sum(counts.values())
        
        # Should be roughly 50/50
        assert 0.4 <= counts.get(0, 0)/total <= 0.6
    
    def test_rx_pi(self):
        """RX(π) should flip |0⟩ to |1⟩."""
        from lret_cirq import LRETSimulator
        
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(
            cirq.rx(np.pi)(q),
            cirq.measure(q, key='m')
        )
        
        sim = LRETSimulator(epsilon=1e-4)
        result = sim.run(circuit, repetitions=100)
        
        counts = result.histogram(key='m')
        
        # Should always be |1⟩
        assert counts.get(1, 0) == 100
    
    def test_ry_creates_superposition(self):
        """RY(θ) should create cos²(θ/2)|0⟩ + sin²(θ/2)|1⟩."""
        from lret_cirq import LRETSimulator
        
        q = cirq.LineQubit(0)
        theta = np.pi / 3  # 60 degrees
        circuit = cirq.Circuit(
            cirq.ry(theta)(q),
            cirq.measure(q, key='m')
        )
        
        sim = LRETSimulator(epsilon=1e-4)
        result = sim.run(circuit, repetitions=1000)
        
        counts = result.histogram(key='m')
        total = sum(counts.values())
        
        # P(0) = cos²(θ/2) = cos²(π/6) ≈ 0.75
        # P(1) = sin²(θ/2) = sin²(π/6) ≈ 0.25
        p0 = counts.get(0, 0)/total
        assert 0.65 <= p0 <= 0.85, f"P(0) = {p0:.3f}, expected ~0.75"


class TestRealBackendMultipleQubits:
    """Multi-qubit circuit tests."""
    
    def test_separate_hadamards(self):
        """Independent H gates should create uncorrelated superpositions."""
        from lret_cirq import LRETSimulator
        
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.H(q1),
            cirq.measure(q0, q1, key='m')
        )
        
        sim = LRETSimulator(epsilon=1e-4)
        result = sim.run(circuit, repetitions=1000)
        
        counts = result.histogram(key='m')
        
        # Should have all 4 outcomes with ~25% each
        total = sum(counts.values())
        for outcome in [0, 1, 2, 3]:  # 00, 01, 10, 11
            p = counts.get(outcome, 0)/total
            assert 0.15 <= p <= 0.35, f"P({outcome}) = {p:.3f}, expected ~0.25"
    
    def test_swap_gate(self):
        """SWAP gate should exchange qubit states."""
        from lret_cirq import LRETSimulator
        
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.X(q0),  # |10⟩ (q0=1, q1=0)
            cirq.SWAP(q0, q1),  # |01⟩ (q0=0, q1=1)
            cirq.measure(q0, q1, key='m')
        )
        
        sim = LRETSimulator(epsilon=1e-4)
        result = sim.run(circuit, repetitions=100)
        
        counts = result.histogram(key='m')
        
        # After SWAP: q0=0, q1=1 → histogram value = 0*2 + 1*1 = 1
        assert counts.get(1, 0) == 100


class TestRealBackendMultipleMeasurements:
    """Tests for circuits with multiple measurement keys."""
    
    def test_separate_measurement_keys(self):
        """Different measurement keys should track independently."""
        from lret_cirq import LRETSimulator
        
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.X(q1),
            cirq.measure(q0, key='first'),
            cirq.measure(q1, key='second')
        )
        
        sim = LRETSimulator(epsilon=1e-4)
        result = sim.run(circuit, repetitions=100)
        
        # First qubit should be 50/50
        first_counts = result.histogram(key='first')
        assert first_counts.get(0, 0) > 0
        assert first_counts.get(1, 0) > 0
        
        # Second qubit should always be 1
        second_counts = result.histogram(key='second')
        assert second_counts.get(1, 0) == 100


class TestRealBackendParameterized:
    """Tests for parameterized circuits."""
    
    def test_parameter_sweep(self):
        """Parameter sweep should work correctly."""
        from lret_cirq import LRETSimulator
        import sympy
        
        q = cirq.LineQubit(0)
        theta = sympy.Symbol('theta')
        circuit = cirq.Circuit(
            cirq.ry(theta)(q),
            cirq.measure(q, key='m')
        )
        
        sim = LRETSimulator(epsilon=1e-4)
        
        # Test at θ = 0 (should always be |0⟩)
        result = sim.run(circuit, repetitions=100, param_resolver={'theta': 0})
        counts = result.histogram(key='m')
        assert counts.get(0, 0) == 100
        
        # Test at θ = π (should always be |1⟩)
        result = sim.run(circuit, repetitions=100, param_resolver={'theta': np.pi})
        counts = result.histogram(key='m')
        assert counts.get(1, 0) == 100


class TestRealBackendQubitTypes:
    """Tests for different Cirq qubit types."""
    
    def test_grid_qubits(self):
        """GridQubits should work correctly."""
        from lret_cirq import LRETSimulator
        
        q00 = cirq.GridQubit(0, 0)
        q01 = cirq.GridQubit(0, 1)
        
        circuit = cirq.Circuit(
            cirq.H(q00),
            cirq.CNOT(q00, q01),
            cirq.measure(q00, q01, key='grid')
        )
        
        sim = LRETSimulator(epsilon=1e-4)
        result = sim.run(circuit, repetitions=100)
        
        counts = result.histogram(key='grid')
        assert set(counts.keys()).issubset({0, 3})
    
    def test_named_qubits(self):
        """NamedQubits should work correctly."""
        from lret_cirq import LRETSimulator
        
        alice = cirq.NamedQubit('alice')
        bob = cirq.NamedQubit('bob')
        
        circuit = cirq.Circuit(
            cirq.H(alice),
            cirq.CNOT(alice, bob),
            cirq.measure(alice, bob, key='ab')
        )
        
        sim = LRETSimulator(epsilon=1e-4)
        result = sim.run(circuit, repetitions=100)
        
        counts = result.histogram(key='ab')
        assert set(counts.keys()).issubset({0, 3})


class TestRealBackendPerformance:
    """Performance and scaling tests."""
    
    @pytest.mark.slow
    def test_5_qubit_ghz(self):
        """5-qubit GHZ state."""
        from lret_cirq import LRETSimulator
        
        qubits = cirq.LineQubit.range(5)
        circuit = cirq.Circuit(
            cirq.H(qubits[0]),
            *[cirq.CNOT(qubits[i], qubits[i+1]) for i in range(4)],
            cirq.measure(*qubits, key='ghz5')
        )
        
        sim = LRETSimulator(epsilon=1e-4)
        result = sim.run(circuit, repetitions=1000)
        
        counts = result.histogram(key='ghz5')
        
        # Should only have |00000⟩ (0) and |11111⟩ (31)
        assert set(counts.keys()).issubset({0, 31})
    
    @pytest.mark.slow
    def test_multiple_repetitions(self):
        """High repetition count should work."""
        from lret_cirq import LRETSimulator
        
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(
            cirq.H(q),
            cirq.measure(q, key='m')
        )
        
        sim = LRETSimulator(epsilon=1e-4)
        result = sim.run(circuit, repetitions=10000)
        
        counts = result.histogram(key='m')
        total = sum(counts.values())
        
        # With 10000 shots, should be very close to 50/50
        assert 0.48 <= counts.get(0, 0)/total <= 0.52


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
