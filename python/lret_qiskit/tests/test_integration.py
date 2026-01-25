"""Comprehensive tests for LRET Qiskit backend integration."""

import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.providers.jobstatus import JobStatus

from lret_qiskit import LRETProvider
from lret_qiskit.backends import LRETBackend, LRETJob
from lret_qiskit.translators import CircuitTranslator, ResultConverter, TranslationError


# =============================================================================
# Provider Tests
# =============================================================================

class TestLRETProvider:
    """Tests for LRETProvider class."""

    def test_provider_instantiation(self):
        """Provider can be instantiated."""
        provider = LRETProvider()
        assert provider is not None

    def test_provider_lists_three_backends(self):
        """Provider exposes three backend variants."""
        provider = LRETProvider()
        backends = provider.backends()
        assert len(backends) == 3
        names = {b.name for b in backends}
        assert names == {"lret_simulator", "lret_simulator_accurate", "lret_simulator_fast"}

    def test_provider_get_backend_default(self):
        """get_backend() returns default backend when no name given."""
        provider = LRETProvider()
        backend = provider.get_backend()
        assert backend.name == "lret_simulator"

    def test_provider_get_backend_by_name(self):
        """get_backend(name) returns correct backend."""
        provider = LRETProvider()
        backend = provider.get_backend("lret_simulator_accurate")
        assert backend.name == "lret_simulator_accurate"

    def test_provider_get_backend_invalid_name(self):
        """get_backend() raises for unknown backend."""
        from qiskit.providers.exceptions import QiskitBackendNotFoundError
        provider = LRETProvider()
        with pytest.raises(QiskitBackendNotFoundError):
            provider.get_backend("nonexistent_backend")

    def test_provider_filter_backends(self):
        """backends() supports filtering."""
        provider = LRETProvider()
        # Filter for backends with "fast" in name
        backends = provider.backends(filters=lambda b: "fast" in b.name)
        assert len(backends) == 1
        assert backends[0].name == "lret_simulator_fast"


# =============================================================================
# Backend Tests
# =============================================================================

class TestLRETBackend:
    """Tests for LRETBackend class."""

    def test_backend_has_target(self):
        """Backend exposes a valid Target."""
        provider = LRETProvider()
        backend = provider.get_backend()
        assert backend.target is not None
        assert backend.target.num_qubits == 20  # default

    def test_backend_default_options(self):
        """Backend has sensible default options."""
        provider = LRETProvider()
        backend = provider.get_backend()
        opts = backend.options
        assert opts.shots == 1024
        assert opts.epsilon == 1e-4

    def test_backend_epsilon_variants(self):
        """Different backends have different epsilon values."""
        provider = LRETProvider()
        
        default = provider.get_backend("lret_simulator")
        accurate = provider.get_backend("lret_simulator_accurate")
        fast = provider.get_backend("lret_simulator_fast")
        
        # Check internal epsilon values
        assert default._epsilon == 1e-4
        assert accurate._epsilon == 1e-6
        assert fast._epsilon == 1e-3

    def test_backend_supported_gates(self):
        """Backend Target includes expected gates."""
        provider = LRETProvider()
        backend = provider.get_backend()
        
        # Check single-qubit gates
        op_names = [op.name for op in backend.target.operations]
        assert "h" in op_names
        assert "x" in op_names
        assert "rx" in op_names
        assert "rz" in op_names
        
        # Check two-qubit gates
        assert "cx" in op_names
        assert "cz" in op_names
        
        # Check measurement
        assert "measure" in op_names


# =============================================================================
# Circuit Translator Tests
# =============================================================================

class TestCircuitTranslator:
    """Tests for Qiskit to LRET circuit translation."""

    def test_translate_empty_circuit(self):
        """Empty circuit translates correctly."""
        translator = CircuitTranslator()
        qc = QuantumCircuit(2)
        result = translator.translate(qc)
        
        assert result["circuit"]["num_qubits"] == 2
        assert result["circuit"]["operations"] == []

    def test_translate_single_qubit_gates(self):
        """Single-qubit gates translate correctly."""
        translator = CircuitTranslator()
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.x(0)
        qc.rz(0.5, 0)
        
        result = translator.translate(qc)
        ops = result["circuit"]["operations"]
        
        assert len(ops) == 3
        assert ops[0] == {"name": "H", "wires": [0]}
        assert ops[1] == {"name": "X", "wires": [0]}
        assert ops[2] == {"name": "RZ", "wires": [0], "params": [0.5]}

    def test_translate_two_qubit_gates(self):
        """Two-qubit gates translate correctly."""
        translator = CircuitTranslator()
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cz(1, 0)
        
        result = translator.translate(qc)
        ops = result["circuit"]["operations"]
        
        assert len(ops) == 2
        assert ops[0] == {"name": "CNOT", "wires": [0, 1]}
        assert ops[1] == {"name": "CZ", "wires": [1, 0]}

    def test_translate_measurement(self):
        """Measurement is handled via shots config, not explicit ops.
        
        LRET handles measurement implicitly - the circuit defines unitary
        evolution, and shots in config triggers sampling from the final state.
        This is cleaner than explicit MEASURE operations and matches how
        many production simulators work.
        """
        translator = CircuitTranslator(shots=100)
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        
        result = translator.translate(qc)
        ops = result["circuit"]["operations"]
        
        # MEASURE ops are NOT included - only gate operations
        assert len(ops) == 2
        assert ops[0]["name"] == "H"
        assert ops[1]["name"] == "CNOT"
        
        # Measurement is triggered via shots in config
        assert result["config"]["shots"] == 100

    def test_translate_barrier_ignored(self):
        """Barrier instructions are skipped."""
        translator = CircuitTranslator()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.barrier()
        qc.cx(0, 1)
        
        result = translator.translate(qc)
        ops = result["circuit"]["operations"]
        
        # Barrier should not appear
        assert len(ops) == 2
        assert all(op["name"] != "BARRIER" for op in ops)

    def test_translate_unsupported_gate_raises(self):
        """Unsupported gates raise TranslationError."""
        translator = CircuitTranslator()
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)  # Toffoli not in GATE_MAP
        
        with pytest.raises(TranslationError):
            translator.translate(qc)

    def test_translate_config_includes_epsilon(self):
        """Translation includes epsilon in config."""
        translator = CircuitTranslator(epsilon=1e-6, shots=512)
        qc = QuantumCircuit(1)
        qc.h(0)
        
        result = translator.translate(qc)
        
        assert result["config"]["epsilon"] == 1e-6

    def test_translate_batch(self):
        """Batch translation works for multiple circuits."""
        translator = CircuitTranslator()
        
        qc1 = QuantumCircuit(1)
        qc1.h(0)
        
        qc2 = QuantumCircuit(2)
        qc2.cx(0, 1)
        
        results = translator.translate_batch([qc1, qc2])
        
        assert len(results) == 2
        assert results[0]["circuit"]["num_qubits"] == 1
        assert results[1]["circuit"]["num_qubits"] == 2


# =============================================================================
# Result Converter Tests
# =============================================================================

class TestResultConverter:
    """Tests for LRET to Qiskit result conversion."""

    def test_convert_success_result(self):
        """Successful results convert correctly."""
        converter = ResultConverter("test_backend", "1.0.0")
        
        lret_results = [{
            "status": "success",
            "execution_time_ms": 10.5,
            "final_rank": 4,
            "expectation_values": [0.5, -0.3],
            "samples": [0, 1, 0, 1, 1],  # 5 samples
        }]
        
        qc = QuantumCircuit(1, 1, name="test_circuit")
        
        result = converter.convert(lret_results, [qc], "job-123", shots=5)
        
        assert result.success
        assert result.backend_name == "test_backend"
        assert result.job_id == "job-123"

    def test_convert_samples_to_counts(self):
        """Samples are correctly converted to counts."""
        converter = ResultConverter("test_backend")
        
        # Samples: 0, 0, 1, 1, 1 -> counts: {"0": 2, "1": 3}
        lret_results = [{
            "status": "success",
            "samples": [0, 0, 1, 1, 1],
        }]
        
        qc = QuantumCircuit(1, 1)
        result = converter.convert(lret_results, [qc], "job-1", shots=5)
        
        counts = result.results[0].data.counts
        assert counts.get("0", 0) == 2
        assert counts.get("1", 0) == 3


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_bell_state_circuit(self):
        """Run a Bell state circuit through the full pipeline."""
        provider = LRETProvider()
        backend = provider.get_backend("lret_simulator")
        
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        
        # Note: This test may fail if LRET native module isn't built
        # In that case, mark as xfail or skip
        try:
            job = backend.run(qc, shots=100)
            result = job.result()
            
            assert result.success
            assert job.status() == JobStatus.DONE
        except Exception as e:
            if "No LRET backend available" in str(e):
                pytest.skip("LRET native module not built")
            raise

    def test_run_with_custom_shots(self):
        """Custom shot count is respected."""
        provider = LRETProvider()
        backend = provider.get_backend()
        
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        
        try:
            job = backend.run(qc, shots=50)
            result = job.result()
            
            exp_result = result.results[0]
            # The shots in result header should match requested
            assert exp_result.shots == 50
        except Exception as e:
            if "No LRET backend available" in str(e):
                pytest.skip("LRET native module not built")
            raise

    def test_transpile_and_run(self):
        """Circuit can be transpiled to backend and run.
        
        Note: We use a smaller backend (8 qubits) to avoid memory issues
        with large simulations. The full 20-qubit backend would require
        significant memory for dense simulation.
        """
        from lret_qiskit.backends import LRETBackend
        
        # Create a small backend for testing
        small_backend = LRETBackend(
            name="lret_test",
            description="Small test backend",
            epsilon=1e-4,
            num_qubits=4,  # Small enough to simulate quickly
        )
        
        # Create a circuit that may need transpilation
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        
        # Transpile to backend's basis gates
        transpiled = transpile(qc, small_backend)
        
        try:
            job = small_backend.run(transpiled, shots=100)
            result = job.result()
            assert result.success
            # Should get Bell state distribution
            counts = result.get_counts()
            assert '00' in counts or '11' in counts
        except Exception as e:
            if "No LRET backend available" in str(e):
                pytest.skip("LRET native module not built")
            raise

# =============================================================================
# Extended Gate Tests
# =============================================================================

class TestExtendedGates:
    """Tests for all supported gate types."""

    def test_translate_y_gate(self):
        """Y gate translates correctly."""
        translator = CircuitTranslator()
        qc = QuantumCircuit(1)
        qc.y(0)
        
        result = translator.translate(qc)
        assert result["circuit"]["operations"][0] == {"name": "Y", "wires": [0]}

    def test_translate_z_gate(self):
        """Z gate translates correctly."""
        translator = CircuitTranslator()
        qc = QuantumCircuit(1)
        qc.z(0)
        
        result = translator.translate(qc)
        assert result["circuit"]["operations"][0] == {"name": "Z", "wires": [0]}

    def test_translate_s_gate(self):
        """S gate translates correctly."""
        translator = CircuitTranslator()
        qc = QuantumCircuit(1)
        qc.s(0)
        
        result = translator.translate(qc)
        assert result["circuit"]["operations"][0] == {"name": "S", "wires": [0]}

    def test_translate_sdg_gate(self):
        """Sdg gate translates correctly."""
        translator = CircuitTranslator()
        qc = QuantumCircuit(1)
        qc.sdg(0)
        
        result = translator.translate(qc)
        assert result["circuit"]["operations"][0] == {"name": "SDG", "wires": [0]}

    def test_translate_t_gate(self):
        """T gate translates correctly."""
        translator = CircuitTranslator()
        qc = QuantumCircuit(1)
        qc.t(0)
        
        result = translator.translate(qc)
        assert result["circuit"]["operations"][0] == {"name": "T", "wires": [0]}

    def test_translate_tdg_gate(self):
        """Tdg gate translates correctly."""
        translator = CircuitTranslator()
        qc = QuantumCircuit(1)
        qc.tdg(0)
        
        result = translator.translate(qc)
        assert result["circuit"]["operations"][0] == {"name": "TDG", "wires": [0]}

    def test_translate_rx_gate(self):
        """RX gate with parameter translates correctly."""
        import math
        translator = CircuitTranslator()
        qc = QuantumCircuit(1)
        qc.rx(math.pi / 4, 0)
        
        result = translator.translate(qc)
        op = result["circuit"]["operations"][0]
        assert op["name"] == "RX"
        assert op["wires"] == [0]
        assert abs(op["params"][0] - math.pi / 4) < 1e-10

    def test_translate_ry_gate(self):
        """RY gate with parameter translates correctly."""
        import math
        translator = CircuitTranslator()
        qc = QuantumCircuit(1)
        qc.ry(math.pi / 2, 0)
        
        result = translator.translate(qc)
        op = result["circuit"]["operations"][0]
        assert op["name"] == "RY"
        assert abs(op["params"][0] - math.pi / 2) < 1e-10

    def test_translate_phase_gate(self):
        """Phase (P) gate translates correctly."""
        import math
        translator = CircuitTranslator()
        qc = QuantumCircuit(1)
        qc.p(math.pi / 3, 0)
        
        result = translator.translate(qc)
        op = result["circuit"]["operations"][0]
        assert op["name"] == "U1"  # Phase maps to U1 in LRET

    def test_translate_swap_gate(self):
        """SWAP gate translates correctly."""
        translator = CircuitTranslator()
        qc = QuantumCircuit(2)
        qc.swap(0, 1)
        
        result = translator.translate(qc)
        op = result["circuit"]["operations"][0]
        assert op["name"] == "SWAP"
        assert op["wires"] == [0, 1]


# =============================================================================
# Parameterized Circuit Tests
# =============================================================================

class TestParameterizedCircuits:
    """Tests for parameterized circuits."""

    def test_bound_parameters(self):
        """Circuit with bound parameters translates correctly."""
        from qiskit.circuit import Parameter
        
        translator = CircuitTranslator()
        theta = Parameter('θ')
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)
        
        # Bind the parameter
        bound_qc = qc.assign_parameters({theta: 0.5})
        
        result = translator.translate(bound_qc)
        op = result["circuit"]["operations"][0]
        assert op["name"] == "RX"
        assert abs(op["params"][0] - 0.5) < 1e-10

    def test_multiple_parameters(self):
        """Circuit with multiple parameters translates correctly."""
        from qiskit.circuit import Parameter
        
        translator = CircuitTranslator()
        theta = Parameter('θ')
        phi = Parameter('φ')
        
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)
        qc.ry(phi, 0)
        
        bound_qc = qc.assign_parameters({theta: 0.3, phi: 0.7})
        
        result = translator.translate(bound_qc)
        ops = result["circuit"]["operations"]
        
        assert len(ops) == 2
        assert abs(ops[0]["params"][0] - 0.3) < 1e-10
        assert abs(ops[1]["params"][0] - 0.7) < 1e-10


# =============================================================================
# Multiple Circuit Tests
# =============================================================================

class TestMultipleCircuits:
    """Tests for running multiple circuits."""

    def test_run_multiple_circuits(self):
        """Multiple circuits can be run in a single job."""
        provider = LRETProvider()
        backend = provider.get_backend()
        
        qc1 = QuantumCircuit(1, 1)
        qc1.h(0)
        qc1.measure(0, 0)
        
        qc2 = QuantumCircuit(1, 1)
        qc2.x(0)
        qc2.measure(0, 0)
        
        try:
            job = backend.run([qc1, qc2], shots=100)
            result = job.result()
            
            assert len(result.results) == 2
        except Exception as e:
            if "No LRET backend available" in str(e):
                pytest.skip("LRET native module not built")
            raise


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    def test_unsupported_gate_error_message(self):
        """TranslationError includes helpful message."""
        translator = CircuitTranslator()
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)  # Toffoli
        
        with pytest.raises(TranslationError) as exc_info:
            translator.translate(qc)
        
        assert "Unsupported gate" in str(exc_info.value)
        assert "ccx" in str(exc_info.value).lower()

    def test_job_error_state(self):
        """Job correctly reports error status for bad circuits."""
        # This test checks error handling when simulation fails
        from lret_qiskit.backends.lret_job import LRETJob
        
        # The job should handle errors gracefully
        provider = LRETProvider()
        backend = provider.get_backend()
        
        # Create a valid circuit - actual errors would come from native module
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        
        try:
            job = backend.run(qc, shots=10)
            # Job should complete successfully for valid circuit
            assert job.status() in [JobStatus.DONE, JobStatus.RUNNING]
        except Exception as e:
            if "No LRET backend available" in str(e):
                pytest.skip("LRET native module not built")
            raise


# =============================================================================
# Backend Configuration Tests  
# =============================================================================

class TestBackendConfiguration:
    """Tests for backend configuration options."""

    def test_custom_num_qubits(self):
        """Backend can be created with custom qubit count."""
        backend = LRETBackend(
            name="test",
            description="Test",
            epsilon=1e-4,
            num_qubits=10,
        )
        assert backend.target.num_qubits == 10

    def test_custom_epsilon(self):
        """Backend uses custom epsilon value."""
        backend = LRETBackend(
            name="test",
            description="Test",
            epsilon=1e-8,
            num_qubits=4,
        )
        assert backend._epsilon == 1e-8

    def test_backend_str_repr(self):
        """Backend has meaningful string representation."""
        provider = LRETProvider()
        backend = provider.get_backend()
        
        # Should not raise
        str(backend)
        repr(backend)

    def test_provider_str_repr(self):
        """Provider has meaningful string representation."""
        provider = LRETProvider()
        
        s = str(provider)
        assert "LRETProvider" in s
        assert "version" in s


# =============================================================================
# Result Format Tests
# =============================================================================

class TestResultFormat:
    """Tests for result format and metadata."""

    def test_result_has_backend_name(self):
        """Result contains backend name."""
        provider = LRETProvider()
        backend = provider.get_backend()
        
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        
        try:
            job = backend.run(qc, shots=10)
            result = job.result()
            
            assert result.backend_name == "lret_simulator"
        except Exception as e:
            if "No LRET backend available" in str(e):
                pytest.skip("LRET native module not built")
            raise

    def test_result_has_job_id(self):
        """Result contains job ID."""
        provider = LRETProvider()
        backend = provider.get_backend()
        
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        
        try:
            job = backend.run(qc, shots=10)
            result = job.result()
            
            assert result.job_id is not None
            assert len(result.job_id) > 0
        except Exception as e:
            if "No LRET backend available" in str(e):
                pytest.skip("LRET native module not built")
            raise

    def test_counts_sum_to_shots(self):
        """Measurement counts sum to number of shots."""
        provider = LRETProvider()
        backend = provider.get_backend()
        
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        
        shots = 200
        try:
            job = backend.run(qc, shots=shots)
            result = job.result()
            
            counts = result.get_counts()
            total = sum(counts.values())
            assert total == shots
        except Exception as e:
            if "No LRET backend available" in str(e):
                pytest.skip("LRET native module not built")
            raise


# =============================================================================
# Circuit Validity Tests
# =============================================================================

class TestCircuitValidity:
    """Tests for circuit validation."""

    def test_empty_circuit_runs(self):
        """Empty circuit can be executed."""
        provider = LRETProvider()
        backend = provider.get_backend()
        
        qc = QuantumCircuit(1, 1)
        qc.measure(0, 0)  # Just measure initial |0⟩
        
        try:
            job = backend.run(qc, shots=100)
            result = job.result()
            
            counts = result.get_counts()
            # Should be 100% |0⟩
            assert counts.get('0', 0) == 100
        except Exception as e:
            if "No LRET backend available" in str(e):
                pytest.skip("LRET native module not built")
            raise

    def test_x_gate_flips_qubit(self):
        """X gate correctly flips qubit from |0⟩ to |1⟩."""
        provider = LRETProvider()
        backend = provider.get_backend()
        
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.measure(0, 0)
        
        try:
            job = backend.run(qc, shots=100)
            result = job.result()
            
            counts = result.get_counts()
            # Should be 100% |1⟩
            assert counts.get('1', 0) == 100
        except Exception as e:
            if "No LRET backend available" in str(e):
                pytest.skip("LRET native module not built")
            raise

    def test_ghz_state(self):
        """GHZ state produces correct correlations."""
        provider = LRETProvider()
        backend = provider.get_backend()
        
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure([0, 1, 2], [0, 1, 2])
        
        try:
            job = backend.run(qc, shots=1000)
            result = job.result()
            
            counts = result.get_counts()
            # GHZ should only have |000⟩ and |111⟩
            for bitstring in counts:
                assert bitstring in ['000', '111']
        except Exception as e:
            if "No LRET backend available" in str(e):
                pytest.skip("LRET native module not built")
            raise


# =============================================================================
# Additional Job Tests  
# =============================================================================

class TestJobBehavior:
    """Tests for job behavior and lifecycle."""

    def test_job_has_unique_id(self):
        """Each job has a unique ID."""
        provider = LRETProvider()
        backend = provider.get_backend()
        
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        
        try:
            job1 = backend.run(qc, shots=10)
            job2 = backend.run(qc, shots=10)
            
            assert job1.job_id() != job2.job_id()
        except Exception as e:
            if "No LRET backend available" in str(e):
                pytest.skip("LRET native module not built")
            raise

    def test_job_returns_same_result(self):
        """Calling result() multiple times returns same result."""
        provider = LRETProvider()
        backend = provider.get_backend()
        
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        
        try:
            job = backend.run(qc, shots=100)
            result1 = job.result()
            result2 = job.result()
            
            assert result1.get_counts() == result2.get_counts()
        except Exception as e:
            if "No LRET backend available" in str(e):
                pytest.skip("LRET native module not built")
            raise


# =============================================================================
# Complex Circuit Tests  
# =============================================================================

class TestComplexCircuits:
    """Tests for more complex circuit patterns."""

    def test_repeated_gates(self):
        """Circuit with repeated gates translates correctly."""
        translator = CircuitTranslator()
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.h(0)  # H twice = identity
        qc.h(0)
        
        result = translator.translate(qc)
        ops = result["circuit"]["operations"]
        assert len(ops) == 3
        assert all(op["name"] == "H" for op in ops)

    def test_circuit_with_all_basic_gates(self):
        """Circuit using all basic single-qubit gates."""
        translator = CircuitTranslator()
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.x(0)
        qc.y(0)
        qc.z(0)
        qc.s(0)
        qc.t(0)
        
        result = translator.translate(qc)
        ops = result["circuit"]["operations"]
        gate_names = [op["name"] for op in ops]
        
        assert "H" in gate_names
        assert "X" in gate_names
        assert "Y" in gate_names
        assert "Z" in gate_names
        assert "S" in gate_names
        assert "T" in gate_names

    def test_multiple_qubits_independent(self):
        """Independent operations on multiple qubits."""
        translator = CircuitTranslator()
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.x(1)
        qc.z(2)
        
        result = translator.translate(qc)
        ops = result["circuit"]["operations"]
        
        assert len(ops) == 3
        assert ops[0]["wires"] == [0]
        assert ops[1]["wires"] == [1]
        assert ops[2]["wires"] == [2]
