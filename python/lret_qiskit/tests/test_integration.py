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
        """Circuit can be transpiled to backend and run."""
        provider = LRETProvider()
        backend = provider.get_backend()
        
        # Create a circuit that may need transpilation
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        
        # Transpile to backend's basis gates
        transpiled = transpile(qc, backend)
        
        try:
            job = backend.run(transpiled, shots=100)
            result = job.result()
            assert result.success
        except Exception as e:
            if "No LRET backend available" in str(e):
                pytest.skip("LRET native module not built")
            raise
