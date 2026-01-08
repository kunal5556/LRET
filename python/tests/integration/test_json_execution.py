"""Integration tests for JSON circuit execution via Python API."""

from __future__ import annotations

import math
import pytest

from qlret import simulate_json, QLRETError


@pytest.mark.subprocess
def test_bell_pair_subprocess(bell_circuit, quantum_sim_path):
    """Bell state via subprocess backend."""
    result = simulate_json(bell_circuit, use_native=False)

    pytest.assert_result_valid(result)
    pytest.assert_bell_state_expectations(result["expectation_values"])


@pytest.mark.native
def test_bell_pair_native(bell_circuit, has_native_module):
    """Bell state via native pybind11 backend."""
    if not has_native_module:
        pytest.skip("Native module not available")

    result = simulate_json(bell_circuit, use_native=True)

    pytest.assert_result_valid(result)
    pytest.assert_bell_state_expectations(result["expectation_values"])


def test_parametric_circuit_rotations():
    """Parameterized single-qubit rotation should give <Z> ≈ cos(theta)."""
    theta = math.pi / 2  # 90 degrees
    circuit = {
        "circuit": {
            "num_qubits": 1,
            "operations": [
                {"name": "RX", "wires": [0], "params": [theta]},
            ],
            "observables": [
                {"type": "PAULI", "operator": "Z", "wires": [0]},
            ],
        },
        "config": {"epsilon": 1e-4},
    }

    result = simulate_json(circuit, use_native=False)

    pytest.assert_result_valid(result)
    # RX(pi/2)|0> => |+Y> gives <Z> ≈ 0
    assert abs(result["expectation_values"][0]) < 0.1


@pytest.mark.slow
def test_sampling_results_have_expected_length():
    """Shot-based sampling returns the requested number of samples."""
    circuit = {
        "circuit": {
            "num_qubits": 2,
            "operations": [
                {"name": "H", "wires": [0]},
                {"name": "CNOT", "wires": [0, 1]},
            ],
            "observables": [],
        },
        "config": {"epsilon": 1e-4, "shots": 100},
    }

    result = simulate_json(circuit, use_native=False)

    pytest.assert_result_valid(result)
    assert "samples" in result, "Missing samples field"
    assert len(result["samples"]) == 100, "Expected 100 samples"


def test_invalid_circuit_error():
    """Invalid circuits should raise QLRETError."""
    invalid = {"circuit": {"num_qubits": -1, "operations": []}, "config": {}}

    with pytest.raises((QLRETError, Exception)):
        simulate_json(invalid, use_native=False)


def test_state_export_optional():
    """State export should succeed or be skipped if unsupported."""
    circuit = {
        "circuit": {
            "num_qubits": 2,
            "operations": [{"name": "H", "wires": [0]}],
            "observables": [],
        },
        "config": {"epsilon": 1e-4},
    }

    result = simulate_json(circuit, use_native=False, export_state=True)
    pytest.assert_result_valid(result)

    # If the backend provides state, validate structure; otherwise allow absence.
    if "state" in result:
        state = result["state"]
        assert "L_real" in state and "L_imag" in state
        # Check for low_rank state format (type field instead of rank)
        assert "type" in state or "rank" in state
        if "rank" in state:
            assert state["rank"] == result["final_rank"]
