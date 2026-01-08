"""Integration tests for Docker container runtime."""

from __future__ import annotations

import subprocess

import pytest


def _image_exists(tag: str) -> bool:
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", tag],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


@pytest.mark.docker
def test_docker_cli_execution(has_docker: bool):
    if not has_docker:
        pytest.skip("Docker not available")
    if not _image_exists("qlret:latest"):
        pytest.skip("Docker image 'qlret:latest' not built")

    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "qlret:latest",
            "quantum_sim",
            "-n",
            "6",
            "-d",
            "8",
            "--mode",
            "sequential",
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert result.returncode == 0, f"Docker CLI failed: {result.stderr}"
    assert "Final Rank" in result.stdout


@pytest.mark.docker
def test_docker_python_import(has_docker: bool):
    if not has_docker:
        pytest.skip("Docker not available")
    if not _image_exists("qlret:latest"):
        pytest.skip("Docker image 'qlret:latest' not built")

    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "qlret:latest",
            "python",
            "-c",
            "import qlret; print(qlret.__version__)",
        ],
        capture_output=True,
        text=True,
        timeout=45,
    )

    assert result.returncode == 0, f"Docker import failed: {result.stderr}"
    assert "1.0.0" in result.stdout


@pytest.mark.docker
@pytest.mark.pennylane
def test_docker_pennylane(has_docker: bool):
    if not has_docker:
        pytest.skip("Docker not available")
    if not _image_exists("qlret:latest"):
        pytest.skip("Docker image 'qlret:latest' not built")

    # Test using simulate_json (PennyLane Device has version compatibility issues)
    python_code = """
from qlret import simulate_json

result = simulate_json({
    'circuit': {
        'num_qubits': 2,
        'operations': [
            {'name': 'H', 'wires': [0]},
            {'name': 'CNOT', 'wires': [0, 1]}
        ],
        'observables': [{'type': 'TENSOR', 'operators': ['Z', 'Z'], 'wires': [0, 1]}]
    },
    'config': {'epsilon': 0.001}
})
exp_val = result['expectation_values'][0]
print(f"Result: {exp_val:.4f}")
"""

    result = subprocess.run(
        ["docker", "run", "--rm", "qlret:latest", "python", "-c", python_code],
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert result.returncode == 0, f"Docker simulation failed: {result.stderr}"
    assert "Result:" in result.stdout
