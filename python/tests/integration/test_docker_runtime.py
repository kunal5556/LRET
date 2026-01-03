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
            "./quantum_sim",
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

    python_code = """
import pennylane as qml
from qlret import QLRETDevice

dev = QLRETDevice(wires=2, shots=None)

@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

print(f"Result: {circuit():.4f}")
"""

    result = subprocess.run(
        ["docker", "run", "--rm", "qlret:latest", "python", "-c", python_code],
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert result.returncode == 0, f"Docker PennyLane failed: {result.stderr}"
    assert "Result:" in result.stdout
