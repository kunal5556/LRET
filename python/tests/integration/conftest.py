"""Shared fixtures for integration tests."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterator

import pytest


# ---------------------------------------------------------------------------
# Paths and discovery helpers
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    # conftest.py is at python/tests/integration/, so repo root is parents[3]
    return Path(__file__).resolve().parents[3]


def _find_quantum_sim() -> Path | None:
    root = _repo_root()
    candidates = [
        root / "build" / "quantum_sim",
        root / "build" / "quantum_sim.exe",
        root / "build" / "Release" / "quantum_sim.exe",
        Path("/usr/local/bin/quantum_sim"),
        Path(shutil.which("quantum_sim")) if shutil.which("quantum_sim") else None,
    ]

    for path in candidates:
        if path and path.exists():
            return path
    return None


# ---------------------------------------------------------------------------
# Session fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return _repo_root()


@pytest.fixture(scope="session")
def quantum_sim_path() -> Path:
    path = _find_quantum_sim()
    if path is None:
        pytest.skip("quantum_sim executable not found; build the project first")
    return path


@pytest.fixture(scope="session")
def samples_dir(repo_root: Path) -> Path:
    path = repo_root / "samples"
    if not path.exists():
        pytest.skip("samples directory not found")
    return path


@pytest.fixture(scope="session")
def has_native_module() -> bool:
    try:
        from qlret import _qlret_native  # noqa: F401
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def has_pennylane() -> bool:
    try:
        import pennylane  # noqa: F401
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def has_docker() -> bool:
    try:
        result = subprocess.run(
            ["docker", "--version"], capture_output=True, timeout=5, text=True
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


@pytest.fixture(scope="session")
def backend_available(has_native_module: bool) -> bool:
    return bool(has_native_module or _find_quantum_sim())


# ---------------------------------------------------------------------------
# Circuit fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bell_circuit() -> dict:
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


@pytest.fixture
def temp_output_dir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def assert_bell_state_expectations(exp_values, tolerance: float = 0.1) -> None:
    assert len(exp_values) >= 2, "Need at least two observables"
    z0, z0z1 = exp_values[0], exp_values[1]
    assert abs(z0) < tolerance, f"<Z0>={z0:.4f}, expected ≈ 0"
    assert abs(z0z1 - 1.0) < tolerance, f"<Z0Z1>={z0z1:.4f}, expected ≈ 1"


def assert_result_valid(result: dict) -> None:
    assert "status" in result, "Missing status"
    assert result["status"] == "success", f"Status: {result['status']}"
    assert "final_rank" in result, "Missing final_rank"
    assert result["final_rank"] > 0, "Rank must be positive"
    assert "execution_time_ms" in result, "Missing execution_time_ms"


# Expose helpers to pytest namespace
pytest.assert_bell_state_expectations = assert_bell_state_expectations
pytest.assert_result_valid = assert_result_valid
