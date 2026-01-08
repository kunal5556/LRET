"""Integration tests for CLI executable."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import pytest


@pytest.mark.subprocess
def test_basic_simulation(quantum_sim_path: Path):
    result = subprocess.run(
        [str(quantum_sim_path), "-n", "6", "-d", "8", "--mode", "sequential"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, f"CLI failed: {result.stderr or result.stdout}"
    assert "Final Rank" in result.stdout
    # Check for time output (format may vary: "Time:" or "Simulation Time")
    assert "Time:" in result.stdout or "Simulation Time" in result.stdout


@pytest.mark.subprocess
def test_parallel_modes(quantum_sim_path: Path):
    modes = ["sequential", "row", "column", "hybrid"]
    for mode in modes:
        result = subprocess.run(
            [str(quantum_sim_path), "-n", "6", "-d", "8", "--mode", mode],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Mode {mode} failed: {result.stderr}"


@pytest.mark.subprocess
def test_csv_output(quantum_sim_path: Path, temp_output_dir: Path):
    output_file = temp_output_dir / "results.csv"

    result = subprocess.run(
        [
            str(quantum_sim_path),
            "-n",
            "6",
            "-d",
            "8",
            "-o",
            str(output_file),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, f"CSV run failed: {result.stderr}"
    assert output_file.exists(), "CSV file not created"

    # Read CSV content - the format is structured with sections
    content = output_file.read_text()
    # Check for key indicators in structured CSV format
    assert "SECTION" in content, "CSV missing SECTION markers"
    assert "num_qubits" in content, "CSV missing num_qubits"
    assert "depth" in content, "CSV missing depth"
    assert "final_rank" in content or "rank" in content, "CSV missing rank info"


@pytest.mark.subprocess
@pytest.mark.slow
def test_fdm_comparison(quantum_sim_path: Path):
    result = subprocess.run(
        [str(quantum_sim_path), "-n", "8", "-d", "10", "--fdm"],
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert result.returncode == 0, f"FDM run failed: {result.stderr}"
    stdout_lower = result.stdout.lower()
    assert "fidelity" in stdout_lower or "fdm" in stdout_lower


@pytest.mark.subprocess
def test_json_io(quantum_sim_path: Path, samples_dir: Path, temp_output_dir: Path):
    bell_json = samples_dir / "json" / "bell_pair.json"
    if not bell_json.exists():
        pytest.skip("Sample bell_pair.json not found")

    output_json = temp_output_dir / "result.json"

    result = subprocess.run(
        [
            str(quantum_sim_path),
            "--input-json",
            str(bell_json),
            "--output-json",
            str(output_json),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, f"JSON IO failed: {result.stderr}"
    assert output_json.exists(), "Output JSON not created"

    with open(output_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data.get("status") == "success"
