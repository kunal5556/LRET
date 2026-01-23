"""Python bridge API for QLRET.

Provides two execution backends:
1. Native: Uses pybind11 bindings (fastest, in-process) - REQUIRED
2. Subprocess: Calls the quantum_sim CLI executable (fallback)

Usage:
    from qlret import simulate_json, load_json_file
    result = simulate_json(circuit_dict)
    result = simulate_json(load_json_file("circuit.json"))
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

__all__ = ["simulate_json", "load_json_file", "set_executable_path", "QLRETError"]


class QLRETError(RuntimeError):
    """Error from QLRET simulation."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_executable_path: Optional[str] = None
_native_module: Optional[Any] = None


def set_executable_path(path: str) -> None:
    """Set explicit path to quantum_sim executable."""
    global _executable_path
    _executable_path = path


def _find_executable() -> Optional[str]:
    """Locate quantum_sim executable in common locations."""
    global _executable_path
    if _executable_path and os.path.isfile(_executable_path):
        return _executable_path

    # Check common locations
    candidates = [
        # Windows Visual Studio builds
        Path(__file__).parent.parent.parent / "build" / "Release" / "quantum_sim.exe",
        Path(__file__).parent.parent.parent / "build" / "Debug" / "quantum_sim.exe",
        Path(__file__).parent.parent.parent / "build" / "RelWithDebInfo" / "quantum_sim.exe",
        # Windows MinGW / Ninja builds
        Path(__file__).parent.parent.parent / "build" / "quantum_sim.exe",
        # Linux / macOS builds
        Path(__file__).parent.parent.parent / "build" / "quantum_sim",
        # System PATH
        shutil.which("quantum_sim"),
    ]

    for c in candidates:
        if c is not None:
            p = Path(c) if isinstance(c, str) else c
            if p.exists() and p.is_file():
                _executable_path = str(p.resolve())
                return _executable_path

    return None


def _get_native_module() -> Optional[Any]:
    """Try to import the native pybind11 module."""
    global _native_module
    if _native_module is not None:
        return _native_module
    try:
        from . import _qlret_native  # pybind11 module
        _native_module = _qlret_native
        return _native_module
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_json_file(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a circuit JSON file into a Python dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def simulate_json(
    circuit: Dict[str, Any],
    *,
    export_state: bool = False,
    use_native: bool = True,
) -> Dict[str, Any]:
    """Run a quantum circuit through QLRET and return results.

    Parameters
    ----------
    circuit : dict
        Circuit specification dict matching the JSON schema:
        {
            "circuit": {
                "num_qubits": int,
                "operations": [...],
                "observables": [...]
            },
            "config": {...}
        }
    export_state : bool
        If True, include the low-rank L matrix in results.
    use_native : bool
        If True, prefer native pybind11 bindings over subprocess.

    Returns
    -------
    dict
        {
            "status": "success",
            "execution_time_ms": float,
            "final_rank": int,
            "expectation_values": List[float],
            "samples": Optional[List[int]],
            "state": Optional[dict]  # if export_state
        }

    Raises
    ------
    QLRETError
        If simulation fails.
    """
    # Try native bindings first
    if use_native:
        native = _get_native_module()
        if native is not None:
            return _simulate_native(native, circuit, export_state)

    # Try subprocess backend
    exe = _find_executable()
    if exe is not None:
        return _simulate_subprocess(circuit, export_state)

    # No backend available
    raise QLRETError(
        "No QLRET backend available. Please build the native module:\n"
        "  cd build && cmake .. -DUSE_PYTHON=ON && make\n"
        "Or ensure quantum_sim executable is in PATH."
    )


# ---------------------------------------------------------------------------
# Subprocess Backend
# ---------------------------------------------------------------------------


def _simulate_subprocess(circuit: Dict[str, Any], export_state: bool) -> Dict[str, Any]:
    """Execute via CLI subprocess."""
    exe = _find_executable()
    if exe is None:
        # Provide detailed diagnostics
        build_dir = Path(__file__).parent.parent.parent / "build"
        error_msg = [
            "quantum_sim executable not found.",
            f"Searched locations:",
            f"  - {build_dir / 'Release' / 'quantum_sim.exe'}",
            f"  - {build_dir / 'Debug' / 'quantum_sim.exe'}",
            f"  - {build_dir / 'quantum_sim.exe'}",
            f"  - {build_dir / 'quantum_sim'}",
            f"  - System PATH",
            f"\nBuild directory exists: {build_dir.exists()}",
        ]
        if build_dir.exists():
            try:
                files = list(build_dir.rglob("quantum_sim*"))
                if files:
                    error_msg.append(f"Found these files in build/:")
                    for f in files[:10]:  # Show first 10
                        error_msg.append(f"  - {f}")
                else:
                    error_msg.append("No quantum_sim files found in build/")
            except Exception as e:
                error_msg.append(f"Could not list build directory: {e}")
        
        raise QLRETError("\n".join(error_msg))

    # Write circuit to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(circuit, f)
        input_path = f.name

    output_path = input_path.replace(".json", "_result.json")

    try:
        cmd = [exe, "--input-json", input_path, "--output-json", output_path]
        if export_state:
            cmd.append("--export-json-state")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode != 0:
            raise QLRETError(f"quantum_sim failed: {result.stderr or result.stdout}")

        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)

    finally:
        # Cleanup temp files
        for p in [input_path, output_path]:
            try:
                os.unlink(p)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Native Backend (pybind11)
# ---------------------------------------------------------------------------


def _simulate_native(
    native: Any, circuit: Dict[str, Any], export_state: bool
) -> Dict[str, Any]:
    """Execute via native pybind11 bindings."""
    json_str = json.dumps(circuit)
    result_str = native.run_circuit_json(json_str, export_state)
    return json.loads(result_str)
