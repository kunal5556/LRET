"""Python bridge API for QLRET.

Provides three execution backends:
1. Native: Uses pybind11 bindings if compiled (fastest, in-process)
2. Subprocess: Calls the quantum_sim CLI executable
3. Fallback: Pure-Python density matrix simulator (slowest, always available)

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
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

__all__ = ["simulate_json", "load_json_file", "set_executable_path", "QLRETError", "set_use_fallback"]


class QLRETError(RuntimeError):
    """Error from QLRET simulation."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_executable_path: Optional[str] = None
_native_module: Optional[Any] = None
_force_fallback: bool = False
_fallback_warned: bool = False


def set_executable_path(path: str) -> None:
    """Set explicit path to quantum_sim executable."""
    global _executable_path
    _executable_path = path


def set_use_fallback(use_fallback: bool) -> None:
    """Force use of the pure-Python fallback simulator.
    
    This is useful for testing or when you don't want to compile the native backend.
    """
    global _force_fallback
    _force_fallback = use_fallback


def _find_executable() -> Optional[str]:
    """Locate quantum_sim executable in common locations."""
    global _executable_path
    if _executable_path and os.path.isfile(_executable_path):
        return _executable_path

    # Check common locations
    candidates = [
        # Relative to package
        Path(__file__).parent.parent.parent / "build" / "quantum_sim",
        Path(__file__).parent.parent.parent / "build" / "quantum_sim.exe",
        Path(__file__).parent.parent.parent / "build" / "Release" / "quantum_sim.exe",
        # System PATH
        shutil.which("quantum_sim"),
    ]

    for c in candidates:
        if c is not None:
            p = Path(c) if isinstance(c, str) else c
            if p.exists():
                _executable_path = str(p)
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
    if use_native and not _force_fallback:
        native = _get_native_module()
        if native is not None:
            return _simulate_native(native, circuit, export_state)

    # Try subprocess backend
    if not _force_fallback:
        exe = _find_executable()
        if exe is not None:
            return _simulate_subprocess(circuit, export_state)

    # Fall back to pure-Python simulator
    return _simulate_fallback(circuit, export_state)


# ---------------------------------------------------------------------------
# Subprocess Backend
# ---------------------------------------------------------------------------


def _simulate_subprocess(circuit: Dict[str, Any], export_state: bool) -> Dict[str, Any]:
    """Execute via CLI subprocess."""
    exe = _find_executable()
    if exe is None:
        # This should not happen since we check in simulate_json
        raise QLRETError(
            "quantum_sim executable not found. "
            "Build the project or call set_executable_path()."
        )

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


# ---------------------------------------------------------------------------
# Fallback Backend (Pure Python)
# ---------------------------------------------------------------------------


def _simulate_fallback(circuit: Dict[str, Any], export_state: bool) -> Dict[str, Any]:
    """Execute via pure-Python density matrix simulator.
    
    This is the slowest backend but always available. It uses full density
    matrices instead of the low-rank representation.
    """
    global _fallback_warned
    
    if not _fallback_warned:
        warnings.warn(
            "Using pure-Python fallback simulator. This is slower than the "
            "native C++ backend and uses full density matrices (not low-rank). "
            "For production use, build the native module with: "
            "cd build && cmake .. && make",
            UserWarning,
            stacklevel=3
        )
        _fallback_warned = True
    
    from .fallback_simulator import simulate_circuit
    result = simulate_circuit(circuit)
    
    # Handle export_state (not fully supported in fallback)
    if export_state:
        result["state"] = {"warning": "Full state export not supported in fallback mode"}
    
    return result
