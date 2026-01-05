"""JAX integration for QLRET autodiff (Phase 8.3).

This module provides a JAX-friendly wrapper around the native autodiff
bindings exposed by the `_qlret_native` pybind module. It defines a custom
VJP (vector-Jacobian product) so `jax.grad`, `jax.jit`, and `jax.vmap` can
be used with QLRET circuits.

API:
    lret_expectation(params, circuit_spec, observable)

`circuit_spec` schema (Python dict):
{
    "num_qubits": int,
    "operations": [
        {"name": "RY", "qubits": [0], "param_idx": 0},
        {"name": "CNOT", "qubits": [0, 1]},
        {"name": "RZ", "qubits": [1], "param_idx": 1},
    ],
}

`observable` schema (Python dict):
- Single-qubit: {"type": "PauliZ", "qubit": 0, "coefficient": 1.0}
- Multi-qubit: {"terms": [{"type": "PauliX", "qubit": 0}, ...], "coefficient": 1.0}

Note: This wrapper requires `_qlret_native` to be built with USE_PYTHON=ON.
"""

from __future__ import annotations

from typing import Any, Dict, List

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "jax_interface requires JAX. Install with `pip install jax jaxlib`."
    ) from exc

try:
    from . import _qlret_native
except ImportError as exc:  # pragma: no cover - native module optional
    raise ImportError(
        "_qlret_native is missing. Build with cmake -DUSE_PYTHON=ON .."
    ) from exc


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _to_py(obj: Any) -> Any:
    """Convert JAX types to Python native types (lists for arrays)."""
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_py(x) for x in obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


def _call_expectation(params: jnp.ndarray, circuit_spec: Dict[str, Any], observable: Dict[str, Any]) -> jnp.ndarray:
    params_list = _to_py(params)
    value = _qlret_native.autodiff_expectation(
        circuit_spec["num_qubits"], circuit_spec["operations"], params_list, observable
    )
    return jnp.asarray(value, dtype=params.dtype)


def _call_gradients(params: jnp.ndarray, circuit_spec: Dict[str, Any], observable: Dict[str, Any]) -> jnp.ndarray:
    params_list = _to_py(params)
    grads = _qlret_native.autodiff_gradients(
        circuit_spec["num_qubits"], circuit_spec["operations"], params_list, observable
    )
    return jnp.asarray(grads, dtype=params.dtype)


# ---------------------------------------------------------------------------
# Public API: custom_vjp for JAX
# ---------------------------------------------------------------------------


@jax.custom_vjp
def lret_expectation(params: jnp.ndarray, circuit_spec: Dict[str, Any], observable: Dict[str, Any]):
    """Compute expectation value using QLRET autodiff with JAX support."""
    return _call_expectation(params, circuit_spec, observable)


def _lret_fwd(params: jnp.ndarray, circuit_spec: Dict[str, Any], observable: Dict[str, Any]):
    value = _call_expectation(params, circuit_spec, observable)
    return value, (params, circuit_spec, observable)


def _lret_bwd(res, g):
    params, circuit_spec, observable = res
    grads = _call_gradients(params, circuit_spec, observable)
    return (g * grads, None, None)


lret_expectation.defvjp(_lret_fwd, _lret_bwd)


__all__ = ["lret_expectation"]
