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
from functools import partial

try:
    import jax
    import jax.numpy as jnp
    from jax import ShapeDtypeStruct
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


def _call_expectation_pure(params_list: List[float], circuit_spec: Dict[str, Any], observable: Dict[str, Any]) -> float:
    """Pure Python function to call native expectation (no JAX types)."""
    value = _qlret_native.autodiff_expectation(
        circuit_spec["num_qubits"], circuit_spec["operations"], list(params_list), observable
    )
    return float(value)


def _call_gradients_pure(params_list: List[float], circuit_spec: Dict[str, Any], observable: Dict[str, Any]) -> List[float]:
    """Pure Python function to call native gradients (no JAX types)."""
    grads = _qlret_native.autodiff_gradients(
        circuit_spec["num_qubits"], circuit_spec["operations"], list(params_list), observable
    )
    return list(grads)


def _call_expectation(params: jnp.ndarray, circuit_spec: Dict[str, Any], observable: Dict[str, Any]) -> jnp.ndarray:
    params_list = _to_py(params)
    value = _call_expectation_pure(params_list, circuit_spec, observable)
    return jnp.asarray(value, dtype=params.dtype)


def _call_gradients(params: jnp.ndarray, circuit_spec: Dict[str, Any], observable: Dict[str, Any]) -> jnp.ndarray:
    params_list = _to_py(params)
    grads = _call_gradients_pure(params_list, circuit_spec, observable)
    return jnp.asarray(grads, dtype=params.dtype)


# ---------------------------------------------------------------------------
# Public API: custom_vjp for JAX with pure_callback for external calls
# ---------------------------------------------------------------------------


def _make_lret_expectation(circuit_spec: Dict[str, Any], observable: Dict[str, Any]):
    """Factory that creates a JAX-differentiable expectation function for a specific circuit."""
    
    def expectation_callback(params_array):
        """Pure callback for forward pass."""
        return jnp.array(_call_expectation_pure(params_array.tolist(), circuit_spec, observable))
    
    def gradients_callback(params_array):
        """Pure callback for gradient computation."""
        return jnp.array(_call_gradients_pure(params_array.tolist(), circuit_spec, observable))
    
    @jax.custom_vjp
    def expectation(params):
        # Use pure_callback to call external code
        return jax.pure_callback(
            expectation_callback,
            jax.ShapeDtypeStruct((), params.dtype),
            params
        )
    
    def expectation_fwd(params):
        value = jax.pure_callback(
            expectation_callback,
            jax.ShapeDtypeStruct((), params.dtype),
            params
        )
        return value, params
    
    def expectation_bwd(params, g):
        grads = jax.pure_callback(
            gradients_callback,
            jax.ShapeDtypeStruct(params.shape, params.dtype),
            params
        )
        return (g * grads,)
    
    expectation.defvjp(expectation_fwd, expectation_bwd)
    return expectation


def lret_expectation(params: jnp.ndarray, circuit_spec: Dict[str, Any], observable: Dict[str, Any]):
    """Compute expectation value using QLRET autodiff with JAX support.
    
    This function supports jax.grad for automatic differentiation.
    
    Args:
        params: JAX array of circuit parameters
        circuit_spec: Circuit specification dict with 'num_qubits' and 'operations'
        observable: Observable specification dict
        
    Returns:
        Expectation value as a JAX scalar
    """
    fn = _make_lret_expectation(circuit_spec, observable)
    return fn(params)


__all__ = ["lret_expectation"]
