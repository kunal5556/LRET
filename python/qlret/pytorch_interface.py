"""PyTorch integration for QLRET autodiff (Phase 8.3).

Provides a custom autograd Function that delegates forward/backward to the
native `_qlret_native` bindings built with USE_PYTHON=ON.

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
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch

try:
    from . import _qlret_native
except ImportError as exc:  # pragma: no cover - native module optional
    raise ImportError(
        "_qlret_native is missing. Build with cmake -DUSE_PYTHON=ON .."
    ) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_py(obj: Any) -> Any:
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_py(x) for x in obj)
    if hasattr(obj, "detach"):
        return obj.detach().cpu().numpy().tolist()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


def _call_expectation(params, circuit_spec: Dict[str, Any], observable: Dict[str, Any], dtype, device):
    params_list = _to_py(params)
    value = _qlret_native.autodiff_expectation(
        circuit_spec["num_qubits"], circuit_spec["operations"], params_list, observable
    )
    return torch.tensor(value, dtype=dtype, device=device)


def _call_gradients(params, circuit_spec: Dict[str, Any], observable: Dict[str, Any], dtype, device):
    params_list = _to_py(params)
    grads = _qlret_native.autodiff_gradients(
        circuit_spec["num_qubits"], circuit_spec["operations"], params_list, observable
    )
    return torch.tensor(grads, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------


class _LRETExpectation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, params, circuit_spec, observable):
        if not params.is_floating_point():
            raise TypeError("params must be a floating point tensor")
        ctx.circuit_spec = circuit_spec
        ctx.observable = observable
        ctx.save_for_backward(params)
        return _call_expectation(params, circuit_spec, observable, params.dtype, params.device)

    @staticmethod
    def backward(ctx, grad_output):
        (params,) = ctx.saved_tensors
        grads = _call_gradients(params, ctx.circuit_spec, ctx.observable, params.dtype, params.device)
        return grad_output * grads, None, None


def lret_expectation(params: torch.Tensor, circuit_spec: Dict[str, Any], observable: Dict[str, Any]) -> torch.Tensor:
    """Compute expectation value with PyTorch autograd support."""
    return _LRETExpectation.apply(params, circuit_spec, observable)


__all__ = ["lret_expectation"]
