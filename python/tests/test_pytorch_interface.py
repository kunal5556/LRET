import pytest

try:
    import torch
except ImportError:  # pragma: no cover
    pytest.skip("PyTorch not installed", allow_module_level=True)

try:
    from qlret.pytorch_interface import lret_expectation
except ImportError as exc:  # pragma: no cover
    pytest.skip(str(exc), allow_module_level=True)


def test_torch_autodiff_single_parameter():
    circuit_spec = {
        "num_qubits": 1,
        "operations": [
            {"name": "RY", "qubits": [0], "param_idx": 0},
        ],
    }
    observable = {"type": "PauliZ", "qubit": 0}

    params = torch.tensor([0.3], requires_grad=True)
    value = lret_expectation(params, circuit_spec, observable)
    value.backward()

    expected_val = torch.cos(params.detach())
    expected_grad = -torch.sin(params.detach())

    assert torch.allclose(value, expected_val, atol=1e-6)
    assert torch.allclose(params.grad, expected_grad, atol=1e-4)
