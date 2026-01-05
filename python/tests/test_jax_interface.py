import pytest

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    pytest.skip("JAX not installed", allow_module_level=True)

try:
    from qlret.jax_interface import lret_expectation
except ImportError as exc:  # pragma: no cover
    pytest.skip(str(exc), allow_module_level=True)


def test_jax_autodiff_single_parameter():
    circuit_spec = {
        "num_qubits": 1,
        "operations": [
            {"name": "RY", "qubits": [0], "param_idx": 0},
        ],
    }
    observable = {"type": "PauliZ", "qubit": 0}

    def energy(theta):
        return lret_expectation(theta, circuit_spec, observable)

    theta0 = jnp.array([0.3])
    val = energy(theta0)
    assert jnp.allclose(val, jnp.cos(theta0), atol=1e-6)

    grad_fn = jax.grad(lambda t: energy(t).sum())
    grad = grad_fn(theta0)
    assert jnp.allclose(grad, -jnp.sin(theta0), atol=1e-4)
