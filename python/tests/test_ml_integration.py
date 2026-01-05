import pytest

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    pytest.skip("JAX not installed", allow_module_level=True)

try:
    import torch
except ImportError:  # pragma: no cover
    pytest.skip("PyTorch not installed", allow_module_level=True)

try:
    from qlret.jax_interface import lret_expectation as jax_expectation
    from qlret.pytorch_interface import lret_expectation as torch_expectation
except ImportError as exc:  # pragma: no cover
    pytest.skip(str(exc), allow_module_level=True)


@pytest.mark.parametrize("theta0, theta1", [(0.15, 0.25), (0.3, -0.4)])
def test_jax_torch_gradient_agreement(theta0, theta1):
    circuit_spec = {
        "num_qubits": 1,
        "operations": [
            {"name": "RY", "qubits": [0], "param_idx": 0},
            {"name": "RZ", "qubits": [0], "param_idx": 1},
        ],
    }
    observable = {"type": "PauliZ", "qubit": 0}

    # Expected analytics
    exp_val_expected = jnp.cos(theta0)
    grad0_expected = -jnp.sin(theta0)
    grad1_expected = 0.0  # RZ commutes with Z for expectation

    # JAX
    params_jax = jnp.array([theta0, theta1])
    val_jax = jax_expectation(params_jax, circuit_spec, observable)
    grad_jax = jax.grad(lambda p: jax_expectation(p, circuit_spec, observable).sum())(params_jax)

    assert jnp.allclose(val_jax, exp_val_expected, atol=1e-6)
    assert jnp.allclose(grad_jax[0], grad0_expected, atol=1e-4)
    assert jnp.allclose(grad_jax[1], grad1_expected, atol=1e-6)

    # PyTorch
    params_torch = torch.tensor([theta0, theta1], requires_grad=True)
    val_torch = torch_expectation(params_torch, circuit_spec, observable)
    val_torch.backward()

    assert torch.allclose(val_torch, torch.tensor(exp_val_expected, dtype=params_torch.dtype), atol=1e-6)
    assert torch.allclose(params_torch.grad[0], torch.tensor(grad0_expected, dtype=params_torch.dtype), atol=1e-4)
    assert torch.allclose(params_torch.grad[1], torch.tensor(grad1_expected, dtype=params_torch.dtype), atol=1e-6)


def test_multi_qubit_pauli_string_expectation():
    # Circuit: H(0); CNOT(0,1) produces Bell state |Phi+>
    circuit_spec = {
        "num_qubits": 2,
        "operations": [
            {"name": "H", "qubits": [0]},
            {"name": "CNOT", "qubits": [0, 1]},
        ],
    }
    observable = {"terms": [{"type": "PauliX", "qubit": 0}, {"type": "PauliX", "qubit": 1}]}

    # JAX
    params_jax = jnp.zeros((0,))  # no params
    val_jax = jax_expectation(params_jax, circuit_spec, observable)
    assert jnp.allclose(val_jax, 1.0, atol=1e-6)

    # PyTorch
    params_torch = torch.zeros((0,), requires_grad=False)
    val_torch = torch_expectation(params_torch, circuit_spec, observable)
    assert torch.allclose(val_torch, torch.tensor(1.0, dtype=val_torch.dtype), atol=1e-6)


def test_vqe_h2_minimal_energy_step():
    # Minimal 2-parameter ansatz for H2 (very simplified, not chemically accurate)
    # Circuit: RY(theta0) on q0, RY(theta1) on q1, CNOT(0,1)
    circuit_spec = {
        "num_qubits": 2,
        "operations": [
            {"name": "RY", "qubits": [0], "param_idx": 0},
            {"name": "RY", "qubits": [1], "param_idx": 1},
            {"name": "CNOT", "qubits": [0, 1]},
        ],
    }

    # Simple H2-like Hamiltonian (Z0 + Z1 + X0X1) with weights
    observable = {
        "terms": [
            {"type": "PauliZ", "qubit": 0},
            {"type": "PauliZ", "qubit": 1},
            {"type": "PauliX", "qubit": 0},
            {"type": "PauliX", "qubit": 1},
        ],
        "coefficient": 0.5,
    }

    theta0 = 0.5
    theta1 = -0.3

    # JAX evaluation
    params_jax = jnp.array([theta0, theta1])
    val_jax = jax_expectation(params_jax, circuit_spec, observable)
    grad_jax = jax.grad(lambda p: jax_expectation(p, circuit_spec, observable).sum())(params_jax)

    # PyTorch evaluation
    params_torch = torch.tensor([theta0, theta1], requires_grad=True)
    val_torch = torch_expectation(params_torch, circuit_spec, observable)
    val_torch.backward()

    # Cross-check expectations within a loose tolerance (toy Hamiltonian)
    assert jnp.allclose(val_jax, val_torch.detach().numpy(), atol=1e-5)

    # Cross-check gradients (directionally) within tolerance
    assert jnp.allclose(grad_jax[0], params_torch.grad[0].detach().numpy(), atol=1e-4)
    assert jnp.allclose(grad_jax[1], params_torch.grad[1].detach().numpy(), atol=1e-4)


def test_vqe_h2_full_hamiltonian_todo():
    # Standard 2-qubit H2 Hamiltonian (JW, R=0.735 Å) coefficients
    coeffs = [
        (-1.052373245772859, "I"),
        (0.39793742484318045, "Z0"),
        (-0.39793742484318045, "Z1"),
        (-0.01128010425623538, "Z0Z1"),
        (0.18093119978423156, "X0X1"),
    ]

    circuit_spec = {
        "num_qubits": 2,
        "operations": [
            {"name": "RY", "qubits": [0], "param_idx": 0},
            {"name": "RY", "qubits": [1], "param_idx": 1},
            {"name": "CNOT", "qubits": [0, 1]},
        ],
    }

    def make_obs(pauli: str):
        if pauli == "Z0":
            return {"terms": [{"type": "PauliZ", "qubit": 0}]}
        if pauli == "Z1":
            return {"terms": [{"type": "PauliZ", "qubit": 1}]}
        if pauli == "Z0Z1":
            return {"terms": [
                {"type": "PauliZ", "qubit": 0},
                {"type": "PauliZ", "qubit": 1},
            ]}
        if pauli == "X0X1":
            return {"terms": [
                {"type": "PauliX", "qubit": 0},
                {"type": "PauliX", "qubit": 1},
            ]}
        raise ValueError("Unsupported term")

    def energy_jax(params):
        total = 0.0
        for c, term in coeffs:
            if term == "I":
                total += c
            else:
                total += c * jax_expectation(params, circuit_spec, make_obs(term))
        return total

    def energy_torch(params):
        total = torch.tensor(0.0, dtype=params.dtype, device=params.device)
        for c, term in coeffs:
            if term == "I":
                total = total + torch.tensor(c, dtype=params.dtype, device=params.device)
            else:
                total = total + torch.tensor(c, dtype=params.dtype, device=params.device) * \
                    torch_expectation(params, circuit_spec, make_obs(term))
        return total

    theta0 = 0.2
    theta1 = -0.4

    params_jax = jnp.array([theta0, theta1])
    e_jax = energy_jax(params_jax)
    grad_jax = jax.grad(lambda p: energy_jax(p))(params_jax)

    params_torch = torch.tensor([theta0, theta1], requires_grad=True)
    e_torch = energy_torch(params_torch)
    e_torch.backward()

    assert jnp.allclose(e_jax, e_torch.detach().numpy(), atol=1e-4)
    assert jnp.allclose(grad_jax[0], params_torch.grad[0].detach().numpy(), atol=1e-4)
    assert jnp.allclose(grad_jax[1], params_torch.grad[1].detach().numpy(), atol=1e-4)


def test_qaoa_maxcut_todo():
    # Simple 6-node graph: edges define the MaxCut problem
    # Nodes 0-5; edges = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,0)]  (cycle)
    # Cost Hamiltonian: 0.5 * sum((1 - Z_i * Z_j) for (i,j) in edges)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]

    # 2-layer QAOA: p=1 (1 pair of gamma, beta)
    circuit_spec = {
        "num_qubits": 3,  # Simplified: use 3 qubits, 3 edges
        "operations": [
            # Initial Hadamard layer (unparameterized)
            {"name": "H", "qubits": [0]},
            {"name": "H", "qubits": [1]},
            {"name": "H", "qubits": [2]},
            # First QAOA layer: mixer (RY)
            {"name": "RY", "qubits": [0], "param_idx": 0},
            {"name": "RY", "qubits": [1], "param_idx": 0},
            {"name": "RY", "qubits": [2], "param_idx": 0},
        ],
    }

    # MaxCut cost on reduced 3-qubit graph: edges (0,1), (1,2), (2,0)
    # Cost = 0.5 * (3 - Z0*Z1 - Z1*Z2 - Z2*Z0)
    def cost_energy_jax(params):
        total = 1.5  # constant offset
        for q1, q2 in [(0, 1), (1, 2), (2, 0)]:
            obs = {"terms": [
                {"type": "PauliZ", "qubit": q1},
                {"type": "PauliZ", "qubit": q2},
            ]}
            total = total - 0.5 * jax_expectation(params, circuit_spec, obs)
        return total

    def cost_energy_torch(params):
        total = torch.tensor(1.5, dtype=params.dtype, device=params.device)
        for q1, q2 in [(0, 1), (1, 2), (2, 0)]:
            obs = {"terms": [
                {"type": "PauliZ", "qubit": q1},
                {"type": "PauliZ", "qubit": q2},
            ]}
            total = total - 0.5 * torch_expectation(params, circuit_spec, obs)
        return total

    beta0 = 0.3
    params_jax = jnp.array([beta0])
    cost_jax = cost_energy_jax(params_jax)
    grad_jax = jax.grad(lambda p: cost_energy_jax(p))(params_jax)

    params_torch = torch.tensor([beta0], requires_grad=True)
    cost_torch = cost_energy_torch(params_torch)
    cost_torch.backward()

    # Cross-check costs and gradients
    assert jnp.allclose(cost_jax, cost_torch.detach().numpy(), atol=1e-4)
    assert jnp.allclose(grad_jax[0], params_torch.grad[0].detach().numpy(), atol=1e-4)
    
    # Verify cost is in expected range [0, 3] for this problem
    assert 0.0 <= cost_jax <= 3.0
