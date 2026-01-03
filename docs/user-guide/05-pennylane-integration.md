# PennyLane Integration

LRET provides a native PennyLane device plugin, enabling seamless integration with PennyLane's quantum machine learning and hybrid quantum-classical computing framework.

## Why PennyLane + LRET?

- **Automatic differentiation:** Gradient-based optimization of variational circuits
- **Noisy simulation:** Realistic device simulation with LRET's efficient noise handling
- **Hybrid algorithms:** VQE, QAOA, quantum machine learning
- **Framework integration:** PyTorch, TensorFlow, JAX compatibility
- **Rank efficiency:** LRET's low-rank decomposition accelerates noisy circuit simulation

---

## Installation

```bash
# Install LRET with PennyLane support
pip install qlret pennylane

# Verify installation
python -c "from qlret import QLRETDevice; print('✓ PennyLane device installed')"
```

---

## Quick Start

```python
import pennylane as qml
from qlret import QLRETDevice

# Create LRET device
dev = QLRETDevice(wires=4, noise_level=0.01)

# Define quantum node
@qml.qnode(dev)
def circuit(params):
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(params[2], wires=1)
    return qml.expval(qml.PauliZ(0))

# Execute
import numpy as np
params = np.array([0.1, 0.2, 0.3])
result = circuit(params)
print(f"Expectation value: {result:.4f}")
```

---

## QLRETDevice Reference

### Constructor

```python
QLRETDevice(
    wires: int | list,
    noise_level: float = 0.0,
    noise_model: Optional[Union[str, dict]] = None,
    parallel_mode: str = "hybrid",
    truncation_threshold: float = 1e-4,
    max_rank: Optional[int] = None,
    shots: Optional[int] = None
)
```

**Parameters:**
- `wires` (int | list): Number of qubits or wire labels
- `noise_level` (float): Global depolarizing noise (0.0-1.0)
- `noise_model` (str | dict): Custom noise model (file or dict)
- `parallel_mode` (str): Parallelization (`"sequential"`, `"row"`, `"column"`, `"hybrid"`)
- `truncation_threshold` (float): SVD truncation threshold
- `max_rank` (int | None): Maximum rank limit
- `shots` (int | None): Number of measurement shots (`None` = analytic mode)

**Example:**
```python
# Basic device
dev = QLRETDevice(wires=4, noise_level=0.01)

# Custom wire labels
dev = QLRETDevice(wires=['q0', 'q1', 'q2', 'q3'], noise_level=0.01)

# IBM device noise
dev = QLRETDevice(wires=5, noise_model="ibmq_manila.json")

# Finite shots (sampling mode)
dev = QLRETDevice(wires=4, noise_level=0.01, shots=1000)
```

### Supported Operations

**Single-qubit gates:**
- `PauliX`, `PauliY`, `PauliZ`
- `Hadamard`
- `RX`, `RY`, `RZ`
- `Rot` (general rotation)
- `S`, `T` (phase gates)
- `SX` (√X gate)

**Two-qubit gates:**
- `CNOT`, `CZ`, `SWAP`
- `CRX`, `CRY`, `CRZ`
- `IsingXX`, `IsingYY`, `IsingZZ`

**Three-qubit gates:**
- `Toffoli`, `CSWAP`

**Observables:**
- `PauliX`, `PauliY`, `PauliZ`
- `Hadamard`, `Identity`
- `Hermitian` (custom observables)
- Tensor products: `@` operator

**Example:**
```python
import pennylane as qml
from qlret import QLRETDevice

dev = QLRETDevice(wires=3, noise_level=0.01)

@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(0.5, wires=2)
    qml.Toffoli(wires=[0, 1, 2])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))  # Tensor product

result = circuit()
print(result)
```

---

## Variational Quantum Eigensolver (VQE)

### Basic VQE

```python
import pennylane as qml
from qlret import QLRETDevice
import numpy as np

# Hamiltonian for H2 molecule
coeffs = [0.2252, 0.3435, -0.4347, 0.5716, 0.0910, 0.0910]
obs = [
    qml.Identity(0),
    qml.PauliZ(0),
    qml.PauliZ(1),
    qml.PauliZ(0) @ qml.PauliZ(1),
    qml.PauliY(0) @ qml.PauliY(1),
    qml.PauliX(0) @ qml.PauliX(1)
]
hamiltonian = qml.Hamiltonian(coeffs, obs)

# Create device with noise
dev = QLRETDevice(wires=2, noise_level=0.005)

@qml.qnode(dev)
def cost_function(params):
    # Hardware-efficient ansatz
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(params[2], wires=0)
    qml.RY(params[3], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(hamiltonian)

# Optimize
opt = qml.GradientDescentOptimizer(stepsize=0.4)
params = np.random.random(4) * 2 * np.pi

max_iterations = 100
conv_tol = 1e-6

for n in range(max_iterations):
    params, prev_energy = opt.step_and_cost(cost_function, params)
    
    if n % 10 == 0:
        print(f"Step {n}: Energy = {prev_energy:.6f}")
    
    if n > 0 and np.abs(prev_energy - energy) < conv_tol:
        print(f"Converged at step {n}")
        break
    
    energy = prev_energy

print(f"\nFinal ground state energy: {energy:.6f}")
print(f"Optimal parameters: {params}")
```

### Adaptive VQE with Noise Calibration

```python
import pennylane as qml
from qlret import QLRETDevice, load_noise_model
import numpy as np

# Load real device noise
noise = load_noise_model("ibmq_manila.json")
dev = QLRETDevice(wires=4, noise_model=noise)

def layered_ansatz(params, n_layers):
    """Parameterized ansatz with multiple layers"""
    n_wires = len(dev.wires)
    
    for layer in range(n_layers):
        # Single-qubit rotations
        for i in range(n_wires):
            qml.RY(params[layer * 2 * n_wires + i], wires=i)
        
        # Entangling layer
        for i in range(n_wires - 1):
            qml.CNOT(wires=[i, i+1])
        
        # Another rotation layer
        for i in range(n_wires):
            qml.RZ(params[layer * 2 * n_wires + n_wires + i], wires=i)

@qml.qnode(dev)
def vqe_circuit(params, hamiltonian, n_layers):
    layered_ansatz(params, n_layers)
    return qml.expval(hamiltonian)

# Define Hamiltonian (example: transverse field Ising)
n_qubits = 4
J = 1.0  # Coupling
h = 0.5  # Transverse field

coeffs = []
obs = []

# ZZ interactions
for i in range(n_qubits - 1):
    coeffs.append(-J)
    obs.append(qml.PauliZ(i) @ qml.PauliZ(i+1))

# X fields
for i in range(n_qubits):
    coeffs.append(-h)
    obs.append(qml.PauliX(i))

hamiltonian = qml.Hamiltonian(coeffs, obs)

# Optimize with adaptive layers
n_layers = 3
n_params = n_layers * 2 * n_qubits
initial_params = np.random.uniform(0, 2*np.pi, n_params)

opt = qml.AdamOptimizer(stepsize=0.1)

for i in range(100):
    params = opt.step(lambda p: vqe_circuit(p, hamiltonian, n_layers), initial_params)
    
    if i % 20 == 0:
        energy = vqe_circuit(initial_params, hamiltonian, n_layers)
        print(f"Step {i}: Energy = {energy:.4f}")
    
    initial_params = params

final_energy = vqe_circuit(params, hamiltonian, n_layers)
print(f"\nFinal energy: {final_energy:.6f}")
```

---

## Quantum Approximate Optimization Algorithm (QAOA)

### MaxCut Problem

```python
import pennylane as qml
from qlret import QLRETDevice
import numpy as np
import networkx as nx

# Define graph
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
graph = nx.Graph(edges)
n_qubits = len(graph.nodes)

# LRET device with noise
dev = QLRETDevice(wires=n_qubits, noise_level=0.01)

def qaoa_maxcut_cost(graph):
    """Cost Hamiltonian for MaxCut"""
    coeffs = []
    obs = []
    
    for edge in graph.edges:
        coeffs.append(0.5)
        obs.append(qml.Identity(edge[0]) - qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1]))
    
    return qml.Hamiltonian(coeffs, obs)

def qaoa_circuit(gammas, betas, graph, p):
    """QAOA circuit with p layers"""
    # Initialize |+⟩ state
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    
    # QAOA layers
    for layer in range(p):
        # Cost Hamiltonian evolution
        for edge in graph.edges:
            qml.CNOT(wires=[edge[0], edge[1]])
            qml.RZ(2 * gammas[layer], wires=edge[1])
            qml.CNOT(wires=[edge[0], edge[1]])
        
        # Mixer Hamiltonian evolution
        for i in range(n_qubits):
            qml.RX(2 * betas[layer], wires=i)

@qml.qnode(dev)
def qaoa_cost(params, graph, p):
    """QAOA cost function"""
    gammas = params[:p]
    betas = params[p:]
    
    qaoa_circuit(gammas, betas, graph, p)
    
    return qml.expval(qaoa_maxcut_cost(graph))

# Optimize
p = 2  # Number of QAOA layers
params = np.random.uniform(0, 2*np.pi, size=2*p)

opt = qml.AdamOptimizer(stepsize=0.1)

for i in range(150):
    params = opt.step(lambda x: qaoa_cost(x, graph, p), params)
    
    if i % 30 == 0:
        cost = qaoa_cost(params, graph, p)
        print(f"Step {i}: Cost = {cost:.4f}")

# Final result
final_cost = qaoa_cost(params, graph, p)
print(f"\nOptimized MaxCut cost: {final_cost:.4f}")
print(f"Optimal parameters: γ={params[:p]}, β={params[p:]}")

# Sample bitstrings to find cut
@qml.qnode(dev, interface="numpy")
def qaoa_sample(params, graph, p):
    gammas = params[:p]
    betas = params[p:]
    qaoa_circuit(gammas, betas, graph, p)
    return qml.sample()

# Run sampling (requires shots)
dev_sample = QLRETDevice(wires=n_qubits, noise_level=0.01, shots=1000)
qnode_sample = qml.QNode(qaoa_sample, dev_sample)

samples = qnode_sample(params, graph, p)
print(f"\nTop 5 bitstrings: {samples[:5]}")
```

---

## Quantum Machine Learning

### Quantum Neural Network Classifier

```python
import pennylane as qml
from qlret import QLRETDevice
import numpy as np

# Generate synthetic dataset
np.random.seed(42)
n_samples = 100
X_train = np.random.randn(n_samples, 4)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

# Create device
dev = QLRETDevice(wires=4, noise_level=0.005)

def quantum_layer(inputs, weights):
    """Quantum feature map + variational layer"""
    # Feature encoding
    for i, x in enumerate(inputs):
        qml.RY(x, wires=i)
    
    # Entangling layer
    for i in range(3):
        qml.CNOT(wires=[i, i+1])
    
    # Variational layer
    for i, w in enumerate(weights):
        qml.RY(w, wires=i)

@qml.qnode(dev)
def quantum_classifier(inputs, weights):
    quantum_layer(inputs, weights)
    return qml.expval(qml.PauliZ(0))

def predict(inputs, weights):
    """Binary classification: 0 or 1"""
    exp_val = quantum_classifier(inputs, weights)
    return 1 if exp_val > 0 else 0

def cost(weights, X, y):
    """Binary cross-entropy loss"""
    predictions = np.array([quantum_classifier(x, weights) for x in X])
    # Map expectation values [-1, 1] to probabilities [0, 1]
    probs = (predictions + 1) / 2
    # Cross-entropy
    loss = -np.mean(y * np.log(probs + 1e-10) + (1 - y) * np.log(1 - probs + 1e-10))
    return loss

# Train
weights = np.random.randn(4) * 0.1
opt = qml.AdamOptimizer(stepsize=0.1)

batch_size = 10
epochs = 20

for epoch in range(epochs):
    # Mini-batch training
    indices = np.random.permutation(n_samples)
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i+batch_size]
        X_batch = X_train[batch_indices]
        y_batch = y_train[batch_indices]
        
        weights = opt.step(lambda w: cost(w, X_batch, y_batch), weights)
    
    # Evaluate
    train_loss = cost(weights, X_train, y_train)
    accuracy = np.mean([predict(x, weights) == y for x, y in zip(X_train, y_train)])
    
    print(f"Epoch {epoch+1}: Loss = {train_loss:.4f}, Accuracy = {accuracy:.2%}")

print(f"\nFinal weights: {weights}")
```

### Data Re-uploading Classifier

```python
import pennylane as qml
from qlret import QLRETDevice
import numpy as np

dev = QLRETDevice(wires=1, noise_level=0.005)

def data_reuploading_layer(x, weights):
    """Single layer with data re-uploading"""
    qml.RX(x * weights[0], wires=0)
    qml.RY(weights[1], wires=0)
    qml.RZ(x * weights[2], wires=0)

@qml.qnode(dev)
def data_reuploading_classifier(x, weights, n_layers):
    """Classifier with multiple re-uploading layers"""
    for i in range(n_layers):
        data_reuploading_layer(x, weights[i])
    return qml.expval(qml.PauliZ(0))

# Generate 1D dataset
np.random.seed(42)
n_samples = 50
X_train = np.random.uniform(-np.pi, np.pi, n_samples)
y_train = (np.sin(X_train) > 0).astype(int) * 2 - 1  # {-1, 1}

# Train
n_layers = 3
weights = np.random.randn(n_layers, 3) * 0.1

def cost(w):
    predictions = np.array([data_reuploading_classifier(x, w, n_layers) for x in X_train])
    return np.mean((predictions - y_train) ** 2)

opt = qml.AdamOptimizer(stepsize=0.1)

for i in range(100):
    weights = opt.step(cost, weights)
    
    if i % 20 == 0:
        loss = cost(weights)
        print(f"Step {i}: Loss = {loss:.4f}")

# Test on new data
X_test = np.linspace(-np.pi, np.pi, 100)
y_pred = np.array([data_reuploading_classifier(x, weights, n_layers) for x in X_test])

import matplotlib.pyplot as plt
plt.plot(X_test, y_pred, label='Prediction')
plt.plot(X_test, np.sin(X_test), 'r--', label='True')
plt.scatter(X_train, y_train, c='black', marker='x', label='Training data')
plt.legend()
plt.savefig('data_reuploading.png')
```

---

## Advanced Features

### Gradient Computation

```python
import pennylane as qml
from qlret import QLRETDevice
import numpy as np

dev = QLRETDevice(wires=2, noise_level=0.01)

@qml.qnode(dev, diff_method="parameter-shift")
def circuit(params):
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

params = np.array([0.5, 0.3])

# Compute gradient
grad_fn = qml.grad(circuit)
gradients = grad_fn(params)

print(f"Circuit output: {circuit(params):.4f}")
print(f"Gradients: {gradients}")
```

### Framework Integration (PyTorch)

```python
import pennylane as qml
from qlret import QLRETDevice
import torch
import torch.nn as nn

# LRET device
dev = QLRETDevice(wires=4, noise_level=0.005)

@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def quantum_layer_torch(inputs, weights):
    # Encode inputs
    for i, x in enumerate(inputs):
        qml.RY(x, wires=i)
    
    # Variational circuit
    for i, w in enumerate(weights):
        qml.RY(w, wires=i)
    
    for i in range(3):
        qml.CNOT(wires=[i, i+1])
    
    return qml.expval(qml.PauliZ(0))

class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 4)
        self.quantum_weights = nn.Parameter(torch.randn(4) * 0.1)
        self.fc2 = nn.Linear(1, 2)
    
    def forward(self, x):
        # Classical preprocessing
        x = torch.relu(self.fc1(x))
        
        # Quantum layer
        x = quantum_layer_torch(x, self.quantum_weights)
        x = x.unsqueeze(-1)
        
        # Classical postprocessing
        x = self.fc2(x)
        return x

# Train hybrid model
model = HybridModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Dummy data
X_train = torch.randn(100, 10)
y_train = torch.randint(0, 2, (100,))

for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
```

---

## Best Practices

### 1. Choose Appropriate Noise Levels

```python
# Too low: no speed benefit from LRET
dev_low = QLRETDevice(wires=10, noise_level=0.0001)

# Realistic: good balance
dev_realistic = QLRETDevice(wires=10, noise_level=0.01)

# Too high: poor fidelity
dev_high = QLRETDevice(wires=10, noise_level=0.1)
```

### 2. Use Analytic Mode When Possible

```python
# Analytic mode (exact expectation values, faster)
dev_analytic = QLRETDevice(wires=4, noise_level=0.01, shots=None)

# Sampling mode (finite shots, slower but more realistic)
dev_sampling = QLRETDevice(wires=4, noise_level=0.01, shots=1000)
```

### 3. Monitor Rank Growth

```python
from qlret import QLRETDevice
import pennylane as qml

dev = QLRETDevice(wires=8, noise_level=0.01, verbose=True)

@qml.qnode(dev)
def circuit():
    for i in range(8):
        qml.Hadamard(wires=i)
    for i in range(7):
        qml.CNOT(wires=[i, i+1])
    return qml.expval(qml.PauliZ(0))

result = circuit()
# Verbose mode prints rank after each gate
```

---

## See Also

- **[Python Interface](04-python-interface.md)** - Core Python API
- **[Noise Models](06-noise-models.md)** - Configuring realistic noise
- **[Examples](../examples/python/)** - More PennyLane examples
- **[PennyLane Documentation](https://pennylane.ai)** - Official PennyLane docs
