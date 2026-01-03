# Quick Start Tutorial

Get started with LRET in 5 minutes! This tutorial covers the basics of simulating noisy quantum circuits using both the CLI and Python interface.

## Prerequisites

Complete the [Installation Guide](01-installation.md) using your preferred method (Docker, Python package, or source build).

---

## Your First Simulation (CLI)

### Step 1: Run a Simple Circuit

```bash
# Simulate 8 qubits, depth 20, with 1% depolarizing noise
quantum_sim -n 8 -d 20 --noise 0.01
```

**Expected Output:**
```
==============================================
LRET Quantum Simulator v1.0.0
==============================================
Configuration:
  Qubits: 8
  Circuit depth: 20
  Noise level: 0.01 (depolarizing)
  Parallel mode: hybrid
  Truncation threshold: 0.0001

Generating random circuit...
Circuit generated: 20 gates

Running LRET simulation...
[====================] 100% (20/20 gates)

Results:
  Simulation time: 0.047 seconds
  Final rank: 12
  Speedup vs FDM: 5.1x
  Memory used: 24.6 MB (FDM would need 268.4 MB)
  Fidelity (vs exact): 0.9998

Circuit diagram saved to: circuit_diagram.txt
State saved to: state_output.csv
```

### Step 2: Customize the Simulation

```bash
# More qubits, deeper circuit, higher noise
quantum_sim -n 10 -d 50 --noise 0.02 --mode hybrid -o results.csv

# Use specific gates (Hadamard + CNOT only)
quantum_sim -n 8 -d 30 --gates H,CNOT --noise 0.01

# Enable verbose output
quantum_sim -n 6 -d 15 --noise 0.01 --verbose
```

### Step 3: Load IBM Device Noise

```bash
# First, download noise model from IBM Quantum
python scripts/download_ibm_noise.py --device ibmq_manila --output manila_noise.json

# Simulate using real device noise
quantum_sim -n 5 -d 30 --noise-file manila_noise.json -o ibm_simulation.csv
```

### Step 4: Compare FDM vs LRET

```bash
# Run both FDM and LRET for comparison
quantum_sim -n 10 -d 30 --noise 0.01 --compare-fdm

# Output shows:
#   FDM time: 1.234 s
#   LRET time: 0.189 s
#   Speedup: 6.5x
#   Fidelity: 0.9997
```

---

## Your First Simulation (Python)

### Step 1: Basic Simulation

```python
from qlret import QuantumSimulator

# Create simulator with 4 qubits
sim = QuantumSimulator(n_qubits=4, noise_level=0.01)

# Apply gates
sim.h(0)              # Hadamard on qubit 0
sim.cnot(0, 1)        # CNOT: control=0, target=1
sim.rx(0.5, 2)        # RX(0.5) on qubit 2
sim.measure_all()     # Measure all qubits

# Get results
print(f"Final rank: {sim.current_rank}")
print(f"Simulation time: {sim.last_runtime:.3f} seconds")
print(f"State fidelity: {sim.fidelity:.4f}")
```

**Expected Output:**
```
Final rank: 8
Simulation time: 0.003 seconds
State fidelity: 0.9998
Measurement results: {'0000': 234, '0011': 241, '1100': 256, '1111': 269}
```

### Step 2: Build Custom Circuits

```python
from qlret import QuantumSimulator
import numpy as np

sim = QuantumSimulator(n_qubits=4, noise_level=0.01)

# Prepare entangled state |Φ+⟩ = (|00⟩ + |11⟩)/√2
sim.h(0)
sim.cnot(0, 1)

# Apply variational layer
params = [0.3, 0.7, 1.2]
sim.ry(params[0], 0)
sim.ry(params[1], 1)
sim.cnot(0, 1)
sim.rz(params[2], 1)

# Get density matrix
rho = sim.get_density_matrix()
print(f"Density matrix shape: {rho.shape}")
print(f"Trace: {np.trace(rho):.4f}")  # Should be ≈ 1.0
```

### Step 3: Use IBM Device Noise

```python
from qlret import QuantumSimulator, load_noise_model

# Load real device noise
noise = load_noise_model("ibmq_manila.json")

# Create simulator with device noise
sim = QuantumSimulator(n_qubits=5, noise_model=noise)

# Build circuit
sim.h(0)
for i in range(4):
    sim.cnot(i, i+1)

# Measure
results = sim.measure_all(shots=1000)
print(results)
```

### Step 4: Batch Simulations

```python
from qlret import QuantumSimulator
import matplotlib.pyplot as plt

# Test different noise levels
noise_levels = [0.001, 0.005, 0.01, 0.02, 0.05]
fidelities = []

for noise in noise_levels:
    sim = QuantumSimulator(n_qubits=8, noise_level=noise)
    
    # Standard circuit
    for i in range(8):
        sim.h(i)
    for i in range(7):
        sim.cnot(i, i+1)
    
    fidelities.append(sim.fidelity)

# Plot results
plt.plot(noise_levels, fidelities, 'o-')
plt.xlabel('Noise Level')
plt.ylabel('Fidelity')
plt.title('Circuit Fidelity vs Noise')
plt.grid(True)
plt.savefig('noise_vs_fidelity.png')
```

---

## PennyLane Integration

### Step 1: Create LRET Device

```python
import pennylane as qml
from qlret import QLRETDevice

# Create device
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
result = circuit([0.1, 0.2, 0.3])
print(f"Expectation value: {result:.4f}")
```

### Step 2: Variational Quantum Eigensolver (VQE)

```python
import pennylane as qml
from qlret import QLRETDevice
import numpy as np

# Define Hamiltonian (H2 molecule)
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

# Create noisy device
dev = QLRETDevice(wires=2, noise_level=0.005)

@qml.qnode(dev)
def cost_function(params):
    # Ansatz
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(params[2], wires=0)
    qml.RY(params[3], wires=1)
    return qml.expval(hamiltonian)

# Optimize
opt = qml.GradientDescentOptimizer(stepsize=0.1)
params = np.random.random(4)

for i in range(50):
    params = opt.step(cost_function, params)
    if i % 10 == 0:
        energy = cost_function(params)
        print(f"Step {i}: Energy = {energy:.4f}")

final_energy = cost_function(params)
print(f"\nFinal ground state energy: {final_energy:.4f}")
```

### Step 3: QAOA for MaxCut

```python
import pennylane as qml
from qlret import QLRETDevice
import numpy as np
import networkx as nx

# Define graph
graph = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
G = nx.Graph(graph)

# QAOA device with noise
dev = QLRETDevice(wires=4, noise_level=0.01)

def qaoa_circuit(params, graph, p=1):
    """QAOA circuit with p layers"""
    gammas = params[:p]
    betas = params[p:]
    
    # Initial state: |+⟩^n
    for i in range(4):
        qml.Hadamard(wires=i)
    
    # QAOA layers
    for layer in range(p):
        # Cost Hamiltonian
        for edge in graph:
            qml.CNOT(wires=[edge[0], edge[1]])
            qml.RZ(2 * gammas[layer], wires=edge[1])
            qml.CNOT(wires=[edge[0], edge[1]])
        
        # Mixer Hamiltonian
        for i in range(4):
            qml.RX(2 * betas[layer], wires=i)

@qml.qnode(dev)
def cost(params):
    qaoa_circuit(params, graph, p=2)
    
    # MaxCut expectation
    H = 0
    for edge in graph:
        H += 0.5 * (qml.Identity(edge[0]) - qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1]))
    
    return qml.expval(H)

# Optimize
opt = qml.AdamOptimizer(stepsize=0.1)
params = np.random.uniform(0, 2*np.pi, size=4)  # 2 gammas + 2 betas

for i in range(100):
    params = opt.step(cost, params)
    if i % 20 == 0:
        print(f"Step {i}: Cost = {cost(params):.4f}")
```

---

## Benchmarking Your Setup

### Performance Test

```bash
# Run quick benchmark
python scripts/benchmark_suite.py --quick --categories scaling,parallel

# Full benchmark suite
python scripts/benchmark_suite.py --output my_benchmark.csv

# Analyze results
python scripts/benchmark_analysis.py my_benchmark.csv --baseline-file baseline.csv

# Generate plots
python scripts/benchmark_visualize.py my_benchmark.csv --output plots/
```

### Expected Performance

On a typical workstation (8-core CPU, 32GB RAM):

| Test | Configuration | Expected Time | Expected Rank |
|------|---------------|---------------|---------------|
| Small | 8 qubits, depth 20 | < 0.1s | 10-15 |
| Medium | 10 qubits, depth 30 | < 0.5s | 15-25 |
| Large | 12 qubits, depth 40 | < 2s | 20-35 |
| XL | 14 qubits, depth 50 | < 10s | 30-50 |

If your results differ significantly, see [Troubleshooting](08-troubleshooting.md).

---

## Common Workflows

### 1. Noise Sensitivity Analysis

```python
from qlret import QuantumSimulator
import numpy as np

def test_circuit_noise_sensitivity(n_qubits, depth, noise_range):
    """Test how circuit fidelity degrades with noise"""
    results = []
    
    for noise in noise_range:
        sim = QuantumSimulator(n_qubits=n_qubits, noise_level=noise)
        
        # Build test circuit
        for _ in range(depth):
            sim.h(np.random.randint(n_qubits))
            sim.cnot(
                np.random.randint(n_qubits),
                np.random.randint(n_qubits)
            )
        
        results.append({
            'noise': noise,
            'fidelity': sim.fidelity,
            'rank': sim.current_rank,
            'runtime': sim.last_runtime
        })
    
    return results

# Run analysis
noise_range = np.logspace(-4, -1, 10)  # 0.0001 to 0.1
results = test_circuit_noise_sensitivity(10, 30, noise_range)

# Print results
for r in results:
    print(f"Noise: {r['noise']:.4f} | Fidelity: {r['fidelity']:.4f} | Rank: {r['rank']}")
```

### 2. Compare Multiple Algorithms

```python
from qlret import QuantumSimulator

def grover_search(sim, target):
    """Simplified Grover's algorithm"""
    n = sim.n_qubits
    
    # Initialize superposition
    for i in range(n):
        sim.h(i)
    
    # Grover iteration (simplified)
    sim.cnot(0, 1)
    sim.h(1)
    
    return sim.measure_all(shots=1000)

def qft(sim):
    """Quantum Fourier Transform"""
    n = sim.n_qubits
    
    for i in range(n):
        sim.h(i)
        for j in range(i+1, n):
            angle = np.pi / (2 ** (j - i))
            sim.crz(angle, i, j)
    
    return sim.get_state()

# Compare both under noise
noise = 0.01
sim1 = QuantumSimulator(n_qubits=4, noise_level=noise)
sim2 = QuantumSimulator(n_qubits=4, noise_level=noise)

grover_results = grover_search(sim1, target=5)
qft_state = qft(sim2)

print(f"Grover rank: {sim1.current_rank}, fidelity: {sim1.fidelity:.4f}")
print(f"QFT rank: {sim2.current_rank}, fidelity: {sim2.fidelity:.4f}")
```

### 3. Calibrate Custom Noise Model

```python
from qlret import QuantumSimulator
import json

# Define custom noise model
custom_noise = {
    "model_type": "mixed",
    "global_depolarizing": 0.001,
    "gate_specific": {
        "H": {"depolarizing": 0.0005},
        "CNOT": {"depolarizing": 0.01},
        "RX": {"amplitude_damping": 0.002},
        "RY": {"amplitude_damping": 0.002}
    },
    "qubit_specific": {
        "0": {"T1": 50e-6, "T2": 70e-6},
        "1": {"T1": 45e-6, "T2": 65e-6}
    }
}

# Save to file
with open("custom_noise.json", "w") as f:
    json.dump(custom_noise, f, indent=2)

# Use in simulation
sim = QuantumSimulator(n_qubits=2, noise_model="custom_noise.json")
sim.h(0)
sim.cnot(0, 1)
print(f"Custom noise simulation rank: {sim.current_rank}")
```

---

## Next Steps

Now that you've run your first simulations:

- **[CLI Reference →](03-cli-reference.md)** - Learn all command-line options
- **[Python Interface →](04-python-interface.md)** - Explore the full Python API
- **[Noise Models →](06-noise-models.md)** - Configure realistic noise
- **[PennyLane Integration →](05-pennylane-integration.md)** - Build hybrid algorithms
- **[Benchmarking Guide →](07-benchmarking.md)** - Measure and optimize performance

## Getting Help

- **[Troubleshooting Guide →](08-troubleshooting.md)** - Common issues and solutions
- **[GitHub Issues](https://github.com/kunal5556/LRET/issues)** - Report bugs
- **[GitHub Discussions](https://github.com/kunal5556/LRET/discussions)** - Ask questions
