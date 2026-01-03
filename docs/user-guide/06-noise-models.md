# Noise Models

LRET supports comprehensive noise modeling to simulate realistic quantum devices. This guide covers built-in noise models, IBM device calibration, and custom noise specification.

## Why Noise Modeling?

Real quantum devices are noisy due to:
- **Decoherence:** T1 (energy relaxation) and T2 (dephasing) processes
- **Gate errors:** Imperfect control pulses
- **Readout errors:** Measurement classification errors
- **Leakage:** Transitions to non-computational states
- **Crosstalk:** Unwanted qubit interactions

LRET efficiently simulates these effects using low-rank decomposition.

---

## Built-in Noise Models

### 1. Depolarizing Noise

**Description:** Each gate is followed by a depolarizing channel with probability $p$:

$$
\mathcal{E}(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)
$$

**Use case:** General noise model; good for circuit robustness testing.

**CLI:**
```bash
quantum_sim -n 8 -d 20 --noise 0.01 --noise-type depolarizing
```

**Python:**
```python
from qlret import QuantumSimulator

sim = QuantumSimulator(n_qubits=8, noise_level=0.01, noise_type="depolarizing")
sim.h(0)
sim.cnot(0, 1)
print(f"Rank: {sim.current_rank}, Fidelity: {sim.fidelity:.4f}")
```

**Typical values:**
- **Low noise:** $p = 0.001$ (0.1% error per gate)
- **Medium noise:** $p = 0.01$ (1% error per gate)
- **High noise:** $p = 0.05$ (5% error per gate)

---

### 2. Amplitude Damping (T1 Decay)

**Description:** Models energy relaxation from |1⟩ to |0⟩ with rate $\Gamma = 1/T_1$:

$$
\mathcal{E}(\rho) = E_0 \rho E_0^\dagger + E_1 \rho E_1^\dagger
$$

where $E_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}$, $E_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$

**Use case:** Simulating energy decay in superconducting qubits.

**CLI:**
```bash
quantum_sim -n 8 -d 20 --noise 0.02 --noise-type amplitude_damping
```

**Python:**
```python
sim = QuantumSimulator(n_qubits=8, noise_level=0.02, noise_type="amplitude_damping")

# Or specify T1 times directly
noise_model = {
    "model_type": "amplitude_damping",
    "qubit_T1": [50e-6, 45e-6, 52e-6, 48e-6, 51e-6, 47e-6, 49e-6, 50e-6],  # seconds
    "gate_time": 50e-9  # 50 nanoseconds per gate
}
sim = QuantumSimulator(n_qubits=8, noise_model=noise_model)
```

**Typical T1 values:**
- **Superconducting qubits:** 20-100 μs
- **Ion traps:** 1-10 seconds
- **Neutral atoms:** 0.1-1 seconds

---

### 3. Phase Damping (T2 Dephasing)

**Description:** Models dephasing without energy loss:

$$
\mathcal{E}(\rho) = \begin{pmatrix} \rho_{00} & \sqrt{1-\lambda}\rho_{01} \\ \sqrt{1-\lambda}\rho_{10} & \rho_{11} \end{pmatrix}
$$

**Use case:** Simulating pure dephasing in quantum devices.

**CLI:**
```bash
quantum_sim -n 8 -d 20 --noise 0.01 --noise-type phase_damping
```

**Python:**
```python
# Using dephasing rate
sim = QuantumSimulator(n_qubits=8, noise_level=0.01, noise_type="phase_damping")

# Or specify T2 times
noise_model = {
    "model_type": "phase_damping",
    "qubit_T2": [70e-6, 65e-6, 68e-6, 66e-6, 72e-6, 69e-6, 67e-6, 71e-6],
    "gate_time": 50e-9
}
sim = QuantumSimulator(n_qubits=8, noise_model=noise_model)
```

**Typical T2 values:**
- **Superconducting qubits:** 20-200 μs (T2 ≤ 2·T1)
- **Ion traps:** 0.1-10 seconds
- **Neutral atoms:** 0.01-1 seconds

---

### 4. Mixed Noise Model

**Description:** Combines multiple noise sources (depolarizing + amplitude damping + phase damping).

**Python:**
```python
noise_model = {
    "model_type": "mixed",
    "global_depolarizing": 0.001,
    "amplitude_damping_rate": 0.005,
    "phase_damping_rate": 0.002
}

sim = QuantumSimulator(n_qubits=8, noise_model=noise_model)
```

---

## IBM Device Noise

### Downloading IBM Noise Data

```bash
# Install IBM Quantum provider
pip install qiskit-ibm-runtime

# Download noise model
python scripts/download_ibm_noise.py --device ibmq_manila --output manila_noise.json

# Available devices: ibmq_manila, ibmq_quito, ibmq_belem, ibm_perth, etc.
```

**Script usage:**
```bash
# Download with authentication
python scripts/download_ibm_noise.py \
    --device ibmq_manila \
    --token YOUR_IBM_TOKEN \
    --output manila_noise.json

# List available devices
python scripts/download_ibm_noise.py --list-devices
```

### JSON Format (IBM Device Noise)

```json
{
  "device_name": "ibmq_manila",
  "n_qubits": 5,
  "coupling_map": [[0, 1], [1, 2], [1, 3], [3, 4]],
  "gate_errors": {
    "u1": [0.00015, 0.00012, 0.00018, 0.00014, 0.00016],
    "u2": [0.00045, 0.00038, 0.00052, 0.00041, 0.00048],
    "u3": [0.00089, 0.00076, 0.00105, 0.00082, 0.00096],
    "cx": {
      "(0, 1)": 0.0124,
      "(1, 2)": 0.0098,
      "(1, 3)": 0.0135,
      "(3, 4)": 0.0112
    }
  },
  "qubit_T1": [0.000052, 0.000048, 0.000056, 0.000050, 0.000054],
  "qubit_T2": [0.000068, 0.000062, 0.000074, 0.000065, 0.000071],
  "readout_errors": [
    [0.0215, 0.0142],
    [0.0189, 0.0167],
    [0.0234, 0.0156],
    [0.0198, 0.0178],
    [0.0221, 0.0165]
  ]
}
```

### Using IBM Noise in Simulations

**CLI:**
```bash
quantum_sim -n 5 -d 30 --noise-file manila_noise.json -o ibm_sim.csv
```

**Python:**
```python
from qlret import QuantumSimulator, load_noise_model

noise = load_noise_model("manila_noise.json")
sim = QuantumSimulator(n_qubits=5, noise_model=noise)

# Build circuit
sim.h(0)
sim.cnot(0, 1)
sim.cnot(1, 2)

# Check fidelity
print(f"Fidelity with IBM noise: {sim.fidelity:.4f}")
```

**PennyLane:**
```python
import pennylane as qml
from qlret import QLRETDevice, load_noise_model

noise = load_noise_model("manila_noise.json")
dev = QLRETDevice(wires=5, noise_model=noise)

@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

result = circuit()
```

---

## Custom Noise Models

### Basic Custom Model

```python
import json

custom_noise = {
    "model_type": "custom",
    "global_depolarizing": 0.001,
    "gate_specific": {
        "H": {"depolarizing": 0.0005},
        "X": {"depolarizing": 0.0004},
        "CNOT": {"depolarizing": 0.01},
        "RX": {"amplitude_damping": 0.002, "phase_damping": 0.001},
        "RY": {"amplitude_damping": 0.002, "phase_damping": 0.001},
        "RZ": {"phase_damping": 0.0008}
    }
}

# Save to file
with open("custom_noise.json", "w") as f:
    json.dump(custom_noise, f, indent=2)

# Use in simulator
from qlret import QuantumSimulator
sim = QuantumSimulator(n_qubits=8, noise_model="custom_noise.json")
```

### Qubit-Specific Noise

```python
custom_noise = {
    "model_type": "qubit_specific",
    "qubit_properties": {
        "0": {
            "T1": 50e-6,
            "T2": 70e-6,
            "single_qubit_error": 0.0005,
            "readout_error": [0.02, 0.01]  # P(0|1), P(1|0)
        },
        "1": {
            "T1": 45e-6,
            "T2": 65e-6,
            "single_qubit_error": 0.0008,
            "readout_error": [0.025, 0.015]
        },
        # ... more qubits
    },
    "two_qubit_gates": {
        "CNOT": {
            "(0, 1)": 0.012,
            "(1, 2)": 0.009,
            "(2, 3)": 0.015
        }
    },
    "gate_time": 50e-9
}

sim = QuantumSimulator(n_qubits=4, noise_model=custom_noise)
```

### Time-Dependent Noise

```python
custom_noise = {
    "model_type": "time_dependent",
    "base_T1": 50e-6,
    "base_T2": 70e-6,
    "drift_rate": 0.01,  # 1% drift per 100 gates
    "gate_time": 50e-9
}

sim = QuantumSimulator(n_qubits=8, noise_model=custom_noise)
```

### Leakage Errors

```python
custom_noise = {
    "model_type": "with_leakage",
    "global_depolarizing": 0.001,
    "leakage_probability": {
        "RX": 0.0001,
        "RY": 0.0001,
        "CNOT": 0.0005
    },
    "leakage_recovery_time": 1e-6
}

sim = QuantumSimulator(n_qubits=8, noise_model=custom_noise)
```

---

## Noise Calibration

### Fitting Noise from Experimental Data

```bash
# Generate calibration data (run on real device or simulator)
python scripts/generate_calibration_data.py --device ibmq_manila --output calib_data.json

# Fit noise model
python scripts/calibrate_noise_model.py --input calib_data.json --output fitted_noise.json

# Validate fit
python scripts/calibrate_noise_model.py --input calib_data.json --validate
```

**Python API:**
```python
from qlret.calibration import fit_noise_model, validate_noise_model

# Load experimental data
import json
with open("experimental_data.json") as f:
    data = json.load(f)

# Fit depolarizing noise
fitted_noise = fit_noise_model(data, model_type="depolarizing")

# Validate
validation_results = validate_noise_model(fitted_noise, data)
print(f"Mean fidelity error: {validation_results['mean_error']:.6f}")
print(f"Max fidelity error: {validation_results['max_error']:.6f}")

# Save fitted model
with open("fitted_noise.json", "w") as f:
    json.dump(fitted_noise, f, indent=2)
```

### Fitting Specific Noise Types

**Depolarizing:**
```bash
python scripts/fit_depolarizing.py --input data.json --output fitted_depol.json
```

**T1/T2 Relaxation:**
```bash
python scripts/fit_t1_t2.py --input data.json --output fitted_t1t2.json
```

**Correlated Errors:**
```bash
python scripts/fit_correlated_errors.py --input data.json --output fitted_corr.json
```

---

## Noise Characterization

### Randomized Benchmarking

```python
from qlret import QuantumSimulator
import numpy as np

def randomized_benchmarking(n_qubits, sequence_lengths, n_trials, noise_level):
    """
    Perform randomized benchmarking to estimate average gate fidelity.
    """
    results = []
    
    for length in sequence_lengths:
        fidelities = []
        
        for _ in range(n_trials):
            sim = QuantumSimulator(n_qubits=n_qubits, noise_level=noise_level)
            
            # Apply random Clifford gates
            for _ in range(length):
                gate = np.random.choice(['H', 'X', 'Y', 'Z', 'S'])
                qubit = np.random.randint(n_qubits)
                
                if gate == 'H':
                    sim.h(qubit)
                elif gate == 'X':
                    sim.x(qubit)
                elif gate == 'Y':
                    sim.y(qubit)
                elif gate == 'Z':
                    sim.z(qubit)
                elif gate == 'S':
                    sim.s(qubit)
            
            fidelities.append(sim.fidelity)
        
        avg_fidelity = np.mean(fidelities)
        results.append({'length': length, 'fidelity': avg_fidelity})
    
    return results

# Run RB
results = randomized_benchmarking(
    n_qubits=2,
    sequence_lengths=[1, 5, 10, 20, 50, 100],
    n_trials=20,
    noise_level=0.01
)

# Plot
import matplotlib.pyplot as plt
lengths = [r['length'] for r in results]
fidelities = [r['fidelity'] for r in results]

plt.plot(lengths, fidelities, 'o-')
plt.xlabel('Sequence Length')
plt.ylabel('Average Fidelity')
plt.title('Randomized Benchmarking')
plt.grid(True)
plt.savefig('rb_results.png')

# Fit exponential decay: F(m) = A * p^m + B
from scipy.optimize import curve_fit

def decay(m, A, p, B):
    return A * p**m + B

popt, _ = curve_fit(decay, lengths, fidelities)
print(f"Fitted parameters: A={popt[0]:.4f}, p={popt[1]:.6f}, B={popt[2]:.4f}")
print(f"Average gate fidelity: {popt[1]:.6f}")
```

### Process Tomography

```python
from qlret import QuantumSimulator
import numpy as np

def process_tomography(gate_func, n_qubits=1):
    """
    Perform quantum process tomography for single-qubit gate.
    """
    # Basis states for single qubit: |0⟩, |1⟩, |+⟩, |+i⟩
    basis_states = [
        lambda sim: None,  # |0⟩
        lambda sim: sim.x(0),  # |1⟩
        lambda sim: sim.h(0),  # |+⟩
        lambda sim: (sim.h(0), sim.s(0))  # |+i⟩
    ]
    
    chi_matrix = np.zeros((4, 4), dtype=complex)
    
    for i, prep in enumerate(basis_states):
        for j, meas in enumerate(basis_states):
            sim = QuantumSimulator(n_qubits=1, noise_level=0.01)
            
            # Prepare
            if prep is not None:
                prep(sim)
            
            # Apply gate
            gate_func(sim)
            
            # Measure (simplified)
            rho = sim.get_density_matrix()
            chi_matrix[i, j] = np.trace(rho)
    
    return chi_matrix

# Example: H gate tomography
def h_gate(sim):
    sim.h(0)

chi = process_tomography(h_gate)
print("Chi matrix (process matrix):")
print(chi)
```

---

## Best Practices

### 1. Choose Realistic Noise Levels

```python
# Too optimistic (unrealistic)
sim = QuantumSimulator(n_qubits=10, noise_level=0.0001)

# Realistic for current devices
sim = QuantumSimulator(n_qubits=10, noise_level=0.01)

# Pessimistic (useful for robustness testing)
sim = QuantumSimulator(n_qubits=10, noise_level=0.05)
```

### 2. Use Device-Specific Noise

```python
# Generic noise
sim_generic = QuantumSimulator(n_qubits=5, noise_level=0.01)

# IBM device noise (more realistic)
sim_ibm = QuantumSimulator(n_qubits=5, noise_model="ibmq_manila.json")
```

### 3. Validate Noise Models

```bash
# Compare simulation vs experimental data
python scripts/compare_fidelities.py \
    --simulation results_sim.json \
    --experimental results_exp.json \
    --output comparison.png
```

### 4. Monitor Rank Growth

```python
sim = QuantumSimulator(n_qubits=8, noise_level=0.01, verbose=True)

for i in range(50):
    sim.h(i % 8)
    sim.cnot(i % 8, (i+1) % 8)
    
    if i % 10 == 0:
        print(f"Step {i}: Rank = {sim.current_rank}")
```

---

## Troubleshooting

### Noise Too Low (Rank Explodes)

**Problem:** Rank grows exponentially with low noise.

**Solution:**
```python
# Increase noise level
sim = QuantumSimulator(n_qubits=10, noise_level=0.005)  # Was 0.0001

# Or use stricter truncation
sim = QuantumSimulator(n_qubits=10, noise_level=0.001, truncation_threshold=1e-3)
```

### Fidelity Too Low

**Problem:** Circuit fidelity drops below acceptable level.

**Solution:**
```python
# Reduce noise
sim = QuantumSimulator(n_qubits=8, noise_level=0.005)  # Was 0.02

# Or use more lenient truncation
sim = QuantumSimulator(n_qubits=8, noise_level=0.01, truncation_threshold=1e-5)
```

### IBM Device Import Fails

**Problem:** `download_ibm_noise.py` fails.

**Solution:**
```bash
# Check credentials
export QISKIT_IBM_TOKEN="your_token_here"

# Verify device name
python scripts/download_ibm_noise.py --list-devices

# Use fallback
quantum_sim -n 5 -d 30 --noise 0.01 --noise-type depolarizing
```

---

## See Also

- **[Installation Guide](01-installation.md)** - Setting up LRET
- **[Python Interface](04-python-interface.md)** - Python API reference
- **[PennyLane Integration](05-pennylane-integration.md)** - Hybrid algorithms
- **[Troubleshooting](08-troubleshooting.md)** - Common issues
