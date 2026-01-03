# LRETSimulator Class

Main simulator class for quantum state evolution using the LRET algorithm.

## Header

```cpp
#include "simulator.h"
```

## Declaration

```cpp
class LRETSimulator {
public:
    // Constructors
    LRETSimulator(int n_qubits, double noise_level = 0.0);
    LRETSimulator(int n_qubits, double noise_level, double truncation_threshold);
    
    // Gate operations
    void apply_gate(const Gate& gate);
    void apply_circuit(const std::vector<Gate>& gates);
    
    // Noise operations
    void apply_noise(const NoiseChannel& noise);
    void apply_global_noise(const NoiseChannel& noise);
    
    // State management
    void reset();
    MatrixXcd get_density_matrix() const;
    VectorXcd get_state_vector() const;
    
    // Measurement
    std::map<std::string, int> measure_all(int shots = 1);
    double get_probability(const std::string& outcome) const;
    std::vector<double> get_probabilities() const;
    
    // Properties
    int get_num_qubits() const;
    int get_rank() const;
    double get_fidelity() const;
    double get_noise_level() const;
    
    // Configuration
    void set_truncation_threshold(double threshold);
    void set_max_rank(int max_rank);
    void set_parallel_mode(ParallelMode mode);
    
    // Advanced
    MatrixXcd get_choi_representation() const;
    void truncate_rank();
    
private:
    MatrixXcd L_;  // Low-rank factor
    int n_qubits_;
    int current_rank_;
    double noise_level_;
    double truncation_threshold_;
    int max_rank_;
    ParallelMode parallel_mode_;
    
    void apply_gate_choi(const MatrixXcd& choi, const std::vector<int>& targets);
    MatrixXcd compute_full_choi(const MatrixXcd& local_choi, const std::vector<int>& targets);
};
```

---

## Constructors

### `LRETSimulator(int n_qubits, double noise_level = 0.0)`

Construct a simulator with specified number of qubits and global noise level.

**Parameters:**
- `n_qubits` - Number of qubits (must be ≥ 1)
- `noise_level` - Global depolarizing noise level (default: 0.0, range: [0, 1])

**Throws:**
- `std::invalid_argument` - If `n_qubits < 1` or `noise_level` out of range

**Example:**
```cpp
LRETSimulator sim(4, 0.01);  // 4 qubits, 1% depolarizing noise
```

---

### `LRETSimulator(int n_qubits, double noise_level, double truncation_threshold)`

Construct a simulator with custom truncation threshold.

**Parameters:**
- `n_qubits` - Number of qubits
- `noise_level` - Global depolarizing noise level
- `truncation_threshold` - Fidelity threshold for rank truncation (default: 1e-6)

**Example:**
```cpp
LRETSimulator sim(10, 0.01, 1e-8);  // Higher fidelity threshold
```

---

## Gate Operations

### `void apply_gate(const Gate& gate)`

Apply a quantum gate to the state.

**Parameters:**
- `gate` - Gate to apply (must have valid targets within [0, n_qubits))

**Throws:**
- `std::invalid_argument` - If gate targets are invalid

**Complexity:** $O(4^{n-k} \cdot 16^k \cdot r)$ for $k$-qubit gate

**Example:**
```cpp
sim.apply_gate(hadamard_gate(0));
sim.apply_gate(cnot_gate(0, 1));
sim.apply_gate(rotation_x_gate(2, M_PI / 4));
```

---

### `void apply_circuit(const std::vector<Gate>& gates)`

Apply a sequence of gates.

**Parameters:**
- `gates` - Vector of gates to apply in order

**Example:**
```cpp
std::vector<Gate> circuit = {
    hadamard_gate(0),
    cnot_gate(0, 1),
    rotation_z_gate(0, M_PI / 2)
};
sim.apply_circuit(circuit);
```

---

## Noise Operations

### `void apply_noise(const NoiseChannel& noise)`

Apply a noise channel to specific qubits.

**Parameters:**
- `noise` - Noise channel with specified target qubits

**Example:**
```cpp
NoiseChannel noise = depolarizing_noise(0, 0.01);
sim.apply_noise(noise);
```

---

### `void apply_global_noise(const NoiseChannel& noise)`

Apply a noise channel to all qubits.

**Parameters:**
- `noise` - Noise channel (targets will be ignored)

**Example:**
```cpp
NoiseChannel noise = amplitude_damping(0, 0.05);
sim.apply_global_noise(noise);  // Applied to all qubits
```

---

## State Management

### `void reset()`

Reset the simulator to the initial state $|0\rangle^{\otimes n}$.

**Example:**
```cpp
sim.reset();
```

---

### `MatrixXcd get_density_matrix() const`

Get the full density matrix representation.

**Returns:** Complex matrix of size $2^n \times 2^n$

**Warning:** Expensive for large $n$ (exponential memory)

**Example:**
```cpp
MatrixXcd rho = sim.get_density_matrix();
std::cout << "Trace: " << rho.trace() << std::endl;  // Should be 1.0
```

---

### `VectorXcd get_state_vector() const`

Get the state vector (only valid for pure states).

**Returns:** Complex vector of size $2^n$

**Throws:** 
- `std::runtime_error` - If state is not pure (rank > 1)

**Example:**
```cpp
try {
    VectorXcd psi = sim.get_state_vector();
    double norm = psi.norm();
    std::cout << "Norm: " << norm << std::endl;
} catch (const std::runtime_error& e) {
    std::cerr << "State is mixed" << std::endl;
}
```

---

## Measurement

### `std::map<std::string, int> measure_all(int shots = 1)`

Measure all qubits in computational basis.

**Parameters:**
- `shots` - Number of measurement shots (default: 1)

**Returns:** Map from outcome strings (e.g., "0101") to counts

**Example:**
```cpp
auto results = sim.measure_all(1000);
for (const auto& [outcome, count] : results) {
    std::cout << outcome << ": " << count << std::endl;
}
```

---

### `double get_probability(const std::string& outcome) const`

Get probability of a specific measurement outcome.

**Parameters:**
- `outcome` - Outcome string (e.g., "0101")

**Returns:** Probability in [0, 1]

**Throws:**
- `std::invalid_argument` - If outcome string is invalid

**Example:**
```cpp
double prob_0000 = sim.get_probability("0000");
double prob_1111 = sim.get_probability("1111");
```

---

### `std::vector<double> get_probabilities() const`

Get probabilities of all measurement outcomes.

**Returns:** Vector of $2^n$ probabilities

**Warning:** Expensive for large $n$

**Example:**
```cpp
auto probs = sim.get_probabilities();
double sum = std::accumulate(probs.begin(), probs.end(), 0.0);
// sum should be 1.0
```

---

## Properties

### `int get_num_qubits() const`

Get the number of qubits.

**Returns:** Number of qubits

---

### `int get_rank() const`

Get the current rank of the state representation.

**Returns:** Current rank $r$

**Example:**
```cpp
std::cout << "Current rank: " << sim.get_rank() << std::endl;
```

---

### `double get_fidelity() const`

Get the fidelity between exact and approximate states.

**Returns:** Fidelity in [0, 1] (1.0 = exact)

**Note:** Only meaningful after truncation

---

### `double get_noise_level() const`

Get the global noise level.

**Returns:** Noise level

---

## Configuration

### `void set_truncation_threshold(double threshold)`

Set the truncation fidelity threshold.

**Parameters:**
- `threshold` - Fidelity threshold (e.g., 1e-6)

**Example:**
```cpp
sim.set_truncation_threshold(1e-8);  // Higher accuracy
```

---

### `void set_max_rank(int max_rank)`

Set the maximum allowed rank.

**Parameters:**
- `max_rank` - Maximum rank

**Example:**
```cpp
sim.set_max_rank(200);  // Allow rank up to 200
```

---

### `void set_parallel_mode(ParallelMode mode)`

Set the parallelization mode.

**Parameters:**
- `mode` - Parallel mode (SEQUENTIAL, ROW_PARALLEL, COLUMN_PARALLEL, HYBRID)

**Example:**
```cpp
sim.set_parallel_mode(ParallelMode::HYBRID);
```

---

## Advanced Methods

### `MatrixXcd get_choi_representation() const`

Get the Choi matrix representation of the current state.

**Returns:** Choi matrix of size $4^n \times 4^n$

**Warning:** Expensive for large $n$

---

### `void truncate_rank()`

Manually trigger rank truncation.

**Note:** Truncation is usually automatic, but can be forced for memory management

**Example:**
```cpp
// After many gates
sim.truncate_rank();
std::cout << "Rank after truncation: " << sim.get_rank() << std::endl;
```

---

## Examples

### Bell State

```cpp
#include "simulator.h"
#include "gates_and_noise.h"

int main() {
    LRETSimulator sim(2, 0.0);
    
    // Create Bell state: (|00⟩ + |11⟩) / √2
    sim.apply_gate(hadamard_gate(0));
    sim.apply_gate(cnot_gate(0, 1));
    
    // Measure
    auto results = sim.measure_all(1000);
    
    std::cout << "Results:" << std::endl;
    for (const auto& [outcome, count] : results) {
        std::cout << outcome << ": " << count << std::endl;
    }
    
    return 0;
}
```

**Output:**
```
Results:
00: 502
11: 498
```

---

### Noisy Simulation

```cpp
#include "simulator.h"
#include "gates_and_noise.h"

int main() {
    LRETSimulator sim(4, 0.01);  // 1% noise
    
    // Apply circuit with noise
    sim.apply_gate(hadamard_gate(0));
    sim.apply_noise(depolarizing_noise(0, 0.05));  // Additional noise
    
    sim.apply_gate(cnot_gate(0, 1));
    sim.apply_global_noise(amplitude_damping(0, 0.02));
    
    std::cout << "Final rank: " << sim.get_rank() << std::endl;
    std::cout << "Fidelity: " << sim.get_fidelity() << std::endl;
    
    return 0;
}
```

---

## See Also

- [Gate Reference](gates.md) - Available quantum gates
- [Noise Reference](noise.md) - Noise channel definitions
- [Examples](../../examples/cpp/) - More code examples
- [User Guide](../../user-guide/02-quick-start.md) - Quick start guide
