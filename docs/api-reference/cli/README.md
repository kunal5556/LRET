# CLI Reference

Command-line interface reference for the LRET quantum simulator.

## Synopsis

```bash
quantum_sim [OPTIONS]
```

## Description

The `quantum_sim` command-line tool provides a convenient interface to run quantum circuit simulations using the LRET algorithm. It supports JSON circuit input, various output formats, and extensive configuration options.

---

## Basic Options

### `-h, --help`

Show help message and exit.

```bash
quantum_sim --help
```

---

### `-v, --version`

Show version information.

```bash
quantum_sim --version
```

**Output:**
```
LRET Quantum Simulator v1.0.0
```

---

## Circuit Specification

### `-i, --input <file>`

Load circuit from JSON file.

```bash
quantum_sim --input circuit.json
```

**JSON format:**
```json
{
  "qubits": 4,
  "gates": [
    {"type": "H", "targets": [0]},
    {"type": "CNOT", "targets": [0, 1]},
    {"type": "RX", "targets": [2], "params": [1.5708]}
  ],
  "shots": 1000
}
```

---

### `-q, --qubits <n>`

Number of qubits (for random circuits).

```bash
quantum_sim --qubits 10
```

---

### `-d, --depth <d>`

Circuit depth (for random circuits).

```bash
quantum_sim --qubits 10 --depth 50
```

---

### `-c, --circuit <string>`

Specify circuit as command-line string.

```bash
quantum_sim --qubits 2 --circuit "H 0; CNOT 0 1"
```

**Syntax:**
- Gates separated by semicolons
- Format: `GATE target1 [target2] [param1] [param2]`

**Examples:**
```bash
# Bell state
quantum_sim -q 2 -c "H 0; CNOT 0 1"

# GHZ state
quantum_sim -q 3 -c "H 0; CNOT 0 1; CNOT 0 2"

# Parametric gates
quantum_sim -q 2 -c "RX 0 1.5708; RY 1 0.7854"
```

---

## Noise Configuration

### `-n, --noise <level>`

Global depolarizing noise level (0 to 1).

```bash
quantum_sim --qubits 4 --depth 20 --noise 0.01
```

---

### `--noise-model <file>`

Load custom noise model from JSON.

```bash
quantum_sim --input circuit.json --noise-model ibmq_noise.json
```

**Noise model JSON format:**
```json
{
  "single_qubit_gates": {
    "depolarizing": 0.001,
    "amplitude_damping": 0.002
  },
  "two_qubit_gates": {
    "depolarizing": 0.01
  },
  "readout_error": 0.02
}
```

---

### `--t1 <time>`, `--t2 <time>`

Relaxation and dephasing times (in microseconds).

```bash
quantum_sim --input circuit.json --t1 50 --t2 30
```

---

## Measurement

### `-s, --shots <n>`

Number of measurement shots.

```bash
quantum_sim --input circuit.json --shots 10000
```

---

### `--measure <qubits>`

Measure specific qubits (comma-separated).

```bash
quantum_sim --input circuit.json --measure 0,2,3
```

---

## Output Options

### `-o, --output <file>`

Output file for results.

```bash
quantum_sim --input circuit.json --output results.json
```

---

### `-f, --format <format>`

Output format: `json`, `csv`, `txt`, `hdf5`.

```bash
quantum_sim --input circuit.json --format csv --output results.csv
```

**Formats:**

**JSON:**
```json
{
  "results": {"00": 487, "11": 513},
  "metadata": {
    "qubits": 2,
    "depth": 2,
    "final_rank": 1,
    "simulation_time": 0.0023
  }
}
```

**CSV:**
```csv
outcome,count,probability
00,487,0.487
11,513,0.513
```

**TXT:**
```
Results:
00: 487 (48.7%)
11: 513 (51.3%)

Metadata:
  Qubits: 2
  Final rank: 1
  Simulation time: 0.0023s
```

---

### `--progressive-csv`

Enable progressive CSV output (streaming).

```bash
quantum_sim --input large_circuit.json --progressive-csv --output results.csv
```

Useful for long-running simulations to monitor progress.

---

### `--verbose`

Enable verbose output.

```bash
quantum_sim --input circuit.json --verbose
```

**Output:**
```
Loading circuit from circuit.json...
Initializing simulator with 4 qubits...
Applying gate H to qubit 0... (rank: 1)
Applying gate CNOT to qubits [0, 1]... (rank: 1)
Measuring 1000 shots...
Writing results to results.json...
Done. Total time: 0.245s
```

---

### `--quiet`

Suppress all output except errors.

```bash
quantum_sim --input circuit.json --quiet
```

---

## Performance Options

### `--threads <n>`

Number of OpenMP threads (0 = auto).

```bash
quantum_sim --input circuit.json --threads 8
```

---

### `--parallel-mode <mode>`

Parallelization strategy: `sequential`, `row`, `column`, `hybrid`.

```bash
quantum_sim --input circuit.json --parallel-mode hybrid
```

---

### `--max-rank <r>`

Maximum allowed rank.

```bash
quantum_sim --input circuit.json --max-rank 200
```

---

### `--truncation-threshold <eps>`

Truncation fidelity threshold.

```bash
quantum_sim --input circuit.json --truncation-threshold 1e-8
```

---

### `--no-optimize`

Disable circuit optimization.

```bash
quantum_sim --input circuit.json --no-optimize
```

---

### `--gpu`

Enable GPU acceleration (if compiled with CUDA).

```bash
quantum_sim --input circuit.json --gpu
```

---

### `--mpi`

Enable MPI distributed computing.

```bash
mpirun -n 4 quantum_sim --input circuit.json --mpi
```

---

## Monitoring and Profiling

### `--monitor`

Enable resource monitoring.

```bash
quantum_sim --input circuit.json --monitor --output-monitor stats.json
```

**Monitor output:**
```json
{
  "peak_memory_mb": 245.7,
  "average_memory_mb": 198.3,
  "cpu_time_s": 12.45,
  "wall_time_s": 3.21,
  "rank_evolution": [1, 2, 4, 8, 12, 10, 8]
}
```

---

### `--profile`

Enable profiling output.

```bash
quantum_sim --input circuit.json --profile --output-profile profile.json
```

**Profile output:**
```json
{
  "gate_times": {
    "H": 0.0023,
    "CNOT": 0.0156,
    "RX": 0.0031
  },
  "truncation_time": 0.245,
  "measurement_time": 0.089
}
```

---

## Benchmarking

### `--benchmark`

Run benchmarking mode.

```bash
quantum_sim --benchmark --qubits 15 --depth 50 --shots 1000
```

---

### `--benchmark-suite <suite>`

Run predefined benchmark suite: `scaling`, `parallel`, `accuracy`, `depth`.

```bash
quantum_sim --benchmark-suite scaling --output benchmark_results.json
```

---

## Examples

### 1. Bell State Simulation

```bash
quantum_sim --qubits 2 \
  --circuit "H 0; CNOT 0 1" \
  --shots 1000 \
  --output bell_results.json
```

---

### 2. Noisy Circuit from JSON

```bash
quantum_sim \
  --input vqe_circuit.json \
  --noise 0.01 \
  --shots 10000 \
  --format csv \
  --output vqe_results.csv
```

---

### 3. Large-Scale Simulation with GPU

```bash
quantum_sim \
  --input deep_circuit.json \
  --gpu \
  --max-rank 500 \
  --threads 16 \
  --progressive-csv \
  --output results.csv \
  --monitor \
  --verbose
```

---

### 4. Distributed MPI Simulation

```bash
mpirun -n 8 quantum_sim \
  --input large_circuit.json \
  --mpi \
  --max-rank 1000 \
  --output results.json
```

---

### 5. Benchmarking

```bash
quantum_sim \
  --benchmark-suite scaling \
  --output benchmark.json \
  --format json
```

---

### 6. Custom Noise Model

```bash
quantum_sim \
  --input circuit.json \
  --noise-model ibmq_manila.json \
  --shots 10000 \
  --output noisy_results.json
```

---

## Environment Variables

### `QLRET_THREADS`

Default number of threads (overridden by `--threads`).

```bash
export QLRET_THREADS=8
quantum_sim --input circuit.json
```

---

### `QLRET_GPU_DEVICE`

GPU device ID (for multi-GPU systems).

```bash
export QLRET_GPU_DEVICE=1
quantum_sim --input circuit.json --gpu
```

---

### `QLRET_CACHE_DIR`

Directory for caching compiled circuits.

```bash
export QLRET_CACHE_DIR=/tmp/qlret_cache
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Invalid arguments |
| 2 | File not found |
| 3 | Parse error |
| 4 | Simulation error |
| 5 | Memory error |
| 6 | Hardware error (GPU/MPI) |

---

## JSON Circuit Format

### Basic Structure

```json
{
  "qubits": <number>,
  "gates": [ <gate1>, <gate2>, ... ],
  "noise": <noise_config>,
  "measurement": <measurement_config>,
  "shots": <number>
}
```

---

### Gate Specification

**Single-qubit gates:**
```json
{"type": "H", "targets": [0]}
{"type": "X", "targets": [1]}
{"type": "RX", "targets": [2], "params": [1.5708]}
```

**Two-qubit gates:**
```json
{"type": "CNOT", "targets": [0, 1]}
{"type": "CZ", "targets": [2, 3]}
{"type": "CRX", "targets": [0, 1], "params": [0.7854]}
```

**Three-qubit gates:**
```json
{"type": "TOFFOLI", "targets": [0, 1, 2]}
{"type": "FREDKIN", "targets": [0, 1, 2]}
```

---

### Noise Specification

```json
{
  "noise": {
    "global_depolarizing": 0.01,
    "per_gate": [
      {"gate_index": 2, "type": "amplitude_damping", "params": [0.05]},
      {"gate_index": 5, "type": "phase_damping", "params": [0.02]}
    ]
  }
}
```

---

### Complete Example

```json
{
  "qubits": 4,
  "gates": [
    {"type": "H", "targets": [0]},
    {"type": "H", "targets": [1]},
    {"type": "CNOT", "targets": [0, 2]},
    {"type": "CNOT", "targets": [1, 3]},
    {"type": "RZ", "targets": [0], "params": [1.5708]},
    {"type": "RZ", "targets": [1], "params": [1.5708]},
    {"type": "CNOT", "targets": [0, 2]},
    {"type": "CNOT", "targets": [1, 3]},
    {"type": "H", "targets": [0]},
    {"type": "H", "targets": [1]}
  ],
  "noise": {
    "global_depolarizing": 0.001,
    "readout_error": 0.02
  },
  "measurement": {
    "qubits": [0, 1, 2, 3],
    "basis": "computational"
  },
  "shots": 10000
}
```

---

## Tips and Best Practices

### Performance Optimization

1. **Use appropriate rank limits:**
   ```bash
   quantum_sim --max-rank 100  # For moderate accuracy
   quantum_sim --max-rank 500  # For high accuracy
   ```

2. **Enable parallelization:**
   ```bash
   quantum_sim --threads 0 --parallel-mode hybrid
   ```

3. **Use GPU for large circuits:**
   ```bash
   quantum_sim --gpu --max-rank 1000
   ```

### Memory Management

1. **Monitor memory usage:**
   ```bash
   quantum_sim --monitor --output-monitor stats.json
   ```

2. **Adjust truncation:**
   ```bash
   quantum_sim --truncation-threshold 1e-5  # Lower memory
   quantum_sim --truncation-threshold 1e-9  # Higher memory
   ```

### Debugging

1. **Verbose output:**
   ```bash
   quantum_sim --verbose --input circuit.json
   ```

2. **Dry run (parse only):**
   ```bash
   quantum_sim --input circuit.json --dry-run
   ```

---

## See Also

- [Python API](../python/simulator.md) - Python interface
- [C++ API](../cpp/simulator.md) - C++ library
- [User Guide](../../user-guide/03-cli-reference.md) - Detailed CLI guide
- [Examples](../../examples/) - Example circuits
