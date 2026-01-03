# CLI Reference

Complete command-line reference for the `quantum_sim` tool.

## Synopsis

```bash
quantum_sim [OPTIONS]
```

## Quick Examples

```bash
# Basic simulation
quantum_sim -n 8 -d 20 --noise 0.01

# Custom output
quantum_sim -n 10 -d 30 --noise 0.02 -o results.csv --format json

# Benchmark mode
quantum_sim --benchmark --qubits 8,10,12 --depths 20,30,40

# IBM device simulation
quantum_sim -n 5 -d 30 --noise-file ibmq_manila.json

# Compare FDM vs LRET
quantum_sim -n 10 -d 30 --noise 0.01 --compare-fdm
```

---

## Core Options

### Circuit Configuration

#### `-n, --qubits <N>`
Number of qubits to simulate.

- **Type:** Integer
- **Range:** 1-20 (practical limit depends on memory)
- **Default:** 8
- **Example:** `-n 12`

```bash
quantum_sim -n 10 -d 30 --noise 0.01
```

#### `-d, --depth <D>`
Circuit depth (number of gate layers).

- **Type:** Integer
- **Range:** 1-10000
- **Default:** 20
- **Example:** `-d 50`

```bash
quantum_sim -n 8 -d 100 --noise 0.01
```

#### `--gates <GATES>`
Comma-separated list of gates to use in random circuit generation.

- **Type:** String (comma-separated)
- **Available gates:** `H, X, Y, Z, RX, RY, RZ, CNOT, CZ, SWAP, TOFFOLI`
- **Default:** `H,CNOT,RX,RY,RZ`
- **Example:** `--gates H,CNOT`

```bash
# Only Hadamard and CNOT gates
quantum_sim -n 8 -d 30 --gates H,CNOT --noise 0.01

# Single-qubit gates only
quantum_sim -n 10 -d 50 --gates H,X,Y,Z,RX,RY,RZ --noise 0.005
```

#### `--circuit-file <FILE>`
Load circuit from JSON file instead of generating random circuit.

- **Type:** File path
- **Format:** JSON circuit specification
- **Example:** `--circuit-file my_circuit.json`

```bash
quantum_sim --circuit-file vqe_ansatz.json --noise 0.01
```

**JSON Format:**
```json
{
  "n_qubits": 4,
  "gates": [
    {"type": "H", "target": 0},
    {"type": "CNOT", "control": 0, "target": 1},
    {"type": "RY", "target": 2, "angle": 0.5},
    {"type": "CNOT", "control": 1, "target": 2}
  ]
}
```

---

### Noise Configuration

#### `--noise <LEVEL>`
Global depolarizing noise level (probability).

- **Type:** Float
- **Range:** 0.0-1.0
- **Default:** 0.0 (noiseless)
- **Example:** `--noise 0.01` (1% noise)

```bash
# 1% depolarizing noise
quantum_sim -n 8 -d 20 --noise 0.01

# 5% noise (very noisy)
quantum_sim -n 8 -d 20 --noise 0.05
```

#### `--noise-file <FILE>`
Load noise model from JSON file (overrides `--noise`).

- **Type:** File path
- **Format:** JSON noise specification
- **Example:** `--noise-file ibm_device.json`

```bash
# Use IBM device noise
quantum_sim -n 5 -d 30 --noise-file ibmq_manila.json

# Custom noise model
quantum_sim -n 8 -d 20 --noise-file custom_noise.json
```

**JSON Format:**
```json
{
  "model_type": "device",
  "global_depolarizing": 0.001,
  "gate_errors": {
    "H": 0.0005,
    "CNOT": 0.01,
    "RX": 0.0008
  },
  "qubit_T1": [50e-6, 45e-6, 52e-6],
  "qubit_T2": [70e-6, 65e-6, 68e-6],
  "readout_errors": [[0.02, 0.01], [0.015, 0.025]]
}
```

#### `--noise-type <TYPE>`
Type of noise model to use.

- **Type:** String
- **Options:** `depolarizing`, `amplitude_damping`, `phase_damping`, `mixed`
- **Default:** `depolarizing`
- **Example:** `--noise-type amplitude_damping`

```bash
# Amplitude damping (T1 decay)
quantum_sim -n 8 -d 20 --noise 0.01 --noise-type amplitude_damping

# Mixed noise model
quantum_sim -n 8 -d 20 --noise 0.01 --noise-type mixed
```

---

### Parallelization

#### `--mode <MODE>`
Parallelization mode.

- **Type:** String
- **Options:**
  - `sequential`: Single-threaded (baseline)
  - `row`: Parallelize row updates (best for small rank)
  - `column`: Parallelize column updates (best for large rank)
  - `hybrid`: Auto-select based on rank (recommended)
- **Default:** `hybrid`
- **Example:** `--mode hybrid`

```bash
# Hybrid mode (recommended)
quantum_sim -n 10 -d 30 --noise 0.01 --mode hybrid

# Force row parallelization
quantum_sim -n 10 -d 30 --noise 0.01 --mode row

# Benchmark all modes
quantum_sim -n 10 -d 30 --noise 0.01 --benchmark-modes
```

#### `--threads <N>`
Number of OpenMP threads (overrides `OMP_NUM_THREADS`).

- **Type:** Integer
- **Range:** 1 to available cores
- **Default:** System default (usually all cores)
- **Example:** `--threads 8`

```bash
# Use 8 threads
quantum_sim -n 10 -d 30 --noise 0.01 --threads 8

# Single-threaded (for profiling)
quantum_sim -n 8 -d 20 --noise 0.01 --threads 1
```

#### `--batch-size <SIZE>`
Batch size for parallel processing.

- **Type:** Integer
- **Range:** 1-1024
- **Default:** Auto-selected based on qubit count
- **Example:** `--batch-size 64`

```bash
# Manual batch size
quantum_sim -n 10 -d 30 --noise 0.01 --batch-size 128

# Auto-select (recommended)
quantum_sim -n 10 -d 30 --noise 0.01  # Uses heuristic
```

---

### LRET Algorithm

#### `--threshold <T>`
SVD truncation threshold for rank reduction.

- **Type:** Float
- **Range:** 1e-10 to 1e-2
- **Default:** 1e-4
- **Example:** `--threshold 1e-5`

```bash
# Stricter truncation (higher fidelity, larger rank)
quantum_sim -n 10 -d 30 --noise 0.01 --threshold 1e-6

# Looser truncation (lower fidelity, smaller rank, faster)
quantum_sim -n 10 -d 30 --noise 0.01 --threshold 1e-3
```

**Effect on fidelity:**
| Threshold | Typical Fidelity | Rank Growth |
|-----------|------------------|-------------|
| 1e-6      | 0.9999           | High        |
| 1e-4      | 0.9997           | Medium      |
| 1e-3      | 0.999            | Low         |
| 1e-2      | 0.99             | Very low    |

#### `--no-truncation`
Disable rank truncation (rank grows unbounded).

- **Type:** Flag (boolean)
- **Default:** false (truncation enabled)

```bash
# Disable truncation
quantum_sim -n 8 -d 20 --noise 0.01 --no-truncation
```

**Warning:** Without truncation, memory usage and runtime grow exponentially!

#### `--max-rank <R>`
Maximum allowed rank (hard limit).

- **Type:** Integer
- **Range:** 1-10000
- **Default:** No limit
- **Example:** `--max-rank 100`

```bash
# Cap rank at 50
quantum_sim -n 10 -d 30 --noise 0.01 --max-rank 50
```

---

### Output Options

#### `-o, --output <FILE>`
Output file path for simulation results.

- **Type:** File path
- **Format:** Determined by extension or `--format`
- **Default:** stdout
- **Example:** `-o results.csv`

```bash
# CSV output
quantum_sim -n 8 -d 20 --noise 0.01 -o results.csv

# JSON output
quantum_sim -n 8 -d 20 --noise 0.01 -o results.json
```

#### `--format <FORMAT>`
Output format (overrides file extension).

- **Type:** String
- **Options:** `csv`, `json`, `hdf5`
- **Default:** Inferred from filename
- **Example:** `--format json`

```bash
# Force JSON format
quantum_sim -n 8 -d 20 --noise 0.01 -o data.txt --format json

# HDF5 for large datasets
quantum_sim -n 12 -d 50 --noise 0.01 -o data.h5 --format hdf5
```

**Output Fields:**
- `qubit_count`: Number of qubits
- `circuit_depth`: Depth of circuit
- `noise_level`: Noise parameter
- `final_rank`: Rank after simulation
- `simulation_time`: Wall-clock time (seconds)
- `fidelity`: Fidelity vs noiseless (if computed)
- `memory_mb`: Peak memory usage (MB)
- `speedup`: Speedup vs FDM (if computed)

#### `--save-state`
Save final quantum state to file.

- **Type:** Flag (boolean)
- **Output:** `state_<timestamp>.npy` (NumPy format)

```bash
quantum_sim -n 8 -d 20 --noise 0.01 --save-state
# Creates: state_20240104_153022.npy
```

#### `--save-diagram`
Save ASCII circuit diagram to file.

- **Type:** Flag (boolean)
- **Output:** `circuit_<timestamp>.txt`

```bash
quantum_sim -n 4 -d 10 --save-diagram
# Creates: circuit_20240104_153022.txt
```

#### `--verbose, -v`
Enable verbose output (gate-by-gate progress).

- **Type:** Flag (boolean)
- **Default:** false

```bash
quantum_sim -n 6 -d 15 --noise 0.01 --verbose
```

**Verbose Output:**
```
Gate 1/15: H on qubit 0 | Rank: 2 → 2 | Time: 0.001s
Gate 2/15: CNOT on qubits 0,1 | Rank: 2 → 4 | Time: 0.002s
Gate 3/15: RY(0.5) on qubit 2 | Rank: 4 → 4 | Time: 0.001s
...
```

#### `--quiet, -q`
Suppress all output except errors.

- **Type:** Flag (boolean)
- **Default:** false

```bash
quantum_sim -n 8 -d 20 --noise 0.01 -o results.csv --quiet
```

---

### Comparison & Validation

#### `--compare-fdm`
Run both FDM and LRET, compare results.

- **Type:** Flag (boolean)
- **Output:** Speedup and fidelity comparison

```bash
quantum_sim -n 10 -d 30 --noise 0.01 --compare-fdm
```

**Output:**
```
FDM Simulation:
  Time: 1.234 seconds
  Memory: 512 MB

LRET Simulation:
  Time: 0.189 seconds
  Memory: 87 MB
  Final rank: 23

Comparison:
  Speedup: 6.5x
  Memory reduction: 5.9x
  Fidelity: 0.9997
  Trace distance: 3.2e-4
```

#### `--fidelity-check`
Compute fidelity against noiseless simulation.

- **Type:** Flag (boolean)
- **Note:** Runs noiseless simulation for comparison

```bash
quantum_sim -n 8 -d 20 --noise 0.01 --fidelity-check
```

#### `--validate`
Run validation tests (compare against known results).

- **Type:** Flag (boolean)

```bash
quantum_sim --validate
```

---

### Benchmarking

#### `--benchmark`
Run benchmarking mode (multiple configurations).

- **Type:** Flag (boolean)
- **Requires:** `--qubits` and `--depths` lists

```bash
quantum_sim --benchmark --qubits 8,10,12 --depths 20,30,40
```

#### `--qubits <LIST>`
Comma-separated list of qubit counts (for benchmarking).

- **Type:** String (comma-separated integers)
- **Example:** `--qubits 8,10,12,14`

```bash
quantum_sim --benchmark --qubits 8,10,12 --depths 20,30,40 -o bench.csv
```

#### `--depths <LIST>`
Comma-separated list of circuit depths (for benchmarking).

- **Type:** String (comma-separated integers)
- **Example:** `--depths 20,30,40,50`

```bash
quantum_sim --benchmark --qubits 8,10 --depths 20,30,40 -o bench.csv
```

#### `--trials <N>`
Number of trials per configuration (for benchmarking).

- **Type:** Integer
- **Default:** 3
- **Example:** `--trials 5`

```bash
quantum_sim --benchmark --qubits 8,10 --depths 20,30 --trials 10
```

#### `--benchmark-modes`
Benchmark all parallelization modes.

- **Type:** Flag (boolean)

```bash
quantum_sim -n 10 -d 30 --noise 0.01 --benchmark-modes -o mode_comparison.csv
```

---

### Advanced Options

#### `--seed <SEED>`
Random seed for reproducibility.

- **Type:** Integer
- **Default:** Random from system clock
- **Example:** `--seed 42`

```bash
# Reproducible results
quantum_sim -n 8 -d 20 --noise 0.01 --seed 42
quantum_sim -n 8 -d 20 --noise 0.01 --seed 42  # Same result
```

#### `--timeout <DURATION>`
Maximum simulation time (kills if exceeded).

- **Type:** String (duration)
- **Format:** `<N>s|m|h|d` (seconds, minutes, hours, days)
- **Default:** No timeout
- **Example:** `--timeout 5m`

```bash
# 10 minute timeout
quantum_sim -n 14 -d 50 --noise 0.01 --timeout 10m

# 2 hour timeout
quantum_sim -n 16 -d 100 --noise 0.01 --timeout 2h
```

#### `--memory-limit <SIZE>`
Memory limit (soft warning, not enforced).

- **Type:** String (size with unit)
- **Format:** `<N>MB|GB|TB`
- **Default:** No limit
- **Example:** `--memory-limit 4GB`

```bash
quantum_sim -n 12 -d 40 --noise 0.01 --memory-limit 8GB
```

#### `--gpu`
Use GPU acceleration (if compiled with CUDA support).

- **Type:** Flag (boolean)
- **Requires:** CUDA-enabled build

```bash
quantum_sim -n 12 -d 50 --noise 0.01 --gpu --device 0
```

#### `--device <ID>`
GPU device ID (for multi-GPU systems).

- **Type:** Integer
- **Default:** 0
- **Requires:** `--gpu`

```bash
# Use second GPU
quantum_sim -n 12 -d 50 --noise 0.01 --gpu --device 1
```

---

## Environment Variables

### `OMP_NUM_THREADS`
Number of OpenMP threads (overridden by `--threads`).

```bash
export OMP_NUM_THREADS=8
quantum_sim -n 10 -d 30 --noise 0.01
```

### `LRET_CACHE_DIR`
Directory for caching compiled circuits.

```bash
export LRET_CACHE_DIR=/tmp/lret_cache
quantum_sim -n 8 -d 20 --noise 0.01
```

### `LRET_LOG_LEVEL`
Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`.

```bash
export LRET_LOG_LEVEL=DEBUG
quantum_sim -n 8 -d 20 --noise 0.01
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0    | Success |
| 1    | Invalid arguments |
| 2    | Simulation error |
| 3    | Timeout exceeded |
| 4    | Memory limit exceeded |
| 5    | File I/O error |

---

## Examples

### Basic Usage

```bash
# Simple 8-qubit simulation
quantum_sim -n 8 -d 20 --noise 0.01

# Save results to CSV
quantum_sim -n 10 -d 30 --noise 0.02 -o results.csv

# Verbose output
quantum_sim -n 6 -d 15 --noise 0.01 --verbose
```

### IBM Device Simulation

```bash
# Download IBM device noise
python scripts/download_ibm_noise.py --device ibmq_manila -o manila.json

# Simulate with device noise
quantum_sim -n 5 -d 30 --noise-file manila.json -o ibm_sim.csv
```

### Benchmarking

```bash
# Quick benchmark
quantum_sim --benchmark --qubits 8,10,12 --depths 20,30 -o bench.csv

# Full sweep
quantum_sim --benchmark \
    --qubits 6,8,10,12,14 \
    --depths 10,20,30,40,50 \
    --trials 5 \
    -o full_benchmark.csv
```

### Performance Comparison

```bash
# Compare FDM vs LRET
quantum_sim -n 10 -d 30 --noise 0.01 --compare-fdm

# Compare parallelization modes
quantum_sim -n 10 -d 30 --noise 0.01 --benchmark-modes
```

### Reproducible Research

```bash
# Fixed seed for reproducibility
quantum_sim -n 8 -d 20 --noise 0.01 --seed 42 -o run1.csv
quantum_sim -n 8 -d 20 --noise 0.01 --seed 42 -o run2.csv
# run1.csv and run2.csv are identical
```

---

## See Also

- **[Quick Start Tutorial](02-quick-start.md)** - Introductory examples
- **[Python Interface](04-python-interface.md)** - Python API reference
- **[Noise Models](06-noise-models.md)** - Configuring noise
- **[Benchmarking Guide](07-benchmarking.md)** - Performance measurement
- **[Troubleshooting](08-troubleshooting.md)** - Common issues
