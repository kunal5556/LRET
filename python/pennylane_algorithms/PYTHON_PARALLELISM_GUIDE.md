# Python Parallelism Implementation Guide

**LRET PennyLane Device - Multi-Level Parallelism Architecture**

This guide explains how Python-level parallelism is implemented in the LRET PennyLane device and the benchmarking suite.

---

## Table of Contents

1. [Overview](#overview)
2. [Two-Level Parallelism Architecture](#two-level-parallelism-architecture)
3. [Python-Level Parallelism (Batch Execution)](#python-level-parallelism-batch-execution)
4. [C++-Level Parallelism (OpenMP)](#c-level-parallelism-openmp)
5. [Thread Allocation Strategy](#thread-allocation-strategy)
6. [Parallel Modes Comparison](#parallel-modes-comparison)
7. [Implementation Details](#implementation-details)
8. [Usage Examples](#usage-examples)
9. [Performance Tuning](#performance-tuning)
10. [Troubleshooting](#troubleshooting)

---

## Overview

LRET implements a **two-level parallelism architecture**:

1. **Python-level parallelism**: Execute multiple circuits concurrently using Python's `ThreadPoolExecutor`
2. **C++-level parallelism**: Execute operations within a single circuit using OpenMP multi-threading

This design allows flexible trade-offs between circuit-level and operation-level parallelism to maximize CPU utilization.

### Key Design Principle

**CPU Budget Management**: The total number of active threads is kept close to the CPU core count to prevent oversubscription, which degrades performance due to context switching.

```
Total Threads = Python Workers × C++ Threads per Worker ≈ CPU Core Count
```

---

## Two-Level Parallelism Architecture

### Level 1: Python-Level (Batch Parallelism)

**Purpose**: Execute multiple quantum circuits concurrently

**Implementation**: `ThreadPoolExecutor` from `concurrent.futures`

**When Used**: When executing batches of circuits (gradient computations, parameter sweeps, ensemble averaging)

**Controlled By**: `max_batch_workers` parameter

### Level 2: C++-Level (OpenMP Parallelism)

**Purpose**: Parallelize operations within a single circuit

**Implementation**: OpenMP in C++ backend (row/column/batch/hybrid modes)

**When Used**: Always (for single circuits or individual circuits in a batch)

**Controlled By**: `num_threads` and `parallel_mode` parameters

---

## Python-Level Parallelism (Batch Execution)

### Device Parameter: `max_batch_workers`

Controls Python-level parallelism for executing multiple circuits concurrently.

#### Values:

| Value | Behavior | Use Case |
|-------|----------|----------|
| `0` (default) | Disabled - sequential execution | Single circuit, minimal overhead |
| `1` | Explicitly sequential | Same as 0, but explicit |
| `N > 1` | Use N Python workers | Fixed worker count |
| `-1` | Auto-tune based on batch size | Adaptive strategy |
| `'max'` | Maximum workers (cpu_count) | Optimal for sequential C++ mode |

### Implementation: `_execute_batch_parallel()`

Located in: `python/qlret/pennylane_device.py` (lines 800-830)

```python
def _execute_batch_parallel(
    self,
    circuits: List[QuantumTape],
    workers: int,
    threads_per_circuit: int,
) -> List[np.ndarray]:
    """Execute a batch of circuits in parallel using ThreadPoolExecutor."""
    
    def execute_single(tape: QuantumTape) -> np.ndarray:
        """Execute a single tape with specified thread count."""
        return self._execute_tape_with_threads(tape, threads_per_circuit)
    
    # ThreadPool works well because C++ work releases GIL
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(execute_single, circuits))
    
    return results
```

### Why ThreadPoolExecutor?

**Reason**: The C++ backend releases the Python GIL (Global Interpreter Lock), allowing true parallel execution despite Python's threading limitations.

**Advantage over ProcessPoolExecutor**: No serialization overhead for passing circuits between processes.

---

## C++-Level Parallelism (OpenMP)

### Device Parameter: `parallel_mode`

Controls C++-level parallelization strategy within each circuit.

#### Modes:

| Mode | Description | Best For |
|------|-------------|----------|
| `sequential` | No parallelism (1 thread) | Small circuits, Python parallelism |
| `row` | Row-wise parallelization | Tall matrices (many operations) |
| `column` | Column-wise parallelization | Wide matrices (many states) |
| `batch` | Gate batching | Circuits with many identical gates |
| `hybrid` (default) | Row + batch combined | General purpose (best balance) |
| `auto` | Automatic selection | Let LRET decide |

### Device Parameter: `num_threads`

Number of OpenMP threads used within each circuit.

| Value | Behavior |
|-------|----------|
| `0` (default) | Auto-detect (use all CPU cores) |
| `1` | Sequential C++ execution |
| `N > 1` | Use N OpenMP threads |

---

## Thread Allocation Strategy

### The `_compute_execution_strategy()` Method

Located in: `python/qlret/pennylane_device.py` (lines 710-790)

This intelligent method detects whether LRET is running in sequential or parallel C++ mode and adjusts the strategy accordingly.

#### Strategy 1: Sequential C++ Mode

**Detected When**: `num_threads=1` OR `parallel_mode='sequential'`

**Optimal Configuration**: Maximize Python workers, 1 thread per circuit

**Example** (8-core CPU):
```python
dev = QLRETDevice(
    wires=4,
    num_threads=1,              # Sequential C++ (no OpenMP)
    max_batch_workers='max'     # Use all CPU cores as workers
)
# Result: 8 workers × 1 thread = 8 total threads (full CPU utilization)
```

**Why**: Sequential C++ mode doesn't benefit from multiple threads per circuit, so maximize circuit-level parallelism instead.

#### Strategy 2: Parallel C++ Mode

**Detected When**: `num_threads > 1` AND `parallel_mode != 'sequential'`

**Optimal Configuration**: Balance workers and threads per circuit

**Example** (8-core CPU):
```python
dev = QLRETDevice(
    wires=4,
    num_threads=8,              # Parallel C++ (8 threads per circuit)
    max_batch_workers=4         # 4 Python workers
)
# Result: 4 workers × 2 threads = 8 total threads (balanced)
```

**Why**: Parallel C++ mode benefits from multiple threads, but too many workers cause oversubscription.

### Automatic Thread Adjustment

When `max_batch_workers > 1`, the device **automatically reduces** `threads_per_circuit` to prevent oversubscription:

```python
threads_per_circuit = max(1, effective_threads // num_workers)
```

**Example**:
- CPU cores: 8
- `num_threads=8` (full parallelism per circuit)
- `max_batch_workers=4` (run 4 circuits in parallel)
- **Adjusted**: Each circuit gets 8/4 = 2 threads
- **Total**: 4 workers × 2 threads = 8 threads (matches CPU count)

---

## Parallel Modes Comparison

### Benchmark Suite Implementation

Located in: `python/pennylane_algorithms/utils/parallel_modes.py`

This module provides utilities for comparing different Python parallelization strategies across the benchmarking suite.

#### Available Parallel Modes

| Mode | Backend | Description | Requires |
|------|---------|-------------|----------|
| `sequential` | N/A | Sequential execution (baseline) | Built-in |
| `threading` | ThreadPoolExecutor | Thread-based parallelism | Built-in |
| `process_pool` | ProcessPoolExecutor | Process-based parallelism | Built-in |
| `multiprocessing` | multiprocessing.Pool | Process-based with shared memory | multiprocessing |
| `joblib_loky` | Joblib (loky backend) | Process-based with robustness | joblib |
| `joblib_threading` | Joblib (threading backend) | Thread-based via joblib | joblib |

### The `ParallelExecutor` Class

A unified interface for different parallelization strategies:

```python
from utils.parallel_modes import ParallelExecutor

executor = ParallelExecutor(mode='threading', n_workers=4)
results = executor.map(my_function, list_of_args)
```

**Supported Methods**:
- `map(func, args_list)`: Apply function to all arguments in parallel
- Automatic fallback to sequential if mode unavailable

### Comparing Parallel Modes

```python
from utils.parallel_modes import run_parallel_comparison

# Test function
def benchmark_circuit(params):
    # Run quantum circuit
    return result

# Compare all available modes
results = run_parallel_comparison(
    func=benchmark_circuit,
    args_list=[(p,) for p in param_list],
    modes=['sequential', 'threading', 'multiprocessing', 'joblib_loky'],
    n_workers=4
)

# Results include timing, speedup, and efficiency metrics
for mode, result in results.items():
    print(f"{mode}: {result.total_time:.3f}s (speedup: {result.speedup:.2f}x)")
```

---

## Implementation Details

### File Locations

#### Core Device Implementation

**File**: `python/qlret/pennylane_device.py`

**Key Components**:

1. **Initialization** (lines 432-520):
   - Parse `max_batch_workers` parameter
   - Detect CPU count
   - Initialize thread settings

2. **Batch Execution** (lines 680-710):
   - Entry point: `execute(circuits, execution_config)`
   - Determines sequential vs parallel execution

3. **Strategy Computation** (lines 710-790):
   - `_compute_execution_strategy(batch_size)`: Intelligent thread allocation
   - Detects C++ sequential mode
   - Computes optimal worker/thread split

4. **Parallel Execution** (lines 800-830):
   - `_execute_batch_parallel(circuits, workers, threads_per_circuit)`
   - Uses `ThreadPoolExecutor`
   - Overrides thread count per circuit

5. **Thread Override** (lines 830-850):
   - `_execute_tape_with_threads(tape, num_threads)`
   - Builds JSON with custom thread count
   - Calls C++ backend

#### Benchmark Suite Implementation

**File**: `python/pennylane_algorithms/utils/parallel_modes.py` (498 lines)

**Key Components**:

1. **ParallelExecutor Class** (lines 85-300):
   - Unified interface for all parallel modes
   - Automatic fallback handling
   - Support for `*args` and `**kwargs`

2. **Parallel Mode Registry** (lines 50-70):
   - `PARALLEL_MODES` dict: mode → metadata
   - `get_parallel_modes()`: List available modes
   - Dependency checking (joblib, multiprocessing)

3. **Comparison Utilities** (lines 260-350):
   - `run_parallel_comparison()`: Compare all modes
   - `measure_parallel_speedup()`: Test worker scaling
   - `format_parallel_comparison()`: Pretty-print results

4. **Convenience Functions** (lines 400-498):
   - `parallel_parameter_sweep()`: Parameter grid evaluation
   - `parallel_gradient_computation()`: Parallel parameter-shift gradients

### Data Flow

```
PennyLane QNode
      ↓
QLRETDevice.execute(circuits)
      ↓
_compute_execution_strategy(batch_size)
      ↓ (decides)
      ├─→ Sequential: _execute_tape(tape) × N
      └─→ Parallel: _execute_batch_parallel(circuits, workers, threads)
                          ↓
                    ThreadPoolExecutor
                          ↓
                    execute_single(tape) × workers
                          ↓
                    _execute_tape_with_threads(tape, threads_per_circuit)
                          ↓
                    simulate_json(circuit_json) [C++ backend]
                          ↓
                    OpenMP parallel execution (row/column/batch/hybrid)
                          ↓
                    Results returned
```

---

## Usage Examples

### Example 1: Single Circuit (No Batch Parallelism)

```python
import pennylane as qml
from qlret import QLRETDevice

# Use all CPU cores for within-circuit parallelism
dev = QLRETDevice(
    wires=8,
    num_threads=0,              # Auto (use all cores)
    parallel_mode='hybrid',     # Best general-purpose mode
    max_batch_workers=0         # Disabled (single circuit)
)

@qml.qnode(dev)
def circuit(theta):
    qml.RY(theta, wires=0)
    for i in range(7):
        qml.CNOT(wires=[i, i+1])
    return qml.expval(qml.PauliZ(0))

result = circuit(0.5)
# Execution: 1 circuit × 8 threads = 8 threads (matches 8-core CPU)
```

### Example 2: Gradient Computation (Batch Parallelism)

**Problem**: Computing gradients requires evaluating 2N shifted circuits (N parameters)

**Solution**: Use batch parallelism to evaluate shifts concurrently

```python
import pennylane as qml
from qlret import QLRETDevice

# 8-core CPU: Use 4 workers × 2 threads
dev = QLRETDevice(
    wires=4,
    num_threads=8,              # 8 threads when running single circuit
    parallel_mode='hybrid',
    max_batch_workers=4         # Automatically reduces to 2 threads per circuit
)

@qml.qnode(dev, diff_method="parameter-shift")
def circuit(params):
    for i in range(4):
        qml.RY(params[i], wires=i)
    for i in range(3):
        qml.CNOT(wires=[i, i+1])
    return qml.expval(qml.PauliZ(0))

params = np.random.random(4)
grad_fn = qml.grad(circuit)
gradients = grad_fn(params)

# Execution: 8 parameter-shift circuits executed in parallel
# → 4 workers × 2 threads = 8 threads (optimal)
```

### Example 3: Sequential C++ Mode (Maximum Python Parallelism)

**When**: C++ parallelism isn't beneficial (small circuits, overhead-dominated)

**Strategy**: Use `num_threads=1` and `max_batch_workers='max'`

```python
import pennylane as qml
from qlret import QLRETDevice

# 8-core CPU: Use 8 workers × 1 thread
dev = QLRETDevice(
    wires=4,
    num_threads=1,              # Sequential C++ (no OpenMP)
    parallel_mode='sequential',
    max_batch_workers='max'     # Use all cores as workers
)

@qml.qnode(dev, diff_method="parameter-shift")
def circuit(params):
    qml.RY(params[0], wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

# Run VQE optimization (many gradient evaluations)
optimizer = qml.GradientDescentOptimizer(0.1)
params = np.random.random(4)

for _ in range(100):
    params = optimizer.step(circuit, params)

# Each gradient evaluation runs 8 circuits in parallel
# → 8 workers × 1 thread = 8 threads (optimal for sequential C++)
```

### Example 4: Auto-Tuning

**When**: Unsure of optimal configuration

**Strategy**: Use `max_batch_workers=-1` for automatic tuning

```python
dev = QLRETDevice(
    wires=6,
    num_threads=0,              # Auto-detect threads
    parallel_mode='auto',       # Auto-select C++ strategy
    max_batch_workers=-1        # Auto-tune Python parallelism
)

# Device will automatically:
# - Detect CPU core count
# - Detect C++ sequential vs parallel mode
# - Adjust worker count based on batch size
# - Balance workers and threads to match CPU cores
```

### Example 5: Benchmark Suite Parallel Modes

**Comparing Python parallelization strategies for algorithm benchmarks**:

```python
from pennylane_algorithms.utils.parallel_modes import run_parallel_comparison

def run_vqe_iteration(params):
    """Single VQE iteration (energy evaluation)."""
    energy = circuit(params)
    return energy

# Test different Python parallelism modes
param_list = [np.random.random(10) for _ in range(50)]

results = run_parallel_comparison(
    func=run_vqe_iteration,
    args_list=[(p,) for p in param_list],
    modes=['sequential', 'threading', 'process_pool', 'joblib_loky'],
    n_workers=4
)

# Output:
# sequential          : 15.234s (speedup: 1.00x, efficiency: 100.0%)
# threading           : 4.102s (speedup: 3.71x, efficiency: 92.8%)
# process_pool        : 4.567s (speedup: 3.34x, efficiency: 83.4%)
# joblib_loky         : 4.201s (speedup: 3.63x, efficiency: 90.7%)
```

---

## Performance Tuning

### Detecting Optimal Configuration

Use the diagnostic tool to find the best settings for your workload:

```python
import time
import numpy as np
from qlret import QLRETDevice
import pennylane as qml

def benchmark_configuration(wires, num_threads, max_batch_workers, batch_size):
    """Benchmark a specific configuration."""
    dev = QLRETDevice(
        wires=wires,
        num_threads=num_threads,
        max_batch_workers=max_batch_workers
    )
    
    @qml.qnode(dev)
    def circuit(theta):
        qml.RY(theta, wires=0)
        for i in range(wires-1):
            qml.CNOT(wires=[i, i+1])
        return qml.expval(qml.PauliZ(0))
    
    # Create batch
    params = [np.random.random() for _ in range(batch_size)]
    
    # Time execution
    start = time.perf_counter()
    for p in params:
        circuit(p)
    elapsed = time.perf_counter() - start
    
    return elapsed

# Test different configurations
configs = [
    (8, 0, "sequential"),          # Full C++ parallelism, no batch
    (8, 1, 'max'),                 # Sequential C++, max Python workers
    (8, 4, 4),                     # Balanced (4 workers × 2 threads)
    (8, 0, -1),                    # Auto-tune
]

print("Configuration Benchmark (8 cores, batch_size=32):")
print("-" * 70)
for num_threads, max_batch_workers, label in configs:
    time_taken = benchmark_configuration(
        wires=6,
        num_threads=num_threads,
        max_batch_workers=max_batch_workers,
        batch_size=32
    )
    print(f"{label:<30}: {time_taken:.3f}s")
```

### General Guidelines

#### For Single Circuits:
- Use `max_batch_workers=0` (disabled)
- Use `num_threads=0` (all cores)
- Use `parallel_mode='hybrid'` (best general-purpose)

#### For Small Batches (< 4 circuits):
- Use `max_batch_workers=-1` (auto-tune)
- Let auto-tuning decide

#### For Large Batches (≥ 4 circuits):
- **If circuits are large** (many qubits/gates):
  - Use `num_threads=8`, `max_batch_workers=4` (balanced)
- **If circuits are small** (few qubits/gates):
  - Use `num_threads=1`, `max_batch_workers='max'` (sequential C++)

#### For Gradient Computations:
- Parameter-shift creates 2N circuits (N parameters)
- Use `max_batch_workers=-1` or explicit workers
- Example: 10 parameters → 20 circuits → use 4-8 workers

---

## Troubleshooting

### Issue 1: Slower with Parallelism Enabled

**Symptom**: Performance degrades when `max_batch_workers > 1`

**Cause**: Thread oversubscription or small circuits with high overhead

**Solution**:
1. Check CPU utilization (should be ~100%, not 200%+)
2. Try `max_batch_workers='max'` with `num_threads=1` (sequential C++)
3. Reduce worker count: `max_batch_workers=2` or `max_batch_workers=4`
4. For very small circuits, disable: `max_batch_workers=0`

### Issue 2: GIL Contention Warning

**Symptom**: `threading` mode performs poorly

**Cause**: Python GIL limiting parallelism (shouldn't happen with LRET)

**Solution**:
1. Verify C++ backend is being used (not subprocess fallback)
2. Check `import _qlret_native` works without error
3. Try `process_pool` or `joblib_loky` instead of `threading`

### Issue 3: High Memory Usage

**Symptom**: Memory consumption increases significantly with parallelism

**Cause**: Multiple circuits held in memory simultaneously

**Solution**:
1. Reduce worker count: `max_batch_workers=2`
2. Process batch in chunks:
   ```python
   def process_batch_chunked(circuits, chunk_size=10):
       results = []
       for i in range(0, len(circuits), chunk_size):
           chunk = circuits[i:i+chunk_size]
           results.extend(dev.execute(chunk))
       return results
   ```

### Issue 4: Inconsistent Results Between Runs

**Symptom**: Different results on repeated runs (should not happen for deterministic circuits)

**Cause**: Race condition or improper thread management

**Solution**:
1. Set `shots=None` for deterministic results
2. Use explicit `max_batch_workers` (avoid auto-tune if problematic)
3. Report as bug (this should not occur)

### Issue 5: Import Error for `joblib`

**Symptom**: `ImportError: No module named 'joblib'`

**Cause**: Optional dependency not installed

**Solution**:
```bash
pip install joblib
```

Or use built-in modes:
```python
# Use only built-in modes
executor = ParallelExecutor(mode='threading', n_workers=4)
```

---

## Advanced Topics

### Custom Parallel Backends

Extend `ParallelExecutor` with custom backends:

```python
from utils.parallel_modes import ParallelExecutor

class CustomParallelExecutor(ParallelExecutor):
    def _run_custom(self, func, args_list, **kwargs):
        # Your custom parallelization logic
        results = []
        # ... implementation ...
        return results
    
    def map(self, func, args_list, **kwargs):
        if self.mode == 'custom':
            return self._run_custom(func, args_list, **kwargs)
        return super().map(func, args_list, **kwargs)
```

### Nested Parallelism

**Warning**: Avoid nested parallelism (parallel device inside parallel executor)

**Problem**:
```python
# BAD: Double parallelism causes oversubscription
dev = QLRETDevice(wires=4, max_batch_workers=4)  # 4 workers
executor = ParallelExecutor(mode='threading', n_workers=4)  # Another 4 workers

def run_circuit(params):
    return dev.execute(circuit(params))  # Already parallel!

executor.map(run_circuit, param_list)  # 4 × 4 = 16 workers (oversubscribed!)
```

**Solution**: Use parallelism at only one level
```python
# GOOD: Parallelism at device level
dev = QLRETDevice(wires=4, max_batch_workers=4)
results = [dev.execute(circuit(p)) for p in param_list]

# OR: Parallelism at executor level
dev = QLRETDevice(wires=4, max_batch_workers=0)  # Disabled
executor = ParallelExecutor(mode='threading', n_workers=4)
results = executor.map(lambda p: dev.execute(circuit(p)), param_list)
```

---

## Summary

### Key Takeaways

1. **Two-level parallelism**: Python (batch) + C++ (OpenMP)
2. **Thread budget**: Total threads ≈ CPU cores (prevent oversubscription)
3. **Sequential C++ mode**: Use `num_threads=1` + `max_batch_workers='max'`
4. **Parallel C++ mode**: Balance workers and threads (e.g., 4 workers × 2 threads)
5. **Auto-tuning**: Use `max_batch_workers=-1` when unsure
6. **ThreadPoolExecutor**: Works well because C++ releases GIL

### Configuration Quick Reference

| Scenario | `num_threads` | `max_batch_workers` | Effective Config |
|----------|---------------|---------------------|------------------|
| Single circuit | `0` (auto) | `0` (disabled) | 1 circuit × 8 threads |
| Small batch, large circuits | `8` | `4` | 4 circuits × 2 threads |
| Large batch, small circuits | `1` | `'max'` | 8 circuits × 1 thread |
| Gradient computation | `8` | `-1` (auto) | Auto-balanced |
| Not sure | `0` (auto) | `-1` (auto) | Fully automatic |

### Related Files

- Device implementation: `python/qlret/pennylane_device.py`
- Parallel modes utility: `python/pennylane_algorithms/utils/parallel_modes.py`
- Benchmark suite: `python/pennylane_algorithms/run_all_benchmarks.py`
- Algorithm examples: `python/pennylane_algorithms/tier{1,2,3}/*.py`

---

**Document Version**: 1.0  
**Last Updated**: February 5, 2026  
**Author**: LRET Development Team  
**Status**: Complete and Production-Ready
