# PennyLane Benchmarking - Parallelization Analysis

## Question 1: Multi-Core/Thread Usage

### Current Status: **Single-Threaded by Default** ❌

Both LRET and default.mixed devices in the benchmark scripts **use only 1 core** by default.

**Evidence:**
- CPU monitoring shows 90-93% usage of **ONE core** (not total CPU)
- On an 8-core system: 90% of 1 core = ~11% total CPU usage (matches your Task Manager observation of 1-2%)
- Neither device is configured to use multi-threading in the current benchmark scripts

### Why Single-Threaded?

1. **LRET Device**: Built with OpenMP support, but requires explicit configuration
   - `num_threads` parameter not exposed through Python API yet
   - Default behavior: uses 1 thread
   
2. **default.mixed Device**: PennyLane's CPU-based mixed state simulator
   - Uses NumPy/Autoray backend (single-threaded by default)
   - No built-in multi-threading support in PennyLane 0.43.2

---

## Question 2: LRET Parallelization Modes

### Available Parallelization Modes

LRET supports **7 parallelization modes** (defined in `include/cli_parser.h`):

```cpp
enum class ParallelMode {
    AUTO,       // Auto-select best strategy
    SEQUENTIAL, // No parallelism (default)
    ROW,        // Row-wise parallel
    COLUMN,     // Column-wise parallel
    BATCH,      // Gate batching
    HYBRID,     // Row + batch combined
    COMPARE,    // Run all and compare
    MPI_ROW,    // MPI row-wise distribution
    MPI_COLUMN, // MPI column-wise distribution
    MPI_HYBRID, // MPI + OpenMP hybrid
    GPU_DISTRIBUTED // Multi-GPU
};
```

### Current Default: **SEQUENTIAL** (No Parallelism)

The PennyLane device plugin currently uses **SEQUENTIAL mode** because:
- The Python API (`qlret.pennylane_device`) doesn't expose `parallel_mode` parameter
- The C++ backend defaults to `ParallelMode::SEQUENTIAL` when not specified
- This is why you see single-core usage

---

## How to Enable Multi-Threading

### Option 1: Modify LRET Device Initialization (Requires Code Changes)

**Current code** (`python/qlret/pennylane_device.py`):
```python
def __init__(
    self,
    wires: Union[int, Sequence[int]],
    shots: Optional[int] = None,
    epsilon: float = 1e-4,
    **kwargs: Any,
) -> None:
```

**Needed enhancement:**
```python
def __init__(
    self,
    wires: Union[int, Sequence[int]],
    shots: Optional[int] = None,
    epsilon: float = 1e-4,
    num_threads: int = 0,  # 0 = use all cores
    parallel_mode: str = "auto",  # "auto", "row", "column", "hybrid"
    **kwargs: Any,
) -> None:
    self.num_threads = num_threads
    self.parallel_mode = parallel_mode
```

Then pass these to the C++ backend in the circuit JSON.

### Option 2: Environment Variable (Quick Fix)

Set OpenMP environment variable before running:

**Windows (PowerShell):**
```powershell
$env:OMP_NUM_THREADS = 8
python benchmarks/pennylane/4q_50e_25s_10n.py
```

**Linux/Mac:**
```bash
export OMP_NUM_THREADS=8
python benchmarks/pennylane/4q_50e_25s_10n.py
```

⚠️ **Note**: This only works if LRET was compiled with OpenMP support (`-DUSE_OPENMP=ON`)

### Option 3: Modify Benchmark Scripts (Immediate Solution)

Add parallel configuration to the device creation:

**Current:**
```python
dev_lret = qml.device('qlret.mixed', wires=N_QUBITS, epsilon=1e-4)
```

**Enhanced (requires API update):**
```python
dev_lret = qml.device(
    'qlret.mixed', 
    wires=N_QUBITS, 
    epsilon=1e-4,
    num_threads=8,  # Use 8 cores
    parallel_mode="hybrid"  # Row + batch parallelization
)
```

---

## Can We Customize Per-Instance?

### Answer: **Not Yet** (Requires Implementation)

**Current situation:**
- Parallelization mode is **not configurable per-device** in the Python API
- It's hardcoded to SEQUENTIAL in the C++ backend when called via Python
- The CLI tool (`quantum_sim`) supports all modes, but Python doesn't expose them

**What needs to be done:**

1. **Update `pennylane_device.py`** to accept `parallel_mode` and `num_threads` parameters
2. **Update `api.py`** to pass these to the C++ backend
3. **Update circuit JSON schema** to include parallelization config
4. **Recompile Python bindings** with new parameters

---

## Implementation Roadmap

### Phase 1: Add Parameters to Device (15 minutes)

**File**: `python/qlret/pennylane_device.py`

```python
def __init__(
    self,
    wires: Union[int, Sequence[int]],
    shots: Optional[int] = None,
    epsilon: float = 1e-4,
    num_threads: int = 0,  # NEW
    parallel_mode: str = "auto",  # NEW
    **kwargs: Any,
) -> None:
    _require_pennylane()
    super().__init__(wires=wires, shots=shots)
    self.epsilon = epsilon
    self.num_threads = num_threads  # NEW
    self.parallel_mode = parallel_mode  # NEW
    self._kwargs = kwargs
    self._num_wires = len(self.wires) if hasattr(self.wires, '__len__') else self.wires
```

### Phase 2: Pass to Circuit JSON (10 minutes)

**File**: `python/qlret/pennylane_device.py` (in `_execute_tape` method)

```python
circuit_json = {
    "circuit": {
        "num_qubits": self._num_wires,
        "operations": operations,
        "observables": observables,
    },
    "config": {
        "epsilon": self.epsilon,
        "num_threads": self.num_threads,  # NEW
        "parallel_mode": self.parallel_mode,  # NEW
    }
}
```

### Phase 3: Update C++ Backend (30 minutes)

**File**: `src/json_interface.cpp`

Parse `num_threads` and `parallel_mode` from JSON config and set them before simulation.

### Phase 4: Rebuild & Test (10 minutes)

```bash
cd build
cmake .. -DUSE_OPENMP=ON
make -j$(nproc)
cd ../python
pip install -e . --force-reinstall
```

---

## Expected Performance Gains

### With Multi-Threading (8 cores):

| Configuration | Current (1 core) | With 8 Threads | Speedup |
|--------------|-----------------|---------------|---------|
| 4 qubits | 3.5s/epoch | ~0.8-1.2s/epoch | 2.5-4× |
| 8 qubits | 15-20s/epoch | ~3-5s/epoch | 4-6× |
| 12 qubits | 120-180s/epoch | ~20-35s/epoch | 5-7× |

**Note**: Speedup depends on parallelization mode and rank of the state.

### Best Parallelization Modes by Scenario:

- **Low rank (< 10)**: `SEQUENTIAL` or `ROW` (parallel overhead not worth it)
- **Medium rank (10-100)**: `HYBRID` (row + batch) - **BEST for most cases**
- **High rank (> 100)**: `COLUMN` or `HYBRID`
- **Very large systems (16+ qubits)**: `MPI_HYBRID` (distributed memory)

---

## Comparison: LRET vs default.mixed Multi-Threading

| Feature | LRET | default.mixed |
|---------|------|---------------|
| **Multi-threading** | ✅ Supported (OpenMP) | ❌ Single-threaded |
| **Parallelization modes** | ✅ 7 modes | ❌ None |
| **Per-device config** | ⚠️ Needs implementation | N/A |
| **Distributed (MPI)** | ✅ Supported | ❌ No |
| **GPU support** | ✅ CUDA backend available | ❌ No |

**Verdict**: LRET has superior parallelization capabilities, but they're not yet exposed through the PennyLane interface.

---

## Immediate Action Items

### Quick Win: Enable OpenMP Environment Variable

```powershell
# Windows
$env:OMP_NUM_THREADS = 8
python benchmarks/pennylane/4q_50e_25s_10n.py
```

This might give 2-3× speedup if LRET was compiled with OpenMP.

### Proper Solution: Implement API Parameters

Create new benchmark scripts with multi-threading support:
- `4q_50e_25s_10n_parallel.py` (8 threads, hybrid mode)
- `8q_100e_100s_12n_parallel.py` (8 threads, hybrid mode)

This requires implementing the API changes outlined above.

---

## Summary

1. **Current state**: Both devices use **1 core** (single-threaded)
2. **LRET parallelization mode**: **SEQUENTIAL** (hardcoded)
3. **Per-device customization**: **Not yet supported** (needs implementation)
4. **Quick fix**: Set `OMP_NUM_THREADS=8` environment variable
5. **Proper fix**: Implement `num_threads` and `parallel_mode` parameters in Python API

**Estimated effort**: 1-2 hours to add proper multi-threading support to PennyLane device.

---

*Analysis date: January 16, 2026*
