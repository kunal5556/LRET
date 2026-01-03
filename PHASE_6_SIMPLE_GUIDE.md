# Phase 6 Docker Strategy: Simple Explanation

## 🎯 Current State (End of Phase 5)

Our project now has **two execution paths:**

### Path 1: CLI (Original - Still Works)
```bash
./quantum_sim -n 10 -d 20 --mode compare
```
Generates results via C++ binary.

### Path 2: JSON (New - Alternative)
```bash
./quantum_sim --input-json circuit.json --output-json result.json
```
Same binary, but driven by JSON file instead of CLI flags.

### Path 3: Python (New - Programmatic)
```python
from qlret import simulate_json, QLRETDevice
import pennylane as qml

# Direct JSON simulation
result = simulate_json(circuit_dict)

# Or PennyLane integration
dev = QLRETDevice(wires=4)
@qml.qnode(dev)
def circuit(): ...
circuit()
```

---

## 🐳 What Phase 6 Does to Docker

### Current Dockerfile (85 lines)
```
Build C++ binary → Run C++ binary with CLI args
```

### Phase 6 Dockerfile (200+ lines, 4 stages)
```
Stage 1 (Builder):   Build C++ binary + Python module
Stage 2 (Python):    Install Python packages + qlret package
Stage 3 (Testing):   Run full integration test suite
Stage 4 (Runtime):   Final image with everything
```

---

## 📋 Phase 6 Tasks (8-10 hours total)

### 6a: Dockerfile Multi-Stage (1-2 hours)
**What:** Rewrite Dockerfile with 4 stages

**Old (simple):**
```dockerfile
FROM ubuntu:24.04 AS builder
  # Build C++
  
FROM ubuntu:24.04 AS runtime
  # Copy binary, run it
```

**New (enhanced):**
```dockerfile
FROM ubuntu:24.04 AS cpp-builder
  # Build quantum_sim + _qlret_native.so
  
FROM ubuntu:24.04 AS python-builder
  # Install Python packages
  
FROM ubuntu:24.04 AS tester
  # Run pytest suite (validation)
  
FROM ubuntu:24.04 AS runtime
  # Final image with everything
```

**Result:** Docker image can now run:
- C++ binary (original capability)
- Python code (new)
- Tests (new)
- Jupyter notebooks (new)

---

### 6b: Integration Tests in Docker (2-3 hours)
**What:** Run full test suite automatically during Docker build

**Why:** Catch bugs early. If tests fail, build fails.

**Tests to run:**
```bash
# In Docker, during build:
pytest python/tests/ -v

# Should validate:
- JSON parsing works
- C++ execution works
- Python bindings work
- PennyLane device works
- Gradients are correct
```

**Result:** Guarantee our code works in container before ship.

---

### 6c: Performance Benchmarks (1-2 hours)
**What:** Compare QLRET vs other simulators in Docker

**Benchmark script:**
```python
import pennylane as qml
from qlret import QLRETDevice
from pennylane import numpy as np

# Run same circuit with multiple backends
backends = ['qlret', 'default.qubit', 'qiskit.aer']
for backend in backends:
    dev = qml.device(backend, wires=4)
    circuit(dev)  # Time it
    # Record: execution time, memory, accuracy
```

**Result:** Proof our simulator is fast (or where improvements needed).

---

### 6d: Documentation (1-2 hours)
**What:** Explain how to use everything

**Files to create:**
- JSON circuit schema (what fields are valid?)
- PennyLane quick start
- Docker usage guide
- Performance tuning tips

**Result:** Users understand how to use our code.

---

### 6e: GitHub Actions CI (1 hour)
**What:** Automate building + testing on every push

**Workflow:**
```
push code → GitHub detects → runs tests in cloud → pass/fail report
```

**Result:** Guarantee code quality on every commit.

---

## 🔄 Phase 6 Impact

### For Users
| Before | After |
|--------|-------|
| Run binary via CLI | **PLUS:** Run via JSON, Python, PennyLane |
| Trust code works | **PLUS:** Automated tests prove it |
| No performance data | **PLUS:** Benchmarks vs competitors |
| Confusing docs | **PLUS:** Clear guides for each mode |

### For Developers
| Before | After |
|--------|-------|
| Test locally | **PLUS:** Docker guarantees portability |
| Manual testing | **PLUS:** Automated pytest suite |
| Slow iteration | **PLUS:** Fast CI/CD feedback |

### For the Project
| Before | After |
|--------|-------|
| 1 way to use | **PLUS:** 4 ways (CLI, JSON, Python, Jupyter) |
| Hard to verify | **PLUS:** Automated testing + benchmarks |
| Docker only runs binary | **PLUS:** Docker is full dev environment |

---

## 🎬 What Docker Does (Simplified)

### Current (Phase 5 end)
```
Docker Container
├── quantum_sim (C++ binary)
└── Python scripts (for noise calibration only)
```
You can only run: `./quantum_sim ...`

### Phase 6
```
Docker Container
├── quantum_sim (C++ binary)
├── _qlret_native.so (Python module)
├── qlret Python package
├── PennyLane, Jax, Jupyter (installed)
├── pytest test suite
├── benchmarks scripts
└── notebooks (example usage)
```
You can now run:
- `./quantum_sim ...` (as before)
- `python -c "import qlret"` (new)
- `pytest tests/` (new)
- `jupyter notebook` (new)
- Benchmark scripts (new)

---

## 💡 Why This Matters

**Problem:** "Does this code actually work in Docker?"
- **Before:** Hope it works, find bugs in production
- **After:** Build fails if tests fail, so bugs caught early

**Problem:** "How fast is QLRET vs competitors?"
- **Before:** No data, hope it's good
- **After:** Docker build runs benchmarks automatically

**Problem:** "How do I use the Python API in Docker?"
- **Before:** Can't, only CLI available
- **After:** Full Python + PennyLane support, Jupyter for experiments

**Problem:** "Will my changes break anything?"
- **Before:** Run tests locally, hope it's fine everywhere
- **After:** GitHub Actions tests automatically, you see results

---

## 📊 Phase 6 Checklist

- [ ] **6a:** Multi-stage Dockerfile (cpp + python + test + runtime stages)
- [ ] **6b:** Docker integration test suite (pytest in build)
- [ ] **6c:** Performance benchmarks (vs Qiskit Aer, PennyLane)
- [ ] **6d:** Documentation (JSON schema, API docs, quickstart)
- [ ] **6e:** GitHub Actions CI/CD workflow
- [ ] **Bonus:** Docker Compose for multi-node MPI testing

---

## 🚀 Next Step

Once Phase 6 is complete:
- Docker image is a full development + runtime environment
- Code is automatically tested on every push
- Performance is tracked and benchmarked
- Users have 4 ways to use the library (CLI, JSON, Python, Jupyter)
- GitHub users can see test results on every PR

**Recommendation:** Start with 6a (Docker refactor) using **GPT-5.1 Codex Max**, then proceed with tests and benchmarks. Switch to **Opus 4.5** if benchmarking shows anomalies.
