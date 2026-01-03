# Troubleshooting

Common issues and solutions when using LRET.

---

## Installation Issues

### Problem: "Eigen3 not found"

**Error:**
```
CMake Error: Could not find Eigen3
```

**Solution (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install libeigen3-dev

# Verify installation
dpkg -L libeigen3-dev | grep Eigen
```

**Solution (macOS):**
```bash
brew install eigen

# If CMake still can't find it
export EIGEN3_INCLUDE_DIR=/opt/homebrew/include/eigen3
cmake .. -DEIGEN3_INCLUDE_DIR=$EIGEN3_INCLUDE_DIR
```

**Solution (Windows vcpkg):**
```powershell
vcpkg install eigen3:x64-windows
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
```

---

### Problem: "OpenMP not supported"

**Error:**
```
warning: OpenMP support not detected
```

**Solution (Ubuntu/Debian):**
```bash
sudo apt install libomp-dev

# Verify
echo '#include <omp.h>' | gcc -xc -fopenmp - -o /dev/null && echo "OpenMP OK"
```

**Solution (macOS):**
```bash
# Install LLVM with OpenMP
brew install libomp

# Configure CMake to use Homebrew OpenMP
cmake .. \
    -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" \
    -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" \
    -DOpenMP_C_LIB_NAMES="omp" \
    -DOpenMP_CXX_LIB_NAMES="omp" \
    -DOpenMP_omp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib
```

**Solution (Windows):**
```powershell
# Visual Studio includes OpenMP by default
# Verify in CMake configuration
cmake .. -DENABLE_OPENMP=ON
```

---

### Problem: Python import fails

**Error:**
```python
>>> from qlret import QuantumSimulator
ModuleNotFoundError: No module named 'qlret'
```

**Solution 1: Check installation**
```bash
pip show qlret
# If not installed:
cd LRET/python
pip install -e .
```

**Solution 2: Check Python path**
```python
import sys
print(sys.path)

# Add LRET to path
sys.path.append('/path/to/LRET/python')
from qlret import QuantumSimulator
```

**Solution 3: Virtual environment**
```bash
# Create fresh environment
python3 -m venv lret-env
source lret-env/bin/activate  # On Windows: lret-env\Scripts\activate
pip install -e python/
```

---

### Problem: "pybind11 not found"

**Error:**
```
CMake Error: Could not find pybind11
```

**Solution:**
```bash
# Install pybind11
pip install pybind11[global]

# Or system-wide
sudo apt install pybind11-dev  # Ubuntu
brew install pybind11          # macOS

# Verify
python3 -m pybind11 --includes
```

---

## Runtime Issues

### Problem: Segmentation fault

**Error:**
```
Segmentation fault (core dumped)
```

**Common Causes:**

**1. Out of memory:**
```bash
# Check available memory
free -h

# Reduce problem size
quantum_sim -n 10 -d 30 --noise 0.01  # Was -n 14
```

**2. Stack overflow (large matrices):**
```bash
# Increase stack size
ulimit -s unlimited  # Linux
export OMP_STACKSIZE=512M  # OpenMP stack
```

**3. Invalid qubit index:**
```python
sim = QuantumSimulator(n_qubits=4)
sim.h(5)  # ❌ Qubit 5 doesn't exist (0-3 valid)
# Solution: Check indices
sim.h(3)  # ✓ Valid
```

**Debugging:**
```bash
# Run with debugger
gdb --args ./quantum_sim -n 8 -d 20
(gdb) run
(gdb) backtrace  # If it crashes
```

---

### Problem: Rank grows exponentially

**Symptoms:**
- Memory usage explodes
- Simulation becomes very slow
- Rank > 1000 for modest circuits

**Cause:** Noise level too low (low-rank advantage lost)

**Solution:**
```bash
# Increase noise
quantum_sim -n 10 -d 50 --noise 0.01  # Was 0.0001

# Or use stricter truncation
quantum_sim -n 10 -d 50 --noise 0.001 --threshold 1e-3  # Was 1e-4
```

**Monitor rank:**
```python
sim = QuantumSimulator(n_qubits=10, noise_level=0.001, verbose=True)
for i in range(50):
    sim.h(i % 10)
    if i % 10 == 0:
        print(f"Step {i}: Rank = {sim.current_rank}")
        if sim.current_rank > 100:
            print("Warning: Rank growing too fast!")
            break
```

---

### Problem: Low fidelity

**Symptoms:**
- Fidelity < 0.99
- Results don't match expected values
- High trace distance vs FDM

**Cause:** Truncation threshold too loose

**Solution:**
```python
# Use stricter truncation
sim = QuantumSimulator(
    n_qubits=8,
    noise_level=0.01,
    truncation_threshold=1e-5  # Was 1e-3
)
```

**Trade-off:**
| Threshold | Fidelity | Rank | Speed |
|-----------|----------|------|-------|
| 1e-6      | 0.9999   | High | Slow  |
| 1e-4      | 0.9997   | Med  | Fast  |
| 1e-3      | 0.999    | Low  | Very fast |

---

### Problem: Simulation hangs / takes forever

**Symptoms:**
- Process doesn't finish
- No output for long time
- High CPU usage

**Solution 1: Enable timeout**
```bash
quantum_sim -n 12 -d 100 --noise 0.01 --timeout 10m
```

**Solution 2: Enable verbose mode**
```bash
quantum_sim -n 12 -d 100 --noise 0.01 --verbose
# Shows progress gate-by-gate
```

**Solution 3: Check rank growth**
```python
sim = QuantumSimulator(n_qubits=12, noise_level=0.01, verbose=True, max_rank=100)
# Aborts if rank exceeds 100
```

**Solution 4: Use simpler circuit**
```bash
# Test with small problem first
quantum_sim -n 8 -d 20 --noise 0.01
# Then scale up gradually
```

---

### Problem: Docker out of memory

**Error:**
```
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
```
or
```
Killed (OOM)
```

**Solution:**
```bash
# 1. Check Docker memory limit
docker info | grep Memory

# 2. Increase Docker memory (Docker Desktop)
# Settings → Resources → Memory → 8GB+

# 3. Run with unlimited memory
docker run --rm -it \
    --memory=0 \
    --memory-swap=-1 \
    -v $(pwd):/app \
    ajs911/lret777:latest \
    quantum_sim -n 12 -d 40

# 4. Use Singularity instead (no memory limits)
singularity run --bind $(pwd):/app lret.sif quantum_sim -n 12 -d 40
```

---

## Performance Issues

### Problem: Slow simulation

**Symptoms:**
- Takes much longer than expected
- No speedup from parallelization

**Diagnosis:**
```bash
# 1. Check CPU usage
top  # Should show high CPU usage on all cores

# 2. Check OpenMP threads
echo $OMP_NUM_THREADS  # Should be number of cores

# 3. Profile the code
quantum_sim -n 8 -d 20 --verbose
```

**Solution 1: Enable parallelization**
```bash
# Explicitly set threads
export OMP_NUM_THREADS=8
quantum_sim -n 10 -d 30 --noise 0.01 --mode hybrid
```

**Solution 2: Optimize parallel mode**
```bash
# Try different modes
quantum_sim -n 10 -d 30 --benchmark-modes
# Use the fastest one
```

**Solution 3: Adjust batch size**
```bash
# Larger batch (more parallelism, more memory)
quantum_sim -n 10 -d 30 --batch-size 128

# Smaller batch (less memory, less parallelism)
quantum_sim -n 10 -d 30 --batch-size 32
```

**Solution 4: Use GPU (if available)**
```bash
quantum_sim -n 12 -d 50 --gpu --device 0
```

---

### Problem: Poor parallel scaling

**Symptoms:**
- 8 threads only 2× faster than 1 thread
- Adding more threads doesn't help

**Cause:** Circuit too small, overhead dominates

**Solution:**
```bash
# Use larger problem size
quantum_sim -n 12 -d 50  # Instead of -n 6 -d 10

# Or use sequential mode for small problems
quantum_sim -n 6 -d 10 --mode sequential
```

**Benchmark parallel efficiency:**
```bash
python scripts/benchmark_suite.py --categories parallel --output parallel.csv
python scripts/benchmark_analysis.py parallel.csv
```

---

## Noise Model Issues

### Problem: IBM noise import fails

**Error:**
```
FileNotFoundError: ibmq_manila.json not found
```

**Solution:**
```bash
# Download noise model first
python scripts/download_ibm_noise.py --device ibmq_manila --output manila_noise.json

# Check if file exists
ls -lh manila_noise.json

# Use in simulation
quantum_sim -n 5 -d 30 --noise-file manila_noise.json
```

---

### Problem: "Invalid noise model format"

**Error:**
```
ValueError: Invalid noise model format
```

**Solution:** Check JSON structure
```bash
# Validate JSON
python -m json.tool custom_noise.json

# Example valid format:
cat > valid_noise.json << EOF
{
  "model_type": "mixed",
  "global_depolarizing": 0.01,
  "gate_errors": {
    "H": 0.0005,
    "CNOT": 0.01
  }
}
EOF
```

---

### Problem: Noise too high/low

**Symptoms:**
- Fidelity = 0 (noise too high)
- Rank explodes (noise too low)

**Solution:** Use realistic noise levels
```python
# Too high (unrealistic)
sim = QuantumSimulator(n_qubits=8, noise_level=0.5)  # 50% error!

# Too low (no low-rank advantage)
sim = QuantumSimulator(n_qubits=8, noise_level=0.00001)

# Realistic (1-5% per gate)
sim = QuantumSimulator(n_qubits=8, noise_level=0.01)  # ✓
```

---

## PennyLane Issues

### Problem: "Device not found"

**Error:**
```python
>>> dev = qml.device('qlret.simulator', wires=4)
DeviceError: Device qlret.simulator does not exist
```

**Solution:**
```bash
# Install LRET Python package
cd LRET/python
pip install -e .

# Verify installation
python -c "from qlret import QLRETDevice; print('OK')"
```

---

### Problem: Gradient computation fails

**Error:**
```
ValueError: Gradient method not supported
```

**Solution:** Use parameter-shift rule
```python
import pennylane as qml
from qlret import QLRETDevice

dev = QLRETDevice(wires=4, noise_level=0.01)

@qml.qnode(dev, diff_method="parameter-shift")  # ← Specify method
def circuit(params):
    qml.RY(params[0], wires=0)
    return qml.expval(qml.PauliZ(0))

# Now gradients work
grad_fn = qml.grad(circuit)
gradients = grad_fn([0.5])
```

---

### Problem: "Shots not supported"

**Error:**
```
NotImplementedError: Shots not yet supported
```

**Solution:** Use analytic mode (shots=None)
```python
# Use analytic mode
dev = QLRETDevice(wires=4, noise_level=0.01, shots=None)

# If you need sampling, use alternative:
from qlret import QuantumSimulator
sim = QuantumSimulator(n_qubits=4, noise_level=0.01)
# ... build circuit ...
results = sim.measure_all(shots=1000)
```

---

## Testing Issues

### Problem: Tests fail to run

**Error:**
```bash
$ ctest
No tests were found!!!
```

**Solution:**
```bash
# Build tests first
cd build
cmake .. -DBUILD_TESTS=ON
make -j$(nproc)
ctest --output-on-failure
```

---

### Problem: Python tests fail

**Error:**
```bash
$ pytest
ERROR: file not found: tests/
```

**Solution:**
```bash
# Run from correct directory
cd LRET/python/tests
pytest -v

# Or specify path
pytest python/tests/ -v
```

---

## Platform-Specific Issues

### macOS: "Library not loaded"

**Error:**
```
dyld: Library not loaded: @rpath/libomp.dylib
```

**Solution:**
```bash
# Install libomp
brew install libomp

# Set library path
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH

# Rebuild with correct paths
cd build
cmake .. -DOpenMP_omp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib
make clean && make -j
```

---

### Windows: "VCRUNTIME140.dll not found"

**Error:**
```
The code execution cannot proceed because VCRUNTIME140.dll was not found.
```

**Solution:**
```powershell
# Install Visual C++ Redistributable
# https://aka.ms/vs/17/release/vc_redist.x64.exe

# Or install Visual Studio with C++ tools
```

---

### Linux: "GLIBC version not found"

**Error:**
```
version 'GLIBC_2.29' not found
```

**Solution:**
```bash
# Check GLIBC version
ldd --version

# If too old, rebuild from source
cd LRET
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Or use Docker
docker pull ajs911/lret777:latest
```

---

## Getting Help

### Collect Debug Information

```bash
# System info
uname -a
lscpu | grep -E "Model name|CPU\(s\)"
free -h

# LRET version
quantum_sim --version

# Python version
python3 --version
pip show qlret

# Dependencies
python3 -c "import numpy; print(numpy.__version__)"
python3 -c "import pennylane; print(pennylane.__version__)"

# CMake cache (for build issues)
cat build/CMakeCache.txt | grep -E "CMAKE_CXX_COMPILER|EIGEN3|OpenMP"
```

### Create Minimal Reproducible Example

```python
# minimal_bug.py
from qlret import QuantumSimulator

sim = QuantumSimulator(n_qubits=4, noise_level=0.01, verbose=True)
sim.h(0)
sim.cnot(0, 1)
print(f"Rank: {sim.current_rank}")

# Run:
# python minimal_bug.py
```

### Report Issue on GitHub

Include:
1. **System info** (OS, CPU, RAM)
2. **LRET version** (`quantum_sim --version`)
3. **Minimal reproducible example**
4. **Expected vs actual behavior**
5. **Error messages** (full stacktrace)
6. **Relevant logs** (use `--verbose`)

**Template:**
```markdown
**Environment:**
- OS: Ubuntu 22.04
- LRET version: 1.0.0
- Python: 3.10.6
- CMake: 3.24.2

**Bug description:**
Simulation crashes with segfault when n_qubits > 12.

**Minimal example:**
```bash
quantum_sim -n 14 -d 20 --noise 0.01
```

**Expected:** Completes successfully
**Actual:** Segmentation fault (core dumped)

**Logs:**
```
...
```
```

---

## Common Error Messages

| Error | Meaning | Solution |
|-------|---------|----------|
| `bad_alloc` | Out of memory | Reduce problem size or increase RAM |
| `invalid_argument` | Invalid parameter | Check qubit indices, noise levels |
| `runtime_error` | Unexpected condition | Check verbose logs, report bug |
| `Assertion failed` | Internal error | Report bug with stacktrace |
| `Floating point exception` | Numerical instability | Adjust truncation threshold |

---

## Performance Checklist

Before reporting "LRET is slow":

- [ ] Enabled parallelization (`--mode hybrid`)
- [ ] Set correct thread count (`export OMP_NUM_THREADS=<cores>`)
- [ ] Used realistic noise (≥0.005)
- [ ] Checked rank growth (`--verbose`)
- [ ] Compared to FDM (`--compare-fdm`)
- [ ] Profiled the code (`--verbose`)
- [ ] Tried different parallel modes (`--benchmark-modes`)
- [ ] Used appropriate truncation threshold

---

## See Also

- **[Installation Guide](01-installation.md)** - Setup instructions
- **[CLI Reference](03-cli-reference.md)** - Command-line options
- **[Python Interface](04-python-interface.md)** - Python API
- **[Benchmarking Guide](07-benchmarking.md)** - Performance testing
- **[GitHub Issues](https://github.com/kunal5556/LRET/issues)** - Report bugs
- **[GitHub Discussions](https://github.com/kunal5556/LRET/discussions)** - Ask questions
