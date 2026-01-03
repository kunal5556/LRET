# Installation Guide

This guide covers three installation methods for LRET:
1. **Docker** (recommended for quick start)
2. **Python Package** (for Python-only users)
3. **Building from Source** (for C++ developers and contributors)

## Prerequisites by Method

| Method | Requirements |
|--------|--------------|
| Docker | Docker 20.10+ |
| Python | Python 3.8+, pip |
| Source | C++17 compiler, CMake 3.18+, Eigen3, OpenMP, pybind11 |

---

## Docker Installation

### Why Docker?
- ✅ No dependency management
- ✅ Works on Linux, macOS, Windows
- ✅ Reproducible environment
- ✅ Ready to use in 30 seconds

### Quick Start

```bash
# Pull the latest image
docker pull ajs911/lret777:latest

# Run interactive shell
docker run --rm -it ajs911/lret777:latest bash

# Inside container: test the simulator
quantum_sim -n 8 -d 20 --noise 0.01
```

### Run Simulations

```bash
# Simulate and save results to host directory
docker run --rm -v $(pwd):/output \
    ajs911/lret777:latest \
    quantum_sim -n 10 -d 30 --noise 0.01 -o /output/results.csv

# Check results
cat results.csv
```

### Python Interface via Docker

```bash
# Run Python scripts
docker run --rm -v $(pwd):/app -w /app \
    ajs911/lret777:latest \
    python3 my_simulation.py

# Interactive Python session
docker run --rm -it ajs911/lret777:latest python3
>>> from qlret import QuantumSimulator
>>> sim = QuantumSimulator(n_qubits=4)
>>> sim.h(0)
>>> print(sim.get_state())
```

### Jupyter Notebook in Docker

```bash
# Start Jupyter server
docker run --rm -p 8888:8888 \
    -v $(pwd):/notebooks \
    ajs911/lret777:latest \
    jupyter notebook --ip=0.0.0.0 --allow-root --no-browser

# Open the URL shown in terminal (http://127.0.0.1:8888/...)
```

### Resource Configuration

By default, Docker limits memory usage. For large simulations:

```bash
# Unlimited memory (recommended for large workloads)
docker run --rm -it \
    --memory=0 \
    --memory-swap=-1 \
    -v $(pwd):/app \
    ajs911/lret777:latest \
    quantum_sim -n 14 -d 50

# Specific limits
docker run --rm -it \
    --memory=16g \        # 16GB RAM
    --memory-swap=32g \   # 32GB total (RAM + swap)
    --cpus=8 \            # 8 CPU cores
    ajs911/lret777:latest \
    quantum_sim -n 12 --mode hybrid
```

### Docker Compose (Multi-Container)

For complex workflows, use Docker Compose:

```yaml
# docker-compose.yml
version: '3.8'
services:
  lret-simulator:
    image: ajs911/lret777:latest
    volumes:
      - ./simulations:/output
      - ./scripts:/scripts
    command: >
      bash -c "
        quantum_sim -n 10 -d 30 --noise 0.01 -o /output/run1.csv &&
        quantum_sim -n 12 -d 40 --noise 0.01 -o /output/run2.csv
      "
    mem_limit: 32g
    cpus: 16
```

```bash
# Run all simulations
docker-compose up
```

---

## Python Package Installation

### System Requirements
- **Python:** 3.8, 3.9, 3.10, or 3.11
- **pip:** 20.0 or newer
- **OS:** Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10+

### Install from PyPI (when available)

```bash
# Install latest release
pip install qlret

# Verify installation
python -c "from qlret import QuantumSimulator; print('✓ LRET installed')"
```

### Install from Source (Current Method)

```bash
# Clone repository
git clone https://github.com/kunal5556/LRET.git
cd LRET

# Install Python package
cd python
pip install -e .

# Test installation
python -c "from qlret import QuantumSimulator; print('✓ LRET installed')"
pytest tests/  # Run tests (optional)
```

### Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv lret-env
source lret-env/bin/activate  # On Windows: lret-env\Scripts\activate

# Install LRET
pip install qlret

# Test
python -c "import qlret; print(qlret.__version__)"
```

### Conda Environment

```bash
# Create Conda environment
conda create -n lret python=3.10
conda activate lret

# Install dependencies
conda install numpy scipy matplotlib pytest

# Install LRET
cd LRET/python
pip install -e .
```

### Dependencies

The Python package automatically installs:
- `numpy >= 1.20.0`
- `pybind11 >= 2.10.0`
- `pennylane >= 0.30.0` (optional, for PennyLane device)

Optional dependencies:
```bash
# For benchmarking tools
pip install scipy matplotlib seaborn pandas

# For Jupyter notebooks
pip install jupyter ipywidgets
```

---

## Building from Source

### System Requirements

| Component | Minimum Version | Recommended |
|-----------|----------------|-------------|
| **Compiler** | GCC 9+ / Clang 10+ / MSVC 2019+ | GCC 11+ |
| **CMake** | 3.18 | 3.25+ |
| **Eigen3** | 3.3 | 3.4+ |
| **OpenMP** | 4.5 | 5.0+ |
| **pybind11** | 2.10 | 2.11+ (for Python) |
| **Python** | 3.8 | 3.10+ (optional) |

### Ubuntu/Debian

```bash
# Install build dependencies
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    libeigen3-dev \
    libomp-dev \
    python3-dev \
    python3-pip \
    git

# Clone repository
git clone https://github.com/kunal5556/LRET.git
cd LRET

# Build C++ core
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Test C++ binaries
./quantum_sim -n 8 -d 20 --noise 0.01
ctest --output-on-failure

# Install Python package (optional)
cd ../python
pip install -e .
```

### macOS

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake eigen libomp python@3.10

# Clone and build
git clone https://github.com/kunal5556/LRET.git
cd LRET
mkdir build && cd build

# Configure with Homebrew OpenMP
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" \
    -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" \
    -DOpenMP_C_LIB_NAMES="omp" \
    -DOpenMP_CXX_LIB_NAMES="omp" \
    -DOpenMP_omp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib

make -j$(sysctl -n hw.ncpu)

# Test
./quantum_sim --help
ctest --output-on-failure
```

### Windows (Visual Studio)

```powershell
# Install Visual Studio 2019+ with C++ tools
# Install CMake from https://cmake.org/download/

# Install vcpkg for dependencies
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install

# Install Eigen3
.\vcpkg install eigen3:x64-windows

# Clone LRET
cd ..
git clone https://github.com/kunal5556/LRET.git
cd LRET

# Build with CMake
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release

# Test
.\Release\quantum_sim.exe -n 8 -d 20 --noise 0.01
```

### Build Options

```bash
# Configure build options
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \      # Release, Debug, RelWithDebInfo
    -DBUILD_PYTHON=ON \               # Build Python bindings
    -DBUILD_TESTS=ON \                # Build unit tests
    -DENABLE_OPENMP=ON \              # Enable parallelization
    -DENABLE_SIMD=ON \                # Enable SIMD optimizations
    -DCMAKE_INSTALL_PREFIX=/usr/local # Install location

make -j$(nproc)
make install  # Install system-wide (requires sudo)
```

### GPU Support (Experimental)

```bash
# Install CUDA Toolkit (11.0+)
# https://developer.nvidia.com/cuda-downloads

# Build with GPU support
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_GPU=ON \
    -DCMAKE_CUDA_ARCHITECTURES=70,75,80  # Adjust for your GPU

make -j$(nproc)

# Test GPU mode
./quantum_sim -n 12 -d 40 --gpu --device 0
```

---

## Verification

### C++ Installation

```bash
# Check binary
quantum_sim --version
# Output: quantum_sim version 1.0.0 (Phase 6d)

# Run quick test
quantum_sim -n 8 -d 20 --noise 0.01
# Should complete in < 1 second with rank ~10-15
```

### Python Installation

```python
import qlret
print(qlret.__version__)  # 1.0.0

# Quick simulation
sim = qlret.QuantumSimulator(n_qubits=4, noise_level=0.01)
sim.h(0)
sim.cnot(0, 1)
print(f"Final rank: {sim.current_rank}")  # Should be < 10

# PennyLane device
import pennylane as qml
dev = qlret.QLRETDevice(wires=4)
print(dev.name)  # qlret.simulator
```

### Running Tests

```bash
# C++ tests
cd build
ctest --output-on-failure

# Python tests
cd python/tests
pytest -v

# Benchmark tests
python scripts/benchmark_suite.py --quick --categories scaling
```

---

## Troubleshooting

### Common Issues

#### "Eigen3 not found"
```bash
# Ubuntu/Debian
sudo apt install libeigen3-dev

# macOS
brew install eigen

# Verify
find /usr -name "Eigen" 2>/dev/null
```

#### "OpenMP not supported"
```bash
# Ubuntu
sudo apt install libomp-dev

# macOS (use Homebrew LLVM)
brew install libomp
export CC=/opt/homebrew/opt/llvm/bin/clang
export CXX=/opt/homebrew/opt/llvm/bin/clang++
```

#### "pybind11 not found"
```bash
pip install pybind11[global]
# Or install system-wide
sudo apt install pybind11-dev  # Ubuntu
brew install pybind11          # macOS
```

#### Python import fails
```python
# Check LD_LIBRARY_PATH (Linux/macOS)
import sys
sys.path.append('/path/to/LRET/python')
from qlret import QuantumSimulator

# Check installation
pip show qlret
```

#### Docker out of memory
```bash
# Check Docker resources
docker info | grep -i memory

# Increase limits
# Linux: Edit /etc/docker/daemon.json
# macOS/Windows: Docker Desktop → Settings → Resources
```

### Platform-Specific Notes

**Linux:**
- Use system package manager for dependencies
- Set `LD_LIBRARY_PATH` if libraries not found
- For HPC: use Singularity instead of Docker

**macOS:**
- Use Homebrew for all dependencies
- M1/M2 Macs: ensure ARM64 compatibility
- OpenMP requires special configuration

**Windows:**
- Use vcpkg for C++ dependencies
- WSL2 recommended for Docker
- Visual Studio 2019+ required for MSVC

---

## Next Steps

- **[Quick Start Tutorial →](02-quick-start.md)** - Run your first simulation
- **[CLI Reference →](03-cli-reference.md)** - Learn command-line options
- **[Python Interface →](04-python-interface.md)** - Explore Python API
- **[Troubleshooting →](08-troubleshooting.md)** - Solve common issues
