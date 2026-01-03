# Building from Source

Comprehensive guide to building LRET from source on all platforms.

## Prerequisites

### Required Dependencies

| Dependency | Minimum Version | Purpose |
|------------|----------------|---------|
| **C++ Compiler** | C++17 support | Core compilation |
| **CMake** | 3.18+ | Build system |
| **Eigen3** | 3.3+ | Linear algebra |
| **OpenMP** | 4.5+ | Parallelization |

### Optional Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| **pybind11** | 2.10+ | Python bindings |
| **Python** | 3.8+ | Python interface |
| **CUDA Toolkit** | 11.0+ | GPU acceleration |
| **MPI** | OpenMPI 4.0+ | Distributed computing |
| **Google Test** | 1.10+ | C++ unit tests |
| **pytest** | 7.0+ | Python tests |

---

## Platform-Specific Setup

### Ubuntu/Debian

```bash
# Update package list
sudo apt update

# Essential build tools
sudo apt install -y \
    build-essential \
    cmake \
    git

# Required dependencies
sudo apt install -y \
    libeigen3-dev \
    libomp-dev

# Optional: Python support
sudo apt install -y \
    python3-dev \
    python3-pip \
    python3-venv

# Optional: Python dependencies
pip3 install pybind11 numpy

# Optional: Testing tools
sudo apt install -y \
    libgtest-dev \
    python3-pytest

# Verify installations
cmake --version        # Should be >= 3.18
g++ --version          # Should support C++17
python3 --version      # Should be >= 3.8
```

**Eigen3 Location:**
```bash
# Check Eigen3 installation
dpkg -L libeigen3-dev | grep Eigen
# Typical location: /usr/include/eigen3/Eigen
```

---

### Fedora/RHEL/CentOS

```bash
# Install build tools
sudo dnf groupinstall "Development Tools"
sudo dnf install cmake

# Required dependencies
sudo dnf install eigen3-devel
sudo dnf install libomp-devel

# Python support (optional)
sudo dnf install python3-devel python3-pip
pip3 install pybind11 numpy

# Testing tools (optional)
sudo dnf install gtest-devel
pip3 install pytest
```

---

### macOS

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Xcode Command Line Tools
xcode-select --install

# Install dependencies
brew install cmake eigen libomp python@3.11

# Optional: Testing tools
brew install googletest
pip3 install pytest pybind11 numpy

# Verify installations
cmake --version
clang++ --version
python3 --version
```

**Important for Apple Silicon (M1/M2):**
```bash
# Set architecture explicitly
export CMAKE_OSX_ARCHITECTURES=arm64

# Or for universal binary
export CMAKE_OSX_ARCHITECTURES="arm64;x86_64"
```

---

### Windows

#### Option 1: Visual Studio (Recommended)

**Prerequisites:**
1. Install [Visual Studio 2019 or later](https://visualstudio.microsoft.com/)
   - Select "Desktop development with C++"
   - Include CMake tools
2. Install [CMake](https://cmake.org/download/) (if not included)

**Install vcpkg for dependencies:**
```powershell
# Clone vcpkg
cd C:\
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# Install dependencies
.\vcpkg install eigen3:x64-windows
.\vcpkg install libomp:x64-windows

# Optional: Python support
.\vcpkg install pybind11:x64-windows

# Integrate with Visual Studio
.\vcpkg integrate install
```

#### Option 2: MSYS2/MinGW

```bash
# Install MSYS2 from https://www.msys2.org/
# Open MSYS2 terminal

# Update package database
pacman -Syu

# Install build tools
pacman -S --needed base-devel mingw-w64-x86_64-toolchain

# Install CMake
pacman -S mingw-w64-x86_64-cmake

# Install dependencies
pacman -S mingw-w64-x86_64-eigen3
pacman -S mingw-w64-x86_64-openmp

# Python support (optional)
pacman -S mingw-w64-x86_64-python
pacman -S mingw-w64-x86_64-python-pip
pip install pybind11 numpy
```

#### Option 3: WSL2 (Windows Subsystem for Linux)

```powershell
# Enable WSL2
wsl --install

# Inside WSL2, follow Ubuntu instructions above
```

---

## Building LRET

### Quick Build (Linux/macOS)

```bash
# Clone repository
git clone https://github.com/kunal5556/LRET.git
cd LRET

# Create build directory
mkdir build && cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build (use all available cores)
make -j$(nproc)  # Linux
make -j$(sysctl -n hw.ncpu)  # macOS

# Test (optional)
ctest --output-on-failure

# Install (optional, requires sudo)
sudo make install
```

**Binaries created:**
- `build/quantum_sim` - CLI tool
- `build/test_*` - Test executables

---

### Configuration Options

#### Build Type

```bash
# Release (optimized, no debug symbols)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Debug (symbols, no optimization)
cmake .. -DCMAKE_BUILD_TYPE=Debug

# RelWithDebInfo (optimized + symbols)
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo

# MinSizeRel (smallest binary size)
cmake .. -DCMAKE_BUILD_TYPE=MinSizeRel
```

#### Feature Flags

```bash
# Enable/disable features
cmake .. \
    -DBUILD_PYTHON=ON \          # Build Python bindings (default: ON)
    -DBUILD_TESTS=ON \           # Build C++ tests (default: ON)
    -DENABLE_OPENMP=ON \         # Enable OpenMP (default: ON)
    -DENABLE_SIMD=ON \           # Enable SIMD (default: ON)
    -DENABLE_GPU=OFF \           # Enable CUDA (default: OFF)
    -DENABLE_MPI=OFF             # Enable MPI (default: OFF)
```

#### Installation Prefix

```bash
# Install to custom location
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/lret

# Install to user directory (no sudo needed)
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/.local
```

#### Compiler Selection

```bash
# Use specific compiler
cmake .. \
    -DCMAKE_C_COMPILER=gcc-11 \
    -DCMAKE_CXX_COMPILER=g++-11

# Use Clang
cmake .. \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++

# Use Intel compiler
cmake .. \
    -DCMAKE_C_COMPILER=icc \
    -DCMAKE_CXX_COMPILER=icpc
```

---

### Platform-Specific Builds

#### macOS with Homebrew OpenMP

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" \
    -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" \
    -DOpenMP_C_LIB_NAMES="omp" \
    -DOpenMP_CXX_LIB_NAMES="omp" \
    -DOpenMP_omp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib

make -j$(sysctl -n hw.ncpu)
```

#### Windows with Visual Studio

```powershell
# Create build directory
mkdir build
cd build

# Configure (generates Visual Studio solution)
cmake .. `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake

# Build
cmake --build . --config Release

# Or open LRET.sln in Visual Studio and build there
```

#### Windows with MinGW

```bash
# From MSYS2 terminal
mkdir build && cd build

cmake .. \
    -G "MinGW Makefiles" \
    -DCMAKE_BUILD_TYPE=Release

mingw32-make -j$(nproc)
```

---

## GPU Support (Optional)

### Prerequisites

1. NVIDIA GPU with CUDA Compute Capability 6.0+
2. [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 11.0+
3. Compatible NVIDIA driver

### Build with CUDA

```bash
# Check CUDA installation
nvcc --version

# Configure with GPU support
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_GPU=ON \
    -DCMAKE_CUDA_ARCHITECTURES=70,75,80,86  # Adjust for your GPU

make -j$(nproc)

# Test GPU version
./quantum_sim -n 12 -d 40 --gpu --device 0
```

**GPU Architecture Codes:**
- 70: Volta (V100)
- 75: Turing (RTX 20xx, T4)
- 80: Ampere (A100)
- 86: Ampere (RTX 30xx)
- 89: Ada Lovelace (RTX 40xx)
- 90: Hopper (H100)

---

## MPI Support (Optional)

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt install libopenmpi-dev

# Fedora
sudo dnf install openmpi-devel

# macOS
brew install open-mpi
```

### Build with MPI

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_MPI=ON \
    -DMPI_C_COMPILER=mpicc \
    -DMPI_CXX_COMPILER=mpicxx

make -j$(nproc)

# Test MPI version
mpirun -np 4 ./quantum_sim -n 14 -d 50 --noise 0.01
```

---

## Python Bindings

### Build and Install Python Package

```bash
# Method 1: Build with CMake (development)
mkdir build && cd build
cmake .. -DBUILD_PYTHON=ON
make -j$(nproc)

# Add to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/python

# Test
python3 -c "from qlret import QuantumSimulator; print('OK')"

# Method 2: Install with pip (production)
cd python
pip install -e .  # Editable install
# OR
pip install .     # Regular install

# Verify
python3 -c "import qlret; print(qlret.__version__)"
```

### Python Package Options

```bash
# Install with specific Python version
python3.10 -m pip install -e python/

# Install without build isolation (uses existing C++ build)
pip install -e python/ --no-build-isolation

# Install with verbose output (for debugging)
pip install -e python/ -v

# Uninstall
pip uninstall qlret
```

---

## Testing

### C++ Unit Tests

```bash
cd build

# Run all tests
ctest --output-on-failure

# Run specific test
./test_simulator
./test_gates
./test_noise

# Run with verbose output
ctest -V

# Run tests in parallel
ctest -j$(nproc)
```

### Python Tests

```bash
cd python/tests

# Run all tests
pytest -v

# Run specific test file
pytest test_simulator.py -v

# Run specific test
pytest test_simulator.py::test_hadamard_gate -v

# Run with coverage
pytest --cov=qlret --cov-report=html

# Run only fast tests (skip slow)
pytest -m "not slow" -v
```

### Integration Tests

```bash
cd python/tests/integration

# Run all integration tests
pytest -v

# Run Docker tests (requires Docker)
pytest test_docker_runtime.py -v

# Run CLI tests
pytest test_cli_regression.py -v

# Run PennyLane tests
pytest test_pennylane_device.py -v
```

---

## Installation

### System-Wide Installation

```bash
cd build

# Install binaries, libraries, and headers
sudo make install

# Default locations:
# - Binaries: /usr/local/bin/quantum_sim
# - Libraries: /usr/local/lib/libqlret.*
# - Headers: /usr/local/include/qlret/
# - CMake files: /usr/local/lib/cmake/qlret/

# Verify
quantum_sim --version
```

### User Installation (No sudo)

```bash
# Configure with user prefix
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/.local

make -j$(nproc)
make install

# Add to PATH
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
quantum_sim --version
```

### Uninstallation

```bash
cd build
sudo make uninstall  # If installed system-wide

# OR manually remove
sudo rm -f /usr/local/bin/quantum_sim
sudo rm -rf /usr/local/include/qlret
sudo rm -f /usr/local/lib/libqlret.*
```

---

## Troubleshooting

### Eigen3 Not Found

**Error:**
```
CMake Error: Could not find Eigen3
```

**Solution:**
```bash
# Ubuntu/Debian
sudo apt install libeigen3-dev

# Set manually if needed
cmake .. -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3

# Verify
find /usr -name "Eigen" -type d 2>/dev/null
```

---

### OpenMP Not Found

**Error:**
```
Could NOT find OpenMP_C
```

**Solution (Ubuntu):**
```bash
sudo apt install libomp-dev

# Verify
echo '#include <omp.h>' | gcc -xc -fopenmp - -o /dev/null && echo "OpenMP OK"
```

**Solution (macOS):**
```bash
brew install libomp

cmake .. \
    -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" \
    -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" \
    -DOpenMP_omp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib
```

---

### Python Bindings Fail to Build

**Error:**
```
Could NOT find pybind11
```

**Solution:**
```bash
pip install pybind11[global]

# Or install system-wide
sudo apt install pybind11-dev  # Ubuntu
brew install pybind11          # macOS

# Verify
python3 -m pybind11 --includes
```

---

### CUDA Not Found

**Error:**
```
Could NOT find CUDA
```

**Solution:**
```bash
# Check CUDA installation
ls /usr/local/cuda/bin/nvcc

# Set CUDA path explicitly
export CUDA_ROOT=/usr/local/cuda
cmake .. -DENABLE_GPU=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
```

---

### Linker Errors

**Error:**
```
undefined reference to 'omp_get_thread_num'
```

**Solution:**
```bash
# Explicitly link OpenMP
cmake .. -DCMAKE_CXX_FLAGS="-fopenmp"

# Or set environment variable
export OMP_NUM_THREADS=1
```

---

### Test Failures

**Check:**
1. Build type (Debug vs Release can affect numerics)
2. OpenMP thread count (set `OMP_NUM_THREADS=1` for reproducibility)
3. Floating-point precision (some tests have tolerance)

```bash
# Run tests with single thread
OMP_NUM_THREADS=1 ctest --output-on-failure

# Increase test timeout
ctest --timeout 300

# Run specific failing test with verbose output
./build/test_simulator --gtest_filter=TestSuiteName.TestName
```

---

## Build Performance Tips

### Faster Compilation

```bash
# Use Ninja instead of Make
cmake .. -GNinja
ninja -j$(nproc)

# Use ccache
sudo apt install ccache
cmake .. -DCMAKE_CXX_COMPILER_LAUNCHER=ccache

# Parallel builds
make -j$(nproc)  # Use all cores
make -j8         # Use 8 cores explicitly
```

### Reduce Build Size

```bash
# Minimal build
cmake .. \
    -DCMAKE_BUILD_TYPE=MinSizeRel \
    -DBUILD_TESTS=OFF \
    -DBUILD_PYTHON=OFF

# Strip binaries
strip build/quantum_sim
```

### Debug Builds

```bash
# Full debug symbols
cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-g -O0"

# Address sanitizer (detect memory errors)
cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-fsanitize=address -g"

# Thread sanitizer (detect race conditions)
cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-fsanitize=thread -g"
```

---

## Cross-Compilation

### For ARM64 (on x86_64)

```bash
# Install cross-compiler
sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# Configure for ARM64
cmake .. \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++

make -j$(nproc)
```

### For Windows (from Linux)

```bash
# Install MinGW cross-compiler
sudo apt install mingw-w64

# Configure for Windows
cmake .. \
    -DCMAKE_SYSTEM_NAME=Windows \
    -DCMAKE_C_COMPILER=x86_64-w64-mingw32-gcc \
    -DCMAKE_CXX_COMPILER=x86_64-w64-mingw32-g++

make -j$(nproc)
```

---

## CMake Cache

### View Configuration

```bash
cd build

# View all CMake variables
cmake -L ..

# View advanced variables
cmake -LA ..

# View cache file
cat CMakeCache.txt | grep -E "CMAKE_|EIGEN|OpenMP|PYTHON"
```

### Clean Build

```bash
# Remove build directory
rm -rf build

# Or clean within build directory
cd build
make clean  # Clean object files only
rm -rf *    # Complete clean
```

### Reconfigure

```bash
cd build

# Reconfigure with new options
cmake .. -DBUILD_PYTHON=OFF

# Force reconfigure (clear cache)
rm CMakeCache.txt
cmake ..
```

---

## See Also

- **[Architecture Overview](00-overview.md)** - System design
- **[Code Structure](02-code-structure.md)** - Repository layout
- **[Testing Framework](05-testing.md)** - Running tests
- **[Performance Guide](06-performance.md)** - Optimization tips
- **[Installation Guide](../user-guide/01-installation.md)** - User installation
