# PennyLane Benchmarking - System Requirements & Setup Guide

This guide provides step-by-step instructions for setting up the LRET PennyLane benchmarking environment on a fresh system.

## Quick Setup (Recommended)

For automated setup, run:
```bash
python setup_pennylane_env.py
```

This will check/install all dependencies, build LRET, and verify the installation.

---

## Manual Setup Instructions

### Prerequisites

#### System Requirements
- **OS**: Linux (Ubuntu 20.04+) or Windows 10/11
- **RAM**: 8GB minimum, 16GB+ recommended
- **Disk**: 2GB free space
- **CPU**: Multi-core processor recommended

#### Required Software

1. **Python 3.8-3.11**
   - Linux (Ubuntu/Debian):
     ```bash
     sudo apt update
     sudo apt install python3 python3-pip python3-dev
     ```
   - Windows:
     - Download from [python.org](https://www.python.org/downloads/)
     - Check "Add Python to PATH" during installation
     - Verify: `python --version`

2. **C++ Compiler**
   - Linux:
     ```bash
     sudo apt install build-essential g++ cmake
     ```
   - Windows:
     - **Recommended:** Visual Studio 2022 Build Tools (minimal installation)
     - **Alternative:** Visual Studio 2022 Community (full IDE)
     - **Minimum:** Visual Studio 2019 or newer
     - Download from: https://visualstudio.microsoft.com/downloads/
     - **Quick Install (PowerShell):**
       ```powershell
       # Build Tools only (minimal, ~2GB)
       Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vs_buildtools.exe" -OutFile "$env:TEMP\vs_buildtools.exe"
       Start-Process -FilePath "$env:TEMP\vs_buildtools.exe" -ArgumentList "--quiet", "--wait", "--add", "Microsoft.VisualStudio.Workload.VCTools" -Wait
       ```

3. **CMake 3.16+**
   - Linux:
     ```bash
     sudo apt install cmake
     cmake --version  # Verify >= 3.16
     ```
   - Windows:
     - Download installer from https://cmake.org/download/
     - Add to PATH during installation
     - Verify in PowerShell: `cmake --version`

4. **Eigen3** (Linear Algebra Library)
   - Linux:
     ```bash
     sudo apt install libeigen3-dev
     ```
   - Windows:
     - Download from http://eigen.tuxfamily.org/
     - Extract to `C:\Program Files\Eigen3`
     - OR let CMake download it automatically (our CMakeLists.txt handles this)

---

### Python Dependencies

Install required Python packages:

```bash
pip install pennylane>=0.30 torch numpy scipy psutil matplotlib pandas
```

**Package Details:**
- `pennylane>=0.30` - Quantum machine learning framework
- `torch` - PyTorch for optimization
- `numpy` - Numerical computing
- `scipy` - Scientific computing
- `psutil>=5.8` - **REQUIRED** System monitoring (memory, CPU) for benchmarks and monitor_cpu.py
- `matplotlib` - Plotting (for results visualization)
- `pandas` - Data analysis (optional, for CSV processing)

**Verification:**
```bash
python -c "import pennylane as qml; print(f'PennyLane {qml.__version__}')"
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

---

### Building LRET C++ Backend

#### Linux

```bash
cd /path/to/LRET
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Build (use all CPU cores)
make -j$(nproc)

# Verify build
ls -lh quantum_sim libquantum_sim.so
```

#### Windows (PowerShell)

```powershell
cd D:\LRET
mkdir build -Force
cd build

# Configure with CMake (auto-detects Visual Studio 2022/2019)
cmake .. -A x64

# Build
cmake --build . --config Release

# Verify build
Get-ChildItem quantum_sim.exe, Release\quantum_sim.lib
```

**Note:** CMake will automatically detect the newest installed Visual Studio version. If you have multiple versions installed, you can specify one explicitly:
```powershell
# Force specific VS version (optional)
cmake .. -G "Visual Studio 17 2022" -A x64  # VS 2022
cmake .. -G "Visual Studio 16 2019" -A x64  # VS 2019
```

**Common Build Issues:**

1. **Eigen3 not found:**
   - Linux: `sudo apt install libeigen3-dev`
   - Windows: Our CMakeLists.txt will auto-download it

2. **CMake version too old:**
   - Update CMake from https://cmake.org/download/

3. **Compiler not found:**
   - Linux: Install `build-essential`
   - Windows: Install Visual Studio Build Tools

---

### Installing LRET as PennyLane Device

After building the C++ backend, install the Python package:

```bash
cd /path/to/LRET/python
pip install -e .
```

This will:
- Build the Python bindings (`_qlret_native` C extension)
- Install the `qlret` Python package
- Register `qlret.mixed` as a PennyLane device

**Verification:**
```bash
python -c "import qlret; print(f'LRET version: {qlret.__version__}')"
python -c "import pennylane as qml; dev = qml.device('qlret.mixed', wires=4); print(f'Device: {dev.name}')"
```

**Expected Output:**
```
LRET version: 0.1.0
Device: qlret.mixed
```

---

### Running Benchmarks

Once setup is complete, you can run benchmarks:

```bash
cd /path/to/LRET

# Light test (4 qubits, ~10-20 minutes)
python benchmarks/pennylane/4q_50e_25s_10n.py

# Medium test (8 qubits, ~1-2 hours)
python benchmarks/pennylane/8q_100e_100s_12n.py

# Heavy test (8 qubits, ~3-5 hours)
python benchmarks/pennylane/8q_200e_200s_15n.py
```

Results are saved to:
```
D:/LRET/results/benchmark_YYYYMMDD_HHMMSS/
  ├── benchmark.log           # Full execution log
  ├── progress.log            # Training progress
  ├── results.json            # Summary statistics
  ├── lret_epochs.csv         # LRET training data
  └── baseline_epochs.csv     # default.mixed training data
```

---

## Troubleshooting

### Python Import Errors

**Error:** `ModuleNotFoundError: No module named 'qlret'`

**Solution:**
```bash
cd /path/to/LRET/python
pip install -e . --force-reinstall
```

### PennyLane Device Not Found

**Error:** `DeviceError: Device 'qlret.mixed' not found`

**Solution:**
1. Verify Python bindings built:
   ```bash
   ls python/qlret/build/*.so     # Linux
   ls python/qlret/build/*.pyd    # Windows
   ```
2. Rebuild if missing:
   ```bash
   cd python && pip install -e . --force-reinstall
   ```

### CMake Configuration Fails

**Error:** `Could not find Eigen3`

**Solution (Linux):**
```bash
sudo apt install libeigen3-dev
```

**Solution (Windows):**
- Let CMake auto-download (it should do this automatically)
- Or manually download Eigen3 and set: `cmake .. -DEIGEN3_INCLUDE_DIR="C:/path/to/eigen3"`

### Build Fails on Windows

**Error:** `error: Microsoft Visual C++ 14.0 or greater is required`

**Solution:**
- Install Visual Studio Build Tools: https://visualstudio.microsoft.com/downloads/
- Select "Desktop development with C++" workload
- Restart terminal after installation

---

## System Requirements Summary

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8 | 3.9-3.11 |
| RAM | 8 GB | 16 GB+ |
| Disk | 2 GB | 5 GB+ |
| CPU | 2 cores | 4+ cores |
| OS | Ubuntu 20.04 / Win10 | Ubuntu 22.04 / Win11 |

---

## Quick Reference

### Installation Commands (Linux)
```bash
# System packages
sudo apt update
sudo apt install python3 python3-pip build-essential cmake libeigen3-dev

# Python packages
pip install pennylane torch numpy scipy psutil matplotlib pandas

# Build LRET
cd /path/to/LRET
mkdir -p build && cd build
cmake .. && make -j$(nproc)

# Install Python package
cd ../python && pip install -e .

# Verify
python -c "import pennylane as qml; dev = qml.device('qlret.mixed', wires=4); print('✓ Setup complete!')"
```

### Installation Commands (Windows PowerShell)
```powershell
# Install Python packages
pip install pennylane torch numpy scipy psutil matplotlib pandas

# Build LRET
cd D:\LRET
mkdir build -Force; cd build
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release

# Install Python package
cd ..\python; pip install -e .

# Verify
python -c "import pennylane as qml; dev = qml.device('qlret.mixed', wires=4); print('✓ Setup complete!')"
```

---

## Next Steps

1. ✅ Complete system setup (automated or manual)
2. ✅ Verify LRET PennyLane device registration
3. ✅ Run a light benchmark to confirm functionality
4. 📊 Analyze results in the output directory
5. 🚀 Run larger benchmarks as needed

For automated setup, use: **`python setup_pennylane_env.py`**

---

*Last updated: January 2026*
