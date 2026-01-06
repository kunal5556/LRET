# LRET Test Execution Plan

**Date Created**: January 6, 2026  
**System**: macOS (ARM64 - Apple Silicon)  
**Branch**: feature/framework-integration  
**Status**: Phase 1 Complete - Ready for Testing

---

## ✅ Phase 1: Setup & Dependencies - COMPLETE

### System Configuration
- **OS**: Darwin Kernel 25.2.0 (ARM64)
- **CMake**: 4.2.1 ✅
- **Python**: 3.9.6 ✅
- **Eigen**: 5.0.1 ✅
- **OpenMP**: 21.1.8 ✅
- **Docker**: Installed ✅
- **Homebrew**: Available ✅

### Python Packages Installed
- ✅ numpy 1.26.4
- ✅ scipy 1.13.1
- ✅ matplotlib 3.9.4
- ✅ pytest 8.4.2
- ✅ pytest-cov 7.0.0
- ✅ pennylane 0.38.0
- ✅ seaborn 0.13.2
- ✅ pandas 2.3.3

### Project Structure Verified
```
LRET/
├── include/           (37 header files) ✅
├── src/               (33 source files) ✅
├── tests/             (21 test files) ✅
├── python/
│   ├── qlret/        ✅
│   ├── tests/        ✅
│   └── setup.py      ✅
├── CMakeLists.txt     ✅
├── test_*.cpp         (4 root test files) ✅
└── TESTING_BACKLOG.md ✅
```

---

## 📋 Test Execution Priority Plan

### **TIER 1: Critical Core Tests** (Execute First - ~2 hours)
**Goal**: Verify basic build and core functionality

#### 1.1 Build Project (30 mins)
```bash
cd /Users/suryanshsingh/Documents/LRET
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j8
```

**Success Criteria**:
- No compilation errors
- All targets build successfully
- Main executable `quantum_sim` created

#### 1.2 Core C++ Tests (1 hour)
Run in order:

**Test 1: test_simple** (5 mins)
```bash
./test_simple
```
Expected: Zero state creation for 2 qubits, dimensions 4x1

**Test 2: test_minimal** (5 mins)
```bash
./test_minimal
```
Expected: LRET vs FDM fidelity = 1.0 for unitary gates

**Test 3: test_fidelity** (10 mins)
```bash
./test_fidelity
```
Expected: Initial fidelity = 1.0, post-noise < 1.0

**Test 4: main benchmark** (15 mins)
```bash
./quantum_sim -n 8 -d 10
./quantum_sim -n 10 -d 15 --mode compare
```
Expected: Parallel speedup 2-5x, no crashes

**Test 5: demo_batch** (5 mins)
```bash
./demo_batch
```
Expected: Auto-tuned batch sizes scale with qubit count

#### 1.3 Initial Test Report (20 mins)
- Document results
- Note any failures
- Update TESTING_BACKLOG.md status

---

### **TIER 2: Python Integration Tests** (~1 hour)
**Goal**: Verify Python bindings and PennyLane device

#### 2.1 Python Package Installation (10 mins)
```bash
cd /Users/suryanshsingh/Documents/LRET/python
pip3 install -e . --user
python3 -c "import qlret; print(qlret.__version__)"
```

**Success Criteria**:
- Package installs without errors
- Version prints correctly
- Imports work

#### 2.2 Python API Tests (30 mins)
```bash
cd tests
pytest test_qlret_device.py -v --tb=short
```

**Expected**:
- 15+ tests pass
- Bell state expectations correct
- Gradients compute correctly

#### 2.3 Integration Tests (20 mins)
```bash
pytest integration/ -v --tb=short -m "not slow"
```

**Expected**:
- JSON execution tests pass
- PennyLane device works
- CLI regression tests pass (if quantum_sim built)

---

### **TIER 3: Docker Tests** (~1.5 hours)
**Goal**: Verify containerized execution

**NOTE**: Docker Desktop must be started manually before these tests!

#### 3.1 Start Docker (5 mins)
```bash
# Open Docker Desktop application
# Wait for Docker daemon to start
docker ps
```

#### 3.2 Build Docker Images (45 mins)
```bash
cd /Users/suryanshsingh/Documents/LRET

# Build base stages
docker build --target cpp-builder -t qlret:cpp-builder .
docker build --target python-builder -t qlret:python-builder .
docker build --target tester -t qlret:tester .

# Build final image
docker build -t qlret:latest .
```

**Success Criteria**:
- All stages complete without errors
- pytest runs in tester stage
- Final image < 2GB

#### 3.3 Docker Runtime Tests (20 mins)
```bash
# Test CLI
docker run --rm qlret:latest ./quantum_sim -n 6 -d 8

# Test Python
docker run --rm qlret:latest python -c "import qlret; print(qlret.__version__)"

# Test PennyLane
docker run --rm qlret:latest python -c "from qlret import QLRETDevice; print('OK')"
```

#### 3.4 Docker Integration Test (20 mins)
```bash
cd python
pytest tests/integration/test_docker_runtime.py -v
```

---

### **TIER 4: Noise & Calibration Tests** (~1 hour)
**Goal**: Verify noise modeling and calibration scripts

#### 4.1 Noise Import Test (10 mins)
```bash
cd build
./test_noise_import
```

**Expected**: JSON parsing, 3 error types recognized

#### 4.2 Advanced Noise Test (10 mins)
```bash
./test_advanced_noise
```

**Expected**: Time-varying, correlated noise work correctly

#### 4.3 Leakage Test (15 mins)
```bash
./test_leakage_measurement
```

**Expected**: 8 subtests pass, trace preservation maintained

#### 4.4 Calibration Scripts (25 mins)
```bash
cd ../scripts

# Test calibration functions
python3 test_calibration.py

# Generate test data
python3 generate_calibration_data.py --num-qubits 5 --depths 5,10,15 --trials 3 --output test_cal.csv

# Fit depolarizing model
python3 fit_depolarizing.py test_cal.csv --output test_depol.json
```

---

### **TIER 5: Benchmarking Tests** (~30 mins)
**Goal**: Performance measurement and analysis

#### 5.1 Quick Benchmark Suite (15 mins)
```bash
cd scripts
python3 benchmark_suite.py --quick --categories scaling,parallel --quantum-sim ../build/quantum_sim
```

**Expected**:
- Completes in < 2 minutes
- CSV results generated
- No benchmark failures

#### 5.2 Benchmark Analysis (10 mins)
```bash
python3 benchmark_analysis.py benchmark_output/benchmark_results.csv --print-summary
```

#### 5.3 Visualization (5 mins)
```bash
python3 benchmark_visualize.py benchmark_output/benchmark_results.csv --output plots/
```

---

### **TIER 6: Optional Advanced Tests** (Skip if no GPU/MPI)

#### 6.1 GPU Tests ⚠️ SKIP (No GPU on macOS)
- test_distributed_gpu
- GPU build tests
- GPU memory tests

#### 6.2 MPI Tests ⚠️ SKIP (No MPI configured)
- Multi-process tests
- Distributed simulation

#### 6.3 Autodiff Tests (15 mins - CPU only)
```bash
cd build
./test_autodiff
./test_autodiff_multi
```

**Expected**:
- Gradient tests pass
- Parameter-shift correct

#### 6.4 QEC Tests (30 mins)
```bash
# Stabilizer codes
./test_qec_stabilizer

# Syndrome extraction
./test_qec_syndrome

# Decoders
./test_qec_decoder

# Logical qubits
./test_qec_logical
```

---

## 🎯 Today's Testing Session Plan (5-6 hours)

### Session 1: Core Build & Tests (2 hours)
1. ✅ Setup complete
2. ⏳ Build project
3. ⏳ Run Tier 1 tests
4. ⏳ Document results

### Session 2: Python & Docker (2.5 hours)
1. ⏳ Python package installation
2. ⏳ Python tests
3. ⏳ Docker build & tests

### Session 3: Advanced Tests (1 hour)
1. ⏳ Noise tests
2. ⏳ Benchmarking
3. ⏳ QEC tests (if time permits)

### Session 4: Documentation (30 mins)
1. ⏳ Update TESTING_BACKLOG.md with results
2. ⏳ Create summary report
3. ⏳ Identify issues for fixing

---

## 📊 Expected Results Summary

### Critical Success Metrics
- ✅ Build completes: 100% required
- ✅ Core tests pass: >90% required
- ✅ Python tests pass: >85% required
- ✅ Docker builds: 100% required

### Known Limitations
- ❌ No GPU tests (no hardware)
- ❌ No MPI tests (not configured)
- ⚠️ Some ML tests may skip (JAX/PyTorch optional)

### Test Coverage Breakdown
| Category | Total Tests | Runnable | GPU-Only | MPI-Only |
|----------|-------------|----------|----------|----------|
| Core C++ | 5 | 5 | 0 | 0 |
| Python | 15 | 15 | 0 | 0 |
| Integration | 23 | 20 | 0 | 3 |
| Docker | 3 | 3 | 0 | 0 |
| Noise | 8 | 8 | 0 | 0 |
| Benchmark | 15 | 15 | 0 | 0 |
| Autodiff | 2 | 2 | 0 | 0 |
| QEC | 90+ | 90+ | 0 | 10 |
| **Total** | **~200** | **~180** | **~10** | **~10** |

---

## 🚨 Failure Response Plan

### If Build Fails
1. Check CMake output for missing dependencies
2. Verify Eigen3 paths
3. Check OpenMP configuration
4. Review compiler errors

### If Core Tests Fail
1. Check executable permissions
2. Verify input files exist
3. Review error messages
4. Check memory/stack limits

### If Python Tests Fail
1. Verify package installation
2. Check for import errors
3. Ensure native module built (if needed)
4. Review pytest output

### If Docker Tests Fail
1. Ensure Docker daemon running
2. Check Dockerfile syntax
3. Review build logs
4. Verify base images available

---

## 📝 Next Steps After Phase 1

1. **Immediate**: Begin Tier 1 tests (build + core tests)
2. **Today**: Complete Tiers 1-3 (core + Python + Docker)
3. **Tomorrow**: Tiers 4-5 (noise + benchmarking)
4. **If issues found**: Switch to GPT-5.1 Codex Max for fixes
5. **Final review**: Switch to Opus 4.5 for optimization

---

## 🔄 Test Status Tracking

Use this checklist during execution:

### Build Status
- [ ] CMake configuration successful
- [ ] C++ compilation successful
- [ ] Python module built
- [ ] All executables created

### Core Tests
- [ ] test_simple PASSED
- [ ] test_minimal PASSED
- [ ] test_fidelity PASSED
- [ ] quantum_sim benchmark PASSED
- [ ] demo_batch PASSED

### Python Tests
- [ ] Package installation PASSED
- [ ] Unit tests PASSED (15/15)
- [ ] Integration tests PASSED (>18/23)

### Docker Tests
- [ ] Build stages PASSED
- [ ] Runtime CLI PASSED
- [ ] Runtime Python PASSED

### Advanced Tests
- [ ] Noise tests PASSED
- [ ] Benchmarking PASSED
- [ ] QEC tests PASSED (optional)

---

**Ready to proceed with Tier 1: Build and Core Tests?**
