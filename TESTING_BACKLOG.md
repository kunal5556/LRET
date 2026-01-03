# LRET Testing Backlog - Comprehensive Test Documentation

**Date Created:** January 3, 2026  
**Purpose:** Document all tests that were skipped due to system configuration issues  
**Target System:** Linux/macOS with proper CMake, Eigen3, Docker setup  
**Status:** PENDING - All tests to be executed on target system

---

## Executive Summary

This document catalogs **all testing tasks** that were planned but not executed during the development of LRET quantum simulator from Phase 1 through Phase 6. Due to Windows PowerShell and system configuration limitations, these tests need to be executed on a properly configured Linux/macOS development environment.

**Test Categories:**
1. Core C++ Unit Tests (Phase 1-2)
2. Advanced Noise Model Tests (Phase 4.1-4.5)
3. Python Integration Tests (Phase 5)
4. Docker Multi-Stage Build Tests (Phase 6)
5. Calibration Script Tests (Phase 4.2)
6. Performance Benchmarks (All Phases)

**Estimated Testing Time:** 6-8 hours for complete test suite execution  
**Prerequisites:** CMake 3.16+, Eigen3, Python 3.11+, Docker, pytest, PennyLane

---

## Table of Contents

1. [Phase 1: Core LRET Tests](#phase-1-core-lret-tests)
2. [Phase 2: GPU Tests (Optional)](#phase-2-gpu-tests-optional)
3. [Phase 3: MPI Tests (Optional)](#phase-3-mpi-tests-optional)
4. [Phase 4.1: Noise Import Tests](#phase-41-noise-import-tests)
5. [Phase 4.2: Calibration Script Tests](#phase-42-calibration-script-tests)
6. [Phase 4.3: Advanced Noise Tests](#phase-43-advanced-noise-tests)
7. [Phase 4.4-4.5: Leakage and Measurement Tests](#phase-44-45-leakage-and-measurement-tests)
8. [Phase 5: Python Integration Tests](#phase-5-python-integration-tests)
9. [Phase 6: Docker Integration Tests](#phase-6-docker-integration-tests)
10. [Performance Benchmarks](#performance-benchmarks)
11. [Execution Instructions](#execution-instructions)

---

## Phase 1: Core LRET Tests

### 1.1 Basic Simulation Tests

**File:** `test_simple.cpp`  
**Status:** ❌ NOT RUN  
**Purpose:** Verify basic state creation and simulation initialization

**Test Commands:**
```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target test_simple
./test_simple
```

**Expected Output:**
```
Starting simple test...
Creating zero state for 2 qubits...
Zero state created. Dimensions: 4x1
Test passed!
```

**Success Criteria:**
- Exit code: 0
- No segmentation faults
- Correct state dimensions (2^n × 1)

---

### 1.2 Fidelity Computation Tests

**File:** `test_fidelity.cpp`  
**Status:** ❌ NOT RUN  
**Purpose:** Validate fidelity calculations between LRET and FDM

**Test Commands:**
```bash
./build/test_fidelity
```

**Expected Output:**
```
=== Fidelity Debug Test ===

1. Creating initial states (n=2 qubits)...
   L initial trace (||L||_F^2): 1.0000
   rho_lret trace: 1.0000
   rho_fdm trace: 1.0000

2. Initial state fidelity: 1.0000 (should be 1.0)

3. Applying Hadamard gate...
   Fidelity after H: 1.0000 (should be 1.0)

4. Applying CNOT gate...
   Fidelity after CNOT: 1.0000 (should be 1.0)

5. Applying depolarizing noise...
   Fidelity after noise: 0.9950 (should be < 1.0)

All fidelity tests passed!
```

**Success Criteria:**
- Initial fidelity = 1.0 (±1e-10)
- Post-unitary fidelity = 1.0 (±1e-10)
- Post-noise fidelity < 1.0
- No NaN or Inf values
- Exit code: 0

**Known Issues to Watch:**
- Trace preservation after noise
- Hermiticity of density matrices
- Numerical precision for small fidelities

---

### 1.3 Minimal LRET vs FDM Test

**File:** `test_minimal.cpp`  
**Status:** ❌ NOT RUN  
**Purpose:** Direct comparison of LRET and FDM evolution on minimal circuit

**Test Commands:**
```bash
./build/test_minimal
```

**Expected Output:**
```
=== Minimal LRET vs FDM Test ===

Initial state (n=2):
LRET L matrix (4x1):
  [0] = 1.0
  [1] = 0.0
  [2] = 0.0
  [3] = 0.0

FDM rho (4x4):
  1.0000  0.0000  0.0000  0.0000
  0.0000  0.0000  0.0000  0.0000
  0.0000  0.0000  0.0000  0.0000
  0.0000  0.0000  0.0000  0.0000

Applying H(0)...
Fidelity after H: 1.0000

Applying CNOT(0,1)...
Fidelity after CNOT: 1.0000

Applying depolarizing noise (p=0.01)...
Fidelity after noise: 0.9900

Test passed! Max deviation: 1.23e-12
```

**Success Criteria:**
- All fidelities match (±1e-10)
- LRET rank grows appropriately with noise
- No crashes or assertion failures
- Exit code: 0

---

### 1.4 Main Benchmark Test

**File:** `main.cpp`  
**Status:** ❌ NOT RUN  
**Purpose:** Full benchmark with parallel modes, FDM comparison, CSV output

**Test Commands:**
```bash
# Basic run (n=11, d=13, default settings)
./build/quantum_sim

# Compare all parallel modes
./build/quantum_sim -n 10 -d 15 --mode compare

# Test FDM integration
./build/quantum_sim -n 8 -d 10 --fdm

# Test CSV output
./build/quantum_sim -n 12 -d 20 -o results.csv
cat results.csv
```

**Expected Output Sections:**
1. **Configuration Summary:**
   ```
   number of qubits: 11
   depth: 13
   batch_size: 64 (auto-selected)
   epsilon: 1e-4
   ```

2. **LRET Simulation:**
   ```
   ===== Running LRET simulation for 11 qubits =====
   Simulation Time: 0.15 seconds
   Final Rank: 13
   Fidelity: 0.9987
   ```

3. **Mode Comparison:**
   ```
   Mode            | Time (s) | Speedup
   ----------------|----------|--------
   sequential      | 0.68     | 1.0x
   row             | 0.22     | 3.1x
   column          | 0.24     | 2.8x
   hybrid          | 0.15     | 4.5x
   ```

4. **FDM Validation (if --fdm):**
   ```
   ===== FDM Simulation =====
   Simulation Time: 0.45 seconds
   Trace Distance: 1.23e-05
   Fidelity: 0.9987
   ```

**Success Criteria:**
- No segmentation faults
- Parallel modes show 2-5x speedup
- FDM fidelity > 0.95 (for low noise)
- CSV output is well-formed
- Exit code: 0

---

### 1.5 Batch Heuristic Demo

**File:** `tests/demo_batch.cpp`  
**Status:** ❌ NOT RUN  
**Purpose:** Verify auto-tuned batch size selection across qubit counts

**Test Commands:**
```bash
./build/demo_batch
```

**Expected Output:**
```
Batch Size Auto-Tuning Demo
============================

INFO: n=11 low-workload, batch_size=64
for 11 qubits number of batches are 64

INFO: n=12 low-workload, batch_size=64
for 12 qubits number of batches are 64

INFO: n=13 medium-workload, batch_size=128
for 13 qubits number of batches are 128

INFO: n=14 medium-workload, batch_size=128
for 14 qubits number of batches are 128

INFO: n=15 high-workload, batch_size=256
for 15 qubits number of batches are 256

INFO: n=16 high-workload, batch_size=256
for 16 qubits number of batches are 256

Summary:
- Low workload (n<=12): batch=64
- Medium workload (n=13-14): batch=128
- High workload (n>=15): batch=256
```

**Success Criteria:**
- Batch sizes scale with qubit count
- No crashes for any n value
- Reasonable memory usage
- Exit code: 0

---

## Phase 2: GPU Tests (Optional)

### 2.1 GPU Build Test

**Status:** ⚠️ OPTIONAL (requires NVIDIA GPU + cuQuantum)  
**Purpose:** Verify GPU compilation and runtime

**Test Commands:**
```bash
# Build with GPU support
cd build
cmake .. -DUSE_GPU=ON -DCMAKE_BUILD_TYPE=Release
cmake --build .

# Check GPU detection
./quantum_sim --gpu-info

# Run GPU simulation
./quantum_sim -n 15 -d 20 --gpu
```

**Expected Output:**
```
GPU Configuration:
- CUDA Version: 12.3
- cuQuantum: FOUND
- GPU: NVIDIA A100 (80GB)
- Compute Capability: 8.0

Running GPU simulation...
Simulation Time: 0.08 seconds (vs 2.3s CPU)
GPU Speedup: 28.7x
```

**Success Criteria:**
- Build completes without CUDA errors
- GPU properly detected
- Simulation runs faster than CPU
- Results match CPU within tolerance (fidelity > 0.99)

---

## Phase 3: MPI Tests (Optional)

### 3.1 MPI Build and Run Test

**Status:** ⚠️ OPTIONAL (requires MPI installation)  
**Purpose:** Verify distributed parallel execution

**Test Commands:**
```bash
# Build with MPI
cd build
cmake .. -DUSE_MPI=ON
cmake --build .

# Run on 4 nodes
mpirun -np 4 ./quantum_sim -n 16 -d 25

# Verify scaling
for np in 1 2 4 8; do
    echo "Testing with $np processes:"
    mpirun -np $np ./quantum_sim -n 14 -d 20
done
```

**Expected Output:**
```
MPI Configuration:
- Processes: 4
- Rank 0: localhost (master)
- Rank 1-3: workers

Running distributed simulation...
- Rank 0: Processing rows 0-511
- Rank 1: Processing rows 512-1023
- Rank 2: Processing rows 1024-1535
- Rank 3: Processing rows 1536-2047

Total Time: 0.65 seconds
Linear Speedup: 3.8x (95% efficiency)
```

**Success Criteria:**
- All ranks complete without deadlock
- Near-linear speedup
- Results identical to single-process run
- Exit code: 0 on all ranks

---

## Phase 4.1: Noise Import Tests

### 4.1.1 Basic Noise Import Test

**File:** `test_noise_import.cpp`  
**Status:** ❌ NOT RUN  
**Purpose:** Validate JSON noise model parsing and conversion

**Test Commands:**
```bash
./build/test_noise_import
```

**Expected Output:**
```
=== LRET Noise Model Import Test ===

Test 1: Parsing JSON...
✓ Parsed 3 errors

Test 2: Validating noise model...
✓ Noise model is valid

Test 3: Noise model summary:
----------------------------------------
Device: test_device
Backend Version: 1.0.0
Total Errors: 3

Error 1: qerror
  Operations: cx
  Gate Qubits: [[0, 1]]
  Probabilities: [0.99, 0.01]

Error 2: depolarizing
  Operations: x, y, z, h
  Gate Qubits: [[0], [1], [2]]
  Param: 0.001

Error 3: thermal_relaxation
  Operations: id
  Gate Qubits: [[0], [1]]
  T1: 50000.0 ns
  T2: 70000.0 ns
  Gate Time: 35.0 ns
----------------------------------------

Test 4: Error lookup...
✓ Found 1 error(s) for CNOT(0,1)

Test 5: Converting Qiskit errors to LRET...
✓ Depolarizing error → 4 LRET noise op(s)
✓ Thermal relaxation → 2 LRET noise op(s)

Test 6: Applying noise to circuit...
✓ Clean circuit: 3 operations
✓ Noisy circuit: 9 operations
  (Noise added: 6 operations)

All tests passed!
```

**Success Criteria:**
- JSON parsing succeeds
- All 3 error types recognized
- Noise model validation passes
- Correct error lookup by gate name
- Circuit operations increase after noise application
- Exit code: 0

---

### 4.1.2 Qiskit Noise Model File Test

**File:** `scripts/sample_noise_with_leakage.json`  
**Status:** ❌ NOT RUN  
**Purpose:** Load and validate production Qiskit noise model

**Test Commands:**
```bash
# Test with sample file
./build/test_noise_import --input scripts/sample_noise_with_leakage.json

# Test with downloaded IBM noise
python3 scripts/download_ibm_noise.py --backend ibmq_manila --output ibm_noise.json
./build/test_noise_import --input ibm_noise.json
```

**Expected Output:**
```
Loading noise model from: scripts/sample_noise_with_leakage.json

Noise Model Summary:
- Device: fake_jakarta
- Errors: 47
  * Depolarizing: 23
  * Thermal Relaxation: 20
  * Readout: 3
  * Leakage: 1

Applying to test circuit (n=5, d=10)...
✓ Circuit size: 50 → 97 operations
✓ Average error rate: 0.0134

Simulation with noise model:
✓ Fidelity: 0.8721 (without noise: 0.9998)
✓ Final rank: 34 (pure state: 1)

Test passed!
```

**Success Criteria:**
- File loads without JSON errors
- All error types parsed
- Circuit successfully modified
- Fidelity degrades with noise
- Rank increases appropriately

---

## Phase 4.2: Calibration Script Tests

### 4.2.1 Unit Tests for Calibration

**File:** `scripts/test_calibration.py`  
**Status:** ❌ NOT RUN  
**Purpose:** Validate calibration fitting functions

**Test Commands:**
```bash
cd scripts
python3 test_calibration.py
```

**Expected Output:**
```
Testing calibration functions...

Test: estimate_depolarizing
  Input fidelity: 0.95
  Estimated p: 0.0167
  ✓ PASS

Test: fit_exponential
  Sample data: [1.0, 0.9, 0.81, 0.73]
  Fitted τ: 10.03
  R²: 0.998
  ✓ PASS

All calibration tests passed!
```

**Success Criteria:**
- All test functions pass
- Fitting converges
- R² > 0.95 for synthetic data
- Exit code: 0

---

### 4.2.2 Generate Calibration Data

**File:** `scripts/generate_calibration_data.py`  
**Status:** ❌ NOT RUN  
**Purpose:** Generate synthetic calibration dataset for testing

**Test Commands:**
```bash
python3 scripts/generate_calibration_data.py \
    --num-qubits 5 \
    --depths 5,10,15,20,25,30 \
    --trials 10 \
    --output calibration_data.csv

# Check output
head -20 calibration_data.csv
```

**Expected Output:**
```
Generating calibration data...
Configuration:
  Qubits: 5
  Depths: [5, 10, 15, 20, 25, 30]
  Trials per depth: 10

Progress: [████████████████████] 60/60 circuits

Output saved to: calibration_data.csv
Columns: depth, trial, fidelity, rank, time_ms

Sample data:
  depth=5:  mean_fidelity=0.9876 ± 0.0023
  depth=10: mean_fidelity=0.9521 ± 0.0045
  depth=20: mean_fidelity=0.8234 ± 0.0089
  depth=30: mean_fidelity=0.6891 ± 0.0134
```

**Success Criteria:**
- CSV file created
- All depths covered
- Fidelity decreases with depth
- No NaN or negative values
- Exit code: 0

---

### 4.2.3 Fit Depolarizing Model

**File:** `scripts/fit_depolarizing.py`  
**Status:** ❌ NOT RUN  
**Purpose:** Fit depolarizing noise parameter from calibration data

**Test Commands:**
```bash
python3 scripts/fit_depolarizing.py calibration_data.csv \
    --output depolarizing_params.json

cat depolarizing_params.json
```

**Expected Output:**
```
Fitting depolarizing model to calibration data...

Data summary:
  Circuits: 60
  Depths: 5-30
  Mean fidelity: 0.8724

Optimization:
  Initial p: 0.01
  Fitted p: 0.00234
  R²: 0.987
  RMSE: 0.0123

Model saved to: depolarizing_params.json

Validation:
  Predicted fidelity (d=15): 0.9245
  Actual mean fidelity:       0.9241
  Error: 0.04%
```

**Success Criteria:**
- Optimization converges
- R² > 0.95
- Fitted p in reasonable range (1e-5 to 0.1)
- JSON output valid
- Exit code: 0

---

### 4.2.4 Fit T1/T2 Relaxation

**File:** `scripts/fit_t1_t2.py`  
**Status:** ❌ NOT RUN  
**Purpose:** Extract T1/T2 parameters from time-dependent data

**Test Commands:**
```bash
python3 scripts/fit_t1_t2.py relaxation_data.csv \
    --output t1_t2_params.json

cat t1_t2_params.json
```

**Expected Output:**
```
Fitting T1/T2 relaxation parameters...

Data:
  Time points: 0-1000 ns
  Samples: 50

T1 Fitting:
  Fitted T1: 47.3 μs
  95% CI: [45.2, 49.4] μs
  R²: 0.994

T2 Fitting:
  Fitted T2: 68.1 μs
  95% CI: [65.8, 70.4] μs
  R²: 0.989

Validation: T2 < 2*T1 ✓

Parameters saved to: t1_t2_params.json
```

**Success Criteria:**
- Both fits converge
- T2 ≤ 2*T1 (physical constraint)
- R² > 0.95 for both
- Confidence intervals reasonable
- Exit code: 0

---

### 4.2.5 Calibrate Full Noise Model

**File:** `scripts/calibrate_noise_model.py`  
**Status:** ❌ NOT RUN  
**Purpose:** End-to-end calibration pipeline

**Test Commands:**
```bash
# Full pipeline
python3 scripts/calibrate_noise_model.py \
    --input calibration_data.csv \
    --output calibrated_model.json \
    --model-type full

# Verify output
python3 -c "import json; print(json.dumps(json.load(open('calibrated_model.json')), indent=2))"
```

**Expected Output:**
```
=== LRET Noise Model Calibration ===

Step 1: Loading calibration data...
  ✓ Loaded 60 data points

Step 2: Fitting depolarizing channel...
  ✓ p = 0.00234 (R² = 0.987)

Step 3: Fitting thermal relaxation...
  ✓ T1 = 47.3 μs (R² = 0.994)
  ✓ T2 = 68.1 μs (R² = 0.989)

Step 4: Fitting time-dependent scaling...
  ✓ α = 1.23 (R² = 0.976)

Step 5: Building noise model...
  ✓ Gates covered: 12
  ✓ Qubits: 5

Step 6: Validating model...
  ✓ All parameters physical
  ✓ Trace preservation verified

Calibrated model saved to: calibrated_model.json

Cross-validation:
  Test set fidelity error: 2.3% (RMSE: 0.019)
  ✓ Model generalizes well
```

**Success Criteria:**
- All fitting steps succeed
- R² > 0.95 for all models
- Output is valid Qiskit noise format
- Cross-validation error < 5%
- Exit code: 0

---

### 4.2.6 Compare Fidelities

**File:** `scripts/compare_fidelities.py`  
**Status:** ❌ NOT RUN  
**Purpose:** Statistical comparison of calibrated vs reference

**Test Commands:**
```bash
python3 scripts/compare_fidelities.py \
    reference.csv \
    calibrated.csv \
    --column fidelity
```

**Expected Output:**
```
Comparing fidelities...

Reference data:
  Mean: 0.8724
  Std:  0.0456
  N:    60

Test data:
  Mean: 0.8698
  Std:  0.0472
  N:    60

Statistics:
  Delta (test - ref): -0.0026
  Relative error:     -0.30%
  t-statistic:        -0.34
  p-value:            0.736

Conclusion: No significant difference (p > 0.05) ✓
```

**Success Criteria:**
- Both datasets load
- Statistical test runs
- Outputs are interpretable
- Exit code: 0

---

## Phase 4.3: Advanced Noise Tests

### 4.3.1 Advanced Noise C++ Tests

**File:** `tests/test_advanced_noise.cpp`  
**Status:** ❌ NOT RUN  
**Purpose:** Test time-dependent, correlated, and memory-effect noise

**Test Commands:**
```bash
./build/test_advanced_noise
```

**Expected Output:**
```
=== Advanced Noise Tests ===

Test 1: Time-varying noise (linear scaling)
  Base rate: 0.01
  Time step: 5
  Scaled rate: 0.015
  ✓ Scaling works correctly

Test 2: Correlated Pauli channel
  Initial rank: 1
  Post-correlation rank: 2
  ✓ Rank doubles as expected

Test 3: Memory effects
  Previous gate: X
  Current gate: Z
  Memory scale: 0.7
  ✓ Error rate reduced by memory

Test 4: Combined noise model
  Clean fidelity: 1.000
  Noisy fidelity: 0.876
  Rank growth: 1 → 12
  ✓ All effects combined correctly

All advanced noise tests passed.
```

**Success Criteria:**
- Time scaling increases error rates
- Correlated noise increases rank
- Memory effects reduce subsequent errors
- Combined model produces expected fidelity
- Exit code: 0

---

### 4.3.2 Correlated Error Fitting

**File:** `scripts/fit_correlated_errors.py`  
**Status:** ❌ NOT RUN  
**Purpose:** Extract two-qubit correlated error rates

**Test Commands:**
```bash
python3 scripts/fit_correlated_errors.py \
    correlated_data.csv \
    --output correlated_params.json
```

**Expected Output:**
```
Fitting correlated error model...

Data:
  Two-qubit gates: CNOT, CZ
  Samples: 120

Single-qubit depolarizing:
  p_single = 0.0023

Two-qubit correlated:
  p_corr(ZZ) = 0.0145
  p_corr(XX) = 0.0089
  p_corr(YY) = 0.0102

Correlation matrix:
        I       X       Y       Z
  I   0.955   0.012   0.010   0.012
  X   0.012   0.003   0.001   0.003
  Y   0.010   0.001   0.002   0.002
  Z   0.012   0.003   0.002   0.009

Model saved to: correlated_params.json
```

**Success Criteria:**
- Fitting converges
- All probabilities sum to 1.0
- Correlation matrix is valid
- Exit code: 0

---

### 4.3.3 Time Scaling Detection

**File:** `scripts/fit_time_scaling.py`  
**Status:** ❌ NOT RUN  
**Purpose:** Detect time-dependent error rate growth

**Test Commands:**
```bash
python3 scripts/fit_time_scaling.py \
    time_series_data.csv \
    --output time_scaling.json
```

**Expected Output:**
```
Fitting time-dependent scaling...

Models tested:
  1. Linear:      α*t
  2. Quadratic:   α*t²
  3. Exponential: exp(α*t)

Best fit: Linear
  α = 1.234
  R² = 0.982
  AIC = -145.3

Model comparison:
  Linear:      R²=0.982, AIC=-145.3 ✓ BEST
  Quadratic:   R²=0.978, AIC=-142.1
  Exponential: R²=0.945, AIC=-128.7

Saved to: time_scaling.json
```

**Success Criteria:**
- At least one model fits (R² > 0.90)
- Best model selected by AIC
- Parameters physically reasonable
- Exit code: 0

---

### 4.3.4 Memory Effect Detection

**File:** `scripts/detect_memory_effects.py`  
**Status:** ❌ NOT RUN  
**Purpose:** Detect gate-sequence dependent error correlations

**Test Commands:**
```bash
python3 scripts/detect_memory_effects.py \
    sequence_data.csv \
    --output memory_effects.json
```

**Expected Output:**
```
Detecting memory effects in gate sequences...

Analyzing sequences:
  Total circuits: 200
  Sequence length: 5-20 gates
  Gate types: H, X, Y, Z, CNOT

Significant memory effects found: 3

Effect 1: X → Z
  Error scale: 0.72
  p-value: 0.002
  Memory depth: 1
  ✓ Statistically significant

Effect 2: CNOT → CNOT (same qubits)
  Error scale: 1.15
  p-value: 0.018
  Memory depth: 2
  ✓ Error rate increases

Effect 3: H → H (same qubit)
  Error scale: 0.85
  p-value: 0.031
  Memory depth: 1
  ✓ Minor reduction

Saved to: memory_effects.json
```

**Success Criteria:**
- At least one effect detected (if present in data)
- p-values < 0.05 for reported effects
- Error scales physically reasonable (0.5-1.5)
- Exit code: 0

---

## Phase 4.4-4.5: Leakage and Measurement Tests

### 4.4.1 Leakage Channel Tests

**File:** `tests/test_leakage_measurement.cpp`  
**Status:** ❌ NOT RUN  
**Purpose:** Validate leakage and measurement error implementations

**Test Commands:**
```bash
./build/test_leakage_measurement
```

**Expected Output:**
```
=== Leakage and Measurement Tests ===

Test 1: Leakage Kraus operators
  Number of Kraus ops: 2
  Trace preservation: ✓ (error < 1e-10)
  ✓ PASS

Test 2: Leakage relaxation Kraus
  Number of Kraus ops: 2
  Trace preservation: ✓
  T2 < 2*T1: ✓
  ✓ PASS

Test 3: Apply leakage channel
  Initial rank: 1
  Post-leakage rank: 2
  Trace before: 1.0000
  Trace after:  1.0000
  ✓ PASS

Test 4: Full leakage model
  p_leak: 0.05
  p_relax: 0.02
  p_phase: 0.01
  Final rank: 4
  Fidelity: 0.9412
  ✓ PASS

Test 5: Measurement error (bitflip)
  p(0→1): 0.02
  p(1→0): 0.03
  Pre-measurement state: |0⟩
  Post-error probabilities: [0.98, 0.02]
  ✓ PASS

Test 6: Measurement error (full POVM)
  Confusion matrix valid: ✓
  Row sums = 1: ✓
  ✓ PASS

Test 7: Readout calibration
  Calibration matrix:
    [0.97  0.03]
    [0.02  0.98]
  Inverted successfully: ✓
  ✓ PASS

Test 8: Full simulation with leakage + measurement
  Initial fidelity: 1.000
  Post-gates fidelity: 0.998
  Post-leakage fidelity: 0.945
  Post-measurement fidelity: 0.928
  ✓ All effects applied correctly

All leakage and measurement tests passed!
```

**Success Criteria:**
- All 8 test sections pass
- Trace preservation maintained
- Rank growth as expected
- Fidelity degrades appropriately
- Exit code: 0

---

### 4.4.2 IBM Noise Download Script

**File:** `scripts/download_ibm_noise.py`  
**Status:** ❌ NOT RUN  
**Purpose:** Download real device noise models from IBM Quantum

**Test Commands:**
```bash
# Test with fake backend (no IBM account needed)
python3 scripts/download_ibm_noise.py \
    --fake-backend FakeQuito \
    --output fake_quito_noise.json

# Verify output
python3 -c "import json; model=json.load(open('fake_quito_noise.json')); print(f'Errors: {len(model[\"errors\"])}')"
```

**Expected Output:**
```
Downloading noise model...

Configuration:
  Backend: FakeQuito (fake backend)
  Output: fake_quito_noise.json

Backend properties:
  Qubits: 5
  Gates: ['id', 'rz', 'sx', 'x', 'cx']
  T1 range: 35-120 μs
  T2 range: 45-95 μs

Extracting noise parameters...
  ✓ Gate errors: 15
  ✓ Thermal relaxation: 5
  ✓ Readout errors: 5

Noise model saved to: fake_quito_noise.json
Total errors: 25

Sample error rates:
  Single-qubit: 0.0003-0.0012
  CNOT: 0.008-0.015
  Readout: 0.01-0.03
```

**Success Criteria:**
- JSON file created
- Valid noise model structure
- Error rates in realistic ranges
- Exit code: 0

---

## Phase 5: Python Integration Tests

### 5.1 Python Package Installation Test

**File:** `python/setup.py`  
**Status:** ❌ NOT RUN  
**Purpose:** Verify Python package installation

**Test Commands:**
```bash
# Install in development mode
cd python
pip install -e .[dev]

# Verify installation
python -c "import qlret; print(qlret.__version__)"
python -c "from qlret import simulate_json, QLRETDevice; print('Imports OK')"
```

**Expected Output:**
```
Installing qlret package...
  ✓ numpy>=1.20
  ✓ pennylane>=0.30 (optional)
  ✓ pytest>=7.0 (dev)

Successfully installed qlret-1.0.0

Import test:
1.0.0
Imports OK
```

**Success Criteria:**
- Installation completes
- Version string correct
- All imports work
- Exit code: 0

---

### 5.2 Python API Tests

**File:** `python/tests/test_qlret_device.py`  
**Status:** ❌ NOT RUN  
**Purpose:** Comprehensive Python bridge testing

**Test Commands:**
```bash
cd python
pytest tests/test_qlret_device.py -v --tb=short
```

**Expected Test Results:**
```
tests/test_qlret_device.py::TestLoadJsonFile::test_load_bell_pair PASSED
tests/test_qlret_device.py::TestSimulateJson::test_simulate_native PASSED
tests/test_qlret_device.py::TestSimulateJson::test_simulate_subprocess PASSED
tests/test_qlret_device.py::TestSimulateJson::test_simulate_with_sampling PASSED
tests/test_qlret_device.py::TestSimulateJson::test_invalid_circuit PASSED
tests/test_qlret_device.py::TestQLRETDevice::test_device_creation PASSED
tests/test_qlret_device.py::TestQLRETDevice::test_device_capabilities PASSED
tests/test_qlret_device.py::TestQLRETDevice::test_tape_to_json PASSED
tests/test_qlret_device.py::TestQLRETDevice::test_bell_state_expectation PASSED
tests/test_qlret_device.py::TestQLRETDevice::test_parametrized_circuit PASSED
tests/test_qlret_device.py::TestGradients::test_gradient_single_param PASSED
tests/test_qlret_device.py::TestGradients::test_gradient_multi_param PASSED
tests/test_qlret_device.py::TestNativeModule::test_version PASSED
tests/test_qlret_device.py::TestNativeModule::test_validate_circuit PASSED
tests/test_qlret_device.py::TestNativeModule::test_validate_invalid PASSED

========================= 15 passed in 12.34s =========================
```

**Success Criteria:**
- All 15 tests pass
- No skipped tests (unless optional features disabled)
- Native module loads (if built)
- Subprocess backend works
- Exit code: 0

---

### 5.3 PennyLane Device Integration Test

**Test Commands:**
```bash
python3 << 'EOF'
import pennylane as qml
from qlret import QLRETDevice
import numpy as np

# Create device
dev = QLRETDevice(wires=2, shots=None)

# Define quantum circuit
@qml.qnode(dev)
def circuit(theta):
    qml.Hadamard(wires=0)
    qml.RX(theta, wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

# Execute
result = circuit(0.5)
print(f"Expectation: {result:.4f}")

# Compute gradient
grad = qml.grad(circuit)(0.5)
print(f"Gradient: {grad:.4f}")

print("PennyLane device test: PASSED")
EOF
```

**Expected Output:**
```
Expectation: 0.8776
Gradient: -0.4794
PennyLane device test: PASSED
```

**Success Criteria:**
- Device creation succeeds
- Circuit executes
- Gradient computation works
- Results are numerically reasonable
- Exit code: 0

---

### 5.4 JSON API Functional Test

**Test Commands:**
```bash
python3 << 'EOF'
from qlret import simulate_json

# Define Bell pair circuit
circuit = {
    "circuit": {
        "num_qubits": 2,
        "operations": [
            {"name": "H", "wires": [0]},
            {"name": "CNOT", "wires": [0, 1]}
        ],
        "observables": [
            {"type": "PAULI", "operator": "Z", "wires": [0]},
            {"type": "TENSOR", "operators": ["Z", "Z"], "wires": [0, 1]}
        ]
    },
    "config": {"epsilon": 1e-4, "initial_rank": 1}
}

# Simulate
result = simulate_json(circuit, use_native=False)

print(f"Status: {result['status']}")
print(f"Execution time: {result['execution_time_ms']:.2f} ms")
print(f"Final rank: {result['final_rank']}")
print(f"Expectation <Z_0>: {result['expectation_values'][0]:.4f}")
print(f"Expectation <Z_0 Z_1>: {result['expectation_values'][1]:.4f}")

# Verify Bell state properties
assert abs(result['expectation_values'][0]) < 0.1  # <Z_0> ≈ 0
assert abs(result['expectation_values'][1] - 1.0) < 0.1  # <Z_0 Z_1> ≈ 1

print("JSON API test: PASSED")
EOF
```

**Expected Output:**
```
Status: success
Execution time: 15.32 ms
Final rank: 1
Expectation <Z_0>: 0.0000
Expectation <Z_0 Z_1>: 1.0000
JSON API test: PASSED
```

**Success Criteria:**
- Simulation succeeds
- Bell state expectations correct
- Execution time reasonable
- Exit code: 0

---

## Phase 6: Docker Integration Tests

### 6.1 Docker Multi-Stage Build Test

**Status:** ❌ NOT RUN  
**Purpose:** Verify all Docker build stages

**Test Commands:**
```bash
# Build cpp-builder stage
docker build --target cpp-builder -t qlret:cpp-builder .

# Build python-builder stage
docker build --target python-builder -t qlret:python-builder .

# Build tester stage (includes pytest gate)
docker build --target tester -t qlret:tester .

# Build final runtime stage
docker build -t qlret:latest .
```

**Expected Output (tester stage):**
```
Step 1/25 : FROM python:3.11-slim AS cpp-builder
...
Step 15/25 : RUN pytest tests/ -v --tb=short
 ---> Running in abc123def456

======================== test session starts ========================
tests/test_qlret_device.py::test_load_bell_pair PASSED
tests/test_qlret_device.py::test_simulate_native PASSED
tests/test_qlret_device.py::test_bell_state_expectation PASSED
...
======================== 15 passed in 8.45s ========================

Successfully built qlret:latest
```

**Success Criteria:**
- All build stages complete
- tester stage runs all tests
- All tests pass in container
- Final image builds
- Exit code: 0

---

### 6.2 Docker Runtime Test - CLI

**Status:** ❌ NOT RUN  
**Purpose:** Verify C++ executable works in container

**Test Commands:**
```bash
# Test basic CLI
docker run --rm qlret:latest ./quantum_sim -n 6 -d 8 --mode sequential

# Test with output
docker run --rm -v $(pwd)/output:/app/output qlret:latest \
    ./quantum_sim -n 10 -d 15 -o /app/output/results.csv

# Verify output file
cat output/results.csv
```

**Expected Output:**
```
--------------------------------------------------------------------------------------------------
number of qubits: 6
INFO: n=6 low-workload, batch_size=32
Generated sequence with total noise perc: 0.000523
batch size: 32
current time == 14:23:17
===== Running LRET simulation for 6 qubits =====
Simulation Time: 0.023 seconds
Final Rank: 7
Fidelity: 0.9989
...
Speed up with batch size 32: 2.134
trace distance: 2.34e-06
```

**Success Criteria:**
- Executable runs in container
- Output appears correct
- CSV file created (if -o flag)
- Exit code: 0

---

### 6.3 Docker Runtime Test - Python

**Status:** ❌ NOT RUN  
**Purpose:** Verify Python integration works in container

**Test Commands:**
```bash
# Test Python import
docker run --rm qlret:latest python -c "import qlret; print(qlret.__version__)"

# Test simulate_json
docker run --rm qlret:latest python << 'EOF'
from qlret import simulate_json

circuit = {
    "circuit": {
        "num_qubits": 2,
        "operations": [
            {"name": "H", "wires": [0]},
            {"name": "CNOT", "wires": [0, 1]}
        ],
        "observables": [
            {"type": "PAULI", "operator": "Z", "wires": [0]}
        ]
    },
    "config": {"epsilon": 1e-4}
}

result = simulate_json(circuit)
print(f"Status: {result['status']}")
print(f"Expectation: {result['expectation_values'][0]:.4f}")
EOF

# Test PennyLane device
docker run --rm qlret:latest python << 'EOF'
import pennylane as qml
from qlret import QLRETDevice

dev = QLRETDevice(wires=2, shots=None)

@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

result = circuit()
print(f"PennyLane result: {result:.4f}")
EOF
```

**Expected Output:**
```
1.0.0

Status: success
Expectation: 0.0000

PennyLane result: 1.0000
```

**Success Criteria:**
- Python imports work
- Native module loads
- JSON API works
- PennyLane device works
- Exit code: 0

---

### 6.4 Docker Image Size Test

**Status:** ❌ NOT RUN  
**Purpose:** Verify image size is reasonable

**Test Commands:**
```bash
docker images | grep qlret
```

**Expected Output:**
```
qlret    latest    abc123def456    2 minutes ago    1.2GB
qlret    tester    def456ghi789    5 minutes ago    1.5GB
```

**Success Criteria:**
- Runtime image < 1.5GB
- Build stages appropriately sized
- No excessive bloat

---

### 6.5 Docker .dockerignore Test

**Status:** ❌ NOT RUN  
**Purpose:** Verify build context is minimal

**Test Commands:**
```bash
# Check what's being sent to Docker daemon
docker build -t qlret:test . 2>&1 | grep "Sending build context"
```

**Expected Output:**
```
Sending build context to Docker daemon  2.5MB
```

**Success Criteria:**
- Build context < 10MB
- No build artifacts included
- No .git directory
- No __pycache__ directories

---

## Performance Benchmarks

### Benchmark 1: Scaling with Qubit Count

**Status:** ❌ NOT RUN  
**Purpose:** Measure time vs qubit count

**Test Commands:**
```bash
for n in 8 9 10 11 12 13 14; do
    echo "Testing n=$n..."
    time ./build/quantum_sim -n $n -d 15 --mode hybrid
done > scaling_results.txt
```

**Expected Results:**
```
n=8:  0.05s
n=9:  0.08s
n=10: 0.15s
n=11: 0.28s
n=12: 0.52s
n=13: 0.98s
n=14: 1.85s
```

**Success Criteria:**
- Time roughly doubles per qubit
- No crashes
- Memory usage acceptable

---

### Benchmark 2: Parallel Speedup

**Status:** ❌ NOT RUN  
**Purpose:** Measure parallel efficiency

**Test Commands:**
```bash
./build/quantum_sim -n 12 -d 20 --mode compare
```

**Expected Output:**
```
Parallel Mode Comparison:
---------------------------------
sequential:  1.234s  (1.00x)
row:         0.423s  (2.92x)
column:      0.398s  (3.10x)
hybrid:      0.276s  (4.47x)
```

**Success Criteria:**
- Hybrid mode shows 3-5x speedup
- All modes produce same results (fidelity)
- Exit code: 0

---

### Benchmark 3: FDM Comparison

**Status:** ❌ NOT RUN  
**Purpose:** Validate accuracy vs full density matrix

**Test Commands:**
```bash
for n in 6 7 8 9 10; do
    ./build/quantum_sim -n $n -d 10 --fdm >> fdm_comparison.txt
done
```

**Expected Output Format:**
```
n=6:  LRET=0.023s, FDM=0.045s, Fidelity=0.9998, Trace Distance=3.2e-06
n=7:  LRET=0.041s, FDM=0.123s, Fidelity=0.9997, Trace Distance=5.1e-06
n=8:  LRET=0.078s, FDM=0.456s, Fidelity=0.9996, Trace Distance=8.3e-06
...
```

**Success Criteria:**
- Fidelity > 0.999 for all n
- Trace distance < 1e-5
- LRET faster than FDM for n > 8

---

## Execution Instructions

### Prerequisites Setup

**Linux/macOS:**
```bash
# Install dependencies
sudo apt-get update  # Ubuntu/Debian
sudo apt-get install -y build-essential cmake git libeigen3-dev libomp-dev python3-dev

# Or on macOS:
brew install cmake eigen libomp python@3.11

# Install Python packages
pip3 install numpy scipy matplotlib pennylane pytest pytest-cov

# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
```

### Build and Test Sequence

**Step 1: Clone and Setup**
```bash
git clone <repository>
cd lret-
mkdir build
```

**Step 2: C++ Tests (Phase 1-4)**
```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_PYTHON=ON
cmake --build . -- -j$(nproc)

# Run all C++ tests in order
./test_simple
./test_minimal
./test_fidelity
./quantum_sim -n 8 -d 10
./test_noise_import
./test_advanced_noise
./test_leakage_measurement
```

**Step 3: Python Tests (Phase 5)**
```bash
cd ../python
pip install -e .[dev]
pytest tests/ -v --tb=short
```

**Step 4: Calibration Script Tests (Phase 4.2)**
```bash
cd ../scripts
python3 test_calibration.py
python3 generate_calibration_data.py --num-qubits 5 --depths 5,10,15 --trials 5 --output test_cal.csv
python3 fit_depolarizing.py test_cal.csv --output test_depol.json
```

**Step 5: Docker Tests (Phase 6)**
```bash
cd ..
docker build --target tester -t qlret:test .
docker build -t qlret:latest .
docker run --rm qlret:latest ./quantum_sim -n 6 -d 8
docker run --rm qlret:latest python -c "import qlret; print('OK')"
```

**Step 6: Benchmarks**
```bash
cd build
./quantum_sim -n 12 -d 20 --mode compare > benchmark_parallel.txt
for n in 8 9 10 11 12; do ./quantum_sim -n $n -d 15; done > benchmark_scaling.txt
```

### Expected Total Execution Time

| Phase | Time | Cumulative |
|-------|------|------------|
| Phase 1 (Core tests) | 30 min | 30 min |
| Phase 4.1 (Noise import) | 15 min | 45 min |
| Phase 4.2 (Calibration) | 45 min | 1h 30m |
| Phase 4.3 (Advanced noise) | 20 min | 1h 50m |
| Phase 4.4-4.5 (Leakage) | 25 min | 2h 15m |
| Phase 5 (Python) | 30 min | 2h 45m |
| Phase 6 (Docker) | 60 min | 3h 45m |
| Benchmarks | 90 min | 5h 15m |
| **Total** | **~5-6 hours** | |

---

## Test Results Documentation

### Create Results Log

```bash
# Create test results directory
mkdir -p test_results

# Run all tests with output logging
bash << 'EOF' | tee test_results/full_test_log.txt
echo "=== LRET Testing Suite ==="
echo "Date: $(date)"
echo "System: $(uname -a)"
echo ""

echo "=== Phase 1: Core Tests ==="
./build/test_simple
./build/test_minimal
./build/test_fidelity
./build/quantum_sim -n 8 -d 10

echo ""
echo "=== Phase 4: Noise Tests ==="
./build/test_noise_import
./build/test_advanced_noise
./build/test_leakage_measurement

echo ""
echo "=== Phase 5: Python Tests ==="
cd python && pytest tests/ -v

echo ""
echo "=== Phase 6: Docker Tests ==="
cd .. && docker build -t qlret:latest .
docker run --rm qlret:latest ./quantum_sim -n 6 -d 8

echo ""
echo "=== All Tests Complete ==="
EOF
```

### Generate Test Report

After completion, generate summary:

```bash
python3 << 'EOF' > test_results/summary.md
import re
from pathlib import Path

log = Path("test_results/full_test_log.txt").read_text()

# Count passed/failed tests
passed = len(re.findall(r"✓|PASSED|passed", log))
failed = len(re.findall(r"✗|FAILED|failed", log, re.IGNORECASE))

print(f"# LRET Test Results Summary")
print(f"\n## Overall Statistics")
print(f"- Total tests passed: {passed}")
print(f"- Total tests failed: {failed}")
print(f"- Success rate: {100*passed/(passed+failed):.1f}%")

# Extract phase summaries
phases = re.findall(r"=== (Phase \d+.*?) ===", log)
print(f"\n## Phase Breakdown")
for phase in phases:
    print(f"- {phase}: ✓")
    
print(f"\n## Detailed Log")
print(f"See `full_test_log.txt` for complete output")
EOF
```

---

## Known Issues and Workarounds

### Issue 1: Eigen3 Not Found
**Symptom:** CMake error "Eigen3 not found"  
**Workaround:**
```bash
# Ubuntu/Debian
sudo apt-get install libeigen3-dev

# macOS
brew install eigen

# Manual install
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar xzf eigen-3.4.0.tar.gz
cd eigen-3.4.0
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
sudo make install
```

### Issue 2: pybind11 Build Fails
**Symptom:** "pybind11 not found" or Python module won't build  
**Workaround:**
```bash
# Ensure Python dev headers installed
sudo apt-get install python3-dev

# Or use system pybind11
sudo apt-get install pybind11-dev
cmake .. -DUSE_PYTHON=ON -DCMAKE_PREFIX_PATH=/usr
```

### Issue 3: Docker Build Out of Memory
**Symptom:** Docker build killed during compilation  
**Workaround:**
```bash
# Increase Docker memory limit
# Docker Desktop: Settings → Resources → Memory → 8GB

# Or build with fewer parallel jobs
docker build --build-arg CMAKE_BUILD_PARALLEL_LEVEL=2 -t qlret:latest .
```

### Issue 4: PennyLane Device Not Found
**Symptom:** "No module named 'pennylane'"  
**Workaround:**
```bash
pip install pennylane>=0.30
# Or install with extras
pip install qlret[pennylane]
```

---

## Success Checklist

Use this checklist to track progress:

### Core Tests (Phase 1)
- [ ] test_simple runs successfully
- [ ] test_minimal produces correct fidelity
- [ ] test_fidelity validates LRET vs FDM
- [ ] quantum_sim benchmark completes
- [ ] demo_batch shows correct batch sizes

### Noise Tests (Phase 4)
- [ ] test_noise_import parses JSON correctly
- [ ] test_advanced_noise validates time/correlation/memory effects
- [ ] test_leakage_measurement validates all 8 subtests
- [ ] Calibration scripts run and produce valid output
- [ ] IBM noise download works (or fake backend)

### Python Tests (Phase 5)
- [ ] qlret package installs via pip
- [ ] All 15 pytest tests pass
- [ ] JSON API works (simulate_json)
- [ ] PennyLane device executes circuits
- [ ] Gradients compute correctly

### Docker Tests (Phase 6)
- [ ] Multi-stage build completes
- [ ] tester stage runs all tests
- [ ] Runtime CLI works
- [ ] Runtime Python works
- [ ] Image size < 1.5GB

### Benchmarks
- [ ] Scaling benchmark shows exponential growth
- [ ] Parallel speedup shows 3-5x improvement
- [ ] FDM comparison shows fidelity > 0.999

---

## Contact and Reporting

If tests fail or unexpected behavior occurs:

1. **Collect logs:** Save full output from all test runs
2. **System info:** Include OS, CMake version, compiler version
3. **Error messages:** Full stack traces for failures
4. **Environment:** Python version, library versions (`pip list`)

Create detailed issue report with:
- Which test failed
- Full error message
- Steps to reproduce
- System configuration

---

## Conclusion

This document provides a comprehensive testing roadmap for LRET quantum simulator. Execute tests in order, document results, and report any failures with detailed logs.

**Estimated Completion Time:** 5-6 hours  
**Required Environment:** Linux/macOS with proper build tools  
**Expected Success Rate:** > 95% (assuming environment is correctly configured)

Good luck with testing! 🚀
