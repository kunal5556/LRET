# LRET vs Cirq Comparison - Execution Roadmap

## 🎯 Overview

This document provides step-by-step instructions for running the LRET vs Cirq FDM comparison based on the [CIRQ_COMPARISON_GUIDE.md](../CIRQ_COMPARISON_GUIDE.md).

**Current Status:** ✅ All infrastructure built and tested  
**Required to Run:** LRET quantum_sim executable  
**Location:** `d:\LRET\cirq_comparison\`

---

## 🚦 Quick Start (5 Steps)

### Step 1: Build LRET (One-Time Setup)

```powershell
# Option A: Using MSBuild (Windows)
cd d:\LRET\build
msbuild QuantumLRET-Sim.sln /p:Configuration=Release /p:Platform=x64 /t:quantum_sim

# Option B: Using CMake
cd d:\LRET\build
cmake --build . --config Release --target quantum_sim

# Verify build
Test-Path d:\LRET\build\Release\quantum_sim.exe  # Should return True
```

### Step 2: Quick Validation Test (2 minutes)

```powershell
cd d:\LRET\cirq_comparison
python test_infrastructure.py
```

**Expected output:** All 6 tests passing ✅

### Step 3: Generate Circuits (1 minute)

```powershell
python circuit_generator.py --max-qubits 10 --noise-levels 0.0 0.001 0.01
```

**Expected output:** 99 circuits generated in `circuits/`

### Step 4: Run Comparison Benchmarks

#### Quick Test (2-6 qubits, ~10 minutes)
```powershell
python run_comparison.py --circuits circuits --output results --trials 3 --timeout 180
```

#### Full Test (2-10 qubits, ~2-6 hours)
```powershell
python run_full_comparison.py
```

### Step 5: Analyze & Visualize

```powershell
# Get the latest results file
$latest = Get-ChildItem results\benchmark_results_*.csv | Sort-Object LastWriteTime -Descending | Select-Object -First 1

# Analyze
python analyze_results.py --input $latest.FullName

# Create plots
python create_plots.py --input $latest.FullName --output plots
```

---

## 📊 What to Expect

### Benchmark Metrics

Each circuit will be tested on both simulators with these metrics:

| Metric | Description | Expected Range |
|--------|-------------|----------------|
| **Execution Time** | Wall-clock time (ms) | 1-10,000 ms |
| **Memory Usage** | Peak RAM (MB) | 0.1-1000 MB |
| **State Fidelity** | \|⟨ψ_LRET\|ψ_Cirq⟩\|² | >0.9999 |
| **Trace Distance** | \|\|ρ_LRET - ρ_Cirq\|\|₁ | <0.001 |
| **Speedup Factor** | time_Cirq / time_LRET | 0.5-5.0× |
| **Memory Efficiency** | mem_Cirq / mem_LRET | 1.0-10× |

### Performance Expectations

Based on the comparison guide and LRET's architecture:

#### Low-Rank Circuits (High LRET Advantage)
- **Bell States:** Speedup 2-5×, Memory 5-10×
- **GHZ States:** Speedup 2-4×, Memory 3-8×
- **Reason:** Highly entangled but low Schmidt rank

#### Moderate-Rank Circuits
- **QFT:** Speedup 1.5-3×, Memory 2-5×
- **Reason:** Moderate entanglement growth

#### High-Rank Circuits (Less LRET Advantage)
- **Random Deep:** Speedup 0.5-1.5×, Memory 1-3×
- **Reason:** Near-maximal entanglement, high rank

#### Breaking Points
- **LRET:** Should work up to ~18-22 qubits with noise
- **Cirq FDM:** Expected OOM at ~12-14 qubits
- **Key finding:** LRET extends simulation range by ~6-10 qubits!

---

## 🔍 Deep Dive: How Each Module Works

### 1. Circuit Generator (`circuit_generator.py`)

**Purpose:** Generate test circuits in LRET JSON format

**Implementation:**
```python
# Bell states (2-10 qubits, pairs only)
operations = [
    {"gate": "H", "targets": [0]},
    {"gate": "CNOT", "control": 0, "targets": [1]},
    # Repeat for pairs...
]

# GHZ states (3-10 qubits)
operations = [
    {"gate": "H", "targets": [0]},
    {"gate": "CNOT", "control": 0, "targets": [1]},
    {"gate": "CNOT", "control": 1, "targets": [2]},
    # CNOT chain...
]

# QFT (3-12 qubits)
for i in range(n_qubits):
    operations.append({"gate": "H", "targets": [i]})
    for j in range(i+1, n_qubits):
        angle = 2*pi / 2^(j-i+1)
        # Controlled phase rotation
```

**Output:** JSON files + manifest.json with metadata

### 2. Cirq Wrapper (`cirq_fdm_wrapper.py`)

**Purpose:** Interface Cirq's DensityMatrixSimulator to match LRET's API

**Key Methods:**
```python
class CirqFDMSimulator:
    def simulate(circuit):
        # Convert LRET JSON → Cirq Circuit
        # Run DensityMatrixSimulator
        # Track time & memory
        # Return: final_state, metadata
    
    def compute_fidelity(state1, state2):
        # F(ρ, σ) = Tr(√(√ρ σ √ρ))²
        
    def compute_trace_distance(state1, state2):
        # D(ρ, σ) = 0.5 * ||ρ - σ||₁
```

**Gate Mapping:**
```python
LRET → Cirq:
- H → cirq.H
- CNOT → cirq.CNOT
- RX/RY/RZ → cirq.rx/ry/rz(angle)
- X/Y/Z → cirq.X/Y/Z
```

### 3. Benchmark Runner (`run_comparison.py`)

**Purpose:** Execute both simulators and collect metrics

**Workflow:**
```python
for circuit in circuits:
    # 1. LRET Benchmark
    for trial in range(trials):
        result = lret.simulate_json(circuit)
        times.append(execution_time)
        states.append(final_state)
    
    # 2. Cirq Benchmark
    cirq_circuit = convert_to_cirq(circuit)
    for trial in range(trials):
        state, metadata = cirq_sim.simulate(cirq_circuit)
        times.append(metadata['time'])
        states.append(state)
    
    # 3. Compare
    fidelity = compute_fidelity(lret_state, cirq_state)
    speedup = cirq_time / lret_time
    
    # 4. Save to CSV
    results.append({
        'circuit': name,
        'lret_time': lret_time,
        'cirq_time': cirq_time,
        'speedup': speedup,
        'fidelity': fidelity
    })
```

### 4. Statistical Analysis (`analyze_results.py`)

**Purpose:** Perform rigorous statistical tests

**Tests Performed:**
```python
# 1. Summary Statistics
mean_speedup = speedup_factors.mean()
median_speedup = speedup_factors.median()
std_speedup = speedup_factors.std()

# 2. T-Test (Paired)
t_stat, p_value = stats.ttest_rel(lret_times, cirq_times)
# H0: No difference in execution times
# H1: Significant difference
# Reject H0 if p < 0.05

# 3. Wilcoxon Signed-Rank Test (Non-parametric)
w_stat, p_value = stats.wilcoxon(lret_times, cirq_times)
# Handles non-normal distributions

# 4. Cohen's d (Effect Size)
cohens_d = (mean_cirq - mean_lret) / pooled_std
# |d| < 0.5: small effect
# |d| > 0.5: medium effect
# |d| > 0.8: large effect

# 5. Correlation Analysis
corr, p = stats.pearsonr(num_qubits, speedup_factors)
# Does speedup increase with problem size?
```

**Output:** Statistical report + LaTeX tables

### 5. Visualization (`create_plots.py`)

**Purpose:** Create publication-quality figures

**Figure Specifications:**

#### Figure 1: Time Comparison
```python
# Line plot: qubits vs time (log scale)
# Lines: LRET (solid) vs Cirq (dashed)
# Error bars: ±1 std deviation
# Separate lines per circuit type
```

#### Figure 2: Memory Comparison
```python
# Bar plot: Circuit types vs memory
# Side-by-side bars: LRET vs Cirq
# Shows memory efficiency clearly
```

#### Figure 3: Speedup Heatmap
```python
# 2D heatmap: qubits × depth
# Color: Speedup factor
# Green (>1): LRET faster
# Red (<1): Cirq faster
```

#### Figure 4: Fidelity Histogram
```python
# Histogram: Distribution of fidelities
# Should be tightly peaked near 1.0
# Red line: Mean fidelity
# Validates correctness
```

#### Figure 5: Scalability
```python
# Log-log plot: qubits vs time
# Fit power laws: O(n^α)
# Shows scaling exponents
# LRET: α_LRET ≈ 2-3 (low-rank)
# Cirq: α_Cirq ≈ 2-3 (FDM)
```

---

## 🎨 Customizing the Comparison

### Test Different Qubit Ranges

```powershell
# Small test (2-6 qubits, fast)
python circuit_generator.py --max-qubits 6 --noise-levels 0.0

# Medium test (2-12 qubits, moderate)
python circuit_generator.py --max-qubits 12 --noise-levels 0.0 0.001

# Large test (2-16 qubits, slow)
python circuit_generator.py --max-qubits 16 --noise-levels 0.0
```

### Test Different Noise Levels

```powershell
# No noise
python circuit_generator.py --noise-levels 0.0

# Light noise
python circuit_generator.py --noise-levels 0.001

# Moderate noise
python circuit_generator.py --noise-levels 0.001 0.01 0.05

# Heavy noise
python circuit_generator.py --noise-levels 0.05 0.1
```

### Adjust Benchmark Parameters

```powershell
# Fewer trials (faster, less statistical power)
python run_comparison.py --trials 3 --timeout 180

# More trials (slower, better statistics)
python run_comparison.py --trials 10 --timeout 600

# Longer timeout (for large circuits)
python run_comparison.py --timeout 1800  # 30 minutes
```

### Test LRET Parallel Modes

To test different LRET execution modes, modify `run_comparison.py`:

```python
# Add parallel mode testing
lret_modes = {
    'serial': {},  # Default
    'openmp': {'OMP_NUM_THREADS': '8'},
    'gpu': {'USE_GPU': '1'},
}

for mode_name, env_vars in lret_modes.items():
    # Set environment variables
    os.environ.update(env_vars)
    
    # Run LRET
    result = simulate_json(circuit)
    
    # Save with mode label
    results.append({
        'mode': mode_name,
        'time': result['execution_time_ms'],
        ...
    })
```

---

## 📈 Interpreting Results

### Statistical Analysis Output

```
============================================================
LRET vs Cirq FDM Comparison - Statistical Analysis
============================================================

SUMMARY STATISTICS
------------------------------------------------------------
Total circuits analyzed: 99
Circuit types: bell, ghz, qft, random
Qubit range: 2-10
Depth range: 2-50
Noise levels: [0.0, 0.001, 0.01]

PERFORMANCE COMPARISON
------------------------------------------------------------
LRET mean time: 123.45 ms
Cirq mean time: 234.56 ms
Mean speedup (LRET vs Cirq): 1.90x (±0.45)
Median speedup: 1.85x

LRET mean memory: 45.67 MB
Cirq mean memory: 123.45 MB
Memory efficiency: 2.70x

CORRECTNESS METRICS
------------------------------------------------------------
Mean fidelity: 0.999876
Min fidelity: 0.999234
Fidelity std: 0.000123

PER CIRCUIT TYPE
------------------------------------------------------------
bell      :  24 circuits, speedup= 2.34x, fidelity=0.999912
ghz       :  24 circuits, speedup= 2.12x, fidelity=0.999845
qft       :  30 circuits, speedup= 1.76x, fidelity=0.999823
random    :  21 circuits, speedup= 1.23x, fidelity=0.999789

STATISTICAL TESTS
------------------------------------------------------------
T-test (time): t=12.456, p=0.000001 (significant)
Wilcoxon test: W=4567.0, p=0.000002 (significant)
Cohen's d (effect size): 1.234 (LARGE effect)
Qubit-speedup correlation: r=0.678, p=0.000123
============================================================
```

**Key Insights:**
- **Speedup >1:** LRET is faster on average
- **p < 0.05:** Difference is statistically significant
- **Cohen's d >0.8:** Large practical effect
- **Fidelity >0.999:** Results are numerically equivalent
- **Positive correlation:** Speedup increases with qubits

### Plot Interpretation

#### Figure 1: Time Comparison
- **Diverging lines:** LRET scales better than Cirq
- **Crossing lines:** Different regimes (small vs large)
- **Steep slope:** Exponential scaling visible

#### Figure 2: Memory Comparison
- **Shorter LRET bars:** Memory advantage
- **Equal bars:** No advantage (high-rank circuits)

#### Figure 3: Speedup Heatmap
- **Green regions:** LRET advantage (low-rank)
- **Red regions:** Cirq advantage (small circuits)
- **Pattern:** Advantage increases with qubits

#### Figure 4: Fidelity Histogram
- **Tight peak near 1.0:** Correct implementation
- **Wide distribution:** Potential numerical issues
- **Multiple peaks:** Different circuit types

#### Figure 5: Scalability
- **Parallel lines:** Similar scaling
- **Diverging lines:** Different scaling exponents
- **Steeper LRET line:** Worse scaling (unlikely)
- **Flatter LRET line:** Better scaling (expected)

---

## ✅ Success Criteria

### ✓ Infrastructure Validated
- [x] Cirq installed (v1.6.1)
- [x] 99 test circuits generated
- [x] Wrapper tested (Bell state: 11ms, fidelity=1.0)
- [x] All 6 validation tests passing

### ✓ Benchmark Execution
- [ ] LRET built successfully
- [ ] Benchmarks completed without errors
- [ ] Results CSV generated
- [ ] No timeouts or crashes

### ✓ Statistical Significance
- [ ] Mean fidelity >0.999
- [ ] p-value <0.05 (time difference)
- [ ] Cohen's d >0.5 (practical significance)

### ✓ Publication Ready
- [ ] All 5 figures generated (PDF + PNG)
- [ ] LaTeX tables created
- [ ] Summary report generated
- [ ] Results reproducible

---

## 🐛 Troubleshooting Guide

### Problem: LRET build fails

**Solution:**
```powershell
# Check CMake configuration
cd d:\LRET\build
cmake .. -DCMAKE_BUILD_TYPE=Release

# Verify Eigen3 is found
cmake .. | Select-String "Eigen3"

# Try clean build
Remove-Item -Recurse -Force CMakeCache.txt, CMakeFiles
cmake ..
cmake --build . --config Release
```

### Problem: "quantum_sim.exe not found"

**Solution:**
```powershell
# Check build output
Get-ChildItem d:\LRET\build -Recurse -Filter quantum_sim.exe

# If found elsewhere, copy to expected location
Copy-Item <found_path>\quantum_sim.exe d:\LRET\build\Release\

# Or set explicit path
$env:LRET_EXECUTABLE = "d:\path\to\quantum_sim.exe"
```

### Problem: Cirq simulation very slow

**Solution:**
```powershell
# Reduce circuit complexity
python circuit_generator.py --max-qubits 8  # Instead of 10

# Reduce trials
python run_comparison.py --trials 3  # Instead of 5

# Skip noisy circuits initially
python circuit_generator.py --noise-levels 0.0  # No noise
```

### Problem: Low fidelity (<0.99)

**Investigation:**
```python
# Check individual circuits
python -c "
from cirq_comparison.cirq_fdm_wrapper import CirqFDMSimulator
# Test Bell state
# Compare against known result
"

# Verify gate definitions match
# Check phase conventions
# Verify noise model implementation
```

### Problem: Out of memory

**Solution:**
```powershell
# Reduce max qubits
python circuit_generator.py --max-qubits 8

# Test fewer circuits
python run_comparison.py --circuits circuits/bell*.json

# Increase system memory (add swap)
```

### Problem: Python import errors

**Solution:**
```powershell
# Ensure correct Python environment
python --version  # Should be 3.8+

# Reinstall dependencies
python -m pip install --upgrade cirq numpy pandas matplotlib scipy seaborn

# Check PYTHONPATH
$env:PYTHONPATH = "d:\LRET\python;$env:PYTHONPATH"
```

---

## 📚 Additional Resources

### Documentation Files
- `CIRQ_COMPARISON_GUIDE.md` - Complete guide (398 lines)
- `SETUP_COMPLETE_REPORT.md` - Infrastructure status
- `test_infrastructure.py` - Validation code examples
- Individual module docstrings

### Comparison Guide Key Sections
- **Metrics Collected** (lines 80-110)
- **Expected Timeline** (lines 150-165)
- **Troubleshooting** (lines 200-250)
- **Verification Steps** (lines 260-290)
- **Customization** (lines 300-330)

### Example Workflows
See `CIRQ_COMPARISON_GUIDE.md`:
- Option 1: Fully automated (lines 5-20)
- Option 2: Step-by-step (lines 22-35)
- Option 3: Manual execution (lines 37-50)

---

## 🎓 Understanding the Comparison

### Why This Comparison Matters

1. **Validation:** Ensures LRET produces correct results
2. **Performance:** Quantifies speedup and memory advantages
3. **Scalability:** Shows extended qubit range capability
4. **Publication:** Provides rigorous comparison data

### What Makes a Good Comparison

1. **Fair:** Same hardware, same circuits, same conditions
2. **Rigorous:** Multiple trials, statistical tests
3. **Comprehensive:** Multiple circuit types, qubit ranges
4. **Reproducible:** All code and data available

### How LRET Can Win

1. **Low-rank states:** Bell, GHZ → Direct advantage
2. **Memory efficiency:** Enables larger simulations
3. **Noise handling:** Maintains low rank under decoherence
4. **Scalability:** Extends practical simulation range

### How Cirq Can Win

1. **Small circuits:** Less overhead for 2-4 qubits
2. **High-rank states:** Full FDM more efficient
3. **Maturity:** Well-optimized, battle-tested
4. **Ecosystem:** Integrated tools, documentation

### Expected Outcome

**Realistic Expectation:**
- LRET faster on low-rank circuits (Bell, GHZ): 2-5× speedup
- LRET enables larger simulations: 18-22 qubits vs 12-14
- Fidelity agreement: >0.9999 on all circuits
- Mixed results on random circuits (rank-dependent)

**Best Case:**
- LRET consistently faster across all types
- 5-10× speedup on entangled states
- Extends range by 10 qubits
- Perfect fidelity agreement

**Worst Case:**
- LRET comparable but not faster
- Speedup only on specific circuit types
- Similar qubit limits
- Still validates correctness!

---

## 🚀 Ready to Run!

**Current Status:** ✅ All infrastructure complete  
**Blocking:** LRET build required  
**Next Step:** Build LRET → Run comparison

**Estimated Timeline:**
- Build LRET: 5-10 minutes
- Quick test: 10-15 minutes
- Full comparison: 2-6 hours
- Analysis & plots: 10-15 minutes

**Total:** ~3-7 hours for complete publication-ready results

---

**Questions?** Check the troubleshooting guide or refer to module docstrings.

**Ready?** Let's build LRET and run the comparison! 🎯

