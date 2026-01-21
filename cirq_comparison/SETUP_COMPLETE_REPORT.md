# LRET vs Cirq FDM Comparison - Setup Complete!

## 🎉 Status: Infrastructure Ready

**Date:** January 21, 2026  
**Location:** `d:\LRET\cirq_comparison\`

---

## ✅ What's Been Completed

### 1. Directory Structure
```
cirq_comparison/
├── circuits/                 # 99 test circuits generated
│   ├── bell_*.json          # Bell state circuits (2-10 qubits)
│   ├── ghz_*.json           # GHZ state circuits (3-10 qubits)
│   ├── qft_*.json           # QFT circuits (3-12 qubits)
│   ├── random_*.json        # Random circuits (4-10 qubits, various depths)
│   └── manifest.json        # Circuit metadata catalog
├── results/                 # Benchmark outputs (generated after run)
├── plots/                   # Publication figures (generated after run)
├── test_circuits/           # Validation test circuits
└── test_results/            # Validation test outputs
```

### 2. Core Modules Created

#### ✅ `cirq_fdm_wrapper.py` (345 lines)
- **Purpose:** Cirq DensityMatrixSimulator wrapper matching LRET's interface
- **Features:**
  - Full density matrix simulation
  - Noise model support (depolarizing, amplitude damping, phase damping)
  - Performance tracking (time, memory)
  - Fidelity and trace distance calculations
  - JSON circuit conversion
- **Tested:** ✓ Bell state simulation working (11ms, 0.1MB, fidelity=1.0)

#### ✅ `circuit_generator.py` (442 lines)
- **Purpose:** Generate test circuits in LRET JSON format
- **Features:**
  - Bell states (2-10 qubits, pairs)
  - GHZ states (3-10 qubits)
  - QFT (3-12 qubits)
  - Random circuits (depth 5, 10, 20)
  - Noise variations (0.0, 0.001, 0.01)
- **Output:** 99 circuits generated ✓

#### ✅ `run_comparison.py` (359 lines)
- **Purpose:** Run both simulators and collect benchmark data
- **Features:**
  - Parallel execution with trials (default: 5)
  - Timeout handling (default: 300s)
  - Metrics: time, memory, fidelity, trace distance
  - CSV output with full results
  - Progress tracking with tqdm
- **Status:** Ready to run (needs LRET build)

#### ✅ `analyze_results.py` (266 lines)
- **Purpose:** Statistical analysis of benchmark results
- **Features:**
  - Summary statistics (mean, median, std)
  - T-tests and Wilcoxon tests
  - Cohen's d effect size
  - Per-circuit-type analysis
  - LaTeX table generation
- **Status:** Ready to use

#### ✅ `create_plots.py` (274 lines)
- **Purpose:** Publication-quality visualization
- **Features:**
  - Figure 1: Time comparison (line plot)
  - Figure 2: Memory comparison (bar plot)
  - Figure 3: Speedup heatmap
  - Figure 4: Fidelity histogram
  - Figure 5: Scalability (log-log)
  - 300 DPI PNG + vector PDF output
- **Status:** Ready to use

#### ✅ `run_full_comparison.py` (143 lines)
- **Purpose:** Master orchestration script
- **Features:**
  - Runs entire pipeline automatically
  - Generates summary report
  - Timestamped outputs
- **Status:** Ready to run (needs LRET build)

#### ✅ `test_infrastructure.py` (164 lines)
- **Purpose:** Validate comparison infrastructure
- **Tested:** All 6 tests passing ✓
  - Circuit generation ✓
  - JSON → Cirq conversion ✓
  - Cirq FDM simulation ✓
  - Metrics collection ✓
  - Results export ✓
  - Fidelity calculations ✓

---

## 📊 Current Test Results (Cirq Only)

From infrastructure validation test:

| Circuit | Qubits | Time (ms) | Memory (MB) | Fidelity |
|---------|--------|-----------|-------------|----------|
| Bell    | 2      | 10.87     | 0.017       | 1.000000 |
| GHZ     | 3      | 7.40      | 0.021       | 1.000000 |
| QFT     | 3      | 13.93     | 0.045       | 1.000000 |

**Cirq Infrastructure:** ✅ Working perfectly!

---

## ⚠️ Next Steps Required

### Critical: Build LRET

The comparison infrastructure is complete, but we need LRET compiled to run the actual comparison. Two options:

#### Option A: Build LRET (Recommended)
```powershell
# Open Visual Studio Developer Command Prompt or use MSBuild
cd d:\LRET\build
msbuild QuantumLRET-Sim.sln /p:Configuration=Release /p:Platform=x64

# Or use CMake + Visual Studio
cmake --build . --config Release --target quantum_sim
```

This will create: `d:\LRET\build\Release\quantum_sim.exe`

#### Option B: Use Python Bindings (If Available)
If LRET has pybind11 bindings, we can use those directly:
```python
from qlret import _qlret_native  # Check if this exists
```

---

## 🚀 Running the Full Comparison

Once LRET is built:

### Quick Test (2-6 qubits, ~5-10 minutes)
```powershell
cd d:\LRET\cirq_comparison

# Generate small circuits
python circuit_generator.py --max-qubits 6 --noise-levels 0.0

# Run comparison
python run_comparison.py --circuits circuits --output results --trials 3 --timeout 180

# Analyze (replace with actual timestamp)
python analyze_results.py --input results/benchmark_results_TIMESTAMP.csv
python create_plots.py --input results/benchmark_results_TIMESTAMP.csv --output plots
```

### Full Comparison (2-10 qubits, ~2-6 hours)
```powershell
cd d:\LRET\cirq_comparison
python run_full_comparison.py
```

This will automatically:
1. Generate 99 circuits (already done ✓)
2. Run benchmarks on LRET and Cirq
3. Perform statistical analysis
4. Create publication plots
5. Generate summary report

---

## 📈 Expected Outputs

### After Running Comparison:

#### Results Files
- `results/benchmark_results_TIMESTAMP.csv` - Raw data
  - Columns: circuit_name, num_qubits, depth, noise_level
  - LRET metrics: time, memory, state
  - Cirq metrics: time, memory, state
  - Comparison: fidelity, trace_distance, speedup, memory_efficiency

#### Analysis Files
- `results/statistical_analysis.txt` - Statistical tests
  - Summary statistics
  - T-tests, Wilcoxon tests
  - Cohen's d effect sizes
  - Per-circuit-type breakdown
- `results/tables_for_paper.tex` - LaTeX tables

#### Plots (PDF + PNG)
- `figure1_time_comparison.pdf` - Execution time vs qubits
- `figure2_memory_comparison.pdf` - Memory usage bars
- `figure3_speedup_heatmap.pdf` - Speedup matrix
- `figure4_fidelity_histogram.pdf` - Fidelity distribution
- `figure5_scalability.pdf` - Scaling analysis

#### Summary Report
- `COMPARISON_SUMMARY_TIMESTAMP.md` - Complete overview

---

## 🔍 What Each Script Does

### Detailed Workflow

#### 1. Circuit Generation (`circuit_generator.py`)
```python
# Generates:
# - Bell states: H + CNOT chain
# - GHZ states: H + CNOT fan-out
# - QFT: Hadamards + controlled rotations + swaps
# - Random: Mix of single/two-qubit gates
# 
# Each circuit type × noise levels → JSON files
```

#### 2. Benchmark Execution (`run_comparison.py`)
```python
# For each circuit:
#   1. Load JSON
#   2. Run LRET:
#      - Convert JSON → LRET format
#      - Execute simulation
#      - Record: time, memory, final_state
#   3. Run Cirq:
#      - Convert JSON → Cirq Circuit
#      - Execute DensityMatrixSimulator
#      - Record: time, memory, final_state
#   4. Compare:
#      - Fidelity: |⟨ψ_LRET|ψ_Cirq⟩|²
#      - Trace distance: ||ρ_L - ρ_C||₁
#      - Speedup: time_Cirq / time_LRET
#   5. Save to CSV
```

#### 3. Statistical Analysis (`analyze_results.py`)
```python
# Load CSV → Filter successful runs → Compute:
#   - Summary stats (mean, std, min, max)
#   - T-test: Are time differences significant?
#   - Wilcoxon: Non-parametric alternative
#   - Cohen's d: Effect size
#   - Correlation: qubits vs speedup
#   - Per-type breakdown
# Output: TXT report + LaTeX tables
```

#### 4. Visualization (`create_plots.py`)
```python
# Load CSV → Create 5 figures:
#   1. Time vs qubits (log scale, lines with error bars)
#   2. Memory bars (LRET vs Cirq, by circuit type)
#   3. Speedup heatmap (qubits × depth, color-coded)
#   4. Fidelity histogram (should peak near 1.0)
#   5. Scalability (log-log, fit power laws)
# Save: 300 DPI PNG + vector PDF
```

---

## 📚 Comparison Guide Deep Dive

Based on `CIRQ_COMPARISON_GUIDE.md`:

### Key Metrics Collected

1. **Performance Metrics**
   - Execution time (ms) - Wall-clock time per simulation
   - Peak memory (MB) - Maximum RAM during simulation
   - Throughput - Circuits/second capability

2. **Correctness Metrics**
   - State fidelity - |⟨ψ_LRET|ψ_Cirq⟩|² (should be >0.9999)
   - Trace distance - ||ρ_LRET - ρ_Cirq||₁ (should be <0.01)
   - Trace preservation - Tr(ρ) ≈ 1.0

3. **Statistical Tests**
   - **T-test:** Are LRET/Cirq times significantly different?
   - **Wilcoxon:** Non-parametric version (handles outliers)
   - **Cohen's d:** Effect size (>0.5 = medium, >0.8 = large)
   - **Correlation:** Does speedup increase with qubits?

### Expected Results (From Guide)

#### Scenario 1: LRET Faster (Low-Rank Advantage)
- **Bell/GHZ states:** Speedup 2-5× (highly entangled but low-rank)
- **QFT:** Speedup 1.5-3× (moderate entanglement)
- **Random circuits:** Speedup 0.5-1.5× (high-rank, less advantage)
- **Memory:** 50-90% less for LRET (low-rank compression)

#### Scenario 2: Cirq Faster (Small Circuits)
- **2-4 qubits:** Cirq may be faster (less overhead)
- **Dense matrices:** Full FDM is efficient for small N
- **LRET overhead:** Rank tracking, truncation, decomposition

#### Scenario 3: Breaking Points
- **LRET:** Should work up to 18-22 qubits (noise-dependent)
- **Cirq FDM:** Likely fails at 12-14 qubits (OOM)
- **Key finding:** LRET extends practical simulation range!

### How Scripts Calculate Metrics

#### Fidelity (in `cirq_fdm_wrapper.py`)
```python
def compute_fidelity(state1, state2):
    # F(ρ, σ) = Tr(√(√ρ σ √ρ))²
    sqrt_rho1 = matrix_sqrt(state1)
    product = sqrt_rho1 @ state2 @ sqrt_rho1
    sqrt_product = matrix_sqrt(product)
    return np.real(np.trace(sqrt_product)) ** 2
```

#### Trace Distance (in `run_comparison.py`)
```python
def compute_trace_distance(state1, state2):
    # D(ρ, σ) = 0.5 * ||ρ - σ||₁
    diff = state1 - state2
    eigenvalues = np.linalg.eigvalsh(diff)
    return 0.5 * np.sum(np.abs(eigenvalues))
```

#### Speedup Factor (in `run_comparison.py`)
```python
speedup = cirq_mean_time / lret_mean_time
# >1.0 means LRET is faster
# <1.0 means Cirq is faster
```

### Parallel Mode Testing (As Requested)

The guide mentions testing different LRET parallel modes. To incorporate this:

1. **Modify `run_comparison.py`** to test multiple LRET configurations:
   ```python
   # Add parallel mode parameter
   lret_modes = ["serial", "openmp", "mpi", "gpu"]
   
   for mode in lret_modes:
       # Run LRET with different parallel backends
       # Compare each mode vs Cirq
   ```

2. **Expected comparisons:**
   - LRET Serial vs Cirq
   - LRET OpenMP vs Cirq
   - LRET GPU vs Cirq (if available)
   - LRET MPI vs Cirq (for distributed)

3. **Additional metrics:**
   - Parallel efficiency
   - Strong scaling (fixed problem size, more cores)
   - Weak scaling (problem size scales with cores)

---

## 🎯 Recommendations

### For Small Test (2-6 qubits)
1. Build LRET: `msbuild /p:Configuration=Release`
2. Run: `python circuit_generator.py --max-qubits 6`
3. Run: `python run_comparison.py --trials 3`
4. Analyze & plot results
5. **Expected time:** 5-15 minutes
6. **Expected outcome:** Validate infrastructure, get preliminary speedup estimates

### For Full Comparison (2-10 qubits)
1. Build LRET
2. Run: `python run_full_comparison.py`
3. Let it run (2-6 hours)
4. Review all outputs
5. **Expected time:** 2-6 hours (depending on LRET speed)
6. **Expected outcome:** Publication-ready results

### For Extended Testing (Parallel Modes)
1. Modify `run_comparison.py` to test different LRET modes
2. Generate separate results for each mode
3. Create comparison plot: Serial vs OpenMP vs GPU vs Cirq
4. Analyze parallel efficiency
5. **Expected time:** 8-24 hours
6. **Expected outcome:** Comprehensive parallel performance study

---

## 📁 Files Created

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `cirq_fdm_wrapper.py` | 345 | Cirq simulator interface | ✅ Tested |
| `circuit_generator.py` | 442 | Test circuit generation | ✅ Generated 99 circuits |
| `run_comparison.py` | 359 | Benchmark runner | ✅ Ready (needs LRET) |
| `analyze_results.py` | 266 | Statistical analysis | ✅ Ready |
| `create_plots.py` | 274 | Publication plots | ✅ Ready |
| `run_full_comparison.py` | 143 | Master orchestration | ✅ Ready (needs LRET) |
| `test_infrastructure.py` | 164 | Validation tests | ✅ All tests passing |
| **TOTAL** | **1,993 lines** | Complete comparison suite | ✅ Infrastructure complete |

---

## ⏭️ Immediate Next Steps

1. **Build LRET** (see Option A above)
2. **Quick test** (6 qubits, 3 trials, 15 min)
3. **Review results** to validate
4. **If good:** Run full comparison
5. **Analyze outputs** for publication

---

## 🆘 Troubleshooting

### Issue: LRET build fails
- Check CMake configuration
- Verify Eigen3 is installed
- Try: `cmake .. -DCMAKE_BUILD_TYPE=Release`

### Issue: Cirq simulation slow
- Reduce max qubits: `--max-qubits 8`
- Reduce trials: `--trials 3`
- Skip noisy circuits initially

### Issue: Low fidelity between LRET and Cirq
- Check gate definitions match
- Verify noise model implementation
- Compare on simple circuits first (Bell, GHZ)

### Issue: Out of memory
- Reduce qubit count
- Test fewer circuits
- Increase system swap space

---

## 📞 Questions?

Refer to:
- `CIRQ_COMPARISON_GUIDE.md` - Complete guide (398 lines)
- `test_infrastructure.py` - See validation code
- `run_comparison.py` - See benchmark implementation

All infrastructure is ready. Just need LRET built to run the full comparison! 🚀

---

**Status:** ✅ Ready to benchmark once LRET is compiled  
**Next:** Build LRET → Test → Full comparison → Publication plots

