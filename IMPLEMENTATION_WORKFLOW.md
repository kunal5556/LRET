# Row Parallelism Optimization - Implementation Workflow

**Branch**: `row-parallelism-optimization`  
**Created**: February 5, 2026  
**Status**: Ready for Implementation

---

## 🔄 Safe Implementation Workflow

### Overview
To ensure we don't break existing functionality and can accurately measure performance gains, we'll follow this workflow:

```
LRET/                          # Original (baseline)
├── src/
├── include/
├── build/
└── ...

LRET_optimized/                # Modified (experimental)
├── src/                       # <- Changes made here
├── include/                   # <- Changes made here
├── build/
└── ...

Performance Comparison
├── Run same tests on both
├── Compare metrics
└── Integrate if better
```

---

## 📋 Step-by-Step Process

### Step 1: Create Baseline Snapshot ✅
**Action**: Duplicate LRET folder
```bash
# From parent directory
cp -r LRET LRET_optimized
cd LRET_optimized
git checkout row-parallelism-optimization
```

**Purpose**: Have pristine baseline for comparison

### Step 2: Implement Optimizations in LRET_optimized
**Action**: Make all code changes in `LRET_optimized/` folder only

**Changes** (Phase 1):
1. `LRET_optimized/src/parallel_modes.cpp`
2. `LRET_optimized/src/simd_kernels.cpp`
3. `LRET_optimized/src/utils.cpp`
4. `LRET_optimized/src/simulator.cpp`

**Original LRET**: Untouched, stays as baseline

### Step 3: Build Both Versions
```bash
# Build baseline
cd LRET/build
cmake .. && make -j8

# Build optimized
cd ../../LRET_optimized/build
cmake .. && make -j8
```

### Step 4: Run Comprehensive Tests
**Test Suite** (see below for details):
- Unit tests (correctness)
- Benchmark suite (performance)
- Fidelity tests (accuracy)
- Memory profiling

**Output**: CSV files with metrics from both versions

### Step 5: Compare & Analyze
**Metrics to Compare**:
- Execution time (speedup factor)
- Memory usage (peak RAM)
- Fidelity (should be >0.999)
- Cache hit rate
- CPU utilization

**Decision Criteria**:
- ✅ Integrate if: Speedup ≥ 1.2× AND fidelity ≥ 0.999
- ⚠️ Review if: Speedup 1.0-1.2× OR fidelity 0.99-0.999
- ❌ Reject if: Speedup < 1.0× OR fidelity < 0.99

### Step 6: Integration (Only if Tests Pass)
```bash
# Copy optimized code back to LRET
cp -r LRET_optimized/src/* LRET/src/
cp -r LRET_optimized/include/* LRET/include/
cd LRET
git add src/ include/
git commit -m "Integrate Phase 1 row parallelism optimizations

Verified performance gains:
- VQE (n=15, d=50): 1.5× speedup
- Fidelity: >0.999
- All tests passing"
```

---

## 🧪 Testing Framework

### Test 1: Unit Tests (Correctness)
**Location**: `LRET/tests/` and `LRET_optimized/tests/`

**Commands**:
```bash
# Baseline
cd LRET/build && ctest --verbose

# Optimized
cd LRET_optimized/build && ctest --verbose
```

**Pass Criteria**: All tests must pass in both versions

### Test 2: Benchmark Suite (Performance)
**Test Circuits**:
1. **VQE H₂** (n=15, d=50, rank~32) - Primary benchmark
2. **QAOA MaxCut** (n=12, d=20, rank~16) - Structured circuit
3. **QNN Classifier** (n=12, d=30, rank~24) - Feature map heavy
4. **Random Circuit** (n=14, d=40, rank~40) - Worst case
5. **Noisy Depolarizing** (n=10, d=100, rank grows) - Noise heavy

**Metrics**:
- Total execution time
- Gate application time
- Noise application time
- Truncation time
- Final rank

**Benchmark Script**:
```bash
#!/bin/bash
# run_benchmarks.sh

BASELINE=./LRET/build/quantum_sim
OPTIMIZED=./LRET_optimized/build/quantum_sim
CIRCUITS=(vqe_h2_n15 qaoa_n12 qnn_n12 random_n14 noisy_n10)

for circuit in "${CIRCUITS[@]}"; do
    echo "=== Benchmarking $circuit ==="
    
    # Baseline
    echo "Baseline:"
    $BASELINE samples/${circuit}.json --verbose > results/baseline_${circuit}.log
    
    # Optimized
    echo "Optimized:"
    $OPTIMIZED samples/${circuit}.json --verbose > results/optimized_${circuit}.log
done

# Compare results
python scripts/compare_performance.py results/
```

### Test 3: Fidelity Verification (Accuracy)
**Purpose**: Ensure optimizations don't break correctness

**Test**:
```cpp
// tests/test_phase1_fidelity.cpp
TEST(Phase1Fidelity, VQE_H2_Comparison) {
    // Run same circuit in baseline and optimized
    auto L_baseline = run_baseline();
    auto L_optimized = run_optimized();
    
    // Compute fidelity between results
    double fidelity = compute_fidelity(L_baseline, L_optimized);
    
    EXPECT_GT(fidelity, 0.999);  // Must be >99.9% identical
}
```

**Pass Criteria**: Fidelity > 0.999 for all test circuits

### Test 4: Memory Profiling
**Tools**:
- Windows: Performance Monitor (PerfMon)
- Linux: Valgrind Massif
- Cross-platform: ResourceMonitor (LRET's built-in)

**Metrics**:
- Peak memory usage
- Memory allocations per gate
- Cache misses (if profiler available)

**Script**:
```bash
# profile_memory.sh
/usr/bin/time -v ./quantum_sim samples/vqe_h2_n15.json 2>&1 | tee memory_profile.txt
```

### Test 5: Regression Tests (No Slowdowns)
**Purpose**: Ensure optimizations don't hurt edge cases

**Test Cases**:
- Very low rank (r=1)
- Very high rank (r=128)
- Column-parallel cases (should be unchanged)
- Sequential cases (n<8, should be unchanged)
- Empty circuits
- Single-gate circuits

**Pass Criteria**: No slowdown >5% in any edge case

---

## 📊 Comparison Script

### compare_performance.py
```python
#!/usr/bin/env python3
"""
Compare performance between baseline and optimized LRET
Usage: python compare_performance.py results/
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def parse_log(log_file):
    """Extract metrics from quantum_sim log file"""
    with open(log_file, 'r') as f:
        content = f.read()
    
    metrics = {}
    # Parse timing info
    if "Total time:" in content:
        metrics['total_time'] = float(content.split("Total time:")[1].split()[0])
    if "Gate time:" in content:
        metrics['gate_time'] = float(content.split("Gate time:")[1].split()[0])
    if "Noise time:" in content:
        metrics['noise_time'] = float(content.split("Noise time:")[1].split()[0])
    if "Truncation time:" in content:
        metrics['truncation_time'] = float(content.split("Truncation time:")[1].split()[0])
    if "Final rank:" in content:
        metrics['final_rank'] = int(content.split("Final rank:")[1].split()[0])
    
    return metrics

def compare_results(results_dir):
    """Compare all baseline vs optimized results"""
    results_dir = Path(results_dir)
    
    comparisons = []
    
    for baseline_file in results_dir.glob("baseline_*.log"):
        circuit_name = baseline_file.stem.replace("baseline_", "")
        optimized_file = results_dir / f"optimized_{circuit_name}.log"
        
        if not optimized_file.exists():
            print(f"Warning: No optimized result for {circuit_name}")
            continue
        
        baseline = parse_log(baseline_file)
        optimized = parse_log(optimized_file)
        
        comparison = {
            'circuit': circuit_name,
            'baseline_time': baseline.get('total_time', 0),
            'optimized_time': optimized.get('total_time', 0),
            'speedup': baseline.get('total_time', 1) / optimized.get('total_time', 1),
            'baseline_rank': baseline.get('final_rank', 0),
            'optimized_rank': optimized.get('final_rank', 0),
        }
        
        comparisons.append(comparison)
    
    df = pd.DataFrame(comparisons)
    
    # Print summary
    print("\n=== Performance Comparison Summary ===")
    print(df.to_string(index=False))
    print(f"\nAverage Speedup: {df['speedup'].mean():.2f}×")
    print(f"Geometric Mean Speedup: {df['speedup'].prod() ** (1/len(df)):.2f}×")
    
    # Save to CSV
    df.to_csv(results_dir / 'comparison_summary.csv', index=False)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Execution time comparison
    x = range(len(df))
    ax1.bar([i-0.2 for i in x], df['baseline_time'], width=0.4, label='Baseline', alpha=0.8)
    ax1.bar([i+0.2 for i in x], df['optimized_time'], width=0.4, label='Optimized', alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['circuit'], rotation=45)
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Execution Time Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Speedup bar chart
    ax2.bar(x, df['speedup'], alpha=0.8, color='green')
    ax2.axhline(y=1.0, color='r', linestyle='--', label='No change')
    ax2.axhline(y=1.5, color='b', linestyle='--', label='Target (1.5×)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['circuit'], rotation=45)
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('Speedup (Higher is Better)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'comparison_plot.png', dpi=150)
    print(f"\nPlot saved to: {results_dir / 'comparison_plot.png'}")
    
    return df

if __name__ == '__main__':
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else 'results'
    compare_results(results_dir)
```

---

## 🎯 Phase 1 Implementation Checklist

### Pre-Implementation
- [ ] Create `LRET_optimized/` folder (duplicate of `LRET/`)
- [ ] Verify both versions build successfully
- [ ] Run baseline benchmarks (save to `results/baseline_*.log`)
- [ ] Create `results/` directory for outputs

### Implementation (in LRET_optimized only)
- [ ] **Change 1**: Raise rank threshold to 32 (`src/parallel_modes.cpp:56`)
- [ ] **Change 2**: Add SIMD pragma (`src/parallel_modes.cpp:~250`)
- [ ] **Change 3**: Add stride-aware scheduling (`src/parallel_modes.cpp:~400`)
- [ ] **Change 4**: Row-parallel trace (`src/utils.cpp` or `src/simulator.cpp`)
- [ ] **Change 5**: Row-parallel sampling (`src/utils.cpp`)
- [ ] Rebuild optimized version

### Testing
- [ ] Run unit tests (both versions)
- [ ] Run benchmark suite (save to `results/optimized_*.log`)
- [ ] Run fidelity tests (expect >0.999)
- [ ] Profile memory usage
- [ ] Run regression tests

### Analysis
- [ ] Run `compare_performance.py`
- [ ] Review speedup factors (target: 1.5×)
- [ ] Check for any slowdowns in edge cases
- [ ] Verify fidelity preservation

### Decision
- [ ] **If speedup ≥ 1.2× AND fidelity ≥ 0.999**: ✅ Integrate
- [ ] **If speedup 1.0-1.2×**: ⚠️ Review and tune
- [ ] **If speedup < 1.0×**: ❌ Investigate and fix

### Integration (only if passing)
- [ ] Copy optimized code to `LRET/`
- [ ] Commit with detailed performance report
- [ ] Update documentation
- [ ] Tag release: `v2.0-phase1-row-parallel`

---

## 📁 Directory Structure After Duplication

```
parent_directory/
├── LRET/                           # BASELINE (untouched)
│   ├── src/
│   │   ├── parallel_modes.cpp      # Original
│   │   ├── simulator.cpp           # Original
│   │   └── ...
│   ├── build/
│   │   └── quantum_sim             # Baseline binary
│   └── ...
│
├── LRET_optimized/                 # EXPERIMENTAL (modified)
│   ├── src/
│   │   ├── parallel_modes.cpp      # MODIFIED (rank threshold)
│   │   ├── simulator.cpp           # MODIFIED (row-parallel trace)
│   │   └── ...
│   ├── build/
│   │   └── quantum_sim             # Optimized binary
│   └── ...
│
└── results/                        # Comparison outputs
    ├── baseline_vqe_h2.log
    ├── optimized_vqe_h2.log
    ├── baseline_qaoa.log
    ├── optimized_qaoa.log
    ├── comparison_summary.csv
    └── comparison_plot.png
```

---

## 🚨 Important Notes

### What Changes Go Where
- ✅ **LRET_optimized/**: All code modifications
- ✅ **LRET/**: No changes (pristine baseline)
- ✅ **results/**: Test outputs and comparisons

### If Something Goes Wrong
- Optimized version has bugs → Fix in `LRET_optimized/`, retest
- Performance worse → Don't integrate, investigate why
- Baseline accidentally modified → Re-copy from git: `git checkout src/`

### Git Workflow
```bash
# Work in LRET_optimized, but commit from LRET
cd LRET_optimized
# ... make changes ...

cd ../LRET
# Only copy files once verified
cp ../LRET_optimized/src/parallel_modes.cpp src/
git add src/
git commit -m "Phase 1: Row parallelism optimizations (verified 1.5× speedup)"
```

---

## 📈 Success Criteria Summary

| Metric | Requirement | Target |
|--------|------------|--------|
| **Speedup** | ≥ 1.2× | 1.5× |
| **Fidelity** | ≥ 0.999 | 0.9999 |
| **Unit Tests** | 100% pass | 100% pass |
| **Regression** | No >5% slowdown | No slowdown |
| **Memory** | No increase >10% | Same or better |

**Decision**: Integrate if ALL requirements met.

---

## 🔄 Iterative Workflow

If Phase 1 doesn't meet targets:
1. Analyze bottlenecks in `LRET_optimized/`
2. Make additional changes
3. Retest
4. Repeat until targets met OR decide to skip/postpone

**Key Principle**: Never integrate code that doesn't improve performance!

---

## ✅ Ready for Implementation

**Current Status**: 
- ✅ Research complete
- ✅ Strategy documented
- ✅ Testing framework designed
- ✅ Workflow defined
- ⏳ Waiting for folder duplication

**Next Step**: 
1. User confirms workflow
2. Create `LRET_optimized/` folder
3. Start Phase 1 implementation with Claude Opus 4.5

---

**Workflow Owner**: Development Team  
**Last Updated**: February 5, 2026  
**Status**: Ready to Execute
