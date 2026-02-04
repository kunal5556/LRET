# Quick Version vs Full Version Comparison

## Overview

Two versions of the parallel modes benchmark are available:

1. **`run_parallel_modes_benchmark.py`** - Full comprehensive benchmark
2. **`run_parallel_modes_benchmark_quick_version.py`** - Optimized quick version

## Parameter Comparison

| Parameter | Full Version | Quick Version | Impact |
|-----------|-------------|---------------|--------|
| **Qubits** | [8,10,12,14,16,18,20,22,24,26,28,30] | [8,12,16,20,24,28,32] | 12→7 points (1.7× fewer) |
| **Depth** | 25 | 15 | 40% fewer gates |
| **Noise** | 1% (0.01) | 0.5% (0.005) | 2-3× less rank growth |
| **Epsilon** | 1e-6 | 1e-5 | 1.2-1.5× faster SVD |
| **Trials** | 3 | 2 | 33% fewer runs |
| **Total Runs** | 216 | 84 | **2.6× fewer runs** |

## Estimated Time Savings

- **Full Version**: ~72+ hours (based on observed performance)
- **Quick Version**: ~12-18 hours (estimated 4-6× faster)

### Time Reduction Breakdown

1. **Fewer qubit points**: 1.7× reduction
2. **Reduced depth**: 1.5-2× faster per run
3. **Lower noise**: 2-3× less rank growth → faster execution
4. **Relaxed epsilon**: 1.2-1.5× faster SVD operations
5. **Fewer trials**: 1.5× reduction

**Combined effect**: ~4-6× overall speedup

## When to Use Each Version

### Use Full Version When:
- Need comprehensive data for publication
- Want to capture all transition points in scalability
- Have 3+ days available for benchmarking
- Need highest precision (epsilon=1e-6)
- Want statistical confidence (3 trials)

### Use Quick Version When:
- Need results quickly for initial analysis
- Want to validate changes/optimizations
- Limited time (12-18 hours available)
- Focus on key scaling points (powers of 4: 8,16,32)
- Acceptable trade-off: slightly less precision

## What You Still Get in Quick Version

✅ **All 6 modes tested**: SEQUENTIAL, ROW, COLUMN, BATCH, HYBRID, CIRQ  
✅ **Scalability curve**: 7 data points from 8-32 qubits  
✅ **Performance comparison**: Accurate relative performance between modes  
✅ **Breaking point detection**: Can still identify OOM failures  
✅ **Same output format**: Plots, CSV, JSON reports  

## What's Slightly Reduced

⚠️ **Statistical confidence**: 2 trials vs 3 (still reasonable)  
⚠️ **Fine-grained transitions**: Missing 10,14,18,22,26,30 qubit points  
⚠️ **Precision**: epsilon 1e-5 vs 1e-6 (minimal impact on results)  
⚠️ **Noise realism**: 0.5% vs 1% (still realistic, less extreme)  

## How to Run

### Quick Version (Recommended for Initial Tests)
```bash
cd cirq_comparison/automated_benchmarks/parallel_modes_comparison
python run_parallel_modes_benchmark_quick_version.py
```

### Full Version (Comprehensive Analysis)
```bash
cd cirq_comparison/automated_benchmarks/parallel_modes_comparison
python run_parallel_modes_benchmark.py
```

Both scripts:
- Open 2 terminal windows (benchmark + CPU monitor)
- Save results to timestamped directories
- Generate plots and reports automatically
- Can be stopped and restarted (results saved per configuration)

## Expected Results

The quick version will still clearly demonstrate:
- **Memory advantage**: LRET vs Cirq at high qubit counts
- **Speed comparison**: Relative performance of all modes
- **Parallel scaling**: Which modes benefit from multi-core
- **Breaking points**: Where each mode/device runs out of memory

## Recommendation

**Start with Quick Version**, then if results are promising or need publication-grade data, run Full Version overnight/weekend.

Quick version provides 90% of the insights in 15-20% of the time!
