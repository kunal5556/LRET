# Parallel Modes Benchmarking System

Comprehensive benchmarking system for comparing the new row-parallel optimizations against traditional parallel methods in the LRET quantum simulator.

## Overview

This benchmarking system tests and validates the **Phase 1-4 row-parallel optimizations**:

- **Phase 1**: Iterative Compression + DLRA (Dynamical Low-Rank Approximation)
- **Phase 2**: CP-ALS tensor decomposition + Sparse Tensor optimization
- **Phase 3**: Distributed Tensor Scatter + Variational Lindblad Evolution
- **Phase 4**: Morton Order Cache Optimization + Parallelism Oracle

### Parallel Modes Compared

1. **SEQUENTIAL** - No parallelism (baseline)
2. **ROW** - Row-wise parallelization (NEW optimizations)
3. **COLUMN** - Column-wise parallelization (traditional)
4. **HYBRID** - Combined row + batch strategy

## Quick Start

### 1. Prerequisites

```bash
# Ensure quantum_sim is built
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

# Install Python dependencies
pip install numpy matplotlib seaborn psutil
```

### 2. Run Quick Benchmark (5-10 minutes)

```bash
# From LRET repository root
python benchmarks/parallel_modes_benchmark_quick.py
```

### 3. Generate Visualizations

```bash
# After benchmark completes
python scripts/benchmark_visualize_modes.py results/parallel_modes_quick/results.json
```

### 4. Validate Results

```bash
python benchmarks/validation_utils.py results/parallel_modes_quick/results.json
```

## File Structure

```
benchmarks/
├── parallel_modes_benchmark.py       # Main orchestrator
├── parallel_modes_benchmark_quick.py # Quick version (5-10 min)
├── validation_utils.py               # Correctness validation
└── README_parallel_benchmarks.md    # This file

scripts/
└── benchmark_visualize_modes.py     # Visualization generation

results/parallel_modes_*/
└── run_YYYYMMDD_HHMMSS/
    ├── config.json                  # Benchmark configuration
    ├── results.json                 # Raw results
    ├── summary.json                 # Aggregated statistics
    ├── validation_report.json       # Validation results
    ├── benchmark.log                # Execution log
    └── plots/                       # Generated visualizations
        ├── mode_comparison_bar.png
        ├── speedup_heatmap.png
        ├── rank_evolution.png
        ├── time_per_state.png
        ├── scaling_comparison.png
        ├── memory_comparison.png
        └── dashboard.png
```

## Usage

###option Quick Benchmark

**Estimated runtime:** 5-10 minutes
**Configuration:** 4-8 qubits, 2 trials, 3 modes

```bash
python benchmarks/parallel_modes_benchmark_quick.py
```

### Comprehensive Benchmark

**Estimated runtime:** 2-8 hours
**Configuration:** 4-16 qubits, 10 trials, 4 modes, multiple circuit types

```bash
python benchmarks/parallel_modes_benchmark.py --comprehensive
```

### Custom Benchmark

```bash
python benchmarks/parallel_modes_benchmark.py \
    --qubits 4,6,8,10 \
    --depths 10,20 \
    --noise 0.01 \
    --epsilon 1e-4 \
    --modes sequential,row,column,hybrid \
    --trials 5 \
    --circuit-types random,qft \
    --output results/custom_benchmark
```

### Command-Line Options

```
--quick                 Run quick benchmark (4-8 qubits, 2 trials)
--comprehensive         Run comprehensive benchmark (4-16 qubits, 10 trials)
--qubits QUBITS         Comma-separated qubit counts (e.g., '4,6,8')
--depths DEPTHS         Comma-separated circuit depths (e.g., '10,20')
--noise NOISE           Comma-separated noise levels (e.g., '0.0,0.01')
--epsilon EPSILON       Comma-separated epsilon values (default: 1e-4)
--modes MODES           Comma-separated modes (e.g., 'sequential,row,hybrid')
--circuit-types TYPES   Comma-separated circuit types (default: random)
--trials TRIALS         Number of trials per configuration (default: 5)
--output OUTPUT         Output directory
--timeout TIMEOUT       Timeout in seconds (default: 3600)
--export-state          Export full quantum state for validation
```

## Visualization

The visualization module generates 8 types of plots:

### 1. Mode Comparison Bar Chart
- **Purpose**: Quick visual comparison of execution times
- **Shows**: Bar chart with speedup annotations
- **File**: `mode_comparison_bar.png`

### 2. Speedup Heatmap
- **Purpose**: Show which modes excel at different scales
- **Shows**: 2D heatmap (modes × qubits) with speedup ratios
- **File**: `speedup_heatmap.png`

### 3. Rank Evolution Trajectories
- **Purpose**: Compare rank growth patterns
- **Shows**: Line plot of rank over operation index
- **File**: `rank_evolution.png`

### 4. Time per Quantum State
- **Purpose**: Efficiency metric (μs/state)
- **Shows**: Time normalized by number of states and depth
- **File**: `time_per_state.png`

### 5. Scaling Comparison
- **Purpose**: Identify complexity trends
- **Shows**: Multi-line time vs qubits (log-log scale)
- **File**: `scaling_comparison.png`

### 6. Memory Comparison
- **Purpose**: Memory efficiency comparison
- **Shows**: Peak memory by mode and qubit count
- **File**: `memory_comparison.png`

### 7. Dashboard
- **Purpose**: Single-page overview
- **Shows**: 2×3 grid combining all key plots
- **File**: `dashboard.png`

## Validation

The validation module checks:

### 1. Trace Normalization
- **Check**: |Tr(ρ) - 1.0| < 1e-6
- **Purpose**: Verify quantum state normalization

### 2. Rank Validity
- **Check**: 1 ≤ rank ≤ 2^n
- **Purpose**: Ensure reasonable rank values

### 3. Purity Bounds
- **Check**: 0 ≤ purity ≤ 1
- **Purpose**: Verify purity is physically valid

### 4. Execution Status
- **Check**: Success rate > 95%
- **Purpose**: Detect systematic failures

### 5. Performance Sanity
- **Check**: Parallel modes not significantly slower
- **Purpose**: Detect performance regressions

## Expected Results

### Performance Improvements

Based on Phase 1-4 optimizations, expected speedups:

| Mode | Expected Speedup | Notes |
|------|-----------------|-------|
| **SEQUENTIAL** | 1.0× | Baseline |
| **ROW** | 1.5-2.5× | With DLRA, Morton order, CP-ALS |
| **COLUMN** | 0.8-1.5× | Better for high-rank states |
| **HYBRID** | 2-3× | Best overall performance |

### Optimization Impact

| Optimization | Benefit |
|-------------|---------|
| **Morton Order** | 50-80% cache miss reduction (n ≥ 14 qubits) |
| **DLRA** | 3-5× rank stabilization, prevents rank explosion |
| **CP-ALS** | 2-5× speedup for Kronecker-separable circuits (QFT, Grover) |
| **Parallelism Oracle** | +20% performance via adaptive mode selection |

### Correctness

All modes should produce identical results:

- **Fidelity**: > 0.9999 between all mode pairs
- **Trace**: |Tr(ρ) - 1.0| < 1e-6
- **Rank Evolution**: Nearly identical trajectories (max diff ≤ 2)
- **Purity**: Values match within 1%

## Troubleshooting

### Issue: quantum_sim.exe not found

**Solution:**
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### Issue: Import errors (matplotlib, numpy, etc.)

**Solution:**
```bash
pip install numpy matplotlib seaborn psutil
```

### Issue: Slow performance / timeouts

**Solution:**
- Use `--quick` for faster testing
- Reduce `--trials` count
- Increase `--timeout` value
- Test with fewer qubits first: `--qubits 4,6,8`

### Issue: Empty plots / missing data

**Solution:**
- Check that benchmark completed successfully
- Verify `results.json` contains successful runs
- Check `benchmark.log` for errors

### Issue: Validation failures

**Solution:**
- Review `validation_report.json` for specific failures
- Check trace normalization: ensure quantum_sim produces normalized states
- Verify circuit generation is correct

## Examples

### Example 1: Quick Test of New ROW Mode

```bash
# Test ROW vs SEQUENTIAL only
python benchmarks/parallel_modes_benchmark.py \
    --qubits 4,6,8 \
    --depths 10 \
    --modes sequential,row \
    --trials 3 \
    --output results/row_vs_sequential

# Visualize
python scripts/benchmark_visualize_modes.py \
    results/row_vs_sequential/run_*/results.json
```

### Example 2: Test Specific Circuit Type

```bash
# QFT circuits only
python benchmarks/parallel_modes_benchmark.py \
    --qubits 6,8,10 \
    --circuit-types qft \
    --modes sequential,row,column,hybrid \
    --trials 5 \
    --output results/qft_benchmark
```

### Example 3: Noise Sensitivity Analysis

```bash
# Test different noise levels
python benchmarks/parallel_modes_benchmark.py \
    --qubits 8 \
    --depths 10 \
    --noise 0.0,0.001,0.01,0.05 \
    --modes row,hybrid \
    --trials 5 \
    --output results/noise_analysis
```

## Interpreting Results

### Good Indicators

✓ **ROW speedup > 1.5×**: New optimizations are effective
✓ **HYBRID fastest**: Combined strategy works well
✓ **Consistent ranks**: All modes converge to similar ranks
✓ **Fidelity ≈ 1.0**: Correctness validated
✓ **Sub-linear scaling**: Efficiency improves with size

### Warning Signs

⚠ **ROW slower than SEQUENTIAL**: Check if optimizations are enabled
⚠ **High rank variance**: Potential numerical instability
⚠ **Low fidelity**: Correctness issues
⚠ **Super-exponential scaling**: Memory/performance problems

## Contributing

To add new circuit types:

1. Add generator function to `parallel_modes_benchmark.py`:
   ```python
   def generate_YOUR_CIRCUIT_json(n_qubits, noise_prob):
       ops = [...]
       return {"operations": ops}
   ```

2. Update `generate_circuit()` function
3. Test with quick benchmark

## References

- **LRET Documentation**: `../ROW_PARALLELISM_OPTIMIZATION_STRATEGY.md`
- **Phase 1-4 Details**: `../ROW_PARALLEL_ADVANCED_OPTIMIZATION_PLAN.md`
- **Quick Reference**: `../ROW_PARALLELISM_QUICK_REFERENCE.md`

## Support

For issues or questions:
1. Check `benchmark.log` in results directory
2. Run validation module for diagnostics
3. Review LRET documentation
4. Report issues with:
   - Full command used
   - Contents of `config.json`
   - Relevant log excerpts
   - System information

---

**Last Updated**: March 2026
**Version**: 1.0
**Author**: LRET Development Team
