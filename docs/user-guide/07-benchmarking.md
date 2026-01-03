# Benchmarking Guide

Comprehensive guide to measuring and analyzing LRET performance using the built-in benchmarking suite.

## Overview

LRET includes three Python modules for benchmarking:
1. **`benchmark_suite.py`** - Run performance tests
2. **`benchmark_analysis.py`** - Statistical analysis and regression detection
3. **`benchmark_visualize.py`** - Generate plots and visualizations

---

## Quick Start

### Run All Benchmarks

```bash
# Full benchmark suite (takes ~30-60 minutes)
python scripts/benchmark_suite.py --output results.csv

# Analyze results
python scripts/benchmark_analysis.py results.csv

# Generate plots
python scripts/benchmark_visualize.py results.csv --output plots/
```

### Quick Benchmark (5 minutes)

```bash
# Reduced configurations for CI/testing
python scripts/benchmark_suite.py --quick --output quick_results.csv
```

---

## Benchmark Categories

### 1. Scaling Benchmarks

**Purpose:** Measure how simulation time scales with qubit count and circuit depth.

**Command:**
```bash
python scripts/benchmark_suite.py --categories scaling --output scaling.csv
```

**Configurations:**
- **Qubits:** 6, 8, 10, 12, 14
- **Depths:** 10, 20, 30, 40, 50
- **Noise:** 1% depolarizing
- **Trials:** 3 per configuration

**Metrics:**
- Simulation time (seconds)
- Final rank
- Memory usage (MB)
- Fidelity (vs noiseless)

**Expected Results:**

| Qubits | Depth | Time (s) | Rank | Memory (MB) |
|--------|-------|----------|------|-------------|
| 6      | 10    | 0.005    | 6    | 12          |
| 8      | 20    | 0.047    | 12   | 25          |
| 10     | 30    | 0.189    | 23   | 58          |
| 12     | 40    | 0.712    | 35   | 142         |
| 14     | 50    | 3.245    | 51   | 321         |

---

### 2. Parallel Benchmarks

**Purpose:** Compare parallelization modes (sequential, row, column, hybrid).

**Command:**
```bash
python scripts/benchmark_suite.py --categories parallel --output parallel.csv
```

**Configurations:**
- **Qubits:** 8, 10, 12
- **Depth:** 30
- **Modes:** sequential, row, column, hybrid
- **Threads:** 1, 2, 4, 8, 16

**Metrics:**
- Time per mode
- Speedup vs sequential
- Parallel efficiency (speedup / threads)

**Expected Results:**

| Qubits | Threads | Sequential | Row  | Column | Hybrid | Best Speedup |
|--------|---------|------------|------|--------|--------|--------------|
| 8      | 4       | 0.120s     | 0.048s | 0.052s | 0.041s | 2.9×         |
| 10     | 8       | 0.512s     | 0.142s | 0.156s | 0.118s | 4.3×         |
| 12     | 8       | 2.145s     | 0.389s | 0.425s | 0.342s | 6.3×         |

---

### 3. Accuracy Benchmarks

**Purpose:** Validate LRET fidelity vs full-density-matrix (FDM) simulation.

**Command:**
```bash
python scripts/benchmark_suite.py --categories accuracy --output accuracy.csv
```

**Configurations:**
- **Qubits:** 6, 8, 10, 12
- **Depth:** 20
- **Noise levels:** 0.001, 0.005, 0.01, 0.02, 0.05
- **Thresholds:** 1e-3, 1e-4, 1e-5, 1e-6

**Metrics:**
- Fidelity (LRET vs FDM)
- Trace distance
- Final rank
- Speedup

**Expected Results:**

| Noise | Threshold | Fidelity | Trace Distance | Rank |
|-------|-----------|----------|----------------|------|
| 0.001 | 1e-4      | 0.99998  | 2.3e-5         | 45   |
| 0.01  | 1e-4      | 0.9997   | 3.1e-4         | 23   |
| 0.05  | 1e-4      | 0.998    | 2.1e-3         | 12   |

---

### 4. Depth Benchmarks

**Purpose:** Test performance on deep circuits.

**Command:**
```bash
python scripts/benchmark_suite.py --categories depth --output depth.csv
```

**Configurations:**
- **Qubits:** 8
- **Depths:** 10, 20, 50, 100, 200, 500, 1000
- **Noise:** 1% depolarizing

**Metrics:**
- Time vs depth
- Rank growth rate
- Memory usage

**Expected Results:**

| Depth | Time (s) | Rank | Memory (MB) | Time/Depth (ms) |
|-------|----------|------|-------------|-----------------|
| 10    | 0.012    | 8    | 18          | 1.2             |
| 50    | 0.098    | 18   | 42          | 2.0             |
| 100   | 0.234    | 25   | 68          | 2.3             |
| 500   | 1.456    | 42   | 156         | 2.9             |
| 1000  | 3.124    | 51   | 245         | 3.1             |

---

### 5. Memory Benchmarks

**Purpose:** Measure memory usage and identify potential leaks.

**Command:**
```bash
python scripts/benchmark_suite.py --categories memory --output memory.csv
```

**Configurations:**
- **Qubits:** 8, 10, 12, 14, 16
- **Depth:** 50
- **Noise:** 1%

**Metrics:**
- Peak memory (MB)
- Memory growth rate
- FDM memory (for comparison)

**Expected Results:**

| Qubits | LRET Memory | FDM Memory | Memory Ratio | Rank |
|--------|-------------|------------|--------------|------|
| 8      | 25 MB       | 268 MB     | 10.7×        | 12   |
| 10     | 58 MB       | 4.3 GB     | 75.9×        | 23   |
| 12     | 142 MB      | 68.7 GB    | 496×         | 35   |
| 14     | 321 MB      | 1.1 TB     | 3516×        | 51   |

---

## Benchmark Analysis

### Statistical Analysis

```bash
# Analyze results with statistics
python scripts/benchmark_analysis.py results.csv --output analysis.json
```

**Output (`analysis.json`):**
```json
{
  "scaling_fit": {
    "time_vs_qubits": {
      "model": "exponential",
      "coefficient": 2.31,
      "base": 1.89,
      "r_squared": 0.998
    },
    "rank_vs_depth": {
      "model": "linear",
      "slope": 0.52,
      "intercept": 3.1,
      "r_squared": 0.987
    }
  },
  "parallel_efficiency": {
    "8_qubits": {
      "max_speedup": 4.3,
      "optimal_threads": 8,
      "efficiency": 0.54
    }
  },
  "accuracy_summary": {
    "mean_fidelity": 0.9997,
    "std_fidelity": 0.0002,
    "min_fidelity": 0.9991,
    "max_fidelity": 0.9999
  }
}
```

### Regression Detection

```bash
# Compare against baseline
python scripts/benchmark_analysis.py results.csv \
    --baseline-file baseline.csv \
    --regression-threshold 0.10  # 10% slowdown = regression
```

**Output:**
```
=== Regression Analysis ===
Configuration: n=10, d=30, noise=0.01
  Baseline time: 0.189s
  Current time:  0.245s
  Regression: 29.6% slower ❌

Configuration: n=12, d=40, noise=0.01
  Baseline time: 0.712s
  Current time:  0.698s
  Improvement: 2.0% faster ✓

Total: 1 regressions, 0 improvements
```

**Exit code 1 if regressions detected** (useful for CI).

---

## Visualization

### Generate All Plots

```bash
python scripts/benchmark_visualize.py results.csv --output plots/
```

**Generated files:**
- `scaling_qubits.png` - Time vs qubit count
- `scaling_depth.png` - Time vs depth
- `parallel_speedup.png` - Speedup vs thread count
- `accuracy_vs_noise.png` - Fidelity vs noise level
- `rank_growth.png` - Rank vs depth
- `memory_usage.png` - Memory vs qubit count
- `summary.png` - Multi-panel summary figure

### Individual Plot Types

```bash
# Scaling plots only
python scripts/benchmark_visualize.py results.csv --plot-types scaling --output plots/

# Parallel speedup
python scripts/benchmark_visualize.py results.csv --plot-types parallel --output plots/

# Accuracy validation
python scripts/benchmark_visualize.py results.csv --plot-types accuracy --output plots/
```

### Custom Plot Options

```bash
# PNG format (default)
python scripts/benchmark_visualize.py results.csv --format png --dpi 300

# SVG for publications
python scripts/benchmark_visualize.py results.csv --format svg

# PDF vector graphics
python scripts/benchmark_visualize.py results.csv --format pdf
```

---

## Advanced Usage

### Custom Benchmark Configurations

```python
from benchmark_suite import BenchmarkSuite

# Create custom suite
suite = BenchmarkSuite(output_file="custom_bench.csv")

# Define custom configurations
configs = [
    {"n_qubits": 10, "depth": 50, "noise": 0.01, "mode": "hybrid"},
    {"n_qubits": 12, "depth": 60, "noise": 0.02, "mode": "hybrid"},
    {"n_qubits": 14, "depth": 70, "noise": 0.015, "mode": "hybrid"}
]

# Run benchmarks
results = suite.run_custom(configs, trials=5)

# Save results
suite.save_results()
```

### Programmatic Analysis

```python
from benchmark_analysis import BenchmarkAnalyzer

# Load results
analyzer = BenchmarkAnalyzer("results.csv")

# Get scaling fit
fit = analyzer.fit_scaling_model(metric="time", variable="n_qubits")
print(f"Time ~ {fit['coefficient']:.2f} * {fit['base']:.2f}^n")

# Get statistics
stats = analyzer.compute_statistics(category="parallel")
print(f"Mean speedup: {stats['speedup']['mean']:.2f}x")
print(f"Max speedup: {stats['speedup']['max']:.2f}x")

# Detect regressions
regressions = analyzer.detect_regressions(
    baseline_file="baseline.csv",
    threshold=0.10
)
print(f"Found {len(regressions)} regressions")
```

### Custom Visualizations

```python
from benchmark_visualize import BenchmarkVisualizer
import matplotlib.pyplot as plt

# Load visualizer
viz = BenchmarkVisualizer("results.csv")

# Create custom plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot scaling with custom styling
viz.plot_scaling(
    ax=ax,
    metric="time",
    variable="n_qubits",
    log_scale=True,
    fit_line=True,
    error_bars=True,
    style="seaborn"
)

ax.set_title("LRET Scaling Performance")
ax.set_xlabel("Number of Qubits")
ax.set_ylabel("Simulation Time (s)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.savefig("custom_scaling.png", dpi=300, bbox_inches="tight")
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
# .github/workflows/benchmark.yml
name: Performance Benchmarks

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          cd python && pip install -e .
      
      - name: Run quick benchmark
        run: |
          python scripts/benchmark_suite.py --quick --output pr_results.csv
      
      - name: Compare against baseline
        run: |
          python scripts/benchmark_analysis.py pr_results.csv \
            --baseline-file benchmarks/baseline.csv \
            --regression-threshold 0.15
      
      - name: Generate plots
        if: always()
        run: |
          mkdir -p plots
          python scripts/benchmark_visualize.py pr_results.csv --output plots/
      
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: |
            pr_results.csv
            plots/*.png
```

### Nightly Benchmarks

```yaml
# .github/workflows/nightly-benchmark.yml
name: Nightly Benchmarks

on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM daily
  workflow_dispatch:

jobs:
  full-benchmark:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Run full benchmark suite
        run: |
          python scripts/benchmark_suite.py --output nightly_results.csv
      
      - name: Analyze results
        run: |
          python scripts/benchmark_analysis.py nightly_results.csv \
            --output analysis.json
      
      - name: Check for regressions
        run: |
          python scripts/benchmark_analysis.py nightly_results.csv \
            --baseline-file benchmarks/baseline.csv \
            --regression-threshold 0.10
      
      - name: Update baseline if improved
        if: success()
        run: |
          python scripts/update_baseline.py \
            --current nightly_results.csv \
            --baseline benchmarks/baseline.csv
      
      - name: Commit updated baseline
        run: |
          git config user.name "Benchmark Bot"
          git config user.email "bot@lret.ai"
          git add benchmarks/baseline.csv
          git commit -m "Update benchmark baseline [skip ci]"
          git push
```

---

## Interpreting Results

### Good Performance Indicators

✅ **Exponential speedup with qubit count**
```
n=8:  0.05s → n=10: 0.19s → n=12: 0.71s
Speedup ratio: ~4× per 2 qubits
```

✅ **Linear rank growth with depth**
```
d=10: rank 8 → d=50: rank 18 → d=100: rank 25
Growth rate: ~0.2 rank/gate
```

✅ **High fidelity (>0.999)**
```
Threshold 1e-4, Noise 1%: Fidelity 0.9997
```

✅ **Parallel scaling efficiency > 50%**
```
8 threads: 4.3× speedup (54% efficiency)
```

### Performance Issues

❌ **Sub-linear speedup**
```
n=8: 0.05s → n=10: 0.08s → n=12: 0.12s
Problem: Rank not growing, noise too high or truncation too aggressive
```

❌ **Exponential rank growth**
```
d=10: rank 8 → d=20: rank 64 → d=30: rank 512
Problem: Noise too low, low-rank advantage lost
```

❌ **Low fidelity (<0.99)**
```
Threshold 1e-3, Noise 1%: Fidelity 0.985
Problem: Truncation threshold too loose
```

❌ **Poor parallel scaling**
```
8 threads: 1.5× speedup (19% efficiency)
Problem: Circuit too small, overhead dominates
```

---

## Best Practices

### 1. Establish Baseline

```bash
# Run benchmark on known-good commit
git checkout v1.0.0
python scripts/benchmark_suite.py --output baseline_v1.0.0.csv

# Use as reference for future comparisons
python scripts/benchmark_analysis.py current_results.csv \
    --baseline-file baseline_v1.0.0.csv
```

### 2. Run Benchmarks Consistently

- Same hardware (CPU, RAM)
- Same system load (no other processes)
- Multiple trials (≥3) for statistical significance
- Fixed random seed for reproducibility

```bash
# Reproducible benchmark
python scripts/benchmark_suite.py \
    --seed 42 \
    --trials 5 \
    --output reproducible.csv
```

### 3. Monitor Trends Over Time

```bash
# Track performance over commits
for commit in $(git log --oneline -n 10 | awk '{print $1}'); do
    git checkout $commit
    python scripts/benchmark_suite.py --quick --output bench_$commit.csv
done

# Plot trend
python scripts/plot_benchmark_history.py bench_*.csv
```

### 4. Profile Before Optimizing

```bash
# Identify bottlenecks
python -m cProfile -o profile.stats scripts/benchmark_suite.py --quick

# Analyze profile
python -m pstats profile.stats
```

---

## Troubleshooting

### Benchmark Takes Too Long

**Solution:** Use quick mode
```bash
python scripts/benchmark_suite.py --quick --categories scaling,parallel
```

### Out of Memory

**Solution:** Reduce max qubits
```bash
python scripts/benchmark_suite.py --max-qubits 12
```

### Inconsistent Results

**Solution:** Increase trials and fix seed
```bash
python scripts/benchmark_suite.py --trials 10 --seed 42
```

### CI Timeouts

**Solution:** Split into multiple jobs
```bash
# Job 1: Scaling
python scripts/benchmark_suite.py --categories scaling --output scaling.csv

# Job 2: Parallel
python scripts/benchmark_suite.py --categories parallel --output parallel.csv
```

---

## See Also

- **[CLI Reference](03-cli-reference.md)** - Command-line options
- **[Python Interface](04-python-interface.md)** - Python API
- **[Troubleshooting](08-troubleshooting.md)** - Common issues
- **[Performance Analysis](../performance/scaling-analysis.md)** - Detailed performance docs
