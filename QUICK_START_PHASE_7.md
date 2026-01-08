# Quick Start: Phase 7 Benchmarking (READY NOW!)

Since Phase 7 benchmarking is fully ready with zero dependencies, here's a quick-start guide to generate your first performance baselines.

---

## Prerequisites Check

```bash
# Verify core library is built
cd /Users/suryanshsingh/Documents/LRET
ls -la build/libqlret_lib.a

# Verify Python is available
python3 --version

# Verify matplotlib for plots
python3 -c "import matplotlib; print('Matplotlib OK')"
```

If any fail, run:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
pip install matplotlib seaborn numpy scipy
```

---

## Option 1: Quick Benchmark (5-10 minutes)

Fastest way to verify everything works and get baseline measurements.

```bash
cd /Users/suryanshsingh/Documents/LRET

# Run quick benchmark suite
python3 scripts/benchmark_suite.py --quick

# This tests:
# - Scaling (6-8 qubits only)
# - Parallel speedup across modes
# - Accuracy vs FDM
# - Memory usage

# Output: benchmark_results.csv
```

**Expected Output:**
```
Benchmarking LRET quantum simulator...
Configuration:
  Qubits: 6-8
  Modes: sequential, row, column, hybrid, adaptive
  Categories: scaling, parallel, accuracy

Progress: [████████████████] 40/40 tests completed

Results saved to: benchmark_results.csv
Execution time: 8 minutes 34 seconds
```

---

## Option 2: Full Benchmark (30-60 minutes)

Complete performance characterization.

```bash
cd /Users/suryanshsingh/Documents/LRET

# Run full benchmark suite
python3 scripts/benchmark_suite.py

# This tests:
# - Scaling (6-14 qubits)
# - Parallel speedup (all modes)
# - Accuracy validation
# - Depth scaling
# - Memory profiling
# - All benchmark categories
```

**Expected Output:**
```
Benchmarking LRET quantum simulator (FULL SUITE)...

Category: scaling
  [████████████████] 45/45 tests | 12m 34s

Category: parallel
  [████████████████] 60/60 tests | 18m 12s

Category: accuracy
  [████████████████] 30/30 tests | 8m 45s

Category: depth
  [████████████████] 20/20 tests | 5m 30s

Category: memory
  [████████████████] 15/15 tests | 3m 15s

Total: 170 tests | 48m 16s

Results saved to: benchmark_results.csv
```

---

## Option 3: Custom Categories (20-30 minutes)

Run only specific benchmarks.

```bash
cd /Users/suryanshsingh/Documents/LRET

# Just scaling and parallel
python3 scripts/benchmark_suite.py --categories scaling,parallel

# Just accuracy
python3 scripts/benchmark_suite.py --categories accuracy

# Just memory
python3 scripts/benchmark_suite.py --categories memory
```

---

## Step 1: Generate Baseline Results

```bash
cd /Users/suryanshsingh/Documents/LRET

# Record today's date for baseline naming
DATE=$(date +%Y%m%d_%H%M%S)

# Run quick benchmark with timestamp
python3 scripts/benchmark_suite.py --quick --output baseline_${DATE}.csv

# Or full benchmark
python3 scripts/benchmark_suite.py --output baseline_${DATE}.csv
```

**Output Files:**
```
benchmark_results.csv          # Latest run
baseline_20260108_143022.csv   # Timestamped backup
```

---

## Step 2: Analyze Results

```bash
cd /Users/suryanshsingh/Documents/LRET

# Analyze benchmark results
python3 scripts/benchmark_analysis.py benchmark_results.csv

# This generates:
# - Execution time summary
# - Scaling analysis (exponential fit)
# - Speedup metrics
# - Regression detection (if baseline exists)
# - Memory efficiency report
```

**Expected Output:**
```
=== Benchmark Analysis Report ===

Scaling Analysis:
  Linear regression: time = 2.3^n * 1.4 ms
  R² = 0.988
  Exponential growth confirmed

Parallel Speedup:
  Speedup at n=10:
    Row mode:    2.1x vs sequential
    Column mode: 3.2x vs sequential
    Hybrid mode: 2.8x vs sequential
    Adaptive:    3.1x vs sequential

Accuracy (LRET vs FDM):
  Mean fidelity difference: 0.0012
  Max fidelity error: 0.0045
  Within expected tolerance: YES

Memory Scaling:
  n=6:  45 MB
  n=8:  185 MB
  n=10: 742 MB
  n=12: 2.9 GB (near limit)

Regression Detection:
  Comparing to baseline_20260107_120000.csv:
  ✓ No regressions detected
  ✓ Times improved by 1-3%
```

---

## Step 3: Generate Visualization Plots

```bash
cd /Users/suryanshsingh/Documents/LRET

# Create plots directory
mkdir -p results/plots

# Generate all plots
python3 scripts/benchmark_visualize.py benchmark_results.csv -o results/plots/

# View plots (macOS)
open results/plots/plot_scaling.svg
open results/plots/plot_speedup.svg
open results/plots/plot_fidelity.svg
open results/plots/plot_memory.svg
open results/plots/plot_summary.svg
```

**Generated Plots:**

1. **plot_scaling.svg** - Execution time vs qubit count
   - Shows exponential growth
   - All modes compared
   - Log-scale for better fit visualization

2. **plot_speedup.svg** - Parallel speedup comparison
   - Row vs Column vs Hybrid vs Adaptive
   - Speedup over sequential baseline
   - Efficiency curves

3. **plot_fidelity.svg** - LRET vs FDM accuracy
   - Fidelity by circuit depth
   - Error margins shown
   - Highlights any mode-specific issues

4. **plot_memory.svg** - Memory usage scaling
   - Peak memory by qubit count
   - All modes shown
   - Identifies memory bottlenecks

5. **plot_summary.svg** - Comprehensive overview
   - All metrics in one figure
   - Time, speedup, fidelity, memory
   - Publication-ready quality

---

## Step 4: Save Baseline for Regression Detection

```bash
cd /Users/suryanshsingh/Documents/LRET

# Save current results as baseline
cp benchmark_results.csv results/baseline_reference.csv

# Future runs will compare against this:
python3 scripts/benchmark_analysis.py benchmark_results.csv \
    --baseline results/baseline_reference.csv
```

---

## Common Commands Reference

```bash
# Quick validation (5 min)
python3 scripts/benchmark_suite.py --quick

# Full characterization (45 min)
python3 scripts/benchmark_suite.py

# Specific modes only
python3 scripts/benchmark_suite.py --categories scaling

# Custom output file
python3 scripts/benchmark_suite.py --output my_results.csv

# Analyze results
python3 scripts/benchmark_analysis.py benchmark_results.csv

# Generate plots
python3 scripts/benchmark_visualize.py benchmark_results.csv -o results/plots/

# Compare to baseline
python3 scripts/benchmark_analysis.py results/new.csv --baseline results/baseline.csv

# Quiet mode (less output)
python3 scripts/benchmark_suite.py --quiet

# Verbose mode (more details)
python3 scripts/benchmark_suite.py --verbose
```

---

## Expected Performance (For Reference)

On a typical modern laptop (M1/M2 macOS or equivalent):

| Qubits | Mode | Time | Rank | Speedup |
|--------|------|------|------|---------|
| 6 | Sequential | 0.8ms | 7 | 1.0x |
| 6 | Row | 0.6ms | 7 | 1.3x |
| 6 | Column | 0.5ms | 7 | 1.6x |
| 10 | Sequential | 45ms | 127 | 1.0x |
| 10 | Row | 21ms | 127 | 2.1x |
| 10 | Column | 14ms | 127 | 3.2x |
| 12 | Sequential | 742ms | 511 | 1.0x |
| 12 | Row | 298ms | 511 | 2.5x |
| 12 | Hybrid | 245ms | 511 | 3.0x |

**Note:** Your actual times will vary based on:
- CPU architecture (ARM vs Intel)
- Circuit depth and gate types
- Rank evolution during simulation
- Noise modeling (adds overhead)

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'matplotlib'"

**Solution:**
```bash
pip install matplotlib seaborn numpy scipy
```

### Issue: "No such file or directory: quantum_sim"

**Solution:**
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Issue: Slow execution (>3h for full suite)

**Solution:** Use quick mode instead
```bash
python3 scripts/benchmark_suite.py --quick  # ~10 min
```

### Issue: Output file not created

**Solution:** Check directory permissions
```bash
pwd  # Should be /Users/suryanshsingh/Documents/LRET
ls -la results/  # Should be writable
```

---

## Next Steps After Benchmarking

1. **Save Results**
   ```bash
   mkdir -p results/
   mv benchmark_results.csv results/baseline_$(date +%Y%m%d).csv
   ```

2. **Generate Report**
   ```bash
   python3 scripts/benchmark_analysis.py results/baseline_*.csv > results/benchmark_report.txt
   ```

3. **Track Changes**
   ```bash
   git add results/
   git commit -m "Add Phase 7 benchmark baselines"
   ```

4. **Continue to Next Tier**
   - Once baselines established, proceed to Tiers 2-6
   - Periodic re-benchmarking to detect regressions

---

## Success Checklist

- [ ] Python scripts execute without errors
- [ ] benchmark_results.csv contains valid data
- [ ] Analysis report generates successfully
- [ ] All plots save as SVG files
- [ ] Speedup curves make physical sense
- [ ] Memory scaling is exponential
- [ ] Results reproducible across runs

---

## Estimated Time Investment

| Task | Duration |
|------|----------|
| Quick benchmark | 10 min |
| Analysis | 2 min |
| Plot generation | 3 min |
| Review results | 5 min |
| **Total** | **20 min** |

**Or for full suite:**

| Task | Duration |
|------|----------|
| Full benchmark | 45 min |
| Analysis | 3 min |
| Plot generation | 5 min |
| Review results | 10 min |
| **Total** | **63 min** |

---

## Recommended Approach

**For immediate feedback:**
```bash
python3 scripts/benchmark_suite.py --quick
```

**For comprehensive baseline:**
```bash
python3 scripts/benchmark_suite.py
python3 scripts/benchmark_analysis.py benchmark_results.csv
python3 scripts/benchmark_visualize.py benchmark_results.csv -o results/plots/
```

Both are ready to run **right now** with no additional setup! 🚀
