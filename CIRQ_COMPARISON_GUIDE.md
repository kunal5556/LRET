# Cirq FDM vs LRET Comparison Guide

## Quick Start

### Option 1: Fully Automated (Recommended)

```bash
# On workstation with GPU
cd /path/to/LRET
export ANTHROPIC_API_KEY="your-key"  # For Claude (better results)
export PATH=$HOME/.opencode/bin:$PATH

# Run setup (creates all infrastructure)
./setup_cirq_comparison.sh

# Run full comparison (may take hours)
cd cirq_comparison
./run_full_comparison.sh

# Results will be in:
# - cirq_comparison/results/benchmark_results_*.csv
# - cirq_comparison/plots/*.pdf (publication-ready)
# - cirq_comparison/results/tables_for_paper_*.tex
```

### Option 2: Step-by-Step with OpenCode Guidance

```bash
# Let OpenCode guide you through each phase
opencode run --model anthropic/claude-sonnet-4 "
I want to compare Cirq FDM and LRET simulators for publication.

Phase 1: Install Cirq and create comparison infrastructure
Phase 2: Generate test circuits (Bell, GHZ, QFT, random)
Phase 3: Run benchmarks on both simulators
Phase 4: Perform statistical analysis
Phase 5: Create publication plots

Start with Phase 1. After each phase, wait for my confirmation before proceeding.
"
```

### Option 3: Manual Execution with OpenCode Assistance

```bash
# Let OpenCode create each component individually
cd /path/to/LRET

# Create Cirq wrapper
opencode run "Create a Cirq FDM wrapper class at cirq_comparison/cirq_fdm_wrapper.py that matches LRET's interface"

# Generate circuits
opencode run "Create circuit generator that produces Bell, GHZ, QFT circuits in both Cirq and LRET JSON format"

# Run benchmarks
opencode run "Run benchmarks comparing LRET and Cirq on all generated circuits, measuring time, memory, and fidelity"

# Analyze and plot
opencode run "Analyze benchmark results statistically and create publication-quality plots"
```

---

## What OpenCode Will Do

### Phase 1: Infrastructure Setup
- ✅ Install Cirq via pip
- ✅ Create `cirq_fdm_wrapper.py` (Cirq interface matching LRET)
- ✅ Create `circuit_generator.py` (test circuit generation)
- ✅ Create `run_comparison.py` (benchmark runner)
- ✅ Create `analyze_results.py` (statistical analysis)
- ✅ Create `create_plots.py` (publication plots)

### Phase 2: Circuit Generation
- ✅ Bell states (2-20 qubits)
- ✅ GHZ states (3-20 qubits)
- ✅ QFT circuits (4-12 qubits)
- ✅ Random circuits (various depths)
- ✅ Noisy versions (depolarizing: 0.001, 0.01, 0.05)
- ✅ Export to both Cirq and LRET JSON formats

### Phase 3: Benchmark Execution
For each circuit:
- ✅ Run on LRET (via python/qlret/api.py)
- ✅ Run on Cirq FDM (via cirq_fdm_wrapper.py)
- ✅ Measure: time, memory, fidelity, trace distance
- ✅ Handle timeouts and errors gracefully
- ✅ Save results to CSV

### Phase 4: Statistical Analysis
- ✅ Compute mean/median/std of execution times
- ✅ Calculate speedup factors (LRET vs Cirq)
- ✅ Perform t-tests and Wilcoxon signed-rank tests
- ✅ Compute effect sizes (Cohen's d)
- ✅ Generate LaTeX tables for manuscript

### Phase 5: Visualization
Publication-quality figures:
1. **Time Comparison** - Line plot (qubits vs time)
2. **Memory Usage** - Bar plot (LRET vs Cirq)
3. **Speedup Heatmap** - (qubits × depth)
4. **Fidelity Agreement** - Histogram (should be ~1.0)
5. **Scalability** - Log-log plot

Output formats:
- High-res PNG (300 DPI)
- Vector PDF
- LaTeX-compatible

---

## Expected Timeline

| Phase | Time | What Happens |
|-------|------|--------------|
| Setup | 5-10 min | Install Cirq, create scripts |
| Circuit Generation | 2-5 min | Generate ~100 test circuits |
| Benchmarks | 2-12 hours | Run all comparisons |
| Analysis | 5-10 min | Statistical tests |
| Plotting | 5 min | Generate figures |
| **Total** | **3-13 hours** | Mostly automated |

**Note:** Time depends on circuit complexity and hardware.

---

## Metrics Collected

### Performance Metrics
- **Execution Time** - Wall-clock time per circuit
- **Peak Memory** - Maximum RAM usage
- **Throughput** - Circuits/second

### Correctness Metrics
- **State Fidelity** - |⟨ψ_LRET|ψ_Cirq⟩|²
- **Trace Distance** - ||ρ_LRET - ρ_Cirq||₁
- **Entanglement Entropy** - Agreement on von Neumann entropy

### Statistical Tests
- **T-test** - Are time differences significant?
- **Wilcoxon** - Non-parametric alternative
- **Cohen's d** - Effect size
- **Confidence Intervals** - 95% CI on speedups

---

## Publication Outputs

### Tables (LaTeX)
```latex
\begin{table}[h]
\caption{Performance comparison: LRET vs Cirq FDM}
\begin{tabular}{lccc}
Circuit Type & Qubits & Speedup & Fidelity \\
\hline
Bell & 10 & 2.3× & 0.9999 \\
GHZ & 15 & 1.8× & 0.9998 \\
...
\end{tabular}
\end{table}
```

### Figures (PDF/PNG)
All figures will be publication-ready:
- 300 DPI resolution
- Vector graphics (PDF)
- Proper axis labels, legends
- Color-blind friendly palette
- Nature/Science journal style

### Data Availability
- Raw CSV with all benchmark results
- Circuit definitions (JSON)
- Analysis scripts (reproducible)

---

## Troubleshooting

### Issue 1: Cirq Installation Fails
```bash
# Manual installation
pip3 install --user cirq==1.4.1
python3 -c "import cirq; print(cirq.__version__)"
```

### Issue 2: Memory Errors on Large Circuits
```bash
# Reduce circuit complexity in circuit_generator.py
opencode run "Modify circuit_generator.py to limit max qubits to 15 and depth to 20"
```

### Issue 3: LRET Crashes on Some Circuits
```bash
# Run in debug mode
opencode run "Add error handling and logging to run_comparison.py. Skip failed circuits and log errors."
```

### Issue 4: Fidelity Too Low (<0.99)
```bash
# Investigate discrepancy
opencode run "Analyze circuits with low fidelity. Check for:
1. Numerical precision issues
2. Different gate definitions
3. State normalization
4. Basis ordering"
```

---

## Verification Steps

Before trusting results:

### 1. Sanity Checks
```bash
# Check fidelity on simple circuits
opencode run "Run comparison on single-qubit Hadamard and 2-qubit CNOT. Fidelity should be >0.99999"
```

### 2. Known Results
```bash
# Verify Bell state
opencode run "Generate Bell state and verify both simulators produce |00⟩+|11⟩ with equal amplitudes"
```

### 3. Noise Models
```bash
# Verify noise implementation
opencode run "Compare noise models: apply same depolarizing channel in both simulators, check fidelity decreases similarly"
```

---

## Customization

### Add New Circuit Types
```bash
opencode run "Add W-state circuits to circuit_generator.py:
- Generate |W⟩ = (|100...⟩ + |010...⟩ + ... + |00...1⟩)/√n
- For n=3,4,5,...,10 qubits
- Save to circuits/w_states/"
```

### Change Noise Parameters
```bash
opencode run "Modify run_comparison.py to test these noise levels: [0.0001, 0.001, 0.01, 0.05, 0.1]"
```

### Add GPU Benchmarks
```bash
opencode run "Create run_comparison_gpu.py that also runs LRET with GPU acceleration and compares to Cirq CPU and LRET CPU"
```

---

## Integration with Paper

### Methods Section Template
```markdown
**Benchmarking Procedure**
We compared LRET against Cirq's full density matrix simulator on 
[N] test circuits spanning [M] qubits and depths [D1-D2]. Circuits 
included Bell states, GHZ states, quantum Fourier transforms, and 
random quantum circuits. Each circuit was executed [K] times, and 
we recorded wall-clock time, peak memory usage, and final state 
fidelity between simulators. All benchmarks were performed on 
[hardware specs]. Statistical significance was assessed using 
Wilcoxon signed-rank tests (α=0.05).
```

### Results Section Template
```markdown
**Performance Analysis**
Figure X shows execution time comparison across circuit types. 
LRET demonstrated a [X.X±Y.Y]× speedup over Cirq for [circuit type] 
circuits (n=[N_min]-[N_max] qubits, p<0.001). Memory consumption 
was [percentage]% lower for LRET (Figure Y). State fidelity between 
simulators exceeded 0.999 for all circuits (Figure Z), confirming 
numerical accuracy.
```

---

## Advanced Usage

### Parallel Execution
```bash
# Run multiple circuits in parallel
opencode run "Modify run_comparison.py to use multiprocessing.Pool with 8 workers for parallel benchmark execution"
```

### Custom Metrics
```bash
# Add entanglement entropy comparison
opencode run "Add compute_entanglement_entropy() to both simulators and compare results in analysis"
```

### Interactive Analysis
```bash
# Use OpenCode for on-the-fly analysis
cd cirq_comparison/results
opencode run "Load benchmark_results_*.csv and create a violin plot comparing LRET vs Cirq time distributions per circuit type"
```

---

## Files Generated

After running `./setup_cirq_comparison.sh`:

```
cirq_comparison/
├── cirq_fdm_wrapper.py          # Cirq FDM interface
├── circuit_generator.py         # Test circuit generator
├── run_comparison.py            # Benchmark runner
├── analyze_results.py           # Statistical analysis
├── create_plots.py              # Plotting code
├── run_full_comparison.sh       # Master execution script
├── circuits/                    # Generated test circuits
│   ├── bell_2q.json
│   ├── ghz_10q.json
│   ├── qft_8q.json
│   └── ...
├── results/                     # Benchmark outputs
│   ├── benchmark_results_<timestamp>.csv
│   ├── statistical_analysis_<timestamp>.txt
│   └── tables_for_paper_<timestamp>.tex
└── plots/                       # Publication figures
    ├── figure1_time_comparison.pdf
    ├── figure2_memory_comparison.pdf
    ├── figure3_speedup_heatmap.pdf
    ├── figure4_fidelity_histogram.pdf
    └── figure5_scalability.pdf
```

---

## Support

If issues arise:

1. **Check logs:**
   ```bash
   tail -100 cirq_comparison/setup_log_*.txt
   tail -100 cirq_comparison/full_comparison_*.log
   ```

2. **Ask OpenCode:**
   ```bash
   opencode run "Debug the error in cirq_comparison/results/. What went wrong and how to fix it?"
   ```

3. **Manual intervention:**
   - Review generated Python scripts
   - Modify parameters as needed
   - Re-run specific phases

---

## FAQ

**Q: How long will this take?**
A: Setup: 10 min. Benchmarks: 3-12 hours (depends on circuits). Analysis/plotting: 10 min.

**Q: Can I run this on a cluster?**
A: Yes! Modify `run_comparison.py` to submit jobs to SLURM/PBS. OpenCode can help with this.

**Q: What if LRET is slower than Cirq?**
A: That's still publishable! Analyze why (scaling regime, noise model overhead) and highlight where LRET excels.

**Q: Can I compare to other simulators (Qiskit, ProjectQ)?**
A: Yes! Create similar wrappers for those simulators using OpenCode.

**Q: How do I ensure reproducibility?**
A: Save all scripts, random seeds, and full benchmark CSV. Include hardware specs.

---

## Citation

When publishing:
- Cite Cirq: "Cirq Developers. Cirq. Version X.X. https://github.com/quantumlib/Cirq (2024)"
- Cite your LRET paper
- Include all generated data as supplementary material
- Make scripts available on GitHub

---

**Ready to start?**

```bash
cd /path/to/LRET
./setup_cirq_comparison.sh
```

OpenCode will guide you through the entire process!
