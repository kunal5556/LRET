#!/bin/bash
# Cirq FDM vs LRET Comparison - Setup and Execution Script
# This script helps OpenCode set up Cirq comparison experiments

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LRET_ROOT="$SCRIPT_DIR"
COMPARISON_DIR="$LRET_ROOT/cirq_comparison"
RESULTS_DIR="$COMPARISON_DIR/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "Cirq FDM vs LRET Comparison Setup"
echo "=========================================="

# Create directories
mkdir -p "$COMPARISON_DIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "$COMPARISON_DIR/circuits"
mkdir -p "$COMPARISON_DIR/benchmarks"
mkdir -p "$COMPARISON_DIR/plots"

cd "$LRET_ROOT"
export PATH=$HOME/.opencode/bin:$PATH

# Check OpenCode
if ! command -v opencode &> /dev/null; then
    echo "ERROR: OpenCode not found. Install with: curl -fsSL https://opencode.ai/install | bash"
    exit 1
fi

# Check API key
if [[ -z "$ANTHROPIC_API_KEY" ]]; then
    echo "WARNING: ANTHROPIC_API_KEY not set. Using free model."
    MODEL_FLAG=""
else
    echo "Using Claude Sonnet 4 for advanced analysis"
    MODEL_FLAG="--model anthropic/claude-sonnet-4-20250514"
fi

echo -e "\n[PHASE 1] Installing Cirq and dependencies..."
opencode run $MODEL_FLAG "
Install Cirq and required dependencies for FDM comparison:

1. Check Python version (need 3.8+)
2. Install Cirq: pip3 install --user cirq cirq-core
3. Install scipy, numpy, matplotlib, pandas for analysis
4. Verify installation: python3 -c 'import cirq; print(cirq.__version__)'
5. Report installation status
" 2>&1 | tee "$COMPARISON_DIR/setup_log_$TIMESTAMP.txt"

echo -e "\n[PHASE 2] Creating Cirq comparison infrastructure..."
opencode run $MODEL_FLAG "
Create a Cirq FDM comparison module at $COMPARISON_DIR/cirq_fdm_wrapper.py:

Requirements:
1. Class CirqFDMSimulator that mimics LRET's interface
2. Methods:
   - simulate(circuit, initial_state=None) → final_state
   - compute_fidelity(state1, state2) → float
   - add_noise(circuit, noise_model) → noisy_circuit
3. Support for:
   - Common gates: H, CNOT, RX, RY, RZ, X, Y, Z
   - Noise models: depolarizing, amplitude_damping, phase_damping
   - Multi-qubit circuits (up to 20 qubits)
4. Use Cirq's DensityMatrixSimulator (FDM)
5. Add timing and memory profiling decorators

Save to: $COMPARISON_DIR/cirq_fdm_wrapper.py
" 2>&1 | tee -a "$COMPARISON_DIR/setup_log_$TIMESTAMP.txt"

echo -e "\n[PHASE 3] Creating test circuit generator..."
opencode run $MODEL_FLAG "
Create a test circuit generator at $COMPARISON_DIR/circuit_generator.py:

Generate parameterized quantum circuits for comparison:

1. Circuit types:
   - Bell states (2-20 qubits)
   - GHZ states (3-20 qubits)
   - Quantum Fourier Transform (4-12 qubits)
   - Random circuits (depth 5, 10, 20, 50)
   - Variational circuits (QAOA, VQE-like)

2. For each circuit type:
   - Generate both noiseless and noisy versions
   - Noise: depolarizing (p=0.001, 0.01, 0.05)
   - Export to JSON (for LRET) and Cirq format

3. Functions:
   - generate_all_circuits() → saves to $COMPARISON_DIR/circuits/
   - circuit_to_json(circuit) → LRET JSON format
   - circuit_to_cirq(circuit) → Cirq Circuit object

Save to: $COMPARISON_DIR/circuit_generator.py
" 2>&1 | tee -a "$COMPARISON_DIR/setup_log_$TIMESTAMP.txt"

echo -e "\n[PHASE 4] Creating benchmark runner..."
opencode run $MODEL_FLAG "
Create a comprehensive benchmark runner at $COMPARISON_DIR/run_comparison.py:

This script should:

1. Load circuits from $COMPARISON_DIR/circuits/
2. For each circuit:
   a) Run on LRET (using python/qlret/api.py)
   b) Run on Cirq FDM (using cirq_fdm_wrapper.py)
   c) Measure:
      - Execution time
      - Memory usage (peak)
      - Final state fidelity (LRET vs Cirq)
      - Trace distance
3. Save results to CSV: $RESULTS_DIR/benchmark_results_$TIMESTAMP.csv
4. Columns: circuit_name, num_qubits, depth, noise_level, lret_time, cirq_time, lret_memory, cirq_memory, fidelity, trace_distance
5. Handle timeouts (max 300s per circuit)
6. Generate progress reports

Usage:
    python3 run_comparison.py --circuits circuits/ --output results/

Save to: $COMPARISON_DIR/run_comparison.py
" 2>&1 | tee -a "$COMPARISON_DIR/setup_log_$TIMESTAMP.txt"

echo -e "\n[PHASE 5] Creating statistical analysis module..."
opencode run $MODEL_FLAG "
Create statistical analysis at $COMPARISON_DIR/analyze_results.py:

Analyze benchmark results for publication:

1. Load results CSV from $RESULTS_DIR/
2. Compute statistics:
   - Mean/median/std of execution times (LRET vs Cirq)
   - Speedup factors per circuit type
   - Memory efficiency ratios
   - Fidelity agreement (should be >0.9999 for correctness)
3. Statistical tests:
   - T-test for time differences
   - Wilcoxon signed-rank test
   - Effect sizes (Cohen's d)
4. Generate tables:
   - Summary statistics table (LaTeX format)
   - Per-circuit-type comparison
5. Output:
   - $RESULTS_DIR/statistical_analysis_$TIMESTAMP.txt
   - $RESULTS_DIR/tables_for_paper_$TIMESTAMP.tex

Save to: $COMPARISON_DIR/analyze_results.py
" 2>&1 | tee -a "$COMPARISON_DIR/setup_log_$TIMESTAMP.txt"

echo -e "\n[PHASE 6] Creating visualization module..."
opencode run $MODEL_FLAG "
Create publication-quality plotting at $COMPARISON_DIR/create_plots.py:

Generate publication-ready figures:

1. Figure 1: Execution time comparison
   - X-axis: Number of qubits
   - Y-axis: Time (log scale)
   - Lines: LRET vs Cirq, by circuit type
   - Error bars: std dev
   
2. Figure 2: Memory usage comparison
   - Bar plot by circuit type
   - LRET vs Cirq side-by-side
   
3. Figure 3: Speedup factor
   - Heatmap: qubits × circuit_depth
   - Color: LRET speedup over Cirq
   
4. Figure 4: Fidelity agreement
   - Histogram of fidelity differences
   - Should show tight peak near 1.0
   
5. Figure 5: Scalability
   - Log-log plot: qubits vs time
   - Show scaling exponents

Output:
- High-res PNG (300 DPI) and PDF
- Style: Nature/Science journal requirements
- Font: Arial 8pt, lines 1.5pt
- Save to: $COMPARISON_DIR/plots/

Save to: $COMPARISON_DIR/create_plots.py
" 2>&1 | tee -a "$COMPARISON_DIR/setup_log_$TIMESTAMP.txt"

echo -e "\n[PHASE 7] Creating master execution script..."
cat > "$COMPARISON_DIR/run_full_comparison.sh" << 'EOF'
#!/bin/bash
# Master script to run full Cirq vs LRET comparison

set -e

COMPARISON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$COMPARISON_DIR/full_comparison_$TIMESTAMP.log"

echo "Starting Cirq vs LRET comparison at $(date)" | tee "$LOG_FILE"

# Step 1: Generate circuits
echo -e "\n=== Generating test circuits ===" | tee -a "$LOG_FILE"
python3 "$COMPARISON_DIR/circuit_generator.py" 2>&1 | tee -a "$LOG_FILE"

# Step 2: Run benchmarks
echo -e "\n=== Running benchmarks (this may take hours) ===" | tee -a "$LOG_FILE"
python3 "$COMPARISON_DIR/run_comparison.py" \
    --circuits "$COMPARISON_DIR/circuits" \
    --output "$COMPARISON_DIR/results" \
    --timestamp "$TIMESTAMP" 2>&1 | tee -a "$LOG_FILE"

# Step 3: Analyze results
echo -e "\n=== Performing statistical analysis ===" | tee -a "$LOG_FILE"
python3 "$COMPARISON_DIR/analyze_results.py" \
    --input "$COMPARISON_DIR/results/benchmark_results_$TIMESTAMP.csv" \
    --output "$COMPARISON_DIR/results" 2>&1 | tee -a "$LOG_FILE"

# Step 4: Create plots
echo -e "\n=== Generating publication plots ===" | tee -a "$LOG_FILE"
python3 "$COMPARISON_DIR/create_plots.py" \
    --input "$COMPARISON_DIR/results/benchmark_results_$TIMESTAMP.csv" \
    --output "$COMPARISON_DIR/plots" 2>&1 | tee -a "$LOG_FILE"

echo -e "\n=== Comparison complete! ===" | tee -a "$LOG_FILE"
echo "Results saved to: $COMPARISON_DIR/results/" | tee -a "$LOG_FILE"
echo "Plots saved to: $COMPARISON_DIR/plots/" | tee -a "$LOG_FILE"
echo "LaTeX tables: $COMPARISON_DIR/results/tables_for_paper_$TIMESTAMP.tex" | tee -a "$LOG_FILE"

# Generate final summary
cat > "$COMPARISON_DIR/COMPARISON_SUMMARY_$TIMESTAMP.md" << SUMMARY
# Cirq FDM vs LRET Comparison Results

**Date:** $(date)
**Timestamp:** $TIMESTAMP

## Files Generated

### Raw Data
- \`results/benchmark_results_$TIMESTAMP.csv\` - Raw benchmark data

### Analysis
- \`results/statistical_analysis_$TIMESTAMP.txt\` - Statistical tests
- \`results/tables_for_paper_$TIMESTAMP.tex\` - LaTeX tables

### Figures (Publication-ready)
- \`plots/figure1_time_comparison.pdf\`
- \`plots/figure2_memory_comparison.pdf\`
- \`plots/figure3_speedup_heatmap.pdf\`
- \`plots/figure4_fidelity_histogram.pdf\`
- \`plots/figure5_scalability.pdf\`

## How to Use in Paper

1. **Methods Section:**
   - Cite: "Benchmarks performed using Cirq v[X.X] and LRET v[Y.Y]"
   - Describe: hardware specs, number of trials, circuit types
   
2. **Results Section:**
   - Include figures 1-5
   - Reference statistical analysis tables
   
3. **Data Availability:**
   - Upload benchmark_results CSV as supplementary data
   
## Next Steps

1. Review plots in \`plots/\` directory
2. Check statistical significance in \`statistical_analysis.txt\`
3. Copy LaTeX tables to manuscript
4. Verify fidelity agreement (should be >0.999)
5. If issues found, re-run specific circuits with:
   \`\`\`bash
   python3 run_comparison.py --circuits circuits/problem_circuit.json
   \`\`\`

## Citation

If LRET shows better performance, emphasize:
- Speedup factors (from Figure 3)
- Memory efficiency (from Figure 2)
- Maintained accuracy (from Figure 4)

If Cirq shows better performance in some regimes:
- Analyze why (depth? qubits? noise?)
- Highlight LRET's strengths in other regimes
- Discuss trade-offs
SUMMARY

cat "$COMPARISON_DIR/COMPARISON_SUMMARY_$TIMESTAMP.md"
EOF

chmod +x "$COMPARISON_DIR/run_full_comparison.sh"

echo -e "\n=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Directory structure created:"
echo "  $COMPARISON_DIR/"
echo "  ├── cirq_fdm_wrapper.py       (Cirq FDM interface)"
echo "  ├── circuit_generator.py      (Test circuit generator)"
echo "  ├── run_comparison.py         (Benchmark runner)"
echo "  ├── analyze_results.py        (Statistical analysis)"
echo "  ├── create_plots.py           (Publication plots)"
echo "  ├── run_full_comparison.sh    (Master script)"
echo "  ├── circuits/                 (Generated test circuits)"
echo "  ├── results/                  (Benchmark results, analysis)"
echo "  └── plots/                    (Publication figures)"
echo ""
echo "Next steps:"
echo "1. Review generated scripts in $COMPARISON_DIR/"
echo "2. Run full comparison:"
echo "   cd $COMPARISON_DIR && ./run_full_comparison.sh"
echo "3. Results will be in results/ and plots/"
echo ""
echo "For targeted comparisons, use:"
echo "  python3 run_comparison.py --help"
echo ""
echo "Setup log saved to: $COMPARISON_DIR/setup_log_$TIMESTAMP.txt"
