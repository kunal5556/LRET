"""Master script to run complete LRET vs Cirq comparison.

This script orchestrates:
1. Circuit generation
2. Benchmark execution
3. Statistical analysis
4. Plot generation
5. Summary report
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n⚠ ERROR: {description} failed!")
        sys.exit(1)
    
    return result.returncode

def main():
    """Run complete comparison pipeline."""
    comparison_dir = Path(__file__).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"""
╔════════════════════════════════════════════════════════════╗
║   LRET vs Cirq FDM Comparison - Full Pipeline             ║
╚════════════════════════════════════════════════════════════╝

Starting comparison at: {datetime.now()}
Output directory: {comparison_dir}
""")
    
    # Step 1: Generate circuits
    run_command(
        f"python {comparison_dir}/circuit_generator.py "
        f"--output {comparison_dir}/circuits "
        f"--max-qubits 10 "
        f"--noise-levels 0.0 0.001 0.01",
        "Step 1: Generating Test Circuits"
    )
    
    # Step 2: Run benchmarks
    run_command(
        f"python {comparison_dir}/run_comparison.py "
        f"--circuits {comparison_dir}/circuits "
        f"--output {comparison_dir}/results "
        f"--trials 3 "
        f"--timeout 300",
        "Step 2: Running Benchmarks (this may take a while...)"
    )
    
    # Find the results file
    results_dir = comparison_dir / "results"
    result_files = sorted(results_dir.glob("benchmark_results_*.csv"))
    
    if not result_files:
        print("\n⚠ ERROR: No benchmark results found!")
        sys.exit(1)
    
    latest_results = result_files[-1]
    print(f"\nUsing results file: {latest_results}")
    
    # Step 3: Statistical analysis
    run_command(
        f"python {comparison_dir}/analyze_results.py "
        f"--input {latest_results}",
        "Step 3: Statistical Analysis"
    )
    
    # Step 4: Create plots
    run_command(
        f"python {comparison_dir}/create_plots.py "
        f"--input {latest_results} "
        f"--output {comparison_dir}/plots",
        "Step 4: Generating Publication Plots"
    )
    
    # Step 5: Create summary
    print(f"\n{'='*60}")
    print("Step 5: Creating Summary Report")
    print(f"{'='*60}")
    
    summary_lines = []
    summary_lines.append("# LRET vs Cirq FDM Comparison - Summary Report")
    summary_lines.append(f"\n**Date:** {datetime.now()}")
    summary_lines.append(f"**Timestamp:** {timestamp}\n")
    
    summary_lines.append("## Files Generated\n")
    summary_lines.append("### Raw Data")
    summary_lines.append(f"- `{latest_results.name}` - Raw benchmark data\n")
    
    summary_lines.append("### Analysis")
    summary_lines.append("- `statistical_analysis.txt` - Statistical tests")
    summary_lines.append("- `tables_for_paper.tex` - LaTeX tables\n")
    
    summary_lines.append("### Figures (Publication-ready)")
    summary_lines.append("- `figure1_time_comparison.pdf`")
    summary_lines.append("- `figure2_memory_comparison.pdf`")
    summary_lines.append("- `figure3_speedup_heatmap.pdf`")
    summary_lines.append("- `figure4_fidelity_histogram.pdf`")
    summary_lines.append("- `figure5_scalability.pdf`\n")
    
    summary_lines.append("## How to Use in Paper\n")
    summary_lines.append("1. **Methods Section:**")
    summary_lines.append("   - Cite: 'Benchmarks performed using Cirq v1.6 and LRET'")
    summary_lines.append("   - Describe: hardware specs, number of trials, circuit types\n")
    
    summary_lines.append("2. **Results Section:**")
    summary_lines.append("   - Include figures 1-5")
    summary_lines.append("   - Reference statistical analysis\n")
    
    summary_lines.append("3. **Data Availability:**")
    summary_lines.append("   - Upload CSV as supplementary data\n")
    
    summary_lines.append("## Quick Results\n")
    summary_lines.append("See `statistical_analysis.txt` for detailed metrics:")
    summary_lines.append("- Mean speedup factor")
    summary_lines.append("- Memory efficiency")
    summary_lines.append("- State fidelity agreement")
    summary_lines.append("- Statistical significance tests\n")
    
    summary_text = "\n".join(summary_lines)
    
    summary_path = comparison_dir / f"COMPARISON_SUMMARY_{timestamp}.md"
    with open(summary_path, "w") as f:
        f.write(summary_text)
    
    print(summary_text)
    print(f"\n✓ Summary saved to: {summary_path}")
    
    # Final message
    print(f"""
╔════════════════════════════════════════════════════════════╗
║                  COMPARISON COMPLETE!                      ║
╚════════════════════════════════════════════════════════════╝

Results location: {comparison_dir}

Key files:
  • Results: {latest_results.relative_to(comparison_dir)}
  • Analysis: results/statistical_analysis.txt
  • Tables: results/tables_for_paper.tex
  • Plots: plots/*.pdf

Next steps:
  1. Review plots in plots/ directory
  2. Check statistical analysis for significance
  3. Copy LaTeX tables to manuscript
  4. Verify fidelity agreement (should be >0.999)

For questions or issues, refer to CIRQ_COMPARISON_GUIDE.md
""")

if __name__ == "__main__":
    main()
