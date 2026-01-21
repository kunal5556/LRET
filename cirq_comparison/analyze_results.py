"""Statistical Analysis for LRET vs Cirq Benchmarks.

Performs statistical tests, computes effect sizes, and generates LaTeX tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


class BenchmarkAnalyzer:
    """Analyze benchmark results with statistical tests."""

    def __init__(self, results_csv: str):
        """Initialize analyzer.

        Args:
            results_csv: Path to benchmark results CSV
        """
        self.results_csv = Path(results_csv)
        self.df = pd.read_csv(results_csv)
        self.output_dir = self.results_csv.parent

        print(f"Loaded {len(self.df)} benchmark results from {results_csv}")

    def analyze_all(self) -> None:
        """Run complete analysis pipeline."""
        print("\nRunning statistical analysis...")

        # Filter successful runs
        self.df_success = self.df[
            (self.df["lret_success"] == True) & (self.df["cirq_success"] == True)
        ]

        print(f"  Successful comparisons: {len(self.df_success)}/{len(self.df)}")

        # Compute summary statistics
        summary = self._compute_summary_stats()

        # Statistical tests
        tests = self._perform_statistical_tests()

        # Generate report
        self._generate_report(summary, tests)

        # Generate LaTeX tables
        self._generate_latex_tables()

    def _compute_summary_stats(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        df = self.df_success

        summary = {
            "total_circuits": len(df),
            "circuit_types": df["circuit_type"].unique().tolist(),
            "qubit_range": (df["num_qubits"].min(), df["num_qubits"].max()),
            "depth_range": (df["depth"].min(), df["depth"].max()),
            "noise_levels": sorted(df["noise_level"].unique().tolist()),
        }

        # Time comparison
        summary["lret_mean_time_ms"] = df["lret_mean_time_ms"].mean()
        summary["cirq_mean_time_ms"] = df["cirq_mean_time_ms"].mean()
        summary["mean_speedup"] = df["speedup_lret_vs_cirq"].mean()
        summary["median_speedup"] = df["speedup_lret_vs_cirq"].median()
        summary["speedup_std"] = df["speedup_lret_vs_cirq"].std()

        # Memory comparison
        summary["lret_mean_memory_mb"] = df["lret_mean_memory_mb"].mean()
        summary["cirq_mean_memory_mb"] = df["cirq_mean_memory_mb"].mean()
        summary["mean_memory_efficiency"] = df["memory_efficiency_lret_vs_cirq"].mean()

        # Fidelity
        summary["mean_fidelity"] = df["fidelity_lret_cirq"].mean()
        summary["min_fidelity"] = df["fidelity_lret_cirq"].min()
        summary["fidelity_std"] = df["fidelity_lret_cirq"].std()

        # Per circuit type
        summary["per_type"] = {}
        for circuit_type in df["circuit_type"].unique():
            df_type = df[df["circuit_type"] == circuit_type]
            summary["per_type"][circuit_type] = {
                "count": len(df_type),
                "mean_speedup": df_type["speedup_lret_vs_cirq"].mean(),
                "mean_fidelity": df_type["fidelity_lret_cirq"].mean(),
            }

        return summary

    def _perform_statistical_tests(self) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        df = self.df_success

        tests = {}

        # T-test: LRET vs Cirq execution time
        lret_times = df["lret_mean_time_ms"].values
        cirq_times = df["cirq_mean_time_ms"].values

        t_stat, t_pvalue = stats.ttest_rel(lret_times, cirq_times)
        tests["ttest_time"] = {
            "statistic": float(t_stat),
            "pvalue": float(t_pvalue),
            "significant": t_pvalue < 0.05,
        }

        # Wilcoxon signed-rank test (non-parametric)
        w_stat, w_pvalue = stats.wilcoxon(lret_times, cirq_times)
        tests["wilcoxon_time"] = {
            "statistic": float(w_stat),
            "pvalue": float(w_pvalue),
            "significant": w_pvalue < 0.05,
        }

        # Cohen's d effect size
        mean_diff = np.mean(cirq_times - lret_times)
        pooled_std = np.sqrt(
            (np.var(lret_times) + np.var(cirq_times)) / 2
        )
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        tests["cohens_d"] = float(cohens_d)

        # Correlation: qubits vs speedup
        corr, corr_pvalue = stats.pearsonr(
            df["num_qubits"],
            df["speedup_lret_vs_cirq"],
        )
        tests["qubit_speedup_correlation"] = {
            "correlation": float(corr),
            "pvalue": float(corr_pvalue),
        }

        return tests

    def _generate_report(
        self,
        summary: Dict[str, Any],
        tests: Dict[str, Any],
    ) -> None:
        """Generate text report."""
        report_lines = []
        report_lines.append("="*60)
        report_lines.append("LRET vs Cirq FDM Comparison - Statistical Analysis")
        report_lines.append("="*60)
        report_lines.append("")

        # Summary
        report_lines.append("SUMMARY STATISTICS")
        report_lines.append("-"*60)
        report_lines.append(f"Total circuits analyzed: {summary['total_circuits']}")
        report_lines.append(f"Circuit types: {', '.join(summary['circuit_types'])}")
        report_lines.append(
            f"Qubit range: {summary['qubit_range'][0]}-{summary['qubit_range'][1]}"
        )
        report_lines.append(
            f"Depth range: {summary['depth_range'][0]}-{summary['depth_range'][1]}"
        )
        report_lines.append(f"Noise levels: {summary['noise_levels']}")
        report_lines.append("")

        # Performance
        report_lines.append("PERFORMANCE COMPARISON")
        report_lines.append("-"*60)
        report_lines.append(f"LRET mean time: {summary['lret_mean_time_ms']:.2f} ms")
        report_lines.append(f"Cirq mean time: {summary['cirq_mean_time_ms']:.2f} ms")
        report_lines.append(
            f"Mean speedup (LRET vs Cirq): {summary['mean_speedup']:.2f}x "
            f"(±{summary['speedup_std']:.2f})"
        )
        report_lines.append(f"Median speedup: {summary['median_speedup']:.2f}x")
        report_lines.append("")

        report_lines.append(f"LRET mean memory: {summary['lret_mean_memory_mb']:.2f} MB")
        report_lines.append(f"Cirq mean memory: {summary['cirq_mean_memory_mb']:.2f} MB")
        report_lines.append(
            f"Memory efficiency: {summary['mean_memory_efficiency']:.2f}x"
        )
        report_lines.append("")

        # Fidelity
        report_lines.append("CORRECTNESS METRICS")
        report_lines.append("-"*60)
        report_lines.append(f"Mean fidelity: {summary['mean_fidelity']:.6f}")
        report_lines.append(f"Min fidelity: {summary['min_fidelity']:.6f}")
        report_lines.append(f"Fidelity std: {summary['fidelity_std']:.6f}")
        report_lines.append("")

        # Per type
        report_lines.append("PER CIRCUIT TYPE")
        report_lines.append("-"*60)
        for circuit_type, stats_dict in summary["per_type"].items():
            report_lines.append(
                f"{circuit_type:10s}: "
                f"{stats_dict['count']:3d} circuits, "
                f"speedup={stats_dict['mean_speedup']:5.2f}x, "
                f"fidelity={stats_dict['mean_fidelity']:.6f}"
            )
        report_lines.append("")

        # Statistical tests
        report_lines.append("STATISTICAL TESTS")
        report_lines.append("-"*60)
        report_lines.append(
            f"T-test (time): t={tests['ttest_time']['statistic']:.3f}, "
            f"p={tests['ttest_time']['pvalue']:.6f} "
            f"({'significant' if tests['ttest_time']['significant'] else 'not significant'})"
        )
        report_lines.append(
            f"Wilcoxon test: W={tests['wilcoxon_time']['statistic']:.1f}, "
            f"p={tests['wilcoxon_time']['pvalue']:.6f} "
            f"({'significant' if tests['wilcoxon_time']['significant'] else 'not significant'})"
        )
        report_lines.append(f"Cohen's d (effect size): {tests['cohens_d']:.3f}")
        report_lines.append(
            f"Qubit-speedup correlation: r={tests['qubit_speedup_correlation']['correlation']:.3f}, "
            f"p={tests['qubit_speedup_correlation']['pvalue']:.6f}"
        )
        report_lines.append("")
        report_lines.append("="*60)

        # Save report
        report_text = "\n".join(report_lines)
        print(report_text)

        output_path = self.output_dir / "statistical_analysis.txt"
        with open(output_path, "w") as f:
            f.write(report_text)

        print(f"\n✓ Report saved to: {output_path}")

    def _generate_latex_tables(self) -> None:
        """Generate LaTeX tables for manuscript."""
        df = self.df_success

        # Table 1: Overall comparison
        latex_lines = []
        latex_lines.append("% Table: LRET vs Cirq FDM Performance Comparison")
        latex_lines.append("\\begin{table}[h]")
        latex_lines.append("\\centering")
        latex_lines.append("\\caption{Performance comparison: LRET vs Cirq FDM}")
        latex_lines.append("\\begin{tabular}{lcccc}")
        latex_lines.append("\\hline")
        latex_lines.append(
            "Circuit Type & N Circuits & Speedup & Memory Efficiency & Fidelity \\\\"
        )
        latex_lines.append("\\hline")

        for circuit_type in sorted(df["circuit_type"].unique()):
            df_type = df[df["circuit_type"] == circuit_type]
            n = len(df_type)
            speedup = df_type["speedup_lret_vs_cirq"].mean()
            mem_eff = df_type["memory_efficiency_lret_vs_cirq"].mean()
            fidelity = df_type["fidelity_lret_cirq"].mean()

            latex_lines.append(
                f"{circuit_type.upper():10s} & {n:3d} & "
                f"{speedup:.2f}$\\times$ & {mem_eff:.2f}$\\times$ & "
                f"{fidelity:.4f} \\\\"
            )

        latex_lines.append("\\hline")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table}")
        latex_lines.append("")

        # Save LaTeX
        latex_text = "\n".join(latex_lines)
        output_path = self.output_dir / "tables_for_paper.tex"

        with open(output_path, "w") as f:
            f.write(latex_text)

        print(f"✓ LaTeX tables saved to: {output_path}")


def main():
    """Run analysis from command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze LRET vs Cirq benchmark results"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to benchmark results CSV",
    )

    args = parser.parse_args()

    analyzer = BenchmarkAnalyzer(args.input)
    analyzer.analyze_all()


if __name__ == "__main__":
    main()
