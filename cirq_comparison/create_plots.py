"""Create Publication-Quality Plots for LRET vs Cirq Comparison.

Generates figures:
1. Execution time comparison (line plot)
2. Memory usage comparison (bar plot)
3. Speedup heatmap (qubits × depth)
4. Fidelity histogram
5. Scalability analysis (log-log plot)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set publication style
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.figsize": (7, 5),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,  # TrueType fonts for compatibility
})


class BenchmarkPlotter:
    """Create publication-quality plots from benchmark results."""

    def __init__(self, results_csv: str, output_dir: str = "plots"):
        """Initialize plotter.

        Args:
            results_csv: Path to benchmark results CSV
            output_dir: Directory to save plots
        """
        self.df = pd.read_csv(results_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Filter successful runs
        self.df_success = self.df[
            (self.df["lret_success"] == True) & (self.df["cirq_success"] == True)
        ]

        print(f"Loaded {len(self.df_success)} successful comparisons for plotting")

    def create_all_plots(self) -> None:
        """Create all publication figures."""
        print("\nGenerating publication figures...")

        self.plot_execution_time_comparison()
        self.plot_memory_comparison()
        self.plot_speedup_heatmap()
        self.plot_fidelity_histogram()
        self.plot_scalability()

        print(f"\n✓ All plots saved to: {self.output_dir}")

    def plot_execution_time_comparison(self) -> None:
        """Figure 1: Execution time vs qubits."""
        fig, ax = plt.subplots(figsize=(8, 5))

        df = self.df_success

        # Plot by circuit type
        for circuit_type in sorted(df["circuit_type"].unique()):
            df_type = df[df["circuit_type"] == circuit_type]

            # Group by qubits
            grouped = df_type.groupby("num_qubits").agg({
                "lret_mean_time_ms": ["mean", "std"],
                "cirq_mean_time_ms": ["mean", "std"],
            })

            qubits = grouped.index.values

            # LRET line
            lret_mean = grouped[("lret_mean_time_ms", "mean")].values
            lret_std = grouped[("lret_mean_time_ms", "std")].values
            ax.plot(
                qubits, lret_mean,
                marker="o", label=f"LRET ({circuit_type})",
                linewidth=2,
            )
            ax.fill_between(
                qubits,
                lret_mean - lret_std,
                lret_mean + lret_std,
                alpha=0.2,
            )

            # Cirq line
            cirq_mean = grouped[("cirq_mean_time_ms", "mean")].values
            cirq_std = grouped[("cirq_mean_time_ms", "std")].values
            ax.plot(
                qubits, cirq_mean,
                marker="s", linestyle="--",
                label=f"Cirq FDM ({circuit_type})",
                linewidth=2,
            )
            ax.fill_between(
                qubits,
                cirq_mean - cirq_std,
                cirq_mean + cirq_std,
                alpha=0.2,
            )

        ax.set_xlabel("Number of Qubits")
        ax.set_ylabel("Execution Time (ms)")
        ax.set_title("LRET vs Cirq FDM: Execution Time Comparison")
        ax.set_yscale("log")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        # Save
        plt.tight_layout()
        plt.savefig(self.output_dir / "figure1_time_comparison.pdf")
        plt.savefig(self.output_dir / "figure1_time_comparison.png")
        plt.close()

        print("  ✓ Figure 1: Time comparison")

    def plot_memory_comparison(self) -> None:
        """Figure 2: Memory usage bar plot."""
        fig, ax = plt.subplots(figsize=(10, 5))

        df = self.df_success

        # Group by circuit type
        grouped = df.groupby("circuit_type").agg({
            "lret_mean_memory_mb": "mean",
            "cirq_mean_memory_mb": "mean",
        })

        circuit_types = grouped.index.values
        x = np.arange(len(circuit_types))
        width = 0.35

        lret_memory = grouped["lret_mean_memory_mb"].values
        cirq_memory = grouped["cirq_mean_memory_mb"].values

        ax.bar(
            x - width/2, lret_memory,
            width, label="LRET", color="steelblue",
        )
        ax.bar(
            x + width/2, cirq_memory,
            width, label="Cirq FDM", color="coral",
        )

        ax.set_xlabel("Circuit Type")
        ax.set_ylabel("Memory Usage (MB)")
        ax.set_title("LRET vs Cirq FDM: Memory Efficiency")
        ax.set_xticks(x)
        ax.set_xticklabels([ct.upper() for ct in circuit_types])
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "figure2_memory_comparison.pdf")
        plt.savefig(self.output_dir / "figure2_memory_comparison.png")
        plt.close()

        print("  ✓ Figure 2: Memory comparison")

    def plot_speedup_heatmap(self) -> None:
        """Figure 3: Speedup heatmap (qubits × depth)."""
        fig, ax = plt.subplots(figsize=(9, 6))

        df = self.df_success

        # Create pivot table
        pivot = df.pivot_table(
            values="speedup_lret_vs_cirq",
            index="depth",
            columns="num_qubits",
            aggfunc="mean",
        )

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            center=1.0,
            cbar_kws={"label": "Speedup (LRET vs Cirq)"},
            ax=ax,
        )

        ax.set_xlabel("Number of Qubits")
        ax.set_ylabel("Circuit Depth")
        ax.set_title("LRET Speedup Factor (>1 means LRET faster)")

        plt.tight_layout()
        plt.savefig(self.output_dir / "figure3_speedup_heatmap.pdf")
        plt.savefig(self.output_dir / "figure3_speedup_heatmap.png")
        plt.close()

        print("  ✓ Figure 3: Speedup heatmap")

    def plot_fidelity_histogram(self) -> None:
        """Figure 4: Fidelity distribution."""
        fig, ax = plt.subplots(figsize=(7, 5))

        df = self.df_success

        fidelities = df["fidelity_lret_cirq"].dropna()

        ax.hist(
            fidelities,
            bins=50,
            color="steelblue",
            edgecolor="black",
            alpha=0.7,
        )

        ax.axvline(
            fidelities.mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {fidelities.mean():.6f}",
        )

        ax.set_xlabel("Fidelity (LRET vs Cirq)")
        ax.set_ylabel("Count")
        ax.set_title("State Fidelity Agreement Between LRET and Cirq")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "figure4_fidelity_histogram.pdf")
        plt.savefig(self.output_dir / "figure4_fidelity_histogram.png")
        plt.close()

        print("  ✓ Figure 4: Fidelity histogram")

    def plot_scalability(self) -> None:
        """Figure 5: Scalability analysis (log-log)."""
        fig, ax = plt.subplots(figsize=(8, 6))

        df = self.df_success

        # Group by qubits
        grouped = df.groupby("num_qubits").agg({
            "lret_mean_time_ms": "mean",
            "cirq_mean_time_ms": "mean",
        })

        qubits = grouped.index.values
        lret_times = grouped["lret_mean_time_ms"].values
        cirq_times = grouped["cirq_mean_time_ms"].values

        # Plot
        ax.loglog(
            qubits, lret_times,
            marker="o", markersize=8,
            label="LRET", linewidth=2,
        )
        ax.loglog(
            qubits, cirq_times,
            marker="s", markersize=8,
            label="Cirq FDM", linewidth=2,
        )

        # Fit power laws
        if len(qubits) >= 3:
            lret_fit = np.polyfit(np.log(qubits), np.log(lret_times), 1)
            cirq_fit = np.polyfit(np.log(qubits), np.log(cirq_times), 1)

            ax.plot(
                qubits,
                np.exp(lret_fit[1]) * qubits ** lret_fit[0],
                "--", alpha=0.5,
                label=f"LRET fit: O(n^{lret_fit[0]:.2f})",
            )
            ax.plot(
                qubits,
                np.exp(cirq_fit[1]) * qubits ** cirq_fit[0],
                "--", alpha=0.5,
                label=f"Cirq fit: O(n^{cirq_fit[0]:.2f})",
            )

        ax.set_xlabel("Number of Qubits (log scale)")
        ax.set_ylabel("Execution Time (ms, log scale)")
        ax.set_title("Scalability: LRET vs Cirq FDM")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "figure5_scalability.pdf")
        plt.savefig(self.output_dir / "figure5_scalability.png")
        plt.close()

        print("  ✓ Figure 5: Scalability")


def main():
    """Create plots from command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create plots from LRET vs Cirq benchmark results"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to benchmark results CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots",
        help="Output directory for plots",
    )

    args = parser.parse_args()

    plotter = BenchmarkPlotter(args.input, args.output)
    plotter.create_all_plots()


if __name__ == "__main__":
    main()
