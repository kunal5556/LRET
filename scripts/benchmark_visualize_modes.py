#!/usr/bin/env python3
"""
Visualization Module for Parallel Modes Benchmarks

Generates publication-quality plots comparing parallel execution modes:
1. Mode comparison bar chart
2. Speedup heatmap
3. Rank evolution trajectories
4. Time per quantum state
5. Scaling comparison
6. Accuracy validation (future: with state export)
7. Memory comparison
8. Dashboard summary

Usage:
    python scripts/benchmark_visualize_modes.py results/parallel_modes_quick/results.json
    python scripts/benchmark_visualize_modes.py results.json --output plots/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Visualization libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Error: matplotlib required. Install with: pip install matplotlib")
    sys.exit(1)

try:
    import seaborn as sns
    HAS_SEABORN = True
    sns.set_style("whitegrid")
except ImportError:
    HAS_SEABORN = False


class ParallelModesVisualizer:
    """Generate visualizations for parallel modes comparison."""

    def __init__(self, results_file: Path, output_dir: Path):
        self.results_file = results_file
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load results
        with open(results_file) as f:
            raw_results = json.load(f)

        # Filter successful results
        self.results = [r for r in raw_results if r.get("status") == "success"]

        print(f"Loaded {len(raw_results)} results ({len(self.results)} successful)")

        # Color scheme
        self.colors = {
            "sequential": "#95a5a6",  # Gray
            "row": "#3498db",         # Blue
            "column": "#e74c3c",      # Red
            "hybrid": "#2ecc71",      # Green
            "batch": "#f39c12"        # Orange
        }

    def generate_all_plots(self) -> List[Path]:
        """Generate all visualization plots."""
        print("\nGenerating plots...")
        plots = []

        plot_funcs = [
            ("mode_comparison_bar", self.plot_mode_comparison_bar),
            ("speedup_heatmap", self.plot_speedup_heatmap),
            ("rank_evolution", self.plot_rank_evolution_trajectories),
            ("time_per_state", self.plot_time_per_state),
            ("scaling_comparison", self.plot_scaling_comparison),
            ("memory_comparison", self.plot_memory_comparison),
            ("dashboard", self.plot_dashboard),
        ]

        for plot_name, plot_func in plot_funcs:
            print(f"  Generating: {plot_name}.png...")
            try:
                output_path = plot_func()
                if output_path:
                    plots.append(output_path)
            except Exception as e:
                print(f"    Warning: Failed to generate {plot_name}: {e}")

        print(f"\nGenerated {len(plots)} plots in: {self.output_dir}")
        return plots

    def _aggregate_by_config(self) -> Dict:
        """Aggregate results by configuration."""
        by_config = defaultdict(list)
        for r in self.results:
            key = (r["mode"], r["circuit_type"], r["n_qubits"], r["depth"], r["noise_prob"])
            by_config[key].append(r)
        return by_config

    def _compute_stats(self, values: List[float]) -> Tuple[float, float]:
        """Compute mean and std."""
        if not values:
            return 0.0, 0.0
        return float(np.mean(values)), float(np.std(values))

    def plot_mode_comparison_bar(self) -> Optional[Path]:
        """Bar chart comparing modes for a fixed configuration."""
        # Find most common configuration
        by_config = self._aggregate_by_config()

        # Pick first config with all modes
        selected_config = None
        for key, results in by_config.items():
            modes = set(r["mode"] for r in results)
            if len(modes) >= 2:  # At least 2 modes
                selected_config = key
                break

        if not selected_config:
            print("    Skipping: No configuration with multiple modes")
            return None

        mode, circuit_type, n_qubits, depth, noise_prob = selected_config

        # Group by mode
        mode_data = defaultdict(list)
        for r in self.results:
            if (r["circuit_type"] == circuit_type and r["n_qubits"] == n_qubits and
                r["depth"] == depth and abs(r["noise_prob"] - noise_prob) < 1e-9):
                mode_data[r["mode"]].append(r["time_wall_ms"])

        # Sort modes
        modes = sorted(mode_data.keys())
        means = [np.mean(mode_data[m]) for m in modes]
        stds = [np.std(mode_data[m]) for m in modes]

        # Compute speedups
        if "sequential" in modes:
            baseline = means[modes.index("sequential")]
            speedups = [baseline / m for m in means]
        else:
            speedups = [1.0] * len(modes)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(modes))
        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8)

        # Color bars
        for bar, mode in zip(bars, modes):
            bar.set_color(self.colors.get(mode, "#34495e"))

        # Annotate with speedup
        for i, (bar, speedup) in enumerate(zip(bars, speedups)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                   f'{speedup:.2f}×', ha='center', va='bottom',
                   fontweight='bold', fontsize=10)

        ax.set_xlabel('Parallel Mode', fontsize=12)
        ax.set_ylabel('Execution Time (ms)', fontsize=12)
        ax.set_title(f'Mode Comparison ({circuit_type}, {n_qubits} qubits, depth={depth})',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in modes])
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = self.output_dir / "mode_comparison_bar.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def plot_speedup_heatmap(self) -> Optional[Path]:
        """2D heatmap: modes × qubit counts with speedup ratios."""
        # Aggregate by mode and qubits
        by_mode_qubits = defaultdict(lambda: defaultdict(list))
        for r in self.results:
            by_mode_qubits[r["mode"]][r["n_qubits"]].append(r["time_wall_ms"])

        # Get baseline (sequential)
        if "sequential" not in by_mode_qubits:
            print("    Skipping: No sequential baseline for speedup")
            return None

        modes = sorted([m for m in by_mode_qubits.keys() if m != "sequential"])
        qubits = sorted(set(q for mode_data in by_mode_qubits.values() for q in mode_data.keys()))

        # Compute speedup matrix
        speedup_matrix = []
        for mode in modes:
            row = []
            for q in qubits:
                if q in by_mode_qubits["sequential"] and q in by_mode_qubits[mode]:
                    seq_mean = np.mean(by_mode_qubits["sequential"][q])
                    mode_mean = np.mean(by_mode_qubits[mode][q])
                    speedup = seq_mean / mode_mean if mode_mean > 0 else 0
                    row.append(speedup)
                else:
                    row.append(np.nan)
            speedup_matrix.append(row)

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(speedup_matrix, cmap='RdYlGn', aspect='auto',
                      vmin=0.5, vmax=2.0, interpolation='nearest')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Speedup vs Sequential', fontsize=12)

        # Annotate cells
        for i, mode in enumerate(modes):
            for j, q in enumerate(qubits):
                text = ax.text(j, i, f'{speedup_matrix[i][j]:.2f}',
                             ha="center", va="center", color="black", fontsize=10)

        ax.set_xticks(range(len(qubits)))
        ax.set_yticks(range(len(modes)))
        ax.set_xticklabels([f'{q}q' for q in qubits])
        ax.set_yticklabels([m.capitalize() for m in modes])
        ax.set_xlabel('Number of Qubits', fontsize=12)
        ax.set_ylabel('Parallel Mode', fontsize=12)
        ax.set_title('Speedup Heatmap (vs Sequential)', fontsize=14, fontweight='bold')

        plt.tight_layout()
        output_path = self.output_dir / "speedup_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def plot_rank_evolution_trajectories(self) -> Optional[Path]:
        """Line plot: rank evolution over operations for different modes."""
        # Find results with rank evolution data
        with_rank_evo = [r for r in self.results if r.get("rank_evolution")]

        if not with_rank_evo:
            print("    Skipping: No rank evolution data")
            return None

        # Pick a representative configuration
        config_key = (with_rank_evo[0]["circuit_type"], with_rank_evo[0]["n_qubits"])
        config_results = [r for r in with_rank_evo
                         if r["circuit_type"] == config_key[0] and r["n_qubits"] == config_key[1]]

        fig, ax = plt.subplots(figsize=(12, 6))

        for mode in set(r["mode"] for r in config_results):
            mode_results = [r for r in config_results if r["mode"] == mode]
            if mode_results:
                # Average rank evolution across trials
                rank_evos = [r["rank_evolution"] for r in mode_results]
                max_len = max(len(evo) for evo in rank_evos)

                # Pad shorter ones with NaN
                padded = [evo + [np.nan] * (max_len - len(evo)) for evo in rank_evos]
                avg_rank = np.nanmean(padded, axis=0)

                ax.plot(range(len(avg_rank)), avg_rank,
                       label=mode.capitalize(), color=self.colors.get(mode, "#34495e"),
                       linewidth=2, marker='o', markersize=3, markevery=max(1, len(avg_rank)//20))

        ax.set_xlabel('Operation Index', fontsize=12)
        ax.set_ylabel('Rank', fontsize=12)
        ax.set_title(f'Rank Evolution Trajectories ({config_key[0]}, {config_key[1]} qubits)',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / "rank_evolution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def plot_time_per_state(self) -> Optional[Path]:
        """Time per quantum state (μs/state)."""
        # Group by mode and qubits
        by_mode_qubits = defaultdict(lambda: defaultdict(list))
        for r in self.results:
            n_states = 2 ** r["n_qubits"]
            depth = r["depth"]
            time_per_state = (r["time_wall_ms"] * 1000) / (n_states * depth)  # μs/state
            by_mode_qubits[r["mode"]][r["n_qubits"]].append(time_per_state)

        fig, ax = plt.subplots(figsize=(12, 7))

        for mode in sorted(by_mode_qubits.keys()):
            qubits = sorted(by_mode_qubits[mode].keys())
            means = [np.mean(by_mode_qubits[mode][q]) for q in qubits]
            stds = [np.std(by_mode_qubits[mode][q]) for q in qubits]

            ax.errorbar(qubits, means, yerr=stds,
                       label=mode.capitalize(), color=self.colors.get(mode, "#34495e"),
                       marker='o', markersize=8, linewidth=2, capsize=5)

        ax.set_yscale('log')
        ax.set_xlabel('Number of Qubits', fontsize=12)
        ax.set_ylabel('Time per State (μs/state, log scale)', fontsize=12)
        ax.set_title('Efficiency: Time per Quantum State', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which='both')

        plt.tight_layout()
        output_path = self.output_dir / "time_per_state.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def plot_scaling_comparison(self) -> Optional[Path]:
        """Multi-line plot: time vs qubits (log-log scale)."""
        by_mode_qubits = defaultdict(lambda: defaultdict(list))
        for r in self.results:
            by_mode_qubits[r["mode"]][r["n_qubits"]].append(r["time_wall_ms"])

        fig, ax = plt.subplots(figsize=(12, 7))

        for mode in sorted(by_mode_qubits.keys()):
            qubits = sorted(by_mode_qubits[mode].keys())
            means = [np.mean(by_mode_qubits[mode][q]) for q in qubits]
            stds = [np.std(by_mode_qubits[mode][q]) for q in qubits]

            ax.errorbar(qubits, means, yerr=stds,
                       label=mode.capitalize(), color=self.colors.get(mode, "#34495e"),
                       marker='o', markersize=8, linewidth=2, capsize=5)

        ax.set_yscale('log')
        ax.set_xlabel('Number of Qubits', fontsize=12)
        ax.set_ylabel('Execution Time (ms, log scale)', fontsize=12)
        ax.set_title('Scaling Comparison Across Modes', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which='both')

        plt.tight_layout()
        output_path = self.output_dir / "scaling_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def plot_memory_comparison(self) -> Optional[Path]:
        """Grouped bar chart: peak memory by mode and qubit count."""
        # Note: This requires memory tracking in results
        # For now, create placeholder noting memory tracking not implemented
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Memory Comparison\n\n(Requires memory tracking in benchmark results)',
               ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        output_path = self.output_dir / "memory_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def plot_dashboard(self) -> Optional[Path]:
        """Combined 2×3 dashboard with key plots."""
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Mode comparison (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_mode_comp_mini(ax1)

        # 2. Speedup heatmap (top-middle)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_speedup_mini(ax2)

        # 3. Time per state (top-right)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_time_per_state_mini(ax3)

        # 4. Scaling comparison (bottom-left)
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_scaling_mini(ax4)

        # 5. Rank evolution (bottom-middle)
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_rank_evo_mini(ax5)

        # 6. Summary stats (bottom-right)
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_summary_stats(ax6)

        fig.suptitle('Parallel Modes Benchmark Dashboard', fontsize=16, fontweight='bold')

        output_path = self.output_dir / "dashboard.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_mode_comp_mini(self, ax):
        """Mini mode comparison for dashboard."""
        by_config = self._aggregate_by_config()
        if not by_config:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            return

        key = list(by_config.keys())[0]
        mode_data = defaultdict(list)
        for r in self.results:
            mode_data[r["mode"]].append(r["time_wall_ms"])

        modes = sorted(mode_data.keys())[:4]  # Max 4 modes
        means = [np.mean(mode_data[m]) for m in modes]

        bars = ax.bar(range(len(modes)), means, alpha=0.8)
        for bar, mode in zip(bars, modes):
            bar.set_color(self.colors.get(mode, "#34495e"))

        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels([m[:3].capitalize() for m in modes], fontsize=8)
        ax.set_ylabel('Time (ms)', fontsize=9)
        ax.set_title('Mode Comparison', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_speedup_mini(self, ax):
        """Mini speedup heatmap for dashboard."""
        ax.text(0.5, 0.5, 'Speedup Heatmap\n(See full plot)', ha='center', va='center',
               transform=ax.transAxes, fontsize=9)
        ax.set_title('Speedup Analysis', fontsize=10, fontweight='bold')
        ax.axis('off')

    def _plot_time_per_state_mini(self, ax):
        """Mini time per state for dashboard."""
        by_mode = defaultdict(list)
        for r in self.results:
            n_states = 2 ** r["n_qubits"]
            tps = (r["time_wall_ms"] * 1000) / (n_states * r["depth"])
            by_mode[r["mode"]].append(tps)

        modes = sorted(by_mode.keys())[:4]
        means = [np.mean(by_mode[m]) for m in modes]

        bars = ax.barh(range(len(modes)), means, alpha=0.8)
        for bar, mode in zip(bars, modes):
            bar.set_color(self.colors.get(mode, "#34495e"))

        ax.set_yticks(range(len(modes)))
        ax.set_yticklabels([m[:3].capitalize() for m in modes], fontsize=8)
        ax.set_xlabel('μs/state', fontsize=9)
        ax.set_title('Efficiency', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

    def _plot_scaling_mini(self, ax):
        """Mini scaling for dashboard."""
        by_mode_qubits = defaultdict(lambda: defaultdict(list))
        for r in self.results:
            by_mode_qubits[r["mode"]][r["n_qubits"]].append(r["time_wall_ms"])

        for mode in sorted(list(by_mode_qubits.keys())[:4]):
            qubits = sorted(by_mode_qubits[mode].keys())
            means = [np.mean(by_mode_qubits[mode][q]) for q in qubits]
            ax.plot(qubits, means, marker='o', label=mode[:3].capitalize(),
                   color=self.colors.get(mode, "#34495e"), linewidth=1.5)

        ax.set_yscale('log')
        ax.set_xlabel('Qubits', fontsize=9)
        ax.set_ylabel('Time (ms)', fontsize=9)
        ax.set_title('Scaling', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)

    def _plot_rank_evo_mini(self, ax):
        """Mini rank evolution for dashboard."""
        with_rank = [r for r in self.results if r.get("rank_evolution")]
        if not with_rank:
            ax.text(0.5, 0.5, 'No rank data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Rank Evolution', fontsize=10, fontweight='bold')
            return

        for r in with_rank[:3]:  # Max 3 trajectories
            evo = r["rank_evolution"]
            ax.plot(range(len(evo)), evo, alpha=0.7, linewidth=1.5,
                   color=self.colors.get(r["mode"], "#34495e"))

        ax.set_xlabel('Operation', fontsize=9)
        ax.set_ylabel('Rank', fontsize=9)
        ax.set_title('Rank Evolution', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _plot_summary_stats(self, ax):
        """Summary statistics for dashboard."""
        modes = set(r["mode"] for r in self.results)
        total_runs = len(self.results)

        # Compute average speedup for each mode
        by_mode = defaultdict(list)
        seq_times = []
        for r in self.results:
            by_mode[r["mode"]].append(r["time_wall_ms"])
            if r["mode"] == "sequential":
                seq_times.append(r["time_wall_ms"])

        seq_mean = np.mean(seq_times) if seq_times else 1.0

        stats_text = "Summary Statistics\n\n"
        stats_text += f"Total runs: {total_runs}\n"
        stats_text += f"Modes tested: {len(modes)}\n\n"

        stats_text += "Average Speedup:\n"
        for mode in sorted(modes):
            if mode != "sequential":
                mode_mean = np.mean(by_mode[mode])
                speedup = seq_mean / mode_mean if mode_mean > 0 else 0
                stats_text += f"  {mode.capitalize()}: {speedup:.2f}×\n"

        ax.text(0.1, 0.9, stats_text, ha='left', va='top', fontsize=9,
               family='monospace', transform=ax.transAxes)
        ax.set_title('Summary', fontsize=10, fontweight='bold')
        ax.axis('off')


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations for parallel modes benchmark"
    )
    parser.add_argument(
        "results_file",
        type=str,
        help="Path to results.json file"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for plots (default: results_dir/plots)"
    )

    args = parser.parse_args()

    results_file = Path(args.results_file)
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        sys.exit(1)

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = results_file.parent / "plots"

    print("=" * 80)
    print("PARALLEL MODES VISUALIZATION")
    print("=" * 80)
    print(f"Results file: {results_file}")
    print(f"Output directory: {output_dir}")
    print("")

    visualizer = ParallelModesVisualizer(results_file, output_dir)
    plots = visualizer.generate_all_plots()

    print("")
    print("=" * 80)
    print(f"Generated {len(plots)} plots successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
