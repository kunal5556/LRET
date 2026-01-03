#!/usr/bin/env python3
"""Visualization module for LRET benchmark results.

Generates publication-quality plots for:
- Scaling analysis (time vs qubit count)
- Parallel speedup comparison
- Accuracy/fidelity validation
- Depth scaling analysis
- Memory profiling

Usage:
    python scripts/benchmark_visualize.py benchmark_results.csv
    python scripts/benchmark_visualize.py results.csv --output plots/
    python scripts/benchmark_visualize.py results.csv --format svg --dpi 300

Examples:
    # Generate all plots with default settings
    python scripts/benchmark_visualize.py benchmark_output/benchmark_results.csv

    # Custom output directory and format
    python scripts/benchmark_visualize.py results.csv --output figures/ --format pdf

    # High-resolution for publication
    python scripts/benchmark_visualize.py results.csv --dpi 300 --format svg

Author: LRET Development Team
Date: January 2026
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

# Try to import visualization libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server/CI use
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Error: matplotlib is required for visualization")
    print("Install with: pip install matplotlib")
    sys.exit(1)

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not available. Using basic matplotlib styles.")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: numpy not available. Some features will be limited.")


@dataclass
class PlotConfig:
    """Configuration for plot generation.
    
    Attributes:
        figsize: Figure size as (width, height) in inches
        dpi: Dots per inch for raster outputs
        format: Output format (png, svg, pdf)
        style: Matplotlib/seaborn style
        palette: Color palette name
        title_fontsize: Font size for titles
        label_fontsize: Font size for axis labels
        tick_fontsize: Font size for tick labels
        legend_fontsize: Font size for legend
        grid_alpha: Grid transparency
        error_capsize: Size of error bar caps
        save_individual: Whether to save individual plots
        save_combined: Whether to save combined summary plot
    """
    figsize: Tuple[float, float] = (10, 6)
    dpi: int = 150
    format: str = "png"
    style: str = "whitegrid"
    palette: str = "husl"
    title_fontsize: int = 14
    label_fontsize: int = 12
    tick_fontsize: int = 10
    legend_fontsize: int = 10
    grid_alpha: float = 0.3
    error_capsize: int = 5
    save_individual: bool = True
    save_combined: bool = True


class BenchmarkVisualizer:
    """Generate visualization plots from benchmark data.
    
    This class creates various plots for analyzing benchmark performance,
    including scaling curves, speedup bar charts, and accuracy validation plots.
    """
    
    # Color schemes for different plot types
    COLORS = {
        "scaling": "#2ecc71",       # Green
        "parallel": ["#95a5a6", "#3498db", "#e74c3c", "#2ecc71"],  # Gray, Blue, Red, Green
        "accuracy": "#9b59b6",      # Purple
        "depth": "#f39c12",         # Orange
        "memory": "#1abc9c",        # Teal
        "baseline": "#e74c3c",      # Red (for threshold lines)
        "fit_line": "#34495e",      # Dark gray
    }
    
    # Mode order for parallel plots
    MODE_ORDER = ["sequential", "row", "column", "hybrid"]
    
    def __init__(
        self, 
        csv_path: Path, 
        output_dir: Path,
        config: Optional[PlotConfig] = None
    ):
        """Initialize visualizer.
        
        Args:
            csv_path: Path to benchmark results CSV
            output_dir: Directory for output plots
            config: Plot configuration (uses defaults if None)
        """
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or PlotConfig()
        
        # Load data
        self.data = self._load_data()
        
        # Apply styles
        self._setup_style()
    
    def _load_data(self) -> List[dict]:
        """Load and parse CSV data."""
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data = list(reader)
        
        # Convert numeric fields
        for row in data:
            for key in ["n_qubits", "depth", "trial", "final_rank"]:
                if key in row and row[key]:
                    try:
                        row[key] = int(row[key])
                    except ValueError:
                        row[key] = -1
            
            for key in ["time_ms", "memory_mb", "fidelity", "trace_distance", 
                        "noise_level", "reported_time_ms"]:
                if key in row and row[key] and row[key] != "None":
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        row[key] = None
                else:
                    row[key] = None
        
        return data
    
    def _setup_style(self):
        """Configure matplotlib/seaborn styles."""
        if HAS_SEABORN:
            sns.set_style(self.config.style)
            sns.set_palette(self.config.palette)
        
        plt.rcParams.update({
            'figure.figsize': self.config.figsize,
            'figure.dpi': self.config.dpi,
            'font.size': self.config.label_fontsize,
            'axes.titlesize': self.config.title_fontsize,
            'axes.labelsize': self.config.label_fontsize,
            'xtick.labelsize': self.config.tick_fontsize,
            'ytick.labelsize': self.config.tick_fontsize,
            'legend.fontsize': self.config.legend_fontsize,
            'figure.autolayout': True,
        })
    
    def get_category_data(self, category: str) -> List[dict]:
        """Filter data by category."""
        return [r for r in self.data if r.get("category") == category]
    
    def get_successful_data(self, category: str) -> List[dict]:
        """Get successful (non-error) results for a category."""
        return [
            r for r in self.get_category_data(category)
            if r.get("time_ms") is not None and r["time_ms"] > 0
        ]
    
    def _save_figure(self, fig: plt.Figure, name: str) -> Path:
        """Save figure to file.
        
        Args:
            fig: Matplotlib figure
            name: Base filename (without extension)
            
        Returns:
            Path to saved file
        """
        filename = f"{name}.{self.config.format}"
        output_path = self.output_dir / filename
        
        fig.savefig(
            output_path, 
            dpi=self.config.dpi, 
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none',
        )
        plt.close(fig)
        
        return output_path
    
    @staticmethod
    def _compute_stats(values: List[float]) -> Tuple[float, float]:
        """Compute mean and standard deviation."""
        if not values:
            return 0.0, 0.0
        
        if HAS_NUMPY:
            arr = np.array(values)
            return float(np.mean(arr)), float(np.std(arr))
        else:
            n = len(values)
            mean = sum(values) / n
            if n > 1:
                variance = sum((x - mean) ** 2 for x in values) / (n - 1)
                std = variance ** 0.5
            else:
                std = 0.0
            return mean, std
    
    # -------------------------------------------------------------------------
    # Scaling Plots
    # -------------------------------------------------------------------------
    
    def plot_scaling(self) -> Optional[Path]:
        """Plot time vs qubit count with linear and log scales.
        
        Creates a two-panel figure showing execution time scaling
        on both linear and logarithmic y-axes.
        
        Returns:
            Path to saved plot, or None if no data
        """
        data = self.get_successful_data("scaling")
        if not data:
            print("  No scaling data to plot")
            return None
        
        # Group by n_qubits
        by_n: Dict[int, List[float]] = {}
        for row in data:
            n = row["n_qubits"]
            if n not in by_n:
                by_n[n] = []
            by_n[n].append(row["time_ms"])
        
        # Compute statistics
        ns = sorted(by_n.keys())
        means = []
        stds = []
        for n in ns:
            m, s = self._compute_stats(by_n[n])
            means.append(m)
            stds.append(s)
        
        # Create two-panel figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        color = self.COLORS["scaling"]
        
        # Linear scale plot
        ax1.errorbar(
            ns, means, yerr=stds, 
            fmt='o-', capsize=self.config.error_capsize,
            color=color, markersize=8, linewidth=2,
            label='LRET Execution Time'
        )
        ax1.set_xlabel('Number of Qubits (n)')
        ax1.set_ylabel('Execution Time (ms)')
        ax1.set_title('Qubit Scaling: Linear Scale')
        ax1.grid(True, alpha=self.config.grid_alpha)
        ax1.legend(loc='upper left')
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Annotate with approximate complexity
        if len(ns) >= 2:
            ratio = means[-1] / means[0]
            qubit_diff = ns[-1] - ns[0]
            approx_base = ratio ** (1.0 / qubit_diff)
            ax1.text(
                0.95, 0.05, f'~{approx_base:.2f}× per qubit',
                transform=ax1.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
        
        # Log scale plot
        ax2.errorbar(
            ns, means, yerr=stds,
            fmt='o-', capsize=self.config.error_capsize,
            color=color, markersize=8, linewidth=2,
            label='LRET Execution Time'
        )
        ax2.set_yscale('log')
        ax2.set_xlabel('Number of Qubits (n)')
        ax2.set_ylabel('Execution Time (ms, log scale)')
        ax2.set_title('Qubit Scaling: Log Scale')
        ax2.grid(True, alpha=self.config.grid_alpha, which='both')
        ax2.legend(loc='upper left')
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add exponential fit line if possible
        if HAS_NUMPY and len(ns) >= 3:
            try:
                # Fit: log(T) = log(a) + b*n
                log_means = np.log(means)
                coeffs = np.polyfit(ns, log_means, 1)
                b, log_a = coeffs
                
                # Plot fit line
                n_fit = np.linspace(min(ns), max(ns), 100)
                t_fit = np.exp(log_a) * np.exp(b * n_fit)
                ax2.plot(
                    n_fit, t_fit, '--', 
                    color=self.COLORS["fit_line"], linewidth=1.5,
                    label=f'Fit: T ∝ e^{{{b:.3f}n}}'
                )
                ax2.legend(loc='upper left')
            except Exception:
                pass
        
        plt.tight_layout()
        output_path = self._save_figure(fig, "scaling_time")
        print(f"  Saved: {output_path}")
        return output_path
    
    def plot_scaling_rank(self) -> Optional[Path]:
        """Plot final rank vs qubit count.
        
        Shows how the density matrix rank grows with system size.
        
        Returns:
            Path to saved plot, or None if no data
        """
        data = self.get_successful_data("scaling")
        if not data:
            return None
        
        # Check for rank data
        data_with_rank = [r for r in data if r.get("final_rank", -1) > 0]
        if not data_with_rank:
            return None
        
        # Group by n_qubits
        by_n: Dict[int, List[int]] = {}
        for row in data_with_rank:
            n = row["n_qubits"]
            if n not in by_n:
                by_n[n] = []
            by_n[n].append(row["final_rank"])
        
        ns = sorted(by_n.keys())
        means = []
        stds = []
        for n in ns:
            ranks = [float(r) for r in by_n[n]]
            m, s = self._compute_stats(ranks)
            means.append(m)
            stds.append(s)
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        ax.errorbar(
            ns, means, yerr=stds,
            fmt='s-', capsize=self.config.error_capsize,
            color='#3498db', markersize=8, linewidth=2,
            label='Final Rank'
        )
        
        ax.set_xlabel('Number of Qubits (n)')
        ax.set_ylabel('Final Density Matrix Rank')
        ax.set_title('Rank Scaling with Qubit Count')
        ax.grid(True, alpha=self.config.grid_alpha)
        ax.legend(loc='upper left')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        output_path = self._save_figure(fig, "scaling_rank")
        print(f"  Saved: {output_path}")
        return output_path
    
    # -------------------------------------------------------------------------
    # Parallel Speedup Plots
    # -------------------------------------------------------------------------
    
    def plot_parallel_speedup(self) -> Optional[Path]:
        """Plot parallel mode speedup comparison.
        
        Creates a bar chart comparing speedup across parallelization modes.
        
        Returns:
            Path to saved plot, or None if no data
        """
        data = self.get_successful_data("parallel")
        if not data:
            print("  No parallel data to plot")
            return None
        
        # Group by mode
        by_mode: Dict[str, List[float]] = {}
        for row in data:
            mode = row["mode"]
            if mode not in by_mode:
                by_mode[mode] = []
            by_mode[mode].append(row["time_ms"])
        
        # Need sequential baseline
        if "sequential" not in by_mode:
            print("  Warning: No sequential baseline for speedup plot")
            return None
        
        seq_mean, _ = self._compute_stats(by_mode["sequential"])
        
        # Compute speedups in order
        modes = [m for m in self.MODE_ORDER if m in by_mode]
        speedups = []
        speedup_stds = []
        times = []
        
        for mode in modes:
            m, s = self._compute_stats(by_mode[mode])
            times.append(m)
            speedup = seq_mean / m if m > 0 else 0
            speedups.append(speedup)
            
            # Propagate error for speedup
            rel_err = s / m if m > 0 else 0
            speedup_stds.append(speedup * rel_err)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        x = range(len(modes))
        colors = [self.COLORS["parallel"][i % len(self.COLORS["parallel"])] 
                  for i in range(len(modes))]
        
        bars = ax.bar(
            x, speedups, 
            yerr=speedup_stds, capsize=self.config.error_capsize,
            color=colors, alpha=0.8, edgecolor='black', linewidth=1
        )
        
        # Add baseline reference line
        ax.axhline(
            1.0, color=self.COLORS["baseline"], linestyle='--', 
            linewidth=2, label='Sequential Baseline'
        )
        
        # Add speedup values on bars
        for bar, speedup in zip(bars, speedups):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2., height + 0.1,
                f'{speedup:.2f}x',
                ha='center', va='bottom', fontsize=11, fontweight='bold'
            )
        
        ax.set_xlabel('Parallelization Mode')
        ax.set_ylabel('Speedup (vs Sequential)')
        ax.set_title('Parallel Mode Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in modes])
        ax.grid(True, alpha=self.config.grid_alpha, axis='y')
        ax.legend(loc='upper left')
        ax.set_ylim(0, max(speedups) * 1.3)
        
        plt.tight_layout()
        output_path = self._save_figure(fig, "parallel_speedup")
        print(f"  Saved: {output_path}")
        return output_path
    
    def plot_parallel_times(self) -> Optional[Path]:
        """Plot raw execution times by parallel mode.
        
        Returns:
            Path to saved plot, or None if no data
        """
        data = self.get_successful_data("parallel")
        if not data:
            return None
        
        # Group by mode
        by_mode: Dict[str, List[float]] = {}
        for row in data:
            mode = row["mode"]
            if mode not in by_mode:
                by_mode[mode] = []
            by_mode[mode].append(row["time_ms"])
        
        modes = [m for m in self.MODE_ORDER if m in by_mode]
        means = []
        stds = []
        
        for mode in modes:
            m, s = self._compute_stats(by_mode[mode])
            means.append(m)
            stds.append(s)
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        x = range(len(modes))
        colors = [self.COLORS["parallel"][i % len(self.COLORS["parallel"])] 
                  for i in range(len(modes))]
        
        ax.bar(
            x, means,
            yerr=stds, capsize=self.config.error_capsize,
            color=colors, alpha=0.8, edgecolor='black', linewidth=1
        )
        
        ax.set_xlabel('Parallelization Mode')
        ax.set_ylabel('Execution Time (ms)')
        ax.set_title('Execution Time by Parallel Mode')
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in modes])
        ax.grid(True, alpha=self.config.grid_alpha, axis='y')
        
        plt.tight_layout()
        output_path = self._save_figure(fig, "parallel_times")
        print(f"  Saved: {output_path}")
        return output_path
    
    # -------------------------------------------------------------------------
    # Accuracy Plots
    # -------------------------------------------------------------------------
    
    def plot_accuracy_fidelity(self) -> Optional[Path]:
        """Plot fidelity validation results.
        
        Shows LRET fidelity across qubit counts with threshold line.
        
        Returns:
            Path to saved plot, or None if no data
        """
        data = self.get_successful_data("accuracy")
        if not data:
            print("  No accuracy data to plot")
            return None
        
        # Group by n_qubits
        by_n: Dict[int, List[float]] = {}
        for row in data:
            fid = row.get("fidelity")
            if fid is not None and fid > 0:
                n = row["n_qubits"]
                if n not in by_n:
                    by_n[n] = []
                by_n[n].append(fid)
        
        if not by_n:
            print("  No fidelity data to plot")
            return None
        
        ns = sorted(by_n.keys())
        means = []
        stds = []
        for n in ns:
            m, s = self._compute_stats(by_n[n])
            means.append(m)
            stds.append(s)
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        ax.errorbar(
            ns, means, yerr=stds,
            fmt='o-', capsize=self.config.error_capsize,
            color=self.COLORS["accuracy"], markersize=10, linewidth=2,
            label='LRET Fidelity'
        )
        
        # Add threshold line
        ax.axhline(
            0.999, color=self.COLORS["baseline"], linestyle='--',
            linewidth=2, label='Threshold (0.999)'
        )
        
        # Mark passing/failing points
        for n, fids in by_n.items():
            for fid in fids:
                if fid < 0.999:
                    ax.scatter(
                        [n], [fid], color='red', marker='x', 
                        s=100, zorder=5, linewidth=2
                    )
        
        ax.set_xlabel('Number of Qubits (n)')
        ax.set_ylabel('Fidelity')
        ax.set_title('LRET vs FDM Fidelity Validation')
        
        # Set y-axis limits to focus on high-fidelity region
        min_fid = min(means) - max(stds) if stds else min(means)
        ax.set_ylim([min(0.99, min_fid - 0.005), 1.001])
        
        ax.grid(True, alpha=self.config.grid_alpha)
        ax.legend(loc='lower left')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add annotation for mean
        overall_mean = sum(f for fids in by_n.values() for f in fids) / sum(len(fids) for fids in by_n.values())
        ax.text(
            0.95, 0.05, f'Mean: {overall_mean:.6f}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        plt.tight_layout()
        output_path = self._save_figure(fig, "accuracy_fidelity")
        print(f"  Saved: {output_path}")
        return output_path
    
    def plot_accuracy_by_noise(self) -> Optional[Path]:
        """Plot fidelity vs noise level.
        
        Returns:
            Path to saved plot, or None if no data
        """
        data = self.get_successful_data("accuracy")
        if not data:
            return None
        
        # Group by noise level
        by_noise: Dict[float, List[float]] = {}
        for row in data:
            fid = row.get("fidelity")
            noise = row.get("noise_level", 0.0) or 0.0
            if fid is not None and fid > 0:
                if noise not in by_noise:
                    by_noise[noise] = []
                by_noise[noise].append(fid)
        
        if len(by_noise) < 2:
            return None  # Not enough data for noise comparison
        
        noise_levels = sorted(by_noise.keys())
        means = []
        stds = []
        for noise in noise_levels:
            m, s = self._compute_stats(by_noise[noise])
            means.append(m)
            stds.append(s)
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        ax.errorbar(
            noise_levels, means, yerr=stds,
            fmt='s-', capsize=self.config.error_capsize,
            color=self.COLORS["accuracy"], markersize=10, linewidth=2,
            label='LRET Fidelity'
        )
        
        ax.axhline(
            0.999, color=self.COLORS["baseline"], linestyle='--',
            linewidth=2, label='Threshold (0.999)'
        )
        
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Fidelity')
        ax.set_title('Fidelity vs Noise Level')
        ax.set_xscale('log') if min(noise_levels) > 0 else None
        ax.grid(True, alpha=self.config.grid_alpha)
        ax.legend(loc='lower left')
        
        plt.tight_layout()
        output_path = self._save_figure(fig, "accuracy_by_noise")
        print(f"  Saved: {output_path}")
        return output_path
    
    # -------------------------------------------------------------------------
    # Depth Scaling Plots
    # -------------------------------------------------------------------------
    
    def plot_depth_scaling(self) -> Optional[Path]:
        """Plot time vs circuit depth.
        
        Should show approximately linear scaling.
        
        Returns:
            Path to saved plot, or None if no data
        """
        data = self.get_successful_data("depth_scaling")
        if not data:
            print("  No depth scaling data to plot")
            return None
        
        # Group by depth
        by_d: Dict[int, List[float]] = {}
        for row in data:
            d = row["depth"]
            if d not in by_d:
                by_d[d] = []
            by_d[d].append(row["time_ms"])
        
        depths = sorted(by_d.keys())
        means = []
        stds = []
        for d in depths:
            m, s = self._compute_stats(by_d[d])
            means.append(m)
            stds.append(s)
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        ax.errorbar(
            depths, means, yerr=stds,
            fmt='o-', capsize=self.config.error_capsize,
            color=self.COLORS["depth"], markersize=8, linewidth=2,
            label='LRET Execution Time'
        )
        
        # Add linear fit
        if HAS_NUMPY and len(depths) >= 3:
            try:
                coeffs = np.polyfit(depths, means, 1)
                slope, intercept = coeffs
                
                d_fit = np.linspace(min(depths), max(depths), 100)
                t_fit = slope * d_fit + intercept
                ax.plot(
                    d_fit, t_fit, '--',
                    color=self.COLORS["fit_line"], linewidth=1.5,
                    label=f'Linear fit: {slope:.2f} ms/depth'
                )
            except Exception:
                pass
        
        ax.set_xlabel('Circuit Depth (d)')
        ax.set_ylabel('Execution Time (ms)')
        ax.set_title('Depth Scaling Analysis')
        ax.grid(True, alpha=self.config.grid_alpha)
        ax.legend(loc='upper left')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        output_path = self._save_figure(fig, "depth_scaling")
        print(f"  Saved: {output_path}")
        return output_path
    
    # -------------------------------------------------------------------------
    # Memory Plots
    # -------------------------------------------------------------------------
    
    def plot_memory_usage(self) -> Optional[Path]:
        """Plot memory usage vs qubit count.
        
        Returns:
            Path to saved plot, or None if no data
        """
        # Collect memory data from all categories
        all_data = []
        for category in ["memory", "scaling", "parallel", "depth_scaling"]:
            all_data.extend(self.get_successful_data(category))
        
        # Filter to rows with memory data
        data = [r for r in all_data if r.get("memory_mb") is not None and r["memory_mb"] > 0]
        
        if not data:
            print("  No memory data to plot")
            return None
        
        # Group by n_qubits
        by_n: Dict[int, List[float]] = {}
        for row in data:
            n = row["n_qubits"]
            if n not in by_n:
                by_n[n] = []
            by_n[n].append(row["memory_mb"])
        
        ns = sorted(by_n.keys())
        means = []
        stds = []
        for n in ns:
            m, s = self._compute_stats(by_n[n])
            means.append(m)
            stds.append(s)
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        ax.errorbar(
            ns, means, yerr=stds,
            fmt='o-', capsize=self.config.error_capsize,
            color=self.COLORS["memory"], markersize=8, linewidth=2,
            label='Peak Memory'
        )
        
        ax.set_xlabel('Number of Qubits (n)')
        ax.set_ylabel('Peak Memory (MB)')
        ax.set_title('Memory Usage Scaling')
        ax.grid(True, alpha=self.config.grid_alpha)
        ax.legend(loc='upper left')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        output_path = self._save_figure(fig, "memory_usage")
        print(f"  Saved: {output_path}")
        return output_path
    
    # -------------------------------------------------------------------------
    # Combined Summary Plot
    # -------------------------------------------------------------------------
    
    def plot_summary(self) -> Optional[Path]:
        """Generate combined summary plot with all categories.
        
        Creates a 2x2 grid with scaling, parallel, accuracy, and depth plots.
        
        Returns:
            Path to saved plot, or None if insufficient data
        """
        # Check what data is available
        has_scaling = bool(self.get_successful_data("scaling"))
        has_parallel = bool(self.get_successful_data("parallel"))
        has_accuracy = bool(self.get_successful_data("accuracy"))
        has_depth = bool(self.get_successful_data("depth_scaling"))
        
        available = sum([has_scaling, has_parallel, has_accuracy, has_depth])
        if available < 2:
            print("  Not enough categories for summary plot")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        ax_idx = 0
        
        # Scaling plot
        if has_scaling and ax_idx < 4:
            data = self.get_successful_data("scaling")
            by_n: Dict[int, List[float]] = {}
            for row in data:
                n = row["n_qubits"]
                if n not in by_n:
                    by_n[n] = []
                by_n[n].append(row["time_ms"])
            
            ns = sorted(by_n.keys())
            means = [self._compute_stats(by_n[n])[0] for n in ns]
            stds = [self._compute_stats(by_n[n])[1] for n in ns]
            
            ax = axes[ax_idx]
            ax.errorbar(ns, means, yerr=stds, fmt='o-', capsize=4, 
                       color=self.COLORS["scaling"], linewidth=2)
            ax.set_yscale('log')
            ax.set_xlabel('Qubits')
            ax.set_ylabel('Time (ms)')
            ax.set_title('Qubit Scaling')
            ax.grid(True, alpha=self.config.grid_alpha)
            ax_idx += 1
        
        # Parallel plot
        if has_parallel and ax_idx < 4:
            data = self.get_successful_data("parallel")
            by_mode: Dict[str, List[float]] = {}
            for row in data:
                mode = row["mode"]
                if mode not in by_mode:
                    by_mode[mode] = []
                by_mode[mode].append(row["time_ms"])
            
            if "sequential" in by_mode:
                seq_mean = self._compute_stats(by_mode["sequential"])[0]
                modes = [m for m in self.MODE_ORDER if m in by_mode]
                speedups = [seq_mean / self._compute_stats(by_mode[m])[0] for m in modes]
                
                ax = axes[ax_idx]
                colors = [self.COLORS["parallel"][i % len(self.COLORS["parallel"])] 
                          for i in range(len(modes))]
                ax.bar(range(len(modes)), speedups, color=colors, alpha=0.8)
                ax.axhline(1.0, color=self.COLORS["baseline"], linestyle='--')
                ax.set_xticks(range(len(modes)))
                ax.set_xticklabels([m[:3].title() for m in modes])
                ax.set_ylabel('Speedup')
                ax.set_title('Parallel Speedup')
                ax.grid(True, alpha=self.config.grid_alpha, axis='y')
                ax_idx += 1
        
        # Accuracy plot
        if has_accuracy and ax_idx < 4:
            data = self.get_successful_data("accuracy")
            by_n: Dict[int, List[float]] = {}
            for row in data:
                fid = row.get("fidelity")
                if fid and fid > 0:
                    n = row["n_qubits"]
                    if n not in by_n:
                        by_n[n] = []
                    by_n[n].append(fid)
            
            if by_n:
                ns = sorted(by_n.keys())
                means = [self._compute_stats(by_n[n])[0] for n in ns]
                stds = [self._compute_stats(by_n[n])[1] for n in ns]
                
                ax = axes[ax_idx]
                ax.errorbar(ns, means, yerr=stds, fmt='o-', capsize=4,
                           color=self.COLORS["accuracy"], linewidth=2)
                ax.axhline(0.999, color=self.COLORS["baseline"], linestyle='--')
                ax.set_xlabel('Qubits')
                ax.set_ylabel('Fidelity')
                ax.set_title('Accuracy Validation')
                ax.set_ylim([0.99, 1.001])
                ax.grid(True, alpha=self.config.grid_alpha)
                ax_idx += 1
        
        # Depth plot
        if has_depth and ax_idx < 4:
            data = self.get_successful_data("depth_scaling")
            by_d: Dict[int, List[float]] = {}
            for row in data:
                d = row["depth"]
                if d not in by_d:
                    by_d[d] = []
                by_d[d].append(row["time_ms"])
            
            depths = sorted(by_d.keys())
            means = [self._compute_stats(by_d[d])[0] for d in depths]
            stds = [self._compute_stats(by_d[d])[1] for d in depths]
            
            ax = axes[ax_idx]
            ax.errorbar(depths, means, yerr=stds, fmt='o-', capsize=4,
                       color=self.COLORS["depth"], linewidth=2)
            ax.set_xlabel('Depth')
            ax.set_ylabel('Time (ms)')
            ax.set_title('Depth Scaling')
            ax.grid(True, alpha=self.config.grid_alpha)
            ax_idx += 1
        
        # Hide unused axes
        for i in range(ax_idx, 4):
            axes[i].set_visible(False)
        
        plt.suptitle('LRET Benchmark Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = self._save_figure(fig, "benchmark_summary")
        print(f"  Saved: {output_path}")
        return output_path
    
    # -------------------------------------------------------------------------
    # Generate All Plots
    # -------------------------------------------------------------------------
    
    def generate_all_plots(self) -> Dict[str, Path]:
        """Generate all available plots.
        
        Returns:
            Dictionary mapping plot names to output paths
        """
        print(f"\nGenerating visualizations to: {self.output_dir}")
        
        outputs: Dict[str, Path] = {}
        
        # Scaling plots
        if self.get_successful_data("scaling"):
            print("  Generating scaling plots...")
            path = self.plot_scaling()
            if path:
                outputs["scaling_time"] = path
            
            path = self.plot_scaling_rank()
            if path:
                outputs["scaling_rank"] = path
        
        # Parallel plots
        if self.get_successful_data("parallel"):
            print("  Generating parallel speedup plots...")
            path = self.plot_parallel_speedup()
            if path:
                outputs["parallel_speedup"] = path
            
            path = self.plot_parallel_times()
            if path:
                outputs["parallel_times"] = path
        
        # Accuracy plots
        if self.get_successful_data("accuracy"):
            print("  Generating accuracy plots...")
            path = self.plot_accuracy_fidelity()
            if path:
                outputs["accuracy_fidelity"] = path
            
            path = self.plot_accuracy_by_noise()
            if path:
                outputs["accuracy_by_noise"] = path
        
        # Depth scaling plot
        if self.get_successful_data("depth_scaling"):
            print("  Generating depth scaling plot...")
            path = self.plot_depth_scaling()
            if path:
                outputs["depth_scaling"] = path
        
        # Memory plot
        path = self.plot_memory_usage()
        if path:
            outputs["memory_usage"] = path
        
        # Summary plot
        if self.config.save_combined:
            print("  Generating summary plot...")
            path = self.plot_summary()
            if path:
                outputs["summary"] = path
        
        print(f"\n✅ Generated {len(outputs)} plots")
        return outputs


def main():
    """Main entry point for benchmark visualization CLI."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations from LRET benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_visualize.py results.csv                  # Generate all plots
  python benchmark_visualize.py results.csv --output plots/  # Custom directory
  python benchmark_visualize.py results.csv --format svg     # SVG format
  python benchmark_visualize.py results.csv --dpi 300        # High resolution
        """
    )
    
    parser.add_argument(
        "results",
        type=Path,
        help="Benchmark results CSV file",
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots"),
        help="Output directory for plots (default: plots/)",
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "svg", "pdf", "eps"],
        default="png",
        help="Output format (default: png)",
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution in DPI (default: 150)",
    )
    
    parser.add_argument(
        "--style",
        type=str,
        default="whitegrid",
        help="Seaborn style (default: whitegrid)",
    )
    
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip combined summary plot",
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.results.exists():
        print(f"Error: Results file not found: {args.results}")
        sys.exit(1)
    
    # Create configuration
    config = PlotConfig(
        format=args.format,
        dpi=args.dpi,
        style=args.style,
        save_combined=not args.no_summary,
    )
    
    # Create visualizer and generate plots
    visualizer = BenchmarkVisualizer(
        csv_path=args.results,
        output_dir=args.output,
        config=config,
    )
    
    outputs = visualizer.generate_all_plots()
    
    if outputs:
        print(f"\nAll plots saved to: {args.output}")
    else:
        print("\nNo plots generated (insufficient data)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
