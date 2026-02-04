"""
Plotting Utilities for LRET PennyLane Algorithm Benchmarks

Generates visualizations for:
- Device comparison charts
- Parallel speedup analysis
- Scaling analysis (time vs qubits)
- Noise resilience curves
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plotting functions will return data only.")


# Color schemes for consistent visualization
DEVICE_COLORS = {
    'qlret.mixed': '#2E86AB',      # Blue
    'default.mixed': '#A23B72',     # Magenta
    'default.qubit': '#F18F01',     # Orange
    'lightning.qubit': '#C73E1D',   # Red
}

MODE_COLORS = {
    'sequential': '#264653',
    'batched': '#2A9D8F',
    'parallel': '#E9C46A',
    'openmp': '#F4A261',
    'multiprocessing': '#E76F51',
    'threading': '#8338EC',
    'joblib_loky': '#FF006E',
    'joblib_threading': '#FB5607',
}


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")


def plot_device_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = 'mean_time',
    title: str = 'Device Comparison',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> Optional[plt.Figure]:
    """
    Plot comparison of devices/modes.
    
    Args:
        results: Dictionary mapping device/mode to metrics
        metric: Metric to plot ('mean_time', 'mean_memory', 'speedup', etc.)
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract data
    labels = list(results.keys())
    values = [results[k].get(metric, 0) for k in labels]
    
    # Assign colors based on device
    colors = []
    for label in labels:
        device = label.split('/')[0]
        if device in DEVICE_COLORS:
            colors.append(DEVICE_COLORS[device])
        else:
            colors.append('#666666')
    
    # Create bar chart
    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor='black', linewidth=0.5)
    
    # Customize
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig


def plot_parallel_speedup(
    results: Dict[str, Dict[str, float]],
    title: str = 'Parallel Mode Comparison',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> Optional[plt.Figure]:
    """
    Plot speedup comparison across parallel modes.
    
    Args:
        results: Dictionary with 'mode' -> {'speedup': float, 'efficiency': float, ...}
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Extract data
    modes = list(results.keys())
    speedups = [results[m].get('speedup', 1.0) for m in modes]
    efficiencies = [results[m].get('efficiency', 1.0) for m in modes]
    
    # Colors
    colors = [MODE_COLORS.get(m, '#666666') for m in modes]
    
    # Speedup chart
    bars1 = ax1.bar(range(len(modes)), speedups, color=colors, edgecolor='black', linewidth=0.5)
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Baseline')
    ax1.set_xticks(range(len(modes)))
    ax1.set_xticklabels(modes, rotation=45, ha='right')
    ax1.set_ylabel('Speedup (x)')
    ax1.set_title('Speedup vs Sequential')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Efficiency chart
    bars2 = ax2.bar(range(len(modes)), [e * 100 for e in efficiencies], color=colors, 
                    edgecolor='black', linewidth=0.5)
    ax2.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Perfect Efficiency')
    ax2.set_xticks(range(len(modes)))
    ax2.set_xticklabels(modes, rotation=45, ha='right')
    ax2.set_ylabel('Efficiency (%)')
    ax2.set_title('Parallel Efficiency')
    ax2.set_ylim(0, 120)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig


def plot_scaling_analysis(
    qubit_counts: List[int],
    device_times: Dict[str, List[float]],
    title: str = 'Scaling Analysis',
    log_scale: bool = True,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> Optional[plt.Figure]:
    """
    Plot execution time vs number of qubits for different devices.
    
    Args:
        qubit_counts: List of qubit counts
        device_times: Dictionary mapping device name to list of times
        title: Plot title
        log_scale: Whether to use log scale for y-axis
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for device_name, times in device_times.items():
        color = DEVICE_COLORS.get(device_name.split('/')[0], '#666666')
        ax.plot(qubit_counts[:len(times)], times, 'o-', 
                label=device_name, color=color, linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Execution Time (s)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if log_scale:
        ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig


def plot_noise_resilience(
    noise_levels: List[float],
    device_results: Dict[str, List[float]],
    exact_value: float,
    metric_name: str = 'Expectation Value',
    title: str = 'Noise Resilience',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> Optional[plt.Figure]:
    """
    Plot algorithm results vs noise level.
    
    Args:
        noise_levels: List of noise strengths
        device_results: Dictionary mapping device name to list of results
        exact_value: Exact/ideal result for reference
        metric_name: Name of the metric being plotted
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Absolute values
    ax1.axhline(y=exact_value, color='black', linestyle='--', linewidth=2, 
                label=f'Exact ({exact_value:.4f})')
    
    for device_name, results in device_results.items():
        color = DEVICE_COLORS.get(device_name.split('/')[0], '#666666')
        ax1.plot(noise_levels[:len(results)], results, 'o-',
                label=device_name, color=color, linewidth=2, markersize=6)
    
    ax1.set_xlabel('Noise Strength')
    ax1.set_ylabel(metric_name)
    ax1.set_title(f'{metric_name} vs Noise')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Relative error
    for device_name, results in device_results.items():
        color = DEVICE_COLORS.get(device_name.split('/')[0], '#666666')
        errors = [abs(r - exact_value) / abs(exact_value) * 100 if exact_value != 0 else 0 
                 for r in results]
        ax2.plot(noise_levels[:len(results)], errors, 'o-',
                label=device_name, color=color, linewidth=2, markersize=6)
    
    ax2.set_xlabel('Noise Strength')
    ax2.set_ylabel('Relative Error (%)')
    ax2.set_title('Error vs Noise')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig


def plot_convergence(
    iterations: List[int],
    device_values: Dict[str, List[float]],
    exact_value: Optional[float] = None,
    title: str = 'Convergence',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> Optional[plt.Figure]:
    """
    Plot algorithm convergence over iterations.
    
    Args:
        iterations: List of iteration numbers
        device_values: Dictionary mapping device name to list of values
        exact_value: Exact value to show as reference line
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if exact_value is not None:
        ax.axhline(y=exact_value, color='black', linestyle='--', 
                   linewidth=2, label=f'Exact ({exact_value:.6f})')
    
    for device_name, values in device_values.items():
        color = DEVICE_COLORS.get(device_name.split('/')[0], '#666666')
        ax.plot(iterations[:len(values)], values, '-',
                label=device_name, color=color, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig


def create_summary_report(
    results: List[Dict[str, Any]],
    algorithm: str,
    output_dir: Path,
    include_plots: bool = True
) -> Path:
    """
    Create a comprehensive summary report with plots and tables.
    
    Args:
        results: List of benchmark results
        algorithm: Algorithm name
        output_dir: Directory to save report
        include_plots: Whether to generate and include plots
    
    Returns:
        Path to generated report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / f"{algorithm}_report.md"
    
    # Generate report content
    lines = []
    lines.append(f"# {algorithm} Benchmark Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("\n## Summary\n")
    
    # Group results by device/mode
    groups = {}
    for r in results:
        key = f"{r.get('device_name', 'unknown')}/{r.get('mode', 'unknown')}"
        if key not in groups:
            groups[key] = []
        groups[key].append(r)
    
    # Create summary table
    lines.append("| Device/Mode | Trials | Mean Time (s) | Std Time | Mean Result | Status |")
    lines.append("|------------|--------|---------------|----------|-------------|--------|")
    
    for key, group in sorted(groups.items()):
        successful = [r for r in group if r.get('success', True)]
        n_trials = len(group)
        
        if successful:
            times = [r.get('execution_time_seconds', 0) for r in successful]
            values = [r.get('result_value', 0) for r in successful]
            
            mean_time = np.mean(times)
            std_time = np.std(times)
            mean_result = np.mean(values)
            status = "✅"
        else:
            mean_time = std_time = mean_result = float('nan')
            status = "❌"
        
        lines.append(f"| {key} | {n_trials} | {mean_time:.4f} | {std_time:.4f} | "
                    f"{mean_result:.6f} | {status} |")
    
    lines.append("\n## Detailed Results\n")
    
    # Add device comparison section
    lines.append("### Device Comparison\n")
    
    if include_plots and HAS_MATPLOTLIB:
        # Generate comparison plot
        comparison_data = {}
        for key, group in groups.items():
            successful = [r for r in group if r.get('success', True)]
            if successful:
                comparison_data[key] = {
                    'mean_time': np.mean([r.get('execution_time_seconds', 0) for r in successful]),
                    'mean_memory': np.mean([r.get('peak_memory_mb', 0) for r in successful]),
                }
        
        if comparison_data:
            fig = plot_device_comparison(
                comparison_data,
                metric='mean_time',
                title=f'{algorithm} - Execution Time Comparison',
                save_path=output_dir / f"{algorithm}_time_comparison.png"
            )
            plt.close(fig)
            lines.append(f"![Time Comparison]({algorithm}_time_comparison.png)\n")
    
    # Write report
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    
    # Save raw results as JSON
    json_path = output_dir / f"{algorithm}_results.json"
    with open(json_path, 'w') as f:
        json.dump({
            'algorithm': algorithm,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2, default=str)
    
    print(f"Report saved to {report_path}")
    print(f"Raw data saved to {json_path}")
    
    return report_path


def plot_algorithm_comparison_grid(
    algorithm_results: Dict[str, Dict[str, float]],
    metric: str = 'speedup',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> Optional[plt.Figure]:
    """
    Create a grid showing performance across all algorithms.
    
    Args:
        algorithm_results: Nested dict {algorithm: {device: metric_value}}
        metric: Metric to plot
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()
    
    algorithms = list(algorithm_results.keys())
    
    # Get all devices
    all_devices = set()
    for alg_data in algorithm_results.values():
        all_devices.update(alg_data.keys())
    devices = sorted(list(all_devices))
    
    # Create matrix
    data = np.zeros((len(algorithms), len(devices)))
    for i, alg in enumerate(algorithms):
        for j, dev in enumerate(devices):
            data[i, j] = algorithm_results[alg].get(dev, 0)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto')
    
    ax.set_xticks(range(len(devices)))
    ax.set_xticklabels(devices, rotation=45, ha='right')
    ax.set_yticks(range(len(algorithms)))
    ax.set_yticklabels(algorithms)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric.replace('_', ' ').title())
    
    # Add value annotations
    for i in range(len(algorithms)):
        for j in range(len(devices)):
            text = ax.text(j, i, f'{data[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=8)
    
    ax.set_title(f'Algorithm Performance Comparison - {metric.replace("_", " ").title()}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig
