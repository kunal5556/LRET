"""
Utility modules for LRET PennyLane Algorithm Benchmarking Suite
"""

from .benchmark_utils import (
    BenchmarkResult,
    BenchmarkRunner,
    format_results_table,
    save_results_json,
    load_results_json,
)

from .device_factory import (
    create_lret_device,
    create_comparison_device,
    get_all_lret_modes,
    get_device_info,
    DeviceConfig,
)

from .parallel_modes import (
    run_parallel_comparison,
    get_parallel_modes,
    ParallelExecutor,
    measure_parallel_speedup,
)

from .plotting import (
    plot_device_comparison,
    plot_parallel_speedup,
    plot_scaling_analysis,
    plot_noise_resilience,
    create_summary_report,
)

__all__ = [
    # Benchmark utilities
    'BenchmarkResult',
    'BenchmarkRunner',
    'format_results_table',
    'save_results_json',
    'load_results_json',
    # Device factory
    'create_lret_device',
    'create_comparison_device',
    'get_all_lret_modes',
    'get_device_info',
    'DeviceConfig',
    # Parallel modes
    'run_parallel_comparison',
    'get_parallel_modes',
    'ParallelExecutor',
    'measure_parallel_speedup',
    # Plotting
    'plot_device_comparison',
    'plot_parallel_speedup',
    'plot_scaling_analysis',
    'plot_noise_resilience',
    'create_summary_report',
]
