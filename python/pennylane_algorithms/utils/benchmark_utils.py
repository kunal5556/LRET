"""
Benchmark Utilities for LRET PennyLane Algorithm Testing

Provides common infrastructure for timing, memory tracking,
result collection, and comparison across devices and modes.
"""

import time
import json
import psutil
import traceback
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Callable, Union
from pathlib import Path
import threading
import os


@dataclass
class BenchmarkResult:
    """Container for benchmark results from a single run."""
    
    # Identification
    algorithm: str
    device_name: str
    mode: str  # 'sequential', 'batched', 'parallel', 'multiprocessing', etc.
    n_qubits: int
    
    # Timing
    execution_time_seconds: float
    setup_time_seconds: float = 0.0
    gradient_time_seconds: float = 0.0
    
    # Memory
    peak_memory_mb: float = 0.0
    memory_delta_mb: float = 0.0
    
    # Algorithm-specific metrics
    result_value: float = 0.0  # Main result (energy, accuracy, fidelity, etc.)
    secondary_value: float = 0.0  # Secondary metric if applicable
    convergence_iterations: int = 0
    
    # Quality metrics
    fidelity: float = 1.0
    error_rate: float = 0.0
    
    # Noise settings
    with_noise: bool = False
    noise_strength: float = 0.0

    # Per-epoch convergence data (loss/energy at each optimisation step)
    convergence_curve: List[float] = field(default_factory=list)

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    success: bool = True
    error_message: str = ""
    extra_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """Create from dictionary."""
        return cls(**data)
    
    def __repr__(self):
        return (f"BenchmarkResult({self.algorithm}, {self.device_name}/{self.mode}, "
                f"{self.n_qubits}q, {self.execution_time_seconds:.2f}s, "
                f"result={self.result_value:.6f})")


class MemoryTracker:
    """Context manager for tracking memory usage."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_memory = 0
        self.peak_memory = 0
        self._monitoring = False
        self._monitor_thread = None
    
    def __enter__(self):
        self.start_memory = self.process.memory_info().rss / (1024 * 1024)
        self.peak_memory = self.start_memory
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        return self
    
    def _monitor(self):
        """Background thread to track peak memory."""
        while self._monitoring:
            try:
                current = self.process.memory_info().rss / (1024 * 1024)
                self.peak_memory = max(self.peak_memory, current)
                time.sleep(0.01)  # 10ms sampling
            except:
                break
    
    def __exit__(self, *args):
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=0.1)
        end_memory = self.process.memory_info().rss / (1024 * 1024)
        self.memory_delta = end_memory - self.start_memory
    
    @property
    def peak_mb(self) -> float:
        return self.peak_memory
    
    @property  
    def delta_mb(self) -> float:
        return getattr(self, 'memory_delta', 0)


class Timer:
    """Context manager for precise timing."""
    
    def __init__(self):
        self.start_time = 0
        self.end_time = 0
        self.elapsed = 0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
    
    @property
    def seconds(self) -> float:
        return self.elapsed


class BenchmarkRunner:
    """
    Unified benchmark runner for comparing devices and modes.
    
    Usage:
        runner = BenchmarkRunner("VQE", n_qubits=4)
        
        # Add benchmark configurations
        runner.add_lret_modes()
        runner.add_comparison_device("default.mixed")
        
        # Run benchmarks
        results = runner.run(circuit_fn, params, n_trials=3)
    """
    
    def __init__(
        self,
        algorithm: str,
        n_qubits: int,
        with_noise: bool = True,
        noise_strength: float = 0.01,
        results_dir: Optional[Path] = None
    ):
        self.algorithm = algorithm
        self.n_qubits = n_qubits
        self.with_noise = with_noise
        self.noise_strength = noise_strength
        self.results_dir = results_dir or Path("results")
        
        self.configurations: List[Dict[str, Any]] = []
        self.results: List[BenchmarkResult] = []
    
    def add_configuration(
        self,
        device_name: str,
        mode: str,
        device_kwargs: Optional[Dict[str, Any]] = None,
        parallel_kwargs: Optional[Dict[str, Any]] = None
    ):
        """Add a device/mode configuration to test."""
        self.configurations.append({
            'device_name': device_name,
            'mode': mode,
            'device_kwargs': device_kwargs or {},
            'parallel_kwargs': parallel_kwargs or {}
        })
    
    def add_lret_modes(self, epsilon: float = 1e-4):
        """Add all LRET device modes for comparison."""
        modes = [
            ('sequential', {}),
            ('batched', {'batch_size': 10}),
            ('parallel', {'n_workers': 4}),
            ('openmp', {'use_openmp': True}),
        ]
        
        for mode_name, kwargs in modes:
            self.add_configuration(
                device_name='qlret.mixed',
                mode=mode_name,
                device_kwargs={'epsilon': epsilon, 'wires': self.n_qubits, **kwargs}
            )
    
    def add_python_parallel_modes(self):
        """Add Python parallelism modes for comparison."""
        modes = [
            ('multiprocessing', {'backend': 'multiprocessing', 'n_workers': 4}),
            ('threading', {'backend': 'threading', 'n_workers': 4}),
            ('joblib', {'backend': 'joblib', 'n_workers': 4}),
            ('sequential', {'backend': 'sequential'}),
        ]
        
        for mode_name, kwargs in modes:
            self.add_configuration(
                device_name='qlret.mixed',
                mode=f'python_{mode_name}',
                parallel_kwargs=kwargs
            )
    
    def add_comparison_devices(self, devices: Optional[List[str]] = None):
        """Add comparison devices (default.mixed, lightning.qubit, etc.)."""
        if devices is None:
            devices = ['default.mixed', 'default.qubit']
        
        for dev_name in devices:
            self.add_configuration(
                device_name=dev_name,
                mode='default',
                device_kwargs={'wires': self.n_qubits}
            )
    
    def run_single(
        self,
        benchmark_fn: Callable,
        config: Dict[str, Any],
        **kwargs
    ) -> BenchmarkResult:
        """Run a single benchmark configuration."""
        device_name = config['device_name']
        mode = config['mode']
        device_kwargs = config['device_kwargs']
        parallel_kwargs = config.get('parallel_kwargs', {})
        
        # Track memory and time
        with MemoryTracker() as mem:
            with Timer() as timer:
                try:
                    # Run the benchmark function
                    result_data = benchmark_fn(
                        device_name=device_name,
                        device_kwargs=device_kwargs,
                        parallel_kwargs=parallel_kwargs,
                        n_qubits=self.n_qubits,
                        with_noise=self.with_noise,
                        noise_strength=self.noise_strength,
                        **kwargs
                    )
                    success = True
                    error_msg = ""
                except Exception as e:
                    result_data = {'result_value': float('nan')}
                    success = False
                    error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        
        # Create result object
        result = BenchmarkResult(
            algorithm=self.algorithm,
            device_name=device_name,
            mode=mode,
            n_qubits=self.n_qubits,
            execution_time_seconds=timer.seconds,
            peak_memory_mb=mem.peak_mb,
            memory_delta_mb=mem.delta_mb,
            with_noise=self.with_noise,
            noise_strength=self.noise_strength,
            success=success,
            error_message=error_msg,
            **{k: v for k, v in result_data.items() if k in BenchmarkResult.__dataclass_fields__}
        )
        
        return result
    
    def run(
        self,
        benchmark_fn: Callable,
        n_trials: int = 3,
        warmup: int = 1,
        **kwargs
    ) -> List[BenchmarkResult]:
        """Run all benchmark configurations."""
        all_results = []
        
        for config in self.configurations:
            print(f"\n  Testing {config['device_name']} ({config['mode']})...")
            
            # Warmup runs
            for _ in range(warmup):
                try:
                    self.run_single(benchmark_fn, config, **kwargs)
                except:
                    pass
            
            # Actual trials
            trial_results = []
            for trial in range(n_trials):
                result = self.run_single(benchmark_fn, config, **kwargs)
                trial_results.append(result)
                
                if result.success:
                    print(f"    Trial {trial+1}: {result.execution_time_seconds:.3f}s, "
                          f"result={result.result_value:.6f}")
                else:
                    print(f"    Trial {trial+1}: FAILED - {result.error_message[:50]}")
            
            all_results.extend(trial_results)
        
        self.results = all_results
        return all_results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all results."""
        summary = {}
        
        # Group by device/mode
        groups = {}
        for r in self.results:
            key = f"{r.device_name}/{r.mode}"
            if key not in groups:
                groups[key] = []
            groups[key].append(r)
        
        for key, results in groups.items():
            successful = [r for r in results if r.success]
            if successful:
                times = [r.execution_time_seconds for r in successful]
                values = [r.result_value for r in successful]
                memory = [r.peak_memory_mb for r in successful]
                
                summary[key] = {
                    'n_trials': len(results),
                    'n_success': len(successful),
                    'mean_time': np.mean(times),
                    'std_time': np.std(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'mean_result': np.mean(values),
                    'std_result': np.std(values),
                    'mean_memory': np.mean(memory),
                }
            else:
                summary[key] = {
                    'n_trials': len(results),
                    'n_success': 0,
                    'error': results[0].error_message if results else 'Unknown'
                }
        
        return summary


def format_results_table(results: List[BenchmarkResult]) -> str:
    """Format benchmark results as a text table."""
    if not results:
        return "No results"
    
    # Group by device/mode
    groups = {}
    for r in results:
        key = f"{r.device_name}/{r.mode}"
        if key not in groups:
            groups[key] = []
        groups[key].append(r)
    
    lines = []
    lines.append("=" * 100)
    lines.append(f"{'Device/Mode':<30} {'Trials':>7} {'Time (s)':>12} {'Result':>15} {'Memory (MB)':>12} {'Status':>10}")
    lines.append("=" * 100)
    
    for key, group in sorted(groups.items()):
        successful = [r for r in group if r.success]
        
        if successful:
            times = [r.execution_time_seconds for r in successful]
            values = [r.result_value for r in successful]
            memory = [r.peak_memory_mb for r in successful]
            
            time_str = f"{np.mean(times):.3f}±{np.std(times):.3f}"
            result_str = f"{np.mean(values):.6f}"
            memory_str = f"{np.mean(memory):.1f}"
            status = "✅ OK"
        else:
            time_str = "N/A"
            result_str = "N/A"
            memory_str = "N/A"
            status = "❌ FAIL"
        
        lines.append(f"{key:<30} {len(group):>7} {time_str:>12} {result_str:>15} {memory_str:>12} {status:>10}")
    
    lines.append("=" * 100)
    return "\n".join(lines)


def save_results_json(
    results: List[BenchmarkResult],
    filepath: Union[str, Path],
    include_summary: bool = True
):
    """Save benchmark results to JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'results': [r.to_dict() for r in results],
        'timestamp': datetime.now().isoformat(),
        'n_results': len(results)
    }
    
    if include_summary:
        runner = BenchmarkRunner("", 0)
        runner.results = results
        data['summary'] = runner.get_summary()
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    print(f"Results saved to {filepath}")


def load_results_json(filepath: Union[str, Path]) -> List[BenchmarkResult]:
    """Load benchmark results from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return [BenchmarkResult.from_dict(r) for r in data['results']]


def compare_results(
    results: List[BenchmarkResult],
    baseline_key: str = "default.mixed/default"
) -> Dict[str, Dict[str, float]]:
    """
    Compare results against a baseline.
    
    Returns speedup and accuracy comparison for each device/mode.
    """
    # Group by device/mode
    groups = {}
    for r in results:
        key = f"{r.device_name}/{r.mode}"
        if key not in groups:
            groups[key] = []
        groups[key].append(r)
    
    # Get baseline statistics
    if baseline_key not in groups:
        print(f"Warning: Baseline {baseline_key} not found")
        return {}
    
    baseline_results = [r for r in groups[baseline_key] if r.success]
    if not baseline_results:
        return {}
    
    baseline_time = np.mean([r.execution_time_seconds for r in baseline_results])
    baseline_result = np.mean([r.result_value for r in baseline_results])
    baseline_memory = np.mean([r.peak_memory_mb for r in baseline_results])
    
    comparisons = {}
    for key, group in groups.items():
        successful = [r for r in group if r.success]
        if successful:
            mean_time = np.mean([r.execution_time_seconds for r in successful])
            mean_result = np.mean([r.result_value for r in successful])
            mean_memory = np.mean([r.peak_memory_mb for r in successful])
            
            comparisons[key] = {
                'speedup': baseline_time / mean_time if mean_time > 0 else 0,
                'memory_ratio': baseline_memory / mean_memory if mean_memory > 0 else 0,
                'result_diff': abs(mean_result - baseline_result),
                'relative_error': abs(mean_result - baseline_result) / abs(baseline_result) if baseline_result != 0 else 0
            }
    
    return comparisons
