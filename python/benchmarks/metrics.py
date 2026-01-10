"""Performance metrics collection utilities.

This module provides functions for measuring execution time, memory usage,
and other performance metrics during benchmark runs.
"""

from __future__ import annotations

import gc
import os
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime
import json

import numpy as np

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class ExecutionMetrics:
    """Metrics from a single circuit execution."""
    
    # Identification
    device_name: str
    circuit_type: str
    num_qubits: int
    circuit_depth: int
    trial: int
    
    # Timing
    execution_time_ms: float
    compile_time_ms: float = 0.0
    total_time_ms: float = 0.0
    
    # Memory
    peak_memory_mb: float = 0.0
    memory_delta_mb: float = 0.0
    
    # Results
    result_value: Optional[float] = None
    result_fidelity: Optional[float] = None
    
    # Status
    success: bool = True
    error_message: str = ""
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class MemoryTracker:
    """Track memory usage during execution."""
    
    def __init__(self):
        self.start_memory = 0.0
        self.peak_memory = 0.0
        self._process = None
        if HAS_PSUTIL:
            self._process = psutil.Process(os.getpid())
    
    def start(self):
        """Start memory tracking."""
        gc.collect()
        if self._process:
            self.start_memory = self._process.memory_info().rss / (1024 * 1024)
            self.peak_memory = self.start_memory
    
    def update(self):
        """Update peak memory."""
        if self._process:
            current = self._process.memory_info().rss / (1024 * 1024)
            self.peak_memory = max(self.peak_memory, current)
    
    def stop(self) -> Tuple[float, float]:
        """Stop tracking and return (peak_mb, delta_mb)."""
        self.update()
        delta = self.peak_memory - self.start_memory
        return self.peak_memory, delta


class Timer:
    """Simple high-resolution timer."""
    
    def __init__(self):
        self.start_time = 0.0
        self.elapsed = 0.0
    
    def start(self):
        """Start the timer."""
        self.start_time = time.perf_counter()
    
    def stop(self) -> float:
        """Stop the timer and return elapsed milliseconds."""
        self.elapsed = (time.perf_counter() - self.start_time) * 1000
        return self.elapsed


def measure_execution(
    func: Callable,
    *args,
    warmup: bool = True,
    **kwargs
) -> Tuple[Any, float, float, float]:
    """Measure execution time and memory of a function.
    
    Parameters
    ----------
    func : callable
        Function to measure.
    *args, **kwargs
        Arguments to pass to function.
    warmup : bool
        If True, run once before timing to warm up JIT, etc.
    
    Returns
    -------
    result : Any
        Function result.
    execution_time_ms : float
        Execution time in milliseconds.
    peak_memory_mb : float
        Peak memory usage in MB.
    memory_delta_mb : float
        Memory increase during execution.
    """
    # Warmup run (optional)
    if warmup:
        try:
            _ = func(*args, **kwargs)
        except Exception:
            pass  # Ignore warmup errors
    
    # Force garbage collection
    gc.collect()
    
    # Set up tracking
    timer = Timer()
    mem_tracker = MemoryTracker()
    
    # Execute with measurement
    mem_tracker.start()
    timer.start()
    
    result = func(*args, **kwargs)
    
    execution_time_ms = timer.stop()
    peak_memory_mb, memory_delta_mb = mem_tracker.stop()
    
    return result, execution_time_ms, peak_memory_mb, memory_delta_mb


def run_with_timeout(
    func: Callable,
    timeout_seconds: float,
    *args,
    **kwargs
) -> Tuple[Any, bool, str]:
    """Run a function with timeout.
    
    Note: This is a simple implementation that doesn't actually enforce
    timeout during execution - it's meant to check after execution.
    For true timeout, would need multiprocessing.
    
    Parameters
    ----------
    func : callable
        Function to run.
    timeout_seconds : float
        Maximum execution time.
    
    Returns
    -------
    result : Any
        Function result (None if failed).
    success : bool
        Whether execution completed successfully.
    error_message : str
        Error message if failed.
    """
    start = time.perf_counter()
    
    try:
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        
        if elapsed > timeout_seconds:
            return result, False, f"Execution took {elapsed:.1f}s (timeout={timeout_seconds}s)"
        
        return result, True, ""
    
    except MemoryError as e:
        return None, False, f"MemoryError: {e}"
    
    except Exception as e:
        return None, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


class BenchmarkResults:
    """Collection of benchmark results with analysis methods."""
    
    def __init__(self):
        self.metrics: List[ExecutionMetrics] = []
        self.metadata: Dict[str, Any] = {
            "start_time": datetime.now().isoformat(),
            "pennylane_version": None,
            "python_version": None,
            "platform": None,
        }
        self._populate_metadata()
    
    def _populate_metadata(self):
        """Populate system metadata."""
        import platform
        import pennylane as qml
        import sys
        
        self.metadata["pennylane_version"] = qml.__version__
        self.metadata["python_version"] = sys.version
        self.metadata["platform"] = platform.platform()
        if HAS_PSUTIL:
            self.metadata["total_memory_gb"] = psutil.virtual_memory().total / (1024**3)
            self.metadata["cpu_count"] = psutil.cpu_count()
    
    def add(self, metric: ExecutionMetrics):
        """Add a metric result."""
        self.metrics.append(metric)
    
    def to_dataframe(self):
        """Convert to pandas DataFrame (if pandas available)."""
        try:
            import pandas as pd
            return pd.DataFrame([m.to_dict() for m in self.metrics])
        except ImportError:
            raise ImportError("pandas required for DataFrame conversion")
    
    def save_json(self, filepath: str):
        """Save results to JSON file."""
        data = {
            "metadata": self.metadata,
            "results": [m.to_dict() for m in self.metrics],
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_csv(self, filepath: str):
        """Save results to CSV file."""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
    
    @classmethod
    def load_json(cls, filepath: str) -> "BenchmarkResults":
        """Load results from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        results = cls()
        results.metadata = data["metadata"]
        for m_dict in data["results"]:
            results.metrics.append(ExecutionMetrics(**m_dict))
        
        return results
    
    def summary_by_device(self) -> Dict[str, Dict[str, Any]]:
        """Get summary statistics by device."""
        from collections import defaultdict
        
        by_device = defaultdict(list)
        for m in self.metrics:
            if m.success:
                by_device[m.device_name].append(m)
        
        summary = {}
        for device, metrics in by_device.items():
            times = [m.execution_time_ms for m in metrics]
            memories = [m.peak_memory_mb for m in metrics]
            
            summary[device] = {
                "num_successful": len(metrics),
                "avg_time_ms": np.mean(times),
                "std_time_ms": np.std(times),
                "min_time_ms": np.min(times),
                "max_time_ms": np.max(times),
                "avg_memory_mb": np.mean(memories),
                "max_memory_mb": np.max(memories),
            }
        
        return summary
    
    def compare_devices(
        self,
        baseline_device: str,
        metric: str = "execution_time_ms"
    ) -> Dict[str, float]:
        """Compare devices to a baseline, returning speedup ratios.
        
        Parameters
        ----------
        baseline_device : str
            Name of baseline device for comparison.
        metric : str
            Metric to compare ("execution_time_ms" or "peak_memory_mb").
        
        Returns
        -------
        Dict mapping device name to speedup/reduction ratio vs baseline.
        Values > 1 mean the device is faster/uses less memory than baseline.
        """
        from collections import defaultdict
        
        # Group by device and qubit count
        by_device_qubits = defaultdict(lambda: defaultdict(list))
        for m in self.metrics:
            if m.success:
                val = getattr(m, metric)
                by_device_qubits[m.device_name][m.num_qubits].append(val)
        
        # Calculate average for each device at each qubit count
        averages = {}
        for device, by_qubits in by_device_qubits.items():
            averages[device] = {q: np.mean(vals) for q, vals in by_qubits.items()}
        
        # Compare to baseline
        if baseline_device not in averages:
            return {}
        
        baseline = averages[baseline_device]
        ratios = {}
        
        for device, device_avgs in averages.items():
            if device == baseline_device:
                ratios[device] = 1.0
                continue
            
            # Average ratio across common qubit counts
            common_qubits = set(baseline.keys()) & set(device_avgs.keys())
            if common_qubits:
                ratio_sum = sum(
                    baseline[q] / device_avgs[q] 
                    for q in common_qubits 
                    if device_avgs[q] > 0
                )
                ratios[device] = ratio_sum / len(common_qubits)
        
        return ratios
