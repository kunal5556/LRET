"""
Python Parallelism Comparison for LRET PennyLane Benchmarks

Compares different Python parallelization strategies:
- Sequential (baseline)
- Multiprocessing (process-based parallelism)
- Threading (thread-based parallelism)  
- Joblib (high-level parallel primitives)
- Concurrent futures (ThreadPoolExecutor, ProcessPoolExecutor)
"""

import time
import numpy as np
from typing import Callable, List, Any, Dict, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import functools

# Optional imports
try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    import multiprocessing as mp
    from multiprocessing import Pool
    HAS_MULTIPROCESSING = True
except ImportError:
    HAS_MULTIPROCESSING = False


@dataclass
class ParallelResult:
    """Result from parallel execution."""
    mode: str
    n_tasks: int
    n_workers: int
    total_time: float
    results: List[Any]
    speedup: float = 1.0
    efficiency: float = 1.0


# Available parallel modes
PARALLEL_MODES = {
    'sequential': {
        'description': 'Sequential execution (baseline)',
        'requires': [],
    },
    'multiprocessing': {
        'description': 'Process-based parallelism using multiprocessing.Pool',
        'requires': ['multiprocessing'],
    },
    'threading': {
        'description': 'Thread-based parallelism using ThreadPoolExecutor',
        'requires': ['concurrent.futures'],
    },
    'process_pool': {
        'description': 'Process-based parallelism using ProcessPoolExecutor',
        'requires': ['concurrent.futures'],
    },
    'joblib_loky': {
        'description': 'Joblib with loky backend (process-based)',
        'requires': ['joblib'],
    },
    'joblib_threading': {
        'description': 'Joblib with threading backend',
        'requires': ['joblib'],
    },
}


def get_parallel_modes() -> List[str]:
    """Get list of available parallel modes."""
    available = ['sequential', 'threading', 'process_pool']
    
    if HAS_MULTIPROCESSING:
        available.append('multiprocessing')
    
    if HAS_JOBLIB:
        available.extend(['joblib_loky', 'joblib_threading'])
    
    return available


class ParallelExecutor:
    """
    Unified executor for different parallelization strategies.
    
    Usage:
        executor = ParallelExecutor(mode='multiprocessing', n_workers=4)
        results = executor.map(my_function, list_of_args)
    """
    
    def __init__(
        self,
        mode: str = 'sequential',
        n_workers: int = 4,
        verbose: bool = False
    ):
        self.mode = mode
        self.n_workers = n_workers
        self.verbose = verbose
        
        if mode not in PARALLEL_MODES and mode not in get_parallel_modes():
            raise ValueError(f"Unknown parallel mode: {mode}. "
                           f"Available: {get_parallel_modes()}")
    
    def map(
        self,
        func: Callable,
        args_list: List[Tuple],
        **kwargs
    ) -> List[Any]:
        """
        Apply function to each argument tuple in parallel.
        
        Args:
            func: Function to apply
            args_list: List of argument tuples
            **kwargs: Additional arguments passed to all calls
        
        Returns:
            List of results in order
        """
        if self.verbose:
            print(f"Running {len(args_list)} tasks with mode={self.mode}, "
                  f"n_workers={self.n_workers}")
        
        if self.mode == 'sequential':
            return self._run_sequential(func, args_list, **kwargs)
        elif self.mode == 'multiprocessing':
            return self._run_multiprocessing(func, args_list, **kwargs)
        elif self.mode == 'threading':
            return self._run_threading(func, args_list, **kwargs)
        elif self.mode == 'process_pool':
            return self._run_process_pool(func, args_list, **kwargs)
        elif self.mode == 'joblib_loky':
            return self._run_joblib(func, args_list, backend='loky', **kwargs)
        elif self.mode == 'joblib_threading':
            return self._run_joblib(func, args_list, backend='threading', **kwargs)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _run_sequential(
        self,
        func: Callable,
        args_list: List[Tuple],
        **kwargs
    ) -> List[Any]:
        """Sequential execution."""
        results = []
        for args in args_list:
            if isinstance(args, tuple):
                result = func(*args, **kwargs)
            else:
                result = func(args, **kwargs)
            results.append(result)
        return results
    
    def _run_multiprocessing(
        self,
        func: Callable,
        args_list: List[Tuple],
        **kwargs
    ) -> List[Any]:
        """Multiprocessing pool execution."""
        if not HAS_MULTIPROCESSING:
            return self._run_sequential(func, args_list, **kwargs)
        
        # Wrap function with kwargs
        if kwargs:
            func = functools.partial(func, **kwargs)
        
        with Pool(processes=self.n_workers) as pool:
            if all(isinstance(a, tuple) for a in args_list):
                results = pool.starmap(func, args_list)
            else:
                results = pool.map(func, args_list)
        
        return results
    
    def _run_threading(
        self,
        func: Callable,
        args_list: List[Tuple],
        **kwargs
    ) -> List[Any]:
        """ThreadPoolExecutor execution."""
        results = [None] * len(args_list)
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {}
            for i, args in enumerate(args_list):
                if isinstance(args, tuple):
                    future = executor.submit(func, *args, **kwargs)
                else:
                    future = executor.submit(func, args, **kwargs)
                futures[future] = i
            
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        
        return results
    
    def _run_process_pool(
        self,
        func: Callable,
        args_list: List[Tuple],
        **kwargs
    ) -> List[Any]:
        """ProcessPoolExecutor execution."""
        results = [None] * len(args_list)
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {}
            for i, args in enumerate(args_list):
                if isinstance(args, tuple):
                    future = executor.submit(func, *args, **kwargs)
                else:
                    future = executor.submit(func, args, **kwargs)
                futures[future] = i
            
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = None
                    print(f"Task {idx} failed: {e}")
        
        return results
    
    def _run_joblib(
        self,
        func: Callable,
        args_list: List[Tuple],
        backend: str = 'loky',
        **kwargs
    ) -> List[Any]:
        """Joblib parallel execution."""
        if not HAS_JOBLIB:
            print("Warning: joblib not available, falling back to sequential")
            return self._run_sequential(func, args_list, **kwargs)
        
        if kwargs:
            func = functools.partial(func, **kwargs)
        
        if all(isinstance(a, tuple) for a in args_list):
            results = Parallel(n_jobs=self.n_workers, backend=backend)(
                delayed(func)(*args) for args in args_list
            )
        else:
            results = Parallel(n_jobs=self.n_workers, backend=backend)(
                delayed(func)(args) for args in args_list
            )
        
        return results


def run_parallel_comparison(
    func: Callable,
    args_list: List[Tuple],
    modes: Optional[List[str]] = None,
    n_workers: int = 4,
    **kwargs
) -> Dict[str, ParallelResult]:
    """
    Compare different parallelization modes on the same workload.
    
    Args:
        func: Function to benchmark
        args_list: List of argument tuples
        modes: List of modes to test (default: all available)
        n_workers: Number of workers for parallel modes
        **kwargs: Additional arguments passed to function
    
    Returns:
        Dictionary mapping mode name to ParallelResult
    """
    if modes is None:
        modes = get_parallel_modes()
    
    results = {}
    baseline_time = None
    
    for mode in modes:
        executor = ParallelExecutor(mode=mode, n_workers=n_workers)
        
        start_time = time.perf_counter()
        task_results = executor.map(func, args_list, **kwargs)
        elapsed = time.perf_counter() - start_time
        
        # Calculate speedup relative to sequential
        if mode == 'sequential':
            baseline_time = elapsed
            speedup = 1.0
        else:
            speedup = baseline_time / elapsed if baseline_time and elapsed > 0 else 1.0
        
        # Calculate efficiency (speedup / n_workers for parallel modes)
        if mode == 'sequential':
            efficiency = 1.0
        else:
            efficiency = speedup / n_workers
        
        results[mode] = ParallelResult(
            mode=mode,
            n_tasks=len(args_list),
            n_workers=n_workers if mode != 'sequential' else 1,
            total_time=elapsed,
            results=task_results,
            speedup=speedup,
            efficiency=efficiency
        )
        
        print(f"  {mode:<20}: {elapsed:.3f}s (speedup: {speedup:.2f}x, "
              f"efficiency: {efficiency:.1%})")
    
    return results


def measure_parallel_speedup(
    func: Callable,
    args_list: List[Tuple],
    worker_counts: List[int] = [1, 2, 4, 8],
    mode: str = 'multiprocessing',
    **kwargs
) -> Dict[int, ParallelResult]:
    """
    Measure speedup as function of worker count.
    
    Args:
        func: Function to benchmark
        args_list: List of argument tuples
        worker_counts: List of worker counts to test
        mode: Parallel mode to use
        **kwargs: Additional arguments passed to function
    
    Returns:
        Dictionary mapping worker count to ParallelResult
    """
    results = {}
    baseline_time = None
    
    for n_workers in worker_counts:
        if n_workers == 1:
            executor = ParallelExecutor(mode='sequential', n_workers=1)
        else:
            executor = ParallelExecutor(mode=mode, n_workers=n_workers)
        
        start_time = time.perf_counter()
        task_results = executor.map(func, args_list, **kwargs)
        elapsed = time.perf_counter() - start_time
        
        if n_workers == 1:
            baseline_time = elapsed
            speedup = 1.0
        else:
            speedup = baseline_time / elapsed if baseline_time and elapsed > 0 else 1.0
        
        efficiency = speedup / n_workers
        
        results[n_workers] = ParallelResult(
            mode=mode if n_workers > 1 else 'sequential',
            n_tasks=len(args_list),
            n_workers=n_workers,
            total_time=elapsed,
            results=task_results,
            speedup=speedup,
            efficiency=efficiency
        )
        
        print(f"  {n_workers} workers: {elapsed:.3f}s (speedup: {speedup:.2f}x, "
              f"efficiency: {efficiency:.1%})")
    
    return results


def format_parallel_comparison(results: Dict[str, ParallelResult]) -> str:
    """Format parallel comparison results as a table."""
    lines = []
    lines.append("\nParallel Mode Comparison:")
    lines.append("=" * 70)
    lines.append(f"{'Mode':<20} {'Time (s)':>10} {'Speedup':>10} {'Efficiency':>12} {'Workers':>8}")
    lines.append("-" * 70)
    
    for mode, result in sorted(results.items(), key=lambda x: x[1].total_time):
        lines.append(
            f"{result.mode:<20} {result.total_time:>10.3f} {result.speedup:>10.2f}x "
            f"{result.efficiency:>11.1%} {result.n_workers:>8}"
        )
    
    lines.append("=" * 70)
    return "\n".join(lines)


# Convenience functions for common patterns

def parallel_parameter_sweep(
    circuit_fn: Callable,
    param_grid: np.ndarray,
    mode: str = 'multiprocessing',
    n_workers: int = 4
) -> np.ndarray:
    """
    Run a circuit over a parameter grid in parallel.
    
    Args:
        circuit_fn: Function that takes parameters and returns result
        param_grid: Array of parameter values to sweep
        mode: Parallelization mode
        n_workers: Number of parallel workers
    
    Returns:
        Array of results
    """
    executor = ParallelExecutor(mode=mode, n_workers=n_workers)
    
    # Convert grid to list of tuples
    if param_grid.ndim == 1:
        args_list = [(p,) for p in param_grid]
    else:
        args_list = [tuple(p) for p in param_grid]
    
    results = executor.map(circuit_fn, args_list)
    return np.array(results)


def parallel_gradient_computation(
    circuit_fn: Callable,
    params: np.ndarray,
    shift: float = np.pi / 2,
    mode: str = 'multiprocessing',
    n_workers: int = 4
) -> np.ndarray:
    """
    Compute gradients using parameter-shift rule in parallel.
    
    Args:
        circuit_fn: Function that takes parameters and returns expectation value
        params: Current parameter values
        shift: Shift amount for parameter-shift rule
        mode: Parallelization mode
        n_workers: Number of parallel workers
    
    Returns:
        Gradient array
    """
    n_params = len(params)
    
    # Create shifted parameter configurations
    shifted_params = []
    for i in range(n_params):
        # Plus shift
        p_plus = params.copy()
        p_plus[i] += shift
        shifted_params.append((p_plus, i, +1))
        
        # Minus shift
        p_minus = params.copy()
        p_minus[i] -= shift
        shifted_params.append((p_minus, i, -1))
    
    # Define worker function
    def evaluate_shift(p, idx, sign):
        return (idx, sign, circuit_fn(p))
    
    # Run in parallel
    executor = ParallelExecutor(mode=mode, n_workers=n_workers)
    results = executor.map(evaluate_shift, shifted_params)
    
    # Compute gradients
    gradients = np.zeros(n_params)
    for idx, sign, value in results:
        gradients[idx] += sign * value / (2 * np.sin(shift))
    
    return gradients


if __name__ == "__main__":
    # Test parallel modes
    print("Testing parallel modes...")
    print("Available modes:", get_parallel_modes())
    
    # Simple test function
    def slow_square(x):
        time.sleep(0.1)
        return x ** 2
    
    args = [(i,) for i in range(20)]
    print("\nRunning comparison with 20 tasks:")
    results = run_parallel_comparison(slow_square, args, n_workers=4)
    print(format_parallel_comparison(results))
