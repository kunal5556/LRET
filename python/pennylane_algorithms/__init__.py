"""
LRET PennyLane Algorithm Testing Suite
=======================================

Comprehensive benchmark suite for comparing LRET device against
PennyLane's default devices across 20 quantum algorithms.

Structure:
    tier1/  - Must Test (7 algorithms) - Critical benchmarks
    tier2/  - Should Test (7 algorithms) - Important applications  
    tier3/  - Nice to Test (6 algorithms) - Extended coverage
    utils/  - Shared utilities for benchmarking

Usage:
    # Run all benchmarks
    python -m pennylane_algorithms.run_all_benchmarks
    
    # Run specific tier
    python -m pennylane_algorithms.tier1.run_tier1
    
    # Run specific algorithm
    from pennylane_algorithms.tier1.vqe import run_vqe_benchmark
    results = run_vqe_benchmark(n_qubits=4, with_noise=True)

Features:
    - LRET device mode comparison (sequential, batched, parallel)
    - Python parallelism comparison (multiprocessing, threading, joblib)
    - Primary device comparison (default.mixed, lightning.qubit)
    - Noise resilience testing
    - Automated result collection and visualization
"""

__version__ = "1.0.0"
__author__ = "LRET Team"

from . import utils
