#!/usr/bin/env python3
"""
auto_tune.py — Automated Performance Tuning for LRET Quantum Simulator

Phase 10, Phase 4B: Performance Tuning Infrastructure
From CSV #3, Technique #22: "1.5-3× throughput improvement through empirical tuning"

This script performs Bayesian optimisation over LRET's tunable parameters
using Gaussian Process regression.  It runs the LRET simulator binary on
a set of test circuits, measures wall-clock time, and iteratively refines
the parameter space to find the fastest configuration.

Output:
    tuned_params.json — JSON file consumed by TunedParameters::load_from_file()

Usage:
    # Basic usage (20 trials, default circuits in samples/)
    python scripts/auto_tune.py --binary build/quantum_sim --circuits samples/ --trials 20

    # More exhaustive (100 trials, custom circuit directory)
    python scripts/auto_tune.py --binary build/quantum_sim --circuits my_circuits/ --trials 100

    # Quick smoke test (5 trials)
    python scripts/auto_tune.py --binary build/quantum_sim --circuits samples/ --trials 5 --quick

Dependencies:
    - Python 3.8+
    - numpy
    - scikit-learn (for GaussianProcessRegressor)
    - subprocess (stdlib)

If scikit-learn is not installed, the script falls back to random search.

@see include/tuning_params.h for the C++ parameter struct
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Try to import GP; fall back to random search if unavailable
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    HAS_GP = True
except ImportError:
    HAS_GP = False


# ═══════════════════════════════════════════════════════════════════════
# Parameter Space Definition
# ═══════════════════════════════════════════════════════════════════════

PARAM_SPACE = {
    "batch_size":             [4, 8, 16, 32, 64, 128],
    "truncation_threshold":   [1e-6, 1e-5, 1e-4, 1e-3],
    "openmp_threads":         [1, 2, 4, 8],
    "row_rank_threshold":     [8, 16, 32, 64],
    "column_rank_threshold":  [32, 64, 128, 256],
    "morton_qubit_threshold": [6, 7, 8, 9, 10],
    "morton_min_qubits":      [12, 14, 16],
    "prefetch_distance":      [2, 4, 8, 16],
    "tile_size":              [4, 8, 16, 32, 64],
}


def encode_params(params: Dict[str, Any]) -> np.ndarray:
    """Encode parameter dict to a numeric vector for GP."""
    keys = sorted(PARAM_SPACE.keys())
    vec = []
    for k in keys:
        choices = PARAM_SPACE[k]
        val = params.get(k, choices[len(choices) // 2])
        # Use index in the choice list as feature
        if val in choices:
            vec.append(float(choices.index(val)))
        else:
            # Find nearest
            diffs = [abs(c - val) if isinstance(c, (int, float)) else 0 for c in choices]
            vec.append(float(np.argmin(diffs)))
    return np.array(vec)


def decode_params(vec: np.ndarray) -> Dict[str, Any]:
    """Decode a numeric vector back to parameter dict."""
    keys = sorted(PARAM_SPACE.keys())
    params = {}
    for i, k in enumerate(keys):
        choices = PARAM_SPACE[k]
        idx = int(round(np.clip(vec[i], 0, len(choices) - 1)))
        params[k] = choices[idx]
    return params


# ═══════════════════════════════════════════════════════════════════════
# Benchmark Runner
# ═══════════════════════════════════════════════════════════════════════

def discover_circuits(circuit_dir: str) -> List[str]:
    """Find all .json circuit files in the given directory."""
    p = Path(circuit_dir)
    if not p.exists():
        print(f"Warning: circuit directory '{circuit_dir}' not found")
        return []
    files = sorted(str(f) for f in p.glob("*.json"))
    return files


def run_benchmark(
    binary: str,
    circuit_file: str,
    params: Dict[str, Any],
    timeout: int = 300,
) -> Optional[float]:
    """
    Run the LRET simulator on a circuit file with given parameters.
    Returns wall-clock time in seconds, or None on failure/timeout.
    """
    cmd = [
        binary,
        circuit_file,
        "--batch-size", str(params.get("batch_size", 64)),
        "--threshold", str(params.get("truncation_threshold", 1e-4)),
    ]

    # Add OpenMP thread control via environment
    env = os.environ.copy()
    omp_threads = params.get("openmp_threads", 0)
    if omp_threads > 0:
        env["OMP_NUM_THREADS"] = str(omp_threads)

    try:
        start = time.perf_counter()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        elapsed = time.perf_counter() - start

        if result.returncode != 0:
            return None

        return elapsed

    except subprocess.TimeoutExpired:
        return None
    except FileNotFoundError:
        print(f"Error: binary '{binary}' not found")
        return None


def evaluate_params(
    binary: str,
    circuits: List[str],
    params: Dict[str, Any],
    timeout: int = 300,
    num_repeats: int = 3,
) -> float:
    """
    Evaluate a parameter set by running all circuits and averaging time.
    Returns average wall-clock time (lower is better).
    Returns float('inf') on failure.
    """
    total_time = 0.0
    count = 0

    for circuit in circuits:
        times = []
        for _ in range(num_repeats):
            t = run_benchmark(binary, circuit, params, timeout)
            if t is not None:
                times.append(t)

        if times:
            total_time += np.median(times)
            count += 1

    if count == 0:
        return float("inf")

    return total_time / count


# ═══════════════════════════════════════════════════════════════════════
# Optimisation Strategies
# ═══════════════════════════════════════════════════════════════════════

def random_sample() -> Dict[str, Any]:
    """Sample a random parameter configuration."""
    params = {}
    for k, choices in PARAM_SPACE.items():
        params[k] = choices[np.random.randint(len(choices))]
    return params


def bayesian_optimize(
    binary: str,
    circuits: List[str],
    num_trials: int,
    timeout: int = 300,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], float]:
    """
    Bayesian optimisation using Gaussian Process regression.
    Falls back to random search if sklearn is not available.
    """
    best_params = random_sample()
    best_time = float("inf")

    X_observed: List[np.ndarray] = []
    y_observed: List[float] = []

    if HAS_GP:
        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=0.1)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3)
    else:
        if verbose:
            print("scikit-learn not found; using random search")
        gp = None

    for trial in range(num_trials):
        # ── Select candidate ──
        if trial < 5 or gp is None:
            # Exploration: random sampling for initial points
            candidate = random_sample()
        else:
            # Exploitation: use GP to select promising candidate
            # Generate 100 random candidates, pick the one with
            # the lowest GP predicted mean (exploitation)
            best_predicted = float("inf")
            best_candidate = random_sample()

            for _ in range(100):
                c = random_sample()
                x = encode_params(c).reshape(1, -1)
                mu, _ = gp.predict(x, return_std=True)
                if mu[0] < best_predicted:
                    best_predicted = mu[0]
                    best_candidate = c

            candidate = best_candidate

        # ── Evaluate ──
        avg_time = evaluate_params(binary, circuits, candidate, timeout, num_repeats=2)

        if verbose:
            status = f"  trial {trial+1}/{num_trials}: time={avg_time:.4f}s"
            if avg_time < best_time:
                status += " ★ NEW BEST"
            print(status)

        # ── Update model ──
        x_vec = encode_params(candidate)
        X_observed.append(x_vec)
        y_observed.append(avg_time if avg_time < float("inf") else 1000.0)

        if gp is not None and len(X_observed) >= 3:
            try:
                gp.fit(np.array(X_observed), np.array(y_observed))
            except Exception:
                pass  # GP fit can occasionally fail; continue with exploration

        # ── Track best ──
        if avg_time < best_time:
            best_time = avg_time
            best_params = candidate.copy()

    return best_params, best_time


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Automated Performance Tuning for LRET Quantum Simulator"
    )
    parser.add_argument(
        "--binary", default="build/quantum_sim",
        help="Path to the LRET simulator binary"
    )
    parser.add_argument(
        "--circuits", default="samples/",
        help="Directory containing .json test circuits"
    )
    parser.add_argument(
        "--trials", type=int, default=20,
        help="Number of optimisation trials"
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Per-circuit timeout in seconds"
    )
    parser.add_argument(
        "--output", default="tuned_params.json",
        help="Output JSON file for tuned parameters"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: fewer repeats, shorter timeout"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Print progress"
    )
    args = parser.parse_args()

    if args.quick:
        args.timeout = 60
        if args.trials > 10:
            args.trials = 10

    print(f"LRET Auto-Tuner")
    print(f"  Binary:   {args.binary}")
    print(f"  Circuits: {args.circuits}")
    print(f"  Trials:   {args.trials}")
    print(f"  Timeout:  {args.timeout}s")
    print(f"  Output:   {args.output}")
    print(f"  GP:       {'enabled' if HAS_GP else 'disabled (random search)'}")
    print()

    # Discover circuits
    circuits = discover_circuits(args.circuits)
    if not circuits:
        print("No .json circuit files found. Creating a minimal test circuit...")
        # Create a minimal test circuit for tuning
        minimal = {
            "num_qubits": 8,
            "operations": [
                {"type": "gate", "gate": "H", "qubits": [0]},
                {"type": "gate", "gate": "CNOT", "qubits": [0, 1]},
                {"type": "noise", "noise": "DEPOLARIZING", "qubits": [0], "probability": 0.01},
                {"type": "gate", "gate": "H", "qubits": [2]},
                {"type": "gate", "gate": "CNOT", "qubits": [2, 3]},
            ]
        }
        os.makedirs("samples", exist_ok=True)
        minimal_path = "samples/auto_tune_test.json"
        with open(minimal_path, "w") as f:
            json.dump(minimal, f, indent=2)
        circuits = [minimal_path]

    print(f"  Found {len(circuits)} circuit(s)")
    print()

    # Run optimisation
    best_params, best_time = bayesian_optimize(
        args.binary, circuits, args.trials, args.timeout, args.verbose
    )

    print()
    print(f"═══════════════════════════════════════")
    print(f"Best parameters (avg time = {best_time:.4f}s):")
    for k, v in sorted(best_params.items()):
        print(f"  {k}: {v}")
    print()

    # Save to JSON (with metadata for C++ loader)
    output = {
        "batch_size":             best_params.get("batch_size", 64),
        "truncation_threshold":   best_params.get("truncation_threshold", 1e-4),
        "openmp_threads":         best_params.get("openmp_threads", 0),
        "row_rank_threshold":     best_params.get("row_rank_threshold", 32),
        "column_rank_threshold":  best_params.get("column_rank_threshold", 64),
        "morton_qubit_threshold": best_params.get("morton_qubit_threshold", 8),
        "morton_min_qubits":      best_params.get("morton_min_qubits", 14),
        "morton_min_batch_gates": best_params.get("morton_min_batch_gates", 2),
        "prefetch_distance":      best_params.get("prefetch_distance", 4),
        "tile_size":              best_params.get("tile_size", 8),
        "noise_threshold_scale":  1.0,
        "max_rank_limit":         0,
        "source":                 "auto_tune",
        "version":                "1.0",
        "_meta": {
            "best_time_seconds": best_time,
            "num_trials":        args.trials,
            "num_circuits":      len(circuits),
            "gp_enabled":        HAS_GP,
            "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved to {args.output}")
    print("Load in C++: TunedParameters::load_from_file(\"" + args.output + "\")")


if __name__ == "__main__":
    main()
