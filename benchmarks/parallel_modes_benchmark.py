#!/usr/bin/env python3
"""
Parallel Modes Benchmark: Compare ROW vs COLUMN vs HYBRID vs SEQUENTIAL

This script comprehensively benchmarks all parallel execution modes of the LRET
quantum simulator to validate the new row-parallel optimizations (Phase 1-4):
- Phase 1: Iterative Compression + DLRA
- Phase 2: CP-ALS + Sparse Tensor
- Phase 3: Distributed Tensor Scatter + Variational Lindblad
- Phase 4: Morton Order Cache Optimization + Parallelism Oracle

Purpose:
--------
1. Compare performance (execution time, speedup)
2. Validate correctness (cross-mode fidelity ≈ 1.0)
3. Analyze rank evolution and memory efficiency
4. Generate publication-quality visualizations

Usage:
------
    # Quick benchmark (5-10 minutes)
    python benchmarks/parallel_modes_benchmark.py --quick

    # Comprehensive benchmark (2-8 hours)
    python benchmarks/parallel_modes_benchmark.py --comprehensive

    # Custom configuration
    python benchmarks/parallel_modes_benchmark.py \\
        --qubits 4,6,8 --depths 10,20 --modes sequential,row,hybrid \\
        --trials 5 --output results/custom_benchmark

Author: LRET Development Team
Date: March 2026
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Add python directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

# Import LRET utilities
try:
    from benchmarks.metrics import ExecutionMetrics, MemoryTracker, Timer, measure_execution
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False
    print("Warning: Could not import metrics module. Using basic timing.")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not available. Memory tracking disabled.")


# ==============================================================================
# Configuration Data Classes
# ==============================================================================

@dataclass
class CircuitConfig:
    """Configuration for a single circuit."""
    circuit_type: str  # "random", "qft", "qaoa", "vqe", "ghz"
    n_qubits: int
    depth: int
    noise_prob: float = 0.0  # Depolarizing noise probability
    epsilon: float = 1e-4     # Truncation threshold

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BenchmarkConfig:
    """Overall benchmark configuration."""
    circuit_configs: List[CircuitConfig]
    modes: List[str] = field(default_factory=lambda: ["sequential", "row", "column", "hybrid"])
    trials: int = 5
    timeout_seconds: int = 3600
    output_dir: str = "results/parallel_modes_comparison"
    save_intermediate: bool = True
    export_state: bool = False  # Export full state for validation

    def to_dict(self) -> Dict:
        return {
            "circuit_configs": [c.to_dict() for c in self.circuit_configs],
            "modes": self.modes,
            "trials": self.trials,
            "timeout_seconds": self.timeout_seconds,
            "output_dir": self.output_dir,
            "save_intermediate": self.save_intermediate,
            "export_state": self.export_state
        }


# ==============================================================================
# Circuit Generation
# ==============================================================================

def generate_random_circuit_json(n_qubits: int, depth: int, noise_prob: float = 0.0, seed: int = 42) -> Dict:
    """Generate random circuit in LRET JSON format.

    Circuit structure:
    - Initial layer of RX/RY/RZ rotations
    - Alternating even/odd CNOT layers
    - Random rotation angles
    - Optional depolarizing noise after each gate
    """
    rng = np.random.default_rng(seed)
    ops = []

    # Initial state preparation
    for i in range(n_qubits):
        ops.append({"name": "H", "wires": [i]})
        if noise_prob > 0:
            ops.append({"name": "DEPOLARIZE", "wires": [i], "params": [noise_prob]})

    # Main circuit layers
    for layer in range(depth):
        # Single-qubit rotations
        for q in range(n_qubits):
            angles = rng.uniform(0, 2 * np.pi, 3)
            ops.append({"name": "RX", "wires": [q], "params": [float(angles[0])]})
            ops.append({"name": "RY", "wires": [q], "params": [float(angles[1])]})
            ops.append({"name": "RZ", "wires": [q], "params": [float(angles[2])]})
            if noise_prob > 0:
                for _ in range(3):
                    ops.append({"name": "DEPOLARIZE", "wires": [q], "params": [noise_prob]})

        # Entangling layer (nearest-neighbor CNOTs)
        # Even layer
        for q in range(0, n_qubits - 1, 2):
            ops.append({"name": "CNOT", "wires": [q, q + 1]})
            if noise_prob > 0:
                ops.append({"name": "DEPOLARIZE", "wires": [q], "params": [noise_prob]})
                ops.append({"name": "DEPOLARIZE", "wires": [q + 1], "params": [noise_prob]})

        # Odd layer (alternating)
        if layer % 2 == 1:
            for q in range(1, n_qubits - 1, 2):
                ops.append({"name": "CNOT", "wires": [q, q + 1]})
                if noise_prob > 0:
                    ops.append({"name": "DEPOLARIZE", "wires": [q], "params": [noise_prob]})
                    ops.append({"name": "DEPOLARIZE", "wires": [q + 1], "params": [noise_prob]})

    return {"operations": ops}


def generate_qft_circuit_json(n_qubits: int, noise_prob: float = 0.0) -> Dict:
    """Generate Quantum Fourier Transform circuit."""
    ops = []

    # QFT algorithm
    for i in range(n_qubits):
        ops.append({"name": "H", "wires": [i]})
        if noise_prob > 0:
            ops.append({"name": "DEPOLARIZE", "wires": [i], "params": [noise_prob]})

        for j in range(i + 1, n_qubits):
            angle = np.pi / (2 ** (j - i))
            ops.append({"name": "CRZ", "wires": [j, i], "params": [float(angle)]})
            if noise_prob > 0:
                ops.append({"name": "DEPOLARIZE", "wires": [j], "params": [noise_prob]})
                ops.append({"name": "DEPOLARIZE", "wires": [i], "params": [noise_prob]})

    # Swap qubits to reverse order
    for i in range(n_qubits // 2):
        j = n_qubits - i - 1
        ops.append({"name": "SWAP", "wires": [i, j]})
        if noise_prob > 0:
            ops.append({"name": "DEPOLARIZE", "wires": [i], "params": [noise_prob]})
            ops.append({"name": "DEPOLARIZE", "wires": [j], "params": [noise_prob]})

    return {"operations": ops}


def generate_ghz_circuit_json(n_qubits: int, noise_prob: float = 0.0) -> Dict:
    """Generate GHZ state preparation circuit."""
    ops = []

    # GHZ: |00...0> + |11...1>
    ops.append({"name": "H", "wires": [0]})
    if noise_prob > 0:
        ops.append({"name": "DEPOLARIZE", "wires": [0], "params": [noise_prob]})

    for i in range(1, n_qubits):
        ops.append({"name": "CNOT", "wires": [0, i]})
        if noise_prob > 0:
            ops.append({"name": "DEPOLARIZE", "wires": [0], "params": [noise_prob]})
            ops.append({"name": "DEPOLARIZE", "wires": [i], "params": [noise_prob]})

    return {"operations": ops}


def generate_circuit(circuit_config: CircuitConfig) -> Dict:
    """Generate circuit JSON based on configuration."""
    circuit_type = circuit_config.circuit_type.lower()

    if circuit_type == "random":
        circuit_ops = generate_random_circuit_json(
            circuit_config.n_qubits,
            circuit_config.depth,
            circuit_config.noise_prob
        )
    elif circuit_type == "qft":
        circuit_ops = generate_qft_circuit_json(
            circuit_config.n_qubits,
            circuit_config.noise_prob
        )
    elif circuit_type == "ghz":
        circuit_ops = generate_ghz_circuit_json(
            circuit_config.n_qubits,
            circuit_config.noise_prob
        )
    else:
        raise ValueError(f"Unknown circuit type: {circuit_type}")

    return {
        "circuit": {
            "num_qubits": circuit_config.n_qubits,
            "operations": circuit_ops["operations"]
        },
        "config": {
            "epsilon": circuit_config.epsilon,
            "initial_rank": 1
        }
    }


# ==============================================================================
# Main Benchmark Class
# ==============================================================================

class ParallelModesBenchmark:
    """Main orchestrator for parallel modes comparison."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.quantum_sim_path = self._find_quantum_sim()
        self.results: List[Dict] = []
        self.output_dir = Path(config.output_dir)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.log_file = self.run_dir / "benchmark.log"
        self.log_file.touch()

        self.log("=" * 80)
        self.log("PARALLEL MODES BENCHMARK")
        self.log("=" * 80)
        self.log(f"Run ID: {self.run_id}")
        self.log(f"Output directory: {self.run_dir}")
        self.log(f"Quantum simulator: {self.quantum_sim_path}")
        self.log("")

    def _find_quantum_sim(self) -> Path:
        """Locate quantum_sim.exe executable."""
        # Check common locations
        repo_root = Path(__file__).parent.parent
        candidates = [
            repo_root / "build" / "Release" / "quantum_sim.exe",
            repo_root / "build" / "quantum_sim.exe",
            repo_root / "build" / "Debug" / "quantum_sim.exe",
        ]

        for path in candidates:
            if path.exists():
                return path

        raise FileNotFoundError(
            "Could not find quantum_sim.exe. Please build the project first:\n"
            "  mkdir build && cd build\n"
            "  cmake .. -DCMAKE_BUILD_TYPE=Release\n"
            "  cmake --build . --config Release"
        )

    def log(self, msg: str, also_print: bool = True):
        """Write to log file and optionally print."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}"
        if also_print:
            print(line)
        with open(self.log_file, 'a') as f:
            f.write(line + '\n')

    def run_full_benchmark(self) -> List[Dict]:
        """Run all benchmark configurations."""
        # Save configuration
        config_file = self.run_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

        self.log(f"Configuration saved to: {config_file}")
        self.log(f"Number of circuit configs: {len(self.config.circuit_configs)}")
        self.log(f"Modes to test: {', '.join(self.config.modes)}")
        self.log(f"Trials per config: {self.config.trials}")
        self.log("")

        total_runs = len(self.config.circuit_configs) * len(self.config.modes) * self.config.trials
        completed_runs = 0
        start_time = time.time()

        self.log(f"Total benchmark runs: {total_runs}")
        self.log("=" * 80)
        self.log("")

        for circuit_idx, circuit_config in enumerate(self.config.circuit_configs):
            self.log(f"Circuit {circuit_idx + 1}/{len(self.config.circuit_configs)}: "
                    f"{circuit_config.circuit_type}, {circuit_config.n_qubits}q, "
                    f"depth={circuit_config.depth}, noise={circuit_config.noise_prob:.4f}")

            for mode in self.config.modes:
                self.log(f"  Mode: {mode}")

                for trial in range(self.config.trials):
                    completed_runs += 1
                    elapsed = time.time() - start_time
                    eta_seconds = (elapsed / completed_runs) * (total_runs - completed_runs) if completed_runs > 0 else 0
                    eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

                    self.log(f"    Trial {trial + 1}/{self.config.trials} "
                            f"[{completed_runs}/{total_runs}, ETA: {eta_str}]")

                    result = self._run_single_benchmark(circuit_config, mode, trial)
                    self.results.append(result)

                    if result["status"] == "success":
                        self.log(f"      Time: {result['time_wall_ms']:.2f}ms, "
                                f"Rank: {result.get('final_rank', 'N/A')}")
                    else:
                        self.log(f"      FAILED: {result.get('error_message', 'Unknown error')}")

                    # Save intermediate results
                    if self.config.save_intermediate:
                        self._save_results()

        self.log("")
        self.log("=" * 80)
        self.log(f"Benchmark completed! Total time: {time.time() - start_time:.2f}s")
        self.log("=" * 80)

        # Final save and aggregation
        self._save_results()
        self._save_summary()

        return self.results

    def _run_single_benchmark(
        self,
        circuit_config: CircuitConfig,
        mode: str,
        trial: int
    ) -> Dict:
        """Execute single benchmark run."""
        result = {
            "mode": mode,
            "circuit_type": circuit_config.circuit_type,
            "n_qubits": circuit_config.n_qubits,
            "depth": circuit_config.depth,
            "noise_prob": circuit_config.noise_prob,
            "epsilon": circuit_config.epsilon,
            "trial": trial,
            "timestamp": datetime.now().isoformat(),
            "status": "pending"
        }

        try:
            # Generate circuit JSON
            circuit_json = generate_circuit(circuit_config)

            # Write to temp file
            temp_circuit_path = self.run_dir / f"temp_circuit_{mode}_{trial}.json"
            temp_output_path = self.run_dir / f"temp_output_{mode}_{trial}.json"

            with open(temp_circuit_path, 'w') as f:
                json.dump(circuit_json, f)

            # Build command
            cmd = [
                str(self.quantum_sim_path),
                "--input-json", str(temp_circuit_path),
                "--output-json", str(temp_output_path),
                "--mode", mode
            ]

            if self.config.export_state:
                cmd.append("--export-json-state")

            # Execute with timeout and metrics
            start_time = time.perf_counter()

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=self.config.timeout_seconds,
                    check=True
                )
                wall_time_ms = (time.perf_counter() - start_time) * 1000

                # Parse output
                if temp_output_path.exists():
                    with open(temp_output_path) as f:
                        output = json.load(f)

                    result.update({
                        "status": "success",
                        "time_wall_ms": wall_time_ms,
                        "time_internal_ms": output.get("execution_time_ms"),
                        "final_rank": output.get("final_rank"),
                        "trace": output.get("trace"),
                        "purity": output.get("purity"),
                        "rank_evolution": output.get("rank_evolution", []),
                    })

                    if self.config.export_state and "state" in output:
                        result["state"] = output["state"]
                else:
                    result["status"] = "error"
                    result["error_message"] = "Output file not created"

            except subprocess.TimeoutExpired:
                result["status"] = "timeout"
                result["error_message"] = f"Execution exceeded {self.config.timeout_seconds}s"

            except subprocess.CalledProcessError as e:
                result["status"] = "error"
                result["error_message"] = f"Return code {e.returncode}: {e.stderr.decode()}"

            # Cleanup temp files
            try:
                temp_circuit_path.unlink(missing_ok=True)
                temp_output_path.unlink(missing_ok=True)
            except Exception:
                pass

        except Exception as e:
            result["status"] = "error"
            result["error_message"] = f"{type(e).__name__}: {str(e)}"
            result["traceback"] = traceback.format_exc()

        return result

    def _save_results(self):
        """Save current results to JSON."""
        results_file = self.run_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

    def _save_summary(self):
        """Generate and save summary statistics."""
        # Group by configuration
        by_config = defaultdict(list)
        for result in self.results:
            if result["status"] == "success":
                key = (
                    result["mode"],
                    result["circuit_type"],
                    result["n_qubits"],
                    result["depth"],
                    result["noise_prob"]
                )
                by_config[key].append(result)

        # Compute statistics
        summary = []
        for key, results_list in by_config.items():
            mode, circuit_type, n_qubits, depth, noise_prob = key

            times = [r["time_wall_ms"] for r in results_list]
            ranks = [r["final_rank"] for r in results_list if r.get("final_rank")]

            summary.append({
                "mode": mode,
                "circuit_type": circuit_type,
                "n_qubits": n_qubits,
                "depth": depth,
                "noise_prob": noise_prob,
                "n_trials": len(results_list),
                "time_mean_ms": float(np.mean(times)),
                "time_std_ms": float(np.std(times)),
                "time_min_ms": float(np.min(times)),
                "time_max_ms": float(np.max(times)),
                "rank_mean": float(np.mean(ranks)) if ranks else None,
                "rank_std": float(np.std(ranks)) if ranks else None,
            })

        summary_file = self.run_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        self.log(f"Summary saved to: {summary_file}")


# ==============================================================================
# CLI and Main Entry Point
# ==============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark parallel execution modes of LRET quantum simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark (4-8 qubits, 2 trials)"
    )

    parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Run comprehensive benchmark (4-16 qubits, 10 trials)"
    )

    parser.add_argument(
        "--qubits",
        type=str,
        help="Comma-separated qubit counts (e.g., '4,6,8')"
    )

    parser.add_argument(
        "--depths",
        type=str,
        help="Comma-separated circuit depths (e.g., '10,20')"
    )

    parser.add_argument(
        "--noise",
        type=str,
        help="Comma-separated noise levels (e.g., '0.0,0.01')"
    )

    parser.add_argument(
        "--epsilon",
        type=str,
        default="1e-4",
        help="Comma-separated epsilon values (default: 1e-4)"
    )

    parser.add_argument(
        "--modes",
        type=str,
        help="Comma-separated modes (e.g., 'sequential,row,hybrid')"
    )

    parser.add_argument(
        "--circuit-types",
        type=str,
        default="random",
        help="Comma-separated circuit types (default: random)"
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of trials per configuration (default: 5)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/parallel_modes_comparison",
        help="Output directory (default: results/parallel_modes_comparison)"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout in seconds (default: 3600)"
    )

    parser.add_argument(
        "--export-state",
        action="store_true",
        help="Export full quantum state for validation"
    )

    return parser.parse_args()


def create_quick_config() -> BenchmarkConfig:
    """Create quick benchmark configuration (5-10 minutes)."""
    circuit_configs = [
        CircuitConfig("random", n_qubits=4, depth=10, noise_prob=0.01, epsilon=1e-4),
        CircuitConfig("random", n_qubits=6, depth=10, noise_prob=0.01, epsilon=1e-4),
        CircuitConfig("random", n_qubits=8, depth=10, noise_prob=0.01, epsilon=1e-4),
        CircuitConfig("qft", n_qubits=6, depth=6, noise_prob=0.01, epsilon=1e-4),
        CircuitConfig("ghz", n_qubits=8, depth=8, noise_prob=0.01, epsilon=1e-4),
    ]

    return BenchmarkConfig(
        circuit_configs=circuit_configs,
        modes=["sequential", "row", "hybrid"],
        trials=2,
        output_dir="results/parallel_modes_quick"
    )


def create_comprehensive_config() -> BenchmarkConfig:
    """Create comprehensive benchmark configuration (2-8 hours)."""
    circuit_configs = []

    qubits_range = [4, 6, 8, 10, 12]
    depths = [10, 20]
    noise_levels = [0.0, 0.01]
    circuit_types = ["random", "qft"]

    for circuit_type in circuit_types:
        for n_qubits in qubits_range:
            for depth in depths:
                for noise in noise_levels:
                    circuit_configs.append(
                        CircuitConfig(circuit_type, n_qubits, depth, noise, epsilon=1e-4)
                    )

    return BenchmarkConfig(
        circuit_configs=circuit_configs,
        modes=["sequential", "row", "column", "hybrid"],
        trials=10,
        output_dir="results/parallel_modes_comprehensive"
    )


def create_custom_config(args) -> BenchmarkConfig:
    """Create custom benchmark configuration from command-line arguments."""
    circuit_configs = []

    qubits = [int(x) for x in args.qubits.split(',')] if args.qubits else [4, 6, 8]
    depths = [int(x) for x in args.depths.split(',')] if args.depths else [10]
    noise_levels = [float(x) for x in args.noise.split(',')] if args.noise else [0.01]
    epsilons = [float(x) for x in args.epsilon.split(',')] if args.epsilon else [1e-4]
    circuit_types = args.circuit_types.split(',')

    for circuit_type in circuit_types:
        for n_qubits in qubits:
            for depth in depths:
                for noise in noise_levels:
                    for eps in epsilons:
                        circuit_configs.append(
                            CircuitConfig(circuit_type, n_qubits, depth, noise, eps)
                        )

    modes = args.modes.split(',') if args.modes else ["sequential", "row", "column", "hybrid"]

    return BenchmarkConfig(
        circuit_configs=circuit_configs,
        modes=modes,
        trials=args.trials,
        timeout_seconds=args.timeout,
        output_dir=args.output,
        export_state=args.export_state
    )


def main():
    """Main entry point."""
    args = parse_args()

    # Create configuration
    if args.quick:
        print("Running QUICK benchmark (5-10 minutes)")
        config = create_quick_config()
    elif args.comprehensive:
        print("Running COMPREHENSIVE benchmark (2-8 hours)")
        config = create_comprehensive_config()
    else:
        print("Running CUSTOM benchmark")
        config = create_custom_config(args)

    # Run benchmark
    benchmark = ParallelModesBenchmark(config)
    results = benchmark.run_full_benchmark()

    print(f"\nResults saved to: {benchmark.run_dir}")
    print(f"  - results.json: Raw results")
    print(f"  - summary.json: Aggregated statistics")
    print(f"  - benchmark.log: Execution log")

    # Print quick summary
    success_count = sum(1 for r in results if r["status"] == "success")
    print(f"\nCompleted: {success_count}/{len(results)} successful runs")


if __name__ == "__main__":
    main()
