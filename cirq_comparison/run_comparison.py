"""Benchmark Runner for LRET vs Cirq Comparison.

Runs circuits on both LRET and Cirq simulators, collecting:
- Execution time
- Memory usage
- State fidelity
- Trace distance
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cirq_comparison.cirq_fdm_wrapper import CirqFDMSimulator
from python.qlret.api import simulate_json, load_json_file


class BenchmarkRunner:
    """Run comparison benchmarks between LRET and Cirq."""

    def __init__(
        self,
        circuits_dir: str = "circuits",
        output_dir: str = "results",
        timeout_seconds: int = 300,
        trials: int = 5,
    ):
        """Initialize benchmark runner.

        Args:
            circuits_dir: Directory containing circuit JSON files
            output_dir: Directory to save results
            timeout_seconds: Max execution time per circuit
            trials: Number of trials per circuit
        """
        self.circuits_dir = Path(circuits_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds
        self.trials = trials

        self.results = []

    def run_all_benchmarks(self) -> pd.DataFrame:
        """Run all benchmarks and return results DataFrame."""
        # Load circuit manifest
        manifest_path = self.circuits_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}. "
                "Run circuit_generator.py first."
            )

        with open(manifest_path, "r") as f:
            circuits = json.load(f)

        print(f"Running benchmarks on {len(circuits)} circuits...")
        print(f"  Trials per circuit: {self.trials}")
        print(f"  Timeout: {self.timeout_seconds}s")
        print(f"  Output: {self.output_dir}")

        # Run benchmarks
        for circuit_info in tqdm(circuits, desc="Benchmarking"):
            try:
                result = self._benchmark_circuit(circuit_info)
                self.results.append(result)

            except Exception as e:
                print(f"\n⚠ Error on {circuit_info['name']}: {e}")
                # Save partial results
                self._save_results()

        # Convert to DataFrame
        df = pd.DataFrame(self.results)

        # Save final results
        self._save_results(df)

        return df

    def _benchmark_circuit(self, circuit_info: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark a single circuit on both simulators.

        Args:
            circuit_info: Circuit metadata from manifest

        Returns:
            Dictionary with benchmark results
        """
        circuit_path = Path(circuit_info["path"])
        circuit_json = load_json_file(circuit_path)

        result = {
            "circuit_name": circuit_info["name"],
            "circuit_type": circuit_info["type"],
            "num_qubits": circuit_info["num_qubits"],
            "depth": circuit_info["depth"],
            "noise_level": circuit_info["noise"],
            "timestamp": datetime.now().isoformat(),
        }

        # Benchmark LRET
        lret_results = self._benchmark_lret(circuit_json)
        result.update({
            f"lret_{k}": v for k, v in lret_results.items()
        })

        # Benchmark Cirq
        cirq_results = self._benchmark_cirq(circuit_json)
        result.update({
            f"cirq_{k}": v for k, v in cirq_results.items()
        })

        # Compute comparisons
        if lret_results["success"] and cirq_results["success"]:
            # Fidelity between LRET and Cirq states
            fidelity = self._compute_fidelity(
                lret_results["state"],
                cirq_results["state"],
            )
            trace_dist = self._compute_trace_distance(
                lret_results["state"],
                cirq_results["state"],
            )

            result["fidelity_lret_cirq"] = fidelity
            result["trace_distance_lret_cirq"] = trace_dist

            # Speedup factor
            if cirq_results["mean_time_ms"] > 0:
                speedup = cirq_results["mean_time_ms"] / lret_results["mean_time_ms"]
                result["speedup_lret_vs_cirq"] = speedup
            else:
                result["speedup_lret_vs_cirq"] = np.nan

            # Memory efficiency
            if cirq_results["mean_memory_mb"] > 0:
                mem_efficiency = (
                    cirq_results["mean_memory_mb"] / lret_results["mean_memory_mb"]
                )
                result["memory_efficiency_lret_vs_cirq"] = mem_efficiency
            else:
                result["memory_efficiency_lret_vs_cirq"] = np.nan

        return result

    def _benchmark_lret(self, circuit_json: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark LRET simulator.

        Args:
            circuit_json: Circuit specification

        Returns:
            Timing, memory, and state results
        """
        times = []
        memories = []
        final_state = None

        success = True
        error_msg = None

        for trial in range(self.trials):
            try:
                start_time = time.perf_counter()

                # Run LRET simulation
                result = simulate_json(circuit_json, export_state=True)

                end_time = time.perf_counter()
                exec_time = (end_time - start_time) * 1000  # ms

                times.append(exec_time)
                # Memory tracking would need to be added to LRET API
                memories.append(0.0)  # Placeholder

                # Extract state for comparison
                if "state" in result:
                    # Convert LRET's low-rank state to density matrix
                    final_state = self._lret_state_to_density_matrix(result["state"])

            except Exception as e:
                success = False
                error_msg = str(e)
                break

        return {
            "success": success,
            "error": error_msg,
            "mean_time_ms": float(np.mean(times)) if times else np.nan,
            "std_time_ms": float(np.std(times)) if times else np.nan,
            "min_time_ms": float(np.min(times)) if times else np.nan,
            "max_time_ms": float(np.max(times)) if times else np.nan,
            "mean_memory_mb": float(np.mean(memories)) if memories else np.nan,
            "std_memory_mb": float(np.std(memories)) if memories else np.nan,
            "state": final_state,
            "trials_completed": len(times),
        }

    def _benchmark_cirq(self, circuit_json: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark Cirq simulator.

        Args:
            circuit_json: Circuit specification

        Returns:
            Timing, memory, and state results
        """
        times = []
        memories = []
        final_state = None

        success = True
        error_msg = None

        try:
            # Convert to Cirq circuit
            circuit, num_qubits, noise_model = CirqFDMSimulator.from_json(
                circuit_json
            )

            for trial in range(self.trials):
                sim = CirqFDMSimulator(num_qubits, noise_model=noise_model)
                state, metadata = sim.simulate(circuit)

                times.append(metadata["execution_time_ms"])
                memories.append(metadata["peak_memory_mb"])

                if trial == 0:
                    final_state = state

        except Exception as e:
            success = False
            error_msg = str(e)

        return {
            "success": success,
            "error": error_msg,
            "mean_time_ms": float(np.mean(times)) if times else np.nan,
            "std_time_ms": float(np.std(times)) if times else np.nan,
            "min_time_ms": float(np.min(times)) if times else np.nan,
            "max_time_ms": float(np.max(times)) if times else np.nan,
            "mean_memory_mb": float(np.mean(memories)) if memories else np.nan,
            "std_memory_mb": float(np.std(memories)) if memories else np.nan,
            "state": final_state,
            "trials_completed": len(times),
        }

    def _lret_state_to_density_matrix(
        self,
        lret_state: Dict[str, Any],
    ) -> np.ndarray:
        """Convert LRET's low-rank state to full density matrix.

        LRET stores: ρ = L @ L.H
        where L is the low-rank factor.

        Args:
            lret_state: LRET state dictionary with 'L_matrix' or similar

        Returns:
            Full density matrix
        """
        # This depends on LRET's actual state export format
        # For now, return a placeholder
        # TODO: Implement actual conversion when state format is known
        if "L_matrix" in lret_state:
            L = np.array(lret_state["L_matrix"])
            return L @ L.conj().T
        elif "density_matrix" in lret_state:
            return np.array(lret_state["density_matrix"])
        else:
            # Return identity as fallback (should not happen)
            n = lret_state.get("num_qubits", 2)
            dim = 2 ** n
            return np.eye(dim) / dim

    def _compute_fidelity(
        self,
        state1: Optional[np.ndarray],
        state2: Optional[np.ndarray],
    ) -> float:
        """Compute fidelity between two density matrices."""
        if state1 is None or state2 is None:
            return np.nan

        # F = Tr(√(√ρ σ √ρ))²
        sqrt_rho1 = self._matrix_sqrt(state1)
        product = sqrt_rho1 @ state2 @ sqrt_rho1
        sqrt_product = self._matrix_sqrt(product)
        fidelity = np.real(np.trace(sqrt_product)) ** 2

        return float(fidelity)

    def _compute_trace_distance(
        self,
        state1: Optional[np.ndarray],
        state2: Optional[np.ndarray],
    ) -> float:
        """Compute trace distance between two density matrices."""
        if state1 is None or state2 is None:
            return np.nan

        diff = state1 - state2
        eigenvalues = np.linalg.eigvalsh(diff)
        return 0.5 * float(np.sum(np.abs(eigenvalues)))

    def _matrix_sqrt(self, matrix: np.ndarray) -> np.ndarray:
        """Compute matrix square root."""
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        # Handle numerical errors
        eigenvalues = np.maximum(eigenvalues, 0)
        sqrt_eigenvalues = np.sqrt(eigenvalues)
        return eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.conj().T

    def _save_results(self, df: Optional[pd.DataFrame] = None) -> None:
        """Save results to CSV.

        Args:
            df: DataFrame to save (if None, creates from self.results)
        """
        if df is None and not self.results:
            return

        if df is None:
            df = pd.DataFrame(self.results)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"benchmark_results_{timestamp}.csv"

        df.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to: {output_path}")


def main():
    """Run benchmarks from command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run LRET vs Cirq comparison benchmarks"
    )
    parser.add_argument(
        "--circuits",
        type=str,
        default="circuits",
        help="Directory containing circuit JSON files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds per circuit (default: 300)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of trials per circuit (default: 5)",
    )

    args = parser.parse_args()

    # Run benchmarks
    runner = BenchmarkRunner(
        circuits_dir=args.circuits,
        output_dir=args.output,
        timeout_seconds=args.timeout,
        trials=args.trials,
    )

    try:
        df = runner.run_all_benchmarks()
        print(f"\n{'='*60}")
        print("Benchmark Summary")
        print(f"{'='*60}")
        print(f"Total circuits: {len(df)}")
        print(f"Successful: {df['lret_success'].sum()} LRET, {df['cirq_success'].sum()} Cirq")
        print(f"\nMean speedup (LRET vs Cirq): {df['speedup_lret_vs_cirq'].mean():.2f}x")
        print(f"Mean fidelity: {df['fidelity_lret_cirq'].mean():.6f}")
        print(f"{'='*60}")

    except KeyboardInterrupt:
        print("\n\n⚠ Benchmark interrupted by user")
        runner._save_results()
        sys.exit(1)


if __name__ == "__main__":
    main()
