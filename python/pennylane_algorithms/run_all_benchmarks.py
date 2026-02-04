"""
Run All PennyLane Algorithm Benchmarks
======================================

Master script to run comprehensive benchmarks across all 20 algorithms.

Usage:
    python run_all_benchmarks.py                    # Run all benchmarks
    python run_all_benchmarks.py --tier 1          # Run only Tier 1
    python run_all_benchmarks.py --tier 1 2        # Run Tier 1 and 2
    python run_all_benchmarks.py --algorithm vqe   # Run specific algorithm
    python run_all_benchmarks.py --quick           # Quick test (1 trial each)
    python run_all_benchmarks.py --full            # Full test (5 trials each)
"""

import argparse
import json
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import asdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from utils.benchmark_utils import BenchmarkResult, save_results_json, format_results_table

# Import all algorithms
from tier1 import (
    run_vqe_benchmark, run_qaoa_benchmark, run_qnn_benchmark,
    run_qft_benchmark, run_qpe_benchmark, run_grover_benchmark,
    run_metrology_benchmark
)
from tier2 import (
    run_uccsd_benchmark, run_portfolio_benchmark, run_qsvm_benchmark,
    run_qae_benchmark, run_vqd_benchmark, run_qgan_benchmark,
    run_number_partitioning_benchmark
)
from tier3 import (
    run_vqt_benchmark, run_quantum_walk_benchmark, run_kernel_alignment_benchmark,
    run_subsampling_qnn_benchmark, run_hea_benchmark, run_adapt_vqe_benchmark
)


# Algorithm registry
ALGORITHMS = {
    # Tier 1 - Must Test
    'vqe': {'fn': run_vqe_benchmark, 'tier': 1, 'name': 'VQE (H2/LiH)', 'qubits': 4},
    'qaoa': {'fn': run_qaoa_benchmark, 'tier': 1, 'name': 'QAOA MaxCut', 'qubits': 6},
    'qnn': {'fn': run_qnn_benchmark, 'tier': 1, 'name': 'QNN Classifier', 'qubits': 4},
    'qft': {'fn': run_qft_benchmark, 'tier': 1, 'name': 'QFT Fidelity', 'qubits': 4},
    'qpe': {'fn': run_qpe_benchmark, 'tier': 1, 'name': 'QPE Accuracy', 'qubits': 5},
    'grover': {'fn': run_grover_benchmark, 'tier': 1, 'name': "Grover's Search", 'qubits': 4},
    'metrology': {'fn': run_metrology_benchmark, 'tier': 1, 'name': 'Quantum Metrology', 'qubits': 4},
    
    # Tier 2 - Should Test
    'uccsd': {'fn': run_uccsd_benchmark, 'tier': 2, 'name': 'UCCSD-VQE', 'qubits': 4},
    'portfolio': {'fn': run_portfolio_benchmark, 'tier': 2, 'name': 'Portfolio Optimization', 'qubits': 6},
    'qsvm': {'fn': run_qsvm_benchmark, 'tier': 2, 'name': 'Quantum SVM', 'qubits': 4},
    'qae': {'fn': run_qae_benchmark, 'tier': 2, 'name': 'Quantum Amplitude Estimation', 'qubits': 6},
    'vqd': {'fn': run_vqd_benchmark, 'tier': 2, 'name': 'Variational Quantum Deflation', 'qubits': 2},
    'qgan': {'fn': run_qgan_benchmark, 'tier': 2, 'name': 'Quantum GAN', 'qubits': 3},
    'number_partitioning': {'fn': run_number_partitioning_benchmark, 'tier': 2, 'name': 'Number Partitioning', 'qubits': 4},
    
    # Tier 3 - Optional
    'vqt': {'fn': run_vqt_benchmark, 'tier': 3, 'name': 'Variational Quantum Thermalizer', 'qubits': 3},
    'quantum_walk': {'fn': run_quantum_walk_benchmark, 'tier': 3, 'name': 'Quantum Walk', 'qubits': 4},
    'kernel_alignment': {'fn': run_kernel_alignment_benchmark, 'tier': 3, 'name': 'Quantum Kernel Alignment', 'qubits': 2},
    'subsampling_qnn': {'fn': run_subsampling_qnn_benchmark, 'tier': 3, 'name': 'Sub-sampling QNN', 'qubits': 4},
    'hea': {'fn': run_hea_benchmark, 'tier': 3, 'name': 'Hardware-Efficient Ansatz', 'qubits': 4},
    'adapt_vqe': {'fn': run_adapt_vqe_benchmark, 'tier': 3, 'name': 'ADAPT-VQE', 'qubits': 4},
}


def get_algorithms_by_tier(tiers: List[int]) -> Dict[str, Dict]:
    """Get algorithms filtered by tier."""
    return {k: v for k, v in ALGORITHMS.items() if v['tier'] in tiers}


def run_benchmark_suite(
    algorithms: Optional[List[str]] = None,
    tiers: Optional[List[int]] = None,
    n_trials: int = 3,
    output_dir: str = 'results',
    verbose: bool = True
) -> Dict[str, Any]:
    """Run the full benchmark suite."""
    
    # Determine which algorithms to run
    if algorithms:
        to_run = {k: v for k, v in ALGORITHMS.items() if k in algorithms}
    elif tiers:
        to_run = get_algorithms_by_tier(tiers)
    else:
        to_run = ALGORITHMS
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    all_results = {
        'metadata': {
            'timestamp': timestamp,
            'n_trials': n_trials,
            'algorithms_run': list(to_run.keys()),
        },
        'results': {}
    }
    
    print("=" * 70)
    print("LRET PennyLane Algorithm Benchmark Suite")
    print("=" * 70)
    print(f"Algorithms to run: {len(to_run)}")
    print(f"Trials per configuration: {n_trials}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    
    for algo_key, algo_info in to_run.items():
        print(f"\n{'='*70}")
        print(f"Running: {algo_info['name']} (Tier {algo_info['tier']})")
        print(f"{'='*70}")
        
        try:
            # Get appropriate qubit count
            n_qubits = algo_info['qubits']
            
            # Run benchmark
            results = algo_info['fn'](n_qubits=n_qubits, n_trials=n_trials)
            
            all_results['results'][algo_key] = {
                'name': algo_info['name'],
                'tier': algo_info['tier'],
                'n_qubits': n_qubits,
                'data': serialize_results(results)
            }
            
            if verbose:
                print_summary(algo_key, results)
                
        except Exception as e:
            print(f"ERROR running {algo_key}: {e}")
            all_results['results'][algo_key] = {
                'name': algo_info['name'],
                'tier': algo_info['tier'],
                'error': str(e)
            }
    
    # Save results
    output_file = os.path.join(output_dir, f'benchmark_results_{timestamp}.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f"Benchmark complete! Results saved to: {output_file}")
    print(f"{'='*70}")
    
    return all_results


def serialize_results(results: Dict) -> Dict:
    """Serialize BenchmarkResult objects to dictionaries."""
    serialized = {}
    for key, value in results.items():
        if isinstance(value, dict):
            serialized[key] = serialize_results(value)
        elif isinstance(value, list):
            serialized[key] = [
                asdict(r) if isinstance(r, BenchmarkResult) else r
                for r in value
            ]
        elif isinstance(value, BenchmarkResult):
            serialized[key] = asdict(value)
        else:
            serialized[key] = value
    return serialized


def print_summary(algo_key: str, results: Dict):
    """Print summary of benchmark results."""
    print(f"\nSummary for {algo_key}:")
    
    if 'lret_modes' in results:
        print("  LRET Modes:")
        for mode, mode_results in results['lret_modes'].items():
            if mode_results:
                avg_time = sum(r.execution_time_seconds for r in mode_results) / len(mode_results)
                success_rate = sum(1 for r in mode_results if r.success) / len(mode_results)
                print(f"    {mode}: avg_time={avg_time:.3f}s, success={success_rate*100:.0f}%")
    
    if 'device_comparison' in results:
        print("  Device Comparison:")
        for device, device_results in results['device_comparison'].items():
            if device_results:
                avg_time = sum(r.execution_time_seconds for r in device_results) / len(device_results)
                success_rate = sum(1 for r in device_results if r.success) / len(device_results)
                print(f"    {device}: avg_time={avg_time:.3f}s, success={success_rate*100:.0f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Run LRET PennyLane Algorithm Benchmarks'
    )
    parser.add_argument(
        '--tier', '-t', type=int, nargs='+',
        help='Tier(s) to run (1, 2, 3)'
    )
    parser.add_argument(
        '--algorithm', '-a', type=str, nargs='+',
        help='Specific algorithm(s) to run'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick test mode (1 trial each)'
    )
    parser.add_argument(
        '--full', action='store_true',
        help='Full test mode (5 trials each)'
    )
    parser.add_argument(
        '--trials', '-n', type=int, default=3,
        help='Number of trials per configuration'
    )
    parser.add_argument(
        '--output', '-o', type=str, default='results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--list', '-l', action='store_true',
        help='List available algorithms'
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable algorithms:")
        print("-" * 60)
        for tier in [1, 2, 3]:
            print(f"\nTier {tier}:")
            for key, info in ALGORITHMS.items():
                if info['tier'] == tier:
                    print(f"  {key:20s} - {info['name']}")
        return
    
    n_trials = args.trials
    if args.quick:
        n_trials = 1
    elif args.full:
        n_trials = 5
    
    run_benchmark_suite(
        algorithms=args.algorithm,
        tiers=args.tier,
        n_trials=n_trials,
        output_dir=args.output,
        verbose=True
    )


if __name__ == "__main__":
    main()
