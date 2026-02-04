"""
Tier 3 Algorithm Benchmarks - Optional/Advanced Tests
======================================================

Tier 3 algorithms are more specialized:
- VQT: Variational Quantum Thermalizer
- Quantum Walk: Continuous-time quantum walks
- Quantum Kernel Alignment: Learnable kernels
- Sub-sampling QNN: Large-scale QNN training
- Hardware Efficient Ansatz: General-purpose ansatz study
- ADAPT-VQE: Adaptive ansatz construction
"""

from .vqt import VQTBenchmark, run_vqt_benchmark
from .quantum_walk import QuantumWalkBenchmark, run_quantum_walk_benchmark
from .kernel_alignment import KernelAlignmentBenchmark, run_kernel_alignment_benchmark
from .subsampling_qnn import SubsamplingQNNBenchmark, run_subsampling_qnn_benchmark
from .hea import HEABenchmark, run_hea_benchmark
from .adapt_vqe import ADAPTVQEBenchmark, run_adapt_vqe_benchmark

__all__ = [
    # VQT
    'VQTBenchmark', 'run_vqt_benchmark',
    # Quantum Walk
    'QuantumWalkBenchmark', 'run_quantum_walk_benchmark',
    # Kernel Alignment
    'KernelAlignmentBenchmark', 'run_kernel_alignment_benchmark',
    # Sub-sampling QNN
    'SubsamplingQNNBenchmark', 'run_subsampling_qnn_benchmark',
    # HEA
    'HEABenchmark', 'run_hea_benchmark',
    # ADAPT-VQE
    'ADAPTVQEBenchmark', 'run_adapt_vqe_benchmark',
]
