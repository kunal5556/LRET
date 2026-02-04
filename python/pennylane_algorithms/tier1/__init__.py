"""
Tier 1 Algorithm Benchmarks - Must Test

These are the highest priority algorithms for LRET benchmarking.
Each algorithm includes:
- LRET device mode comparison
- Python parallelism comparison
- Comparison with default.mixed and lightning.qubit
"""

from .vqe import run_vqe_benchmark, VQEBenchmark
from .qaoa import run_qaoa_benchmark, QAOABenchmark
from .qnn import run_qnn_benchmark, QNNBenchmark
from .qft import run_qft_benchmark, QFTBenchmark
from .qpe import run_qpe_benchmark, QPEBenchmark
from .grover import run_grover_benchmark, GroverBenchmark
from .metrology import run_metrology_benchmark, MetrologyBenchmark

__all__ = [
    # VQE
    'run_vqe_benchmark',
    'VQEBenchmark',
    # QAOA
    'run_qaoa_benchmark',
    'QAOABenchmark',
    # QNN
    'run_qnn_benchmark',
    'QNNBenchmark',
    # QFT
    'run_qft_benchmark',
    'QFTBenchmark',
    # QPE
    'run_qpe_benchmark',
    'QPEBenchmark',
    # Grover
    'run_grover_benchmark',
    'GroverBenchmark',
    # Metrology
    'run_metrology_benchmark',
    'MetrologyBenchmark',
]
