# Quick Start Guide

## Setup (One-time)

```batch
cd d:\LRET\cirq_comparison\automated_benchmarks
01_setup.bat
```

**What it does:**
- ✓ Installs Python dependencies (cirq, matplotlib, numpy, psutil)
- ✓ Builds qlret_lib.lib (DEPOLARIZE support)
- ✓ Builds quantum_sim.exe
- ✓ Tests basic circuit
- ✓ Tests DEPOLARIZE gate

**Time:** ~2 minutes

## Run Benchmark

```batch
python 02_run_benchmark.py
```

**Configuration:**
- Qubits: 10, 12, 14, 16, 18, 20
- Depth: 20
- Noise: 0.01% per gate
- Epsilon: 1e-6

**Time:** ~30-60 minutes

## Output

Results saved to `results_YYYYMMDD_HHMMSS/`:
- `benchmark.log` - Detailed log
- `results.json` - Raw data
- `REPORT.md` - Analysis
- `*.png` - Plots (4 files)

## Customization

Edit `CONFIG` in `02_run_benchmark.py`:
```python
CONFIG = {
    'qubits': [10, 12, 14, 16, 18, 20],
    'depth': 20,
    'noise_prob': 0.0001,  # 0.01%
    'epsilon': 1e-6,
    'n_trials': 3,
    'timeout': 300,
}
```

## Troubleshooting

**Setup fails:** Check Visual Studio 2019 Build Tools installed  
**Timeout:** Increase `timeout` in CONFIG  
**Out of memory:** Reduce `qubits` range
