# LRET Parallel Modes Comparison Benchmark

This folder contains automated benchmarking scripts for comparing **ALL LRET parallelization modes** against each other and against **Cirq's DensityMatrixSimulator**.

## Overview

The benchmark compares:

### LRET Modes
| Mode | Description |
|------|-------------|
| **SEQUENTIAL** | Single-threaded baseline (no parallelism) |
| **ROW** | Row-wise parallel operations on density matrix |
| **COLUMN** | Column-wise parallel operations |
| **BATCH** | Gate batching parallelism |
| **HYBRID** | Combined row + batch (LRET's default mode) |

### Baseline
| Mode | Description |
|------|-------------|
| **CIRQ** | Cirq's DensityMatrixSimulator (full density matrix) |

## Features

- **Dual Terminal Windows**: Opens separate windows for:
  1. **Benchmark Output** - Real-time test progress and results
  2. **CPU Monitor** - Live CPU usage tracking (overall + per-core)

- **Comprehensive Metrics**:
  - Execution time (mean ± std)
  - Memory usage
  - Final rank (LRET)
  - Speedup calculations

- **Rich Visualizations**:
  - Time comparison (all modes)
  - Speedup vs Sequential
  - Speedup vs Cirq
  - Memory comparison
  - Rank evolution
  - Comprehensive 2×3 summary grid

## Quick Start

### 1. Run the Benchmark

```powershell
cd cirq_comparison\automated_benchmarks\parallel_modes_comparison
python run_parallel_modes_benchmark.py
```

This will:
1. Open a new PowerShell window with benchmark output
2. Open another PowerShell window with CPU monitoring
3. Run all parallel modes for qubits 8-20
4. Generate plots and reports

### 2. View Results

Results are saved to `results/run_parallel_modes_benchmark_YYYYMMDD_HHMMSS/`:

- `benchmark.log` - Detailed execution log
- `results.json` - Raw data (JSON format)
- `results.csv` - Tabular data (spreadsheet compatible)
- `REPORT.md` - Summary report with analysis
- `cpu_usage.csv` - CPU monitoring data
- `time_comparison_all.png` - All modes timing plot
- `speedup_vs_sequential.png` - LRET internal speedup
- `speedup_vs_cirq.png` - Speedup over Cirq baseline
- `memory_comparison.png` - Memory usage
- `rank_evolution.png` - LRET rank tracking
- `comprehensive_summary.png` - 2×3 summary grid

## Configuration

Edit `run_parallel_modes_benchmark.py` to customize:

```python
CONFIG = {
    'qubits': [8, 10, 12, 14, 16, 18, 20],  # Qubit range to test
    'depth': 20,                             # CNOT layers
    'noise_prob': 0.0001,                    # 0.01% depolarizing
    'epsilon': 1e-6,                         # Rank truncation threshold
    'n_trials': 3,                           # Trials per configuration
    'timeout': 300,                          # Timeout in seconds
}
```

## Expected Results

For typical configurations:

| Qubits | Best LRET Mode | Expected Speedup vs Cirq |
|--------|----------------|--------------------------|
| 8-10   | HYBRID/BATCH   | 1-2× |
| 12-14  | HYBRID         | 2-5× |
| 16-18  | HYBRID/ROW     | 5-20× |
| 20+    | HYBRID         | 10-100×+ (Cirq may OOM) |

## File Structure

```
parallel_modes_comparison/
├── launcher_utils.py            # Cross-platform terminal launcher
├── monitor_cpu.py               # CPU usage monitoring script
├── run_parallel_modes_benchmark.py  # Main benchmark script
├── README.md                    # This file
└── results/                     # Output directory (auto-created)
    └── run_parallel_modes_benchmark_YYYYMMDD_HHMMSS/
        ├── benchmark.log
        ├── results.json
        ├── results.csv
        ├── cpu_usage.csv
        ├── REPORT.md
        └── *.png
```

## Requirements

- Python 3.8+
- cirq
- matplotlib
- numpy
- psutil

Install with:
```bash
pip install cirq matplotlib numpy psutil
```

## Cross-Platform Support

The scripts work on:
- **Windows**: Opens PowerShell windows
- **Linux**: Uses gnome-terminal, konsole, or xterm
- **macOS**: Uses Terminal.app via AppleScript
