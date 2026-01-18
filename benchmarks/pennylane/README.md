# PennyLane Benchmarking Suite

This directory contains a complete benchmarking suite for comparing LRET's performance against PennyLane's `default.mixed` device.

## 📁 Directory Contents

### Documentation
- **`REQUIREMENTS.md`** - Complete setup guide for fresh systems (Linux/Windows)
  - Manual installation instructions
  - System requirements
  - Troubleshooting guide
  - Quick reference commands

### Automated Setup
- **`setup_pennylane_env.py`** - Cross-platform setup automation script
  - Checks Python version
  - Installs Python dependencies (PennyLane, PyTorch, etc.)
  - Builds LRET C++ backend
  - Installs LRET Python package
  - Verifies device registration
  - Runs smoke test
  
  **Usage:**
  ```bash
  python setup_pennylane_env.py              # Full setup
  python setup_pennylane_env.py --skip-build # Skip C++ build
  python setup_pennylane_env.py --test-only  # Only run verification
  ```

### Benchmark Scripts

All benchmark scripts perform **actual QNN training** with numerical gradient computation, providing realistic performance comparisons.

#### Standard Benchmarks (Renamed with CSV Logging)

All scripts now follow the naming convention: `pennylane_<params>_<mode>.py`

| Script | Qubits | Epochs | Batch | Noise | Est. LRET Time | Est. Baseline Time |
|--------|--------|--------|-------|-------|----------------|-------------------|
| `pennylane_4q_50e_25s_10n.py` | 4 | 50 | 25 | 10% | ~2-3 min | ~6-8 min |
| `pennylane_8q_100e_100s_12n.py` | 8 | 100 | 100 | 12% | ~15-30 min | ~45-90 min (may OOM) |
| `pennylane_8q_200e_200s_15n.py` | 8 | 200 | 200 | 15% | ~30-60 min | Likely OOM |

#### Parallelized Benchmarks (ROW Mode)

ROW mode parallelizes operations across matrix rows.

| Script | Qubits | Epochs | Batch | Noise | Expected Speedup |
|--------|--------|--------|-------|-------|------------------|
| `pennylane_4q_50e_25s_10n_row.py` | 4 | 50 | 25 | 10% | ~2.5-3× |
| `pennylane_8q_100e_100s_12n_row.py` | 8 | 100 | 100 | 12% | ~2.5-3× |
| `pennylane_8q_200e_200s_15n_row.py` | 8 | 200 | 200 | 15% | ∞ (baseline OOM) |

#### Parallelized Benchmarks (HYBRID Mode - Recommended)

HYBRID mode combines row and batch parallelization - best overall performance.

| Script | Qubits | Epochs | Batch | Noise | Expected Speedup |
|--------|--------|--------|-------|-------|------------------|
| `pennylane_4q_50e_25s_10n_hybrid.py` | 4 | 50 | 25 | 10% | ~2.5-3× |
| `pennylane_8q_100e_100s_12n_hybrid.py` | 8 | 100 | 100 | 12% | ~2.5-3× |
| `pennylane_8q_200e_200s_15n_hybrid.py` | 8 | 200 | 200 | 15% | ∞ (baseline OOM) |

#### Comprehensive Comparison Scripts (NEW!)

These scripts test ALL LRET parallelization modes against the baseline in one run:

| Script | Tests | Output Files |
|--------|-------|--------------|
| `pennylane_4q_50e_25s_10n_compare_all.py` | 6 devices (5 LRET modes + baseline) | 6 CSV files + results.json |
| `pennylane_8q_100e_100s_12n_compare_all.py` | 6 devices | 6 CSV files + results.json |
| `pennylane_8q_200e_200s_15n_compare_all.py` | 6 devices | 6 CSV files + results.json |
| `pennylane_parallel_modes_comparison.py` | 5 LRET modes only (no baseline) | 5 CSV files + results.json |

#### What the Benchmarks Measure

Each benchmark performs:
1. **QNN Training Loop**: Actual variational quantum classifier training
2. **Numerical Gradients**: Finite-difference gradient computation (33 circuit calls per sample)
3. **Noisy Simulation**: DepolarizingChannel noise after data encoding
4. **Loss Convergence**: Tracks training loss over epochs

Circuit calls per epoch = `batch_size × (1 + 2 × num_params)` = `25 × 33 = 825` (4-qubit case)

---

## 🔀 Parallelization Options

LRET supports multi-threading via OpenMP. Configure via device parameters:

```python
import pennylane as qml

# Default: HYBRID mode with auto thread count (uses all CPU cores)
dev = qml.device("qlret.mixed", wires=8, epsilon=1e-4)

# Explicit configuration
dev = qml.device("qlret.mixed", wires=8, epsilon=1e-4,
                 num_threads=8,           # 0 = auto (all cores)
                 parallel_mode="hybrid")  # or "row", "sequential"
```

### Parallelization Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `hybrid` | Combined row + batch | General purpose (default) |
| `row` | Parallelize matrix rows | Wide matrices, many qubits |
| `column` | Parallelize matrix columns | Tall matrices |
| `batch` | Parallelize operation batches | Deep circuits |
| `sequential` | Single-threaded | Debugging, baseline comparison |

### Thread Count

- `num_threads=0` (default): Auto-detect, use all available CPU cores
- `num_threads=4`: Use exactly 4 threads
- `num_threads=1`: Single-threaded (same as sequential mode)

---

## 🚀 Quick Start

### For Fresh System Setup

```bash
# 1. Clone/navigate to LRET repository
cd /path/to/LRET

# 2. Switch to pennylane-documentation-benchmarking branch
git checkout pennylane-documentation-benchmarking

# 3. Run automated setup
python benchmarks/pennylane/setup_pennylane_env.py

# 4. Run light benchmark to verify
python benchmarks/pennylane/pennylane_4q_50e_25s_10n.py
```

### For Running Benchmarks (if already set up)

```bash
# Light test (~1-2 hours) - Standard benchmark
python benchmarks/pennylane/pennylane_4q_50e_25s_10n.py

# Light test - Comprehensive comparison (ALL modes)
python benchmarks/pennylane/pennylane_4q_50e_25s_10n_compare_all.py

# Medium test (~3-5 hours)
python benchmarks/pennylane/pennylane_8q_100e_100s_12n.py

# Heavy test (~6-10 hours)
python benchmarks/pennylane/pennylane_8q_200e_200s_15n.py

# Parallelization-only comparison (no baseline)
python benchmarks/pennylane/pennylane_parallel_modes_comparison.py
```

**Note:** All scripts now launch two PowerShell windows:
1. Benchmark execution with live progress
2. CPU monitoring (saves to `cpu_usage.csv`)

---

## 📊 Output Structure

Each benchmark creates a timestamped directory with detailed results:

```
D:/LRET/results/benchmark_YYYYMMDD_HHMMSS/
├── benchmark.log           # Full execution log
├── progress.log            # Training progress only
├── results.json            # Summary statistics (JSON)
├── epochs_lret.csv         # LRET training data per epoch (4q, 8q main scripts)
├── epochs_default_mixed.csv # Baseline training data per epoch
├── epochs_lret_*.csv       # Per-mode epoch data (compare_all scripts)
└── cpu_usage.csv           # CPU monitoring data (NEW!)
```

### CSV Logging (NEW!)

All benchmark scripts now save detailed epoch-by-epoch data to CSV files for analysis:

**Epoch CSV Format** (`epochs_*.csv`):
```csv
epoch,loss,time_seconds,elapsed_seconds,eta_seconds
1,2.456789,1.23,1.23,61.50
2,2.234567,1.18,2.41,59.00
...
```

**CPU Monitoring CSV Format** (`cpu_usage.csv`):
```csv
timestamp,overall_cpu,process_cpu,process_status,core_0,core_1,core_2,...
2026-01-19 12:34:56,45.2,78.5,running,52.1,38.9,46.3,...
```

### CPU Monitoring (NEW!)

The `monitor_cpu.py` script runs automatically in a separate window for all benchmarks:
- Tracks overall CPU usage and per-core utilization
- Monitors the benchmark process specifically
- Saves data to `cpu_usage.csv` with 1-second granularity
- Useful for analyzing parallelization efficiency

**Manual usage:**
```bash
python monitor_cpu.py                    # Live monitoring only
python monitor_cpu.py /path/to/log_dir   # Save to log_dir/cpu_usage.csv
```
└── baseline_epochs.csv     # default.mixed training data per epoch
```

### Sample Output (results.json)
```json
{
  "run_id": "20260116_123456",
  "config": {
    "n_qubits": 4,
    "n_epochs": 50,
    "n_samples": 25,
    "noise_rate": 0.10
  },
  "lret": {
    "status": "completed",
    "total_time_seconds": 3600,
    "avg_epoch_time": 72,
    "final_loss": 0.152,
    "memory_delta_mb": 280
  },
  "baseline": {
    "status": "completed",
    "total_time_seconds": 36000,
    "avg_epoch_time": 720,
    "final_loss": 0.148,
    "memory_delta_mb": 2400
  },
  "summary": {
    "speedup": 10.0,
    "loss_difference": 0.004,
    "lret_faster": true,
    "results_match": true
  }
}
```

---

## 🔧 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.8 | 3.9-3.11 |
| **RAM** | 8 GB | 16 GB+ |
| **Disk** | 2 GB | 5 GB+ |
| **CPU** | 2 cores | 4+ cores |
| **OS** | Ubuntu 20.04 / Win10 | Ubuntu 22.04 / Win11 |

### Required Dependencies
- Python 3.8-3.11
- CMake 3.16+
- C++ compiler (g++ on Linux, Visual Studio on Windows)
- Eigen3 (auto-downloaded by CMake if not found)

### Python Packages
```
pennylane>=0.30
torch
numpy
scipy
psutil>=5.8     # Required for CPU monitoring (monitor_cpu.py)
matplotlib
pandas
```

---

## 📈 Expected Results

### Performance Comparison (Training-Based Benchmarks)

| Benchmark | LRET Time/Epoch | Baseline Time/Epoch | Speedup | Memory LRET | Memory Baseline |
|-----------|-----------------|---------------------|---------|-------------|-----------------|
| Light (4q, 50 epochs) | ~1.5-2s | ~4-5s | ~2.5-3× | ~100-200 MB | ~400-800 MB |
| Medium (8q, 100 epochs) | ~10-20s | ~30-60s (may OOM) | ~2-3× | ~500-800 MB | ~4-8 GB |
| Heavy (8q, 200 epochs) | ~15-25s | OOM (fails) | ∞ | ~800-1500 MB | N/A |

### Key Findings
1. **Training Speedup**: LRET is 2.5-3× faster for actual QNN training workloads
2. **Memory Efficiency**: LRET uses 5-10× less memory than default.mixed
3. **Accuracy**: Loss convergence nearly identical (difference < 0.01)
4. **Scalability**: LRET handles 8+ qubit training; default.mixed struggles or OOMs

### Understanding the Numbers

The benchmarks measure **real QNN training** with gradient computation:
- Each epoch processes `batch_size` samples
- Each sample requires `1 + 2×num_params` circuit evaluations (numerical gradients)
- For 4 qubits: `25 samples × 33 circuits = 825 circuit calls/epoch`
- For 8 qubits: `100 samples × 65 circuits = 6,500 circuit calls/epoch`

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. LRET Device Not Found
```python
DeviceError: Device 'qlret.mixed' not found
```
**Solution:**
```bash
cd python
pip install -e . --force-reinstall
```

#### 2. CMake Not Found
```
'cmake' is not recognized as an internal or external command
```
**Solution:**
- Linux: `sudo apt install cmake`
- Windows: Download from https://cmake.org/download/ and add to PATH

#### 3. Out of Memory (OOM)
```
MemoryError: Unable to allocate array
```
**Expected for default.mixed at 8+ qubits!** This demonstrates LRET's advantage.

#### 4. Build Fails on Windows
```
error: Microsoft Visual C++ 14.0 or greater is required
```
**Solution:** Install Visual Studio Build Tools with "Desktop development with C++"

#### 5. psutil ImportError
```
ModuleNotFoundError: No module named 'psutil'
```
**Solution:**
```bash
pip install psutil>=5.8
```
Note: psutil is now required for CPU monitoring (monitor_cpu.py)

#### 6. CSV Files Not Created
If CSV files (`epochs_*.csv`, `cpu_usage.csv`) are not being created:
- Ensure you're running scripts from the benchmarks/pennylane directory
- Check that the results directory has write permissions
- Verify log_dir is being created: look for "Results directory:" in launcher output

---

## 📝 Notes

### Benchmark Design
- All benchmarks use identical circuits, data, and parameters
- Training uses numerical gradients (finite differences)
- Same random seed ensures reproducibility
- Fair comparison: same hardware, same software versions

### Why These Parameters?
- **4 qubits:** Safe for both devices, shows LRET is competitive
- **8 qubits:** Pushes default.mixed to limits, LRET shines
- **Noise levels:** Realistic for NISQ devices (10-15%)
- **Epochs/samples:** Enough to show training convergence

### Interpretation
- **Speedup > 1:** LRET is faster
- **Loss difference < 0.01:** Results match (both devices work correctly)
- **OOM failure:** Device ran out of memory (expected for default.mixed at 8+ qubits)

---

## 🔗 Additional Resources

- **LRET Documentation:** `../../docs/`
- **PennyLane Docs:** https://pennylane.ai/
- **Parallelization Analysis:** `PARALLELIZATION_ANALYSIS.md`
- **CPU Monitoring Tool:** `monitor_cpu.py`

---

## 📞 Support

For issues or questions:
1. Check `REQUIREMENTS.md` for detailed troubleshooting
2. Review logs in `D:/LRET/results/{script_name}_{timestamp}/benchmark.log`
3. Verify setup with: `python setup_pennylane_env.py --test-only`
4. Check CPU usage: Review `cpu_usage.csv` in results directory

---

*Last updated: January 19, 2026*
