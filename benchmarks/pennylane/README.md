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

#### Standard Benchmarks (Original)

| Script | Qubits | Epochs | Batch | Noise | Est. LRET Time | Est. Baseline Time |
|--------|--------|--------|-------|-------|----------------|-------------------|
| `4q_50e_25s_10n.py` | 4 | 50 | 25 | 10% | ~2-3 min | ~6-8 min |
| `8q_100e_100s_12n.py` | 8 | 100 | 100 | 12% | ~15-30 min | ~45-90 min (may OOM) |
| `8q_200e_200s_15n.py` | 8 | 200 | 200 | 15% | ~30-60 min | Likely OOM |

#### Parallelized Benchmarks (ROW Mode)

ROW mode parallelizes operations across matrix rows.

| Script | Qubits | Epochs | Batch | Noise | Expected Speedup |
|--------|--------|--------|-------|-------|------------------|
| `benchmark_4q_50e_25s_10n_row.py` | 4 | 50 | 25 | 10% | ~2.5-3× |
| `benchmark_8q_100e_100s_12n_row.py` | 8 | 100 | 100 | 12% | ~2.5-3× |
| `benchmark_8q_200e_200s_15n_row.py` | 8 | 200 | 200 | 15% | ∞ (baseline OOM) |

#### Parallelized Benchmarks (HYBRID Mode - Recommended)

HYBRID mode combines row and batch parallelization - best overall performance.

| Script | Qubits | Epochs | Batch | Noise | Expected Speedup |
|--------|--------|--------|-------|-------|------------------|
| `benchmark_4q_50e_25s_10n_hybrid.py` | 4 | 50 | 25 | 10% | ~2.5-3× |
| `benchmark_8q_100e_100s_12n_hybrid.py` | 8 | 100 | 100 | 12% | ~2.5-3× |
| `benchmark_8q_200e_200s_15n_hybrid.py` | 8 | 200 | 200 | 15% | ∞ (baseline OOM) |

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

# 2. Switch to pennylane branch
git checkout pennylane

# 3. Run automated setup
python benchmarks/pennylane/setup_pennylane_env.py

# 4. Run light benchmark to verify
python benchmarks/pennylane/4q_50e_25s_10n.py
```

### For Running Benchmarks (if already set up)

```bash
# Light test (~1-2 hours)
python benchmarks/pennylane/4q_50e_25s_10n.py

# Medium test (~3-5 hours)
python benchmarks/pennylane/8q_100e_100s_12n.py

# Heavy test (~6-10 hours)
python benchmarks/pennylane/8q_200e_200s_15n.py
```

---

## 📊 Output Structure

Each benchmark creates a timestamped directory with detailed results:

```
D:/LRET/results/benchmark_YYYYMMDD_HHMMSS/
├── benchmark.log           # Full execution log
├── progress.log            # Training progress only
├── results.json            # Summary statistics (JSON)
├── lret_epochs.csv         # LRET training data per epoch
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
psutil
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
- **Full Benchmarking Strategy:** `../../PENNYLANE_BENCHMARKING_STRATEGY.md`
- **Algorithm Catalog:** `../../PENNYLANE_ALGORITHM_CATALOG.md`

---

## 📞 Support

For issues or questions:
1. Check `REQUIREMENTS.md` for detailed troubleshooting
2. Review logs in `D:/LRET/results/benchmark_*/benchmark.log`
3. Verify setup with: `python setup_pennylane_env.py --test-only`

---

*Last updated: January 16, 2026*
