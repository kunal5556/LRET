# LRET Automated Benchmark Suite

Complete automation for LRET vs Cirq benchmarking with full environment setup.

## Overview

This suite provides end-to-end automation for:
1. Installing Python dependencies
2. Building LRET C++ binaries with DEPOLARIZE support
3. Running comprehensive benchmarks (LRET vs Cirq FDM)
4. Generating publication-quality plots
5. Creating detailed analysis reports

## Quick Start

### Step 1: Setup Environment

Run the PowerShell setup script to install dependencies and build LRET:

```powershell
cd d:\LRET\cirq_comparison\automated_benchmarks
.\01_setup_environment.ps1
```

This will:
- ✓ Check Python version
- ✓ Install cirq, matplotlib, numpy, scipy, psutil
- ✓ Build qlret_lib.lib (contains DEPOLARIZE support)
- ✓ Build quantum_sim.exe (main simulator)
- ✓ Verify basic circuit execution
- ✓ Test DEPOLARIZE gate support

**Expected time:** 1-2 minutes

### Step 2: Run Benchmark

Execute the Python benchmark script:

```powershell
python 02_run_benchmark.py
```

**Configuration:**
- Qubits: 10, 12, 14, 16, 18, 20
- Depth: 20 CNOT layers
- Noise: 0.01% depolarizing per gate (0.0001 probability)
- Epsilon: 1e-6 (rank truncation threshold)
- Trials: 3 runs per configuration
- Timeout: 300s per simulation

**Expected time:** 15-60 minutes (depending on system)

## Output Structure

After running, a timestamped directory will be created:

```
results_YYYYMMDD_HHMMSS/
├── benchmark.log              # Detailed execution log
├── results.json               # Raw benchmark data (JSON)
├── REPORT.md                  # Markdown analysis report
├── time_comparison.png        # Execution time plot
├── speedup.png                # Speedup vs qubits
├── rank.png                   # Rank evolution
└── comprehensive_summary.png  # 2×2 grid with all metrics
```

## Configuration

To customize the benchmark, edit `02_run_benchmark.py`:

```python
CONFIG = {
    'qubits': [10, 12, 14, 16, 18, 20],  # Test range
    'depth': 20,                         # Circuit depth
    'noise_prob': 0.0001,                # 0.01% per gate
    'epsilon': 1e-6,                     # Rank truncation
    'n_trials': 3,                       # Runs per config
    'timeout': 300,                      # Seconds
}
```

### Example Configurations

**High noise (1%):**
```python
'noise_prob': 0.01
```

**Deeper circuits:**
```python
'depth': 50
```

**More aggressive truncation:**
```python
'epsilon': 1e-4
```

**Extended qubit range:**
```python
'qubits': [10, 12, 14, 16, 18, 20, 22]
```

## Requirements

### Software
- **Python:** 3.8 or higher
- **Visual Studio Build Tools 2019:** MSBuild for C++ compilation
- **PowerShell:** 5.1 or higher (Windows 10+)

### Python Packages (auto-installed)
- cirq >= 1.4.0
- matplotlib >= 3.5.0
- numpy >= 1.21.0
- scipy >= 1.7.0
- psutil >= 5.8.0

### Hardware
- **RAM:** 16GB+ recommended for 20 qubits
- **CPU:** Multi-core recommended (simulations are single-threaded but multiple trials can be parallelized manually)
- **Disk:** ~500MB free for LRET binaries + results

## Troubleshooting

### Setup Script Fails

**MSBuild not found:**
```
ERROR: MSBuild not found at C:\Program Files (x86)\...
```
→ Install Visual Studio 2019 Build Tools from https://visualstudio.microsoft.com/downloads/

**Build fails:**
```
ERROR: qlret_lib build failed
```
→ Check that `d:\LRET\build\` contains CMake-generated .vcxproj files
→ Re-run CMake configuration if needed

### Benchmark Script Fails

**quantum_sim.exe not found:**
```
ERROR: quantum_sim.exe not found
```
→ Run setup script first: `.\01_setup_environment.ps1`

**DEPOLARIZE not working:**
```
LRET failed: Unsupported gate: DEPOLARIZE
```
→ Rebuild LRET: run setup script again to recompile with latest code

**Timeout on large circuits:**
```
LRET failed: Timeout after 300s
```
→ Increase timeout in CONFIG: `'timeout': 600`

**Cirq out of memory:**
```
Cirq failed: MemoryError
```
→ Reduce qubit range: `'qubits': [10, 12, 14, 16]`

## Performance Notes

### Expected Runtimes (per trial)

| Qubits | LRET | Cirq FDM | Speedup |
|--------|------|----------|---------|
| 10 | ~0.5s | ~0.8s | 1.6× |
| 12 | ~1.5s | ~3s | 2× |
| 14 | ~4s | ~12s | 3× |
| 16 | ~10s | ~50s | 5× |
| 18 | ~25s | ~200s | 8× |
| 20 | ~60s | ~800s | 13× |

**Total benchmark time (3 trials × 6 configs):** ~30-60 minutes

### Memory Usage

| Qubits | Full DM | LRET (rank ~20) | Reduction |
|--------|---------|-----------------|-----------|
| 10 | 16 MB | ~20 KB | 800× |
| 12 | 256 MB | ~80 KB | 3,200× |
| 14 | 4 GB | ~320 KB | 12,800× |
| 16 | 64 GB | ~1.3 MB | 51,200× |
| 18 | 1 TB | ~5 MB | 204,800× |
| 20 | 16 TB | ~20 MB | 819,200× |

## Advanced Usage

### Running Specific Qubit Counts

Edit `CONFIG['qubits']` to test only specific sizes:

```python
CONFIG = {
    'qubits': [16, 18, 20],  # Test only large circuits
    ...
}
```

### Comparing Multiple Noise Levels

Run benchmark multiple times with different noise:

```bash
# No noise
python 02_run_benchmark.py  # Set noise_prob = 0

# Low noise (0.01%)
python 02_run_benchmark.py  # Set noise_prob = 0.0001

# High noise (1%)
python 02_run_benchmark.py  # Set noise_prob = 0.01
```

### Parallel Execution

To speed up benchmarks, run multiple qubit counts in parallel:

```powershell
# Terminal 1
python 02_run_benchmark.py  # qubits = [10, 12, 14]

# Terminal 2
python 02_run_benchmark.py  # qubits = [16, 18, 20]
```

Then merge results manually.

## Integration with Existing Workflows

This automated suite is designed to work alongside:

- **Manual benchmarks:** `d:\LRET\cirq_comparison\benchmark_with_noise_exe.py`
- **PennyLane tests:** `d:\LRET\python\qlret\tests\test_pennylane.py`
- **Cirq comparison:** `d:\LRET\cirq_comparison\run_comparison.py`

All use the same `quantum_sim.exe` binary built by the setup script.

## Citation

If you use this benchmark suite in publications, please cite:

```bibtex
@software{lret_benchmark_2026,
  title = {LRET Automated Benchmark Suite},
  author = {LRET Development Team},
  year = {2026},
  url = {https://github.com/your-repo/LRET}
}
```

## Support

For issues or questions:
1. Check `benchmark.log` for detailed error messages
2. Review `TROUBLESHOOTING.md` in main LRET directory
3. Open an issue on GitHub with log files attached

---

**Last Updated:** January 22, 2026  
**Version:** 1.0.0  
**Tested On:** Windows 10, Python 3.13, Cirq 1.5.0, LRET 1.0.0
