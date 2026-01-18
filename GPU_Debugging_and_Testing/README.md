# GPU Debugging & Testing Scripts

This directory contains comprehensive testing infrastructure for LRET's hardware-dependent tests, specifically GPU, Multi-GPU, MPI, and NCCL tests that cannot be run on standard development machines.

## Overview

All GPU/MPI tests in LRET currently output only plain text to stdout/stderr. These scripts provide:
- Automated dependency installation
- Systematic test execution
- **CSV output generation** for all test results
- Structured logging and error capture
- HTML reporting with visualization

## Files

### 1. `run_hardware_dependent_tests.py`
**Main test execution script** - A comprehensive Python framework that:
- Detects system capabilities (GPUs, CUDA, MPI, memory)
- Installs dependencies (Linux only)
- Builds tests with appropriate CMake flags
- Runs all hardware-dependent tests
- Captures stdout/stderr for each test
- Parses output to extract metrics
- Generates multiple output formats:
  - `test_results.csv` - Detailed test results
  - `system_info.csv` - Hardware configuration
  - `summary.csv` - Pass/fail statistics
  - `report.html` - Interactive HTML report
  - Per-test folders with logs and metrics

**Usage:**
```bash
# Full run with automatic dependency installation (Linux)
python run_hardware_dependent_tests.py

# Skip installation (dependencies already present)
python run_hardware_dependent_tests.py --skip-install

# Skip build (already built)
python run_hardware_dependent_tests.py --skip-install --skip-build

# Custom output directory
python run_hardware_dependent_tests.py --output-dir /path/to/results

# Run specific tests only
python run_hardware_dependent_tests.py --tests test_distributed_gpu test_multi_gpu_sync
```

### 2. `setup_and_run_hardware_tests.sh`
**Linux automated setup script** - Bash script for fresh system setup:
- Installs NVIDIA drivers (if needed)
- Installs CUDA Toolkit 12.4
- Installs cuDNN and NCCL
- Installs OpenMPI
- Sets up Python virtual environment
- Calls the main test script

**Usage:**
```bash
# Make executable
chmod +x setup_and_run_hardware_tests.sh

# Run with automatic reboot handling
./setup_and_run_hardware_tests.sh

# Skip reboot prompt (not recommended if drivers installed)
./setup_and_run_hardware_tests.sh --skip-reboot

# Custom output directory
./setup_and_run_hardware_tests.sh --output-dir /results/gpu_tests
```

**Supported Platforms:**
- Ubuntu 22.04 LTS
- Ubuntu 24.04 LTS
- Other Debian-based distributions (may require adjustments)

### 3. `setup_and_run_hardware_tests.ps1`
**Windows automated setup script** - PowerShell script for Windows systems:
- Checks NVIDIA drivers
- Installs/verifies CUDA Toolkit
- Installs MS-MPI (Windows MPI implementation)
- Downloads Eigen3
- Sets up Python virtual environment
- Calls the main test script

**Usage:**
```powershell
# Run as Administrator
.\setup_and_run_hardware_tests.ps1

# Skip installation steps
.\setup_and_run_hardware_tests.ps1 -SkipInstall

# Custom output directory
.\setup_and_run_hardware_tests.ps1 -OutputDir "C:\Results\GPU_Tests"
```

## Hardware-Dependent Tests Covered

### GPU Tests (1 GPU required)
- `test_distributed_gpu` - Single GPU scaffold validation

### Multi-GPU Tests (2+ GPUs required)
- `test_distributed_gpu_mpi` - 2 GPUs + MPI + NCCL
- `test_autodiff_multi_gpu` - Multi-GPU autodiff for VQE/QAOA
- `test_multi_gpu_collectives` - NCCL collective operations
- `test_multi_gpu_load_balance` - 4 GPU load balancing
- `test_multi_gpu_sync` - GPU synchronization primitives
- `test_fault_tolerance` - Fault-tolerant simulation

### MPI Tests (MPI required)
- `distributed_mpi_simulation` - Multi-process execution

## Output Structure

After running tests, you'll get a timestamped output directory:

```
test_output_YYYYMMDD_HHMMSS/
├── report.html              # Interactive HTML report (open in browser)
├── test_results.csv         # All test results with metrics
├── system_info.csv          # Hardware/software configuration
├── summary.csv              # Aggregated statistics
│
├── test_distributed_gpu/
│   ├── stdout.log          # Standard output
│   ├── stderr.log          # Error output
│   ├── metrics.json        # Extracted metrics
│   └── summary.txt         # Test summary
│
├── test_autodiff_multi_gpu/
│   └── ...
│
└── ... (one folder per test)
```

## CSV Output Formats

### test_results.csv
| Column | Description |
|--------|-------------|
| Test Name | Name of the test |
| Category | GPU, Multi-GPU, or MPI |
| Status | PASSED, FAILED, SKIPPED, ERROR |
| Duration (s) | Execution time |
| Return Code | Process exit code |
| Error Message | Error details if failed |
| Hardware Requirements | Required hardware |
| Build Flags | CMake build flags used |
| Run Command | Command executed |
| Timestamp | When test ran |
| Pass Count | Number of [PASS] markers |
| Fail Count | Number of [FAIL] markers |
| Fidelity | Extracted fidelity metric |
| Speedup | Extracted speedup metric |

### system_info.csv
System configuration details including:
- OS and kernel version
- CPU model and core count
- Total memory
- GPU count, models, and memory
- CUDA, cuDNN, NCCL versions
- MPI implementation and version
- Build tool versions

### summary.csv
Aggregated statistics:
- Total tests run
- Pass/fail/skip/error counts
- Pass rate percentage
- Total duration

## Requirements

### Hardware Requirements
- **Minimum:** 1 NVIDIA GPU with 8GB+ VRAM
- **Recommended:** 2-4 NVIDIA GPUs with 16GB+ VRAM each
- **Memory:** 32GB+ system RAM
- **CPU:** Multi-core processor (8+ cores recommended)

### Software Requirements

**Linux:**
- Ubuntu 22.04/24.04 or similar
- NVIDIA Driver 535+
- CUDA Toolkit 12.4+
- cuDNN 8.x
- NCCL 2.x
- OpenMPI 4.x or MPICH
- CMake 3.16+
- GCC 9+
- Python 3.9+
- Eigen3

**Windows:**
- Windows 10/11
- NVIDIA Driver 535+
- CUDA Toolkit 12.4+
- Visual Studio 2022 with C++ tools
- MS-MPI
- CMake 3.16+
- Python 3.9+
- Eigen3

## Important Notes

### Original Test Output
The C++ tests in `tests/` directory (test_distributed_gpu.cpp, etc.) **do not** have built-in CSV output. They only print to stdout with messages like:
- `"test_distributed_gpu (single GPU) passed"`
- `"[PASS] AllReduce expectation: 6.0 == 6.0"`
- `"Skipping: requires USE_GPU, USE_MPI, USE_NCCL."`

This test framework **wraps** those tests to:
1. Capture their stdout/stderr
2. Parse output for metrics
3. Generate structured CSV reports
4. Create HTML visualizations

### Why These Tests Are Separate
These tests require:
1. **NVIDIA GPUs** - Not available on all development machines
2. **CUDA/NCCL** - Requires NVIDIA proprietary libraries
3. **Multiple GPUs** - Many tests need 2-4 GPUs
4. **MPI** - Multi-node execution capabilities

Most developers cannot run these on their local machines, hence this dedicated testing infrastructure.

## Troubleshooting

### Test Skipped
If tests show `SKIPPED` status:
- Check GPU count: `nvidia-smi --list-gpus`
- Verify CUDA: `nvcc --version`
- Check MPI: `mpirun --version`
- Review system_info.csv for missing components

### Build Failures
If tests show `ERROR` with build failures:
- Ensure CUDA paths are in environment variables
- Check CMake can find Eigen3
- Verify compiler compatibility (GCC 9+ or MSVC 2019+)

### Runtime Errors
If tests `FAILED`:
- Check stderr.log in test folder for error details
- Verify GPU memory is sufficient
- Check NCCL can communicate between GPUs
- Review MPI hostfile configuration

## Example: Running on Fresh Ubuntu 24.04 System

```bash
# 1. Clone repository
git clone <repo-url>
cd LRET

# 2. Switch to framework-integration branch
git checkout feature/framework-integration

# 3. Run automated setup (installs everything)
cd GPU_Debugging_and_Testing
chmod +x setup_and_run_hardware_tests.sh
./setup_and_run_hardware_tests.sh

# 4. View results
firefox test_output_*/report.html
```

## Integration with CI/CD

For continuous integration on GPU servers:

```yaml
# .github/workflows/gpu-tests.yml (example)
name: GPU Tests
on: [push, pull_request]
jobs:
  gpu-tests:
    runs-on: [self-hosted, linux, gpu]
    steps:
      - uses: actions/checkout@v2
      - name: Run GPU Tests
        run: |
          cd GPU_Debugging_and_Testing
          python run_hardware_dependent_tests.py --skip-install --output-dir $GITHUB_WORKSPACE/gpu_results
      - name: Upload Results
        uses: actions/upload-artifact@v2
        with:
          name: gpu-test-results
          path: gpu_results/
```

## Support

For issues with these scripts:
1. Check that all hardware requirements are met
2. Review the HTML report for detailed error information
3. Examine per-test stderr.log files
4. Verify CUDA/MPI installations with system commands

For issues with the actual LRET tests themselves, refer to:
- `HARDWARE_DEPENDENT_TESTS.md` in project root
- `TESTING_BACKLOG.md` in project root
- Individual test source files in `tests/` directory

## License

Same as LRET project license.
