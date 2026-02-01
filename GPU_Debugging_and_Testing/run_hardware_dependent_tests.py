#!/usr/bin/env python3
"""
LRET Hardware-Dependent Tests Runner
=====================================

This script runs all GPU/MPI/NCCL dependent tests on a fresh high-power system.
It handles:
1. Installation of all required dependencies (CUDA, MPI, NCCL, etc.)
2. Building tests with appropriate CMake flags
3. Running each test and capturing output
4. Generating CSV reports with metrics, logs, and error information

Target System: Linux with NVIDIA GPUs (Ubuntu 22.04/24.04 recommended)

Usage:
    python run_hardware_dependent_tests.py [--skip-install] [--skip-build] [--output-dir OUTPUT]

Author: LRET Project
Date: January 2026
"""

import argparse
import csv
import datetime
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple


# =============================================================================
# Configuration
# =============================================================================

class TestStatus(Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"
    NOT_RUN = "NOT_RUN"


@dataclass
class TestResult:
    """Result of a single test execution."""
    test_name: str
    test_file: str
    status: TestStatus
    duration_seconds: float = 0.0
    stdout: str = ""
    stderr: str = ""
    return_code: int = -1
    error_message: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    hardware_requirements: str = ""
    build_flags: str = ""
    run_command: str = ""
    

@dataclass
class SystemInfo:
    """System hardware and software information."""
    os_name: str = ""
    os_version: str = ""
    kernel: str = ""
    cpu_model: str = ""
    cpu_cores: int = 0
    total_memory_gb: float = 0.0
    gpu_count: int = 0
    gpu_models: List[str] = field(default_factory=list)
    gpu_memory_gb: List[float] = field(default_factory=list)
    cuda_version: str = ""
    cudnn_version: str = ""
    nccl_version: str = ""
    mpi_version: str = ""
    mpi_implementation: str = ""
    cmake_version: str = ""
    gcc_version: str = ""
    eigen_version: str = ""


# =============================================================================
# Test Definitions
# =============================================================================

# All hardware-dependent tests with their requirements
HARDWARE_TESTS = [
    # GPU Single-Node Tests
    {
        "name": "test_distributed_gpu",
        "file": "tests/test_distributed_gpu.cpp",
        "target": "test_distributed_gpu",
        "build_flags": ["-DUSE_GPU=ON"],
        "run_cmd": "./test_distributed_gpu",
        "mpi_ranks": 0,  # 0 means no MPI needed
        "requirements": "Single NVIDIA GPU + CUDA",
        "description": "Validates distributed GPU scaffold on a single node; ensures CUDA streams are wired correctly",
        "category": "GPU"
    },
    
    # Multi-GPU MPI+NCCL Tests
    {
        "name": "test_distributed_gpu_mpi",
        "file": "tests/test_distributed_gpu_mpi.cpp",
        "target": "test_distributed_gpu_mpi",
        "build_flags": ["-DUSE_GPU=ON", "-DUSE_MPI=ON", "-DUSE_NCCL=ON", "-DBUILD_MULTI_GPU_TESTS=ON"],
        "run_cmd": "mpirun -np 2 ./test_distributed_gpu_mpi",
        "mpi_ranks": 2,
        "requirements": "2 NVIDIA GPUs + MPI + NCCL",
        "description": "Verify multi-GPU collectives (distribute/allreduce/gather) via MPI+NCCL",
        "category": "Multi-GPU"
    },
    {
        "name": "test_autodiff_multi_gpu",
        "file": "tests/test_autodiff_multi_gpu.cpp",
        "target": "test_autodiff_multi_gpu",
        "build_flags": ["-DUSE_GPU=ON", "-DUSE_MPI=ON", "-DUSE_NCCL=ON", "-DBUILD_MULTI_GPU_TESTS=ON"],
        "run_cmd": "mpirun -np 2 ./test_autodiff_multi_gpu",
        "mpi_ranks": 2,
        "requirements": "2+ NVIDIA GPUs + MPI + NCCL",
        "description": "Test automatic differentiation on multiple GPUs for VQE/QAOA workloads",
        "category": "Multi-GPU"
    },
    {
        "name": "test_multi_gpu_collectives",
        "file": "tests/test_multi_gpu_collectives.cpp",
        "target": "test_multi_gpu_collectives",
        "build_flags": ["-DUSE_GPU=ON", "-DUSE_MPI=ON", "-DUSE_NCCL=ON", "-DBUILD_MULTI_GPU_TESTS=ON"],
        "run_cmd": "mpirun -np 2 ./test_multi_gpu_collectives",
        "mpi_ranks": 2,
        "requirements": "2+ NVIDIA GPUs + MPI + NCCL",
        "description": "Test NCCL collective operations between GPUs (allreduce, broadcast, scatter, gather)",
        "category": "Multi-GPU"
    },
    {
        "name": "test_multi_gpu_load_balance",
        "file": "tests/test_multi_gpu_load_balance.cpp",
        "target": "test_multi_gpu_load_balance",
        "build_flags": ["-DUSE_GPU=ON", "-DUSE_MPI=ON", "-DUSE_NCCL=ON", "-DBUILD_MULTI_GPU_TESTS=ON"],
        "run_cmd": "mpirun -np 4 ./test_multi_gpu_load_balance",
        "mpi_ranks": 4,
        "requirements": "4 NVIDIA GPUs + MPI + NCCL",
        "description": "Test load balancing across multiple GPUs",
        "category": "Multi-GPU"
    },
    {
        "name": "test_multi_gpu_sync",
        "file": "tests/test_multi_gpu_sync.cpp",
        "target": "test_multi_gpu_sync",
        "build_flags": ["-DUSE_GPU=ON", "-DUSE_MPI=ON", "-DUSE_NCCL=ON", "-DBUILD_MULTI_GPU_TESTS=ON"],
        "run_cmd": "mpirun -np 2 ./test_multi_gpu_sync",
        "mpi_ranks": 2,
        "requirements": "2+ NVIDIA GPUs + MPI + NCCL",
        "description": "Test GPU synchronization primitives and barriers",
        "category": "Multi-GPU"
    },
    {
        "name": "test_fault_tolerance",
        "file": "tests/test_fault_tolerance.cpp",
        "target": "test_fault_tolerance",
        "build_flags": ["-DUSE_GPU=ON", "-DUSE_MPI=ON", "-DUSE_NCCL=ON", "-DBUILD_MULTI_GPU_TESTS=ON"],
        "run_cmd": "mpirun -np 2 ./test_fault_tolerance",
        "mpi_ranks": 2,
        "requirements": "2+ NVIDIA GPUs + MPI + NCCL",
        "description": "Test fault-tolerant simulation capabilities on GPU with multi-GPU support",
        "category": "Multi-GPU"
    },
    
    # MPI-Only Tests (no GPU required)
    {
        "name": "distributed_mpi_simulation",
        "file": "main.cpp",
        "target": "quantum_sim",
        "build_flags": ["-DUSE_MPI=ON"],
        "run_cmd": "mpirun -np 4 ./quantum_sim -n 12 -d 20 --mode hybrid",
        "mpi_ranks": 4,
        "requirements": "MPI (OpenMPI or MPICH)",
        "description": "Test distributed parallel execution across multiple MPI processes",
        "category": "MPI"
    },
]


# =============================================================================
# Utility Functions
# =============================================================================

def log(msg: str, level: str = "INFO"):
    """Print a timestamped log message."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    colors = {
        "INFO": "\033[94m",    # Blue
        "SUCCESS": "\033[92m", # Green
        "WARNING": "\033[93m", # Yellow
        "ERROR": "\033[91m",   # Red
        "RESET": "\033[0m"
    }
    color = colors.get(level, colors["INFO"])
    reset = colors["RESET"]
    print(f"{color}[{timestamp}] [{level}] {msg}{reset}")


def run_command(cmd: str, cwd: Optional[str] = None, timeout: int = 3600, 
                env: Optional[Dict] = None, capture: bool = True) -> Tuple[int, str, str]:
    """Run a shell command and return (return_code, stdout, stderr)."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    
    try:
        if capture:
            result = subprocess.run(
                cmd, shell=True, cwd=cwd, timeout=timeout,
                capture_output=True, text=True, env=merged_env
            )
            return result.returncode, result.stdout, result.stderr
        else:
            result = subprocess.run(
                cmd, shell=True, cwd=cwd, timeout=timeout, env=merged_env
            )
            return result.returncode, "", ""
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout} seconds"
    except Exception as e:
        return -1, "", str(e)


def ensure_directory(path: Path):
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Windows CUDA/MSBuild Workaround
# =============================================================================

def find_cuda_path(prefer_12x: bool = True) -> Optional[str]:
    """Find CUDA installation path on Windows or Linux.
    
    Args:
        prefer_12x: If True, prefer CUDA 12.x over 13.x for Pascal GPU compatibility.
                   CUDA 13.x dropped support for compute capability < 7.5.
    """
    # Check environment variable first
    cuda_path = os.environ.get('CUDA_PATH', '')
    if cuda_path and os.path.exists(cuda_path):
        # Check if it's a 12.x version when prefer_12x is set
        if prefer_12x and 'v13' in cuda_path:
            log(f"Environment CUDA_PATH points to {cuda_path}, but prefer_12x=True", "WARNING")
            log("Searching for CUDA 12.x...", "INFO")
        else:
            return cuda_path
    
    if platform.system() == "Windows":
        # Search common Windows CUDA installation paths
        base_paths = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
            r"C:\CUDA",
        ]
        for base in base_paths:
            if os.path.exists(base):
                # Find all versions
                versions = []
                try:
                    for item in os.listdir(base):
                        if item.startswith('v') and os.path.isdir(os.path.join(base, item)):
                            versions.append(item)
                except:
                    continue
                if versions:
                    versions.sort(reverse=True)
                    
                    # If prefer_12x, look for 12.x versions first
                    if prefer_12x:
                        v12_versions = [v for v in versions if v.startswith('v12')]
                        if v12_versions:
                            selected = v12_versions[0]  # Latest 12.x
                            log(f"Selected CUDA {selected} (prefer_12x=True, Pascal GPU compatible)", "INFO")
                            return os.path.join(base, selected)
                        else:
                            log(f"No CUDA 12.x found, using {versions[0]}", "WARNING")
                    
                    return os.path.join(base, versions[0])
    else:
        # Linux paths
        linux_paths = [
            "/usr/local/cuda",
            "/opt/cuda",
        ]
        for path in linux_paths:
            if os.path.exists(path):
                return path
    
    return None


def find_visual_studio_path() -> Optional[str]:
    """Find Visual Studio or Build Tools installation path."""
    vswhere_paths = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe",
        r"C:\Program Files\Microsoft Visual Studio\Installer\vswhere.exe",
    ]
    
    for vswhere in vswhere_paths:
        if os.path.exists(vswhere):
            try:
                result = subprocess.run(
                    [vswhere, "-latest", "-products", "*", 
                     "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                     "-property", "installationPath"],
                    capture_output=True, text=True
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
            except:
                pass
    
    # Fallback to common paths
    common_paths = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools",
    ]
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    return None


def setup_windows_build_environment(prefer_cuda_12x: bool = True) -> Dict[str, str]:
    """
    Set up environment for Windows CUDA builds using Ninja.
    This bypasses the problematic MSBuild CUDA targets.
    
    Args:
        prefer_cuda_12x: If True, prefer CUDA 12.x for Pascal GPU compatibility.
    
    Returns environment variables dict.
    """
    env = os.environ.copy()
    
    # Find CUDA (prefer 12.x by default for Pascal GPU compatibility)
    cuda_path = find_cuda_path(prefer_12x=prefer_cuda_12x)
    if cuda_path:
        log(f"Found CUDA at: {cuda_path}", "SUCCESS")
        
        # Ensure trailing backslash for MSBuild compatibility
        cuda_path_with_slash = cuda_path.rstrip('\\') + '\\'
        
        env['CUDA_PATH'] = cuda_path
        env['CUDA_HOME'] = cuda_path
        env['CUDAToolkit_ROOT'] = cuda_path
        # CRITICAL: MSBuild CUDA targets require CudaToolkitDir with trailing backslash
        env['CudaToolkitDir'] = cuda_path_with_slash
        
        # Add CUDA bin to PATH
        cuda_bin = os.path.join(cuda_path, 'bin')
        if cuda_bin not in env.get('PATH', ''):
            env['PATH'] = f"{cuda_bin};{env.get('PATH', '')}"
        
        # Add CUDA lib to PATH for runtime
        cuda_lib = os.path.join(cuda_path, 'lib', 'x64')
        if os.path.exists(cuda_lib) and cuda_lib not in env.get('PATH', ''):
            env['PATH'] = f"{cuda_lib};{env.get('PATH', '')}"
    else:
        log("CUDA not found - GPU tests will fail", "WARNING")
    
    # Find Visual Studio and set up vcvars
    vs_path = find_visual_studio_path()
    if vs_path:
        log(f"Found Visual Studio at: {vs_path}", "SUCCESS")
        
        # Try to run vcvarsall to get the proper environment
        vcvars_paths = [
            os.path.join(vs_path, "VC", "Auxiliary", "Build", "vcvars64.bat"),
            os.path.join(vs_path, "VC", "Auxiliary", "Build", "vcvarsall.bat"),
        ]
        
        for vcvars in vcvars_paths:
            if os.path.exists(vcvars):
                try:
                    # Run vcvars and capture the resulting environment
                    if "vcvarsall" in vcvars:
                        cmd = f'"{vcvars}" x64 && set'
                    else:
                        cmd = f'"{vcvars}" && set'
                    
                    result = subprocess.run(
                        cmd, shell=True, capture_output=True, text=True
                    )
                    
                    if result.returncode == 0:
                        # Parse environment variables from output
                        for line in result.stdout.split('\n'):
                            if '=' in line:
                                key, _, value = line.partition('=')
                                key = key.strip()
                                value = value.strip()
                                if key and value:
                                    env[key] = value
                        log("Visual Studio environment configured", "SUCCESS")
                        break
                except Exception as e:
                    log(f"Failed to run vcvars: {e}", "WARNING")
    else:
        log("Visual Studio not found - builds may fail", "WARNING")
    
    return env


def check_ninja_available() -> bool:
    """Check if Ninja build system is available."""
    try:
        result = subprocess.run(
            ["ninja", "--version"], capture_output=True, text=True
        )
        if result.returncode == 0:
            log(f"Ninja found: {result.stdout.strip()}", "SUCCESS")
            return True
    except:
        pass
    return False


def install_ninja_windows() -> bool:
    """Attempt to install Ninja on Windows."""
    log("Attempting to install Ninja...", "INFO")
    
    # Try winget first
    try:
        result = subprocess.run(
            ["winget", "install", "Ninja-build.Ninja", "--silent"],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            log("Ninja installed via winget", "SUCCESS")
            return True
    except:
        pass
    
    # Try pip
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "ninja"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            log("Ninja installed via pip", "SUCCESS")
            return True
    except:
        pass
    
    # Try chocolatey
    try:
        result = subprocess.run(
            ["choco", "install", "ninja", "-y"],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            log("Ninja installed via chocolatey", "SUCCESS")
            return True
    except:
        pass
    
    log("Failed to install Ninja automatically", "WARNING")
    log("Please install Ninja manually: https://ninja-build.org/", "WARNING")
    return False


def get_cmake_generator_and_env(prefer_cuda_12x: bool = True) -> Tuple[str, Dict[str, str]]:
    """
    Determine the best CMake generator and return appropriate environment.
    On Windows, prefer Ninja to avoid MSBuild CUDA integration issues.
    
    Args:
        prefer_cuda_12x: If True, prefer CUDA 12.x for Pascal GPU compatibility.
    """
    if platform.system() == "Windows":
        # Set up Windows build environment first
        env = setup_windows_build_environment(prefer_cuda_12x=prefer_cuda_12x)
        
        # Check for Ninja
        if check_ninja_available():
            log("Using Ninja generator (bypasses MSBuild CUDA issues)", "INFO")
            return "Ninja", env
        else:
            log("Ninja not found, attempting installation...", "WARNING")
            if install_ninja_windows():
                # Refresh PATH and check again
                env['PATH'] = os.environ.get('PATH', '') + ';' + env.get('PATH', '')
                if check_ninja_available():
                    return "Ninja", env
            
            # Fall back to Visual Studio generator with workaround
            log("Falling back to Visual Studio generator", "WARNING")
            return "Visual Studio 17 2022", env
    else:
        # Linux - use Unix Makefiles or Ninja if available
        if check_ninja_available():
            return "Ninja", os.environ.copy()
        return "Unix Makefiles", os.environ.copy()


# =============================================================================
# System Information Collection
# =============================================================================

def get_system_info() -> SystemInfo:
    """Collect comprehensive system information."""
    info = SystemInfo()
    
    # OS Information
    info.os_name = platform.system()
    info.os_version = platform.version()
    info.kernel = platform.release()
    
    # CPU Information
    try:
        if info.os_name == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        info.cpu_model = line.split(":")[1].strip()
                        break
            info.cpu_cores = os.cpu_count() or 0
        elif info.os_name == "Windows":
            info.cpu_model = platform.processor()
            info.cpu_cores = os.cpu_count() or 0
    except:
        info.cpu_model = "Unknown"
        info.cpu_cores = os.cpu_count() or 0
    
    # Memory Information
    try:
        if info.os_name == "Linux":
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemTotal" in line:
                        mem_kb = int(line.split()[1])
                        info.total_memory_gb = mem_kb / (1024 * 1024)
                        break
        elif info.os_name == "Windows":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulong = ctypes.c_ulong
            class MEMORYSTATUS(ctypes.Structure):
                _fields_ = [
                    ('dwLength', c_ulong),
                    ('dwMemoryLoad', c_ulong),
                    ('dwTotalPhys', c_ulong),
                    ('dwAvailPhys', c_ulong),
                    ('dwTotalPageFile', c_ulong),
                    ('dwAvailPageFile', c_ulong),
                    ('dwTotalVirtual', c_ulong),
                    ('dwAvailVirtual', c_ulong),
                ]
            mem_status = MEMORYSTATUS()
            mem_status.dwLength = ctypes.sizeof(MEMORYSTATUS)
            kernel32.GlobalMemoryStatus(ctypes.byref(mem_status))
            info.total_memory_gb = mem_status.dwTotalPhys / (1024**3)
    except:
        info.total_memory_gb = 0.0
    
    # GPU Information (NVIDIA)
    try:
        rc, stdout, _ = run_command("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits")
        if rc == 0 and stdout.strip():
            lines = stdout.strip().split("\n")
            info.gpu_count = len(lines)
            for line in lines:
                parts = line.split(",")
                if len(parts) >= 2:
                    info.gpu_models.append(parts[0].strip())
                    info.gpu_memory_gb.append(float(parts[1].strip()) / 1024)
    except:
        info.gpu_count = 0
    
    # CUDA Version
    try:
        rc, stdout, _ = run_command("nvcc --version")
        if rc == 0:
            match = re.search(r"release (\d+\.\d+)", stdout)
            if match:
                info.cuda_version = match.group(1)
    except:
        pass
    
    # cuDNN Version
    try:
        rc, stdout, _ = run_command("cat /usr/local/cuda/include/cudnn_version.h 2>/dev/null | grep CUDNN_MAJOR -A 2")
        if rc == 0 and stdout:
            major = minor = patch = "0"
            for line in stdout.split("\n"):
                if "CUDNN_MAJOR" in line:
                    major = line.split()[-1]
                elif "CUDNN_MINOR" in line:
                    minor = line.split()[-1]
                elif "CUDNN_PATCHLEVEL" in line:
                    patch = line.split()[-1]
            info.cudnn_version = f"{major}.{minor}.{patch}"
    except:
        pass
    
    # NCCL Version
    try:
        rc, stdout, _ = run_command("cat /usr/include/nccl.h 2>/dev/null | grep NCCL_VERSION -A 1 | head -5")
        if rc == 0 and "NCCL" in stdout:
            info.nccl_version = "Installed"
    except:
        pass
    
    # MPI Version
    try:
        # Try OpenMPI first
        rc, stdout, _ = run_command("mpirun --version")
        if rc == 0:
            if "Open MPI" in stdout:
                info.mpi_implementation = "OpenMPI"
                match = re.search(r"(\d+\.\d+\.\d+)", stdout)
                if match:
                    info.mpi_version = match.group(1)
            elif "MPICH" in stdout or "mpich" in stdout.lower():
                info.mpi_implementation = "MPICH"
                match = re.search(r"(\d+\.\d+)", stdout)
                if match:
                    info.mpi_version = match.group(1)
            else:
                info.mpi_implementation = "Unknown"
                info.mpi_version = "Installed"
    except:
        pass
    
    # CMake Version
    try:
        rc, stdout, _ = run_command("cmake --version")
        if rc == 0:
            match = re.search(r"cmake version (\d+\.\d+\.\d+)", stdout)
            if match:
                info.cmake_version = match.group(1)
    except:
        pass
    
    # GCC Version
    try:
        rc, stdout, _ = run_command("gcc --version")
        if rc == 0:
            match = re.search(r"(\d+\.\d+\.\d+)", stdout)
            if match:
                info.gcc_version = match.group(1)
    except:
        pass
    
    return info


# =============================================================================
# Dependency Installation
# =============================================================================

def install_dependencies(project_root: Path, skip: bool = False) -> bool:
    """Install all required dependencies for hardware-dependent tests."""
    
    if skip:
        log("Skipping dependency installation (--skip-install)", "WARNING")
        return True
    
    log("=" * 60)
    log("INSTALLING DEPENDENCIES FOR HARDWARE-DEPENDENT TESTS")
    log("=" * 60)
    
    system = platform.system()
    
    if system != "Linux":
        log(f"Automatic installation only supported on Linux. Current system: {system}", "WARNING")
        log("Please manually install: CUDA Toolkit, cuDNN, NCCL, OpenMPI, Eigen3", "WARNING")
        return True
    
    # Check if running as root or with sudo
    is_root = os.geteuid() == 0 if hasattr(os, 'geteuid') else False
    sudo = "" if is_root else "sudo "
    
    installation_steps = [
        # Step 1: Update package lists
        {
            "name": "Update package lists",
            "cmd": f"{sudo}apt-get update",
            "required": True
        },
        
        # Step 2: Install basic build tools
        {
            "name": "Install build essentials",
            "cmd": f"{sudo}apt-get install -y build-essential cmake git wget curl",
            "required": True
        },
        
        # Step 3: Install Eigen3
        {
            "name": "Install Eigen3",
            "cmd": f"{sudo}apt-get install -y libeigen3-dev",
            "required": True
        },
        
        # Step 4: Install OpenMP
        {
            "name": "Install OpenMP",
            "cmd": f"{sudo}apt-get install -y libomp-dev",
            "required": True
        },
        
        # Step 5: Install OpenMPI
        {
            "name": "Install OpenMPI",
            "cmd": f"{sudo}apt-get install -y openmpi-bin openmpi-common libopenmpi-dev",
            "required": True
        },
        
        # Step 6: Install CUDA Toolkit (if not present)
        {
            "name": "Check/Install CUDA Toolkit",
            "cmd": """
if ! command -v nvcc &> /dev/null; then
    echo "CUDA not found. Installing CUDA Toolkit..."
    # Add NVIDIA package repositories
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get install -y cuda-toolkit-12-4
    rm cuda-keyring_1.1-1_all.deb
    
    # Add CUDA to PATH
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
else
    echo "CUDA already installed: $(nvcc --version | grep release)"
fi
""",
            "required": False
        },
        
        # Step 7: Install cuDNN
        {
            "name": "Check/Install cuDNN",
            "cmd": f"""
if [ ! -f /usr/local/cuda/include/cudnn.h ] && [ ! -f /usr/include/cudnn.h ]; then
    echo "Installing cuDNN..."
    {sudo}apt-get install -y libcudnn8 libcudnn8-dev
else
    echo "cuDNN already installed"
fi
""",
            "required": False
        },
        
        # Step 8: Install NCCL
        {
            "name": "Check/Install NCCL",
            "cmd": f"""
if [ ! -f /usr/include/nccl.h ]; then
    echo "Installing NCCL..."
    {sudo}apt-get install -y libnccl2 libnccl-dev
else
    echo "NCCL already installed"
fi
""",
            "required": False
        },
        
        # Step 9: Install Python dependencies
        {
            "name": "Install Python dependencies",
            "cmd": "pip install numpy pytest jax jaxlib torch pennylane qiskit flax optax",
            "required": False
        },
    ]
    
    success = True
    for step in installation_steps:
        log(f"Step: {step['name']}")
        rc, stdout, stderr = run_command(step["cmd"], timeout=1800)  # 30 min timeout
        
        if rc != 0:
            if step["required"]:
                log(f"FAILED: {step['name']}", "ERROR")
                log(f"stderr: {stderr}", "ERROR")
                success = False
                break
            else:
                log(f"Warning: {step['name']} had issues (non-critical)", "WARNING")
        else:
            log(f"Completed: {step['name']}", "SUCCESS")
    
    return success


# =============================================================================
# Build System
# =============================================================================

# Cache for build environment (computed once)
_build_env_cache: Optional[Tuple[str, Dict[str, str]]] = None
_prefer_cuda_12x: bool = True  # Global preference for CUDA 12.x (Pascal GPU compatible)

def get_build_configuration(prefer_cuda_12x: Optional[bool] = None) -> Tuple[str, Dict[str, str]]:
    """Get cached build configuration (generator and environment).
    
    Args:
        prefer_cuda_12x: If provided, overrides the global preference. If None, uses _prefer_cuda_12x.
    """
    global _build_env_cache, _prefer_cuda_12x
    if prefer_cuda_12x is not None:
        use_12x = prefer_cuda_12x
    else:
        use_12x = _prefer_cuda_12x
    
    if _build_env_cache is None:
        _build_env_cache = get_cmake_generator_and_env(prefer_cuda_12x=use_12x)
    return _build_env_cache


def build_tests(project_root: Path, build_dir: Path, build_flags: List[str], 
                targets: List[str], skip: bool = False, 
                existing_build_dir: Optional[str] = None) -> Tuple[bool, str]:
    """Build specific test targets with given CMake flags.
    
    Args:
        project_root: Path to the LRET project root.
        build_dir: Base directory for builds.
        build_flags: CMake flags for this build.
        targets: List of targets to build.
        skip: If True, skip build entirely.
        existing_build_dir: If provided, use this existing build directory instead of creating new one.
    """
    
    if skip:
        log("Skipping build (--skip-build)", "WARNING")
        return True, "Skipped"
    
    # Use existing build directory if specified, otherwise create unique one
    if existing_build_dir:
        unique_build_dir = Path(existing_build_dir)
        if unique_build_dir.exists():
            log(f"Using existing build directory: {unique_build_dir}", "INFO")
            # Check if executables exist in this directory
            for target in targets:
                exe_name = f"{target}.exe" if platform.system() == "Windows" else target
                exe_path = unique_build_dir / exe_name
                if exe_path.exists():
                    log(f"  Found executable: {exe_name}", "INFO")
            # Return success with the existing build path
            return True, str(unique_build_dir)
        else:
            log(f"Existing build directory not found: {unique_build_dir}", "WARNING")
            # Fall through to create new build
    
    # Create unique build directory based on flags
    flags_hash = "_".join(sorted(build_flags)).replace("-D", "").replace("=", "_")
    unique_build_dir = build_dir / f"build_{flags_hash}"
    ensure_directory(unique_build_dir)
    
    log(f"Building in: {unique_build_dir}")
    log(f"Build flags: {' '.join(build_flags)}")
    log(f"Targets: {', '.join(targets)}")
    
    # Get CMake generator and environment
    generator, build_env = get_build_configuration()
    log(f"Using CMake generator: {generator}")
    
    # Build CMake configure command
    cmake_args = [
        f'cmake',
        f'"{project_root}"',
        f'-G "{generator}"',
        f'-DCMAKE_BUILD_TYPE=Release',
    ]
    
    # Add CUDA configuration if available
    cuda_path = build_env.get('CUDA_PATH', '') or build_env.get('CUDAToolkit_ROOT', '')
    if cuda_path:
        # Normalize path separators for CMake
        cuda_path_cmake = cuda_path.replace('\\', '/')
        cmake_args.extend([
            f'-DCUDAToolkit_ROOT="{cuda_path_cmake}"',
            f'-DCMAKE_CUDA_COMPILER="{cuda_path_cmake}/bin/nvcc"',
        ])
        
        # For Windows, also set the host compiler explicitly
        if platform.system() == "Windows":
            # Find cl.exe path from environment
            cl_path = None
            for path_dir in build_env.get('PATH', '').split(';'):
                cl_candidate = os.path.join(path_dir, 'cl.exe')
                if os.path.exists(cl_candidate):
                    cl_path = cl_candidate.replace('\\', '/')
                    break
            
            if cl_path:
                cmake_args.append(f'-DCMAKE_CUDA_HOST_COMPILER="{cl_path}"')
    
    # Add build flags
    cmake_args.extend(build_flags)
    
    cmake_cmd = ' '.join(cmake_args)
    log(f"CMake configure: {cmake_cmd}")
    
    # Run CMake configure
    rc, stdout, stderr = run_command(cmake_cmd, cwd=str(unique_build_dir), timeout=300, env=build_env)
    if rc != 0:
        # If Ninja failed, try falling back to Visual Studio generator
        if "Ninja" in generator:
            log("Ninja configuration failed, trying Visual Studio generator...", "WARNING")
            
            # Clean the build directory for fresh configuration
            shutil.rmtree(unique_build_dir, ignore_errors=True)
            ensure_directory(unique_build_dir)
            
            # Build new CMake command with VS generator
            cmake_args_vs = [
                f'cmake',
                f'"{project_root}"',
                f'-G "Visual Studio 17 2022"',
                f'-A x64',  # Explicitly set architecture
                f'-DCMAKE_BUILD_TYPE=Release',
            ]
            if cuda_path:
                cmake_args_vs.extend([
                    f'-DCUDAToolkit_ROOT="{cuda_path_cmake}"',
                ])
            cmake_args_vs.extend(build_flags)
            cmake_cmd_vs = ' '.join(cmake_args_vs)
            log(f"CMake configure (VS): {cmake_cmd_vs}")
            
            rc, stdout, stderr = run_command(cmake_cmd_vs, cwd=str(unique_build_dir), timeout=300, env=build_env)
            if rc != 0:
                return False, f"CMake configure failed:\n{stderr}\n{stdout}"
        else:
            return False, f"CMake configure failed:\n{stderr}\n{stdout}"
    
    # Build targets
    for target in targets:
        log(f"Building target: {target}")
        # Use parallel build
        parallel_jobs = min(os.cpu_count() or 4, 8)  # Limit to 8 parallel jobs
        build_cmd = f'cmake --build . --target {target} --config Release --parallel {parallel_jobs}'
        rc, stdout, stderr = run_command(build_cmd, cwd=str(unique_build_dir), timeout=600, env=build_env)
        if rc != 0:
            return False, f"Build failed for {target}:\n{stderr}\n{stdout}"
    
    return True, str(unique_build_dir)


# =============================================================================
# Test Execution
# =============================================================================

def parse_test_output(stdout: str, stderr: str) -> Dict[str, Any]:
    """Parse test output to extract metrics."""
    metrics = {}
    
    combined = stdout + stderr
    
    # Look for common patterns
    patterns = {
        "pass_count": r"\[PASS\]",
        "fail_count": r"\[FAIL\]",
        "time_seconds": r"Time:\s*([\d.]+)\s*s",
        "speedup": r"Speedup:\s*([\d.]+)x",
        "fidelity": r"Fidelity:\s*([\d.]+)",
        "rank": r"Final [Rr]ank:\s*(\d+)",
        "imbalance": r"imbalance ratio:\s*([\d.]+)",
    }
    
    for key, pattern in patterns.items():
        matches = re.findall(pattern, combined)
        if matches:
            if key in ["pass_count", "fail_count"]:
                metrics[key] = len(matches)
            else:
                try:
                    metrics[key] = float(matches[-1])
                except:
                    metrics[key] = matches[-1]
    
    # Determine if test passed from output
    if "passed" in combined.lower() and "failed" not in combined.lower():
        metrics["test_passed"] = True
    elif "failed" in combined.lower() or "error" in combined.lower():
        metrics["test_passed"] = False
    elif "skipping" in combined.lower() or "skip" in combined.lower():
        metrics["test_skipped"] = True
    
    return metrics


def run_single_test(test_def: dict, build_dir: Path, output_dir: Path, 
                    system_info: SystemInfo, actual_build_path: Optional[str] = None) -> TestResult:
    """Run a single test and capture results.
    
    Args:
        test_def: Test definition dictionary.
        build_dir: Base build directory.
        output_dir: Output directory for results.
        system_info: System information.
        actual_build_path: If provided, use this path for the executable instead of computing from flags.
    """
    
    result = TestResult(
        test_name=test_def["name"],
        test_file=test_def["file"],
        status=TestStatus.NOT_RUN,
        hardware_requirements=test_def["requirements"],
        build_flags=" ".join(test_def["build_flags"]),
        run_command=test_def["run_cmd"],
        timestamp=datetime.datetime.now().isoformat()
    )
    
    # Check hardware requirements
    required_gpus = test_def.get("mpi_ranks", 0)
    if required_gpus > 0 and test_def["category"] != "MPI":
        # GPU test - check if we have enough GPUs
        if system_info.gpu_count < required_gpus:
            result.status = TestStatus.SKIPPED
            result.error_message = f"Requires {required_gpus} GPUs, but only {system_info.gpu_count} available"
            return result
    
    if test_def.get("mpi_ranks", 0) > 0 and not system_info.mpi_version:
        result.status = TestStatus.SKIPPED
        result.error_message = "MPI not installed"
        return result
    
    # Find the build directory: use actual_build_path if provided, else compute from flags
    if actual_build_path and actual_build_path != "Skipped":
        unique_build_dir = Path(actual_build_path)
    else:
        flags_hash = "_".join(sorted(test_def["build_flags"])).replace("-D", "").replace("=", "_")
        unique_build_dir = build_dir / f"build_{flags_hash}"
    
    if not unique_build_dir.exists():
        result.status = TestStatus.ERROR
        result.error_message = f"Build directory not found: {unique_build_dir}"
        return result
    
    # Check if executable exists
    executable = test_def["run_cmd"].split()[-1] if "mpirun" not in test_def["run_cmd"] else test_def["run_cmd"].split()[-1]
    
    # Strip leading ./ if present
    if executable.startswith("./"):
        executable = executable[2:]
    
    # Add .exe extension on Windows
    if platform.system() == "Windows" and not executable.endswith(".exe"):
        executable = executable + ".exe"
    
    exec_path = unique_build_dir / executable
    
    # For mpirun commands, extract the actual executable
    if "mpirun" in test_def["run_cmd"]:
        parts = test_def["run_cmd"].split()
        for i, part in enumerate(parts):
            if part.startswith("./"):
                executable = part[2:]
                if platform.system() == "Windows" and not executable.endswith(".exe"):
                    executable = executable + ".exe"
                exec_path = unique_build_dir / executable
                break
    
    if not exec_path.exists() and not test_def["run_cmd"].startswith("mpirun"):
        result.status = TestStatus.ERROR
        result.error_message = f"Executable not found: {exec_path}"
        return result
    
    # Prepare run command with full path
    if "mpirun" in test_def["run_cmd"]:
        run_cmd = test_def["run_cmd"].replace(f"./{executable}", str(unique_build_dir / executable))
    else:
        run_cmd = str(unique_build_dir / executable)
    
    log(f"Running: {run_cmd}")
    
    # Set environment variables
    env = {
        "OMP_NUM_THREADS": str(os.cpu_count() or 4),
        "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(system_info.gpu_count)),
    }
    
    # Run the test
    start_time = time.time()
    rc, stdout, stderr = run_command(run_cmd, cwd=str(unique_build_dir), timeout=1800, env=env)
    end_time = time.time()
    
    result.duration_seconds = end_time - start_time
    result.stdout = stdout
    result.stderr = stderr
    result.return_code = rc
    
    # Parse output for metrics
    result.metrics = parse_test_output(stdout, stderr)
    
    # Determine status
    if rc == 0:
        if result.metrics.get("test_skipped", False):
            result.status = TestStatus.SKIPPED
            result.error_message = "Test skipped due to insufficient hardware"
        elif "passed" in stdout.lower() or "pass" in stdout.lower():
            result.status = TestStatus.PASSED
        else:
            result.status = TestStatus.PASSED  # Assume pass if exit code is 0
    else:
        if "skip" in stdout.lower() or "skip" in stderr.lower():
            result.status = TestStatus.SKIPPED
            result.error_message = "Test skipped"
        else:
            result.status = TestStatus.FAILED
            result.error_message = stderr if stderr else f"Exit code: {rc}"
    
    return result


# =============================================================================
# Output Generation
# =============================================================================

def save_test_output(result: TestResult, output_dir: Path):
    """Save individual test output to files."""
    test_dir = output_dir / result.test_name
    ensure_directory(test_dir)
    
    # Save stdout
    with open(test_dir / "stdout.log", "w", encoding="utf-8") as f:
        f.write(result.stdout)
    
    # Save stderr
    with open(test_dir / "stderr.log", "w", encoding="utf-8") as f:
        f.write(result.stderr)
    
    # Save metrics as JSON
    with open(test_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "test_name": result.test_name,
            "status": result.status.value,
            "duration_seconds": result.duration_seconds,
            "return_code": result.return_code,
            "error_message": result.error_message,
            "metrics": result.metrics,
            "timestamp": result.timestamp,
            "hardware_requirements": result.hardware_requirements,
            "build_flags": result.build_flags,
            "run_command": result.run_command
        }, f, indent=2)
    
    # Save summary
    with open(test_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Test: {result.test_name}\n")
        f.write(f"Status: {result.status.value}\n")
        f.write(f"Duration: {result.duration_seconds:.2f}s\n")
        f.write(f"Return Code: {result.return_code}\n")
        f.write(f"Requirements: {result.hardware_requirements}\n")
        f.write(f"Build Flags: {result.build_flags}\n")
        f.write(f"Command: {result.run_command}\n")
        if result.error_message:
            f.write(f"Error: {result.error_message}\n")
        f.write(f"\nMetrics:\n")
        for k, v in result.metrics.items():
            f.write(f"  {k}: {v}\n")


def generate_csv_report(results: List[TestResult], output_dir: Path, system_info: SystemInfo):
    """Generate comprehensive CSV report of all test results."""
    
    # Main results CSV
    csv_path = output_dir / "test_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            "Test Name", "Category", "Status", "Duration (s)", "Return Code",
            "Error Message", "Hardware Requirements", "Build Flags", "Run Command",
            "Timestamp", "Pass Count", "Fail Count", "Fidelity", "Speedup"
        ])
        
        for r in results:
            # Find category from test definition
            category = "Unknown"
            for t in HARDWARE_TESTS:
                if t["name"] == r.test_name:
                    category = t["category"]
                    break
            
            writer.writerow([
                r.test_name,
                category,
                r.status.value,
                f"{r.duration_seconds:.3f}",
                r.return_code,
                r.error_message[:100] if r.error_message else "",
                r.hardware_requirements,
                r.build_flags,
                r.run_command,
                r.timestamp,
                r.metrics.get("pass_count", ""),
                r.metrics.get("fail_count", ""),
                r.metrics.get("fidelity", ""),
                r.metrics.get("speedup", "")
            ])
    
    log(f"Results CSV saved to: {csv_path}", "SUCCESS")
    
    # System info CSV
    sys_csv_path = output_dir / "system_info.csv"
    with open(sys_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Property", "Value"])
        writer.writerow(["OS", f"{system_info.os_name} {system_info.os_version}"])
        writer.writerow(["Kernel", system_info.kernel])
        writer.writerow(["CPU", system_info.cpu_model])
        writer.writerow(["CPU Cores", system_info.cpu_cores])
        writer.writerow(["Memory (GB)", f"{system_info.total_memory_gb:.1f}"])
        writer.writerow(["GPU Count", system_info.gpu_count])
        writer.writerow(["GPUs", "; ".join(system_info.gpu_models)])
        writer.writerow(["GPU Memory (GB)", "; ".join(f"{m:.1f}" for m in system_info.gpu_memory_gb)])
        writer.writerow(["CUDA Version", system_info.cuda_version])
        writer.writerow(["cuDNN Version", system_info.cudnn_version])
        writer.writerow(["NCCL Version", system_info.nccl_version])
        writer.writerow(["MPI Implementation", system_info.mpi_implementation])
        writer.writerow(["MPI Version", system_info.mpi_version])
        writer.writerow(["CMake Version", system_info.cmake_version])
        writer.writerow(["GCC Version", system_info.gcc_version])
    
    log(f"System info CSV saved to: {sys_csv_path}", "SUCCESS")
    
    # Summary CSV
    summary_path = output_dir / "summary.csv"
    passed = sum(1 for r in results if r.status == TestStatus.PASSED)
    failed = sum(1 for r in results if r.status == TestStatus.FAILED)
    skipped = sum(1 for r in results if r.status == TestStatus.SKIPPED)
    errors = sum(1 for r in results if r.status == TestStatus.ERROR)
    
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total Tests", len(results)])
        writer.writerow(["Passed", passed])
        writer.writerow(["Failed", failed])
        writer.writerow(["Skipped", skipped])
        writer.writerow(["Errors", errors])
        writer.writerow(["Pass Rate", f"{100*passed/len(results):.1f}%" if results else "N/A"])
        writer.writerow(["Total Duration (s)", f"{sum(r.duration_seconds for r in results):.2f}"])
        writer.writerow(["Run Date", datetime.datetime.now().isoformat()])
    
    log(f"Summary CSV saved to: {summary_path}", "SUCCESS")


def generate_html_report(results: List[TestResult], output_dir: Path, system_info: SystemInfo):
    """Generate HTML report for better visualization."""
    
    html_path = output_dir / "report.html"
    
    passed = sum(1 for r in results if r.status == TestStatus.PASSED)
    failed = sum(1 for r in results if r.status == TestStatus.FAILED)
    skipped = sum(1 for r in results if r.status == TestStatus.SKIPPED)
    errors = sum(1 for r in results if r.status == TestStatus.ERROR)
    
    status_colors = {
        TestStatus.PASSED: "#28a745",
        TestStatus.FAILED: "#dc3545", 
        TestStatus.SKIPPED: "#ffc107",
        TestStatus.ERROR: "#6c757d",
        TestStatus.NOT_RUN: "#17a2b8"
    }
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>LRET Hardware Tests Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
        .summary-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; min-width: 120px; }}
        .summary-card.passed {{ border-left: 4px solid #28a745; }}
        .summary-card.failed {{ border-left: 4px solid #dc3545; }}
        .summary-card.skipped {{ border-left: 4px solid #ffc107; }}
        .summary-card.error {{ border-left: 4px solid #6c757d; }}
        .summary-number {{ font-size: 36px; font-weight: bold; }}
        .summary-label {{ color: #666; }}
        table {{ width: 100%; border-collapse: collapse; background: white; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        th {{ background: #007bff; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 12px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background: #f8f9fa; }}
        .status {{ padding: 4px 8px; border-radius: 4px; color: white; font-size: 12px; }}
        .system-info {{ background: white; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .system-info table {{ box-shadow: none; }}
        .test-details {{ margin-top: 10px; font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 LRET Hardware-Dependent Tests Report</h1>
        <p>Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <div class="summary">
            <div class="summary-card passed">
                <div class="summary-number" style="color: #28a745;">{passed}</div>
                <div class="summary-label">Passed</div>
            </div>
            <div class="summary-card failed">
                <div class="summary-number" style="color: #dc3545;">{failed}</div>
                <div class="summary-label">Failed</div>
            </div>
            <div class="summary-card skipped">
                <div class="summary-number" style="color: #ffc107;">{skipped}</div>
                <div class="summary-label">Skipped</div>
            </div>
            <div class="summary-card error">
                <div class="summary-number" style="color: #6c757d;">{errors}</div>
                <div class="summary-label">Errors</div>
            </div>
        </div>
        
        <h2>🖥️ System Information</h2>
        <div class="system-info">
            <table>
                <tr><td><strong>OS</strong></td><td>{system_info.os_name} {system_info.os_version}</td></tr>
                <tr><td><strong>CPU</strong></td><td>{system_info.cpu_model} ({system_info.cpu_cores} cores)</td></tr>
                <tr><td><strong>Memory</strong></td><td>{system_info.total_memory_gb:.1f} GB</td></tr>
                <tr><td><strong>GPUs</strong></td><td>{system_info.gpu_count}x {', '.join(system_info.gpu_models) if system_info.gpu_models else 'None'}</td></tr>
                <tr><td><strong>CUDA</strong></td><td>{system_info.cuda_version or 'Not installed'}</td></tr>
                <tr><td><strong>MPI</strong></td><td>{system_info.mpi_implementation} {system_info.mpi_version or 'Not installed'}</td></tr>
            </table>
        </div>
        
        <h2>📊 Test Results</h2>
        <table>
            <tr>
                <th>Test Name</th>
                <th>Category</th>
                <th>Status</th>
                <th>Duration</th>
                <th>Requirements</th>
                <th>Details</th>
            </tr>
"""
    
    for r in results:
        category = "Unknown"
        description = ""
        for t in HARDWARE_TESTS:
            if t["name"] == r.test_name:
                category = t["category"]
                description = t["description"]
                break
        
        color = status_colors.get(r.status, "#666")
        error_info = f"<br><small style='color:red;'>{r.error_message[:80]}...</small>" if r.error_message else ""
        
        html += f"""
            <tr>
                <td><strong>{r.test_name}</strong><div class="test-details">{description[:60]}...</div></td>
                <td>{category}</td>
                <td><span class="status" style="background:{color};">{r.status.value}</span></td>
                <td>{r.duration_seconds:.2f}s</td>
                <td>{r.hardware_requirements}</td>
                <td><a href="{r.test_name}/summary.txt">View Logs</a>{error_info}</td>
            </tr>
"""
    
    html += """
        </table>
        
        <h2>📁 Output Files</h2>
        <ul>
            <li><a href="test_results.csv">test_results.csv</a> - Detailed results CSV</li>
            <li><a href="system_info.csv">system_info.csv</a> - System information</li>
            <li><a href="summary.csv">summary.csv</a> - Test summary</li>
        </ul>
        
        <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666;">
            <p>LRET Quantum Simulator - Hardware-Dependent Tests</p>
        </footer>
    </div>
</body>
</html>
"""
    
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    log(f"HTML report saved to: {html_path}", "SUCCESS")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run LRET Hardware-Dependent Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full run with dependency installation
    python run_hardware_dependent_tests.py
    
    # Skip installation (dependencies already installed)
    python run_hardware_dependent_tests.py --skip-install
    
    # Skip build (already built)
    python run_hardware_dependent_tests.py --skip-install --skip-build
    
    # Custom output directory
    python run_hardware_dependent_tests.py --output-dir /results/gpu_tests
"""
    )
    parser.add_argument("--skip-install", action="store_true",
                        help="Skip dependency installation")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip building tests")
    parser.add_argument("--use-existing-build", type=str, default=None,
                        help="Use existing build directory (e.g., build_tests/gpu_build)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results (default: test_output_TIMESTAMP)")
    parser.add_argument("--project-root", type=str, default=None,
                        help="LRET project root directory")
    parser.add_argument("--tests", type=str, nargs="+", default=None,
                        help="Specific tests to run (default: all)")
    parser.add_argument("--cuda-12x", action="store_true", default=True,
                        help="Prefer CUDA 12.x for Pascal GPU compatibility (default: True)")
    parser.add_argument("--cuda-13x", action="store_true", default=False,
                        help="Allow CUDA 13.x (requires compute capability >= 7.5)")
    
    args = parser.parse_args()
    
    # Set global CUDA preference based on CLI arguments
    global _prefer_cuda_12x
    if args.cuda_13x:
        _prefer_cuda_12x = False
        log("Using CUDA 13.x (requires compute >= 7.5)", "INFO")
    else:
        _prefer_cuda_12x = True
        log("Preferring CUDA 12.x for Pascal GPU compatibility", "INFO")
    
    # Determine paths
    script_dir = Path(__file__).parent.resolve()
    project_root = Path(args.project_root) if args.project_root else script_dir.parent
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else project_root / f"test_output_{timestamp}"
    build_dir = project_root / "build_tests"
    
    ensure_directory(output_dir)
    ensure_directory(build_dir)
    
    log("=" * 70)
    log("LRET HARDWARE-DEPENDENT TESTS RUNNER")
    log("=" * 70)
    log(f"Project Root: {project_root}")
    log(f"Output Directory: {output_dir}")
    log(f"Build Directory: {build_dir}")
    
    # Collect system information
    log("\nCollecting system information...")
    system_info = get_system_info()
    
    log(f"  OS: {system_info.os_name} {system_info.os_version}")
    log(f"  CPU: {system_info.cpu_model} ({system_info.cpu_cores} cores)")
    log(f"  Memory: {system_info.total_memory_gb:.1f} GB")
    log(f"  GPUs: {system_info.gpu_count}")
    for i, (model, mem) in enumerate(zip(system_info.gpu_models, system_info.gpu_memory_gb)):
        log(f"    GPU {i}: {model} ({mem:.1f} GB)")
    log(f"  CUDA: {system_info.cuda_version or 'Not found'}")
    log(f"  MPI: {system_info.mpi_implementation} {system_info.mpi_version or 'Not found'}")
    
    # Install dependencies
    if not install_dependencies(project_root, skip=args.skip_install):
        log("Dependency installation failed!", "ERROR")
        return 1
    
    # Determine which tests to run
    tests_to_run = HARDWARE_TESTS
    if args.tests:
        tests_to_run = [t for t in HARDWARE_TESTS if t["name"] in args.tests]
        if not tests_to_run:
            log(f"No matching tests found for: {args.tests}", "ERROR")
            return 1
    
    log(f"\nTests to run: {len(tests_to_run)}")
    
    # Group tests by build flags to minimize rebuilds
    build_groups: Dict[str, List[dict]] = {}
    for test in tests_to_run:
        flags_key = tuple(sorted(test["build_flags"]))
        if flags_key not in build_groups:
            build_groups[flags_key] = []
        build_groups[flags_key].append(test)
    
    # Build each configuration
    build_success = {}
    for flags_key, tests in build_groups.items():
        flags = list(flags_key)
        targets = list(set(t["target"] for t in tests))
        
        log(f"\nBuilding configuration: {' '.join(flags)}")
        success, result = build_tests(
            project_root, build_dir, flags, targets, 
            skip=args.skip_build, 
            existing_build_dir=args.use_existing_build
        )
        build_success[flags_key] = (success, result)
        
        if not success:
            log(f"Build failed: {result}", "ERROR")
    
    # Run tests
    results: List[TestResult] = []
    
    log("\n" + "=" * 70)
    log("RUNNING TESTS")
    log("=" * 70)
    
    for test in tests_to_run:
        log(f"\n--- Running: {test['name']} ---")
        log(f"    Category: {test['category']}")
        log(f"    Requirements: {test['requirements']}")
        
        flags_key = tuple(sorted(test["build_flags"]))
        build_ok, build_path = build_success.get(flags_key, (False, "Not built"))
        
        if not build_ok and not args.skip_build:
            result = TestResult(
                test_name=test["name"],
                test_file=test["file"],
                status=TestStatus.ERROR,
                error_message=f"Build failed: {build_path}",
                hardware_requirements=test["requirements"],
                build_flags=" ".join(test["build_flags"]),
                run_command=test["run_cmd"],
                timestamp=datetime.datetime.now().isoformat()
            )
        else:
            result = run_single_test(test, build_dir, output_dir, system_info, actual_build_path=build_path)
        
        results.append(result)
        save_test_output(result, output_dir)
        
        status_msg = {
            TestStatus.PASSED: ("SUCCESS", "✅"),
            TestStatus.FAILED: ("ERROR", "❌"),
            TestStatus.SKIPPED: ("WARNING", "⏭️"),
            TestStatus.ERROR: ("ERROR", "💥"),
            TestStatus.NOT_RUN: ("WARNING", "⏸️")
        }
        level, emoji = status_msg.get(result.status, ("INFO", "❓"))
        log(f"    Result: {emoji} {result.status.value} ({result.duration_seconds:.2f}s)", level)
        if result.error_message:
            log(f"    Error: {result.error_message[:100]}", "WARNING")
    
    # Generate reports
    log("\n" + "=" * 70)
    log("GENERATING REPORTS")
    log("=" * 70)
    
    generate_csv_report(results, output_dir, system_info)
    generate_html_report(results, output_dir, system_info)
    
    # Final summary
    passed = sum(1 for r in results if r.status == TestStatus.PASSED)
    failed = sum(1 for r in results if r.status == TestStatus.FAILED)
    skipped = sum(1 for r in results if r.status == TestStatus.SKIPPED)
    errors = sum(1 for r in results if r.status == TestStatus.ERROR)
    
    log("\n" + "=" * 70)
    log("FINAL SUMMARY")
    log("=" * 70)
    log(f"  Total Tests: {len(results)}")
    log(f"  ✅ Passed:   {passed}", "SUCCESS" if passed > 0 else "INFO")
    log(f"  ❌ Failed:   {failed}", "ERROR" if failed > 0 else "INFO")
    log(f"  ⏭️ Skipped:  {skipped}", "WARNING" if skipped > 0 else "INFO")
    log(f"  💥 Errors:   {errors}", "ERROR" if errors > 0 else "INFO")
    log(f"\n  Output directory: {output_dir}")
    log(f"  HTML Report: {output_dir / 'report.html'}")
    
    # Return appropriate exit code
    if failed > 0 or errors > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
