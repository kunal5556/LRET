#!/usr/bin/env python3
"""
LRET Automated Benchmark Setup - Cirq Comparison
=================================================
This script automates the complete setup process:
  1. Check Python version
  2. Install Python dependencies (cirq, matplotlib, numpy, scipy, psutil)
  3. Build LRET C++ backend
  4. Verify quantum_sim.exe
  5. Test DEPOLARIZE gate support

Usage:
  python 01_setup.py
  python 01_setup.py --skip-build  # Skip C++ build if already done

Adapted from setup_pennylane_env.py which works on this system.
"""

import os
import sys
import subprocess
import json
import platform
import shutil
import time
from pathlib import Path

# ANSI color codes
class Colors:
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def colored(text, color):
    """Return colored text (degrades gracefully on Windows)"""
    if platform.system() == "Windows":
        return text
    return f"{color}{text}{Colors.ENDC}"

def print_header(msg):
    """Print formatted header"""
    print("\n" + "=" * 72)
    print(colored(msg, Colors.BOLD))
    print("=" * 72 + "\n")

def print_step(step, total, msg):
    """Print step progress"""
    print("\n" + "=" * 72)
    print(colored(f"[{step}/{total}] {msg}", Colors.OKBLUE))
    print("=" * 72)

def check_command_exists(command):
    """Check if a command exists in PATH"""
    return shutil.which(command) is not None

def get_python_executable():
    """Get the current Python executable path"""
    return sys.executable

def find_msbuild():
    """Find MSBuild.exe and VsDevCmd.bat in various Visual Studio installations"""
    
    # Potential locations for VsDevCmd.bat (preferred - sets up full environment)
    vsdevcmd_paths = [
        # VS 2022 (version-numbered, non-standard)
        r"C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat",
        r"C:\Program Files\Microsoft Visual Studio\17\Community\Common7\Tools\VsDevCmd.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat",
        
        # VS 2022 (standard locations)
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\VsDevCmd.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat",
        
        # VS 2019
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\Common7\Tools\VsDevCmd.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\Tools\VsDevCmd.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\Tools\VsDevCmd.bat",
    ]
    
    # Check for VsDevCmd.bat first (preferred)
    for path in vsdevcmd_paths:
        if os.path.exists(path):
            print(f"  Found VS Developer Command Prompt: {path}")
            return ("vsdevcmd", path)
    
    # Fallback: Find MSBuild directly (less reliable)
    msbuild_paths = [
        # VS 2022 (version-numbered, non-standard)
        r"C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe",
        r"C:\Program Files\Microsoft Visual Studio\17\Community\MSBuild\Current\Bin\MSBuild.exe",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe",
        
        # VS 2022 (standard locations)
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\MSBuild\Current\Bin\MSBuild.exe",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\MSBuild\Current\Bin\MSBuild.exe",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe",
        
        # VS 2019
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\MSBuild\Current\Bin\MSBuild.exe",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\MSBuild\Current\Bin\MSBuild.exe",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin\MSBuild.exe",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\MSBuild\Current\Bin\MSBuild.exe",
        
        # VS 2017
        r"C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\MSBuild\15.0\Bin\MSBuild.exe",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\MSBuild\15.0\Bin\MSBuild.exe",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\MSBuild\15.0\Bin\MSBuild.exe",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\MSBuild\15.0\Bin\MSBuild.exe",
    ]
    
    # Check each MSBuild path
    for path in msbuild_paths:
        if os.path.exists(path):
            print(f"  Found MSBuild: {path}")
            return ("msbuild", path)
    
    # Try PATH
    msbuild = shutil.which("msbuild")
    if msbuild:
        print(f"  Found MSBuild in PATH: {msbuild}")
        return ("msbuild", msbuild)
    
    return None

def run_command(cmd, cwd=None, check=True):
    """Run command and return result"""
    print(f"  Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=check,
            capture_output=True,
            text=True,
            shell=True if isinstance(cmd, str) else False
        )
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"\n  ERROR: Command failed with exit code {e.returncode}")
        if e.stdout:
            print("  STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("  STDERR:")
            print(e.stderr)
        if check:
            raise
        return e

def main():
    # Parse command line arguments
    skip_build = '--skip-build' in sys.argv
    
    print_header("LRET Cirq Benchmark Setup")
    
    # Detect system
    system = platform.system()
    print(f"Detected OS: {colored(system, Colors.OKCYAN)}")
    print(f"Python: {colored(sys.version.split()[0], Colors.OKCYAN)} at {sys.executable}")
    
    # Detect LRET root (3 levels up from script directory)
    script_dir = Path(__file__).parent.resolve()
    lret_root = script_dir.parent.parent
    print(f"LRET Root: {colored(str(lret_root), Colors.OKCYAN)}\n")
    
    # Verify LRET root
    if not (lret_root / "CMakeLists.txt").exists():
        print(colored(f"ERROR: Cannot find LRET root directory", Colors.FAIL))
        print(f"Expected CMakeLists.txt at: {lret_root}")
        return 1
    
    build_dir = lret_root / "build"
    
    total_steps = 5 if skip_build else 6
    current_step = 0
    
    # Step 1: Check Python
    current_step += 1
    print_step(current_step, total_steps, "Check Python Version")
    
    py_version = sys.version_info
    if py_version < (3, 8):
        print(colored(f"✗ Python {py_version.major}.{py_version.minor} is too old. Need Python 3.8+", Colors.FAIL))
        return 1
    elif py_version >= (3, 12):
        print(colored(f"⚠ Python {py_version.major}.{py_version.minor} is very new. Tested on 3.8-3.11", Colors.WARNING))
    else:
        print(colored(f"✓ Python {py_version.major}.{py_version.minor} is compatible", Colors.OKGREEN))
    
    # Step 2: Install dependencies
    current_step += 1
    print_step(current_step, total_steps, "Install Python Dependencies")
    
    packages = ["cirq", "matplotlib", "numpy", "scipy", "psutil>=5.8"]
    print(f"Installing packages: {', '.join(packages)}")
    pip_cmd = [get_python_executable(), "-m", "pip", "install"] + packages
    
    try:
        subprocess.run(pip_cmd, check=True)
        print(colored("✓ Python dependencies installed successfully", Colors.OKGREEN))
    except subprocess.CalledProcessError:
        print(colored("⚠ Some dependencies may already be installed", Colors.WARNING))
    
    # Verify installations
    print("\nVerifying installations...")
    try:
        import cirq
        print(colored(f"  ✓ Cirq {cirq.__version__}", Colors.OKGREEN))
    except ImportError:
        print(colored("  ✗ Cirq not found", Colors.FAIL))
        return 1
    
    # Step 4: Build LRET binaries
    print_step(4, 6, "Building LRET C++ binaries...")
    
    # Verify vcxproj exists
    qlret_vcxproj = build_dir / "qlret_lib.vcxproj"
    if not qlret_vcxproj.exists():
        print(f"ERROR: {qlret_vcxproj} not found")
        print("Please run CMake to generate Visual Studio project files")
        return 1
    
    # Build qlret_lib
    print("  Building qlret_lib.lib...")
    result = run_msbuild(qlret_vcxproj, build_dir)
    
    if result.returncode != 0:
        print("\n  ERROR: qlret_lib build failed")
        if result.stdout:
            print("  STDOUT:")
            print(result.stdout)
        if result.stderr:
            print("  STDERR:")
            print(result.stderr)
        return 1
    
    if result.stdout:
        print(result.stdout)
    
    # Build quantum_sim
    print("  Building quantum_sim.exe...")
    quantum_sim_vcxproj = build_dir / "quantum_sim.vcxproj"
    result = run_msbuild(quantum_sim_vcxproj, build_dir)
    
    if result.returncode != 0:
        print("\n  ERROR: quantum_sim build failed")
        if result.stdout:
            print("  STDOUT:")
            print(result.stdout)
        if result.stderr:
            print("  STDERR:")
            print(result.stderr)
        return 1
    
    if result.stdout:
        print(result.stdout)
    
    print("  Binaries built successfully")
    
    # Step 5: Verify binary
    print_step(5, 6, "Verifying quantum_sim.exe...")
    exe_path = build_dir / "Release" / "quantum_sim.exe"
    if not exe_path.exists():
        print(f"  ERROR: quantum_sim.exe not found at {exe_path}")
        return 1
    print(f"  Found: {exe_path}")
    
    # Test basic circuit
    test_basic_json = {
        "circuit": {
            "num_qubits": 2,
            "operations": [
                {"name": "H", "wires": [0]},
                {"name": "CNOT", "wires": [0, 1]}
            ]
        },
        "config": {"epsilon": 0.0001, "initial_rank": 1}
    }
    
    test_file = build_dir / "test_basic.json"
    test_output = build_dir / "test_basic_out.json"
    
    with open(test_file, 'w') as f:
        json.dump(test_basic_json, f)
    
    result = run_command([
        str(exe_path),
        "--input-json", str(test_file),
        "--output", str(test_output)
    ], check=False)
    
    if not test_output.exists():
        print("  ERROR: Basic test failed")
        return 1
    print("  Basic circuit test passed")
    
    # Step 6: Test DEPOLARIZE
    print_step(6, 6, "Testing DEPOLARIZE gate...")
    test_depol_json = {
        "circuit": {
            "num_qubits": 2,
            "operations": [
                {"name": "H", "wires": [0]},
                {"name": "DEPOLARIZE", "wires": [0], "params": [0.1]},
                {"name": "CNOT", "wires": [0, 1]}
            ]
        },
        "config": {"epsilon": 0.0001, "initial_rank": 1}
    }
    
    test_depol_file = build_dir / "test_depol.json"
    test_depol_output = build_dir / "test_depol_out.json"
    
    with open(test_depol_file, 'w') as f:
        json.dump(test_depol_json, f)
    
    result = run_command([
        str(exe_path),
        "--input-json", str(test_depol_file),
        "--output", str(test_depol_output)
    ], check=False)
    
    if not test_depol_output.exists():
        print("  ERROR: DEPOLARIZE test failed")
        return 1
    print("  DEPOLARIZE gate working")
    
    # Summary
    print_header("SETUP COMPLETE - Ready to run benchmarks!")
    print("Summary:")
    print("  ✓ Python dependencies: OK")
    print("  ✓ LRET binaries: OK")
    print("  ✓ quantum_sim.exe: OK")
    print("  ✓ DEPOLARIZE support: OK")
    print("\nNext: python 02_run_benchmark.py\n")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
