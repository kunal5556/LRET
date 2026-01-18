#!/usr/bin/env python3
"""
Automated Setup Script for LRET PennyLane Benchmarking
=======================================================
This script automates the complete setup process:
  1. Check Python version
  2. Install Python dependencies
  3. Build LRET C++ backend
  4. Install LRET Python package
  5. Verify PennyLane device registration
  6. Run smoke test

Usage:
  python setup_pennylane_env.py
  python setup_pennylane_env.py --skip-build  # Skip C++ build if already done
  python setup_pennylane_env.py --test-only   # Only run verification test
"""

import sys
import os
import platform
import subprocess
import shutil
from pathlib import Path

# ANSI color codes for pretty output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def colored(text, color):
    """Return colored text (works on Linux/Mac, degrades gracefully on Windows)"""
    if platform.system() == "Windows":
        # Windows terminal might not support ANSI, so just return plain text
        return text
    return f"{color}{text}{Colors.ENDC}"

def print_step(step_num, total_steps, message):
    """Print a formatted step header"""
    print(f"\n{'='*70}")
    print(colored(f"Step {step_num}/{total_steps}: {message}", Colors.HEADER))
    print('='*70)

def run_command(cmd, shell=False, check=True, capture_output=False):
    """Run a shell command and return result"""
    print(colored(f"Running: {cmd if isinstance(cmd, str) else ' '.join(cmd)}", Colors.OKBLUE))
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=shell, check=check, 
                                  capture_output=True, text=True)
            return result
        else:
            result = subprocess.run(cmd, shell=shell, check=check)
            return result
    except subprocess.CalledProcessError as e:
        print(colored(f"✗ Command failed with exit code {e.returncode}", Colors.FAIL))
        if capture_output and e.stderr:
            print(colored(f"Error output: {e.stderr}", Colors.FAIL))
        raise

def check_command_exists(command):
    """Check if a command exists in PATH"""
    return shutil.which(command) is not None

def get_python_executable():
    """Get the current Python executable path"""
    return sys.executable

def main():
    # Parse command line arguments
    skip_build = '--skip-build' in sys.argv
    test_only = '--test-only' in sys.argv
    
    print(colored("\n╔═══════════════════════════════════════════════════════════════════╗", Colors.BOLD))
    print(colored("║   LRET PennyLane Benchmarking - Automated Setup Script          ║", Colors.BOLD))
    print(colored("╚═══════════════════════════════════════════════════════════════════╝", Colors.BOLD))
    
    # Detect system
    system = platform.system()
    print(f"\nDetected OS: {colored(system, Colors.OKCYAN)}")
    print(f"Python: {colored(sys.version.split()[0], Colors.OKCYAN)} at {sys.executable}")
    
    # Determine project root
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent  # benchmarks/pennylane -> benchmarks -> LRET
    print(f"Project root: {colored(str(project_root), Colors.OKCYAN)}")
    
    total_steps = 3 if test_only else (5 if skip_build else 6)
    current_step = 0
    
    # ==========================================================================
    # STEP 1: Check Python Version
    # ==========================================================================
    if not test_only:
        current_step += 1
        print_step(current_step, total_steps, "Check Python Version")
        
        py_version = sys.version_info
        if py_version < (3, 8):
            print(colored(f"✗ Python {py_version.major}.{py_version.minor} is too old. Need Python 3.8+", Colors.FAIL))
            sys.exit(1)
        elif py_version >= (3, 12):
            print(colored(f"⚠ Python {py_version.major}.{py_version.minor} is very new. Recommended: 3.8-3.11", Colors.WARNING))
        else:
            print(colored(f"✓ Python {py_version.major}.{py_version.minor} is compatible", Colors.OKGREEN))
    
    # ==========================================================================
    # STEP 2: Install Python Dependencies
    # ==========================================================================
    if not test_only:
        current_step += 1
        print_step(current_step, total_steps, "Install Python Dependencies")
        
        packages = [
            "pennylane>=0.30",
            "torch",
            "numpy",
            "scipy",
            "psutil>=5.8",  # Required for CPU monitoring
            "matplotlib",
            "pandas"
        ]
        
        print(f"Installing packages: {', '.join(packages)}")
        pip_cmd = [get_python_executable(), "-m", "pip", "install"] + packages
        
        try:
            run_command(pip_cmd)
            print(colored("✓ Python dependencies installed successfully", Colors.OKGREEN))
        except subprocess.CalledProcessError:
            print(colored("✗ Failed to install Python dependencies", Colors.FAIL))
            print(colored("Try manually: pip install pennylane torch numpy scipy psutil matplotlib pandas", Colors.WARNING))
            sys.exit(1)
        
        # Verify installations
        print("\nVerifying installations...")
        try:
            import pennylane as qml
            print(colored(f"  ✓ PennyLane {qml.__version__}", Colors.OKGREEN))
        except ImportError:
            print(colored("  ✗ PennyLane not found", Colors.FAIL))
            sys.exit(1)
        
        try:
            import torch
            print(colored(f"  ✓ PyTorch {torch.__version__}", Colors.OKGREEN))
        except ImportError:
            print(colored("  ✗ PyTorch not found", Colors.FAIL))
            sys.exit(1)
    
    # ==========================================================================
    # STEP 3: Build LRET C++ Backend (if not skipped)
    # ==========================================================================
    if not test_only and not skip_build:
        current_step += 1
        print_step(current_step, total_steps, "Build LRET C++ Backend")
        
        # Check for CMake
        if not check_command_exists("cmake"):
            print(colored("✗ CMake not found in PATH", Colors.FAIL))
            print(colored("Please install CMake 3.16+ and add to PATH", Colors.WARNING))
            print(colored("  Linux: sudo apt install cmake", Colors.WARNING))
            print(colored("  Windows: Download from https://cmake.org/download/", Colors.WARNING))
            sys.exit(1)
        
        # Check for compiler
        if system == "Linux":
            if not check_command_exists("g++"):
                print(colored("✗ g++ not found. Install build tools:", Colors.FAIL))
                print(colored("  sudo apt install build-essential", Colors.WARNING))
                sys.exit(1)
        elif system == "Windows":
            # Check for Visual Studio or Build Tools
            # This is approximate - full check would require registry inspection
            print(colored("⚠ Make sure Visual Studio or Build Tools are installed", Colors.WARNING))
        
        # Build
        build_dir = project_root / "build"
        build_dir.mkdir(exist_ok=True)
        
        print(f"\nConfiguring CMake in {build_dir}...")
        os.chdir(build_dir)
        
        # CMake configure
        if system == "Windows":
            cmake_config = ["cmake", "..", "-G", "Visual Studio 16 2019", "-A", "x64"]
        else:
            cmake_config = ["cmake", ".."]
        
        try:
            run_command(cmake_config)
            print(colored("✓ CMake configuration successful", Colors.OKGREEN))
        except subprocess.CalledProcessError:
            print(colored("✗ CMake configuration failed", Colors.FAIL))
            print(colored("Check REQUIREMENTS.md for manual build instructions", Colors.WARNING))
            sys.exit(1)
        
        # CMake build
        print("\nBuilding LRET (this may take several minutes)...")
        if system == "Windows":
            cmake_build = ["cmake", "--build", ".", "--config", "Release"]
        else:
            import multiprocessing
            n_cores = multiprocessing.cpu_count()
            cmake_build = ["make", f"-j{n_cores}"]
        
        try:
            run_command(cmake_build)
            print(colored("✓ LRET C++ backend built successfully", Colors.OKGREEN))
        except subprocess.CalledProcessError:
            print(colored("✗ Build failed", Colors.FAIL))
            print(colored("Check REQUIREMENTS.md for troubleshooting", Colors.WARNING))
            sys.exit(1)
        
        os.chdir(project_root)
    
    # ==========================================================================
    # STEP 4: Install LRET Python Package
    # ==========================================================================
    if not test_only:
        current_step += 1
        print_step(current_step, total_steps, "Install LRET Python Package")
        
        python_dir = project_root / "python"
        if not python_dir.exists():
            print(colored(f"✗ Python directory not found: {python_dir}", Colors.FAIL))
            sys.exit(1)
        
        os.chdir(python_dir)
        
        print("Installing LRET package (building Python bindings)...")
        pip_install = [get_python_executable(), "-m", "pip", "install", "-e", "."]
        
        try:
            run_command(pip_install)
            print(colored("✓ LRET Python package installed", Colors.OKGREEN))
        except subprocess.CalledProcessError:
            print(colored("✗ Failed to install LRET Python package", Colors.FAIL))
            print(colored("Try manually: cd python && pip install -e .", Colors.WARNING))
            sys.exit(1)
        
        os.chdir(project_root)
    
    # ==========================================================================
    # STEP 5: Verify PennyLane Device Registration
    # ==========================================================================
    current_step += 1
    print_step(current_step, total_steps, "Verify PennyLane Device Registration")
    
    try:
        import pennylane as qml
        print("Testing LRET device creation...")
        dev = qml.device('qlret.mixed', wires=4, epsilon=1e-4)
        print(colored(f"✓ LRET device created: {dev.name}", Colors.OKGREEN))
        print(colored(f"  Device type: {type(dev)}", Colors.OKCYAN))
        print(colored(f"  Number of wires: {dev.num_wires}", Colors.OKCYAN))
    except Exception as e:
        print(colored(f"✗ Failed to create LRET device: {e}", Colors.FAIL))
        print(colored("\nTroubleshooting:", Colors.WARNING))
        print(colored("  1. Verify build artifacts exist:", Colors.WARNING))
        print(colored("     ls python/qlret/build/*.so (Linux)", Colors.WARNING))
        print(colored("     ls python/qlret/build/*.pyd (Windows)", Colors.WARNING))
        print(colored("  2. Rebuild: cd python && pip install -e . --force-reinstall", Colors.WARNING))
        sys.exit(1)
    
    # ==========================================================================
    # STEP 6: Run Smoke Test
    # ==========================================================================
    current_step += 1
    print_step(current_step, total_steps, "Run Smoke Test")
    
    try:
        import pennylane as qml
        import numpy as np
        
        print("Creating 4-qubit LRET device...")
        dev = qml.device('qlret.mixed', wires=4, epsilon=1e-4)
        
        print("Defining quantum circuit...")
        @qml.qnode(dev)
        def circuit(params):
            for i in range(4):
                qml.RY(params[i], wires=i)
            for i in range(3):
                qml.CNOT(wires=[i, i+1])
            return qml.expval(qml.PauliZ(0))
        
        print("Running test circuit...")
        params = np.random.random(4)
        result = circuit(params)
        
        print(colored(f"✓ Circuit executed successfully!", Colors.OKGREEN))
        print(colored(f"  Result: {result:.6f}", Colors.OKCYAN))
        print(colored(f"  Circuit executed on {dev.name} device", Colors.OKCYAN))
        
    except Exception as e:
        print(colored(f"✗ Smoke test failed: {e}", Colors.FAIL))
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ==========================================================================
    # SUCCESS!
    # ==========================================================================
    print("\n" + "="*70)
    print(colored("🎉 SETUP COMPLETE! 🎉", Colors.OKGREEN + Colors.BOLD))
    print("="*70)
    print(colored("\n✓ All checks passed. Ready to run benchmarks!\n", Colors.OKGREEN))
    
    print("Next steps:")
    print(colored("  1. Run light benchmark:  python benchmarks/pennylane/4q_50e_25s_10n.py", Colors.OKCYAN))
    print(colored("  2. Run medium benchmark: python benchmarks/pennylane/8q_100e_100s_12n.py", Colors.OKCYAN))
    print(colored("  3. Run heavy benchmark:  python benchmarks/pennylane/8q_200e_200s_15n.py", Colors.OKCYAN))
    print(colored("\nResults will be saved to: D:/LRET/results/benchmark_YYYYMMDD_HHMMSS/\n", Colors.OKCYAN))
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(colored("\n\n✗ Setup interrupted by user", Colors.WARNING))
        sys.exit(1)
    except Exception as e:
        print(colored(f"\n✗ Unexpected error: {e}", Colors.FAIL))
        import traceback
        traceback.print_exc()
        sys.exit(1)
