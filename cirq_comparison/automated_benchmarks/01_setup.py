#!/usr/bin/env python3
"""
LRET Cirq Benchmark Setup (Based on working setup_pennylane_env.py)
===================================================================
This script automates the complete setup process:
  1. Check Python version
  2. Install Python dependencies (cirq, matplotlib, numpy, scipy, psutil)
  3. Build LRET C++ backend with CMake
  4. Verify quantum_sim.exe
  5. Test DEPOLARIZE gate support

Usage:
  python 01_setup.py
  python 01_setup.py --skip-build  # Skip C++ build if already done
"""

import sys
import os
import platform
import subprocess
import shutil
import json
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

def print_step(step_num, total_steps, message):
    """Print a formatted step header"""
    print(f"\n{'='*70}")
    print(colored(f"Step {step_num}/{total_steps}: {message}", Colors.BOLD))
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
    
    print(colored("\n╔═══════════════════════════════════════════════════════════════════╗", Colors.BOLD))
    print(colored("║   LRET Cirq Benchmarking - Automated Setup Script               ║", Colors.BOLD))
    print(colored("╚═══════════════════════════════════════════════════════════════════╝", Colors.BOLD))
    
    # Detect system
    system = platform.system()
    print(f"\nDetected OS: {colored(system, Colors.OKCYAN)}")
    print(f"Python: {colored(sys.version.split()[0], Colors.OKCYAN)} at {sys.executable}")
    
    # Determine project root (3 levels up: automated_benchmarks -> cirq_comparison -> LRET)
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent
    print(f"Project root: {colored(str(project_root), Colors.OKCYAN)}")
    
    # Verify LRET root
    if not (project_root / "CMakeLists.txt").exists():
        print(colored(f"✗ Cannot find LRET root directory", Colors.FAIL))
        print(f"Expected CMakeLists.txt at: {project_root}")
        return 1
    
    total_steps = 5 if skip_build else 6
    current_step = 0
    
    # ==========================================================================
    # STEP 1: Check Python Version
    # ==========================================================================
    current_step += 1
    print_step(current_step, total_steps, "Check Python Version")
    
    py_version = sys.version_info
    if py_version < (3, 8):
        print(colored(f"✗ Python {py_version.major}.{py_version.minor} is too old. Need Python 3.8+", Colors.FAIL))
        return 1
    elif py_version >= (3, 13):
        print(colored(f"⚠ Python {py_version.major}.{py_version.minor} - very new, tested on 3.8-3.12", Colors.WARNING))
    else:
        print(colored(f"✓ Python {py_version.major}.{py_version.minor} is compatible", Colors.OKGREEN))
    
    # ==========================================================================
    # STEP 2: Install Python Dependencies
    # ==========================================================================
    current_step += 1
    print_step(current_step, total_steps, "Install Python Dependencies")
    
    packages = [
        "cirq",
        "matplotlib",
        "numpy",
        "scipy",
        "psutil>=5.8",
    ]
    
    print(f"Installing packages: {', '.join(packages)}")
    pip_cmd = [get_python_executable(), "-m", "pip", "install"] + packages
    
    try:
        run_command(pip_cmd)
        print(colored("✓ Python dependencies installed successfully", Colors.OKGREEN))
    except subprocess.CalledProcessError:
        print(colored("⚠ Some packages may already be installed (continuing)", Colors.WARNING))
    
    # Verify installations
    print("\nVerifying installations...")
    try:
        import cirq
        print(colored(f"  ✓ Cirq {cirq.__version__}", Colors.OKGREEN))
    except ImportError:
        print(colored("  ✗ Cirq not found", Colors.FAIL))
        return 1
    
    try:
        import numpy as np
        print(colored(f"  ✓ NumPy {np.__version__}", Colors.OKGREEN))
    except ImportError:
        print(colored("  ✗ NumPy not found", Colors.FAIL))
        return 1
    
    # ==========================================================================
    # STEP 3: Build LRET C++ Backend (if not skipped)
    # ==========================================================================
    if not skip_build:
        current_step += 1
        print_step(current_step, total_steps, "Build LRET C++ Backend")
        
        # Check for CMake
        if not check_command_exists("cmake"):
            print(colored("✗ CMake not found in PATH", Colors.FAIL))
            print(colored("Please install CMake 3.16+ and add to PATH", Colors.WARNING))
            print(colored("  Windows: Download from https://cmake.org/download/", Colors.WARNING))
            return 1
        
        build_dir = project_root / "build"
        
        # On Windows, clear build directory if it exists (to avoid cached generator issues)
        if system == "Windows":
            if build_dir.exists():
                print(colored("\n⚠ Removing existing build directory to clear CMake cache...", Colors.WARNING))
                try:
                    shutil.rmtree(build_dir, ignore_errors=True)
                    import time
                    time.sleep(0.5)  # Give filesystem time to settle
                except Exception as e:
                    print(colored(f"  Warning: Could not fully clear build directory: {e}", Colors.WARNING))
        
        build_dir.mkdir(exist_ok=True)
        
        print(f"\nConfiguring CMake in {build_dir}...")
        os.chdir(build_dir)
        
        # CMake configure
        if system == "Windows":
            print("Detecting Visual Studio installation...")
            
            # Try to find vswhere (official VS locator tool)
            vswhere_path = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
            vs_path = None
            if vswhere_path.exists():
                try:
                    result = subprocess.run(
                        [str(vswhere_path), "-latest", "-property", "installationPath"],
                        capture_output=True, text=True, check=True
                    )
                    vs_path = result.stdout.strip()
                    print(colored(f"  Found Visual Studio at: {vs_path}", Colors.OKGREEN))
                except:
                    pass
            
            # Try Ninja generator first (recommended - works with any VS installation)
            print(colored("\n  Strategy 1: Trying Ninja generator (recommended)...", Colors.OKBLUE))
            success = False
            if check_command_exists("ninja"):
                # Setup MSVC environment for Ninja
                if vs_path:
                    vcvars_path = Path(vs_path) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                    if vcvars_path.exists():
                        print(colored(f"    Using MSVC environment from: {vcvars_path}", Colors.OKBLUE))
                        # Call vcvars64 and then cmake with Ninja
                        cmake_ninja = f'"{vcvars_path}" && cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release'
                        try:
                            result = subprocess.run(cmake_ninja, shell=True, check=True, 
                                                  capture_output=True, text=True)
                            print(colored("  ✓ Successfully configured with Ninja generator", Colors.OKGREEN))
                            success = True
                        except subprocess.CalledProcessError as e:
                            print(colored(f"  ✗ Ninja generator failed: {e.stderr[:200]}", Colors.WARNING))
                else:
                    # Try Ninja without vcvars (might work if VS is in PATH)
                    try:
                        run_command(["cmake", "..", "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release"], 
                                  capture_output=True)
                        print(colored("  ✓ Successfully configured with Ninja generator", Colors.OKGREEN))
                        success = True
                    except:
                        pass
            else:
                print(colored("    Ninja not found, skipping...", Colors.WARNING))
            
            # Fallback: Try Visual Studio generators
            if not success:
                print(colored("\n  Strategy 2: Trying Visual Studio generators...", Colors.OKBLUE))
                
                for generator_name, generator_arg in [
                    ("Visual Studio 17 2022", "Visual Studio 17 2022"),
                    ("Visual Studio 16 2019", "Visual Studio 16 2019"),
                ]:
                    try:
                        print(colored(f"    Trying {generator_name}...", Colors.OKBLUE))
                        result = subprocess.run(
                            ["cmake", "..", "-G", generator_arg, "-A", "x64"],
                            capture_output=True, text=True, check=True
                        )
                        success = True
                        print(colored(f"  ✓ Successfully configured with {generator_name}", Colors.OKGREEN))
                        break
                    except subprocess.CalledProcessError:
                        print(colored(f"    ✗ {generator_name} failed", Colors.WARNING))
                        continue
            
            if not success:
                print(colored("\n✗ Could not configure CMake with any generator", Colors.FAIL))
                print(colored("\nSolutions:", Colors.OKGREEN))
                print(colored("  1. Repair Visual Studio installation", Colors.OKGREEN))
                print(colored("  2. Ensure 'Desktop development with C++' is installed", Colors.OKGREEN))
                print(colored("  3. Or install Ninja: choco install ninja", Colors.OKGREEN))
                return 1
        else:
            # Linux/Mac
            cmake_config = ["cmake", ".."]
            try:
                run_command(cmake_config)
                success = True
            except subprocess.CalledProcessError:
                success = False
        
        if not success:
            print(colored("✗ CMake configuration failed", Colors.FAIL))
            return 1
        
        # CMake build
        print("\nBuilding LRET (this may take several minutes)...")
        
        # Detect if we used Ninja or VS generator
        cmake_cache = build_dir / "CMakeCache.txt"
        using_ninja = False
        if cmake_cache.exists():
            with open(cmake_cache, 'r') as f:
                if 'Ninja' in f.read():
                    using_ninja = True
        
        if system == "Windows":
            if using_ninja:
                cmake_build = ["cmake", "--build", "."]
            else:
                cmake_build = ["cmake", "--build", ".", "--config", "Release"]
        else:
            import multiprocessing
            n_cores = multiprocessing.cpu_count()
            cmake_build = ["cmake", "--build", ".", "-j", str(n_cores)]
        
        try:
            run_command(cmake_build)
            print(colored("✓ LRET C++ backend built successfully", Colors.OKGREEN))
        except subprocess.CalledProcessError:
            print(colored("✗ Build failed", Colors.FAIL))
            return 1
        
        os.chdir(project_root)
    
    # ==========================================================================
    # STEP 4: Verify quantum_sim.exe
    # ==========================================================================
    current_step += 1
    print_step(current_step, total_steps, "Verify quantum_sim.exe")
    
    build_dir = project_root / "build"
    if not build_dir.exists():
        print(colored(f"✗ Build directory not found: {build_dir}", Colors.FAIL))
        print(colored("Please run CMake configuration first or remove --skip-build", Colors.WARNING))
        return 1
    
    if system == "Windows":
        exe_path = build_dir / "Release" / "quantum_sim.exe"
    else:
        exe_path = build_dir / "quantum_sim"
    
    if not exe_path.exists():
        print(colored(f"✗ quantum_sim executable not found at {exe_path}", Colors.FAIL))
        return 1
    print(colored(f"✓ Found: {exe_path}", Colors.OKGREEN))
    
    # Test basic circuit
    print("\nTesting basic circuit...")
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
    
    try:
        result = subprocess.run([
            str(exe_path),
            "--input-json", str(test_file),
            "--output", str(test_output)
        ], capture_output=True, text=True, check=True)
        
        if not test_output.exists():
            print(colored("✗ Basic test failed - no output file", Colors.FAIL))
            return 1
        print(colored("✓ Basic circuit test passed", Colors.OKGREEN))
    except subprocess.CalledProcessError as e:
        print(colored(f"✗ Basic test failed: {e.stderr}", Colors.FAIL))
        return 1
    
    # ==========================================================================
    # STEP 5: Test DEPOLARIZE Gate
    # ==========================================================================
    current_step += 1
    print_step(current_step, total_steps, "Test DEPOLARIZE Gate Support")
    
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
    
    try:
        result = subprocess.run([
            str(exe_path),
            "--input-json", str(test_depol_file),
            "--output", str(test_depol_output)
        ], capture_output=True, text=True, check=True)
        
        if not test_depol_output.exists():
            print(colored("✗ DEPOLARIZE test failed - no output file", Colors.FAIL))
            return 1
        print(colored("✓ DEPOLARIZE gate working", Colors.OKGREEN))
    except subprocess.CalledProcessError as e:
        print(colored(f"✗ DEPOLARIZE test failed: {e.stderr}", Colors.FAIL))
        return 1
    
    # ==========================================================================
    # SUCCESS!
    # ==========================================================================
    print("\n" + "="*70)
    print(colored("🎉 SETUP COMPLETE! 🎉", Colors.OKGREEN + Colors.BOLD))
    print("="*70)
    print(colored("\n✓ All checks passed. Ready to run benchmarks!\n", Colors.OKGREEN))
    
    print("Next step:")
    print(colored("  python 02_run_benchmark.py", Colors.OKCYAN))
    print(colored("\nResults will be saved to: cirq_comparison/benchmark_results_TIMESTAMP/\n", Colors.OKCYAN))
    
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
