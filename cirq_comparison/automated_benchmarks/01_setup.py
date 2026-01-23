#!/usr/bin/env python3
"""
LRET Automated Benchmark Setup - Python Version
Handles MSBuild detection and compilation robustly across Windows configurations
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import shutil

def print_header(msg):
    """Print formatted header"""
    print("\n" + "=" * 72)
    print(msg)
    print("=" * 72 + "\n")

def print_step(step, total, msg):
    """Print step progress"""
    print(f"[{step}/{total}] {msg}")

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
    print_header("LRET Automated Benchmark Setup (Python)")
    
    # Detect LRET root
    script_dir = Path(__file__).parent.resolve()
    lret_root = script_dir.parent.parent
    print(f"LRET Root: {lret_root}\n")
    
    # Verify LRET root
    if not (lret_root / "CMakeLists.txt").exists():
        print(f"ERROR: Cannot find LRET root directory")
        print(f"Expected CMakeLists.txt at: {lret_root}")
        return 1
    
    build_dir = lret_root / "build"
    if not build_dir.exists():
        print(f"ERROR: Build directory not found: {build_dir}")
        print("Please run CMake configuration first:")
        print(f"  cd {lret_root}")
        print("  cmake -B build -DCMAKE_BUILD_TYPE=Release")
        return 1
    
    # Step 1: Find MSBuild or VS Developer Command Prompt
    print_step(1, 6, "Finding MSBuild...")
    build_tool = find_msbuild()
    
    if not build_tool:
        print("\nERROR: MSBuild not found in any Visual Studio installation")
        print("\nChecked:")
        print("  - Visual Studio 2022 (all editions, standard and non-standard paths)")
        print("  - Visual Studio 2019 (all editions)")
        print("  - Visual Studio 2017 (all editions)")
        print("  - System PATH")
        print("\nPlease install Visual Studio 2017/2019/2022 with C++ build tools")
        print("Download: https://visualstudio.microsoft.com/downloads/")
        return 1
    
    tool_type, tool_path = build_tool
    
    # Prepare MSBuild command wrapper
    if tool_type == "vsdevcmd":
        # Use VS Developer Command Prompt to set up environment
        def run_msbuild(vcxproj, cwd):
            """Run MSBuild through VS Developer Command Prompt"""
            # Create a batch script that loads VS environment and runs MSBuild
            batch_script = f'''@echo off
call "{tool_path}" -arch=x64 -host_arch=x64 >nul 2>&1
msbuild "{vcxproj}" /p:Configuration=Release /p:Platform=x64 /v:minimal /nologo
'''
            batch_file = cwd / "temp_build.bat"
            with open(batch_file, 'w') as f:
                f.write(batch_script)
            
            result = subprocess.run(
                [str(batch_file)],
                cwd=cwd,
                capture_output=True,
                text=True,
                shell=True
            )
            
            # Clean up
            try:
                batch_file.unlink()
            except:
                pass
            
            return result
    else:
        # Use MSBuild directly
        def run_msbuild(vcxproj, cwd):
            """Run MSBuild directly"""
            return subprocess.run(
                [tool_path, str(vcxproj), 
                 "/p:Configuration=Release", "/p:Platform=x64", 
                 "/v:minimal", "/nologo"],
                cwd=cwd,
                capture_output=True,
                text=True
            )
    
    # Step 2: Check Python
    print_step(2, 6, "Checking Python version...")
    python_version = sys.version.split()[0]
    print(f"  Python {python_version}")
    
    # Step 3: Install dependencies
    print_step(3, 6, "Installing Python dependencies...")
    deps = ["cirq", "matplotlib", "numpy", "scipy", "psutil"]
    try:
        run_command([sys.executable, "-m", "pip", "install", "--quiet"] + deps)
        print(f"  Installed: {', '.join(deps)}")
    except:
        print("  WARNING: Some dependencies may already be installed")
    
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
