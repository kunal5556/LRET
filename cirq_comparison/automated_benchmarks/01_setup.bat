@echo off
REM LRET Automated Benchmark - Environment Setup
REM This script installs dependencies and builds LRET binaries

echo ========================================================================
echo LRET Automated Benchmark Setup
echo ========================================================================
echo.

REM Auto-detect LRET root directory (go up 2 levels from automated_benchmarks)
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%..\.."
set LRET_ROOT=%CD%
cd /d "%SCRIPT_DIR%"

echo LRET Root: %LRET_ROOT%
echo.

set BUILD_DIR=%LRET_ROOT%\build
set MSBUILD="C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\MSBuild\Current\Bin\MSBuild.exe"

REM Verify LRET root exists
if not exist "%LRET_ROOT%\CMakeLists.txt" (
    echo ERROR: Cannot find LRET root directory
    echo Expected CMakeLists.txt at: %LRET_ROOT%
    exit /b 1
)

REM Verify build directory exists
if not exist "%BUILD_DIR%" (
    echo ERROR: Build directory not found: %BUILD_DIR%
    echo Please run CMake configuration first
    exit /b 1
)

REM Step 1: Check Python
echo [1/6] Checking Python version...
python --version
if errorlevel 1 (
    echo ERROR: Python not found
    exit /b 1
)

REM Step 2: Install dependencies
echo.
echo [2/6] Installing Python dependencies...
python -m pip install --user --upgrade --quiet cirq matplotlib numpy scipy psutil
echo   Done

REM Step 3: Build LRET binaries
echo.
echo [3/6] Building LRET C++ binaries...

cd /d "%BUILD_DIR%"

echo   Building qlret_lib.lib...
%MSBUILD% qlret_lib.vcxproj /p:Configuration=Release /p:Platform=x64 /v:minimal /nologo
if errorlevel 1 (
    echo   ERROR: qlret_lib build failed
    exit /b 1
)

echo   Building quantum_sim.exe...
%MSBUILD% quantum_sim.vcxproj /p:Configuration=Release /p:Platform=x64 /v:minimal /nologo
if errorlevel 1 (
    echo   ERROR: quantum_sim build failed
    exit /b 1
)

echo   Binaries built successfully

cd /d "%LRET_ROOT%"

REM Step 4: Verify binary
echo.
echo [4/6] Verifying quantum_sim.exe...

set EXE=%BUILD_DIR%\Release\quantum_sim.exe
if not exist "%EXE%" (
    echo   ERROR: quantum_sim.exe not found
    exit /b 1
)

REM Step 5: Test basic circuit
echo {"circuit":{"num_qubits":2,"operations":[{"name":"H","wires":[0]},{"name":"CNOT","wires":[0,1]}]},"config":{"epsilon":0.0001,"initial_rank":1}} > %BUILD_DIR%\test_basic.json
"%EXE%" --input-json %BUILD_DIR%\test_basic.json --output %BUILD_DIR%\test_basic_out.json > nul 2>&1
if not exist "%BUILD_DIR%\test_basic_out.json" (
    echo   ERROR: Basic test failed
    exit /b 1
)
echo   Basic circuit test passed

REM Step 6: Test DEPOLARIZE
echo.
echo [5/6] Testing DEPOLARIZE gate...

echo {"circuit":{"num_qubits":2,"operations":[{"name":"H","wires":[0]},{"name":"DEPOLARIZE","wires":[0],"params":[0.1]},{"name":"CNOT","wires":[0,1]}]},"config":{"epsilon":0.0001,"initial_rank":1}} > %BUILD_DIR%\test_depol.json
"%EXE%" --input-json %BUILD_DIR%\test_depol.json --output %BUILD_DIR%\test_depol_out.json > nul 2>&1
if not exist "%BUILD_DIR%\test_depol_out.json" (
    echo   ERROR: DEPOLARIZE test failed
    exit /b 1
)
echo   DEPOLARIZE gate working

REM Summary
echo.
echo [6/6] Setup Summary
echo   Python dependencies: OK
echo   LRET binaries: OK
echo   quantum_sim.exe: OK
echo   DEPOLARIZE support: OK

echo.
echo ========================================================================
echo SETUP COMPLETE - Ready to run benchmarks!
echo ========================================================================
echo.
echo Next: python 02_run_benchmark.py
echo.

exit /b 0
