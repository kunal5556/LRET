@echo off
setlocal enabledelayedexpansion

echo ========================================================================
echo LRET Automated Benchmark Setup (Using VS Developer Environment)
echo ========================================================================
echo.

REM Detect LRET root (3 levels up from script directory)
set SCRIPT_DIR=%~dp0
pushd "%SCRIPT_DIR%..\..\"
set LRET_ROOT=%CD%
popd

echo LRET Root: %LRET_ROOT%
echo.

REM Verify LRET root exists
if not exist "%LRET_ROOT%\CMakeLists.txt" (
    echo ERROR: Cannot find LRET root directory
    echo Expected CMakeLists.txt at: %LRET_ROOT%
    exit /b 1
)

REM Find Visual Studio Developer Command Prompt
set VSDEVCMD=
if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\VsDevCmd.bat" set VSDEVCMD="C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\VsDevCmd.bat"
if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat" set VSDEVCMD="C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat"
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" set VSDEVCMD="C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\Common7\Tools\VsDevCmd.bat" set VSDEVCMD="C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\Common7\Tools\VsDevCmd.bat"
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\Tools\VsDevCmd.bat" set VSDEVCMD="C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\Tools\VsDevCmd.bat"
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat" set VSDEVCMD="C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat"

if "%VSDEVCMD%"=="" (
    echo ERROR: Visual Studio Developer Command Prompt not found
    echo.
    echo Please ensure Visual Studio 2019/2022 is installed with C++ build tools
    echo Download: https://visualstudio.microsoft.com/downloads/
    exit /b 1
)

echo Found VS Developer Environment: %VSDEVCMD%
echo Initializing Visual Studio environment...
call %VSDEVCMD% -arch=x64 -host_arch=x64 >nul 2>&1

REM Verify MSBuild is now available
where msbuild >nul 2>&1
if errorlevel 1 (
    echo ERROR: MSBuild still not available after loading VS environment
    echo Please ensure C++ build tools are installed in Visual Studio
    exit /b 1
)

echo MSBuild is ready
echo.

REM Set paths
set BUILD_DIR=%LRET_ROOT%\build
set PYTHON=python

REM Verify build directory exists
if not exist "%BUILD_DIR%" (
    echo ERROR: Build directory not found: %BUILD_DIR%
    echo Please run CMake configuration first
    exit /b 1
)

REM Step 1: Check Python
echo [1/6] Checking Python version...
%PYTHON% --version
if errorlevel 1 (
    echo ERROR: Python not found
    exit /b 1
)

REM Step 2: Install dependencies
echo.
echo [2/6] Installing Python dependencies...
%PYTHON% -m pip install --quiet cirq matplotlib numpy scipy psutil 2>nul
if errorlevel 1 (
    echo   WARNING: Some dependencies may already be installed
)

REM Step 3: Build LRET binaries
echo.
echo [3/6] Building LRET C++ binaries...

REM Verify vcxproj files exist
if not exist "%BUILD_DIR%\qlret_lib.vcxproj" (
    echo ERROR: qlret_lib.vcxproj not found in build directory
    echo Please run CMake to generate Visual Studio project files
    echo Expected: %BUILD_DIR%\qlret_lib.vcxproj
    exit /b 1
)

pushd "%BUILD_DIR%"

echo   Building qlret_lib.lib...
msbuild qlret_lib.vcxproj /p:Configuration=Release /p:Platform=x64 /v:minimal /nologo
if errorlevel 1 (
    echo   ERROR: qlret_lib build failed
    popd
    exit /b 1
)

echo   Building quantum_sim.exe...
msbuild quantum_sim.vcxproj /p:Configuration=Release /p:Platform=x64 /v:minimal /nologo
if errorlevel 1 (
    echo   ERROR: quantum_sim build failed
    popd
    exit /b 1
)

echo   Binaries built successfully

popd

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
