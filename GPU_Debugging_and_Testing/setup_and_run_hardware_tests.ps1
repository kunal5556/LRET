# =============================================================================
# LRET Hardware-Dependent Tests - Windows Setup and Run Script
# =============================================================================
#
# This script sets up a Windows system for running LRET GPU/MPI tests.
# Requirements: Windows 10/11 with NVIDIA GPU
#
# Usage:
#   .\setup_and_run_hardware_tests.ps1 [-SkipInstall] [-OutputDir <path>]
#
# Prerequisites:
#   - Run as Administrator
#   - Internet connection for downloads
#   - NVIDIA GPU with drivers installed
#
# =============================================================================

param(
    [switch]$SkipInstall,
    [string]$OutputDir = "",
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"

# Colors
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Log-Info($message) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] " -NoNewline -ForegroundColor Cyan
    Write-Host $message
}

function Log-Success($message) {
    Write-Host "[SUCCESS] " -NoNewline -ForegroundColor Green
    Write-Host $message
}

function Log-Warning($message) {
    Write-Host "[WARNING] " -NoNewline -ForegroundColor Yellow
    Write-Host $message
}

function Log-Error($message) {
    Write-Host "[ERROR] " -NoNewline -ForegroundColor Red
    Write-Host $message
}

# Get script and project directories
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrEmpty($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent $ScriptDir
}

Log-Info "=============================================="
Log-Info "LRET Hardware Tests - Windows Setup and Run"
Log-Info "=============================================="
Log-Info "Project Root: $ProjectRoot"
Log-Info "Script Directory: $ScriptDir"

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin -and -not $SkipInstall) {
    Log-Warning "Not running as Administrator. Some installations may fail."
    Log-Warning "Consider running: Start-Process powershell -Verb runAs -ArgumentList '-File', '$($MyInvocation.MyCommand.Path)'"
}

# =============================================================================
# Step 1: Check Prerequisites
# =============================================================================

Log-Info "Step 1: Checking prerequisites..."

# Check for NVIDIA GPU
try {
    $gpuInfo = nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>$null
    if ($gpuInfo) {
        Log-Success "NVIDIA GPU detected:"
        $gpuInfo | ForEach-Object { Write-Host "  $_" }
    }
} catch {
    Log-Error "NVIDIA GPU not found or driver not installed"
    Log-Error "Please install NVIDIA drivers from: https://www.nvidia.com/drivers"
    exit 1
}

# Check for CUDA
$cudaPath = $env:CUDA_PATH
if ([string]::IsNullOrEmpty($cudaPath)) {
    $cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
}

if (Test-Path "$cudaPath\bin\nvcc.exe") {
    $nvccVersion = & "$cudaPath\bin\nvcc.exe" --version 2>&1 | Select-String "release"
    Log-Success "CUDA found: $nvccVersion"
} else {
    Log-Warning "CUDA not found at $cudaPath"
    Log-Warning "Please install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads"
    if (-not $SkipInstall) {
        Log-Info "Attempting to install CUDA via winget..."
        try {
            winget install NVIDIA.CUDA --silent
        } catch {
            Log-Warning "winget installation failed. Please install CUDA manually."
        }
    }
}

# =============================================================================
# Step 2: Install Build Tools
# =============================================================================

if (-not $SkipInstall) {
    Log-Info "Step 2: Checking/Installing build tools..."
    
    # Check for Visual Studio or Build Tools
    $vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vsWhere) {
        $vsPath = & $vsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($vsPath) {
            Log-Success "Visual Studio found: $vsPath"
        }
    } else {
        Log-Warning "Visual Studio not found. Please install Visual Studio 2022 with C++ tools"
        Log-Warning "Download from: https://visualstudio.microsoft.com/"
    }
    
    # Check for CMake
    try {
        $cmakeVersion = cmake --version 2>&1 | Select-Object -First 1
        Log-Success "CMake found: $cmakeVersion"
    } catch {
        Log-Warning "CMake not found. Installing via winget..."
        try {
            winget install Kitware.CMake --silent
        } catch {
            Log-Warning "Please install CMake manually from: https://cmake.org/download/"
        }
    }
    
    # Check for Git
    try {
        $gitVersion = git --version 2>&1
        Log-Success "Git found: $gitVersion"
    } catch {
        Log-Warning "Git not found. Installing via winget..."
        try {
            winget install Git.Git --silent
        } catch {
            Log-Warning "Please install Git manually from: https://git-scm.com/"
        }
    }
}

# =============================================================================
# Step 3: Install MS-MPI
# =============================================================================

if (-not $SkipInstall) {
    Log-Info "Step 3: Checking/Installing MS-MPI..."
    
    $msmpiPath = "C:\Program Files\Microsoft MPI\Bin\mpiexec.exe"
    if (Test-Path $msmpiPath) {
        Log-Success "MS-MPI found"
    } else {
        Log-Warning "MS-MPI not found. Installing..."
        
        # Download MS-MPI
        $msmpiUrl = "https://github.com/microsoft/Microsoft-MPI/releases/download/v10.1.1/msmpisetup.exe"
        $msmpiInstaller = "$env:TEMP\msmpisetup.exe"
        
        try {
            Invoke-WebRequest -Uri $msmpiUrl -OutFile $msmpiInstaller
            Start-Process -FilePath $msmpiInstaller -ArgumentList "-unattend" -Wait
            Remove-Item $msmpiInstaller -ErrorAction SilentlyContinue
            Log-Success "MS-MPI installed"
        } catch {
            Log-Warning "MS-MPI installation failed. Please install manually from:"
            Log-Warning "https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi"
        }
        
        # Also install SDK
        $msmpiSdkUrl = "https://github.com/microsoft/Microsoft-MPI/releases/download/v10.1.1/msmpisdk.msi"
        $msmpiSdkInstaller = "$env:TEMP\msmpisdk.msi"
        
        try {
            Invoke-WebRequest -Uri $msmpiSdkUrl -OutFile $msmpiSdkInstaller
            Start-Process msiexec.exe -ArgumentList "/i", $msmpiSdkInstaller, "/quiet" -Wait
            Remove-Item $msmpiSdkInstaller -ErrorAction SilentlyContinue
            Log-Success "MS-MPI SDK installed"
        } catch {
            Log-Warning "MS-MPI SDK installation failed."
        }
    }
}

# =============================================================================
# Step 4: Install Python Dependencies
# =============================================================================

Log-Info "Step 4: Installing Python dependencies..."

try {
    $pythonVersion = python --version 2>&1
    Log-Success "Python found: $pythonVersion"
} catch {
    Log-Error "Python not found. Please install Python 3.10+ from https://python.org"
    exit 1
}

# Create virtual environment
$venvPath = Join-Path $ProjectRoot "venv"
if (-not (Test-Path $venvPath)) {
    Log-Info "Creating virtual environment..."
    python -m venv $venvPath
}

# Activate and install packages
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    . $activateScript
    
    pip install --upgrade pip
    pip install numpy pytest scipy matplotlib
    
    # Try to install ML frameworks (may fail without proper CUDA)
    try {
        pip install torch --index-url https://download.pytorch.org/whl/cu121
    } catch {
        Log-Warning "PyTorch installation failed. GPU support may not be available."
        pip install torch
    }
    
    pip install pennylane jax jaxlib qiskit flax optax
    
    Log-Success "Python dependencies installed"
}

# =============================================================================
# Step 5: Download Eigen
# =============================================================================

Log-Info "Step 5: Checking/Installing Eigen3..."

$eigenPath = Join-Path $env:USERPROFILE "eigen-3.4.0"
if (-not (Test-Path $eigenPath)) {
    Log-Info "Downloading Eigen3..."
    $eigenUrl = "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"
    $eigenZip = "$env:TEMP\eigen-3.4.0.zip"
    
    try {
        Invoke-WebRequest -Uri $eigenUrl -OutFile $eigenZip
        Expand-Archive -Path $eigenZip -DestinationPath $env:USERPROFILE -Force
        Remove-Item $eigenZip -ErrorAction SilentlyContinue
        Log-Success "Eigen3 installed to: $eigenPath"
    } catch {
        Log-Warning "Eigen download failed. Please download manually."
    }
} else {
    Log-Success "Eigen3 found at: $eigenPath"
}

# =============================================================================
# Step 6: Build and Run Tests
# =============================================================================

Log-Info "Step 6: Building and running tests..."

Set-Location $ProjectRoot

# Set output directory
if ([string]::IsNullOrEmpty($OutputDir)) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputDir = Join-Path $ProjectRoot "test_output_$timestamp"
}

# Create output directory
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

# Run the Python test script
$testScript = Join-Path $ScriptDir "run_hardware_dependent_tests.py"

if (Test-Path $testScript) {
    $args = @(
        $testScript,
        "--skip-install",
        "--project-root", $ProjectRoot,
        "--output-dir", $OutputDir
    )
    
    Log-Info "Running test script..."
    python @args
    $testExitCode = $LASTEXITCODE
} else {
    Log-Error "Test script not found: $testScript"
    $testExitCode = 1
}

# =============================================================================
# Step 7: Summary
# =============================================================================

Write-Host ""
Log-Info "=============================================="
Log-Info "TEST EXECUTION COMPLETE"
Log-Info "=============================================="
Write-Host ""
Log-Info "Output directory: $OutputDir"
Log-Info "HTML Report: $OutputDir\report.html"
Log-Info "CSV Results: $OutputDir\test_results.csv"
Write-Host ""

if ($testExitCode -eq 0) {
    Log-Success "All tests completed successfully!"
} else {
    Log-Warning "Some tests failed or had errors. Check the reports for details."
}

# Open HTML report in browser
$htmlReport = Join-Path $OutputDir "report.html"
if (Test-Path $htmlReport) {
    $openReport = Read-Host "Open HTML report in browser? (Y/n)"
    if ($openReport -ne "n" -and $openReport -ne "N") {
        Start-Process $htmlReport
    }
}

exit $testExitCode
