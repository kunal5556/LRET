#!/bin/bash
# =============================================================================
# LRET Hardware-Dependent Tests - Full Setup and Run Script
# =============================================================================
#
# This script sets up a fresh Linux system for running LRET GPU/MPI tests.
# Designed for: Ubuntu 22.04/24.04 with NVIDIA GPUs
#
# Usage:
#   chmod +x setup_and_run_hardware_tests.sh
#   ./setup_and_run_hardware_tests.sh [--skip-reboot] [--output-dir DIR]
#
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
SKIP_REBOOT=false
OUTPUT_DIR=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-reboot)
            SKIP_REBOOT=true
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

log "=============================================="
log "LRET Hardware Tests - Full Setup and Run"
log "=============================================="
log "Project Root: $PROJECT_ROOT"
log "Script Directory: $SCRIPT_DIR"

# Check if running on Linux
if [[ "$(uname)" != "Linux" ]]; then
    error "This script only supports Linux"
    exit 1
fi

# Check if running as root or with sudo available
if [[ $EUID -ne 0 ]]; then
    if ! command -v sudo &> /dev/null; then
        error "This script requires root privileges or sudo"
        exit 1
    fi
    SUDO="sudo"
else
    SUDO=""
fi

# =============================================================================
# Step 1: System Update and Basic Tools
# =============================================================================

log "Step 1: Updating system and installing basic tools..."

$SUDO apt-get update
$SUDO apt-get upgrade -y

$SUDO apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    pkg-config \
    libeigen3-dev \
    libomp-dev \
    python3 \
    python3-pip \
    python3-venv

success "Basic tools installed"

# =============================================================================
# Step 2: Install NVIDIA Drivers and CUDA
# =============================================================================

log "Step 2: Checking/Installing NVIDIA drivers and CUDA..."

# Check if NVIDIA driver is installed
if ! command -v nvidia-smi &> /dev/null; then
    warn "NVIDIA driver not found. Installing..."
    
    # Add NVIDIA repository
    $SUDO apt-get install -y linux-headers-$(uname -r)
    
    # Install driver
    $SUDO apt-get install -y nvidia-driver-535 nvidia-utils-535
    
    warn "NVIDIA driver installed. A REBOOT may be required."
    if [[ "$SKIP_REBOOT" != "true" ]]; then
        warn "Please reboot and run this script again."
        warn "Use --skip-reboot to continue without rebooting (not recommended)"
        exit 0
    fi
else
    success "NVIDIA driver already installed"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
fi

# Check/Install CUDA Toolkit
if ! command -v nvcc &> /dev/null; then
    log "Installing CUDA Toolkit 12.4..."
    
    # Download and install CUDA keyring
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    $SUDO dpkg -i cuda-keyring_1.1-1_all.deb
    rm cuda-keyring_1.1-1_all.deb
    
    $SUDO apt-get update
    $SUDO apt-get install -y cuda-toolkit-12-4
    
    # Add to PATH
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    
    success "CUDA Toolkit installed"
else
    success "CUDA already installed: $(nvcc --version | grep release)"
fi

# =============================================================================
# Step 3: Install cuDNN
# =============================================================================

log "Step 3: Checking/Installing cuDNN..."

if [ ! -f /usr/include/cudnn.h ] && [ ! -f /usr/local/cuda/include/cudnn.h ]; then
    $SUDO apt-get install -y libcudnn8 libcudnn8-dev
    success "cuDNN installed"
else
    success "cuDNN already installed"
fi

# =============================================================================
# Step 4: Install NCCL
# =============================================================================

log "Step 4: Checking/Installing NCCL..."

if [ ! -f /usr/include/nccl.h ]; then
    $SUDO apt-get install -y libnccl2 libnccl-dev
    success "NCCL installed"
else
    success "NCCL already installed"
fi

# =============================================================================
# Step 5: Install MPI
# =============================================================================

log "Step 5: Checking/Installing OpenMPI..."

if ! command -v mpirun &> /dev/null; then
    $SUDO apt-get install -y openmpi-bin openmpi-common libopenmpi-dev
    success "OpenMPI installed"
else
    success "MPI already installed: $(mpirun --version | head -1)"
fi

# =============================================================================
# Step 6: Install Python Dependencies
# =============================================================================

log "Step 6: Installing Python dependencies..."

# Create virtual environment (optional)
if [ ! -d "$PROJECT_ROOT/venv" ]; then
    python3 -m venv "$PROJECT_ROOT/venv"
fi
source "$PROJECT_ROOT/venv/bin/activate"

pip install --upgrade pip
pip install numpy pytest jax jaxlib torch pennylane qiskit flax optax scipy matplotlib

success "Python dependencies installed"

# =============================================================================
# Step 7: Verify Installation
# =============================================================================

log "Step 7: Verifying installation..."

echo ""
echo "=== System Verification ==="
echo ""

echo "NVIDIA Driver:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo ""
echo "CUDA Version:"
nvcc --version | grep release

echo ""
echo "cuDNN:"
if [ -f /usr/include/cudnn_version.h ]; then
    cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2 | head -3
else
    echo "cuDNN headers found at alternate location"
fi

echo ""
echo "NCCL:"
if [ -f /usr/include/nccl.h ]; then
    grep "NCCL_VERSION" /usr/include/nccl.h | head -1
else
    echo "NCCL not found"
fi

echo ""
echo "MPI:"
mpirun --version | head -2

echo ""
echo "CMake:"
cmake --version | head -1

echo ""
echo "GCC:"
gcc --version | head -1

echo ""
success "All dependencies verified"

# =============================================================================
# Step 8: Run Hardware Tests
# =============================================================================

log "Step 8: Running hardware-dependent tests..."

cd "$PROJECT_ROOT"

# Determine output directory
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$PROJECT_ROOT/test_output_$(date +%Y%m%d_%H%M%S)"
fi

# Run the Python test script
python3 scripts/run_hardware_dependent_tests.py \
    --skip-install \
    --project-root "$PROJECT_ROOT" \
    --output-dir "$OUTPUT_DIR"

TEST_EXIT_CODE=$?

# =============================================================================
# Step 9: Summary
# =============================================================================

echo ""
log "=============================================="
log "TEST EXECUTION COMPLETE"
log "=============================================="
echo ""
log "Output directory: $OUTPUT_DIR"
log "HTML Report: $OUTPUT_DIR/report.html"
log "CSV Results: $OUTPUT_DIR/test_results.csv"
echo ""

if [ $TEST_EXIT_CODE -eq 0 ]; then
    success "All tests completed successfully!"
else
    warn "Some tests failed or had errors. Check the reports for details."
fi

exit $TEST_EXIT_CODE
