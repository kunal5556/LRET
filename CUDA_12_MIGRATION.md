# CUDA 12.x Migration Guide for Quadro P400

## Current Situation
- GPU: NVIDIA Quadro P400 (Compute Capability 6.1 - Pascal architecture)
- Installed: CUDA 13.1 (does NOT support Pascal GPUs)
- Issue: CUDA 13.x dropped support for compute capabilities < 7.5

## Solution: Install CUDA 12.x

### Option 1: Keep Both CUDA Versions (Recommended)
1. Download CUDA 12.6 (latest 12.x): https://developer.nvidia.com/cuda-12-6-0-download-archive
2. Install CUDA 12.6 alongside CUDA 13.1
3. Set environment variable to use CUDA 12.6 when building:
   `powershell
   $env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6'
   $env:CudaToolkitDir='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\'
   `

### Option 2: Replace CUDA 13.1 with CUDA 12.6
1. Uninstall CUDA 13.1 from Control Panel
2. Install CUDA 12.6
3. Restart system

## Code Changes Made
 Updated CMakeLists.txt to include compute capability 61 (Pascal)
  - Changed: CUDA_ARCHITECTURES '61;70;75;80;86;89;90'
  - Was: CUDA_ARCHITECTURES '70;75;80;86;89;90'

## What Does NOT Need to Change
 Source code (.cu files) - uses standard CUDA APIs compatible with 12.x
 Build system - already configured correctly
 Python wrapper - no changes needed
 Test infrastructure - works with any CUDA version

## Testing After CUDA 12.x Installation
`powershell
cd GPU_Debugging_and_Testing
$env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6'
$env:CudaToolkitDir='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\'
python run_hardware_dependent_tests.py --skip-install
`

## Compatibility Matrix
| CUDA Version | Pascal (6.1) | Volta (7.0) | Turing (7.5) | Ampere (8.0+) |
|--------------|--------------|-------------|--------------|---------------|
| CUDA 12.x    |  Yes        |  Yes       |  Yes        |  Yes         |
| CUDA 13.x    |  No         |  No        |  Yes        |  Yes         |

Your Quadro P400 (6.1) requires CUDA 12.x or earlier.
