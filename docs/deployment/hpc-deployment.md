# HPC Deployment Guide

Deploy LRET on High-Performance Computing (HPC) clusters and supercomputers for extremely large-scale quantum simulations.

## Overview

HPC deployment enables:
- **Massive simulations** (25+ qubits)
- **Distributed memory** parallelism with MPI
- **Node-level parallelism** with OpenMP
- **GPU acceleration** across multiple nodes
- **Job scheduling** integration

## Supported HPC Systems

- **Slurm** - Most common HPC scheduler
- **PBS/Torque** - Traditional HPC scheduler
- **LSF** - IBM Platform LSF
- **SGE** - Sun Grid Engine
- **MOAB** - Adaptive Computing

---

## Building LRET for HPC

### Module System

Most HPC systems use module systems:

```bash
# Load required modules
module load gcc/11.2.0
module load cmake/3.22
module load openmpi/4.1.2
module load cuda/12.0  # If GPU support needed

# Verify
module list
```

### Build with MPI

```bash
# Clone repository
git clone https://github.com/kunal5556/LRET.git
cd LRET

# Create build directory
mkdir build && cd build

# Configure with MPI and optimizations
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_MPI=ON \
    -DENABLE_OPENMP=ON \
    -DENABLE_AVX2=ON \
    -DCMAKE_CXX_COMPILER=mpic++ \
    -DCMAKE_INSTALL_PREFIX=$HOME/lret

# Build with multiple cores
make -j 16

# Install
make install
```

### Build with GPU Support

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_MPI=ON \
    -DENABLE_GPU=ON \
    -DCMAKE_CUDA_ARCHITECTURES=80 \  # Adjust for your GPU
    -DCMAKE_INSTALL_PREFIX=$HOME/lret

make -j 16
make install
```

---

## Slurm Integration

### Basic Job Script

`lret_job.slurm`:
```bash
#!/bin/bash
#SBATCH --job-name=lret_simulation
#SBATCH --output=lret_%j.out
#SBATCH --error=lret_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --partition=standard
#SBATCH --account=my_project

# Load modules
module load gcc/11.2.0 openmpi/4.1.2

# Set environment
export OMP_NUM_THREADS=1
export LRET_THREADS=32

# Run simulation
mpirun -np 128 lret-mpi \
    --qubits 25 \
    --gates-file circuit.txt \
    --output results.csv \
    --shots 10000
```

Submit:
```bash
sbatch lret_job.slurm
```

### GPU Job Script

`lret_gpu.slurm`:
```bash
#!/bin/bash
#SBATCH --job-name=lret_gpu
#SBATCH --output=lret_gpu_%j.out
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --partition=gpu

module load gcc/11.2.0 cuda/12.0 openmpi/4.1.2

# 4 GPUs per node, 2 nodes = 8 total GPUs
mpirun -np 8 lret-mpi \
    --qubits 28 \
    --use-gpu \
    --gates-file large_circuit.txt \
    --output gpu_results.csv
```

### Hybrid MPI+OpenMP

`lret_hybrid.slurm`:
```bash
#!/bin/bash
#SBATCH --job-name=lret_hybrid
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00

module load gcc/11.2.0 openmpi/4.1.2

# 4 MPI ranks per node, 8 OpenMP threads each
export OMP_NUM_THREADS=8
export OMP_PLACES=cores
export OMP_PROC_BIND=close

mpirun -np 32 lret-mpi \
    --qubits 30 \
    --gates-file massive_circuit.txt \
    --hybrid-parallel \
    --output hybrid_results.csv
```

### Array Jobs

Process multiple circuits:

`lret_array.slurm`:
```bash
#!/bin/bash
#SBATCH --job-name=lret_array
#SBATCH --output=lret_array_%A_%a.out
#SBATCH --array=1-100
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=01:00:00

module load gcc/11.2.0 openmpi/4.1.2

# Process circuit based on array index
CIRCUIT_FILE="circuits/circuit_${SLURM_ARRAY_TASK_ID}.json"
OUTPUT_FILE="results/result_${SLURM_ARRAY_TASK_ID}.csv"

mpirun -np 32 lret-mpi \
    --input $CIRCUIT_FILE \
    --output $OUTPUT_FILE
```

Submit:
```bash
sbatch lret_array.slurm
```

### Job Management

```bash
# Submit job
sbatch lret_job.slurm

# Check job status
squeue -u $USER

# Cancel job
scancel <job_id>

# View output (for running job)
tail -f lret_<job_id>.out

# Job efficiency report
seff <job_id>
```

---

## PBS/Torque Integration

### PBS Job Script

`lret_job.pbs`:
```bash
#!/bin/bash
#PBS -N lret_simulation
#PBS -l nodes=4:ppn=32
#PBS -l walltime=04:00:00
#PBS -q standard
#PBS -A my_project

cd $PBS_O_WORKDIR

module load gcc/11.2.0 openmpi/4.1.2

mpirun -np 128 lret-mpi \
    --qubits 25 \
    --gates-file circuit.txt \
    --output results.csv
```

Submit:
```bash
qsub lret_job.pbs
```

### PBS Job Management

```bash
# Submit
qsub lret_job.pbs

# Status
qstat -u $USER

# Delete
qdel <job_id>

# Hold/Release
qhold <job_id>
qrls <job_id>
```

---

## LSF Integration

### LSF Job Script

`lret_job.lsf`:
```bash
#!/bin/bash
#BSUB -J lret_simulation
#BSUB -n 128
#BSUB -R "span[ptile=32]"
#BSUB -W 4:00
#BSUB -o lret_%J.out
#BSUB -e lret_%J.err

module load gcc/11.2.0 openmpi/4.1.2

mpirun lret-mpi \
    --qubits 25 \
    --gates-file circuit.txt \
    --output results.csv
```

Submit:
```bash
bsub < lret_job.lsf
```

---

## MPI Parallel Execution

### Distribution Strategies

LRET supports multiple MPI distribution strategies:

#### 1. State Vector Distribution

```bash
mpirun -np 16 lret-mpi \
    --qubits 28 \
    --distribution state-vector \
    --gates-file circuit.txt
```

Each rank holds portion of state vector.
- **Best for**: Large qubit counts (25+)
- **Memory**: O(2^n / P) per process
- **Communication**: High

#### 2. Shot Distribution

```bash
mpirun -np 32 lret-mpi \
    --qubits 20 \
    --shots 10000 \
    --distribution shots \
    --gates-file circuit.txt
```

Each rank simulates subset of shots.
- **Best for**: Many shots, moderate qubits
- **Memory**: O(2^n) per process
- **Communication**: Low

#### 3. Hybrid Distribution

```bash
mpirun -np 64 lret-mpi \
    --qubits 26 \
    --shots 1000 \
    --distribution hybrid \
    --state-ranks 16 \
    --shot-ranks 4 \
    --gates-file circuit.txt
```

Combines both strategies.
- **Best for**: Very large simulations
- **Memory**: Flexible
- **Communication**: Medium

### MPI Configuration

```bash
# Set MPI environment
export OMPI_MCA_btl=^openib  # Disable InfiniBand (if issues)
export OMPI_MCA_btl_tcp_if_include=eth0  # Specify network interface

# Binding
mpirun --bind-to core -np 128 lret-mpi ...
mpirun --map-by socket -np 64 lret-mpi ...

# Node file
mpirun -np 128 --hostfile nodes.txt lret-mpi ...
```

---

## Performance Tuning

### Node-Level Optimization

#### Core Binding

```bash
# Slurm
#SBATCH --cpu-bind=cores

# PBS
#PBS -l place=scatter:excl

# Manual
export OMP_PROC_BIND=true
export OMP_PLACES=cores
```

#### NUMA Awareness

```bash
# Check NUMA topology
numactl --hardware

# Run with NUMA binding
numactl --cpunodebind=0 --membind=0 lret ...
```

#### Huge Pages

```bash
# Enable transparent huge pages
echo always > /sys/kernel/mm/transparent_hugepage/enabled

# Or pre-allocate
hugeadm --pool-pages-min 2MB:1024
```

### Network Optimization

#### InfiniBand

```bash
# Enable InfiniBand
export OMPI_MCA_btl=openib,self,sm

# Tune parameters
export OMPI_MCA_btl_openib_receive_queues=P,128,32:P,2048,32:P,12288,32
```

#### High-Speed Networks

```bash
# For Cray systems
export MPICH_GNI_NDREG_ENTRIES=1024

# For Intel Omni-Path
export PSM2_MEMORY=large
```

### Memory Management

```bash
# Increase locked memory limit
ulimit -l unlimited

# Set memory allocation strategy
export MALLOC_MMAP_THRESHOLD_=131072
export MALLOC_TRIM_THRESHOLD_=131072
```

---

## Specific HPC Systems

### NERSC (Perlmutter)

```bash
# Load modules
module load PrgEnv-gnu
module load cmake/3.24
module load cudatoolkit

# Build
CC=cc CXX=CC cmake .. -DENABLE_GPU=ON
make -j 16

# Submit job
sbatch -C gpu -A myaccount lret_gpu.slurm
```

### TACC (Stampede2/Frontera)

```bash
# Stampede2
module load gcc/11.2.0 impi/19.0.9 cmake/3.24

# Frontera
module load gcc/11.2.0 mvapich2/2.3.7

# Submit
sbatch -p normal lret_job.slurm
```

### OLCF (Summit)

```bash
# Load modules
module load gcc/11.2.0 cuda/12.0 cmake/3.23

# Build with GPU
cmake .. -DENABLE_GPU=ON -DCMAKE_CUDA_ARCHITECTURES=70
make -j 16

# Submit with jsrun
jsrun -n 24 -g 6 -a 1 lret-mpi --qubits 30 --use-gpu
```

### ALCF (Theta/Polaris)

```bash
# Polaris
module load gcc/11.2.0 openmpi/4.1.4 cuda/11.8

# Submit
qsub -A myproject -q gpu lret_job.pbs
```

---

## Monitoring and Profiling

### Resource Usage

```bash
# Monitor job
watch -n 1 'squeue -j <job_id> -o "%.18i %.9P %.50j %.8u %.2t %.10M %.6D %.4C %.8m"'

# Detailed stats
sacct -j <job_id> --format=JobID,JobName,Partition,State,Elapsed,MaxRSS,MaxVMSize
```

### Performance Profiling

```bash
# Profile with HPCToolkit
srun hpcrun -e WALLCLOCK@5000 lret-mpi ...
hpcstruct lret-mpi
hpcprof -S lret-mpi.hpcstruct hpcrun-measurements

# Profile with TAU
tau_exec -T mpi,openmp lret-mpi ...
pprof

# Profile with Score-P
scorep mpirun -np 128 lret-mpi ...
```

### GPU Profiling

```bash
# NVIDIA Nsight
nsys profile --stats=true mpirun -np 4 lret-mpi --use-gpu ...

# nvprof (deprecated but still useful)
nvprof --print-gpu-trace mpirun -np 4 lret-mpi --use-gpu ...
```

---

## Checkpointing

### Enable Checkpointing

```bash
lret-mpi \
    --qubits 28 \
    --checkpoint-freq 1000 \
    --checkpoint-dir /scratch/checkpoints \
    --gates-file circuit.txt
```

### Restart from Checkpoint

```bash
lret-mpi \
    --qubits 28 \
    --restart-from /scratch/checkpoints/checkpoint_5000.dat \
    --gates-file circuit.txt
```

### Job Script with Checkpointing

```bash
#!/bin/bash
#SBATCH --job-name=lret_checkpoint
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@300  # Signal 5 min before timeout

# Cleanup handler
cleanup() {
    echo "Time limit approaching, creating checkpoint..."
    # LRET handles checkpoint automatically on signal
}

trap cleanup USR1

mpirun -np 128 lret-mpi \
    --qubits 30 \
    --checkpoint-freq 500 \
    --checkpoint-dir $SCRATCH/checkpoints \
    --gates-file large_circuit.txt
```

---

## Best Practices

### 1. Start Small

Test on single node before scaling:
```bash
# Single node test
sbatch --nodes=1 --ntasks=32 lret_test.slurm

# Verify correctness and scaling
# Then scale to multiple nodes
```

### 2. Use Scratch Storage

```bash
# Set scratch directory
export LRET_SCRATCH=$SCRATCH/lret_runs

# Write outputs to scratch
lret-mpi --output $SCRATCH/results.csv
```

### 3. Monitor Efficiency

```bash
# After job completes
seff <job_id>

# Look for:
# - CPU efficiency > 90%
# - Memory usage appropriate for problem size
```

### 4. Job Chaining

For long simulations:
```bash
#!/bin/bash
#SBATCH --dependency=afterok:<previous_job_id>

# Continue from previous job's checkpoint
```

### 5. Data Management

```bash
# Archive results
tar czf results_${SLURM_JOB_ID}.tar.gz results/

# Transfer off cluster
scp results_*.tar.gz user@external:/path/
```

---

## Troubleshooting

### Out of Memory

```bash
# Reduce memory usage
lret-mpi --sparse-threshold 0.01 \
         --use-disk-cache \
         --cache-dir $SCRATCH/cache

# Request more memory
#SBATCH --mem=500G
```

### MPI Communication Errors

```bash
# Enable verbose MPI output
mpirun --mca btl_base_verbose 30 -np 128 lret-mpi ...

# Test MPI installation
mpirun -np 128 hostname
```

### GPU Not Detected

```bash
# Check GPU visibility
nvidia-smi

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

### Slow Performance

```bash
# Profile to identify bottleneck
lret-mpi --profile --profile-output profile.json ...

# Check for load imbalance
```

---

## Example Workflows

### Workflow 1: Parameter Sweep

```bash
#!/bin/bash
#SBATCH --array=0-99
#SBATCH --nodes=2
#SBATCH --ntasks=64

# Sweep noise parameter
NOISE=$(echo "scale=4; $SLURM_ARRAY_TASK_ID / 1000" | bc)

mpirun -np 64 lret-mpi \
    --qubits 24 \
    --noise-level $NOISE \
    --gates-file circuit.txt \
    --output results/noise_${NOISE}.csv
```

### Workflow 2: Multi-Stage Simulation

```bash
# Stage 1: Coarse simulation
sbatch --job-name=stage1 stage1.slurm

# Stage 2: Refine (depends on stage 1)
sbatch --dependency=afterok:$STAGE1_ID stage2.slurm

# Stage 3: Analysis
sbatch --dependency=afterok:$STAGE2_ID analyze.slurm
```

---

## Further Reading

- [Docker Deployment](docker-guide.md)
- [Cloud Deployment](cloud-deployment.md)
- [Performance Guide](../developer-guide/06-performance.md)
- [MPI Parallelization](../developer-guide/02-code-structure.md#mpi)
