# Docker Deployment Guide

This guide covers deploying LRET using Docker for development and production environments.

## Quick Start

### Pull Pre-built Image

```bash
docker pull kunal5556/lret:latest
docker run -it kunal5556/lret:latest
```

### Build from Source

```bash
cd lret/
docker build -t lret:local .
docker run -it lret:local
```

## Available Docker Images

| Image Tag | Description | Size | Use Case |
|-----------|-------------|------|----------|
| `latest` | Production build with all features | ~500MB | Production deployment |
| `dev` | Development build with debugging | ~800MB | Development |
| `gpu` | GPU-accelerated build (CUDA) | ~2GB | GPU workloads |
| `minimal` | Minimal build (no Python) | ~200MB | CLI-only usage |

## Dockerfile Configurations

### Production Dockerfile

The main `Dockerfile` provides a production-ready build:

```dockerfile
FROM ubuntu:22.04
# Optimized for size and performance
# Includes: C++ library, Python bindings, CLI
```

**Features:**
- Multi-stage build for smaller image
- AVX2/SIMD optimizations enabled
- Python 3.10 with qlret package
- All dependencies included

**Build:**
```bash
docker build -t lret:prod .
```

### Development Dockerfile

The `Dockerfile.dev` includes development tools:

```dockerfile
FROM ubuntu:22.04
# Includes debugging tools and test suites
```

**Additional features:**
- GDB, Valgrind for debugging
- Test suites and benchmarks
- Development headers
- Hot reload capabilities

**Build:**
```bash
docker build -f Dockerfile.dev -t lret:dev .
```

### GPU Dockerfile

The `Dockerfile.gpu` enables GPU acceleration:

```dockerfile
FROM nvidia/cuda:12.0-devel-ubuntu22.04
# CUDA-enabled build for GPU simulations
```

**Requirements:**
- NVIDIA GPU with CUDA support
- nvidia-docker installed

**Build:**
```bash
docker build -f Dockerfile.gpu -t lret:gpu .
```

**Run:**
```bash
docker run --gpus all -it lret:gpu
```

## Running Simulations in Docker

### Interactive Shell

```bash
docker run -it lret:latest /bin/bash
```

Inside container:
```bash
# C++ simulator
lret --qubits 10 --gates "H 0; CNOT 0 1" --shots 1000

# Python
python3 -c "from qlret import QuantumSimulator; sim = QuantumSimulator(2); sim.h(0); sim.cx(0,1); print(sim.measure())"
```

### Single Command Execution

```bash
docker run lret:latest lret --qubits 5 --gates "H 0; H 1; H 2; H 3; H 4" --shots 100
```

### Python Script Execution

```bash
# Mount local script
docker run -v $(pwd):/workspace lret:latest python3 /workspace/my_circuit.py
```

## Volume Mounts

### Mount Local Data

```bash
docker run -v /path/to/local/data:/data lret:latest lret --input /data/circuit.json
```

### Mount Output Directory

```bash
docker run -v $(pwd)/results:/output lret:latest lret \
    --qubits 10 \
    --gates "..." \
    --output /output/results.csv
```

### Mount Configuration

```bash
docker run -v $(pwd)/config:/config lret:latest lret --config /config/settings.json
```

## Docker Compose

### Basic Compose File

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  lret:
    image: lret:latest
    volumes:
      - ./data:/data
      - ./results:/output
    environment:
      - LRET_THREADS=4
      - LRET_NOISE_LEVEL=0.01
    command: lret --qubits 10 --gates "H 0; CNOT 0 1" --shots 1000
```

Run:
```bash
docker-compose up
```

### Multi-Container Setup

For distributed simulations:

```yaml
version: '3.8'

services:
  master:
    image: lret:latest
    command: lret-mpi --master --workers 3
    ports:
      - "8080:8080"
  
  worker1:
    image: lret:latest
    command: lret-mpi --worker --master-host master
    depends_on:
      - master
  
  worker2:
    image: lret:latest
    command: lret-mpi --worker --master-host master
    depends_on:
      - master
  
  worker3:
    image: lret:latest
    command: lret-mpi --worker --master-host master
    depends_on:
      - master
```

## Environment Variables

Configure LRET behavior via environment variables:

```bash
docker run -e LRET_THREADS=8 \
           -e LRET_MEMORY_LIMIT=16GB \
           -e LRET_NOISE_LEVEL=0.01 \
           -e LRET_PRECISION=double \
           lret:latest
```

Available variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LRET_THREADS` | # of CPU cores | Number of threads |
| `LRET_MEMORY_LIMIT` | System memory | Maximum memory usage |
| `LRET_NOISE_LEVEL` | 0.0 | Default noise level |
| `LRET_PRECISION` | double | float or double |
| `LRET_SIMD` | auto | SIMD level (SSE, AVX2) |
| `LRET_GPU` | auto | Enable GPU (0/1/auto) |

## Resource Limits

### CPU Limits

```bash
docker run --cpus="4.0" lret:latest
```

### Memory Limits

```bash
docker run --memory="8g" --memory-swap="8g" lret:latest
```

### Combined Limits

```bash
docker run \
    --cpus="4.0" \
    --memory="8g" \
    --memory-swap="8g" \
    lret:latest lret --qubits 15
```

## GPU Support

### Prerequisites

Install nvidia-docker:
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Run with GPU

```bash
docker run --gpus all lret:gpu lret --qubits 20 --use-gpu
```

### Specific GPU

```bash
docker run --gpus '"device=0,2"' lret:gpu lret --qubits 20 --use-gpu
```

## Networking

### Expose Ports

For REST API or monitoring:

```bash
docker run -p 8080:8080 lret:latest lret-server --port 8080
```

### Connect Containers

Create network:
```bash
docker network create lret-net
```

Run containers:
```bash
docker run --network lret-net --name lret-master lret:latest
docker run --network lret-net --name lret-worker lret:latest
```

## Persistent Data

### Named Volumes

```bash
# Create volume
docker volume create lret-data

# Use volume
docker run -v lret-data:/data lret:latest

# List volumes
docker volume ls

# Remove volume
docker volume rm lret-data
```

### Backup and Restore

Backup:
```bash
docker run -v lret-data:/data -v $(pwd):/backup ubuntu \
    tar czf /backup/lret-data-backup.tar.gz -C /data .
```

Restore:
```bash
docker run -v lret-data:/data -v $(pwd):/backup ubuntu \
    tar xzf /backup/lret-data-backup.tar.gz -C /data
```

## Image Management

### Build Multi-Platform Images

```bash
docker buildx create --use
docker buildx build --platform linux/amd64,linux/arm64 -t lret:multiarch .
```

### Push to Registry

```bash
# Tag image
docker tag lret:latest myregistry.com/lret:v1.0

# Push
docker push myregistry.com/lret:v1.0
```

### Clean Up

```bash
# Remove unused images
docker image prune

# Remove all LRET images
docker rmi $(docker images 'lret' -q)

# System-wide cleanup
docker system prune -a
```

## Performance Tuning

### Optimize for Speed

```bash
docker run \
    --cpus="8.0" \
    --memory="16g" \
    --shm-size="8g" \
    -e LRET_THREADS=8 \
    -e LRET_SIMD=AVX2 \
    lret:latest lret --qubits 20 --optimize-speed
```

### Optimize for Memory

```bash
docker run \
    --memory="4g" \
    -e LRET_SPARSE_THRESHOLD=0.1 \
    -e LRET_CHECKPOINT_FREQ=100 \
    lret:latest lret --qubits 25 --optimize-memory
```

## Debugging

### Enable Debug Mode

```bash
docker run -e LRET_DEBUG=1 -e LRET_LOG_LEVEL=DEBUG lret:dev
```

### Attach Debugger

```bash
# Run with security privileges
docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -it lret:dev /bin/bash

# Inside container
gdb --args lret --qubits 10 ...
```

### Inspect Container

```bash
# Running container
docker exec -it <container-id> /bin/bash

# Inspect logs
docker logs <container-id>

# Resource usage
docker stats <container-id>
```

## Health Checks

Add health check to Dockerfile:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD lret --version || exit 1
```

Check health:
```bash
docker ps
docker inspect --format='{{.State.Health.Status}}' <container-id>
```

## Security

### Run as Non-Root

```dockerfile
RUN useradd -m -u 1000 lret
USER lret
```

### Read-Only Filesystem

```bash
docker run --read-only --tmpfs /tmp lret:latest
```

### Limit Capabilities

```bash
docker run --cap-drop=ALL --cap-add=NET_BIND_SERVICE lret:latest
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs <container-id>

# Inspect configuration
docker inspect <container-id>

# Run with bash to debug
docker run -it --entrypoint /bin/bash lret:latest
```

### Out of Memory

```bash
# Increase memory limit
docker run --memory="16g" --memory-swap="16g" lret:latest

# Use sparse matrices
docker run -e LRET_SPARSE=1 lret:latest
```

### Performance Issues

```bash
# Check resource usage
docker stats

# Profile simulation
docker run lret:latest lret --qubits 15 --profile --output /dev/null
```

## Best Practices

1. **Use specific tags**: `lret:v1.0` instead of `lret:latest`
2. **Set resource limits**: Prevent runaway containers
3. **Use volumes for data**: Don't store data in containers
4. **Multi-stage builds**: Keep images small
5. **Health checks**: Monitor container health
6. **Security**: Run as non-root user
7. **Logs**: Stream logs to external service
8. **Updates**: Regularly update base images

## Examples

### Run VQE Simulation

```bash
docker run -v $(pwd)/vqe.py:/workspace/vqe.py lret:latest \
    python3 /workspace/vqe.py
```

### Batch Processing

```bash
for file in circuits/*.json; do
    docker run -v $(pwd):/data lret:latest \
        lret --input /data/$file --output /data/results/$(basename $file .json).csv
done
```

### Continuous Integration

```yaml
# .github/workflows/test.yml
- name: Run LRET tests
  run: |
    docker run lret:latest lret --test
```

## Further Reading

- [Cloud Deployment Guide](cloud-deployment.md)
- [HPC Deployment Guide](hpc-deployment.md)
- [C++ API Reference](../api-reference/cpp/)
- [CLI Reference](../api-reference/cli/)
