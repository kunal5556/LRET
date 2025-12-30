#!/bin/bash
# run_unlimited.sh - Run LRET with full system resources (TB RAM, all cores)
# 
# Usage:
#   ./run_unlimited.sh -n 10 -d 15 --fdm -o /output/results.csv
#   ./run_unlimited.sh --benchmark-all -o /output/benchmark.csv
#   ./run_unlimited.sh -n 20 -d 100 --sweep-qubits "15:25:1" -o /output/sweep.csv

set -e  # Exit on error

# Configuration
IMAGE="lreto9"  # Change to your image name (e.g., "username/quantum-lret:latest")
OUTPUT_DIR="$(pwd)"  # Current directory will be mounted as /output
CONTAINER_NAME="quantum-lret-unlimited"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo -e "${YELLOW}Warning: Unlimited resources work best on Linux.${NC}"
    echo "On Windows/Mac Docker Desktop, resources are limited by VM settings."
    echo ""
fi

# Display system resources
echo -e "${BLUE}===========================================================${NC}"
echo -e "${BLUE}           LRET Simulator - Unlimited Resources${NC}"
echo -e "${BLUE}===========================================================${NC}"
echo ""
echo -e "${GREEN}System Resources Available:${NC}"
echo "  CPUs: $(nproc) cores"
echo "  RAM: $(free -h 2>/dev/null | awk '/^Mem:/ {print $2}' || echo 'N/A')"
echo "  Swap: $(free -h 2>/dev/null | awk '/^Swap:/ {print $2}' || echo 'N/A')"
echo "  Disk: $(df -h "$OUTPUT_DIR" | awk 'NR==2 {print $4}') free"
echo ""

# Check if image exists
if ! docker image inspect "$IMAGE" &> /dev/null; then
    echo -e "${YELLOW}Image '$IMAGE' not found locally.${NC}"
    echo "Attempting to pull from registry..."
    if ! docker pull "$IMAGE" 2>/dev/null; then
        echo -e "${YELLOW}Warning: Could not pull image. Using local image if available.${NC}"
    fi
fi

# Parse arguments to show what will run
echo -e "${GREEN}Configuration:${NC}"
echo "  Docker image: $IMAGE"
echo "  Output mounted: $OUTPUT_DIR → /output (in container)"
echo "  Arguments: $@"
echo ""

# Run with unlimited resources
echo -e "${BLUE}===========================================================${NC}"
echo -e "${BLUE}  Running with UNLIMITED resources (no memory/CPU limits)${NC}"
echo -e "${BLUE}===========================================================${NC}"
echo ""

docker run --rm \
  --name "$CONTAINER_NAME" \
  --memory unlimited \
  --memory-swap -1 \
  --memory-swappiness 0 \
  --cpus "$(nproc)" \
  --cpu-shares 1024 \
  --oom-kill-disable \
  --ulimit memlock=-1:-1 \
  --ulimit nofile=1048576:1048576 \
  --ulimit nproc=-1:-1 \
  --ulimit core=-1:-1 \
  --ulimit stack=-1:-1 \
  -v "$OUTPUT_DIR:/output" \
  -e OMP_NUM_THREADS=$(nproc) \
  -e OMP_PROC_BIND=true \
  -e OMP_PLACES=cores \
  "$IMAGE" "$@"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}===========================================================${NC}"
    echo -e "${GREEN}  Completed successfully!${NC}"
    echo -e "${GREEN}===========================================================${NC}"
    echo ""
    echo "Output files in: $OUTPUT_DIR"
    echo ""
    # List CSV files if any were created
    if ls "$OUTPUT_DIR"/*.csv 1> /dev/null 2>&1; then
        echo "CSV files created:"
        ls -lh "$OUTPUT_DIR"/*.csv | awk '{print "  " $9 " (" $5 ")"}'
    fi
else
    echo -e "${YELLOW}===========================================================${NC}"
    echo -e "${YELLOW}  Exited with code: $EXIT_CODE${NC}"
    echo -e "${YELLOW}===========================================================${NC}"
fi

echo ""
