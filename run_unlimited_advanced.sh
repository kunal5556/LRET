#!/bin/bash
# run_unlimited_advanced.sh - Advanced version with more features
#
# Features:
#   - Automatic resource detection
#   - Progress monitoring
#   - Multiple output directory support
#   - Image auto-pull
#   - Resource limit warnings

set -e

# ============================================================================
# Configuration
# ============================================================================

IMAGE="${DOCKER_IMAGE:-lreto9}"  # Set via env or use default
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)}"
CONTAINER_NAME="quantum-lret-unlimited-$$"  # Unique name per process
AUTO_PULL="${AUTO_PULL:-false}"
SHOW_RESOURCES="${SHOW_RESOURCES:-true}"

# ============================================================================
# Helper Functions
# ============================================================================

print_banner() {
    echo "=============================================================="
    echo "$1"
    echo "=============================================================="
}

print_section() {
    echo ""
    echo ">> $1"
    echo ""
}

get_cpu_count() {
    nproc 2>/dev/null || echo "1"
}

get_memory_gb() {
    free -h 2>/dev/null | awk '/^Mem:/ {print $2}' || echo "N/A"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        echo "Error: Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker ps &> /dev/null; then
        echo "Error: Docker daemon is not running or permission denied"
        echo "Try: sudo systemctl start docker"
        exit 1
    fi
}

check_image() {
    if ! docker image inspect "$IMAGE" &> /dev/null; then
        echo "Warning: Image '$IMAGE' not found locally."
        
        if [ "$AUTO_PULL" = "true" ]; then
            echo "Attempting to pull..."
            docker pull "$IMAGE" || {
                echo "Error: Failed to pull image '$IMAGE'"
                exit 1
            }
        else
            echo "Run with AUTO_PULL=true to auto-pull or pull manually:"
            echo "  docker pull $IMAGE"
            exit 1
        fi
    fi
}

show_help() {
    cat << 'EOF'
LRET Simulator - Docker with Unlimited Resources

USAGE:
    ./run_unlimited_advanced.sh [OPTIONS] -- [SIMULATOR_ARGS]

ENVIRONMENT VARIABLES:
    DOCKER_IMAGE      Docker image name (default: lreto9)
    OUTPUT_DIR        Output directory to mount (default: current dir)
    AUTO_PULL         Auto-pull image if missing (default: false)
    SHOW_RESOURCES    Show resource summary (default: true)

EXAMPLES:
    # Basic run with all benchmarks
    ./run_unlimited_advanced.sh -- --benchmark-all -o /output/results.csv

    # Custom configuration
    ./run_unlimited_advanced.sh -- -n 20 -d 100 --fdm -o /output/test.csv

    # Epsilon sweep
    ./run_unlimited_advanced.sh -- --sweep-epsilon "1e-7:1e-2:10" -o /output/sweep.csv

    # With custom image and output directory
    DOCKER_IMAGE=myrepo/lret:v2 OUTPUT_DIR=/data/results \
        ./run_unlimited_advanced.sh -- --benchmark-all -o /output/bench.csv

    # Auto-pull if image missing
    AUTO_PULL=true ./run_unlimited_advanced.sh -- -n 15 -o /output/run.csv

RESOURCE FLAGS (automatically applied):
    --memory unlimited      No RAM limit (uses all TB)
    --memory-swap -1        No swap limit
    --cpus $(nproc)         All CPU cores
    --oom-kill-disable      Don't kill on OOM
    --ulimit memlock=-1     Unlimited locked memory

NOTES:
    - Works best on Linux (native Docker)
    - Windows/Mac: Limited by Docker Desktop VM settings
    - Output directory $OUTPUT_DIR is mounted as /output in container
    - All simulator arguments are passed directly to the container

EOF
    exit 0
}

# ============================================================================
# Main Script
# ============================================================================

# Parse script arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use: $0 [OPTIONS] -- [SIMULATOR_ARGS]"
            echo "Try: $0 --help"
            exit 1
            ;;
    esac
    shift
done

# Check prerequisites
check_docker
check_image

# Show configuration
if [ "$SHOW_RESOURCES" = "true" ]; then
    print_banner "LRET Simulator - Unlimited Resources Mode"
    
    echo ""
    echo "Configuration:"
    echo "  Docker Image:    $IMAGE"
    echo "  Output Dir:      $OUTPUT_DIR → /output"
    echo "  Container Name:  $CONTAINER_NAME"
    echo ""
    
    echo "System Resources:"
    echo "  CPUs:            $(get_cpu_count) cores (all available to container)"
    echo "  RAM:             $(get_memory_gb) (unlimited access)"
    echo "  Disk Space:      $(df -h "$OUTPUT_DIR" | awk 'NR==2 {print $4}') free"
    echo ""
    
    echo "Simulator Arguments:"
    echo "  $@"
    echo ""
    
    print_section "Starting simulation..."
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run container with unlimited resources
docker run --rm \
  --name "$CONTAINER_NAME" \
  --memory unlimited \
  --memory-swap -1 \
  --memory-swappiness 0 \
  --cpus "$(get_cpu_count)" \
  --cpu-shares 1024 \
  --oom-kill-disable \
  --ulimit memlock=-1:-1 \
  --ulimit nofile=1048576:1048576 \
  --ulimit nproc=-1:-1 \
  --ulimit core=-1:-1 \
  --ulimit stack=-1:-1 \
  -v "$OUTPUT_DIR:/output" \
  -e OMP_NUM_THREADS=$(get_cpu_count) \
  -e OMP_PROC_BIND=true \
  -e OMP_PLACES=cores \
  "$IMAGE" "$@"

EXIT_CODE=$?

# Show results
if [ "$SHOW_RESOURCES" = "true" ]; then
    echo ""
    if [ $EXIT_CODE -eq 0 ]; then
        print_banner "✓ Simulation Completed Successfully"
        
        # Show output files
        if ls "$OUTPUT_DIR"/*.csv 1> /dev/null 2>&1; then
            echo ""
            echo "Output files created:"
            ls -lh "$OUTPUT_DIR"/*.csv | tail -10 | awk '{print "  📄 " $9 " (" $5 ")"}'
        fi
    else
        print_banner "✗ Simulation Failed (Exit Code: $EXIT_CODE)"
    fi
    echo ""
fi

exit $EXIT_CODE
