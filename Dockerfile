# QuantumLRET-Sim Dockerfile
# Supports CLI arguments and structured CSV output
#
# Build:
#   docker build -t quantum-lret .
#
# Run examples:
#   docker run quantum-lret -n 10 -d 15 --mode compare --fdm
#   docker run -v $(pwd)/output:/app/output quantum-lret -n 12 -d 20 -o /app/output/results.csv
#
# For CSV persistence, mount a volume to /app/output

FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies (git + ca-certificates for nlohmann/json FetchContent)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ca-certificates \
    libeigen3-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy CMakeLists first for better layer caching
COPY CMakeLists.txt .

# Copy headers and sources separately for caching
COPY include/ include/
COPY src/ src/
COPY scripts/ scripts/
COPY main.cpp .
COPY test_noise_import.cpp .

# Build the project with OpenMP support
RUN mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

# ---- Runtime stage (smaller final image) ----
FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies (including Python for noise model download script)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libomp5 \
    libgomp1 \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binaries from builder stage
COPY --from=builder /app/build/quantum_sim ./quantum_sim
COPY --from=builder /app/build/test_noise_import ./test_noise_import

# Copy Python scripts for noise model management
COPY scripts/ scripts/

# Create output directory for CSV files
RUN mkdir -p /app/output

# Use ENTRYPOINT so arguments can be passed via docker run
ENTRYPOINT ["./quantum_sim"]

# Default arguments (can be overridden)
# Note: No -o flag by default. User must add -o to generate output file.
# Examples:
#   docker run quantum-lret                           # No output file
#   docker run quantum-lret -o                        # Default filename in container
#   docker run quantum-lret -o results.csv            # Custom filename in container
#   docker run -v /host/path:/app/output quantum-lret -o /app/output/results.csv  # Save to host
CMD ["-n", "8", "-d", "13", "--mode", "compare", "--fdm"]
