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

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libeigen3-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy CMakeLists first for better layer caching
COPY CMakeLists.txt .

# Copy headers and sources separately for caching
COPY include/ include/
COPY src/ src/
COPY main.cpp .

# Build the project with OpenMP support
RUN mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

# ---- Runtime stage (smaller final image) ----
FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libomp5 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the built binary from builder stage
COPY --from=builder /app/build/quantum_sim ./quantum_sim

# Create output directory for CSV files
RUN mkdir -p /app/output

# Use ENTRYPOINT so arguments can be passed via docker run
ENTRYPOINT ["./quantum_sim"]

# Default arguments (can be overridden)
# Runs compare mode with FDM and auto-generated CSV filename
CMD ["-n", "8", "-d", "13", "--mode", "compare", "--fdm"]
