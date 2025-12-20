# QuantumLRET-Sim Dockerfile
# Supports CLI arguments for simulation parameters

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libeigen3-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source code
COPY . /app

# Build the project
RUN mkdir -p build && cd build && cmake .. && make -j$(nproc)

# Use ENTRYPOINT so arguments can be passed via docker run
ENTRYPOINT ["./build/quantum_sim"]

# Default arguments (can be overridden)
CMD ["-n", "8", "-d", "13", "--mode", "auto"]
