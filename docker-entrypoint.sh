#!/bin/bash
# Docker entrypoint for development mode
# Builds if needed, then runs with provided arguments

cd /app/build

# Build if not built or source changed
if [ ! -f quantum_sim ] || [ /app/main.cpp -nt quantum_sim ]; then
    echo "Building project..."
    cmake .. && make -j$(nproc)
fi

# Run with all provided arguments
./quantum_sim "$@"
