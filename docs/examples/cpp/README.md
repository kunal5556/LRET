# C++ Examples for LRET

This directory contains C++ examples demonstrating the LRET simulator API.

## Building Examples

### Using CMake

```bash
cd docs/examples/cpp
mkdir build && cd build
cmake ..
cmake --build .
```

### Manual Compilation

```bash
# With installed LRET
g++ -std=c++17 -O3 -o example 01_basic_simulation.cpp -llret

# With local build
g++ -std=c++17 -O3 -I../../../include -o example 01_basic_simulation.cpp -L../../../build -llret
```

## Examples

1. **01_basic_simulation.cpp** - Basic simulator usage and gate operations
2. **02_custom_gates.cpp** - Implementing custom gates and operations
3. **03_noise_models.cpp** - Advanced noise modeling and characterization
4. **04_parallel_simulation.cpp** - Parallelization with OpenMP and MPI
5. **05_benchmarking.cpp** - Performance benchmarking and profiling

## Running Examples

```bash
./build/01_basic_simulation
./build/02_custom_gates
./build/03_noise_models
./build/04_parallel_simulation
./build/05_benchmarking
```

## API Documentation

See [C++ API Reference](../../api-reference/cpp/) for complete documentation.
