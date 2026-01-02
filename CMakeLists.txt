cmake_minimum_required(VERSION 3.16)

# GPU Support Option (must be set before project())
option(USE_GPU "Enable NVIDIA GPU acceleration via cuQuantum" OFF)

if(USE_GPU)
    project(QuantumLRET-Sim VERSION 1.0.0 LANGUAGES CXX CUDA)
else()
    project(QuantumLRET-Sim VERSION 1.0.0 LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find Eigen3 - try find_package first, fall back to manual path
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    # Manual Eigen path for systems without CMake config
    set(EIGEN3_INCLUDE_DIR "$ENV{USERPROFILE}/eigen-3.4.0" CACHE PATH "Eigen3 include directory")
    if(EXISTS "${EIGEN3_INCLUDE_DIR}/Eigen/Dense")
        message(STATUS "Found Eigen3 at: ${EIGEN3_INCLUDE_DIR}")
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen3::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
        )
    else()
        message(FATAL_ERROR "Eigen3 not found. Set EIGEN3_INCLUDE_DIR to your Eigen installation.")
    endif()
endif()

find_package(OpenMP)

#==============================================================================
# MPI Configuration (Phase 3)
#==============================================================================
option(USE_MPI "Enable MPI distributed parallelization" OFF)

if(USE_MPI)
    find_package(MPI REQUIRED)
    
    if(MPI_CXX_FOUND)
        message(STATUS "Found MPI: ${MPI_CXX_COMPILER}")
        message(STATUS "  MPI Include: ${MPI_CXX_INCLUDE_DIRS}")
        message(STATUS "  MPI Libraries: ${MPI_CXX_LIBRARIES}")
    endif()
endif()

#==============================================================================
# GPU/CUDA Configuration (Phase 2)
#==============================================================================
if(USE_GPU)
    # CUDA settings
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    # Find CUDA Toolkit
    find_package(CUDAToolkit REQUIRED)
    
    # Find cuQuantum (NVIDIA's quantum simulation library)
    # cuQuantum provides custatevec and cutensornet
    find_path(CUQUANTUM_INCLUDE_DIR
        NAMES custatevec.h
        HINTS
            ${CUQUANTUM_ROOT}/include
            $ENV{CUQUANTUM_ROOT}/include
            /usr/local/cuda/include
            /opt/nvidia/cuquantum/include
    )
    
    find_library(CUSTATEVEC_LIBRARY
        NAMES custatevec
        HINTS
            ${CUQUANTUM_ROOT}/lib
            ${CUQUANTUM_ROOT}/lib64
            $ENV{CUQUANTUM_ROOT}/lib
            $ENV{CUQUANTUM_ROOT}/lib64
            /usr/local/cuda/lib64
            /opt/nvidia/cuquantum/lib64
    )
    
    if(CUQUANTUM_INCLUDE_DIR AND CUSTATEVEC_LIBRARY)
        message(STATUS "Found cuQuantum: ${CUSTATEVEC_LIBRARY}")
        set(CUQUANTUM_FOUND TRUE)
    else()
        message(WARNING "cuQuantum not found. GPU simulation will use custom CUDA kernels.")
        set(CUQUANTUM_FOUND FALSE)
    endif()
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Source files
set(LIB_SOURCES
    src/gates_and_noise.cpp
    src/gate_fusion.cpp
    src/circuit_optimizer.cpp
    src/simd_kernels.cpp
    src/simulator.cpp
    src/utils.cpp
    src/cli_parser.cpp
    src/output_formatter.cpp
    src/parallel_modes.cpp
    src/fdm_simulator.cpp
    src/resource_monitor.cpp
    src/progressive_csv.cpp
    src/structured_csv.cpp
    src/benchmark_types.cpp
    src/benchmark_runner.cpp
)

# Add MPI source if enabled
if(USE_MPI)
    list(APPEND LIB_SOURCES src/mpi_parallel.cpp)
endif()

# Create static library
add_library(qlret_lib STATIC ${LIB_SOURCES})
target_link_libraries(qlret_lib PUBLIC Eigen3::Eigen)

if(OpenMP_CXX_FOUND)
    target_link_libraries(qlret_lib PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(qlret_lib PUBLIC USE_OPENMP)
endif()

#==============================================================================
# MPI Library Linking (Phase 3)
#==============================================================================
if(USE_MPI AND MPI_CXX_FOUND)
    target_include_directories(qlret_lib PUBLIC ${MPI_CXX_INCLUDE_DIRS})
    target_link_libraries(qlret_lib PUBLIC ${MPI_CXX_LIBRARIES})
    target_compile_definitions(qlret_lib PUBLIC USE_MPI)
    
    # Add MPI compile flags
    if(MPI_CXX_COMPILE_FLAGS)
        target_compile_options(qlret_lib PUBLIC ${MPI_CXX_COMPILE_FLAGS})
    endif()
endif()

#==============================================================================
# GPU Library (Phase 2)
#==============================================================================
if(USE_GPU)
    # GPU simulator CUDA source
    set(GPU_SOURCES
        src/gpu_simulator.cu
    )
    
    # Create GPU library
    add_library(qlret_gpu STATIC ${GPU_SOURCES})
    
    # CUDA compile options
    set_target_properties(qlret_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "70;75;80;86;89;90"  # Volta, Turing, Ampere, Ada, Hopper
    )
    
    target_include_directories(qlret_gpu PUBLIC 
        ${CMAKE_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    
    target_link_libraries(qlret_gpu PUBLIC 
        Eigen3::Eigen
        CUDA::cudart
        CUDA::cublas
    )
    
    # Link cuQuantum if available
    if(CUQUANTUM_FOUND)
        target_include_directories(qlret_gpu PUBLIC ${CUQUANTUM_INCLUDE_DIR})
        target_link_libraries(qlret_gpu PUBLIC ${CUSTATEVEC_LIBRARY})
        target_compile_definitions(qlret_gpu PUBLIC USE_CUQUANTUM)
    endif()
    
    target_compile_definitions(qlret_gpu PUBLIC USE_GPU)
    
    # Link GPU lib to main lib
    target_link_libraries(qlret_lib PUBLIC qlret_gpu)
    target_compile_definitions(qlret_lib PUBLIC USE_GPU)
    
    if(CUQUANTUM_FOUND)
        target_compile_definitions(qlret_lib PUBLIC USE_CUQUANTUM)
    endif()
endif()

# Main benchmark executable
add_executable(quantum_sim main.cpp)
target_link_libraries(quantum_sim PRIVATE qlret_lib)

# Batch demo executable (only if tests directory exists)
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/demo_batch.cpp")
    add_executable(demo_batch tests/demo_batch.cpp)
    target_link_libraries(demo_batch PRIVATE qlret_lib)
    set(BUILD_DEMO_BATCH TRUE)
else()
    set(BUILD_DEMO_BATCH FALSE)
endif()

# Fidelity debug test
if(EXISTS "${CMAKE_SOURCE_DIR}/test_fidelity.cpp")
    add_executable(test_fidelity test_fidelity.cpp)
    target_link_libraries(test_fidelity PRIVATE qlret_lib)
endif()

# Minimal debug test
if(EXISTS "${CMAKE_SOURCE_DIR}/test_minimal.cpp")
    add_executable(test_minimal test_minimal.cpp)
    target_link_libraries(test_minimal PRIVATE qlret_lib)
endif()

# Compiler optimizations for Release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(qlret_lib PRIVATE -O3 -march=native)
        target_compile_options(quantum_sim PRIVATE -O3 -march=native)
        if(BUILD_DEMO_BATCH)
            target_compile_options(demo_batch PRIVATE -O3 -march=native)
        endif()
    elseif(MSVC)
        target_compile_options(qlret_lib PRIVATE /O2 /arch:AVX2)
        target_compile_options(quantum_sim PRIVATE /O2 /arch:AVX2)
        if(BUILD_DEMO_BATCH)
            target_compile_options(demo_batch PRIVATE /O2 /arch:AVX2)
        endif()
    endif()
endif()

# Installation rules
install(TARGETS quantum_sim qlret_lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
if(BUILD_DEMO_BATCH)
    install(TARGETS demo_batch RUNTIME DESTINATION bin)
endif()

install(DIRECTORY include/ DESTINATION include)

# Print configuration summary
message(STATUS "========================================")
message(STATUS "QuantumLRET-Sim Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Eigen3: ${EIGEN3_VERSION_STRING}")
message(STATUS "  OpenMP: ${OpenMP_CXX_FOUND}")
if(OpenMP_CXX_FOUND)
    message(STATUS "  OpenMP Version: ${OpenMP_CXX_VERSION}")
endif()
message(STATUS "  MPI Support: ${USE_MPI}")
if(USE_MPI AND MPI_CXX_FOUND)
    message(STATUS "  MPI Compiler: ${MPI_CXX_COMPILER}")
endif()
message(STATUS "  GPU Support: ${USE_GPU}")
if(USE_GPU)
    message(STATUS "  CUDA Toolkit: ${CUDAToolkit_VERSION}")
    message(STATUS "  cuQuantum: ${CUQUANTUM_FOUND}")
endif()
message(STATUS "========================================")
