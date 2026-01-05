# Minimum CMake version
cmake_minimum_required(VERSION 3.16)

# Enable position independent code so static libs can link into Python module (_qlret_native)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
# nlohmann/json for Phase 4.1 (Noise Model Import)
#==============================================================================
include(FetchContent)

FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)

FetchContent_MakeAvailable(json)

#==============================================================================
# Python Bindings (Phase 5)
#==============================================================================
option(USE_PYTHON "Enable Python bindings via pybind11" OFF)

if(USE_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
    
    message(STATUS "Python bindings enabled")
    message(STATUS "  Python: ${Python3_EXECUTABLE}")
    message(STATUS "  Python Include: ${Python3_INCLUDE_DIRS}")
endif()

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
# GPU/CUDA Configuration (Phase 2) + NCCL (Phase 8.1)
#==============================================================================
if(USE_GPU)
    option(USE_NCCL "Enable NCCL for multi-GPU collectives" OFF)

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

    if(USE_NCCL)
        find_path(NCCL_INCLUDE_DIR
            NAMES nccl.h
            HINTS
                ${NCCL_ROOT}/include
                $ENV{NCCL_ROOT}/include
                /usr/local/cuda/include
                /usr/include
        )
        
        find_library(NCCL_LIBRARY
            NAMES nccl
            HINTS
                ${NCCL_ROOT}/lib
                ${NCCL_ROOT}/lib64
                $ENV{NCCL_ROOT}/lib
                $ENV{NCCL_ROOT}/lib64
                /usr/local/cuda/lib64
                /usr/lib/x86_64-linux-gnu
        )

        if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
            message(STATUS "Found NCCL: ${NCCL_LIBRARY}")
            set(NCCL_FOUND TRUE)
        else()
            message(WARNING "NCCL requested but not found. Multi-GPU collectives will be disabled.")
            set(NCCL_FOUND FALSE)
        endif()
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
    src/autodiff.cpp
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
    src/mpi_parallel.cpp
    src/noise_import.cpp
    src/json_interface.cpp
    # Phase 8: Fault tolerance
    src/checkpoint.cpp
    src/distributed_perf.cpp
    src/distributed_autodiff.cpp
    src/fault_tolerance.cpp
    # Phase 9: QEC
    src/qec_types.cpp
    src/qec_stabilizer.cpp
    src/qec_syndrome.cpp
    src/qec_decoder.cpp
    src/qec_logical.cpp
    # Phase 9.2: Distributed QEC
    src/qec_distributed.cpp
)

# Create static library
add_library(qlret_lib STATIC ${LIB_SOURCES})
target_link_libraries(qlret_lib PUBLIC Eigen3::Eigen nlohmann_json::nlohmann_json)

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
        src/distributed_gpu.cu
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

    if(USE_NCCL AND NCCL_FOUND)
        target_include_directories(qlret_gpu PUBLIC ${NCCL_INCLUDE_DIR})
        target_link_libraries(qlret_gpu PUBLIC ${NCCL_LIBRARY})
        target_compile_definitions(qlret_gpu PUBLIC USE_NCCL)
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

#==============================================================================
# Python Module (Phase 5 - pybind11)
#==============================================================================
if(USE_PYTHON)
    pybind11_add_module(_qlret_native src/python_bindings.cpp)
    target_link_libraries(_qlret_native PRIVATE qlret_lib)
    target_compile_definitions(_qlret_native PRIVATE USE_PYTHON)
    
    # Install to python/qlret directory for development
    set_target_properties(_qlret_native PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/qlret"
    )
    
    message(STATUS "Python module will be built: _qlret_native")
endif()

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

# Noise import test (Phase 4.1)
if(EXISTS "${CMAKE_SOURCE_DIR}/test_noise_import.cpp")
    add_executable(test_noise_import test_noise_import.cpp)
    target_link_libraries(test_noise_import PRIVATE qlret_lib)
endif()

# Distributed GPU smoke test (Phase 8.1)
if(USE_GPU AND EXISTS "${CMAKE_SOURCE_DIR}/tests/test_distributed_gpu.cpp")
    add_executable(test_distributed_gpu tests/test_distributed_gpu.cpp)
    target_link_libraries(test_distributed_gpu PRIVATE qlret_lib)
endif()

option(BUILD_MULTI_GPU_TESTS "Build multi-GPU MPI/NCCL tests" OFF)
if(BUILD_MULTI_GPU_TESTS AND USE_GPU AND USE_MPI)
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_distributed_gpu_mpi.cpp")
        add_executable(test_distributed_gpu_mpi tests/test_distributed_gpu_mpi.cpp)
        target_link_libraries(test_distributed_gpu_mpi PRIVATE qlret_lib)
        if(NCCL_FOUND)
            target_compile_definitions(test_distributed_gpu_mpi PRIVATE USE_NCCL)
        endif()
    endif()

    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_multi_gpu_sync.cpp")
        add_executable(test_multi_gpu_sync tests/test_multi_gpu_sync.cpp)
        target_link_libraries(test_multi_gpu_sync PRIVATE qlret_lib)
        if(NCCL_FOUND)
            target_compile_definitions(test_multi_gpu_sync PRIVATE USE_NCCL)
        endif()
    endif()

    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_autodiff_multi_gpu.cpp")
        add_executable(test_autodiff_multi_gpu tests/test_autodiff_multi_gpu.cpp)
        target_link_libraries(test_autodiff_multi_gpu PRIVATE qlret_lib)
        if(NCCL_FOUND)
            target_compile_definitions(test_autodiff_multi_gpu PRIVATE USE_NCCL)
        endif()
    endif()

    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_multi_gpu_collectives.cpp")
        add_executable(test_multi_gpu_collectives tests/test_multi_gpu_collectives.cpp)
        target_link_libraries(test_multi_gpu_collectives PRIVATE qlret_lib)
        if(NCCL_FOUND)
            target_compile_definitions(test_multi_gpu_collectives PRIVATE USE_NCCL)
        endif()
    endif()

    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_multi_gpu_load_balance.cpp")
        add_executable(test_multi_gpu_load_balance tests/test_multi_gpu_load_balance.cpp)
        target_link_libraries(test_multi_gpu_load_balance PRIVATE qlret_lib)
        if(NCCL_FOUND)
            target_compile_definitions(test_multi_gpu_load_balance PRIVATE USE_NCCL)
        endif()
    endif()

    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_fault_tolerance.cpp")
        add_executable(test_fault_tolerance tests/test_fault_tolerance.cpp src/fault_tolerance.cpp src/checkpoint.cpp)
        target_link_libraries(test_fault_tolerance PRIVATE qlret_lib)
        if(NCCL_FOUND)
            target_compile_definitions(test_fault_tolerance PRIVATE USE_NCCL)
        endif()
    endif()
endif()

# Advanced noise test (Phase 4.3)
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_advanced_noise.cpp")
    add_executable(test_advanced_noise tests/test_advanced_noise.cpp)
    target_link_libraries(test_advanced_noise PRIVATE qlret_lib)
endif()

# Autodiff gradient test (Phase 8.3)
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_autodiff.cpp")
    add_executable(test_autodiff tests/test_autodiff.cpp)
    target_link_libraries(test_autodiff PRIVATE qlret_lib)
endif()

# Autodiff multi-parameter / multi-qubit test (Phase 8.3)
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_autodiff_multi.cpp")
    add_executable(test_autodiff_multi tests/test_autodiff_multi.cpp)
    target_link_libraries(test_autodiff_multi PRIVATE qlret_lib)
endif()

# Leakage and measurement test (Phase 4.4/4.5)
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_leakage_measurement.cpp")
    add_executable(test_leakage_measurement tests/test_leakage_measurement.cpp)
    target_link_libraries(test_leakage_measurement PRIVATE qlret_lib)
endif()

# Checkpoint test (Phase 8.3+)
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_checkpoint.cpp")
    add_executable(test_checkpoint tests/test_checkpoint.cpp src/checkpoint.cpp)
    target_link_libraries(test_checkpoint PRIVATE qlret_lib)
endif()

# Scheduler test (Phase 8.3+)
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_scheduler.cpp")
    add_executable(test_scheduler tests/test_scheduler.cpp)
    target_link_libraries(test_scheduler PRIVATE qlret_lib)
endif()

#==============================================================================
# Phase 9: Quantum Error Correction Tests
#==============================================================================
# QEC Stabilizer test (Phase 9.1)
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_qec_stabilizer.cpp")
    add_executable(test_qec_stabilizer tests/test_qec_stabilizer.cpp)
    target_link_libraries(test_qec_stabilizer PRIVATE qlret_lib)
endif()

# QEC Syndrome test (Phase 9.1)
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_qec_syndrome.cpp")
    add_executable(test_qec_syndrome tests/test_qec_syndrome.cpp)
    target_link_libraries(test_qec_syndrome PRIVATE qlret_lib)
endif()

# QEC Decoder test (Phase 9.1)
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_qec_decoder.cpp")
    add_executable(test_qec_decoder tests/test_qec_decoder.cpp)
    target_link_libraries(test_qec_decoder PRIVATE qlret_lib)
endif()

# QEC Logical Qubit test (Phase 9.1)
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_qec_logical.cpp")
    add_executable(test_qec_logical tests/test_qec_logical.cpp)
    target_link_libraries(test_qec_logical PRIVATE qlret_lib)
endif()

# QEC Distributed test (Phase 9.2)
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_qec_distributed.cpp")
    add_executable(test_qec_distributed tests/test_qec_distributed.cpp)
    target_link_libraries(test_qec_distributed PRIVATE qlret_lib)
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
