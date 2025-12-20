cmake_minimum_required(VERSION 3.16)
project(QuantumLRET-Sim VERSION 1.0.0 LANGUAGES CXX)

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

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Source files
set(LIB_SOURCES
    src/gates_and_noise.cpp
    src/simulator.cpp
    src/utils.cpp
    src/cli_parser.cpp
    src/output_formatter.cpp
    src/parallel_modes.cpp
    src/fdm_simulator.cpp
)

# Create static library
add_library(qlret_lib STATIC ${LIB_SOURCES})
target_link_libraries(qlret_lib PUBLIC Eigen3::Eigen)

if(OpenMP_CXX_FOUND)
    target_link_libraries(qlret_lib PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(qlret_lib PUBLIC USE_OPENMP)
endif()

# Main benchmark executable
add_executable(quantum_sim main.cpp)
target_link_libraries(quantum_sim PRIVATE qlret_lib)

# Batch demo executable
add_executable(demo_batch tests/demo_batch.cpp)
target_link_libraries(demo_batch PRIVATE qlret_lib)

# Compiler optimizations for Release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(qlret_lib PRIVATE -O3 -march=native)
        target_compile_options(quantum_sim PRIVATE -O3 -march=native)
        target_compile_options(demo_batch PRIVATE -O3 -march=native)
    elseif(MSVC)
        target_compile_options(qlret_lib PRIVATE /O2 /arch:AVX2)
        target_compile_options(quantum_sim PRIVATE /O2 /arch:AVX2)
        target_compile_options(demo_batch PRIVATE /O2 /arch:AVX2)
    endif()
endif()

# Installation rules
install(TARGETS quantum_sim demo_batch qlret_lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

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
message(STATUS "========================================")
