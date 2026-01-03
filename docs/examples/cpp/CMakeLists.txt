cmake_minimum_required(VERSION 3.15)
project(LRET_Examples CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find LRET library
find_path(LRET_INCLUDE_DIR simulator.h
    PATHS ${CMAKE_SOURCE_DIR}/../../../include
    DOC "LRET include directory")

find_library(LRET_LIBRARY
    NAMES lret
    PATHS ${CMAKE_SOURCE_DIR}/../../../build
    DOC "LRET library")

if(NOT LRET_INCLUDE_DIR OR NOT LRET_LIBRARY)
    message(STATUS "LRET not found, using local paths")
    set(LRET_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/../../../include)
    set(LRET_LIBRARY ${CMAKE_SOURCE_DIR}/../../../build/liblret.a)
endif()

include_directories(${LRET_INCLUDE_DIR})

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -march=native -Wall -Wextra)
endif()

# Example executables
add_executable(01_basic_simulation 01_basic_simulation.cpp)
target_link_libraries(01_basic_simulation ${LRET_LIBRARY})

add_executable(02_custom_gates 02_custom_gates.cpp)
target_link_libraries(02_custom_gates ${LRET_LIBRARY})

add_executable(03_noise_models 03_noise_models.cpp)
target_link_libraries(03_noise_models ${LRET_LIBRARY})

add_executable(04_parallel_simulation 04_parallel_simulation.cpp)
target_link_libraries(04_parallel_simulation ${LRET_LIBRARY})
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(04_parallel_simulation OpenMP::OpenMP_CXX)
endif()

add_executable(05_benchmarking 05_benchmarking.cpp)
target_link_libraries(05_benchmarking ${LRET_LIBRARY})

# Installation
install(TARGETS 
    01_basic_simulation
    02_custom_gates
    03_noise_models
    04_parallel_simulation
    05_benchmarking
    DESTINATION bin/examples)
