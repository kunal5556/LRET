/**
 * @file gpu_simulator.cu
 * @brief GPU-Accelerated Quantum Simulation Implementation (Phase 2)
 * 
 * This file contains:
 * 1. CUDA kernel implementations for gate application
 * 2. cuQuantum integration (if available)
 * 3. GPU memory management
 * 4. Host-device data transfer
 * 
 * BUILD:
 * Compiled only when USE_GPU=ON in CMake
 * 
 * @author LRET Team
 * @date January 2026
 */

#include "gpu_simulator.h"
#include "gates_and_noise.h"

#ifdef USE_GPU

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <iostream>
#include <iomanip>
#include <stdexcept>

#ifdef USE_CUQUANTUM
#include <custatevec.h>
#endif

namespace qlret {

//==============================================================================
// CUDA Error Checking
//==============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + \
                cudaGetErrorString(err) + " at " + __FILE__ + ":" + \
                std::to_string(__LINE__)); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            throw std::runtime_error(std::string("cuBLAS error at ") + \
                __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while(0)

#ifdef USE_CUQUANTUM
#define CUSTATEVEC_CHECK(call) \
    do { \
        custatevecStatus_t status = call; \
        if (status != CUSTATEVEC_STATUS_SUCCESS) { \
            throw std::runtime_error(std::string("cuStateVec error at ") + \
                __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while(0)
#endif

//==============================================================================
// GPU Memory Manager Implementation
//==============================================================================

size_t GPUMemoryManager::estimate_memory(size_t num_qubits, size_t rank) {
    size_t dim = 1ULL << num_qubits;
    // L matrix: dim × rank complex doubles (16 bytes each)
    size_t L_size = dim * rank * sizeof(cuDoubleComplex);
    // Workspace for operations (2x L size for safety)
    size_t workspace = L_size * 2;
    // Gate matrices and other small buffers
    size_t overhead = 1024 * 1024;  // 1 MB overhead
    
    return L_size + workspace + overhead;
}

bool GPUMemoryManager::can_fit_on_gpu(size_t num_qubits, size_t rank, int device_id) {
    size_t required = estimate_memory(num_qubits, rank);
    
    cudaSetDevice(device_id);
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    
    // Leave 10% buffer for system
    return required < (free_mem * 0.9);
}

std::string GPUMemoryManager::recommend_device(size_t num_qubits, size_t rank) {
    // For small problems, CPU may be faster due to transfer overhead
    if (num_qubits < 10) {
        return "cpu";
    }
    
    // Check if GPU is available and has enough memory
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    
    if (device_count == 0) {
        return "cpu";
    }
    
    if (can_fit_on_gpu(num_qubits, rank, 0)) {
        return "gpu";
    }
    
    return "cpu";
}

void GPUMemoryManager::print_memory_estimate(size_t num_qubits, size_t rank) {
    size_t required = estimate_memory(num_qubits, rank);
    double required_gb = required / (1024.0 * 1024.0 * 1024.0);
    
    std::cout << "GPU Memory Estimate for n=" << num_qubits 
              << ", rank=" << rank << ":\n";
    std::cout << "  Required: " << std::fixed << std::setprecision(2) 
              << required_gb << " GB\n";
    
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    
    std::cout << "  Available: " << (free_mem / (1024.0 * 1024.0 * 1024.0)) 
              << " GB / " << (total_mem / (1024.0 * 1024.0 * 1024.0)) << " GB\n";
    std::cout << "  Status: " << (required < free_mem * 0.9 ? "FITS" : "TOO LARGE") << "\n";
}

//==============================================================================
// GPU Device Query Functions
//==============================================================================

bool is_gpu_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

int get_gpu_count() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return device_count;
}

GPUDeviceInfo get_gpu_info(int device_id) {
    GPUDeviceInfo info;
    
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    
    if (err != cudaSuccess) {
        return info;  // Return empty info
    }
    
    info.device_id = device_id;
    info.name = prop.name;
    info.total_memory = prop.totalGlobalMem;
    info.compute_capability_major = prop.major;
    info.compute_capability_minor = prop.minor;
    info.multiprocessor_count = prop.multiProcessorCount;
    info.supports_double = (prop.major >= 2);  // SM 2.0+
    
    // Get free memory
    cudaSetDevice(device_id);
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    info.free_memory = free_mem;
    
    return info;
}

std::vector<GPUDeviceInfo> get_all_gpu_info() {
    std::vector<GPUDeviceInfo> devices;
    int count = get_gpu_count();
    
    for (int i = 0; i < count; ++i) {
        devices.push_back(get_gpu_info(i));
    }
    
    return devices;
}

bool is_cuquantum_available() {
#ifdef USE_CUQUANTUM
    return true;
#else
    return false;
#endif
}

void print_gpu_info() {
    std::cout << "===== GPU Information =====\n";
    
    if (!is_gpu_available()) {
        std::cout << "No NVIDIA GPU detected.\n";
        return;
    }
    
    auto devices = get_all_gpu_info();
    std::cout << "Found " << devices.size() << " GPU(s):\n";
    
    for (const auto& dev : devices) {
        std::cout << "\nDevice " << dev.device_id << ": " << dev.name << "\n";
        std::cout << "  Compute Capability: " << dev.compute_capability_str() << "\n";
        std::cout << "  Total Memory: " << std::fixed << std::setprecision(2) 
                  << dev.total_memory_gb() << " GB\n";
        std::cout << "  Free Memory: " << dev.free_memory_gb() << " GB\n";
        std::cout << "  Multiprocessors: " << dev.multiprocessor_count << "\n";
    }
    
    std::cout << "\ncuQuantum: " << (is_cuquantum_available() ? "Available" : "Not Available") << "\n";
    std::cout << "===========================\n";
}

//==============================================================================
// CUDA Kernels for Gate Application
//==============================================================================

/**
 * @brief CUDA kernel: Apply single-qubit gate to L matrix (column-major, coalesced)
 * 
 * Column-major layout: L[row, col] stored at L[col * dim + row]
 * Adjacent threads access adjacent memory locations for coalesced reads/writes.
 * 
 * Each thread handles one (row_pair, column) combination.
 * Gate matrix cached in shared memory.
 */
__global__ void apply_single_qubit_gate_kernel_colmajor(
    cuDoubleComplex* L,
    size_t dim,
    size_t rank,
    size_t qubit,
    cuDoubleComplex u00, cuDoubleComplex u01,
    cuDoubleComplex u10, cuDoubleComplex u11
) {
    // Cache gate matrix in shared memory
    __shared__ cuDoubleComplex gate_shared[4];
    if (threadIdx.x == 0) {
        gate_shared[0] = u00;
        gate_shared[1] = u01;
        gate_shared[2] = u10;
        gate_shared[3] = u11;
    }
    __syncthreads();

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t step = 1ULL << qubit;
    size_t num_pairs = dim / 2;
    
    // Total work: num_pairs * rank
    size_t total_work = num_pairs * rank;
    if (idx >= total_work) return;
    
    // Decode: column-first ordering for coalesced access
    size_t col = idx / num_pairs;
    size_t pair_idx = idx % num_pairs;
    
    // Calculate row indices for the pair
    size_t block_idx = pair_idx / step;
    size_t within_block = pair_idx % step;
    size_t row0 = block_idx * (2 * step) + within_block;
    size_t row1 = row0 + step;
    
    // Read gate from shared memory
    cuDoubleComplex g00 = gate_shared[0];
    cuDoubleComplex g01 = gate_shared[1];
    cuDoubleComplex g10 = gate_shared[2];
    cuDoubleComplex g11 = gate_shared[3];
    
    // Column-major access: L[row, col] = L[col * dim + row]
    cuDoubleComplex v0 = L[col * dim + row0];
    cuDoubleComplex v1 = L[col * dim + row1];
    
    // Apply 2x2 unitary
    L[col * dim + row0] = cuCadd(cuCmul(g00, v0), cuCmul(g01, v1));
    L[col * dim + row1] = cuCadd(cuCmul(g10, v0), cuCmul(g11, v1));
}

/**
 * @brief CUDA kernel: Apply Kraus expansion (column-major, coalesced)
 * 
 * Column-major layout with coalesced memory access pattern.
 * Adjacent threads process adjacent rows within same column for coalescing.
 */
__global__ void apply_kraus_expansion_kernel_colmajor(
    const cuDoubleComplex* L_in,   // Input L matrix (dim x old_rank), column-major
    cuDoubleComplex* L_out,        // Output L matrix (dim x new_rank), column-major
    size_t dim,
    size_t old_rank,
    size_t num_kraus,
    size_t qubit,
    const cuDoubleComplex* kraus_ops  // num_kraus * 4 elements
) {
    // Cache Kraus operators in shared memory
    extern __shared__ cuDoubleComplex kraus_shared[];
    size_t total_kraus_elems = num_kraus * 4;
    for (size_t i = threadIdx.x; i < total_kraus_elems; i += blockDim.x) {
        kraus_shared[i] = kraus_ops[i];
    }
    __syncthreads();

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t step = 1ULL << qubit;
    size_t num_pairs = dim / 2;
    
    // Total work: num_pairs * old_rank * num_kraus
    size_t total_work = num_pairs * old_rank * num_kraus;
    if (idx >= total_work) return;
    
    // Decode with pair_idx varying fastest for coalesced row access
    size_t pair_idx = idx % num_pairs;
    size_t remainder = idx / num_pairs;
    size_t col_in = remainder % old_rank;
    size_t kraus_idx = remainder / old_rank;
    
    // Calculate row indices
    size_t block_idx = pair_idx / step;
    size_t within_block = pair_idx % step;
    size_t row0 = block_idx * (2 * step) + within_block;
    size_t row1 = row0 + step;
    
    // Load Kraus operator from shared memory
    cuDoubleComplex k00 = kraus_shared[kraus_idx * 4 + 0];
    cuDoubleComplex k01 = kraus_shared[kraus_idx * 4 + 1];
    cuDoubleComplex k10 = kraus_shared[kraus_idx * 4 + 2];
    cuDoubleComplex k11 = kraus_shared[kraus_idx * 4 + 3];
    
    // Column-major access for input
    cuDoubleComplex v0 = L_in[col_in * dim + row0];
    cuDoubleComplex v1 = L_in[col_in * dim + row1];
    
    // Compute output column index
    size_t new_rank = old_rank * num_kraus;
    size_t col_out = kraus_idx * old_rank + col_in;
    
    // Column-major access for output
    L_out[col_out * dim + row0] = cuCadd(cuCmul(k00, v0), cuCmul(k01, v1));
    L_out[col_out * dim + row1] = cuCadd(cuCmul(k10, v0), cuCmul(k11, v1));
}

/**
 * @brief CUDA kernel: Apply single-qubit gate to L matrix (shared-memory optimized)
 * 
 * For L matrix (dim × rank), applies U to qubit q:
 * L[i, :] and L[i ^ (1 << q), :] are mixed according to U
 * 
 * Each thread handles one pair of rows.
 * Gate matrix cached in shared memory to avoid redundant global reads.
 */
__global__ void apply_single_qubit_gate_kernel(
    cuDoubleComplex* L,
    size_t dim,
    size_t rank,
    size_t qubit,
    cuDoubleComplex u00, cuDoubleComplex u01,
    cuDoubleComplex u10, cuDoubleComplex u11
) {
    // Cache gate matrix in shared memory (all threads in block share)
    __shared__ cuDoubleComplex gate_shared[4];
    if (threadIdx.x == 0) {
        gate_shared[0] = u00;
        gate_shared[1] = u01;
        gate_shared[2] = u10;
        gate_shared[3] = u11;
    }
    __syncthreads();

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t step = 1ULL << qubit;
    
    // Each thread processes one pair
    size_t num_pairs = dim / 2;
    if (idx >= num_pairs) return;
    
    // Calculate row indices for the pair
    size_t block_idx = idx / step;
    size_t within_block = idx % step;
    size_t row0 = block_idx * (2 * step) + within_block;
    size_t row1 = row0 + step;
    
    // Read gate from shared memory
    cuDoubleComplex g00 = gate_shared[0];
    cuDoubleComplex g01 = gate_shared[1];
    cuDoubleComplex g10 = gate_shared[2];
    cuDoubleComplex g11 = gate_shared[3];
    
    // Process all columns (rank elements)
    for (size_t c = 0; c < rank; ++c) {
        cuDoubleComplex v0 = L[row0 * rank + c];
        cuDoubleComplex v1 = L[row1 * rank + c];
        
        // Apply 2x2 unitary: [g00 g01; g10 g11] * [v0; v1]
        L[row0 * rank + c] = cuCadd(cuCmul(g00, v0), cuCmul(g01, v1));
        L[row1 * rank + c] = cuCadd(cuCmul(g10, v0), cuCmul(g11, v1));
    }
}

/**
 * @brief CUDA kernel: Apply multiple Kraus operators to expand L matrix rank
 * 
 * LRET Kraus expansion: L_out[:, k*old_rank : (k+1)*old_rank] = K_k ⊗ I * L_in
 * Output matrix has rank = old_rank * num_kraus
 * 
 * Each thread handles one (row_pair, column, kraus_index) combination.
 * Kraus operators cached in shared memory for efficiency.
 */
__global__ void apply_kraus_expansion_kernel(
    const cuDoubleComplex* L_in,   // Input L matrix (dim x old_rank)
    cuDoubleComplex* L_out,        // Output L matrix (dim x new_rank)
    size_t dim,
    size_t old_rank,
    size_t num_kraus,
    size_t qubit,
    const cuDoubleComplex* kraus_ops  // num_kraus * 4 elements (K0, K1, ...)
) {
    // Cache all Kraus operators in shared memory (max 16 Kraus ops = 64 elements)
    extern __shared__ cuDoubleComplex kraus_shared[];
    size_t total_kraus_elems = num_kraus * 4;
    for (size_t i = threadIdx.x; i < total_kraus_elems; i += blockDim.x) {
        kraus_shared[i] = kraus_ops[i];
    }
    __syncthreads();

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t step = 1ULL << qubit;
    size_t num_pairs = dim / 2;
    
    // Total work items: num_pairs * old_rank * num_kraus
    size_t total_work = num_pairs * old_rank * num_kraus;
    if (idx >= total_work) return;
    
    // Decode work item
    size_t kraus_idx = idx % num_kraus;
    size_t remainder = idx / num_kraus;
    size_t col_in = remainder % old_rank;
    size_t pair_idx = remainder / old_rank;
    
    // Calculate row indices for the pair
    size_t block_idx = pair_idx / step;
    size_t within_block = pair_idx % step;
    size_t row0 = block_idx * (2 * step) + within_block;
    size_t row1 = row0 + step;
    
    // Load Kraus operator from shared memory
    cuDoubleComplex k00 = kraus_shared[kraus_idx * 4 + 0];
    cuDoubleComplex k01 = kraus_shared[kraus_idx * 4 + 1];
    cuDoubleComplex k10 = kraus_shared[kraus_idx * 4 + 2];
    cuDoubleComplex k11 = kraus_shared[kraus_idx * 4 + 3];
    
    // Load input values
    cuDoubleComplex v0 = L_in[row0 * old_rank + col_in];
    cuDoubleComplex v1 = L_in[row1 * old_rank + col_in];
    
    // Compute output column index
    size_t new_rank = old_rank * num_kraus;
    size_t col_out = kraus_idx * old_rank + col_in;
    
    // Apply Kraus: K * [v0; v1]
    L_out[row0 * new_rank + col_out] = cuCadd(cuCmul(k00, v0), cuCmul(k01, v1));
    L_out[row1 * new_rank + col_out] = cuCadd(cuCmul(k10, v0), cuCmul(k11, v1));
}

/**
 * @brief CUDA kernel: Apply two-qubit gate to L matrix
 * 
 * Processes 4 rows at a time for the 4x4 gate.
 */
__global__ void apply_two_qubit_gate_kernel(
    cuDoubleComplex* L,
    size_t dim,
    size_t rank,
    size_t qubit1,  // Lower qubit index
    size_t qubit2,  // Higher qubit index
    const cuDoubleComplex* gate  // 4x4 gate matrix in row-major order
) {
    // Cache 4x4 gate in shared memory once per block to avoid repeated global reads
    __shared__ cuDoubleComplex gate_shared[16];
    if (threadIdx.x < 16) {
        gate_shared[threadIdx.x] = gate[threadIdx.x];
    }
    __syncthreads();

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    size_t step1 = 1ULL << qubit1;
    size_t step2 = 1ULL << qubit2;

    // Number of 4-tuples
    size_t num_quads = dim / 4;
    if (idx >= num_quads) return;

    // Compute base row by inserting zero bits at qubit1 and qubit2 positions
    size_t low_mask = (1ULL << qubit1) - 1ULL;
    size_t middle_bits = (qubit2 > qubit1 + 1) ? (qubit2 - qubit1 - 1) : 0;
    size_t mid_mask = (middle_bits == 0) ? 0ULL : ((1ULL << middle_bits) - 1ULL);

    size_t low = idx & low_mask;
    size_t mid = (middle_bits == 0) ? 0ULL : ((idx >> qubit1) & mid_mask);
    size_t high = idx >> (qubit1 + middle_bits);

    size_t base = low;
    base |= mid << (qubit1 + 1);          // insert zero at qubit1
    base |= high << (qubit2 + 1);         // insert zero at qubit2 (after first insert)

    // Get the 4 row indices
    size_t rows[4];
    rows[0] = base;                        // |00⟩
    rows[1] = base | step1;                // |01⟩
    rows[2] = base | step2;                // |10⟩
    rows[3] = base | step1 | step2;        // |11⟩

    // Process all columns
    for (size_t c = 0; c < rank; ++c) {
        // Load 4 values
        cuDoubleComplex v[4];
        for (int i = 0; i < 4; ++i) {
            v[i] = L[rows[i] * rank + c];
        }

        // Apply 4x4 gate using cached matrix
        cuDoubleComplex result[4];
        for (int i = 0; i < 4; ++i) {
            result[i] = make_cuDoubleComplex(0.0, 0.0);
            for (int j = 0; j < 4; ++j) {
                result[i] = cuCadd(result[i], cuCmul(gate_shared[i * 4 + j], v[j]));
            }
        }

        // Store results
        for (int i = 0; i < 4; ++i) {
            L[rows[i] * rank + c] = result[i];
        }
    }
}

//==============================================================================
// GPU Simulator Implementation
//==============================================================================

class GPUSimulatorImpl {
public:
    size_t num_qubits_;
    size_t dim_;
    size_t rank_;
    GPUConfig config_;
    
    // Device memory
    cuDoubleComplex* d_L_ = nullptr;
    cuDoubleComplex* d_workspace_ = nullptr;
    
    // cuBLAS handle
    cublasHandle_t cublas_handle_ = nullptr;
    
#ifdef USE_CUQUANTUM
    custatevecHandle_t custatevec_handle_ = nullptr;
#endif
    
    size_t allocated_bytes_ = 0;
    bool gpu_active_ = false;
    
    GPUSimulatorImpl(size_t num_qubits, const GPUConfig& config)
        : num_qubits_(num_qubits), config_(config) {
        
        dim_ = 1ULL << num_qubits;
        rank_ = 1;  // Initial rank
        
        // Set GPU device
        int device = config.device_id;
        if (device < 0) {
            device = 0;  // Default to first GPU
        }
        CUDA_CHECK(cudaSetDevice(device));
        
        // Create cuBLAS handle
        CUBLAS_CHECK(cublasCreate(&cublas_handle_));
        
#ifdef USE_CUQUANTUM
        if (config.use_cuquantum) {
            CUSTATEVEC_CHECK(custatevecCreate(&custatevec_handle_));
        }
#endif
        
        gpu_active_ = true;
        
        if (config.verbose) {
            std::cout << "GPU Simulator initialized:\n";
            std::cout << "  Qubits: " << num_qubits << "\n";
            std::cout << "  Dimension: " << dim_ << "\n";
            std::cout << "  Device: " << device << "\n";
            std::cout << "  cuQuantum: " << (is_cuquantum_available() ? "Yes" : "No") << "\n";
            std::cout << "  Layout: " << (config.use_column_major ? "column-major (coalesced)" : "row-major") << "\n";
        }
    }
    
    ~GPUSimulatorImpl() {
        if (d_L_) cudaFree(d_L_);
        if (d_workspace_) cudaFree(d_workspace_);
        if (cublas_handle_) cublasDestroy(cublas_handle_);
        
#ifdef USE_CUQUANTUM
        if (custatevec_handle_) custatevecDestroy(custatevec_handle_);
#endif
    }
    
    void upload(const MatrixXcd& L) {
        rank_ = L.cols();
        size_t size = dim_ * rank_ * sizeof(cuDoubleComplex);
        
        // Allocate GPU memory
        if (d_L_) cudaFree(d_L_);
        CUDA_CHECK(cudaMalloc(&d_L_, size));
        allocated_bytes_ = size;
        
        std::vector<cuDoubleComplex> host_data(dim_ * rank_);
        
        if (config_.use_column_major) {
            // Column-major: L[row, col] at L[col * dim + row] (matches Eigen)
            for (size_t c = 0; c < rank_; ++c) {
                for (size_t r = 0; r < dim_; ++r) {
                    host_data[c * dim_ + r] = make_cuDoubleComplex(
                        L(r, c).real(), L(r, c).imag()
                    );
                }
            }
        } else {
            // Row-major: L[row, col] at L[row * rank + col]
            for (size_t r = 0; r < dim_; ++r) {
                for (size_t c = 0; c < rank_; ++c) {
                    host_data[r * rank_ + c] = make_cuDoubleComplex(
                        L(r, c).real(), L(r, c).imag()
                    );
                }
            }
        }
        
        CUDA_CHECK(cudaMemcpy(d_L_, host_data.data(), size, cudaMemcpyHostToDevice));
        
        if (config_.verbose) {
            std::cout << "Uploaded L matrix: " << dim_ << " x " << rank_ 
                      << " (" << (size / (1024.0 * 1024.0)) << " MB, "
                      << (config_.use_column_major ? "col-major" : "row-major") << ")\n";
        }
    }
    
    MatrixXcd download() const {
        size_t size = dim_ * rank_ * sizeof(cuDoubleComplex);
        std::vector<cuDoubleComplex> host_data(dim_ * rank_);
        
        CUDA_CHECK(cudaMemcpy(host_data.data(), d_L_, size, cudaMemcpyDeviceToHost));
        
        MatrixXcd L(dim_, rank_);
        
        if (config_.use_column_major) {
            // Column-major: L[row, col] at L[col * dim + row]
            for (size_t c = 0; c < rank_; ++c) {
                for (size_t r = 0; r < dim_; ++r) {
                    auto& val = host_data[c * dim_ + r];
                    L(r, c) = Complex(cuCreal(val), cuCimag(val));
                }
            }
        } else {
            // Row-major: L[row, col] at L[row * rank + col]
            for (size_t r = 0; r < dim_; ++r) {
                for (size_t c = 0; c < rank_; ++c) {
                    auto& val = host_data[r * rank_ + c];
                    L(r, c) = Complex(cuCreal(val), cuCimag(val));
                }
            }
        }
        
        return L;
    }
    
    void apply_single_qubit(size_t qubit, const Matrix2cd& gate) {
#ifdef USE_CUQUANTUM
        if (config_.use_cuquantum && custatevec_handle_) {
            apply_single_qubit_cuquantum(qubit, gate);
            return;
        }
#endif
        apply_single_qubit_custom(qubit, gate);
    }
    
    void apply_single_qubit_custom(size_t qubit, const Matrix2cd& gate) {
        // Extract gate elements
        cuDoubleComplex u00 = make_cuDoubleComplex(gate(0,0).real(), gate(0,0).imag());
        cuDoubleComplex u01 = make_cuDoubleComplex(gate(0,1).real(), gate(0,1).imag());
        cuDoubleComplex u10 = make_cuDoubleComplex(gate(1,0).real(), gate(1,0).imag());
        cuDoubleComplex u11 = make_cuDoubleComplex(gate(1,1).real(), gate(1,1).imag());
        
        if (config_.use_column_major) {
            // Column-major kernel with coalesced access
            size_t num_pairs = dim_ / 2;
            size_t total_work = num_pairs * rank_;
            int threads = 256;
            int blocks = (total_work + threads - 1) / threads;
            
            apply_single_qubit_gate_kernel_colmajor<<<blocks, threads>>>(
                d_L_, dim_, rank_, qubit, u00, u01, u10, u11
            );
        } else {
            // Row-major kernel
            size_t num_pairs = dim_ / 2;
            int threads = 256;
            int blocks = (num_pairs + threads - 1) / threads;
            
            apply_single_qubit_gate_kernel<<<blocks, threads>>>(
                d_L_, dim_, rank_, qubit, u00, u01, u10, u11
            );
        }
        
        CUDA_CHECK(cudaGetLastError());
    }
    
#ifdef USE_CUQUANTUM
    void apply_single_qubit_cuquantum(size_t qubit, const Matrix2cd& gate) {
        // cuQuantum gate matrix (column-major)
        cuDoubleComplex gate_data[4];
        gate_data[0] = make_cuDoubleComplex(gate(0,0).real(), gate(0,0).imag());
        gate_data[1] = make_cuDoubleComplex(gate(1,0).real(), gate(1,0).imag());
        gate_data[2] = make_cuDoubleComplex(gate(0,1).real(), gate(0,1).imag());
        gate_data[3] = make_cuDoubleComplex(gate(1,1).real(), gate(1,1).imag());
        
        int32_t target = static_cast<int32_t>(qubit);
        
        // Apply to each column of L separately
        // cuQuantum operates on state vectors, we have L columns
        // NOTE: cuQuantum expects contiguous state vectors. For L matrix:
        //   - Column-major: column c starts at d_L_ + c * dim_ (columns are contiguous)
        //   - Row-major: elements of column c are at d_L_[row * rank_ + c] (NOT contiguous!)
        // cuQuantum can only be used efficiently with column-major layout.
        if (!config_.use_column_major) {
            // Fall back to custom kernel for row-major layout
            apply_single_qubit_custom(qubit, gate);
            return;
        }
        
        for (size_t c = 0; c < rank_; ++c) {
            cuDoubleComplex* column = d_L_ + c * dim_;  // Column-major: column c starts at c*dim
            
            CUSTATEVEC_CHECK(custatevecApplyMatrix(
                custatevec_handle_,
                column,
                CUDA_C_64F,
                num_qubits_,
                gate_data,
                CUDA_C_64F,
                CUSTATEVEC_MATRIX_LAYOUT_COL,
                0,  // adjoint
                &target,
                1,  // num targets
                nullptr,  // controls
                nullptr,  // control bit values
                0,  // num controls
                CUSTATEVEC_COMPUTE_64F,
                nullptr,  // workspace
                0  // workspace size
            ));
        }
    }
#endif
    
    void apply_two_qubit(size_t qubit1, size_t qubit2, const Matrix4cd& gate) {
        // Ensure qubit1 < qubit2
        if (qubit1 > qubit2) std::swap(qubit1, qubit2);
        
        // Upload gate to device
        cuDoubleComplex h_gate[16];
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                h_gate[i * 4 + j] = make_cuDoubleComplex(
                    gate(i, j).real(), gate(i, j).imag()
                );
            }
        }
        
        cuDoubleComplex* d_gate;
        CUDA_CHECK(cudaMalloc(&d_gate, 16 * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMemcpy(d_gate, h_gate, 16 * sizeof(cuDoubleComplex), 
                              cudaMemcpyHostToDevice));
        
        size_t num_quads = dim_ / 4;
        int threads = 256;
        int blocks = (num_quads + threads - 1) / threads;
        
        apply_two_qubit_gate_kernel<<<blocks, threads>>>(
            d_L_, dim_, rank_, qubit1, qubit2, d_gate
        );
        
        CUDA_CHECK(cudaGetLastError());
        cudaFree(d_gate);
    }
    
    /**
     * @brief Apply Kraus operators with proper LRET rank expansion on GPU
     * 
     * For LRET: applying k Kraus operators multiplies rank by k.
     * L_out has dimensions (dim x new_rank) where new_rank = old_rank * num_kraus.
     * Supports both row-major and column-major layouts.
     */
    void apply_kraus_expand(size_t qubit, const std::vector<Matrix2cd>& kraus_ops) {
        if (kraus_ops.empty()) return;
        
        size_t num_kraus = kraus_ops.size();
        size_t new_rank = rank_ * num_kraus;
        size_t new_size = dim_ * new_rank * sizeof(cuDoubleComplex);
        
        // Allocate output buffer
        cuDoubleComplex* d_L_out;
        CUDA_CHECK(cudaMalloc(&d_L_out, new_size));
        
        // Prepare and upload Kraus operators (row-major: K[i][j] = kraus[k](i,j))
        std::vector<cuDoubleComplex> h_kraus(num_kraus * 4);
        for (size_t k = 0; k < num_kraus; ++k) {
            const auto& K = kraus_ops[k];
            h_kraus[k * 4 + 0] = make_cuDoubleComplex(K(0,0).real(), K(0,0).imag());
            h_kraus[k * 4 + 1] = make_cuDoubleComplex(K(0,1).real(), K(0,1).imag());
            h_kraus[k * 4 + 2] = make_cuDoubleComplex(K(1,0).real(), K(1,0).imag());
            h_kraus[k * 4 + 3] = make_cuDoubleComplex(K(1,1).real(), K(1,1).imag());
        }
        
        cuDoubleComplex* d_kraus;
        CUDA_CHECK(cudaMalloc(&d_kraus, h_kraus.size() * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMemcpy(d_kraus, h_kraus.data(), 
                              h_kraus.size() * sizeof(cuDoubleComplex),
                              cudaMemcpyHostToDevice));
        
        // Launch appropriate kernel based on memory layout
        size_t num_pairs = dim_ / 2;
        size_t total_work = num_pairs * rank_ * num_kraus;
        int threads = 256;
        int blocks = (total_work + threads - 1) / threads;
        size_t shared_mem = num_kraus * 4 * sizeof(cuDoubleComplex);
        
        if (config_.use_column_major) {
            apply_kraus_expansion_kernel_colmajor<<<blocks, threads, shared_mem>>>(
                d_L_, d_L_out, dim_, rank_, num_kraus, qubit, d_kraus
            );
        } else {
            apply_kraus_expansion_kernel<<<blocks, threads, shared_mem>>>(
                d_L_, d_L_out, dim_, rank_, num_kraus, qubit, d_kraus
            );
        }
        
        CUDA_CHECK(cudaGetLastError());
        
        // Swap buffers and update rank
        cudaFree(d_L_);
        cudaFree(d_kraus);
        d_L_ = d_L_out;
        rank_ = new_rank;
        allocated_bytes_ = new_size;
        
        if (config_.verbose) {
            std::cout << "Kraus expansion: rank " << (rank_ / num_kraus) 
                      << " -> " << rank_ << " (" << num_kraus << " operators, "
                      << (config_.use_column_major ? "col-major" : "row-major") << ")\n";
        }
    }
};

//==============================================================================
// GPUSimulator Public Interface
//==============================================================================

GPUSimulator::GPUSimulator(size_t num_qubits, const GPUConfig& config) {
    impl_ = std::make_unique<GPUSimulatorImpl>(num_qubits, config);
}

GPUSimulator::~GPUSimulator() = default;

GPUSimulator::GPUSimulator(GPUSimulator&&) noexcept = default;
GPUSimulator& GPUSimulator::operator=(GPUSimulator&&) noexcept = default;

void GPUSimulator::upload_state(const MatrixXcd& L) {
    impl_->upload(L);
}

MatrixXcd GPUSimulator::download_state() const {
    return impl_->download();
}

size_t GPUSimulator::get_rank() const {
    return impl_->rank_;
}

size_t GPUSimulator::get_num_qubits() const {
    return impl_->num_qubits_;
}

void GPUSimulator::apply_gate(const GateOp& gate) {
    if (gate.qubits.size() == 1) {
        Matrix2cd U = get_single_qubit_gate(gate.type, gate.params);
        apply_single_qubit_gate(gate.qubits[0], U);
    } else if (gate.qubits.size() == 2) {
        Matrix4cd U = get_two_qubit_gate(gate.type, gate.params);
        apply_two_qubit_gate(gate.qubits[0], gate.qubits[1], U);
    }
}

void GPUSimulator::apply_single_qubit_gate(size_t qubit, const Matrix2cd& gate_matrix) {
    impl_->apply_single_qubit(qubit, gate_matrix);
}

void GPUSimulator::apply_two_qubit_gate(size_t qubit1, size_t qubit2, const Matrix4cd& gate_matrix) {
    impl_->apply_two_qubit(qubit1, qubit2, gate_matrix);
}

void GPUSimulator::apply_noise(const NoiseOp& noise) {
    // For LRET noise: Apply Kraus operators with proper rank expansion
    // L_new = [K_0 ⊗ I * L | K_1 ⊗ I * L | ... | K_k ⊗ I * L]
    // Rank increases by factor of num_kraus_operators
    auto kraus_full = get_noise_kraus_operators(noise.type, noise.probability, noise.params);
    if (!kraus_full.empty()) {
        // Convert MatrixXcd to Matrix2cd for single-qubit Kraus operators
        std::vector<Matrix2cd> kraus;
        kraus.reserve(kraus_full.size());
        for (const auto& K : kraus_full) {
            if (K.rows() == 2 && K.cols() == 2) {
                kraus.push_back(K);
            }
        }
        if (!kraus.empty()) {
            size_t target = noise.qubits.empty() ? 0 : noise.qubits[0];
            apply_kraus(target, kraus);
        }
    }
}

void GPUSimulator::apply_kraus(size_t qubit, const std::vector<Matrix2cd>& kraus_ops) {
    // Full LRET Kraus expansion on GPU
    // This increases rank: L_new has rank = old_rank * num_kraus_ops
    impl_->apply_kraus_expand(qubit, kraus_ops);
}

size_t GPUSimulator::truncate(double threshold) {
    // Download, truncate on CPU, upload
    // TODO: Implement GPU-based SVD truncation
    MatrixXcd L = download_state();
    
    // Simple norm-based truncation (fast approximation)
    Eigen::VectorXd col_norms(L.cols());
    for (int c = 0; c < L.cols(); ++c) {
        col_norms(c) = L.col(c).norm();
    }
    
    // Find columns above threshold
    std::vector<int> keep_cols;
    for (int c = 0; c < L.cols(); ++c) {
        if (col_norms(c) > threshold) {
            keep_cols.push_back(c);
        }
    }
    
    if (keep_cols.empty()) {
        keep_cols.push_back(0);  // Keep at least one column
    }
    
    // Build truncated L
    MatrixXcd L_trunc(L.rows(), keep_cols.size());
    for (size_t i = 0; i < keep_cols.size(); ++i) {
        L_trunc.col(i) = L.col(keep_cols[i]);
    }
    
    upload_state(L_trunc);
    return L_trunc.cols();
}

bool GPUSimulator::is_gpu_active() const {
    return impl_->gpu_active_;
}

GPUDeviceInfo GPUSimulator::get_device_info() const {
    return get_gpu_info(impl_->config_.device_id);
}

void GPUSimulator::synchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

size_t GPUSimulator::get_memory_usage() const {
    return impl_->allocated_bytes_;
}

//==============================================================================
// High-Level Simulation Functions
//==============================================================================

MatrixXcd simulate_gpu(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    const SimConfig& config,
    const GPUConfig& gpu_config
) {
    GPUSimulator gpu(num_qubits, gpu_config);
    gpu.upload_state(L_init);
    
    size_t step = 0;
    size_t total = sequence.operations.size();
    
    for (const auto& op : sequence.operations) {
        step++;
        
        if (std::holds_alternative<GateOp>(op)) {
            const auto& gate = std::get<GateOp>(op);
            gpu.apply_gate(gate);
        } else {
            const auto& noise = std::get<NoiseOp>(op);
            gpu.apply_noise(noise);
            
            // Truncation after noise
            if (config.do_truncation) {
                gpu.truncate(config.truncation_threshold);
            }
        }
        
        if (gpu_config.verbose && step % 100 == 0) {
            std::cout << "GPU step " << step << "/" << total 
                      << " rank=" << gpu.get_rank() << std::endl;
        }
    }
    
    gpu.synchronize();
    return gpu.download_state();
}

MatrixXcd simulate_auto(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    const SimConfig& config,
    bool prefer_gpu
) {
    // Check if GPU is available and suitable
    bool use_gpu = prefer_gpu && is_gpu_available();
    
    if (use_gpu) {
        size_t rank_estimate = std::max(L_init.cols(), static_cast<Eigen::Index>(10));
        use_gpu = GPUMemoryManager::can_fit_on_gpu(num_qubits, rank_estimate);
    }
    
    if (use_gpu) {
        GPUConfig gpu_config;
        gpu_config.verbose = config.verbose;
        return simulate_gpu(L_init, sequence, num_qubits, config, gpu_config);
    } else {
        // Fall back to CPU simulation
        // Import from simulator.h
        extern MatrixXcd run_lret_simulation(
            const MatrixXcd&, const QuantumSequence&, size_t, const SimConfig&);
        return run_lret_simulation(L_init, sequence, num_qubits, config);
    }
}

}  // namespace qlret

#else  // !USE_GPU

//==============================================================================
// CPU-Only Stubs (when GPU not enabled)
//==============================================================================

namespace qlret {

// Stub implementations when GPU is not available
size_t GPUMemoryManager::estimate_memory(size_t, size_t) { return 0; }
bool GPUMemoryManager::can_fit_on_gpu(size_t, size_t, int) { return false; }
std::string GPUMemoryManager::recommend_device(size_t, size_t) { return "cpu"; }
void GPUMemoryManager::print_memory_estimate(size_t, size_t) {
    std::cout << "GPU support not compiled. Use -DUSE_GPU=ON\n";
}

bool is_gpu_available() { return false; }
int get_gpu_count() { return 0; }
std::vector<GPUDeviceInfo> get_all_gpu_info() { return {}; }
GPUDeviceInfo get_gpu_info(int) { return GPUDeviceInfo(); }
bool is_cuquantum_available() { return false; }
void print_gpu_info() {
    std::cout << "GPU support not compiled. Rebuild with -DUSE_GPU=ON\n";
}

// GPUSimulator stubs
class GPUSimulatorImpl {};

GPUSimulator::GPUSimulator(size_t, const GPUConfig&) {
    throw std::runtime_error("GPU support not compiled. Use -DUSE_GPU=ON");
}
GPUSimulator::~GPUSimulator() = default;
GPUSimulator::GPUSimulator(GPUSimulator&&) noexcept = default;
GPUSimulator& GPUSimulator::operator=(GPUSimulator&&) noexcept = default;
void GPUSimulator::upload_state(const MatrixXcd&) {}
MatrixXcd GPUSimulator::download_state() const { return MatrixXcd(); }
size_t GPUSimulator::get_rank() const { return 0; }
size_t GPUSimulator::get_num_qubits() const { return 0; }
void GPUSimulator::apply_gate(const GateOp&) {}
void GPUSimulator::apply_single_qubit_gate(size_t, const Matrix2cd&) {}
void GPUSimulator::apply_two_qubit_gate(size_t, size_t, const Matrix4cd&) {}
void GPUSimulator::apply_noise(const NoiseOp&) {}
void GPUSimulator::apply_kraus(size_t, const std::vector<Matrix2cd>&) {}
size_t GPUSimulator::truncate(double) { return 0; }
bool GPUSimulator::is_gpu_active() const { return false; }
GPUDeviceInfo GPUSimulator::get_device_info() const { return GPUDeviceInfo(); }
void GPUSimulator::synchronize() {}
size_t GPUSimulator::get_memory_usage() const { return 0; }

MatrixXcd simulate_gpu(const MatrixXcd&, const QuantumSequence&, size_t, 
                       const SimConfig&, const GPUConfig&) {
    throw std::runtime_error("GPU support not compiled");
}

MatrixXcd simulate_auto(const MatrixXcd& L_init, const QuantumSequence& sequence,
                        size_t num_qubits, const SimConfig& config, bool) {
    // Always use CPU
    extern MatrixXcd run_lret_simulation(
        const MatrixXcd&, const QuantumSequence&, size_t, const SimConfig&);
    return run_lret_simulation(L_init, sequence, num_qubits, config);
}

}  // namespace qlret

#endif  // USE_GPU
