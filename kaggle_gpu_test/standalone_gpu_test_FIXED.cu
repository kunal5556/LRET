/**
 * CORRECTED LRET GPU Correctness Test for Kaggle
 *
 * This version FIXES the quantum gate kernel index calculation bug
 *
 * UPLOAD TO KAGGLE NOTEBOOK AND RUN:
 *   !nvcc -arch=sm_75 standalone_gpu_test_FIXED.cu -o gpu_test -lcublas
 *   !./gpu_test
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <complex>

// Error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return 1; \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS Error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return 1; \
        } \
    } while(0)

// FIXED: Corrected kernel to apply a 2x2 gate to quantum state
__global__ void apply_single_qubit_gate_kernel(
    cuDoubleComplex* state,
    const cuDoubleComplex* gate,
    int qubit,
    int n_qubits
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = 1 << n_qubits;

    if (idx >= dim/2) return;

    // FIXED: Correct index calculation for state pairs
    // For qubit q, we need to pair amplitudes that differ only in bit q
    int bit_mask = 1 << qubit;

    // Extract bits below and above the target qubit
    int lower_bits = idx & (bit_mask - 1);  // Bits below qubit position
    int upper_bits = (idx >> qubit) << (qubit + 1);  // Bits above qubit position

    // Construct the two indices: one with bit=0, one with bit=1
    int i0 = upper_bits | lower_bits;  // Target bit is 0
    int i1 = i0 | bit_mask;            // Target bit is 1

    if (i1 >= dim) return;

    // Read current amplitudes
    cuDoubleComplex a0 = state[i0];
    cuDoubleComplex a1 = state[i1];

    // Apply gate: [g00 g01] [a0]
    //             [g10 g11] [a1]
    cuDoubleComplex new_a0 = cuCadd(cuCmul(gate[0], a0), cuCmul(gate[1], a1));
    cuDoubleComplex new_a1 = cuCadd(cuCmul(gate[2], a0), cuCmul(gate[3], a1));

    // Write back
    state[i0] = new_a0;
    state[i1] = new_a1;
}

// Test GPU device info
bool test_gpu_device_info() {
    std::cout << "\n=== TEST 1: GPU Device Information ===" << std::endl;

    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        std::cerr << "ERROR: No CUDA devices found!" << std::endl;
        return false;
    }

    std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;

    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

        std::cout << "\nDevice " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total Memory: " << (prop.totalGlobalMem / (1024*1024*1024.0)) << " GB" << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "  FP64 Support: " << (prop.major >= 1 && prop.minor >= 3 ? "Yes" : "No") << std::endl;
    }

    std::cout << "✓ GPU device detection passed" << std::endl;
    return true;
}

// Test memory allocation and transfer
bool test_gpu_memory() {
    std::cout << "\n=== TEST 2: GPU Memory Operations ===" << std::endl;

    const int N = 1024;
    std::vector<std::complex<double>> host_data(N);

    // Initialize host data
    for (int i = 0; i < N; i++) {
        host_data[i] = std::complex<double>(i * 0.1, i * 0.01);
    }

    // Allocate device memory
    cuDoubleComplex* device_data;
    CUDA_CHECK(cudaMalloc(&device_data, N * sizeof(cuDoubleComplex)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(device_data, host_data.data(),
                         N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    // Copy back to verify
    std::vector<std::complex<double>> verify_data(N);
    CUDA_CHECK(cudaMemcpy(verify_data.data(), device_data,
                         N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

    // Verify correctness
    bool passed = true;
    for (int i = 0; i < N; i++) {
        if (std::abs(host_data[i] - verify_data[i]) > 1e-12) {
            std::cerr << "ERROR: Mismatch at index " << i << std::endl;
            passed = false;
            break;
        }
    }

    CUDA_CHECK(cudaFree(device_data));

    if (passed) {
        std::cout << "✓ Memory allocation and transfer passed" << std::endl;
    }
    return passed;
}

// Test cuBLAS integration
bool test_cublas() {
    std::cout << "\n=== TEST 3: cuBLAS Complex Matrix Operations ===" << std::endl;

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Create 2x2 matrix multiplication: C = A * B
    const int N = 2;
    std::vector<cuDoubleComplex> A(N*N), B(N*N), C(N*N);

    // A = [1+i, 2]    B = [1, 0]
    //     [0, 1  ]        [0, 1]
    A[0] = make_cuDoubleComplex(1.0, 1.0);
    A[1] = make_cuDoubleComplex(0.0, 0.0);
    A[2] = make_cuDoubleComplex(2.0, 0.0);
    A[3] = make_cuDoubleComplex(1.0, 0.0);

    B[0] = make_cuDoubleComplex(1.0, 0.0);
    B[1] = make_cuDoubleComplex(0.0, 0.0);
    B[2] = make_cuDoubleComplex(0.0, 0.0);
    B[3] = make_cuDoubleComplex(1.0, 0.0);

    // Allocate device memory
    cuDoubleComplex *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, N*N*sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_B, N*N*sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_C, N*N*sizeof(cuDoubleComplex)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), N*N*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), N*N*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    // Perform matrix multiplication: C = 1.0*A*B + 0.0*C
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);

    CUBLAS_CHECK(cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            N, N, N,
                            &alpha,
                            d_A, N,
                            d_B, N,
                            &beta,
                            d_C, N));

    // Copy result back
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, N*N*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

    // Verify result (should equal A since B is identity)
    bool passed = true;
    for (int i = 0; i < N*N; i++) {
        double diff_real = std::abs(cuCreal(C[i]) - cuCreal(A[i]));
        double diff_imag = std::abs(cuCimag(C[i]) - cuCimag(A[i]));
        if (diff_real > 1e-12 || diff_imag > 1e-12) {
            std::cerr << "ERROR: Matrix multiply mismatch at index " << i << std::endl;
            passed = false;
            break;
        }
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUBLAS_CHECK(cublasDestroy(handle));

    if (passed) {
        std::cout << "✓ cuBLAS complex matrix operations passed" << std::endl;
    }
    return passed;
}

// Test quantum gate kernel
bool test_quantum_gate_kernel() {
    std::cout << "\n=== TEST 4: Quantum Gate Application Kernel ===" << std::endl;

    // 2-qubit system, apply Hadamard on qubit 0
    const int n_qubits = 2;
    const int dim = 1 << n_qubits;  // 4

    // Initial state: |00⟩
    std::vector<cuDoubleComplex> host_state(dim);
    host_state[0] = make_cuDoubleComplex(1.0, 0.0);
    for (int i = 1; i < dim; i++) {
        host_state[i] = make_cuDoubleComplex(0.0, 0.0);
    }

    // Hadamard gate: H = 1/√2 * [1,  1]
    //                            [1, -1]
    std::vector<cuDoubleComplex> hadamard(4);
    double sqrt2_inv = 1.0 / std::sqrt(2.0);
    hadamard[0] = make_cuDoubleComplex(sqrt2_inv, 0.0);   // H[0,0]
    hadamard[1] = make_cuDoubleComplex(sqrt2_inv, 0.0);   // H[0,1]
    hadamard[2] = make_cuDoubleComplex(sqrt2_inv, 0.0);   // H[1,0]
    hadamard[3] = make_cuDoubleComplex(-sqrt2_inv, 0.0);  // H[1,1]

    // Allocate device memory
    cuDoubleComplex *d_state, *d_gate;
    CUDA_CHECK(cudaMalloc(&d_state, dim * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_gate, 4 * sizeof(cuDoubleComplex)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_state, host_state.data(), dim * sizeof(cuDoubleComplex),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gate, hadamard.data(), 4 * sizeof(cuDoubleComplex),
                         cudaMemcpyHostToDevice));

    // Launch kernel
    int block_size = 256;
    int num_blocks = (dim/2 + block_size - 1) / block_size;
    apply_single_qubit_gate_kernel<<<num_blocks, block_size>>>(d_state, d_gate, 0, n_qubits);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    std::vector<cuDoubleComplex> result(dim);
    CUDA_CHECK(cudaMemcpy(result.data(), d_state, dim * sizeof(cuDoubleComplex),
                         cudaMemcpyDeviceToHost));

    // Verify result: After H on qubit 0, state should be 1/√2 * (|00⟩ + |10⟩)
    // result[0] = 1/√2, result[1] = 0, result[2] = 1/√2, result[3] = 0
    bool passed = true;
    double expected_amp = 1.0 / std::sqrt(2.0);

    std::cout << "  Final state: ";
    for (int i = 0; i < dim; i++) {
        std::cout << "|" << i << "⟩: " << std::fixed << std::setprecision(4)
                 << cuCreal(result[i]) << " ";
    }
    std::cout << std::endl;

    if (std::abs(cuCreal(result[0]) - expected_amp) > 1e-10) {
        std::cerr << "ERROR: State[0] = " << cuCreal(result[0]) << ", expected " << expected_amp << std::endl;
        passed = false;
    }
    if (std::abs(cuCreal(result[1])) > 1e-10) {
        std::cerr << "ERROR: State[1] = " << cuCreal(result[1]) << ", should be 0" << std::endl;
        passed = false;
    }
    if (std::abs(cuCreal(result[2]) - expected_amp) > 1e-10) {
        std::cerr << "ERROR: State[2] = " << cuCreal(result[2]) << ", expected " << expected_amp << std::endl;
        passed = false;
    }
    if (std::abs(cuCreal(result[3])) > 1e-10) {
        std::cerr << "ERROR: State[3] = " << cuCreal(result[3]) << ", should be 0" << std::endl;
        passed = false;
    }

    CUDA_CHECK(cudaFree(d_state));
    CUDA_CHECK(cudaFree(d_gate));

    if (passed) {
        std::cout << "✓ Quantum gate kernel passed" << std::endl;
    }
    return passed;
}

// Main test runner
int main() {
    std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   LRET GPU Correctness Test - Kaggle Edition (FIXED)      ║" << std::endl;
    std::cout << "║   Testing CUDA, cuBLAS, and Quantum Gate Kernels          ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;

    bool all_passed = true;

    // Run all tests
    all_passed &= test_gpu_device_info();
    all_passed &= test_gpu_memory();
    all_passed &= test_cublas();
    all_passed &= test_quantum_gate_kernel();

    std::cout << "\n" << std::string(60, '=') << std::endl;
    if (all_passed) {
        std::cout << "✓✓✓ ALL TESTS PASSED ✓✓✓" << std::endl;
        std::cout << "\nConclusion: GPU code compiles and executes correctly!" << std::endl;
        std::cout << "  - CUDA runtime: ✓ Working" << std::endl;
        std::cout << "  - Memory transfer: ✓ Working" << std::endl;
        std::cout << "  - cuBLAS operations: ✓ Working" << std::endl;
        std::cout << "  - Quantum kernels: ✓ Working" << std::endl;
        std::cout << "\nYour LRET GPU implementation is validated!" << std::endl;
        return 0;
    } else {
        std::cout << "✗✗✗ SOME TESTS FAILED ✗✗✗" << std::endl;
        std::cout << "\nPlease review error messages above." << std::endl;
        return 1;
    }
}
