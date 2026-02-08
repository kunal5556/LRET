/**
 * @file distributed_tensor_scatter.cpp
 * @brief Implementation of Distributed Tensor Scattering (Phase 3A)
 * 
 * Provides MPI-based tensor distribution for multi-level parallelism.
 * When USE_MPI is not defined, provides single-process fallback.
 * 
 * @see distributed_tensor_scatter.h for API documentation
 * @see ROW_PARALLEL_ADVANCED_OPTIMIZATION_PLAN.md Phase 3A
 */

#include "distributed_tensor_scatter.h"
#include "gates_and_noise.h"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cassert>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace qlret {

//==============================================================================
// ScatterStats
//==============================================================================

void ScatterStats::print() const {
    std::cout << "=== Distributed Tensor Scatter Statistics ===" << std::endl;
    std::cout << "  Scatter operations:  " << scatter_count << std::endl;
    std::cout << "  Reduce operations:   " << reduce_count << std::endl;
    std::cout << "  Broadcast operations:" << broadcast_count << std::endl;
    std::cout << "  Scatter time:        " << std::fixed << std::setprecision(4) 
              << scatter_time << " s" << std::endl;
    std::cout << "  Reduce time:         " << reduce_time << " s" << std::endl;
    std::cout << "  Compute time:        " << compute_time << " s" << std::endl;
    std::cout << "  Total time:          " << total_time() << " s" << std::endl;
    std::cout << "  Comm fraction:       " << std::setprecision(1) 
              << (comm_fraction() * 100.0) << "%" << std::endl;
    std::cout << "  Bytes scattered:     " << total_bytes_scattered << std::endl;
    std::cout << "  Bytes reduced:       " << total_bytes_reduced << std::endl;
    std::cout << "==============================================" << std::endl;
}

//==============================================================================
// MPI Implementation
//==============================================================================

#ifdef USE_MPI

// MS-MPI (Windows) does not reliably support MPI_CXX_DOUBLE_COMPLEX or
// MPI_C_DOUBLE_COMPLEX in collective operations. Use MPI_DOUBLE with
// doubled element counts instead (std::complex<double> == double[2]).
#ifndef LRET_MPI_COMPLEX_COUNT
#define LRET_MPI_COMPLEX_COUNT(n) ((n) * 2)
#define LRET_MPI_COMPLEX_TYPE MPI_DOUBLE
#endif

DistributedTensorScatter::DistributedTensorScatter(MPI_Comm comm, 
                                                     const ScatterConfig& config)
    : comm_(comm), config_(config) {
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);
    
    if (config_.multilevel) {
        coordinate_hybrid_parallelism();
    }
    
    if (config_.verbose && rank_ == 0) {
        std::cout << "[TensorScatter] Initialized with " << size_ << " MPI ranks"
                  << (config_.multilevel ? " (multi-level mode)" : "")
                  << std::endl;
    }
}

DistributedTensorScatter::~DistributedTensorScatter() = default;

DistributedTensorScatter::DistributedTensorScatter(DistributedTensorScatter&&) noexcept = default;
DistributedTensorScatter& DistributedTensorScatter::operator=(DistributedTensorScatter&&) noexcept = default;

//------------------------------------------------------------------------------
// scatter_tensors: distribute tensors across ranks using LPT scheduling
//------------------------------------------------------------------------------

void DistributedTensorScatter::scatter_tensors(
    const std::vector<MatrixXcd>& tensors,
    std::vector<MatrixXcd>& local_tensors,
    int root
) {
    auto t_start = std::chrono::steady_clock::now();
    
    // Step 1: Root computes scatter pattern
    int num_tensors = 0;
    ScatterPattern pattern;
    
    if (rank_ == root) {
        num_tensors = static_cast<int>(tensors.size());
        pattern = compute_scatter_pattern(tensors);
    }
    
    // Step 2: Broadcast number of tensors and assignment
    MPI_Bcast(&num_tensors, 1, MPI_INT, root, comm_);
    
    if (num_tensors == 0) {
        local_tensors.clear();
        return;
    }
    
    // Broadcast tensor-to-rank assignment
    if (rank_ != root) {
        pattern.tensor_to_rank.resize(num_tensors);
    }
    MPI_Bcast(pattern.tensor_to_rank.data(), num_tensors, MPI_INT, root, comm_);
    
    // Step 3: Broadcast tensor dimensions (rows, cols for each tensor)
    std::vector<int> tensor_dims(num_tensors * 2);  // [rows0, cols0, rows1, cols1, ...]
    if (rank_ == root) {
        for (int i = 0; i < num_tensors; ++i) {
            tensor_dims[2 * i]     = static_cast<int>(tensors[i].rows());
            tensor_dims[2 * i + 1] = static_cast<int>(tensors[i].cols());
        }
    }
    MPI_Bcast(tensor_dims.data(), num_tensors * 2, MPI_INT, root, comm_);
    stats_.broadcast_count++;
    
    // Step 4: Send each tensor to its assigned rank
    local_tensors.clear();
    
    for (int i = 0; i < num_tensors; ++i) {
        int target_rank = pattern.tensor_to_rank[i];
        int rows = tensor_dims[2 * i];
        int cols = tensor_dims[2 * i + 1];
        int elems = rows * cols;
        
        if (target_rank == root && rank_ == root) {
            // Tensor stays on root
            local_tensors.push_back(tensors[i]);
        } else if (rank_ == root) {
            // Root sends to target
            MPI_Send(tensors[i].data(), LRET_MPI_COMPLEX_COUNT(elems), LRET_MPI_COMPLEX_TYPE,
                     target_rank, i /*tag*/, comm_);
            stats_.total_bytes_scattered += elems * sizeof(Complex);
        } else if (rank_ == target_rank) {
            // This rank receives
            MatrixXcd tensor(rows, cols);
            MPI_Recv(tensor.data(), LRET_MPI_COMPLEX_COUNT(elems), LRET_MPI_COMPLEX_TYPE,
                     root, i /*tag*/, comm_, MPI_STATUS_IGNORE);
            local_tensors.push_back(std::move(tensor));
        }
    }
    
    last_pattern_ = pattern;
    stats_.scatter_count++;
    stats_.scatter_time += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
}

//------------------------------------------------------------------------------
// broadcast_scatter_hybrid: broadcast metadata, scatter L rows
//------------------------------------------------------------------------------

void DistributedTensorScatter::broadcast_scatter_hybrid(
    const MatrixXcd& L,
    MatrixXcd& local_L,
    int root
) {
    auto t_start = std::chrono::steady_clock::now();
    
    // Phase 1: Broadcast dimensions
    int dims[2] = {0, 0};
    if (rank_ == root) {
        dims[0] = static_cast<int>(L.rows());
        dims[1] = static_cast<int>(L.cols());
    }
    MPI_Bcast(dims, 2, MPI_INT, root, comm_);
    stats_.broadcast_count++;
    
    int global_rows = dims[0];
    int cols = dims[1];
    
    // Phase 2: Broadcast the full L matrix to all ranks
    // (Eigen is column-major, so direct MPI_Scatterv of rows would
    //  scatter non-contiguous data. Broadcasting full L is simpler
    //  and still efficient for moderate sizes.)
    MatrixXcd L_full(global_rows, cols);
    if (rank_ == root) {
        L_full = L;
    }
    MPI_Bcast(
        L_full.data(),
        LRET_MPI_COMPLEX_COUNT(global_rows * cols),
        LRET_MPI_COMPLEX_TYPE,
        root,
        comm_
    );
    
    // Phase 3: Each rank extracts its row slab
    int base_rows = global_rows / size_;
    int remainder = global_rows % size_;
    int local_rows = base_rows + (rank_ < remainder ? 1 : 0);
    int local_start = rank_ * base_rows + std::min(rank_, remainder);
    
    local_L = L_full.middleRows(local_start, local_rows);
    
    stats_.scatter_count++;
    stats_.total_bytes_scattered += local_rows * cols * sizeof(Complex);
    stats_.scatter_time += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
}

//------------------------------------------------------------------------------
// contract_and_reduce: local tensor contraction + MPI_Allreduce
//------------------------------------------------------------------------------

MatrixXcd DistributedTensorScatter::contract_and_reduce(
    const std::vector<MatrixXcd>& local_tensors,
    const MatrixXcd& local_L,
    size_t num_qubits,
    size_t target_qubit
) {
    auto t_compute_start = std::chrono::steady_clock::now();
    
    size_t dim = local_L.rows();
    size_t rank = local_L.cols();
    size_t num_local = local_tensors.size();
    
    // Local contraction: apply each local Kraus operator to L
    MatrixXcd local_result;
    
    if (num_local == 0) {
        local_result = MatrixXcd::Zero(dim, 0);
    } else {
        local_result.resize(dim, rank * num_local);
        
        #ifdef USE_OPENMP
        if (config_.multilevel) {
            #pragma omp parallel for schedule(dynamic)
            for (int k = 0; k < static_cast<int>(num_local); ++k) {
                MatrixXcd L_k = apply_kraus_to_L(
                    local_tensors[k], local_L, target_qubit, num_qubits
                );
                local_result.block(0, k * rank, dim, rank) = L_k;
            }
        } else
        #endif
        {
            for (size_t k = 0; k < num_local; ++k) {
                MatrixXcd L_k = apply_kraus_to_L(
                    local_tensors[k], local_L, target_qubit, num_qubits
                );
                local_result.block(0, k * rank, dim, rank) = L_k;
            }
        }
    }
    
    stats_.compute_time += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_compute_start).count();
    
    // Allgatherv: each rank contributes its local_result columns
    auto t_reduce_start = std::chrono::steady_clock::now();
    
    int local_cols = static_cast<int>(local_result.cols());
    
    // Gather all local column counts
    std::vector<int> all_cols(size_);
    MPI_Allgather(&local_cols, 1, MPI_INT, all_cols.data(), 1, MPI_INT, comm_);
    
    // Compute total columns and displacements (in elements, column-major)
    int total_cols = 0;
    std::vector<int> recv_counts(size_);
    std::vector<int> recv_displs(size_);
    for (int r = 0; r < size_; ++r) {
        recv_counts[r] = static_cast<int>(dim) * all_cols[r];
        recv_displs[r] = static_cast<int>(dim) * total_cols;
        total_cols += all_cols[r];
    }
    
    MatrixXcd global_result(dim, total_cols);
    
    // Double counts/displacements for MPI_DOUBLE encoding of complex
    std::vector<int> recv_counts_d(size_), recv_displs_d(size_);
    for (int r = 0; r < size_; ++r) {
        recv_counts_d[r] = LRET_MPI_COMPLEX_COUNT(recv_counts[r]);
        recv_displs_d[r] = LRET_MPI_COMPLEX_COUNT(recv_displs[r]);
    }
    MPI_Allgatherv(
        local_result.data(),
        LRET_MPI_COMPLEX_COUNT(static_cast<int>(dim) * local_cols),
        LRET_MPI_COMPLEX_TYPE,
        global_result.data(),
        recv_counts_d.data(),
        recv_displs_d.data(),
        LRET_MPI_COMPLEX_TYPE,
        comm_
    );
    
    stats_.reduce_count++;
    stats_.total_bytes_reduced += total_cols * dim * sizeof(Complex);
    stats_.reduce_time += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_reduce_start).count();
    
    return global_result;
}

//------------------------------------------------------------------------------
// scatter_apply_reduce: high-level convenience for Kraus distribution
//------------------------------------------------------------------------------

MatrixXcd DistributedTensorScatter::scatter_apply_reduce(
    const std::vector<MatrixXcd>& kraus_ops,
    const MatrixXcd& local_L,
    size_t num_qubits,
    int root,
    size_t target_qubit
) {
    // Step 1: Scatter Kraus operators across ranks
    std::vector<MatrixXcd> local_kraus;
    scatter_tensors(kraus_ops, local_kraus, root);
    
    // Step 2: Contract locally and reduce globally
    return contract_and_reduce(local_kraus, local_L, num_qubits, target_qubit);
}

//------------------------------------------------------------------------------
// compute_scatter_pattern: LPT scheduling for load balance
//------------------------------------------------------------------------------

ScatterPattern DistributedTensorScatter::compute_scatter_pattern(
    const std::vector<MatrixXcd>& tensors
) {
    ScatterPattern pattern;
    size_t n = tensors.size();
    
    pattern.tensor_to_rank.resize(n);
    pattern.tensor_sizes.resize(n);
    pattern.rank_workload.resize(size_, 0);
    
    // Compute sizes
    for (size_t i = 0; i < n; ++i) {
        pattern.tensor_sizes[i] = tensors[i].rows() * tensors[i].cols();
    }
    
    // Sort indices by size (descending) for LPT scheduling
    std::vector<size_t> sorted_indices(n);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [&](size_t a, size_t b) {
                  return pattern.tensor_sizes[a] > pattern.tensor_sizes[b];
              });
    
    // Greedy LPT: assign each tensor (largest first) to least-loaded rank
    for (size_t idx : sorted_indices) {
        // Find rank with minimum current workload
        int min_rank = 0;
        size_t min_load = pattern.rank_workload[0];
        for (int r = 1; r < size_; ++r) {
            if (pattern.rank_workload[r] < min_load) {
                min_load = pattern.rank_workload[r];
                min_rank = r;
            }
        }
        
        pattern.tensor_to_rank[idx] = min_rank;
        pattern.rank_workload[min_rank] += pattern.tensor_sizes[idx];
    }
    
    if (config_.verbose && rank_ == 0) {
        std::cout << "[TensorScatter] Scatter pattern for " << n << " tensors:"
                  << std::endl;
        for (int r = 0; r < size_; ++r) {
            auto indices = pattern.tensors_for_rank(r);
            std::cout << "  Rank " << r << ": " << indices.size() << " tensors, "
                      << pattern.rank_workload[r] << " elements" << std::endl;
        }
        std::cout << "  Load imbalance: " << std::fixed << std::setprecision(2)
                  << (pattern.imbalance() * 100.0) << "%" << std::endl;
    }
    
    return pattern;
}

//------------------------------------------------------------------------------
// coordinate_hybrid_parallelism: MPI + OpenMP coordination
//------------------------------------------------------------------------------

void DistributedTensorScatter::coordinate_hybrid_parallelism() {
    // Query shared-memory topology using MPI_Comm_split_type
    MPI_Comm node_comm;
    MPI_Comm_split_type(comm_, MPI_COMM_TYPE_SHARED, rank_, MPI_INFO_NULL, &node_comm);
    
    int node_size = 1;
    MPI_Comm_size(node_comm, &node_size);
    MPI_Comm_free(&node_comm);
    
    // Adjust OpenMP threads: if multiple ranks share a node, reduce threads
    #ifdef USE_OPENMP
    int total_cores = omp_get_max_threads();
    int threads_per_rank = std::max(1, total_cores / node_size);
    omp_set_num_threads(threads_per_rank);
    
    if (config_.verbose && rank_ == 0) {
        std::cout << "[TensorScatter] Multi-level: " << node_size 
                  << " ranks/node, " << threads_per_rank 
                  << " OpenMP threads/rank" << std::endl;
    }
    #endif
}

//------------------------------------------------------------------------------
// apply_kraus_to_L: single Kraus operator application
//------------------------------------------------------------------------------

MatrixXcd DistributedTensorScatter::apply_kraus_to_L(
    const MatrixXcd& K,
    const MatrixXcd& L,
    size_t qubit,
    size_t num_qubits
) {
    // Delegate to the existing optimized single-gate/two-qubit application
    // Kraus operators have the same mathematical structure as gate matrices
    size_t k_dim = K.rows();
    
    if (k_dim == 2) {
        // Single-qubit Kraus operator: use apply_single_gate_direct
        return apply_single_gate_direct(L, K, qubit, num_qubits);
    } else if (k_dim == 4) {
        // Two-qubit Kraus operator: use apply_two_qubit_gate_direct
        // For two-qubit, assume qubits are (qubit, qubit+1)
        Matrix4cd K4;
        K4 << K(0,0), K(0,1), K(0,2), K(0,3),
              K(1,0), K(1,1), K(1,2), K(1,3),
              K(2,0), K(2,1), K(2,2), K(2,3),
              K(3,0), K(3,1), K(3,2), K(3,3);
        return apply_two_qubit_gate_direct(L, K4, qubit, qubit + 1, num_qubits);
    } else {
        // General case: full matrix-vector product (rare)
        // K acts on the full space, just do L_out = K * L
        return K * L;
    }
}

#else  // USE_MPI not defined

//==============================================================================
// Non-MPI Fallback: contract_and_reduce (single process)
//==============================================================================

MatrixXcd DistributedTensorScatter::contract_and_reduce(
    const std::vector<MatrixXcd>& local_tensors,
    const MatrixXcd& local_L,
    size_t num_qubits,
    size_t target_qubit
) {
    size_t dim = local_L.rows();
    size_t rank = local_L.cols();
    size_t num_tensors = local_tensors.size();
    
    if (num_tensors == 0) {
        return local_L;
    }
    
    // Single process: apply all Kraus operators and concatenate
    MatrixXcd result(dim, rank * num_tensors);
    
    for (size_t k = 0; k < num_tensors; ++k) {
        const MatrixXcd& K = local_tensors[k];
        size_t k_dim = K.rows();
        
        MatrixXcd L_k;
        if (k_dim == 2) {
            L_k = apply_single_gate_direct(local_L, K, target_qubit, num_qubits);
        } else if (k_dim == 4) {
            Matrix4cd K4;
            K4 << K(0,0), K(0,1), K(0,2), K(0,3),
                  K(1,0), K(1,1), K(1,2), K(1,3),
                  K(2,0), K(2,1), K(2,2), K(2,3),
                  K(3,0), K(3,1), K(3,2), K(3,3);
            L_k = apply_two_qubit_gate_direct(local_L, K4, 0, 1, num_qubits);
        } else {
            L_k = K * local_L;
        }
        
        result.block(0, k * rank, dim, rank) = L_k;
    }
    
    return result;
}

#endif  // USE_MPI

//==============================================================================
// Free function: apply_noise_distributed
//==============================================================================

MatrixXcd apply_noise_distributed(
    const MatrixXcd& L,
    const NoiseOp& noise,
    size_t num_qubits,
    const ScatterConfig& config
) {
    // Get Kraus operators for this noise channel
    std::vector<MatrixXcd> kraus_ops;
    if (noise.type == NoiseType::CUSTOM && !noise.custom_kraus.empty()) {
        kraus_ops = noise.custom_kraus;
    } else {
        kraus_ops = get_noise_kraus_operators(noise.type, noise.probability, noise.params);
    }
    
    if (kraus_ops.empty()) {
        return L;  // Identity channel
    }
    
    #ifdef USE_MPI
    // Check if MPI is initialized
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    
    if (mpi_initialized) {
        int world_size = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        
        // Only use distributed scatter if we have multiple ranks
        // AND enough Kraus operators to distribute
        if (world_size > 1 && kraus_ops.size() >= static_cast<size_t>(world_size)) {
            DistributedTensorScatter scatter(MPI_COMM_WORLD, config);
            size_t target_q = noise.qubits.empty() ? 0 : noise.qubits[0];
            return scatter.scatter_apply_reduce(kraus_ops, L, num_qubits, 0, target_q);
        }
    }
    #endif
    
    // Fallback: apply all Kraus operators locally (standard path)
    return apply_noise_to_L(L, noise, num_qubits);
}

}  // namespace qlret
