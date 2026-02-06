#pragma once

/**
 * @file distributed_tensor_scatter.h
 * @brief Distributed Tensor Scattering for Multi-Level MPI Parallelism
 * 
 * Phase 3A of Advanced Row-Parallel Optimization.
 * 
 * BACKGROUND:
 * The existing MPISimulator (in mpi_parallel.h) distributes the L matrix
 * row-wise: each MPI rank owns 2^n / P contiguous rows of L ∈ C^(2^n × r).
 * This works well for n ≤ 16, but for n = 16-24 the communication overhead
 * grows because remote gate operations require exchanging full row slabs.
 * 
 * DISTRIBUTED TENSOR SCATTERING takes a different approach:
 * Instead of distributing rows, we distribute *tensors* (gate Choi matrices,
 * Kraus operators, partial density subblocks) across ranks. This enables:
 * 
 * 1. PER-TENSOR SCATTERING: Scatter individual operator tensors across ranks,
 *    each rank applies its subset locally, then reduce (allreduce) the result.
 *    Avoids the need to exchange L rows for every remote gate.
 * 
 * 2. BROADCAST-SCATTER HYBRID: Broadcast small metadata (tensor dimensions,
 *    sparsity patterns) to all ranks, then scatter only the data payload.
 *    Reduces latency for small-tensor operations.
 * 
 * 3. CONTRACT-AND-REDUCE: Each rank contracts its local tensors with its
 *    local L chunk, then MPI_Allreduce combines partial results.
 *    O(r × local_ops) per rank instead of O(r × total_ops).
 * 
 * 4. MULTI-LEVEL PARALLELISM: MPI across nodes + OpenMP within nodes.
 *    The scatter pattern is topology-aware, placing related tensors
 *    on the same node to minimize inter-node communication.
 * 
 * KEY INSIGHT:
 * For a circuit with d depth and k Kraus operators per noise layer:
 *   - MPISimulator: O(d × 2^n / P) communication per gate
 *   - TensorScatter: O(d × r / P) communication (rank-proportional)
 * Since r << 2^n for LRET, tensor scattering has much lower communication.
 * 
 * USAGE (requires USE_MPI):
 * @code
 * DistributedTensorScatter scatter(MPI_COMM_WORLD);
 * scatter.set_multilevel_mode(true);  // Enable MPI+OpenMP
 * 
 * // Scatter operator tensors
 * scatter.scatter_tensors(kraus_ops, local_ops);
 * 
 * // Contract locally and reduce globally
 * MatrixXcd L_new = scatter.contract_and_reduce(local_ops, local_L);
 * @endcode
 * 
 * Expected Gain: 2-5× better MPI scaling for n=16-24 qubits.
 * 
 * @see ROW_PARALLEL_ADVANCED_OPTIMIZATION_PLAN.md Phase 3A
 * @see mpi_parallel.h for the existing row-wise MPI distribution
 */

#include "types.h"
#include <vector>
#include <memory>
#include <functional>

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace qlret {

//==============================================================================
// Scatter Configuration
//==============================================================================

/**
 * @brief Configuration for distributed tensor scattering
 */
struct ScatterConfig {
    bool verbose = false;                 ///< Print progress/diagnostics
    bool multilevel = false;              ///< Enable MPI + OpenMP multi-level mode
    size_t min_tensors_per_rank = 1;      ///< Minimum tensors assigned per rank
    double load_balance_threshold = 0.1;  ///< Imbalance ratio triggering rebalance
    bool overlap_comm_compute = true;     ///< Overlap MPI communication with computation
    bool topology_aware = true;           ///< Place related tensors on same node
    size_t allreduce_chunk_size = 0;      ///< Chunk size for pipelined allreduce (0=auto)
};

//==============================================================================
// Scatter Pattern (tensor-to-rank assignment)
//==============================================================================

/**
 * @brief Mapping of tensors to MPI ranks with size metadata
 * 
 * Computed by compute_scatter_pattern() based on tensor sizes
 * and available ranks. Attempts to balance total FLOPs per rank.
 */
struct ScatterPattern {
    std::vector<int> tensor_to_rank;      ///< tensor_to_rank[i] = rank owning tensor i
    std::vector<size_t> tensor_sizes;     ///< tensor_sizes[i] = elements in tensor i
    std::vector<size_t> rank_workload;    ///< rank_workload[r] = total elements on rank r
    
    /// Get tensors assigned to a specific rank
    std::vector<size_t> tensors_for_rank(int rank) const {
        std::vector<size_t> indices;
        for (size_t i = 0; i < tensor_to_rank.size(); ++i) {
            if (tensor_to_rank[i] == rank) {
                indices.push_back(i);
            }
        }
        return indices;
    }
    
    /// Compute load imbalance ratio (max/avg - 1)
    double imbalance() const {
        if (rank_workload.empty()) return 0.0;
        size_t total = 0;
        size_t max_load = 0;
        for (size_t w : rank_workload) {
            total += w;
            max_load = std::max(max_load, w);
        }
        double avg = static_cast<double>(total) / rank_workload.size();
        return (avg > 0) ? (static_cast<double>(max_load) / avg - 1.0) : 0.0;
    }
};

//==============================================================================
// Scatter Statistics
//==============================================================================

/**
 * @brief Performance statistics for scatter operations
 */
struct ScatterStats {
    size_t scatter_count = 0;             ///< Number of scatter operations
    size_t reduce_count = 0;              ///< Number of allreduce operations
    size_t broadcast_count = 0;           ///< Number of broadcast operations
    double scatter_time = 0.0;            ///< Total time in scatter (seconds)
    double reduce_time = 0.0;             ///< Total time in allreduce (seconds)
    double compute_time = 0.0;            ///< Total time in local compute (seconds)
    size_t total_bytes_scattered = 0;     ///< Total bytes scattered
    size_t total_bytes_reduced = 0;       ///< Total bytes reduced
    
    double total_time() const { return scatter_time + reduce_time + compute_time; }
    double comm_fraction() const {
        double t = total_time();
        return (t > 0) ? ((scatter_time + reduce_time) / t) : 0.0;
    }
    
    void reset() { *this = ScatterStats{}; }
    void print() const;
};

//==============================================================================
// DistributedTensorScatter Class
//==============================================================================

#ifdef USE_MPI

/**
 * @brief Advanced MPI tensor scattering for multi-level parallelism
 * 
 * Extends the row-wise MPISimulator with tensor-level distribution:
 * - Per-tensor scattering (not just row-slice-based)
 * - Broadcast-scatter hybrid pattern for metadata + data
 * - Contract-and-reduce for efficient global accumulation
 * - Multi-level parallelism (MPI ranks + OpenMP threads)
 * 
 * From CSV #2: "2-5× better load balance, scales to exascale"
 */
class DistributedTensorScatter {
public:
    /**
     * @brief Construct with MPI communicator
     * @param comm MPI communicator (usually MPI_COMM_WORLD)
     * @param config Scatter configuration
     */
    explicit DistributedTensorScatter(MPI_Comm comm, 
                                       const ScatterConfig& config = ScatterConfig{});
    
    ~DistributedTensorScatter();
    
    // Non-copyable, movable
    DistributedTensorScatter(const DistributedTensorScatter&) = delete;
    DistributedTensorScatter& operator=(const DistributedTensorScatter&) = delete;
    DistributedTensorScatter(DistributedTensorScatter&&) noexcept;
    DistributedTensorScatter& operator=(DistributedTensorScatter&&) noexcept;
    
    //--------------------------------------------------------------------------
    // Core Scatter Operations
    //--------------------------------------------------------------------------
    
    /**
     * @brief Scatter individual tensors (matrices) across MPI ranks
     * 
     * Distributes tensors so each rank owns a balanced subset.
     * Uses greedy load-balancing: assign largest unassigned tensor
     * to the rank with the least current workload.
     * 
     * @param tensors   Input tensors (all on root, or each rank sends its own)
     * @param[out] local_tensors  Tensors assigned to this rank
     * @param root      Root rank that holds all input tensors (default 0)
     */
    void scatter_tensors(
        const std::vector<MatrixXcd>& tensors,
        std::vector<MatrixXcd>& local_tensors,
        int root = 0
    );
    
    /**
     * @brief Broadcast-scatter hybrid: broadcast metadata, scatter data
     * 
     * Two-phase distribution:
     * 1. Broadcast: small metadata (dimensions, pattern) to all ranks
     * 2. Scatter: large data payload only to owning rank
     * 
     * Efficient for operators where metadata is needed by all ranks
     * (e.g., to compute local contributions) but data is only
     * applied on the owning rank.
     * 
     * @param L         Full L matrix (only valid on root)
     * @param[out] local_L  Local chunk of L for this rank
     * @param root      Root rank holding the full L
     */
    void broadcast_scatter_hybrid(
        const MatrixXcd& L,
        MatrixXcd& local_L,
        int root = 0
    );
    
    /**
     * @brief Contract local tensors with local L, then allreduce
     * 
     * Each rank:
     * 1. Applies its local tensors to its local L chunk
     * 2. MPI_Allreduce (SUM) to combine partial results
     * 
     * This implements: L_new = Σ_k K_k · L  distributed across ranks.
     * Each rank computes a subset of the K_k terms, then allreduce sums them.
     * 
     * @param local_tensors  Tensors (Kraus operators) assigned to this rank
     * @param local_L        Local L chunk
     * @param num_qubits     Number of qubits
     * @return Reduced L_new (same on all ranks after allreduce)
     */
    MatrixXcd contract_and_reduce(
        const std::vector<MatrixXcd>& local_tensors,
        const MatrixXcd& local_L,
        size_t num_qubits
    );
    
    /**
     * @brief Scatter Kraus operators and apply with reduction
     * 
     * High-level convenience: scatter + contract + reduce in one call.
     * Distributes Kraus operators K_k across ranks, each rank applies
     * its subset to L, then allreduce to get the final L_new = [K_0·L | K_1·L | ...].
     * 
     * @param kraus_ops   All Kraus operators (on root)
     * @param local_L     Local L chunk on this rank
     * @param num_qubits  Number of qubits
     * @param root        Root rank with all Kraus operators
     * @return Concatenated result after scatter+contract+reduce
     */
    MatrixXcd scatter_apply_reduce(
        const std::vector<MatrixXcd>& kraus_ops,
        const MatrixXcd& local_L,
        size_t num_qubits,
        int root = 0
    );
    
    //--------------------------------------------------------------------------
    // Configuration
    //--------------------------------------------------------------------------
    
    /// Enable/disable multi-level parallelism (MPI + OpenMP)
    void set_multilevel_mode(bool enable) { config_.multilevel = enable; }
    
    /// Update configuration
    void set_config(const ScatterConfig& config) { config_ = config; }
    
    /// Get current configuration
    const ScatterConfig& get_config() const { return config_; }
    
    //--------------------------------------------------------------------------
    // Statistics & Diagnostics
    //--------------------------------------------------------------------------
    
    /// Get accumulated statistics
    const ScatterStats& get_stats() const { return stats_; }
    
    /// Reset statistics
    void reset_stats() { stats_.reset(); }
    
    /// Get the last computed scatter pattern
    const ScatterPattern& get_last_pattern() const { return last_pattern_; }
    
    /// Get this rank's MPI rank number
    int rank() const { return rank_; }
    
    /// Get total number of MPI ranks
    int size() const { return size_; }
    
    /// Check if this is the root rank
    bool is_root() const { return rank_ == 0; }

private:
    MPI_Comm comm_;
    int rank_;
    int size_;
    ScatterConfig config_;
    ScatterStats stats_;
    ScatterPattern last_pattern_;
    
    /**
     * @brief Compute optimal scatter pattern for load balancing
     * 
     * Uses greedy longest-processing-time-first (LPT) algorithm:
     * 1. Sort tensors by size (descending)
     * 2. Assign each tensor to the rank with minimum current load
     * 
     * This gives ≤ 4/3 × optimal makespan for uniform sizes.
     * 
     * @param tensors  Tensors to distribute
     * @return ScatterPattern with tensor-to-rank mapping
     */
    ScatterPattern compute_scatter_pattern(
        const std::vector<MatrixXcd>& tensors
    );
    
    /**
     * @brief Coordinate hybrid MPI+OpenMP parallelism
     * 
     * Queries MPI topology for shared-memory regions (MPI_Comm_split_type)
     * and adjusts OpenMP thread count accordingly:
     * - If multiple ranks share a node → reduce threads per rank
     * - If one rank per node → maximize threads
     */
    void coordinate_hybrid_parallelism();
    
    /**
     * @brief Apply a single Kraus operator to L using row-level expansion
     * 
     * For a single-qubit Kraus operator K on qubit q:
     *   L_out[i, :] += Σ_j K[i_q, j_q] * L[i⊕(i_q⊕j_q)·2^q, :]
     * where i_q is the q-th bit of row index i.
     * 
     * @param K          Kraus operator matrix (2×2 for single-qubit)
     * @param L          Input L matrix (local chunk)
     * @param qubit      Target qubit
     * @param num_qubits Number of qubits
     * @return K applied to L
     */
    MatrixXcd apply_kraus_to_L(
        const MatrixXcd& K,
        const MatrixXcd& L,
        size_t qubit,
        size_t num_qubits
    );
};

#else  // USE_MPI not defined

/**
 * @brief Stub class when MPI is not available
 * 
 * Provides the same interface but operates on a single process.
 * scatter_tensors simply copies all tensors to local_tensors.
 * contract_and_reduce applies all tensors locally.
 */
class DistributedTensorScatter {
public:
    explicit DistributedTensorScatter(const ScatterConfig& config = ScatterConfig{})
        : config_(config) {}
    
    ~DistributedTensorScatter() = default;
    
    void scatter_tensors(
        const std::vector<MatrixXcd>& tensors,
        std::vector<MatrixXcd>& local_tensors,
        int root = 0
    ) {
        (void)root;
        local_tensors = tensors;  // Single process owns everything
    }
    
    void broadcast_scatter_hybrid(
        const MatrixXcd& L,
        MatrixXcd& local_L,
        int root = 0
    ) {
        (void)root;
        local_L = L;  // Single process owns everything
    }
    
    MatrixXcd contract_and_reduce(
        const std::vector<MatrixXcd>& local_tensors,
        const MatrixXcd& local_L,
        size_t num_qubits
    );
    
    MatrixXcd scatter_apply_reduce(
        const std::vector<MatrixXcd>& kraus_ops,
        const MatrixXcd& local_L,
        size_t num_qubits,
        int root = 0
    ) {
        (void)root;
        // Single process: just apply all Kraus ops locally
        return contract_and_reduce(kraus_ops, local_L, num_qubits);
    }
    
    void set_multilevel_mode(bool enable) { config_.multilevel = enable; }
    void set_config(const ScatterConfig& config) { config_ = config; }
    const ScatterConfig& get_config() const { return config_; }
    const ScatterStats& get_stats() const { return stats_; }
    void reset_stats() { stats_.reset(); }
    const ScatterPattern& get_last_pattern() const { return last_pattern_; }
    int rank() const { return 0; }
    int size() const { return 1; }
    bool is_root() const { return true; }

private:
    ScatterConfig config_;
    ScatterStats stats_;
    ScatterPattern last_pattern_;
};

#endif  // USE_MPI

//==============================================================================
// Free Functions
//==============================================================================

/**
 * @brief Apply distributed tensor scattering to a noise channel
 * 
 * High-level entry point: takes a NoiseOp and L, distributes the
 * Kraus operators across MPI ranks (if available), applies them,
 * and returns the updated L.
 * 
 * Falls back to single-process application when MPI is not available.
 * 
 * @param L           Current L factor
 * @param noise       Noise operation with Kraus operators
 * @param num_qubits  Number of qubits
 * @param config      Scatter configuration
 * @return Updated L after noise application
 */
MatrixXcd apply_noise_distributed(
    const MatrixXcd& L,
    const NoiseOp& noise,
    size_t num_qubits,
    const ScatterConfig& config = ScatterConfig{}
);

}  // namespace qlret
