/**
 * @file advanced_optimizations.h
 * @brief Phase 3 Advanced Optimizations for Row Parallelism
 * 
 * Implements three key optimizations from ROW_PARALLELISM_OPTIMIZATION_STRATEGY.md:
 * 
 * 1. Cholesky QR Orthonormalization (2.5× faster truncation)
 *    - Uses Cholesky decomposition for row-parallel orthonormalization
 *    - Optimal when rank < 64 and L is well-conditioned
 * 
 * 2. Qubit Reordering (1.8× for QNN circuits)
 *    - Reorders qubits so most-used qubits → lowest indices
 *    - Improves cache locality for gate application
 * 
 * 3. Community Detection Batching (2× for random circuits)
 *    - Groups rows by gate connectivity patterns
 *    - Processes communities in parallel for better load balance
 * 
 * Expected cumulative gain: 2.5× from baseline (with Phases 1-2)
 */

#pragma once

#include "types.h"
#include <vector>
#include <numeric>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace qlret {

//==============================================================================
// Configuration
//==============================================================================

/**
 * @brief Configuration for advanced optimizations
 */
struct AdvancedOptConfig {
    // Cholesky QR settings
    bool enable_cholesky_qr = true;      // Use Cholesky QR for orthonormalization
    size_t cholesky_max_rank = 64;       // Max rank for Cholesky QR (larger → use HouseholderQR)
    double condition_threshold = 1e10;    // Condition number threshold for fallback
    
    // Qubit reordering settings
    bool enable_qubit_reordering = true;  // Reorder qubits for cache locality
    size_t min_qubits_for_reorder = 8;    // Min qubits to enable reordering
    double usage_imbalance_threshold = 2.0;  // Min usage ratio to trigger reordering
    
    // Community batching settings
    bool enable_community_batching = false;  // Community detection (experimental)
    size_t min_community_size = 64;          // Min rows per community
    size_t max_communities = 16;             // Max number of communities
    
    // Verbose logging
    bool verbose = false;
};

//==============================================================================
// Cholesky QR Orthonormalization
//==============================================================================

/**
 * @brief Orthonormalize L matrix using row-parallel Cholesky QR
 * 
 * Algorithm:
 * 1. Compute Gram matrix G = L† L (rank × rank)
 * 2. Cholesky decomposition: G = R† R (R upper triangular)
 * 3. Compute R⁻¹ (rank × rank triangular inverse)
 * 4. Q = L × R⁻¹ (row-parallel multiplication)
 * 
 * Advantages over HouseholderQR:
 * - Row-parallel: each row of Q computed independently
 * - 2-3× faster for dim >> rank
 * - Better cache locality
 * 
 * Limitations:
 * - Requires G to be positive definite (well-conditioned L)
 * - Less numerically stable for ill-conditioned matrices
 * 
 * @param L Input matrix (dim × rank)
 * @return Q matrix with orthonormal columns (dim × rank)
 */
MatrixXcd orthonormalize_cholesky_qr(const MatrixXcd& L);

/**
 * @brief Orthonormalize with fallback to HouseholderQR if Cholesky fails
 * 
 * @param L Input matrix
 * @param config Optimization configuration
 * @return Orthonormalized matrix
 */
MatrixXcd orthonormalize_adaptive(const MatrixXcd& L, const AdvancedOptConfig& config);

/**
 * @brief Check if matrix is suitable for Cholesky QR
 * 
 * Checks:
 * - Rank is within threshold
 * - Estimated condition number is acceptable
 * 
 * @param L Input matrix
 * @param config Configuration with thresholds
 * @return True if Cholesky QR is suitable
 */
bool is_cholesky_suitable(const MatrixXcd& L, const AdvancedOptConfig& config);

//==============================================================================
// Qubit Reordering
//==============================================================================

/**
 * @brief Tracks qubit usage to determine optimal ordering
 * 
 * Most-used qubits are mapped to lowest indices for better cache locality.
 * Low-indexed qubits have small stride (2^t), keeping row pairs in L2 cache.
 */
class QubitUsageTracker {
public:
    /**
     * @brief Constructor
     * @param num_qubits Number of qubits in the system
     */
    explicit QubitUsageTracker(size_t num_qubits);
    
    /**
     * @brief Record gate usage
     * @param gate Gate operation to record
     */
    void record_gate(const GateOp& gate);
    
    /**
     * @brief Record noise operation
     * @param noise Noise operation to record
     */
    void record_noise(const NoiseOp& noise);
    
    /**
     * @brief Analyze entire quantum sequence
     * @param sequence Quantum sequence to analyze
     */
    void analyze_sequence(const QuantumSequence& sequence);
    
    /**
     * @brief Get optimal qubit permutation
     * 
     * Returns permutation where perm[logical] = physical.
     * Most-used logical qubits map to lowest physical indices.
     * 
     * @return Permutation vector
     */
    std::vector<size_t> get_optimal_permutation() const;
    
    /**
     * @brief Get inverse permutation
     * @param perm Forward permutation
     * @return Inverse permutation where inv[physical] = logical
     */
    static std::vector<size_t> invert_permutation(const std::vector<size_t>& perm);
    
    /**
     * @brief Check if reordering would be beneficial
     * 
     * Reordering is beneficial if there's significant usage imbalance
     * between qubits.
     * 
     * @param threshold Minimum usage ratio (max/min) to trigger reordering
     * @return True if reordering recommended
     */
    bool is_reordering_beneficial(double threshold = 2.0) const;
    
    /**
     * @brief Get gate counts per qubit
     * @return Vector of gate counts
     */
    const std::vector<size_t>& get_gate_counts() const { return gate_counts_; }
    
    /**
     * @brief Reset tracker
     */
    void reset();

private:
    size_t num_qubits_;
    std::vector<size_t> gate_counts_;  // gate_counts_[q] = # operations on qubit q
};

/**
 * @brief Permute L matrix rows according to qubit reordering
 * 
 * For each row index i (binary representation corresponds to qubit states),
 * compute new index by permuting bits according to qubit permutation.
 * 
 * @param L Input L matrix
 * @param perm Qubit permutation (perm[logical] = physical)
 * @param num_qubits Number of qubits
 * @return Permuted L matrix
 */
MatrixXcd permute_L_qubits(const MatrixXcd& L, 
                           const std::vector<size_t>& perm,
                           size_t num_qubits);

/**
 * @brief Permute gate operation according to qubit reordering
 * 
 * Updates qubit indices in gate to match new ordering.
 * 
 * @param gate Original gate
 * @param perm Qubit permutation
 * @return Gate with permuted qubit indices
 */
GateOp permute_gate(const GateOp& gate, const std::vector<size_t>& perm);

/**
 * @brief Permute noise operation according to qubit reordering
 * @param noise Original noise operation
 * @param perm Qubit permutation
 * @return Noise operation with permuted qubit indices
 */
NoiseOp permute_noise(const NoiseOp& noise, const std::vector<size_t>& perm);

/**
 * @brief Permute entire quantum sequence
 * @param sequence Original sequence
 * @param perm Qubit permutation
 * @return Sequence with permuted operations
 */
QuantumSequence permute_sequence(const QuantumSequence& sequence,
                                  const std::vector<size_t>& perm);

//==============================================================================
// Community Detection Batching
//==============================================================================

/**
 * @brief Represents a community of rows with similar connectivity patterns
 */
struct RowCommunity {
    std::vector<size_t> row_indices;  // Indices of rows in this community
    std::unordered_set<size_t> affected_qubits;  // Qubits affecting this community
    size_t size() const { return row_indices.size(); }
};

/**
 * @brief Detects communities of rows based on gate connectivity
 * 
 * Rows that are frequently accessed together by gates form communities.
 * Processing communities in parallel improves cache locality.
 */
class CommunityDetector {
public:
    /**
     * @brief Constructor
     * @param num_qubits Number of qubits
     * @param config Configuration
     */
    CommunityDetector(size_t num_qubits, const AdvancedOptConfig& config = {});
    
    /**
     * @brief Analyze gate connectivity patterns
     * @param gates Vector of gate operations
     */
    void analyze_gates(const std::vector<GateOp>& gates);
    
    /**
     * @brief Detect row communities
     * 
     * Uses a simple partitioning based on high-order qubit bits.
     * Rows with same high bits tend to be accessed together.
     * 
     * @return Vector of detected communities
     */
    std::vector<RowCommunity> detect_communities() const;
    
    /**
     * @brief Get community assignment for each row
     * @param num_communities Number of communities to create
     * @return Vector where result[row] = community_id
     */
    std::vector<size_t> get_community_assignment(size_t num_communities) const;

private:
    size_t num_qubits_;
    size_t dim_;
    AdvancedOptConfig config_;
    std::unordered_map<size_t, size_t> qubit_gate_counts_;
};

//==============================================================================
// Optimized Simulation with Advanced Features
//==============================================================================

/**
 * @brief Run simulation with all Phase 3 optimizations
 * 
 * Combines:
 * - Cholesky QR for fast truncation
 * - Qubit reordering for cache locality
 * - Community batching for load balance
 * 
 * @param L_init Initial L matrix
 * @param sequence Quantum sequence
 * @param num_qubits Number of qubits
 * @param config Simulation configuration
 * @param opt_config Advanced optimization configuration
 * @return Final L matrix
 */
MatrixXcd run_with_advanced_optimizations(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    const SimConfig& config,
    const AdvancedOptConfig& opt_config = {}
);

/**
 * @brief Enhanced truncation with Cholesky QR
 * 
 * Uses Cholesky QR for orthonormalization when rank < threshold,
 * providing 2-3× speedup over HouseholderQR.
 * 
 * @param L Input L matrix
 * @param threshold Eigenvalue threshold for truncation
 * @param max_rank Maximum rank (0 = no limit)
 * @param use_cholesky Whether to use Cholesky QR
 * @return Truncated L matrix
 */
MatrixXcd truncate_L_enhanced(
    const MatrixXcd& L,
    double threshold,
    size_t max_rank = 0,
    bool use_cholesky = true
);

//==============================================================================
// Statistics and Diagnostics
//==============================================================================

/**
 * @brief Statistics for advanced optimizations
 */
struct AdvancedOptStats {
    // Cholesky QR stats
    size_t cholesky_calls = 0;
    size_t cholesky_fallbacks = 0;  // Times fell back to HouseholderQR
    double cholesky_total_time_ms = 0.0;
    
    // Qubit reordering stats
    size_t reorder_applied = 0;
    size_t reorder_skipped = 0;
    double reorder_speedup_estimate = 1.0;
    
    // Community batching stats
    size_t communities_created = 0;
    double community_balance = 0.0;  // Std dev / mean of community sizes
    
    void reset() {
        cholesky_calls = 0;
        cholesky_fallbacks = 0;
        cholesky_total_time_ms = 0.0;
        reorder_applied = 0;
        reorder_skipped = 0;
        reorder_speedup_estimate = 1.0;
        communities_created = 0;
        community_balance = 0.0;
    }
};

/**
 * @brief Get global statistics for advanced optimizations
 * @return Reference to statistics struct
 */
AdvancedOptStats& get_advanced_opt_stats();

}  // namespace qlret
