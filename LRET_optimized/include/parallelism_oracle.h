/**
 * @file parallelism_oracle.h
 * @brief Parallelism Oracle for Runtime Mode Selection
 * 
 * Phase 2 of Row Parallelism Optimization (ROW_PARALLELISM_OPTIMIZATION_STRATEGY.md).
 * 
 * The oracle makes runtime decisions between row vs column parallelism based on:
 * - Current rank of the L matrix (low rank → row parallelism)
 * - Target qubit indices (low qubits → row parallelism for cache locality)
 * - Cache size considerations (stride fits in L2 → row parallelism)
 * 
 * Expected gain: +20% (1.8× total from baseline) when combined with Phase 1.
 * 
 * Key insight from MPS research: Row-parallel mode is more cache-efficient
 * for LRET's typical low-rank scenarios (r=16-64), avoiding L2 cache thrashing
 * through better spatial locality.
 */

#pragma once

#include "types.h"
#include "cli_parser.h"
#include <vector>
#include <algorithm>
#include <cstddef>

namespace qlret {

//==============================================================================
// Oracle Configuration
//==============================================================================

/**
 * @brief Configuration parameters for the Parallelism Oracle
 * 
 * These thresholds can be tuned based on hardware characteristics:
 * - L2 cache size
 * - Number of cores
 * - Memory bandwidth
 */
struct OracleConfig {
    // Rank threshold: below this → row parallelism preferred
    // Based on analysis: rank < 32 → each row fits in L2 cache
    size_t row_rank_threshold = 32;
    
    // Rank threshold: above this → column parallelism preferred
    // Column parallelism excels when rank > 64 AND dimension is large
    size_t column_rank_threshold = 64;
    
    // Qubit threshold: gates on qubits 0-(max_qubit_for_row-1) prefer row mode
    // Low qubit indices → small stride (2^t) → cache-line friendly
    size_t max_qubit_for_row = 5;  // stride ≤ 32 rows
    
    // Minimum dimension for any parallelism (OpenMP overhead threshold)
    size_t min_dim_for_parallel = 256;  // 2^8, so n >= 8 qubits
    
    // L2 cache size in bytes (used for stride analysis)
    // Default: 256 KB (typical per-core L2 cache)
    size_t l2_cache_bytes = 256 * 1024;
    
    // Bytes per complex<double> element
    static constexpr size_t BYTES_PER_ELEMENT = 16;  // 2 * sizeof(double)
    
    // Whether to enable verbose logging of oracle decisions
    bool verbose = false;
    
    /**
     * @brief Default constructor with sensible defaults
     */
    OracleConfig() = default;
    
    /**
     * @brief Constructor with custom L2 cache size
     * @param cache_kb L2 cache size in KB
     */
    explicit OracleConfig(size_t cache_kb)
        : l2_cache_bytes(cache_kb * 1024) {}
};

//==============================================================================
// Oracle Statistics
//==============================================================================

/**
 * @brief Statistics collected by the oracle for analysis
 */
struct OracleStats {
    size_t total_decisions = 0;
    size_t row_decisions = 0;
    size_t column_decisions = 0;
    size_t sequential_decisions = 0;
    
    // Reason breakdown
    size_t row_by_rank = 0;          // Decided ROW because rank < threshold
    size_t row_by_qubit = 0;         // Decided ROW because max_qubit < threshold
    size_t row_by_cache = 0;         // Decided ROW because stride fits in L2
    size_t column_by_rank = 0;       // Decided COLUMN because rank > threshold
    size_t sequential_by_dim = 0;    // Decided SEQUENTIAL because dim too small
    
    void reset() {
        total_decisions = 0;
        row_decisions = 0;
        column_decisions = 0;
        sequential_decisions = 0;
        row_by_rank = 0;
        row_by_qubit = 0;
        row_by_cache = 0;
        column_by_rank = 0;
        sequential_by_dim = 0;
    }
    
    double row_percentage() const {
        return total_decisions > 0 
            ? 100.0 * row_decisions / total_decisions 
            : 0.0;
    }
    
    double column_percentage() const {
        return total_decisions > 0 
            ? 100.0 * column_decisions / total_decisions 
            : 0.0;
    }
};

//==============================================================================
// Oracle Decision Reasons (for logging/debugging)
//==============================================================================

enum class OracleReason {
    SMALL_DIM,              // Dimension too small → SEQUENTIAL
    LOW_RANK,               // rank < row_rank_threshold → ROW
    LOW_QUBIT_INDEX,        // max(gate.qubits) < max_qubit_for_row → ROW
    STRIDE_FITS_CACHE,      // row stride fits in L2 cache → ROW
    HIGH_RANK,              // rank > column_rank_threshold → COLUMN
    DEFAULT_ROW             // No specific reason, default to ROW
};

/**
 * @brief Get human-readable string for oracle reason
 */
inline const char* oracle_reason_to_string(OracleReason reason) {
    switch (reason) {
        case OracleReason::SMALL_DIM:        return "small_dim";
        case OracleReason::LOW_RANK:         return "low_rank";
        case OracleReason::LOW_QUBIT_INDEX:  return "low_qubit";
        case OracleReason::STRIDE_FITS_CACHE: return "cache_fit";
        case OracleReason::HIGH_RANK:        return "high_rank";
        case OracleReason::DEFAULT_ROW:      return "default";
        default: return "unknown";
    }
}

//==============================================================================
// Parallelism Oracle Class
//==============================================================================

/**
 * @brief Runtime oracle for selecting optimal parallelism mode
 * 
 * The oracle uses three primary heuristics:
 * 1. Low rank (< 32): Row parallelism → each row fits in L2 cache
 * 2. Low qubit index (< 5): Row parallelism → cache-line friendly stride
 * 3. High rank (> 64): Column parallelism → better work distribution
 * 
 * Usage:
 *   ParallelismOracle oracle;  // or oracle(config)
 *   ParallelMode mode = oracle.select_mode(L, gate);
 */
class ParallelismOracle {
public:
    /**
     * @brief Constructor with default configuration
     */
    ParallelismOracle() : config_(), stats_() {}
    
    /**
     * @brief Constructor with custom configuration
     */
    explicit ParallelismOracle(const OracleConfig& config)
        : config_(config), stats_() {}
    
    /**
     * @brief Select optimal parallelism mode for a single gate operation
     * 
     * @param L Current L matrix (dim × rank)
     * @param gate Gate operation to apply
     * @return Optimal ParallelMode for this gate
     */
    ParallelMode select_mode(const MatrixXcd& L, const GateOp& gate);
    
    /**
     * @brief Select optimal parallelism mode for a batch of gates
     * 
     * Considers the average characteristics of the gate batch
     * 
     * @param L Current L matrix (dim × rank)
     * @param gates Batch of gate operations
     * @return Optimal ParallelMode for this batch
     */
    ParallelMode select_mode_batch(const MatrixXcd& L, 
                                   const std::vector<GateOp>& gates);
    
    /**
     * @brief Select mode with reason (for debugging/logging)
     * 
     * @param L Current L matrix
     * @param gate Gate operation
     * @param[out] reason Reason for the decision
     * @return Optimal ParallelMode
     */
    ParallelMode select_mode_with_reason(const MatrixXcd& L, 
                                          const GateOp& gate,
                                          OracleReason& reason);
    
    /**
     * @brief Get current configuration
     */
    const OracleConfig& config() const { return config_; }
    
    /**
     * @brief Get mutable configuration reference
     */
    OracleConfig& config() { return config_; }
    
    /**
     * @brief Get oracle statistics
     */
    const OracleStats& stats() const { return stats_; }
    
    /**
     * @brief Reset statistics
     */
    void reset_stats() { stats_.reset(); }
    
    /**
     * @brief Set verbose mode
     */
    void set_verbose(bool verbose) { config_.verbose = verbose; }

private:
    OracleConfig config_;
    mutable OracleStats stats_;  // Mutable for const select_mode tracking
    
    /**
     * @brief Check if row stride fits in L2 cache
     * 
     * @param rank Current rank
     * @param stride Gate stride (2^target_qubit for single-qubit gates)
     * @return True if stride × rank × sizeof(complex) < L2 cache size
     */
    bool stride_fits_cache(size_t rank, size_t stride) const;
    
    /**
     * @brief Get maximum qubit index from gate operation
     */
    size_t get_max_qubit(const GateOp& gate) const;
    
    /**
     * @brief Compute stride for a gate operation
     * 
     * For single-qubit gates: stride = 2^target
     * For two-qubit gates: stride = max(2^q1, 2^q2)
     */
    size_t compute_stride(const GateOp& gate) const;
};

//==============================================================================
// Free Functions for Simple Usage
//==============================================================================

/**
 * @brief Simple mode selection without oracle object
 * 
 * Convenience function using default configuration.
 * For repeated calls, prefer creating a ParallelismOracle instance.
 * 
 * @param L Current L matrix
 * @param gate Gate operation
 * @return Optimal ParallelMode
 */
ParallelMode oracle_select_mode(const MatrixXcd& L, const GateOp& gate);

/**
 * @brief Mode selection with custom rank thresholds
 * 
 * @param L Current L matrix
 * @param gate Gate operation
 * @param row_threshold Rank threshold for row mode (default: 32)
 * @param column_threshold Rank threshold for column mode (default: 64)
 * @return Optimal ParallelMode
 */
ParallelMode oracle_select_mode_with_thresholds(
    const MatrixXcd& L, 
    const GateOp& gate,
    size_t row_threshold = 32,
    size_t column_threshold = 64);

}  // namespace qlret
