/**
 * @file parallelism_oracle.cpp
 * @brief Implementation of Parallelism Oracle for Runtime Mode Selection
 * 
 * Phase 2 of Row Parallelism Optimization (ROW_PARALLELISM_OPTIMIZATION_STRATEGY.md).
 * 
 * This oracle implements three primary heuristics based on MPS research and
 * Grok analysis findings:
 * 
 * 1. LOW RANK (< 32):
 *    - Each row of L is 32-64 complex numbers (512-1024 bytes)
 *    - Fits perfectly in L2 cache (256-512 KB per core)
 *    - Row-parallel: minimal cache misses
 *    - Column-parallel: cache thrashing
 *    → Choose ROW parallelism
 * 
 * 2. LOW QUBIT INDEX (< 5):
 *    - Stride = 2^target ≤ 32 rows
 *    - Cache-line friendly access pattern (~512B-4KB jumps)
 *    - Row parallelism over independent row pairs has high data locality
 *    → Choose ROW parallelism
 * 
 * 3. HIGH RANK (> 64):
 *    - Column parallelism provides better work distribution
 *    - Each column is independent, good for thread scaling
 *    → Choose COLUMN parallelism
 * 
 * Default: ROW parallelism (LRET's low-rank nature means row mode usually wins)
 */

#include "parallelism_oracle.h"
#include <iostream>
#include <algorithm>
#include <cmath>

namespace qlret {

//==============================================================================
// ParallelismOracle Implementation
//==============================================================================

ParallelMode ParallelismOracle::select_mode(const MatrixXcd& L, const GateOp& gate) {
    OracleReason reason;
    return select_mode_with_reason(L, gate, reason);
}

ParallelMode ParallelismOracle::select_mode_with_reason(
    const MatrixXcd& L,
    const GateOp& gate,
    OracleReason& reason
) {
    stats_.total_decisions++;
    
    const size_t dim = static_cast<size_t>(L.rows());
    const size_t rank = static_cast<size_t>(L.cols());
    
    // ----- Heuristic 0: Small dimension → Sequential (OpenMP overhead) -----
    if (dim < config_.min_dim_for_parallel) {
        reason = OracleReason::SMALL_DIM;
        stats_.sequential_decisions++;
        stats_.sequential_by_dim++;
        if (config_.verbose) {
            std::cout << "[Oracle] SEQUENTIAL: dim=" << dim 
                      << " < " << config_.min_dim_for_parallel << "\n";
        }
        return ParallelMode::SEQUENTIAL;
    }
    
    // ----- Heuristic 1: Low rank → Row parallelism -----
    // For rank < 32, each row fits in L2 cache (32 × 16 bytes = 512 bytes)
    if (rank < config_.row_rank_threshold) {
        reason = OracleReason::LOW_RANK;
        stats_.row_decisions++;
        stats_.row_by_rank++;
        if (config_.verbose) {
            std::cout << "[Oracle] ROW: rank=" << rank 
                      << " < " << config_.row_rank_threshold << "\n";
        }
        return ParallelMode::ROW;
    }
    
    // ----- Heuristic 2: Low qubit index → Row parallelism -----
    // Gates on qubits 0-4 have stride ≤ 32, cache-line friendly
    const size_t max_qubit = get_max_qubit(gate);
    if (max_qubit < config_.max_qubit_for_row) {
        reason = OracleReason::LOW_QUBIT_INDEX;
        stats_.row_decisions++;
        stats_.row_by_qubit++;
        if (config_.verbose) {
            std::cout << "[Oracle] ROW: max_qubit=" << max_qubit 
                      << " < " << config_.max_qubit_for_row << "\n";
        }
        return ParallelMode::ROW;
    }
    
    // ----- Heuristic 2b: Stride fits in cache → Row parallelism -----
    const size_t stride = compute_stride(gate);
    if (stride_fits_cache(rank, stride)) {
        reason = OracleReason::STRIDE_FITS_CACHE;
        stats_.row_decisions++;
        stats_.row_by_cache++;
        if (config_.verbose) {
            std::cout << "[Oracle] ROW: stride=" << stride 
                      << " × rank=" << rank << " fits L2 cache\n";
        }
        return ParallelMode::ROW;
    }
    
    // ----- Heuristic 3: High rank → Column parallelism -----
    // For rank > 64 with large dimension, column parallelism scales better
    if (rank > config_.column_rank_threshold && dim >= 8192) {
        reason = OracleReason::HIGH_RANK;
        stats_.column_decisions++;
        stats_.column_by_rank++;
        if (config_.verbose) {
            std::cout << "[Oracle] COLUMN: rank=" << rank 
                      << " > " << config_.column_rank_threshold 
                      << ", dim=" << dim << " >= 8192\n";
        }
        return ParallelMode::COLUMN;
    }
    
    // ----- Default: Row parallelism -----
    // LRET's low-rank nature means row mode is usually the best default
    reason = OracleReason::DEFAULT_ROW;
    stats_.row_decisions++;
    if (config_.verbose) {
        std::cout << "[Oracle] ROW: default (rank=" << rank 
                  << ", dim=" << dim << ")\n";
    }
    return ParallelMode::ROW;
}

ParallelMode ParallelismOracle::select_mode_batch(
    const MatrixXcd& L,
    const std::vector<GateOp>& gates
) {
    if (gates.empty()) {
        return ParallelMode::SEQUENTIAL;
    }
    
    const size_t dim = static_cast<size_t>(L.rows());
    const size_t rank = static_cast<size_t>(L.cols());
    
    // For small dimension, always sequential
    if (dim < config_.min_dim_for_parallel) {
        return ParallelMode::SEQUENTIAL;
    }
    
    // Low rank → row parallelism (highest priority heuristic)
    if (rank < config_.row_rank_threshold) {
        return ParallelMode::ROW;
    }
    
    // Analyze batch: compute average max qubit and stride
    size_t total_max_qubit = 0;
    size_t low_qubit_count = 0;
    
    for (const auto& gate : gates) {
        size_t max_q = get_max_qubit(gate);
        total_max_qubit += max_q;
        if (max_q < config_.max_qubit_for_row) {
            low_qubit_count++;
        }
    }
    
    // If majority of gates are low-qubit, use row parallelism
    if (low_qubit_count > gates.size() / 2) {
        return ParallelMode::ROW;
    }
    
    // High rank with large dimension → column parallelism
    if (rank > config_.column_rank_threshold && dim >= 8192) {
        return ParallelMode::COLUMN;
    }
    
    // Default to row (LRET's low-rank nature)
    return ParallelMode::ROW;
}

//==============================================================================
// Private Helper Methods
//==============================================================================

bool ParallelismOracle::stride_fits_cache(size_t rank, size_t stride) const {
    // Each row segment = rank × sizeof(complex<double>) = rank × 16 bytes
    // For a strided access, we access (stride × row_size) bytes per iteration
    // If this fits in L2 cache, row parallelism is efficient
    
    const size_t row_bytes = rank * OracleConfig::BYTES_PER_ELEMENT;
    const size_t strided_bytes = stride * row_bytes;
    
    // Conservative: use 1/4 of L2 cache as threshold
    // This accounts for working set of multiple threads
    const size_t cache_threshold = config_.l2_cache_bytes / 4;
    
    return strided_bytes < cache_threshold;
}

size_t ParallelismOracle::get_max_qubit(const GateOp& gate) const {
    if (gate.qubits.empty()) {
        return 0;
    }
    return *std::max_element(gate.qubits.begin(), gate.qubits.end());
}

size_t ParallelismOracle::compute_stride(const GateOp& gate) const {
    if (gate.qubits.empty()) {
        return 1;
    }
    
    // For single-qubit gate: stride = 2^target
    // For two-qubit gate: stride = max(2^q1, 2^q2)
    size_t max_stride = 1;
    for (size_t q : gate.qubits) {
        size_t stride = 1ULL << q;
        if (stride > max_stride) {
            max_stride = stride;
        }
    }
    return max_stride;
}

//==============================================================================
// Free Functions
//==============================================================================

ParallelMode oracle_select_mode(const MatrixXcd& L, const GateOp& gate) {
    static ParallelismOracle default_oracle;
    return default_oracle.select_mode(L, gate);
}

ParallelMode oracle_select_mode_with_thresholds(
    const MatrixXcd& L,
    const GateOp& gate,
    size_t row_threshold,
    size_t column_threshold
) {
    OracleConfig config;
    config.row_rank_threshold = row_threshold;
    config.column_rank_threshold = column_threshold;
    ParallelismOracle oracle(config);
    return oracle.select_mode(L, gate);
}

}  // namespace qlret
