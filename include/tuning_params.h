#pragma once

/**
 * @file tuning_params.h
 * @brief Performance Tuning Infrastructure for LRET
 *
 * Phase 10, Phase 4B: Performance Tuning Infrastructure
 * From CSV #3, Technique #22: "1.5-3× throughput improvement through empirical tuning"
 *
 * Provides:
 *   1. TunedParameters struct — holds tunable knobs for the simulation.
 *   2. JSON persistence — load/save tuned parameters for reproducibility.
 *   3. Heuristic defaults — get_optimal() returns good defaults based on
 *      circuit characteristics when no tuning data is available.
 *   4. Cache-aware batch sizing — batch_size_for_cache() computes the
 *      optimal gate batch size for a given L matrix shape and cache size.
 *   5. Adaptive truncation thresholds — noise-aware threshold selection.
 *
 * The companion Python script (scripts/auto_tune.py) performs Bayesian
 * optimization over the parameter space using Gaussian Process regression,
 * producing a tuned_params.json file that C++ loads at runtime.
 *
 * @see ROW_PARALLEL_ADVANCED_OPTIMIZATION_PLAN.md Phase 4B
 * @see scripts/auto_tune.py
 */

#include "types.h"
#include <string>
#include <map>
#include <vector>
#include <cstdint>

namespace qlret {

/**
 * @brief Tunable simulation parameters.
 *
 * These parameters control the runtime behaviour of the LRET simulation:
 * - batch_size: how many gates to accumulate before flushing to L.
 * - truncation_threshold: eigenvalue cutoff for rank truncation.
 * - openmp_threads: number of OpenMP threads (0 = system default).
 * - row_rank_threshold: maximum rank for row-parallel mode preference.
 * - column_rank_threshold: minimum rank for column-parallel mode preference.
 * - morton_qubit_threshold: minimum target qubit for Morton reordering.
 * - morton_min_qubits: minimum circuit size for Morton reordering.
 * - prefetch_distance: rows to prefetch ahead in gate loops.
 * - tile_size: L-matrix column tile for cache blocking.
 */
struct TunedParameters {
    // ────────────── Core parameters ──────────────
    size_t batch_size              = 64;
    double truncation_threshold    = 1e-4;
    size_t openmp_threads          = 0;       ///< 0 = use OMP_NUM_THREADS or all cores

    // ────────────── Parallelism thresholds ──────────────
    size_t row_rank_threshold      = 32;      ///< Use row-parallel when rank <= this
    size_t column_rank_threshold   = 64;      ///< Use column-parallel when rank >= this

    // ────────────── Morton order parameters ──────────────
    size_t morton_qubit_threshold  = 8;       ///< Min target qubit for Morton reorder
    size_t morton_min_qubits       = 14;      ///< Min num_qubits for Morton benefit
    size_t morton_min_batch_gates  = 2;       ///< Min qualifying gates in batch

    // ────────────── Cache tuning ──────────────
    size_t prefetch_distance       = 4;       ///< Rows to prefetch ahead
    size_t tile_size               = 8;       ///< Column tile size for cache blocking
    size_t l1_cache_bytes          = 32768;   ///< L1 data cache size estimate
    size_t l2_cache_bytes          = 262144;  ///< L2 cache size estimate
    size_t cache_line_bytes        = 64;      ///< Cache line size

    // ────────────── Truncation strategy ──────────────
    double noise_threshold_scale   = 1.0;     ///< Scale truncation threshold by noise level
    size_t max_rank_limit          = 0;       ///< Hard cap on rank (0 = unlimited)

    // ────────────── Version / metadata ──────────────
    std::string source;                       ///< "default", "heuristic", or "auto_tune"
    std::string version = "1.0";

    // ────────────── Construction ──────────────

    /** Default constructor — all defaults. */
    TunedParameters() : source("default") {}

    // ────────────── JSON persistence ──────────────

    /**
     * @brief Load tuned parameters from a JSON file.
     *
     * Expected format:
     * {
     *   "batch_size": 32,
     *   "truncation_threshold": 1e-6,
     *   "openmp_threads": 8,
     *   "row_rank_threshold": 16,
     *   "column_rank_threshold": 128,
     *   "morton_qubit_threshold": 8,
     *   "morton_min_qubits": 14,
     *   "prefetch_distance": 4,
     *   "tile_size": 8,
     *   "l1_cache_bytes": 32768,
     *   "l2_cache_bytes": 262144,
     *   "source": "auto_tune",
     *   "version": "1.0"
     * }
     *
     * Missing keys retain their defaults.
     *
     * @param path  Path to JSON file.
     * @return Loaded TunedParameters.
     * @throws std::runtime_error if file cannot be opened.
     */
    static TunedParameters load_from_file(const std::string& path);

    /**
     * @brief Save tuned parameters to a JSON file.
     *
     * @param path  Path to JSON file (will be created/overwritten).
     */
    void save_to_file(const std::string& path) const;

    // ────────────── Heuristic selection ──────────────

    /**
     * @brief Get heuristic-optimal parameters for a given circuit.
     *
     * Selects parameters based on empirical rules:
     *
     *   - batch_size:  Proportional to depth, capped by cache capacity.
     *   - truncation_threshold:  Tighter for low-noise circuits, looser
     *     for heavily noisy circuits (where rank growth is dominant cost).
     *   - openmp_threads:  Capped by hardware concurrency.
     *   - Morton parameters:  Adjusted based on qubit count.
     *   - tile_size:  Chosen so that one tile fits in L1 cache.
     *
     * @param num_qubits      Number of qubits in the circuit.
     * @param circuit_depth   Total depth (gates + noise ops).
     * @param noise_probability  Average per-gate noise probability.
     * @return TunedParameters with heuristic values.
     */
    static TunedParameters get_optimal(
        size_t num_qubits,
        size_t circuit_depth,
        double noise_probability
    );

    // ────────────── Cache-aware helpers ──────────────

    /**
     * @brief Compute the optimal gate batch size for cache efficiency.
     *
     * The batch should be small enough that all rows touched by the
     * batched gate application fit in L2 cache.  For a single-qubit
     * gate on qubit t, we touch 2^(n-1) pairs of rows, each of size
     * rank × 16 bytes.  We want:
     *
     *   batch_size × 2 × rank × 16 <= L2_cache_bytes
     *
     * @param dim   Hilbert space dimension (2^n).
     * @param rank  Current rank of L.
     * @return Optimal batch size (at least 1).
     */
    size_t batch_size_for_cache(size_t dim, size_t rank) const;

    /**
     * @brief Compute the optimal column tile size for cache blocking.
     *
     * Tile size chosen so that one tile of two paired rows fits in L1:
     *   tile × 16 bytes × 2 (pair) <= L1_cache_bytes
     *
     * @return Optimal tile size (at least 1).
     */
    size_t tile_size_for_cache() const;

    /**
     * @brief Get adapted truncation threshold based on noise level.
     *
     * For heavily noisy circuits, we use a looser threshold to prevent
     * rank explosion, trading a small amount of accuracy for speed.
     *
     * @param noise_probability  Average per-gate noise probability.
     * @return Adapted truncation threshold.
     */
    double adapted_threshold(double noise_probability) const;
};

}  // namespace qlret
