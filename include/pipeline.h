#pragma once

/**
 * @file pipeline.h
 * @brief Unified Optimized Simulation Pipeline (Phase 6A)
 *
 * Phase 6 of Advanced Row-Parallel Optimization.
 *
 * BACKGROUND:
 * Phases 1-5 each provide independent optimizations:
 *   Phase 1A: Iterative compression (smaller Gram matrices)
 *   Phase 1B: DLRA evolution (SVD-based tangent-space truncation)
 *   Phase 2A: CP decomposition (Kronecker-structured circuits)
 *   Phase 2B: Sparse tensor mode (high-noise circuits)
 *   Phase 3A: Distributed tensor scatter (MPI multi-level)
 *   Phase 3B: Variational Lindblad (fixed-rank evolution)
 *   Phase 4A: Morton order (cache-friendly gate application)
 *   Phase 4B: Tuned parameters (empirically optimized knobs)
 *   Phase 5A: Matrix completion (partial Pauli measurements)
 *   Phase 5B: Quantum state tomography (compressed sensing)
 *
 * The OptimizedPipeline class automatically selects and chains the best
 * combination of these optimizations based on circuit characteristics.
 *
 * SELECTION LOGIC:
 *   1. Analyze the circuit (noise ratio, gate pattern, qubit count)
 *   2. Select noise handling: iterative compression vs DLRA vs sparse
 *   3. Select truncation: Gram eigendecomp vs CP vs DLRA-SVD
 *   4. Select gate application: row-parallel ± Morton order
 *   5. Optionally run tomography on the final state
 *
 * This is the "production" entry point that ties everything together.
 *
 * @see ROW_PARALLEL_ADVANCED_OPTIMIZATION_PLAN.md Phase 6
 */

#include "types.h"
#include "simulator.h"
#include "gates_and_noise.h"
#include "iterative_compression.h"
#include "dlra_evolution.h"
#include "cp_decomposition.h"
#include "sparse_tensor_sim.h"
#include "morton_order.h"
#include "tuning_params.h"
#include "matrix_completion.h"
#include <string>
#include <chrono>
#include <vector>
#include <functional>

namespace qlret {

//==============================================================================
// Strategy Enumerations
//==============================================================================

/**
 * @brief Noise handling strategy for the pipeline
 */
enum class NoiseStrategy {
    /// Standard LRET: concatenate all Kraus results, then truncate_L
    Standard,

    /// Phase 1A: Iterative compression — apply Kraus one-by-one with
    /// incremental Gram truncation.  Best general-purpose strategy.
    IterativeCompression,

    /// Phase 1B: DLRA — SVD-based tangent-space projection.
    /// More stable for rank-sensitive circuits.
    DLRA,

    /// Phase 2B: Sparse tensor — zero-out small elements.
    /// Best for high-noise (>50% sparse) circuits.
    Sparse,

    /// Let the pipeline auto-select based on circuit analysis.
    Auto
};

/**
 * @brief Truncation strategy for the pipeline
 */
enum class TruncationStrategy {
    /// Standard Gram eigendecomposition
    GramEigen,

    /// Phase 2A: CP decomposition for Kronecker-structured circuits (QFT, Grover)
    CPDecomposition,

    /// Phase 1B: SVD-based (part of DLRA)
    SVD,

    /// Let the pipeline auto-select
    Auto
};

/**
 * @brief Gate application strategy
 */
enum class GateStrategy {
    /// Standard row-parallel with OpenMP
    RowParallel,

    /// Phase 4A: Morton order for cache-friendly high-stride access
    MortonOrder,

    /// Let the pipeline auto-select
    Auto
};

//==============================================================================
// Pipeline Configuration
//==============================================================================

/**
 * @brief Configuration for the OptimizedPipeline
 */
struct PipelineConfig {
    // ── Strategy selection ──
    NoiseStrategy noise_strategy = NoiseStrategy::Auto;
    TruncationStrategy truncation_strategy = TruncationStrategy::Auto;
    GateStrategy gate_strategy = GateStrategy::Auto;

    // ── Core simulation parameters ──
    double truncation_threshold = 1e-4;
    size_t max_rank = 0;             ///< 0 = no hard limit
    size_t batch_size = 64;
    bool use_parallel = true;
    bool verbose = false;

    // ── Phase 4B: Tuning ──
    bool use_tuned_params = true;    ///< Use TunedParameters heuristics
    std::string tuned_params_file;   ///< Path to tuned_params.json (empty = heuristic)

    // ── Phase 5: Tomography ──
    bool run_tomography = false;     ///< Run compressed tomography on final state
    double tomography_fraction = 0.5; ///< Fraction of Paulis to measure

    // ── Validation ──
    bool validate_output = true;     ///< Validate ρ properties at end
    double validation_tolerance = 1e-6; ///< Tolerance for DM validation
};

//==============================================================================
// Pipeline Statistics
//==============================================================================

/**
 * @brief Detailed statistics from a pipeline run
 */
struct PipelineStats {
    // ── Strategies used ──
    NoiseStrategy noise_strategy_used = NoiseStrategy::Standard;
    TruncationStrategy truncation_strategy_used = TruncationStrategy::GramEigen;
    GateStrategy gate_strategy_used = GateStrategy::RowParallel;

    // ── Timing breakdown (seconds) ──
    double total_time = 0.0;
    double gate_time = 0.0;
    double noise_time = 0.0;
    double truncation_time = 0.0;
    double tomography_time = 0.0;
    double validation_time = 0.0;
    double strategy_selection_time = 0.0;

    // ── Rank tracking ──
    size_t initial_rank = 0;
    size_t final_rank = 0;
    size_t max_rank_reached = 0;
    size_t truncation_count = 0;

    // ── Quality metrics ──
    double trace_deviation = 0.0;     ///< |Tr[ρ] - 1|
    bool is_hermitian = false;
    bool is_psd = false;
    bool is_valid_dm = false;

    // ── Tomography results (if run) ──
    double tomography_fidelity = 0.0; ///< F(ρ_tomo, ρ_direct)
    double tomography_trace_distance = 0.0;
    size_t tomography_measurements_used = 0;

    // ── Circuit characteristics (detected) ──
    size_t num_gates = 0;
    size_t num_noise_ops = 0;
    double noise_ratio = 0.0;         ///< noise_ops / total_ops
    CircuitPattern detected_pattern = CircuitPattern::UNKNOWN;

    /**
     * @brief Get human-readable summary string
     */
    std::string summary() const;
};

//==============================================================================
// Pipeline Result
//==============================================================================

/**
 * @brief Complete result from a pipeline run
 */
struct PipelineResult {
    MatrixXcd L_final;          ///< Final low-rank factor
    PipelineStats stats;        ///< Detailed statistics

    /// Tomography result (if run_tomography was enabled)
    MatrixXcd rho_tomography;   ///< Reconstructed density matrix (may be empty)
    CompletionStats tomography_stats; ///< Tomography statistics
};

//==============================================================================
// OptimizedPipeline Class
//==============================================================================

/**
 * @brief Unified simulation pipeline with automatic Phase 1-5 optimization selection
 *
 * Usage:
 * @code
 *   // Simple usage with auto-selection
 *   OptimizedPipeline pipeline(num_qubits);
 *   auto result = pipeline.run(L_init, sequence);
 *   // result.L_final, result.stats.summary()
 *
 *   // Custom configuration
 *   PipelineConfig config;
 *   config.noise_strategy = NoiseStrategy::IterativeCompression;
 *   config.truncation_strategy = TruncationStrategy::CPDecomposition;
 *   config.run_tomography = true;
 *   OptimizedPipeline pipeline2(num_qubits, config);
 *   auto result2 = pipeline2.run(L_init, sequence);
 * @endcode
 */
class OptimizedPipeline {
public:
    /**
     * @brief Construct pipeline for a given qubit count
     * @param num_qubits  Number of qubits
     * @param config  Pipeline configuration
     */
    OptimizedPipeline(size_t num_qubits, const PipelineConfig& config = PipelineConfig());

    /**
     * @brief Run the full optimized simulation pipeline
     *
     * Steps:
     * 1. Analyze the circuit and select strategies
     * 2. Optionally load tuned parameters
     * 3. Execute the simulation with selected optimizations
     * 4. Validate the output state
     * 5. Optionally run compressed tomography
     * 6. Return result with detailed statistics
     *
     * @param L_init  Initial low-rank factor (dim × rank)
     * @param sequence  Quantum circuit to simulate
     * @return PipelineResult with final state and statistics
     */
    PipelineResult run(const MatrixXcd& L_init, const QuantumSequence& sequence);

    /**
     * @brief Analyze a circuit and determine optimal strategies
     *
     * Can be called independently to preview what the pipeline would do
     * without actually running the simulation.
     *
     * @param sequence  The circuit to analyze
     * @return PipelineStats with strategy selections (timing fields will be zero)
     */
    PipelineStats analyze(const QuantumSequence& sequence) const;

    /**
     * @brief Run just the validation step on a final L factor
     *
     * Checks: Hermitian, PSD, trace = 1
     *
     * @param L  Low-rank factor to validate
     * @param tolerance  Numerical tolerance
     * @return true if valid density matrix
     */
    bool validate(const MatrixXcd& L, double tolerance = 1e-6) const;

    /**
     * @brief Get current configuration
     */
    const PipelineConfig& config() const { return config_; }

    /**
     * @brief Get number of qubits
     */
    size_t num_qubits() const { return num_qubits_; }

    /**
     * @brief Get the Hilbert space dimension (2^num_qubits)
     */
    size_t dimension() const { return dim_; }

    /**
     * @brief Get human-readable description of selected strategies
     */
    std::string strategy_description(const PipelineStats& stats) const;

private:
    size_t num_qubits_;
    size_t dim_;
    PipelineConfig config_;
    TunedParameters tuned_params_;

    /**
     * @brief Select the best noise handling strategy for a circuit
     */
    NoiseStrategy select_noise_strategy(const QuantumSequence& sequence) const;

    /**
     * @brief Select the best truncation strategy for a circuit
     * @param sequence Circuit to analyze
     * @param resolved_noise The already-resolved noise strategy (used for synergy)
     */
    TruncationStrategy select_truncation_strategy(
        const QuantumSequence& sequence,
        NoiseStrategy resolved_noise
    ) const;

    /**
     * @brief Select the best gate application strategy for a circuit
     */
    GateStrategy select_gate_strategy(const QuantumSequence& sequence) const;

    /**
     * @brief Execute simulation with the selected strategies
     */
    MatrixXcd execute(
        const MatrixXcd& L_init,
        const QuantumSequence& sequence,
        NoiseStrategy noise_strat,
        TruncationStrategy trunc_strat,
        GateStrategy gate_strat,
        PipelineStats& stats
    );

    /**
     * @brief Apply a single noise operation using the selected strategy
     */
    MatrixXcd apply_noise_with_strategy(
        const MatrixXcd& L,
        const NoiseOp& noise_op,
        NoiseStrategy strategy
    );

    /**
     * @brief Apply truncation using the selected strategy
     */
    MatrixXcd apply_truncation_with_strategy(
        const MatrixXcd& L,
        TruncationStrategy strategy
    );

    /**
     * @brief Apply a gate operation using the selected strategy
     */
    MatrixXcd apply_gate_with_strategy(
        const MatrixXcd& L,
        const GateOp& gate_op,
        GateStrategy strategy
    );

    /**
     * @brief Compute circuit characteristics for strategy selection
     */
    void compute_circuit_stats(
        const QuantumSequence& sequence,
        PipelineStats& stats
    ) const;
};

//==============================================================================
// Convenience Functions
//==============================================================================

/**
 * @brief Run a simulation with all optimizations auto-selected
 *
 * Convenience wrapper for OptimizedPipeline::run().
 *
 * @param L_init  Initial low-rank factor
 * @param sequence  Quantum circuit
 * @param num_qubits  Number of qubits
 * @param threshold  Truncation threshold (default 1e-4)
 * @param verbose  Print progress (default false)
 * @return PipelineResult
 */
PipelineResult run_optimized_pipeline(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    double threshold = 1e-4,
    bool verbose = false
);

/**
 * @brief Run simulation and compare baseline vs pipeline for validation
 *
 * Runs both standard (naive) simulation and the optimized pipeline,
 * then compares the results for correctness.
 *
 * @param L_init  Initial low-rank factor
 * @param sequence  Quantum circuit
 * @param num_qubits  Number of qubits
 * @param threshold  Truncation threshold
 * @return Pair of (fidelity between results, PipelineResult from optimized run)
 */
std::pair<double, PipelineResult> run_and_validate_pipeline(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    double threshold = 1e-4
);

}  // namespace qlret
