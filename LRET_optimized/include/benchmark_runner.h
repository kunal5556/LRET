#pragma once

#include "benchmark_types.h"
#include "cli_parser.h"
#include "types.h"
#include "output_formatter.h"
#include <functional>

namespace qlret {

//==============================================================================
// Benchmark Runner - Executes Parameter Sweeps
//==============================================================================

/**
 * @brief Run a single simulation with rank evolution tracking
 * 
 * This is a modified simulation that records rank after each operation.
 * Used for generating Paper Figure: rank vs circuit depth.
 * 
 * @param L_init Initial low-rank factor
 * @param sequence Quantum sequence to simulate
 * @param num_qubits Number of qubits
 * @param config Simulation configuration
 * @param track_rank Whether to track rank evolution
 * @return SweepPointResult with timing, rank, and optionally evolution data
 */
SweepPointResult run_single_benchmark(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    const SimConfig& config,
    bool track_rank = false,
    bool include_fdm = false
);

/**
 * @brief Run all LRET modes for a single benchmark point
 * 
 * Runs sequential, row, column, hybrid, and adaptive modes
 * and returns results for each.
 */
std::vector<ModePointResult> run_all_modes_benchmark(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    const SimConfig& config,
    size_t batch_size,
    bool include_fdm = false,
    const MatrixXcd* fdm_rho_final = nullptr
);

/**
 * @brief Run a complete parameter sweep
 * 
 * Executes the simulation at each sweep point and collects results.
 * Supports:
 * - Epsilon sweep (truncation threshold)
 * - Noise probability sweep
 * - Qubit count sweep
 * - Depth sweep
 * - LRET vs FDM crossover analysis
 * 
 * @param opts CLI options containing sweep configuration
 * @param callback Optional callback for progress reporting (point_idx, total_points, current_result)
 * @return SweepResults containing all sweep point results
 */
SweepResults run_parameter_sweep(
    const CLIOptions& opts,
    std::function<void(size_t, size_t, const SweepPointResult&)> callback = nullptr
);

/**
 * @brief Run epsilon (truncation threshold) sweep
 * 
 * Paper Figure: Runtime and rank vs. epsilon
 * Fixed: n, d, noise_prob
 * Vary: epsilon = {1e-7, 1e-6, ..., 1e-2}
 */
SweepResults run_epsilon_sweep(
    const CLIOptions& opts,
    const std::vector<double>& epsilon_values
);

/**
 * @brief Run noise probability sweep
 * 
 * Paper Figure: Runtime and rank vs. noise probability
 * Fixed: n, d, epsilon
 * Vary: p = {0.0, 0.01, 0.05, 0.1, 0.2, ...}
 */
SweepResults run_noise_sweep(
    const CLIOptions& opts,
    const std::vector<double>& noise_values
);

/**
 * @brief Run initial rank sweep
 * 
 * Paper Figure: Runtime vs. initial rank
 * Fixed: n, d, epsilon, noise_prob
 * Vary: rank = {1, 2, 4, 8, 16, ...}
 */
SweepResults run_rank_sweep(
    const CLIOptions& opts,
    const std::vector<size_t>& rank_values
);

/**
 * @brief Run qubit count sweep
 * 
 * Paper Figure: Runtime vs. number of qubits
 * Fixed: d, noise_prob, epsilon
 * Vary: n = {5, 6, 7, ..., 20}
 */
SweepResults run_qubit_sweep(
    const CLIOptions& opts,
    const std::vector<size_t>& qubit_values
);

/**
 * @brief Run circuit depth sweep
 * 
 * Paper Figure: Runtime vs. circuit depth
 * Fixed: n, noise_prob, epsilon
 * Vary: d = {10, 20, 50, 100, ...}
 */
SweepResults run_depth_sweep(
    const CLIOptions& opts,
    const std::vector<size_t>& depth_values
);

/**
 * @brief Run LRET vs FDM crossover analysis
 * 
 * Paper Figure: LRET time and FDM time vs. qubits
 * Identifies crossover point where LRET becomes faster
 * 
 * Both LRET and FDM are run at each point.
 */
SweepResults run_crossover_analysis(
    const CLIOptions& opts,
    size_t min_qubits = 5,
    size_t max_qubits = 15
);

//==============================================================================
// Rank Evolution Simulator
//==============================================================================

/**
 * @brief Run simulation with detailed rank evolution tracking
 * 
 * Records rank before and after each:
 * - Gate application
 * - Noise/Kraus application  
 * - Truncation
 * 
 * @param L_init Initial low-rank factor
 * @param sequence Quantum sequence
 * @param num_qubits Number of qubits
 * @param config Simulation config
 * @return RankEvolution with complete history
 */
RankEvolution run_with_rank_tracking(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    const SimConfig& config
);

/**
 * @brief Compute memory usage for LRET simulation
 * 
 * @param L Current L matrix
 * @return Memory in bytes for L matrix
 */
size_t compute_L_memory_bytes(const MatrixXcd& L);

/**
 * @brief Compute theoretical FDM memory
 * 
 * FDM memory = 2^n × 2^n × sizeof(complex<double>)
 * 
 * @param num_qubits Number of qubits
 * @return Theoretical memory in bytes
 */
size_t compute_fdm_memory_bytes(size_t num_qubits);

/**
 * @brief Create MemoryComparison for LRET vs FDM
 */
MemoryComparison create_memory_comparison(
    const MatrixXcd& L_final,
    size_t num_qubits,
    bool fdm_was_run = false,
    size_t fdm_peak_bytes = 0
);

//==============================================================================
// Timing Analysis
//==============================================================================

/**
 * @brief Create detailed timing breakdown from rank evolution
 */
TimingBreakdown create_timing_breakdown(const RankEvolution& evolution);

/**
 * @brief Format timing breakdown as string for output
 */
std::string format_timing_breakdown(const TimingBreakdown& timing);

//==============================================================================
// Statistical Analysis (for multiple trials)
//==============================================================================

/**
 * @brief Compute mean and standard deviation of times
 */
struct TrialStats {
    double mean = 0.0;
    double stddev = 0.0;
    double min_val = 0.0;
    double max_val = 0.0;
    size_t count = 0;
};

TrialStats compute_trial_stats(const std::vector<double>& values);

}  // namespace qlret
