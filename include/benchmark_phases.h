#pragma once

/**
 * @file benchmark_phases.h
 * @brief Phase-by-Phase Benchmark Suite (Phase 6B)
 *
 * Provides systematic performance comparison of each optimization phase
 * (1A, 1B, 2A, 2B, 4A, 4B, 5A/B, 6A pipeline) against the baseline
 * simulator. Measures time, rank, memory proxy, and fidelity.
 *
 * @see ROW_PARALLEL_ADVANCED_OPTIMIZATION_PLAN.md Phase 6B
 */

#include "pipeline.h"
#include <string>
#include <vector>
#include <functional>

namespace qlret {

//==============================================================================
// Benchmark Result
//==============================================================================

/**
 * @brief Result of a single benchmark run
 */
struct BenchmarkResult {
    /// Label identifying the method (e.g. "Baseline", "Phase1A", "Pipeline")
    std::string label;

    /// Number of qubits
    size_t num_qubits = 0;

    /// Circuit depth (total operations)
    size_t circuit_depth = 0;

    /// Noise probability used
    double noise_probability = 0.0;

    /// Execution time in seconds
    double elapsed_seconds = 0.0;

    /// Final rank of L
    size_t final_rank = 0;

    /// Peak rank during simulation
    size_t peak_rank = 0;

    /// Memory proxy: dim × final_rank × sizeof(complex<double>)
    size_t memory_bytes_proxy = 0;

    /// Fidelity vs baseline (1.0 = perfect match)
    double fidelity_vs_baseline = 1.0;

    /// Trace deviation |Tr[ρ] - 1|
    double trace_deviation = 0.0;

    /// Whether the output is a valid density matrix
    bool is_valid_dm = false;

    /// Extra info string (strategy description, etc.)
    std::string notes;

    /**
     * @brief Get one-line summary
     */
    std::string one_line() const;
};

//==============================================================================
// Benchmark Configuration
//==============================================================================

/**
 * @brief Configuration for the benchmark suite
 */
struct BenchmarkConfig {
    /// Qubit counts to test
    std::vector<size_t> qubit_counts = {2, 4, 6, 8};

    /// Circuit depths (gates per qubit)
    std::vector<size_t> depths_per_qubit = {4, 8};

    /// Noise probabilities to test
    std::vector<double> noise_probs = {0.0, 0.01, 0.05};

    /// Truncation threshold
    double truncation_threshold = 1e-4;

    /// Number of repetitions per configuration (for timing stability)
    size_t num_reps = 1;

    /// Whether to compute fidelity vs baseline (slower but more informative)
    bool compute_fidelity = true;

    /// Print progress to stdout
    bool verbose = true;

    /// Which phases to benchmark (empty = all)
    std::vector<std::string> phases_to_run;
};

//==============================================================================
// Circuit Generators
//==============================================================================

/**
 * @brief Generate a random circuit with Hadamard, CNOT, RZ, and noise
 *
 * @param num_qubits Number of qubits
 * @param depth Number of layers (each layer: random gates on all qubits + noise)
 * @param noise_prob Probability for depolarizing noise after each gate
 * @param seed Random seed (0 = nondeterministic)
 * @return QuantumSequence
 */
QuantumSequence generate_random_circuit(
    size_t num_qubits,
    size_t depth,
    double noise_prob,
    unsigned seed = 42
);

/**
 * @brief Generate a QFT-like circuit
 *
 * @param num_qubits Number of qubits
 * @param noise_prob Probability for depolarizing noise after each gate
 * @return QuantumSequence
 */
QuantumSequence generate_qft_circuit(
    size_t num_qubits,
    double noise_prob = 0.0
);

/**
 * @brief Generate a heavily noisy circuit (>50% noise ops)
 *
 * @param num_qubits Number of qubits
 * @param depth Number of layers
 * @param noise_prob Per-gate noise probability (high, e.g. 0.1)
 * @return QuantumSequence
 */
QuantumSequence generate_noisy_circuit(
    size_t num_qubits,
    size_t depth,
    double noise_prob = 0.1
);

//==============================================================================
// PhaseBenchmark Class
//==============================================================================

/**
 * @brief Systematic benchmark comparing all optimization phases
 *
 * Usage:
 * @code
 *   BenchmarkConfig cfg;
 *   cfg.qubit_counts = {2, 4, 6, 8};
 *   cfg.noise_probs = {0.0, 0.01, 0.05};
 *
 *   PhaseBenchmark bench(cfg);
 *   auto results = bench.run_all();
 *   bench.print_table(results);
 *   bench.save_csv(results, "benchmark_results.csv");
 * @endcode
 */
class PhaseBenchmark {
public:
    /**
     * @brief Construct benchmark suite
     * @param config Benchmark configuration
     */
    explicit PhaseBenchmark(const BenchmarkConfig& config = BenchmarkConfig());

    /**
     * @brief Run all benchmarks across all configurations
     * @return Vector of results
     */
    std::vector<BenchmarkResult> run_all();

    /**
     * @brief Run benchmark for a single (qubits, depth, noise) configuration
     * @param num_qubits Number of qubits
     * @param depth Circuit depth
     * @param noise_prob Noise probability
     * @return Vector of results (one per phase)
     */
    std::vector<BenchmarkResult> run_single(
        size_t num_qubits,
        size_t depth,
        double noise_prob
    );

    /**
     * @brief Print results as a formatted table
     */
    void print_table(const std::vector<BenchmarkResult>& results) const;

    /**
     * @brief Save results to CSV
     * @param results Benchmark results
     * @param path Output CSV path
     */
    void save_csv(
        const std::vector<BenchmarkResult>& results,
        const std::string& path
    ) const;

    /**
     * @brief Generate a markdown summary of results
     * @param results Benchmark results
     * @return Markdown string
     */
    std::string markdown_summary(const std::vector<BenchmarkResult>& results) const;

private:
    BenchmarkConfig config_;

    /**
     * @brief Run baseline simulation and return result + rho for fidelity
     */
    std::pair<BenchmarkResult, MatrixXcd> run_baseline(
        const MatrixXcd& L_init,
        const QuantumSequence& sequence,
        size_t num_qubits
    );

    /**
     * @brief Run Phase 1A (Iterative Compression) benchmark
     */
    BenchmarkResult run_phase1a(
        const MatrixXcd& L_init,
        const QuantumSequence& sequence,
        size_t num_qubits,
        const MatrixXcd& rho_baseline
    );

    /**
     * @brief Run Phase 1B (DLRA) benchmark
     */
    BenchmarkResult run_phase1b(
        const MatrixXcd& L_init,
        const QuantumSequence& sequence,
        size_t num_qubits,
        const MatrixXcd& rho_baseline
    );

    /**
     * @brief Run Phase 2A (CP Decomposition) benchmark
     */
    BenchmarkResult run_phase2a(
        const MatrixXcd& L_init,
        const QuantumSequence& sequence,
        size_t num_qubits,
        const MatrixXcd& rho_baseline
    );

    /**
     * @brief Run Phase 2B (Sparse Tensor) benchmark
     */
    BenchmarkResult run_phase2b(
        const MatrixXcd& L_init,
        const QuantumSequence& sequence,
        size_t num_qubits,
        const MatrixXcd& rho_baseline
    );

    /**
     * @brief Run Phase 6A (Unified Pipeline) benchmark
     */
    BenchmarkResult run_pipeline(
        const MatrixXcd& L_init,
        const QuantumSequence& sequence,
        size_t num_qubits,
        const MatrixXcd& rho_baseline
    );

    /**
     * @brief Compute fidelity between two density matrices
     */
    double compute_fidelity(const MatrixXcd& rho_a, const MatrixXcd& rho_b) const;

    /**
     * @brief Check if a phase label should be run
     */
    bool should_run(const std::string& phase_label) const;
};

}  // namespace qlret
