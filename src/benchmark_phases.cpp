/**
 * @file benchmark_phases.cpp
 * @brief Implementation of Phase-by-Phase Benchmark Suite (Phase 6B)
 */

#include "benchmark_phases.h"
#include "iterative_compression.h"
#include "dlra_evolution.h"
#include "cp_decomposition.h"
#include "sparse_tensor_sim.h"
#include "morton_order.h"
#include "tuning_params.h"
#include "matrix_completion.h"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>

namespace qlret {

//==============================================================================
// BenchmarkResult::one_line
//==============================================================================

std::string BenchmarkResult::one_line() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << std::setw(20) << label
       << " | " << std::setw(2) << num_qubits << "q"
       << " | " << std::setw(4) << circuit_depth << " ops"
       << " | p=" << std::setw(5) << noise_probability
       << " | " << std::setw(10) << elapsed_seconds << "s"
       << " | rank=" << std::setw(4) << final_rank
       << " | peak=" << std::setw(4) << peak_rank
       << " | F=" << std::setw(8) << fidelity_vs_baseline
       << " | valid=" << (is_valid_dm ? "Y" : "N");
    return ss.str();
}

//==============================================================================
// Circuit Generators
//==============================================================================

QuantumSequence generate_random_circuit(
    size_t num_qubits,
    size_t depth,
    double noise_prob,
    unsigned seed
) {
    QuantumSequence seq;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * PI);

    for (size_t layer = 0; layer < depth; ++layer) {
        // Single-qubit gates on all qubits
        for (size_t q = 0; q < num_qubits; ++q) {
            double theta = angle_dist(rng);

            // Alternating H and RZ gates
            if (layer % 2 == 0) {
                seq.operations.push_back(GateOp(GateType::H, q));
            } else {
                seq.operations.push_back(GateOp(GateType::RZ, q, std::vector<double>{theta}));
            }

            // Add noise after each gate
            if (noise_prob > 0) {
                NoiseOp noise(NoiseType::DEPOLARIZING, q, noise_prob);
                seq.operations.push_back(noise);
            }
        }

        // CNOT layer (entangling)
        for (size_t q = 0; q + 1 < num_qubits; q += 2) {
            seq.operations.push_back(GateOp(GateType::CNOT, q, q + 1));

            if (noise_prob > 0) {
                NoiseOp noise(NoiseType::DEPOLARIZING, q, noise_prob);
                seq.operations.push_back(noise);
            }
        }
    }

    return seq;
}

QuantumSequence generate_qft_circuit(size_t num_qubits, double noise_prob) {
    QuantumSequence seq;

    for (size_t i = 0; i < num_qubits; ++i) {
        // H gate on qubit i
        seq.operations.push_back(GateOp(GateType::H, i));

        if (noise_prob > 0) {
            seq.operations.push_back(NoiseOp(NoiseType::DEPOLARIZING, i, noise_prob));
        }

        // Controlled-phase gates (CZ with phase angle)
        for (size_t j = i + 1; j < num_qubits; ++j) {
            double angle = PI / static_cast<double>(size_t(1) << (j - i));

            // Use RZ on target qubit as approximation of controlled-phase
            // (For benchmarking purposes, the exact gate structure matters
            //  less than having the right gate/noise ratio)
            seq.operations.push_back(GateOp(GateType::CZ, i, j));
            seq.operations.push_back(GateOp(GateType::RZ, j, std::vector<double>{angle}));

            if (noise_prob > 0) {
                seq.operations.push_back(NoiseOp(NoiseType::DEPOLARIZING, i, noise_prob));
            }
        }
    }

    return seq;
}

QuantumSequence generate_noisy_circuit(
    size_t num_qubits,
    size_t depth,
    double noise_prob
) {
    QuantumSequence seq;

    for (size_t layer = 0; layer < depth; ++layer) {
        // One gate per layer, but multiple noise ops
        size_t q = layer % num_qubits;

        // RZ gate
        double theta = PI / 4.0;
        seq.operations.push_back(GateOp(GateType::RZ, q, std::vector<double>{theta}));

        // Multiple noise ops (making noise ratio > 50%)
        for (size_t nq = 0; nq < num_qubits; ++nq) {
            NoiseOp noise(NoiseType::DEPOLARIZING, nq, noise_prob);
            seq.operations.push_back(noise);
        }
    }

    return seq;
}

//==============================================================================
// PhaseBenchmark
//==============================================================================

PhaseBenchmark::PhaseBenchmark(const BenchmarkConfig& config)
    : config_(config)
{
}

bool PhaseBenchmark::should_run(const std::string& phase_label) const {
    if (config_.phases_to_run.empty()) return true;
    for (const auto& p : config_.phases_to_run) {
        if (p == phase_label) return true;
    }
    return false;
}

double PhaseBenchmark::compute_fidelity(
    const MatrixXcd& rho_a,
    const MatrixXcd& rho_b
) const {
    if (rho_a.rows() == 0 || rho_b.rows() == 0) return 0.0;
    return QuantumStateTomography::fidelity(rho_a, rho_b);
}

//==============================================================================
// Baseline
//==============================================================================

std::pair<BenchmarkResult, MatrixXcd> PhaseBenchmark::run_baseline(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits
) {
    BenchmarkResult result;
    result.label = "Baseline";
    result.num_qubits = num_qubits;
    result.circuit_depth = sequence.operations.size();

    auto start = std::chrono::steady_clock::now();

    MatrixXcd L_final = run_simulation_optimized(
        L_init, sequence, num_qubits,
        64, true, false, config_.truncation_threshold
    );

    auto end = std::chrono::steady_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(end - start).count();

    result.final_rank = static_cast<size_t>(L_final.cols());
    result.peak_rank = result.final_rank; // baseline doesn't track peak
    result.fidelity_vs_baseline = 1.0;

    size_t dim = size_t(1) << num_qubits;
    result.memory_bytes_proxy = dim * result.final_rank * sizeof(Complex);

    // Validate
    if (dim <= 4096) {
        MatrixXcd rho = L_final * L_final.adjoint();
        result.trace_deviation = std::abs(rho.trace().real() - 1.0);
        result.is_valid_dm = validate_density_matrix(rho, 1e-6);
        return {result, rho};
    } else {
        MatrixXcd G = L_final.adjoint() * L_final;
        result.trace_deviation = std::abs(G.trace().real() - 1.0);
        result.is_valid_dm = (result.trace_deviation < 1e-6);
        return {result, MatrixXcd()};
    }
}

//==============================================================================
// Phase 1A: Iterative Compression
//==============================================================================

BenchmarkResult PhaseBenchmark::run_phase1a(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    const MatrixXcd& rho_baseline
) {
    BenchmarkResult result;
    result.label = "Phase1A-IterComp";
    result.num_qubits = num_qubits;
    result.circuit_depth = sequence.operations.size();

    PipelineConfig cfg;
    cfg.noise_strategy = NoiseStrategy::IterativeCompression;
    cfg.truncation_strategy = TruncationStrategy::GramEigen;
    cfg.gate_strategy = GateStrategy::RowParallel;
    cfg.truncation_threshold = config_.truncation_threshold;
    cfg.validate_output = false;
    cfg.run_tomography = false;
    cfg.verbose = false;

    auto start = std::chrono::steady_clock::now();

    OptimizedPipeline pipeline(num_qubits, cfg);
    PipelineResult pr = pipeline.run(L_init, sequence);

    auto end = std::chrono::steady_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(end - start).count();

    result.final_rank = pr.stats.final_rank;
    result.peak_rank = pr.stats.max_rank_reached;

    size_t dim = size_t(1) << num_qubits;
    result.memory_bytes_proxy = dim * result.final_rank * sizeof(Complex);

    // Fidelity
    if (config_.compute_fidelity && rho_baseline.rows() > 0 && dim <= 4096) {
        MatrixXcd rho = pr.L_final * pr.L_final.adjoint();
        result.fidelity_vs_baseline = compute_fidelity(rho, rho_baseline);
        result.trace_deviation = std::abs(rho.trace().real() - 1.0);
        result.is_valid_dm = validate_density_matrix(rho, 1e-6);
    } else {
        MatrixXcd G = pr.L_final.adjoint() * pr.L_final;
        result.trace_deviation = std::abs(G.trace().real() - 1.0);
        result.is_valid_dm = (result.trace_deviation < 1e-6);
    }

    return result;
}

//==============================================================================
// Phase 1B: DLRA
//==============================================================================

BenchmarkResult PhaseBenchmark::run_phase1b(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    const MatrixXcd& rho_baseline
) {
    BenchmarkResult result;
    result.label = "Phase1B-DLRA";
    result.num_qubits = num_qubits;
    result.circuit_depth = sequence.operations.size();

    PipelineConfig cfg;
    cfg.noise_strategy = NoiseStrategy::DLRA;
    cfg.truncation_strategy = TruncationStrategy::SVD;
    cfg.gate_strategy = GateStrategy::RowParallel;
    cfg.truncation_threshold = config_.truncation_threshold;
    cfg.validate_output = false;
    cfg.run_tomography = false;
    cfg.verbose = false;

    auto start = std::chrono::steady_clock::now();

    OptimizedPipeline pipeline(num_qubits, cfg);
    PipelineResult pr = pipeline.run(L_init, sequence);

    auto end = std::chrono::steady_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(end - start).count();

    result.final_rank = pr.stats.final_rank;
    result.peak_rank = pr.stats.max_rank_reached;

    size_t dim = size_t(1) << num_qubits;
    result.memory_bytes_proxy = dim * result.final_rank * sizeof(Complex);

    if (config_.compute_fidelity && rho_baseline.rows() > 0 && dim <= 4096) {
        MatrixXcd rho = pr.L_final * pr.L_final.adjoint();
        result.fidelity_vs_baseline = compute_fidelity(rho, rho_baseline);
        result.trace_deviation = std::abs(rho.trace().real() - 1.0);
        result.is_valid_dm = validate_density_matrix(rho, 1e-6);
    } else {
        MatrixXcd G = pr.L_final.adjoint() * pr.L_final;
        result.trace_deviation = std::abs(G.trace().real() - 1.0);
        result.is_valid_dm = (result.trace_deviation < 1e-6);
    }

    return result;
}

//==============================================================================
// Phase 2A: CP Decomposition
//==============================================================================

BenchmarkResult PhaseBenchmark::run_phase2a(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    const MatrixXcd& rho_baseline
) {
    BenchmarkResult result;
    result.label = "Phase2A-CP";
    result.num_qubits = num_qubits;
    result.circuit_depth = sequence.operations.size();

    PipelineConfig cfg;
    cfg.noise_strategy = NoiseStrategy::IterativeCompression;
    cfg.truncation_strategy = TruncationStrategy::CPDecomposition;
    cfg.gate_strategy = GateStrategy::RowParallel;
    cfg.truncation_threshold = config_.truncation_threshold;
    cfg.validate_output = false;
    cfg.run_tomography = false;
    cfg.verbose = false;

    auto start = std::chrono::steady_clock::now();

    OptimizedPipeline pipeline(num_qubits, cfg);
    PipelineResult pr = pipeline.run(L_init, sequence);

    auto end = std::chrono::steady_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(end - start).count();

    result.final_rank = pr.stats.final_rank;
    result.peak_rank = pr.stats.max_rank_reached;

    size_t dim = size_t(1) << num_qubits;
    result.memory_bytes_proxy = dim * result.final_rank * sizeof(Complex);

    if (config_.compute_fidelity && rho_baseline.rows() > 0 && dim <= 4096) {
        MatrixXcd rho = pr.L_final * pr.L_final.adjoint();
        result.fidelity_vs_baseline = compute_fidelity(rho, rho_baseline);
        result.trace_deviation = std::abs(rho.trace().real() - 1.0);
        result.is_valid_dm = validate_density_matrix(rho, 1e-6);
    } else {
        MatrixXcd G = pr.L_final.adjoint() * pr.L_final;
        result.trace_deviation = std::abs(G.trace().real() - 1.0);
        result.is_valid_dm = (result.trace_deviation < 1e-6);
    }

    return result;
}

//==============================================================================
// Phase 2B: Sparse Tensor
//==============================================================================

BenchmarkResult PhaseBenchmark::run_phase2b(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    const MatrixXcd& rho_baseline
) {
    BenchmarkResult result;
    result.label = "Phase2B-Sparse";
    result.num_qubits = num_qubits;
    result.circuit_depth = sequence.operations.size();

    PipelineConfig cfg;
    cfg.noise_strategy = NoiseStrategy::Sparse;
    cfg.truncation_strategy = TruncationStrategy::GramEigen;
    cfg.gate_strategy = GateStrategy::RowParallel;
    cfg.truncation_threshold = config_.truncation_threshold;
    cfg.validate_output = false;
    cfg.run_tomography = false;
    cfg.verbose = false;

    auto start = std::chrono::steady_clock::now();

    OptimizedPipeline pipeline(num_qubits, cfg);
    PipelineResult pr = pipeline.run(L_init, sequence);

    auto end = std::chrono::steady_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(end - start).count();

    result.final_rank = pr.stats.final_rank;
    result.peak_rank = pr.stats.max_rank_reached;

    size_t dim = size_t(1) << num_qubits;
    result.memory_bytes_proxy = dim * result.final_rank * sizeof(Complex);

    if (config_.compute_fidelity && rho_baseline.rows() > 0 && dim <= 4096) {
        MatrixXcd rho = pr.L_final * pr.L_final.adjoint();
        result.fidelity_vs_baseline = compute_fidelity(rho, rho_baseline);
        result.trace_deviation = std::abs(rho.trace().real() - 1.0);
        result.is_valid_dm = validate_density_matrix(rho, 1e-6);
    } else {
        MatrixXcd G = pr.L_final.adjoint() * pr.L_final;
        result.trace_deviation = std::abs(G.trace().real() - 1.0);
        result.is_valid_dm = (result.trace_deviation < 1e-6);
    }

    return result;
}

//==============================================================================
// Phase 6A: Unified Pipeline (Auto)
//==============================================================================

BenchmarkResult PhaseBenchmark::run_pipeline(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    const MatrixXcd& rho_baseline
) {
    BenchmarkResult result;
    result.label = "Phase6-Pipeline";
    result.num_qubits = num_qubits;
    result.circuit_depth = sequence.operations.size();

    PipelineConfig cfg;
    // All Auto — let the pipeline decide
    cfg.noise_strategy = NoiseStrategy::Auto;
    cfg.truncation_strategy = TruncationStrategy::Auto;
    cfg.gate_strategy = GateStrategy::Auto;
    cfg.truncation_threshold = config_.truncation_threshold;
    cfg.use_tuned_params = true;
    cfg.validate_output = true;
    cfg.run_tomography = false;
    cfg.verbose = false;

    auto start = std::chrono::steady_clock::now();

    OptimizedPipeline pipeline(num_qubits, cfg);
    PipelineResult pr = pipeline.run(L_init, sequence);

    auto end = std::chrono::steady_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(end - start).count();

    result.final_rank = pr.stats.final_rank;
    result.peak_rank = pr.stats.max_rank_reached;
    result.trace_deviation = pr.stats.trace_deviation;
    result.is_valid_dm = pr.stats.is_valid_dm;

    size_t dim = size_t(1) << num_qubits;
    result.memory_bytes_proxy = dim * result.final_rank * sizeof(Complex);

    if (config_.compute_fidelity && rho_baseline.rows() > 0 && dim <= 4096) {
        MatrixXcd rho = pr.L_final * pr.L_final.adjoint();
        result.fidelity_vs_baseline = compute_fidelity(rho, rho_baseline);
    }

    // Note which strategies were auto-selected
    result.notes = pr.stats.summary();

    return result;
}

//==============================================================================
// Run Single Configuration
//==============================================================================

std::vector<BenchmarkResult> PhaseBenchmark::run_single(
    size_t num_qubits,
    size_t depth,
    double noise_prob
) {
    std::vector<BenchmarkResult> results;

    size_t dim = size_t(1) << num_qubits;

    // Prepare initial |0...0><0...0| state as L = e_0
    MatrixXcd L_init = MatrixXcd::Zero(static_cast<Eigen::Index>(dim), 1);
    L_init(0, 0) = Complex(1.0, 0.0);

    // Generate circuit
    QuantumSequence sequence = generate_random_circuit(num_qubits, depth, noise_prob, 42);

    if (config_.verbose) {
        std::cout << "\n--- " << num_qubits << " qubits, depth=" << depth
                  << ", noise=" << noise_prob << " ---\n";
    }

    // Run baseline first
    auto [baseline_result, rho_baseline] = run_baseline(L_init, sequence, num_qubits);
    baseline_result.noise_probability = noise_prob;
    results.push_back(baseline_result);

    if (config_.verbose) {
        std::cout << baseline_result.one_line() << "\n";
    }

    // Phase 1A
    if (should_run("Phase1A") && noise_prob > 0) {
        auto r = run_phase1a(L_init, sequence, num_qubits, rho_baseline);
        r.noise_probability = noise_prob;
        results.push_back(r);
        if (config_.verbose) std::cout << r.one_line() << "\n";
    }

    // Phase 1B
    if (should_run("Phase1B") && noise_prob > 0) {
        auto r = run_phase1b(L_init, sequence, num_qubits, rho_baseline);
        r.noise_probability = noise_prob;
        results.push_back(r);
        if (config_.verbose) std::cout << r.one_line() << "\n";
    }

    // Phase 2A
    if (should_run("Phase2A")) {
        auto r = run_phase2a(L_init, sequence, num_qubits, rho_baseline);
        r.noise_probability = noise_prob;
        results.push_back(r);
        if (config_.verbose) std::cout << r.one_line() << "\n";
    }

    // Phase 2B
    if (should_run("Phase2B") && noise_prob > 0) {
        auto r = run_phase2b(L_init, sequence, num_qubits, rho_baseline);
        r.noise_probability = noise_prob;
        results.push_back(r);
        if (config_.verbose) std::cout << r.one_line() << "\n";
    }

    // Phase 6A: Unified Pipeline
    if (should_run("Pipeline")) {
        auto r = run_pipeline(L_init, sequence, num_qubits, rho_baseline);
        r.noise_probability = noise_prob;
        results.push_back(r);
        if (config_.verbose) std::cout << r.one_line() << "\n";
    }

    return results;
}

//==============================================================================
// Run All
//==============================================================================

std::vector<BenchmarkResult> PhaseBenchmark::run_all() {
    std::vector<BenchmarkResult> all_results;

    if (config_.verbose) {
        std::cout << "=== Phase Benchmark Suite ===\n";
        std::cout << "Qubits: ";
        for (auto q : config_.qubit_counts) std::cout << q << " ";
        std::cout << "\nDepths/q: ";
        for (auto d : config_.depths_per_qubit) std::cout << d << " ";
        std::cout << "\nNoise: ";
        for (auto n : config_.noise_probs) std::cout << n << " ";
        std::cout << "\n";
    }

    for (size_t nq : config_.qubit_counts) {
        for (size_t dpq : config_.depths_per_qubit) {
            size_t depth = dpq * nq;
            for (double np : config_.noise_probs) {
                for (size_t rep = 0; rep < config_.num_reps; ++rep) {
                    auto results = run_single(nq, depth, np);
                    all_results.insert(all_results.end(), results.begin(), results.end());
                }
            }
        }
    }

    return all_results;
}

//==============================================================================
// Print Table
//==============================================================================

void PhaseBenchmark::print_table(const std::vector<BenchmarkResult>& results) const {
    std::cout << "\n" << std::string(120, '=') << "\n";
    std::cout << std::left
              << std::setw(20) << "Method"
              << " | " << std::setw(4) << "Q"
              << " | " << std::setw(6) << "Depth"
              << " | " << std::setw(7) << "Noise"
              << " | " << std::setw(12) << "Time(s)"
              << " | " << std::setw(6) << "Rank"
              << " | " << std::setw(6) << "Peak"
              << " | " << std::setw(10) << "Fidelity"
              << " | " << std::setw(10) << "Trace Dev"
              << " | " << std::setw(5) << "Valid"
              << "\n";
    std::cout << std::string(120, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::left
                  << std::setw(20) << r.label
                  << " | " << std::setw(4) << r.num_qubits
                  << " | " << std::setw(6) << r.circuit_depth
                  << " | " << std::fixed << std::setprecision(3) << std::setw(7) << r.noise_probability
                  << " | " << std::scientific << std::setprecision(4) << std::setw(12) << r.elapsed_seconds
                  << " | " << std::setw(6) << r.final_rank
                  << " | " << std::setw(6) << r.peak_rank
                  << " | " << std::fixed << std::setprecision(6) << std::setw(10) << r.fidelity_vs_baseline
                  << " | " << std::scientific << std::setprecision(2) << std::setw(10) << r.trace_deviation
                  << " | " << std::setw(5) << (r.is_valid_dm ? "YES" : "NO")
                  << "\n";
    }
    std::cout << std::string(120, '=') << "\n";
}

//==============================================================================
// Save CSV
//==============================================================================

void PhaseBenchmark::save_csv(
    const std::vector<BenchmarkResult>& results,
    const std::string& path
) const {
    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "Warning: could not open " << path << " for writing\n";
        return;
    }

    out << "method,num_qubits,circuit_depth,noise_probability,"
        << "elapsed_seconds,final_rank,peak_rank,"
        << "fidelity_vs_baseline,trace_deviation,is_valid_dm,"
        << "memory_bytes_proxy\n";

    for (const auto& r : results) {
        out << r.label << ","
            << r.num_qubits << ","
            << r.circuit_depth << ","
            << std::fixed << std::setprecision(6) << r.noise_probability << ","
            << std::scientific << std::setprecision(8) << r.elapsed_seconds << ","
            << r.final_rank << ","
            << r.peak_rank << ","
            << std::fixed << std::setprecision(8) << r.fidelity_vs_baseline << ","
            << std::scientific << std::setprecision(8) << r.trace_deviation << ","
            << (r.is_valid_dm ? 1 : 0) << ","
            << r.memory_bytes_proxy << "\n";
    }

    out.close();
    if (config_.verbose) {
        std::cout << "Results saved to " << path << "\n";
    }
}

//==============================================================================
// Markdown Summary
//==============================================================================

std::string PhaseBenchmark::markdown_summary(
    const std::vector<BenchmarkResult>& results
) const {
    std::ostringstream md;
    md << std::fixed;

    md << "# Phase Benchmark Results\n\n";

    md << "| Method | Qubits | Depth | Noise | Time (s) | Rank | Fidelity | Valid |\n";
    md << "|--------|--------|-------|-------|----------|------|----------|-------|\n";

    for (const auto& r : results) {
        md << "| " << r.label
           << " | " << r.num_qubits
           << " | " << r.circuit_depth
           << " | " << std::setprecision(3) << r.noise_probability
           << " | " << std::setprecision(4) << r.elapsed_seconds
           << " | " << r.final_rank
           << " | " << std::setprecision(6) << r.fidelity_vs_baseline
           << " | " << (r.is_valid_dm ? "✓" : "✗")
           << " |\n";
    }

    md << "\n";

    // Compute speedup and rank reduction summaries per method
    // Group by (num_qubits, circuit_depth, noise) → find baseline
    std::map<std::string, std::vector<double>> speedups;
    std::map<std::string, std::vector<double>> fidelities;

    for (size_t i = 0; i < results.size(); ++i) {
        if (results[i].label == "Baseline") {
            double base_time = results[i].elapsed_seconds;
            // Look at subsequent results with same config
            for (size_t j = i + 1; j < results.size(); ++j) {
                if (results[j].label == "Baseline") break;
                if (base_time > 0) {
                    speedups[results[j].label].push_back(base_time / results[j].elapsed_seconds);
                }
                fidelities[results[j].label].push_back(results[j].fidelity_vs_baseline);
            }
        }
    }

    if (!speedups.empty()) {
        md << "## Summary\n\n";
        md << "| Method | Avg Speedup | Min Fidelity | Avg Fidelity |\n";
        md << "|--------|-------------|--------------|---------------|\n";

        for (const auto& [label, spds] : speedups) {
            double avg_spd = 0;
            for (double s : spds) avg_spd += s;
            avg_spd /= static_cast<double>(spds.size());

            const auto& fids = fidelities[label];
            double min_fid = 1.0, avg_fid = 0;
            for (double f : fids) {
                avg_fid += f;
                if (f < min_fid) min_fid = f;
            }
            avg_fid /= static_cast<double>(fids.size());

            md << "| " << label
               << " | " << std::setprecision(2) << avg_spd << "x"
               << " | " << std::setprecision(6) << min_fid
               << " | " << std::setprecision(6) << avg_fid
               << " |\n";
        }
    }

    return md.str();
}

}  // namespace qlret
