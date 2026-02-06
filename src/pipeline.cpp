/**
 * @file pipeline.cpp
 * @brief Implementation of the Unified Optimized Simulation Pipeline (Phase 6A)
 *
 * Ties together Phases 1-5 into a single, auto-configuring simulation engine.
 */

#include "pipeline.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cmath>

namespace qlret {

//==============================================================================
// Timer helper
//==============================================================================

namespace {

struct ScopedTimer {
    double& target;
    std::chrono::steady_clock::time_point start;

    explicit ScopedTimer(double& t)
        : target(t), start(std::chrono::steady_clock::now()) {}

    ~ScopedTimer() {
        auto end = std::chrono::steady_clock::now();
        target += std::chrono::duration<double>(end - start).count();
    }
};

} // anonymous namespace

//==============================================================================
// PipelineStats::summary
//==============================================================================

std::string PipelineStats::summary() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);

    ss << "=== Pipeline Summary ===\n";

    // Strategies
    auto noise_name = [](NoiseStrategy s) -> const char* {
        switch (s) {
            case NoiseStrategy::Standard:             return "Standard";
            case NoiseStrategy::IterativeCompression: return "IterativeCompression (Phase 1A)";
            case NoiseStrategy::DLRA:                 return "DLRA (Phase 1B)";
            case NoiseStrategy::Sparse:               return "Sparse (Phase 2B)";
            default:                                  return "Auto";
        }
    };
    auto trunc_name = [](TruncationStrategy s) -> const char* {
        switch (s) {
            case TruncationStrategy::GramEigen:       return "Gram Eigendecomp";
            case TruncationStrategy::CPDecomposition: return "CP Decomposition (Phase 2A)";
            case TruncationStrategy::SVD:             return "SVD (Phase 1B)";
            default:                                  return "Auto";
        }
    };
    auto gate_name = [](GateStrategy s) -> const char* {
        switch (s) {
            case GateStrategy::RowParallel:           return "Row-Parallel";
            case GateStrategy::MortonOrder:           return "Morton Order (Phase 4A)";
            default:                                  return "Auto";
        }
    };

    ss << "  Noise strategy:      " << noise_name(noise_strategy_used) << "\n";
    ss << "  Truncation strategy: " << trunc_name(truncation_strategy_used) << "\n";
    ss << "  Gate strategy:       " << gate_name(gate_strategy_used) << "\n";

    if (detected_pattern != CircuitPattern::UNKNOWN) {
        ss << "  Detected pattern:    " << circuit_pattern_name(detected_pattern) << "\n";
    }

    // Circuit info
    ss << "  Circuit: " << num_gates << " gates, " << num_noise_ops
       << " noise ops (ratio=" << noise_ratio << ")\n";

    // Timing
    ss << "  Total time:          " << total_time << "s\n";
    ss << "    Gate application:  " << gate_time << "s\n";
    ss << "    Noise handling:    " << noise_time << "s\n";
    ss << "    Truncation:        " << truncation_time << "s\n";
    if (tomography_time > 0) {
        ss << "    Tomography:        " << tomography_time << "s\n";
    }

    // Rank
    ss << "  Rank: " << initial_rank << " -> " << final_rank
       << " (max=" << max_rank_reached << ", truncations=" << truncation_count << ")\n";

    // Validation
    ss << "  Valid DM: " << (is_valid_dm ? "YES" : "NO")
       << " (Hermitian=" << is_hermitian << " PSD=" << is_psd
       << " trace_dev=" << trace_deviation << ")\n";

    // Tomography
    if (tomography_fidelity > 0) {
        ss << "  Tomography: fidelity=" << tomography_fidelity
           << " trace_dist=" << tomography_trace_distance
           << " measurements=" << tomography_measurements_used << "\n";
    }

    return ss.str();
}

//==============================================================================
// OptimizedPipeline Constructor
//==============================================================================

OptimizedPipeline::OptimizedPipeline(size_t num_qubits, const PipelineConfig& config)
    : num_qubits_(num_qubits)
    , dim_(size_t(1) << num_qubits)
    , config_(config)
{
    // Load tuned parameters if configured
    if (config_.use_tuned_params) {
        if (!config_.tuned_params_file.empty()) {
            try {
                tuned_params_ = TunedParameters::load_from_file(config_.tuned_params_file);
            } catch (...) {
                // Fall back to heuristic defaults
                tuned_params_ = TunedParameters();
            }
        }
        // Will be refined in run() once we know the circuit
    }
}

//==============================================================================
// Strategy Selection
//==============================================================================

NoiseStrategy OptimizedPipeline::select_noise_strategy(const QuantumSequence& sequence) const {
    if (config_.noise_strategy != NoiseStrategy::Auto) {
        return config_.noise_strategy;
    }

    // Count noise operations and analyze characteristics
    size_t noise_ops = 0;
    size_t total_ops = 0;
    double avg_noise_prob = 0.0;
    bool has_high_noise = false;

    for (const auto& op : sequence.operations) {
        ++total_ops;
        if (std::holds_alternative<NoiseOp>(op)) {
            ++noise_ops;
            const auto& noise = std::get<NoiseOp>(op);
            avg_noise_prob += noise.probability;
            if (noise.probability > 0.1) {
                has_high_noise = true;
            }
        }
    }

    if (noise_ops > 0) {
        avg_noise_prob /= static_cast<double>(noise_ops);
    }

    double noise_ratio = (total_ops > 0)
        ? static_cast<double>(noise_ops) / static_cast<double>(total_ops)
        : 0.0;

    // Decision tree:
    // High noise ratio (>50%) + high individual probabilities → Sparse mode
    if (noise_ratio > 0.5 && has_high_noise && num_qubits_ >= 6) {
        return NoiseStrategy::Sparse;
    }

    // Medium noise with structured circuits → DLRA
    if (noise_ratio > 0.2 && avg_noise_prob > 0.01 && num_qubits_ >= 8) {
        return NoiseStrategy::DLRA;
    }

    // Default: Iterative compression (best general-purpose)
    if (noise_ops > 0) {
        return NoiseStrategy::IterativeCompression;
    }

    // No noise: Standard (no Kraus ops to handle)
    return NoiseStrategy::Standard;
}

TruncationStrategy OptimizedPipeline::select_truncation_strategy(
    const QuantumSequence& sequence,
    NoiseStrategy resolved_noise
) const {
    if (config_.truncation_strategy != TruncationStrategy::Auto) {
        return config_.truncation_strategy;
    }

    // Check for CP-friendly patterns
    CircuitPattern pattern = detect_circuit_pattern(sequence);
    if (pattern == CircuitPattern::QFT || pattern == CircuitPattern::GROVER ||
        pattern == CircuitPattern::PERIODIC) {
        return TruncationStrategy::CPDecomposition;
    }

    // For DLRA noise strategy, use SVD truncation (they work together)
    if (resolved_noise == NoiseStrategy::DLRA) {
        return TruncationStrategy::SVD;
    }

    // Default: Gram eigendecomposition (fastest for unstructured circuits)
    return TruncationStrategy::GramEigen;
}

GateStrategy OptimizedPipeline::select_gate_strategy(const QuantumSequence& sequence) const {
    if (config_.gate_strategy != GateStrategy::Auto) {
        return config_.gate_strategy;
    }

    // Morton order is beneficial for large systems with high-stride gates
    if (num_qubits_ >= MortonOrderManager::MIN_QUBITS_FOR_MORTON) {
        // Check if circuit has high-stride gates
        size_t high_stride_count = 0;
        for (const auto& op : sequence.operations) {
            if (std::holds_alternative<GateOp>(op)) {
                const auto& gate = std::get<GateOp>(op);
                for (size_t q : gate.qubits) {
                    if (q >= MortonOrderManager::MIN_TARGET_FOR_MORTON) {
                        ++high_stride_count;
                    }
                }
            }
        }
        if (high_stride_count >= 4) {
            return GateStrategy::MortonOrder;
        }
    }

    return GateStrategy::RowParallel;
}

//==============================================================================
// Circuit Analysis
//==============================================================================

void OptimizedPipeline::compute_circuit_stats(
    const QuantumSequence& sequence,
    PipelineStats& stats
) const {
    stats.num_gates = 0;
    stats.num_noise_ops = 0;

    for (const auto& op : sequence.operations) {
        if (std::holds_alternative<GateOp>(op)) {
            ++stats.num_gates;
        } else if (std::holds_alternative<NoiseOp>(op)) {
            ++stats.num_noise_ops;
        }
    }

    size_t total = stats.num_gates + stats.num_noise_ops;
    stats.noise_ratio = (total > 0)
        ? static_cast<double>(stats.num_noise_ops) / static_cast<double>(total)
        : 0.0;

    stats.detected_pattern = detect_circuit_pattern(sequence);
}

PipelineStats OptimizedPipeline::analyze(const QuantumSequence& sequence) const {
    PipelineStats stats;
    compute_circuit_stats(sequence, stats);

    stats.noise_strategy_used = select_noise_strategy(sequence);
    stats.truncation_strategy_used = select_truncation_strategy(sequence, stats.noise_strategy_used);
    stats.gate_strategy_used = select_gate_strategy(sequence);

    return stats;
}

std::string OptimizedPipeline::strategy_description(const PipelineStats& stats) const {
    return stats.summary();
}

//==============================================================================
// Noise Application with Strategy
//==============================================================================

MatrixXcd OptimizedPipeline::apply_noise_with_strategy(
    const MatrixXcd& L,
    const NoiseOp& noise_op,
    NoiseStrategy strategy
) {
    switch (strategy) {
        case NoiseStrategy::IterativeCompression: {
            return apply_noise_iterative_simple(
                L, noise_op, num_qubits_,
                config_.truncation_threshold,
                config_.max_rank
            );
        }

        case NoiseStrategy::DLRA: {
            return apply_noise_dlra_simple(
                L, noise_op, num_qubits_,
                config_.truncation_threshold,
                config_.max_rank > 0 ? config_.max_rank : 0
            );
        }

        case NoiseStrategy::Sparse: {
            SparseConfig sparse_cfg;
            sparse_cfg.sparsity_threshold = config_.truncation_threshold * 0.01;
            sparse_cfg.min_qubits = 4;
            return apply_noise_sparse(L, noise_op, num_qubits_, sparse_cfg);
        }

        case NoiseStrategy::Standard:
        default: {
            MatrixXcd L_new = apply_noise_to_L(L, noise_op, num_qubits_);
            return L_new;
        }
    }
}

//==============================================================================
// Truncation with Strategy
//==============================================================================

MatrixXcd OptimizedPipeline::apply_truncation_with_strategy(
    const MatrixXcd& L,
    TruncationStrategy strategy
) {
    if (L.cols() <= 1) return L;

    switch (strategy) {
        case TruncationStrategy::CPDecomposition: {
            CPConfig cp_cfg;
            cp_cfg.target_rank = (config_.max_rank > 0)
                ? std::min(config_.max_rank, static_cast<size_t>(L.cols()))
                : std::min(static_cast<size_t>(L.cols()), size_t(16));
            cp_cfg.tolerance = config_.truncation_threshold;
            cp_cfg.max_iterations = 50;
            return truncate_cp(L, num_qubits_, cp_cfg);
        }

        case TruncationStrategy::SVD: {
            DLRAConfig dlra_cfg;
            dlra_cfg.threshold = config_.truncation_threshold;
            dlra_cfg.target_rank = (config_.max_rank > 0)
                ? config_.max_rank
                : compute_optimal_rank(L, config_.truncation_threshold);
            return truncate_dlra(L, dlra_cfg);
        }

        case TruncationStrategy::GramEigen:
        default: {
            return truncate_L(L, config_.truncation_threshold, config_.max_rank);
        }
    }
}

//==============================================================================
// Gate Application with Strategy
//==============================================================================

MatrixXcd OptimizedPipeline::apply_gate_with_strategy(
    const MatrixXcd& L,
    const GateOp& gate_op,
    GateStrategy strategy
) {
    // Morton order is applied at batch level, not individual gate level.
    // For individual gates, always use the standard apply_gate_to_L.
    // Morton batching is handled in execute() when accumulating gates.
    return apply_gate_to_L(L, gate_op, num_qubits_);
}

//==============================================================================
// Main Execution Loop
//==============================================================================

MatrixXcd OptimizedPipeline::execute(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    NoiseStrategy noise_strat,
    TruncationStrategy trunc_strat,
    GateStrategy gate_strat,
    PipelineStats& stats
) {
    MatrixXcd L = L_init;
    stats.initial_rank = static_cast<size_t>(L.cols());
    stats.max_rank_reached = stats.initial_rank;

    // Accumulate gates for batched Morton application
    std::vector<GateOp> gate_batch;

    auto flush_gate_batch = [&]() {
        if (gate_batch.empty()) return;

        ScopedTimer timer(stats.gate_time);

        if (gate_strat == GateStrategy::MortonOrder &&
            MortonOrderManager::should_use_morton_batch(gate_batch, num_qubits_)) {
            MortonOrderManager mom(dim_, static_cast<size_t>(L.cols()));
            L = mom.apply_gate_batch_morton(L, gate_batch, num_qubits_);
        } else {
            // Standard application
            for (const auto& g : gate_batch) {
                L = apply_gate_to_L(L, g, num_qubits_);
            }
        }

        gate_batch.clear();

        // Track max rank
        size_t r = static_cast<size_t>(L.cols());
        if (r > stats.max_rank_reached) {
            stats.max_rank_reached = r;
        }
    };

    size_t step = 0;
    for (const auto& op : sequence.operations) {
        ++step;

        if (std::holds_alternative<GateOp>(op)) {
            gate_batch.push_back(std::get<GateOp>(op));

            // Flush batch when it reaches batch size
            if (gate_batch.size() >= config_.batch_size) {
                flush_gate_batch();
            }
        }
        else if (std::holds_alternative<NoiseOp>(op)) {
            // Flush any pending gates first
            flush_gate_batch();

            const auto& noise = std::get<NoiseOp>(op);

            {
                ScopedTimer timer(stats.noise_time);
                L = apply_noise_with_strategy(L, noise, noise_strat);
            }

            // Track max rank after noise (rank may have grown)
            size_t r = static_cast<size_t>(L.cols());
            if (r > stats.max_rank_reached) {
                stats.max_rank_reached = r;
            }

            // Apply truncation if using Standard noise (others self-truncate)
            if (noise_strat == NoiseStrategy::Standard && L.cols() > 1) {
                ScopedTimer timer(stats.truncation_time);
                L = apply_truncation_with_strategy(L, trunc_strat);
                ++stats.truncation_count;
            }

            // Track max rank after truncation
            r = static_cast<size_t>(L.cols());
            if (r > stats.max_rank_reached) {
                stats.max_rank_reached = r;
            }
        }
        else if (std::holds_alternative<MeasurementOp>(op)) {
            flush_gate_batch();
            // Measurements are handled but we don't collapse state in pipeline mode
            // (would require probabilistic branching)
        }

        if (config_.verbose && step % 100 == 0) {
            std::cout << "  Pipeline step " << step << "/" << sequence.operations.size()
                      << ": rank=" << L.cols() << std::endl;
        }
    }

    // Flush remaining gates
    flush_gate_batch();

    // Final truncation
    if (L.cols() > 1) {
        ScopedTimer timer(stats.truncation_time);
        L = apply_truncation_with_strategy(L, trunc_strat);
        ++stats.truncation_count;
    }

    stats.final_rank = static_cast<size_t>(L.cols());

    return L;
}

//==============================================================================
// Validation
//==============================================================================

bool OptimizedPipeline::validate(const MatrixXcd& L, double tolerance) const {
    if (L.rows() == 0 || L.cols() == 0) return false;

    // Reconstruct ρ = L L†  (only for small systems)
    if (static_cast<size_t>(L.rows()) > 4096) {
        // For large systems, check properties via L directly
        // Tr[ρ] = Tr[L†L] = sum of squared singular values
        MatrixXcd G = L.adjoint() * L;
        double trace = G.trace().real();
        return std::abs(trace - 1.0) < tolerance;
    }

    MatrixXcd rho = L * L.adjoint();
    return validate_density_matrix(rho, tolerance);
}

//==============================================================================
// Main Run Method
//==============================================================================

PipelineResult OptimizedPipeline::run(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence
) {
    PipelineResult result;
    PipelineStats& stats = result.stats;
    auto total_start = std::chrono::steady_clock::now();

    // Step 1: Analyze circuit and select strategies
    {
        ScopedTimer timer(stats.strategy_selection_time);
        compute_circuit_stats(sequence, stats);

        stats.noise_strategy_used = select_noise_strategy(sequence);
        stats.truncation_strategy_used = select_truncation_strategy(sequence, stats.noise_strategy_used);
        stats.gate_strategy_used = select_gate_strategy(sequence);

        // Refine tuned parameters based on circuit characteristics
        if (config_.use_tuned_params && config_.tuned_params_file.empty()) {
            double avg_noise_prob = 0.0;
            if (stats.num_noise_ops > 0) {
                for (const auto& op : sequence.operations) {
                    if (std::holds_alternative<NoiseOp>(op)) {
                        avg_noise_prob += std::get<NoiseOp>(op).probability;
                    }
                }
                avg_noise_prob /= static_cast<double>(stats.num_noise_ops);
            }

            tuned_params_ = TunedParameters::get_optimal(
                num_qubits_,
                sequence.operations.size(),
                avg_noise_prob
            );

            // Apply tuned batch_size if using heuristic defaults
            if (config_.batch_size == 64) {
                config_.batch_size = tuned_params_.batch_size;
            }
        }
    }

    if (config_.verbose) {
        std::cout << "Pipeline: " << num_qubits_ << " qubits, "
                  << stats.num_gates << " gates, "
                  << stats.num_noise_ops << " noise ops" << std::endl;
        std::cout << strategy_description(stats);
    }

    // Step 2: Execute simulation
    result.L_final = execute(
        L_init, sequence,
        stats.noise_strategy_used,
        stats.truncation_strategy_used,
        stats.gate_strategy_used,
        stats
    );

    // Step 3: Validate output
    if (config_.validate_output) {
        ScopedTimer timer(stats.validation_time);

        // Compute trace via Gram matrix (always feasible)
        MatrixXcd G = result.L_final.adjoint() * result.L_final;
        double trace = G.trace().real();
        stats.trace_deviation = std::abs(trace - 1.0);

        // Full validation only for small systems
        if (dim_ <= 4096) {
            MatrixXcd rho = result.L_final * result.L_final.adjoint();

            // Hermitian check
            stats.is_hermitian = (rho - rho.adjoint()).norm() < config_.validation_tolerance;

            // PSD check
            Eigen::SelfAdjointEigenSolver<MatrixXcd> solver(rho);
            if (solver.info() == Eigen::Success) {
                double min_eigenval = solver.eigenvalues().minCoeff();
                stats.is_psd = (min_eigenval >= -config_.validation_tolerance);
            }

            stats.is_valid_dm = stats.is_hermitian && stats.is_psd &&
                                (stats.trace_deviation < config_.validation_tolerance);
        } else {
            // For large systems, assume Hermitian and PSD from LRET structure
            stats.is_hermitian = true;
            stats.is_psd = true;
            stats.is_valid_dm = (stats.trace_deviation < config_.validation_tolerance);
        }
    }

    // Step 4: Run tomography if requested
    if (config_.run_tomography && dim_ <= 65536) { // limit to 16 qubits
        ScopedTimer timer(stats.tomography_time);

        CompletionConfig tomo_cfg;
        tomo_cfg.solver = CompletionSolver::AlternatingProjection;
        tomo_cfg.max_iterations = 500;
        tomo_cfg.tolerance = 1e-8;
        tomo_cfg.enforce_dm_constraints = true;

        QuantumStateTomography tomo(num_qubits_, tomo_cfg);
        auto [rho_tomo, tomo_stats] = tomo.compressed_tomography_from_L(
            result.L_final, config_.tomography_fraction
        );

        result.rho_tomography = rho_tomo;
        result.tomography_stats = tomo_stats;

        // Compute fidelity with direct reconstruction
        if (dim_ <= 4096) {
            MatrixXcd rho_direct = result.L_final * result.L_final.adjoint();
            stats.tomography_fidelity = QuantumStateTomography::fidelity(rho_tomo, rho_direct);
            stats.tomography_trace_distance = QuantumStateTomography::trace_distance(rho_tomo, rho_direct);
        }

        // Count measurements used
        size_t total_paulis = 1;
        for (size_t i = 0; i < num_qubits_; ++i) total_paulis *= 4;
        stats.tomography_measurements_used = static_cast<size_t>(
            std::ceil(config_.tomography_fraction * static_cast<double>(total_paulis))
        );
    }

    // Record total time
    auto total_end = std::chrono::steady_clock::now();
    stats.total_time = std::chrono::duration<double>(total_end - total_start).count();

    return result;
}

//==============================================================================
// Convenience Functions
//==============================================================================

PipelineResult run_optimized_pipeline(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    double threshold,
    bool verbose
) {
    PipelineConfig config;
    config.truncation_threshold = threshold;
    config.verbose = verbose;
    config.run_tomography = false;

    OptimizedPipeline pipeline(num_qubits, config);
    return pipeline.run(L_init, sequence);
}

std::pair<double, PipelineResult> run_and_validate_pipeline(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    double threshold
) {
    // Run optimized pipeline
    PipelineConfig config;
    config.truncation_threshold = threshold;
    config.verbose = false;

    OptimizedPipeline pipeline(num_qubits, config);
    PipelineResult result = pipeline.run(L_init, sequence);

    // Run naive simulation for comparison
    MatrixXcd L_naive = run_simulation_optimized(
        L_init, sequence, num_qubits,
        64, true, false, threshold
    );

    // Compute fidelity between results
    size_t dim = size_t(1) << num_qubits;
    double fidelity = 0.0;

    if (dim <= 4096) {
        MatrixXcd rho_opt = result.L_final * result.L_final.adjoint();
        MatrixXcd rho_naive = L_naive * L_naive.adjoint();
        fidelity = QuantumStateTomography::fidelity(rho_opt, rho_naive);
    } else {
        // For large systems, compare Gram matrices as proxy
        MatrixXcd G_opt = result.L_final.adjoint() * result.L_final;
        MatrixXcd G_naive = L_naive.adjoint() * L_naive;

        // Rough fidelity proxy from Gram matrix overlap
        double norm_diff = (G_opt - G_naive).norm();
        double norm_sum = G_opt.norm() + G_naive.norm();
        fidelity = (norm_sum > 0) ? std::max(0.0, 1.0 - norm_diff / norm_sum) : 1.0;
    }

    return {fidelity, result};
}

}  // namespace qlret
