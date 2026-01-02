#include "benchmark_runner.h"
#include "simulator.h"
#include "fdm_simulator.h"
#include "parallel_modes.h"
#include "utils.h"
#include "gates_and_noise.h"
#include "resource_monitor.h"
#include "structured_csv.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numeric>

namespace qlret {

//==============================================================================
// Memory Utilities
//==============================================================================

size_t compute_L_memory_bytes(const MatrixXcd& L) {
    // L is dim × rank matrix of complex<double>
    return L.rows() * L.cols() * sizeof(Complex);
}

size_t compute_fdm_memory_bytes(size_t num_qubits) {
    // ρ is 2^n × 2^n matrix of complex<double>
    size_t dim = 1ULL << num_qubits;
    return dim * dim * sizeof(Complex);
}

MemoryComparison create_memory_comparison(
    const MatrixXcd& L_final,
    size_t num_qubits,
    bool fdm_was_run,
    size_t fdm_peak_bytes
) {
    MemoryComparison mc;
    mc.num_qubits = num_qubits;
    mc.lret_L_matrix_bytes = compute_L_memory_bytes(L_final);
    mc.lret_final_rank = L_final.cols();
    mc.lret_peak_bytes = mc.lret_L_matrix_bytes;  // Approximate
    
    // Estimate work memory for SVD (O(rank^2) for Gram matrix)
    mc.lret_work_memory_bytes = L_final.cols() * L_final.cols() * sizeof(Complex);
    
    // FDM memory
    mc.fdm_rho_matrix_bytes = compute_fdm_memory_bytes(num_qubits);
    mc.fdm_peak_bytes = fdm_was_run ? fdm_peak_bytes : mc.fdm_rho_matrix_bytes;
    
    return mc;
}

//==============================================================================
// Timing Analysis
//==============================================================================

TimingBreakdown create_timing_breakdown(const RankEvolution& evolution) {
    TimingBreakdown tb;
    
    for (const auto& event : evolution.events) {
        if (event.operation_type == "gate") {
            tb.gate_time += event.time_seconds;
            tb.gate_count++;
        } else if (event.operation_type == "noise" || event.operation_type == "kraus") {
            tb.noise_time += event.time_seconds;
            tb.noise_count++;
        } else if (event.operation_type == "truncation") {
            tb.truncation_time += event.time_seconds;
            tb.truncation_count++;
        }
    }
    
    if (!evolution.events.empty()) {
        tb.total_time = evolution.events.back().cumulative_time;
    }
    
    tb.compute_overhead();
    return tb;
}

std::string format_timing_breakdown(const TimingBreakdown& timing) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "Timing Breakdown:\n";
    oss << "  Gate application:  " << timing.gate_time << "s (" 
        << std::setprecision(1) << timing.gate_percent() << "%)\n";
    oss << "  Noise/Kraus:       " << std::setprecision(4) << timing.noise_time << "s (" 
        << std::setprecision(1) << timing.noise_percent() << "%)\n";
    oss << "  Truncation/SVD:    " << std::setprecision(4) << timing.truncation_time << "s (" 
        << std::setprecision(1) << timing.truncation_percent() << "%)\n";
    oss << "  Overhead:          " << std::setprecision(4) << timing.overhead_time << "s (" 
        << std::setprecision(1) << timing.overhead_percent() << "%)\n";
    oss << "  Total:             " << timing.total_time << "s\n";
    return oss.str();
}

//==============================================================================
// Statistical Analysis
//==============================================================================

TrialStats compute_trial_stats(const std::vector<double>& values) {
    TrialStats stats;
    if (values.empty()) return stats;
    
    stats.count = values.size();
    
    // Mean
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    stats.mean = sum / values.size();
    
    // Min/max
    auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
    stats.min_val = *min_it;
    stats.max_val = *max_it;
    
    // Standard deviation
    if (values.size() > 1) {
        double sq_sum = 0.0;
        for (double v : values) {
            sq_sum += (v - stats.mean) * (v - stats.mean);
        }
        stats.stddev = std::sqrt(sq_sum / (values.size() - 1));
    }
    
    return stats;
}

//==============================================================================
// Rank Evolution Tracking Simulation
//==============================================================================

RankEvolution run_with_rank_tracking(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    const SimConfig& config
) {
    RankEvolution evolution;
    evolution.initial_rank = L_init.cols();
    
    MatrixXcd L = L_init;
    size_t step = 0;
    size_t layer = 0;
    auto sim_start = std::chrono::steady_clock::now();
    
    for (const auto& op : sequence.operations) {
        step++;
        
        // Check for abort
        if (should_abort()) break;
        
        size_t rank_before = L.cols();
        auto op_start = std::chrono::steady_clock::now();
        
        if (std::holds_alternative<GateOp>(op)) {
            const auto& gate = std::get<GateOp>(op);
            L = apply_gate_to_L(L, gate, num_qubits);
            
            auto op_end = std::chrono::steady_clock::now();
            double op_time = std::chrono::duration<double>(op_end - op_start).count();
            double cumulative = std::chrono::duration<double>(op_end - sim_start).count();
            
            // Build operation name
            std::string op_name = gate_type_to_string(gate.type);
            if (gate.qubits.size() == 1) {
                op_name += "(" + std::to_string(gate.qubits[0]) + ")";
            } else if (gate.qubits.size() == 2) {
                op_name += "(" + std::to_string(gate.qubits[0]) + "," 
                         + std::to_string(gate.qubits[1]) + ")";
            }
            
            evolution.add_event(RankEvent(
                step, layer, "gate", op_name,
                rank_before, L.cols(), op_time, cumulative,
                compute_L_memory_bytes(L)
            ));
            
        } else {
            const auto& noise = std::get<NoiseOp>(op);
            L = apply_noise_to_L(L, noise, num_qubits);
            
            auto op_end = std::chrono::steady_clock::now();
            double op_time = std::chrono::duration<double>(op_end - op_start).count();
            double cumulative = std::chrono::duration<double>(op_end - sim_start).count();
            
            std::string op_name = noise_type_to_string(noise.type);
            op_name += "(" + std::to_string(noise.qubits[0]) + ")";
            
            evolution.add_event(RankEvent(
                step, layer, "noise", op_name,
                rank_before, L.cols(), op_time, cumulative,
                compute_L_memory_bytes(L)
            ));
            
            // Truncation after noise
            if (config.do_truncation && L.cols() > 1) {
                size_t rank_before_trunc = L.cols();
                auto trunc_start = std::chrono::steady_clock::now();
                L = truncate_L(L, config.truncation_threshold);
                auto trunc_end = std::chrono::steady_clock::now();
                
                double trunc_time = std::chrono::duration<double>(trunc_end - trunc_start).count();
                double cumulative_trunc = std::chrono::duration<double>(trunc_end - sim_start).count();
                
                if (L.cols() < rank_before_trunc) {
                    evolution.add_event(RankEvent(
                        step, layer, "truncation", "truncate",
                        rank_before_trunc, L.cols(), trunc_time, cumulative_trunc,
                        compute_L_memory_bytes(L)
                    ));
                }
            }
            
            layer++;  // Layer increments after noise (end of circuit layer)
        }
    }
    
    evolution.final_rank = L.cols();
    evolution.compute_average_rank();
    
    return evolution;
}

//==============================================================================
// Single Benchmark Point
//==============================================================================

SweepPointResult run_single_benchmark(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    const SimConfig& config,
    bool track_rank,
    bool include_fdm
) {
    SweepPointResult result;
    result.num_qubits = num_qubits;
    result.depth = sequence.depth;
    result.epsilon = config.truncation_threshold;
    result.noise_prob = sequence.total_noise_probability / std::max(size_t(1), sequence.size());
    
    // Run LRET simulation
    Timer lret_timer;
    MatrixXcd L_final;
    
    if (track_rank) {
        result.rank_evolution = run_with_rank_tracking(L_init, sequence, num_qubits, config);
        L_final = L_init;  // Need to re-run to get final state
        
        // Extract timing breakdown
        TimingBreakdown timing = create_timing_breakdown(result.rank_evolution);
        result.gate_time = timing.gate_time;
        result.noise_time = timing.noise_time;
        result.truncation_time = timing.truncation_time;
        result.truncation_count = timing.truncation_count;
        
        // Re-run simulation to get final L (rank tracking version is for monitoring)
        L_final = run_simulation_optimized(
            L_init, sequence, num_qubits,
            config.batch_size, config.do_truncation,
            false, config.truncation_threshold
        );
        
        result.lret_time_seconds = lret_timer.elapsed_seconds();
    } else {
        // Run with timing instrumentation (without full rank tracking)
        L_final = run_simulation_with_timing(
            L_init, sequence, num_qubits, config,
            result.gate_time, result.noise_time, 
            result.truncation_time, result.truncation_count
        );
        result.lret_time_seconds = lret_timer.elapsed_seconds();
    }
    
    result.lret_final_rank = L_final.cols();
    result.lret_memory_bytes = compute_L_memory_bytes(L_final);
    
    // Compute state metrics
    result.lret_purity = compute_purity(L_final);
    result.lret_entropy = compute_entropy(L_final);
    result.lret_linear_entropy = compute_linear_entropy(L_final);
    
    // Negativity for bipartite split (half-half)
    if (num_qubits >= 2) {
        size_t split = num_qubits / 2;
        result.lret_negativity = compute_negativity(L_final, split, num_qubits);
    }
    
    // Run FDM if requested and feasible
    if (include_fdm) {
        auto fdm_check = check_fdm_feasibility(num_qubits, true, false);
        if (fdm_check.should_run) {
            try {
                // Convert L_init to density matrix for FDM (same initial state!)
                MatrixXcd rho_init = L_init * L_init.adjoint();
                auto fdm_result = run_fdm_simulation(sequence, num_qubits, rho_init, false);
                result.fdm_run = fdm_result.was_run;
                result.fdm_time_seconds = fdm_result.time_seconds;
                result.fdm_memory_bytes = compute_fdm_memory_bytes(num_qubits);
                
                // Compute LRET vs FDM comparison metrics
                if (fdm_result.was_run) {
                    MatrixXcd rho_lret = L_final * L_final.adjoint();
                    result.fidelity_vs_fdm = compute_fidelity_rho(rho_lret, fdm_result.rho_final);
                    result.trace_distance_vs_fdm = compute_trace_distance_rho(rho_lret, fdm_result.rho_final);
                }
            } catch (...) {
                result.fdm_run = false;
            }
        }
    }
    
    return result;
}

//==============================================================================
// All Modes Benchmark (runs sequential, row, column, hybrid, adaptive)
//==============================================================================

std::vector<ModePointResult> run_all_modes_benchmark(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    const SimConfig& config,
    size_t batch_size,
    bool include_fdm,
    const MatrixXcd* fdm_rho_final  // Optional: pre-computed FDM result
) {
    std::vector<ModePointResult> results;
    
    // Define modes to run
    std::vector<ParallelMode> modes = {
        ParallelMode::SEQUENTIAL,
        ParallelMode::ROW,
        ParallelMode::COLUMN,
        ParallelMode::BATCH,
        ParallelMode::HYBRID
    };
    
    double seq_time = 0.0;  // For computing speedup
    
    for (auto mode : modes) {
        if (should_abort()) break;
        
        ModePointResult mpr;
        mpr.mode_name = parallel_mode_to_string(mode);
        
        // Run this mode
        auto mode_result = run_with_mode(L_init, sequence, num_qubits, mode, config, batch_size);
        
        mpr.time_seconds = mode_result.time_seconds;
        mpr.final_rank = mode_result.final_rank;
        mpr.purity = compute_purity(mode_result.L_final);
        mpr.entropy = compute_entropy(mode_result.L_final);
        
        // Compute speedup vs sequential
        if (mode == ParallelMode::SEQUENTIAL) {
            seq_time = mpr.time_seconds;
            mpr.speedup_vs_seq = 1.0;
        } else if (seq_time > 0) {
            mpr.speedup_vs_seq = seq_time / mpr.time_seconds;
        }
        
        // Compute fidelity vs FDM if available
        if (include_fdm && fdm_rho_final) {
            MatrixXcd rho_lret = mode_result.L_final * mode_result.L_final.adjoint();
            mpr.fidelity_vs_fdm = compute_fidelity_rho(rho_lret, *fdm_rho_final);
            mpr.trace_distance_vs_fdm = compute_trace_distance_rho(rho_lret, *fdm_rho_final);
        }
        
        results.push_back(mpr);
    }
    
    return results;
}

//==============================================================================
// Parameter Sweep Runners
//==============================================================================

SweepResults run_epsilon_sweep(
    const CLIOptions& opts,
    const std::vector<double>& epsilon_values
) {
    SweepResults results;
    results.type = SweepType::EPSILON;
    results.sweep_parameter_name = "epsilon";
    results.start_time = std::chrono::system_clock::now();
    
    // Generate circuit once (same circuit for all epsilon values)
    auto sequence = generate_quantum_sequences(
        opts.num_qubits, opts.depth, true, opts.noise_prob, opts.random_seed
    );
    
    // Initial state
    MatrixXcd L_init;
    if (opts.initial_rank > 1) {
        L_init = create_random_mixed_state(opts.num_qubits, opts.initial_rank, opts.random_seed);
    } else {
        L_init = create_zero_state(opts.num_qubits);
    }
    
    std::cout << "Running epsilon sweep: " << epsilon_values.size() << " points\n";
    std::cout << "  Fixed: n=" << opts.num_qubits << ", d=" << opts.depth 
              << ", noise=" << opts.noise_prob << "\n\n";
    
    for (size_t i = 0; i < epsilon_values.size(); ++i) {
        double eps = epsilon_values[i];
        
        std::cout << "  [" << (i+1) << "/" << epsilon_values.size() << "] "
                  << "epsilon=" << std::scientific << std::setprecision(1) << eps 
                  << " ... " << std::flush;
        
        SimConfig config;
        config.truncation_threshold = eps;
        config.do_truncation = true;
        config.batch_size = opts.batch_size ? opts.batch_size : auto_select_batch_size(opts.num_qubits);
        
        // Cache FDM result for first trial (only run once per parameter point)
        SweepPointResult first_trial_result;
        MatrixXcd fdm_rho_storage;
        bool have_fdm = false;
        
        // Run each trial and output individually (no averaging - let Excel handle it)
        for (size_t trial = 0; trial < opts.sweep_trials; ++trial) {
            auto point = run_single_benchmark(
                L_init, sequence, opts.num_qubits, config,
                opts.track_rank_evolution && trial == 0,  // Track rank only on first trial
                opts.enable_fdm && trial == 0             // FDM only on first trial
            );
            
            point.trial_id = trial;
            point.total_trials = opts.sweep_trials;
            point.epsilon = eps;
            
            // For trials after first, copy FDM results from first trial
            if (trial == 0) {
                first_trial_result = point;
                point.fdm_executed = point.fdm_run;  // FDM actually executed on trial 0
                if (point.fdm_run) {
                    have_fdm = true;
                    // Store FDM rho for all-modes comparison
                    MatrixXcd rho_init = L_init * L_init.adjoint();
                    auto fdm_result = run_fdm_simulation(sequence, opts.num_qubits, rho_init, false);
                    if (fdm_result.was_run) {
                        fdm_rho_storage = fdm_result.rho_final;
                    }
                }
            } else if (have_fdm) {
                // Copy FDM metrics from first trial (FDM is deterministic)
                point.fdm_run = true;
                point.fdm_executed = false;  // FDM was NOT executed this trial, just copied
                point.fdm_time_seconds = first_trial_result.fdm_time_seconds;
                point.fdm_memory_bytes = first_trial_result.fdm_memory_bytes;
                point.fidelity_vs_fdm = first_trial_result.fidelity_vs_fdm;
                point.trace_distance_vs_fdm = first_trial_result.trace_distance_vs_fdm;
            }
            
            // Run all LRET modes for EVERY trial (needed for proper statistical analysis)
            if (opts.sweep_config.run_all_modes) {
                size_t batch = opts.batch_size ? opts.batch_size : auto_select_batch_size(opts.num_qubits);
                const MatrixXcd* fdm_rho = have_fdm ? &fdm_rho_storage : nullptr;
                point.all_modes_results = run_all_modes_benchmark(
                    L_init, sequence, opts.num_qubits, config, batch, opts.enable_fdm, fdm_rho
                );
            }
            
            results.add_point(point);
        }
        
        // Console output (summary for this epsilon value)
        std::cout << std::fixed << std::setprecision(4) << first_trial_result.lret_time_seconds << "s";
        if (opts.sweep_trials > 1) {
            std::cout << " (" << opts.sweep_trials << " trials)";
        }
        std::cout << ", rank=" << first_trial_result.lret_final_rank;
        if (first_trial_result.fdm_run) {
            std::cout << ", speedup=" << std::setprecision(1) << first_trial_result.speedup_vs_fdm() << "x";
        }
        if (!first_trial_result.all_modes_results.empty()) {
            std::cout << ", modes=" << first_trial_result.all_modes_results.size();
        }
        std::cout << "\n";
    }
    
    results.end_time = std::chrono::system_clock::now();
    results.total_wall_time_seconds = std::chrono::duration<double>(
        results.end_time - results.start_time).count();
    
    return results;
}

SweepResults run_noise_sweep(
    const CLIOptions& opts,
    const std::vector<double>& noise_values
) {
    SweepResults results;
    results.type = SweepType::NOISE_PROB;
    results.sweep_parameter_name = "noise_prob";
    results.start_time = std::chrono::system_clock::now();
    
    // Initial state
    MatrixXcd L_init;
    if (opts.initial_rank > 1) {
        L_init = create_random_mixed_state(opts.num_qubits, opts.initial_rank, opts.random_seed);
    } else {
        L_init = create_zero_state(opts.num_qubits);
    }
    
    std::cout << "Running noise probability sweep: " << noise_values.size() << " points\n";
    std::cout << "  Fixed: n=" << opts.num_qubits << ", d=" << opts.depth 
              << ", epsilon=" << std::scientific << opts.truncation_threshold << "\n\n";
    
    for (size_t i = 0; i < noise_values.size(); ++i) {
        double noise = noise_values[i];
        
        std::cout << "  [" << (i+1) << "/" << noise_values.size() << "] "
                  << "noise=" << std::fixed << std::setprecision(3) << noise 
                  << " ... " << std::flush;
        
        // Generate new circuit with this noise level
        auto sequence = generate_quantum_sequences(
            opts.num_qubits, opts.depth, true, noise, opts.random_seed
        );
        
        SimConfig config;
        config.truncation_threshold = opts.truncation_threshold;
        config.do_truncation = true;
        config.batch_size = opts.batch_size ? opts.batch_size : auto_select_batch_size(opts.num_qubits);
        
        // Cache FDM result for first trial (only run once per parameter point)
        SweepPointResult first_trial_result;
        MatrixXcd fdm_rho_storage;
        bool have_fdm = false;
        
        // Run each trial and output individually (no averaging - let Excel handle it)
        for (size_t trial = 0; trial < opts.sweep_trials; ++trial) {
            auto point = run_single_benchmark(
                L_init, sequence, opts.num_qubits, config,
                opts.track_rank_evolution && trial == 0,
                opts.enable_fdm && trial == 0
            );
            
            point.trial_id = trial;
            point.total_trials = opts.sweep_trials;
            point.noise_prob = noise;
            
            // For trials after first, copy FDM results from first trial
            if (trial == 0) {
                first_trial_result = point;
                point.fdm_executed = point.fdm_run;
                if (point.fdm_run) {
                    have_fdm = true;
                    MatrixXcd rho_init = L_init * L_init.adjoint();
                    auto fdm_result = run_fdm_simulation(sequence, opts.num_qubits, rho_init, false);
                    if (fdm_result.was_run) {
                        fdm_rho_storage = fdm_result.rho_final;
                    }
                }
            } else if (have_fdm) {
                point.fdm_run = true;
                point.fdm_executed = false;
                point.fdm_time_seconds = first_trial_result.fdm_time_seconds;
                point.fdm_memory_bytes = first_trial_result.fdm_memory_bytes;
                point.fidelity_vs_fdm = first_trial_result.fidelity_vs_fdm;
                point.trace_distance_vs_fdm = first_trial_result.trace_distance_vs_fdm;
            }
            
            // Run all LRET modes for EVERY trial
            if (opts.sweep_config.run_all_modes) {
                size_t batch = opts.batch_size ? opts.batch_size : auto_select_batch_size(opts.num_qubits);
                const MatrixXcd* fdm_rho = have_fdm ? &fdm_rho_storage : nullptr;
                point.all_modes_results = run_all_modes_benchmark(
                    L_init, sequence, opts.num_qubits, config, batch, opts.enable_fdm, fdm_rho
                );
            }
            
            results.add_point(point);
        }
        
        // Console output
        std::cout << std::fixed << std::setprecision(4) << first_trial_result.lret_time_seconds << "s";
        if (opts.sweep_trials > 1) std::cout << " (" << opts.sweep_trials << " trials)";
        std::cout << ", rank=" << first_trial_result.lret_final_rank;
        if (first_trial_result.fdm_run) {
            std::cout << ", FDM=" << first_trial_result.fdm_time_seconds << "s"
                      << ", speedup=" << std::setprecision(1) << first_trial_result.speedup_vs_fdm() << "x";
        }
        if (!first_trial_result.all_modes_results.empty()) {
            std::cout << ", modes=" << first_trial_result.all_modes_results.size();
        }
        std::cout << "\n";
    }
    
    results.end_time = std::chrono::system_clock::now();
    results.total_wall_time_seconds = std::chrono::duration<double>(
        results.end_time - results.start_time).count();
    
    return results;
}

SweepResults run_rank_sweep(
    const CLIOptions& opts,
    const std::vector<size_t>& rank_values
) {
    SweepResults results;
    results.type = SweepType::INITIAL_RANK;
    results.sweep_parameter_name = "initial_rank";
    results.start_time = std::chrono::system_clock::now();
    
    std::cout << "Running initial rank sweep: " << rank_values.size() << " points\n";
    std::cout << "  Fixed: n=" << opts.num_qubits << ", d=" << opts.depth 
              << ", noise=" << opts.noise_prob 
              << ", epsilon=" << std::scientific << opts.truncation_threshold << "\n\n";
    
    // Generate circuit once (same for all rank values)
    auto sequence = generate_quantum_sequences(
        opts.num_qubits, opts.depth, true, opts.noise_prob, opts.random_seed
    );
    
    SimConfig config;
    config.truncation_threshold = opts.truncation_threshold;
    config.do_truncation = true;
    config.batch_size = opts.batch_size ? opts.batch_size : auto_select_batch_size(opts.num_qubits);
    
    size_t max_rank = 1ULL << opts.num_qubits;
    
    for (size_t i = 0; i < rank_values.size(); ++i) {
        size_t rank = rank_values[i];
        size_t actual_rank = std::min(rank, max_rank);
        
        std::cout << "  [" << (i+1) << "/" << rank_values.size() << "] "
                  << "rank=" << std::setw(4) << actual_rank << " ... " << std::flush;
        
        // Create initial state with specified rank
        MatrixXcd L_init;
        if (actual_rank > 1) {
            L_init = create_random_mixed_state(opts.num_qubits, actual_rank, opts.random_seed);
        } else {
            L_init = create_zero_state(opts.num_qubits);
        }
        
        // Cache FDM result for first trial
        SweepPointResult first_trial_result;
        MatrixXcd fdm_rho_storage;
        bool have_fdm = false;
        
        // Run each trial and output individually
        for (size_t trial = 0; trial < opts.sweep_trials; ++trial) {
            auto point = run_single_benchmark(
                L_init, sequence, opts.num_qubits, config,
                opts.track_rank_evolution && trial == 0,
                opts.enable_fdm && trial == 0
            );
            
            point.trial_id = trial;
            point.total_trials = opts.sweep_trials;
            point.initial_rank = actual_rank;
            
            if (trial == 0) {
                first_trial_result = point;
                point.fdm_executed = point.fdm_run;
                if (point.fdm_run) {
                    have_fdm = true;
                    MatrixXcd rho_init = L_init * L_init.adjoint();
                    auto fdm_result = run_fdm_simulation(sequence, opts.num_qubits, rho_init, false);
                    if (fdm_result.was_run) {
                        fdm_rho_storage = fdm_result.rho_final;
                    }
                }
            } else if (have_fdm) {
                point.fdm_run = true;
                point.fdm_executed = false;
                point.fdm_time_seconds = first_trial_result.fdm_time_seconds;
                point.fdm_memory_bytes = first_trial_result.fdm_memory_bytes;
                point.fidelity_vs_fdm = first_trial_result.fidelity_vs_fdm;
                point.trace_distance_vs_fdm = first_trial_result.trace_distance_vs_fdm;
            }
            
            if (opts.sweep_config.run_all_modes) {
                size_t batch = opts.batch_size ? opts.batch_size : auto_select_batch_size(opts.num_qubits);
                const MatrixXcd* fdm_rho = have_fdm ? &fdm_rho_storage : nullptr;
                point.all_modes_results = run_all_modes_benchmark(
                    L_init, sequence, opts.num_qubits, config, batch, opts.enable_fdm, fdm_rho
                );
            }
            
            results.add_point(point);
        }
        
        std::cout << std::fixed << std::setprecision(4) << first_trial_result.lret_time_seconds << "s";
        if (opts.sweep_trials > 1) std::cout << " (" << opts.sweep_trials << " trials)";
        std::cout << ", final_rank=" << first_trial_result.lret_final_rank;
        if (first_trial_result.fdm_run) {
            std::cout << ", FDM=" << first_trial_result.fdm_time_seconds << "s"
                      << ", speedup=" << std::setprecision(1) << first_trial_result.speedup_vs_fdm() << "x";
        }
        if (!first_trial_result.all_modes_results.empty()) {
            std::cout << ", modes=" << first_trial_result.all_modes_results.size();
        }
        std::cout << "\n";
    }
    
    results.end_time = std::chrono::system_clock::now();
    results.total_wall_time_seconds = std::chrono::duration<double>(
        results.end_time - results.start_time).count();
    
    return results;
}

SweepResults run_qubit_sweep(
    const CLIOptions& opts,
    const std::vector<size_t>& qubit_values
) {
    SweepResults results;
    results.type = SweepType::QUBITS;
    results.sweep_parameter_name = "num_qubits";
    results.start_time = std::chrono::system_clock::now();
    
    std::cout << "Running qubit count sweep: " << qubit_values.size() << " points\n";
    std::cout << "  Fixed: d=" << opts.depth << ", noise=" << opts.noise_prob 
              << ", epsilon=" << std::scientific << opts.truncation_threshold << "\n\n";
    
    for (size_t i = 0; i < qubit_values.size(); ++i) {
        size_t n = qubit_values[i];
        
        std::cout << "  [" << (i+1) << "/" << qubit_values.size() << "] "
                  << "n=" << std::setw(2) << n << " qubits ... " << std::flush;
        
        // Generate circuit and initial state for this qubit count
        auto sequence = generate_quantum_sequences(
            n, opts.depth, true, opts.noise_prob, opts.random_seed
        );
        
        MatrixXcd L_init;
        if (opts.initial_rank > 1) {
            size_t max_rank = 1ULL << n;
            size_t actual_rank = std::min(opts.initial_rank, max_rank);
            L_init = create_random_mixed_state(n, actual_rank, opts.random_seed);
        } else {
            L_init = create_zero_state(n);
        }
        
        SimConfig config;
        config.truncation_threshold = opts.truncation_threshold;
        config.do_truncation = true;
        config.batch_size = opts.batch_size ? opts.batch_size : auto_select_batch_size(n);
        
        // Cache FDM result for first trial
        SweepPointResult first_trial_result;
        MatrixXcd fdm_rho_storage;
        bool have_fdm = false;
        
        // Run each trial and output individually
        for (size_t trial = 0; trial < opts.sweep_trials; ++trial) {
            auto point = run_single_benchmark(
                L_init, sequence, n, config,
                opts.track_rank_evolution && trial == 0,
                opts.enable_fdm && trial == 0
            );
            
            point.trial_id = trial;
            point.total_trials = opts.sweep_trials;
            
            if (trial == 0) {
                first_trial_result = point;
                point.fdm_executed = point.fdm_run;
                if (point.fdm_run) {
                    have_fdm = true;
                    MatrixXcd rho_init = L_init * L_init.adjoint();
                    auto fdm_result = run_fdm_simulation(sequence, n, rho_init, false);
                    if (fdm_result.was_run) {
                        fdm_rho_storage = fdm_result.rho_final;
                    }
                }
            } else if (have_fdm) {
                point.fdm_run = true;
                point.fdm_executed = false;
                point.fdm_time_seconds = first_trial_result.fdm_time_seconds;
                point.fdm_memory_bytes = first_trial_result.fdm_memory_bytes;
                point.fidelity_vs_fdm = first_trial_result.fidelity_vs_fdm;
                point.trace_distance_vs_fdm = first_trial_result.trace_distance_vs_fdm;
            }
            
            if (opts.sweep_config.run_all_modes) {
                size_t batch = opts.batch_size ? opts.batch_size : auto_select_batch_size(n);
                const MatrixXcd* fdm_rho = have_fdm ? &fdm_rho_storage : nullptr;
                point.all_modes_results = run_all_modes_benchmark(
                    L_init, sequence, n, config, batch, opts.enable_fdm, fdm_rho
                );
            }
            
            results.add_point(point);
        }
        
        std::cout << std::fixed << std::setprecision(4) << first_trial_result.lret_time_seconds << "s";
        if (opts.sweep_trials > 1) std::cout << " (" << opts.sweep_trials << " trials)";
        std::cout << ", rank=" << first_trial_result.lret_final_rank;
        if (first_trial_result.fdm_run) {
            std::cout << ", FDM=" << first_trial_result.fdm_time_seconds << "s"
                      << ", speedup=" << std::setprecision(1) << first_trial_result.speedup_vs_fdm() << "x";
        }
        if (!first_trial_result.all_modes_results.empty()) {
            std::cout << ", modes=" << first_trial_result.all_modes_results.size();
        }
        std::cout << "\n";
    }
    
    results.end_time = std::chrono::system_clock::now();
    results.total_wall_time_seconds = std::chrono::duration<double>(
        results.end_time - results.start_time).count();
    
    // Find crossover if both LRET and FDM were run
    results.find_crossover();
    
    return results;
}

SweepResults run_depth_sweep(
    const CLIOptions& opts,
    const std::vector<size_t>& depth_values
) {
    SweepResults results;
    results.type = SweepType::DEPTH;
    results.sweep_parameter_name = "depth";
    results.start_time = std::chrono::system_clock::now();
    
    // Initial state
    MatrixXcd L_init;
    if (opts.initial_rank > 1) {
        L_init = create_random_mixed_state(opts.num_qubits, opts.initial_rank, opts.random_seed);
    } else {
        L_init = create_zero_state(opts.num_qubits);
    }
    
    std::cout << "Running depth sweep: " << depth_values.size() << " points\n";
    std::cout << "  Fixed: n=" << opts.num_qubits << ", noise=" << opts.noise_prob << "\n\n";
    
    for (size_t i = 0; i < depth_values.size(); ++i) {
        size_t d = depth_values[i];
        
        std::cout << "  [" << (i+1) << "/" << depth_values.size() << "] "
                  << "d=" << std::setw(3) << d << " ... " << std::flush;
        
        auto sequence = generate_quantum_sequences(
            opts.num_qubits, d, true, opts.noise_prob, opts.random_seed
        );
        
        SimConfig config;
        config.truncation_threshold = opts.truncation_threshold;
        config.do_truncation = true;
        config.batch_size = opts.batch_size ? opts.batch_size : auto_select_batch_size(opts.num_qubits);
        
        // Cache FDM result for first trial
        SweepPointResult first_trial_result;
        MatrixXcd fdm_rho_storage;
        bool have_fdm = false;
        
        // Run each trial and output individually
        for (size_t trial = 0; trial < opts.sweep_trials; ++trial) {
            auto point = run_single_benchmark(
                L_init, sequence, opts.num_qubits, config,
                opts.track_rank_evolution && trial == 0,
                opts.enable_fdm && trial == 0
            );
            
            point.trial_id = trial;
            point.total_trials = opts.sweep_trials;
            point.depth = d;
            
            if (trial == 0) {
                first_trial_result = point;
                point.fdm_executed = point.fdm_run;
                if (point.fdm_run) {
                    have_fdm = true;
                    MatrixXcd rho_init = L_init * L_init.adjoint();
                    auto fdm_result = run_fdm_simulation(sequence, opts.num_qubits, rho_init, false);
                    if (fdm_result.was_run) {
                        fdm_rho_storage = fdm_result.rho_final;
                    }
                }
            } else if (have_fdm) {
                point.fdm_run = true;
                point.fdm_executed = false;
                point.fdm_time_seconds = first_trial_result.fdm_time_seconds;
                point.fdm_memory_bytes = first_trial_result.fdm_memory_bytes;
                point.fidelity_vs_fdm = first_trial_result.fidelity_vs_fdm;
                point.trace_distance_vs_fdm = first_trial_result.trace_distance_vs_fdm;
            }
            
            if (opts.sweep_config.run_all_modes) {
                size_t batch = opts.batch_size ? opts.batch_size : auto_select_batch_size(opts.num_qubits);
                const MatrixXcd* fdm_rho = have_fdm ? &fdm_rho_storage : nullptr;
                point.all_modes_results = run_all_modes_benchmark(
                    L_init, sequence, opts.num_qubits, config, batch, opts.enable_fdm, fdm_rho
                );
            }
            
            results.add_point(point);
        }
        
        std::cout << std::fixed << std::setprecision(4) << first_trial_result.lret_time_seconds << "s";
        if (opts.sweep_trials > 1) std::cout << " (" << opts.sweep_trials << " trials)";
        std::cout << ", rank=" << first_trial_result.lret_final_rank;
        if (first_trial_result.fdm_run) {
            std::cout << ", FDM=" << first_trial_result.fdm_time_seconds << "s"
                      << ", speedup=" << std::setprecision(1) << first_trial_result.speedup_vs_fdm() << "x";
        }
        if (!first_trial_result.all_modes_results.empty()) {
            std::cout << ", modes=" << first_trial_result.all_modes_results.size();
        }
        std::cout << "\n";
    }
    
    results.end_time = std::chrono::system_clock::now();
    results.total_wall_time_seconds = std::chrono::duration<double>(
        results.end_time - results.start_time).count();
    
    return results;
}

SweepResults run_crossover_analysis(
    const CLIOptions& opts,
    size_t min_qubits,
    size_t max_qubits
) {
    std::cout << "\n=== LRET vs FDM Crossover Analysis ===\n";
    std::cout << "Finding where LRET becomes faster than FDM...\n\n";
    
    // Use qubit_values from config if available, otherwise build from min/max
    std::vector<size_t> qubit_values;
    if (!opts.sweep_config.qubit_values.empty()) {
        qubit_values = opts.sweep_config.qubit_values;
    } else {
        for (size_t n = min_qubits; n <= max_qubits; ++n) {
            qubit_values.push_back(n);
        }
    }
    
    // Create modified options with FDM enabled
    CLIOptions sweep_opts = opts;
    sweep_opts.enable_fdm = true;
    
    auto results = run_qubit_sweep(sweep_opts, qubit_values);
    results.type = SweepType::CROSSOVER;
    
    // Print crossover analysis
    std::cout << "\n=== Crossover Analysis Results ===\n";
    if (results.crossover_found) {
        std::cout << "LRET becomes faster than FDM at n=" << results.crossover_qubit_count << " qubits\n";
    } else {
        std::cout << "No crossover found in range n=" << min_qubits << " to " << max_qubits << "\n";
        std::cout << "(FDM may be faster throughout this range, or couldn't run at larger n)\n";
    }
    
    // Print comparison table
    std::cout << "\n";
    std::cout << std::setw(8) << "Qubits" << " | "
              << std::setw(12) << "LRET (s)" << " | "
              << std::setw(12) << "FDM (s)" << " | "
              << std::setw(10) << "Speedup" << " | "
              << std::setw(10) << "Winner" << "\n";
    std::cout << std::string(60, '-') << "\n";
    
    for (const auto& p : results.points) {
        std::cout << std::setw(8) << p.num_qubits << " | "
                  << std::setw(12) << std::fixed << std::setprecision(4) << p.lret_time_seconds << " | ";
        
        if (p.fdm_run) {
            std::cout << std::setw(12) << p.fdm_time_seconds << " | "
                      << std::setw(9) << std::setprecision(1) << p.speedup_vs_fdm() << "x | "
                      << std::setw(10) << (p.lret_time_seconds < p.fdm_time_seconds ? "LRET" : "FDM");
        } else {
            std::cout << std::setw(12) << "N/A" << " | "
                      << std::setw(10) << "N/A" << " | "
                      << std::setw(10) << "LRET only";
        }
        std::cout << "\n";
    }
    
    return results;
}

//==============================================================================
// Main Parameter Sweep Runner
//==============================================================================

SweepResults run_parameter_sweep(
    const CLIOptions& opts,
    std::function<void(size_t, size_t, const SweepPointResult&)> callback
) {
    const SweepConfig& config = opts.sweep_config;
    
    switch (config.type) {
        case SweepType::EPSILON:
            return run_epsilon_sweep(opts, config.epsilon_values);
            
        case SweepType::NOISE_PROB:
            return run_noise_sweep(opts, config.noise_values);
            
        case SweepType::QUBITS:
            return run_qubit_sweep(opts, config.qubit_values);
            
        case SweepType::DEPTH:
            return run_depth_sweep(opts, config.depth_values);
            
        case SweepType::INITIAL_RANK:
            return run_rank_sweep(opts, config.rank_values);
            
        case SweepType::CROSSOVER:
            return run_crossover_analysis(
                opts, 
                config.crossover_min_qubits, 
                config.crossover_max_qubits
            );
            
        default:
            // No sweep - return empty results
            return SweepResults();
    }
}

}  // namespace qlret
