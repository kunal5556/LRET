/**
 * QuantumLRET-Sim - Main Program
 * Multi-mode parallelization with CLI support
 * Parameter sweeps for LRET paper benchmarking
 */
#include "cli_parser.h"
#include "parallel_modes.h"
#include "fdm_simulator.h"
#include "output_formatter.h"
#include "gates_and_noise.h"
#include "simulator.h"
#include "utils.h"
#include "resource_monitor.h"
#include "structured_csv.h"
#include "benchmark_runner.h"
#include <iostream>
#include <iomanip>
#include <memory>
#include <sys/stat.h>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace qlret;

// Forward declaration for sweep mode handling
int run_sweep_mode(const CLIOptions& opts, StructuredCSVWriter* csv_writer);

// Forward declaration for benchmark-all mode (runs all LRET paper benchmarks)
int run_all_benchmarks(const CLIOptions& opts, StructuredCSVWriter* csv_writer);

int main(int argc, char* argv[]) {
    auto program_start_time = std::chrono::steady_clock::now();
    
    try {
        // Setup signal handlers for graceful Ctrl+C handling
        setup_signal_handlers();
        
        // Parse command line
        CLIOptions opts = parse_arguments(argc, argv);
        
        if (opts.show_help) {
            print_help();
            return 0;
        }
        
        if (opts.show_version) {
            print_version();
            return 0;
        }
        
        // Validate options
        std::string error;
        if (!validate_options(opts, error)) {
            std::cerr << "Error: " << error << std::endl;
            return 1;
        }
        
        // Initialize timeout if specified
        if (!opts.timeout_str.empty()) {
            auto timeout_duration = parse_timeout_string(opts.timeout_str);
            if (timeout_duration.count() > 0) {
                g_timeout.enabled = true;
                g_timeout.duration = timeout_duration;
                g_timeout.start();
                std::cout << "Timeout set: " << timeout_duration.count() << " seconds\n";
            } else {
                std::cerr << "Warning: Could not parse timeout '" << opts.timeout_str << "'\n";
            }
        }
        
        // Check for swap usage (unless --allow-swap or --non-interactive)
        if (!opts.allow_swap && !opts.non_interactive) {
            if (!check_swap_and_prompt(opts.allow_swap, !opts.non_interactive)) {
                std::cout << "Exiting due to swap memory concerns.\n";
                return 2;
            }
        }
        
        // Output file handling
        // Only generate output if -o flag was explicitly given
        std::unique_ptr<StructuredCSVWriter> csv_writer;
        std::string csv_filename;
        
        // Helper lambda to check if path is a directory
        auto is_directory = [](const std::string& path) -> bool {
            struct stat info;
            if (stat(path.c_str(), &info) != 0) return false;
            return (info.st_mode & S_IFDIR) != 0;
        };
        
        // Helper lambda to check if path exists
        auto path_exists = [](const std::string& path) -> bool {
            struct stat info;
            return stat(path.c_str(), &info) == 0;
        };
        
        if (opts.generate_output) {
            // -o flag was given
            std::string default_filename = generate_default_csv_filename(opts);
            
            if (opts.output_file && !opts.output_file->empty()) {
                std::string provided_path = *opts.output_file;
                
                // Check if the provided path is a directory
                if (is_directory(provided_path)) {
                    // It's a directory - append default filename
                    if (provided_path.back() != '/' && provided_path.back() != '\\') {
                        provided_path += '/';
                    }
                    csv_filename = provided_path + default_filename;
                    std::cout << "Output file (auto in dir): " << csv_filename << "\n";
                } else {
                    // It's a file path - use as-is
                    csv_filename = provided_path;
                    std::cout << "Output file (custom): " << csv_filename << "\n";
                }
            } else {
                // No path provided: -o (alone)
                // Check for common mounted directories in order of preference
                std::vector<std::string> output_dirs = {"/app/output", "/myoutput", "/output", "."};
                std::string output_dir = ".";  // fallback
                
                for (const auto& dir : output_dirs) {
                    if (is_directory(dir)) {
                        output_dir = dir;
                        break;
                    }
                }
                
                if (output_dir != "." && output_dir.back() != '/') {
                    output_dir += '/';
                } else if (output_dir == ".") {
                    output_dir = "./";
                }
                
                csv_filename = output_dir + default_filename;
                std::cout << "Output file (default): " << csv_filename << "\n";
            }
            
            // Initialize structured CSV writer
            csv_writer = std::make_unique<StructuredCSVWriter>(csv_filename);
            if (!csv_writer->is_open()) {
                std::cerr << "Warning: Could not open CSV file for writing: " << csv_filename << "\n";
                csv_writer.reset();
            } else {
                g_structured_csv = csv_writer.get();
                
                // Print CSV file path immediately so user can monitor during long runs
                std::cout << "========================================\n";
                std::cout << "CSV Output: " << csv_writer->get_filepath() << "\n";
                std::cout << "  (File updates in real-time)\n";
                std::cout << "  Monitor with: tail -f " << csv_filename << "\n";
                std::cout << "========================================\n\n";
            }
        } else {
            // No -o flag: no output file will be created
            std::cout << "No output file requested (use -o to generate CSV output)\n\n";
        }
        
        // Configure threading
        if (opts.num_threads > 0) {
#ifdef _OPENMP
            omp_set_num_threads(opts.num_threads);
#endif
        }
        
        //======================================================================
        // Check for Benchmark-All Mode (Run all LRET paper benchmarks)
        //======================================================================
        if (opts.benchmark_all) {
            return run_all_benchmarks(opts, csv_writer.get());
        }
        
        //======================================================================
        // Check for Parameter Sweep Mode (LRET Paper Benchmarking)
        //======================================================================
        if (opts.sweep_config.is_active()) {
            return run_sweep_mode(opts, csv_writer.get());
        }
        
        //======================================================================
        // Regular Single-Run Simulation Mode
        //======================================================================
        
        // Setup batch size
        size_t batch_size = opts.batch_size ? opts.batch_size 
                                            : auto_select_batch_size(opts.num_qubits);
        
        // Generate circuit
        auto sequence = generate_quantum_sequences(
            opts.num_qubits, opts.depth, true, opts.noise_prob
        );
        double noise_in_circuit = sequence.total_noise_probability;
        
        // Write CSV header section
        if (g_structured_csv) {
            g_structured_csv->write_header(opts, sequence.noise_stats);
        }
        
        // Print circuit if verbose
        if (opts.verbose) {
            std::cout << "Circuit diagram:\n";
            print_circuit_diagram(opts.num_qubits, sequence, 80);
            std::cout << "\n";
        }
        
        // Initial state - pure state or random mixed state for high-rank testing
        MatrixXcd L_init;
        if (opts.initial_rank > 1) {
            L_init = create_random_mixed_state(opts.num_qubits, opts.initial_rank, opts.random_seed);
            std::cout << "Using random mixed state with initial rank=" << opts.initial_rank << "\n";
            if (opts.verbose) {
                std::cout << "  (This enables meaningful parallel benchmarking since pure states have rank=1)\n";
            }
        } else {
            L_init = create_zero_state(opts.num_qubits);
        }
        
        // Simulation config
        SimConfig config;
        config.truncation_threshold = opts.truncation_threshold;
        config.verbose = opts.verbose;
        config.do_truncation = true;
        
        // Check FDM feasibility (pass force flag to bypass memory check)
        auto fdm_check = check_fdm_feasibility(opts.num_qubits, opts.enable_fdm, opts.fdm_force);
        
        std::optional<FDMResult> fdm_result;
        std::optional<MetricsResult> fdm_metrics;
        
        // Check for abort before FDM
        if (should_abort()) {
            std::cout << "\nAborted before FDM simulation.\n";
            if (g_structured_csv) {
                g_structured_csv->log_interrupt("Aborted before FDM");
            }
            return 130;  // Standard interrupt exit code
        }
        
        // Run FDM if applicable (wrapped in try-catch for graceful failure)
        if (fdm_check.should_run) {
            std::cout << "Running FDM simulation..." << std::flush;
            
            // Start FDM progress logging
            if (g_structured_csv) {
                g_structured_csv->begin_fdm_progress(opts.num_qubits, opts.depth);
            }
            
            try {
                fdm_result = run_fdm_simulation(sequence, opts.num_qubits, opts.verbose);
                std::cout << " done (" << std::fixed << std::setprecision(3) 
                          << fdm_result->time_seconds << "s)\n";
                
                // End FDM progress and write metrics
                if (g_structured_csv) {
                    g_structured_csv->end_fdm_progress(fdm_result->time_seconds, true);
                    g_structured_csv->write_fdm_metrics(*fdm_result, opts.num_qubits, sequence.noise_stats);
                }
            } catch (const std::bad_alloc& e) {
                std::cout << " FAILED (memory allocation error)\n";
                std::cerr << "FDM aborted: Could not allocate memory. "
                          << "Required ~" << fdm_check.estimated_memory_mb << " MB.\n";
                FDMResult failed;
                failed.was_run = false;
                failed.skip_reason = "Memory allocation failed during execution";
                fdm_result = failed;
                
                if (g_structured_csv) {
                    g_structured_csv->end_fdm_progress(0, false, "Memory allocation failed");
                }
            } catch (const std::exception& e) {
                std::cout << " FAILED\n";
                std::cerr << "FDM aborted: " << e.what() << "\n";
                FDMResult failed;
                failed.was_run = false;
                failed.skip_reason = std::string("Runtime error: ") + e.what();
                fdm_result = failed;
                
                if (g_structured_csv) {
                    g_structured_csv->end_fdm_progress(0, false, e.what());
                }
            }
        } else if (opts.enable_fdm) {
            FDMResult skipped;
            skipped.was_run = false;
            skipped.skip_reason = fdm_check.skip_reason;
            fdm_result = skipped;
            
            if (g_structured_csv) {
                g_structured_csv->write_fdm_metrics(skipped, opts.num_qubits, sequence.noise_stats);
            }
        }
        
        // Check for abort before LRET simulation
        if (should_abort()) {
            std::cout << "\nAborted before LRET simulation.\n";
            if (g_structured_csv) {
                g_structured_csv->log_interrupt("Aborted before LRET");
            }
            return 130;
        }
        
        // Run based on mode
        if (opts.parallel_mode == ParallelMode::COMPARE) {
            // Comparison mode - run all strategies
            std::cout << "\n";
            auto results = run_all_modes_comparison(
                L_init, sequence, opts.num_qubits, config, batch_size
            );
            
            // Compute FDM metrics if available
            if (fdm_result && fdm_result->was_run) {
                auto fastest = std::min_element(results.begin(), results.end(),
                    [](const auto& a, const auto& b) { 
                        return a.time_seconds < b.time_seconds; 
                    });
                
                MatrixXcd rho_lret = L_to_density_matrix(fastest->L_final);
                fdm_metrics = MetricsResult{
                    compute_fidelity_rho(rho_lret, fdm_result->rho_final),
                    compute_trace_distance_rho(rho_lret, fdm_result->rho_final),
                    compute_frobenius_distance_rho(rho_lret, fdm_result->rho_final),
                    compute_variational_distance(rho_lret, fdm_result->rho_final)
                };
            }
            
            // Write structured CSV comparison tables
            if (g_structured_csv) {
                g_structured_csv->write_mode_comparison(results);
                if (fdm_result && fdm_result->was_run) {
                    g_structured_csv->write_fdm_comparison(results, *fdm_result);
                }
                
                // Write summary
                auto wall_clock_end = std::chrono::steady_clock::now();
                double total_wall_time = std::chrono::duration<double>(wall_clock_end - program_start_time).count();
                
                // Compute total LRET time
                double total_lret_time = 0;
                for (const auto& r : results) {
                    total_lret_time += r.time_seconds;
                }
                
                double fdm_time = (fdm_result && fdm_result->was_run) ? fdm_result->time_seconds : 0;
                g_structured_csv->write_summary(total_wall_time, total_lret_time, fdm_time, true, "");
            }
            
            std::cout << "\n";
            print_comparison_output(opts, results, noise_in_circuit, 
                                    sequence.noise_stats, fdm_result, fdm_metrics);
        } else {
            // Single mode run
            ParallelMode mode = opts.parallel_mode;
            if (mode == ParallelMode::AUTO) {
                mode = auto_select_mode(opts.num_qubits, opts.depth, 10);
            }
            
            std::cout << "Running " << parallel_mode_to_string(mode) << " mode...\n";
            
            auto result = run_with_mode(L_init, sequence, opts.num_qubits, 
                                        mode, config, batch_size);
            
            // If not sequential, run a quick sequential baseline for speedup comparison
            if (mode != ParallelMode::SEQUENTIAL) {
                std::cout << "Running sequential baseline for speedup comparison...\n";
                auto seq_result = run_with_mode(L_init, sequence, opts.num_qubits, 
                                                ParallelMode::SEQUENTIAL, config, batch_size);
                result.speedup = seq_result.time_seconds / result.time_seconds;
                
                // Compute distortion vs sequential
                double seq_norm = seq_result.L_final.norm();
                if (seq_norm > 1e-15) {
                    result.distortion = (result.L_final - seq_result.L_final).norm() / seq_norm;
                }
            }
            
            // Compute metrics vs initial state
            MetricsResult metrics{
                compute_fidelity(L_init, result.L_final),
                compare_L_matrices_trace(L_init, result.L_final),
                compute_frobenius_distance(L_init, result.L_final),
                compute_variational_distance_L(L_init, result.L_final)
            };
            
            // Compute state metrics for final state
            StateMetrics state_metrics;
            state_metrics.purity = compute_purity(result.L_final);
            state_metrics.entropy = compute_entropy(result.L_final);
            state_metrics.linear_entropy = compute_linear_entropy(result.L_final);
            state_metrics.rank = result.final_rank;
            
            // Concurrence only for 2-qubit systems
            if (opts.num_qubits == 2) {
                state_metrics.concurrence = compute_concurrence(result.L_final);
            }
            
            // Negativity for bipartite split (half-half)
            if (opts.num_qubits >= 2) {
                size_t split = opts.num_qubits / 2;
                state_metrics.negativity = compute_negativity(result.L_final, split, opts.num_qubits);
            }
            
            // Compute FDM metrics if available
            if (fdm_result && fdm_result->was_run) {
                MatrixXcd rho_lret = L_to_density_matrix(result.L_final);
                fdm_metrics = MetricsResult{
                    compute_fidelity_rho(rho_lret, fdm_result->rho_final),
                    compute_trace_distance_rho(rho_lret, fdm_result->rho_final),
                    compute_frobenius_distance_rho(rho_lret, fdm_result->rho_final),
                    compute_variational_distance(rho_lret, fdm_result->rho_final)
                };
            }
            
            print_standard_output(opts, result, metrics, noise_in_circuit, 
                                  sequence.noise_stats, fdm_result, fdm_metrics, state_metrics);
            
            // Log LRET metrics and summary to structured CSV
            if (g_structured_csv) {
                // Write comprehensive LRET mode metrics (single mode)
                g_structured_csv->write_lret_mode_metrics(
                    parallel_mode_to_string(mode), result, metrics, state_metrics, sequence.noise_stats);
                
                // Write FDM comparison if FDM was run
                if (fdm_result && fdm_result->was_run && fdm_metrics) {
                    std::vector<ModeResult> single_result = {result};
                    std::map<std::string, MetricsResult> fdm_metrics_map;
                    fdm_metrics_map[parallel_mode_to_string(mode)] = *fdm_metrics;
                    g_structured_csv->write_fdm_comparison(single_result, *fdm_result, fdm_metrics_map);
                }
                
                // Write summary
                auto wall_clock_end = std::chrono::steady_clock::now();
                double total_wall_time = std::chrono::duration<double>(wall_clock_end - program_start_time).count();
                double fdm_time = (fdm_result && fdm_result->was_run) ? fdm_result->time_seconds : 0;
                g_structured_csv->write_summary(total_wall_time, result.time_seconds, fdm_time, true, "");
            }
        }
        
        // Print final export message
        if (g_structured_csv && g_structured_csv->is_open()) {
            std::cout << "Results exported to: " << g_structured_csv->get_filepath() << "\n";
        }
        
        // Check if we were interrupted during computation
        if (is_interrupted()) {
            std::cout << "\nSimulation was interrupted by user.\n";
            return 130;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        if (g_structured_csv) {
            g_structured_csv->log_error(e.what());
        }
        return 1;
    }
    
    return 0;
}

//==============================================================================
// Parameter Sweep Mode (LRET Paper Benchmarking)
//==============================================================================

/**
 * @brief Run parameter sweep mode for LRET paper benchmarking
 * 
 * Supports sweeps over:
 * - Truncation threshold (epsilon): Shows accuracy vs speed tradeoff
 * - Noise probability: Shows how noise affects rank growth
 * - Number of qubits: Shows scaling behavior
 * - Circuit depth: Shows time vs depth relationship
 * - LRET vs FDM crossover: Finds where LRET becomes faster
 */
int run_sweep_mode(const CLIOptions& opts, StructuredCSVWriter* csv_writer) {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "   LRET Paper Benchmarking Mode\n";
    std::cout << "========================================\n\n";
    
    auto sweep_start = std::chrono::steady_clock::now();
    
    // Print sweep configuration
    std::cout << "Sweep type: " << sweep_type_to_string(opts.sweep_config.type) << "\n";
    std::cout << "Sweep points: " << opts.sweep_config.num_points() << "\n";
    if (opts.sweep_trials > 1) {
        std::cout << "Trials per point: " << opts.sweep_trials << "\n";
    }
    if (opts.track_rank_evolution) {
        std::cout << "Rank evolution tracking: ENABLED\n";
    }
    if (opts.enable_fdm) {
        std::cout << "FDM comparison: ENABLED\n";
    }
    std::cout << "\n";
    
    // Write sweep header to CSV
    if (csv_writer) {
        csv_writer->write_sweep_header(
            sweep_type_to_string(opts.sweep_config.type),
            opts.sweep_config.num_points(),
            opts.num_qubits,
            opts.depth,
            opts.noise_prob
        );
    }
    
    // Run the sweep
    SweepResults results = run_parameter_sweep(opts);
    
    // Write results to CSV
    if (csv_writer && !results.points.empty()) {
        // Build data tuples for CSV
        std::vector<std::tuple<double, double, size_t, double>> sweep_data;
        
        for (const auto& p : results.points) {
            double param_value;
            switch (opts.sweep_config.type) {
                case SweepType::EPSILON:
                    param_value = p.epsilon;
                    break;
                case SweepType::NOISE_PROB:
                    param_value = p.noise_prob;
                    break;
                case SweepType::QUBITS:
                case SweepType::CROSSOVER:
                    param_value = static_cast<double>(p.num_qubits);
                    break;
                case SweepType::DEPTH:
                    param_value = static_cast<double>(p.depth);
                    break;
                default:
                    param_value = 0.0;
            }
            
            sweep_data.emplace_back(
                param_value,
                p.lret_time_seconds,
                p.lret_final_rank,
                p.fidelity_vs_fdm
            );
        }
        
        csv_writer->write_sweep_results(
            sweep_type_to_string(opts.sweep_config.type),
            sweep_data
        );
        
        // Write rank evolution if tracked (for first point only, to avoid huge files)
        if (opts.track_rank_evolution && !results.points.empty() && 
            !results.points[0].rank_evolution.empty()) {
            
            const auto& evolution = results.points[0].rank_evolution;
            std::vector<std::tuple<size_t, std::string, size_t, size_t, double>> events;
            
            for (const auto& e : evolution.events) {
                events.emplace_back(
                    e.step,
                    e.operation_name,
                    e.rank_before,
                    e.rank_after,
                    e.time_seconds
                );
            }
            
            csv_writer->write_rank_evolution(events);
            
            // Write timing breakdown
            TimingBreakdown timing = create_timing_breakdown(evolution);
            csv_writer->write_timing_breakdown(
                timing.gate_time,
                timing.noise_time,
                timing.truncation_time,
                timing.total_time,
                timing.truncation_count
            );
        }
        
        // Write crossover summary if applicable
        if (opts.sweep_config.type == SweepType::CROSSOVER) {
            csv_writer->write_crossover_summary(
                results.crossover_qubit_count,
                results.crossover_found
            );
        }
        
        // Write memory comparison for a representative point
        if (!results.points.empty()) {
            const auto& p = results.points.back();  // Use largest point
            csv_writer->write_memory_comparison(
                p.num_qubits,
                p.lret_memory_bytes,
                p.fdm_run ? p.fdm_memory_bytes : compute_fdm_memory_bytes(p.num_qubits),
                p.lret_final_rank
            );
        }
        
        // Write summary
        auto sweep_end = std::chrono::steady_clock::now();
        double total_wall_time = std::chrono::duration<double>(sweep_end - sweep_start).count();
        
        double total_lret_time = 0.0, total_fdm_time = 0.0;
        for (const auto& p : results.points) {
            total_lret_time += p.lret_time_seconds;
            if (p.fdm_run) total_fdm_time += p.fdm_time_seconds;
        }
        
        csv_writer->write_summary(total_wall_time, total_lret_time, total_fdm_time, true, "");
    }
    
    // Print summary
    auto sweep_end = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration<double>(sweep_end - sweep_start).count();
    
    std::cout << "\n========================================\n";
    std::cout << "   Sweep Complete\n";
    std::cout << "========================================\n";
    std::cout << "Total points: " << results.points.size() << "\n";
    std::cout << "Total time: " << std::fixed << std::setprecision(2) << total_time << "s\n";
    
    if (csv_writer) {
        std::cout << "Results exported to: " << csv_writer->get_filepath() << "\n";
    }
    
    std::cout << "\n";
    
    return 0;
}

//==============================================================================
// Run All Benchmarks Mode (--benchmark-all)
// Runs all LRET paper benchmarks: epsilon sweep, noise sweep, qubit sweep,
// crossover analysis, and rank evolution tracking
//==============================================================================
int run_all_benchmarks(const CLIOptions& original_opts, StructuredCSVWriter* csv_writer) {
    std::cout << "\n";
    std::cout << "========================================================\n";
    std::cout << "   LRET Paper: COMPLETE BENCHMARK SUITE\n";
    std::cout << "========================================================\n\n";
    
    auto total_start = std::chrono::steady_clock::now();
    
    std::cout << "This will run ALL paper benchmarks in sequence:\n";
    std::cout << "  1. Epsilon (truncation threshold) sweep\n";
    std::cout << "  2. Noise probability sweep\n";
    std::cout << "  3. Qubit count sweep\n";
    std::cout << "  4. LRET vs FDM crossover analysis\n";
    std::cout << "  5. Rank evolution tracking\n";
    std::cout << "\n";
    
    int overall_result = 0;
    
    //--------------------------------------------------------------------------
    // 1. Epsilon Sweep: 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2
    //--------------------------------------------------------------------------
    {
        std::cout << "------------------------------------------------------\n";
        std::cout << "  [1/5] Epsilon Sweep\n";
        std::cout << "------------------------------------------------------\n";
        
        CLIOptions opts = original_opts;
        opts.sweep_config.type = SweepType::EPSILON;
        opts.sweep_config.epsilon_values = {1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2};
        opts.num_qubits = std::min(original_opts.num_qubits, static_cast<size_t>(12));
        opts.depth = std::min(original_opts.depth, static_cast<size_t>(20));
        
        std::cout << "Parameters: n=" << opts.num_qubits << ", depth=" << opts.depth << "\n";
        std::cout << "Epsilon values: 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2\n\n";
        
        SweepResults results = run_parameter_sweep(opts);
        
        if (csv_writer && !results.points.empty()) {
            std::vector<std::tuple<double, double, size_t, double>> sweep_data;
            for (const auto& p : results.points) {
                sweep_data.emplace_back(p.epsilon, p.lret_time_seconds, p.lret_final_rank, p.fidelity_vs_fdm);
            }
            csv_writer->write_sweep_header("EPSILON", results.points.size(), opts.num_qubits, opts.depth, opts.noise_prob);
            csv_writer->write_sweep_results("EPSILON", sweep_data);
        }
        
        std::cout << "  Completed " << results.points.size() << " epsilon points.\n\n";
    }
    
    //--------------------------------------------------------------------------
    // 2. Noise Probability Sweep: 0.0, 0.01, 0.02, 0.05, 0.1, 0.2
    //--------------------------------------------------------------------------
    {
        std::cout << "------------------------------------------------------\n";
        std::cout << "  [2/5] Noise Probability Sweep\n";
        std::cout << "------------------------------------------------------\n";
        
        CLIOptions opts = original_opts;
        opts.sweep_config.type = SweepType::NOISE_PROB;
        opts.sweep_config.noise_values = {0.0, 0.01, 0.02, 0.05, 0.1, 0.2};
        opts.num_qubits = std::min(original_opts.num_qubits, static_cast<size_t>(10));
        opts.depth = std::min(original_opts.depth, static_cast<size_t>(15));
        
        std::cout << "Parameters: n=" << opts.num_qubits << ", depth=" << opts.depth << "\n";
        std::cout << "Noise values: 0.0, 0.01, 0.02, 0.05, 0.1, 0.2\n\n";
        
        SweepResults results = run_parameter_sweep(opts);
        
        if (csv_writer && !results.points.empty()) {
            std::vector<std::tuple<double, double, size_t, double>> sweep_data;
            for (const auto& p : results.points) {
                sweep_data.emplace_back(p.noise_prob, p.lret_time_seconds, p.lret_final_rank, p.fidelity_vs_fdm);
            }
            csv_writer->write_sweep_header("NOISE_PROB", results.points.size(), opts.num_qubits, opts.depth, opts.noise_prob);
            csv_writer->write_sweep_results("NOISE_PROB", sweep_data);
        }
        
        std::cout << "  Completed " << results.points.size() << " noise points.\n\n";
    }
    
    //--------------------------------------------------------------------------
    // 3. Qubit Count Sweep: 5, 6, 7, ..., 15 (or user limit)
    //--------------------------------------------------------------------------
    {
        std::cout << "------------------------------------------------------\n";
        std::cout << "  [3/5] Qubit Count Sweep\n";
        std::cout << "------------------------------------------------------\n";
        
        CLIOptions opts = original_opts;
        opts.sweep_config.type = SweepType::QUBITS;
        opts.sweep_config.qubit_values.clear();
        
        size_t max_qubits = std::min(original_opts.num_qubits, static_cast<size_t>(15));
        for (size_t n = 5; n <= max_qubits; ++n) {
            opts.sweep_config.qubit_values.push_back(n);
        }
        opts.depth = std::min(original_opts.depth, static_cast<size_t>(15));
        
        std::cout << "Parameters: depth=" << opts.depth << "\n";
        std::cout << "Qubit range: 5 to " << max_qubits << "\n\n";
        
        SweepResults results = run_parameter_sweep(opts);
        
        if (csv_writer && !results.points.empty()) {
            std::vector<std::tuple<double, double, size_t, double>> sweep_data;
            for (const auto& p : results.points) {
                sweep_data.emplace_back(static_cast<double>(p.num_qubits), p.lret_time_seconds, p.lret_final_rank, p.fidelity_vs_fdm);
            }
            csv_writer->write_sweep_header("QUBITS", results.points.size(), opts.num_qubits, opts.depth, opts.noise_prob);
            csv_writer->write_sweep_results("QUBITS", sweep_data);
        }
        
        std::cout << "  Completed " << results.points.size() << " qubit sweep points.\n\n";
    }
    
    //--------------------------------------------------------------------------
    // 4. LRET vs FDM Crossover Analysis (5-12 qubits)
    //--------------------------------------------------------------------------
    {
        std::cout << "------------------------------------------------------\n";
        std::cout << "  [4/5] LRET vs FDM Crossover Analysis\n";
        std::cout << "------------------------------------------------------\n";
        
        CLIOptions opts = original_opts;
        opts.sweep_config.type = SweepType::CROSSOVER;
        opts.sweep_config.include_fdm = true;
        opts.enable_fdm = true;
        opts.sweep_config.qubit_values.clear();
        
        // Crossover limited to 5-12 qubits (FDM memory limits)
        for (size_t n = 5; n <= 12; ++n) {
            opts.sweep_config.qubit_values.push_back(n);
        }
        opts.depth = std::min(original_opts.depth, static_cast<size_t>(20));
        
        std::cout << "Parameters: depth=" << opts.depth << "\n";
        std::cout << "Qubit range: 5 to 12 (FDM memory limit)\n\n";
        
        SweepResults results = run_parameter_sweep(opts);
        
        if (csv_writer && !results.points.empty()) {
            std::vector<std::tuple<double, double, size_t, double>> sweep_data;
            for (const auto& p : results.points) {
                sweep_data.emplace_back(static_cast<double>(p.num_qubits), p.lret_time_seconds, p.lret_final_rank, p.fidelity_vs_fdm);
            }
            csv_writer->write_sweep_header("CROSSOVER", results.points.size(), opts.num_qubits, opts.depth, opts.noise_prob);
            csv_writer->write_sweep_results("CROSSOVER", sweep_data);
            csv_writer->write_crossover_summary(results.crossover_qubit_count, results.crossover_found);
        }
        
        std::cout << "  Crossover point: ";
        if (results.crossover_found) {
            std::cout << results.crossover_qubit_count << " qubits\n";
        } else {
            std::cout << "Not found in range\n";
        }
        std::cout << "\n";
    }
    
    //--------------------------------------------------------------------------
    // 5. Rank Evolution Tracking (single run with tracking)
    //--------------------------------------------------------------------------
    {
        std::cout << "------------------------------------------------------\n";
        std::cout << "  [5/5] Rank Evolution Tracking\n";
        std::cout << "------------------------------------------------------\n";
        
        CLIOptions opts = original_opts;
        opts.sweep_config.type = SweepType::DEPTH;
        opts.sweep_config.depth_values = {50};  // Single deep circuit
        opts.sweep_config.track_rank_evolution = true;
        opts.track_rank_evolution = true;
        opts.num_qubits = std::min(original_opts.num_qubits, static_cast<size_t>(10));
        
        std::cout << "Parameters: n=" << opts.num_qubits << ", depth=50\n";
        std::cout << "Tracking rank after each gate/noise/truncation\n\n";
        
        SweepResults results = run_parameter_sweep(opts);
        
        if (csv_writer && !results.points.empty() && !results.points[0].rank_evolution.empty()) {
            const auto& evolution = results.points[0].rank_evolution;
            std::vector<std::tuple<size_t, std::string, size_t, size_t, double>> events;
            
            for (const auto& e : evolution.events) {
                events.emplace_back(e.step, e.operation_name, e.rank_before, e.rank_after, e.time_seconds);
            }
            csv_writer->write_rank_evolution(events);
            
            TimingBreakdown timing = create_timing_breakdown(evolution);
            csv_writer->write_timing_breakdown(
                timing.gate_time, timing.noise_time, timing.truncation_time,
                timing.total_time, timing.truncation_count
            );
            
            std::cout << "  Final rank: " << results.points[0].lret_final_rank << "\n";
            std::cout << "  Operations tracked: " << evolution.events.size() << "\n";
        }
        std::cout << "\n";
    }
    
    //--------------------------------------------------------------------------
    // Summary
    //--------------------------------------------------------------------------
    auto total_end = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration<double>(total_end - total_start).count();
    
    std::cout << "========================================================\n";
    std::cout << "   COMPLETE BENCHMARK SUITE FINISHED\n";
    std::cout << "========================================================\n";
    std::cout << "Total wall time: " << std::fixed << std::setprecision(2) << total_time << " seconds\n";
    
    if (csv_writer) {
        csv_writer->write_summary(total_time, 0.0, 0.0, true, "Benchmark-all complete");
        std::cout << "All results exported to: " << csv_writer->get_filepath() << "\n";
    }
    
    std::cout << "\n";
    
    return overall_result;
}
