/**
 * QuantumLRET-Sim - Main Program
 * Multi-mode parallelization with CLI support
 */
#include "cli_parser.h"
#include "parallel_modes.h"
#include "fdm_simulator.h"
#include "output_formatter.h"
#include "gates_and_noise.h"
#include "simulator.h"
#include "utils.h"
#include "resource_monitor.h"
#include "progressive_csv.h"
#include <iostream>
#include <iomanip>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace qlret;

int main(int argc, char* argv[]) {
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
        
        // Initialize progressive CSV writer if output file specified
        std::unique_ptr<ProgressiveCSVWriter> csv_writer;
        if (opts.output_file) {
            csv_writer = std::make_unique<ProgressiveCSVWriter>(*opts.output_file);
            if (!csv_writer->is_open()) {
                std::cerr << "Warning: Could not open CSV file for writing\n";
                csv_writer.reset();
            } else {
                g_csv_writer = csv_writer.get();
                
                // Print CSV file path immediately so user can monitor during long runs
                std::cout << "========================================\n";
                std::cout << "CSV Output: " << csv_writer->get_filepath() << "\n";
                std::cout << "  (File updates in real-time - you can monitor with: tail -f " 
                          << *opts.output_file << ")\n";
                std::cout << "========================================\n\n";
                
                g_csv_writer->log_start(opts.num_qubits, opts.depth, 
                                        parallel_mode_to_string(opts.parallel_mode), opts.noise_prob);
            }
        }
        
        // Configure threading
        if (opts.num_threads > 0) {
#ifdef _OPENMP
            omp_set_num_threads(opts.num_threads);
#endif
        }
        
        // Setup batch size
        size_t batch_size = opts.batch_size ? opts.batch_size 
                                            : auto_select_batch_size(opts.num_qubits);
        
        // Generate circuit
        auto sequence = generate_quantum_sequences(
            opts.num_qubits, opts.depth, true, opts.noise_prob
        );
        double noise_in_circuit = sequence.total_noise_probability;
        
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
            if (g_csv_writer) {
                g_csv_writer->log_interrupt("Aborted before FDM");
            }
            return 130;  // Standard interrupt exit code
        }
        
        // Run FDM if applicable (wrapped in try-catch for graceful failure)
        if (fdm_check.should_run) {
            std::cout << "Running FDM simulation..." << std::flush;
            try {
                fdm_result = run_fdm_simulation(sequence, opts.num_qubits, opts.verbose);
                std::cout << " done (" << std::fixed << std::setprecision(3) 
                          << fdm_result->time_seconds << "s)\n";
            } catch (const std::bad_alloc& e) {
                std::cout << " FAILED (memory allocation error)\n";
                std::cerr << "FDM aborted: Could not allocate memory. "
                          << "Required ~" << fdm_check.estimated_memory_mb << " MB.\n";
                FDMResult failed;
                failed.was_run = false;
                failed.skip_reason = "Memory allocation failed during execution";
                fdm_result = failed;
            } catch (const std::exception& e) {
                std::cout << " FAILED\n";
                std::cerr << "FDM aborted: " << e.what() << "\n";
                FDMResult failed;
                failed.was_run = false;
                failed.skip_reason = std::string("Runtime error: ") + e.what();
                fdm_result = failed;
            }
        } else if (opts.enable_fdm) {
            FDMResult skipped;
            skipped.was_run = false;
            skipped.skip_reason = fdm_check.skip_reason;
            fdm_result = skipped;
        }
        
        // Check for abort before LRET simulation
        if (should_abort()) {
            std::cout << "\nAborted before LRET simulation.\n";
            if (g_csv_writer) {
                g_csv_writer->log_interrupt("Aborted before LRET");
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
            
            std::cout << "\n";
            print_comparison_output(opts, results, noise_in_circuit, 
                                    sequence.noise_stats, fdm_result, fdm_metrics);
            
            if (opts.output_file) {
                export_to_csv(*opts.output_file, opts, results, noise_in_circuit,
                              sequence.noise_stats, fdm_result);
            }
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
            
            // Log summary to progressive CSV
            if (g_csv_writer) {
                g_csv_writer->log_summary(result.final_rank, 
                                          compute_purity(result.L_final), 
                                          result.time_seconds, "SUCCESS");
            }
            
            if (opts.output_file) {
                std::vector<ModeResult> single_result = {result};
                export_to_csv(*opts.output_file, opts, single_result, 
                              noise_in_circuit, sequence.noise_stats, fdm_result);
            }
        }
        
        // Check if we were interrupted during computation
        if (is_interrupted()) {
            std::cout << "\nSimulation was interrupted by user.\n";
            return 130;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        if (g_csv_writer) {
            g_csv_writer->log_error(e.what());
        }
        return 1;
    }
    
    return 0;
}
