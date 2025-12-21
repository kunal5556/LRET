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
#include <iostream>
#include <iomanip>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace qlret;

int main(int argc, char* argv[]) {
    try {
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
        
        // Initial state
        MatrixXcd L_init = create_zero_state(opts.num_qubits);
        
        // Simulation config
        SimConfig config;
        config.truncation_threshold = opts.truncation_threshold;
        config.verbose = opts.verbose;
        config.do_truncation = true;
        
        // Check FDM feasibility
        auto fdm_check = check_fdm_feasibility(
            opts.num_qubits, opts.fdm_threshold, opts.enable_fdm
        );
        
        std::optional<FDMResult> fdm_result;
        std::optional<MetricsResult> fdm_metrics;
        
        // Run FDM if applicable
        if (fdm_check.should_run) {
            std::cout << "Running FDM simulation..." << std::flush;
            fdm_result = run_fdm_simulation(sequence, opts.num_qubits, opts.verbose);
            std::cout << " done (" << std::fixed << std::setprecision(3) 
                      << fdm_result->time_seconds << "s)\n";
        } else if (opts.enable_fdm) {
            FDMResult skipped;
            skipped.was_run = false;
            skipped.skip_reason = fdm_check.skip_reason;
            fdm_result = skipped;
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
            print_comparison_output(opts, results, noise_in_circuit, fdm_result, fdm_metrics);
            
            if (opts.output_file) {
                export_to_csv(*opts.output_file, opts, results, noise_in_circuit, fdm_result);
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
                                  fdm_result, fdm_metrics);
            
            if (opts.output_file) {
                std::vector<ModeResult> single_result = {result};
                export_to_csv(*opts.output_file, opts, single_result, 
                              noise_in_circuit, fdm_result);
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
