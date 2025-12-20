#include "output_formatter.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>

namespace qlret {

void print_separator(size_t width, char c) {
    std::cout << std::string(width, c) << "\n";
}

void print_config_header(const CLIOptions& opts, double noise_in_circuit, bool fdm_enabled) {
    std::cout << "Configuration:\n";
    std::cout << "  Qubits: " << opts.num_qubits 
              << " | Depth: " << opts.depth
              << " | Noise: " << std::fixed << std::setprecision(6) << noise_in_circuit
              << " | Mode: " << parallel_mode_to_string(opts.parallel_mode)
              << " | FDM: " << (fdm_enabled ? "ENABLED" : "DISABLED") << "\n";
}

void print_standard_output(
    const CLIOptions& opts,
    const ModeResult& result,
    const MetricsResult& metrics,
    double noise_in_circuit,
    const std::optional<FDMResult>& fdm_result,
    const std::optional<MetricsResult>& fdm_metrics
) {
    print_separator(80, '=');
    std::cout << "                    QuantumLRET-Sim Results\n";
    print_separator(80, '=');
    
    bool fdm_enabled = fdm_result.has_value() && fdm_result->was_run;
    print_config_header(opts, noise_in_circuit, fdm_enabled);
    print_separator(80, '-');
    
    // LRET results
    std::cout << "\nLRET Simulation:\n";
    std::cout << std::fixed;
    std::cout << "  Time:        " << std::setprecision(6) << result.time_seconds << " s\n";
    std::cout << "  Final Rank:  " << result.final_rank << "\n";
    std::cout << "  Final Trace: " << std::setprecision(5) << result.trace_value << "\n";
    
    // Metrics vs initial state
    std::cout << "\nMetrics (vs initial state):\n";
    std::cout << "  Fidelity:              " << std::setprecision(6) << metrics.fidelity << "\n";
    std::cout << "  Trace Distance:        " << std::scientific << std::setprecision(2) 
              << metrics.trace_distance << "\n";
    std::cout << "  Frobenius Distance:    " << metrics.frobenius_distance << "\n";
    std::cout << "  Variational Distance:  " << metrics.variational_distance << "\n";
    
    // FDM comparison if available
    if (fdm_result.has_value()) {
        std::cout << "\nFDM Comparison:\n";
        if (fdm_result->was_run) {
            std::cout << "  FDM Time:     " << std::fixed << std::setprecision(6) 
                      << fdm_result->time_seconds << " s\n";
            std::cout << "  FDM Trace:    " << std::setprecision(5) 
                      << fdm_result->trace_value << "\n";
            
            double speedup = fdm_result->time_seconds / result.time_seconds;
            std::cout << "  LRET Speedup: " << std::setprecision(2) << speedup << "x\n";
            
            if (fdm_metrics.has_value()) {
                std::cout << "\n  Accuracy (LRET vs FDM):\n";
                std::cout << "    Fidelity:           " << std::fixed << std::setprecision(6) 
                          << fdm_metrics->fidelity << "\n";
                std::cout << "    Trace Distance:     " << std::scientific << std::setprecision(2) 
                          << fdm_metrics->trace_distance << "\n";
                std::cout << "    Frobenius Distance: " << fdm_metrics->frobenius_distance << "\n";
                std::cout << "    Variational Dist:   " << fdm_metrics->variational_distance << "\n";
            }
        } else {
            std::cout << "  Status: NOT RUN\n";
            std::cout << "  Reason: " << fdm_result->skip_reason << "\n";
        }
    }
    
    print_separator(80, '=');
}

void print_comparison_output(
    const CLIOptions& opts,
    const std::vector<ModeResult>& results,
    double noise_in_circuit,
    const std::optional<FDMResult>& fdm_result,
    const std::optional<MetricsResult>& fdm_metrics
) {
    print_separator(80, '=');
    std::cout << "               Parallelization Strategy Comparison\n";
    print_separator(80, '=');
    
    bool fdm_enabled = fdm_result.has_value() && fdm_result->was_run;
    print_config_header(opts, noise_in_circuit, fdm_enabled);
    print_separator(80, '-');
    
    // Find sequential time for speedup calculation
    double seq_time = 1.0;
    for (const auto& r : results) {
        if (r.mode == ParallelMode::SEQUENTIAL) {
            seq_time = r.time_seconds;
            break;
        }
    }
    
    // Find fastest mode
    auto fastest = std::min_element(results.begin(), results.end(),
        [](const auto& a, const auto& b) { return a.time_seconds < b.time_seconds; });
    
    // Print comparison table
    std::cout << "\nPerformance Results:\n";
    std::cout << "+--------------+------------+------------+------------+\n";
    std::cout << "| Strategy     | Time (s)   | Speedup    | Final Rank |\n";
    std::cout << "+--------------+------------+------------+------------+\n";
    
    for (const auto& r : results) {
        double speedup = seq_time / r.time_seconds;
        std::cout << "| " << std::left << std::setw(12) << r.mode_name()
                  << " | " << std::right << std::fixed << std::setprecision(4) 
                  << std::setw(10) << r.time_seconds
                  << " | " << std::setw(9) << std::setprecision(2) << speedup << "x"
                  << " | " << std::setw(10) << r.final_rank << " |\n";
    }
    std::cout << "+--------------+------------+------------+------------+\n";
    
    std::cout << "\nWinner: " << fastest->mode_name() 
              << " (" << std::fixed << std::setprecision(2) 
              << seq_time / fastest->time_seconds << "x speedup)\n";
    
    // Row vs Column analysis
    double row_time = 0, col_time = 0;
    for (const auto& r : results) {
        if (r.mode == ParallelMode::ROW) row_time = r.time_seconds;
        if (r.mode == ParallelMode::COLUMN) col_time = r.time_seconds;
    }
    
    if (row_time > 0 && col_time > 0) {
        std::cout << "\nRow vs Column Analysis:\n";
        if (row_time < col_time) {
            std::cout << "  Row is " << std::setprecision(2) << col_time / row_time 
                      << "x faster than Column\n";
            std::cout << "  Recommendation: ROW parallelization\n";
        } else {
            std::cout << "  Column is " << std::setprecision(2) << row_time / col_time 
                      << "x faster than Row\n";
            std::cout << "  Recommendation: COLUMN parallelization\n";
        }
    }
    
    // FDM comparison if available
    if (fdm_result.has_value()) {
        std::cout << "\n";
        print_separator(80, '-');
        std::cout << "LRET vs FDM Comparison:\n";
        
        if (fdm_result->was_run) {
            std::cout << "+-----------------------+----------------+----------------+\n";
            std::cout << "| Metric                | LRET (best)    | FDM            |\n";
            std::cout << "+-----------------------+----------------+----------------+\n";
            
            std::cout << "| Time (s)              | " << std::right << std::setw(14) 
                      << std::fixed << std::setprecision(4) << fastest->time_seconds
                      << " | " << std::setw(14) << fdm_result->time_seconds << " |\n";
            
            std::cout << "| Final Trace           | " << std::setw(14) 
                      << std::setprecision(5) << fastest->trace_value
                      << " | " << std::setw(14) << fdm_result->trace_value << " |\n";
            
            std::cout << "+-----------------------+----------------+----------------+\n";
            
            double speedup = fdm_result->time_seconds / fastest->time_seconds;
            std::cout << "\nLRET Speedup over FDM: " << std::setprecision(2) << speedup << "x\n";
            
            if (fdm_metrics.has_value()) {
                std::cout << "\nAccuracy (LRET vs FDM):\n";
                std::cout << "  Fidelity:             " << std::fixed << std::setprecision(6) 
                          << fdm_metrics->fidelity << "\n";
                std::cout << "  Trace Distance:       " << std::scientific << std::setprecision(2) 
                          << fdm_metrics->trace_distance << "\n";
                std::cout << "  Frobenius Distance:   " << fdm_metrics->frobenius_distance << "\n";
                std::cout << "  Variational Distance: " << fdm_metrics->variational_distance << "\n";
            }
        } else {
            std::cout << "  Status: NOT RUN\n";
            std::cout << "  Reason: " << fdm_result->skip_reason << "\n";
        }
    }
    
    print_separator(80, '=');
}

void export_to_csv(
    const std::string& filename,
    const CLIOptions& opts,
    const std::vector<ModeResult>& results,
    double noise_in_circuit,
    const std::optional<FDMResult>& fdm_result
) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << "\n";
        return;
    }
    
    // Header
    file << "mode,qubits,depth,noise,time_seconds,final_rank,trace\n";
    
    // Results
    for (const auto& r : results) {
        file << r.mode_name() << ","
             << opts.num_qubits << ","
             << opts.depth << ","
             << std::fixed << std::setprecision(6) << noise_in_circuit << ","
             << std::setprecision(6) << r.time_seconds << ","
             << r.final_rank << ","
             << std::setprecision(5) << r.trace_value << "\n";
    }
    
    // FDM result
    if (fdm_result.has_value() && fdm_result->was_run) {
        file << "fdm,"
             << opts.num_qubits << ","
             << opts.depth << ","
             << noise_in_circuit << ","
             << fdm_result->time_seconds << ","
             << "N/A,"
             << fdm_result->trace_value << "\n";
    }
    
    file.close();
    std::cout << "Results exported to: " << filename << "\n";
}

}  // namespace qlret
