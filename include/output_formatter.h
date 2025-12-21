#pragma once

#include "types.h"
#include "cli_parser.h"
#include <vector>
#include <string>
#include <optional>

namespace qlret {

// Result from a single parallelization mode run
struct ModeResult {
    ParallelMode mode;
    double time_seconds;
    MatrixXcd L_final;
    size_t final_rank;
    double trace_value;
    
    // Metrics for comparison (computed vs sequential baseline)
    double speedup = 1.0;           // time_sequential / time_this
    double fidelity = 0.0;          // vs sequential result
    double trace_distance = 0.0;
    double frobenius_distance = 0.0;
    double distortion = 0.0;        // ||L_this - L_seq||_F / ||L_seq||_F
    
    std::string mode_name() const { return parallel_mode_to_string(mode); }
};

// FDM simulation result
struct FDMResult {
    bool was_run = false;
    std::string skip_reason;
    double time_seconds = 0.0;
    MatrixXcd rho_final;
    double trace_value = 0.0;
};

// Metrics comparing two states
struct MetricsResult {
    double fidelity = 0.0;
    double trace_distance = 0.0;
    double frobenius_distance = 0.0;
    double variational_distance = 0.0;
};

// Print standard output for single mode run
void print_standard_output(
    const CLIOptions& opts,
    const ModeResult& result,
    const MetricsResult& metrics,
    double noise_in_circuit,
    const std::optional<FDMResult>& fdm_result = std::nullopt,
    const std::optional<MetricsResult>& fdm_metrics = std::nullopt
);

// Print comparison table for --mode compare
void print_comparison_output(
    const CLIOptions& opts,
    const std::vector<ModeResult>& results,
    double noise_in_circuit,
    const std::optional<FDMResult>& fdm_result = std::nullopt,
    const std::optional<MetricsResult>& fdm_metrics = std::nullopt
);

// Export results to CSV
void export_to_csv(
    const std::string& filename,
    const CLIOptions& opts,
    const std::vector<ModeResult>& results,
    double noise_in_circuit,
    const std::optional<FDMResult>& fdm_result = std::nullopt
);

// Print configuration header
void print_config_header(const CLIOptions& opts, double noise_in_circuit, bool fdm_enabled);

// Print horizontal separator
void print_separator(size_t width = 80, char c = '=');

}  // namespace qlret
