#pragma once

#include "types.h"
#include <string>
#include <optional>

namespace qlret {

// Parallelization modes
enum class ParallelMode {
    AUTO,       // Auto-select best strategy
    SEQUENTIAL, // No parallelism
    ROW,        // Row-wise parallel
    COLUMN,     // Column-wise parallel
    BATCH,      // Gate batching
    HYBRID,     // Row + batch combined
    COMPARE     // Run all and compare
};

// Command-line options
struct CLIOptions {
    // Simulation parameters
    size_t num_qubits = 11;
    size_t depth = 13;
    double noise_prob = 0.01;
    double truncation_threshold = 1e-4;
    size_t batch_size = 0;  // 0 = auto
    
    // High-rank testing (for parallelization benchmarking)
    // When > 1, starts with a random mixed state of given rank instead of |0...0>
    // This enables meaningful parallelization testing since rank=1 has no parallel benefit
    size_t initial_rank = 1;
    unsigned int random_seed = 0;  // 0 = time-based seed
    
    // Parallelization
    ParallelMode parallel_mode = ParallelMode::AUTO;
    size_t num_threads = 0;  // 0 = use all cores
    
    // FDM options
    bool enable_fdm = false;
    size_t fdm_threshold = 10;
    
    // Output options
    bool verbose = false;
    std::optional<std::string> output_file;
    
    // Flags
    bool show_help = false;
    bool show_version = false;
};

// Parse command line arguments
CLIOptions parse_arguments(int argc, char* argv[]);

// Print help message
void print_help();

// Print version
void print_version();

// Validate options
bool validate_options(const CLIOptions& opts, std::string& error_msg);

// Mode string conversion
std::string parallel_mode_to_string(ParallelMode mode);
ParallelMode string_to_parallel_mode(const std::string& str);

}  // namespace qlret
