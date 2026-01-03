#pragma once

#include "types.h"
#include "benchmark_types.h"
#include <string>
#include <optional>
#include <vector>

namespace qlret {

//==============================================================================
// Compound Benchmark Specification
// Allows each benchmark to have its own range AND fixed parameters
//==============================================================================
struct BenchmarkSpec {
    SweepType type = SweepType::NONE;
    std::string range_str;              // e.g., "1e-7:1e-2:6" or "0.0,0.01,0.05"
    
    // Fixed parameters for this specific benchmark (optional overrides)
    std::optional<size_t> fixed_qubits;
    std::optional<size_t> fixed_depth;
    std::optional<double> fixed_noise;
    std::optional<double> fixed_epsilon;
    std::optional<size_t> fixed_rank;
    std::optional<size_t> trials;       // Per-benchmark trial count
    
    // Parse a compound spec string like "range=1e-7:1e-2:6,n=12,d=20,noise=0.01,trials=5"
    static BenchmarkSpec parse(const std::string& spec_str, SweepType type);
};

// Parallelization modes
enum class ParallelMode {
    AUTO,       // Auto-select best strategy
    SEQUENTIAL, // No parallelism
    ROW,        // Row-wise parallel
    COLUMN,     // Column-wise parallel
    BATCH,      // Gate batching
    HYBRID,     // Row + batch combined
    COMPARE,    // Run all and compare
    MPI_ROW,    // MPI row-wise distribution (Phase 3)
    MPI_COLUMN, // MPI column-wise distribution (Phase 3)
    MPI_HYBRID  // MPI + OpenMP hybrid (Phase 3)
};

// Noise selection for CLI (which noise types to enable)
enum class NoiseSelection {
    ALL,           // All noise types (default)
    DEPOLARIZING,  // Only depolarizing noise
    AMPLITUDE,     // Only amplitude damping
    PHASE,         // Only phase damping
    REALISTIC,     // Realistic mix (different rates)
    NONE           // No noise (for pure unitary evolution)
};

// Command-line options
struct CLIOptions {
    // Simulation parameters
    size_t num_qubits = 11;
    size_t depth = 13;
    double noise_prob = 0.01;
    double truncation_threshold = 1e-4;
    size_t batch_size = 0;  // 0 = auto
    
    // Gate fusion optimization (Phase 1.1 of roadmap)
    bool enable_fusion = true;         // --fuse-gates / --no-fuse
    size_t min_fusion_gates = 2;       // --min-fusion N
    size_t max_fusion_depth = 50;      // --max-fusion-depth N
    
    // Circuit stratification (Phase 1.3 of roadmap - Cirq pattern)
    bool enable_stratify = true;       // --stratify / --no-stratify
    bool greedy_layers = true;         // --greedy-layers / --asap-layers
    size_t min_layer_size = 1;         // --min-layer-size N
    
    // Noise type selection
    NoiseSelection noise_selection = NoiseSelection::ALL;
    
    // High-rank testing (for parallelization benchmarking)
    // When > 1, starts with a random mixed state of given rank instead of |0...0>
    // This enables meaningful parallelization testing since rank=1 has no parallel benefit
    size_t initial_rank = 1;
    unsigned int random_seed = 0;  // 0 = time-based seed
    
    // Parallelization
    ParallelMode parallel_mode = ParallelMode::AUTO;
    size_t num_threads = 0;  // 0 = use all cores
    
    // MPI options (Phase 3 of roadmap)
    bool enable_mpi = false;           // --mpi / --mode=mpi-row
    bool mpi_row_dist = true;          // --mpi-row (default) / --mpi-column
    bool mpi_verbose = false;          // --mpi-verbose
    bool mpi_validate = false;         // --mpi-validate (check vs local)
    
    // GPU options (Phase 2 of roadmap)
    bool enable_gpu = false;           // --gpu / --device=gpu
    bool auto_device = true;           // --device=auto (default)
    int gpu_device_id = 0;             // --gpu-id N
    bool use_cuquantum = true;         // --cuquantum / --no-cuquantum
    size_t gpu_memory_limit = 0;       // --gpu-memory-limit N (in GB, 0=no limit)
    
    // Noise model import (Phase 4.1 of roadmap)
    std::string noise_model_path;      // --noise-model path/to/noise.json
    bool validate_noise_model = true;  // --validate-noise / --no-validate-noise
    bool print_noise_summary = false;  // --print-noise-summary
    bool enable_correlated_errors = false;   // --enable-correlated-errors
    bool enable_time_dependent_noise = false; // --enable-time-dependent
    bool enable_memory_effects = false;      // --enable-memory-effects
    size_t max_memory_depth = 2;             // --max-memory-depth

    // Leakage and measurement (Phase 4.4/4.5)
    bool enable_leakage = false;             // --enable-leakage
    bool enable_measurement_errors = false;  // --enable-measurement-errors
    bool enable_conditional_measurement = false; // --enable-conditional-measurement
    
    // FDM options
    bool enable_fdm = false;
    bool fdm_force = false;  // Bypass memory check, attempt FDM anyway
    
    // Resource management
    bool allow_swap = false;          // Skip swap warning prompt
    std::string timeout_str = "";     // Timeout string (e.g., "1h", "2d")
    bool non_interactive = false;     // Skip all prompts
    
    // Output options
    bool verbose = false;
    bool show_timing = false;      // --show-timing (display timing breakdown)
    std::optional<std::string> output_file;
    bool generate_output = false;  // True if -o flag was given (with or without filename)

    // JSON/PennyLane bridge (Phase 5)
    std::string input_json_path;            // --input-json path/to/circuit.json
    std::optional<std::string> output_json_path; // --output-json path/to/result.json (optional)
    bool json_export_state = false;         // --export-json-state
    
    //==========================================================================
    // Parameter Sweep Options (LRET Paper Benchmarking)
    //==========================================================================
    SweepConfig sweep_config;          // Sweep configuration
    
    // Sweep strings (parsed into sweep_config)
    std::string sweep_epsilon_str;     // --sweep-epsilon "1e-7:1e-2:6"
    std::string sweep_noise_str;       // --sweep-noise "0.0,0.01,0.05,0.1,0.2"
    std::string sweep_qubits_str;      // --sweep-qubits "5:20:1"
    std::string sweep_depth_str;       // --sweep-depth "10,20,50,100"
    std::string sweep_rank_str;        // --sweep-rank "1,2,4,8,16"
    
    bool sweep_crossover = false;      // --sweep-crossover (LRET vs FDM analysis)
    bool track_rank_evolution = false; // --track-rank (record rank after each op)
    size_t sweep_trials = 1;           // --sweep-trials N (repeat for statistics)
    
    bool benchmark_all = false;        // --benchmark-all (run all paper benchmarks)
    
    //==========================================================================
    // Compound Benchmark Specs (--bench-* options with custom ranges + params)
    //==========================================================================
    std::vector<BenchmarkSpec> benchmark_specs;  // Multiple benchmarks to run
    
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

// Noise selection conversion
std::string noise_selection_to_string(NoiseSelection sel);
NoiseSelection string_to_noise_selection(const std::string& str);

}  // namespace qlret
