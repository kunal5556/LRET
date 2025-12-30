#include "cli_parser.h"
#include <iostream>
#include <algorithm>
#include <cstring>

namespace qlret {

std::string parallel_mode_to_string(ParallelMode mode) {
    switch (mode) {
        case ParallelMode::AUTO:       return "auto";
        case ParallelMode::SEQUENTIAL: return "sequential";
        case ParallelMode::ROW:        return "row";
        case ParallelMode::COLUMN:     return "column";
        case ParallelMode::BATCH:      return "batch";
        case ParallelMode::HYBRID:     return "hybrid";
        case ParallelMode::COMPARE:    return "compare";
        default:                       return "unknown";
    }
}

ParallelMode string_to_parallel_mode(const std::string& str) {
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    if (lower == "auto")       return ParallelMode::AUTO;
    if (lower == "sequential") return ParallelMode::SEQUENTIAL;
    if (lower == "row")        return ParallelMode::ROW;
    if (lower == "column")     return ParallelMode::COLUMN;
    if (lower == "batch")      return ParallelMode::BATCH;
    if (lower == "hybrid")     return ParallelMode::HYBRID;
    if (lower == "compare")    return ParallelMode::COMPARE;
    
    return ParallelMode::AUTO;  // Default fallback
}

std::string noise_selection_to_string(NoiseSelection sel) {
    switch (sel) {
        case NoiseSelection::ALL:          return "all";
        case NoiseSelection::DEPOLARIZING: return "depolarizing";
        case NoiseSelection::AMPLITUDE:    return "amplitude";
        case NoiseSelection::PHASE:        return "phase";
        case NoiseSelection::REALISTIC:    return "realistic";
        case NoiseSelection::NONE:         return "none";
        default:                           return "unknown";
    }
}

NoiseSelection string_to_noise_selection(const std::string& str) {
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    if (lower == "all")          return NoiseSelection::ALL;
    if (lower == "depolarizing") return NoiseSelection::DEPOLARIZING;
    if (lower == "amplitude")    return NoiseSelection::AMPLITUDE;
    if (lower == "phase")        return NoiseSelection::PHASE;
    if (lower == "realistic")    return NoiseSelection::REALISTIC;
    if (lower == "none")         return NoiseSelection::NONE;
    
    return NoiseSelection::ALL;  // Default fallback
}

void print_help() {
    std::cout << R"(
QuantumLRET-Sim - Low-Rank Exact Tensor Quantum Simulator
Version 1.0.0

USAGE:
    quantum_sim [OPTIONS]

SIMULATION OPTIONS:
    -n, --qubits N        Number of qubits (1-20, default: 11)
    -d, --depth N         Circuit depth (default: 13)
    -b, --batch N         Batch size for parallel processing (0=auto, default: 0)
    -t, --threshold F     Truncation threshold for rank control (default: 1e-4)
    --threads N           Limit OpenMP thread count (0=all cores, default: 0)

NOISE OPTIONS:
    --noise F             Noise probability per gate (0.0-1.0, default: 0.01)
    --noise-type TYPE     Type of noise to apply:
                          all         - All noise types randomly (default)
                          depolarizing - Depolarizing channel only
                          amplitude   - Amplitude damping only
                          phase       - Phase damping only
                          realistic   - Realistic mix with varied rates
                          none        - No noise (pure unitary evolution)

PARALLELIZATION OPTIONS:
    --mode MODE           Parallelization strategy:
                          auto       - Auto-select best mode (default)
                          sequential - No parallelization (baseline)
                          row        - Parallelize over matrix rows
                          column     - Parallelize over matrix columns
                          batch      - Batch gate application
                          hybrid     - Combined row/column strategy
                          compare    - Run all modes and compare performance

INITIAL STATE OPTIONS:
    --initial-rank N      Start with random mixed state of rank N (default: 1)
                          Rank=1 is pure state |0...0>.
                          Higher rank enables meaningful parallel benchmarking.
    --seed N              Random seed for mixed state generation (0=time-based)

FDM (FULL DENSITY MATRIX) OPTIONS:
    --fdm                 Enable FDM comparison (if memory permits)
    --fdm-force           Force FDM even with insufficient memory (test limits)

PARAMETER SWEEP OPTIONS (LRET Paper Benchmarking):
    --benchmark-all       Run ALL paper benchmarks with default settings.
                          Includes: epsilon sweep, noise sweep, qubit sweep,
                          crossover analysis, and rank tracking. Output saved
                          to benchmark_results.csv (or use -o to specify).
    --sweep-epsilon STR   Sweep truncation threshold (epsilon).
                          Format: "1e-7,1e-6,1e-5,1e-4" or "1e-7:1e-2:6" (log-spaced)
    --sweep-noise STR     Sweep noise probability.
                          Format: "0.0,0.01,0.05,0.1,0.2" or "0.0:0.2:11"
    --sweep-qubits STR    Sweep number of qubits.
                          Format: "5,8,10,12,15" or "5:20:1" (step=1)
    --sweep-depth STR     Sweep circuit depth.
                          Format: "10,20,50,100" or "10:100:10"
    --sweep-rank STR      Sweep initial rank for parallel benchmarking.
                          Format: "1,2,4,8,16,32" or "1:64:2" (powers)
    --sweep-crossover STR LRET vs FDM crossover analysis.
                          Format: "5:15" (min:max qubits) or "5:15:1" (min:max:step)
                          Default: "5:15:1" if no argument given
    --track-rank          Track rank evolution after each operation
    --sweep-trials N      Number of trials per sweep point (for statistics)

RESOURCE MANAGEMENT OPTIONS:
    --allow-swap          Continue even if system is using swap memory
    --timeout TIME        Timeout for simulation. Formats supported:
                          60        - 60 seconds
                          5m        - 5 minutes
                          2h        - 2 hours
                          1d        - 1 day
    --non-interactive     Skip all prompts (for scripted/automated runs)

OUTPUT OPTIONS:
    -o, --output FILE     Export results to CSV file. Features:
                          - Progressive: writes after each operation
                          - Absolute path shown at start for monitoring
                          - Can use 'tail -f FILE' to watch progress
    -v, --verbose         Show detailed step-by-step output

HELP OPTIONS:
    -h, --help            Show this help message
    --version             Show version information

EXAMPLES:
    # Basic simulation with 11 qubits and depth 13
    quantum_sim -n 11 -d 13

    # Compare all parallelization modes with FDM validation
    quantum_sim -n 10 --mode compare --fdm

    # Force FDM at memory limit for testing
    quantum_sim -n 14 --fdm-force

    # Detailed output with CSV export
    quantum_sim -n 8 --mode row -v --output results.csv

    # Specific noise model with custom probability
    quantum_sim -n 10 --noise-type depolarizing --noise 0.05

    # Long-running simulation with 2 hour timeout
    quantum_sim -n 20 --timeout 2h -o long_run.csv

    # Automated/scripted run (no prompts)
    quantum_sim -n 15 --allow-swap --non-interactive

    # High-rank initial state for parallel benchmarking
    quantum_sim -n 12 --initial-rank 16 --mode compare

    # ============ LRET Paper Benchmarking Examples ============

    # Run ALL paper benchmarks at once (comprehensive analysis)
    quantum_sim --benchmark-all -o paper_benchmarks.csv

    # Sweep truncation threshold (Figure: epsilon vs time/rank)
    quantum_sim -n 12 -d 20 --sweep-epsilon "1e-7:1e-2:6" -o epsilon_sweep.csv

    # Sweep noise probability (Figure: noise vs time/rank)
    quantum_sim -n 10 -d 15 --sweep-noise "0.0,0.01,0.05,0.1,0.2" -o noise_sweep.csv

    # Sweep qubit count (Figure: n vs time)
    quantum_sim -d 15 --sweep-qubits "5:18:1" --fdm -o qubit_sweep.csv

    # LRET vs FDM crossover analysis (custom range)
    quantum_sim -d 20 --sweep-crossover "8:20:1" --fdm -o crossover.csv

    # Track rank evolution through circuit (Figure: rank vs depth)
    quantum_sim -n 12 -d 50 --track-rank -o rank_evolution.csv

For more information, see: https://github.com/kunal5556/LRET

)";
}

void print_version() {
    std::cout << "QuantumLRET-Sim v1.0.0\n";
    std::cout << "Low-Rank Exact Tensor Quantum Circuit Simulator\n";
}

CLIOptions parse_arguments(int argc, char* argv[]) {
    CLIOptions opts;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        // Help
        if (arg == "-h" || arg == "--help") {
            opts.show_help = true;
            return opts;
        }
        
        // Version
        if (arg == "--version") {
            opts.show_version = true;
            return opts;
        }
        
        // Qubits
        if ((arg == "-n" || arg == "--qubits") && i + 1 < argc) {
            opts.num_qubits = std::stoul(argv[++i]);
            continue;
        }
        
        // Depth
        if ((arg == "-d" || arg == "--depth") && i + 1 < argc) {
            opts.depth = std::stoul(argv[++i]);
            continue;
        }
        
        // Batch size
        if ((arg == "-b" || arg == "--batch") && i + 1 < argc) {
            opts.batch_size = std::stoul(argv[++i]);
            continue;
        }
        
        // Truncation threshold
        if ((arg == "-t" || arg == "--threshold") && i + 1 < argc) {
            opts.truncation_threshold = std::stod(argv[++i]);
            continue;
        }
        
        // Noise probability
        if (arg == "--noise" && i + 1 < argc) {
            opts.noise_prob = std::stod(argv[++i]);
            continue;
        }
        
        // Initial rank (for high-rank testing)
        if (arg == "--initial-rank" && i + 1 < argc) {
            opts.initial_rank = std::stoul(argv[++i]);
            continue;
        }
        
        // Random seed
        if (arg == "--seed" && i + 1 < argc) {
            opts.random_seed = static_cast<unsigned int>(std::stoul(argv[++i]));
            continue;
        }
        
        // Parallel mode
        if (arg == "--mode" && i + 1 < argc) {
            opts.parallel_mode = string_to_parallel_mode(argv[++i]);
            continue;
        }
        
        // Noise type
        if (arg == "--noise-type" && i + 1 < argc) {
            opts.noise_selection = string_to_noise_selection(argv[++i]);
            continue;
        }
        
        // FDM enable
        if (arg == "--fdm") {
            opts.enable_fdm = true;
            continue;
        }
        
        // FDM force (bypass memory check)
        if (arg == "--fdm-force") {
            opts.enable_fdm = true;
            opts.fdm_force = true;
            continue;
        }
        
        // Output file
        // -o or --output: enables output generation
        // -o filename: uses custom filename
        // -o (alone or followed by another flag): uses default filename
        if (arg == "-o" || arg == "--output") {
            opts.generate_output = true;
            
            // Check if next argument exists and is not a flag (doesn't start with -)
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                opts.output_file = argv[++i];
            }
            // If no filename provided, output_file stays empty (will use default)
            continue;
        }
        
        // Verbose
        if (arg == "-v" || arg == "--verbose") {
            opts.verbose = true;
            continue;
        }
        
        // Thread count
        if (arg == "--threads" && i + 1 < argc) {
            opts.num_threads = std::stoul(argv[++i]);
            continue;
        }
        
        // Allow swap memory (skip swap warning)
        if (arg == "--allow-swap") {
            opts.allow_swap = true;
            continue;
        }
        
        // Timeout for simulation
        if (arg == "--timeout" && i + 1 < argc) {
            opts.timeout_str = argv[++i];
            continue;
        }
        
        // Non-interactive mode (skip all prompts)
        if (arg == "--non-interactive") {
            opts.non_interactive = true;
            continue;
        }
        
        //======================================================================
        // Parameter Sweep Options (LRET Paper Benchmarking)
        //======================================================================
        
        // Run ALL paper benchmarks
        if (arg == "--benchmark-all") {
            opts.benchmark_all = true;
            // This sets up a special mode that runs all sweeps sequentially
            // Individual sweep values will be set up later in main.cpp
            continue;
        }
        
        // Sweep truncation threshold (epsilon)
        if (arg == "--sweep-epsilon" && i + 1 < argc) {
            opts.sweep_epsilon_str = argv[++i];
            opts.sweep_config.type = SweepType::EPSILON;
            opts.sweep_config.epsilon_values = SweepConfig::parse_double_sweep(opts.sweep_epsilon_str);
            continue;
        }
        
        // Sweep noise probability
        if (arg == "--sweep-noise" && i + 1 < argc) {
            opts.sweep_noise_str = argv[++i];
            opts.sweep_config.type = SweepType::NOISE_PROB;
            opts.sweep_config.noise_values = SweepConfig::parse_double_sweep(opts.sweep_noise_str);
            continue;
        }
        
        // Sweep qubit count
        if (arg == "--sweep-qubits" && i + 1 < argc) {
            opts.sweep_qubits_str = argv[++i];
            opts.sweep_config.type = SweepType::QUBITS;
            opts.sweep_config.qubit_values = SweepConfig::parse_size_sweep(opts.sweep_qubits_str);
            continue;
        }
        
        // Sweep circuit depth
        if (arg == "--sweep-depth" && i + 1 < argc) {
            opts.sweep_depth_str = argv[++i];
            opts.sweep_config.type = SweepType::DEPTH;
            opts.sweep_config.depth_values = SweepConfig::parse_size_sweep(opts.sweep_depth_str);
            continue;
        }
        
        // Sweep initial rank
        if (arg == "--sweep-rank" && i + 1 < argc) {
            opts.sweep_rank_str = argv[++i];
            opts.sweep_config.type = SweepType::INITIAL_RANK;
            opts.sweep_config.rank_values = SweepConfig::parse_size_sweep(opts.sweep_rank_str);
            continue;
        }
        
        // LRET vs FDM crossover analysis
        if (arg == "--sweep-crossover") {
            opts.sweep_crossover = true;
            opts.sweep_config.type = SweepType::CROSSOVER;
            opts.sweep_config.include_fdm = true;
            
            // Check if next argument is a range specification (not another flag)
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                std::string range_str = argv[++i];
                // Parse range: "min:max" or "min:max:step"
                auto values = SweepConfig::parse_size_sweep(range_str);
                if (!values.empty()) {
                    opts.sweep_config.qubit_values = values;
                    opts.sweep_config.crossover_min_qubits = values.front();
                    opts.sweep_config.crossover_max_qubits = values.back();
                }
            } else {
                // Default crossover range: 5-15 qubits with step 1
                opts.sweep_config.qubit_values.clear();
                for (size_t n = opts.sweep_config.crossover_min_qubits; 
                     n <= opts.sweep_config.crossover_max_qubits; ++n) {
                    opts.sweep_config.qubit_values.push_back(n);
                }
            }
            continue;
        }
        
        // Track rank evolution
        if (arg == "--track-rank") {
            opts.track_rank_evolution = true;
            opts.sweep_config.track_rank_evolution = true;
            continue;
        }
        
        // Number of trials per sweep point
        if (arg == "--sweep-trials" && i + 1 < argc) {
            opts.sweep_trials = std::stoul(argv[++i]);
            opts.sweep_config.num_trials = opts.sweep_trials;
            continue;
        }
        
        // Unknown option
        std::cerr << "Warning: Unknown option '" << arg << "'\n";
    }
    
    // Post-processing: if FDM is enabled and sweep is active, propagate flag
    if (opts.enable_fdm && opts.sweep_config.is_active()) {
        opts.sweep_config.include_fdm = true;
    }
    
    return opts;
}

bool validate_options(const CLIOptions& opts, std::string& error_msg) {
    if (opts.num_qubits < 1) {
        error_msg = "Qubits must be at least 1";
        return false;
    }
    
    // Warning for large qubit counts (memory scales as 2^n)
    if (opts.num_qubits > 20) {
        std::cerr << "Warning: " << opts.num_qubits << " qubits requires ~" 
                  << (1ULL << opts.num_qubits) * sizeof(std::complex<double>) / (1024*1024) 
                  << " MB for state vector. Ensure sufficient memory.\n";
    }
    
    if (opts.depth < 1) {
        error_msg = "Depth must be at least 1";
        return false;
    }
    
    if (opts.noise_prob < 0.0 || opts.noise_prob > 1.0) {
        error_msg = "Noise probability must be between 0 and 1";
        return false;
    }
    
    if (opts.truncation_threshold <= 0.0) {
        error_msg = "Truncation threshold must be positive";
        return false;
    }
    
    // Initial rank validation
    size_t dim = 1ULL << opts.num_qubits;
    if (opts.initial_rank < 1 || opts.initial_rank > dim) {
        error_msg = "Initial rank must be between 1 and 2^n (n=num_qubits)";
        return false;
    }
    
    return true;
}

}  // namespace qlret
