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

USAGE:
    quantum_sim [OPTIONS]

OPTIONS:
    -n, --qubits N        Number of qubits (default: 11)
    -d, --depth N         Circuit depth (default: 13)
    -b, --batch N         Batch size, 0=auto (default: 0)
    -t, --threshold F     Truncation threshold (default: 1e-4)
    --noise F             Noise probability (default: 0.01)

    --noise-type TYPE     Type of noise to apply:
                          all         - All noise types (default)
                          depolarizing - Only depolarizing noise
                          amplitude   - Only amplitude damping
                          phase       - Only phase damping
                          realistic   - Realistic mix (varied rates)
                          none        - No noise (pure unitary)

    --mode MODE           Parallelization mode:
                          auto|sequential|row|column|batch|hybrid|compare
                          (default: auto)

    --initial-rank N      Start with random mixed state of rank N (default: 1)
                          Rank=1 is pure state |0...0>.
                          Higher rank enables meaningful parallel benchmarking
                          since pure states have only 1 column to process.
    --seed N              Random seed for mixed state (default: 0=time-based)

    --fdm                 Enable FDM comparison (memory permitting)
    --fdm-force           Force FDM even with insufficient memory (test limits)

    -o, --output FILE     Export results to CSV
    -v, --verbose         Detailed output
    --threads N           Limit thread count, 0=all cores (default: 0)

    -h, --help            Show this help
    --version             Show version

EXAMPLES:
    quantum_sim -n 11 -d 13
    quantum_sim -n 10 --mode compare --fdm
    quantum_sim -n 14 --fdm-force   # Test FDM at the memory limit
    quantum_sim -n 8 --mode row -v --output results.csv
    quantum_sim -n 10 --noise-type depolarizing --noise 0.05

)";
}
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
        if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            opts.output_file = argv[++i];
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
        
        // Unknown option
        std::cerr << "Warning: Unknown option '" << arg << "'\n";
    }
    
    return opts;
}

bool validate_options(const CLIOptions& opts, std::string& error_msg) {
    if (opts.num_qubits < 1 || opts.num_qubits > 20) {
        error_msg = "Qubits must be between 1 and 20";
        return false;
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
