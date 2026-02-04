#include "benchmark_types.h"
#include <sstream>
#include <algorithm>
#include <cmath>
#include <iomanip>

namespace qlret {

//==============================================================================
// Sweep Type Conversions
//==============================================================================

std::string sweep_type_to_string(SweepType type) {
    switch (type) {
        case SweepType::NONE:         return "none";
        case SweepType::EPSILON:      return "epsilon";
        case SweepType::NOISE_PROB:   return "noise";
        case SweepType::QUBITS:       return "qubits";
        case SweepType::DEPTH:        return "depth";
        case SweepType::INITIAL_RANK: return "rank";
        case SweepType::CROSSOVER:    return "crossover";
        default:                      return "unknown";
    }
}

SweepType string_to_sweep_type(const std::string& str) {
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    if (lower == "epsilon" || lower == "eps" || lower == "threshold") return SweepType::EPSILON;
    if (lower == "noise" || lower == "noise_prob" || lower == "p") return SweepType::NOISE_PROB;
    if (lower == "qubits" || lower == "n" || lower == "qubit") return SweepType::QUBITS;
    if (lower == "depth" || lower == "d") return SweepType::DEPTH;
    if (lower == "rank" || lower == "initial_rank") return SweepType::INITIAL_RANK;
    if (lower == "crossover") return SweepType::CROSSOVER;
    
    return SweepType::NONE;
}

//==============================================================================
// Sweep Value Parsing
//==============================================================================

/**
 * Parse sweep string in two formats:
 * 1. Comma-separated: "1e-7,1e-6,1e-5,1e-4,1e-3,1e-2"
 * 2. Range format: "1e-7:1e-2:6" (start:end:count, logarithmic spacing for doubles)
 */
std::vector<double> SweepConfig::parse_double_sweep(const std::string& str) {
    std::vector<double> values;
    
    // Check for range format (contains exactly 2 colons)
    size_t colon_count = std::count(str.begin(), str.end(), ':');
    
    if (colon_count == 2) {
        // Range format: start:end:count
        size_t pos1 = str.find(':');
        size_t pos2 = str.find(':', pos1 + 1);
        
        double start = std::stod(str.substr(0, pos1));
        double end = std::stod(str.substr(pos1 + 1, pos2 - pos1 - 1));
        size_t count = std::stoul(str.substr(pos2 + 1));
        
        if (count < 2) {
            values.push_back(start);
            return values;
        }
        
        // Use logarithmic spacing for values spanning orders of magnitude
        // log-space: log(start), log(end), interpolate
        if (start > 0 && end > 0) {
            double log_start = std::log10(start);
            double log_end = std::log10(end);
            double log_step = (log_end - log_start) / (count - 1);
            
            for (size_t i = 0; i < count; ++i) {
                double log_val = log_start + i * log_step;
                values.push_back(std::pow(10.0, log_val));
            }
        } else {
            // Linear spacing for negative or zero values
            double step = (end - start) / (count - 1);
            for (size_t i = 0; i < count; ++i) {
                values.push_back(start + i * step);
            }
        }
    } else {
        // Comma-separated format
        std::stringstream ss(str);
        std::string token;
        
        while (std::getline(ss, token, ',')) {
            // Trim whitespace
            token.erase(0, token.find_first_not_of(" \t"));
            token.erase(token.find_last_not_of(" \t") + 1);
            
            if (!token.empty()) {
                values.push_back(std::stod(token));
            }
        }
    }
    
    return values;
}

/**
 * Parse sweep string for size_t values:
 * 1. Comma-separated: "5,8,10,12,15,18,20"
 * 2. Range format: "5:20:4" (start:end:step)
 */
std::vector<size_t> SweepConfig::parse_size_sweep(const std::string& str) {
    std::vector<size_t> values;
    
    // Check for range format
    size_t colon_count = std::count(str.begin(), str.end(), ':');
    
    if (colon_count == 2) {
        // Range format: start:end:step
        size_t pos1 = str.find(':');
        size_t pos2 = str.find(':', pos1 + 1);
        
        size_t start = std::stoul(str.substr(0, pos1));
        size_t end = std::stoul(str.substr(pos1 + 1, pos2 - pos1 - 1));
        size_t step = std::stoul(str.substr(pos2 + 1));
        
        if (step == 0) step = 1;
        
        for (size_t v = start; v <= end; v += step) {
            values.push_back(v);
        }
    } else {
        // Comma-separated format
        std::stringstream ss(str);
        std::string token;
        
        while (std::getline(ss, token, ',')) {
            token.erase(0, token.find_first_not_of(" \t"));
            token.erase(token.find_last_not_of(" \t") + 1);
            
            if (!token.empty()) {
                values.push_back(std::stoul(token));
            }
        }
    }
    
    return values;
}

//==============================================================================
// Memory Utilities
//==============================================================================

std::string bytes_to_human_readable(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB", "PB"};
    int unit_idx = 0;
    double size = static_cast<double>(bytes);
    
    while (size >= 1024.0 && unit_idx < 5) {
        size /= 1024.0;
        unit_idx++;
    }
    
    std::ostringstream oss;
    if (unit_idx == 0) {
        oss << bytes << " B";
    } else {
        oss << std::fixed << std::setprecision(2) << size << " " << units[unit_idx];
    }
    
    return oss.str();
}

std::string MemoryComparison::lret_memory_str() const {
    return bytes_to_human_readable(lret_peak_bytes > 0 ? lret_peak_bytes : lret_L_matrix_bytes);
}

std::string MemoryComparison::fdm_memory_str() const {
    size_t mem = fdm_peak_bytes > 0 ? fdm_peak_bytes : fdm_theoretical_bytes();
    return bytes_to_human_readable(mem);
}

}  // namespace qlret
