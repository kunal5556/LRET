#pragma once

#include "types.h"
#include <string>
#include <chrono>
#include <atomic>
#include <csignal>
#include <functional>

namespace qlret {

//==============================================================================
// Global Interrupt Flag (for signal handling)
//==============================================================================

// Atomic flag set by signal handler
extern std::atomic<bool> g_interrupted;

// Initialize signal handlers (call once at program start)
void setup_signal_handlers();

// Check if interrupted (use in loops)
bool is_interrupted();

// Reset interrupt flag
void reset_interrupt();

//==============================================================================
// Swap Detection
//==============================================================================

struct SwapStatus {
    bool is_using_swap = false;
    size_t swap_used_mb = 0;
    size_t swap_total_mb = 0;
    double swap_percent = 0.0;
};

// Get current swap usage
SwapStatus get_swap_status();

// Check swap and prompt user if needed
// Returns true if should continue, false if user wants to abort
bool check_swap_and_prompt(bool allow_swap_flag, bool interactive = true);

//==============================================================================
// Timeout Management
//==============================================================================

struct TimeoutConfig {
    bool enabled = false;
    std::chrono::seconds duration{0};
    std::chrono::steady_clock::time_point start_time;
    
    void start() {
        start_time = std::chrono::steady_clock::now();
    }
    
    bool is_expired() const {
        if (!enabled) return false;
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        return elapsed >= duration;
    }
    
    double remaining_seconds() const {
        if (!enabled) return -1.0;
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        auto remaining = duration - std::chrono::duration_cast<std::chrono::seconds>(elapsed);
        return std::max(0.0, static_cast<double>(remaining.count()));
    }
    
    std::string elapsed_string() const {
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        auto secs = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
        
        if (secs < 60) return std::to_string(secs) + "s";
        if (secs < 3600) return std::to_string(secs / 60) + "m " + std::to_string(secs % 60) + "s";
        if (secs < 86400) return std::to_string(secs / 3600) + "h " + std::to_string((secs % 3600) / 60) + "m";
        return std::to_string(secs / 86400) + "d " + std::to_string((secs % 86400) / 3600) + "h";
    }
};

// Global timeout config
extern TimeoutConfig g_timeout;

// Parse timeout string (e.g., "60", "5m", "2h", "1d")
std::chrono::seconds parse_timeout_string(const std::string& str);

// Check if should abort (timeout or interrupt)
bool should_abort();

//==============================================================================
// Container Detection
//==============================================================================

struct ContainerInfo {
    bool is_container = false;
    std::string container_type;  // "docker", "podman", "singularity", "lxc", ""
    size_t memory_limit_mb = 0;  // 0 = unlimited or unknown
    size_t memory_available_mb = 0;
};

// Detect if running in a container
ContainerInfo detect_container();

//==============================================================================
// LRET Resource Monitoring
//==============================================================================

struct LRETResourceEstimate {
    size_t base_memory_mb;      // Initial L matrix
    size_t estimated_peak_mb;   // Rough estimate based on depth and noise
    size_t current_memory_mb;   // Actual current usage
    size_t max_observed_rank;
    bool memory_warning;        // True if approaching limits
    std::string warning_message;
};

// Estimate LRET memory requirements
LRETResourceEstimate estimate_lret_resources(
    size_t num_qubits, 
    size_t depth, 
    double noise_prob,
    size_t initial_rank = 1
);

// Update estimate with actual observation
void update_lret_estimate(LRETResourceEstimate& estimate, size_t current_rank, size_t num_qubits);

// Get current process memory usage
size_t get_current_memory_usage_mb();

//==============================================================================
// Resource Check Result
//==============================================================================

struct ResourceCheckResult {
    bool should_continue = true;
    bool has_warnings = false;
    std::vector<std::string> warnings;
    std::vector<std::string> errors;
    
    void add_warning(const std::string& msg) {
        has_warnings = true;
        warnings.push_back(msg);
    }
    
    void add_error(const std::string& msg) {
        should_continue = false;
        errors.push_back(msg);
    }
};

// Comprehensive resource check at program start
ResourceCheckResult check_all_resources(
    size_t num_qubits,
    size_t depth,
    double noise_prob,
    bool fdm_enabled,
    bool allow_swap
);

}  // namespace qlret
