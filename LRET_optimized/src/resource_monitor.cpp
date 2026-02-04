#include "resource_monitor.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cctype>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <unistd.h>
#include <sys/resource.h>
#else
#include <sys/sysinfo.h>
#include <unistd.h>
#include <sys/resource.h>
#endif

namespace qlret {

//==============================================================================
// Global Variables
//==============================================================================

std::atomic<bool> g_interrupted{false};
TimeoutConfig g_timeout;

//==============================================================================
// Signal Handling
//==============================================================================

void signal_handler(int sig) {
    g_interrupted = true;
    std::cout << "\n[!] Interrupt received (signal " << sig << "). Finishing current operation...\n";
    std::cout << "    (Press Ctrl+C again to force quit)\n";
    
    // Restore default handler so second Ctrl+C kills immediately
    std::signal(sig, SIG_DFL);
}

void setup_signal_handlers() {
    std::signal(SIGINT, signal_handler);   // Ctrl+C
    std::signal(SIGTERM, signal_handler);  // Termination request
#ifndef _WIN32
    std::signal(SIGHUP, signal_handler);   // Terminal closed
#endif
}

bool is_interrupted() {
    return g_interrupted.load();
}

void reset_interrupt() {
    g_interrupted = false;
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
}

//==============================================================================
// Swap Detection
//==============================================================================

SwapStatus get_swap_status() {
    SwapStatus status;
    
#ifdef _WIN32
    MEMORYSTATUSEX mem;
    mem.dwLength = sizeof(mem);
    if (GlobalMemoryStatusEx(&mem)) {
        // Windows uses "page file" instead of swap
        size_t page_total = mem.ullTotalPageFile / (1024 * 1024);
        size_t page_avail = mem.ullAvailPageFile / (1024 * 1024);
        size_t phys_total = mem.ullTotalPhys / (1024 * 1024);
        
        // Page file > physical memory means swap is configured
        if (page_total > phys_total) {
            status.swap_total_mb = page_total - phys_total;
            status.swap_used_mb = (page_total - page_avail) > phys_total ? 
                                  (page_total - page_avail - phys_total) : 0;
            status.is_using_swap = status.swap_used_mb > 100;  // > 100MB used
            if (status.swap_total_mb > 0) {
                status.swap_percent = 100.0 * status.swap_used_mb / status.swap_total_mb;
            }
        }
    }
#else
    // Linux: read /proc/meminfo
    std::ifstream meminfo("/proc/meminfo");
    if (meminfo.is_open()) {
        std::string line;
        size_t swap_total = 0, swap_free = 0;
        
        while (std::getline(meminfo, line)) {
            if (line.find("SwapTotal:") == 0) {
                std::istringstream iss(line.substr(10));
                iss >> swap_total;  // In kB
            } else if (line.find("SwapFree:") == 0) {
                std::istringstream iss(line.substr(9));
                iss >> swap_free;
            }
        }
        
        status.swap_total_mb = swap_total / 1024;
        status.swap_used_mb = (swap_total - swap_free) / 1024;
        status.is_using_swap = status.swap_used_mb > 100;  // > 100MB used
        if (status.swap_total_mb > 0) {
            status.swap_percent = 100.0 * status.swap_used_mb / status.swap_total_mb;
        }
    }
#endif
    
    return status;
}

bool check_swap_and_prompt(bool allow_swap_flag, bool interactive) {
    SwapStatus swap = get_swap_status();
    
    if (!swap.is_using_swap) {
        return true;  // No swap usage, continue
    }
    
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    SWAP USAGE WARNING                        ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  System is currently using swap memory.                      ║\n";
    std::cout << "║  Swap used: " << std::setw(6) << swap.swap_used_mb << " MB / " 
              << std::setw(6) << swap.swap_total_mb << " MB (" 
              << std::fixed << std::setprecision(1) << swap.swap_percent << "%)           ║\n";
    std::cout << "║                                                              ║\n";
    std::cout << "║  This can cause simulations to run 100-1000x SLOWER!         ║\n";
    std::cout << "║  Consider freeing memory before running large simulations.   ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    
    if (allow_swap_flag) {
        std::cout << "[--allow-swap] Continuing despite swap usage.\n\n";
        return true;
    }
    
    if (!interactive) {
        std::cout << "Non-interactive mode: use --allow-swap to bypass this check.\n";
        return false;
    }
    
    std::cout << "Continue anyway? [y/N]: ";
    std::string response;
    std::getline(std::cin, response);
    
    if (response.empty() || (response[0] != 'y' && response[0] != 'Y')) {
        std::cout << "Aborting. Free memory or use --allow-swap to continue.\n";
        return false;
    }
    
    return true;
}

//==============================================================================
// Timeout Management
//==============================================================================

std::chrono::seconds parse_timeout_string(const std::string& str) {
    if (str.empty()) return std::chrono::seconds(0);
    
    std::string num_part;
    char unit = 's';  // Default to seconds
    
    for (char c : str) {
        if (std::isdigit(c) || c == '.') {
            num_part += c;
        } else {
            unit = std::tolower(c);
            break;
        }
    }
    
    if (num_part.empty()) return std::chrono::seconds(0);
    
    double value = std::stod(num_part);
    
    switch (unit) {
        case 's': return std::chrono::seconds(static_cast<long long>(value));
        case 'm': return std::chrono::seconds(static_cast<long long>(value * 60));
        case 'h': return std::chrono::seconds(static_cast<long long>(value * 3600));
        case 'd': return std::chrono::seconds(static_cast<long long>(value * 86400));
        default:  return std::chrono::seconds(static_cast<long long>(value));
    }
}

bool should_abort() {
    if (is_interrupted()) return true;
    if (g_timeout.enabled && g_timeout.is_expired()) {
        std::cout << "\n[!] Timeout expired after " << g_timeout.elapsed_string() << "\n";
        return true;
    }
    return false;
}

//==============================================================================
// Container Detection
//==============================================================================

ContainerInfo detect_container() {
    ContainerInfo info;
    
#ifdef _WIN32
    // Windows: Check for Hyper-V or Docker Desktop
    // Harder to detect, usually not containerized
    info.is_container = false;
#else
    // Check for Docker
    std::ifstream cgroup("/proc/1/cgroup");
    if (cgroup.is_open()) {
        std::string line;
        while (std::getline(cgroup, line)) {
            if (line.find("docker") != std::string::npos) {
                info.is_container = true;
                info.container_type = "docker";
                break;
            } else if (line.find("lxc") != std::string::npos) {
                info.is_container = true;
                info.container_type = "lxc";
                break;
            } else if (line.find("kubepods") != std::string::npos) {
                info.is_container = true;
                info.container_type = "kubernetes";
                break;
            }
        }
    }
    
    // Check for Singularity
    if (getenv("SINGULARITY_CONTAINER") != nullptr) {
        info.is_container = true;
        info.container_type = "singularity";
    }
    
    // Check /.dockerenv file
    std::ifstream dockerenv("/.dockerenv");
    if (dockerenv.good()) {
        info.is_container = true;
        info.container_type = "docker";
    }
    
    // Get container memory limit from cgroups
    if (info.is_container) {
        // Try cgroups v2 first
        std::ifstream memlimit("/sys/fs/cgroup/memory.max");
        if (memlimit.is_open()) {
            std::string val;
            memlimit >> val;
            if (val != "max") {
                info.memory_limit_mb = std::stoull(val) / (1024 * 1024);
            }
        } else {
            // Try cgroups v1
            std::ifstream memlimit_v1("/sys/fs/cgroup/memory/memory.limit_in_bytes");
            if (memlimit_v1.is_open()) {
                size_t limit;
                memlimit_v1 >> limit;
                // Check if it's effectively unlimited (very large number)
                if (limit < 1ULL << 60) {  // Less than exabyte
                    info.memory_limit_mb = limit / (1024 * 1024);
                }
            }
        }
    }
#endif
    
    return info;
}

//==============================================================================
// LRET Resource Monitoring
//==============================================================================

LRETResourceEstimate estimate_lret_resources(
    size_t num_qubits, 
    size_t depth, 
    double noise_prob,
    size_t initial_rank
) {
    LRETResourceEstimate est;
    
    size_t dim = 1ULL << num_qubits;
    
    // Base memory: L matrix is dim x rank
    est.base_memory_mb = (dim * initial_rank * 16) / (1024 * 1024);
    
    // Estimate peak rank based on noise and depth
    // Rough heuristic: rank can grow up to O(depth * noise_channels)
    // but truncation keeps it bounded
    size_t noise_ops_per_depth = static_cast<size_t>(num_qubits * noise_prob * 8);  // ~8 ops per layer
    size_t estimated_max_rank = (std::min)(
        initial_rank + depth * noise_ops_per_depth / 4,  // Growth estimate
        dim  // Can't exceed dimension
    );
    
    // Peak memory estimate
    est.estimated_peak_mb = (dim * estimated_max_rank * 16) / (1024 * 1024);
    
    // Also need temporary for Gram matrix: rank x rank
    size_t gram_temp = (estimated_max_rank * estimated_max_rank * 16) / (1024 * 1024);
    est.estimated_peak_mb += gram_temp;
    
    est.current_memory_mb = est.base_memory_mb;
    est.max_observed_rank = initial_rank;
    est.memory_warning = false;
    
    return est;
}

void update_lret_estimate(LRETResourceEstimate& estimate, size_t current_rank, size_t num_qubits) {
    size_t dim = 1ULL << num_qubits;
    
    estimate.max_observed_rank = (std::max)(estimate.max_observed_rank, current_rank);
    estimate.current_memory_mb = (dim * current_rank * 16) / (1024 * 1024);
    
    // Warn if rank is getting very high
    if (current_rank > dim / 4) {
        estimate.memory_warning = true;
        estimate.warning_message = "Rank (" + std::to_string(current_rank) + 
                                   ") exceeding 25% of dimension (" + std::to_string(dim) + ")";
    }
}

size_t get_current_memory_usage_mb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize / (1024 * 1024);
    }
    return 0;
#else
    std::ifstream status("/proc/self/status");
    if (status.is_open()) {
        std::string line;
        while (std::getline(status, line)) {
            if (line.find("VmRSS:") == 0) {
                std::istringstream iss(line.substr(6));
                size_t kb;
                iss >> kb;
                return kb / 1024;
            }
        }
    }
    return 0;
#endif
}

//==============================================================================
// Comprehensive Resource Check
//==============================================================================

ResourceCheckResult check_all_resources(
    size_t num_qubits,
    size_t depth,
    double noise_prob,
    bool fdm_enabled,
    bool allow_swap
) {
    ResourceCheckResult result;
    
    // 1. Check swap usage
    SwapStatus swap = get_swap_status();
    if (swap.is_using_swap && !allow_swap) {
        result.add_warning("System is using swap (" + std::to_string(swap.swap_used_mb) + 
                          " MB). This may cause severe slowdowns.");
    }
    
    // 2. Check container limits
    ContainerInfo container = detect_container();
    if (container.is_container) {
        std::cout << "[Info] Running in " << container.container_type << " container.\n";
        if (container.memory_limit_mb > 0) {
            std::cout << "[Info] Container memory limit: " << container.memory_limit_mb << " MB\n";
        }
    }
    
    // 3. Estimate LRET requirements
    LRETResourceEstimate lret_est = estimate_lret_resources(num_qubits, depth, noise_prob);
    if (lret_est.estimated_peak_mb > 1024) {  // > 1GB
        result.add_warning("LRET estimated peak memory: ~" + 
                          std::to_string(lret_est.estimated_peak_mb) + " MB");
    }
    
    // 4. Check FDM requirements if enabled
    if (fdm_enabled) {
        size_t fdm_req = (1ULL << (2 * num_qubits)) * 16 / (1024 * 1024);
        
        size_t available = container.memory_limit_mb > 0 ? 
                          container.memory_limit_mb : 
                          get_current_memory_usage_mb();
        
        if (fdm_req > 10 * 1024) {  // > 10 GB
            result.add_warning("FDM requires ~" + std::to_string(fdm_req / 1024) + " GB");
        }
    }
    
    return result;
}

}  // namespace qlret
