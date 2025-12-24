#pragma once

#include "types.h"
#include "cli_parser.h"
#include <string>
#include <fstream>
#include <chrono>
#include <mutex>

namespace qlret {

/**
 * Progressive CSV Writer
 * 
 * Appends results to CSV file immediately after each operation completes.
 * This ensures data is saved even if the program crashes or is interrupted.
 * 
 * CSV Format:
 * timestamp,event_type,mode,step,operation,qubits,depth,rank_before,rank_after,
 * time_seconds,memory_mb,trace,status,message
 */
class ProgressiveCSVWriter {
public:
    ProgressiveCSVWriter() = default;
    
    // Open file and write header
    bool open(const std::string& filename, const CLIOptions& opts);
    
    // Close file
    void close();
    
    // Check if file is open
    bool is_open() const { return file_.is_open(); }
    
    // Get filename
    const std::string& filename() const { return filename_; }
    
    //==========================================================================
    // Event Logging Methods
    //==========================================================================
    
    // Log simulation start
    void log_start(size_t num_qubits, size_t depth, double noise_prob, 
                   const std::string& mode);
    
    // Log a step beginning (e.g., "=== Step 5 ===")
    void log_step_start(size_t step_num, const std::string& mode);
    
    // Log an operation within a step
    void log_operation(
        const std::string& mode,
        size_t step_num,
        const std::string& operation,  // "gate", "kraus", "truncation"
        size_t rank_before,
        size_t rank_after,
        double time_seconds,
        size_t memory_mb = 0
    );
    
    // Log mode completion
    void log_mode_complete(
        const std::string& mode,
        double total_time_seconds,
        size_t final_rank,
        double trace_value,
        double fidelity = -1.0
    );
    
    // Log FDM start
    void log_fdm_start(size_t num_qubits);
    
    // Log FDM operation
    void log_fdm_operation(
        size_t step_num,
        const std::string& operation,
        double time_seconds,
        size_t memory_mb = 0
    );
    
    // Log FDM completion or failure
    void log_fdm_complete(double total_time, double trace_value, bool success,
                          const std::string& message = "");
    
    // Log error or warning
    void log_error(const std::string& mode, const std::string& message);
    void log_warning(const std::string& mode, const std::string& message);
    
    // Log final summary
    void log_summary(
        const std::string& best_mode,
        double best_time,
        double fdm_time = -1.0,
        double fidelity_vs_fdm = -1.0
    );
    
    // Log interrupt/timeout
    void log_interrupt(const std::string& reason);
    
private:
    std::ofstream file_;
    std::string filename_;
    std::mutex mutex_;  // Thread safety
    std::chrono::steady_clock::time_point start_time_;
    CLIOptions opts_;
    
    // Write a row to CSV
    void write_row(
        const std::string& event_type,
        const std::string& mode,
        int step,  // -1 for N/A
        const std::string& operation,
        int rank_before,  // -1 for N/A
        int rank_after,   // -1 for N/A
        double time_seconds,
        size_t memory_mb,
        double trace_value,  // NaN for N/A
        const std::string& status,
        const std::string& message
    );
    
    // Get current timestamp string
    std::string get_timestamp() const;
    
    // Get elapsed time since start
    double elapsed_seconds() const;
};

// Global progressive writer (optional, for convenience)
extern ProgressiveCSVWriter g_csv_writer;

}  // namespace qlret
