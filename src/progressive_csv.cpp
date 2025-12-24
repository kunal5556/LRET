#include "progressive_csv.h"
#include "resource_monitor.h"
#include <iomanip>
#include <ctime>
#include <sstream>
#include <cmath>
#include <filesystem>

namespace qlret {

// Global pointer (nullptr when not using CSV output)
ProgressiveCSVWriter* g_csv_writer = nullptr;

// Constructor that opens file immediately
ProgressiveCSVWriter::ProgressiveCSVWriter(const std::string& filename) {
    open(filename);
}

// Simple open without CLIOptions
bool ProgressiveCSVWriter::open(const std::string& filename) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    filename_ = filename;
    start_time_ = std::chrono::steady_clock::now();
    
    file_.open(filename, std::ios::out | std::ios::trunc);
    if (!file_.is_open()) {
        return false;
    }
    
    // Write header
    file_ << "timestamp,elapsed_s,event_type,mode,step,operation,"
          << "qubits,depth,rank_before,rank_after,time_s,memory_mb,"
          << "trace,status,message\n";
    file_.flush();
    
    return true;
}

bool ProgressiveCSVWriter::open(const std::string& filename, const CLIOptions& opts) {
    opts_ = opts;
    return open(filename);
}

std::string ProgressiveCSVWriter::get_filepath() const {
    try {
        return std::filesystem::absolute(filename_).string();
    } catch (...) {
        return filename_;  // Fallback to relative path
    }
}

void ProgressiveCSVWriter::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_.is_open()) {
        file_.flush();
        file_.close();
    }
}

std::string ProgressiveCSVWriter::get_timestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time);
    
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

double ProgressiveCSVWriter::elapsed_seconds() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - start_time_).count();
}

void ProgressiveCSVWriter::write_row(
    const std::string& event_type,
    const std::string& mode,
    int step,
    const std::string& operation,
    int rank_before,
    int rank_after,
    double time_seconds,
    size_t memory_mb,
    double trace_value,
    const std::string& status,
    const std::string& message
) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!file_.is_open()) return;
    
    file_ << get_timestamp() << ","
          << std::fixed << std::setprecision(3) << elapsed_seconds() << ","
          << event_type << ","
          << mode << ","
          << (step >= 0 ? std::to_string(step) : "") << ","
          << operation << ","
          << opts_.num_qubits << ","
          << opts_.depth << ","
          << (rank_before >= 0 ? std::to_string(rank_before) : "") << ","
          << (rank_after >= 0 ? std::to_string(rank_after) : "") << ","
          << std::setprecision(6) << time_seconds << ","
          << memory_mb << ","
          << (std::isnan(trace_value) ? "" : std::to_string(trace_value)) << ","
          << status << ","
          << "\"" << message << "\"\n";
    
    file_.flush();  // Ensure immediate write to disk
}

void ProgressiveCSVWriter::log_start(
    size_t num_qubits, 
    size_t depth, 
    const std::string& mode,
    double noise_prob
) {
    std::ostringstream msg;
    msg << "Starting simulation: " << num_qubits << " qubits, depth=" << depth 
        << ", noise=" << noise_prob << ", mode=" << mode;
    
    write_row("START", mode, -1, "init", -1, -1, 0.0, 
              get_current_memory_usage_mb(), std::nan(""), "OK", msg.str());
}

void ProgressiveCSVWriter::log_step_start(size_t step_num, const std::string& mode) {
    write_row("STEP_START", mode, static_cast<int>(step_num), "", -1, -1, 
              0.0, get_current_memory_usage_mb(), std::nan(""), "OK", 
              "Step " + std::to_string(step_num) + " started");
}

void ProgressiveCSVWriter::log_operation(
    const std::string& mode,
    size_t step_num,
    const std::string& operation,
    size_t rank_before,
    size_t rank_after,
    double time_seconds,
    size_t memory_mb
) {
    size_t mem = memory_mb > 0 ? memory_mb : get_current_memory_usage_mb();
    write_row("OPERATION", mode, static_cast<int>(step_num), operation,
              static_cast<int>(rank_before), static_cast<int>(rank_after),
              time_seconds, mem, std::nan(""), "OK", "");
}

void ProgressiveCSVWriter::log_operation(
    size_t step_num,
    const std::string& operation,
    size_t rank_before,
    size_t rank_after,
    double time_seconds
) {
    log_operation("", step_num, operation, rank_before, rank_after, time_seconds, 0);
}

void ProgressiveCSVWriter::log_mode_complete(
    const std::string& mode,
    double total_time_seconds,
    size_t final_rank,
    double trace_value,
    double fidelity
) {
    std::ostringstream msg;
    msg << "Mode complete. Final rank=" << final_rank;
    if (fidelity >= 0) {
        msg << ", fidelity=" << std::fixed << std::setprecision(6) << fidelity;
    }
    
    write_row("MODE_COMPLETE", mode, -1, "", -1, static_cast<int>(final_rank),
              total_time_seconds, get_current_memory_usage_mb(), trace_value,
              "OK", msg.str());
}

void ProgressiveCSVWriter::log_fdm_start(size_t num_qubits) {
    size_t estimated_mb = (1ULL << (2 * num_qubits)) * 16 / (1024 * 1024);
    std::ostringstream msg;
    msg << "FDM starting. Estimated memory: " << estimated_mb << " MB";
    
    write_row("FDM_START", "fdm", -1, "init", -1, -1, 0.0,
              get_current_memory_usage_mb(), std::nan(""), "OK", msg.str());
}

void ProgressiveCSVWriter::log_fdm_operation(
    size_t step_num,
    const std::string& operation,
    double time_seconds,
    size_t memory_mb
) {
    size_t mem = memory_mb > 0 ? memory_mb : get_current_memory_usage_mb();
    write_row("FDM_OP", "fdm", static_cast<int>(step_num), operation,
              -1, -1, time_seconds, mem, std::nan(""), "OK", "");
}

void ProgressiveCSVWriter::log_fdm_complete(
    double total_time, 
    double trace_value, 
    bool success,
    const std::string& message
) {
    write_row("FDM_COMPLETE", "fdm", -1, "", -1, -1, total_time,
              get_current_memory_usage_mb(), trace_value,
              success ? "OK" : "FAILED", message);
}

void ProgressiveCSVWriter::log_error(const std::string& message) {
    write_row("ERROR", "", -1, "", -1, -1, 0.0,
              get_current_memory_usage_mb(), std::nan(""), "ERROR", message);
}

void ProgressiveCSVWriter::log_error(const std::string& mode, const std::string& message) {
    write_row("ERROR", mode, -1, "", -1, -1, 0.0,
              get_current_memory_usage_mb(), std::nan(""), "ERROR", message);
}

void ProgressiveCSVWriter::log_warning(const std::string& mode, const std::string& message) {
    write_row("WARNING", mode, -1, "", -1, -1, 0.0,
              get_current_memory_usage_mb(), std::nan(""), "WARNING", message);
}

void ProgressiveCSVWriter::log_summary(
    const std::string& best_mode,
    double best_time,
    double fdm_time,
    double fidelity_vs_fdm
) {
    std::ostringstream msg;
    msg << "Best: " << best_mode << " in " << std::fixed << std::setprecision(3) 
        << best_time << "s";
    if (fdm_time > 0) {
        msg << ". FDM: " << fdm_time << "s. Speedup: " 
            << std::setprecision(2) << (fdm_time / best_time) << "x";
    }
    if (fidelity_vs_fdm >= 0) {
        msg << ". Fidelity: " << std::setprecision(6) << fidelity_vs_fdm;
    }
    
    write_row("SUMMARY", best_mode, -1, "", -1, -1, best_time,
              get_current_memory_usage_mb(), std::nan(""), "OK", msg.str());
}

void ProgressiveCSVWriter::log_summary(
    size_t final_rank, 
    double trace_value, 
    double time_seconds, 
    const std::string& status
) {
    std::ostringstream msg;
    msg << "Final rank=" << final_rank << ", trace=" << std::fixed 
        << std::setprecision(6) << trace_value << ", time=" 
        << std::setprecision(3) << time_seconds << "s";
    
    write_row("SUMMARY", "", -1, "", -1, static_cast<int>(final_rank), time_seconds,
              get_current_memory_usage_mb(), trace_value, status, msg.str());
}

void ProgressiveCSVWriter::log_interrupt(const std::string& reason) {
    write_row("INTERRUPT", "", -1, "", -1, -1, elapsed_seconds(),
              get_current_memory_usage_mb(), std::nan(""), "INTERRUPTED", reason);
}

}  // namespace qlret
