#include "structured_csv.h"
#include "resource_monitor.h"
#include <iomanip>
#include <sstream>
#include <ctime>
#include <cmath>
#include <filesystem>
#include <algorithm>

namespace qlret {

// Global pointer
StructuredCSVWriter* g_structured_csv = nullptr;

//==============================================================================
// Default Filename Generator
//==============================================================================

std::string generate_default_csv_filename(const CLIOptions& opts) {
    std::ostringstream oss;
    
    // Get current timestamp
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time);
    
    // Build filename: lret_n{qubits}_d{depth}_{mode}_{extras}_{timestamp}.csv
    oss << "lret_n" << opts.num_qubits 
        << "_d" << opts.depth
        << "_" << parallel_mode_to_string(opts.parallel_mode);
    
    // Add FDM indicator
    if (opts.enable_fdm) {
        oss << "_fdm";
    }
    
    // Add noise type if not default
    if (opts.noise_selection != NoiseSelection::ALL) {
        oss << "_" << noise_selection_to_string(opts.noise_selection);
    }
    
    // Add timestamp: YYYYMMDD_HHMMSS
    oss << "_" << std::put_time(&tm, "%Y%m%d_%H%M%S");
    
    oss << ".csv";
    
    return oss.str();
}

//==============================================================================
// Constructor / Destructor
//==============================================================================

StructuredCSVWriter::StructuredCSVWriter(const std::string& filename) {
    open(filename);
}

StructuredCSVWriter::~StructuredCSVWriter() {
    close();
}

bool StructuredCSVWriter::open(const std::string& filename) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    filename_ = filename;
    start_time_ = std::chrono::steady_clock::now();
    
    file_.open(filename, std::ios::out | std::ios::trunc);
    if (!file_.is_open()) {
        return false;
    }
    
    // Write file format comment (for parsers)
    file_ << "# QuantumLRET-Sim Structured CSV Output\n";
    file_ << "# Format Version: 2.0\n";
    file_ << "# Generated: " << get_timestamp() << "\n";
    file_ << "# Sections can be parsed independently for Excel conversion\n";
    file_ << "#\n";
    file_.flush();
    
    return true;
}

void StructuredCSVWriter::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!current_section_.empty()) {
        file_ << "END_SECTION," << current_section_ << "\n";
        current_section_.clear();
    }
    
    if (file_.is_open()) {
        file_.flush();
        file_.close();
    }
}

std::string StructuredCSVWriter::get_filepath() const {
    try {
        return std::filesystem::absolute(filename_).string();
    } catch (...) {
        return filename_;
    }
}

//==============================================================================
// Helper Methods
//==============================================================================

void StructuredCSVWriter::begin_section(const std::string& section_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!current_section_.empty()) {
        file_ << "END_SECTION," << current_section_ << "\n\n";
    }
    
    current_section_ = section_name;
    file_ << "SECTION," << section_name << "\n";
    file_.flush();
}

void StructuredCSVWriter::end_section() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!current_section_.empty()) {
        file_ << "END_SECTION," << current_section_ << "\n\n";
        file_.flush();
        current_section_.clear();
    }
}

void StructuredCSVWriter::write_line(const std::string& line) {
    std::lock_guard<std::mutex> lock(mutex_);
    file_ << line << "\n";
    file_.flush();
}

void StructuredCSVWriter::write_csv_row(const std::vector<std::string>& fields) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (size_t i = 0; i < fields.size(); ++i) {
        if (i > 0) file_ << ",";
        file_ << escape_csv(fields[i]);
    }
    file_ << "\n";
    file_.flush();
}

std::string StructuredCSVWriter::get_timestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time);
    
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

double StructuredCSVWriter::elapsed_seconds() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - start_time_).count();
}

std::string StructuredCSVWriter::escape_csv(const std::string& s) const {
    if (s.find(',') != std::string::npos || 
        s.find('"') != std::string::npos || 
        s.find('\n') != std::string::npos) {
        std::string escaped = s;
        size_t pos = 0;
        while ((pos = escaped.find('"', pos)) != std::string::npos) {
            escaped.replace(pos, 1, "\"\"");
            pos += 2;
        }
        return "\"" + escaped + "\"";
    }
    return s;
}

std::string StructuredCSVWriter::format_double(double val, int precision) const {
    if (std::isnan(val) || std::isinf(val)) return "";
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << val;
    return oss.str();
}

//==============================================================================
// Section 1: Header
//==============================================================================

void StructuredCSVWriter::write_header(const CLIOptions& opts, const NoiseStats& noise_stats) {
    begin_section("HEADER");
    
    // Configuration parameters as key-value pairs
    write_csv_row({"parameter", "value"});
    write_csv_row({"timestamp", get_timestamp()});
    write_csv_row({"num_qubits", std::to_string(opts.num_qubits)});
    write_csv_row({"depth", std::to_string(opts.depth)});
    write_csv_row({"batch_size", std::to_string(opts.batch_size)});
    write_csv_row({"truncation_threshold", format_double(opts.truncation_threshold)});
    write_csv_row({"noise_probability", format_double(opts.noise_prob)});
    write_csv_row({"noise_type", noise_selection_to_string(opts.noise_selection)});
    write_csv_row({"parallel_mode", parallel_mode_to_string(opts.parallel_mode)});
    write_csv_row({"initial_rank", std::to_string(opts.initial_rank)});
    write_csv_row({"fdm_enabled", opts.enable_fdm ? "true" : "false"});
    write_csv_row({"num_threads", std::to_string(opts.num_threads)});
    
    // Noise statistics
    write_csv_row({"total_noise_events", std::to_string(noise_stats.total_count())});
    write_csv_row({"depolarizing_count", std::to_string(noise_stats.depolarizing_count)});
    write_csv_row({"amplitude_damping_count", std::to_string(noise_stats.amplitude_damping_count)});
    write_csv_row({"phase_damping_count", std::to_string(noise_stats.phase_damping_count)});
    
    end_section();
}

//==============================================================================
// Section 2: FDM Progress
//==============================================================================

void StructuredCSVWriter::begin_fdm_progress(size_t num_qubits, size_t depth) {
    begin_section("FDM_PROGRESS");
    
    // Write header row
    write_csv_row({"timestamp", "elapsed_s", "step", "operation", "detail", 
                   "time_s", "memory_mb", "cumulative_time_s"});
    
    // Log start
    write_csv_row({get_timestamp(), format_double(elapsed_seconds(), 3), 
                   "0", "START", "FDM simulation starting", 
                   "0", std::to_string(get_current_memory_usage_mb()), "0"});
}

void StructuredCSVWriter::log_fdm_step(size_t step, const std::string& operation,
                                        double time_seconds, size_t memory_mb) {
    size_t mem = memory_mb > 0 ? memory_mb : get_current_memory_usage_mb();
    write_csv_row({get_timestamp(), format_double(elapsed_seconds(), 3),
                   std::to_string(step), operation, "",
                   format_double(time_seconds, 6), std::to_string(mem),
                   format_double(elapsed_seconds(), 3)});
}

void StructuredCSVWriter::log_fdm_gate(size_t step, size_t gate_idx, 
                                        const std::string& gate_name,
                                        const std::vector<size_t>& qubits, 
                                        double time_seconds) {
    std::ostringstream detail;
    detail << gate_name << " on qubit(s) ";
    for (size_t i = 0; i < qubits.size(); ++i) {
        if (i > 0) detail << ",";
        detail << qubits[i];
    }
    
    write_csv_row({get_timestamp(), format_double(elapsed_seconds(), 3),
                   std::to_string(step), "GATE", detail.str(),
                   format_double(time_seconds, 6), 
                   std::to_string(get_current_memory_usage_mb()),
                   format_double(elapsed_seconds(), 3)});
}

void StructuredCSVWriter::log_fdm_noise(size_t step, const std::string& noise_type,
                                         size_t qubit, double time_seconds) {
    std::ostringstream detail;
    detail << noise_type << " on qubit " << qubit;
    
    write_csv_row({get_timestamp(), format_double(elapsed_seconds(), 3),
                   std::to_string(step), "NOISE", detail.str(),
                   format_double(time_seconds, 6),
                   std::to_string(get_current_memory_usage_mb()),
                   format_double(elapsed_seconds(), 3)});
}

// Simplified FDM gate logging (step and time only)
void StructuredCSVWriter::log_fdm_gate(size_t step, double time_seconds) {
    write_csv_row({get_timestamp(), format_double(elapsed_seconds(), 3),
                   std::to_string(step), "GATE", "",
                   format_double(time_seconds, 6),
                   std::to_string(get_current_memory_usage_mb()),
                   format_double(elapsed_seconds(), 3)});
}

// Simplified FDM noise logging (step and time only)
void StructuredCSVWriter::log_fdm_noise(size_t step, double time_seconds) {
    write_csv_row({get_timestamp(), format_double(elapsed_seconds(), 3),
                   std::to_string(step), "NOISE", "",
                   format_double(time_seconds, 6),
                   std::to_string(get_current_memory_usage_mb()),
                   format_double(elapsed_seconds(), 3)});
}

void StructuredCSVWriter::end_fdm_progress(double total_time, bool success,
                                            const std::string& message) {
    write_csv_row({get_timestamp(), format_double(elapsed_seconds(), 3),
                   "", "END", success ? "FDM completed successfully" : message,
                   format_double(total_time, 6),
                   std::to_string(get_current_memory_usage_mb()),
                   format_double(elapsed_seconds(), 3)});
    end_section();
}

//==============================================================================
// Section 3: FDM Metrics
//==============================================================================

void StructuredCSVWriter::write_fdm_metrics(const FDMResult& fdm_result, 
                                             size_t num_qubits,
                                             const NoiseStats& noise_stats) {
    begin_section("FDM_METRICS");
    
    write_csv_row({"metric", "value", "unit", "description"});
    
    if (fdm_result.was_run) {
        write_csv_row({"status", "SUCCESS", "", "FDM simulation completed"});
        write_csv_row({"time_seconds", format_double(fdm_result.time_seconds, 6), "s", "Total FDM execution time"});
        write_csv_row({"trace", format_double(fdm_result.trace_value, 10), "", "Tr(rho)"});
        write_csv_row({"dimension", std::to_string(1ULL << num_qubits), "", "Hilbert space dimension 2^n"});
        write_csv_row({"matrix_size_bytes", std::to_string((1ULL << (2*num_qubits)) * 16), "bytes", "Full density matrix size"});
        write_csv_row({"matrix_size_gb", format_double((1ULL << (2*num_qubits)) * 16.0 / (1024*1024*1024), 3), "GB", "Full density matrix size"});
        
        // Noise breakdown
        write_csv_row({"total_noise_ops", std::to_string(noise_stats.total_count()), "", "Total noise operations applied"});
        write_csv_row({"depolarizing_ops", std::to_string(noise_stats.depolarizing_count), "", "Depolarizing noise count"});
        write_csv_row({"amplitude_damping_ops", std::to_string(noise_stats.amplitude_damping_count), "", "Amplitude damping count"});
        write_csv_row({"phase_damping_ops", std::to_string(noise_stats.phase_damping_count), "", "Phase damping count"});
    } else {
        write_csv_row({"status", "SKIPPED", "", fdm_result.skip_reason});
    }
    
    end_section();
}

//==============================================================================
// Section 4: LRET Progress
//==============================================================================

void StructuredCSVWriter::begin_lret_progress(const std::string& mode, 
                                               size_t num_qubits, size_t depth) {
    begin_section("LRET_PROGRESS_" + mode);
    
    write_csv_row({"timestamp", "elapsed_s", "step", "operation", "noise_type",
                   "rank_before", "rank_after", "time_s", "memory_mb"});
    
    write_csv_row({get_timestamp(), format_double(elapsed_seconds(), 3),
                   "0", "START", "", "1", "1", "0", 
                   std::to_string(get_current_memory_usage_mb())});
}

// Simplified begin_lret_progress with ParallelMode enum
void StructuredCSVWriter::begin_lret_progress(size_t num_qubits, size_t depth, ParallelMode mode) {
    begin_lret_progress(parallel_mode_to_string(mode), num_qubits, depth);
}

void StructuredCSVWriter::log_lret_step_start(const std::string& mode, size_t step) {
    write_csv_row({get_timestamp(), format_double(elapsed_seconds(), 3),
                   std::to_string(step), "STEP_START", "", "", "", "0",
                   std::to_string(get_current_memory_usage_mb())});
}

void StructuredCSVWriter::log_lret_gate(const std::string& mode, size_t step,
                                         size_t rank_before, size_t rank_after,
                                         double time_seconds) {
    write_csv_row({get_timestamp(), format_double(elapsed_seconds(), 3),
                   std::to_string(step), "GATE", "",
                   std::to_string(rank_before), std::to_string(rank_after),
                   format_double(time_seconds, 6),
                   std::to_string(get_current_memory_usage_mb())});
}

// Simplified log_lret_gate (no mode parameter)
void StructuredCSVWriter::log_lret_gate(size_t step, double time_seconds,
                                         size_t rank_before, size_t rank_after) {
    write_csv_row({get_timestamp(), format_double(elapsed_seconds(), 3),
                   std::to_string(step), "GATE", "",
                   std::to_string(rank_before), std::to_string(rank_after),
                   format_double(time_seconds, 6),
                   std::to_string(get_current_memory_usage_mb())});
}

void StructuredCSVWriter::log_lret_kraus(const std::string& mode, size_t step,
                                          const std::string& noise_type,
                                          size_t rank_before, size_t rank_after,
                                          double time_seconds) {
    write_csv_row({get_timestamp(), format_double(elapsed_seconds(), 3),
                   std::to_string(step), "KRAUS", noise_type,
                   std::to_string(rank_before), std::to_string(rank_after),
                   format_double(time_seconds, 6),
                   std::to_string(get_current_memory_usage_mb())});
}

// Simplified log_lret_kraus (no mode or noise_type parameter)
void StructuredCSVWriter::log_lret_kraus(size_t step, double time_seconds,
                                          size_t rank_before, size_t rank_after) {
    write_csv_row({get_timestamp(), format_double(elapsed_seconds(), 3),
                   std::to_string(step), "KRAUS", "",
                   std::to_string(rank_before), std::to_string(rank_after),
                   format_double(time_seconds, 6),
                   std::to_string(get_current_memory_usage_mb())});
}

void StructuredCSVWriter::log_lret_truncation(const std::string& mode, size_t step,
                                               size_t rank_before, size_t rank_after,
                                               double time_seconds) {
    write_csv_row({get_timestamp(), format_double(elapsed_seconds(), 3),
                   std::to_string(step), "TRUNCATION", "",
                   std::to_string(rank_before), std::to_string(rank_after),
                   format_double(time_seconds, 6),
                   std::to_string(get_current_memory_usage_mb())});
}

// Simplified log_lret_truncation (no mode parameter)
void StructuredCSVWriter::log_lret_truncation(size_t step, double time_seconds,
                                               size_t rank_before, size_t rank_after) {
    write_csv_row({get_timestamp(), format_double(elapsed_seconds(), 3),
                   std::to_string(step), "TRUNCATION", "",
                   std::to_string(rank_before), std::to_string(rank_after),
                   format_double(time_seconds, 6),
                   std::to_string(get_current_memory_usage_mb())});
}

void StructuredCSVWriter::end_lret_progress(const std::string& mode, double total_time,
                                             size_t final_rank, bool success) {
    write_csv_row({get_timestamp(), format_double(elapsed_seconds(), 3),
                   "", "END", success ? "completed" : "failed",
                   "", std::to_string(final_rank),
                   format_double(total_time, 6),
                   std::to_string(get_current_memory_usage_mb())});
    end_section();
}

// Simplified end_lret_progress with ParallelMode enum
void StructuredCSVWriter::end_lret_progress(double total_time, bool success, ParallelMode mode) {
    // Just end the section - the mode was already used to name it
    write_csv_row({get_timestamp(), format_double(elapsed_seconds(), 3),
                   "", "END", success ? "completed" : "failed",
                   "", "",
                   format_double(total_time, 6),
                   std::to_string(get_current_memory_usage_mb())});
    end_section();
}

//==============================================================================
// Section 5: LRET Mode Metrics
//==============================================================================

void StructuredCSVWriter::write_lret_mode_metrics(const std::string& mode,
                                                   const ModeResult& result,
                                                   const MetricsResult& vs_initial,
                                                   const StateMetrics& state_metrics,
                                                   const NoiseStats& noise_stats) {
    begin_section("LRET_METRICS_" + mode);
    
    write_csv_row({"metric", "value", "unit", "description"});
    
    // Execution metrics
    write_csv_row({"mode", mode, "", "Parallelization mode"});
    write_csv_row({"time_seconds", format_double(result.time_seconds, 6), "s", "Execution time"});
    write_csv_row({"final_rank", std::to_string(result.final_rank), "", "Final L-matrix rank"});
    write_csv_row({"trace", format_double(result.trace_value, 10), "", "Tr(rho)"});
    write_csv_row({"speedup", format_double(result.speedup, 4), "x", "vs sequential baseline"});
    
    // Distance metrics vs initial state
    write_csv_row({"fidelity_vs_initial", format_double(vs_initial.fidelity, 10), "", "F(rho_final, rho_initial)"});
    write_csv_row({"trace_distance_vs_initial", format_double(vs_initial.trace_distance, 10), "", "T(rho_final, rho_initial)"});
    write_csv_row({"frobenius_distance_vs_initial", format_double(vs_initial.frobenius_distance, 10), "", "||rho_final - rho_initial||_F"});
    write_csv_row({"variational_distance_vs_initial", format_double(vs_initial.variational_distance, 10), "", "D_var(rho_final, rho_initial)"});
    
    // State metrics
    write_csv_row({"purity", format_double(state_metrics.purity, 10), "", "Tr(rho^2)"});
    write_csv_row({"von_neumann_entropy", format_double(state_metrics.entropy, 10), "bits", "S = -Tr(rho log2 rho)"});
    write_csv_row({"linear_entropy", format_double(state_metrics.linear_entropy, 10), "", "S_L = 1 - Tr(rho^2)"});
    
    if (state_metrics.concurrence >= 0) {
        write_csv_row({"concurrence", format_double(state_metrics.concurrence, 10), "", "Entanglement (2-qubit only)"});
    }
    if (state_metrics.negativity >= 0) {
        write_csv_row({"negativity", format_double(state_metrics.negativity, 10), "", "Bipartite negativity"});
    }
    
    write_csv_row({"rank", std::to_string(state_metrics.rank), "", "Effective rank of state"});
    
    // Noise statistics
    write_csv_row({"total_noise_ops", std::to_string(noise_stats.total_count()), "", "Total noise operations"});
    
    // Distortion if computed
    if (result.distortion > 0) {
        write_csv_row({"distortion_vs_sequential", format_double(result.distortion, 10), "", "||L_mode - L_seq||/||L_seq||"});
    }
    
    end_section();
}

// Simplified write_lret_mode_metrics (result and mode only)
void StructuredCSVWriter::write_lret_mode_metrics(const ModeResult& result, ParallelMode mode) {
    std::string mode_str = parallel_mode_to_string(mode);
    begin_section("LRET_METRICS_" + mode_str);
    
    write_csv_row({"metric", "value", "unit", "description"});
    
    // Execution metrics
    write_csv_row({"mode", mode_str, "", "Parallelization mode"});
    write_csv_row({"time_seconds", format_double(result.time_seconds, 6), "s", "Execution time"});
    write_csv_row({"final_rank", std::to_string(result.final_rank), "", "Final L-matrix rank"});
    write_csv_row({"trace", format_double(result.trace_value, 10), "", "Tr(rho)"});
    write_csv_row({"speedup", format_double(result.speedup, 4), "x", "vs sequential baseline"});
    write_csv_row({"fidelity", format_double(result.fidelity, 10), "", "Fidelity vs baseline"});
    write_csv_row({"trace_distance", format_double(result.trace_distance, 10), "", "Trace distance vs baseline"});
    write_csv_row({"frobenius_distance", format_double(result.frobenius_distance, 10), "", "Frobenius distance vs baseline"});
    write_csv_row({"distortion", format_double(result.distortion, 10), "", "Relative distortion"});
    
    end_section();
}

// Full write_lret_mode_metrics with all state metrics
void StructuredCSVWriter::write_lret_mode_metrics_full(const ModeResult& result, size_t num_qubits,
                                                        double purity, double entropy, double linear_entropy,
                                                        double concurrence, double negativity) {
    std::string mode_str = parallel_mode_to_string(result.mode);
    begin_section("LRET_METRICS_" + mode_str);
    
    write_csv_row({"metric", "value", "unit", "description"});
    
    // Execution metrics
    write_csv_row({"mode", mode_str, "", "Parallelization mode"});
    write_csv_row({"num_qubits", std::to_string(num_qubits), "", "Number of qubits"});
    write_csv_row({"time_seconds", format_double(result.time_seconds, 6), "s", "Execution time"});
    write_csv_row({"final_rank", std::to_string(result.final_rank), "", "Final L-matrix rank"});
    write_csv_row({"trace", format_double(result.trace_value, 10), "", "Tr(rho) - should be 1"});
    write_csv_row({"speedup", format_double(result.speedup, 4), "x", "vs sequential baseline"});
    
    // Distance metrics vs sequential baseline
    write_csv_row({"fidelity_vs_seq", format_double(result.fidelity, 10), "", "Fidelity vs sequential"});
    write_csv_row({"trace_distance_vs_seq", format_double(result.trace_distance, 10), "", "Trace distance vs sequential"});
    write_csv_row({"frobenius_distance_vs_seq", format_double(result.frobenius_distance, 10), "", "||L - L_seq||_F"});
    write_csv_row({"distortion_vs_seq", format_double(result.distortion, 10), "", "Relative distortion vs sequential"});
    
    // State metrics (from quantum information theory)
    write_csv_row({"purity", format_double(purity, 10), "", "Tr(rho^2) - 1 for pure, 1/d for maximally mixed"});
    write_csv_row({"von_neumann_entropy", format_double(entropy, 10), "bits", "S = -Tr(rho log2 rho)"});
    write_csv_row({"linear_entropy", format_double(linear_entropy, 10), "", "S_L = 1 - Tr(rho^2)"});
    
    // Entanglement metrics (conditional)
    if (concurrence >= 0) {
        write_csv_row({"concurrence", format_double(concurrence, 10), "", "Entanglement measure (2-qubit only)"});
    }
    if (negativity >= 0) {
        write_csv_row({"negativity", format_double(negativity, 10), "", "Bipartite negativity (half-half split)"});
    }
    
    end_section();
}

//==============================================================================
// Section 6: Mode Comparison
//==============================================================================

void StructuredCSVWriter::write_mode_comparison(const std::vector<ModeResult>& results,
                                                 const std::string& baseline_mode) {
    begin_section("MODE_COMPARISON");
    
    // Header row
    write_csv_row({"mode", "time_s", "speedup", "final_rank", "trace", 
                   "fidelity_vs_baseline", "trace_distance", "frobenius_distance", "distortion"});
    
    // Find baseline result
    const ModeResult* baseline = nullptr;
    for (const auto& r : results) {
        if (parallel_mode_to_string(r.mode) == baseline_mode) {
            baseline = &r;
            break;
        }
    }
    
    // Write each mode
    for (const auto& r : results) {
        double speedup = baseline ? baseline->time_seconds / r.time_seconds : 1.0;
        
        write_csv_row({
            parallel_mode_to_string(r.mode),
            format_double(r.time_seconds, 6),
            format_double(speedup, 4),
            std::to_string(r.final_rank),
            format_double(r.trace_value, 10),
            format_double(r.fidelity, 10),
            format_double(r.trace_distance, 10),
            format_double(r.frobenius_distance, 10),
            format_double(r.distortion, 10)
        });
    }
    
    end_section();
}

//==============================================================================
// Section 7: FDM Comparison
//==============================================================================

void StructuredCSVWriter::write_fdm_comparison(const std::vector<ModeResult>& results,
                                                const FDMResult& fdm_result,
                                                const std::map<std::string, MetricsResult>& fdm_metrics) {
    begin_section("FDM_COMPARISON");
    
    write_csv_row({"mode", "lret_time_s", "fdm_time_s", "speedup_vs_fdm",
                   "fidelity_vs_fdm", "trace_distance_vs_fdm", 
                   "frobenius_distance_vs_fdm", "variational_distance_vs_fdm"});
    
    for (const auto& r : results) {
        std::string mode_name = parallel_mode_to_string(r.mode);
        double speedup = fdm_result.time_seconds / r.time_seconds;
        
        auto it = fdm_metrics.find(mode_name);
        if (it != fdm_metrics.end()) {
            const auto& m = it->second;
            write_csv_row({
                mode_name,
                format_double(r.time_seconds, 6),
                format_double(fdm_result.time_seconds, 6),
                format_double(speedup, 4),
                format_double(m.fidelity, 10),
                format_double(m.trace_distance, 10),
                format_double(m.frobenius_distance, 10),
                format_double(m.variational_distance, 10)
            });
        } else {
            write_csv_row({
                mode_name,
                format_double(r.time_seconds, 6),
                format_double(fdm_result.time_seconds, 6),
                format_double(speedup, 4),
                "", "", "", ""
            });
        }
    }
    
    end_section();
}

// Simplified FDM comparison (no metrics map, just timing comparison)
void StructuredCSVWriter::write_fdm_comparison(const std::vector<ModeResult>& results,
                                                const FDMResult& fdm_result) {
    begin_section("FDM_COMPARISON");
    
    write_csv_row({"mode", "lret_time_s", "fdm_time_s", "speedup_vs_fdm"});
    
    for (const auto& r : results) {
        std::string mode_name = parallel_mode_to_string(r.mode);
        double speedup = fdm_result.time_seconds / r.time_seconds;
        
        write_csv_row({
            mode_name,
            format_double(r.time_seconds, 6),
            format_double(fdm_result.time_seconds, 6),
            format_double(speedup, 4)
        });
    }
    
    end_section();
}

//==============================================================================
// Section 8: Summary
//==============================================================================

void StructuredCSVWriter::write_summary(const CLIOptions& opts,
                                         const std::vector<ModeResult>& results,
                                         const std::optional<FDMResult>& fdm_result,
                                         double total_wall_time) {
    begin_section("SUMMARY");
    
    write_csv_row({"metric", "value"});
    write_csv_row({"total_wall_time_s", format_double(total_wall_time, 3)});
    write_csv_row({"num_qubits", std::to_string(opts.num_qubits)});
    write_csv_row({"depth", std::to_string(opts.depth)});
    
    if (!results.empty()) {
        // Find fastest mode
        auto fastest = std::min_element(results.begin(), results.end(),
            [](const auto& a, const auto& b) { return a.time_seconds < b.time_seconds; });
        
        write_csv_row({"fastest_mode", parallel_mode_to_string(fastest->mode)});
        write_csv_row({"fastest_time_s", format_double(fastest->time_seconds, 6)});
        write_csv_row({"modes_compared", std::to_string(results.size())});
    }
    
    if (fdm_result && fdm_result->was_run) {
        write_csv_row({"fdm_time_s", format_double(fdm_result->time_seconds, 6)});
        write_csv_row({"fdm_status", "SUCCESS"});
    } else if (fdm_result) {
        write_csv_row({"fdm_status", "SKIPPED"});
        write_csv_row({"fdm_skip_reason", fdm_result->skip_reason});
    }
    
    write_csv_row({"completion_timestamp", get_timestamp()});
    
    end_section();
}

// Simplified write_summary (timing only)
void StructuredCSVWriter::write_summary(double total_wall_time, double lret_time, double fdm_time,
                                         bool success, const std::string& message) {
    begin_section("SUMMARY");
    
    write_csv_row({"metric", "value"});
    write_csv_row({"total_wall_time_s", format_double(total_wall_time, 3)});
    write_csv_row({"lret_time_s", format_double(lret_time, 6)});
    write_csv_row({"fdm_time_s", format_double(fdm_time, 6)});
    write_csv_row({"status", success ? "SUCCESS" : "FAILED"});
    if (!message.empty()) {
        write_csv_row({"message", message});
    }
    write_csv_row({"completion_timestamp", get_timestamp()});
    
    end_section();
}

//==============================================================================
// Error/Warning/Interrupt
//==============================================================================

void StructuredCSVWriter::log_error(const std::string& context, const std::string& message) {
    write_csv_row({"ERROR", context, message, get_timestamp()});
}

// Simplified log_error (message only)
void StructuredCSVWriter::log_error(const std::string& message) {
    write_csv_row({"ERROR", message, get_timestamp()});
}

void StructuredCSVWriter::log_warning(const std::string& context, const std::string& message) {
    write_csv_row({"WARNING", context, message, get_timestamp()});
}

void StructuredCSVWriter::log_interrupt(const std::string& reason) {
    write_csv_row({"INTERRUPT", reason, get_timestamp(), format_double(elapsed_seconds(), 3)});
}

}  // namespace qlret
