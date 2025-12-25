#pragma once

#include "types.h"
#include "cli_parser.h"
#include "output_formatter.h"
#include <string>
#include <fstream>
#include <chrono>
#include <mutex>
#include <vector>
#include <map>

namespace qlret {

/**
 * Generate default CSV filename based on simulation parameters
 * Format: lret_n{qubits}_d{depth}_{mode}_{fdm}_{timestamp}.csv
 * Example: lret_n15_d10_compare_fdm_20251225_143022.csv
 */
std::string generate_default_csv_filename(const CLIOptions& opts);

/**
 * Structured CSV Writer for Excel-Compatible Output
 * 
 * The CSV is organized into clearly marked sections that can be easily
 * parsed and converted to Excel sheets:
 * 
 * Section Types:
 * - HEADER: Simulation configuration and parameters
 * - FDM_PROGRESS: Step-by-step FDM execution (if enabled)
 * - FDM_METRICS: Complete FDM metrics table
 * - LRET_PROGRESS_{MODE}: Step-by-step LRET execution per mode
 * - LRET_METRICS_{MODE}: Complete metrics table per mode
 * - MODE_COMPARISON: Comparison table across all LRET modes
 * - FDM_COMPARISON: LRET vs FDM comparison for each mode
 * - SUMMARY: Final summary
 * 
 * Each section starts with: SECTION,{section_name}
 * Each section ends with: END_SECTION,{section_name}
 * 
 * This format allows easy splitting into multiple Excel sheets.
 */
class StructuredCSVWriter {
public:
    StructuredCSVWriter() = default;
    explicit StructuredCSVWriter(const std::string& filename);
    ~StructuredCSVWriter();
    
    // Disable copy
    StructuredCSVWriter(const StructuredCSVWriter&) = delete;
    StructuredCSVWriter& operator=(const StructuredCSVWriter&) = delete;
    
    // Open/close
    bool open(const std::string& filename);
    void close();
    bool is_open() const { return file_.is_open(); }
    std::string get_filepath() const;
    const std::string& filename() const { return filename_; }
    
    //==========================================================================
    // Section 1: Header - Simulation Configuration
    //==========================================================================
    void write_header(const CLIOptions& opts, const NoiseStats& noise_stats);
    
    //==========================================================================
    // Section 2: FDM Progress (Step-by-Step)
    //==========================================================================
    void begin_fdm_progress(size_t num_qubits, size_t depth);
    void log_fdm_step(size_t step, const std::string& operation, 
                      double time_seconds, size_t memory_mb = 0);
    void log_fdm_gate(size_t step, size_t gate_idx, const std::string& gate_name,
                      const std::vector<size_t>& qubits, double time_seconds);
    void log_fdm_noise(size_t step, const std::string& noise_type, size_t qubit,
                       double time_seconds);
    void end_fdm_progress(double total_time, bool success, const std::string& message = "");
    
    //==========================================================================
    // Section 3: FDM Metrics (Complete Table)
    //==========================================================================
    void write_fdm_metrics(const FDMResult& fdm_result, size_t num_qubits,
                           const NoiseStats& noise_stats);
    
    //==========================================================================
    // Section 4: LRET Progress (Step-by-Step per Mode)
    //==========================================================================
    void begin_lret_progress(const std::string& mode, size_t num_qubits, size_t depth);
    void log_lret_step_start(const std::string& mode, size_t step);
    void log_lret_gate(const std::string& mode, size_t step, 
                       size_t rank_before, size_t rank_after, double time_seconds);
    void log_lret_kraus(const std::string& mode, size_t step, const std::string& noise_type,
                        size_t rank_before, size_t rank_after, double time_seconds);
    void log_lret_truncation(const std::string& mode, size_t step,
                             size_t rank_before, size_t rank_after, double time_seconds);
    void end_lret_progress(const std::string& mode, double total_time, 
                           size_t final_rank, bool success);
    
    //==========================================================================
    // Section 5: LRET Metrics per Mode (Complete Table)
    //==========================================================================
    void write_lret_mode_metrics(const std::string& mode, const ModeResult& result,
                                 const MetricsResult& vs_initial,
                                 const StateMetrics& state_metrics,
                                 const NoiseStats& noise_stats);
    
    //==========================================================================
    // Section 6: Mode Comparison Table (All LRET Modes)
    //==========================================================================
    void write_mode_comparison(const std::vector<ModeResult>& results,
                               const std::string& baseline_mode = "sequential");
    
    //==========================================================================
    // Section 7: FDM Comparison Table (Each LRET Mode vs FDM)
    //==========================================================================
    void write_fdm_comparison(const std::vector<ModeResult>& results,
                              const FDMResult& fdm_result,
                              const std::map<std::string, MetricsResult>& fdm_metrics);
    
    //==========================================================================
    // Section 8: Final Summary
    //==========================================================================
    void write_summary(const CLIOptions& opts, 
                       const std::vector<ModeResult>& results,
                       const std::optional<FDMResult>& fdm_result,
                       double total_wall_time);
    
    //==========================================================================
    // Error/Warning/Interrupt Logging
    //==========================================================================
    void log_error(const std::string& context, const std::string& message);
    void log_warning(const std::string& context, const std::string& message);
    void log_interrupt(const std::string& reason);
    
private:
    std::ofstream file_;
    std::string filename_;
    std::mutex mutex_;
    std::chrono::steady_clock::time_point start_time_;
    
    // Section tracking
    std::string current_section_;
    
    // Helper methods
    void begin_section(const std::string& section_name);
    void end_section();
    void write_line(const std::string& line);
    void write_csv_row(const std::vector<std::string>& fields);
    std::string get_timestamp() const;
    double elapsed_seconds() const;
    std::string escape_csv(const std::string& s) const;
    std::string format_double(double val, int precision = 6) const;
};

// Global structured CSV writer pointer
extern StructuredCSVWriter* g_structured_csv;

}  // namespace qlret
