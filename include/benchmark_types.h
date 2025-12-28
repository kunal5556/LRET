#pragma once

#include "types.h"
#include <vector>
#include <string>
#include <chrono>

namespace qlret {

//==============================================================================
// Rank Evolution Tracking (Paper Figure: Rank vs Circuit Layer)
//==============================================================================

/**
 * @brief Single event in rank evolution history
 * 
 * Tracks rank changes after each operation in the circuit.
 * This enables plotting rank vs. circuit depth/layer (key LRET paper figure).
 */
struct RankEvent {
    size_t step;                    // Operation index in circuit
    size_t layer;                   // Circuit layer (depth index)
    std::string operation_type;     // "gate", "noise", "truncation"
    std::string operation_name;     // "H(0)", "CNOT(1,2)", "depolarizing(3)", "truncate"
    size_t rank_before;             // Rank before this operation
    size_t rank_after;              // Rank after this operation
    double time_seconds;            // Time for this operation
    double cumulative_time;         // Total time up to this point
    
    // Memory tracking
    size_t memory_bytes;            // Current L matrix memory usage
    
    RankEvent() = default;
    RankEvent(size_t s, size_t l, const std::string& op_type, const std::string& op_name,
              size_t r_before, size_t r_after, double t, double cum_t, size_t mem = 0)
        : step(s), layer(l), operation_type(op_type), operation_name(op_name),
          rank_before(r_before), rank_after(r_after), time_seconds(t), 
          cumulative_time(cum_t), memory_bytes(mem) {}
};

/**
 * @brief Complete rank evolution history for a simulation run
 * 
 * Captures the full trajectory of rank changes during circuit execution.
 * Essential for reproducing LRET paper Figure 2 (rank vs. depth).
 */
struct RankEvolution {
    std::vector<RankEvent> events;  // All rank-changing events
    
    // Summary statistics
    size_t initial_rank = 1;
    size_t final_rank = 1;
    size_t max_rank = 1;
    size_t min_rank = 1;
    double avg_rank = 1.0;
    
    // Timing breakdown (Paper analysis)
    double total_gate_time = 0.0;       // Time spent applying gates
    double total_noise_time = 0.0;      // Time spent applying noise (Kraus)
    double total_truncation_time = 0.0; // Time spent in SVD/truncation
    size_t truncation_count = 0;        // Number of truncation events
    
    // Memory tracking
    size_t peak_memory_bytes = 0;
    
    void add_event(const RankEvent& event) {
        events.push_back(event);
        
        // Update summary stats
        if (events.size() == 1) {
            initial_rank = event.rank_before;
            min_rank = event.rank_after;
        }
        final_rank = event.rank_after;
        max_rank = std::max(max_rank, event.rank_after);
        min_rank = std::min(min_rank, event.rank_after);
        peak_memory_bytes = std::max(peak_memory_bytes, event.memory_bytes);
        
        // Update timing breakdown
        if (event.operation_type == "gate") {
            total_gate_time += event.time_seconds;
        } else if (event.operation_type == "noise" || event.operation_type == "kraus") {
            total_noise_time += event.time_seconds;
        } else if (event.operation_type == "truncation") {
            total_truncation_time += event.time_seconds;
            truncation_count++;
        }
    }
    
    void compute_average_rank() {
        if (events.empty()) return;
        double sum = 0.0;
        for (const auto& e : events) {
            sum += e.rank_after;
        }
        avg_rank = sum / events.size();
    }
    
    size_t size() const { return events.size(); }
    bool empty() const { return events.empty(); }
    
    // Get rank at specific layer/step
    size_t rank_at_layer(size_t layer) const {
        for (auto it = events.rbegin(); it != events.rend(); ++it) {
            if (it->layer <= layer) return it->rank_after;
        }
        return initial_rank;
    }
};

//==============================================================================
// Parameter Sweep Configuration (Paper Figures: ε vs time, p vs time, n vs time)
//==============================================================================

/**
 * @brief Type of parameter being swept
 */
enum class SweepType {
    NONE,           // No sweep (single run)
    EPSILON,        // Truncation threshold sweep (ε)
    NOISE_PROB,     // Noise probability sweep (p)
    QUBITS,         // Number of qubits sweep (n)
    DEPTH,          // Circuit depth sweep (d)
    INITIAL_RANK,   // Initial rank sweep (for parallel benchmarking)
    CROSSOVER       // LRET vs FDM crossover analysis
};

/**
 * @brief Configuration for parameter sweeps
 * 
 * Enables systematic benchmarking as done in the LRET paper:
 * - Figure: Runtime vs. truncation threshold (ε)
 * - Figure: Runtime vs. noise probability (p)
 * - Figure: Runtime vs. number of qubits (n)
 * - Figure: LRET vs FDM crossover point
 */
struct SweepConfig {
    SweepType type = SweepType::NONE;
    
    // Sweep values (one of these is used based on type)
    std::vector<double> epsilon_values;     // Truncation thresholds: 1e-7, 1e-6, ..., 1e-2
    std::vector<double> noise_values;       // Noise probabilities: 0.0, 0.01, 0.05, 0.1, ...
    std::vector<size_t> qubit_values;       // Qubit counts: 5, 8, 10, 12, 15, 18, 20
    std::vector<size_t> depth_values;       // Circuit depths: 10, 20, 50, 100
    std::vector<size_t> rank_values;        // Initial ranks: 1, 2, 4, 8, 16, 32
    
    // Crossover analysis bounds (for LRET vs FDM)
    size_t crossover_min_qubits = 5;
    size_t crossover_max_qubits = 15;
    
    // Common options
    size_t num_trials = 1;          // Repeat each point for statistics
    bool include_fdm = false;       // Compare with FDM at each point
    bool track_rank_evolution = false; // Track rank at each operation
    
    // Output
    std::string output_prefix = "sweep";  // Prefix for output files
    
    // Helper: Check if sweep is active
    bool is_active() const { return type != SweepType::NONE; }
    
    // Helper: Get number of sweep points
    size_t num_points() const {
        switch (type) {
            case SweepType::EPSILON:     return epsilon_values.size();
            case SweepType::NOISE_PROB:  return noise_values.size();
            case SweepType::QUBITS:      return qubit_values.size();
            case SweepType::DEPTH:       return depth_values.size();
            case SweepType::INITIAL_RANK: return rank_values.size();
            case SweepType::CROSSOVER:   return crossover_max_qubits - crossover_min_qubits + 1;
            default: return 0;
        }
    }
    
    // Parse sweep string: "1e-7,1e-6,1e-5,1e-4" or "1e-7:1e-2:6" (start:end:count)
    static std::vector<double> parse_double_sweep(const std::string& str);
    static std::vector<size_t> parse_size_sweep(const std::string& str);
};

//==============================================================================
// Single Sweep Point Result
//==============================================================================

/**
 * @brief Result from a single point in a parameter sweep
 */
struct SweepPointResult {
    // Parameter values at this point
    double epsilon = 1e-4;
    double noise_prob = 0.01;
    size_t num_qubits = 11;
    size_t depth = 13;
    size_t initial_rank = 1;
    
    // LRET results
    double lret_time_seconds = 0.0;
    size_t lret_final_rank = 0;
    size_t lret_max_rank = 0;
    double lret_purity = 0.0;
    double lret_entropy = 0.0;
    size_t lret_memory_bytes = 0;
    
    // Timing breakdown
    double gate_time = 0.0;
    double noise_time = 0.0;
    double truncation_time = 0.0;
    size_t truncation_count = 0;
    
    // FDM results (if enabled)
    bool fdm_run = false;
    double fdm_time_seconds = 0.0;
    size_t fdm_memory_bytes = 0;
    
    // Comparison metrics (LRET vs FDM)
    double fidelity_vs_fdm = 0.0;
    double trace_distance_vs_fdm = 0.0;
    
    // Trial statistics (if num_trials > 1)
    double lret_time_stddev = 0.0;
    double fdm_time_stddev = 0.0;
    
    // Rank evolution (if tracking enabled)
    RankEvolution rank_evolution;
    
    // Speedup ratio: FDM_time / LRET_time
    double speedup_vs_fdm() const {
        if (lret_time_seconds <= 0) return 0.0;
        if (!fdm_run || fdm_time_seconds <= 0) return 0.0;
        return fdm_time_seconds / lret_time_seconds;
    }
    
    // Memory ratio: FDM_memory / LRET_memory
    double memory_ratio_vs_fdm() const {
        if (lret_memory_bytes == 0) return 0.0;
        if (!fdm_run || fdm_memory_bytes == 0) return 0.0;
        return static_cast<double>(fdm_memory_bytes) / lret_memory_bytes;
    }
};

//==============================================================================
// Complete Sweep Results
//==============================================================================

/**
 * @brief Complete results from a parameter sweep
 */
struct SweepResults {
    SweepType type;
    std::vector<SweepPointResult> points;
    
    // Metadata
    std::string sweep_parameter_name;  // "epsilon", "noise_prob", "qubits", etc.
    std::chrono::system_clock::time_point start_time;
    std::chrono::system_clock::time_point end_time;
    double total_wall_time_seconds = 0.0;
    
    // For crossover analysis
    size_t crossover_qubit_count = 0;  // Qubit count where LRET becomes faster than FDM
    bool crossover_found = false;
    
    void add_point(const SweepPointResult& point) {
        points.push_back(point);
    }
    
    size_t size() const { return points.size(); }
    
    // Find crossover point (LRET faster than FDM)
    void find_crossover() {
        crossover_found = false;
        for (const auto& p : points) {
            if (p.fdm_run && p.lret_time_seconds < p.fdm_time_seconds) {
                crossover_qubit_count = p.num_qubits;
                crossover_found = true;
                return;
            }
        }
    }
};

//==============================================================================
// Detailed Timing Breakdown (Paper: Where is time spent?)
//==============================================================================

/**
 * @brief Detailed timing breakdown for simulation analysis
 * 
 * Shows where computation time is spent:
 * - Gate application (matrix multiplication)
 * - Noise/Kraus application (rank expansion)
 * - Truncation/SVD (rank compression)
 */
struct TimingBreakdown {
    // Absolute times
    double gate_time = 0.0;           // Time in gate application
    double noise_time = 0.0;          // Time in noise/Kraus application
    double truncation_time = 0.0;     // Time in SVD/truncation
    double overhead_time = 0.0;       // Other overhead (memory, etc.)
    double total_time = 0.0;          // Total simulation time
    
    // Percentages
    double gate_percent() const { return total_time > 0 ? 100.0 * gate_time / total_time : 0; }
    double noise_percent() const { return total_time > 0 ? 100.0 * noise_time / total_time : 0; }
    double truncation_percent() const { return total_time > 0 ? 100.0 * truncation_time / total_time : 0; }
    double overhead_percent() const { return total_time > 0 ? 100.0 * overhead_time / total_time : 0; }
    
    // Counts
    size_t gate_count = 0;
    size_t noise_count = 0;
    size_t truncation_count = 0;
    
    // Per-operation averages
    double avg_gate_time() const { return gate_count > 0 ? gate_time / gate_count : 0; }
    double avg_noise_time() const { return noise_count > 0 ? noise_time / noise_count : 0; }
    double avg_truncation_time() const { return truncation_count > 0 ? truncation_time / truncation_count : 0; }
    
    void compute_overhead() {
        overhead_time = total_time - gate_time - noise_time - truncation_time;
        if (overhead_time < 0) overhead_time = 0;
    }
};

//==============================================================================
// Memory Usage Tracking (Paper: LRET vs FDM memory scaling)
//==============================================================================

/**
 * @brief Memory usage comparison between LRET and FDM
 * 
 * LRET Memory: O(2^n × rank) - depends on rank, not full 4^n
 * FDM Memory: O(4^n) - full density matrix
 */
struct MemoryComparison {
    size_t num_qubits = 0;
    
    // LRET memory
    size_t lret_L_matrix_bytes = 0;     // L matrix storage
    size_t lret_work_memory_bytes = 0;  // Working memory (SVD, etc.)
    size_t lret_peak_bytes = 0;         // Peak memory usage
    size_t lret_final_rank = 0;
    
    // FDM memory (theoretical or measured)
    size_t fdm_rho_matrix_bytes = 0;    // Full density matrix
    size_t fdm_work_memory_bytes = 0;   // Working memory
    size_t fdm_peak_bytes = 0;          // Peak memory usage
    
    // Theoretical estimates
    size_t fdm_theoretical_bytes() const {
        // ρ is 2^n × 2^n complex doubles
        size_t dim = 1ULL << num_qubits;
        return dim * dim * sizeof(std::complex<double>);
    }
    
    size_t lret_theoretical_bytes(size_t rank) const {
        // L is 2^n × rank complex doubles
        size_t dim = 1ULL << num_qubits;
        return dim * rank * sizeof(std::complex<double>);
    }
    
    // Memory savings ratio
    double memory_savings_ratio() const {
        if (lret_peak_bytes == 0) return 0.0;
        size_t fdm_mem = fdm_peak_bytes > 0 ? fdm_peak_bytes : fdm_theoretical_bytes();
        return static_cast<double>(fdm_mem) / lret_peak_bytes;
    }
    
    // Human-readable memory strings
    std::string lret_memory_str() const;
    std::string fdm_memory_str() const;
};

//==============================================================================
// Helper Functions
//==============================================================================

std::string sweep_type_to_string(SweepType type);
SweepType string_to_sweep_type(const std::string& str);

// Memory size to human-readable string
std::string bytes_to_human_readable(size_t bytes);

}  // namespace qlret
