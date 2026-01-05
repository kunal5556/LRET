#pragma once

#include "qec_types.h"
#include "qec_stabilizer.h"
#include "qec_syndrome.h"
#include "qec_decoder.h"
#include "simulator.h"
#include <memory>
#include <vector>
#include <string>
#include <optional>

namespace qlret {

//==============================================================================
// Logical Qubit State
//==============================================================================

/**
 * @brief Represents the logical state of an encoded qubit
 */
struct LogicalState {
    CMatrix L;           // Low-rank L-factor for logical state
    size_t code_distance;
    QECCodeType code_type;
    
    // Logical Pauli expectation values
    double logical_x_expectation() const;
    double logical_y_expectation() const;
    double logical_z_expectation() const;
    
    // Logical fidelity with target state
    double fidelity_with(const LogicalState& target) const;
};

//==============================================================================
// Logical Gate Operations
//==============================================================================

/**
 * @brief Transversal and fault-tolerant logical gates
 */
enum class LogicalGateType {
    LOGICAL_I,      // Identity
    LOGICAL_X,      // Transversal X
    LOGICAL_Y,      // Transversal Y
    LOGICAL_Z,      // Transversal Z
    LOGICAL_H,      // Transversal H (code-specific)
    LOGICAL_S,      // Transversal S (for surface code)
    LOGICAL_CNOT,   // Transversal CNOT between logical qubits
    LOGICAL_MEASURE // Logical measurement
};

struct LogicalGate {
    LogicalGateType type;
    std::vector<size_t> targets;  // Logical qubit indices
    std::optional<int> measurement_result;
};

//==============================================================================
// QEC Round Result
//==============================================================================

struct QECRoundResult {
    Syndrome syndrome;
    Correction correction;
    bool detected_error = false;
    bool logical_error = false;
    double round_time_ms = 0.0;
};

//==============================================================================
// Logical Qubit Interface
//==============================================================================

/**
 * @brief High-level interface for fault-tolerant quantum computation
 * 
 * Manages logical qubits encoded in stabilizer codes with:
 * - Automatic syndrome extraction and correction
 * - Transversal logical gate application
 * - Error tracking and fidelity estimation
 */
class LogicalQubit {
public:
    struct Config {
        QECCodeType code_type = QECCodeType::SURFACE;
        size_t distance = 3;
        DecoderType decoder_type = DecoderType::MWPM;
        double physical_error_rate = 0.001;
        double measurement_error_rate = 0.001;
        size_t syndrome_rounds = 1;  // Rounds per QEC cycle
        bool auto_correct = true;    // Apply corrections automatically
    };

    explicit LogicalQubit(Config config = {});
    LogicalQubit(const LogicalQubit&) = delete;
    LogicalQubit& operator=(const LogicalQubit&) = delete;
    LogicalQubit(LogicalQubit&&) = default;
    LogicalQubit& operator=(LogicalQubit&&) = default;

    //--------------------------------------------------------------------------
    // Initialization
    //--------------------------------------------------------------------------

    /**
     * @brief Initialize logical qubit in |0_L⟩ state
     */
    void initialize_zero();

    /**
     * @brief Initialize logical qubit in |1_L⟩ state
     */
    void initialize_one();

    /**
     * @brief Initialize logical qubit in |+_L⟩ state
     */
    void initialize_plus();

    /**
     * @brief Initialize logical qubit in |-_L⟩ state
     */
    void initialize_minus();

    /**
     * @brief Initialize from arbitrary (encoded) state
     * @param L Low-rank L-factor (must match code dimensions)
     */
    void initialize_from_state(const CMatrix& L);

    //--------------------------------------------------------------------------
    // Logical Gate Operations
    //--------------------------------------------------------------------------

    /**
     * @brief Apply transversal logical X gate
     */
    void apply_logical_x();

    /**
     * @brief Apply transversal logical Y gate
     */
    void apply_logical_y();

    /**
     * @brief Apply transversal logical Z gate
     */
    void apply_logical_z();

    /**
     * @brief Apply transversal logical Hadamard (code-specific)
     */
    void apply_logical_h();

    /**
     * @brief Apply transversal logical S gate
     */
    void apply_logical_s();

    /**
     * @brief Measure logical qubit in Z basis
     * @return Measurement outcome (0 or 1)
     */
    int measure_logical_z();

    /**
     * @brief Measure logical qubit in X basis
     * @return Measurement outcome (0 or 1)
     */
    int measure_logical_x();

    //--------------------------------------------------------------------------
    // Error Correction
    //--------------------------------------------------------------------------

    /**
     * @brief Perform one round of QEC
     * @return Result containing syndrome, correction, and error info
     */
    QECRoundResult qec_round();

    /**
     * @brief Perform multiple rounds of QEC with time-domain decoding
     * @param rounds Number of syndrome measurement rounds
     * @return Results for each round
     */
    std::vector<QECRoundResult> qec_rounds(size_t rounds);

    /**
     * @brief Inject random error for testing
     * @param p Physical error probability
     */
    void inject_error(double p);

    /**
     * @brief Inject specific Pauli error
     * @param error PauliString to apply
     */
    void inject_error(const PauliString& error);

    //--------------------------------------------------------------------------
    // State Inspection
    //--------------------------------------------------------------------------

    /**
     * @brief Get current logical state
     */
    LogicalState get_state() const;

    /**
     * @brief Estimate logical fidelity with ideal state
     * @param ideal_state Target state for comparison
     */
    double estimate_fidelity(const CMatrix& ideal_state) const;

    /**
     * @brief Get accumulated error since last correction
     */
    PauliString get_accumulated_error() const;

    /**
     * @brief Check if logical error has occurred
     */
    bool has_logical_error() const;

    //--------------------------------------------------------------------------
    // Configuration
    //--------------------------------------------------------------------------

    const StabilizerCode& code() const { return *code_; }
    const QECDecoder& decoder() const { return *decoder_; }
    const Config& config() const { return config_; }

    void set_physical_error_rate(double p);
    void set_measurement_error_rate(double p);
    void set_syndrome_rounds(size_t rounds);
    void set_auto_correct(bool enable);

    //--------------------------------------------------------------------------
    // Statistics
    //--------------------------------------------------------------------------

    struct Stats {
        size_t total_qec_rounds = 0;
        size_t detected_errors = 0;
        size_t logical_errors = 0;
        double total_qec_time_ms = 0.0;
        
        double detection_rate() const {
            return total_qec_rounds > 0 ? 
                static_cast<double>(detected_errors) / total_qec_rounds : 0.0;
        }
        double logical_error_rate() const {
            return total_qec_rounds > 0 ? 
                static_cast<double>(logical_errors) / total_qec_rounds : 0.0;
        }
    };

    const Stats& stats() const { return stats_; }
    void reset_stats() { stats_ = Stats{}; }

private:
    Config config_;
    std::unique_ptr<StabilizerCode> code_;
    std::unique_ptr<QECDecoder> decoder_;
    std::unique_ptr<SyndromeExtractor> extractor_;
    std::unique_ptr<ErrorInjector> error_injector_;

    CMatrix L_;                    // Current state L-factor
    PauliString accumulated_error_; // Error tracking for simulation
    Stats stats_;

    // Apply Pauli string to state (for transversal gates)
    void apply_pauli_string(const PauliString& pauli);

    // Update state from physical qubit operations
    void apply_physical_gate(const std::string& gate, size_t qubit);
    void apply_physical_gate(const std::string& gate, size_t ctrl, size_t tgt);
};

//==============================================================================
// Multi-Qubit Logical Operations
//==============================================================================

/**
 * @brief Manager for multiple logical qubits
 */
class LogicalRegister {
public:
    struct Config {
        QECCodeType code_type = QECCodeType::SURFACE;
        size_t distance = 3;
        DecoderType decoder_type = DecoderType::MWPM;
        double physical_error_rate = 0.001;
    };

    LogicalRegister(size_t num_qubits, Config config = {});

    /**
     * @brief Get logical qubit by index
     */
    LogicalQubit& qubit(size_t idx);
    const LogicalQubit& qubit(size_t idx) const;

    /**
     * @brief Apply transversal CNOT between logical qubits
     * @param control Control logical qubit index
     * @param target Target logical qubit index
     */
    void apply_logical_cnot(size_t control, size_t target);

    /**
     * @brief Perform QEC round on all qubits
     */
    std::vector<QECRoundResult> qec_round_all();

    /**
     * @brief Initialize all qubits to |0_L⟩
     */
    void initialize_all_zero();

    size_t size() const { return qubits_.size(); }

private:
    std::vector<LogicalQubit> qubits_;
    Config config_;
};

//==============================================================================
// QEC Simulation Runner
//==============================================================================

/**
 * @brief Run Monte Carlo QEC simulations
 */
class QECSimulator {
public:
    struct SimConfig {
        QECCodeType code_type = QECCodeType::SURFACE;
        size_t distance = 3;
        DecoderType decoder_type = DecoderType::MWPM;
        double physical_error_rate = 0.001;
        size_t num_trials = 10000;
        size_t qec_rounds_per_trial = 1;
        unsigned seed = 42;
    };

    struct SimResult {
        size_t num_trials;
        size_t num_logical_errors;
        double logical_error_rate;
        double logical_error_rate_std;  // Standard deviation
        double avg_decode_time_ms;
        std::vector<size_t> errors_per_round;  // Distribution across rounds
    };

    explicit QECSimulator(SimConfig config = {});

    /**
     * @brief Run QEC simulation to estimate logical error rate
     */
    SimResult run();

    /**
     * @brief Estimate threshold error rate
     * @param distances Vector of code distances to test
     * @param error_rates Vector of physical error rates to test
     * @return Map of distance -> (error_rate -> logical_error_rate)
     */
    std::map<size_t, std::map<double, double>> 
    estimate_threshold(const std::vector<size_t>& distances,
                       const std::vector<double>& error_rates);

    const SimConfig& config() const { return config_; }

private:
    SimConfig config_;
};

}  // namespace qlret
