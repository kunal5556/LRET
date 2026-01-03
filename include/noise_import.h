#pragma once

/**
 * @file noise_import.h
 * @brief Import and Convert Qiskit Noise Models to LRET Format (Phase 4.1)
 * 
 * Enables import of real quantum device noise profiles from IBM Quantum (IBMQ)
 * and other sources that use the Qiskit Aer noise model format.
 * 
 * ============================================================================
 * QISKIT AER NOISE MODEL FORMAT
 * ============================================================================
 * 
 * Standard JSON schema from Qiskit's NoiseModel.to_dict():
 * 
 * @code{.json}
 * {
 *   "errors": [
 *     {
 *       "type": "qerror",
 *       "operations": ["x", "y", "h", "rx", "ry", "rz"],
 *       "gate_qubits": [[0], [1], [2]],
 *       "probabilities": [0.001, 0.0005],
 *       "instructions": [
 *         [{"name": "x", "qubits": [0]}],
 *         [{"name": "pauli", "params": ["X"], "qubits": [0]}]
 *       ]
 *     },
 *     {
 *       "type": "thermal_relaxation_error",
 *       "operations": ["id"],
 *       "gate_qubits": [[0]],
 *       "gate_time": 50e-9,
 *       "T1": 50e-6,
 *       "T2": 70e-6
 *     },
 *     {
 *       "type": "depolarizing_error",
 *       "operations": ["cx"],
 *       "gate_qubits": [[0, 1], [1, 2]],
 *       "param": 0.01
 *     }
 *   ]
 * }
 * @endcode
 * 
 * ============================================================================
 * CONVERSION STRATEGY
 * ============================================================================
 * 
 * Qiskit Error Type → LRET Kraus Operators:
 * 
 * 1. **depolarizing_error(p):**
 *    - Kraus: {√(1-3p/4)·I, √(p/4)·X, √(p/4)·Y, √(p/4)·Z}
 *    - Maps to: LRET's NoiseType::DEPOLARIZING
 * 
 * 2. **thermal_relaxation_error(T1, T2, gate_time):**
 *    - p_reset = 1 - exp(-gate_time/T1)
 *    - p_dephase = 0.5 * (1 - exp(-gate_time/T2))
 *    - Maps to: Combined AMPLITUDE_DAMPING + PHASE_DAMPING
 * 
 * 3. **amplitude_damping_error(gamma):**
 *    - Kraus: {[[1, 0], [0, √(1-γ)]], [[0, √γ], [0, 0]]}
 *    - Maps to: NoiseType::AMPLITUDE_DAMPING
 * 
 * 4. **phase_damping_error(lambda):**
 *    - Kraus: {[[1, 0], [0, √(1-λ)]], [[0, 0], [0, √λ]]}
 *    - Maps to: NoiseType::PHASE_DAMPING
 * 
 * 5. **pauli_error(probs):**
 *    - Kraus: {√p_I·I, √p_X·X, √p_Y·Y, √p_Z·Z}
 *    - Maps to: Generic Kraus application
 * 
 * ============================================================================
 * USAGE EXAMPLES
 * ============================================================================
 * 
 * @code
 * // Import noise model from JSON file
 * NoiseModelImporter importer;
 * NoiseModel noise = importer.load_from_json("ibmq_bogota_noise.json");
 * 
 * // Apply noise to circuit
 * QuantumSequence noisy_circuit = importer.apply_noise_model(
 *     clean_circuit, 
 *     noise
 * );
 * 
 * // Run simulation with real device noise
 * SimResult result = run_simulation(L_init, noisy_circuit, num_qubits, config);
 * @endcode
 * 
 * Command line:
 * @code
 * ./lret --qubits=5 --depth=100 --noise-model=ibmq_bogota.json
 * @endcode
 * 
 * @author LRET Team (Phase 4.1)
 * @date January 2026
 * @version 1.0
 */

#include "types.h"
#include "gates_and_noise.h"
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <vector>
#include <map>
#include <memory>

namespace qlret {

//==============================================================================
// Noise Model Structures
//==============================================================================

/**
 * @brief Types of noise errors from Qiskit
 */
enum class QiskitErrorType {
    QERROR,                      ///< Generic quantum error with Kraus operators
    DEPOLARIZING,                ///< Depolarizing channel
    THERMAL_RELAXATION,          ///< T1 and T2 relaxation
    AMPLITUDE_DAMPING,           ///< Amplitude damping (T1)
    PHASE_DAMPING,               ///< Phase damping (T2*)
    PAULI,                       ///< Pauli channel (X, Y, Z with probabilities)
    READOUT,                     ///< Measurement error (not applicable to LRET)
    RESET,                       ///< Reset error
    UNKNOWN                      ///< Unrecognized error type
};

/**
 * @brief A single noise error specification from Qiskit
 */
struct QiskitNoiseError {
    QiskitErrorType type;
    std::vector<std::string> operations;        ///< Gate names this error applies to
    std::vector<std::vector<size_t>> gate_qubits; ///< Qubit indices for each gate
    
    // Type-specific parameters
    double probability = 0.0;                   ///< For depolarizing, amplitude damping, etc.
    std::vector<double> probabilities;          ///< For Pauli channels
    double gate_time = 0.0;                     ///< For thermal relaxation (seconds)
    double T1 = 0.0;                            ///< Relaxation time (seconds)
    double T2 = 0.0;                            ///< Dephasing time (seconds)
    
    // Kraus operators (for generic qerror)
    std::vector<Matrix2cd> kraus_ops;
    
    // Instructions (Qiskit format - for future reference)
    std::vector<std::vector<std::string>> instructions;
};

/**
 * @brief Complete noise model (collection of errors)
 */
struct NoiseModel {
    std::string device_name;                    ///< Source device (e.g., "ibmq_bogota")
    std::string backend_version;                ///< Backend version
    std::string noise_model_version;            ///< Noise model version
    
    std::vector<QiskitNoiseError> errors;       ///< All error specifications
    
    // Qubit-specific error lookup (for fast access)
    std::map<std::string, std::vector<QiskitNoiseError*>> gate_errors;  // gate_name -> errors
    std::map<size_t, std::vector<QiskitNoiseError*>> qubit_errors;      // qubit_id -> errors
    
    bool is_empty() const { return errors.empty(); }
    size_t num_errors() const { return errors.size(); }
    
    /**
     * @brief Find errors applicable to a specific gate and qubits
     */
    std::vector<const QiskitNoiseError*> find_applicable_errors(
        const std::string& gate_name,
        const std::vector<size_t>& qubits
    ) const;
};

//==============================================================================
// Noise Model Importer
//==============================================================================

/**
 * @brief Main class for importing Qiskit noise models
 */
class NoiseModelImporter {
public:
    NoiseModelImporter() = default;
    ~NoiseModelImporter() = default;
    
    //==========================================================================
    // JSON Import
    //==========================================================================
    
    /**
     * @brief Load noise model from Qiskit JSON file
     * 
     * Supports:
     * - Direct NoiseModel.to_dict() output
     * - IBMQ backend calibration data
     * - Custom noise specifications
     * 
     * @param filepath Path to JSON file
     * @return Parsed noise model
     * @throws std::runtime_error if file not found or JSON invalid
     */
    NoiseModel load_from_json(const std::string& filepath);
    
    /**
     * @brief Parse noise model from JSON string
     */
    NoiseModel load_from_json_string(const std::string& json_str);
    
    //==========================================================================
    // Qiskit Error Conversion
    //==========================================================================
    
    /**
     * @brief Convert Qiskit error to LRET NoiseOp sequence
     * 
     * Maps Qiskit error types to appropriate LRET noise operations.
     * Some errors (like thermal_relaxation) may produce multiple NoiseOps.
     * 
     * @param error Qiskit error specification
     * @param qubit Target qubit for noise application
     * @return Vector of LRET NoiseOps
     */
    std::vector<NoiseOp> convert_qiskit_error_to_lret(
        const QiskitNoiseError& error,
        size_t qubit
    );
    
    /**
     * @brief Convert thermal relaxation to amplitude + phase damping
     */
    std::vector<NoiseOp> convert_thermal_relaxation(
        double T1, 
        double T2, 
        double gate_time,
        size_t qubit
    );
    
    /**
     * @brief Convert Pauli channel to depolarizing (approximation)
     * 
     * If probabilities are roughly equal, maps to depolarizing.
     * Otherwise, applies weighted Pauli errors.
     */
    std::vector<NoiseOp> convert_pauli_channel(
        const std::vector<double>& probs,  // [p_I, p_X, p_Y, p_Z]
        size_t qubit
    );
    
    //==========================================================================
    // Circuit Noise Application
    //==========================================================================
    
    /**
     * @brief Apply noise model to clean circuit
     * 
     * Inserts appropriate noise operations after each gate based on
     * the noise model specifications.
     * 
     * @param clean_circuit Circuit without noise
     * @param noise_model Noise specifications
     * @return Circuit with noise operations inserted
     */
    QuantumSequence apply_noise_model(
        const QuantumSequence& clean_circuit,
        const NoiseModel& noise_model
    );
    
    /**
     * @brief Apply noise to a single gate operation
     * 
     * Looks up applicable errors from noise model and inserts them.
     * 
     * @param gate Gate operation
     * @param noise_model Noise specifications
     * @param noisy_sequence Output sequence (append gate + noise)
     */
    void apply_noise_to_gate(
        const GateOp& gate,
        const NoiseModel& noise_model,
        QuantumSequence& noisy_sequence
    );
    
    //==========================================================================
    // Utilities
    //==========================================================================
    
    /**
     * @brief Print noise model statistics
     */
    void print_noise_model_summary(const NoiseModel& model) const;
    
    /**
     * @brief Validate noise model (check probabilities sum to 1, etc.)
     */
    bool validate_noise_model(const NoiseModel& model, bool verbose = false) const;
    
    /**
     * @brief Convert Qiskit gate name to LRET GateType
     * 
     * Maps: "x" -> X, "cx" -> CNOT, "h" -> H, etc.
     */
    GateType qiskit_gate_name_to_lret(const std::string& qiskit_name) const;

private:
    /**
     * @brief Parse JSON error object
     */
    QiskitNoiseError parse_error_from_json(const nlohmann::json& error_json);
    
    /**
     * @brief Determine error type from JSON
     */
    QiskitErrorType determine_error_type(const nlohmann::json& error_json);
    
    /**
     * @brief Build lookup tables after parsing
     */
    void build_lookup_tables(NoiseModel& model);
};

//==============================================================================
// Convenience Functions
//==============================================================================

/**
 * @brief Quick load: filename -> noise model
 */
NoiseModel load_noise_model(const std::string& filepath);

/**
 * @brief Quick apply: circuit + noise model -> noisy circuit
 */
QuantumSequence apply_noise(
    const QuantumSequence& circuit,
    const NoiseModel& noise_model
);

/**
 * @brief Quick apply: circuit + noise file -> noisy circuit
 */
QuantumSequence apply_noise_from_file(
    const QuantumSequence& circuit,
    const std::string& noise_json_path
);

} // namespace qlret
