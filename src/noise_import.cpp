#include "noise_import.h"
#include "utils.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdexcept>

using json = nlohmann::json;

namespace qlret {

//==============================================================================
// Error Type Determination
//==============================================================================

QiskitErrorType NoiseModelImporter::determine_error_type(const json& error_json) {
    if (!error_json.contains("type")) {
        return QiskitErrorType::UNKNOWN;
    }
    
    std::string type_str = error_json["type"].get<std::string>();
    
    // Convert to lowercase for case-insensitive comparison
    std::transform(type_str.begin(), type_str.end(), type_str.begin(), ::tolower);
    
    if (type_str.find("depolarizing") != std::string::npos) {
        return QiskitErrorType::DEPOLARIZING;
    }
    if (type_str.find("thermal") != std::string::npos || type_str.find("relaxation") != std::string::npos) {
        return QiskitErrorType::THERMAL_RELAXATION;
    }
    if (type_str.find("amplitude") != std::string::npos && type_str.find("damping") != std::string::npos) {
        return QiskitErrorType::AMPLITUDE_DAMPING;
    }
    if (type_str.find("phase") != std::string::npos && type_str.find("damping") != std::string::npos) {
        return QiskitErrorType::PHASE_DAMPING;
    }
    if (type_str.find("pauli") != std::string::npos) {
        return QiskitErrorType::PAULI;
    }
    if (type_str.find("readout") != std::string::npos) {
        return QiskitErrorType::READOUT;
    }
    if (type_str.find("reset") != std::string::npos) {
        return QiskitErrorType::RESET;
    }
    if (type_str == "qerror") {
        return QiskitErrorType::QERROR;
    }
    
    return QiskitErrorType::UNKNOWN;
}

//==============================================================================
// JSON Parsing
//==============================================================================

QiskitNoiseError NoiseModelImporter::parse_error_from_json(const json& error_json) {
    QiskitNoiseError error;
    
    // Determine error type
    error.type = determine_error_type(error_json);
    
    // Parse operations (gate names)
    if (error_json.contains("operations")) {
        error.operations = error_json["operations"].get<std::vector<std::string>>();
    }
    
    // Parse gate_qubits (which qubits this error applies to)
    if (error_json.contains("gate_qubits")) {
        for (const auto& qubit_list : error_json["gate_qubits"]) {
            std::vector<size_t> qubits;
            for (const auto& q : qubit_list) {
                qubits.push_back(q.get<size_t>());
            }
            error.gate_qubits.push_back(qubits);
        }
    }
    
    // Parse type-specific parameters
    switch (error.type) {
        case QiskitErrorType::DEPOLARIZING:
            if (error_json.contains("param")) {
                error.probability = error_json["param"].get<double>();
            } else if (error_json.contains("probability")) {
                error.probability = error_json["probability"].get<double>();
            }
            break;
            
        case QiskitErrorType::THERMAL_RELAXATION:
            if (error_json.contains("T1")) {
                error.T1 = error_json["T1"].get<double>();
            }
            if (error_json.contains("T2")) {
                error.T2 = error_json["T2"].get<double>();
            }
            if (error_json.contains("gate_time")) {
                error.gate_time = error_json["gate_time"].get<double>();
            }
            break;
            
        case QiskitErrorType::AMPLITUDE_DAMPING:
        case QiskitErrorType::PHASE_DAMPING:
            if (error_json.contains("param")) {
                error.probability = error_json["param"].get<double>();
            } else if (error_json.contains("gamma")) {
                error.probability = error_json["gamma"].get<double>();
            } else if (error_json.contains("lambda")) {
                error.probability = error_json["lambda"].get<double>();
            }
            break;
            
        case QiskitErrorType::PAULI:
            if (error_json.contains("probabilities")) {
                error.probabilities = error_json["probabilities"].get<std::vector<double>>();
            }
            break;
            
        case QiskitErrorType::QERROR:
            // For generic qerror, parse probabilities and instructions
            if (error_json.contains("probabilities")) {
                error.probabilities = error_json["probabilities"].get<std::vector<double>>();
            }
            if (error_json.contains("instructions")) {
                // Parse Kraus operators from instructions (simplified)
                // Full implementation would construct matrices from instruction sequences
                // For now, store instruction strings for reference
                for (const auto& inst_list : error_json["instructions"]) {
                    std::vector<std::string> inst_names;
                    for (const auto& inst : inst_list) {
                        if (inst.contains("name")) {
                            inst_names.push_back(inst["name"].get<std::string>());
                        }
                    }
                    error.instructions.push_back(inst_names);
                }
            }
            break;
            
        default:
            break;
    }
    
    return error;
}

NoiseModel NoiseModelImporter::load_from_json_string(const std::string& json_str) {
    json j;
    try {
        j = json::parse(json_str);
    } catch (const json::parse_error& e) {
        throw std::runtime_error("JSON parse error: " + std::string(e.what()));
    }
    
    NoiseModel model;
    
    // Parse metadata
    if (j.contains("device_name")) {
        model.device_name = j["device_name"].get<std::string>();
    }
    if (j.contains("backend_version")) {
        model.backend_version = j["backend_version"].get<std::string>();
    }
    if (j.contains("noise_model_version")) {
        model.noise_model_version = j["noise_model_version"].get<std::string>();
    }
    
    // Parse errors
    if (j.contains("errors")) {
        for (const auto& error_json : j["errors"]) {
            try {
                QiskitNoiseError error = parse_error_from_json(error_json);
                model.errors.push_back(error);
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to parse error: " << e.what() << std::endl;
                // Continue parsing other errors
            }
        }
    }
    
    // Build lookup tables for fast access
    build_lookup_tables(model);
    
    return model;
}

NoiseModel NoiseModelImporter::load_from_json(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open noise model file: " + filepath);
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();
    
    return load_from_json_string(buffer.str());
}

//==============================================================================
// Lookup Table Building
//==============================================================================

void NoiseModelImporter::build_lookup_tables(NoiseModel& model) {
    model.gate_errors.clear();
    model.qubit_errors.clear();
    
    for (auto& error : model.errors) {
        // Index by gate name
        for (const auto& op_name : error.operations) {
            model.gate_errors[op_name].push_back(&error);
        }
        
        // Index by qubit
        for (const auto& qubit_list : error.gate_qubits) {
            for (size_t q : qubit_list) {
                model.qubit_errors[q].push_back(&error);
            }
        }
    }
}

//==============================================================================
// Error Lookup
//==============================================================================

std::vector<const QiskitNoiseError*> NoiseModel::find_applicable_errors(
    const std::string& gate_name,
    const std::vector<size_t>& qubits
) const {
    std::vector<const QiskitNoiseError*> applicable;
    
    // Find errors for this gate type
    auto it = gate_errors.find(gate_name);
    if (it == gate_errors.end()) {
        return applicable;  // No errors for this gate
    }
    
    // Filter by qubit indices
    for (const auto* error : it->second) {
        // Check if error applies to these specific qubits
        bool applies = false;
        for (const auto& gate_qubit_list : error->gate_qubits) {
            if (gate_qubit_list.size() != qubits.size()) {
                continue;
            }
            
            bool match = true;
            for (size_t i = 0; i < qubits.size(); ++i) {
                if (gate_qubit_list[i] != qubits[i]) {
                    match = false;
                    break;
                }
            }
            
            if (match) {
                applies = true;
                break;
            }
        }
        
        if (applies) {
            applicable.push_back(error);
        }
    }
    
    return applicable;
}

//==============================================================================
// Qiskit Error → LRET Conversion
//==============================================================================

std::vector<NoiseOp> NoiseModelImporter::convert_thermal_relaxation(
    double T1,
    double T2,
    double gate_time,
    size_t qubit
) {
    std::vector<NoiseOp> noise_ops;
    
    // Calculate decay probabilities
    double p_reset = 1.0 - std::exp(-gate_time / T1);      // Amplitude damping
    double p_dephase = 0.5 * (1.0 - std::exp(-gate_time / T2));  // Phase damping
    
    // Apply amplitude damping if significant
    if (p_reset > 1e-10) {
        NoiseOp amplitude_noise(NoiseType::AMPLITUDE_DAMPING, qubit, p_reset);
        noise_ops.push_back(amplitude_noise);
    }
    
    // Apply phase damping if significant
    if (p_dephase > 1e-10) {
        NoiseOp phase_noise(NoiseType::PHASE_DAMPING, qubit, p_dephase);
        noise_ops.push_back(phase_noise);
    }
    
    return noise_ops;
}

std::vector<NoiseOp> NoiseModelImporter::convert_pauli_channel(
    const std::vector<double>& probs,
    size_t qubit
) {
    std::vector<NoiseOp> noise_ops;
    
    // Pauli channel: {p_I, p_X, p_Y, p_Z}
    if (probs.size() != 4) {
        std::cerr << "Warning: Pauli channel expects 4 probabilities (I, X, Y, Z), got " 
                  << probs.size() << std::endl;
        return noise_ops;
    }
    
    double p_I = probs[0];
    double p_X = probs[1];
    double p_Y = probs[2];
    double p_Z = probs[3];
    
    // Check if approximately depolarizing (equal X, Y, Z probabilities)
    double avg_pauli = (p_X + p_Y + p_Z) / 3.0;
    double variance = (std::pow(p_X - avg_pauli, 2) + 
                       std::pow(p_Y - avg_pauli, 2) + 
                       std::pow(p_Z - avg_pauli, 2)) / 3.0;
    
    if (variance < 1e-6 && avg_pauli > 1e-10) {
        // Approximately depolarizing
        // depolarizing(p) has equal X, Y, Z with p/3 each
        // So p_total = 3 * p_pauli_avg
        double p_depol = 3.0 * avg_pauli;
        NoiseOp depol_noise(NoiseType::DEPOLARIZING, qubit, p_depol);
        noise_ops.push_back(depol_noise);
    } else {
        // For non-uniform Pauli channels, approximate with depolarizing
        // using the total error probability
        double p_total = 1.0 - p_I;
        if (p_total > 1e-10) {
            NoiseOp depol_noise(NoiseType::DEPOLARIZING, qubit, p_total);
            noise_ops.push_back(depol_noise);
        }
    }
    
    return noise_ops;
}

std::vector<NoiseOp> NoiseModelImporter::convert_qiskit_error_to_lret(
    const QiskitNoiseError& error,
    size_t qubit
) {
    std::vector<NoiseOp> noise_ops;
    
    switch (error.type) {
        case QiskitErrorType::DEPOLARIZING:
            if (error.probability > 1e-10) {
                noise_ops.push_back(NoiseOp(NoiseType::DEPOLARIZING, qubit, error.probability));
            }
            break;
            
        case QiskitErrorType::AMPLITUDE_DAMPING:
            if (error.probability > 1e-10) {
                noise_ops.push_back(NoiseOp(NoiseType::AMPLITUDE_DAMPING, qubit, error.probability));
            }
            break;
            
        case QiskitErrorType::PHASE_DAMPING:
            if (error.probability > 1e-10) {
                noise_ops.push_back(NoiseOp(NoiseType::PHASE_DAMPING, qubit, error.probability));
            }
            break;
            
        case QiskitErrorType::THERMAL_RELAXATION:
            noise_ops = convert_thermal_relaxation(error.T1, error.T2, error.gate_time, qubit);
            break;
            
        case QiskitErrorType::PAULI:
            noise_ops = convert_pauli_channel(error.probabilities, qubit);
            break;
            
        case QiskitErrorType::QERROR:
            // Generic qerror - approximate with depolarizing based on probabilities
            if (!error.probabilities.empty()) {
                // Use 1 - p_identity as error probability
                double p_error = error.probabilities.empty() ? 0.0 : (1.0 - error.probabilities[0]);
                if (p_error > 1e-10) {
                    noise_ops.push_back(NoiseOp(NoiseType::DEPOLARIZING, qubit, p_error));
                }
            }
            break;
            
        default:
            std::cerr << "Warning: Unknown or unsupported error type for qubit " << qubit << std::endl;
            break;
    }
    
    return noise_ops;
}

//==============================================================================
// Circuit Noise Application
//==============================================================================

GateType NoiseModelImporter::qiskit_gate_name_to_lret(const std::string& qiskit_name) const {
    // Convert to lowercase for case-insensitive comparison
    std::string lower_name = qiskit_name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
    
    // Single-qubit gates
    if (lower_name == "id" || lower_name == "i") return GateType::CUSTOM; // Treat identity as no-op/custom
    if (lower_name == "x") return GateType::X;
    if (lower_name == "y") return GateType::Y;
    if (lower_name == "z") return GateType::Z;
    if (lower_name == "h") return GateType::H;
    if (lower_name == "s") return GateType::S;
    if (lower_name == "t") return GateType::T;
    if (lower_name == "rx") return GateType::RX;
    if (lower_name == "ry") return GateType::RY;
    if (lower_name == "rz") return GateType::RZ;
    
    // Two-qubit gates
    if (lower_name == "cx" || lower_name == "cnot") return GateType::CNOT;
    if (lower_name == "cz") return GateType::CZ;
    if (lower_name == "swap") return GateType::SWAP;
    
    // Default to custom for unknown gates
    return GateType::CUSTOM;
}

void NoiseModelImporter::apply_noise_to_gate(
    const GateOp& gate,
    const NoiseModel& noise_model,
    QuantumSequence& noisy_sequence
) {
    // Add the gate itself
    noisy_sequence.operations.push_back(gate);
    
    // Find applicable errors for this gate
    std::string gate_name = gate_type_to_string(gate.type);
    auto applicable_errors = noise_model.find_applicable_errors(gate_name, gate.qubits);
    
    // Apply noise to each qubit involved in the gate
    for (size_t qubit : gate.qubits) {
        for (const auto* error : applicable_errors) {
            // Convert Qiskit error to LRET noise operations
            std::vector<NoiseOp> noise_ops = convert_qiskit_error_to_lret(*error, qubit);
            
            // Add noise operations to sequence
            for (const auto& noise_op : noise_ops) {
                noisy_sequence.operations.push_back(noise_op);
            }
        }
    }
}

QuantumSequence NoiseModelImporter::apply_noise_model(
    const QuantumSequence& clean_circuit,
    const NoiseModel& noise_model
) {
    QuantumSequence noisy_circuit;
    noisy_circuit.depth = clean_circuit.depth;
    
    // Process each operation in the clean circuit
    for (const auto& op : clean_circuit.operations) {
        if (std::holds_alternative<GateOp>(op)) {
            const auto& gate = std::get<GateOp>(op);
            apply_noise_to_gate(gate, noise_model, noisy_circuit);
        } else {
            // Noise operation - pass through unchanged
            noisy_circuit.operations.push_back(op);
        }
    }
    
    return noisy_circuit;
}

//==============================================================================
// Utilities
//==============================================================================

void NoiseModelImporter::print_noise_model_summary(const NoiseModel& model) const {
    std::cout << "=== Noise Model Summary ===" << std::endl;
    if (!model.device_name.empty()) {
        std::cout << "Device: " << model.device_name << std::endl;
    }
    if (!model.backend_version.empty()) {
        std::cout << "Backend Version: " << model.backend_version << std::endl;
    }
    std::cout << "Total Errors: " << model.num_errors() << std::endl;
    
    // Count by type
    std::map<QiskitErrorType, size_t> type_counts;
    for (const auto& error : model.errors) {
        type_counts[error.type]++;
    }
    
    std::cout << "\nError Types:" << std::endl;
    for (const auto& [type, count] : type_counts) {
        std::string type_name;
        switch (type) {
            case QiskitErrorType::DEPOLARIZING: type_name = "Depolarizing"; break;
            case QiskitErrorType::THERMAL_RELAXATION: type_name = "Thermal Relaxation"; break;
            case QiskitErrorType::AMPLITUDE_DAMPING: type_name = "Amplitude Damping"; break;
            case QiskitErrorType::PHASE_DAMPING: type_name = "Phase Damping"; break;
            case QiskitErrorType::PAULI: type_name = "Pauli Channel"; break;
            case QiskitErrorType::QERROR: type_name = "Generic QError"; break;
            default: type_name = "Unknown"; break;
        }
        std::cout << "  " << type_name << ": " << count << std::endl;
    }
    
    std::cout << "\nAffected Gates:" << std::endl;
    for (const auto& [gate_name, errors] : model.gate_errors) {
        std::cout << "  " << gate_name << ": " << errors.size() << " error(s)" << std::endl;
    }
}

bool NoiseModelImporter::validate_noise_model(const NoiseModel& model, bool verbose) const {
    bool valid = true;
    
    for (size_t i = 0; i < model.errors.size(); ++i) {
        const auto& error = model.errors[i];
        
        // Check probability bounds
        if (error.type == QiskitErrorType::DEPOLARIZING || 
            error.type == QiskitErrorType::AMPLITUDE_DAMPING ||
            error.type == QiskitErrorType::PHASE_DAMPING) {
            if (error.probability < 0.0 || error.probability > 1.0) {
                if (verbose) {
                    std::cerr << "Error " << i << ": Invalid probability " << error.probability << std::endl;
                }
                valid = false;
            }
        }
        
        // Check Pauli probabilities sum to 1
        if (error.type == QiskitErrorType::PAULI && !error.probabilities.empty()) {
            double sum = 0.0;
            for (double p : error.probabilities) {
                sum += p;
            }
            if (std::abs(sum - 1.0) > 1e-6) {
                if (verbose) {
                    std::cerr << "Error " << i << ": Pauli probabilities sum to " << sum << " (should be 1.0)" << std::endl;
                }
                valid = false;
            }
        }
        
        // Check thermal relaxation parameters
        if (error.type == QiskitErrorType::THERMAL_RELAXATION) {
            if (error.T1 <= 0.0 || error.T2 <= 0.0) {
                if (verbose) {
                    std::cerr << "Error " << i << ": Invalid T1/T2 values" << std::endl;
                }
                valid = false;
            }
        }
    }
    
    return valid;
}

//==============================================================================
// Convenience Functions
//==============================================================================

NoiseModel load_noise_model(const std::string& filepath) {
    NoiseModelImporter importer;
    return importer.load_from_json(filepath);
}

QuantumSequence apply_noise(
    const QuantumSequence& circuit,
    const NoiseModel& noise_model
) {
    NoiseModelImporter importer;
    return importer.apply_noise_model(circuit, noise_model);
}

QuantumSequence apply_noise_from_file(
    const QuantumSequence& circuit,
    const std::string& noise_json_path
) {
    NoiseModel noise_model = load_noise_model(noise_json_path);
    return apply_noise(circuit, noise_model);
}

} // namespace qlret
