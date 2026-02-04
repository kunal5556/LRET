#include "noise_import.h"
#include "utils.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <set>

using json = nlohmann::json;

namespace qlret {

namespace {

int pauli_index(char c) {
    switch (std::toupper(static_cast<unsigned char>(c))) {
        case 'I': return 0;
        case 'X': return 1;
        case 'Y': return 2;
        case 'Z': return 3;
        default: return -1;
    }
}

} // namespace

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
    if (type_str.find("correlated") != std::string::npos || type_str.find("zz") != std::string::npos) {
        return QiskitErrorType::CORRELATED_PAULI;
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
                QiskitErrorType type = determine_error_type(error_json);
                if (type == QiskitErrorType::CORRELATED_PAULI) {
                    parse_correlated_error(error_json, model);
                    continue;
                }

                QiskitNoiseError error = parse_error_from_json(error_json);
                model.errors.push_back(error);
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to parse error: " << e.what() << std::endl;
                // Continue parsing other errors
            }
        }
    }

    // Parse advanced noise sections (Phase 4.3)
    if (j.contains("time_dependent_noise")) {
        parse_time_dependent_noise(j, model);
    }
    if (j.contains("memory_effects")) {
        parse_memory_effects(j, model);
    }
    if (j.contains("leakage")) {
        parse_leakage(j, model);
    }
    if (j.contains("measurement_confusion")) {
        parse_measurement_confusion(j, model);
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
// Advanced Noise Parsing (Phase 4.3)
//==============================================================================

void NoiseModelImporter::parse_correlated_error(const json& error_json, NoiseModel& model) {
    // Gather qubit pairs
    std::vector<std::pair<size_t, size_t>> pairs;
    auto parse_pair_list = [&](const json& arr) {
        for (const auto& entry : arr) {
            if (!entry.is_array() || entry.size() != 2) continue;
            size_t q0 = entry[0].get<size_t>();
            size_t q1 = entry[1].get<size_t>();
            size_t qmin = std::min(q0, q1);
            size_t qmax = std::max(q0, q1);
            pairs.emplace_back(qmin, qmax);
        }
    };

    if (error_json.contains("qubit_pairs")) {
        parse_pair_list(error_json["qubit_pairs"]);
    } else if (error_json.contains("gate_qubits")) {
        parse_pair_list(error_json["gate_qubits"]);
    }
    if (pairs.empty()) {
        std::cerr << "Warning: correlated_pauli error missing qubit_pairs" << std::endl;
        return;
    }

    CorrelatedError corr;
    if (error_json.contains("operations")) {
        corr.applicable_gates = error_json["operations"].get<std::vector<std::string>>();
    }

    if (error_json.contains("parameters") && error_json["parameters"].is_object()) {
        const auto& params = error_json["parameters"];
        if (params.contains("zz_rate_hz")) {
            corr.coupling_strength_hz = params["zz_rate_hz"].get<double>();
        }
    }

    // Default joint probabilities: identity channel
    for (auto& row : corr.joint_probs) {
        row.fill(0.0);
    }
    corr.joint_probs[0][0] = 1.0;

    if (error_json.contains("joint_probabilities")) {
        const auto& probs_obj = error_json["joint_probabilities"];
        if (probs_obj.is_object()) {
            corr.joint_probs[0][0] = 0.0; // reset to parse actual values
            for (auto it = probs_obj.begin(); it != probs_obj.end(); ++it) {
                const std::string key = it.key();
                if (key.size() != 2) continue;
                int p0 = pauli_index(key[0]);
                int p1 = pauli_index(key[1]);
                if (p0 < 0 || p1 < 0) continue;
                double val = it.value().get<double>();
                corr.joint_probs[static_cast<size_t>(p0)][static_cast<size_t>(p1)] = val;
                if (val > 0.0) {
                    corr.sparse_probs.emplace_back(p0, p1, val);
                }
            }
        }
    }

    // If sparse_probs empty, populate from dense table for convenience
    if (corr.sparse_probs.empty()) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                double val = corr.joint_probs[static_cast<size_t>(i)][static_cast<size_t>(j)];
                if (val > 0.0) {
                    corr.sparse_probs.emplace_back(i, j, val);
                }
            }
        }
    }

    // Register into model
    for (const auto& [qmin, qmax] : pairs) {
        model.correlated_errors[{qmin, qmax}].push_back(corr);
        // Neighbor adjacency (deduplicated)
        auto& neigh_i = model.qubit_neighbors[qmin];
        auto& neigh_j = model.qubit_neighbors[qmax];
        if (std::find(neigh_i.begin(), neigh_i.end(), qmax) == neigh_i.end()) {
            neigh_i.push_back(qmax);
        }
        if (std::find(neigh_j.begin(), neigh_j.end(), qmin) == neigh_j.end()) {
            neigh_j.push_back(qmin);
        }
    }
}

void NoiseModelImporter::parse_time_dependent_noise(const json& root, NoiseModel& model) {
    const auto& section = root.at("time_dependent_noise");
    bool enabled = section.value("enabled", false);
    model.use_time_dependent_noise = enabled;
    if (!enabled) return;

    if (!section.contains("gate_parameters")) {
        return;
    }

    for (const auto& item : section["gate_parameters"].items()) {
        const std::string gate_name = item.key();
        const auto& params_json = item.value();

        TimeVaryingNoiseParams params;
        params.base_t1_ns = params_json.value("base_t1_ns", 0.0);
        params.base_t2_ns = params_json.value("base_t2_ns", 0.0);
        params.base_depol_prob = params_json.value("base_depol_prob", 0.0);
        params.alpha = params_json.value("alpha", 0.0);
        params.beta = params_json.value("beta", 0.0);
        params.max_depth = std::max<size_t>(1, params_json.value("max_depth", 1));

        std::string model_str = params_json.value("scaling_model", std::string("linear"));
        std::transform(model_str.begin(), model_str.end(), model_str.begin(), ::tolower);
        if (model_str == "linear") {
            params.model = TimeScalingModel::LINEAR;
        } else if (model_str == "exponential") {
            params.model = TimeScalingModel::EXPONENTIAL;
        } else if (model_str == "polynomial") {
            params.model = TimeScalingModel::POLYNOMIAL;
        }

        model.time_varying_params[gate_name] = params;
    }
}

void NoiseModelImporter::parse_memory_effects(const json& root, NoiseModel& model) {
    const auto& section = root.at("memory_effects");
    bool enabled = section.value("enabled", false);
    model.use_memory_effects = enabled;
    if (!enabled) return;

    model.max_memory_depth = section.value("max_memory_depth", static_cast<size_t>(model.max_memory_depth));

    if (!section.contains("rules")) {
        return;
    }

    for (const auto& rule : section["rules"]) {
        if (!rule.contains("trigger") || !rule.contains("effect")) {
            continue;
        }

        MemoryEffect effect;
        const auto& trigger = rule["trigger"];
        const auto& eff = rule["effect"];

        if (trigger.contains("gate_type")) {
            effect.prev_gate_type = trigger["gate_type"].get<std::string>();
        }
        if (trigger.contains("qubits") && trigger["qubits"].is_array()) {
            for (const auto& q : trigger["qubits"]) {
                effect.prev_qubits.push_back(q.get<size_t>());
            }
        }

        if (eff.contains("affected_gate")) {
            effect.affected_gate_type = eff["affected_gate"].get<std::string>();
        } else {
            effect.affected_gate_type = "any";
        }
        if (eff.contains("affected_qubits") && eff["affected_qubits"].is_array()) {
            for (const auto& q : eff["affected_qubits"]) {
                effect.affected_qubits.push_back(q.get<size_t>());
            }
        }
        effect.error_scale_factor = eff.value("error_scale", 1.0);

        if (rule.contains("memory_depth")) {
            effect.memory_depth = rule["memory_depth"].get<size_t>();
        } else {
            effect.memory_depth = 1;
        }

        model.memory_effects.push_back(effect);
    }
}

void NoiseModelImporter::parse_leakage(const json& root, NoiseModel& model) {
    if (!root.contains("leakage")) {
        return;
    }

    const auto& section = root.at("leakage");
    model.use_leakage = section.value("enabled", true);
    if (!model.use_leakage) {
        return;
    }

    // Default parameters
    if (section.contains("default")) {
        const auto& d = section["default"];
        model.default_leakage.p_leak = d.value("p_leak", model.default_leakage.p_leak);
        model.default_leakage.p_relax = d.value("p_relax", model.default_leakage.p_relax);
        model.default_leakage.p_phase = d.value("p_phase", model.default_leakage.p_phase);
        model.default_leakage.use_qutrit = d.value("use_qutrit", model.default_leakage.use_qutrit);
    }

    // Per-qubit overrides
    if (section.contains("qubits") && section["qubits"].is_array()) {
        for (const auto& entry : section["qubits"]) {
            if (!entry.contains("id")) continue;
            size_t q = entry["id"].get<size_t>();
            LeakageChannel ch = model.default_leakage;
            ch.p_leak = entry.value("p_leak", ch.p_leak);
            ch.p_relax = entry.value("p_relax", ch.p_relax);
            ch.p_phase = entry.value("p_phase", ch.p_phase);
            ch.use_qutrit = entry.value("use_qutrit", ch.use_qutrit);
            model.leakage_channels[q] = ch;
        }
    } else {
        // If no qubit array, allow top-level params as defaults
        model.default_leakage.p_leak = section.value("p_leak", model.default_leakage.p_leak);
        model.default_leakage.p_relax = section.value("p_relax", model.default_leakage.p_relax);
        model.default_leakage.p_phase = section.value("p_phase", model.default_leakage.p_phase);
        model.default_leakage.use_qutrit = section.value("use_qutrit", model.default_leakage.use_qutrit);
    }
}

void NoiseModelImporter::parse_measurement_confusion(const json& root, NoiseModel& model) {
    if (!root.contains("measurement_confusion")) {
        return;
    }

    const auto& section = root.at("measurement_confusion");
    model.use_measurement_confusion = section.value("enabled", true);
    if (!model.use_measurement_confusion) {
        return;
    }

    if (!section.contains("entries")) {
        return;  // nothing to do
    }

    for (const auto& entry : section["entries"]) {
        if (!entry.contains("qubit")) continue;
        size_t q = entry["qubit"].get<size_t>();
        bool conditional = entry.value("conditional", false);

        if (!entry.contains("matrix") || !entry["matrix"].is_array()) continue;
        const auto& mat_json = entry["matrix"];
        size_t rows = mat_json.size();
        if (rows == 0) continue;
        size_t cols = mat_json[0].size();
        MatrixXd confusion = MatrixXd::Zero(rows, cols);
        for (size_t r = 0; r < rows; ++r) {
            const auto& row = mat_json[r];
            for (size_t c = 0; c < row.size(); ++c) {
                confusion(static_cast<int>(r), static_cast<int>(c)) = row[c].get<double>();
            }
        }

        MeasurementSpec spec;
        spec.qubit = q;
        spec.confusion = confusion;
        spec.conditional = conditional;
        model.measurement_specs[q] = spec;
    }
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
    QuantumSequence& noisy_sequence,
    CircuitMemoryState& memory_state
) {
    // Add the gate itself
    noisy_sequence.operations.push_back(gate);
    
    // Find applicable errors for this gate
    std::string gate_name = gate_type_to_string(gate.type);
    auto applicable_errors = noise_model.find_applicable_errors(gate_name, gate.qubits);

    // Time-dependent scaling for this gate (if configured)
    double time_scale = 1.0;
    auto tv_it = noise_model.time_varying_params.find(gate_name);
    if (noise_model.use_time_dependent_noise && tv_it != noise_model.time_varying_params.end()) {
        time_scale = compute_time_scaled_rate(1.0, memory_state.current_depth, tv_it->second);
    }

    // Memory-dependent scaling (non-Markovian)
    double memory_scale = 1.0;
    if (noise_model.use_memory_effects) {
        memory_scale = evaluate_memory_scale(memory_state, gate_name, gate.qubits, noise_model.memory_effects);
    }
    double total_scale = time_scale * memory_scale;
    
    // Apply noise to each qubit involved in the gate
    for (size_t qubit : gate.qubits) {
        for (const auto* error : applicable_errors) {
            QiskitNoiseError error_scaled = *error;

            // Apply time scaling overrides for base parameters when available
            if (noise_model.use_time_dependent_noise && tv_it != noise_model.time_varying_params.end()) {
                const auto& params = tv_it->second;
                switch (error_scaled.type) {
                    case QiskitErrorType::THERMAL_RELAXATION: {
                        if (params.base_t1_ns > 0.0) {
                            error_scaled.T1 = params.base_t1_ns * 1e-9; // ns -> s
                        }
                        if (params.base_t2_ns > 0.0) {
                            error_scaled.T2 = params.base_t2_ns * 1e-9;
                        }
                        error_scaled.gate_time = error_scaled.gate_time * time_scale;
                        break;
                    }
                    case QiskitErrorType::DEPOLARIZING:
                    case QiskitErrorType::AMPLITUDE_DAMPING:
                    case QiskitErrorType::PHASE_DAMPING:
                        if (params.base_depol_prob > 0.0) {
                            error_scaled.probability = compute_time_scaled_rate(params.base_depol_prob, memory_state.current_depth, params);
                        } else {
                            error_scaled.probability = std::clamp(error_scaled.probability * total_scale, 0.0, 1.0);
                        }
                        break;
                    case QiskitErrorType::PAULI:
                        if (!error_scaled.probabilities.empty() && error_scaled.probabilities.size() == 4) {
                            double non_id_sum = 0.0;
                            for (size_t idx = 1; idx < 4; ++idx) {
                                error_scaled.probabilities[idx] = std::clamp(error_scaled.probabilities[idx] * total_scale, 0.0, 1.0);
                                non_id_sum += error_scaled.probabilities[idx];
                            }
                            error_scaled.probabilities[0] = std::max(0.0, 1.0 - non_id_sum);
                        }
                        break;
                    default:
                        break;
                }
            }
            
            // Convert Qiskit error to LRET noise operations
            std::vector<NoiseOp> noise_ops = convert_qiskit_error_to_lret(error_scaled, qubit);
            if (total_scale != 1.0) {
                for (auto& op_variant : noise_ops) {
                    op_variant.probability = std::clamp(op_variant.probability * total_scale, 0.0, 1.0);
                }
            }
            
            // Add noise operations to sequence
            for (const auto& noise_op : noise_ops) {
                noisy_sequence.operations.push_back(noise_op);
            }
        }
    }

    // Correlated errors (two-qubit Pauli channels)
    if (gate.qubits.size() == 2 && !noise_model.correlated_errors.empty()) {
        size_t q0 = gate.qubits[0];
        size_t q1 = gate.qubits[1];
        size_t qmin = std::min(q0, q1);
        size_t qmax = std::max(q0, q1);
        auto corr_it = noise_model.correlated_errors.find({qmin, qmax});
        if (corr_it != noise_model.correlated_errors.end()) {
            std::string gate_name_lower = gate_name;
            std::transform(gate_name_lower.begin(), gate_name_lower.end(), gate_name_lower.begin(), ::tolower);
            for (const auto& corr : corr_it->second) {
                bool gate_match = corr.applicable_gates.empty();
                if (!gate_match) {
                    for (const auto& op : corr.applicable_gates) {
                        std::string op_lower = op;
                        std::transform(op_lower.begin(), op_lower.end(), op_lower.begin(), ::tolower);
                        if (op_lower == gate_name_lower) {
                            gate_match = true;
                            break;
                        }
                    }
                }
                if (!gate_match) continue;

                std::vector<double> flattened(16, 0.0);
                double p_error_total = 0.0;
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        double p = corr.joint_probs[static_cast<size_t>(i)][static_cast<size_t>(j)];
                        flattened[static_cast<size_t>(i * 4 + j)] = p;
                        if (!(i == 0 && j == 0)) {
                            p_error_total += p;
                        }
                    }
                }

                if (total_scale != 1.0) {
                    double non_id_sum = 0.0;
                    for (size_t idx = 1; idx < flattened.size(); ++idx) {
                        flattened[idx] = std::clamp(flattened[idx] * total_scale, 0.0, 1.0);
                        non_id_sum += flattened[idx];
                    }
                    flattened[0] = std::max(0.0, 1.0 - non_id_sum);
                    p_error_total = non_id_sum;
                }

                NoiseOp corr_noise(NoiseType::CORRELATED_PAULI, std::vector<size_t>{qmin, qmax}, p_error_total, flattened);
                noisy_sequence.operations.push_back(corr_noise);
            }
        }
    }

    // Leakage channels (Phase 4.4) - inject after each gate on involved qubits
    if (noise_model.use_leakage) {
        for (size_t qubit : gate.qubits) {
            LeakageChannel ch = noise_model.default_leakage;
            auto lk_it = noise_model.leakage_channels.find(qubit);
            if (lk_it != noise_model.leakage_channels.end()) {
                ch = lk_it->second;
            }
            
            // Apply leakage noise if p_leak > 0
            if (ch.p_leak > 1e-12) {
                double scaled_leak = std::clamp(ch.p_leak * total_scale, 0.0, 1.0);
                NoiseOp leak_noise(NoiseType::LEAKAGE, qubit, scaled_leak);
                noisy_sequence.operations.push_back(leak_noise);
            }
            
            // Apply relaxation noise if p_relax > 0
            if (ch.p_relax > 1e-12) {
                double scaled_relax = std::clamp(ch.p_relax * total_scale, 0.0, 1.0);
                NoiseOp relax_noise(NoiseType::LEAKAGE_RELAXATION, qubit, scaled_relax);
                noisy_sequence.operations.push_back(relax_noise);
            }
            
            // Apply phase noise on leaked population
            if (ch.p_phase > 1e-12) {
                double scaled_phase = std::clamp(ch.p_phase * total_scale, 0.0, 1.0);
                NoiseOp phase_noise(NoiseType::PHASE_DAMPING, qubit, scaled_phase);
                noisy_sequence.operations.push_back(phase_noise);
            }
        }
    }

    // Update memory history after processing this gate
    append_memory_history(memory_state, gate_name, gate.qubits);
}

QuantumSequence NoiseModelImporter::apply_noise_model(
    const QuantumSequence& clean_circuit,
    const NoiseModel& noise_model
) {
    QuantumSequence noisy_circuit;
    noisy_circuit.depth = clean_circuit.depth;
    noisy_circuit.num_qubits = clean_circuit.num_qubits;
    CircuitMemoryState memory_state;
    memory_state.max_memory_depth = noise_model.max_memory_depth;
    memory_state.current_depth = 0;
    
    // Process each operation in the clean circuit
    for (const auto& op : clean_circuit.operations) {
        if (std::holds_alternative<GateOp>(op)) {
            const auto& gate = std::get<GateOp>(op);
            memory_state.current_depth++;
            apply_noise_to_gate(gate, noise_model, noisy_circuit, memory_state);
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

    if (!model.correlated_errors.empty()) {
        size_t corr_count = 0;
        for (const auto& entry : model.correlated_errors) {
            corr_count += entry.second.size();
        }
        std::cout << "\nCorrelated Errors: " << corr_count << " pair entries" << std::endl;
    }
    if (model.use_time_dependent_noise) {
        std::cout << "Time-Dependent Noise: enabled (" << model.time_varying_params.size()
                  << " gate parameter sets)" << std::endl;
    }
    if (model.use_memory_effects) {
        std::cout << "Memory Effects: enabled (" << model.memory_effects.size() << " rules, depth="
                  << model.max_memory_depth << ")" << std::endl;
    }
    if (model.use_leakage) {
        std::cout << "Leakage Channels: enabled (default p_leak=" << model.default_leakage.p_leak
                  << ", p_relax=" << model.default_leakage.p_relax << ")" << std::endl;
        if (!model.leakage_channels.empty()) {
            std::cout << "  Per-qubit overrides: " << model.leakage_channels.size() << " qubits" << std::endl;
        }
    }
    if (model.use_measurement_confusion) {
        std::cout << "Measurement Confusion: enabled (" << model.measurement_specs.size() 
                  << " qubit specs)" << std::endl;
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

    // Correlated probabilities sanity check
    for (const auto& [pair_key, entries] : model.correlated_errors) {
        for (const auto& corr : entries) {
            double sum = 0.0;
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    sum += corr.joint_probs[static_cast<size_t>(i)][static_cast<size_t>(j)];
                }
            }
            if (std::abs(sum - 1.0) > 1e-3) {
                valid = false;
                if (verbose) {
                    std::cerr << "Correlated error (" << pair_key.first << "," << pair_key.second
                              << "): joint probabilities sum to " << sum << std::endl;
                }
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
