#include "json_interface.h"
#include "simulator.h"
#include "gates_and_noise.h"
#include "utils.h"

#include <chrono>
#include <cctype>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace qlret {
namespace {

//------------------------------------------------------------------------------
// Helpers
//------------------------------------------------------------------------------

std::string to_upper_copy(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

GateType gate_type_from_string(const std::string& name) {
    static const std::unordered_map<std::string, GateType> kGateMap = {
        {"H", GateType::H},
        {"X", GateType::X},
        {"Y", GateType::Y},
        {"Z", GateType::Z},
        {"S", GateType::S},
        {"T", GateType::T},
        {"SDG", GateType::Sdg},
        {"TDG", GateType::Tdg},
        {"SX", GateType::SX},
        {"RX", GateType::RX},
        {"RY", GateType::RY},
        {"RZ", GateType::RZ},
        {"U1", GateType::U1},
        {"U2", GateType::U2},
        {"U3", GateType::U3},
        {"CNOT", GateType::CNOT},
        {"CX", GateType::CNOT},
        {"CZ", GateType::CZ},
        {"CY", GateType::CY},
        {"SWAP", GateType::SWAP},
        {"ISWAP", GateType::ISWAP},
    };

    auto upper = to_upper_copy(name);
    auto it = kGateMap.find(upper);
    if (it == kGateMap.end()) {
        throw std::invalid_argument("Unsupported gate: " + name);
    }
    return it->second;
}

Matrix2cd pauli_from_char(char c) {
    switch (c) {
        case 'X': {
            Matrix2cd m; m << 0, 1, 1, 0; return m;
        }
        case 'Y': {
            Matrix2cd m; m << 0, Complex(0, -1), Complex(0, 1), 0; return m;
        }
        case 'Z': {
            Matrix2cd m; m << 1, 0, 0, -1; return m;
        }
        case 'I': {
            Matrix2cd m; m << 1, 0, 0, 1; return m;
        }
        default:
            throw std::invalid_argument("Invalid Pauli operator: " + std::string(1, c));
    }
}

MatrixXcd kron(const MatrixXcd& a, const MatrixXcd& b) {
    MatrixXcd result(a.rows() * b.rows(), a.cols() * b.cols());
    for (int i = 0; i < a.rows(); ++i) {
        for (int j = 0; j < a.cols(); ++j) {
            result.block(i * b.rows(), j * b.cols(), b.rows(), b.cols()) = a(i, j) * b;
        }
    }
    return result;
}

MatrixXcd build_observable_matrix(const JsonObservable& obs, size_t num_qubits) {
    const size_t dim = 1ULL << num_qubits;

    if (obs.type == JsonObservableType::HERMITIAN) {
        if (obs.matrix.rows() != static_cast<int>(dim) || obs.matrix.cols() != static_cast<int>(dim)) {
            throw std::invalid_argument("Hermitian observable dimension mismatch for num_qubits");
        }
        return obs.coefficient * obs.matrix;
    }

    // Build per-qubit operator list (default identity)
    std::vector<Matrix2cd> per_qubit(num_qubits, pauli_from_char('I'));
    for (size_t idx = 0; idx < obs.wires.size(); ++idx) {
        size_t wire = obs.wires[idx];
        if (wire >= num_qubits) {
            throw std::invalid_argument("Observable wire index out of range");
        }
        char p = obs.operators[idx].empty() ? 'I' : static_cast<char>(std::toupper(static_cast<unsigned char>(obs.operators[idx][0])));
        per_qubit[wire] = pauli_from_char(p);
    }

    // Build tensor product with qubit 0 as LSB convention:
    // Observable matrix is per_qubit[n-1] ⊗ ... ⊗ per_qubit[1] ⊗ per_qubit[0]
    // This matches state vector ordering where index = sum(bit_q * 2^q)
    MatrixXcd result = per_qubit[num_qubits - 1];
    for (int q = static_cast<int>(num_qubits) - 2; q >= 0; --q) {
        result = kron(result, per_qubit[q]);
    }
    return obs.coefficient * result;
}

double compute_expectation(const MatrixXcd& L, const JsonObservable& obs, size_t num_qubits) {
    MatrixXcd obs_matrix = build_observable_matrix(obs, num_qubits);
    MatrixXcd middle = obs_matrix * L;
    Complex tr = (L.adjoint() * middle).trace();
    return tr.real();
}

std::vector<int> sample_measurements(const MatrixXcd& L, size_t shots, size_t num_qubits) {
    const size_t dim = 1ULL << num_qubits;
    MatrixXcd rho = L * L.adjoint();
    VectorXd probs(dim);
    for (size_t i = 0; i < dim; ++i) {
        probs(static_cast<Eigen::Index>(i)) = std::max(0.0, rho(i, i).real());
    }
    // Normalize in case of numerical drift
    double total = probs.sum();
    if (total <= 0) {
        throw std::runtime_error("Invalid probability distribution for sampling");
    }
    probs /= total;

    std::mt19937 rng(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    std::discrete_distribution<size_t> dist(probs.data(), probs.data() + probs.size());

    std::vector<int> samples;
    samples.reserve(shots);
    for (size_t s = 0; s < shots; ++s) {
        samples.push_back(static_cast<int>(dist(rng)));
    }
    return samples;
}

std::vector<double> flatten_real(const MatrixXcd& m) {
    std::vector<double> out;
    out.reserve(static_cast<size_t>(m.size()));
    for (int r = 0; r < m.rows(); ++r) {
        for (int c = 0; c < m.cols(); ++c) {
            out.push_back(m(r, c).real());
        }
    }
    return out;
}

std::vector<double> flatten_imag(const MatrixXcd& m) {
    std::vector<double> out;
    out.reserve(static_cast<size_t>(m.size()));
    for (int r = 0; r < m.rows(); ++r) {
        for (int c = 0; c < m.cols(); ++c) {
            out.push_back(m(r, c).imag());
        }
    }
    return out;
}

} // namespace

//------------------------------------------------------------------------------
// Parsing
//------------------------------------------------------------------------------

JsonCircuitSpec parse_circuit_json(const std::string& json_text) {
    auto j = nlohmann::json::parse(json_text);
    return parse_circuit_json(j);
}

JsonCircuitSpec parse_circuit_json_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("Could not open JSON file: " + path);
    }
    std::stringstream buffer;
    buffer << f.rdbuf();
    return parse_circuit_json(buffer.str());
}

JsonCircuitSpec parse_circuit_json(const nlohmann::json& j) {
    JsonCircuitSpec spec;

    if (!j.contains("circuit")) {
        throw std::invalid_argument("JSON missing 'circuit' field");
    }
    const auto& jcirc = j.at("circuit");
    spec.num_qubits = jcirc.at("num_qubits").get<size_t>();

    // Operations
    if (jcirc.contains("operations")) {
        for (const auto& op_json : jcirc.at("operations")) {
            JsonOperation op;
            op.name = op_json.at("name").get<std::string>();
            op.wires = op_json.at("wires").get<std::vector<size_t>>();
            if (op_json.contains("params")) {
                op.params = op_json.at("params").get<std::vector<double>>();
            }
            spec.operations.push_back(std::move(op));
        }
    }

    // Observables
    if (jcirc.contains("observables")) {
        for (const auto& obs_json : jcirc.at("observables")) {
            JsonObservable obs;
            std::string type = obs_json.at("type").get<std::string>();
            std::string type_upper = to_upper_copy(type);

            if (type_upper == "PAULI") {
                obs.type = JsonObservableType::PAULI;
                obs.operators = {obs_json.at("operator").get<std::string>()};
                obs.wires = obs_json.at("wires").get<std::vector<size_t>>();
                obs.coefficient = obs_json.value("coefficient", 1.0);
            } else if (type_upper == "TENSOR") {
                obs.type = JsonObservableType::TENSOR;
                obs.operators = obs_json.at("operators").get<std::vector<std::string>>();
                obs.wires = obs_json.at("wires").get<std::vector<size_t>>();
                obs.coefficient = obs_json.value("coefficient", 1.0);
            } else if (type_upper == "HERMITIAN") {
                obs.type = JsonObservableType::HERMITIAN;
                obs.wires = obs_json.at("wires").get<std::vector<size_t>>();
                obs.coefficient = obs_json.value("coefficient", 1.0);

                auto real = obs_json.at("matrix_real").get<std::vector<std::vector<double>>>();
                auto imag = obs_json.at("matrix_imag").get<std::vector<std::vector<double>>>();
                size_t rows = real.size();
                size_t cols = rows ? real[0].size() : 0;
                obs.matrix = MatrixXcd(rows, cols);
                for (size_t r = 0; r < rows; ++r) {
                    if (real[r].size() != cols || imag[r].size() != cols) {
                        throw std::invalid_argument("Hermitian matrix shape mismatch");
                    }
                    for (size_t c = 0; c < cols; ++c) {
                        obs.matrix(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) =
                            Complex(real[r][c], imag[r][c]);
                    }
                }
            } else {
                throw std::invalid_argument("Unsupported observable type: " + type);
            }

            spec.observables.push_back(std::move(obs));
        }
    }

    // Config
    if (j.contains("config")) {
        const auto& cfg = j.at("config");
        spec.config.truncation_threshold = cfg.value("epsilon", 1e-4);
        spec.initial_rank = cfg.value("initial_rank", static_cast<size_t>(1));
        spec.export_state = cfg.value("export_state", false);
        if (cfg.contains("shots") && !cfg["shots"].is_null()) {
            spec.shots = cfg.at("shots").get<size_t>();
        }
    }

    return spec;
}

//------------------------------------------------------------------------------
// Execution
//------------------------------------------------------------------------------

QuantumSequence build_sequence(const JsonCircuitSpec& spec) {
    QuantumSequence seq(spec.num_qubits);
    seq.depth = spec.operations.size();

    for (const auto& op : spec.operations) {
        GateType gtype = gate_type_from_string(op.name);
        if (op.wires.empty()) {
            throw std::invalid_argument("Operation missing wires: " + op.name);
        }
        if (gtype == GateType::CNOT || gtype == GateType::CZ || gtype == GateType::CY || gtype == GateType::SWAP || gtype == GateType::ISWAP) {
            if (op.wires.size() != 2) {
                throw std::invalid_argument("Two-qubit gate requires two wires: " + op.name);
            }
            seq.add_gate(GateOp(gtype, op.wires[0], op.wires[1]));
        } else {
            seq.add_gate(GateOp(gtype, op.wires, op.params));
        }
    }

    return seq;
}

JsonRunResult run_json_circuit(const JsonCircuitSpec& spec) {
    if (spec.num_qubits == 0) {
        throw std::invalid_argument("num_qubits must be > 0");
    }

    QuantumSequence seq = build_sequence(spec);

    // Initial state
    MatrixXcd L_init;
    if (spec.initial_rank > 1) {
        L_init = create_random_mixed_state(spec.num_qubits, spec.initial_rank, 0);
    } else {
        L_init = create_zero_state(spec.num_qubits);
    }

    SimConfig cfg = spec.config;

    auto start = std::chrono::high_resolution_clock::now();
    SimResult sim_res = run_simulation(L_init, seq, spec.num_qubits, cfg);
    auto end = std::chrono::high_resolution_clock::now();

    JsonRunResult result;
    result.L_final = sim_res.L_final;
    result.execution_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Expectations
    for (const auto& obs : spec.observables) {
        result.expectations.push_back(compute_expectation(result.L_final, obs, spec.num_qubits));
    }

    // Optional sampling
    if (spec.shots.has_value() && spec.shots.value() > 0) {
        result.samples = sample_measurements(result.L_final, spec.shots.value(), spec.num_qubits);
    }

    return result;
}

//------------------------------------------------------------------------------
// Export
//------------------------------------------------------------------------------

std::string export_result_json(
    const JsonRunResult& result,
    bool include_state,
    int indent
) {
    nlohmann::json j;
    j["status"] = "success";
    j["execution_time_ms"] = result.execution_time_ms;
    j["final_rank"] = result.L_final.cols();
    j["expectation_values"] = result.expectations;

    if (!result.samples.empty()) {
        j["samples"] = result.samples;
    } else {
        j["samples"] = nullptr;
    }

    if (include_state) {
        j["state"]["type"] = "low_rank";
        j["state"]["rows"] = result.L_final.rows();
        j["state"]["cols"] = result.L_final.cols();
        j["state"]["L_real"] = flatten_real(result.L_final);
        j["state"]["L_imag"] = flatten_imag(result.L_final);
    }

    return j.dump(indent);
}

}  // namespace qlret
