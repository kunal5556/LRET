/**
 * QLRET Python Bindings (pybind11)
 * Phase 5b: Native Python bindings for JSON-based circuit execution
 *
 * Build with USE_PYTHON=ON in CMake:
 *   cmake -DUSE_PYTHON=ON ..
 *   cmake --build .
 *
 * The resulting _qlret_native module exposes:
 *   - run_circuit_json(json_str, export_state) -> result_json_str
 */

#ifdef USE_PYTHON

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "json_interface.h"
#include "autodiff.h"
#include "gates_and_noise.h"
#include <string>
#include <stdexcept>
#include <algorithm>

namespace py = pybind11;

namespace qlret {

//------------------------------------------------------------------------------
// Helpers for autodiff bindings
//------------------------------------------------------------------------------

static ObservableType parse_observable_type(const std::string& name) {
    std::string key = name;
    std::transform(key.begin(), key.end(), key.begin(), ::toupper);
    if (key == "PAULIZ" || key == "Z") return ObservableType::PauliZ;
    if (key == "PAULIX" || key == "X") return ObservableType::PauliX;
    if (key == "PAULIY" || key == "Y") return ObservableType::PauliY;
    throw std::invalid_argument("Unsupported observable type: " + name);
}

static Observable parse_observable(const py::dict& obs_dict) {
    Observable obs;
    obs.coefficient = obs_dict.contains("coefficient")
        ? obs_dict["coefficient"].cast<double>()
        : 1.0;

    if (obs_dict.contains("terms")) {
        auto terms = obs_dict["terms"].cast<py::list>();
        for (const auto& term_obj : terms) {
            auto term = term_obj.cast<py::dict>();
            if (!term.contains("type") || !term.contains("qubit")) {
                throw std::invalid_argument("Observable term requires type and qubit");
            }
            auto t = parse_observable_type(term["type"].cast<std::string>());
            size_t q = term["qubit"].cast<size_t>();
            obs.terms.push_back({t, q});
        }
        return obs;
    }

    if (!obs_dict.contains("type") || !obs_dict.contains("qubit")) {
        throw std::invalid_argument("Observable requires type and qubit");
    }
    obs.type = parse_observable_type(obs_dict["type"].cast<std::string>());
    obs.qubit = obs_dict["qubit"].cast<size_t>();
    return obs;
}

static GateType parse_gate_type(const std::string& name) {
    auto it = gate_name_to_type.find(name);
    if (it != gate_name_to_type.end()) return it->second;

    // Try uppercase lookup
    std::string upper = name;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
    it = gate_name_to_type.find(upper);
    if (it != gate_name_to_type.end()) return it->second;

    throw std::invalid_argument("Unsupported gate name: " + name);
}

static void build_sequence_from_ops(size_t num_qubits,
                                    const py::list& ops,
                                    QuantumSequence& seq,
                                    std::vector<int>& param_indices) {
    seq = QuantumSequence(num_qubits);
    param_indices.clear();
    param_indices.reserve(py::len(ops));

    for (const auto& obj : ops) {
        auto op = obj.cast<py::dict>();
        if (!op.contains("name") || !op.contains("qubits")) {
            throw std::invalid_argument("Each operation requires name and qubits");
        }

        auto gate_type = parse_gate_type(op["name"].cast<std::string>());
        auto qubits = op["qubits"].cast<std::vector<size_t>>();

        std::vector<double> params;
        if (op.contains("params")) {
            params = op["params"].cast<std::vector<double>>();
        }

        int param_idx = -1;
        if (op.contains("param_idx")) {
            param_idx = op["param_idx"].cast<int>();
        }

        GateOp gate(gate_type, qubits, params);
        seq.operations.push_back(gate);
        param_indices.push_back(param_idx);
    }
}

static double autodiff_expectation_internal(size_t num_qubits,
                                            const py::list& ops,
                                            const std::vector<double>& params,
                                            const py::dict& observable) {
    QuantumSequence seq;
    std::vector<int> param_indices;
    build_sequence_from_ops(num_qubits, ops, seq, param_indices);

    AutoDiffCircuit circuit(num_qubits, seq, param_indices);
    auto obs = parse_observable(observable);
    return circuit.forward(params, obs);
}

static std::vector<double> autodiff_gradients_internal(size_t num_qubits,
                                                       const py::list& ops,
                                                       const std::vector<double>& params,
                                                       const py::dict& observable) {
    QuantumSequence seq;
    std::vector<int> param_indices;
    build_sequence_from_ops(num_qubits, ops, seq, param_indices);

    AutoDiffCircuit circuit(num_qubits, seq, param_indices);
    auto obs = parse_observable(observable);
    return circuit.backward(params, obs);
}

/**
 * Run a circuit from JSON string and return result as JSON string.
 *
 * @param json_str  JSON circuit specification
 * @param export_state  If true, include low-rank state in output
 * @return JSON result string
 */
std::string run_circuit_json(const std::string& json_str, bool export_state) {
    try {
        JsonCircuitSpec spec = parse_circuit_json(json_str);
        
        // Override export_state from Python if requested
        if (export_state) {
            spec.export_state = true;
        }
        
        JsonRunResult result = run_json_circuit(spec);
        return export_result_json(result, export_state || spec.export_state, 2);
    } catch (const std::exception& e) {
        // Return error as JSON
        nlohmann::json err;
        err["status"] = "error";
        err["message"] = e.what();
        return err.dump(2);
    }
}

/**
 * Parse a circuit JSON and validate it without running.
 * Returns an empty string on success, or an error message.
 */
std::string validate_circuit_json(const std::string& json_str) {
    try {
        JsonCircuitSpec spec = parse_circuit_json(json_str);
        if (spec.num_qubits == 0) {
            return "Invalid circuit: num_qubits must be > 0";
        }
        return "";  // Success
    } catch (const std::exception& e) {
        return e.what();
    }
}

/**
 * Get version information.
 */
std::string get_version() {
    return "QLRET 1.0.0 (Phase 5 Python Bindings)";
}

}  // namespace qlret

PYBIND11_MODULE(_qlret_native, m) {
    m.doc() = "QLRET native Python bindings for low-rank quantum simulation";
    
    m.def("run_circuit_json", &qlret::run_circuit_json,
          py::arg("json_str"),
          py::arg("export_state") = false,
          R"doc(
Run a quantum circuit from JSON specification.

Parameters
----------
json_str : str
    JSON string containing circuit specification with keys:
    - circuit: {num_qubits, operations, observables}
    - config: {epsilon, initial_rank, export_state, shots}

export_state : bool, optional
    If True, include the low-rank L matrix in the result.

Returns
-------
str
    JSON string containing results:
    - status: "success" or "error"
    - execution_time_ms: float
    - final_rank: int
    - expectation_values: List[float]
    - samples: List[int] or null
    - state: dict (if export_state)
)doc");

    m.def("validate_circuit_json", &qlret::validate_circuit_json,
          py::arg("json_str"),
          R"doc(
Validate a circuit JSON without executing.

Parameters
----------
json_str : str
    JSON string to validate.

Returns
-------
str
    Empty string if valid, error message otherwise.
)doc");

    m.def("get_version", &qlret::get_version,
          "Get QLRET version string.");

    // Autodiff interfaces (Phase 8.3)
    m.def("autodiff_expectation", &qlret::autodiff_expectation_internal,
          py::arg("num_qubits"),
          py::arg("operations"),
          py::arg("params"),
          py::arg("observable"),
          R"doc(
Compute expectation value using tape-based autodiff.

Parameters
----------
num_qubits : int
    Number of qubits in the circuit.
operations : list[dict]
    Sequence of gate operations. Each dict must contain:
        - name: str (gate name, e.g., "RY", "CNOT")
        - qubits: list[int]
        - param_idx: int (optional, >=0 if parameterized, -1 or absent if fixed)
        - params: list[float] (optional fixed parameters)
params : list[float]
    Parameter vector used for gates with param_idx >= 0.
observable : dict
    Observable specification. Either:
        - {"type": "PauliZ"|"PauliX"|"PauliY", "qubit": int, "coefficient": float}
        - {"terms": [{"type": str, "qubit": int}, ...], "coefficient": float}

Returns
-------
float
    Expectation value.
)doc");

    m.def("autodiff_gradients", &qlret::autodiff_gradients_internal,
          py::arg("num_qubits"),
          py::arg("operations"),
          py::arg("params"),
          py::arg("observable"),
          R"doc(
Compute parameter-shift gradients for a circuit.

Parameters are identical to autodiff_expectation.

Returns
-------
list[float]
    Gradient vector (same length as params input).
)doc");
}

#endif  // USE_PYTHON
