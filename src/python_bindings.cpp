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
#include <string>
#include <stdexcept>

namespace py = pybind11;

namespace qlret {

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
}

#endif  // USE_PYTHON
