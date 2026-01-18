#pragma once

#include "types.h"
#include "cli_parser.h"
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace qlret {

// Observable types supported by JSON interface
enum class JsonObservableType {
    PAULI,
    TENSOR,
    HERMITIAN
};

struct JsonOperation {
    std::string name;
    std::vector<size_t> wires;
    std::vector<double> params;
    std::vector<MatrixXcd> kraus_matrices;  // For KRAUS channel operations
    bool is_channel = false;  // True if this is a noise channel
};

struct JsonObservable {
    JsonObservableType type = JsonObservableType::PAULI;
    std::vector<std::string> operators;  // Pauli symbols ("X","Y","Z","I")
    std::vector<size_t> wires;
    double coefficient = 1.0;
    MatrixXcd matrix;  // Used for HERMITIAN
};

struct JsonCircuitSpec {
    size_t num_qubits = 0;
    std::vector<JsonOperation> operations;
    std::vector<JsonObservable> observables;
    SimConfig config;               // Uses truncation_threshold, batch_size, etc.
    size_t initial_rank = 1;
    bool export_state = false;
    std::optional<size_t> shots;    // Optional sampling shots
    
    // Parallelization settings
    int num_threads = 0;            // 0 = auto (use all available cores)
    std::string parallel_mode = "hybrid";  // "sequential", "row", "column", "batch", "hybrid"
};

struct JsonRunResult {
    MatrixXcd L_final;
    std::vector<double> expectations;
    double execution_time_ms = 0.0;
    std::vector<int> samples;  // Optional samples (flattened bitstrings)
};

// Parse circuit specification from JSON text
JsonCircuitSpec parse_circuit_json(const std::string& json_text);
JsonCircuitSpec parse_circuit_json(const nlohmann::json& j);
JsonCircuitSpec parse_circuit_json_file(const std::string& path);

// Execute circuit using LRET core
JsonRunResult run_json_circuit(const JsonCircuitSpec& spec);

// Export result to JSON (pretty if indent > 0)
std::string export_result_json(
    const JsonRunResult& result,
    bool include_state,
    int indent = 2
);

}  // namespace qlret
