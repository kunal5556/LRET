#include "noise_import.h"
#include <iostream>

using namespace qlret;

int main() {
    std::cout << "=== LRET Noise Model Import Test ===" << std::endl;
    
    // Test JSON string (minimal Qiskit Aer format)
    std::string test_json = R"({
        "device_name": "test_device",
        "backend_version": "1.0.0",
        "noise_model_version": "1.0",
        "errors": [
            {
                "type": "qerror",
                "operations": ["cx"],
                "gate_qubits": [[0, 1]],
                "probabilities": [0.99, 0.01],
                "instructions": [
                    [{"name": "id"}],
                    [{"name": "pauli", "params": [1, 0, 0]}]
                ]
            },
            {
                "type": "depolarizing",
                "operations": ["x", "y", "z", "h"],
                "gate_qubits": [[0], [1], [2]],
                "param": 0.001
            },
            {
                "type": "thermal_relaxation",
                "operations": ["id"],
                "gate_qubits": [[0], [1]],
                "T1": 50000.0,
                "T2": 70000.0,
                "gate_time": 35.0
            }
        ]
    })";
    
    try {
        NoiseModelImporter importer;
        
        // Test 1: Parse JSON
        std::cout << "\nTest 1: Parsing JSON..." << std::endl;
        NoiseModel model = importer.load_from_json_string(test_json);
        std::cout << "✓ Parsed " << model.num_errors() << " errors" << std::endl;
        
        // Test 2: Validate model
        std::cout << "\nTest 2: Validating noise model..." << std::endl;
        bool valid = importer.validate_noise_model(model, true);
        if (valid) {
            std::cout << "✓ Noise model is valid" << std::endl;
        } else {
            std::cout << "✗ Noise model validation failed" << std::endl;
        }
        
        // Test 3: Print summary
        std::cout << "\nTest 3: Noise model summary:" << std::endl;
        importer.print_noise_model_summary(model);
        
        // Test 4: Error lookup
        std::cout << "\nTest 4: Error lookup..." << std::endl;
        auto cx_errors = model.find_applicable_errors("cx", {0, 1});
        std::cout << "✓ Found " << cx_errors.size() << " error(s) for CNOT(0,1)" << std::endl;
        
        // Test 5: Error conversion
        std::cout << "\nTest 5: Converting Qiskit errors to LRET..." << std::endl;
        for (const auto& error : model.errors) {
            if (error.type == QiskitErrorType::DEPOLARIZING) {
                auto noise_ops = importer.convert_qiskit_error_to_lret(error, 0);
                std::cout << "✓ Depolarizing error → " << noise_ops.size() << " LRET noise op(s)" << std::endl;
            }
            if (error.type == QiskitErrorType::THERMAL_RELAXATION) {
                auto noise_ops = importer.convert_qiskit_error_to_lret(error, 0);
                std::cout << "✓ Thermal relaxation → " << noise_ops.size() << " LRET noise op(s)" << std::endl;
            }
        }
        
        // Test 6: Apply noise to a test circuit
        std::cout << "\nTest 6: Applying noise to circuit..." << std::endl;
        QuantumSequence clean_circuit;
        clean_circuit.depth = 3;
        
        // Add a few gates
        GateOp h_gate(GateType::H, {0});
        GateOp cnot_gate(GateType::CNOT, {0, 1});
        GateOp x_gate(GateType::X, {1});
        
        clean_circuit.operations.push_back(h_gate);
        clean_circuit.operations.push_back(cnot_gate);
        clean_circuit.operations.push_back(x_gate);
        
        QuantumSequence noisy_circuit = importer.apply_noise_model(clean_circuit, model);
        
        std::cout << "✓ Clean circuit: " << clean_circuit.operations.size() << " operations" << std::endl;
        std::cout << "✓ Noisy circuit: " << noisy_circuit.operations.size() << " operations" << std::endl;
        std::cout << "  (Noise added: " << (noisy_circuit.operations.size() - clean_circuit.operations.size()) 
                  << " operations)" << std::endl;
        
        std::cout << "\n=== All Tests Passed ===" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n✗ Error: " << e.what() << std::endl;
        return 1;
    }
}
