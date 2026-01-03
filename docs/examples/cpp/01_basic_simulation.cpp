/**
 * @file 01_basic_simulation.cpp
 * @brief Basic LRET simulator usage examples
 * 
 * This example demonstrates:
 * - Creating a simulator instance
 * - Applying single and two-qubit gates
 * - Measuring qubits
 * - Getting state vectors and probabilities
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include "simulator.h"

using namespace std;
using namespace lret;

void print_separator(const string& title) {
    cout << "\n" << string(60, '=') << "\n";
    cout << title << "\n";
    cout << string(60, '=') << "\n\n";
}

void print_state(const LRETSimulator& sim) {
    auto state = sim.get_statevector();
    cout << "State vector:\n";
    for (size_t i = 0; i < state.size(); ++i) {
        auto& amp = state[i];
        double prob = norm(amp);
        if (prob > 1e-10) {  // Only print non-zero amplitudes
            cout << "  |" << bitset<3>(i) << ">: "
                 << fixed << setprecision(4)
                 << amp.real() << (amp.imag() >= 0 ? "+" : "")
                 << amp.imag() << "i"
                 << " (prob: " << prob << ")\n";
        }
    }
}

void example_bell_state() {
    print_separator("Example 1: Creating a Bell State");
    
    // Create 2-qubit simulator
    LRETSimulator sim(2);
    
    cout << "Creating Bell state |Φ+> = (|00> + |11>)/√2\n\n";
    
    // Apply Hadamard to qubit 0
    sim.h(0);
    cout << "After H(0):\n";
    print_state(sim);
    
    // Apply CNOT
    sim.cx(0, 1);
    cout << "\nAfter CNOT(0, 1):\n";
    print_state(sim);
    
    // Measure multiple times
    cout << "\nMeasurement results (100 shots):\n";
    map<string, int> counts;
    for (int i = 0; i < 100; ++i) {
        LRETSimulator sim_copy = sim;  // Copy state
        vector<int> results = sim_copy.measure();
        string outcome = to_string(results[0]) + to_string(results[1]);
        counts[outcome]++;
    }
    
    for (const auto& [outcome, count] : counts) {
        cout << "  |" << outcome << ">: " << count << " (" 
             << fixed << setprecision(3) << count/100.0 << ")\n";
    }
}

void example_ghz_state() {
    print_separator("Example 2: Creating a GHZ State");
    
    int n_qubits = 3;
    LRETSimulator sim(n_qubits);
    
    cout << "Creating GHZ state |GHZ> = (|000> + |111>)/√2\n\n";
    
    // Create GHZ state
    sim.h(0);
    for (int i = 0; i < n_qubits - 1; ++i) {
        sim.cx(i, i + 1);
    }
    
    cout << "Final state:\n";
    print_state(sim);
    
    // Verify entanglement by measuring correlations
    cout << "\nMeasuring correlations (100 shots):\n";
    map<string, int> counts;
    
    for (int i = 0; i < 100; ++i) {
        LRETSimulator sim_copy = sim;
        vector<int> results = sim_copy.measure();
        string outcome;
        for (int r : results) outcome += to_string(r);
        counts[outcome]++;
    }
    
    for (const auto& [outcome, count] : counts) {
        cout << "  |" << outcome << ">: " << count << "\n";
    }
}

void example_rotation_gates() {
    print_separator("Example 3: Rotation Gates");
    
    LRETSimulator sim(1);
    
    cout << "Applying rotation gates to create arbitrary state\n\n";
    
    // Create state α|0> + β|1> with specific angle
    double theta = M_PI / 4;  // 45 degrees
    double phi = M_PI / 2;    // 90 degrees
    
    cout << "Applying RY(" << theta << "):\n";
    sim.ry(theta, 0);
    print_state(sim);
    
    cout << "\nApplying RZ(" << phi << "):\n";
    sim.rz(phi, 0);
    print_state(sim);
    
    // Calculate expected probabilities
    double prob_0 = pow(cos(theta/2), 2);
    double prob_1 = pow(sin(theta/2), 2);
    
    cout << "\nExpected probabilities:\n";
    cout << "  |0>: " << prob_0 << "\n";
    cout << "  |1>: " << prob_1 << "\n";
}

void example_expectation_values() {
    print_separator("Example 4: Expectation Values");
    
    LRETSimulator sim(2);
    
    // Create |+> state
    sim.h(0);
    
    cout << "State: |+>|0>\n\n";
    
    // Measure expectation values
    double exp_z0 = sim.expectation_value("Z", {0});
    double exp_x0 = sim.expectation_value("X", {0});
    double exp_y0 = sim.expectation_value("Y", {0});
    
    cout << "Expectation values for qubit 0:\n";
    cout << "  <Z>: " << exp_z0 << " (expected: 0.0)\n";
    cout << "  <X>: " << exp_x0 << " (expected: 1.0)\n";
    cout << "  <Y>: " << exp_y0 << " (expected: 0.0)\n";
    
    // Two-qubit observable
    sim.h(1);
    sim.cx(0, 1);
    
    cout << "\nAfter creating Bell state |Φ+>:\n";
    double exp_zz = sim.expectation_value("ZZ", {0, 1});
    double exp_xx = sim.expectation_value("XX", {0, 1});
    
    cout << "  <ZZ>: " << exp_zz << " (expected: 1.0)\n";
    cout << "  <XX>: " << exp_xx << " (expected: 1.0)\n";
}

void example_circuit_composition() {
    print_separator("Example 5: Circuit Composition");
    
    LRETSimulator sim(3);
    
    cout << "Building a complex quantum circuit\n\n";
    
    // Layer 1: Hadamards
    cout << "Layer 1: Hadamard gates\n";
    for (int i = 0; i < 3; ++i) {
        sim.h(i);
    }
    
    // Layer 2: Entangling gates
    cout << "Layer 2: CNOT gates\n";
    sim.cx(0, 1);
    sim.cx(1, 2);
    
    // Layer 3: Rotation layer
    cout << "Layer 3: Rotation gates\n";
    sim.ry(M_PI / 4, 0);
    sim.ry(M_PI / 3, 1);
    sim.ry(M_PI / 6, 2);
    
    // Layer 4: More entangling
    cout << "Layer 4: More entangling\n";
    sim.cx(2, 0);
    
    cout << "\nFinal state:\n";
    print_state(sim);
    
    // Get probabilities directly
    cout << "\nMeasurement probabilities:\n";
    auto probs = sim.get_probabilities();
    for (size_t i = 0; i < probs.size(); ++i) {
        if (probs[i] > 1e-10) {
            cout << "  |" << bitset<3>(i) << ">: " 
                 << fixed << setprecision(4) << probs[i] << "\n";
        }
    }
}

void example_partial_measurement() {
    print_separator("Example 6: Partial Measurement");
    
    LRETSimulator sim(3);
    
    // Create GHZ-like state
    sim.h(0);
    sim.cx(0, 1);
    sim.cx(1, 2);
    
    cout << "Initial GHZ state:\n";
    print_state(sim);
    
    // Measure only first qubit
    cout << "\nMeasuring qubit 0:\n";
    int result = sim.measure_single(0);
    cout << "  Result: " << result << "\n\n";
    
    cout << "State after measurement:\n";
    print_state(sim);
}

void example_reset_and_reuse() {
    print_separator("Example 7: Reset and Reuse");
    
    LRETSimulator sim(2);
    
    // First circuit
    cout << "First circuit: Bell state\n";
    sim.h(0);
    sim.cx(0, 1);
    print_state(sim);
    
    // Reset
    cout << "\nResetting simulator...\n";
    sim.reset();
    
    // Second circuit
    cout << "\nSecond circuit: Different state\n";
    sim.x(0);
    sim.h(1);
    print_state(sim);
}

void example_properties() {
    print_separator("Example 8: Simulator Properties");
    
    LRETSimulator sim(4, 0.01);  // 4 qubits, 1% noise
    
    cout << "Simulator properties:\n";
    cout << "  Number of qubits: " << sim.get_num_qubits() << "\n";
    cout << "  Noise level: " << sim.get_noise_level() << "\n";
    cout << "  State dimension: " << (1 << sim.get_num_qubits()) << "\n";
    cout << "  Memory usage: ~" << (1 << sim.get_num_qubits()) * 16 << " bytes\n";
}

int main() {
    cout << "LRET Basic Simulation Examples\n";
    cout << "==============================\n";
    
    try {
        example_bell_state();
        example_ghz_state();
        example_rotation_gates();
        example_expectation_values();
        example_circuit_composition();
        example_partial_measurement();
        example_reset_and_reuse();
        example_properties();
        
        cout << "\n" << string(60, '=') << "\n";
        cout << "All examples completed successfully!\n";
        cout << string(60, '=') << "\n";
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
