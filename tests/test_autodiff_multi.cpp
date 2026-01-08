#include "autodiff.h"
#include <iostream>
#include <vector>
#include <cmath>

using namespace qlret;

namespace {

bool approx_equal(double a, double b, double tol) {
    return std::abs(a - b) <= tol;
}

int fail(const std::string& msg) {
    std::cerr << "[FAIL] " << msg << std::endl;
    return 1;
}

}  // namespace

int main() {
    try {
        // Observable: X0 X1
        Observable obs;
        obs.terms = {{ObservableType::PauliX, 0}, {ObservableType::PauliX, 1}};

        // Circuit: RY(theta0) on q0, CNOT(0,1), RZ(theta1) on q1
        QuantumSequence seq(2);
        seq.add_gate(GateOp(GateType::RY, 0));      // param 0
        seq.add_gate(GateOp(GateType::CNOT, 0, 1));
        seq.add_gate(GateOp(GateType::RZ, 1));      // param 1

        std::vector<int> pidx = {0, -1, 1};
        AutoDiffCircuit circuit(2, seq, pidx);

        double theta0 = 0.3;  // controls superposition amplitude
        double theta1 = 0.4;  // phase between |00> and |11>
        std::vector<double> params = {theta0, theta1};

        double exp_val = circuit.forward(params, obs);
        // The expectation for this circuit is sin(theta0) * cos(theta1)
        // (verified empirically to match simulator convention)
        double expected_exp = std::sin(theta0) * std::cos(theta1);
        if (!approx_equal(exp_val, expected_exp, 1e-6)) {
            return fail("Two-qubit X0X1 expectation mismatch");
        }

        auto grads = circuit.backward(params, obs);
        if (grads.size() != 2) {
            return fail("Unexpected gradient size for two-parameter test");
        }

        // Gradients for f = sin(theta0) * cos(theta1)
        double d_theta0 = std::cos(theta0) * std::cos(theta1);
        double d_theta1 = -std::sin(theta0) * std::sin(theta1);

        if (!approx_equal(grads[0], d_theta0, 1e-4)) {
            return fail("Gradient mismatch for theta0");
        }
        if (!approx_equal(grads[1], d_theta1, 1e-4)) {
            return fail("Gradient mismatch for theta1");
        }

        std::cout << "Autodiff multi-parameter multi-qubit tests passed" << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Exception: " << ex.what() << std::endl;
        return 1;
    }
}
