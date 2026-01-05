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
        Observable obs{ObservableType::PauliZ, 0};

        // Test 1: single RY(theta) on |0> should give <Z> = cos(theta)
        {
            QuantumSequence seq(1);
            seq.add_gate(GateOp(GateType::RY, 0));
            std::vector<int> pidx = {0};
            AutoDiffCircuit circuit(1, seq, pidx);

            double theta = PI / 4.0;  // 45 degrees
            std::vector<double> params = {theta};

            double exp_val = circuit.forward(params, obs);
            double expected_exp = std::cos(theta);
            if (!approx_equal(exp_val, expected_exp, 1e-6)) {
                return fail("Single RY expectation mismatch");
            }

            auto grads = circuit.backward(params, obs);
            if (grads.size() != 1) {
                return fail("Unexpected gradient size for single RY");
            }
            double expected_grad = -std::sin(theta);
            if (!approx_equal(grads[0], expected_grad, 1e-4)) {
                return fail("Single RY gradient mismatch");
            }
        }

        // Test 2: shared parameter across two RY gates -> RY(2*theta) total
        {
            QuantumSequence seq(1);
            seq.add_gate(GateOp(GateType::RY, 0));
            seq.add_gate(GateOp(GateType::RY, 0));
            std::vector<int> pidx = {0, 0};
            AutoDiffCircuit circuit(1, seq, pidx);

            double theta = 0.3;
            std::vector<double> params = {theta};

            double exp_val = circuit.forward(params, obs);
            double expected_exp = std::cos(2.0 * theta);
            if (!approx_equal(exp_val, expected_exp, 1e-6)) {
                return fail("Shared-parameter expectation mismatch");
            }

            auto grads = circuit.backward(params, obs);
            if (grads.size() != 1) {
                return fail("Unexpected gradient size for shared-parameter test");
            }
            double expected_grad = -2.0 * std::sin(2.0 * theta);
            if (!approx_equal(grads[0], expected_grad, 1e-4)) {
                return fail("Shared-parameter gradient mismatch");
            }
        }

        std::cout << "Autodiff tests passed" << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Exception: " << ex.what() << std::endl;
        return 1;
    }
}
