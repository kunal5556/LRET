#include "advanced_noise.h"
#include <cassert>
#include <iostream>

using namespace qlret;

int main() {
    std::cout << "=== Advanced Noise Tests ===" << std::endl;

    // Time-dependent scaling (linear)
    TimeVaryingNoiseParams params;
    params.base_depol_prob = 0.01;
    params.alpha = 1.0;
    params.model = TimeScalingModel::LINEAR;
    params.max_depth = 10;
    double scaled = compute_time_scaled_rate(params.base_depol_prob, 5, params);
    assert(scaled > params.base_depol_prob);

    // Correlated Pauli channel doubles rank when two Kraus ops are present
    MatrixXcd L = MatrixXcd::Zero(4, 1);
    L(0, 0) = 1.0; // |00>
    std::vector<double> probs(16, 0.0);
    probs[0] = 0.9;   // II
    probs[15] = 0.1;  // ZZ
    MatrixXcd L_corr = apply_correlated_pauli_channel(L, 0, 1, probs, 2);
    assert(L_corr.cols() == 2);

    // Memory effect scaling
    MemoryEffect effect;
    effect.prev_gate_type = "x";
    effect.affected_gate_type = "z";
    effect.error_scale_factor = 0.7;
    effect.memory_depth = 1;

    CircuitMemoryState mem;
    mem.max_memory_depth = 2;
    mem.current_depth = 1;
    append_memory_history(mem, "x", {0});
    double mem_scale = evaluate_memory_scale(mem, "z", {0}, {effect});
    assert(mem_scale < 1.0);

    std::cout << "All advanced noise tests passed.\n";
    return 0;
}
