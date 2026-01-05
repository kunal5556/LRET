#include "distributed_autodiff.h"
#include "mpi_parallel.h"
#include <iostream>
#include <cmath>
#include <cassert>

using namespace qlret;

namespace {

bool approx_equal(double a, double b, double tol = 1e-4) {
    return std::abs(a - b) <= tol;
}

}  // namespace

int main(int argc, char** argv) {
#if defined(USE_GPU) && defined(USE_MPI) && defined(USE_NCCL)
    mpi_init(&argc, &argv);
    int world = get_mpi_size();
    int rank = get_mpi_rank();

    if (world < 2) {
        if (rank == 0) {
            std::cout << "Skipping: requires >=2 MPI ranks for distributed autodiff gradient check\n";
        }
        mpi_finalize();
        return 0;
    }

    // Test 1: Simple RY circuit gradient comparison
    {
        QuantumSequence seq(2);
        seq.add_gate(GateOp(GateType::RY, 0));      // param 0
        seq.add_gate(GateOp(GateType::CNOT, 0, 1));
        seq.add_gate(GateOp(GateType::RY, 1));      // param 1

        std::vector<int> pidx = {0, -1, 1};

        DistributedGPUConfig cfg;
        cfg.world_size = world;
        cfg.rank = rank;
        cfg.device_id = rank;
        cfg.enable_collectives = true;

        DistributedAutoDiffCircuit dist_circuit(2, seq, pidx, cfg);

        std::vector<double> params = {0.3, 0.5};
        Observable obs{ObservableType::PauliZ, 0};

        double exp_val = dist_circuit.forward(params, obs);
        auto grads = dist_circuit.backward(params, obs);

        if (rank == 0) {
            // Reference: single-GPU autodiff
            AutoDiffCircuit ref_circuit(2, seq, pidx);
            double ref_exp = ref_circuit.forward(params, obs);
            auto ref_grads = ref_circuit.backward(params, obs);

            assert(approx_equal(exp_val, ref_exp));
            assert(grads.size() == ref_grads.size());
            for (size_t i = 0; i < grads.size(); ++i) {
                assert(approx_equal(grads[i], ref_grads[i]));
            }
            std::cout << "[PASS] Distributed autodiff matches single-GPU reference\n";
        }
    }

    // Test 2: Validate using helper function
    {
        bool ok = validate_distributed_gradients(2, 2, 1e-4);
        if (rank == 0 && !ok) {
            std::cerr << "[FAIL] validate_distributed_gradients failed\n";
            mpi_finalize();
            return 1;
        }
    }

    if (rank == 0) {
        std::cout << "test_autodiff_multi_gpu passed\n";
    }
    mpi_finalize();
    return 0;
#else
    (void)argc;
    (void)argv;
    std::cout << "Skipping: requires USE_GPU, USE_MPI, and USE_NCCL.\n";
    return 0;
#endif
}
