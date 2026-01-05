#include "distributed_gpu.h"
#include "mpi_parallel.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace qlret;

namespace {

bool approx_equal(double a, double b, double tol = 1e-9) {
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
            std::cout << "Skipping: requires >=2 MPI ranks for collective tests\n";
        }
        mpi_finalize();
        return 0;
    }

    DistributedGPUConfig cfg;
    cfg.world_size = world;
    cfg.rank = rank;
    cfg.device_id = rank;
    cfg.enable_collectives = true;
    cfg.verbose = false;

    // Test 1: AllReduce expectation (sum of ranks)
    {
        DistributedGPUSimulator dist(cfg);
        double local = static_cast<double>(rank + 1);
        double reduced = dist.all_reduce_expectation(local);
        double expected = static_cast<double>(world * (world + 1) / 2);
        assert(approx_equal(reduced, expected));
        if (rank == 0) {
            std::cout << "[PASS] AllReduce expectation: " << reduced << " == " << expected << "\n";
        }
    }

    // Test 2: Distribute + Gather roundtrip (state integrity)
    {
        MatrixXcd L(4, 1);
        L << 1.0, 0.0, 0.0, 0.0;  // |00>

        DistributedGPUSimulator dist(cfg);
        dist.distribute_state(L);
        MatrixXcd gathered = dist.gather_state();

        if (rank == 0) {
            assert(gathered.rows() == L.rows());
            assert(gathered.cols() == L.cols());
            assert(gathered.isApprox(L, 1e-12));
            std::cout << "[PASS] Distribute + Gather roundtrip matches input\n";
        }
    }

    // Test 3: P2P overlap hint (no-op correctness check)
    {
        MatrixXcd L(4, 1);
        L << 0.5, 0.5, 0.5, 0.5;

        DistributedGPUSimulator dist(cfg);
        dist.distribute_state(L);
        dist.overlap_for_two_qubit(true);   // hint for upcoming remote dependency
        dist.overlap_for_two_qubit(false);  // hint for local gate

        MatrixXcd gathered = dist.gather_state();
        if (rank == 0) {
            assert(gathered.isApprox(L, 1e-12));
            std::cout << "[PASS] P2P overlap hint did not corrupt state\n";
        }
    }

    if (rank == 0) {
        std::cout << "test_multi_gpu_collectives passed\n";
    }
    mpi_finalize();
    return 0;
#else
    (void)argc;
    (void)argv;
    std::cout << "Skipping: requires USE_GPU, USE_MPI, USE_NCCL.\n";
    return 0;
#endif
}
