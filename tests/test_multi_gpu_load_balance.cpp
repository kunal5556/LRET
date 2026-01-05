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
            std::cout << "Skipping: requires >=2 MPI ranks for load balance test\n";
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

    // Simple load balance smoke test: distribute, gather, verify
    // Future: add imbalance metrics and rebalance logic

    MatrixXcd L(8, 1);  // 3-qubit state
    L << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;  // |000>

    DistributedGPUSimulator dist(cfg);
    dist.distribute_state(L);

    size_t local_rows = dist.local_rows();
    size_t global_rows = dist.global_rows();

    // Simple imbalance metric: max local / (global / world)
    double ideal = static_cast<double>(global_rows) / static_cast<double>(world);
    double imbalance = static_cast<double>(local_rows) / ideal;

    // Gather imbalance metrics via all-reduce (max)
    double max_imbalance = dist.all_reduce_expectation(imbalance);  // sum; divide by world for avg
    max_imbalance /= static_cast<double>(world);  // approximate avg (not true max, but indicative)

    MatrixXcd gathered = dist.gather_state();
    if (rank == 0) {
        assert(gathered.rows() == L.rows());
        assert(gathered.cols() == L.cols());
        assert(gathered.isApprox(L, 1e-12));
        std::cout << "[INFO] Avg imbalance ratio: " << max_imbalance << "\n";
        std::cout << "[PASS] Load balance smoke test passed\n";
    }

    if (rank == 0) {
        std::cout << "test_multi_gpu_load_balance passed\n";
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
