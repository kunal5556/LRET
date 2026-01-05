#include "distributed_gpu.h"
#include "mpi_parallel.h"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace qlret;

int main(int argc, char** argv) {
#if defined(USE_GPU) && defined(USE_MPI) && defined(USE_NCCL)
    mpi_init(&argc, &argv);
    int world = get_mpi_size();
    int rank = get_mpi_rank();

    if (world < 2) {
        if (rank == 0) {
            std::cout << "Skipping: requires >=2 MPI ranks for sync/collective test\n";
        }
        mpi_finalize();
        return 0;
    }

    DistributedGPUConfig cfg;
    cfg.world_size = world;
    cfg.rank = rank;
    cfg.device_id = rank;  // assume one GPU per rank
    cfg.enable_collectives = true;
    cfg.overlap_comm_compute = true;
    cfg.verbose = false;

    // Simple 2-qubit |00> state to distribute
    MatrixXcd L(4, 1);
    L << 1.0, 0.0, 0.0, 0.0;

    DistributedGPUSimulator dist(cfg);
    dist.distribute_state(L);

    // Hint to overlap next two-qubit gate communication path
    dist.overlap_for_two_qubit(true);

    // All-reduce sync: sum of ranks (1..world)
    double local = static_cast<double>(rank + 1);
    double reduced = dist.all_reduce_expectation(local);
    double expected = static_cast<double>(world * (world + 1) / 2);
    assert(std::abs(reduced - expected) < 1e-9);

    // Gather check: rank 0 should see full state
    MatrixXcd gathered = dist.gather_state();
    if (rank == 0) {
        assert(gathered.rows() == L.rows());
        assert(gathered.cols() == L.cols());
        assert(gathered.isApprox(L, 1e-12));
        std::cout << "test_multi_gpu_sync passed\n";
    } else {
        // Non-root ranks may return empty; just ensure no crash
        assert(gathered.size() == 0 || gathered.rows() == 0 || gathered.cols() == 0);
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
