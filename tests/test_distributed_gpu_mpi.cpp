#include "distributed_gpu.h"
#include "mpi_parallel.h"
#include <cassert>
#include <iostream>

using namespace qlret;

int main(int argc, char* argv[]) {
#if defined(USE_GPU) && defined(USE_MPI) && defined(USE_NCCL)
    mpi_init(&argc, &argv);
    int world = get_mpi_size();
    int rank = get_mpi_rank();

    if (world != 2) {
        if (rank == 0) {
            std::cout << "Skipping: requires exactly 2 MPI ranks for this smoke test\n";
        }
        mpi_finalize();
        return 0;
    }

    MatrixXcd L(4, 1);
    L << 1.0, 0.0, 0.0, 0.0;  // |00>

    DistributedGPUConfig cfg;
    cfg.world_size = world;
    cfg.rank = rank;
    cfg.enable_collectives = true;
    cfg.overlap_comm_compute = true;

    DistributedGPUSimulator dist(cfg);
    dist.distribute_state(L);

    // Collective sanity check
    double reduced = dist.all_reduce_expectation(1.0);
    assert(reduced == static_cast<double>(world));

    MatrixXcd gathered = dist.gather_state();
    if (rank == 0) {
        assert(gathered.rows() == L.rows());
        assert(gathered.cols() == L.cols());
    }

    if (rank == 0) {
        std::cout << "test_distributed_gpu_mpi (2-GPU/MPI) passed\n";
    }
    mpi_finalize();
    return 0;
#else
    (void)argc; (void)argv;
    std::cout << "Skipping: requires USE_GPU, USE_MPI, and USE_NCCL.\n";
    return 0;
#endif
}
