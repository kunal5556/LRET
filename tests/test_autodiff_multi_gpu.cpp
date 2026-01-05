#include "autodiff.h"
#include "distributed_gpu.h"
#include "mpi_parallel.h"
#include <iostream>

using namespace qlret;

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

    if (rank == 0) {
        std::cout << "Distributed autodiff multi-GPU path not implemented yet; skipping test.\n";
        std::cout << "Planned: compare parameter-shift gradients vs single-GPU reference for depth-2 circuits.\n";
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
