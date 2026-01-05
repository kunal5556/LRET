#include "fault_tolerance.h"
#include "mpi_parallel.h"
#include <iostream>
#include <filesystem>

using namespace qlret;

int main(int argc, char** argv) {
#if defined(USE_GPU) && defined(USE_MPI) && defined(USE_NCCL)
    mpi_init(&argc, &argv);
    int rank = get_mpi_rank();
    int world = get_mpi_size();

    if (world < 2) {
        if (rank == 0) {
            std::cout << "Skipping: requires >=2 MPI ranks for fault tolerance test\n";
        }
        mpi_finalize();
        return 0;
    }

    // Test fault recovery: 100 ops, fail at step 35
    bool ok = test_fault_recovery(2, 100, 35);

    // Cleanup checkpoint directory
    if (rank == 0) {
        std::filesystem::remove_all("./ckpt_test");
    }

    mpi_finalize();
    return ok ? 0 : 1;
#else
    (void)argc;
    (void)argv;
    std::cout << "Skipping: requires USE_GPU, USE_MPI, and USE_NCCL.\n";
    return 0;
#endif
}
