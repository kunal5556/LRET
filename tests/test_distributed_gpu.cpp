#include "distributed_gpu.h"
#include <Eigen/Dense>
#include <cassert>
#include <iostream>

using namespace qlret;

int main() {
#ifdef USE_GPU
    DistributedGPUConfig cfg;
    cfg.world_size = 1;  // single GPU smoke test (no MPI/NCCL required)
    cfg.rank = 0;
    cfg.device_id = 0;
    cfg.verbose = false;

    // 2-qubit, rank-2 example: L is 4 x 2
    MatrixXcd L(4, 2);
    L <<
        std::complex<double>(1.0, 0.0), std::complex<double>(0.0, 0.0),
        std::complex<double>(0.0, 0.0), std::complex<double>(1.0, 0.0),
        std::complex<double>(0.5, 0.1), std::complex<double>(-0.2, 0.3),
        std::complex<double>(-0.4, 0.2), std::complex<double>(0.7, -0.1);

    DistributedGPUSimulator dist(cfg);
    dist.distribute_state(L);
    MatrixXcd gathered = dist.gather_state();

    // On single GPU, gathered should equal input
    assert(gathered.rows() == L.rows());
    assert(gathered.cols() == L.cols());
    assert(gathered.isApprox(L, 1e-12));

    // All-reduce should return same value when world_size=1
    double local_exp = 3.14;
    double reduced = dist.all_reduce_expectation(local_exp);
    assert(std::abs(reduced - local_exp) < 1e-12);

    std::cout << "test_distributed_gpu (single GPU) passed\n";
    return 0;
#else
    std::cout << "USE_GPU not enabled; skipping distributed GPU test.\n";
    return 0;
#endif
}
