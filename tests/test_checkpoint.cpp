#include "checkpoint.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>

using namespace qlret;

namespace {

bool approx_equal(double a, double b, double tol = 1e-12) {
    return std::abs(a - b) <= tol;
}

}  // namespace

int main() {
    try {
        // Create test L matrix (2-qubit, rank 2)
        MatrixXcd L(4, 2);
        L << Complex(1.0, 0.0), Complex(0.0, 0.0),
             Complex(0.0, 0.1), Complex(0.5, 0.0),
             Complex(0.2, -0.3), Complex(0.1, 0.2),
             Complex(0.0, 0.0), Complex(0.8, -0.1);

        CheckpointMeta meta;
        meta.step = 42;
        meta.num_qubits = 2;
        meta.rank = 2;
        meta.config_json = R"({"depth": 10, "noise": true})";

        std::string path = "test_checkpoint.bin";

        // Save checkpoint
        bool ok = save_checkpoint(path, L, meta);
        assert(ok);

        // Load checkpoint
        MatrixXcd L_loaded;
        CheckpointMeta meta_loaded;
        ok = load_checkpoint(path, L_loaded, meta_loaded);
        assert(ok);

        // Verify metadata
        assert(meta_loaded.step == meta.step);
        assert(meta_loaded.num_qubits == meta.num_qubits);
        assert(meta_loaded.rank == meta.rank);
        assert(meta_loaded.config_json == meta.config_json);

        // Verify L matrix
        assert(L_loaded.rows() == L.rows());
        assert(L_loaded.cols() == L.cols());
        assert(L_loaded.isApprox(L, 1e-12));

        // Cleanup
        std::remove(path.c_str());

        // Async checkpoint test
        {
            AsyncCheckpointWriter writer;
            std::string async_path = "test_checkpoint_async.bin";
            writer.start(async_path, L, meta);
            assert(writer.is_busy() || !writer.is_busy());  // may finish quickly
            bool success = writer.wait();
            assert(success);

            MatrixXcd L2;
            CheckpointMeta meta2;
            ok = load_checkpoint(async_path, L2, meta2);
            assert(ok);
            assert(L2.isApprox(L, 1e-12));

            std::remove(async_path.c_str());
        }

        std::cout << "Checkpoint tests passed\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Exception: " << ex.what() << "\n";
        return 1;
    }
}
