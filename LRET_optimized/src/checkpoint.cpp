#include "checkpoint.h"
#include <fstream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace qlret {

//==============================================================================
// Synchronous Checkpoint I/O
//==============================================================================

bool save_checkpoint(const std::string& path, const MatrixXcd& L, const CheckpointMeta& meta) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;

    // Header: magic + version
    const char magic[8] = "QLRETCK";
    uint32_t version = 1;
    ofs.write(magic, 8);
    ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));

    // Metadata
    ofs.write(reinterpret_cast<const char*>(&meta.step), sizeof(meta.step));
    ofs.write(reinterpret_cast<const char*>(&meta.num_qubits), sizeof(meta.num_qubits));
    ofs.write(reinterpret_cast<const char*>(&meta.rank), sizeof(meta.rank));

    size_t json_len = meta.config_json.size();
    ofs.write(reinterpret_cast<const char*>(&json_len), sizeof(json_len));
    ofs.write(meta.config_json.data(), static_cast<std::streamsize>(json_len));

    // L matrix dimensions
    Eigen::Index rows = L.rows();
    Eigen::Index cols = L.cols();
    ofs.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    ofs.write(reinterpret_cast<const char*>(&cols), sizeof(cols));

    // L matrix data (column-major, complex<double>)
    ofs.write(reinterpret_cast<const char*>(L.data()),
              static_cast<std::streamsize>(rows * cols * sizeof(Complex)));

    return ofs.good();
}

bool load_checkpoint(const std::string& path, MatrixXcd& L, CheckpointMeta& meta) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;

    char magic[8] = {0};
    ifs.read(magic, 8);
    if (std::string(magic, 7) != "QLRETCK") return false;

    uint32_t version = 0;
    ifs.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 1) return false;

    ifs.read(reinterpret_cast<char*>(&meta.step), sizeof(meta.step));
    ifs.read(reinterpret_cast<char*>(&meta.num_qubits), sizeof(meta.num_qubits));
    ifs.read(reinterpret_cast<char*>(&meta.rank), sizeof(meta.rank));

    size_t json_len = 0;
    ifs.read(reinterpret_cast<char*>(&json_len), sizeof(json_len));
    meta.config_json.resize(json_len);
    ifs.read(meta.config_json.data(), static_cast<std::streamsize>(json_len));

    Eigen::Index rows = 0, cols = 0;
    ifs.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    ifs.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    L.resize(rows, cols);
    ifs.read(reinterpret_cast<char*>(L.data()),
             static_cast<std::streamsize>(rows * cols * sizeof(Complex)));

    return ifs.good();
}

//==============================================================================
// Async Checkpoint Writer
//==============================================================================

class AsyncCheckpointWriter::Impl {
public:
    Impl() = default;
    ~Impl() { wait(); }

    void start(const std::string& path, const MatrixXcd& L, const CheckpointMeta& meta) {
        wait();  // ensure previous write is done
        {
            std::lock_guard<std::mutex> lock(mtx_);
            busy_ = true;
            success_ = false;
        }
        worker_ = std::thread([this, path, L, meta]() {
            bool ok = save_checkpoint(path, L, meta);
            {
                std::lock_guard<std::mutex> lock(mtx_);
                success_ = ok;
                busy_ = false;
            }
            cv_.notify_all();
        });
    }

    bool wait() {
        if (worker_.joinable()) {
            worker_.join();
        }
        std::lock_guard<std::mutex> lock(mtx_);
        return success_;
    }

    bool is_busy() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return busy_;
    }

private:
    std::thread worker_;
    mutable std::mutex mtx_;
    std::condition_variable cv_;
    bool busy_ = false;
    bool success_ = false;
};

AsyncCheckpointWriter::AsyncCheckpointWriter() = default;
AsyncCheckpointWriter::~AsyncCheckpointWriter() = default;

void AsyncCheckpointWriter::start(const std::string& path, const MatrixXcd& L, const CheckpointMeta& meta) {
    if (!impl_) impl_ = std::make_unique<Impl>();
    impl_->start(path, L, meta);
}

bool AsyncCheckpointWriter::wait() {
    if (!impl_) return true;
    return impl_->wait();
}

bool AsyncCheckpointWriter::is_busy() const {
    if (!impl_) return false;
    return impl_->is_busy();
}

}  // namespace qlret
