#pragma once

#include "types.h"
#include "cli_parser.h"
#include "output_formatter.h"
#include "gate_fusion.h"
#include "circuit_optimizer.h"
#include "gpu_simulator.h"
#include <vector>

namespace qlret {

// Device selection
enum class DeviceType {
    CPU,
    GPU,
    AUTO
};

// Auto-select best mode based on problem size
ParallelMode auto_select_mode(size_t num_qubits, size_t depth, size_t rank_estimate = 10);

// Auto-select batch size
size_t auto_select_batch_size(size_t num_qubits);

// Get workload class string
std::string get_workload_class(size_t num_qubits);

// Combined optimization configuration for run_with_mode
struct OptimizationConfig {
    FusionConfig fusion;
    StratificationConfig stratification;
    GPUConfig gpu;
    DeviceType device = DeviceType::AUTO;
    
    OptimizationConfig() = default;
    
    // Constructor from CLI options
    explicit OptimizationConfig(const CLIOptions& opts) {
        fusion.enable_fusion = opts.enable_fusion;
        fusion.min_gates_to_fuse = opts.min_fusion_gates;
        fusion.max_fusion_depth = opts.max_fusion_depth;
        fusion.verbose = opts.verbose;
        
        stratification.enable_stratification = opts.enable_stratify;
        stratification.use_greedy_assignment = opts.greedy_layers;
        stratification.min_layer_size = opts.min_layer_size;
        stratification.verbose = opts.verbose;
        
        // GPU configuration
        gpu.enable_gpu = opts.enable_gpu;
        gpu.device_id = opts.gpu_device_id;
        gpu.use_cuquantum = opts.use_cuquantum;
        gpu.max_gpu_memory = opts.gpu_memory_limit * 1024ULL * 1024ULL * 1024ULL;
        gpu.verbose = opts.verbose;
        
        // Device selection
        if (opts.auto_device) {
            device = DeviceType::AUTO;
        } else if (opts.enable_gpu) {
            device = DeviceType::GPU;
        } else {
            device = DeviceType::CPU;
        }
    }
};

// Run simulation with specific mode (with optional gate fusion and stratification)
ModeResult run_with_mode(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    ParallelMode mode,
    const SimConfig& config,
    size_t batch_size = 0,
    const FusionConfig* fusion_config = nullptr  // Optional fusion configuration
);

// Run simulation with combined optimization config
ModeResult run_optimized(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    ParallelMode mode,
    const SimConfig& config,
    const OptimizationConfig& opt_config,
    size_t batch_size = 0
);

// Run simulation with pre-fused sequence
ModeResult run_with_fused_sequence(
    const MatrixXcd& L_init,
    const FusedSequence& fused_seq,
    size_t num_qubits,
    ParallelMode mode,
    const SimConfig& config,
    size_t batch_size = 0
);

// Run all modes for comparison
std::vector<ModeResult> run_all_modes_comparison(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    const SimConfig& config,
    size_t batch_size
);

// Individual mode implementations
namespace modes {

// Sequential (no parallelism)
MatrixXcd run_sequential(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    const SimConfig& config
);

// Row-wise parallelization
MatrixXcd run_row_parallel(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    const SimConfig& config
);

// Column-wise parallelization
MatrixXcd run_column_parallel(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    const SimConfig& config
);

// Gate batching parallelization
MatrixXcd run_batch_parallel(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    size_t batch_size,
    const SimConfig& config
);

// Hybrid: row parallel + batch organization
MatrixXcd run_hybrid(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    size_t batch_size,
    const SimConfig& config
);

}  // namespace modes

// Gate application with different parallelization strategies
namespace parallel_ops {

// Apply gate to L using row parallelization
MatrixXcd apply_gate_row_parallel(
    const MatrixXcd& L,
    const GateOp& gate,
    size_t num_qubits
);

// Apply gate to L using column parallelization
MatrixXcd apply_gate_column_parallel(
    const MatrixXcd& L,
    const GateOp& gate,
    size_t num_qubits
);

// Apply gate sequentially (baseline)
MatrixXcd apply_gate_sequential(
    const MatrixXcd& L,
    const GateOp& gate,
    size_t num_qubits
);

}  // namespace parallel_ops

}  // namespace qlret
