#pragma once

#include "types.h"
#include "cli_parser.h"
#include "output_formatter.h"
#include "gate_fusion.h"
#include <vector>

namespace qlret {

// Auto-select best mode based on problem size
ParallelMode auto_select_mode(size_t num_qubits, size_t depth, size_t rank_estimate = 10);

// Auto-select batch size
size_t auto_select_batch_size(size_t num_qubits);

// Get workload class string
std::string get_workload_class(size_t num_qubits);

// Run simulation with specific mode (with optional gate fusion)
ModeResult run_with_mode(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    ParallelMode mode,
    const SimConfig& config,
    size_t batch_size = 0,
    const FusionConfig* fusion_config = nullptr  // Optional fusion configuration
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
