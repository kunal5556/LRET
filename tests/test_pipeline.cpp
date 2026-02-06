/**
 * @file test_pipeline.cpp
 * @brief Tests for Phase 6: Production Hardening & Performance Validation
 *
 * Validates:
 * - PipelineConfig defaults and construction
 * - OptimizedPipeline auto-strategy selection
 * - Manual strategy forcing (each noise/truncation/gate strategy)
 * - Execution of pipelines with gates only, noise only, mixed circuits
 * - Validation of density matrix properties after pipeline run
 * - Edge cases: empty circuit, single gate, single noise, 1-qubit
 * - Fidelity vs baseline (run_and_validate_pipeline)
 * - PipelineStats summary generation
 * - Benchmark circuit generators
 * - PhaseBenchmark integration smoke test
 */

#include "pipeline.h"
#include "benchmark_phases.h"
#include "simulator.h"
#include "gates_and_noise.h"
#include <iostream>
#include <string>
#include <cmath>
#include <vector>

using namespace qlret;

namespace {

int tests_passed = 0;
int tests_failed = 0;

bool approx_equal(double a, double b, double tol = 1e-6) {
    return std::abs(a - b) <= tol;
}

void check(bool condition, const std::string& test_name) {
    if (condition) {
        std::cout << "  [PASS] " << test_name << std::endl;
        ++tests_passed;
    } else {
        std::cerr << "  [FAIL] " << test_name << std::endl;
        ++tests_failed;
    }
}

// Helper: create |0...0⟩ as L factor
MatrixXcd make_zero_state_L(size_t num_qubits) {
    size_t dim = size_t(1) << num_qubits;
    MatrixXcd L = MatrixXcd::Zero(static_cast<Eigen::Index>(dim), 1);
    L(0, 0) = Complex(1.0, 0.0);
    return L;
}

// Helper: create a simple noisy circuit
QuantumSequence make_simple_circuit(size_t num_qubits, double noise_prob = 0.01) {
    QuantumSequence seq;

    // H on all qubits
    for (size_t q = 0; q < num_qubits; ++q) {
        seq.operations.push_back(GateOp(GateType::H, q));
    }

    // CNOT chain
    for (size_t q = 0; q + 1 < num_qubits; ++q) {
        seq.operations.push_back(GateOp(GateType::CNOT, q, q + 1));
    }

    // Noise on all qubits
    if (noise_prob > 0) {
        for (size_t q = 0; q < num_qubits; ++q) {
            seq.operations.push_back(NoiseOp(NoiseType::DEPOLARIZING, q, noise_prob));
        }
    }

    return seq;
}

} // anonymous namespace

//==============================================================================
// Test 1: PipelineConfig defaults
//==============================================================================

void test_config_defaults() {
    std::cout << "\n[Test 1] PipelineConfig defaults\n";

    PipelineConfig cfg;
    check(cfg.noise_strategy == NoiseStrategy::Auto, "noise_strategy default is Auto");
    check(cfg.truncation_strategy == TruncationStrategy::Auto, "truncation_strategy default is Auto");
    check(cfg.gate_strategy == GateStrategy::Auto, "gate_strategy default is Auto");
    check(approx_equal(cfg.truncation_threshold, 1e-4), "truncation_threshold default is 1e-4");
    check(cfg.max_rank == 0, "max_rank default is 0");
    check(cfg.batch_size == 64, "batch_size default is 64");
    check(cfg.use_tuned_params == true, "use_tuned_params default is true");
    check(cfg.run_tomography == false, "run_tomography default is false");
    check(cfg.validate_output == true, "validate_output default is true");
}

//==============================================================================
// Test 2: OptimizedPipeline construction
//==============================================================================

void test_pipeline_construction() {
    std::cout << "\n[Test 2] OptimizedPipeline construction\n";

    OptimizedPipeline pipe(4);
    check(pipe.num_qubits() == 4, "num_qubits() returns 4");
    check(pipe.dimension() == 16, "dimension() returns 16");

    PipelineConfig cfg;
    cfg.truncation_threshold = 1e-6;
    OptimizedPipeline pipe2(6, cfg);
    check(pipe2.num_qubits() == 6, "custom config: num_qubits() returns 6");
    check(pipe2.dimension() == 64, "custom config: dimension() returns 64");
    check(approx_equal(pipe2.config().truncation_threshold, 1e-6), "custom config: threshold is 1e-6");
}

//==============================================================================
// Test 3: Circuit analysis and strategy selection
//==============================================================================

void test_strategy_selection() {
    std::cout << "\n[Test 3] Strategy selection\n";

    // Gates only — should select Standard noise (no noise ops)
    {
        QuantumSequence seq;
        for (size_t q = 0; q < 4; ++q) {
            seq.operations.push_back(GateOp(GateType::H, q));
        }
        OptimizedPipeline pipe(4);
        PipelineStats stats = pipe.analyze(seq);
        check(stats.noise_strategy_used == NoiseStrategy::Standard,
              "Gates only → Standard noise strategy");
        check(stats.num_gates == 4, "Circuit has 4 gates");
        check(stats.num_noise_ops == 0, "Circuit has 0 noise ops");
    }

    // With noise → should select IterativeCompression
    {
        QuantumSequence seq = make_simple_circuit(4, 0.01);
        OptimizedPipeline pipe(4);
        PipelineStats stats = pipe.analyze(seq);
        check(stats.noise_strategy_used == NoiseStrategy::IterativeCompression,
              "Low noise → IterativeCompression");
        check(stats.num_noise_ops > 0, "Circuit has noise ops");
    }

    // Manual override
    {
        PipelineConfig cfg;
        cfg.noise_strategy = NoiseStrategy::DLRA;
        OptimizedPipeline pipe(4, cfg);

        QuantumSequence seq = make_simple_circuit(4, 0.01);
        PipelineStats stats = pipe.analyze(seq);
        check(stats.noise_strategy_used == NoiseStrategy::DLRA,
              "Manual override: DLRA noise strategy");
    }
}

//==============================================================================
// Test 4: Pipeline run - gates only
//==============================================================================

void test_gates_only() {
    std::cout << "\n[Test 4] Pipeline run - gates only\n";

    size_t nq = 2;
    MatrixXcd L = make_zero_state_L(nq);

    // Just Hadamard on qubit 0
    QuantumSequence seq;
    seq.operations.push_back(GateOp(GateType::H, size_t(0)));

    PipelineConfig cfg;
    cfg.validate_output = true;
    cfg.run_tomography = false;

    OptimizedPipeline pipe(nq, cfg);
    PipelineResult result = pipe.run(L, seq);

    check(result.L_final.rows() == 4, "L_final has dim=4 rows");
    check(result.L_final.cols() >= 1, "L_final has rank >= 1");
    check(result.stats.num_gates == 1, "stats: 1 gate");
    check(result.stats.num_noise_ops == 0, "stats: 0 noise ops");
    check(result.stats.total_time > 0, "stats: total_time > 0");

    // Validate density matrix
    MatrixXcd rho = result.L_final * result.L_final.adjoint();
    double trace = rho.trace().real();
    check(approx_equal(trace, 1.0, 1e-10), "Tr[rho] = 1");

    // H|0⟩ = |+⟩ = (|0⟩+|1⟩)/√2  →  ρ₀₀ = ρ₁₁ = 0.5
    check(approx_equal(rho(0, 0).real(), 0.5, 1e-10), "rho(0,0) = 0.5 (|+⟩ state)");
    check(approx_equal(rho(1, 1).real(), 0.5, 1e-10), "rho(1,1) = 0.5 (|+⟩ state)");
}

//==============================================================================
// Test 5: Pipeline run - with noise
//==============================================================================

void test_with_noise() {
    std::cout << "\n[Test 5] Pipeline run - with noise\n";

    size_t nq = 2;
    MatrixXcd L = make_zero_state_L(nq);

    QuantumSequence seq = make_simple_circuit(nq, 0.01);

    PipelineConfig cfg;
    cfg.validate_output = true;
    cfg.verbose = false;

    OptimizedPipeline pipe(nq, cfg);
    PipelineResult result = pipe.run(L, seq);

    check(result.L_final.rows() == 4, "L_final has dim=4 rows");
    check(result.stats.num_noise_ops > 0, "has noise ops");
    check(result.stats.total_time > 0, "total_time > 0");
    check(result.stats.noise_time >= 0, "noise_time >= 0");

    // Trace should be ~1
    MatrixXcd G = result.L_final.adjoint() * result.L_final;
    double trace = G.trace().real();
    check(approx_equal(trace, 1.0, 0.01), "Tr[rho] ≈ 1 after noise");
}

//==============================================================================
// Test 6: Pipeline run - each noise strategy
//==============================================================================

void test_noise_strategies() {
    std::cout << "\n[Test 6] Each noise strategy\n";

    size_t nq = 3;
    MatrixXcd L = make_zero_state_L(nq);
    QuantumSequence seq = make_simple_circuit(nq, 0.02);

    NoiseStrategy strategies[] = {
        NoiseStrategy::Standard,
        NoiseStrategy::IterativeCompression,
        NoiseStrategy::DLRA,
        NoiseStrategy::Sparse
    };
    const char* names[] = {"Standard", "IterComp", "DLRA", "Sparse"};

    for (int i = 0; i < 4; ++i) {
        PipelineConfig cfg;
        cfg.noise_strategy = strategies[i];
        cfg.truncation_strategy = TruncationStrategy::GramEigen;
        cfg.validate_output = true;
        cfg.verbose = false;

        OptimizedPipeline pipe(nq, cfg);
        PipelineResult result = pipe.run(L, seq);

        // Check trace
        MatrixXcd G = result.L_final.adjoint() * result.L_final;
        double trace = G.trace().real();

        std::string label = std::string(names[i]) + " noise: Tr≈1";
        check(approx_equal(trace, 1.0, 0.05), label);

        label = std::string(names[i]) + " noise: rank > 0";
        check(result.stats.final_rank > 0, label);
    }
}

//==============================================================================
// Test 7: Pipeline run - each truncation strategy
//==============================================================================

void test_truncation_strategies() {
    std::cout << "\n[Test 7] Each truncation strategy\n";

    size_t nq = 3;
    MatrixXcd L = make_zero_state_L(nq);
    QuantumSequence seq = make_simple_circuit(nq, 0.02);

    TruncationStrategy strategies[] = {
        TruncationStrategy::GramEigen,
        TruncationStrategy::CPDecomposition,
        TruncationStrategy::SVD
    };
    const char* names[] = {"GramEigen", "CP", "SVD"};

    for (int i = 0; i < 3; ++i) {
        PipelineConfig cfg;
        cfg.noise_strategy = NoiseStrategy::Standard;
        cfg.truncation_strategy = strategies[i];
        cfg.validate_output = true;
        cfg.verbose = false;

        OptimizedPipeline pipe(nq, cfg);
        PipelineResult result = pipe.run(L, seq);

        MatrixXcd G = result.L_final.adjoint() * result.L_final;
        double trace = G.trace().real();

        std::string label = std::string(names[i]) + " trunc: Tr≈1";
        check(approx_equal(trace, 1.0, 0.05), label);
    }
}

//==============================================================================
// Test 8: Empty circuit
//==============================================================================

void test_empty_circuit() {
    std::cout << "\n[Test 8] Empty circuit\n";

    size_t nq = 2;
    MatrixXcd L = make_zero_state_L(nq);
    QuantumSequence seq; // empty

    OptimizedPipeline pipe(nq);
    PipelineResult result = pipe.run(L, seq);

    check(result.L_final.rows() == 4, "L_final has correct dims");
    check(result.L_final.cols() == 1, "L_final rank = 1 (unchanged)");
    check(result.stats.num_gates == 0, "0 gates");
    check(result.stats.num_noise_ops == 0, "0 noise ops");

    // State should be unchanged
    double diff = (result.L_final - L).norm();
    check(diff < 1e-10, "State unchanged after empty circuit");
}

//==============================================================================
// Test 9: Single gate
//==============================================================================

void test_single_gate() {
    std::cout << "\n[Test 9] Single gate\n";

    size_t nq = 1;
    MatrixXcd L = make_zero_state_L(nq);

    QuantumSequence seq;
    seq.operations.push_back(GateOp(GateType::X, size_t(0)));

    OptimizedPipeline pipe(nq);
    PipelineResult result = pipe.run(L, seq);

    // X|0⟩ = |1⟩
    MatrixXcd rho = result.L_final * result.L_final.adjoint();
    check(approx_equal(rho(1, 1).real(), 1.0, 1e-10), "X|0⟩ = |1⟩: rho(1,1) = 1");
    check(approx_equal(rho(0, 0).real(), 0.0, 1e-10), "X|0⟩ = |1⟩: rho(0,0) = 0");
}

//==============================================================================
// Test 10: Validation
//==============================================================================

void test_validation() {
    std::cout << "\n[Test 10] Validation\n";

    size_t nq = 2;
    MatrixXcd L = make_zero_state_L(nq);
    QuantumSequence seq = make_simple_circuit(nq, 0.01);

    PipelineConfig cfg;
    cfg.validate_output = true;

    OptimizedPipeline pipe(nq, cfg);
    PipelineResult result = pipe.run(L, seq);

    check(result.stats.is_hermitian, "Output is Hermitian");
    check(result.stats.is_psd, "Output is PSD");
    check(result.stats.trace_deviation < 0.01, "Trace deviation < 0.01");

    // Direct validation
    bool valid = pipe.validate(result.L_final, 1e-4);
    check(valid, "pipe.validate() returns true");
}

//==============================================================================
// Test 11: PipelineStats summary
//==============================================================================

void test_stats_summary() {
    std::cout << "\n[Test 11] PipelineStats summary\n";

    size_t nq = 3;
    MatrixXcd L = make_zero_state_L(nq);
    QuantumSequence seq = make_simple_circuit(nq, 0.01);

    OptimizedPipeline pipe(nq);
    PipelineResult result = pipe.run(L, seq);

    std::string summary = result.stats.summary();
    check(!summary.empty(), "Summary is non-empty");
    check(summary.find("Pipeline Summary") != std::string::npos, "Summary contains 'Pipeline Summary'");
    check(summary.find("Noise strategy") != std::string::npos, "Summary contains strategy info");
}

//==============================================================================
// Test 12: strategy_description
//==============================================================================

void test_strategy_description() {
    std::cout << "\n[Test 12] strategy_description\n";

    size_t nq = 3;
    MatrixXcd L = make_zero_state_L(nq);
    QuantumSequence seq = make_simple_circuit(nq, 0.01);

    OptimizedPipeline pipe(nq);
    PipelineStats stats = pipe.analyze(seq);

    std::string desc = pipe.strategy_description(stats);
    check(!desc.empty(), "Description is non-empty");
}

//==============================================================================
// Test 13: run_optimized_pipeline convenience function
//==============================================================================

void test_convenience_function() {
    std::cout << "\n[Test 13] run_optimized_pipeline convenience\n";

    size_t nq = 2;
    MatrixXcd L = make_zero_state_L(nq);
    QuantumSequence seq = make_simple_circuit(nq, 0.01);

    PipelineResult result = run_optimized_pipeline(L, seq, nq);

    check(result.L_final.rows() == 4, "Convenience: correct dim");
    check(result.stats.final_rank > 0, "Convenience: rank > 0");
}

//==============================================================================
// Test 14: run_and_validate_pipeline
//==============================================================================

void test_validate_pipeline() {
    std::cout << "\n[Test 14] run_and_validate_pipeline\n";

    size_t nq = 2;
    MatrixXcd L = make_zero_state_L(nq);
    QuantumSequence seq = make_simple_circuit(nq, 0.01);

    auto [fidelity, result] = run_and_validate_pipeline(L, seq, nq);

    check(fidelity > 0.9, "Fidelity vs baseline > 0.9");
    check(result.stats.final_rank > 0, "Result has positive rank");
}

//==============================================================================
// Test 15: Circuit generators
//==============================================================================

void test_circuit_generators() {
    std::cout << "\n[Test 15] Circuit generators\n";

    // Random circuit
    QuantumSequence rand_seq = generate_random_circuit(4, 3, 0.01);
    check(rand_seq.operations.size() > 0, "Random circuit has operations");

    size_t gates = 0, noise = 0;
    for (const auto& op : rand_seq.operations) {
        if (std::holds_alternative<GateOp>(op)) ++gates;
        if (std::holds_alternative<NoiseOp>(op)) ++noise;
    }
    check(gates > 0, "Random circuit has gates");
    check(noise > 0, "Random circuit has noise");

    // QFT circuit
    QuantumSequence qft_seq = generate_qft_circuit(4, 0.01);
    check(qft_seq.operations.size() > 0, "QFT circuit has operations");

    // Noisy circuit
    QuantumSequence noisy_seq = generate_noisy_circuit(4, 3, 0.1);
    check(noisy_seq.operations.size() > 0, "Noisy circuit has operations");

    size_t n_gates = 0, n_noise = 0;
    for (const auto& op : noisy_seq.operations) {
        if (std::holds_alternative<GateOp>(op)) ++n_gates;
        if (std::holds_alternative<NoiseOp>(op)) ++n_noise;
    }
    double ratio = static_cast<double>(n_noise) / static_cast<double>(n_gates + n_noise);
    check(ratio > 0.5, "Noisy circuit has noise ratio > 50%");
}

//==============================================================================
// Test 16: PhaseBenchmark smoke test
//==============================================================================

void test_benchmark_smoke() {
    std::cout << "\n[Test 16] PhaseBenchmark smoke test\n";

    // Instead of calling run_single (which invokes all phases and one crashes),
    // test each strategy individually via OptimizedPipeline (which works in tests 4-14).
    // This isolates strategy issues from benchmark wrapper issues.

    size_t nq = 2;
    MatrixXcd L = make_zero_state_L(nq);
    QuantumSequence seq = generate_random_circuit(nq, 4, 0.01, 42);
    std::cout << "  Circuit: " << seq.operations.size() << " ops\n" << std::flush;

    // Test each strategy that run_single would use:
    struct StrategyTest {
        std::string name;
        NoiseStrategy noise;
        TruncationStrategy trunc;
    };
    std::vector<StrategyTest> strategies = {
        {"IterComp+GramEigen", NoiseStrategy::IterativeCompression, TruncationStrategy::GramEigen},
        {"DLRA+SVD",           NoiseStrategy::DLRA,                  TruncationStrategy::SVD},
        {"IterComp+CP",        NoiseStrategy::IterativeCompression, TruncationStrategy::CPDecomposition},
        {"Sparse+GramEigen",   NoiseStrategy::Sparse,               TruncationStrategy::GramEigen},
        {"Auto+Auto",          NoiseStrategy::Auto,                  TruncationStrategy::Auto},
    };

    size_t passed = 0;
    for (const auto& st : strategies) {
        try {
            std::cout << "  Testing " << st.name << "..." << std::flush;
            PipelineConfig cfg;
            cfg.noise_strategy = st.noise;
            cfg.truncation_strategy = st.trunc;
            cfg.gate_strategy = GateStrategy::RowParallel;
            cfg.truncation_threshold = 1e-4;
            cfg.validate_output = false;
            cfg.run_tomography = false;
            cfg.verbose = false;

            OptimizedPipeline pipe(nq, cfg);
            PipelineResult pr = pipe.run(L, seq);
            std::cout << " rank=" << pr.stats.final_rank
                      << " time=" << pr.stats.total_time << "s OK\n" << std::flush;
            ++passed;
        } catch (const std::exception& e) {
            std::cerr << " FAIL: " << e.what() << "\n" << std::flush;
        }
    }

    check(passed == strategies.size(), "All strategy combos ran without crash");

    // Now test run_single (the benchmark wrapper)
    BenchmarkConfig bcfg;
    bcfg.verbose = true;  // Verbose to see which phase fails
    bcfg.compute_fidelity = true;
    PhaseBenchmark bench(bcfg);

    try {
        std::cout << "  Running bench.run_single(2, 4, 0.01)...\n" << std::flush;
        auto results = bench.run_single(nq, 4, 0.01);
        std::cout << "  Got " << results.size() << " results\n" << std::flush;
        check(results.size() >= 2, "At least baseline + 1 phase");
    } catch (const std::exception& e) {
        std::cerr << "  bench.run_single FAIL: " << e.what() << "\n" << std::flush;
        check(false, "run_single threw exception");
    }
}

//==============================================================================
// Test 17: BenchmarkResult one_line
//==============================================================================

void test_result_one_line() {
    std::cout << "\n[Test 17] BenchmarkResult one_line\n";

    BenchmarkResult r;
    r.label = "TestLabel";
    r.num_qubits = 4;
    r.circuit_depth = 32;
    r.noise_probability = 0.05;
    r.elapsed_seconds = 1.234;
    r.final_rank = 8;
    r.peak_rank = 16;
    r.fidelity_vs_baseline = 0.999;
    r.is_valid_dm = true;

    std::string line = r.one_line();
    check(!line.empty(), "one_line is non-empty");
    check(line.find("TestLabel") != std::string::npos, "Contains label");
}

//==============================================================================
// Test 18: 4-qubit pipeline with noise (larger test)
//==============================================================================

void test_4qubit_noisy() {
    std::cout << "\n[Test 18] 4-qubit noisy pipeline\n";

    size_t nq = 4;
    MatrixXcd L = make_zero_state_L(nq);

    // Build a moderate circuit
    QuantumSequence seq;
    for (size_t layer = 0; layer < 3; ++layer) {
        for (size_t q = 0; q < nq; ++q) {
            seq.operations.push_back(GateOp(GateType::H, q));
            seq.operations.push_back(NoiseOp(NoiseType::DEPOLARIZING, q, 0.01));
        }
        for (size_t q = 0; q + 1 < nq; q += 2) {
            seq.operations.push_back(GateOp(GateType::CNOT, q, q + 1));
        }
    }

    PipelineConfig cfg;
    cfg.validate_output = true;
    cfg.verbose = false;

    OptimizedPipeline pipe(nq, cfg);
    PipelineResult result = pipe.run(L, seq);

    check(result.L_final.rows() == 16, "4q: dim = 16");
    check(result.stats.final_rank > 0, "4q: rank > 0");
    check(result.stats.is_valid_dm || result.stats.trace_deviation < 0.01,
          "4q: valid DM or trace close to 1");
    check(result.stats.total_time > 0, "4q: time > 0");
}

//==============================================================================
// Test 19: Timing breakdown consistency
//==============================================================================

void test_timing_breakdown() {
    std::cout << "\n[Test 19] Timing breakdown consistency\n";

    size_t nq = 3;
    MatrixXcd L = make_zero_state_L(nq);
    QuantumSequence seq = make_simple_circuit(nq, 0.02);

    PipelineConfig cfg;
    cfg.validate_output = true;
    cfg.verbose = false;

    OptimizedPipeline pipe(nq, cfg);
    PipelineResult result = pipe.run(L, seq);

    const auto& s = result.stats;
    double component_sum = s.gate_time + s.noise_time + s.truncation_time
                          + s.tomography_time + s.validation_time
                          + s.strategy_selection_time;

    // Component sum should be <= total (there's some overhead not tracked)
    check(component_sum <= s.total_time * 1.1, "Component times <= total (with margin)");
    check(s.total_time > 0, "Total time is positive");
}

//==============================================================================
// Test 20: Markdown summary from benchmark
//==============================================================================

void test_markdown_summary() {
    std::cout << "\n[Test 20] Markdown summary\n";

    // Build results manually to avoid re-running the full benchmark
    std::vector<BenchmarkResult> results;

    BenchmarkResult r1;
    r1.label = "Baseline";
    r1.num_qubits = 2;
    r1.circuit_depth = 24;
    r1.noise_probability = 0.01;
    r1.elapsed_seconds = 0.05;
    r1.final_rank = 3;
    r1.peak_rank = 4;
    r1.fidelity_vs_baseline = 1.0;
    r1.is_valid_dm = true;
    r1.trace_deviation = 1e-12;
    results.push_back(r1);

    BenchmarkResult r2;
    r2.label = "Phase1A-IterComp";
    r2.num_qubits = 2;
    r2.circuit_depth = 24;
    r2.noise_probability = 0.01;
    r2.elapsed_seconds = 0.03;
    r2.final_rank = 2;
    r2.peak_rank = 4;
    r2.fidelity_vs_baseline = 0.998;
    r2.is_valid_dm = true;
    r2.trace_deviation = 1e-10;
    results.push_back(r2);

    BenchmarkConfig cfg;
    PhaseBenchmark bench(cfg);

    std::string md = bench.markdown_summary(results);
    check(!md.empty(), "Markdown summary non-empty");
    check(md.find("Phase Benchmark Results") != std::string::npos ||
          md.find("Benchmark") != std::string::npos, "Contains title");
    check(md.find("Baseline") != std::string::npos, "Contains Baseline");
}

//==============================================================================
// Main
//==============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "Phase 6: Pipeline & Benchmark Tests\n";
    std::cout << "========================================\n";

    test_config_defaults();
    test_pipeline_construction();
    test_strategy_selection();
    test_gates_only();
    test_with_noise();
    test_noise_strategies();
    test_truncation_strategies();
    test_empty_circuit();
    test_single_gate();
    test_validation();
    test_stats_summary();
    test_strategy_description();
    test_convenience_function();
    test_validate_pipeline();
    test_circuit_generators();
    test_benchmark_smoke();
    test_result_one_line();
    test_4qubit_noisy();
    test_timing_breakdown();
    test_markdown_summary();

    std::cout << "\n========================================\n";
    std::cout << "Results: " << tests_passed << " passed, "
              << tests_failed << " failed out of "
              << (tests_passed + tests_failed) << " tests\n";
    std::cout << "========================================\n";

    return tests_failed;
}
