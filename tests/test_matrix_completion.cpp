/**
 * @file test_matrix_completion.cpp
 * @brief Tests for Phase 5: Low-Rank Matrix Completion & Quantum State Tomography
 * 
 * Validates:
 * - Pauli utility functions (pauli_string_matrix, enumerate_pauli_strings, etc.)
 * - Density matrix constraint enforcement
 * - Matrix completion from Pauli measurements (SVT + AltProj solvers)
 * - 2-RDM completion from partial elements
 * - Measurement suggestion (leverage-score heuristic)
 * - Compressed tomography pipeline (from L factor)
 * - Adaptive measurement selection
 * - Denoising via low-rank projection
 * - Fidelity and trace distance
 * - Solver comparison (SVT vs AlternatingProjection)
 * - Round-trip: L → ρ → partial measurements → completion → ρ' → L' → fidelity
 */

#include "matrix_completion.h"
#include "simulator.h"
#include <set>
#include "gates_and_noise.h"
#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <map>
#include <random>

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

// Helper: create |0...0⟩ state as L factor
MatrixXcd make_zero_state_L(size_t num_qubits) {
    size_t dim = static_cast<size_t>(1) << num_qubits;
    MatrixXcd L = MatrixXcd::Zero(dim, 1);
    L(0, 0) = Complex(1.0, 0.0);
    return L;
}

// Helper: create |+⟩ = (|0⟩+|1⟩)/√2 state as L factor (1 qubit)
MatrixXcd make_plus_state_L() {
    MatrixXcd L(2, 1);
    L(0, 0) = Complex(INV_SQRT2, 0.0);
    L(1, 0) = Complex(INV_SQRT2, 0.0);
    return L;
}

// Helper: create Bell state (|00⟩+|11⟩)/√2 as L factor (2 qubits)
MatrixXcd make_bell_state_L() {
    MatrixXcd L(4, 1);
    L(0, 0) = Complex(INV_SQRT2, 0.0);
    L(1, 0) = Complex(0.0, 0.0);
    L(2, 0) = Complex(0.0, 0.0);
    L(3, 0) = Complex(INV_SQRT2, 0.0);
    return L;
}

// Helper: create maximally mixed state ρ = I/d as L factor
MatrixXcd make_mixed_state_L(size_t num_qubits) {
    size_t dim = static_cast<size_t>(1) << num_qubits;
    MatrixXcd L = MatrixXcd::Zero(dim, dim);
    double coeff = 1.0 / std::sqrt(static_cast<double>(dim));
    for (size_t i = 0; i < dim; ++i) {
        L(i, i) = Complex(coeff, 0.0);
    }
    return L;
}

}  // namespace

int main() {
    try {
        std::cout << "=== Phase 5: Matrix Completion & Tomography Tests ===" << std::endl;

        //======================================================================
        // Section 1: Pauli Utilities
        //======================================================================
        std::cout << "\n--- Section 1: Pauli Utilities ---" << std::endl;

        // Test 1.1: Single-qubit Pauli matrices
        {
            MatrixXcd I = pauli_string_matrix("I", 1);
            MatrixXcd X = pauli_string_matrix("X", 1);
            MatrixXcd Y = pauli_string_matrix("Y", 1);
            MatrixXcd Z = pauli_string_matrix("Z", 1);

            // I should be identity
            check(approx_equal((I - MatrixXcd::Identity(2, 2)).norm(), 0.0, 1e-12),
                  "1.1a: Pauli I is identity");

            // X² = Y² = Z² = I
            check(approx_equal((X * X - MatrixXcd::Identity(2, 2)).norm(), 0.0, 1e-12),
                  "1.1b: X^2 = I");
            check(approx_equal((Y * Y - MatrixXcd::Identity(2, 2)).norm(), 0.0, 1e-12),
                  "1.1c: Y^2 = I");
            check(approx_equal((Z * Z - MatrixXcd::Identity(2, 2)).norm(), 0.0, 1e-12),
                  "1.1d: Z^2 = I");

            // Anticommutation: {X,Y} = 0, {Y,Z} = 0, {X,Z} = 0
            check(approx_equal((X * Y + Y * X).norm(), 0.0, 1e-12),
                  "1.1e: {X,Y} = 0");
            check(approx_equal((Y * Z + Z * Y).norm(), 0.0, 1e-12),
                  "1.1f: {Y,Z} = 0");
            check(approx_equal((X * Z + Z * X).norm(), 0.0, 1e-12),
                  "1.1g: {X,Z} = 0");

            // XY = iZ
            MatrixXcd XY = X * Y;
            MatrixXcd iZ = Complex(0, 1) * Z;
            check(approx_equal((XY - iZ).norm(), 0.0, 1e-12),
                  "1.1h: XY = iZ");
        }

        // Test 1.2: Two-qubit Pauli string
        {
            MatrixXcd ZI = pauli_string_matrix("ZI", 2);
            MatrixXcd IZ = pauli_string_matrix("IZ", 2);
            MatrixXcd ZZ = pauli_string_matrix("ZZ", 2);

            // ZI ⊗ IZ should equal ZZ
            check(approx_equal((ZI * IZ - ZZ).norm(), 0.0, 1e-12),
                  "1.2a: ZI * IZ = ZZ");

            // ZI is 4x4 diagonal with entries (1,1,-1,-1) ← Z on qubit 0, I on qubit 1
            check(approx_equal(ZI(0, 0).real(), 1.0) && approx_equal(ZI(1, 1).real(), 1.0) &&
                  approx_equal(ZI(2, 2).real(), -1.0) && approx_equal(ZI(3, 3).real(), -1.0),
                  "1.2b: ZI diagonal correct");
        }

        // Test 1.3: Enumerate Pauli strings
        {
            auto ps1 = enumerate_pauli_strings(1);
            check(ps1.size() == 4, "1.3a: 1-qubit has 4 Pauli strings");

            auto ps2 = enumerate_pauli_strings(2);
            check(ps2.size() == 16, "1.3b: 2-qubit has 16 Pauli strings");

            auto ps3 = enumerate_pauli_strings(3);
            check(ps3.size() == 64, "1.3c: 3-qubit has 64 Pauli strings");
        }

        // Test 1.4: Pauli expectation from L
        {
            // |0⟩ state: ⟨Z⟩ = 1, ⟨X⟩ = 0, ⟨Y⟩ = 0
            MatrixXcd L0 = make_zero_state_L(1);
            MatrixXcd Z1 = pauli_string_matrix("Z", 1);
            MatrixXcd X1 = pauli_string_matrix("X", 1);
            MatrixXcd Y1 = pauli_string_matrix("Y", 1);

            double expZ = pauli_expectation_from_L(L0, Z1);
            double expX = pauli_expectation_from_L(L0, X1);
            double expY = pauli_expectation_from_L(L0, Y1);

            check(approx_equal(expZ, 1.0, 1e-10), "1.4a: |0> has <Z> = 1");
            check(approx_equal(expX, 0.0, 1e-10), "1.4b: |0> has <X> = 0");
            check(approx_equal(expY, 0.0, 1e-10), "1.4c: |0> has <Y> = 0");

            // |+⟩ state: ⟨Z⟩ = 0, ⟨X⟩ = 1
            MatrixXcd Lp = make_plus_state_L();
            double expZp = pauli_expectation_from_L(Lp, Z1);
            double expXp = pauli_expectation_from_L(Lp, X1);

            check(approx_equal(expZp, 0.0, 1e-10), "1.4d: |+> has <Z> = 0");
            check(approx_equal(expXp, 1.0, 1e-10), "1.4e: |+> has <X> = 1");

            // Bell state: ⟨ZZ⟩ = 1, ⟨ZI⟩ = 0, ⟨IZ⟩ = 0
            MatrixXcd Lb = make_bell_state_L();
            MatrixXcd ZZ = pauli_string_matrix("ZZ", 2);
            MatrixXcd ZI = pauli_string_matrix("ZI", 2);
            MatrixXcd XX = pauli_string_matrix("XX", 2);

            double expZZb = pauli_expectation_from_L(Lb, ZZ);
            double expZIb = pauli_expectation_from_L(Lb, ZI);
            double expXXb = pauli_expectation_from_L(Lb, XX);

            check(approx_equal(expZZb, 1.0, 1e-10), "1.4f: Bell has <ZZ> = 1");
            check(approx_equal(expZIb, 0.0, 1e-10), "1.4g: Bell has <ZI> = 0");
            check(approx_equal(expXXb, 1.0, 1e-10), "1.4h: Bell has <XX> = 1");
        }

        //======================================================================
        // Section 2: Density Matrix Constraints
        //======================================================================
        std::cout << "\n--- Section 2: DM Constraint Enforcement ---" << std::endl;

        {
            CompletionConfig config;
            MatrixCompletion mc(1, config);

            // Test 2.1: Already valid density matrix should be unchanged
            {
                MatrixXcd rho = MatrixXcd::Zero(2, 2);
                rho(0, 0) = Complex(0.7, 0.0);
                rho(1, 1) = Complex(0.3, 0.0);
                MatrixXcd result = mc.enforce_dm_constraints(rho);

                check(approx_equal(result.trace().real(), 1.0, 1e-12),
                      "2.1a: Valid DM trace preserved");
                check(approx_equal((result - rho).norm(), 0.0, 1e-10),
                      "2.1b: Valid DM unchanged");
            }

            // Test 2.2: Non-PSD matrix should be projected to PSD
            {
                MatrixXcd rho(2, 2);
                rho << Complex(0.6, 0), Complex(0.5, 0),
                       Complex(0.5, 0), Complex(0.4, 0);
                // This has eigenvalues 1.0 and 0.0, but with off-diag 0.5
                // let's make it non-PSD explicitly
                MatrixXcd bad(2, 2);
                bad << Complex(0.8, 0), Complex(0.9, 0),
                       Complex(0.9, 0), Complex(0.2, 0);
                // eigenvalues: ~1.62, ~-0.62 → has a negative eigenvalue

                MatrixXcd result = mc.enforce_dm_constraints(bad);

                // Should be PSD now
                Eigen::SelfAdjointEigenSolver<MatrixXcd> solver(result);
                bool psd = (solver.eigenvalues().array() >= -1e-12).all();
                check(psd, "2.2a: Non-PSD projected to PSD");
                check(approx_equal(result.trace().real(), 1.0, 1e-12),
                      "2.2b: Non-PSD projected to trace 1");
            }

            // Test 2.3: Non-Hermitian matrix should be Hermitianized
            {
                MatrixXcd bad(2, 2);
                bad << Complex(0.5, 0), Complex(0.3, 0.2),
                       Complex(0.1, -0.1), Complex(0.5, 0);
                // Not Hermitian: bad(0,1) ≠ conj(bad(1,0))

                MatrixXcd result = mc.enforce_dm_constraints(bad);

                // Should be Hermitian
                double herm_err = (result - result.adjoint()).norm();
                check(herm_err < 1e-12, "2.3: Non-Hermitian projected to Hermitian");
            }
        }

        //======================================================================
        // Section 3: Matrix Completion from Pauli Measurements
        //======================================================================
        std::cout << "\n--- Section 3: Matrix Completion ---" << std::endl;

        // Test 3.1: Complete |0⟩ from all 1-qubit Pauli measurements
        {
            CompletionConfig config;
            config.solver = CompletionSolver::AlternatingProjection;
            config.max_iterations = 500;
            config.tolerance = 1e-8;
            MatrixCompletion mc(1, config);

            // Full measurements for |0⟩
            std::map<std::string, double> measurements;
            measurements["I"] = 1.0;  // Tr[ρ] = 1
            measurements["X"] = 0.0;
            measurements["Y"] = 0.0;
            measurements["Z"] = 1.0;

            auto [rho, stats] = mc.complete_from_paulis(measurements);

            // Should recover |0⟩⟨0|
            MatrixXcd expected(2, 2);
            expected << Complex(1, 0), Complex(0, 0),
                        Complex(0, 0), Complex(0, 0);

            double error = (rho - expected).norm();
            check(error < 0.01, "3.1a: |0> recovered from full Pauli (err=" +
                  std::to_string(error) + ")");
            check(stats.recovered_rank == 1, "3.1b: Rank-1 recovered");
        }

        // Test 3.2: Complete |+⟩ from all 1-qubit Pauli measurements
        {
            CompletionConfig config;
            config.solver = CompletionSolver::AlternatingProjection;
            config.max_iterations = 500;
            config.tolerance = 1e-8;
            MatrixCompletion mc(1, config);

            std::map<std::string, double> measurements;
            measurements["I"] = 1.0;
            measurements["X"] = 1.0;
            measurements["Y"] = 0.0;
            measurements["Z"] = 0.0;

            auto [rho, stats] = mc.complete_from_paulis(measurements);

            // Should recover |+⟩⟨+| = [[0.5, 0.5],[0.5, 0.5]]
            MatrixXcd expected(2, 2);
            expected << Complex(0.5, 0), Complex(0.5, 0),
                        Complex(0.5, 0), Complex(0.5, 0);

            double error = (rho - expected).norm();
            check(error < 0.01, "3.2: |+> recovered from full Pauli (err=" +
                  std::to_string(error) + ")");
        }

        // Test 3.3: Complete Bell state from 2-qubit Pauli measurements
        {
            CompletionConfig config;
            config.solver = CompletionSolver::AlternatingProjection;
            config.max_iterations = 1000;
            config.tolerance = 1e-8;
            MatrixCompletion mc(2, config);

            // Measure all 16 Pauli operators on Bell state
            MatrixXcd Lb = make_bell_state_L();
            std::map<std::string, double> measurements;
            auto all_paulis = enumerate_pauli_strings(2);
            for (const auto& label : all_paulis) {
                MatrixXcd P = pauli_string_matrix(label, 2);
                measurements[label] = pauli_expectation_from_L(Lb, P);
            }

            auto [rho, stats] = mc.complete_from_paulis(measurements);

            // Compare to actual Bell state density matrix
            MatrixXcd rho_exact = Lb * Lb.adjoint();
            double error = (rho - rho_exact).norm();
            check(error < 0.01, "3.3a: Bell state recovered from full Pauli (err=" +
                  std::to_string(error) + ")");
            check(stats.recovered_rank <= 2, "3.3b: Low rank recovered (rank=" +
                  std::to_string(stats.recovered_rank) + ")");
        }

        // Test 3.4: SVT solver — complete |0⟩
        {
            CompletionConfig config;
            config.solver = CompletionSolver::SVDThreshold;
            config.max_iterations = 500;
            config.tolerance = 1e-8;
            MatrixCompletion mc(1, config);

            std::map<std::string, double> measurements;
            measurements["I"] = 1.0;
            measurements["X"] = 0.0;
            measurements["Y"] = 0.0;
            measurements["Z"] = 1.0;

            auto [rho, stats] = mc.complete_from_paulis(measurements);

            MatrixXcd expected(2, 2);
            expected << Complex(1, 0), Complex(0, 0),
                        Complex(0, 0), Complex(0, 0);

            double error = (rho - expected).norm();
            check(error < 0.05, "3.4: SVT solver recovers |0> (err=" +
                  std::to_string(error) + ")");
        }

        // Test 3.5: Partial measurements — can we still recover?
        {
            CompletionConfig config;
            config.solver = CompletionSolver::AlternatingProjection;
            config.max_iterations = 1000;
            config.tolerance = 1e-8;
            MatrixCompletion mc(2, config);

            // Bell state, but only give 12 out of 16 measurements (75%)
            MatrixXcd Lb = make_bell_state_L();
            auto all_paulis = enumerate_pauli_strings(2);

            // Deliberately omit some higher-weight operators
            std::set<std::string> omitted = {"XY", "YX", "YY", "YZ"};

            std::map<std::string, double> measurements;
            for (const auto& label : all_paulis) {
                if (omitted.count(label) > 0) continue;
                MatrixXcd P = pauli_string_matrix(label, 2);
                measurements[label] = pauli_expectation_from_L(Lb, P);
            }

            auto [rho, stats] = mc.complete_from_paulis(measurements);
            MatrixXcd rho_exact = Lb * Lb.adjoint();
            double error = (rho - rho_exact).norm();

            // With 75% measurements on a rank-1 state, should be close
            check(error < 0.15, "3.5: Partial measurements (75%) still recover Bell (err=" +
                  std::to_string(error) + ")");
        }

        //======================================================================
        // Section 4: 2-RDM Completion
        //======================================================================
        std::cout << "\n--- Section 4: 2-RDM Completion ---" << std::endl;

        {
            CompletionConfig config;
            config.solver = CompletionSolver::AlternatingProjection;
            config.max_iterations = 500;
            config.tolerance = 1e-8;
            MatrixCompletion mc(2, config);

            // Create a known 2-RDM (4x4 for 2-qubit system)
            MatrixXcd rho_true(4, 4);
            rho_true << Complex(0.5, 0),  Complex(0, 0),   Complex(0, 0),   Complex(0.5, 0),
                        Complex(0, 0),    Complex(0, 0),   Complex(0, 0),   Complex(0, 0),
                        Complex(0, 0),    Complex(0, 0),   Complex(0, 0),   Complex(0, 0),
                        Complex(0.5, 0),  Complex(0, 0),   Complex(0, 0),   Complex(0.5, 0);
            // This is the Bell state density matrix

            // Provide partial elements
            std::vector<std::tuple<size_t, size_t, Complex>> partial;
            partial.emplace_back(0, 0, rho_true(0, 0));
            partial.emplace_back(3, 3, rho_true(3, 3));
            partial.emplace_back(1, 1, rho_true(1, 1));
            partial.emplace_back(2, 2, rho_true(2, 2));
            partial.emplace_back(0, 3, rho_true(0, 3));

            auto [rho_completed, stats] = mc.complete_2rdm(partial, 4);

            // Should recover the off-diagonal structure
            double error = (rho_completed - rho_true).norm();
            check(error < 0.2, "4.1: 2-RDM completion from partial elements (err=" +
                  std::to_string(error) + ")");
            check(approx_equal(rho_completed.trace().real(), 1.0, 1e-6),
                  "4.2: 2-RDM has trace 1");
        }

        //======================================================================
        // Section 5: Measurement Suggestion
        //======================================================================
        std::cout << "\n--- Section 5: Measurement Suggestion ---" << std::endl;

        {
            CompletionConfig config;
            MatrixCompletion mc(2, config);

            // Test 5.1: Suggest measurements for 2-qubit system
            auto suggestions = mc.suggest_measurements(10);
            check(suggestions.size() == 10, "5.1a: Got 10 suggestions");

            // All suggestions should be valid 2-character Pauli strings
            bool all_valid = true;
            for (const auto& s : suggestions) {
                if (s.size() != 2) { all_valid = false; break; }
                for (char c : s) {
                    if (c != 'I' && c != 'X' && c != 'Y' && c != 'Z') {
                        all_valid = false;
                        break;
                    }
                }
            }
            check(all_valid, "5.1b: All suggestions are valid Pauli strings");

            // Test 5.2: Suggestions should include weight-1 operators first
            // For 2 qubits: XI, YI, ZI, IX, IY, IZ (6 weight-1 operators)
            std::set<std::string> weight1 = {"XI", "YI", "ZI", "IX", "IY", "IZ"};
            size_t w1_count = 0;
            for (const auto& s : suggestions) {
                if (weight1.count(s) > 0) ++w1_count;
            }
            check(w1_count == 6, "5.2: All 6 weight-1 operators suggested (got " +
                  std::to_string(w1_count) + ")");

            // Test 5.3: Already measured operators should not be re-suggested
            std::map<std::string, double> already = {{"ZI", 0.5}, {"IZ", 0.3}};
            auto suggestions2 = mc.suggest_measurements(6, already);
            bool no_dups = true;
            for (const auto& s : suggestions2) {
                if (already.count(s) > 0) { no_dups = false; break; }
            }
            check(no_dups, "5.3: Already-measured operators excluded from suggestions");
        }

        //======================================================================
        // Section 6: Compressed Tomography Pipeline
        //======================================================================
        std::cout << "\n--- Section 6: Compressed Tomography ---" << std::endl;

        // Test 6.1: Full tomography (100% measurements) — should be exact
        {
            CompletionConfig config;
            config.solver = CompletionSolver::AlternatingProjection;
            config.max_iterations = 1000;
            config.tolerance = 1e-8;
            QuantumStateTomography tomo(1, config);

            MatrixXcd L0 = make_zero_state_L(1);
            auto [rho, stats] = tomo.compressed_tomography_from_L(L0, 1.0);

            MatrixXcd rho_exact = L0 * L0.adjoint();
            double error = (rho - rho_exact).norm();
            check(error < 0.01, "6.1: Full tomography recovers |0> (err=" +
                  std::to_string(error) + ")");
        }

        // Test 6.2: Compressed tomography (75% measurements) for Bell state (2-qubit)
        // 75% of 16 Paulis = 12 measurements — good compressed ratio for rank-1
        {
            CompletionConfig config;
            config.solver = CompletionSolver::AlternatingProjection;
            config.max_iterations = 1000;
            config.tolerance = 1e-8;
            QuantumStateTomography tomo(2, config);

            MatrixXcd Lb = make_bell_state_L();
            auto [rho, stats] = tomo.compressed_tomography_from_L(Lb, 0.75);

            MatrixXcd rho_exact = Lb * Lb.adjoint();
            double error = (rho - rho_exact).norm();

            // With 75% measurements (12 out of 16) on a rank-1 state, should recover well
            check(error < 0.3, "6.2: 75% compressed tomography 2-qubit Bell (err=" +
                  std::to_string(error) + ")");
        }

        // Test 6.3: Tomography with measurement oracle
        {
            CompletionConfig config;
            config.solver = CompletionSolver::AlternatingProjection;
            config.max_iterations = 1000;
            config.tolerance = 1e-8;
            QuantumStateTomography tomo(1, config);

            // Oracle: simulate measuring |+⟩ state
            MatrixXcd Lp = make_plus_state_L();
            MatrixXcd rho_plus = Lp * Lp.adjoint();

            auto measure_fn = [&](const std::string& label) -> double {
                MatrixXcd P = pauli_string_matrix(label, 1);
                return pauli_expectation_from_L(Lp, P);
            };

            auto [rho, stats] = tomo.compressed_tomography(measure_fn, 1.0);
            double error = (rho - rho_plus).norm();
            check(error < 0.01, "6.3: Oracle tomography recovers |+> (err=" +
                  std::to_string(error) + ")");
        }

        // Test 6.4: Two-qubit compressed tomography (Bell state)
        {
            CompletionConfig config;
            config.solver = CompletionSolver::AlternatingProjection;
            config.max_iterations = 1000;
            config.tolerance = 1e-8;
            QuantumStateTomography tomo(2, config);

            MatrixXcd Lb = make_bell_state_L();
            auto [rho, stats] = tomo.compressed_tomography_from_L(Lb, 1.0);

            MatrixXcd rho_exact = Lb * Lb.adjoint();
            double error = (rho - rho_exact).norm();
            check(error < 0.05, "6.4: 2-qubit full tomography Bell state (err=" +
                  std::to_string(error) + ")");
        }

        //======================================================================
        // Section 7: Adaptive Measurements
        //======================================================================
        std::cout << "\n--- Section 7: Adaptive Measurements ---" << std::endl;

        {
            CompletionConfig config;
            QuantumStateTomography tomo(2, config);

            // Start with a rough estimate (maximally mixed)
            size_t dim = 4;
            MatrixXcd estimate = MatrixXcd::Identity(dim, dim) / 4.0;

            std::map<std::string, double> already;
            already["II"] = 1.0;
            already["ZI"] = 0.0;
            already["IZ"] = 0.0;

            auto suggestions = tomo.adaptive_measurements(5, estimate, already);

            check(suggestions.size() == 5, "7.1: Got 5 adaptive suggestions");

            // Should not repeat already measured
            bool no_repeats = true;
            for (const auto& s : suggestions) {
                if (already.count(s) > 0) { no_repeats = false; break; }
            }
            check(no_repeats, "7.2: Adaptive suggestions don't repeat measurements");
        }

        //======================================================================
        // Section 8: Denoising
        //======================================================================
        std::cout << "\n--- Section 8: Denoising ---" << std::endl;

        {
            CompletionConfig config;
            QuantumStateTomography tomo(1, config);

            // Create a noisy version of |0⟩⟨0|
            MatrixXcd rho_true(2, 2);
            rho_true << Complex(1, 0), Complex(0, 0),
                        Complex(0, 0), Complex(0, 0);

            // Add noise
            std::mt19937 rng(42);
            std::normal_distribution<double> noise_dist(0.0, 0.05);
            MatrixXcd rho_noisy = rho_true;
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    rho_noisy(i, j) += Complex(noise_dist(rng), noise_dist(rng));
                }
            }

            MatrixXcd rho_denoised = tomo.denoise(rho_noisy, 1);

            double error_before = (rho_noisy - rho_true).norm();
            double error_after = (rho_denoised - rho_true).norm();

            check(error_after < error_before, "8.1: Denoising reduces error (before=" +
                  std::to_string(error_before) + " after=" + std::to_string(error_after) + ")");

            // Denoised should be valid DM
            check(approx_equal(rho_denoised.trace().real(), 1.0, 1e-10),
                  "8.2: Denoised has trace 1");

            Eigen::SelfAdjointEigenSolver<MatrixXcd> solver(rho_denoised);
            bool psd = (solver.eigenvalues().array() >= -1e-10).all();
            check(psd, "8.3: Denoised is PSD");
        }

        // Test 8.4: Auto-rank detection
        {
            CompletionConfig config;
            QuantumStateTomography tomo(2, config);

            // Rank-2 mixed state: ρ = 0.7|00⟩⟨00| + 0.3|11⟩⟨11|
            MatrixXcd rho_true(4, 4);
            rho_true.setZero();
            rho_true(0, 0) = Complex(0.7, 0);
            rho_true(3, 3) = Complex(0.3, 0);

            // Add noise and denoise with auto-rank detection
            std::mt19937 rng(123);
            std::normal_distribution<double> noise_dist(0.0, 0.02);
            MatrixXcd rho_noisy = rho_true;
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    rho_noisy(i, j) += Complex(noise_dist(rng), noise_dist(rng));
                }
            }

            MatrixXcd rho_denoised = tomo.denoise(rho_noisy, 0);  // auto-rank

            double error = (rho_denoised - rho_true).norm();
            check(error < 0.3, "8.4: Auto-rank denoising (err=" + std::to_string(error) + ")");
        }

        //======================================================================
        // Section 9: Fidelity and Trace Distance
        //======================================================================
        std::cout << "\n--- Section 9: Fidelity & Trace Distance ---" << std::endl;

        {
            // Test 9.1: Fidelity of identical states = 1
            MatrixXcd rho0(2, 2);
            rho0 << Complex(1, 0), Complex(0, 0),
                    Complex(0, 0), Complex(0, 0);

            double f_same = QuantumStateTomography::fidelity(rho0, rho0);
            check(approx_equal(f_same, 1.0, 1e-6), "9.1: F(ρ, ρ) = 1");

            // Test 9.2: Fidelity of orthogonal pure states = 0
            MatrixXcd rho1(2, 2);
            rho1 << Complex(0, 0), Complex(0, 0),
                    Complex(0, 0), Complex(1, 0);

            double f_orth = QuantumStateTomography::fidelity(rho0, rho1);
            check(approx_equal(f_orth, 0.0, 1e-6), "9.2: F(|0>, |1>) = 0");

            // Test 9.3: Fidelity of pure state with mixed state
            MatrixXcd rho_mixed = MatrixXcd::Identity(2, 2) / 2.0;
            double f_mixed = QuantumStateTomography::fidelity(rho0, rho_mixed);
            check(approx_equal(f_mixed, 0.5, 1e-6), "9.3: F(|0>, I/2) = 0.5");

            // Test 9.4: Trace distance of identical states = 0
            double d_same = QuantumStateTomography::trace_distance(rho0, rho0);
            check(approx_equal(d_same, 0.0, 1e-6), "9.4: D(ρ, ρ) = 0");

            // Test 9.5: Trace distance of orthogonal pure states = 1
            double d_orth = QuantumStateTomography::trace_distance(rho0, rho1);
            check(approx_equal(d_orth, 1.0, 1e-6), "9.5: D(|0>, |1>) = 1");

            // Test 9.6: Trace distance of pure vs mixed = 0.5
            double d_mixed = QuantumStateTomography::trace_distance(rho0, rho_mixed);
            check(approx_equal(d_mixed, 0.5, 1e-6), "9.6: D(|0>, I/2) = 0.5");
        }

        //======================================================================
        // Section 10: L ↔ ρ Round-Trip
        //======================================================================
        std::cout << "\n--- Section 10: L <-> rho Round-Trip ---" << std::endl;

        {
            CompletionConfig config;
            MatrixCompletion mc(2, config);

            // Test 10.1: L → ρ → L' → ρ' preserves state
            MatrixXcd L_orig = make_bell_state_L();
            MatrixXcd rho = L_orig * L_orig.adjoint();
            MatrixXcd L_recovered = mc.rho_to_L(rho);
            MatrixXcd rho_recovered = L_recovered * L_recovered.adjoint();

            double error = (rho - rho_recovered).norm();
            check(error < 1e-10, "10.1: L -> rho -> L' -> rho' round-trip (err=" +
                  std::to_string(error) + ")");

            // Test 10.2: Rank is preserved
            check(L_recovered.cols() == 1, "10.2: Rank preserved (got " +
                  std::to_string(L_recovered.cols()) + ")");
        }

        {
            CompletionConfig config;
            MatrixCompletion mc(2, config);

            // Test 10.3: Mixed state round-trip
            MatrixXcd L_mixed = make_mixed_state_L(2);
            MatrixXcd rho_mixed = L_mixed * L_mixed.adjoint();
            MatrixXcd L_rec = mc.rho_to_L(rho_mixed);
            MatrixXcd rho_rec = L_rec * L_rec.adjoint();

            double error = (rho_mixed - rho_rec).norm();
            check(error < 1e-10, "10.3: Mixed state round-trip (err=" +
                  std::to_string(error) + ")");
        }

        //======================================================================
        // Section 11: End-to-End — LRET Simulation → Tomography
        //======================================================================
        std::cout << "\n--- Section 11: LRET Simulation -> Tomography ---" << std::endl;

        {
            // Simulate a simple 2-qubit circuit with noise, then use tomography
            // to reconstruct the density matrix.
            const size_t nq = 2;
            MatrixXcd L = make_zero_state_L(nq);

            // Apply H on qubit 0 → |+0⟩
            L = apply_gate_to_L(L, GateOp(GateType::H, size_t(0)), nq);

            // Apply CNOT(0,1) → Bell state
            L = apply_gate_to_L(L, GateOp(GateType::CNOT, size_t(0), size_t(1)), nq);

            // Apply light depolarizing noise
            L = apply_noise_to_L(L, NoiseOp(NoiseType::DEPOLARIZING, size_t(0), 0.01), nq);

            // Truncate to keep things clean
            L = truncate_L(L, 1e-6);

            // Get the "true" density matrix for comparison
            MatrixXcd rho_true = L * L.adjoint();

            // Perform compressed tomography at 100% measurements
            CompletionConfig config;
            config.solver = CompletionSolver::AlternatingProjection;
            config.max_iterations = 1000;
            config.tolerance = 1e-8;
            QuantumStateTomography tomo(nq, config);

            auto [rho_recovered, stats] = tomo.compressed_tomography_from_L(L, 1.0);

            // Compute fidelity
            double fid = QuantumStateTomography::fidelity(rho_true, rho_recovered);
            double tdist = QuantumStateTomography::trace_distance(rho_true, rho_recovered);

            check(fid > 0.99, "11.1: LRET->Tomography fidelity > 99% (F=" +
                  std::to_string(fid) + ")");
            check(tdist < 0.05, "11.2: LRET->Tomography trace distance < 0.05 (D=" +
                  std::to_string(tdist) + ")");

            std::cout << "  [INFO] Tomography stats: " << stats.iterations << " iters, "
                      << stats.elapsed_seconds << "s, rank=" << stats.recovered_rank << std::endl;
        }

        //======================================================================
        // Section 12: Solver Comparison (SVT vs AltProj)
        //======================================================================
        std::cout << "\n--- Section 12: Solver Comparison ---" << std::endl;

        {
            const size_t nq = 2;
            MatrixXcd Lb = make_bell_state_L();
            MatrixXcd rho_true = Lb * Lb.adjoint();

            // Full Pauli measurements
            std::map<std::string, double> measurements;
            auto all_paulis = enumerate_pauli_strings(nq);
            for (const auto& label : all_paulis) {
                MatrixXcd P = pauli_string_matrix(label, nq);
                measurements[label] = pauli_expectation_from_L(Lb, P);
            }

            // Solver 1: SVT
            {
                CompletionConfig config;
                config.solver = CompletionSolver::SVDThreshold;
                config.max_iterations = 500;
                config.tolerance = 1e-8;
                MatrixCompletion mc(nq, config);
                auto [rho, stats] = mc.complete_from_paulis(measurements);

                double error = (rho - rho_true).norm();
                std::cout << "  [INFO] SVT: err=" << error << " iters=" << stats.iterations
                          << " time=" << stats.elapsed_seconds << "s" << std::endl;
                check(error < 0.1, "12.1: SVT recovers Bell state (err=" +
                      std::to_string(error) + ")");
            }

            // Solver 2: Alternating Projection
            {
                CompletionConfig config;
                config.solver = CompletionSolver::AlternatingProjection;
                config.max_iterations = 500;
                config.tolerance = 1e-8;
                MatrixCompletion mc(nq, config);
                auto [rho, stats] = mc.complete_from_paulis(measurements);

                double error = (rho - rho_true).norm();
                std::cout << "  [INFO] AltProj: err=" << error << " iters=" << stats.iterations
                          << " time=" << stats.elapsed_seconds << "s" << std::endl;
                check(error < 0.1, "12.2: AltProj recovers Bell state (err=" +
                      std::to_string(error) + ")");
            }
        }

        //======================================================================
        // Summary
        //======================================================================
        std::cout << "\n========================================" << std::endl;
        std::cout << "Phase 5 Tests: " << tests_passed << "/" << (tests_passed + tests_failed)
                  << " passed" << std::endl;
        std::cout << "========================================" << std::endl;

        return tests_failed > 0 ? 1 : 0;

    } catch (const std::exception& e) {
        std::cerr << "[FATAL] Exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "[FATAL] Unknown exception" << std::endl;
        return 1;
    }
}
