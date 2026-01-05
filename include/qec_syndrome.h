#pragma once

#include "qec_types.h"
#include "qec_stabilizer.h"
#include "types.h"
#include <complex>
#include <vector>
#include <functional>
#include <random>

namespace qlret {

//==============================================================================
// Syndrome Extraction Circuit
//==============================================================================

/**
 * @brief Generates and executes syndrome extraction circuits
 * 
 * Creates ancilla-based syndrome measurement circuits for stabilizer codes.
 * Supports both ideal and noisy syndrome extraction.
 */
class SyndromeExtractor {
public:
    struct NoiseParams {
        double measurement_error = 0.0;   // Probability of measurement flip
        double cnot_error = 0.0;          // Depolarizing error per CNOT
        double idle_error = 0.0;          // Idling error per time step
        double reset_error = 0.0;         // Ancilla reset error
    };

    explicit SyndromeExtractor(const StabilizerCode& code, 
                               NoiseParams noise = {},
                               unsigned seed = 42);

    /**
     * @brief Extract syndrome from Pauli error pattern
     * @param error PauliString representing data qubit errors
     * @return Syndrome with X and Z syndrome bits
     */
    Syndrome extract(const PauliString& error) const;

    /**
     * @brief Extract syndrome from density matrix (LRET)
     * @param L Low-rank L-factor (n_data x rank)
     * @return Syndrome with probabilistic measurement outcomes
     */
    Syndrome extract_from_state(const CMatrix& L);

    /**
     * @brief Generate stabilizer measurement circuit instructions
     * @param stab_idx Stabilizer index
     * @param is_x True for X-stabilizer, false for Z-stabilizer
     * @return Vector of gate instructions (type, control, target)
     */
    std::vector<std::tuple<std::string, int, int>> 
    measurement_circuit(size_t stab_idx, bool is_x) const;

    /**
     * @brief Perform multiple syndrome rounds (for time-domain decoding)
     * @param error Current error pattern
     * @param rounds Number of measurement rounds
     * @return Vector of syndromes across time
     */
    std::vector<Syndrome> extract_multiple_rounds(const PauliString& error, 
                                                   size_t rounds);

    /**
     * @brief Get detection events between consecutive syndromes
     * @param prev Previous syndrome
     * @param curr Current syndrome
     * @return Detection event syndrome (XOR of consecutive)
     */
    static Syndrome detection_events(const Syndrome& prev, const Syndrome& curr);

    const StabilizerCode& code() const { return code_; }
    const NoiseParams& noise() const { return noise_; }

private:
    const StabilizerCode& code_;
    NoiseParams noise_;
    mutable std::mt19937 rng_;

    // Apply measurement error
    int noisy_measure(int ideal_result) const;

    // Apply gate noise
    void apply_cnot_noise(PauliString& error, size_t ctrl, size_t tgt);
};

//==============================================================================
// Syndrome Graph (for MWPM)
//==============================================================================

/**
 * @brief Graph representation for matching-based decoding
 */
struct SyndromeGraph {
    struct Edge {
        size_t u;         // First vertex (defect index or boundary)
        size_t v;         // Second vertex
        double weight;    // Edge weight (typically -log(p))
        size_t boundary_u = SIZE_MAX;  // Boundary vertex index if applicable
        size_t boundary_v = SIZE_MAX;
    };

    size_t num_defects;
    std::vector<Edge> edges;
    
    // Build graph from syndrome
    static SyndromeGraph from_syndrome(const Syndrome& syn, 
                                       const StabilizerCode& code,
                                       double physical_error_rate);

    // Build 3D graph for time-domain decoding
    static SyndromeGraph from_detection_events(
        const std::vector<Syndrome>& syndromes,
        const StabilizerCode& code,
        double physical_error_rate,
        double measurement_error_rate);
};

//==============================================================================
// Error Injection (for testing)
//==============================================================================

/**
 * @brief Generate random Pauli errors for QEC simulation
 */
class ErrorInjector {
public:
    explicit ErrorInjector(size_t num_qubits, unsigned seed = 42);

    /**
     * @brief Generate depolarizing error
     * @param p Per-qubit error probability
     * @return Random PauliString
     */
    PauliString depolarizing(double p);

    /**
     * @brief Generate biased noise (X vs Z)
     * @param px X-error probability
     * @param pz Z-error probability
     * @return Random PauliString
     */
    PauliString biased_noise(double px, double pz);

    /**
     * @brief Generate single qubit error at specific location
     * @param qubit Target qubit
     * @param pauli Error type
     * @return PauliString with single error
     */
    PauliString single_error(size_t qubit, Pauli pauli);

    /**
     * @brief Generate error chain (for logical error testing)
     * @param qubits Chain of qubits
     * @param pauli Error type
     * @return PauliString with chain of errors
     */
    PauliString error_chain(const std::vector<size_t>& qubits, Pauli pauli);

private:
    size_t num_qubits_;
    std::mt19937 rng_;
    std::uniform_real_distribution<> uniform_;
};

}  // namespace qlret
