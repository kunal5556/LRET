#include "qec_syndrome.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace qlret {

//==============================================================================
// SyndromeExtractor Implementation
//==============================================================================

SyndromeExtractor::SyndromeExtractor(const StabilizerCode& code, 
                                     NoiseParams noise,
                                     unsigned seed)
    : code_(code), noise_(noise), rng_(seed) {}

Syndrome SyndromeExtractor::extract(const PauliString& error) const {
    Syndrome syn;
    const auto& x_stabs = code_.x_stabilizers();
    const auto& z_stabs = code_.z_stabilizers();

    // X-syndrome: measure Z-stabilizers (detect X errors)
    // Each Z-stabilizer anti-commutes with X errors in its support
    syn.z_syndrome.resize(z_stabs.size(), 0);
    for (size_t i = 0; i < z_stabs.size(); ++i) {
        // Z-stabilizer detects X and Y errors
        const auto& stab = z_stabs[i];
        int parity = 0;
        for (size_t q : stab.support()) {
            if (q < error.size()) {
                Pauli e = error[q];
                if (e == Pauli::X || e == Pauli::Y) {
                    parity ^= 1;
                }
            }
        }
        syn.z_syndrome[i] = noisy_measure(parity);
    }

    // Z-syndrome: measure X-stabilizers (detect Z errors)
    syn.x_syndrome.resize(x_stabs.size(), 0);
    for (size_t i = 0; i < x_stabs.size(); ++i) {
        const auto& stab = x_stabs[i];
        int parity = 0;
        for (size_t q : stab.support()) {
            if (q < error.size()) {
                Pauli e = error[q];
                if (e == Pauli::Z || e == Pauli::Y) {
                    parity ^= 1;
                }
            }
        }
        syn.x_syndrome[i] = noisy_measure(parity);
    }

    return syn;
}

Syndrome SyndromeExtractor::extract_from_state(const CMatrix& /* L */) {
    // LRET-based syndrome extraction
    // For pure stabilizer states, this projects onto syndrome eigenspaces
    // Full implementation requires state projection and measurement sampling
    
    // Placeholder: return empty syndrome (no errors detected)
    Syndrome syn;
    syn.x_syndrome.resize(code_.x_stabilizers().size(), 0);
    syn.z_syndrome.resize(code_.z_stabilizers().size(), 0);
    return syn;
}

std::vector<std::tuple<std::string, int, int>> 
SyndromeExtractor::measurement_circuit(size_t stab_idx, bool is_x) const {
    std::vector<std::tuple<std::string, int, int>> circuit;
    
    const auto& stabs = is_x ? code_.x_stabilizers() : code_.z_stabilizers();
    if (stab_idx >= stabs.size()) return circuit;

    const auto& stab = stabs[stab_idx];
    int ancilla = static_cast<int>(code_.num_data_qubits() + stab_idx);

    // Initialize ancilla in |0⟩
    circuit.push_back({"RESET", ancilla, -1});

    if (is_x) {
        // X-stabilizer measurement: H-ancilla, CNOTs, H-ancilla
        circuit.push_back({"H", ancilla, -1});
        for (size_t q : stab.support()) {
            if (stab[q] == Pauli::X) {
                circuit.push_back({"CNOT", ancilla, static_cast<int>(q)});
            }
        }
        circuit.push_back({"H", ancilla, -1});
    } else {
        // Z-stabilizer measurement: CNOTs from data to ancilla
        for (size_t q : stab.support()) {
            if (stab[q] == Pauli::Z) {
                circuit.push_back({"CNOT", static_cast<int>(q), ancilla});
            }
        }
    }

    circuit.push_back({"MEASURE", ancilla, -1});
    return circuit;
}

std::vector<Syndrome> SyndromeExtractor::extract_multiple_rounds(
    const PauliString& error, size_t rounds) {
    std::vector<Syndrome> syndromes;
    syndromes.reserve(rounds);

    PauliString accumulated_error = error;

    for (size_t r = 0; r < rounds; ++r) {
        syndromes.push_back(extract(accumulated_error));
        // Note: In a real simulation, we would apply mid-circuit noise here
    }

    return syndromes;
}

Syndrome SyndromeExtractor::detection_events(const Syndrome& prev, 
                                              const Syndrome& curr) {
    Syndrome events;
    events.x_syndrome.resize(curr.x_syndrome.size());
    events.z_syndrome.resize(curr.z_syndrome.size());

    for (size_t i = 0; i < events.x_syndrome.size(); ++i) {
        int p = (i < prev.x_syndrome.size()) ? prev.x_syndrome[i] : 0;
        events.x_syndrome[i] = p ^ curr.x_syndrome[i];
    }

    for (size_t i = 0; i < events.z_syndrome.size(); ++i) {
        int p = (i < prev.z_syndrome.size()) ? prev.z_syndrome[i] : 0;
        events.z_syndrome[i] = p ^ curr.z_syndrome[i];
    }

    return events;
}

int SyndromeExtractor::noisy_measure(int ideal_result) const {
    if (noise_.measurement_error > 0.0) {
        std::uniform_real_distribution<> dist(0.0, 1.0);
        if (dist(rng_) < noise_.measurement_error) {
            return 1 - ideal_result;
        }
    }
    return ideal_result;
}

void SyndromeExtractor::apply_cnot_noise(PauliString& error, 
                                          size_t ctrl, size_t tgt) {
    if (noise_.cnot_error <= 0.0) return;

    std::uniform_real_distribution<> dist(0.0, 1.0);
    double r = dist(rng_);

    // Two-qubit depolarizing: 15 non-trivial Paulis with equal probability
    if (r < noise_.cnot_error) {
        int pauli_idx = static_cast<int>(dist(rng_) * 15);
        Pauli p1 = static_cast<Pauli>((pauli_idx / 4) + 1);
        Pauli p2 = static_cast<Pauli>((pauli_idx % 4));
        
        if (ctrl < error.size()) {
            error.set(ctrl, pauli_mult(error[ctrl], p1));
        }
        if (tgt < error.size()) {
            error.set(tgt, pauli_mult(error[tgt], p2));
        }
    }
}

//==============================================================================
// SyndromeGraph Implementation
//==============================================================================

SyndromeGraph SyndromeGraph::from_syndrome(const Syndrome& syn, 
                                           const StabilizerCode& code,
                                           double physical_error_rate) {
    SyndromeGraph graph;
    double weight = -std::log(physical_error_rate);

    // Collect X and Z defects
    auto x_defects = syn.x_defect_indices();
    auto z_defects = syn.z_defect_indices();

    // For now, handle each type separately
    // Build edges for Z-defects (for X-error decoding)
    graph.num_defects = z_defects.size();

    for (size_t i = 0; i < z_defects.size(); ++i) {
        for (size_t j = i + 1; j < z_defects.size(); ++j) {
            // Distance based on stabilizer positions
            auto [r1, c1] = code.qubit_coords(z_defects[i]);
            auto [r2, c2] = code.qubit_coords(z_defects[j]);
            int dist = std::abs(r1 - r2) + std::abs(c1 - c2);
            
            Edge e;
            e.u = i;
            e.v = j;
            e.weight = weight * dist;
            graph.edges.push_back(e);
        }

        // Boundary edges
        auto [r, c] = code.qubit_coords(z_defects[i]);
        int dist_to_boundary = std::min({r, c, 
            static_cast<int>(code.distance()) - 1 - r,
            static_cast<int>(code.distance()) - 1 - c});
        
        Edge boundary_edge;
        boundary_edge.u = i;
        boundary_edge.v = graph.num_defects;  // Virtual boundary vertex
        boundary_edge.weight = weight * dist_to_boundary;
        graph.edges.push_back(boundary_edge);
    }

    return graph;
}

SyndromeGraph SyndromeGraph::from_detection_events(
    const std::vector<Syndrome>& syndromes,
    const StabilizerCode& code,
    double physical_error_rate,
    double measurement_error_rate) {
    
    SyndromeGraph graph;
    double space_weight = -std::log(physical_error_rate);
    double time_weight = -std::log(measurement_error_rate);

    // 3D graph: collect all detection events across time
    std::vector<std::tuple<size_t, size_t, size_t>> defects;  // (type, space_idx, time_idx)

    Syndrome prev;
    prev.x_syndrome.resize(syndromes[0].x_syndrome.size(), 0);
    prev.z_syndrome.resize(syndromes[0].z_syndrome.size(), 0);

    for (size_t t = 0; t < syndromes.size(); ++t) {
        Syndrome events = SyndromeExtractor::detection_events(prev, syndromes[t]);
        
        for (size_t i = 0; i < events.z_syndrome.size(); ++i) {
            if (events.z_syndrome[i]) {
                defects.push_back({0, i, t});  // Z-syndrome defect
            }
        }
        for (size_t i = 0; i < events.x_syndrome.size(); ++i) {
            if (events.x_syndrome[i]) {
                defects.push_back({1, i, t});  // X-syndrome defect
            }
        }
        prev = syndromes[t];
    }

    graph.num_defects = defects.size();

    // Build edges (space and time)
    for (size_t i = 0; i < defects.size(); ++i) {
        for (size_t j = i + 1; j < defects.size(); ++j) {
            auto [type_i, space_i, time_i] = defects[i];
            auto [type_j, space_j, time_j] = defects[j];

            if (type_i != type_j) continue;  // Only match same type

            auto [r1, c1] = code.qubit_coords(space_i);
            auto [r2, c2] = code.qubit_coords(space_j);
            
            int space_dist = std::abs(r1 - r2) + std::abs(c1 - c2);
            int time_dist = std::abs(static_cast<int>(time_i) - static_cast<int>(time_j));

            Edge e;
            e.u = i;
            e.v = j;
            e.weight = space_weight * space_dist + time_weight * time_dist;
            graph.edges.push_back(e);
        }
    }

    return graph;
}

//==============================================================================
// ErrorInjector Implementation
//==============================================================================

ErrorInjector::ErrorInjector(size_t num_qubits, unsigned seed)
    : num_qubits_(num_qubits), rng_(seed), uniform_(0.0, 1.0) {}

PauliString ErrorInjector::depolarizing(double p) {
    PauliString error(num_qubits_);
    
    for (size_t i = 0; i < num_qubits_; ++i) {
        double r = uniform_(rng_);
        if (r < p) {
            // Error occurred, choose X, Y, or Z with equal probability
            double r2 = uniform_(rng_);
            if (r2 < 1.0/3.0) {
                error.set(i, Pauli::X);
            } else if (r2 < 2.0/3.0) {
                error.set(i, Pauli::Y);
            } else {
                error.set(i, Pauli::Z);
            }
        }
    }
    return error;
}

PauliString ErrorInjector::biased_noise(double px, double pz) {
    PauliString error(num_qubits_);
    
    for (size_t i = 0; i < num_qubits_; ++i) {
        bool has_x = uniform_(rng_) < px;
        bool has_z = uniform_(rng_) < pz;
        
        if (has_x && has_z) {
            error.set(i, Pauli::Y);
        } else if (has_x) {
            error.set(i, Pauli::X);
        } else if (has_z) {
            error.set(i, Pauli::Z);
        }
    }
    return error;
}

PauliString ErrorInjector::single_error(size_t qubit, Pauli pauli) {
    PauliString error(num_qubits_);
    if (qubit < num_qubits_) {
        error.set(qubit, pauli);
    }
    return error;
}

PauliString ErrorInjector::error_chain(const std::vector<size_t>& qubits, Pauli pauli) {
    PauliString error(num_qubits_);
    for (size_t q : qubits) {
        if (q < num_qubits_) {
            error.set(q, pauli);
        }
    }
    return error;
}

}  // namespace qlret
