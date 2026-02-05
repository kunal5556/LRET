/**
 * @file phase5_optimizations.h
 * @brief Phase 5: Advanced Optimization Techniques for Row Parallelism
 * 
 * This file implements three advanced optimization techniques from
 * ROW_PARALLELISM_OPTIMIZATION_STRATEGY.md Phase 5:
 * 
 * 1. COMMUNITY DETECTION BATCHING (1.5-2× for random circuits)
 *    - Graph-based row community detection using BFS clustering
 *    - Dynamic scheduling with better load balance
 *    - Reduces cache thrashing for unstructured circuits
 * 
 * 2. ML-BASED ADAPTIVE RANK PREDICTION (5-10% speedup)
 *    - Lightweight C++ inference engine (no external dependencies)
 *    - Predicts rank growth to enable proactive truncation
 *    - Adaptive truncation threshold based on predictions
 * 
 * 3. HYBRID TREE TENSOR NETWORK (2.5× for depth > 100)
 *    - Hierarchical SVD decomposition of L matrix
 *    - Binary tree tensor structure for deep circuits
 *    - Automatic switching between LRET and TTN modes
 * 
 * @author LRET Team (Phase 5 - Row Parallelism Advanced)
 * @date February 2026
 * @version 1.0
 */

#pragma once

#include "types.h"
#include "gates_and_noise.h"
#include "advanced_optimizations.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <functional>
#include <chrono>
#include <cmath>
#include <array>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace qlret {

//==============================================================================
// Phase 5 Configuration
//==============================================================================

/**
 * @brief Configuration for Phase 5 advanced optimizations
 */
struct Phase5Config {
    //--------------------------------------------------------------------------
    // Community Detection Settings
    //--------------------------------------------------------------------------
    bool enable_community_detection = true;   ///< Enable graph-based community detection
    size_t min_dim_for_community = 4096;      ///< Min dimension (2^n) to use community detection
    size_t min_community_size = 64;           ///< Min rows per community
    size_t max_communities = 32;              ///< Max number of communities
    double community_merge_threshold = 0.3;   ///< Merge communities if overlap > threshold
    bool use_louvain_clustering = false;      ///< Use Louvain algorithm (slower but better)
    
    //--------------------------------------------------------------------------
    // ML Rank Prediction Settings
    //--------------------------------------------------------------------------
    bool enable_rank_prediction = true;       ///< Enable ML-based rank prediction
    size_t prediction_lookahead = 5;          ///< Gates to look ahead for prediction
    double proactive_truncation_factor = 0.5; ///< Tighten threshold by this factor proactively
    size_t rank_explosion_threshold = 128;    ///< Predict explosion if rank exceeds this
    bool use_adaptive_threshold = true;       ///< Adjust truncation threshold based on predictions
    
    //--------------------------------------------------------------------------
    // Hybrid TTN Settings
    //--------------------------------------------------------------------------
    bool enable_hybrid_ttn = true;            ///< Enable Tree Tensor Network mode
    size_t ttn_depth_threshold = 50;          ///< Switch to TTN after this many gates
    size_t ttn_rank_threshold = 64;           ///< Minimum rank to consider TTN
    size_t ttn_max_bond_dim = 256;            ///< Maximum bond dimension in TTN
    bool ttn_auto_switch = true;              ///< Automatically switch between LRET and TTN
    
    //--------------------------------------------------------------------------
    // General Settings
    //--------------------------------------------------------------------------
    bool verbose = false;                     ///< Print diagnostic messages
    bool collect_stats = true;                ///< Collect performance statistics
    
    Phase5Config() = default;
    
    // Fluent API
    Phase5Config& set_community_detection(bool v) { enable_community_detection = v; return *this; }
    Phase5Config& set_rank_prediction(bool v) { enable_rank_prediction = v; return *this; }
    Phase5Config& set_hybrid_ttn(bool v) { enable_hybrid_ttn = v; return *this; }
    Phase5Config& set_verbose(bool v) { verbose = v; return *this; }
};

//==============================================================================
// Part 1: Community Detection Batching
//==============================================================================

/**
 * @brief Represents a connectivity graph of L matrix rows based on gate operations
 * 
 * Nodes: Rows of L (0 to 2^n - 1)
 * Edges: Gates connecting row pairs
 *   - Single-qubit gate on qubit t: connects rows (i, i XOR 2^t)
 *   - Two-qubit gate on qubits c,t: connects 4-way row groups
 */
class CircuitConnectivityGraph {
public:
    /**
     * @brief Constructor
     * @param num_qubits Number of qubits
     */
    explicit CircuitConnectivityGraph(size_t num_qubits);
    
    /**
     * @brief Build graph from quantum sequence
     * @param sequence Quantum sequence to analyze
     */
    void build_from_sequence(const QuantumSequence& sequence);
    
    /**
     * @brief Add edge for single-qubit gate
     * @param target Target qubit index
     */
    void add_single_qubit_gate(size_t target);
    
    /**
     * @brief Add edges for two-qubit gate
     * @param control Control qubit index
     * @param target Target qubit index
     */
    void add_two_qubit_gate(size_t control, size_t target);
    
    /**
     * @brief Get neighbors of a row
     * @param row Row index
     * @return Set of neighboring row indices
     */
    const std::unordered_set<size_t>& get_neighbors(size_t row) const;
    
    /**
     * @brief Get degree of a row (number of connections)
     * @param row Row index
     * @return Number of neighbors
     */
    size_t get_degree(size_t row) const;
    
    /**
     * @brief Get number of nodes (rows)
     */
    size_t num_nodes() const { return dim_; }
    
    /**
     * @brief Get total number of edges
     */
    size_t num_edges() const { return total_edges_; }
    
    /**
     * @brief Reset graph
     */
    void reset();

private:
    size_t num_qubits_;
    size_t dim_;
    size_t total_edges_;
    std::vector<std::unordered_set<size_t>> adjacency_;
    static std::unordered_set<size_t> empty_set_;
};

/**
 * @brief Advanced community detection using BFS clustering and optional Louvain
 */
class AdvancedCommunityDetector {
public:
    /**
     * @brief Constructor
     * @param graph Connectivity graph
     * @param config Configuration
     */
    AdvancedCommunityDetector(const CircuitConnectivityGraph& graph, 
                               const Phase5Config& config = {});
    
    /**
     * @brief Detect communities using BFS clustering
     * 
     * Algorithm:
     * 1. Start from highest-degree node as seed
     * 2. BFS expand until community reaches max size
     * 3. Mark visited nodes, repeat with next unvisited seed
     * 4. Merge small communities if beneficial
     * 
     * @return Vector of communities (each community = vector of row indices)
     */
    std::vector<std::vector<size_t>> detect_bfs() const;
    
    /**
     * @brief Detect communities using Louvain algorithm
     * 
     * More sophisticated than BFS, optimizes modularity.
     * Better quality but O(n log n) complexity.
     * 
     * @return Vector of communities
     */
    std::vector<std::vector<size_t>> detect_louvain() const;
    
    /**
     * @brief Get community assignment vector
     * 
     * @return Vector where result[row] = community_id
     */
    std::vector<size_t> get_assignment() const;
    
    /**
     * @brief Calculate community quality metric (modularity)
     * @param communities Community assignment
     * @return Modularity score in [-0.5, 1.0]
     */
    double calculate_modularity(const std::vector<std::vector<size_t>>& communities) const;

private:
    const CircuitConnectivityGraph& graph_;
    Phase5Config config_;
    mutable std::vector<std::vector<size_t>> cached_communities_;
    mutable bool communities_cached_ = false;
};

/**
 * @brief Apply gate using community-based parallelism
 * 
 * Instead of fixed OpenMP chunks, process each community in parallel.
 * Communities have high intra-connectivity → good cache reuse.
 * 
 * @param L Input L matrix
 * @param gate Gate operation
 * @param num_qubits Number of qubits
 * @param communities Community assignment (communities[c] = vector of rows in community c)
 * @return Result matrix after gate application
 */
MatrixXcd apply_gate_community_parallel(
    const MatrixXcd& L,
    const GateOp& gate,
    size_t num_qubits,
    const std::vector<std::vector<size_t>>& communities
);

//==============================================================================
// Part 2: ML-Based Adaptive Rank Prediction
//==============================================================================

/**
 * @brief Features for rank prediction model
 */
struct RankPredictionFeatures {
    size_t current_rank;           ///< Current rank of L
    size_t gate_type_encoded;      ///< 0=H, 1=X, 2=Y, 3=Z, 4=RX, 5=RY, 6=RZ, 7=CNOT, ...
    size_t target_qubit;           ///< Target qubit index
    size_t control_qubit;          ///< Control qubit (0 if single-qubit)
    double noise_probability;      ///< Noise probability
    size_t noise_type_encoded;     ///< 0=none, 1=depolarizing, 2=amplitude_damping, ...
    size_t depth_so_far;           ///< Gates applied so far
    size_t num_qubits;             ///< Total qubits
    double truncation_threshold;   ///< Current truncation threshold
    size_t gates_since_truncation; ///< Gates since last truncation
    
    /**
     * @brief Convert to fixed-size feature vector
     * @return Array of 10 normalized features
     */
    std::array<double, 10> to_vector() const;
};

/**
 * @brief Lightweight neural network for rank prediction
 * 
 * Architecture: 10 → 32 → 16 → 1 (fully connected)
 * - No external dependencies (pure C++ implementation)
 * - Pre-trained weights stored in static arrays
 * - Inference time: ~1 µs
 */
class RankPredictorNN {
public:
    /**
     * @brief Constructor (loads pre-trained weights)
     */
    RankPredictorNN();
    
    /**
     * @brief Predict rank after gate/noise operation
     * @param features Input features
     * @return Predicted rank
     */
    size_t predict(const RankPredictionFeatures& features) const;
    
    /**
     * @brief Predict rank growth rate
     * @param features Input features
     * @return Predicted growth factor (e.g., 1.5 means rank will grow 1.5×)
     */
    double predict_growth_rate(const RankPredictionFeatures& features) const;
    
    /**
     * @brief Check if rank explosion is predicted
     * @param features Input features
     * @param threshold Explosion threshold
     * @return True if explosion predicted
     */
    bool predict_explosion(const RankPredictionFeatures& features, size_t threshold) const;
    
    /**
     * @brief Get recommended truncation threshold
     * @param features Input features
     * @param base_threshold Base truncation threshold
     * @return Adjusted threshold
     */
    double get_adaptive_threshold(const RankPredictionFeatures& features, 
                                   double base_threshold) const;

private:
    // Pre-trained weights (10 → 32 → 16 → 1)
    std::array<std::array<double, 10>, 32> W1_;  // First layer weights
    std::array<double, 32> b1_;                   // First layer biases
    std::array<std::array<double, 32>, 16> W2_;  // Second layer weights
    std::array<double, 16> b2_;                   // Second layer biases
    std::array<double, 16> W3_;                   // Output layer weights
    double b3_;                                    // Output layer bias
    
    /**
     * @brief ReLU activation
     */
    static double relu(double x) { return x > 0 ? x : 0; }
    
    /**
     * @brief Forward pass
     * @param input Input features
     * @return Output value
     */
    double forward(const std::array<double, 10>& input) const;
    
    /**
     * @brief Initialize with pre-trained weights
     */
    void initialize_weights();
};

/**
 * @brief Heuristic rank predictor (no ML, rule-based)
 * 
 * Fallback predictor using simple heuristics:
 * - Depolarizing noise: rank *= 4
 * - Amplitude damping: rank *= 2
 * - Single-qubit gate: rank unchanged
 * - Two-qubit gate: rank *= 1.1 (small increase due to entanglement)
 */
class HeuristicRankPredictor {
public:
    /**
     * @brief Predict rank after operation
     * @param current_rank Current rank
     * @param op Operation (gate or noise)
     * @return Predicted rank
     */
    static size_t predict(size_t current_rank, const std::variant<GateOp, NoiseOp>& op);
    
    /**
     * @brief Predict if truncation is needed
     * @param current_rank Current rank
     * @param ops Next few operations
     * @param threshold Rank threshold
     * @return True if truncation recommended
     */
    static bool should_truncate_proactively(
        size_t current_rank,
        const std::vector<std::variant<GateOp, NoiseOp>>& ops,
        size_t threshold
    );
};

/**
 * @brief Adaptive truncation manager using rank predictions
 */
class AdaptiveTruncationManager {
public:
    /**
     * @brief Constructor
     * @param config Configuration
     */
    explicit AdaptiveTruncationManager(const Phase5Config& config = {});
    
    /**
     * @brief Update state with new operation
     * @param op Operation just applied
     * @param current_rank Current rank after operation
     */
    void update(const std::variant<GateOp, NoiseOp>& op, size_t current_rank);
    
    /**
     * @brief Get adaptive truncation threshold
     * @param base_threshold Base threshold from config
     * @param upcoming_ops Next operations (for lookahead)
     * @return Adjusted threshold
     */
    double get_threshold(double base_threshold,
                         const std::vector<std::variant<GateOp, NoiseOp>>& upcoming_ops = {}) const;
    
    /**
     * @brief Check if proactive truncation is recommended
     * @param current_rank Current rank
     * @param upcoming_ops Next operations
     * @return True if should truncate now
     */
    bool should_truncate_now(size_t current_rank,
                              const std::vector<std::variant<GateOp, NoiseOp>>& upcoming_ops) const;
    
    /**
     * @brief Reset manager state
     */
    void reset();
    
    /**
     * @brief Get statistics
     */
    struct Stats {
        size_t predictions_made = 0;
        size_t proactive_truncations = 0;
        double avg_threshold_adjustment = 0.0;
    };
    
    const Stats& get_stats() const { return stats_; }

private:
    Phase5Config config_;
    RankPredictorNN nn_predictor_;
    size_t gates_since_truncation_ = 0;
    size_t last_rank_ = 1;
    std::vector<size_t> rank_history_;
    Stats stats_;
};

//==============================================================================
// Part 3: Hybrid Tree Tensor Network (TTN)
//==============================================================================

/**
 * @brief Node in the Tree Tensor Network
 */
struct TTNNode {
    MatrixXcd tensor;              ///< Tensor data (various shapes depending on node type)
    size_t left_bond_dim = 0;      ///< Bond dimension to left child
    size_t right_bond_dim = 0;     ///< Bond dimension to right child
    size_t parent_bond_dim = 0;    ///< Bond dimension to parent
    size_t physical_dim = 0;       ///< Physical dimension (if leaf)
    std::unique_ptr<TTNNode> left_child;   ///< Left child
    std::unique_ptr<TTNNode> right_child;  ///< Right child
    TTNNode* parent = nullptr;     ///< Parent (weak reference)
    bool is_leaf = false;          ///< True if this is a leaf node
    size_t qubit_start = 0;        ///< First qubit covered (for leaves)
    size_t qubit_end = 0;          ///< Last qubit covered (for leaves)
    
    TTNNode() = default;
    TTNNode(const TTNNode&) = delete;
    TTNNode& operator=(const TTNNode&) = delete;
    TTNNode(TTNNode&&) = default;
    TTNNode& operator=(TTNNode&&) = default;
};

/**
 * @brief Tree Tensor Network representation of quantum state
 * 
 * Converts L matrix to hierarchical binary tree structure:
 * 
 *                    Root
 *                   /    \
 *                Node1   Node2
 *               /    \   /    \
 *             L01   L23 L45   L67
 * 
 * Each node stores a tensor contracted from its children.
 * Gates affecting a subtree are applied locally.
 */
class TreeTensorNetwork {
public:
    /**
     * @brief Constructor
     * @param num_qubits Number of qubits
     * @param config Configuration
     */
    TreeTensorNetwork(size_t num_qubits, const Phase5Config& config = {});
    
    /**
     * @brief Destructor
     */
    ~TreeTensorNetwork() = default;
    
    // Move semantics
    TreeTensorNetwork(TreeTensorNetwork&&) = default;
    TreeTensorNetwork& operator=(TreeTensorNetwork&&) = default;
    
    // No copy (TTN can be large)
    TreeTensorNetwork(const TreeTensorNetwork&) = delete;
    TreeTensorNetwork& operator=(const TreeTensorNetwork&) = delete;
    
    /**
     * @brief Convert L matrix to TTN representation
     * 
     * Algorithm (Hierarchical SVD):
     * 1. Split L into left and right halves (by qubits)
     * 2. SVD each half → truncate to max_bond_dim
     * 3. Recursively decompose each half
     * 4. Store tensors in binary tree structure
     * 
     * @param L Input L matrix (dim × rank)
     */
    void from_L_matrix(const MatrixXcd& L);
    
    /**
     * @brief Convert TTN back to L matrix
     * 
     * Algorithm (Tensor Contraction):
     * 1. Contract leaves with their parent
     * 2. Repeat upward until reaching root
     * 3. Result is L matrix
     * 
     * @return Reconstructed L matrix
     */
    MatrixXcd to_L_matrix() const;
    
    /**
     * @brief Apply gate in TTN representation
     * 
     * 1. Find minimal subtree containing gate qubits
     * 2. Contract subtree to local tensor
     * 3. Apply gate to local tensor
     * 4. Re-decompose with SVD
     * 
     * @param gate Gate operation
     */
    void apply_gate(const GateOp& gate);
    
    /**
     * @brief Apply noise in TTN representation
     * 
     * For noise, we must convert back to L, apply noise, and re-decompose.
     * This is because noise is not unitary and increases rank.
     * 
     * @param noise Noise operation
     * @param truncation_threshold Truncation threshold
     */
    void apply_noise(const NoiseOp& noise, double truncation_threshold);
    
    /**
     * @brief Get total memory usage (bytes)
     */
    size_t memory_usage() const;
    
    /**
     * @brief Get maximum bond dimension currently used
     */
    size_t max_bond_dim() const;
    
    /**
     * @brief Check if TTN mode is beneficial for current state
     */
    bool is_ttn_beneficial() const;
    
    /**
     * @brief Truncate bond dimensions to limit memory
     * @param max_dim Maximum allowed bond dimension
     */
    void truncate_bonds(size_t max_dim);

private:
    std::unique_ptr<TTNNode> root_;
    size_t num_qubits_;
    size_t dim_;
    Phase5Config config_;
    
    /**
     * @brief Recursively build TTN from L matrix
     */
    std::unique_ptr<TTNNode> build_tree(const MatrixXcd& L, 
                                         size_t qubit_start, 
                                         size_t qubit_end);
    
    /**
     * @brief Contract node with children to get local tensor
     */
    MatrixXcd contract_to_local(TTNNode* node) const;
    
    /**
     * @brief Decompose local tensor back to node with children
     */
    void decompose_from_local(TTNNode* node, const MatrixXcd& local);
    
    /**
     * @brief Find minimal subtree containing given qubits
     */
    TTNNode* find_minimal_subtree(const std::vector<size_t>& qubits) const;
    
    /**
     * @brief Calculate memory usage of subtree
     */
    size_t subtree_memory(const TTNNode* node) const;
};

/**
 * @brief Hybrid simulator that switches between LRET and TTN modes
 */
class HybridLRETTTN {
public:
    /**
     * @brief Constructor
     * @param num_qubits Number of qubits
     * @param config Configuration
     */
    HybridLRETTTN(size_t num_qubits, const Phase5Config& config = {});
    
    /**
     * @brief Run simulation with automatic mode switching
     * 
     * Starts in LRET mode. Switches to TTN when:
     * - Depth exceeds threshold AND
     * - Rank exceeds threshold
     * 
     * Switches back to LRET when:
     * - Noise operation encountered (TTN doesn't handle noise well)
     * 
     * @param L_init Initial L matrix
     * @param sequence Quantum sequence
     * @param sim_config Simulation config
     * @return Final L matrix
     */
    MatrixXcd run(const MatrixXcd& L_init,
                  const QuantumSequence& sequence,
                  const SimConfig& sim_config);
    
    /**
     * @brief Force a specific mode
     */
    enum class Mode { LRET, TTN };
    void set_mode(Mode mode);
    
    /**
     * @brief Get current mode
     */
    Mode get_mode() const { return mode_; }
    
    /**
     * @brief Get statistics
     */
    struct Stats {
        size_t lret_gates = 0;
        size_t ttn_gates = 0;
        size_t mode_switches = 0;
        double lret_time_ms = 0.0;
        double ttn_time_ms = 0.0;
        double conversion_time_ms = 0.0;
    };
    
    const Stats& get_stats() const { return stats_; }
    
private:
    size_t num_qubits_;
    Phase5Config config_;
    Mode mode_ = Mode::LRET;
    std::unique_ptr<TreeTensorNetwork> ttn_;
    Stats stats_;
    size_t depth_counter_ = 0;
    
    /**
     * @brief Check if should switch to TTN mode
     */
    bool should_switch_to_ttn(size_t current_rank) const;
    
    /**
     * @brief Check if should switch back to LRET mode
     */
    bool should_switch_to_lret(const std::variant<GateOp, NoiseOp>& next_op) const;
};

//==============================================================================
// Phase 5 Unified Simulation Interface
//==============================================================================

/**
 * @brief Phase 5 statistics
 */
struct Phase5Stats {
    // Community detection stats
    size_t communities_detected = 0;
    double community_modularity = 0.0;
    double community_detection_time_ms = 0.0;
    
    // Rank prediction stats
    size_t predictions_made = 0;
    size_t accurate_predictions = 0;  // Within 20% of actual
    size_t proactive_truncations = 0;
    
    // TTN stats
    size_t ttn_activations = 0;
    size_t ttn_gates_applied = 0;
    double ttn_memory_savings_mb = 0.0;
    
    // Overall timing
    double total_simulation_time_ms = 0.0;
    double speedup_vs_baseline = 1.0;
    
    void reset() { *this = Phase5Stats(); }
};

/**
 * @brief Get global Phase 5 statistics
 */
Phase5Stats& get_phase5_stats();

/**
 * @brief Run simulation with all Phase 5 optimizations
 * 
 * Combines:
 * - Community detection for parallel gate application
 * - ML-based adaptive truncation
 * - Hybrid TTN mode for deep circuits
 * 
 * @param L_init Initial L matrix
 * @param sequence Quantum sequence
 * @param num_qubits Number of qubits
 * @param sim_config Simulation configuration
 * @param phase5_config Phase 5 configuration
 * @return Final L matrix
 */
MatrixXcd run_with_phase5_optimizations(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    const SimConfig& sim_config,
    const Phase5Config& phase5_config = {}
);

/**
 * @brief High-level simulation with automatic optimization selection
 * 
 * Automatically enables Phase 5 optimizations based on circuit characteristics:
 * - Community detection: enabled for n ≥ 12 with random circuits
 * - Rank prediction: always enabled (low overhead)
 * - TTN: enabled for depth > 50 circuits
 * 
 * @param L_init Initial L matrix
 * @param sequence Quantum sequence
 * @param num_qubits Number of qubits
 * @param sim_config Simulation configuration
 * @return Final L matrix
 */
MatrixXcd simulate_phase5_auto(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    const SimConfig& sim_config
);

}  // namespace qlret
