#pragma once

/**
 * @file qec_adaptive.h
 * @brief Phase 9.3: Adaptive & ML-Driven Quantum Error Correction
 * 
 * Implements runtime-adaptive QEC with:
 * - Noise-aware code selection (Surface, Repetition, Color, Concatenated)
 * - ML-based decoder (Transformer via pybind11)
 * - Closed-loop calibration with drift detection
 * - Dynamic distance scaling
 * 
 * Dependencies:
 * - Phase 4: Noise calibration (NoiseProfile)
 * - Phase 5: Python bindings (pybind11)
 * - Phase 8: Fault tolerance (checkpointing)
 * - Phase 9.1/9.2: QEC infrastructure
 */

#include "qec_types.h"
#include "qec_stabilizer.h"
#include "qec_syndrome.h"
#include "qec_decoder.h"
#include "qec_logical.h"
#include "qec_distributed.h"
#include "advanced_noise.h"

#include <nlohmann/json.hpp>
#include <Eigen/Dense>

#include <chrono>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace qlret {

//==============================================================================
// Forward Declarations
//==============================================================================

class MLDecoder;
class AdaptiveCodeSelector;
class ClosedLoopController;
class AdaptiveQECController;

//==============================================================================
// NoiseProfile - Unified representation of device noise characteristics
//==============================================================================

/**
 * @brief Aggregated noise characteristics from Phase 4 calibration
 * 
 * Integrates with:
 * - Phase 4.1: JSON import from Qiskit noise models
 * - Phase 4.2: Calibration scripts (calibrate_noise_model.py)
 * - Phase 4.3: Advanced noise (correlations, time-varying, memory effects)
 */
struct NoiseProfile {
    // Metadata
    std::string device_name;
    std::string calibration_timestamp;
    size_t num_qubits = 0;
    
    // Single-qubit coherence times (per qubit, in nanoseconds)
    std::vector<double> t1_times_ns;
    std::vector<double> t2_times_ns;
    
    // Single-qubit gate errors (averaged across H, X, RZ, etc.)
    std::vector<double> single_gate_errors;
    
    // Two-qubit gate errors (CNOT/CZ)
    std::map<std::pair<size_t, size_t>, double> two_qubit_errors;
    
    // Measurement/readout errors (P(1|0) + P(0|1))
    std::vector<double> readout_errors;
    
    // Advanced noise features (from Phase 4.3)
    std::vector<CorrelatedError> correlated_errors;
    TimeVaryingNoiseParams time_varying;
    std::vector<MemoryEffect> memory_effects;
    
    //==========================================================================
    // Derived Statistics
    //==========================================================================
    
    /// Average single-qubit gate error across all qubits
    double avg_gate_error() const;
    
    /// Maximum single-qubit gate error (worst qubit)
    double max_gate_error() const;
    
    /// Average two-qubit gate error
    double avg_two_qubit_error() const;
    
    /// Maximum two-qubit gate error
    double max_two_qubit_error() const;
    
    /// Average T1 coherence time (ns)
    double avg_t1() const;
    
    /// Average T2 coherence time (ns)
    double avg_t2() const;
    
    /// T1/T2 ratio (indicates noise bias direction)
    double t1_t2_ratio() const;
    
    /// Check if noise is biased (|T1/T2 - 1| > threshold)
    bool is_biased(double threshold = 0.2) const;
    
    /// Check if correlated errors are present
    bool has_correlations() const;
    
    /// Check if time-varying noise is configured
    bool has_time_varying() const;
    
    /// Check if memory effects are present
    bool has_memory_effects() const;
    
    /// Average readout error
    double avg_readout_error() const;
    
    /// Estimate effective physical error rate for QEC threshold comparison
    double effective_physical_error_rate() const;
    
    //==========================================================================
    // Serialization
    //==========================================================================
    
    /// Convert to JSON
    nlohmann::json to_json() const;
    
    /// Construct from JSON
    static NoiseProfile from_json(const nlohmann::json& j);
    
    /// Load from Phase 4 calibration output file
    static NoiseProfile load_from_calibration(const std::string& calib_file);
    
    /// Save to file
    void save(const std::string& path) const;
    
    //==========================================================================
    // Comparison & Analysis
    //==========================================================================
    
    /// Compute relative difference from another profile (for drift detection)
    double relative_difference(const NoiseProfile& other) const;
    
    /// Check if profile differs significantly from baseline
    bool differs_from(const NoiseProfile& baseline, double threshold = 0.15) const;
};

//==============================================================================
// AdaptiveCodeSelector - Noise-aware code selection
//==============================================================================

/**
 * @brief Selects optimal QEC code based on noise characteristics
 * 
 * Decision tree:
 * 1. High error rate → concatenated or high-distance codes
 * 2. Biased noise (T1 ≠ T2) → repetition or biased surface codes
 * 3. Correlated errors → surface code (better at handling clusters)
 * 4. Default → standard rotated surface code
 */
class AdaptiveCodeSelector {
public:
    struct Config {
        double bias_threshold = 0.2;           // T1/T2 ratio threshold for bias detection
        double high_error_threshold = 0.05;    // Switch to higher distance or concat codes
        double low_error_threshold = 0.001;    // Can use smaller distance codes
        bool prefer_low_overhead = true;       // Minimize qubit count when possible
        size_t min_distance = 3;               // Minimum code distance
        size_t max_distance = 9;               // Maximum code distance
    };
    
    explicit AdaptiveCodeSelector(Config config = {});
    
    //==========================================================================
    // Main Selection Interface
    //==========================================================================
    
    /// Select optimal code type based on noise profile
    StabilizerCodeType select_code(const NoiseProfile& noise);
    
    /// Select optimal code distance for given code type and target error rate
    size_t select_distance(
        StabilizerCodeType code,
        const NoiseProfile& noise,
        double target_logical_error_rate
    );
    
    /// Combined selection: returns (code_type, distance)
    std::pair<StabilizerCodeType, size_t> select_code_and_distance(
        const NoiseProfile& noise,
        double target_logical_error_rate
    );
    
    //==========================================================================
    // Decision Tree Components
    //==========================================================================
    
    /// Select code for biased noise (T1 << T2 or T1 >> T2)
    StabilizerCodeType select_for_biased_noise(const NoiseProfile& noise);
    
    /// Select code for correlated noise
    StabilizerCodeType select_for_correlated_noise(const NoiseProfile& noise);
    
    /// Select code for high error rate (> high_error_threshold)
    StabilizerCodeType select_for_high_error_rate(const NoiseProfile& noise);
    
    /// Select code for low error rate (< low_error_threshold)
    StabilizerCodeType select_for_low_error_rate(const NoiseProfile& noise);
    
    //==========================================================================
    // Prediction & Ranking
    //==========================================================================
    
    /// Predict logical error rate for given code + noise combination
    double predict_logical_error_rate(
        StabilizerCodeType code,
        size_t distance,
        const NoiseProfile& noise
    );
    
    /// Rank all codes by predicted logical error rate (ascending)
    std::vector<std::pair<StabilizerCodeType, double>> rank_codes(
        const NoiseProfile& noise,
        size_t distance
    );
    
    /// Get configuration
    const Config& config() const { return config_; }
    
private:
    Config config_;
    
    /// Compute effective error rate for code type given noise profile
    double compute_effective_error_rate(
        StabilizerCodeType code,
        const NoiseProfile& noise
    );
    
    /// Get code-specific threshold constant (A in p_L ≈ A * p_phys^exp)
    double get_threshold_constant(StabilizerCodeType code);
    
    /// Get code-specific exponent (e.g., (d+1)/2 for surface, d for repetition)
    double get_exponent(StabilizerCodeType code, size_t distance);
};

//==============================================================================
// MLDecoder - Neural network-based syndrome decoder
//==============================================================================

/**
 * @brief ML-based decoder using neural networks (via pybind11)
 * 
 * Architecture:
 * - Transformer encoder for syndrome → error mapping
 * - Trained on synthetic error patterns from noise model
 * - Outperforms MWPM on correlated noise patterns
 * 
 * Integration:
 * - C++ inference wrapper calls Python model via pybind11
 * - Supports batch inference for efficiency
 * - Model checkpoints stored as .pkl files
 */
class MLDecoder : public Decoder {
public:
    struct Config {
        std::string model_path;               // Path to saved .pkl checkpoint
        std::string backend = "jax";          // "jax" or "pytorch"
        bool use_gpu = true;                  // GPU inference
        size_t batch_size = 1;                // Inference batch size
        double fallback_threshold = 0.5;      // Confidence threshold for MWPM fallback
        bool enable_fallback = true;          // Fall back to MWPM on low confidence
    };
    
    /// Construct ML decoder for given code
    MLDecoder(const StabilizerCode& code, Config config);
    
    /// Destructor (releases Python resources)
    ~MLDecoder() override;
    
    //==========================================================================
    // Decoder Interface
    //==========================================================================
    
    /// Decode syndrome to correction (single syndrome)
    Correction decode(const Syndrome& syndrome) override;
    
    /// Batch decode for efficiency
    std::vector<Correction> decode_batch(const std::vector<Syndrome>& syndromes);
    
    /// Get decoder name
    std::string name() const override { return "MLDecoder"; }
    
    //==========================================================================
    // Model Management
    //==========================================================================
    
    /// Load model from checkpoint file
    void load_model(const std::string& path);
    
    /// Reload current model (for closed-loop updates)
    void reload_model();
    
    /// Check if model is loaded and ready
    bool is_ready() const { return model_loaded_; }
    
    /// Get path to current model
    const std::string& model_path() const { return config_.model_path; }
    
    //==========================================================================
    // Statistics
    //==========================================================================
    
    struct Stats {
        size_t num_inferences = 0;
        size_t num_fallbacks = 0;
        double total_inference_time_ms = 0.0;
        
        double avg_inference_time_ms() const {
            return num_inferences ? total_inference_time_ms / num_inferences : 0.0;
        }
        
        double fallback_rate() const {
            return num_inferences ? static_cast<double>(num_fallbacks) / num_inferences : 0.0;
        }
    };
    
    Stats stats() const { return stats_; }
    void reset_stats() { stats_ = Stats{}; }
    
private:
    const StabilizerCode& code_;
    Config config_;
    Stats stats_;
    bool model_loaded_ = false;
    
    // Fallback decoder (MWPM)
    std::unique_ptr<MWPMDecoder> fallback_decoder_;
    
    // Python model handle (opaque pointer)
    struct PythonModel;
    std::unique_ptr<PythonModel> py_model_;
    
    /// Convert syndrome to model input format
    std::vector<float> syndrome_to_tensor(const Syndrome& syndrome);
    
    /// Convert model output to Correction
    Correction tensor_to_correction(const std::vector<float>& logits);
    
    /// Check confidence and decide on fallback
    bool should_fallback(const std::vector<float>& logits);
};

//==============================================================================
// ClosedLoopController - Drift detection and recalibration
//==============================================================================

/**
 * @brief Monitors QEC performance and triggers recalibration on drift
 * 
 * Algorithm:
 * 1. Track logical error rate in moving window
 * 2. Compare current rate to baseline
 * 3. If relative change > threshold, trigger recalibration
 * 4. Update decoder and code selection based on new noise profile
 */
class ClosedLoopController {
public:
    struct Config {
        size_t window_size = 100;               // Moving average window
        double drift_threshold = 0.15;          // Relative change threshold (15%)
        size_t recalibration_interval = 1000;   // Check every N cycles
        bool auto_recalibrate = true;           // Trigger Phase 4 automatically
        std::string calibration_script_path = "scripts/calibrate_noise_model.py";
        std::string calibration_output_path = "recalibration_output.json";
        size_t min_cycles_before_recalib = 200; // Minimum cycles before first recalib
    };
    
    explicit ClosedLoopController(Config config = {});
    
    //==========================================================================
    // Update & Detection
    //==========================================================================
    
    /// Update with QEC round result
    void update(const QECRoundResult& result);
    
    /// Check if recalibration is needed
    bool should_recalibrate() const;
    
    /// Force recalibration check (ignores interval)
    bool check_drift() const;
    
    //==========================================================================
    // Recalibration
    //==========================================================================
    
    /// Trigger recalibration (calls Phase 4.2 script)
    NoiseProfile recalibrate();
    
    /// Update baseline with new noise profile
    void update_baseline(const NoiseProfile& new_noise);
    
    /// Reset controller state
    void reset();
    
    //==========================================================================
    // Decoder Updates
    //==========================================================================
    
    /// Select and load appropriate model for new noise profile
    std::string select_model_for_noise(const NoiseProfile& noise);
    
    //==========================================================================
    // Statistics
    //==========================================================================
    
    struct Stats {
        size_t total_cycles = 0;
        size_t logical_errors = 0;
        double current_logical_error_rate = 0.0;
        double baseline_logical_error_rate = 0.0;
        size_t num_recalibrations = 0;
        
        double avg_logical_error_rate() const {
            return total_cycles ? static_cast<double>(logical_errors) / total_cycles : 0.0;
        }
    };
    
    Stats stats() const { return stats_; }
    const Config& config() const { return config_; }
    
    /// Get current noise profile
    const NoiseProfile& current_noise() const { return current_noise_; }
    
    /// Set initial noise profile
    void set_noise_profile(const NoiseProfile& noise);
    
private:
    Config config_;
    Stats stats_;
    NoiseProfile current_noise_;
    NoiseProfile baseline_noise_;
    
    // Moving window for error rate tracking
    std::deque<bool> recent_errors_;
    
    /// Detect drift based on recent error rate
    bool detect_drift() const;
    
    /// Compute error rate from recent window
    double compute_current_rate() const;
};

//==============================================================================
// DynamicDistanceSelector - Runtime distance adjustment
//==============================================================================

/**
 * @brief Dynamically adjusts code distance based on real-time error rates
 * 
 * Strategy:
 * - Start with minimal distance that meets target
 * - Increase distance if logical error rate exceeds target
 * - Decrease distance if error rate is much better than target (save resources)
 */
class DynamicDistanceSelector {
public:
    struct Config {
        size_t min_distance = 3;
        size_t max_distance = 9;
        double increase_threshold = 2.0;   // Increase if p_L > target * threshold
        double decrease_threshold = 0.1;   // Decrease if p_L < target * threshold
        size_t evaluation_window = 500;    // Cycles before distance change
    };
    
    explicit DynamicDistanceSelector(Config config = {});
    
    /// Update with QEC result
    void update(const QECRoundResult& result);
    
    /// Get recommended distance (may differ from current)
    size_t recommended_distance() const;
    
    /// Check if distance change is recommended
    bool should_change_distance() const;
    
    /// Set current distance
    void set_current_distance(size_t d);
    
    /// Set target logical error rate
    void set_target_error_rate(double target);
    
    /// Get current distance
    size_t current_distance() const { return current_distance_; }
    
private:
    Config config_;
    size_t current_distance_ = 3;
    double target_error_rate_ = 1e-6;
    std::deque<bool> recent_errors_;
};

//==============================================================================
// AdaptiveQECController - Orchestrates adaptive QEC
//==============================================================================

/**
 * @brief Master controller for adaptive QEC system
 * 
 * Coordinates:
 * - Code selection based on noise
 * - Decoder selection (ML vs MWPM)
 * - Closed-loop calibration
 * - Dynamic distance adjustment
 */
class AdaptiveQECController {
public:
    struct Config {
        // Initial settings
        StabilizerCodeType initial_code = StabilizerCodeType::SURFACE;
        size_t initial_distance = 3;
        double target_logical_error_rate = 1e-6;
        
        // Decoder settings
        bool use_ml_decoder = true;
        std::string ml_model_dir = "models/";
        
        // Adaptation settings
        bool enable_code_switching = true;
        bool enable_distance_adaptation = true;
        bool enable_closed_loop = true;
        
        // Component configs
        AdaptiveCodeSelector::Config selector_config;
        ClosedLoopController::Config closed_loop_config;
        DynamicDistanceSelector::Config distance_config;
    };
    
    explicit AdaptiveQECController(Config config = {});
    
    //==========================================================================
    // Initialization
    //==========================================================================
    
    /// Initialize with noise profile
    void initialize(const NoiseProfile& noise);
    
    /// Reset controller state
    void reset();
    
    //==========================================================================
    // QEC Execution
    //==========================================================================
    
    /// Process a single QEC round result
    void process_round(const QECRoundResult& result);
    
    /// Decode syndrome using current decoder
    Correction decode(const Syndrome& syndrome);
    
    //==========================================================================
    // Adaptation
    //==========================================================================
    
    /// Check if any adaptation is needed
    bool needs_adaptation() const;
    
    /// Apply pending adaptations (code switch, distance change, recalibration)
    void apply_adaptations();
    
    /// Force adaptation check
    void check_adaptations();
    
    //==========================================================================
    // State Access
    //==========================================================================
    
    /// Get current code type
    StabilizerCodeType current_code_type() const { return current_code_type_; }
    
    /// Get current distance
    size_t current_distance() const { return current_distance_; }
    
    /// Get current decoder
    Decoder& current_decoder();
    
    /// Get noise profile
    const NoiseProfile& noise_profile() const { return noise_profile_; }
    
    //==========================================================================
    // Statistics
    //==========================================================================
    
    struct Stats {
        size_t total_rounds = 0;
        size_t code_switches = 0;
        size_t distance_changes = 0;
        size_t recalibrations = 0;
        size_t ml_decodes = 0;
        size_t mwpm_decodes = 0;
    };
    
    Stats stats() const { return stats_; }
    
private:
    Config config_;
    Stats stats_;
    
    // Current state
    StabilizerCodeType current_code_type_;
    size_t current_distance_;
    NoiseProfile noise_profile_;
    
    // Components
    std::unique_ptr<StabilizerCode> current_code_;
    std::unique_ptr<AdaptiveCodeSelector> code_selector_;
    std::unique_ptr<ClosedLoopController> closed_loop_;
    std::unique_ptr<DynamicDistanceSelector> distance_selector_;
    std::unique_ptr<MLDecoder> ml_decoder_;
    std::unique_ptr<MWPMDecoder> mwpm_decoder_;
    
    // Pending adaptations
    bool pending_code_switch_ = false;
    bool pending_distance_change_ = false;
    bool pending_recalibration_ = false;
    StabilizerCodeType pending_code_type_;
    size_t pending_distance_;
    
    /// Update code based on new selection
    void switch_code(StabilizerCodeType new_type, size_t new_distance);
    
    /// Create decoder for current code
    void create_decoder();
    
    /// Get ML model path for code type and distance
    std::string get_model_path(StabilizerCodeType code, size_t distance);
};

//==============================================================================
// Training Data Types (for Python bridge)
//==============================================================================

/**
 * @brief Training sample for ML decoder
 */
struct TrainingSample {
    std::vector<int> syndrome;   // Concatenated X + Z syndrome bits
    std::vector<int> error;      // Error pattern (0=I, 1=X, 2=Z, 3=Y per qubit)
    
    nlohmann::json to_json() const;
    static TrainingSample from_json(const nlohmann::json& j);
};

/**
 * @brief Training dataset metadata
 */
struct TrainingDatasetMetadata {
    std::string code_type;
    size_t code_distance;
    size_t num_samples;
    NoiseProfile noise_profile;
    std::string creation_timestamp;
    
    nlohmann::json to_json() const;
    static TrainingDatasetMetadata from_json(const nlohmann::json& j);
};

//==============================================================================
// Utility Functions
//==============================================================================

/// Generate synthetic training data for ML decoder
std::vector<TrainingSample> generate_training_data(
    const StabilizerCode& code,
    const NoiseProfile& noise,
    size_t num_samples,
    unsigned int seed = 42
);

/// Evaluate decoder accuracy on test data
double evaluate_decoder_accuracy(
    Decoder& decoder,
    const std::vector<TrainingSample>& test_data
);

/// Compare two decoders on same test data
std::pair<double, double> compare_decoders(
    Decoder& decoder1,
    Decoder& decoder2,
    const std::vector<TrainingSample>& test_data
);

}  // namespace qlret
