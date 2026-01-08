#include "qec_adaptive.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace qlret {

//==============================================================================
// NoiseProfile Implementation
//==============================================================================

double NoiseProfile::avg_gate_error() const {
    if (single_gate_errors.empty()) return 0.0;
    double sum = std::accumulate(single_gate_errors.begin(), single_gate_errors.end(), 0.0);
    return sum / single_gate_errors.size();
}

double NoiseProfile::max_gate_error() const {
    if (single_gate_errors.empty()) return 0.0;
    return *std::max_element(single_gate_errors.begin(), single_gate_errors.end());
}

double NoiseProfile::avg_two_qubit_error() const {
    if (two_qubit_errors.empty()) return 0.0;
    double sum = 0.0;
    for (const auto& [pair, error] : two_qubit_errors) {
        sum += error;
    }
    return sum / two_qubit_errors.size();
}

double NoiseProfile::max_two_qubit_error() const {
    if (two_qubit_errors.empty()) return 0.0;
    double max_err = 0.0;
    for (const auto& [pair, error] : two_qubit_errors) {
        max_err = std::max(max_err, error);
    }
    return max_err;
}

double NoiseProfile::avg_t1() const {
    if (t1_times_ns.empty()) return 0.0;
    double sum = std::accumulate(t1_times_ns.begin(), t1_times_ns.end(), 0.0);
    return sum / t1_times_ns.size();
}

double NoiseProfile::avg_t2() const {
    if (t2_times_ns.empty()) return 0.0;
    double sum = std::accumulate(t2_times_ns.begin(), t2_times_ns.end(), 0.0);
    return sum / t2_times_ns.size();
}

double NoiseProfile::t1_t2_ratio() const {
    double t2 = avg_t2();
    if (t2 < 1e-9) return 1.0;  // Avoid division by zero
    return avg_t1() / t2;
}

bool NoiseProfile::is_biased(double threshold) const {
    double ratio = t1_t2_ratio();
    return std::abs(ratio - 1.0) > threshold;
}

bool NoiseProfile::has_correlations() const {
    return !correlated_errors.empty();
}

bool NoiseProfile::has_time_varying() const {
    return time_varying.alpha > 0.0 || time_varying.beta > 0.0;
}

bool NoiseProfile::has_memory_effects() const {
    return !memory_effects.empty();
}

double NoiseProfile::avg_readout_error() const {
    if (readout_errors.empty()) return 0.0;
    double sum = std::accumulate(readout_errors.begin(), readout_errors.end(), 0.0);
    return sum / readout_errors.size();
}

double NoiseProfile::effective_physical_error_rate() const {
    // Weighted combination of different error sources
    double gate_err = avg_gate_error();
    double two_q_err = avg_two_qubit_error();
    double readout_err = avg_readout_error();
    
    // Weight two-qubit errors more heavily (they dominate in QEC)
    return gate_err + 2.0 * two_q_err + 0.5 * readout_err;
}

nlohmann::json NoiseProfile::to_json() const {
    nlohmann::json j;
    
    j["device_name"] = device_name;
    j["calibration_timestamp"] = calibration_timestamp;
    j["num_qubits"] = num_qubits;
    j["t1_times_ns"] = t1_times_ns;
    j["t2_times_ns"] = t2_times_ns;
    j["single_gate_errors"] = single_gate_errors;
    j["readout_errors"] = readout_errors;
    
    // Two-qubit errors (convert pair keys to strings)
    nlohmann::json two_q_json;
    for (const auto& [pair, error] : two_qubit_errors) {
        std::string key = "(" + std::to_string(pair.first) + "," + 
                          std::to_string(pair.second) + ")";
        two_q_json[key] = error;
    }
    j["two_qubit_errors"] = two_q_json;
    
    // Correlated errors (simplified)
    if (!correlated_errors.empty()) {
        nlohmann::json corr_json = nlohmann::json::array();
        for (const auto& ce : correlated_errors) {
            nlohmann::json ce_json;
            ce_json["qubit_i"] = ce.qubit_i;
            ce_json["qubit_j"] = ce.qubit_j;
            ce_json["coupling_strength_hz"] = ce.coupling_strength_hz;
            corr_json.push_back(ce_json);
        }
        j["correlated_errors"] = corr_json;
    }
    
    // Time-varying
    if (has_time_varying()) {
        nlohmann::json tv_json;
        tv_json["base_t1_ns"] = time_varying.base_t1_ns;
        tv_json["base_t2_ns"] = time_varying.base_t2_ns;
        tv_json["base_depol_prob"] = time_varying.base_depol_prob;
        tv_json["model"] = static_cast<int>(time_varying.model);
        tv_json["alpha"] = time_varying.alpha;
        tv_json["beta"] = time_varying.beta;
        j["time_varying"] = tv_json;
    }
    
    return j;
}

NoiseProfile NoiseProfile::from_json(const nlohmann::json& j) {
    NoiseProfile profile;
    
    if (j.contains("device_name")) profile.device_name = j["device_name"];
    if (j.contains("calibration_timestamp")) profile.calibration_timestamp = j["calibration_timestamp"];
    if (j.contains("num_qubits")) profile.num_qubits = j["num_qubits"];
    if (j.contains("t1_times_ns")) profile.t1_times_ns = j["t1_times_ns"].get<std::vector<double>>();
    if (j.contains("t2_times_ns")) profile.t2_times_ns = j["t2_times_ns"].get<std::vector<double>>();
    if (j.contains("single_gate_errors")) profile.single_gate_errors = j["single_gate_errors"].get<std::vector<double>>();
    if (j.contains("readout_errors")) profile.readout_errors = j["readout_errors"].get<std::vector<double>>();
    
    // Two-qubit errors
    if (j.contains("two_qubit_errors")) {
        for (auto& [key, value] : j["two_qubit_errors"].items()) {
            // Parse "(i,j)" format
            size_t comma = key.find(',');
            if (comma != std::string::npos && key.front() == '(' && key.back() == ')') {
                size_t i = std::stoul(key.substr(1, comma - 1));
                size_t jj = std::stoul(key.substr(comma + 1, key.size() - comma - 2));
                profile.two_qubit_errors[{i, jj}] = value.get<double>();
            }
        }
    }
    
    // Correlated errors
    if (j.contains("correlated_errors")) {
        for (const auto& ce_json : j["correlated_errors"]) {
            CorrelatedError ce;
            ce.qubit_i = ce_json["qubit_i"];
            ce.qubit_j = ce_json["qubit_j"];
            if (ce_json.contains("coupling_strength_hz")) {
                ce.coupling_strength_hz = ce_json["coupling_strength_hz"];
            }
            profile.correlated_errors.push_back(ce);
        }
    }
    
    // Time-varying
    if (j.contains("time_varying") && !j["time_varying"].is_null()) {
        const auto& tv = j["time_varying"];
        profile.time_varying.base_t1_ns = tv.value("base_t1_ns", 0.0);
        profile.time_varying.base_t2_ns = tv.value("base_t2_ns", 0.0);
        profile.time_varying.base_depol_prob = tv.value("base_depol_prob", 0.0);
        profile.time_varying.model = static_cast<TimeScalingModel>(tv.value("model", 0));
        profile.time_varying.alpha = tv.value("alpha", 0.0);
        profile.time_varying.beta = tv.value("beta", 0.0);
    }
    
    return profile;
}

NoiseProfile NoiseProfile::load_from_calibration(const std::string& calib_file) {
    std::ifstream file(calib_file);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open calibration file: " + calib_file);
    }
    
    nlohmann::json j;
    file >> j;
    return from_json(j);
}

void NoiseProfile::save(const std::string& path) const {
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    file << to_json().dump(2);
}

double NoiseProfile::relative_difference(const NoiseProfile& other) const {
    // Compare key metrics
    double diff = 0.0;
    int count = 0;
    
    // Gate error difference
    double this_err = avg_gate_error();
    double other_err = other.avg_gate_error();
    if (this_err > 1e-9) {
        diff += std::abs(this_err - other_err) / this_err;
        count++;
    }
    
    // Two-qubit error difference
    double this_2q = avg_two_qubit_error();
    double other_2q = other.avg_two_qubit_error();
    if (this_2q > 1e-9) {
        diff += std::abs(this_2q - other_2q) / this_2q;
        count++;
    }
    
    // T1/T2 ratio difference
    double this_ratio = t1_t2_ratio();
    double other_ratio = other.t1_t2_ratio();
    if (this_ratio > 1e-9) {
        diff += std::abs(this_ratio - other_ratio) / this_ratio;
        count++;
    }
    
    return count > 0 ? diff / count : 0.0;
}

bool NoiseProfile::differs_from(const NoiseProfile& baseline, double threshold) const {
    return relative_difference(baseline) > threshold;
}

//==============================================================================
// AdaptiveCodeSelector Implementation
//==============================================================================

AdaptiveCodeSelector::AdaptiveCodeSelector()
    : AdaptiveCodeSelector(Config{}) {}

AdaptiveCodeSelector::AdaptiveCodeSelector(Config config)
    : config_(config) {}

QECCodeType AdaptiveCodeSelector::select_code(const NoiseProfile& noise) {
    // Decision tree based on noise characteristics
    
    // 1. High error rate → prefer higher distance or concatenated
    if (noise.avg_gate_error() > config_.high_error_threshold) {
        return select_for_high_error_rate(noise);
    }
    
    // 2. Biased noise → repetition or biased codes
    if (noise.is_biased(config_.bias_threshold)) {
        return select_for_biased_noise(noise);
    }
    
    // 3. Correlated errors → surface code handles clusters better
    if (noise.has_correlations()) {
        return select_for_correlated_noise(noise);
    }
    
    // 4. Low error rate → can use smaller codes
    if (noise.avg_gate_error() < config_.low_error_threshold) {
        return select_for_low_error_rate(noise);
    }
    
    // 5. Default: standard rotated surface code
    return QECCodeType::SURFACE;
}

size_t AdaptiveCodeSelector::select_distance(
    QECCodeType code,
    const NoiseProfile& noise,
    double target_logical_error_rate
) {
    // Binary search for minimum distance that meets target
    for (size_t d = config_.min_distance; d <= config_.max_distance; d += 2) {
        double predicted = predict_logical_error_rate(code, d, noise);
        if (predicted <= target_logical_error_rate) {
            return d;
        }
    }
    
    // Return max if target can't be met
    return config_.max_distance;
}

std::pair<QECCodeType, size_t> AdaptiveCodeSelector::select_code_and_distance(
    const NoiseProfile& noise,
    double target_logical_error_rate
) {
    QECCodeType code = select_code(noise);
    size_t distance = select_distance(code, noise, target_logical_error_rate);
    return {code, distance};
}

QECCodeType AdaptiveCodeSelector::select_for_biased_noise(const NoiseProfile& noise) {
    double ratio = noise.t1_t2_ratio();
    
    // T1 >> T2: Z errors dominate → repetition code for bit-flip
    // T1 << T2: X errors dominate → repetition code for phase-flip
    // Either way, repetition code can be more efficient for single-type errors
    
    if (ratio > 1.5 || ratio < 0.67) {
        return QECCodeType::REPETITION;
    }
    
    // Moderate bias: surface code is still good
    return QECCodeType::SURFACE;
}

QECCodeType AdaptiveCodeSelector::select_for_correlated_noise(const NoiseProfile& noise) {
    // Surface code with its 2D structure handles correlated errors better
    // than 1D repetition codes
    return QECCodeType::SURFACE;
}

QECCodeType AdaptiveCodeSelector::select_for_high_error_rate(const NoiseProfile& noise) {
    // For very high error rates, we may need concatenated codes
    // or simply larger distance surface codes
    // For now, return SURFACE with expectation of higher distance
    return QECCodeType::SURFACE;
}

QECCodeType AdaptiveCodeSelector::select_for_low_error_rate(const NoiseProfile& noise) {
    // Low error rate: surface code is efficient
    // Could use smaller distance
    if (config_.prefer_low_overhead) {
        return QECCodeType::SURFACE;
    }
    return QECCodeType::SURFACE;
}

double AdaptiveCodeSelector::predict_logical_error_rate(
    QECCodeType code,
    size_t distance,
    const NoiseProfile& noise
) {
    double p_phys = compute_effective_error_rate(code, noise);
    double A = get_threshold_constant(code);
    double exp = get_exponent(code, distance);
    
    // p_L ≈ A * p_phys^exp
    return A * std::pow(p_phys, exp);
}

std::vector<std::pair<QECCodeType, double>> AdaptiveCodeSelector::rank_codes(
    const NoiseProfile& noise,
    size_t distance
) {
    std::vector<std::pair<QECCodeType, double>> rankings;
    
    std::vector<QECCodeType> codes = {
        QECCodeType::SURFACE,
        QECCodeType::REPETITION
    };
    
    for (auto code : codes) {
        double error_rate = predict_logical_error_rate(code, distance, noise);
        rankings.push_back({code, error_rate});
    }
    
    // Sort by error rate (ascending)
    std::sort(rankings.begin(), rankings.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    return rankings;
}

double AdaptiveCodeSelector::compute_effective_error_rate(
    QECCodeType code,
    const NoiseProfile& noise
) {
    // Different codes have different sensitivities to different error types
    double gate_err = noise.avg_gate_error();
    double two_q_err = noise.avg_two_qubit_error();
    
    switch (code) {
        case QECCodeType::REPETITION:
            // Repetition code only protects against one type of error
            // Effective rate is dominated by single-qubit errors
            return gate_err + two_q_err;
            
        case QECCodeType::SURFACE:
        default:
            // Surface code protects against both X and Z
            // Weight two-qubit errors heavily (syndrome circuits use CNOT)
            return gate_err + 2.0 * two_q_err;
    }
}

double AdaptiveCodeSelector::get_threshold_constant(QECCodeType code) {
    // Code-specific threshold constants (empirically determined)
    switch (code) {
        case QECCodeType::REPETITION:
            return 0.5;   // Higher constant (less protection)
        case QECCodeType::SURFACE:
        default:
            return 0.1;   // Lower constant (better protection)
    }
}

double AdaptiveCodeSelector::get_exponent(QECCodeType code, size_t distance) {
    switch (code) {
        case QECCodeType::REPETITION:
            // p_L ≈ A * p^d for repetition code (bit-flip only)
            return static_cast<double>(distance);
        case QECCodeType::SURFACE:
        default:
            // p_L ≈ A * p^((d+1)/2) for surface code
            return (distance + 1.0) / 2.0;
    }
}

//==============================================================================
// MLDecoder Implementation (Stub - Python bridge to be added)
//==============================================================================

struct MLDecoder::PythonModel {
    // Placeholder for Python model handle
    // In full implementation, this would hold pybind11 objects
    bool loaded = false;
    std::string path;
};

MLDecoder::MLDecoder(const StabilizerCode& code, Config config)
    : code_(code), config_(config) {
    
    py_model_ = std::make_unique<PythonModel>();
    
    // Create fallback decoder
    MWPMDecoder::Config mwpm_config;
    mwpm_config.physical_error_rate = 0.001;  // Default
    fallback_decoder_ = std::make_unique<MWPMDecoder>(code, mwpm_config);
    
    // Load model if path provided
    if (!config_.model_path.empty()) {
        load_model(config_.model_path);
    }
}

MLDecoder::~MLDecoder() = default;

Correction MLDecoder::decode(const Syndrome& syndrome) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // If model not loaded, use fallback
    if (!model_loaded_ || !py_model_->loaded) {
        stats_.num_inferences++;
        stats_.num_fallbacks++;
        return fallback_decoder_->decode(syndrome);
    }
    
    // Convert syndrome to tensor
    std::vector<float> input = syndrome_to_tensor(syndrome);
    
    // In full implementation: call Python model
    // For now, use fallback
    std::vector<float> logits(code_.num_data_qubits() * 4, 0.0f);
    
    // Check confidence and potentially fallback
    if (config_.enable_fallback && should_fallback(logits)) {
        stats_.num_fallbacks++;
        auto end = std::chrono::high_resolution_clock::now();
        stats_.total_inference_time_ms += 
            std::chrono::duration<double, std::milli>(end - start).count();
        stats_.num_inferences++;
        return fallback_decoder_->decode(syndrome);
    }
    
    Correction correction = tensor_to_correction(logits);
    
    auto end = std::chrono::high_resolution_clock::now();
    stats_.total_inference_time_ms += 
        std::chrono::duration<double, std::milli>(end - start).count();
    stats_.num_inferences++;
    
    return correction;
}

std::vector<Correction> MLDecoder::decode_batch(const std::vector<Syndrome>& syndromes) {
    std::vector<Correction> corrections;
    corrections.reserve(syndromes.size());
    
    // In full implementation: batch inference
    // For now: sequential fallback
    for (const auto& syn : syndromes) {
        corrections.push_back(decode(syn));
    }
    
    return corrections;
}

bool MLDecoder::has_logical_error(const Correction& correction, 
                                   const PauliString& actual_error) const {
    // Delegate to fallback decoder for logical error checking
    return fallback_decoder_->has_logical_error(correction, actual_error);
}

void MLDecoder::load_model(const std::string& path) {
    // In full implementation: load Python model via pybind11
    // For now: just mark as loaded
    py_model_->path = path;
    py_model_->loaded = true;
    model_loaded_ = true;
    config_.model_path = path;
}

void MLDecoder::reload_model() {
    if (!config_.model_path.empty()) {
        load_model(config_.model_path);
    }
}

std::vector<float> MLDecoder::syndrome_to_tensor(const Syndrome& syndrome) {
    std::vector<float> tensor;
    tensor.reserve(syndrome.x_syndrome.size() + syndrome.z_syndrome.size());
    
    for (int s : syndrome.x_syndrome) {
        tensor.push_back(static_cast<float>(s));
    }
    for (int s : syndrome.z_syndrome) {
        tensor.push_back(static_cast<float>(s));
    }
    
    return tensor;
}

Correction MLDecoder::tensor_to_correction(const std::vector<float>& logits) {
    size_t n_qubits = code_.num_data_qubits();
    Correction correction;
    correction.x_correction = PauliString(n_qubits);
    correction.z_correction = PauliString(n_qubits);
    
    for (size_t q = 0; q < n_qubits; ++q) {
        size_t offset = q * 4;
        if (offset + 3 >= logits.size()) break;
        
        // Find argmax for this qubit
        int max_idx = 0;
        float max_val = logits[offset];
        for (int i = 1; i < 4; ++i) {
            if (logits[offset + i] > max_val) {
                max_val = logits[offset + i];
                max_idx = i;
            }
        }
        
        // Map: 0=I, 1=X, 2=Z, 3=Y
        if (max_idx == 1) {
            correction.x_correction.set(q, Pauli::X);
        } else if (max_idx == 2) {
            correction.z_correction.set(q, Pauli::Z);
        } else if (max_idx == 3) {
            correction.x_correction.set(q, Pauli::X);
            correction.z_correction.set(q, Pauli::Z);
        }
    }
    
    return correction;
}

bool MLDecoder::should_fallback(const std::vector<float>& logits) {
    // Check if maximum confidence is below threshold
    float max_conf = 0.0f;
    for (size_t q = 0; q < code_.num_data_qubits(); ++q) {
        size_t offset = q * 4;
        if (offset + 3 >= logits.size()) break;
        
        float qmax = *std::max_element(logits.begin() + offset, 
                                        logits.begin() + offset + 4);
        max_conf = std::max(max_conf, qmax);
    }
    
    return max_conf < config_.fallback_threshold;
}

//==============================================================================
// ClosedLoopController Implementation
//==============================================================================

ClosedLoopController::ClosedLoopController()
    : ClosedLoopController(Config{}) {}

ClosedLoopController::ClosedLoopController(Config config)
    : config_(config) {}

void ClosedLoopController::update(const QECRoundResult& result) {
    stats_.total_cycles++;
    
    if (result.logical_error) {
        stats_.logical_errors++;
        recent_errors_.push_back(true);
    } else {
        recent_errors_.push_back(false);
    }
    
    // Maintain window size
    while (recent_errors_.size() > config_.window_size) {
        recent_errors_.pop_front();
    }
    
    // Update current rate
    stats_.current_logical_error_rate = compute_current_rate();
    
    // Set baseline after first window
    if (stats_.total_cycles == config_.window_size && 
        stats_.baseline_logical_error_rate < 1e-12) {
        stats_.baseline_logical_error_rate = stats_.current_logical_error_rate;
    }
}

bool ClosedLoopController::should_recalibrate() const {
    // Check interval
    if (stats_.total_cycles % config_.recalibration_interval != 0) {
        return false;
    }
    
    // Check minimum cycles
    if (stats_.total_cycles < config_.min_cycles_before_recalib) {
        return false;
    }
    
    // Check if enough data in window
    if (recent_errors_.size() < config_.window_size) {
        return false;
    }
    
    return detect_drift();
}

bool ClosedLoopController::check_drift() const {
    if (recent_errors_.size() < config_.window_size) {
        return false;
    }
    return detect_drift();
}

bool ClosedLoopController::detect_drift() const {
    double current = stats_.current_logical_error_rate;
    double baseline = stats_.baseline_logical_error_rate;
    
    if (baseline < 1e-12) {
        return false;  // Baseline not established
    }
    
    double relative_change = std::abs(current - baseline) / baseline;
    return relative_change > config_.drift_threshold;
}

double ClosedLoopController::compute_current_rate() const {
    if (recent_errors_.empty()) return 0.0;
    
    size_t errors = 0;
    for (bool e : recent_errors_) {
        if (e) errors++;
    }
    
    return static_cast<double>(errors) / recent_errors_.size();
}

NoiseProfile ClosedLoopController::recalibrate() {
    if (!config_.auto_recalibrate) {
        return current_noise_;
    }
    
    // Call Phase 4.2 calibration script
    std::string cmd = "python " + config_.calibration_script_path + 
                      " --output " + config_.calibration_output_path;
    
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        // Calibration failed, return current noise
        return current_noise_;
    }
    
    // Load new noise profile
    try {
        NoiseProfile new_profile = NoiseProfile::load_from_calibration(
            config_.calibration_output_path
        );
        
        stats_.num_recalibrations++;
        update_baseline(new_profile);
        
        return new_profile;
    } catch (const std::exception&) {
        return current_noise_;
    }
}

void ClosedLoopController::update_baseline(const NoiseProfile& new_noise) {
    current_noise_ = new_noise;
    baseline_noise_ = new_noise;
    stats_.baseline_logical_error_rate = stats_.current_logical_error_rate;
}

void ClosedLoopController::reset() {
    stats_ = Stats{};
    recent_errors_.clear();
}

std::string ClosedLoopController::select_model_for_noise(const NoiseProfile& noise) {
    // Select pre-trained model based on noise characteristics
    // In production, would have library of models for different noise regimes
    
    std::string model_base = "models/decoder";
    
    if (noise.is_biased()) {
        model_base += "_biased";
    } else if (noise.has_correlations()) {
        model_base += "_correlated";
    } else {
        model_base += "_standard";
    }
    
    return model_base + ".pkl";
}

void ClosedLoopController::set_noise_profile(const NoiseProfile& noise) {
    current_noise_ = noise;
    if (stats_.total_cycles == 0) {
        baseline_noise_ = noise;
    }
}

//==============================================================================
// DynamicDistanceSelector Implementation
//==============================================================================

DynamicDistanceSelector::DynamicDistanceSelector()
    : DynamicDistanceSelector(Config{}) {}

DynamicDistanceSelector::DynamicDistanceSelector(Config config)
    : config_(config), current_distance_(config.min_distance) {}

void DynamicDistanceSelector::update(const QECRoundResult& result) {
    recent_errors_.push_back(result.logical_error);
    
    while (recent_errors_.size() > config_.evaluation_window) {
        recent_errors_.pop_front();
    }
}

size_t DynamicDistanceSelector::recommended_distance() const {
    if (recent_errors_.size() < config_.evaluation_window) {
        return current_distance_;
    }
    
    // Compute current error rate
    size_t errors = 0;
    for (bool e : recent_errors_) {
        if (e) errors++;
    }
    double current_rate = static_cast<double>(errors) / recent_errors_.size();
    
    // Compare to target
    if (current_rate > target_error_rate_ * config_.increase_threshold) {
        // Increase distance
        size_t new_d = current_distance_ + 2;
        return std::min(new_d, config_.max_distance);
    } else if (current_rate < target_error_rate_ * config_.decrease_threshold) {
        // Decrease distance
        if (current_distance_ > config_.min_distance) {
            return current_distance_ - 2;
        }
    }
    
    return current_distance_;
}

bool DynamicDistanceSelector::should_change_distance() const {
    return recommended_distance() != current_distance_;
}

void DynamicDistanceSelector::set_current_distance(size_t d) {
    current_distance_ = d;
    recent_errors_.clear();  // Reset window on distance change
}

void DynamicDistanceSelector::set_target_error_rate(double target) {
    target_error_rate_ = target;
}

//==============================================================================
// AdaptiveQECController Implementation
//==============================================================================

AdaptiveQECController::AdaptiveQECController()
    : AdaptiveQECController(Config{}) {}

AdaptiveQECController::AdaptiveQECController(Config config)
    : config_(config),
      current_code_type_(config.initial_code),
      current_distance_(config.initial_distance) {
    
    code_selector_ = std::make_unique<AdaptiveCodeSelector>(config.selector_config);
    closed_loop_ = std::make_unique<ClosedLoopController>(config.closed_loop_config);
    distance_selector_ = std::make_unique<DynamicDistanceSelector>(config.distance_config);
    
    distance_selector_->set_target_error_rate(config.target_logical_error_rate);
    distance_selector_->set_current_distance(config.initial_distance);
}

void AdaptiveQECController::initialize(const NoiseProfile& noise) {
    noise_profile_ = noise;
    closed_loop_->set_noise_profile(noise);
    
    // Select initial code based on noise
    if (config_.enable_code_switching) {
        auto [code_type, distance] = code_selector_->select_code_and_distance(
            noise, config_.target_logical_error_rate);
        current_code_type_ = code_type;
        current_distance_ = distance;
    }
    
    // Create code and decoder
    current_code_ = create_stabilizer_code(current_code_type_, current_distance_);
    create_decoder();
}

void AdaptiveQECController::reset() {
    stats_ = Stats{};
    closed_loop_->reset();
    pending_code_switch_ = false;
    pending_distance_change_ = false;
    pending_recalibration_ = false;
}

void AdaptiveQECController::process_round(const QECRoundResult& result) {
    stats_.total_rounds++;
    
    // Update components
    closed_loop_->update(result);
    distance_selector_->update(result);
    
    // Check for needed adaptations
    check_adaptations();
}

Correction AdaptiveQECController::decode(const Syndrome& syndrome) {
    if (config_.use_ml_decoder && ml_decoder_ && ml_decoder_->is_ready()) {
        stats_.ml_decodes++;
        return ml_decoder_->decode(syndrome);
    } else if (mwpm_decoder_) {
        stats_.mwpm_decodes++;
        return mwpm_decoder_->decode(syndrome);
    }
    
    // Fallback: empty correction
    return Correction{
        PauliString(current_code_->num_data_qubits()),
        PauliString(current_code_->num_data_qubits())
    };
}

bool AdaptiveQECController::needs_adaptation() const {
    return pending_code_switch_ || pending_distance_change_ || pending_recalibration_;
}

void AdaptiveQECController::check_adaptations() {
    // Check for recalibration need
    if (config_.enable_closed_loop && closed_loop_->should_recalibrate()) {
        pending_recalibration_ = true;
    }
    
    // Check for distance change
    if (config_.enable_distance_adaptation && distance_selector_->should_change_distance()) {
        pending_distance_change_ = true;
        pending_distance_ = distance_selector_->recommended_distance();
    }
    
    // Check for code switch (only on recalibration)
    if (pending_recalibration_ && config_.enable_code_switching) {
        auto new_code = code_selector_->select_code(noise_profile_);
        if (new_code != current_code_type_) {
            pending_code_switch_ = true;
            pending_code_type_ = new_code;
        }
    }
}

void AdaptiveQECController::apply_adaptations() {
    if (pending_recalibration_) {
        noise_profile_ = closed_loop_->recalibrate();
        stats_.recalibrations++;
        pending_recalibration_ = false;
    }
    
    if (pending_code_switch_ || pending_distance_change_) {
        QECCodeType new_type = pending_code_switch_ ? pending_code_type_ : current_code_type_;
        size_t new_distance = pending_distance_change_ ? pending_distance_ : current_distance_;
        
        switch_code(new_type, new_distance);
        
        if (pending_code_switch_) stats_.code_switches++;
        if (pending_distance_change_) stats_.distance_changes++;
        
        pending_code_switch_ = false;
        pending_distance_change_ = false;
    }
}

QECDecoder& AdaptiveQECController::current_decoder() {
    if (config_.use_ml_decoder && ml_decoder_ && ml_decoder_->is_ready()) {
        return *ml_decoder_;
    }
    return *mwpm_decoder_;
}

void AdaptiveQECController::switch_code(QECCodeType new_type, size_t new_distance) {
    current_code_type_ = new_type;
    current_distance_ = new_distance;
    distance_selector_->set_current_distance(new_distance);
    
    current_code_ = create_stabilizer_code(new_type, new_distance);
    create_decoder();
}

void AdaptiveQECController::create_decoder() {
    // Create MWPM decoder
    MWPMDecoder::Config mwpm_config;
    mwpm_config.physical_error_rate = noise_profile_.effective_physical_error_rate();
    mwpm_decoder_ = std::make_unique<MWPMDecoder>(*current_code_, mwpm_config);
    
    // Create ML decoder if enabled
    if (config_.use_ml_decoder) {
        MLDecoder::Config ml_config;
        ml_config.model_path = get_model_path(current_code_type_, current_distance_);
        ml_config.enable_fallback = true;
        
        try {
            ml_decoder_ = std::make_unique<MLDecoder>(*current_code_, ml_config);
        } catch (const std::exception&) {
            // ML decoder not available, will use MWPM fallback
            ml_decoder_.reset();
        }
    }
}

std::string AdaptiveQECController::get_model_path(QECCodeType code, size_t distance) {
    std::string code_name;
    switch (code) {
        case QECCodeType::REPETITION:
            code_name = "repetition";
            break;
        case QECCodeType::SURFACE:
        default:
            code_name = "surface";
            break;
    }
    
    return config_.ml_model_dir + "decoder_" + code_name + "_d" + 
           std::to_string(distance) + ".pkl";
}

//==============================================================================
// Training Data Types Implementation
//==============================================================================

nlohmann::json TrainingSample::to_json() const {
    return {
        {"syndrome", syndrome},
        {"error", error}
    };
}

TrainingSample TrainingSample::from_json(const nlohmann::json& j) {
    TrainingSample sample;
    sample.syndrome = j["syndrome"].get<std::vector<int>>();
    sample.error = j["error"].get<std::vector<int>>();
    return sample;
}

nlohmann::json TrainingDatasetMetadata::to_json() const {
    return {
        {"code_type", code_type},
        {"code_distance", code_distance},
        {"num_samples", num_samples},
        {"noise_profile", noise_profile.to_json()},
        {"creation_timestamp", creation_timestamp}
    };
}

TrainingDatasetMetadata TrainingDatasetMetadata::from_json(const nlohmann::json& j) {
    TrainingDatasetMetadata meta;
    meta.code_type = j["code_type"];
    meta.code_distance = j["code_distance"];
    meta.num_samples = j["num_samples"];
    meta.noise_profile = NoiseProfile::from_json(j["noise_profile"]);
    meta.creation_timestamp = j["creation_timestamp"];
    return meta;
}

//==============================================================================
// Utility Functions Implementation
//==============================================================================

std::vector<TrainingSample> generate_training_data(
    const StabilizerCode& code,
    const NoiseProfile& noise,
    size_t num_samples,
    unsigned int seed
) {
    std::vector<TrainingSample> samples;
    samples.reserve(num_samples);
    
    std::mt19937 rng(seed);
    double p = noise.effective_physical_error_rate();
    std::uniform_real_distribution<> uniform(0.0, 1.0);
    std::uniform_int_distribution<> pauli_dist(1, 3);  // X, Z, Y
    
    ErrorInjector injector(code.num_data_qubits());
    SyndromeExtractor extractor(code);
    
    for (size_t i = 0; i < num_samples; ++i) {
        TrainingSample sample;
        
        // Generate random error pattern
        PauliString error(code.num_data_qubits());
        sample.error.resize(code.num_data_qubits(), 0);
        
        for (size_t q = 0; q < code.num_data_qubits(); ++q) {
            if (uniform(rng) < p) {
                int pauli = pauli_dist(rng);
                sample.error[q] = pauli;
                
                if (pauli == 1) error.set(q, Pauli::X);
                else if (pauli == 2) error.set(q, Pauli::Z);
                else error.set(q, Pauli::Y);
            }
        }
        
        // Extract syndrome
        Syndrome syn = extractor.extract(error);
        
        // Add measurement noise
        double p_meas = noise.avg_readout_error();
        for (int& s : syn.x_syndrome) {
            if (uniform(rng) < p_meas) s ^= 1;
        }
        for (int& s : syn.z_syndrome) {
            if (uniform(rng) < p_meas) s ^= 1;
        }
        
        // Flatten syndrome
        sample.syndrome.reserve(syn.x_syndrome.size() + syn.z_syndrome.size());
        sample.syndrome.insert(sample.syndrome.end(), 
                               syn.x_syndrome.begin(), syn.x_syndrome.end());
        sample.syndrome.insert(sample.syndrome.end(), 
                               syn.z_syndrome.begin(), syn.z_syndrome.end());
        
        samples.push_back(std::move(sample));
    }
    
    return samples;
}

double evaluate_decoder_accuracy(
    QECDecoder& decoder,
    const std::vector<TrainingSample>& test_data
) {
    size_t correct = 0;
    
    for (const auto& sample : test_data) {
        // Reconstruct syndrome
        Syndrome syn;
        size_t mid = sample.syndrome.size() / 2;
        syn.x_syndrome.assign(sample.syndrome.begin(), sample.syndrome.begin() + mid);
        syn.z_syndrome.assign(sample.syndrome.begin() + mid, sample.syndrome.end());
        
        // Decode
        Correction corr = decoder.decode(syn);
        
        // Check if correction matches error (simplified)
        bool correct_decode = true;
        for (size_t q = 0; q < sample.error.size() && q < corr.x_correction.size(); ++q) {
            int expected = sample.error[q];
            bool has_x = (corr.x_correction[q] == Pauli::X);
            bool has_z = (corr.z_correction[q] == Pauli::Z);
            
            int decoded = 0;
            if (has_x && has_z) decoded = 3;
            else if (has_x) decoded = 1;
            else if (has_z) decoded = 2;
            
            if (decoded != expected) {
                correct_decode = false;
                break;
            }
        }
        
        if (correct_decode) correct++;
    }
    
    return static_cast<double>(correct) / test_data.size();
}

std::pair<double, double> compare_decoders(
    QECDecoder& decoder1,
    QECDecoder& decoder2,
    const std::vector<TrainingSample>& test_data
) {
    double acc1 = evaluate_decoder_accuracy(decoder1, test_data);
    double acc2 = evaluate_decoder_accuracy(decoder2, test_data);
    return {acc1, acc2};
}

}  // namespace qlret
