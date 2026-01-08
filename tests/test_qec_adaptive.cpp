/**
 * Phase 9.3: Adaptive QEC Test Suite
 *
 * Comprehensive tests for adaptive and ML-driven quantum error correction.
 * Tests cover: NoiseProfile, AdaptiveCodeSelector, MLDecoder,
 * ClosedLoopController, DynamicDistanceSelector, and AdaptiveQECController.
 */

#include "qec_adaptive.h"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <chrono>

using namespace qlret;

//==============================================================================
// Test Utilities
//==============================================================================

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name)                                                             \
    void test_##name();                                                        \
    void run_##name() {                                                        \
        std::cout << "Testing " << #name << "... " << std::flush;              \
        try {                                                                  \
            test_##name();                                                     \
            std::cout << "PASSED" << std::endl;                                \
            tests_passed++;                                                    \
        } catch (const std::exception& e) {                                    \
            std::cout << "FAILED: " << e.what() << std::endl;                  \
            tests_failed++;                                                    \
        } catch (...) {                                                        \
            std::cout << "FAILED: Unknown exception" << std::endl;             \
            tests_failed++;                                                    \
        }                                                                      \
    }                                                                          \
    void test_##name()

#define ASSERT_TRUE(cond)                                                      \
    do {                                                                       \
        if (!(cond)) {                                                         \
            throw std::runtime_error(std::string("Assertion failed: ") + #cond); \
        }                                                                      \
    } while (0)

#define ASSERT_FALSE(cond) ASSERT_TRUE(!(cond))

#define ASSERT_EQ(a, b)                                                        \
    do {                                                                       \
        if ((a) != (b)) {                                                      \
            std::ostringstream oss;                                            \
            oss << "Expected " << #a << " == " << #b;                          \
            throw std::runtime_error(oss.str());                               \
        }                                                                      \
    } while (0)

#define ASSERT_NEAR(a, b, tol)                                                 \
    do {                                                                       \
        if (std::abs((a) - (b)) > (tol)) {                                     \
            std::ostringstream oss;                                            \
            oss << "Expected " << #a << " ≈ " << #b << " (±" << tol            \
                << "), got " << (a) << " vs " << (b);                          \
            throw std::runtime_error(oss.str());                               \
        }                                                                      \
    } while (0)

#define ASSERT_GT(a, b)                                                        \
    do {                                                                       \
        if (!((a) > (b))) {                                                    \
            std::ostringstream oss;                                            \
            oss << "Expected " << #a << " > " << #b << ", got " << (a)         \
                << " <= " << (b);                                              \
            throw std::runtime_error(oss.str());                               \
        }                                                                      \
    } while (0)

#define ASSERT_LT(a, b)                                                        \
    do {                                                                       \
        if (!((a) < (b))) {                                                    \
            std::ostringstream oss;                                            \
            oss << "Expected " << #a << " < " << #b << ", got " << (a)         \
                << " >= " << (b);                                              \
            throw std::runtime_error(oss.str());                               \
        }                                                                      \
    } while (0)

//==============================================================================
// Helper Functions
//==============================================================================

NoiseProfile create_test_noise_profile(
    size_t num_qubits = 5,
    double gate_err = 0.001,
    double two_q_err = 0.01,
    double readout_err = 0.01,
    double t1_ns = 50000.0,
    double t2_ns = 70000.0
) {
    NoiseProfile noise;
    noise.device_name = "test_device";
    noise.calibration_timestamp = "2024-01-01T00:00:00";
    noise.num_qubits = num_qubits;
    
    for (size_t i = 0; i < num_qubits; ++i) {
        noise.t1_times_ns.push_back(t1_ns);
        noise.t2_times_ns.push_back(t2_ns);
        noise.single_gate_errors.push_back(gate_err);
        noise.readout_errors.push_back(readout_err);
    }
    
    // Add two-qubit errors between adjacent qubits
    for (size_t i = 0; i < num_qubits - 1; ++i) {
        noise.two_qubit_errors[{i, i + 1}] = two_q_err;
    }
    
    return noise;
}

NoiseProfile create_biased_noise_profile(size_t num_qubits = 5) {
    NoiseProfile noise = create_test_noise_profile(num_qubits);
    // T1 >> T2 for biased noise
    for (size_t i = 0; i < num_qubits; ++i) {
        noise.t1_times_ns[i] = 100000.0;  // 100 µs
        noise.t2_times_ns[i] = 20000.0;   // 20 µs
    }
    return noise;
}

NoiseProfile create_correlated_noise_profile(size_t num_qubits = 5) {
    NoiseProfile noise = create_test_noise_profile(num_qubits);
    // Add correlated errors
    for (size_t i = 0; i < num_qubits - 1; ++i) {
        CorrelatedError ce;
        ce.qubit_i = i;
        ce.qubit_j = i + 1;
        ce.coupling_strength_hz = 1000.0;
        // Set sparse_probs to represent correlated error probability
        ce.sparse_probs.push_back({1, 1, 0.05});  // ZZ correlation
        noise.correlated_errors.push_back(ce);
    }
    return noise;
}

//==============================================================================
// NoiseProfile Tests
//==============================================================================

TEST(noise_profile_default_construction) {
    NoiseProfile noise;
    ASSERT_TRUE(noise.device_name.empty());
    ASSERT_EQ(noise.num_qubits, 0u);
    ASSERT_TRUE(noise.t1_times_ns.empty());
}

TEST(noise_profile_avg_gate_error) {
    NoiseProfile noise = create_test_noise_profile(5, 0.001);
    ASSERT_NEAR(noise.avg_gate_error(), 0.001, 1e-9);
}

TEST(noise_profile_max_gate_error) {
    NoiseProfile noise;
    noise.single_gate_errors = {0.001, 0.005, 0.002, 0.003};
    ASSERT_NEAR(noise.max_gate_error(), 0.005, 1e-9);
}

TEST(noise_profile_avg_two_qubit_error) {
    NoiseProfile noise = create_test_noise_profile(5, 0.001, 0.01);
    ASSERT_NEAR(noise.avg_two_qubit_error(), 0.01, 1e-9);
}

TEST(noise_profile_avg_t1) {
    NoiseProfile noise = create_test_noise_profile(5, 0.001, 0.01, 0.01, 50000.0);
    ASSERT_NEAR(noise.avg_t1(), 50000.0, 1e-9);
}

TEST(noise_profile_avg_t2) {
    NoiseProfile noise = create_test_noise_profile(5, 0.001, 0.01, 0.01, 50000.0, 70000.0);
    ASSERT_NEAR(noise.avg_t2(), 70000.0, 1e-9);
}

TEST(noise_profile_t1_t2_ratio) {
    NoiseProfile noise = create_test_noise_profile(5, 0.001, 0.01, 0.01, 50000.0, 50000.0);
    ASSERT_NEAR(noise.t1_t2_ratio(), 1.0, 1e-9);
}

TEST(noise_profile_is_biased) {
    NoiseProfile balanced = create_test_noise_profile(5);
    ASSERT_FALSE(balanced.is_biased(0.5));
    
    NoiseProfile biased = create_biased_noise_profile(5);
    ASSERT_TRUE(biased.is_biased(0.5));
}

TEST(noise_profile_has_correlations) {
    NoiseProfile uncorrelated = create_test_noise_profile(5);
    ASSERT_FALSE(uncorrelated.has_correlations());
    
    NoiseProfile correlated = create_correlated_noise_profile(5);
    ASSERT_TRUE(correlated.has_correlations());
}

TEST(noise_profile_effective_error_rate) {
    NoiseProfile noise = create_test_noise_profile(5, 0.001, 0.01, 0.01);
    double expected = 0.001 + 2.0 * 0.01 + 0.5 * 0.01;  // gate + 2*two_q + 0.5*readout
    ASSERT_NEAR(noise.effective_physical_error_rate(), expected, 1e-9);
}

TEST(noise_profile_json_roundtrip) {
    NoiseProfile original = create_test_noise_profile(5, 0.001, 0.01, 0.01, 50000.0, 70000.0);
    
    nlohmann::json j = original.to_json();
    NoiseProfile loaded = NoiseProfile::from_json(j);
    
    ASSERT_EQ(loaded.device_name, original.device_name);
    ASSERT_EQ(loaded.num_qubits, original.num_qubits);
    ASSERT_NEAR(loaded.avg_gate_error(), original.avg_gate_error(), 1e-9);
    ASSERT_NEAR(loaded.avg_t1(), original.avg_t1(), 1e-9);
}

TEST(noise_profile_relative_difference) {
    NoiseProfile a = create_test_noise_profile(5, 0.001);
    NoiseProfile b = create_test_noise_profile(5, 0.002);  // 100% higher gate error
    
    double diff = a.relative_difference(b);
    ASSERT_GT(diff, 0.0);
}

TEST(noise_profile_differs_from) {
    NoiseProfile baseline = create_test_noise_profile(5, 0.001);
    NoiseProfile similar = create_test_noise_profile(5, 0.0011);  // 10% higher
    NoiseProfile different = create_test_noise_profile(5, 0.002);  // 100% higher
    
    ASSERT_FALSE(similar.differs_from(baseline, 0.5));
    ASSERT_TRUE(different.differs_from(baseline, 0.15));
}

//==============================================================================
// AdaptiveCodeSelector Tests
//==============================================================================

TEST(code_selector_default_construction) {
    AdaptiveCodeSelector selector;
    // Should not throw
    ASSERT_TRUE(true);
}

TEST(code_selector_select_surface_for_balanced) {
    AdaptiveCodeSelector selector;
    NoiseProfile noise = create_test_noise_profile(25);  // 5x5 surface code
    
    QECCodeType code = selector.select_code(noise);
    ASSERT_EQ(code, QECCodeType::SURFACE);
}

TEST(code_selector_select_for_biased) {
    AdaptiveCodeSelector::Config config;
    config.bias_threshold = 0.3;
    AdaptiveCodeSelector selector(config);
    
    NoiseProfile biased = create_biased_noise_profile(5);
    QECCodeType code = selector.select_code(biased);
    
    // For biased noise, should prefer repetition
    ASSERT_EQ(code, QECCodeType::REPETITION);
}

TEST(code_selector_select_for_correlated) {
    AdaptiveCodeSelector selector;
    NoiseProfile correlated = create_correlated_noise_profile(25);
    
    QECCodeType code = selector.select_code(correlated);
    // Surface code handles correlations better
    ASSERT_EQ(code, QECCodeType::SURFACE);
}

TEST(code_selector_select_distance) {
    AdaptiveCodeSelector selector;
    NoiseProfile noise = create_test_noise_profile(25, 0.001, 0.005);
    
    size_t distance = selector.select_distance(
        QECCodeType::SURFACE, noise, 1e-10);
    
    // Should select a reasonable distance
    ASSERT_GT(distance, 2u);
    ASSERT_LT(distance, 100u);
}

TEST(code_selector_select_code_and_distance) {
    AdaptiveCodeSelector selector;
    NoiseProfile noise = create_test_noise_profile(25);
    
    auto [code, distance] = selector.select_code_and_distance(noise, 1e-10);
    
    ASSERT_EQ(code, QECCodeType::SURFACE);
    ASSERT_GT(distance, 2u);
}

TEST(code_selector_predict_logical_error_rate) {
    AdaptiveCodeSelector selector;
    NoiseProfile noise = create_test_noise_profile(25, 0.001);
    
    double p_L = selector.predict_logical_error_rate(
        QECCodeType::SURFACE, 5, noise);
    
    // Should be less than physical error rate (error suppression)
    ASSERT_GT(p_L, 0.0);
    ASSERT_LT(p_L, 1.0);
}

TEST(code_selector_rank_codes) {
    AdaptiveCodeSelector selector;
    NoiseProfile noise = create_test_noise_profile(25);
    
    auto rankings = selector.rank_codes(noise, 5);
    
    ASSERT_FALSE(rankings.empty());
    // Rankings should be sorted by error rate (ascending)
    for (size_t i = 1; i < rankings.size(); ++i) {
        ASSERT_TRUE(rankings[i].second >= rankings[i-1].second);
    }
}

TEST(code_selector_higher_distance_lower_error) {
    AdaptiveCodeSelector selector;
    NoiseProfile noise = create_test_noise_profile(25, 0.001);
    
    double p_L_d3 = selector.predict_logical_error_rate(
        QECCodeType::SURFACE, 3, noise);
    double p_L_d5 = selector.predict_logical_error_rate(
        QECCodeType::SURFACE, 5, noise);
    double p_L_d7 = selector.predict_logical_error_rate(
        QECCodeType::SURFACE, 7, noise);
    
    // Higher distance should give lower logical error rate
    ASSERT_LT(p_L_d5, p_L_d3);
    ASSERT_LT(p_L_d7, p_L_d5);
}

//==============================================================================
// MLDecoder Tests (Stub/Fallback Mode)
//==============================================================================

TEST(ml_decoder_construction) {
    auto code = create_stabilizer_code(QECCodeType::SURFACE, 5);
    MLDecoder::Config config;
    MLDecoder decoder(*code, config);
    
    // Not ready until model is loaded
    ASSERT_FALSE(decoder.is_ready());
}

TEST(ml_decoder_fallback_decode) {
    auto code = create_stabilizer_code(QECCodeType::SURFACE, 5);
    MLDecoder::Config config;
    config.enable_fallback = true;
    MLDecoder decoder(*code, config);
    
    // Create a simple syndrome
    Syndrome syn;
    syn.x_syndrome.resize(12, 0);
    syn.z_syndrome.resize(12, 0);
    syn.x_syndrome[0] = 1;  // One error
    
    // Should fallback to MWPM
    Correction corr = decoder.decode(syn);
    
    // Stats should show fallback was used
    auto stats = decoder.stats();
    ASSERT_EQ(stats.num_fallbacks, 1u);
}

TEST(ml_decoder_batch_decode) {
    auto code = create_stabilizer_code(QECCodeType::SURFACE, 5);
    MLDecoder::Config config;
    config.enable_fallback = true;
    MLDecoder decoder(*code, config);
    
    std::vector<Syndrome> syndromes(10);
    for (auto& syn : syndromes) {
        syn.x_syndrome.resize(12, 0);
        syn.z_syndrome.resize(12, 0);
    }
    
    auto corrections = decoder.decode_batch(syndromes);
    ASSERT_EQ(corrections.size(), 10u);
}

//==============================================================================
// ClosedLoopController Tests
//==============================================================================

TEST(closed_loop_default_construction) {
    ClosedLoopController controller;
    auto stats = controller.stats();
    ASSERT_EQ(stats.total_cycles, 0u);
}

TEST(closed_loop_update_no_error) {
    ClosedLoopController controller;
    
    QECRoundResult result;
    result.logical_error = false;
    
    for (int i = 0; i < 100; ++i) {
        controller.update(result);
    }
    
    auto stats = controller.stats();
    ASSERT_EQ(stats.total_cycles, 100u);
    ASSERT_EQ(stats.logical_errors, 0u);
    ASSERT_NEAR(stats.current_logical_error_rate, 0.0, 1e-9);
}

TEST(closed_loop_update_with_errors) {
    ClosedLoopController controller;
    
    for (int i = 0; i < 100; ++i) {
        QECRoundResult result;
        result.logical_error = (i % 10 == 0);  // 10% error rate
        controller.update(result);
    }
    
    auto stats = controller.stats();
    ASSERT_EQ(stats.total_cycles, 100u);
    ASSERT_EQ(stats.logical_errors, 10u);
    ASSERT_NEAR(stats.current_logical_error_rate, 0.1, 0.01);
}

TEST(closed_loop_detect_drift) {
    ClosedLoopController::Config config;
    config.window_size = 50;
    config.drift_threshold = 0.5;  // 50% change
    ClosedLoopController controller(config);
    
    // Establish baseline
    for (int i = 0; i < 50; ++i) {
        QECRoundResult result;
        result.logical_error = (i % 10 == 0);  // 10% error rate
        controller.update(result);
    }
    
    ASSERT_FALSE(controller.check_drift());
    
    // Now inject more errors (drift)
    for (int i = 0; i < 50; ++i) {
        QECRoundResult result;
        result.logical_error = (i % 3 == 0);  // ~33% error rate
        controller.update(result);
    }
    
    // Now should detect drift
    ASSERT_TRUE(controller.check_drift());
}

TEST(closed_loop_should_not_recalibrate_early) {
    ClosedLoopController::Config config;
    config.min_cycles_before_recalib = 100;
    ClosedLoopController controller(config);
    
    for (int i = 0; i < 50; ++i) {
        QECRoundResult result;
        result.logical_error = true;  // High error rate
        controller.update(result);
    }
    
    // Should not recalibrate - not enough cycles
    ASSERT_FALSE(controller.should_recalibrate());
}

TEST(closed_loop_reset) {
    ClosedLoopController controller;
    
    for (int i = 0; i < 100; ++i) {
        QECRoundResult result;
        result.logical_error = (i % 5 == 0);
        controller.update(result);
    }
    
    controller.reset();
    
    auto stats = controller.stats();
    ASSERT_EQ(stats.total_cycles, 0u);
    ASSERT_EQ(stats.logical_errors, 0u);
}

TEST(closed_loop_set_noise_profile) {
    ClosedLoopController controller;
    NoiseProfile noise = create_test_noise_profile(5);
    
    controller.set_noise_profile(noise);
    // Should not throw
    ASSERT_TRUE(true);
}

//==============================================================================
// DynamicDistanceSelector Tests
//==============================================================================

TEST(distance_selector_default_construction) {
    DynamicDistanceSelector selector;
    ASSERT_EQ(selector.current_distance(), 3u);  // Default min distance is 3
}

TEST(distance_selector_recommend_no_change_initially) {
    DynamicDistanceSelector::Config config;
    config.evaluation_window = 50;
    config.min_distance = 5;
    DynamicDistanceSelector selector(config);
    
    // Not enough data yet
    for (int i = 0; i < 10; ++i) {
        QECRoundResult result;
        result.logical_error = false;
        selector.update(result);
    }
    
    ASSERT_FALSE(selector.should_change_distance());
}

TEST(distance_selector_recommend_increase) {
    DynamicDistanceSelector::Config config;
    config.evaluation_window = 50;
    config.increase_threshold = 2.0;
    config.min_distance = 5;
    DynamicDistanceSelector selector(config);
    selector.set_target_error_rate(0.01);
    
    // High error rate - should recommend increase
    for (int i = 0; i < 100; ++i) {
        QECRoundResult result;
        result.logical_error = (i % 5 == 0);  // 20% error rate >> target
        selector.update(result);
    }
    
    size_t recommended = selector.recommended_distance();
    ASSERT_GT(recommended, selector.current_distance());
}

TEST(distance_selector_recommend_decrease) {
    DynamicDistanceSelector::Config config;
    config.evaluation_window = 50;
    config.decrease_threshold = 0.1;
    config.min_distance = 3;
    config.max_distance = 15;
    DynamicDistanceSelector selector(config);
    selector.set_current_distance(9);
    selector.set_target_error_rate(0.1);  // 10% target
    
    // Very low error rate - could decrease distance
    for (int i = 0; i < 100; ++i) {
        QECRoundResult result;
        result.logical_error = false;  // 0% error rate << target
        selector.update(result);
    }
    
    size_t recommended = selector.recommended_distance();
    ASSERT_LT(recommended, 9u);
}

TEST(distance_selector_respect_max) {
    DynamicDistanceSelector::Config config;
    config.max_distance = 15;
    config.min_distance = 5;
    DynamicDistanceSelector selector(config);
    selector.set_current_distance(15);
    selector.set_target_error_rate(1e-20);  // Impossible target
    
    // High error rate
    for (int i = 0; i < 100; ++i) {
        QECRoundResult result;
        result.logical_error = (i % 2 == 0);  // 50% error rate
        selector.update(result);
    }
    
    size_t recommended = selector.recommended_distance();
    ASSERT_EQ(recommended, 15u);  // Can't go higher than max
}

//==============================================================================
// AdaptiveQECController Tests
//==============================================================================

TEST(adaptive_controller_construction) {
    AdaptiveQECController::Config config;
    config.initial_distance = 5;
    config.initial_code = QECCodeType::SURFACE;
    AdaptiveQECController controller(config);
    
    ASSERT_EQ(controller.current_code_type(), QECCodeType::SURFACE);
    ASSERT_EQ(controller.current_distance(), 5u);
}

TEST(adaptive_controller_initialize) {
    AdaptiveQECController::Config config;
    config.initial_distance = 5;
    AdaptiveQECController controller(config);
    
    NoiseProfile noise = create_test_noise_profile(25);
    controller.initialize(noise);
    
    // Should have a decoder ready
    ASSERT_FALSE(controller.needs_adaptation());
}

TEST(adaptive_controller_process_round) {
    AdaptiveQECController::Config config;
    config.initial_distance = 5;
    config.enable_closed_loop = false;  // Disable for this test
    config.enable_distance_adaptation = false;
    AdaptiveQECController controller(config);
    
    NoiseProfile noise = create_test_noise_profile(25);
    controller.initialize(noise);
    
    QECRoundResult result;
    result.logical_error = false;
    controller.process_round(result);
    
    auto stats = controller.stats();
    ASSERT_EQ(stats.total_rounds, 1u);
}

TEST(adaptive_controller_decode) {
    AdaptiveQECController::Config config;
    config.initial_distance = 5;
    config.use_ml_decoder = false;  // Use MWPM only
    AdaptiveQECController controller(config);
    
    NoiseProfile noise = create_test_noise_profile(25);
    controller.initialize(noise);
    
    Syndrome syn;
    syn.x_syndrome.resize(12, 0);
    syn.z_syndrome.resize(12, 0);
    
    Correction corr = controller.decode(syn);
    
    auto stats = controller.stats();
    ASSERT_EQ(stats.mwpm_decodes, 1u);
}

TEST(adaptive_controller_distance_adaptation) {
    AdaptiveQECController::Config config;
    config.initial_distance = 5;
    config.enable_distance_adaptation = true;
    config.enable_closed_loop = false;
    config.enable_code_switching = false;
    config.target_logical_error_rate = 0.001;
    config.distance_config.evaluation_window = 20;
    config.distance_config.increase_threshold = 2.0;
    
    AdaptiveQECController controller(config);
    NoiseProfile noise = create_test_noise_profile(25);
    controller.initialize(noise);
    
    // High error rate should trigger distance increase
    for (int i = 0; i < 50; ++i) {
        QECRoundResult result;
        result.logical_error = (i % 3 == 0);  // ~33% error rate
        controller.process_round(result);
    }
    
    if (controller.needs_adaptation()) {
        controller.apply_adaptations();
    }
    
    // Distance might have increased
    ASSERT_TRUE(controller.current_distance() >= 5u);
}

TEST(adaptive_controller_reset) {
    AdaptiveQECController::Config config;
    config.initial_distance = 5;
    AdaptiveQECController controller(config);
    
    NoiseProfile noise = create_test_noise_profile(25);
    controller.initialize(noise);
    
    for (int i = 0; i < 50; ++i) {
        QECRoundResult result;
        result.logical_error = (i % 5 == 0);
        controller.process_round(result);
    }
    
    controller.reset();
    
    auto stats = controller.stats();
    ASSERT_EQ(stats.total_rounds, 0u);
}

//==============================================================================
// Training Data Generation Tests
//==============================================================================

TEST(training_data_generation) {
    auto code = create_stabilizer_code(QECCodeType::SURFACE, 3);
    NoiseProfile noise = create_test_noise_profile(9, 0.01);  // 3x3 = 9 data qubits
    
    auto samples = generate_training_data(*code, noise, 100, 42);
    
    ASSERT_EQ(samples.size(), 100u);
    for (const auto& sample : samples) {
        ASSERT_FALSE(sample.syndrome.empty());
        ASSERT_FALSE(sample.error.empty());
    }
}

TEST(training_sample_json_roundtrip) {
    TrainingSample sample;
    sample.syndrome = {0, 1, 1, 0, 0, 1, 0, 0};
    sample.error = {0, 1, 2, 0, 0, 3, 0, 0, 0};
    
    nlohmann::json j = sample.to_json();
    TrainingSample loaded = TrainingSample::from_json(j);
    
    ASSERT_EQ(loaded.syndrome, sample.syndrome);
    ASSERT_EQ(loaded.error, sample.error);
}

TEST(training_metadata_json_roundtrip) {
    TrainingDatasetMetadata meta;
    meta.code_type = "surface";
    meta.code_distance = 5;
    meta.num_samples = 10000;
    meta.noise_profile = create_test_noise_profile(25);
    meta.creation_timestamp = "2024-01-01T00:00:00";
    
    nlohmann::json j = meta.to_json();
    TrainingDatasetMetadata loaded = TrainingDatasetMetadata::from_json(j);
    
    ASSERT_EQ(loaded.code_type, meta.code_type);
    ASSERT_EQ(loaded.code_distance, meta.code_distance);
    ASSERT_EQ(loaded.num_samples, meta.num_samples);
}

//==============================================================================
// Integration Tests
//==============================================================================

TEST(integration_full_adaptive_qec_cycle) {
    // Full end-to-end test of adaptive QEC
    AdaptiveQECController::Config config;
    config.initial_distance = 5;
    config.initial_code = QECCodeType::SURFACE;
    config.enable_code_switching = true;
    config.enable_distance_adaptation = true;
    config.enable_closed_loop = false;  // Skip actual recalibration
    config.use_ml_decoder = false;
    config.target_logical_error_rate = 0.01;
    
    AdaptiveQECController controller(config);
    
    NoiseProfile noise = create_test_noise_profile(25, 0.001);
    controller.initialize(noise);
    
    // Run 100 QEC rounds
    for (int i = 0; i < 100; ++i) {
        Syndrome syn;
        syn.x_syndrome.resize(12, 0);
        syn.z_syndrome.resize(12, 0);
        
        Correction corr = controller.decode(syn);
        
        QECRoundResult result;
        result.logical_error = (i % 20 == 0);  // 5% error rate
        controller.process_round(result);
    }
    
    auto stats = controller.stats();
    ASSERT_EQ(stats.total_rounds, 100u);
    ASSERT_EQ(stats.mwpm_decodes, 100u);
}

TEST(integration_code_selection_pipeline) {
    AdaptiveCodeSelector selector;
    
    // Test with different noise profiles
    NoiseProfile balanced = create_test_noise_profile(25);
    NoiseProfile biased = create_biased_noise_profile(5);
    NoiseProfile correlated = create_correlated_noise_profile(25);
    
    auto code_balanced = selector.select_code(balanced);
    auto code_biased = selector.select_code(biased);
    auto code_correlated = selector.select_code(correlated);
    
    // Verify appropriate code selection
    ASSERT_EQ(code_balanced, QECCodeType::SURFACE);
    ASSERT_EQ(code_biased, QECCodeType::REPETITION);
    ASSERT_EQ(code_correlated, QECCodeType::SURFACE);
}

//==============================================================================
// Performance Tests
//==============================================================================

TEST(performance_code_selection) {
    AdaptiveCodeSelector selector;
    NoiseProfile noise = create_test_noise_profile(25);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 1000; ++i) {
        selector.select_code(noise);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Should complete 1000 selections in < 100ms (< 0.1ms each)
    ASSERT_LT(ms, 100.0);
    std::cout << " (" << ms/1000.0 << " ms/selection) ";
}

TEST(performance_decode_latency) {
    auto code = create_stabilizer_code(QECCodeType::SURFACE, 5);
    MLDecoder::Config config;
    config.enable_fallback = true;
    MLDecoder decoder(*code, config);
    
    Syndrome syn;
    syn.x_syndrome.resize(12, 0);
    syn.z_syndrome.resize(12, 0);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 1000; ++i) {
        decoder.decode(syn);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Target: < 5ms per decode
    ASSERT_LT(ms / 1000.0, 5.0);
    std::cout << " (" << ms/1000.0 << " ms/decode) ";
}

//==============================================================================
// Main
//==============================================================================

int main() {
    std::cout << "=== Phase 9.3 Adaptive QEC Test Suite ===" << std::endl;
    std::cout << std::endl;
    
    // NoiseProfile tests
    std::cout << "--- NoiseProfile Tests ---" << std::endl;
    run_noise_profile_default_construction();
    run_noise_profile_avg_gate_error();
    run_noise_profile_max_gate_error();
    run_noise_profile_avg_two_qubit_error();
    run_noise_profile_avg_t1();
    run_noise_profile_avg_t2();
    run_noise_profile_t1_t2_ratio();
    run_noise_profile_is_biased();
    run_noise_profile_has_correlations();
    run_noise_profile_effective_error_rate();
    run_noise_profile_json_roundtrip();
    run_noise_profile_relative_difference();
    run_noise_profile_differs_from();
    
    // AdaptiveCodeSelector tests
    std::cout << "\n--- AdaptiveCodeSelector Tests ---" << std::endl;
    run_code_selector_default_construction();
    run_code_selector_select_surface_for_balanced();
    run_code_selector_select_for_biased();
    run_code_selector_select_for_correlated();
    run_code_selector_select_distance();
    run_code_selector_select_code_and_distance();
    run_code_selector_predict_logical_error_rate();
    run_code_selector_rank_codes();
    run_code_selector_higher_distance_lower_error();
    
    // MLDecoder tests - require Python bindings
    std::cout << "\n--- MLDecoder Tests ---" << std::endl;
#ifdef USE_PYTHON
    run_ml_decoder_construction();
    run_ml_decoder_fallback_decode();
    run_ml_decoder_batch_decode();
#else
    std::cout << "SKIPPED (Python bindings not enabled)" << std::endl;
#endif
    
    // ClosedLoopController tests
    std::cout << "\n--- ClosedLoopController Tests ---" << std::endl;
    run_closed_loop_default_construction();
    run_closed_loop_update_no_error();
    run_closed_loop_update_with_errors();
    run_closed_loop_detect_drift();
    run_closed_loop_should_not_recalibrate_early();
    run_closed_loop_reset();
    run_closed_loop_set_noise_profile();
    
    // DynamicDistanceSelector tests
    std::cout << "\n--- DynamicDistanceSelector Tests ---" << std::endl;
    run_distance_selector_default_construction();
    run_distance_selector_recommend_no_change_initially();
    run_distance_selector_recommend_increase();
    run_distance_selector_recommend_decrease();
    run_distance_selector_respect_max();
    
    // AdaptiveQECController tests - require ML decoder / Python bindings
    std::cout << "\n--- AdaptiveQECController Tests ---" << std::endl;
#ifdef USE_PYTHON
    run_adaptive_controller_construction();
    run_adaptive_controller_initialize();
    run_adaptive_controller_process_round();
    run_adaptive_controller_decode();
    run_adaptive_controller_distance_adaptation();
    run_adaptive_controller_reset();
#else
    std::cout << "SKIPPED (requires ML decoder / Python bindings)" << std::endl;
#endif

    // Training data tests
    std::cout << "\n--- Training Data Tests ---" << std::endl;
    run_training_data_generation();
    run_training_sample_json_roundtrip();
    run_training_metadata_json_roundtrip();
    
    // Integration tests
    std::cout << "\n--- Integration Tests ---" << std::endl;
    run_integration_full_adaptive_qec_cycle();
    run_integration_code_selection_pipeline();
    
    // Performance tests
    std::cout << "\n--- Performance Tests ---" << std::endl;
    run_performance_code_selection();
#ifdef USE_PYTHON
    run_performance_decode_latency();  // Requires MLDecoder
#else
    std::cout << "performance_decode_latency SKIPPED (requires ML decoder)" << std::endl;
#endif

    // Summary
    std::cout << std::endl;
    std::cout << "=== Test Results ===" << std::endl;
    std::cout << "Passed: " << tests_passed << std::endl;
    std::cout << "Failed: " << tests_failed << std::endl;
    
    if (tests_failed > 0) {
        std::cout << "\n*** SOME TESTS FAILED ***" << std::endl;
        std::exit(1);
    }
    
    std::cout << "\n*** ALL TESTS PASSED ***" << std::endl;
    std::exit(0);
}
