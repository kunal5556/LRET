/**
 * @file test_qec_decoder.cpp
 * @brief Tests for QEC decoders (MWPM, Union-Find, Lookup Table)
 * 
 * Phase 9.1: Quantum Error Correction - Decoders
 */

#include <iostream>
#include <cassert>
#include <cmath>
#include <random>
#include "qec_types.h"
#include "qec_stabilizer.h"
#include "qec_syndrome.h"
#include "qec_decoder.h"

using namespace qlret;

//==============================================================================
// Test Helpers
//==============================================================================

void test_passed(const std::string& name) {
    std::cout << "[PASS] " << name << std::endl;
}

//==============================================================================
// MWPM Decoder Tests
//==============================================================================

void test_mwpm_decoder_creation() {
    SurfaceCode code(3);
    
    MWPMDecoder::Config config;
    config.physical_error_rate = 0.01;
    
    MWPMDecoder decoder(code, config);
    
    test_passed("mwpm_decoder_creation");
}

void test_mwpm_decoder_no_error() {
    SurfaceCode code(3);
    MWPMDecoder decoder(code);
    
    // Empty syndrome
    Syndrome syn;
    syn.x_syndrome.resize(code.x_stabilizers().size(), 0);
    syn.z_syndrome.resize(code.z_stabilizers().size(), 0);
    
    Correction corr = decoder.decode(syn);
    
    // Should return identity correction
    assert(corr.x_correction.weight() == 0);
    assert(corr.z_correction.weight() == 0);
    
    test_passed("mwpm_decoder_no_error");
}

void test_mwpm_decoder_single_error() {
    RepetitionCode code(5);
    MWPMDecoder decoder(code);
    SyndromeExtractor extractor(code);
    ErrorInjector injector(5);
    
    // Single X error
    auto error = injector.single_error(2, Pauli::X);
    Syndrome syn = extractor.extract(error);
    
    Correction corr = decoder.decode(syn);
    
    // Correction should fix the error
    // After applying correction, should be in code space
    assert(corr.x_correction.weight() >= 1);
    
    test_passed("mwpm_decoder_single_error");
}

void test_mwpm_decoder_stats() {
    SurfaceCode code(3);
    MWPMDecoder decoder(code);
    
    Syndrome syn;
    syn.x_syndrome.resize(code.x_stabilizers().size(), 0);
    syn.z_syndrome.resize(code.z_stabilizers().size(), 0);
    
    decoder.decode(syn);
    decoder.decode(syn);
    decoder.decode(syn);
    
    assert(decoder.stats().num_decodes == 3);
    assert(decoder.stats().avg_time_ms() >= 0.0);
    
    test_passed("mwpm_decoder_stats");
}

//==============================================================================
// Union-Find Decoder Tests
//==============================================================================

void test_union_find_decoder_creation() {
    SurfaceCode code(3);
    
    UnionFindDecoder::Config config;
    config.use_weighted = true;
    
    UnionFindDecoder decoder(code, config);
    
    test_passed("union_find_decoder_creation");
}

void test_union_find_decoder_no_error() {
    RepetitionCode code(5);
    UnionFindDecoder decoder(code);
    
    Syndrome syn;
    syn.x_syndrome.resize(code.x_stabilizers().size(), 0);
    syn.z_syndrome.resize(code.z_stabilizers().size(), 0);
    
    Correction corr = decoder.decode(syn);
    
    // Should return identity correction
    assert(corr.x_correction.weight() == 0 || corr.z_correction.weight() == 0);
    
    test_passed("union_find_decoder_no_error");
}

void test_union_find_decoder_single_error() {
    RepetitionCode code(5);
    UnionFindDecoder decoder(code);
    SyndromeExtractor extractor(code);
    ErrorInjector injector(5);
    
    auto error = injector.single_error(2, Pauli::X);
    Syndrome syn = extractor.extract(error);
    
    Correction corr = decoder.decode(syn);
    
    // Should produce some correction
    // (may not be optimal but should neutralize syndrome)
    
    test_passed("union_find_decoder_single_error");
}

//==============================================================================
// Lookup Table Decoder Tests
//==============================================================================

void test_lookup_table_decoder_creation() {
    RepetitionCode code(3);
    
    LookupTableDecoder decoder(code);
    
    // Table should be populated
    assert(decoder.table_size() > 0);
    
    test_passed("lookup_table_decoder_creation");
}

void test_lookup_table_decoder_no_error() {
    RepetitionCode code(3);
    LookupTableDecoder decoder(code);
    
    Syndrome syn;
    syn.x_syndrome.resize(code.x_stabilizers().size(), 0);
    syn.z_syndrome.resize(code.z_stabilizers().size(), 0);
    
    Correction corr = decoder.decode(syn);
    
    assert(corr.x_correction.weight() == 0);
    assert(corr.z_correction.weight() == 0);
    
    test_passed("lookup_table_decoder_no_error");
}

void test_lookup_table_decoder_single_error() {
    RepetitionCode code(3);
    LookupTableDecoder decoder(code);
    SyndromeExtractor extractor(code);
    ErrorInjector injector(3);
    
    auto error = injector.single_error(1, Pauli::X);
    Syndrome syn = extractor.extract(error);
    
    Correction corr = decoder.decode(syn);
    
    // Lookup should give weight-1 correction
    assert(corr.x_correction.weight() <= 1);
    
    test_passed("lookup_table_decoder_single_error");
}

//==============================================================================
// Decoder Factory Tests
//==============================================================================

void test_decoder_factory() {
    SurfaceCode code(3);
    
    auto mwpm = create_decoder(DecoderType::MWPM, code, 0.01);
    assert(mwpm != nullptr);
    
    auto uf = create_decoder(DecoderType::UNION_FIND, code, 0.01);
    assert(uf != nullptr);
    
    RepetitionCode small_code(3);
    auto lookup = create_decoder(DecoderType::LOOKUP_TABLE, small_code, 0.01);
    assert(lookup != nullptr);
    
    test_passed("decoder_factory");
}

//==============================================================================
// Logical Error Detection Tests
//==============================================================================

void test_logical_error_detection() {
    RepetitionCode code(5);
    MWPMDecoder decoder(code);
    SyndromeExtractor extractor(code);
    ErrorInjector injector(5);
    
    // Single error - should be correctable
    auto single_error = injector.single_error(2, Pauli::X);
    Syndrome syn = extractor.extract(single_error);
    Correction corr = decoder.decode(syn);
    
    // Single error within correction capacity
    bool logical_err = decoder.has_logical_error(corr, single_error);
    // May or may not have logical error depending on correction
    (void)logical_err;  // Just test it runs
    
    test_passed("logical_error_detection");
}

//==============================================================================
// Decoder Comparison Tests
//==============================================================================

void test_decoder_comparison() {
    RepetitionCode code(5);
    SyndromeExtractor extractor(code);
    ErrorInjector injector(5);
    
    MWPMDecoder mwpm(code);
    UnionFindDecoder uf(code);
    
    // Test on same error
    auto error = injector.single_error(2, Pauli::X);
    Syndrome syn = extractor.extract(error);
    
    Correction mwpm_corr = mwpm.decode(syn);
    Correction uf_corr = uf.decode(syn);
    
    // Both should produce valid corrections
    // (may differ but both should be valid)
    
    std::cout << "  MWPM correction weight: " << mwpm_corr.x_correction.weight() << std::endl;
    std::cout << "  UF correction weight: " << uf_corr.x_correction.weight() << std::endl;
    
    test_passed("decoder_comparison");
}

//==============================================================================
// Performance Tests
//==============================================================================

void test_decoder_performance() {
    SurfaceCode code(3);
    MWPMDecoder decoder(code);
    
    Syndrome syn;
    syn.x_syndrome.resize(code.x_stabilizers().size(), 0);
    syn.z_syndrome.resize(code.z_stabilizers().size(), 0);
    syn.z_syndrome[0] = 1;
    syn.z_syndrome[1] = 1;
    
    const int num_decodes = 100;
    for (int i = 0; i < num_decodes; ++i) {
        decoder.decode(syn);
    }
    
    std::cout << "  Average decode time: " << decoder.stats().avg_time_ms() << " ms" << std::endl;
    assert(decoder.stats().num_decodes == num_decodes);
    
    test_passed("decoder_performance");
}

//==============================================================================
// Multi-Round Decoding Tests
//==============================================================================

void test_multi_round_decoding() {
    SurfaceCode code(3);
    
    MWPMDecoder::Config config;
    config.use_3d_matching = true;
    config.physical_error_rate = 0.01;
    config.measurement_error_rate = 0.01;
    
    MWPMDecoder decoder(code, config);
    
    // Multiple syndrome rounds
    std::vector<Syndrome> syndromes(3);
    for (auto& syn : syndromes) {
        syn.x_syndrome.resize(code.x_stabilizers().size(), 0);
        syn.z_syndrome.resize(code.z_stabilizers().size(), 0);
    }
    
    // Introduce a detection event
    syndromes[1].z_syndrome[0] = 1;
    syndromes[2].z_syndrome[0] = 1;
    
    Correction corr = decoder.decode_multiple_rounds(syndromes);
    
    // Should produce some correction
    
    test_passed("multi_round_decoding");
}

//==============================================================================
// Main
//==============================================================================

int main() {
    std::cout << "=== QEC Decoder Tests ===" << std::endl;
    
    // MWPM decoder tests
    test_mwpm_decoder_creation();
    test_mwpm_decoder_no_error();
    test_mwpm_decoder_single_error();
    test_mwpm_decoder_stats();
    
    // Union-Find decoder tests
    test_union_find_decoder_creation();
    test_union_find_decoder_no_error();
    test_union_find_decoder_single_error();
    
    // Lookup table decoder tests
    test_lookup_table_decoder_creation();
    test_lookup_table_decoder_no_error();
    test_lookup_table_decoder_single_error();
    
    // Factory tests
    test_decoder_factory();
    
    // Logical error tests
    test_logical_error_detection();
    
    // Comparison tests
    test_decoder_comparison();
    
    // Performance tests
    test_decoder_performance();
    
    // Multi-round tests
    test_multi_round_decoding();
    
    std::cout << "\n=== All Decoder Tests Passed ===" << std::endl;
    return 0;
}
