#!/bin/bash
# Comprehensive Test Runner for LRET
# Can be executed by OpenCode via natural language commands

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LRET_ROOT="$SCRIPT_DIR"
BUILD_DIR="$LRET_ROOT/build"
PYTHON_DIR="$LRET_ROOT/python"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TEST_LOG="$LRET_ROOT/test_results_$TIMESTAMP.log"
SUMMARY_FILE="$LRET_ROOT/test_summary_$TIMESTAMP.txt"

echo "=========================================="
echo "LRET Comprehensive Test Suite"
echo "Started: $(date)"
echo "=========================================="
echo ""

# Initialize counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# Function to run a test and record result
run_test() {
    local test_name="$1"
    local test_command="$2"
    local test_type="$3"  # C++ or Python
    
    echo "───────────────────────────────────────"
    echo "Running: $test_name"
    echo "Type: $test_type"
    echo "Command: $test_command"
    echo ""
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    # Create temp file for test output
    local temp_output=$(mktemp)
    
    if eval "$test_command" > "$temp_output" 2>&1; then
        echo "✅ PASS: $test_name"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        echo "[PASS] $test_name" >> "$TEST_LOG"
    else
        local exit_code=$?
        echo "❌ FAIL: $test_name (exit code: $exit_code)"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        echo "[FAIL] $test_name (exit code: $exit_code)" >> "$TEST_LOG"
        echo "Error output:" >> "$TEST_LOG"
        cat "$temp_output" >> "$TEST_LOG"
        echo "───────────────────────────────────────" >> "$TEST_LOG"
    fi
    
    rm -f "$temp_output"
    echo ""
}

# Initialize log
echo "LRET Test Execution Log - $TIMESTAMP" > "$TEST_LOG"
echo "========================================" >> "$TEST_LOG"
echo "" >> "$TEST_LOG"

cd "$LRET_ROOT"

# ============================================
# Phase 1: Build the Project
# ============================================
echo "Phase 1: Building Project"
echo "=========================================="

if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"
echo "Running cmake..."
if cmake .. >> "$TEST_LOG" 2>&1; then
    echo "✅ CMake configuration successful"
else
    echo "❌ CMake failed. See $TEST_LOG for details."
    exit 1
fi

echo "Building with make..."
if make -j$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4) >> "$TEST_LOG" 2>&1; then
    echo "✅ Build successful"
else
    echo "❌ Build failed. See $TEST_LOG for details."
    exit 1
fi

echo ""

# ============================================
# Phase 2: C++ Tests
# ============================================
echo "Phase 2: C++ Test Binaries"
echo "=========================================="

# Find all test binaries
TEST_BINARIES=$(find "$BUILD_DIR" -name "test_*" -type f -executable 2>/dev/null | sort)

if [ -z "$TEST_BINARIES" ]; then
    echo "⚠️  No C++ test binaries found"
else
    echo "Found $(echo "$TEST_BINARIES" | wc -l) test binaries"
    echo ""
    
    for test_bin in $TEST_BINARIES; do
        test_name=$(basename "$test_bin")
        run_test "$test_name" "$test_bin" "C++"
    done
fi

echo ""

# ============================================
# Phase 3: Python Tests
# ============================================
echo "Phase 3: Python Tests"
echo "=========================================="

if [ -d "$PYTHON_DIR/tests" ]; then
    cd "$LRET_ROOT"
    
    # Check if pytest is available
    if command -v pytest &> /dev/null; then
        echo "Running pytest..."
        run_test "Python Test Suite" "python3 -m pytest python/tests/ -v --tb=short" "Python"
    elif python3 -m pytest --version &> /dev/null; then
        echo "Running pytest via python3 -m..."
        run_test "Python Test Suite" "python3 -m pytest python/tests/ -v --tb=short" "Python"
    else
        echo "⚠️  pytest not found. Skipping Python tests."
        echo "Install with: pip install pytest"
        SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
    fi
else
    echo "⚠️  No Python tests directory found"
fi

echo ""

# ============================================
# Phase 4: Integration Tests (Optional)
# ============================================
echo "Phase 4: Integration Tests"
echo "=========================================="

# Test quantum_sim with sample circuits
if [ -f "$BUILD_DIR/quantum_sim" ] && [ -d "$LRET_ROOT/samples" ]; then
    SAMPLE_CIRCUITS=$(find "$LRET_ROOT/samples" -name "*.json" -type f 2>/dev/null | head -5)
    
    if [ -n "$SAMPLE_CIRCUITS" ]; then
        echo "Testing quantum_sim with sample circuits..."
        for circuit_file in $SAMPLE_CIRCUITS; do
            circuit_name=$(basename "$circuit_file")
            run_test "quantum_sim: $circuit_name" "$BUILD_DIR/quantum_sim $circuit_file" "Integration"
        done
    else
        echo "⚠️  No sample circuits found in samples/"
        SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
    fi
else
    echo "⚠️  quantum_sim binary or samples/ not found. Skipping integration tests."
    SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
fi

echo ""

# ============================================
# Phase 5: GPU Tests (if available)
# ============================================
echo "Phase 5: GPU Tests"
echo "=========================================="

if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1
    echo ""
    
    GPU_TESTS=$(find "$BUILD_DIR" -name "*gpu*" -o -name "*distributed_gpu*" -type f -executable 2>/dev/null)
    
    if [ -n "$GPU_TESTS" ]; then
        for test_bin in $GPU_TESTS; do
            test_name=$(basename "$test_bin")
            run_test "$test_name" "$test_bin" "GPU"
        done
    else
        echo "⚠️  No GPU test binaries found"
        SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
    fi
else
    echo "⚠️  No NVIDIA GPU detected. Skipping GPU tests."
    SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
fi

echo ""

# ============================================
# Generate Summary
# ============================================
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo ""
echo "Total Tests:   $TOTAL_TESTS"
echo "Passed:        $PASSED_TESTS ($(awk "BEGIN {printf \"%.1f\", ($PASSED_TESTS/$TOTAL_TESTS)*100}")%)"
echo "Failed:        $FAILED_TESTS"
echo "Skipped:       $SKIPPED_TESTS"
echo ""
echo "Completed:     $(date)"
echo "Duration:      $SECONDS seconds"
echo ""

# Create detailed summary file
cat > "$SUMMARY_FILE" << EOF
# LRET Test Execution Summary

**Date:** $(date)
**Timestamp:** $TIMESTAMP
**Duration:** $SECONDS seconds

## Results

| Metric | Count | Percentage |
|--------|-------|------------|
| Total Tests | $TOTAL_TESTS | 100% |
| ✅ Passed | $PASSED_TESTS | $(awk "BEGIN {printf \"%.1f\", ($PASSED_TESTS/$TOTAL_TESTS)*100}")% |
| ❌ Failed | $FAILED_TESTS | $(awk "BEGIN {printf \"%.1f\", ($FAILED_TESTS/$TOTAL_TESTS)*100}")% |
| ⏭️ Skipped | $SKIPPED_TESTS | - |

## Test Phases

1. ✅ Build: CMake + Make
2. C++ Tests: $(echo "$TEST_BINARIES" | wc -l | xargs) binaries
3. Python Tests: pytest suite
4. Integration: Sample circuits
5. GPU Tests: CUDA binaries (if available)

## Detailed Log

See full execution log: \`$TEST_LOG\`

## Failed Tests

EOF

# Append failed test details
if [ $FAILED_TESTS -gt 0 ]; then
    echo "### Failed Test Details" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    grep "^\[FAIL\]" "$TEST_LOG" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    echo "See $TEST_LOG for full error output." >> "$SUMMARY_FILE"
else
    echo "🎉 **All tests passed!**" >> "$SUMMARY_FILE"
fi

echo "## Next Steps" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

if [ $FAILED_TESTS -gt 0 ]; then
    cat >> "$SUMMARY_FILE" << EOF
1. Review failed tests in $TEST_LOG
2. Debug failing tests individually
3. Re-run after fixes: ./run_all_tests.sh
4. Consider using OpenCode for automated debugging:
   \`\`\`bash
   opencode run "Analyze test failures in $TEST_LOG and suggest fixes"
   \`\`\`
EOF
else
    cat >> "$SUMMARY_FILE" << EOF
1. All tests passed! Ready for:
   - Cirq comparison: ./setup_cirq_comparison.sh
   - GPU benchmarking: ./gpu_test_automated.sh
   - Production deployment
EOF
fi

# Display summary
cat "$SUMMARY_FILE"
echo ""
echo "=========================================="
echo "Summary saved to: $SUMMARY_FILE"
echo "Full log saved to: $TEST_LOG"
echo "=========================================="

# Exit with appropriate code
if [ $FAILED_TESTS -gt 0 ]; then
    exit 1
else
    exit 0
fi
