#!/bin/bash
# GPU Testing and Error Collection Script for OpenCode
# Usage: ./gpu_test_automated.sh [--auto-fix]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LRET_ROOT="$SCRIPT_DIR"
BUILD_DIR="$LRET_ROOT/build"
LOG_DIR="$LRET_ROOT/gpu_test_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SESSION_LOG="$LOG_DIR/session_$TIMESTAMP.log"
ERROR_SUMMARY="$LOG_DIR/errors_$TIMESTAMP.txt"
AUTO_FIX=false

# Parse arguments
if [[ "$1" == "--auto-fix" ]]; then
    AUTO_FIX=true
    echo "Auto-fix mode enabled"
fi

# Setup
mkdir -p "$LOG_DIR"
cd "$LRET_ROOT"
export PATH=$HOME/.opencode/bin:$PATH

# Check OpenCode
if ! command -v opencode &> /dev/null; then
    echo "ERROR: OpenCode not found. Install with: curl -fsSL https://opencode.ai/install | bash"
    exit 1
fi

# Check API key for commercial LLM (optional but recommended)
if [[ -z "$ANTHROPIC_API_KEY" ]]; then
    echo "WARNING: ANTHROPIC_API_KEY not set. Using free model (limited capability)."
    MODEL_FLAG=""
else
    echo "Using Claude Sonnet 4 for advanced debugging"
    MODEL_FLAG="--model anthropic/claude-sonnet-4-20250514"
fi

echo "===================================================" | tee -a "$SESSION_LOG"
echo "GPU Testing Session: $TIMESTAMP" | tee -a "$SESSION_LOG"
echo "===================================================" | tee -a "$SESSION_LOG"

# Step 1: Build GPU targets
echo -e "\n[STEP 1] Building GPU targets..." | tee -a "$SESSION_LOG"
opencode run $MODEL_FLAG "
Build GPU-enabled targets:
1. cd $BUILD_DIR
2. cmake -DGPU_SUPPORT=ON ..
3. make -j\$(nproc) gpu_simulator distributed_gpu test_advanced_noise
4. Report build status and any warnings
" 2>&1 | tee -a "$SESSION_LOG"

BUILD_STATUS=$?
if [[ $BUILD_STATUS -ne 0 ]]; then
    echo "ERROR: Build failed. See $SESSION_LOG" | tee -a "$ERROR_SUMMARY"
    exit 1
fi

# Step 2: Identify GPU test binaries
echo -e "\n[STEP 2] Identifying GPU test binaries..." | tee -a "$SESSION_LOG"
GPU_TESTS=$(find "$BUILD_DIR" -name "test_*" -type f -executable | grep -E "(gpu|distributed)" || true)

if [[ -z "$GPU_TESTS" ]]; then
    echo "WARNING: No GPU test binaries found" | tee -a "$SESSION_LOG"
    GPU_TESTS="$BUILD_DIR/quantum_sim"
fi

echo "GPU tests to run:" | tee -a "$SESSION_LOG"
echo "$GPU_TESTS" | tee -a "$SESSION_LOG"

# Step 3: Run tests and collect errors
echo -e "\n[STEP 3] Running GPU tests..." | tee -a "$SESSION_LOG"
FAILED_TESTS=""

for test_bin in $GPU_TESTS; do
    test_name=$(basename "$test_bin")
    echo -e "\n--- Testing: $test_name ---" | tee -a "$SESSION_LOG"
    
    TEST_OUTPUT_FILE="$LOG_DIR/${test_name}_$TIMESTAMP.log"
    
    if $test_bin > "$TEST_OUTPUT_FILE" 2>&1; then
        echo "✅ PASS: $test_name" | tee -a "$SESSION_LOG"
    else
        echo "❌ FAIL: $test_name" | tee -a "$SESSION_LOG"
        FAILED_TESTS="$FAILED_TESTS $test_name"
        
        # Collect error details
        echo -e "\n=== ERROR: $test_name ===" >> "$ERROR_SUMMARY"
        echo "Output saved to: $TEST_OUTPUT_FILE" >> "$ERROR_SUMMARY"
        echo "--- Last 50 lines ---" >> "$ERROR_SUMMARY"
        tail -50 "$TEST_OUTPUT_FILE" >> "$ERROR_SUMMARY"
        echo -e "\n" >> "$ERROR_SUMMARY"
    fi
done

# Step 4: Analyze errors with OpenCode
if [[ -n "$FAILED_TESTS" ]]; then
    echo -e "\n[STEP 4] Analyzing errors with OpenCode..." | tee -a "$SESSION_LOG"
    
    opencode run $MODEL_FLAG "
Analyze GPU test failures:

1. Read the error summary file: $ERROR_SUMMARY
2. For each failed test, categorize the error:
   - CUDA runtime error (cudaMalloc, cudaMemcpy, kernel launch)
   - Compilation error (nvcc issues)
   - Logic error (incorrect results)
   - Memory error (out of bounds, leaks)
   - Hardware limitation (insufficient GPU memory, compute capability)
3. For each error:
   a) Identify the root cause
   b) Find the relevant source file(s)
   c) Suggest a fix with code snippets
4. Create a prioritized fix list in $LOG_DIR/fix_recommendations_$TIMESTAMP.md

Output format:
## Failed Test: <test_name>
**Error Type:** <category>
**Root Cause:** <explanation>
**Affected Files:** <file paths>
**Suggested Fix:**
\`\`\`cpp
<code snippet with fix>
\`\`\`
**Priority:** <High/Medium/Low>
" 2>&1 | tee -a "$SESSION_LOG"
    
    # Step 5: Auto-fix if requested
    if $AUTO_FIX; then
        echo -e "\n[STEP 5] Attempting automatic fixes..." | tee -a "$SESSION_LOG"
        
        opencode run $MODEL_FLAG "
Auto-fix GPU errors:

1. Read fix recommendations: $LOG_DIR/fix_recommendations_$TIMESTAMP.md
2. For each HIGH priority fix:
   a) Read the affected source file
   b) Apply the suggested fix using Edit tool
   c) Rebuild: cd $BUILD_DIR && make -j\$(nproc)
   d) Re-run the failed test
   e) If test passes: log success
   f) If test fails: revert change, try alternative approach
3. Save results to $LOG_DIR/auto_fix_results_$TIMESTAMP.md

Stop after 3 failed fix attempts per test.
" 2>&1 | tee -a "$SESSION_LOG"
    fi
else
    echo -e "\n✅ All GPU tests passed!" | tee -a "$SESSION_LOG"
fi

# Step 6: Generate summary report
echo -e "\n[STEP 6] Generating summary report..." | tee -a "$SESSION_LOG"

cat > "$LOG_DIR/summary_$TIMESTAMP.md" << EOF
# GPU Testing Summary - $TIMESTAMP

## Session Details
- **Date:** $(date)
- **LRET Root:** $LRET_ROOT
- **Build Dir:** $BUILD_DIR
- **Model:** ${MODEL_FLAG:-Free model}
- **Auto-fix:** $AUTO_FIX

## Test Results

$(if [[ -z "$FAILED_TESTS" ]]; then
    echo "✅ **All tests passed**"
else
    echo "❌ **Failed tests:**"
    for test in $FAILED_TESTS; do
        echo "- $test"
    done
fi)

## Logs Generated
- Session log: $SESSION_LOG
- Error summary: $ERROR_SUMMARY
- Fix recommendations: $LOG_DIR/fix_recommendations_$TIMESTAMP.md
$(if $AUTO_FIX; then
    echo "- Auto-fix results: $LOG_DIR/auto_fix_results_$TIMESTAMP.md"
fi)

## Next Steps

$(if [[ -n "$FAILED_TESTS" ]]; then
    echo "1. Review error details in $ERROR_SUMMARY"
    echo "2. Check fix recommendations in fix_recommendations_$TIMESTAMP.md"
    if ! $AUTO_FIX; then
        echo "3. Run with --auto-fix to attempt automatic corrections"
    else
        echo "3. Review auto-fix results and manually address remaining issues"
    fi
else
    echo "No issues detected. GPU code is working correctly."
fi)

## Files for Remote Debugging

If you need to debug on your personal system:
\`\`\`bash
# Copy these files to your local machine:
scp user@workstation:$ERROR_SUMMARY ./
scp user@workstation:$LOG_DIR/fix_recommendations_$TIMESTAMP.md ./
scp user@workstation:$SESSION_LOG ./
\`\`\`
EOF

echo -e "\n===================================================" | tee -a "$SESSION_LOG"
echo "Summary report: $LOG_DIR/summary_$TIMESTAMP.md" | tee -a "$SESSION_LOG"
echo "===================================================" | tee -a "$SESSION_LOG"

cat "$LOG_DIR/summary_$TIMESTAMP.md"

exit 0
