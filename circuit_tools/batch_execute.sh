#!/bin/bash
# Batch execute multiple quantum circuits
# Usage: ./batch_execute.sh <circuit_directory> [output_directory]

set -e

CIRCUIT_DIR="${1:-.}"
OUTPUT_DIR="${2:-./results}"
SIMULATOR="../build/quantum_sim"

# Check if simulator exists
if [ ! -f "$SIMULATOR" ]; then
    echo "❌ Error: quantum_sim binary not found at $SIMULATOR"
    echo "   Please build the project first: cd build && make quantum_sim"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Find all JSON circuit files
CIRCUITS=$(find "$CIRCUIT_DIR" -name "*.json" -type f)
TOTAL=$(echo "$CIRCUITS" | wc -l | tr -d ' ')

if [ "$TOTAL" -eq 0 ]; then
    echo "❌ No JSON circuit files found in $CIRCUIT_DIR"
    exit 1
fi

echo "🚀 Batch Execution Started"
echo "   Directory: $CIRCUIT_DIR"
echo "   Circuits: $TOTAL"
echo "   Output: $OUTPUT_DIR"
echo ""

SUCCESS=0
FAILED=0

for circuit in $CIRCUITS; do
    filename=$(basename "$circuit" .json)
    output_file="$OUTPUT_DIR/${filename}_output.txt"
    
    echo -n "⏳ Executing $filename... "
    
    if "$SIMULATOR" "$circuit" > "$output_file" 2>&1; then
        echo "✅ Success"
        ((SUCCESS++))
    else
        echo "❌ Failed"
        ((FAILED++))
        echo "   Error log: $output_file"
    fi
done

echo ""
echo "📊 Batch Execution Complete"
echo "   ✅ Success: $SUCCESS / $TOTAL"
echo "   ❌ Failed: $FAILED / $TOTAL"
echo ""

# Generate summary
SUMMARY_FILE="$OUTPUT_DIR/batch_summary.txt"
{
    echo "Batch Execution Summary"
    echo "======================="
    echo "Date: $(date)"
    echo "Directory: $CIRCUIT_DIR"
    echo "Total circuits: $TOTAL"
    echo "Success: $SUCCESS"
    echo "Failed: $FAILED"
    echo ""
    echo "Results:"
    for circuit in $CIRCUITS; do
        filename=$(basename "$circuit" .json)
        output_file="$OUTPUT_DIR/${filename}_output.txt"
        if grep -q "Simulation complete" "$output_file" 2>/dev/null; then
            echo "  ✅ $filename"
        else
            echo "  ❌ $filename"
        fi
    done
} > "$SUMMARY_FILE"

echo "📝 Summary saved to: $SUMMARY_FILE"

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
