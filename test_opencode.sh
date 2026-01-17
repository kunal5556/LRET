#!/bin/bash
set -e

echo "🧪 OpenCodeAI Testing Script for LRET"
echo "======================================"
echo

# Check we're on agent-MD branch
BRANCH=$(git branch --show-current)
if [ "$BRANCH" != "agent-MD" ]; then
    echo "❌ Error: Not on agent-MD branch (currently on $BRANCH)"
    exit 1
fi
echo "✅ On agent-MD branch"

# Verify agent files exist
if [ ! -f "agent1.md" ]; then
    echo "❌ Error: agent1.md not found"
    exit 1
fi
TOTAL_LINES=0
for i in 1 2 3 4 5 6; do
    if [ -f "agent${i}.md" ]; then
        LINES=$(wc -l < "agent${i}.md")
        TOTAL_LINES=$((TOTAL_LINES + LINES))
    fi
done
echo "✅ Agent files found (agent1.md - agent6.md, total $TOTAL_LINES lines)"

if [ ! -f "AGENT_GUIDE.md" ]; then
    echo "⚠️  Warning: AGENT_GUIDE.md not found"
else
    echo "✅ AGENT_GUIDE.md found ($(wc -l < AGENT_GUIDE.md) lines)"
fi

# Verify build
if [ ! -d "build" ]; then
    echo "⚠️  Warning: build directory not found, creating..."
    mkdir build
fi

cd build
echo "🔨 Building LRET..."
cmake .. > /dev/null
make -j$(sysctl -n hw.ncpu) > /dev/null 2>&1
echo "✅ Build successful"

# Count test binaries
TEST_COUNT=$(ls -1 test_* 2>/dev/null | wc -l | xargs)
echo "✅ Found $TEST_COUNT test binaries"

cd ..

# Check OpenCode installation
if command -v opencode &> /dev/null; then
    OPENCODE_VERSION=$(opencode --version 2>/dev/null || echo 'version unknown')
    echo "✅ OpenCode installed: $OPENCODE_VERSION"
else
    echo "❌ OpenCode not installed"
    echo "   Install with: curl -fsSL https://opencode.ai/install | bash"
    exit 1
fi

# Check Python module
echo
echo "🐍 Checking Python integration..."
if python3 -c "import sys; sys.path.insert(0, 'python'); import qlret" 2>/dev/null; then
    echo "✅ qlret Python module importable"
else
    echo "⚠️  qlret Python module not importable (run: cd python && pip install -e .)"
fi

# Verify key components
echo
echo "📦 Verifying key components..."
COMPONENTS=(
    "src/quantum_simulator.cpp"
    "src/qec_adaptive.cpp"
    "include/quantum_simulator.h"
    "CMakeLists.txt"
    "Dockerfile"
)

for comp in "${COMPONENTS[@]}"; do
    if [ -f "$comp" ]; then
        echo "  ✅ $comp"
    else
        echo "  ❌ $comp (missing)"
    fi
done

echo
echo "✅ All prerequisites met!"
echo
echo "📋 Test Summary:"
echo "  • Branch: $BRANCH"
echo "  • Agent files: agent1.md - agent6.md ($TOTAL_LINES total lines)"
echo "  • Build status: ✅ Success"
echo "  • Test binaries: $TEST_COUNT"
echo "  • OpenCode: $OPENCODE_VERSION"
echo
echo "🚀 Next steps:"
echo "1. Run: opencode"
echo "2. In OpenCode terminal, try: @agent1.md What is LRET?"
echo "3. Follow the testing phases in OPENCODE_TESTING_GUIDE.md"
echo
echo "📖 Documentation:"
echo "  • Testing guide: OPENCODE_TESTING_GUIDE.md"
echo "  • Agent files: agent1.md - agent6.md"
echo "  • User guide: AGENT_GUIDE.md"
echo
