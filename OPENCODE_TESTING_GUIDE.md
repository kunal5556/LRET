# 🧪 OpenCodeAI Testing & Debugging Guide for LRET

## ✅ Status: agent-MD Branch Synced Successfully

**Date:** January 17, 2026  
**Branch:** `agent-MD` (rebased on latest `feature/framework-integration`)  
**Build Status:** ✅ All targets compiled successfully  
**Agent Files:** `agent1.md` through `agent6.md` (split documentation), `AGENT_GUIDE.md` (13.9KB)

---

## 📋 What We Did

1. ✅ Rebased `agent-MD` onto `feature/framework-integration` (11 commits of bug fixes)
2. ✅ Force-pushed to `origin/agent-MD`
3. ✅ Verified build compiles with all latest fixes
4. ✅ Split agent.md into 6 parts for easier navigation
5. ✅ Agent files preserved and ready for testing

---

## 🎯 Testing Strategy for OpenCodeAI

### Phase 1: Installation & Setup Verification

#### Step 1.1: Install OpenCode (if not already installed)

**macOS:**

```bash
curl -fsSL https://opencode.ai/install | bash
```

**Verify installation:**

```bash
opencode --version
which opencode
```

#### Step 1.2: Configure OpenCode for LRET

Check if OpenCode can recognize the repository:

```bash
cd /Users/suryanshsingh/Documents/LRET
opencode --help
```

---

### Phase 2: Test Agent File Recognition

#### Test 2.1: Verify Agent Files are Valid

**Check file structure:**

```bash
cd /Users/suryanshsingh/Documents/LRET
ls -la agent*.md
wc -l agent1.md agent2.md agent3.md agent4.md agent5.md agent6.md
```

**Validate markdown syntax:**

````bash
# Check for common issues
grep -n "```" agent1.md | head -20  # Check code blocks
grep -n "^#" agent1.md | head -30   # Check headers
````

#### Test 2.2: Test OpenCode Agent Initialization

**Start OpenCode with agent files:**

```bash
cd /Users/suryanshsingh/Documents/LRET
opencode
```

Once in OpenCode terminal, try:

```
@agent1.md What is LRET?
```

**Expected behavior:** Agent should recognize the context from the agent files and provide relevant information about LRET.

---

### Phase 3: Basic Functionality Tests

#### Test 3.1: Simple Query Test

**In OpenCode terminal:**

```
@agent1.md List the main components of the LRET quantum simulator
```

**What to verify:**

- [ ] Agent responds with information from agent files
- [ ] References correct file paths
- [ ] Understands LRET architecture

#### Test 3.2: Build Command Test

**In OpenCode terminal:**

```
@agent1.md How do I build the LRET project?
```

**What to verify:**

- [ ] Provides correct cmake commands
- [ ] References CMakeLists.txt correctly
- [ ] Mentions dependencies (Eigen3, MPI, etc.)

#### Test 3.3: Code Navigation Test

**In OpenCode terminal:**

```
@agent3.md Show me the quantum error correction implementation
```

**What to verify:**

- [ ] Navigates to correct files (src/qec*\*.cpp, include/qec*\*.h)
- [ ] Understands QEC subsystem
- [ ] Can explain the code

---

### Phase 4: Advanced Integration Tests

#### Test 4.1: Run a Test Binary

**In OpenCode terminal:**

```
@agent1.md Run the test_simple binary and explain the results
```

**What to verify:**

- [ ] Navigates to build/ directory
- [ ] Executes ./test_simple correctly
- [ ] Interprets output

#### Test 4.2: Python Integration Test

**In OpenCode terminal:**

```
@agent2.md Run a simple PennyLane quantum circuit using LRET device
```

**What to verify:**

- [ ] Uses python/qlret/pennylane_device.py
- [ ] Creates valid Python code
- [ ] Handles imports correctly

#### Test 4.3: Docker Workflow Test

**In OpenCode terminal:**

```
@agent1.md How would I run LRET in a Docker container?
```

**What to verify:**

- [ ] References Dockerfile correctly
- [ ] Provides docker build/run commands
- [ ] Mentions volumes for data persistence

---

### Phase 5: Error Handling & Debugging

#### Test 5.1: File Not Found Scenario

**In OpenCode terminal:**

```
@agent2.md Show me the file src/nonexistent_file.cpp
```

**Expected behavior:**

- [ ] Gracefully handles missing file
- [ ] Suggests similar files if available
- [ ] Doesn't crash

#### Test 5.2: Compilation Error Scenario

**Introduce a bug:**

```bash
# Create a test file with syntax error
echo "int main() { return 0 // missing semicolon" > test_error.cpp
```

**In OpenCode terminal:**

```
@agent2.md Compile test_error.cpp and fix any errors
```

**What to verify:**

- [ ] Detects compilation error
- [ ] Suggests fix
- [ ] Can apply fix automatically

#### Test 5.3: Complex Query Handling

**In OpenCode terminal:**

```
@agent3.md Compare the performance of LRET vs Cirq for a 10-qubit QFT circuit with depolarizing noise
```

**What to verify:**

- [ ] Breaks down complex request
- [ ] Uses benchmark_runner.h if relevant
- [ ] Provides meaningful comparison

---

### Phase 6: Agent Files Content Validation

#### Checklist for Agent Files Quality:

```bash
# Check key sections exist across agent files
grep -n "Getting Started" agent1.md
grep -n "Architecture" agent2.md
grep -n "API Reference" agent4.md
grep -n "Troubleshooting" agent1.md
grep -n "Examples" agent6.md
```

**Manual review:**

- [ ] All code examples use correct syntax
- [ ] File paths are accurate (use absolute paths)
- [ ] No broken markdown links
- [ ] Tool descriptions match actual functionality
- [ ] Examples are runnable

---

### Phase 7: Performance & Context Tests

#### Test 7.1: Large File Handling

**In OpenCode terminal:**

```
@agent2.md Summarize the entire src/fdm_simulator.cpp file
```

**What to verify:**

- [ ] Handles large file without timeout
- [ ] Provides coherent summary
- [ ] Doesn't truncate important info

#### Test 7.2: Multi-File Context

**In OpenCode terminal:**

```
@agent3.md How does checkpoint.cpp interact with qec_adaptive.cpp?
```

**What to verify:**

- [ ] Searches multiple files
- [ ] Understands relationships
- [ ] Provides accurate flow diagram

#### Test 7.3: Context Switching

**In OpenCode terminal:**

```
@agent2.md First explain quantum_sim.cpp, then switch to python bindings
```

**What to verify:**

- [ ] Maintains context between queries
- [ ] Smoothly transitions topics
- [ ] References previous responses

---

## 🔧 Common Issues & Fixes

### Issue 1: Agent Files Not Recognized

**Symptoms:**

- OpenCode doesn't respond to @agent1.md
- "File not found" errors

**Debug steps:**

```bash
# Verify files exist
ls -lh agent*.md

# Check OpenCode config
opencode config list

# Try explicit path
opencode --agent-file=/Users/suryanshsingh/Documents/LRET/agent1.md
```

**Fix:**
Update agent files frontmatter if needed.

### Issue 2: Incorrect File Paths in Responses

**Symptoms:**

- Agent references wrong directories
- Build commands fail

**Debug steps:**

```bash
# Verify current structure
tree -L 2 -d /Users/suryanshsingh/Documents/LRET

# Check agent file paths
grep -n "src/" agent2.md | head -20
```

**Fix:**
Update file paths in agent files to use absolute paths or workspace-relative paths.

### Issue 3: Build Commands Fail

**Symptoms:**

- cmake/make commands suggested by agent fail
- Dependency issues

**Debug steps:**

```bash
# Verify build system
cd /Users/suryanshsingh/Documents/LRET
cmake --version
make --version

# Check dependencies
brew list | grep eigen
which mpicxx
```

**Fix:**
Update build instructions in agent1.md to match actual system.

### Issue 4: Python Integration Fails

**Symptoms:**

- Import errors for qlret
- Device not found

**Debug steps:**

```bash
# Check Python module
python3 -c "import qlret; print(qlret.__file__)"
python3 -c "from qlret import QLRETDevice"

# Verify build
ls -la python/qlret/_qlret_native*.so
```

**Fix:**
Rebuild Python bindings:

```bash
cd build && make _qlret_native
cd ../python && pip install -e .
```

---

## 📊 Test Tracking Template

Use this checklist to track testing progress:

```markdown
## OpenCodeAI Test Results

**Test Date:** ****\_\_\_****
**OpenCode Version:** ****\_\_\_****
**Tester:** ****\_\_\_****

### Phase 1: Installation ✅ ❌ ⚠️

- [ ] OpenCode installed
- [ ] Repository recognized
- [ ] Agent file loaded

### Phase 2: Recognition ✅ ❌ ⚠️

- [ ] Agent files validated (agent1-6.md)
- [ ] Agent initialization
- [ ] Context loading

### Phase 3: Basic Functionality ✅ ❌ ⚠️

- [ ] Simple queries (score: \_\_/10)
- [ ] Build commands (score: \_\_/10)
- [ ] Code navigation (score: \_\_/10)

### Phase 4: Advanced Tests ✅ ❌ ⚠️

- [ ] Binary execution (score: \_\_/10)
- [ ] Python integration (score: \_\_/10)
- [ ] Docker workflows (score: \_\_/10)

### Phase 5: Error Handling ✅ ❌ ⚠️

- [ ] Missing files (score: \_\_/10)
- [ ] Compilation errors (score: \_\_/10)
- [ ] Complex queries (score: \_\_/10)

### Phase 6: Content Quality ✅ ❌ ⚠️

- [ ] Sections complete
- [ ] Examples runnable
- [ ] Paths accurate

### Phase 7: Performance ✅ ❌ ⚠️

- [ ] Large files (score: \_\_/10)
- [ ] Multi-file context (score: \_\_/10)
- [ ] Context switching (score: \_\_/10)

### Overall Score: \_\_\_/100

### Issues Found:

1. ***
2. ***
3. ***

### Recommended Fixes:

1. ***
2. ***
3. ***
```

---

## 🚀 Quick Start Testing Script

Save this as `test_opencode.sh`:

```bash
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
echo "✅ Agent files found"
wc -l agent1.md agent2.md agent3.md agent4.md agent5.md agent6.md

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
    echo "✅ OpenCode installed: $(opencode --version 2>/dev/null || echo 'version unknown')"
else
    echo "❌ OpenCode not installed"
    echo "   Install with: curl -fsSL https://opencode.ai/install | bash"
    exit 1
fi

echo
echo "✅ All prerequisites met!"
echo
echo "Next steps:"
echo "1. Run: opencode"
echo "2. In OpenCode terminal, try: @agent1.md What is LRET?"
echo "3. Follow the testing phases in OPENCODE_TESTING_GUIDE.md"
echo
```

Make it executable:

```bash
chmod +x test_opencode.sh
./test_opencode.sh
```

---

## 📝 Updating Agent Files Based on Test Results

### When to Update Agent Files:

1. **Incorrect file paths** → Update path references
2. **Outdated commands** → Update build/run instructions
3. **Missing features** → Add documentation for new features
4. **Wrong API usage** → Correct examples
5. **Poor explanations** → Improve clarity

### Update Workflow:

```bash
# 1. Make changes to relevant agent file
vim agent1.md  # or your editor

# 2. Test the changes
opencode  # Start fresh session to reload agent files

# 3. If good, commit
git add agent*.md
git commit -m "Update agent files: [describe changes]"
git push origin agent-MD

# 4. Retest to confirm
```

---

## 🔄 Syncing Future Changes

If `feature/framework-integration` gets more updates:

```bash
# Switch to agent-MD
git checkout agent-MD

# Fetch latest
git fetch origin

# Rebase again
git rebase origin/feature/framework-integration

# Force push
git push origin agent-MD --force-with-lease

# Rebuild
cd build && cmake .. && make -j$(sysctl -n hw.ncpu)
```

---

## 📞 Support & Resources

**Agent Files:**

- [agent1.md](agent1.md) - Quick Reference & Getting Started (~600 lines)
- [agent2.md](agent2.md) - Core Implementation Phases 1-4 (~900 lines)
- [agent3.md](agent3.md) - Backend & Execution Phases 5-10 (~1000 lines)
- [agent4.md](agent4.md) - Session, Batch & API Phases 11-13 (~950 lines)
- [agent5.md](agent5.md) - Visualization, Cloud & ML Phases 14-16 (~1300 lines)
- [agent6.md](agent6.md) - Collaboration & User Guide Phase 17 (~750 lines)
- [AGENT_GUIDE.md](AGENT_GUIDE.md) - User guide (510 lines)

**Documentation:**

- [README.md](README.md) - Project overview
- [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) - All docs
- [QUICK_START_PHASE_7.md](QUICK_START_PHASE_7.md) - Quick start

**OpenCode Resources:**

- GitHub: https://github.com/anomalyco/opencode
- Docs: https://docs.opencode.ai
- Discord: https://discord.gg/opencode

---

## ✅ Success Criteria

OpenCodeAI integration is successful when:

- [ ] Agent responds accurately to 90%+ queries
- [ ] Can build project from agent instructions
- [ ] Can run tests via agent commands
- [ ] Handles errors gracefully
- [ ] Provides correct file paths
- [ ] Python integration works
- [ ] Docker workflows execute
- [ ] Performance is acceptable (<5s response time)
- [ ] No crashes or hangs
- [ ] Context switching works smoothly

**Target Score: 85/100 or higher**

---

**Last Updated:** January 17, 2026  
**Status:** Ready for Testing ✅
