# LRET AI Agent - Part 1: Quick Reference & Getting Started

**AI Agent for Quantum Simulation Research**  
_Intelligent assistant for LRET (Low-Rank Entanglement Tracking) quantum simulator_  
_Designed for [OpenCode AI](https://github.com/anomalyco/opencode) Integration_

---

## 📌 Quick Reference

| Item                     | Value                                                                                  |
| ------------------------ | -------------------------------------------------------------------------------------- |
| **Target Repository**    | [kunal5556/LRET](https://github.com/kunal5556/LRET/tree/feature/framework-integration) |
| **OpenCode Integration** | [anomalyco/opencode](https://github.com/anomalyco/opencode)                            |
| **Branch**               | `feature/framework-integration`                                                        |
| **Agent Mode**           | Primary + Subagent support                                                             |
| **Default Model**        | `anthropic/claude-sonnet-4-20250514`                                                   |

---

## 📚 Document Structure

This implementation guide is split into 6 parts for easier navigation:

| Part  | File        | Contents                                                       |
| ----- | ----------- | -------------------------------------------------------------- |
| **1** | `agent1.md` | Quick Reference, Getting Started, Setup, Common Commands       |
| **2** | `agent2.md` | Phase 1-4: Foundation, Intent, Execution, Backend Management   |
| **3** | `agent3.md` | Phase 5-10: Safety, Memory, Code Editing, Testing              |
| **4** | `agent4.md` | Phase 11-13: Session Management, Batch Execution, Web API      |
| **5** | `agent5.md` | Phase 14-16: Visualization, Cloud Integration, ML Optimization |
| **6** | `agent6.md` | Phase 17: Collaboration, User Guide, Documentation             |

---

## 🚀 Getting Started Guide (For Non-Technical Users)

This guide will help you set up and use the LRET AI Agent with OpenCode AI - no technical expertise required! Just follow these steps in order.

### Step 1: Install OpenCode AI (One-Time Setup)

**What is OpenCode?** OpenCode is an AI-powered coding assistant that works in your terminal. We'll use it to talk to the LRET quantum simulator.

**Install on Windows:**

1. Open PowerShell (search for "PowerShell" in Windows Start menu)
2. Copy and paste this command, then press Enter:
   ```powershell
   irm https://opencode.ai/install.ps1 | iex
   ```
3. Wait for it to finish (it will download and install OpenCode)
4. Close and reopen PowerShell

**Install on Mac/Linux:**

1. Open Terminal (search for "Terminal" in your applications)
2. Copy and paste this command, then press Enter:
   ```bash
   curl -fsSL https://opencode.ai/install | bash
   ```
3. Wait for it to finish
4. Close and reopen Terminal

**Verify Installation:**
Type `opencode --version` and press Enter. You should see a version number.

---

### Step 2: Get Your API Key

**What's an API key?** It's like a password that lets OpenCode talk to the AI (Claude).

1. Go to https://console.anthropic.com/
2. Sign up for an account (or log in if you have one)
3. Click "API Keys" in the left sidebar
4. Click "Create Key"
5. Copy the key (it looks like `sk-ant-...`)

**Save Your API Key:**

**On Windows:**

```powershell
# Open PowerShell and run:
setx ANTHROPIC_API_KEY "your-key-here"
```

_(Replace `your-key-here` with your actual key)_

**On Mac/Linux:**

```bash
# Open Terminal and run:
echo 'export ANTHROPIC_API_KEY="your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

_(Replace `your-key-here` with your actual key)_

---

### Step 3: Download the LRET Repository

**What's a repository?** Think of it as a project folder containing all the quantum simulator code.

1. **Install Git** (if you don't have it):
   - Windows: Download from https://git-scm.com/download/win
   - Mac: Type `git --version` in Terminal (it will prompt you to install)
   - Linux: Type `sudo apt-get install git` (Ubuntu) or `sudo yum install git` (RedHat)

2. **Download LRET:**

   ```bash
   # Choose a folder where you want to download LRET
   cd ~

   # Download the LRET repository
   git clone https://github.com/kunal5556/LRET.git
   cd LRET

   # Switch to the correct branch
   git checkout feature/framework-integration
   ```

---

### Step 4: Set Up the Agent File

**What are the agent files?** They are instruction manuals that teach OpenCode how to work with LRET. The documentation is split into 6 parts for easier navigation.

1. **Create the folder:**

   ```bash
   mkdir -p .opencode/agent
   ```

2. **Copy the agent files:**
   ```bash
   # Copy all agent parts to the OpenCode agent folder
   cp agent1.md agent2.md agent3.md agent4.md agent5.md agent6.md .opencode/agent/
   # Or combine them into a single file
   cat agent1.md agent2.md agent3.md agent4.md agent5.md agent6.md > .opencode/agent/lret.md
   ```

**Verify Setup:**

```bash
ls -la .opencode/agent/
```

---

### Step 5: Build LRET (One-Time Setup)

**What's building?** It's like compiling all the code into a program you can run.

1. **Install Prerequisites:**

   **On Ubuntu/Debian Linux:**

   ```bash
   sudo apt-get update
   sudo apt-get install -y cmake g++ libeigen3-dev libhdf5-dev python3-pip
   ```

   **On Mac:**

   ```bash
   brew install cmake eigen hdf5
   ```

   **On Windows:**
   - Install Visual Studio Community Edition from https://visualstudio.microsoft.com/
   - During installation, select "Desktop development with C++"
   - Install CMake from https://cmake.org/download/

2. **Build the Project:**

   ```bash
   cd ~/LRET
   mkdir build && cd build
   cmake ..
   cmake --build . -j4

   # Test that it worked
   ./quantum_sim --help
   ```

---

### Step 6: Start Using the Agent

1. **Open OpenCode:**

   ```bash
   cd ~/LRET
   opencode
   ```

2. **The OpenCode Interface:**
   - You'll see a prompt like: `You>`
   - Type your questions or requests in natural language
   - Press Enter to send
   - Type `exit` to quit

---

### Step 7: Your First Simulation

**Type this at the OpenCode prompt:**

```
run a 10 qubit simulation with depth 20
```

**What happens:**

1. The AI understands your request
2. It shows you a plan (what it will do)
3. It asks for permission to run the simulation
4. Type `y` (yes) and press Enter
5. The simulation runs and you'll see results!

---

## 📖 Common Natural Language Commands

### Running Simulations

```bash
# Simple simulations
> run 8 qubit simulation
> simulate 12 qubits with depth 30

# With noise
> run 10 qubits with 1% noise
> simulate 8 qubits with amplitude damping noise

# With specific settings
> run 10 qubits, depth 20, hybrid mode, save to results.csv
> simulate with GPU acceleration
```

### Benchmarking (Performance Tests)

```bash
> benchmark qubits 8,10,12 with depths 20,30,40
> run performance test with 5 trials
> compare all parallelization modes
```

### Comparing Results

```bash
> compare the last two experiments
> explain why this run was slower than the previous one
> show me the differences in fidelity
```

### Learning About the Code

```bash
> explain how the truncation algorithm works
> show me where the noise model is implemented
> what does the rank threshold parameter do?
```

### Getting Help

```bash
> what can you do?
> how do I run a simulation with custom noise?
> explain the output metrics
```

---

## 💡 Pro Tips

### Permission System

- OpenCode will ask permission before:
  - Running experiments
  - Modifying code
  - Installing packages
- Type `y` for yes, `n` for no
- Type `always` to allow similar actions without asking again

### Saving Results

```bash
> run 10 qubits and save results to my_experiment.json
> run benchmark and save to benchmark_results.csv
```

### Dry-Run Mode (Preview Without Running)

```bash
# Start OpenCode with dry-run flag:
opencode --dry-run

# Now type commands - it will show what it WOULD do
> run 10 qubit simulation
# (You'll see the plan but nothing will execute)
```

### Session History

- OpenCode remembers your conversation
- You can refer to previous experiments:
  ```
  > run the same experiment again but with 2% noise
  > increase the qubit count from the last run
  ```

---

## 🔧 Troubleshooting

### "Command not found: opencode"

**Fix:** OpenCode isn't installed or not in your PATH.

- Reinstall using Step 1
- Close and reopen your terminal

### "API key not found"

**Fix:** Your API key isn't set up.

- Redo Step 2
- Verify with: `echo $ANTHROPIC_API_KEY`

### "quantum_sim: command not found"

**Fix:** LRET isn't built or you're in the wrong folder.

- Go to the LRET folder: `cd ~/LRET`
- Check if the build exists: `ls build/quantum_sim`

### "Agent file not found"

**Fix:** The agent files aren't in the right place.

- Check: `ls .opencode/agent/`
- Verify all agent files exist: `ls agent*.md`

### Simulation Fails or Crashes

**Try:**

1. Start with smaller simulations (6-8 qubits)
2. Ask the agent: `why did the last simulation fail?`
3. Check your computer has enough RAM (at least 8GB recommended)

---

## 🎯 Strategic Vision

This document outlines the complete implementation strategy for building an intelligent AI agent that integrates with **OpenCode AI** to assist researchers and developers in using the **LRET quantum simulator**.

### Core Objectives

1. **Democratize Quantum Research**: Enable researchers to run complex quantum simulations through natural language
2. **Ensure Safety**: Maintain human-in-the-loop control for all risky operations
3. **Maximize Productivity**: Remember context, compare experiments, and provide insights
4. **Guarantee Reproducibility**: Log all actions, decisions, and results for scientific rigor
5. **Support Multiple Workflows**: From quick experiments to multi-day research campaigns
6. **OpenCode Native**: Seamless integration with OpenCode's agent system

---

## 📋 Complete Capability Overview

### ✅ 1. User Interaction & Interface

- Accept natural-language instructions from terminal / VS Code
- Accept structured CLI flags (backend, model, dry-run, configs, etc.)
- Support follow-up questions in the same session
- Clearly show plans, actions, and outcomes to the user

### ✅ 2. Intent Understanding & Planning

- Understand user intent: run experiment, compare, explain, inspect code
- Convert user intent into step-by-step plans
- Show the plan before execution
- Re-plan if a step fails

### ✅ 3. Safety & Control (Human-in-the-Loop)

- Categorize actions: read (safe), run (medium), write (high risk)
- Enforce permissions for each category
- Require explicit user confirmation for risky actions
- Support dry-run mode

### ✅ 4. Repo-Aware Code Interaction

- Read files, search for symbols
- Propose code changes as diffs
- Apply changes only after approval
- Roll back changes if build fails

### ✅ 5. Experiment / Simulator Execution

- Build the LRET project
- Run experiments deterministically
- Capture stdout, stderr, exit codes
- Support long-running experiments

### ✅ 6. Backend Support

- Support State Vector, Density Matrix, LRET, Cirq backends
- Allow user to explicitly select backend
- Automatically choose best backend based on query
- Compare results across backends

### ✅ 7. Canonical Result Handling

- Collect raw experiment outputs
- Map to canonical LRET result schema
- Support partial results (crashed runs)

### ✅ 8. Validation & Schema Enforcement

- Validate results before explanation
- Detect missing/invalid fields
- Block explanations on invalid results

### ✅ 9. Explanation & Analysis

- Explain results in plain language
- Compare multiple runs
- Explain performance differences

### ✅ 10. Memory & State Management

- Remember past prompts, configs, results
- Enable comparisons with previous runs
- Persist memory across sessions

### ✅ 11. Multi-Model LLM Support

- Support Claude, GPT, Gemini
- Use common LLM interface
- Allow switching models

### ✅ 12. Code Editing & Improvement

- Suggest improvements
- Generate minimal diffs
- Rebuild and test after edits

### ✅ 13. Rollback & Recovery

- Snapshot state before risky actions
- Roll back failed changes
- Clean up temporary files

### ✅ 14. Observability & Auditability

- Generate structured logs
- Maintain audit trails
- Capture timing metrics

### ✅ 15. Research & Productivity Features

- Compare experiments across time
- Track experiment lineage
- Support reproducibility

---

## 🔧 OpenCode AI Integration

### Agent Configuration Files

OpenCode supports two formats for agent configuration:

- **Global**: `~/.config/opencode/agent/lret.md`
- **Per-project**: `.opencode/agent/lret.md`

### OpenCode Agent Definition (Markdown Format)

**File: `.opencode/agent/lret.md`**

```markdown
---
description: AI agent for LRET quantum simulation - run experiments, analyze results, compare backends
mode: primary
model: anthropic/claude-sonnet-4-20250514
temperature: 0.2
maxSteps: 50
tools:
  read: true
  write: true
  edit: true
  bash: true
  webfetch: false
permission:
  edit: ask
  bash:
    "cmake*": allow
    "make*": allow
    "./quantum_sim*": allow
    "python*": allow
    "ctest*": allow
    "pytest*": allow
    "git status": allow
    "git diff*": allow
    "rm -rf*": deny
    "*": ask
---

You are the LRET AI Agent, an intelligent assistant for quantum simulation research supporting both LRET (Low-Rank Entanglement Tracking) and Google Cirq backends.

## Your Capabilities

1. **Run Experiments**: Execute quantum simulations on LRET or Cirq
2. **Compare Backends**: Run simulations on both backends, compare results
3. **Explain Metrics**: Interpret simulation results
4. **Inspect Code**: Navigate and explain the codebase
5. **Propose Changes**: Suggest optimizations with diff-based changes

## Backend Support

### LRET Backends

- **State Vector**: Best for small circuits (< 12 qubits)
- **Density Matrix (LRET)**: Best for noisy circuits
- **FDM**: Full density matrix for validation

### Cirq Backends

- **Simulator**: Cirq's default state vector simulator
- **qsim**: Google's fast C++ simulator
- **Hardware**: Real Google quantum processors

## Safety Rules

- ALWAYS show execution plan before running experiments
- NEVER modify code without explicit user approval
- ALWAYS validate results before explanation
- For long-running experiments (> 5 minutes), ask for confirmation
```

### OpenCode Agent Definition (JSON Format)

**File: `opencode.json`**

```json
{
  "$schema": "https://opencode.ai/config.json",
  "agent": {
    "lret": {
      "description": "LRET quantum simulator agent",
      "mode": "primary",
      "model": "anthropic/claude-sonnet-4-20250514",
      "temperature": 0.2,
      "maxSteps": 50,
      "tools": {
        "read": true,
        "write": true,
        "edit": true,
        "bash": true
      },
      "permission": {
        "edit": "ask",
        "bash": {
          "cmake*": "allow",
          "make*": "allow",
          "./quantum_sim*": "allow",
          "*": "ask"
        }
      }
    },
    "lret-readonly": {
      "description": "Read-only LRET agent for exploration",
      "mode": "subagent",
      "model": "anthropic/claude-haiku-4-20250514",
      "tools": {
        "read": true,
        "write": false,
        "bash": false
      }
    }
  }
}
```

---

## 🗓️ Implementation Phases Overview

| Phase     | Weeks | Focus                                                              |
| --------- | ----- | ------------------------------------------------------------------ |
| **1**     | 1-2   | Foundation & Core Infrastructure                                   |
| **2**     | 3-4   | Intent Understanding & Planning                                    |
| **3**     | 5-6   | Experiment Execution & Results                                     |
| **4**     | 7-8   | Backend Management & Validation                                    |
| **5**     | 9-10  | Safety & Permissions                                               |
| **6**     | 11-12 | Memory & State                                                     |
| **7**     | 13-14 | Code Interaction & Editing                                         |
| **8**     | 15-16 | Rollback & Auditability                                            |
| **9**     | 17-18 | Research Features                                                  |
| **10**    | 19-20 | Polish & Testing                                                   |
| **11-17** | 21+   | Advanced Features (Sessions, Batch, API, Cloud, ML, Collaboration) |

---

_Continue to [agent2.md](agent2.md) for Phase 1-4 implementation details._
