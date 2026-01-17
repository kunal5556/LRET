# 🤖 Quantum Simulation Agent - Complete Beginner's Guide

Welcome! This guide will help you use the Quantum Simulation Agent, even if you have no prior experience with quantum computing or programming.

---

## 📋 Table of Contents

1. [What is This Agent?](#what-is-this-agent)
2. [What Can It Do?](#what-can-it-do)
3. [Prerequisites](#prerequisites)
4. [Installation & Setup](#installation--setup)
5. [Basic Usage Examples](#basic-usage-examples)
6. [Understanding the Output](#understanding-the-output)
7. [Advanced Examples](#advanced-examples)
8. [Troubleshooting](#troubleshooting)
9. [FAQ](#faq)
10. [Glossary](#glossary)

---

## 🎯 What is This Agent?

The **Quantum Simulation Agent** is an AI-powered assistant that helps you run quantum computing simulations using two powerful frameworks:

- **LRET (Low-Rank Environment Truncation)**: A specialized framework for simulating quantum systems efficiently
- **Cirq**: Google's quantum computing framework

Think of it as a translator - you can describe what you want in plain English, and it converts your request into the technical commands needed to run quantum simulations.

---

## ✨ What Can It Do?

### For Complete Beginners:

- Run simple quantum simulations by just describing what you want
- Visualize quantum circuits (the "programs" for quantum computers)
- Compare different simulation methods
- Test quantum algorithms with realistic noise

### For Intermediate Users:

- Benchmark performance of different approaches
- Optimize quantum circuits
- Run batch simulations with different parameters
- Export results in various formats (CSV, JSON, plots)

### For Advanced Users:

- Custom noise models
- Multi-GPU acceleration
- Integration with existing quantum workflows
- Hardware execution on Google quantum processors

---

## 📦 Prerequisites

### Required Software

1. **Docker** (Recommended for beginners)
   - Download from: https://www.docker.com/get-started
   - Why? It packages everything you need in one container
   - No complex setup required!

2. **Python 3.8 or higher** (Alternative approach)
   - Download from: https://www.python.org/downloads/
   - Make sure to check "Add Python to PATH" during installation

### Optional But Helpful

- **Git**: For downloading examples and updates
- **VS Code**: A friendly code editor
- **Basic terminal knowledge**: Don't worry, we'll guide you!

---

## 🚀 Installation & Setup

### Option 1: Using Docker (Easiest!)

#### Step 1: Install Docker

Download and install Docker Desktop for your operating system.

#### Step 2: Pull the LRET Image

Open your terminal (PowerShell on Windows, Terminal on Mac/Linux) and run:

```bash
docker pull ghcr.io/kunal5556/lret:latest
```

This downloads the quantum simulation environment (may take 5-10 minutes).

#### Step 3: Verify Installation

```bash
docker run ghcr.io/kunal5556/lret:latest quantum_sim --help
```

If you see a help message, you're ready to go! 🎉

### Option 2: Using Python (More Control)

#### Step 1: Install Python Packages

```bash
pip install cirq cirq-google qsimcirq numpy matplotlib scipy
```

#### Step 2: Clone LRET Repository

```bash
git clone https://github.com/kunal5556/LRET.git
cd LRET
pip install -e .
```

#### Step 3: Test Installation

```bash
quantum_sim --version
```

---

## 🎓 Basic Usage Examples

### Example 1: Your First Quantum Simulation (Super Simple!)

**What you want to do:** Run a basic quantum circuit with 3 qubits

**How to ask the agent:**

```
"Run a simple 3-qubit quantum simulation with Hadamard gates using LRET"
```

**What the agent does:**

1. Creates a 3-qubit quantum circuit
2. Applies Hadamard gates (creates quantum superposition)
3. Runs the simulation
4. Shows you the results

**Expected output:**

```
Simulation Results:
- Number of qubits: 3
- Circuit depth: 1
- Final state vector: [0.35+0j, 0.35+0j, 0.35+0j, ...]
- Execution time: 0.05 seconds
```

### Example 2: Comparing Two Simulators

**What you want to do:** See which simulator is faster

**How to ask the agent:**

```
"Compare LRET and Cirq performance for a 5-qubit circuit with CNOT gates"
```

**What the agent does:**

1. Creates identical 5-qubit circuits
2. Runs on both LRET and Cirq
3. Compares speed, accuracy, and memory usage
4. Shows you a comparison report

**Expected output:**

```
Performance Comparison:
┌──────────────┬──────────┬──────────┐
│ Framework    │ Time (s) │ Memory   │
├──────────────┼──────────┼──────────┤
│ LRET         │ 0.12     │ 45 MB    │
│ Cirq         │ 0.18     │ 62 MB    │
└──────────────┴──────────┴──────────┘

LRET is 33% faster!
```

### Example 3: Adding Realistic Noise

**What you want to do:** Simulate a real quantum computer with errors

**How to ask the agent:**

```
"Simulate a 4-qubit GHZ state with depolarizing noise at 1% error rate"
```

**What the agent does:**

1. Creates a GHZ state (maximally entangled state)
2. Adds noise to mimic real quantum hardware
3. Shows how noise affects the results

**Expected output:**

```
GHZ State with Noise:
- Ideal fidelity: 1.00 (perfect)
- Noisy fidelity: 0.94 (realistic)
- Noise degradation: 6%

Recommendation: This circuit is robust to noise!
```

### Example 4: Visualizing Your Circuit

**What you want to do:** See what your quantum circuit looks like

**How to ask the agent:**

```
"Create a 3-qubit Quantum Fourier Transform and show me the circuit diagram"
```

**What the agent does:**

1. Generates the circuit
2. Creates a visual diagram
3. Saves it as an image file

**Expected output:**

```
Circuit created: quantum_fourier_transform.png

q0: ───H───@───────────────@───────
           │               │
q1: ───────X───H───@───────┼───────
                   │       │
q2: ───────────────X───H───X───────

File saved to: ./quantum_fourier_transform.png
```

---

## 📊 Understanding the Output

### Key Terms in Results

**State Vector**

- The complete quantum state description
- Complex numbers representing probability amplitudes
- Example: `[0.71+0j, 0+0.71j]` means 71% chance for each state

**Fidelity**

- How close your result is to the ideal (perfect) result
- Range: 0.0 (terrible) to 1.0 (perfect)
- Above 0.95 is usually considered good

**Circuit Depth**

- How many "layers" of gates in your circuit
- Lower is usually better (less error accumulation)

**Execution Time**

- How long the simulation took
- Depends on number of qubits and circuit complexity

---

## 🔥 Advanced Examples

### Example 5: Batch Processing

**Scenario:** You want to test different noise levels

```
"Run a benchmark of a 6-qubit random circuit with noise levels from 0% to 5% in 1% steps"
```

**What happens:**

- Runs 6 simulations automatically
- Creates a comparison chart
- Shows optimal noise threshold

### Example 6: GPU Acceleration

**Scenario:** You have large simulations and a GPU

```
"Simulate a 12-qubit circuit with random gates using GPU acceleration"
```

**What happens:**

- Automatically detects your GPU
- Offloads computation to GPU
- Can be 10-100x faster than CPU!

### Example 7: Custom Circuit from File

**Scenario:** You have a circuit saved as QASM

```
"Load circuit from my_circuit.qasm and run it with density matrix simulation"
```

**What happens:**

- Reads your QASM file
- Converts to appropriate format
- Runs with noise support (density matrix)

### Example 8: Export Results

**Scenario:** You need data for a report

```
"Run a variational quantum eigensolver simulation and export results as CSV and plot"
```

**What happens:**

- Runs the VQE algorithm
- Saves numerical data to CSV
- Creates publication-quality plots
- Generates summary report

---

## 🔧 Troubleshooting

### Problem: "Docker is not recognized"

**Solution:**

1. Make sure Docker Desktop is running
2. Restart your terminal
3. On Windows: Check if Docker is in your PATH

### Problem: "Module 'cirq' not found"

**Solution:**

```bash
pip install cirq --upgrade
```

### Problem: "Out of memory error"

**Solution:**

- Reduce number of qubits (try n-2 qubits)
- Use LRET instead of Cirq (more memory efficient)
- Add `--memory-limit 8GB` flag
- Enable GPU if available

### Problem: "Simulation is very slow"

**Solution:**

- Use LRET's `--mode batch` for parallelization
- Enable GPU: `--gpu`
- Reduce circuit depth
- Use approximate methods: `--threshold 1e-6`

### Problem: "Results don't match expected"

**Solution:**

- Check if noise is enabled (might be unintentional)
- Verify qubit ordering (Cirq uses different convention)
- Compare with `--compare-fdm` flag
- Use `--fidelity-check` to validate

---

## ❓ FAQ

### Q: Do I need to know quantum mechanics?

**A:** Not at all! The agent handles the technical details. You just describe what you want.

### Q: How many qubits can I simulate?

**A:**

- **Laptop (8GB RAM)**: Up to 15 qubits comfortably
- **Desktop (32GB RAM)**: Up to 20 qubits
- **Server/GPU**: 25+ qubits possible with LRET
- **Note**: Each additional qubit doubles memory needs!

### Q: Is this using a real quantum computer?

**A:** By default, no - it's a simulation. But you can connect to real Google quantum hardware if you have access credentials.

### Q: Can I use this for my research paper?

**A:** Yes! Just cite both LRET and Cirq appropriately:

- LRET: Check the GitHub repository for citation info
- Cirq: https://quantumai.google/cirq/citing

### Q: What's the difference between LRET and Cirq?

**A:**
| Feature | LRET | Cirq |
|---------|------|------|
| Speed for large systems | ⚡⚡⚡ Faster | ⚡⚡ Fast |
| Memory efficiency | ✅ Better | ⚫ Standard |
| Noise simulation | ✅ Advanced | ✅ Good |
| Hardware access | ⚫ No | ✅ Google QPU |
| Community size | 🔹 Smaller | 🔹🔹🔹 Large |

**Rule of thumb:** Use LRET for large simulations, Cirq for Google hardware access.

### Q: Are the results accurate?

**A:** Yes! Both frameworks are scientifically validated. You can verify accuracy using the `--validate` flag which compares multiple methods.

### Q: Can I run this on a cloud service?

**A:** Absolutely! Works on:

- Google Cloud Platform
- AWS
- Azure
- Any service that supports Docker

---

## 📚 Glossary

**Qubit**: Quantum bit - the basic unit of quantum information (like a regular bit but can be both 0 and 1 simultaneously)

**Gate**: An operation applied to qubits (like quantum "instructions")

**Circuit**: A sequence of gates applied to qubits (like a quantum "program")

**Superposition**: When a qubit is in multiple states at once (the "magic" of quantum computing)

**Entanglement**: When qubits become correlated (change one, affects the other instantly)

**Measurement**: Reading the qubit state (collapses superposition to 0 or 1)

**Noise**: Errors that occur in real quantum computers

**Fidelity**: How accurate your simulation is compared to the ideal

**Density Matrix**: Mathematical representation that includes noise effects

**QASM**: Quantum Assembly Language - a text format for quantum circuits

**VQE**: Variational Quantum Eigensolver - an algorithm for finding ground states

**GHZ State**: Greenberger-Horne-Zeilinger state - maximally entangled state

**Hadamard Gate (H)**: Creates superposition - puts qubit in equal probability of 0 and 1

**CNOT Gate**: Controlled-NOT - two-qubit gate for creating entanglement

---

## 🎯 Quick Reference Commands

### Simple Simulations

```
"Run a 5-qubit circuit with random gates"
"Simulate a Bell state pair"
"Create a 3-qubit GHZ state"
```

### With Noise

```
"Add 2% depolarizing noise to a 4-qubit circuit"
"Simulate IBM noise model for 6 qubits"
"Test amplitude damping with T1 = 50 microseconds"
```

### Performance Testing

```
"Benchmark LRET vs Cirq for 10 qubits"
"Compare CPU and GPU execution for 12 qubits"
"Test memory usage for circuits with 15 qubits"
```

### Visualization

```
"Show me the circuit diagram"
"Plot the state vector as a bar chart"
"Generate a Bloch sphere visualization"
```

### Export & Analysis

```
"Export results as CSV"
"Save the circuit as QASM"
"Generate a summary report in JSON"
```

---

## 🆘 Getting Help

### Resources

1. **LRET Documentation**: https://github.com/kunal5556/LRET
2. **Cirq Documentation**: https://quantumai.google/cirq
3. **Quantum Computing Basics**:
   - IBM Quantum Learning: https://learning.quantum.ibm.com/
   - Qiskit Textbook: https://qiskit.org/learn

### Contact & Support

- **LRET Issues**: Open an issue on GitHub
- **Cirq Questions**: Use the Cirq discussion forum
- **General Quantum Help**: Quantum Computing Stack Exchange

---

## 🎉 You're Ready!

Start with simple examples and gradually explore more complex features. Remember:

1. **Start Small**: Begin with 2-3 qubits
2. **Experiment**: Try different gates and configurations
3. **Compare**: Use both LRET and Cirq to understand differences
4. **Visualize**: Always look at circuit diagrams
5. **Document**: Keep notes on what works well

Happy quantum computing! 🚀✨

---

**Last Updated**: January 17, 2026  
**Version**: 1.1  
**Agent Files**: agent1.md - agent6.md
