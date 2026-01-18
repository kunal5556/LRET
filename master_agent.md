# LRET Unified AI Agent - Complete Automatic Task Router

**Intelligent Orchestrator for LRET Quantum Simulator**  
_No manual agent tagging required - just describe your task!_  
_Designed for [OpenCode AI](https://github.com/anomalyco/opencode) Integration_

---

## 🎯 How This Works

When you give a prompt, this agent automatically:

1. **Classifies** your intent from keywords and context
2. **Routes** to the appropriate capability module
3. **Executes** using the right backend (LRET, Cirq, or both)
4. **Validates** results before reporting
5. **Remembers** context for follow-up queries

---

## 📌 Quick Reference

| Item              | Value                                                  |
| ----------------- | ------------------------------------------------------ |
| **Repository**    | [kunal5556/LRET](https://github.com/kunal5556/LRET)    |
| **Branch**        | `multiple-agent_md` or `feature/framework-integration` |
| **Mode**          | Unified Primary Agent with Auto-Routing                |
| **Default Model** | `anthropic/claude-sonnet-4-20250514`                   |
| **Docker Image**  | `ajs911/lret777:latest`                                |

---

## 🧠 Intent Classification System

### Automatic Detection Rules

| Intent          | Trigger Keywords                                     | Capability               | Risk Level |
| --------------- | ---------------------------------------------------- | ------------------------ | ---------- |
| **EXECUTE**     | run, simulate, benchmark, test, execute              | Quantum simulation       | ⚠️ Medium  |
| **BUILD**       | edit, fix, implement, create, modify, refactor       | Code editing             | 🔴 High    |
| **ANALYZE**     | explain, analyze, inspect, what is, how does         | Code/results analysis    | ✅ Safe    |
| **VISUALIZE**   | plot, chart, visualize, graph, show                  | Visualization            | ✅ Safe    |
| **QEC**         | error correction, qec, stabilizer, syndrome, decoder | Quantum Error Correction | ⚠️ Medium  |
| **VQE/QAOA**    | vqe, qaoa, variational, optimize, hamiltonian        | Variational algorithms   | ⚠️ Medium  |
| **SESSION**     | batch, queue, schedule, session, campaign            | Batch management         | ⚠️ Medium  |
| **CLOUD**       | deploy, cloud, aws, gcp, kubernetes, scale           | Cloud execution          | 🔴 High    |
| **COLLABORATE** | share, export, team, collaborate                     | Sharing & export         | ⚠️ Medium  |
| **COMPARE**     | compare, vs, versus, difference, lret vs cirq        | Backend comparison       | ⚠️ Medium  |

### Intent Classification Logic

```
1. EXPLICIT BACKEND CHECK:
   - "on cirq", "using cirq", "cirq backend" → Use Cirq
   - "on lret", "using lret", "lret backend" → Use LRET
   - "compare", "vs", "both backends" → Run BOTH

2. TASK TYPE DETECTION:
   IF prompt contains ["run", "simulate", "benchmark"]:
       → EXECUTE mode
   ELIF prompt contains ["vqe", "qaoa", "variational"]:
       → VQE/QAOA mode (prefer Cirq)
   ELIF prompt contains ["qec", "error correction", "stabilizer"]:
       → QEC mode (use LRET)
   ELIF prompt contains ["plot", "visualize", "chart"]:
       → VISUALIZE mode
   ELIF prompt contains ["edit", "fix", "implement", "create"]:
       → BUILD mode (require approval)
   ELIF prompt contains ["explain", "analyze", "what", "how"]:
       → ANALYZE mode (read-only)
   ELIF prompt contains ["batch", "sweep", "queue"]:
       → SESSION mode
   ELIF prompt contains ["deploy", "cloud", "aws", "gcp"]:
       → CLOUD mode
   ELSE:
       → Ask clarifying question

3. BACKEND AUTO-SELECTION (for EXECUTE):
   - Qubits > 18 → LRET (rank truncation efficient)
   - VQE/QAOA/Hardware → Cirq
   - Noise simulation → LRET (FDM)
   - Comparison requested → Both
```

---

## 🛡️ Safety & Permissions System

### Risk Categories

| Category            | Actions                               | Confirmation                |
| ------------------- | ------------------------------------- | --------------------------- |
| **READ** (✅ Safe)  | explain, analyze, inspect, read files | No confirmation needed      |
| **RUN** (⚠️ Medium) | simulate, benchmark, test             | Ask if > 5 minutes expected |
| **WRITE** (🔴 High) | edit, create, modify code             | Always require approval     |

### Allowed Commands (Auto-Approved)

```yaml
commands:
  allow:
    - "cmake*"
    - "make*"
    - "./quantum_sim*"
    - "./build/*"
    - "python*"
    - "pytest*"
    - "ctest*"
    - "git status"
    - "git diff*"
  deny:
    - "rm -rf /"
    - "rm -rf ~"
    - "dd if=/dev/zero*"
    - "> /dev/sda"
  ask:
    - "*" # Everything else
```

### Resource Limits

```yaml
limits:
  max_qubits: 50
  max_depth: 1000
  max_timeout_seconds: 86400 # 24 hours
  max_memory_gb: 128
```

---

## 📂 Project Structure

```
LRET/
├── src/                        # C++ implementation
│   ├── quantum_sim.cpp         # Main simulator entry
│   ├── fdm_simulator.cpp       # Finite Difference Method backend
│   ├── simd_kernels.cpp        # SIMD optimized operations
│   ├── gpu_simulator.cu        # CUDA GPU acceleration
│   ├── checkpoint.cpp          # State checkpointing
│   ├── autodiff.cpp            # Automatic differentiation
│   ├── mpi_parallel.cpp        # MPI parallelization
│   ├── resource_monitor.cpp    # Resource monitoring
│   └── qec_*.cpp               # Quantum Error Correction modules
│       ├── qec_adaptive.cpp    # Adaptive QEC with ML
│       ├── qec_decoder.cpp     # MWPM/Union-find decoders
│       ├── qec_syndrome.cpp    # Syndrome extraction
│       ├── qec_stabilizer.cpp  # Stabilizer measurements
│       ├── qec_logical.cpp     # Logical qubit operations
│       └── qec_distributed.cpp # Distributed QEC (MPI)
├── include/                    # C++ headers
├── python/qlret/               # Python package
│   ├── pennylane_device.py     # PennyLane integration
│   ├── jax_interface.py        # JAX autodiff integration
│   └── cirq_compare.py         # Cirq comparison utilities
├── tests/                      # Test binaries
├── scripts/                    # Python utility scripts
│   ├── noise_calibration.py    # Noise model calibration
│   └── ml_decoder_train.py     # ML decoder training
├── samples/                    # JSON configurations
├── build/                      # Compiled binaries
│   ├── quantum_sim             # Main executable
│   ├── test_simple             # Basic tests
│   ├── test_fidelity           # Fidelity tests
│   ├── test_autodiff           # Autodiff tests
│   ├── test_checkpoint         # Checkpoint tests
│   └── test_qec_*              # QEC tests
└── docs/                       # Documentation
```

---

## 🚀 Capabilities by Domain

### 1. EXECUTE - Quantum Simulation

**Triggers:** "run", "simulate", "benchmark", "test", "execute"

#### LRET Backend Commands

```bash
# Basic simulation
./build/quantum_sim samples/basic_gates.json

# With parameters
./build/quantum_sim -n 10 -d 20 --mode hybrid --threshold 1e-4

# Full options
./build/quantum_sim \
  -n <qubits> \
  -d <depth> \
  --mode <state_vector|density_matrix|hybrid|mps> \
  --noise <noise_level> \
  --threshold <rank_threshold> \
  --backend <lret|fdm> \
  --format <json|csv> \
  --compare-fdm \
  --gpu \
  -o <output_file>
```

#### Cirq Backend

```python
# Random circuit
python scripts/cirq_benchmark.py --qubits 10 --circuit random --depth 20

# VQE simulation
python scripts/cirq_benchmark.py --qubits 8 --circuit vqe --optimizer COBYLA

# With noise
python scripts/cirq_benchmark.py --qubits 10 --noise depolarizing --noise-strength 0.01
```

#### Backend Selection Logic

| Condition            | Backend | Reason                          |
| -------------------- | ------- | ------------------------------- |
| Qubits > 18          | LRET    | Rank truncation scales better   |
| VQE/QAOA             | Cirq    | Optimized variational circuits  |
| Hardware execution   | Cirq    | Google quantum processor access |
| Noise simulation     | LRET    | FDM density matrix support      |
| Comparison requested | Both    | Side-by-side analysis           |

---

### 2. QEC - Quantum Error Correction

**Triggers:** "qec", "error correction", "stabilizer", "syndrome", "decoder", "surface code"

#### Available Tests

```bash
# Surface code with specific distance
./build/test_qec_surface --distance 5 --rounds 10

# Decoder comparison
./build/test_qec_decoder --decoder mwpm
./build/test_qec_decoder --decoder union-find

# Syndrome extraction
./build/test_qec_syndrome

# Stabilizer measurements
./build/test_qec_stabilizer

# Full QEC cycle
./build/test_qec_full --distance 3 --noise 0.01 --rounds 5
```

#### QEC Modules

- **Adaptive QEC** (`qec_adaptive.cpp`): ML-driven decoder selection
- **Decoders** (`qec_decoder.cpp`): MWPM, Union-Find implementations
- **Syndrome** (`qec_syndrome.cpp`): Efficient syndrome extraction
- **Stabilizer** (`qec_stabilizer.cpp`): Stabilizer group operations
- **Logical** (`qec_logical.cpp`): Logical qubit encoding/operations
- **Distributed** (`qec_distributed.cpp`): MPI-based distributed QEC

---

### 3. VQE/QAOA - Variational Algorithms

**Triggers:** "vqe", "qaoa", "variational", "optimize", "hamiltonian", "maxcut"

#### VQE (Variational Quantum Eigensolver)

```python
# H2 molecule ground state
from agent.optimization.vqe import VQEOptimizer

vqe = VQEOptimizer(executor, optimizer="COBYLA")
result = vqe.optimize(hamiltonian, n_qubits=4, n_layers=3)
print(f"Ground state energy: {result['optimal_value']}")
```

#### QAOA (Quantum Approximate Optimization)

```python
# MaxCut problem
from agent.optimization.qaoa import QAOAOptimizer

qaoa = QAOAOptimizer(executor)
result = qaoa.solve_maxcut(
    graph=[(0,1), (1,2), (2,3), (3,0)],
    n_layers=2
)
print(f"Best cut: {result['best_cut_value']}")
```

#### Supported Optimizers

- COBYLA (default for QAOA)
- BFGS (default for VQE)
- Nelder-Mead
- Powell
- L-BFGS-B

---

### 4. BUILD - Code Development

**Triggers:** "edit", "fix", "implement", "create", "modify", "refactor"

#### Safety Rules

1. **Always create backup** before modifying files
2. **Show diff** before applying changes
3. **Require explicit approval** for all writes
4. **Run tests** after modifications
5. **Rollback** if build/tests fail

#### Workflow

```
1. Read current file content
2. Generate proposed changes
3. Show unified diff to user
4. Wait for approval (y/n/always)
5. Create backup (.bak file)
6. Apply changes
7. Run related tests
8. Rollback if tests fail
```

#### Backup Location

```
~/.lret_backups/<filename>.<timestamp>.bak
```

---

### 5. VISUALIZE - Results & Charts

**Triggers:** "plot", "chart", "visualize", "graph", "show results", "histogram"

#### Available Visualizations

```python
from agent.visualization import ResultsVisualizer

viz = ResultsVisualizer(output_dir="./outputs/plots")

# Measurement histogram
viz.plot_histogram(counts, title="10-Qubit GHZ", filename="ghz.png")

# Fidelity over time
viz.plot_fidelity_over_time(results, filename="fidelity.png")

# Scaling analysis
viz.plot_scaling_analysis(results, metric="simulation_time_seconds")

# Backend comparison
viz.plot_backend_comparison(comparison_data)

# Noise impact
viz.plot_noise_impact(noise_sweep_results)

# Complete report
viz.create_summary_report(experiment, filename="report.png")
```

#### ASCII Charts (Terminal)

```python
from agent.visualization import ASCIICharts

# Histogram
print(ASCIICharts.histogram(counts, width=50, top_n=16))

# Progress bar
print(ASCIICharts.progress_bar(75, 100, prefix="Progress: "))

# Data table
print(ASCIICharts.table(data, columns=["run_id", "qubits", "fidelity"]))

# Sparkline
print(f"Trend: {ASCIICharts.sparkline(values)}")
```

---

### 6. SESSION - Batch Management

**Triggers:** "batch", "queue", "schedule", "session", "campaign", "sweep"

#### Parameter Sweeps

```python
from agent.batch import ParameterSweep, BatchExecutor

# Grid sweep (all combinations)
configs = ParameterSweep.grid_sweep(
    base_config={"mode": "hybrid"},
    sweep_params={"n_qubits": [8, 10, 12], "depth": [20, 30]}
)
# → 6 configurations

# Scaling sweep
configs = ParameterSweep.scaling_sweep(
    base_config={"mode": "hybrid"},
    qubit_range=[8, 10, 12, 14, 16],
    depth_scaling="linear"  # depth = 2 * qubits
)

# Noise sweep
configs = ParameterSweep.noise_sweep(
    base_config={"n_qubits": 10},
    noise_levels=[0.0, 0.001, 0.01, 0.05],
    noise_types=["depolarizing", "amplitude_damping"]
)

# Backend comparison sweep
configs = ParameterSweep.backend_comparison_sweep(
    base_config={"n_qubits": 10},
    backends=["lret", "cirq"]
)
```

#### Batch Execution

```python
executor = BatchExecutor(runner, max_workers=4)

# Create and run batch
batch = executor.create_batch(configs, parallel_mode="threaded")
results = executor.run_batch(batch.batch_id, progress_callback=print_progress)

# Parallel modes: "sequential", "threaded", "process"
```

---

### 7. CLOUD - Cloud Integration

**Triggers:** "deploy", "cloud", "aws", "gcp", "kubernetes", "scale"

#### AWS EC2

```python
from agent.cloud import AWSRunner, AWSConfig

config = AWSConfig(
    region="us-east-1",
    instance_type="c5.4xlarge",
    use_spot=True,
    spot_max_price=0.5
)

runner = AWSRunner(config)
instance_id = runner.launch_instance(experiment_config)
runner.wait_for_completion(instance_id)
results = runner.get_results(run_id)
```

#### GCP Compute Engine

```python
from agent.cloud import GCPRunner, GCPConfig

config = GCPConfig(
    project_id="lret-quantum",
    zone="us-central1-a",
    machine_type="n2-standard-8",
    use_preemptible=True
)

runner = GCPRunner(config)
instance_name = runner.launch_instance(experiment_config)
```

#### Kubernetes

```python
from agent.cloud import KubernetesRunner, K8sConfig

config = K8sConfig(
    namespace="lret",
    image="lret/agent:latest",
    cpu_limit="4",
    memory_limit="8Gi"
)

runner = KubernetesRunner(config)
job_name = runner.submit_job(experiment_config)
status = runner.get_job_status(job_name)
```

---

### 8. ANALYZE - Code Inspection

**Triggers:** "explain", "analyze", "inspect", "what is", "how does", "debug"

#### Analysis Capabilities

- **Explain code**: Describe how a function/module works
- **Trace execution**: Follow code flow for a specific operation
- **Debug issues**: Investigate errors and suggest fixes
- **Compare implementations**: Analyze differences between backends
- **Performance analysis**: Identify bottlenecks

#### Key Files for Analysis

| Component      | File                    | Purpose                  |
| -------------- | ----------------------- | ------------------------ |
| Main simulator | `src/quantum_sim.cpp`   | Entry point, CLI parsing |
| FDM backend    | `src/fdm_simulator.cpp` | Density matrix evolution |
| SIMD kernels   | `src/simd_kernels.cpp`  | Vectorized operations    |
| GPU backend    | `src/gpu_simulator.cu`  | CUDA acceleration        |
| Checkpoints    | `src/checkpoint.cpp`    | State serialization      |
| Autodiff       | `src/autodiff.cpp`      | Gradient computation     |

#### LLM-Powered Result Explanation

When you ask "explain these results", the agent uses LLM to provide:

```python
# Example explanation output
"""
📊 Simulation Results Explanation

Your 10-qubit simulation with depth 20 completed successfully.

Key Findings:
• Fidelity: 0.995 - This is excellent! Your simulation achieved 99.5%
  overlap with the ideal quantum state, indicating minimal errors.

• Simulation Time: 1.23 seconds - This is faster than expected for
  10 qubits. The hybrid mode efficiently managed the state space.

• Final Rank: 32 - The density matrix was truncated to rank 32
  (max possible: 1024). This 97% compression shows LRET's efficiency.

• Memory Usage: 45 MB - Well within normal range for this qubit count.

Recommendations:
1. Your fidelity is high - you could try increasing depth to 30
2. Consider adding 1% noise to test error resilience
3. For comparison, run the same circuit on Cirq backend
"""
```

---

### 9. COLLABORATE - Sharing & Export

**Triggers:** "share", "export", "team", "collaborate", "package"

#### Export Experiment

```python
from agent.collaboration import ExperimentSharing

sharing = ExperimentSharing()

# Export as shareable package
package_path = sharing.export_experiment(
    experiment=result.to_dict(),
    title="10-Qubit GHZ Analysis",
    description="Comparison of LRET vs Cirq for GHZ state preparation",
    author="researcher@lab.edu",
    include_code=True
)
# Creates: ~/.lret/shared/exp_20240115_abc123.lret
```

#### Team Workspace

```python
from agent.collaboration import TeamWorkspace

workspace = TeamWorkspace("quantum-research-team")

# Add members
workspace.add_member("alice", "Alice", "alice@lab.edu", role="admin")
workspace.add_member("bob", "Bob", "bob@lab.edu", role="researcher")

# Share experiment
workspace.share_experiment("exp_001", experiment, author_id="alice")

# Comments & collaboration
workspace.add_comment("exp_001", "bob", "Great results! Can you try with more qubits?")

# Leaderboard
leaderboard = workspace.get_leaderboard(metric="fidelity")
```

---

## 🔧 Common Commands Reference

### Build Commands

```bash
# Full build
cd build && cmake .. && make -j$(nproc)

# Debug build
cmake .. -DCMAKE_BUILD_TYPE=Debug && make -j$(nproc)

# With MPI
cmake .. -DWITH_MPI=ON && make -j$(nproc)

# With CUDA
cmake .. -DWITH_CUDA=ON && make -j$(nproc)
```

### Test Commands

```bash
# All tests
cd build && ctest --output-on-failure

# Specific test
./test_simple
./test_fidelity
./test_autodiff
./test_checkpoint
./test_qec_surface

# Python tests
cd python && pytest tests/ -v
```

### Simulation Commands

```bash
# Basic run
./build/quantum_sim samples/basic_gates.json

# With parameters
./build/quantum_sim -n 10 -d 20 --mode hybrid

# Compare to FDM (validation)
./build/quantum_sim -n 8 -d 15 --compare-fdm

# Save output
./build/quantum_sim -n 10 -d 20 -o results.json --format json
```

---

## 💡 Example Prompts → Auto-Routing

| Your Prompt                          | Detected Intent | Backend | Action                                  |
| ------------------------------------ | --------------- | ------- | --------------------------------------- |
| "Run a 10-qubit benchmark"           | EXECUTE         | LRET    | `./build/quantum_sim -n 10 -d 20`       |
| "Compare LRET vs Cirq for 12 qubits" | COMPARE         | Both    | Run both, show comparison               |
| "Run VQE for H2 molecule"            | VQE             | Cirq    | `VQEOptimizer` with H2 hamiltonian      |
| "Test surface code distance 5"       | QEC             | LRET    | `./build/test_qec_surface --distance 5` |
| "Fix the bug in qec_decoder.cpp"     | BUILD           | -       | Show diff, require approval             |
| "Explain how FDM simulator works"    | ANALYZE         | -       | Read and explain code                   |
| "Plot the fidelity results"          | VISUALIZE       | -       | Generate matplotlib chart               |
| "Run batch with qubits 8,10,12"      | SESSION         | LRET    | `ParameterSweep` + `BatchExecutor`      |
| "Deploy to AWS with 20 qubits"       | CLOUD           | AWS     | Launch EC2 spot instance                |
| "Export results for the team"        | COLLABORATE     | -       | Create `.lret` package                  |

---

## 📝 Memory & Context

### Session Memory

- Conversation history (last 50 messages)
- Previous experiment results (indexed by run_id)
- User preferences (backend, output format)
- Current working session

### Cross-Session Persistence

```
~/.lret/
├── sessions/           # Session state
├── experiments/        # Experiment results
├── memory/             # Conversation cache
├── shared/             # Shared packages
└── logs/               # Audit logs
```

### Context References

- "Run the same experiment again" → Retrieve last config
- "Increase qubits from the last run" → Modify previous config
- "Compare with yesterday's results" → Load from session storage

---

## ⚙️ Configuration

### OpenCode Integration (`opencode.json`)

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "anthropic/claude-sonnet-4-20250514",
  "instructions": ["master_agent.md", "AGENTS.md", "README.md"],
  "agent": {
    "default": {
      "mode": "primary",
      "description": "Unified LRET agent with automatic task routing",
      "tools": {
        "write": true,
        "edit": true,
        "bash": true
      }
    }
  }
}
```

### Environment Variables

```bash
export ANTHROPIC_API_KEY="sk-ant-..."      # For Claude
export OPENAI_API_KEY="sk-..."             # For GPT (optional)
export GOOGLE_API_KEY="..."                # For Gemini (optional)
export LRET_DOCKER_IMAGE="ajs911/lret777:latest"
```

---

## 🆘 Troubleshooting

### Common Issues

| Issue                   | Solution                                      |
| ----------------------- | --------------------------------------------- |
| "quantum_sim not found" | Run `cd build && cmake .. && make -j$(nproc)` |
| "Cirq not installed"    | Run `pip install cirq qsimcirq`               |
| "Out of memory"         | Use `--mode mps` or reduce qubits             |
| "Simulation timeout"    | Increase `--timeout` or use smaller circuit   |
| "Permission denied"     | Check file permissions, use `chmod +x`        |

### Debug Mode

```bash
# Verbose output
./build/quantum_sim -n 10 -d 20 --verbose

# Dry run (plan only)
lret-agent --dry-run "run 10 qubit simulation"
```

---

## 📊 Metrics & Validation

### Key Metrics

- **Fidelity**: Overlap with ideal state (0.0 - 1.0)
- **Simulation Time**: Wall clock time in seconds
- **Final Rank**: Rank of truncated density matrix
- **Memory Usage**: Peak memory in MB
- **Speedup**: Ratio vs reference implementation

### Validation

- All results validated against schema before reporting
- Fidelity must be in [0, 1]
- Simulation time must be positive
- Failed runs clearly marked with error details

---

## 📚 Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                    LRET AI Agent                            │
├─────────────────────────────────────────────────────────────┤
│ EXECUTE:  run, simulate, benchmark, test                    │
│ BUILD:    edit, fix, implement, create (requires approval)  │
│ ANALYZE:  explain, analyze, what is, how does               │
│ VISUALIZE: plot, chart, graph, show results                 │
│ QEC:      error correction, stabilizer, syndrome            │
│ VQE/QAOA: variational, optimize, hamiltonian                │
│ SESSION:  batch, sweep, queue, schedule                     │
│ CLOUD:    deploy, aws, gcp, kubernetes                      │
│ COLLABORATE: share, export, team                            │
│ COMPARE:  compare, vs, both backends                        │
├─────────────────────────────────────────────────────────────┤
│ Backend Selection:                                          │
│   • Qubits > 18 → LRET                                      │
│   • VQE/QAOA → Cirq                                         │
│   • Hardware → Cirq                                         │
│   • Noise sim → LRET                                        │
│   • "compare" → Both                                        │
├─────────────────────────────────────────────────────────────┤
│ Safety:                                                     │
│   ✅ READ: No confirmation                                  │
│   ⚠️ RUN: Ask if > 5 min                                    │
│   🔴 WRITE: Always confirm                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🌐 REST API (Remote Access)

**Triggers:** "api", "server", "remote", "endpoint", "http"

### Start API Server

```bash
# Using uvicorn
uvicorn agent.api.app:app --host 0.0.0.0 --port 8000 --reload

# Or using CLI
lret-agent serve --port 8000
```

### API Endpoints

| Method | Endpoint                | Description            |
| ------ | ----------------------- | ---------------------- |
| GET    | `/`                     | API information        |
| GET    | `/health`               | Health check           |
| POST   | `/experiments`          | Submit experiment      |
| GET    | `/experiments/{run_id}` | Get experiment result  |
| GET    | `/experiments`          | List all experiments   |
| POST   | `/batches`              | Submit batch job       |
| GET    | `/batches/{batch_id}`   | Get batch status       |
| POST   | `/query`                | Natural language query |

### Example API Calls

```bash
# Submit experiment
curl -X POST http://localhost:8000/experiments \
  -H "Content-Type: application/json" \
  -d '{"n_qubits": 10, "depth": 20, "backend": "lret"}'

# Get results
curl http://localhost:8000/experiments/run_20240115_abc123

# Submit batch
curl -X POST http://localhost:8000/batches \
  -H "Content-Type: application/json" \
  -d '{
    "experiments": [
      {"n_qubits": 8, "depth": 20},
      {"n_qubits": 10, "depth": 20}
    ],
    "parallel_mode": "threaded"
  }'

# Natural language query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Run a 10 qubit GHZ simulation"}'
```

### Python API Client

```python
from agent.api.client import LRETClient, APIConfig

config = APIConfig(base_url="http://localhost:8000", timeout=60)
client = LRETClient(config)

# Run experiment and wait
result = client.run_experiment(n_qubits=10, depth=20)
print(f"Fidelity: {result['metrics']['fidelity']}")

# Run batch
batch_result = client.run_batch([
    {"n_qubits": 8, "depth": 20},
    {"n_qubits": 10, "depth": 20},
])
```

---

## 🤖 ML Optimization

**Triggers:** "optimize", "tune", "automl", "hyperparameter", "search", "genetic"

### Parameter Optimization Strategies

```python
from agent.ml import ParameterOptimizer

# Define objective function
def objective(params):
    result = run_simulation(params)
    return result['fidelity']

# Define search space
param_space = {
    "n_qubits": (8, 20),           # Range
    "depth": (10, 50),             # Range
    "noise_level": (0.0, 0.1),     # Range
    "mode": ["hybrid", "mps"]      # Discrete choices
}

optimizer = ParameterOptimizer(objective, param_space, maximize=True)

# Random search
result = optimizer.random_search(n_iterations=50)

# Grid search
result = optimizer.grid_search(grid_points=5)

# Bayesian optimization (most efficient)
result = optimizer.bayesian_optimization(n_iterations=30, n_initial=10)

print(f"Best params: {result.best_params}")
print(f"Best score: {result.best_score}")
```

### Circuit AutoML (Genetic Algorithm)

```python
from agent.ml import CircuitAutoML

def fitness_fn(circuit_template):
    # Evaluate circuit quality
    return simulate_and_score(circuit_template)

automl = CircuitAutoML(
    n_qubits=10,
    target_depth=20,
    fitness_fn=fitness_fn
)

# Genetic algorithm evolution
best_circuit = automl.evolve(
    population_size=20,
    generations=50,
    mutation_rate=0.3
)

# Neural architecture search
best_circuit = automl.search_architecture(search_budget=100)

print(f"Best circuit: {best_circuit.layers}")
```

### Optimization Strategies

| Strategy              | Best For              | Iterations         |
| --------------------- | --------------------- | ------------------ |
| Random Search         | Initial exploration   | 50-100             |
| Grid Search           | Small param spaces    | Depends on grid    |
| Bayesian Optimization | Expensive evaluations | 20-50              |
| Genetic Algorithm     | Circuit architecture  | 50-100 generations |

---

## 🔄 Experiment Tracking & Lineage

**Triggers:** "track", "lineage", "history", "parent", "compare experiments"

### Record Experiments

```python
from agent.session import ExperimentTracker

tracker = ExperimentTracker(session_id="research_001", storage_path=Path("~/.lret"))

# Record with lineage
tracker.record_experiment(
    run_id="exp_002",
    config={"n_qubits": 12, "depth": 30},
    result={"fidelity": 0.995},
    parent_id="exp_001"  # Links to previous experiment
)

# Get lineage chain
lineage = tracker.get_lineage("exp_002")
# Returns: [exp_002, exp_001, ...]

# Tag experiments
tracker.tag_experiment("exp_002", ["production", "noise-study"])

# Search by tag
results = tracker.search_by_tag("noise-study")

# Compare experiments
comparison = tracker.compare_experiments(["exp_001", "exp_002", "exp_003"])
print(comparison["result_comparison"])  # Shows metric improvements
```

### Experiment Comparison Output

```json
{
  "experiments": ["exp_001", "exp_002", "exp_003"],
  "config_diff": {
    "n_qubits": [10, 12, 14],
    "depth": [20, 20, 20]
  },
  "result_comparison": {
    "fidelity": {
      "values": [0.99, 0.995, 0.992],
      "min": 0.99,
      "max": 0.995,
      "improvement": "0.5%"
    },
    "simulation_time_seconds": {
      "values": [1.2, 2.1, 3.5],
      "min": 1.2,
      "max": 3.5
    }
  }
}
```

---

## 🔙 Rollback & Recovery System

### Automatic Backups

All code modifications create automatic backups:

```
~/.lret_backups/
├── quantum_sim.cpp.20240115_143022.bak
├── qec_decoder.cpp.20240115_150315.bak
└── ...
```

### Rollback Commands

```python
from agent.code import CodeEditor

editor = CodeEditor(repo_root=Path("."))

# View pending changes
for change in editor.pending_changes:
    print(f"ID: {change['id']}, File: {change['file_path']}")
    print(f"Diff:\n{change['diff']}")

# Rollback specific change
success, message = editor.rollback_change(change_id=0)
print(message)

# Manual restore from backup
import shutil
shutil.copy("~/.lret_backups/file.cpp.timestamp.bak", "src/file.cpp")
```

### Recovery Workflow

```
1. Change fails or causes issues
2. Agent automatically detects failure
3. Rolls back to backup
4. Reports what was reverted
5. Suggests alternative approach
```

---

## 🔌 Multi-LLM Provider Support

### Supported Providers

| Provider         | Model                    | API Key Env Variable |
| ---------------- | ------------------------ | -------------------- |
| Anthropic Claude | claude-sonnet-4-20250514 | `ANTHROPIC_API_KEY`  |
| OpenAI GPT       | gpt-4-turbo              | `OPENAI_API_KEY`     |
| Google Gemini    | gemini-pro               | `GOOGLE_API_KEY`     |

### Configuration

```python
from agent.config import AgentConfig

config = AgentConfig()

# Set provider
config.set("llm.provider", "claude")  # or "gpt", "gemini"
config.set("llm.model", "claude-sonnet-4-20250514")
config.set("llm.temperature", 0.2)
```

### LLM-Powered Features

- **Intent Classification**: Understands natural language queries
- **Plan Generation**: Creates step-by-step execution plans
- **Result Explanation**: Explains simulation results in plain language
- **Code Analysis**: Understands and explains codebase
- **Error Diagnosis**: Suggests fixes for failures

---

## 🔢 Supported Circuit Types

### Cirq Circuit Types

| Type     | Trigger              | Description               |
| -------- | -------------------- | ------------------------- |
| `random` | "random circuit"     | Random gate sequence      |
| `ghz`    | "ghz state"          | GHZ entangled state       |
| `qft`    | "qft", "fourier"     | Quantum Fourier Transform |
| `grover` | "grover", "search"   | Grover's search algorithm |
| `vqe`    | "vqe", "variational" | Variational ansatz        |
| `qaoa`   | "qaoa", "maxcut"     | QAOA optimization circuit |

### Example Commands

```bash
# GHZ state
python scripts/cirq_benchmark.py --qubits 10 --circuit ghz

# Quantum Fourier Transform
python scripts/cirq_benchmark.py --qubits 8 --circuit qft

# Grover's algorithm
python scripts/cirq_benchmark.py --qubits 6 --circuit grover

# Random circuit with depth
python scripts/cirq_benchmark.py --qubits 10 --circuit random --depth 30
```

---

## 🛠️ Installation & Setup Guide

### Prerequisites

- Python 3.9+
- CMake 3.16+
- C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- Eigen3 library

### Quick Install

**Option 1: Docker (Recommended)**

```bash
docker pull ajs911/lret777:latest
docker run -it ajs911/lret777:latest quantum_sim --help
```

**Option 2: Manual Build**

```bash
# Clone repository
git clone https://github.com/kunal5556/LRET.git
cd LRET
git checkout multiple-agent_md

# Install dependencies (Ubuntu)
sudo apt-get install cmake g++ libeigen3-dev libhdf5-dev python3-pip

# Install dependencies (macOS)
brew install cmake eigen hdf5

# Build
mkdir build && cd build
cmake ..
make -j$(nproc)

# Verify
./quantum_sim --help
```

**Option 3: Python Package**

```bash
pip install lret cirq numpy scipy matplotlib

# For full agent capabilities
pip install anthropic openai google-generativeai
pip install fastapi uvicorn  # For API server
```

### Verify Installation

```bash
# C++ simulator
./build/quantum_sim -n 8 -d 10 --mode hybrid

# Python
python -c "import lret; print(lret.__version__)"
python -c "import cirq; print(cirq.__version__)"

# Run tests
cd build && ctest --output-on-failure
```

### OpenCode AI Setup

```bash
# Install OpenCode
# Windows:
irm https://opencode.ai/install.ps1 | iex

# Mac/Linux:
curl -fsSL https://opencode.ai/install | bash

# Set API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Start with agent
cd LRET
opencode
```

---

## 📋 Complete Feature Matrix

| Phase | Feature                        | Status | Domain        |
| ----- | ------------------------------ | ------ | ------------- |
| 1     | Configuration Management       | ✅     | Foundation    |
| 1     | CLI Interface                  | ✅     | Foundation    |
| 1     | Structured Logging             | ✅     | Foundation    |
| 2     | Intent Classification          | ✅     | NLP           |
| 2     | Plan Generation                | ✅     | NLP           |
| 3     | LRET Execution                 | ✅     | Simulation    |
| 3     | Result Parsing                 | ✅     | Simulation    |
| 4     | Backend Routing                | ✅     | Simulation    |
| 4     | Result Validation              | ✅     | Simulation    |
| 5     | Cirq Integration               | ✅     | Backends      |
| 5     | Circuit Types (GHZ, QFT, etc.) | ✅     | Backends      |
| 6     | VQE Optimizer                  | ✅     | Algorithms    |
| 6     | QAOA Optimizer                 | ✅     | Algorithms    |
| 7     | Permission System              | ✅     | Safety        |
| 7     | Safety Validation              | ✅     | Safety        |
| 8     | Session Memory                 | ✅     | State         |
| 8     | Conversation Context           | ✅     | State         |
| 9     | Code Editing                   | ✅     | Development   |
| 9     | Rollback System                | ✅     | Development   |
| 10    | Test Runner                    | ✅     | Testing       |
| 11    | Session Management             | ✅     | Research      |
| 11    | Experiment Tracking            | ✅     | Research      |
| 12    | Batch Execution                | ✅     | Execution     |
| 12    | Parameter Sweeps               | ✅     | Execution     |
| 13    | REST API                       | ✅     | Integration   |
| 13    | API Client                     | ✅     | Integration   |
| 14    | Matplotlib Visualization       | ✅     | Visualization |
| 14    | ASCII Charts                   | ✅     | Visualization |
| 15    | AWS Integration                | ✅     | Cloud         |
| 15    | GCP Integration                | ✅     | Cloud         |
| 15    | Kubernetes Runner              | ✅     | Cloud         |
| 16    | Parameter Optimization         | ✅     | ML            |
| 16    | Circuit AutoML                 | ✅     | ML            |
| 17    | Experiment Sharing             | ✅     | Collaboration |
| 17    | Team Workspaces                | ✅     | Collaboration |

---

## 📚 Document References

For detailed implementation code, refer to the source agent files:

| Document    | Contents                                | Use When                               |
| ----------- | --------------------------------------- | -------------------------------------- |
| `agent1.md` | Setup, CLI, Getting Started             | Initial setup, common commands         |
| `agent2.md` | LLM, Intent, Planning (Phases 1-4)      | Understanding NLP components           |
| `agent3.md` | Backends, Safety, Memory (Phases 5-10)  | Backend details, safety rules          |
| `agent4.md` | Sessions, Batch, API (Phases 11-13)     | API development, batch jobs            |
| `agent5.md` | Visualization, Cloud, ML (Phases 14-16) | Charts, cloud deployment, optimization |
| `agent6.md` | Collaboration, User Guide (Phase 17)    | Sharing, team features                 |

---

## 📚 Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                    LRET AI Agent                            │
├─────────────────────────────────────────────────────────────┤
│ EXECUTE:  run, simulate, benchmark, test                    │
│ BUILD:    edit, fix, implement, create (requires approval)  │
│ ANALYZE:  explain, analyze, what is, how does               │
│ VISUALIZE: plot, chart, graph, show results                 │
│ QEC:      error correction, stabilizer, syndrome            │
│ VQE/QAOA: variational, optimize, hamiltonian                │
│ SESSION:  batch, sweep, queue, schedule                     │
│ CLOUD:    deploy, aws, gcp, kubernetes                      │
│ COLLABORATE: share, export, team                            │
│ COMPARE:  compare, vs, both backends                        │
│ API:      serve, endpoint, remote, http                     │
│ ML:       tune, automl, genetic, bayesian                   │
├─────────────────────────────────────────────────────────────┤
│ Backend Selection:                                          │
│   • Qubits > 18 → LRET                                      │
│   • VQE/QAOA → Cirq                                         │
│   • Hardware → Cirq                                         │
│   • Noise sim → LRET                                        │
│   • "compare" → Both                                        │
├─────────────────────────────────────────────────────────────┤
│ Safety:                                                     │
│   ✅ READ: No confirmation                                  │
│   ⚠️ RUN: Ask if > 5 min                                    │
│   🔴 WRITE: Always confirm                                  │
├─────────────────────────────────────────────────────────────┤
│ API: http://localhost:8000                                  │
│   POST /experiments - Submit job                            │
│   GET  /experiments/{id} - Get result                       │
│   POST /batches - Submit batch                              │
│   POST /query - NL query                                    │
└─────────────────────────────────────────────────────────────┘
```

---

_Just describe what you want to do - the agent handles the rest!_
