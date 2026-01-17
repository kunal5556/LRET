# LRET AI Agent - Part 6: Collaboration & User Guide (Phase 17 + Documentation)

_Previous: [agent5.md](agent5.md) - Visualization, Cloud & ML_  
_Return to: [agent1.md](agent1.md) - Quick Reference & Getting Started_

---

## 📚 PHASE 17: Collaboration & Sharing

### Overview

Enable team collaboration, experiment sharing, and reproducible research workflows.

### 17.1 Experiment Sharing

**File: `agent/collaboration/sharing.py`**

```python
"""Experiment sharing and collaboration."""
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import json
import hashlib
import zipfile

@dataclass
class SharedExperiment:
    """Shareable experiment package."""
    experiment_id: str
    title: str
    description: str
    author: str
    created_at: str
    config: Dict[str, Any]
    results: Dict[str, Any]
    code_snapshot: Optional[str] = None
    requirements: List[str] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.requirements is None:
            self.requirements = []
        if self.tags is None:
            self.tags = []

class ExperimentSharing:
    """Share and import experiments."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / ".lret" / "shared"
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def export_experiment(self, experiment: Dict[str, Any],
                         title: str, description: str,
                         author: str, output_path: Optional[Path] = None,
                         include_code: bool = True) -> Path:
        """Export experiment as shareable package."""
        # Generate unique ID
        content_hash = hashlib.md5(
            json.dumps(experiment, sort_keys=True).encode()
        ).hexdigest()[:8]
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d')}_{content_hash}"

        # Create shared experiment
        shared = SharedExperiment(
            experiment_id=experiment_id,
            title=title,
            description=description,
            author=author,
            created_at=datetime.utcnow().isoformat(),
            config=experiment.get("config", {}),
            results=experiment.get("results", {}),
            tags=experiment.get("tags", [])
        )

        # Detect requirements
        shared.requirements = self._detect_requirements()

        # Create output path
        if output_path is None:
            output_path = self.storage_path / f"{experiment_id}.lret"

        # Package as zip
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Main metadata
            zf.writestr("metadata.json", json.dumps(asdict(shared), indent=2))

            # Full results
            if "full_results" in experiment:
                zf.writestr("results.json", json.dumps(experiment["full_results"], indent=2))

            # Code snapshot
            if include_code:
                self._add_code_snapshot(zf)

        return output_path

    def import_experiment(self, package_path: Path) -> SharedExperiment:
        """Import shared experiment package."""
        with zipfile.ZipFile(package_path, 'r') as zf:
            metadata = json.loads(zf.read("metadata.json"))

            shared = SharedExperiment(**metadata)

            # Load full results if available
            if "results.json" in zf.namelist():
                shared.results = json.loads(zf.read("results.json"))

        return shared

    def _detect_requirements(self) -> List[str]:
        """Detect Python requirements."""
        try:
            import pkg_resources
            installed = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

            # Core requirements
            core = ["lret", "cirq", "numpy", "scipy"]
            requirements = []

            for pkg in core:
                if pkg in installed:
                    requirements.append(f"{pkg}=={installed[pkg]}")
                else:
                    requirements.append(pkg)

            return requirements
        except:
            return ["lret", "cirq", "numpy"]

    def _add_code_snapshot(self, zf: zipfile.ZipFile):
        """Add code snapshot to package."""
        # Find agent module
        import agent
        agent_path = Path(agent.__file__).parent

        for py_file in agent_path.rglob("*.py"):
            rel_path = py_file.relative_to(agent_path.parent)
            zf.write(py_file, f"code/{rel_path}")

    def list_shared(self) -> List[Dict[str, Any]]:
        """List all shared experiments."""
        shared = []

        for package_path in self.storage_path.glob("*.lret"):
            try:
                exp = self.import_experiment(package_path)
                shared.append({
                    "path": str(package_path),
                    "id": exp.experiment_id,
                    "title": exp.title,
                    "author": exp.author,
                    "created_at": exp.created_at,
                    "tags": exp.tags
                })
            except:
                continue

        return sorted(shared, key=lambda x: x["created_at"], reverse=True)
```

### 17.2 Team Collaboration

**File: `agent/collaboration/team.py`**

```python
"""Team collaboration features."""
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import json

@dataclass
class TeamMember:
    """Team member info."""
    user_id: str
    name: str
    email: str
    role: str  # "admin", "researcher", "viewer"
    joined_at: str

@dataclass
class Comment:
    """Comment on experiment."""
    comment_id: str
    author: str
    content: str
    created_at: str
    experiment_id: str

class TeamWorkspace:
    """Collaborative team workspace."""

    def __init__(self, workspace_id: str, storage_path: Optional[Path] = None):
        self.workspace_id = workspace_id
        self.storage_path = storage_path or Path.home() / ".lret" / "workspaces" / workspace_id
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.members: Dict[str, TeamMember] = {}
        self.experiments: Dict[str, Dict[str, Any]] = {}
        self.comments: Dict[str, List[Comment]] = {}

        self._load_workspace()

    def add_member(self, user_id: str, name: str, email: str,
                  role: str = "researcher") -> TeamMember:
        """Add team member."""
        member = TeamMember(
            user_id=user_id,
            name=name,
            email=email,
            role=role,
            joined_at=datetime.utcnow().isoformat()
        )

        self.members[user_id] = member
        self._save_members()

        return member

    def share_experiment(self, experiment_id: str, experiment: Dict[str, Any],
                        author_id: str) -> bool:
        """Share experiment with team."""
        if author_id not in self.members:
            return False

        self.experiments[experiment_id] = {
            "experiment": experiment,
            "author": author_id,
            "shared_at": datetime.utcnow().isoformat(),
            "views": 0,
            "stars": []
        }

        self._save_experiments()
        return True

    def add_comment(self, experiment_id: str, author_id: str, content: str) -> Comment:
        """Add comment to experiment."""
        import uuid

        comment = Comment(
            comment_id=str(uuid.uuid4())[:8],
            author=author_id,
            content=content,
            created_at=datetime.utcnow().isoformat(),
            experiment_id=experiment_id
        )

        if experiment_id not in self.comments:
            self.comments[experiment_id] = []

        self.comments[experiment_id].append(comment)
        self._save_comments()

        return comment

    def get_comments(self, experiment_id: str) -> List[Comment]:
        """Get comments for experiment."""
        return self.comments.get(experiment_id, [])

    def star_experiment(self, experiment_id: str, user_id: str) -> bool:
        """Star/favorite an experiment."""
        if experiment_id not in self.experiments:
            return False

        stars = self.experiments[experiment_id]["stars"]
        if user_id not in stars:
            stars.append(user_id)
            self._save_experiments()

        return True

    def get_leaderboard(self, metric: str = "fidelity") -> List[Dict[str, Any]]:
        """Get experiment leaderboard."""
        results = []

        for exp_id, exp_data in self.experiments.items():
            exp = exp_data["experiment"]
            results.append({
                "experiment_id": exp_id,
                "author": exp_data["author"],
                "metric": exp.get("results", {}).get(metric, 0),
                "stars": len(exp_data["stars"])
            })

        return sorted(results, key=lambda x: x["metric"], reverse=True)

    def _save_members(self):
        """Save members to disk."""
        data = {uid: {
            "user_id": m.user_id,
            "name": m.name,
            "email": m.email,
            "role": m.role,
            "joined_at": m.joined_at
        } for uid, m in self.members.items()}

        with open(self.storage_path / "members.json", 'w') as f:
            json.dump(data, f, indent=2)

    def _save_experiments(self):
        """Save experiments to disk."""
        with open(self.storage_path / "experiments.json", 'w') as f:
            json.dump(self.experiments, f, indent=2)

    def _save_comments(self):
        """Save comments to disk."""
        data = {exp_id: [{
            "comment_id": c.comment_id,
            "author": c.author,
            "content": c.content,
            "created_at": c.created_at,
            "experiment_id": c.experiment_id
        } for c in comments] for exp_id, comments in self.comments.items()}

        with open(self.storage_path / "comments.json", 'w') as f:
            json.dump(data, f, indent=2)

    def _load_workspace(self):
        """Load workspace from disk."""
        # Load members
        members_file = self.storage_path / "members.json"
        if members_file.exists():
            with open(members_file) as f:
                data = json.load(f)
            self.members = {uid: TeamMember(**m) for uid, m in data.items()}

        # Load experiments
        exp_file = self.storage_path / "experiments.json"
        if exp_file.exists():
            with open(exp_file) as f:
                self.experiments = json.load(f)

        # Load comments
        comments_file = self.storage_path / "comments.json"
        if comments_file.exists():
            with open(comments_file) as f:
                data = json.load(f)
            self.comments = {exp_id: [Comment(**c) for c in comments]
                           for exp_id, comments in data.items()}
```

---

## 📖 COMPLETE USER GUIDE

### Getting Started

#### Prerequisites

```bash
# Python 3.9+
python --version

# Install LRET and dependencies
pip install lret cirq numpy scipy matplotlib

# Clone agent repository
git clone https://github.com/your-org/lret-agent.git
cd lret-agent
pip install -e .
```

#### Quick Start

```python
from agent import LRETAgent

# Initialize agent
agent = LRETAgent()

# Simple query
response = agent.query("Run a 10-qubit GHZ state simulation")
print(response.result)

# More specific request
response = agent.query("""
    Compare LRET and Cirq backends for:
    - 12 qubits
    - depth 30
    - with 1% depolarizing noise
""")

print(f"Best backend: {response.recommended_backend}")
print(f"Fidelity: {response.fidelity}")
```

### CLI Reference

```bash
# Basic experiment
lret-agent run --qubits 10 --depth 20

# With backend selection
lret-agent run --qubits 12 --backend lret --mode hybrid

# Backend comparison
lret-agent compare --qubits 10 --backends lret,cirq

# Batch execution
lret-agent batch --config experiments.yaml --parallel 4

# Parameter sweep
lret-agent sweep --qubits 8,10,12,14 --depths 20,30,40

# Start API server
lret-agent serve --port 8000

# Interactive mode
lret-agent interactive
```

### Configuration Files

**`lret_config.yaml`** - Main configuration:

```yaml
# LRET AI Agent Configuration

# LLM Settings
llm:
  provider: claude # claude, gpt, gemini
  model: claude-sonnet-4-20250514
  temperature: 0.1
  max_tokens: 4096

# Backend Preferences
backends:
  default: auto # auto, lret, cirq
  fallback: cirq
  comparison_enabled: true

# LRET Specific
lret:
  default_mode: hybrid
  rank_threshold: 1e-4
  max_bond_dimension: 256

# Cirq Specific
cirq:
  default_simulator: density_matrix
  noise_model: depolarizing

# Execution
execution:
  timeout_seconds: 3600
  max_retries: 3
  docker_enabled: true

# Output
output:
  format: json # json, csv, markdown
  save_histograms: true
  plot_results: true
```

**`experiments.yaml`** - Batch experiments:

```yaml
experiments:
  - name: "GHZ Scaling"
    type: ghz
    qubits: [8, 10, 12, 14, 16]
    backend: lret

  - name: "VQE H2"
    type: vqe
    molecule: H2
    optimizer: COBYLA

  - name: "Noise Analysis"
    type: random_circuit
    qubits: 10
    depth: 30
    noise_levels: [0, 0.001, 0.01, 0.05]
```

### Python API

#### Basic Usage

```python
from agent.core import LRETAgent
from agent.execution import ExperimentConfig

# Initialize
agent = LRETAgent()

# Run experiment
config = ExperimentConfig(
    n_qubits=10,
    depth=20,
    backend_preference="auto",
    mode="hybrid"
)

result = agent.run_experiment(config)

print(f"Run ID: {result.run_id}")
print(f"Time: {result.simulation_time_seconds:.2f}s")
print(f"Fidelity: {result.fidelity:.4f}")
```

#### Natural Language Queries

```python
# Simple queries
agent.query("Run a 10-qubit GHZ simulation")
agent.query("Compare backends for 12 qubits with noise")
agent.query("What's the best backend for 20 qubits?")

# Complex queries
agent.query("""
    I need to run a VQE simulation for H2 molecule.
    Use COBYLA optimizer with 1000 iterations.
    Compare results between LRET and Cirq backends.
    Plot the energy convergence.
""")
```

#### Batch Processing

```python
from agent.batch import BatchExecutor, ParameterSweep

# Create parameter sweep
configs = ParameterSweep.grid_sweep(
    base_config={"backend": "lret", "mode": "hybrid"},
    sweep_params={
        "n_qubits": [8, 10, 12],
        "depth": [20, 30, 40]
    }
)

# Run batch
executor = BatchExecutor(agent.experiment_runner, max_workers=4)
batch = executor.create_batch(configs, parallel_mode="threaded")
results = executor.run_batch(batch.batch_id)

# Analyze
for result in results.results:
    print(f"Qubits: {result['config']['n_qubits']}, "
          f"Time: {result['metrics']['simulation_time']:.2f}s")
```

#### Cloud Execution

```python
from agent.cloud import AWSRunner, AWSConfig

# Configure AWS
config = AWSConfig(
    region="us-east-1",
    instance_type="c5.4xlarge",
    use_spot=True
)

runner = AWSRunner(config)

# Launch experiment
instance_id = runner.launch_instance({
    "run_id": "cloud_exp_001",
    "n_qubits": 20,
    "depth": 50,
    "backend": "lret"
})

# Wait and get results
runner.wait_for_completion(instance_id)
results = runner.get_results("cloud_exp_001")
```

### Visualization

```python
from agent.visualization import ResultsVisualizer

viz = ResultsVisualizer()

# Plot histogram
viz.plot_histogram(
    counts=result.counts,
    title="10-Qubit GHZ State",
    filename="ghz_histogram.png"
)

# Scaling analysis
viz.plot_scaling_analysis(
    results=scaling_results,
    metric="simulation_time_seconds",
    filename="scaling.png"
)

# Summary report
viz.create_summary_report(
    experiment=result.to_dict(),
    filename="experiment_report.png"
)
```

### REST API

#### Start Server

```bash
lret-agent serve --port 8000
# or
uvicorn agent.api.app:app --reload --port 8000
```

#### Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Submit experiment
curl -X POST http://localhost:8000/experiments \
  -H "Content-Type: application/json" \
  -d '{"n_qubits": 10, "depth": 20, "backend": "lret"}'

# Get results
curl http://localhost:8000/experiments/exp_20240115_abc123

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

### Troubleshooting

#### Common Issues

**1. Import Errors**

```python
# Check installation
python -c "import lret; print(lret.__version__)"
python -c "import cirq; print(cirq.__version__)"

# Reinstall if needed
pip install --upgrade lret cirq
```

**2. Memory Issues**

```python
# For large simulations, use LRET with low-rank mode
config = ExperimentConfig(
    n_qubits=20,
    mode="mps",  # Matrix Product State - memory efficient
    max_bond_dimension=128
)
```

**3. Docker Issues**

```bash
# Check Docker is running
docker ps

# Pull LRET image
docker pull lret/quantum:latest

# Run manually
docker run -it lret/quantum python -c "import lret; print('OK')"
```

**4. API Connection Issues**

```python
from agent.api.client import LRETClient, APIConfig

# Configure with retry
config = APIConfig(
    base_url="http://localhost:8000",
    timeout=60,
    poll_interval=2.0
)

client = LRETClient(config)
```

### Best Practices

1. **Start Small**: Begin with 8-10 qubits to verify setup
2. **Use Auto Backend**: Let the agent choose the best backend
3. **Save Results**: Always save experiment results for reproducibility
4. **Monitor Resources**: Use resource monitor for large experiments
5. **Version Control**: Keep experiment configs in version control
6. **Document**: Add descriptions and tags to experiments

---

## 📊 Complete Phase Summary

| Phase | Title                | Status | Key Features                              |
| ----- | -------------------- | ------ | ----------------------------------------- |
| 1     | Foundation           | ✅     | Config, logging, CLI, LLM interface       |
| 2     | Intent Understanding | ✅     | Query classification, experiment planning |
| 3     | Experiment Execution | ✅     | Config management, Docker runner          |
| 4     | Backend Management   | ✅     | Routing, validation, monitoring           |
| 5     | Cirq Integration     | ✅     | Circuit execution, hardware support       |
| 6     | VQE/QAOA             | ✅     | Variational algorithms, optimizers        |
| 7     | Safety & Permissions | ✅     | Sandboxing, resource limits               |
| 8     | Memory & State       | ✅     | Caching, session persistence              |
| 9     | Code Editing         | ✅     | File operations, rollback                 |
| 10    | Testing              | ✅     | Unit tests, integration tests             |
| 11    | Session Management   | ✅     | Multi-session, experiment tracking        |
| 12    | Batch Execution      | ✅     | Parallel processing, sweeps               |
| 13    | Web API              | ✅     | REST API, client library                  |
| 14    | Visualization        | ✅     | Charts, reports, ASCII output             |
| 15    | Cloud Integration    | ✅     | AWS, GCP, Kubernetes                      |
| 16    | ML Optimization      | ✅     | Parameter tuning, AutoML                  |
| 17    | Collaboration        | ✅     | Sharing, team workspaces                  |

---

## 🔗 Document Index

| Document               | Contents                                              | Lines |
| ---------------------- | ----------------------------------------------------- | ----- |
| [agent1.md](agent1.md) | Quick Reference, Setup, Overview                      | ~300  |
| [agent2.md](agent2.md) | Phases 1-4: Foundation, Intent, Execution, Backend    | ~400  |
| [agent3.md](agent3.md) | Phases 5-10: Cirq, VQE, Safety, Memory, Code, Testing | ~500  |
| [agent4.md](agent4.md) | Phases 11-13: Session, Batch, API                     | ~450  |
| [agent5.md](agent5.md) | Phases 14-16: Visualization, Cloud, ML                | ~500  |
| [agent6.md](agent6.md) | Phase 17 + Complete User Guide                        | ~400  |

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- LRET quantum simulation team
- Google Cirq team
- Anthropic Claude for AI assistance

---

_End of LRET AI Agent Documentation_

_For questions or contributions, please open an issue or pull request._
