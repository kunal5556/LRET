# LRET AI Agent - Part 4: Session, Batch Execution & API (Phases 11-13)

_Previous: [agent3.md](agent3.md) - Backend & Execution Systems_  
_Next: [agent5.md](agent5.md) - Visualization, Cloud & ML_

---

## 📚 PHASE 11: Session Management

### Overview

Implement comprehensive session management for multi-day research campaigns with state persistence.

### 11.1 Session Manager

**File: `agent/session/manager.py`**

```python
"""Session management for research campaigns."""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json
import uuid
from dataclasses import dataclass, asdict

@dataclass
class SessionConfig:
    """Configuration for a session."""
    name: str
    description: str = ""
    timeout_hours: int = 24
    auto_save: bool = True
    persist_results: bool = True

@dataclass
class SessionState:
    """Current state of a session."""
    session_id: str
    config: SessionConfig
    created_at: str
    last_accessed: str
    experiment_count: int = 0
    status: str = "active"

class SessionManager:
    """Manage research sessions with persistence."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / ".lret" / "sessions"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.active_sessions: Dict[str, SessionState] = {}
        self._load_active_sessions()

    def create_session(self, config: SessionConfig) -> SessionState:
        """Create a new research session."""
        session_id = str(uuid.uuid4())[:8]
        now = datetime.utcnow().isoformat()

        state = SessionState(
            session_id=session_id,
            config=config,
            created_at=now,
            last_accessed=now
        )

        self.active_sessions[session_id] = state
        self._save_session(state)

        return state

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get session by ID."""
        if session_id in self.active_sessions:
            state = self.active_sessions[session_id]
            state.last_accessed = datetime.utcnow().isoformat()
            return state
        return self._load_session(session_id)

    def list_sessions(self, include_expired: bool = False) -> List[SessionState]:
        """List all sessions."""
        sessions = list(self.active_sessions.values())

        if not include_expired:
            sessions = [s for s in sessions if not self._is_expired(s)]

        return sorted(sessions, key=lambda s: s.last_accessed, reverse=True)

    def close_session(self, session_id: str) -> bool:
        """Close and archive a session."""
        state = self.get_session(session_id)
        if not state:
            return False

        state.status = "closed"
        self._save_session(state)

        if session_id in self.active_sessions:
            del self.active_sessions[session_id]

        return True

    def resume_session(self, session_id: str) -> Optional[SessionState]:
        """Resume a closed or expired session."""
        state = self._load_session(session_id)
        if not state:
            return None

        state.status = "active"
        state.last_accessed = datetime.utcnow().isoformat()
        self.active_sessions[session_id] = state
        self._save_session(state)

        return state

    def _is_expired(self, state: SessionState) -> bool:
        """Check if session has expired."""
        last_accessed = datetime.fromisoformat(state.last_accessed)
        timeout = timedelta(hours=state.config.timeout_hours)
        return datetime.utcnow() - last_accessed > timeout

    def _save_session(self, state: SessionState):
        """Save session to disk."""
        session_file = self.storage_path / f"{state.session_id}.json"
        with open(session_file, 'w') as f:
            json.dump(asdict(state), f, indent=2)

    def _load_session(self, session_id: str) -> Optional[SessionState]:
        """Load session from disk."""
        session_file = self.storage_path / f"{session_id}.json"
        if not session_file.exists():
            return None

        with open(session_file, 'r') as f:
            data = json.load(f)

        config = SessionConfig(**data.pop("config"))
        return SessionState(config=config, **data)

    def _load_active_sessions(self):
        """Load all active sessions from disk."""
        for session_file in self.storage_path.glob("*.json"):
            try:
                state = self._load_session(session_file.stem)
                if state and state.status == "active" and not self._is_expired(state):
                    self.active_sessions[state.session_id] = state
            except (json.JSONDecodeError, KeyError):
                continue
```

### 11.2 Experiment Tracker

**File: `agent/session/tracker.py`**

```python
"""Track experiments within a session."""
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json

class ExperimentTracker:
    """Track experiment lineage and comparisons."""

    def __init__(self, session_id: str, storage_path: Path):
        self.session_id = session_id
        self.storage_path = storage_path / session_id / "experiments"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.experiments: Dict[str, Dict[str, Any]] = {}
        self._load_experiments()

    def record_experiment(self, run_id: str, config: Dict[str, Any],
                         result: Dict[str, Any], parent_id: Optional[str] = None):
        """Record an experiment with optional lineage."""
        experiment = {
            "run_id": run_id,
            "config": config,
            "result": result,
            "parent_id": parent_id,
            "created_at": datetime.utcnow().isoformat(),
            "tags": [],
            "notes": ""
        }

        self.experiments[run_id] = experiment
        self._save_experiment(run_id, experiment)

    def get_experiment(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment by run ID."""
        return self.experiments.get(run_id)

    def get_lineage(self, run_id: str) -> List[Dict[str, Any]]:
        """Get experiment lineage (ancestors)."""
        lineage = []
        current_id = run_id

        while current_id:
            experiment = self.get_experiment(current_id)
            if not experiment:
                break
            lineage.append(experiment)
            current_id = experiment.get("parent_id")

        return lineage

    def compare_experiments(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple experiments."""
        experiments = [self.get_experiment(rid) for rid in run_ids]
        experiments = [e for e in experiments if e]

        if len(experiments) < 2:
            return {"error": "Need at least 2 experiments to compare"}

        comparison = {
            "experiments": run_ids,
            "config_diff": self._compare_configs([e["config"] for e in experiments]),
            "result_comparison": self._compare_results([e["result"] for e in experiments]),
        }

        return comparison

    def _compare_configs(self, configs: List[Dict]) -> Dict[str, Any]:
        """Find differences in configurations."""
        if not configs:
            return {}

        all_keys = set()
        for config in configs:
            all_keys.update(config.keys())

        diff = {}
        for key in all_keys:
            values = [c.get(key) for c in configs]
            if len(set(str(v) for v in values)) > 1:
                diff[key] = values

        return diff

    def _compare_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Compare result metrics."""
        metrics = ["simulation_time_seconds", "fidelity", "final_rank", "memory_used_mb"]
        comparison = {}

        for metric in metrics:
            values = [r.get(metric) for r in results if r.get(metric) is not None]
            if values:
                comparison[metric] = {
                    "values": values,
                    "min": min(values),
                    "max": max(values),
                    "improvement": (max(values) - min(values)) / min(values) * 100 if min(values) > 0 else 0
                }

        return comparison

    def tag_experiment(self, run_id: str, tags: List[str]):
        """Add tags to an experiment."""
        if run_id in self.experiments:
            self.experiments[run_id]["tags"] = tags
            self._save_experiment(run_id, self.experiments[run_id])

    def search_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """Find experiments by tag."""
        return [e for e in self.experiments.values() if tag in e.get("tags", [])]

    def _save_experiment(self, run_id: str, experiment: Dict):
        """Save experiment to disk."""
        exp_file = self.storage_path / f"{run_id}.json"
        with open(exp_file, 'w') as f:
            json.dump(experiment, f, indent=2)

    def _load_experiments(self):
        """Load all experiments from disk."""
        for exp_file in self.storage_path.glob("*.json"):
            try:
                with open(exp_file, 'r') as f:
                    experiment = json.load(f)
                self.experiments[experiment["run_id"]] = experiment
            except (json.JSONDecodeError, KeyError):
                continue
```

---

## 📚 PHASE 12: Batch Execution

### Overview

Implement parallel batch execution for running multiple experiments efficiently.

### 12.1 Batch Executor

**File: `agent/batch/executor.py`**

```python
"""Batch execution for multiple experiments."""
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import uuid
import time

@dataclass
class BatchJob:
    """A batch of experiments to run."""
    batch_id: str
    experiments: List[Dict[str, Any]]
    parallel_mode: str  # "sequential", "threaded", "process"
    max_workers: int
    status: str = "pending"
    progress: int = 0
    results: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.results is None:
            self.results = []

class BatchExecutor:
    """Execute batches of experiments in parallel."""

    def __init__(self, experiment_runner, max_workers: int = 4):
        self.experiment_runner = experiment_runner
        self.max_workers = max_workers
        self.active_jobs: Dict[str, BatchJob] = {}

    def create_batch(self, experiments: List[Dict[str, Any]],
                    parallel_mode: str = "threaded") -> BatchJob:
        """Create a new batch job."""
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"

        job = BatchJob(
            batch_id=batch_id,
            experiments=experiments,
            parallel_mode=parallel_mode,
            max_workers=self.max_workers
        )

        self.active_jobs[batch_id] = job
        return job

    def run_batch(self, batch_id: str, progress_callback: Callable = None) -> BatchJob:
        """Execute a batch job."""
        job = self.active_jobs.get(batch_id)
        if not job:
            raise ValueError(f"Batch {batch_id} not found")

        job.status = "running"

        if job.parallel_mode == "sequential":
            self._run_sequential(job, progress_callback)
        elif job.parallel_mode == "threaded":
            self._run_threaded(job, progress_callback)
        elif job.parallel_mode == "process":
            self._run_multiprocess(job, progress_callback)

        job.status = "completed"
        return job

    def _run_sequential(self, job: BatchJob, callback: Callable):
        """Run experiments sequentially."""
        for i, exp_config in enumerate(job.experiments):
            result = self._run_single(exp_config)
            job.results.append(result)
            job.progress = (i + 1) / len(job.experiments) * 100

            if callback:
                callback(job.progress, result)

    def _run_threaded(self, job: BatchJob, callback: Callable):
        """Run experiments using thread pool."""
        with ThreadPoolExecutor(max_workers=job.max_workers) as executor:
            futures = {
                executor.submit(self._run_single, exp): i
                for i, exp in enumerate(job.experiments)
            }

            completed = 0
            for future in as_completed(futures):
                result = future.result()
                job.results.append(result)
                completed += 1
                job.progress = completed / len(job.experiments) * 100

                if callback:
                    callback(job.progress, result)

    def _run_multiprocess(self, job: BatchJob, callback: Callable):
        """Run experiments using process pool."""
        with ProcessPoolExecutor(max_workers=job.max_workers) as executor:
            futures = {
                executor.submit(self._run_single, exp): i
                for i, exp in enumerate(job.experiments)
            }

            completed = 0
            for future in as_completed(futures):
                result = future.result()
                job.results.append(result)
                completed += 1
                job.progress = completed / len(job.experiments) * 100

                if callback:
                    callback(job.progress, result)

    def _run_single(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single experiment."""
        from agent.execution.runner import ExperimentConfig

        exp_config = ExperimentConfig(**config)
        result = self.experiment_runner.run_experiment(exp_config)

        return {
            "run_id": result.run_id,
            "status": result.status,
            "config": config,
            "metrics": {
                "simulation_time": result.simulation_time_seconds,
                "fidelity": result.fidelity,
                "final_rank": result.final_rank,
            }
        }

    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Get status of a batch job."""
        job = self.active_jobs.get(batch_id)
        if not job:
            return {"error": "Batch not found"}

        return {
            "batch_id": job.batch_id,
            "status": job.status,
            "progress": job.progress,
            "total_experiments": len(job.experiments),
            "completed": len(job.results),
            "parallel_mode": job.parallel_mode,
        }

    def cancel_batch(self, batch_id: str) -> bool:
        """Cancel a running batch."""
        job = self.active_jobs.get(batch_id)
        if job and job.status == "running":
            job.status = "cancelled"
            return True
        return False
```

### 12.2 Parameter Sweep

**File: `agent/batch/sweep.py`**

```python
"""Parameter sweep for systematic experiments."""
from typing import Dict, Any, List, Generator
from itertools import product

class ParameterSweep:
    """Generate parameter combinations for sweeps."""

    @staticmethod
    def grid_sweep(base_config: Dict[str, Any],
                   sweep_params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters (grid search).

        Example:
            base_config = {"mode": "hybrid"}
            sweep_params = {"n_qubits": [8, 10, 12], "depth": [20, 30]}

            Returns 6 configurations:
            - {mode: hybrid, n_qubits: 8, depth: 20}
            - {mode: hybrid, n_qubits: 8, depth: 30}
            - ... etc
        """
        configs = []

        param_names = list(sweep_params.keys())
        param_values = list(sweep_params.values())

        for combo in product(*param_values):
            config = base_config.copy()
            for name, value in zip(param_names, combo):
                config[name] = value
            configs.append(config)

        return configs

    @staticmethod
    def linear_sweep(base_config: Dict[str, Any],
                    param_name: str,
                    values: List[Any]) -> List[Dict[str, Any]]:
        """Sweep a single parameter."""
        configs = []
        for value in values:
            config = base_config.copy()
            config[param_name] = value
            configs.append(config)
        return configs

    @staticmethod
    def scaling_sweep(base_config: Dict[str, Any],
                     qubit_range: List[int],
                     depth_scaling: str = "linear") -> List[Dict[str, Any]]:
        """Generate configs for scaling analysis.

        Args:
            base_config: Base configuration
            qubit_range: List of qubit counts
            depth_scaling: "linear" (depth = qubits), "quadratic" (depth = qubits^2),
                          "constant" (use base depth)
        """
        configs = []
        base_depth = base_config.get("depth", 20)

        for n_qubits in qubit_range:
            config = base_config.copy()
            config["n_qubits"] = n_qubits

            if depth_scaling == "linear":
                config["depth"] = n_qubits * 2
            elif depth_scaling == "quadratic":
                config["depth"] = n_qubits ** 2
            else:
                config["depth"] = base_depth

            configs.append(config)

        return configs

    @staticmethod
    def noise_sweep(base_config: Dict[str, Any],
                   noise_levels: List[float] = None,
                   noise_types: List[str] = None) -> List[Dict[str, Any]]:
        """Sweep noise parameters."""
        if noise_levels is None:
            noise_levels = [0.0, 0.001, 0.01, 0.05, 0.1]
        if noise_types is None:
            noise_types = ["depolarizing"]

        configs = []
        for noise_type in noise_types:
            for noise_level in noise_levels:
                config = base_config.copy()
                config["noise_type"] = noise_type
                config["noise_level"] = noise_level
                configs.append(config)

        return configs

    @staticmethod
    def backend_comparison_sweep(base_config: Dict[str, Any],
                                 backends: List[str] = None) -> List[Dict[str, Any]]:
        """Generate configs for comparing backends."""
        if backends is None:
            backends = ["lret", "cirq"]

        configs = []
        for backend in backends:
            config = base_config.copy()
            config["backend_preference"] = backend
            configs.append(config)

        return configs
```

---

## 📚 PHASE 13: Web API

### Overview

Implement REST API for remote access and integration with other tools.

### 13.1 FastAPI Application

**File: `agent/api/app.py`**

```python
"""FastAPI application for LRET AI Agent."""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

app = FastAPI(
    title="LRET AI Agent API",
    description="REST API for quantum simulation with LRET and Cirq",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class ExperimentRequest(BaseModel):
    n_qubits: int
    depth: int = 20
    backend: str = "auto"
    noise_level: float = 0.001
    mode: str = "hybrid"
    rank_threshold: float = 1e-4
    compare_to_exact: bool = False

class BatchRequest(BaseModel):
    experiments: List[ExperimentRequest]
    parallel_mode: str = "threaded"
    max_workers: int = 4

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    dry_run: bool = False

class ExperimentResponse(BaseModel):
    run_id: str
    status: str
    config: Dict[str, Any]
    metrics: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

# In-memory storage (replace with database in production)
experiments_db: Dict[str, Dict[str, Any]] = {}
batches_db: Dict[str, Dict[str, Any]] = {}

# API Endpoints
@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "LRET AI Agent API",
        "version": "1.0.0",
        "endpoints": [
            "/experiments",
            "/batches",
            "/query",
            "/health"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/experiments", response_model=ExperimentResponse)
async def create_experiment(request: ExperimentRequest,
                           background_tasks: BackgroundTasks):
    """Submit a new experiment."""
    run_id = f"api_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"

    experiment = {
        "run_id": run_id,
        "status": "pending",
        "config": request.dict(),
        "created_at": datetime.utcnow().isoformat(),
        "metrics": None
    }

    experiments_db[run_id] = experiment

    # Run experiment in background
    background_tasks.add_task(run_experiment_task, run_id, request.dict())

    return ExperimentResponse(
        run_id=run_id,
        status="pending",
        config=request.dict(),
        message="Experiment submitted"
    )

@app.get("/experiments/{run_id}", response_model=ExperimentResponse)
async def get_experiment(run_id: str):
    """Get experiment status and results."""
    if run_id not in experiments_db:
        raise HTTPException(status_code=404, detail="Experiment not found")

    exp = experiments_db[run_id]
    return ExperimentResponse(
        run_id=exp["run_id"],
        status=exp["status"],
        config=exp["config"],
        metrics=exp.get("metrics")
    )

@app.get("/experiments")
async def list_experiments(limit: int = 10, status: str = None):
    """List recent experiments."""
    experiments = list(experiments_db.values())

    if status:
        experiments = [e for e in experiments if e["status"] == status]

    experiments.sort(key=lambda x: x["created_at"], reverse=True)
    return {"experiments": experiments[:limit], "total": len(experiments)}

@app.post("/batches")
async def create_batch(request: BatchRequest, background_tasks: BackgroundTasks):
    """Submit a batch of experiments."""
    batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"

    batch = {
        "batch_id": batch_id,
        "status": "pending",
        "experiments": [exp.dict() for exp in request.experiments],
        "parallel_mode": request.parallel_mode,
        "max_workers": request.max_workers,
        "progress": 0,
        "results": [],
        "created_at": datetime.utcnow().isoformat()
    }

    batches_db[batch_id] = batch

    # Run batch in background
    background_tasks.add_task(run_batch_task, batch_id)

    return {"batch_id": batch_id, "status": "submitted",
            "total_experiments": len(request.experiments)}

@app.get("/batches/{batch_id}")
async def get_batch(batch_id: str):
    """Get batch status and results."""
    if batch_id not in batches_db:
        raise HTTPException(status_code=404, detail="Batch not found")

    return batches_db[batch_id]

@app.post("/query")
async def process_query(request: QueryRequest):
    """Process a natural language query."""
    # This would integrate with the agent's NLP pipeline
    return {
        "query": request.query,
        "session_id": request.session_id or str(uuid.uuid4())[:8],
        "response": "Query processing will be implemented with full agent integration",
        "dry_run": request.dry_run
    }

# Background task functions
async def run_experiment_task(run_id: str, config: Dict[str, Any]):
    """Background task to run an experiment."""
    import time

    experiments_db[run_id]["status"] = "running"

    # Simulate experiment (replace with actual runner)
    time.sleep(2)

    experiments_db[run_id]["status"] = "completed"
    experiments_db[run_id]["metrics"] = {
        "simulation_time_seconds": 1.5,
        "fidelity": 0.995,
        "final_rank": 32
    }
    experiments_db[run_id]["completed_at"] = datetime.utcnow().isoformat()

async def run_batch_task(batch_id: str):
    """Background task to run a batch."""
    import time

    batch = batches_db[batch_id]
    batch["status"] = "running"

    for i, exp_config in enumerate(batch["experiments"]):
        # Simulate each experiment
        time.sleep(1)

        batch["results"].append({
            "config": exp_config,
            "status": "completed",
            "metrics": {"simulation_time_seconds": 1.0}
        })
        batch["progress"] = (i + 1) / len(batch["experiments"]) * 100

    batch["status"] = "completed"
    batch["completed_at"] = datetime.utcnow().isoformat()

# Run with: uvicorn agent.api.app:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 13.2 API Client

**File: `agent/api/client.py`**

```python
"""Python client for LRET AI Agent API."""
import requests
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time

@dataclass
class APIConfig:
    """API client configuration."""
    base_url: str = "http://localhost:8000"
    timeout: int = 30
    poll_interval: float = 1.0

class LRETClient:
    """Client for LRET AI Agent API."""

    def __init__(self, config: APIConfig = None):
        self.config = config or APIConfig()
        self.base_url = self.config.base_url.rstrip('/')

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request."""
        url = f"{self.base_url}{endpoint}"
        kwargs.setdefault('timeout', self.config.timeout)

        response = requests.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()

    def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        return self._request('GET', '/health')

    def run_experiment(self, n_qubits: int, depth: int = 20,
                      backend: str = "auto", **kwargs) -> Dict[str, Any]:
        """Submit and wait for experiment."""
        # Submit
        data = {"n_qubits": n_qubits, "depth": depth, "backend": backend, **kwargs}
        result = self._request('POST', '/experiments', json=data)

        # Poll for completion
        run_id = result["run_id"]
        while True:
            status = self.get_experiment(run_id)
            if status["status"] in ["completed", "failed"]:
                return status
            time.sleep(self.config.poll_interval)

    def submit_experiment(self, n_qubits: int, depth: int = 20,
                         backend: str = "auto", **kwargs) -> str:
        """Submit experiment without waiting."""
        data = {"n_qubits": n_qubits, "depth": depth, "backend": backend, **kwargs}
        result = self._request('POST', '/experiments', json=data)
        return result["run_id"]

    def get_experiment(self, run_id: str) -> Dict[str, Any]:
        """Get experiment status."""
        return self._request('GET', f'/experiments/{run_id}')

    def list_experiments(self, limit: int = 10, status: str = None) -> List[Dict]:
        """List experiments."""
        params = {"limit": limit}
        if status:
            params["status"] = status
        return self._request('GET', '/experiments', params=params)["experiments"]

    def run_batch(self, experiments: List[Dict[str, Any]],
                 parallel_mode: str = "threaded",
                 wait: bool = True) -> Dict[str, Any]:
        """Run batch of experiments."""
        data = {
            "experiments": experiments,
            "parallel_mode": parallel_mode,
            "max_workers": 4
        }
        result = self._request('POST', '/batches', json=data)

        if not wait:
            return result

        # Poll for completion
        batch_id = result["batch_id"]
        while True:
            status = self.get_batch(batch_id)
            if status["status"] in ["completed", "failed"]:
                return status
            time.sleep(self.config.poll_interval)

    def get_batch(self, batch_id: str) -> Dict[str, Any]:
        """Get batch status."""
        return self._request('GET', f'/batches/{batch_id}')

    def query(self, query: str, session_id: str = None,
             dry_run: bool = False) -> Dict[str, Any]:
        """Send natural language query."""
        data = {"query": query, "session_id": session_id, "dry_run": dry_run}
        return self._request('POST', '/query', json=data)

# Usage example
if __name__ == "__main__":
    client = LRETClient()

    # Check health
    print(client.health_check())

    # Run experiment
    result = client.run_experiment(n_qubits=10, depth=20)
    print(f"Experiment completed: {result}")

    # Run batch
    experiments = [
        {"n_qubits": 8, "depth": 20},
        {"n_qubits": 10, "depth": 20},
        {"n_qubits": 12, "depth": 20},
    ]
    batch_result = client.run_batch(experiments)
    print(f"Batch completed: {batch_result}")
```

---

## 📚 API Reference

### Endpoints

| Method | Endpoint                | Description            |
| ------ | ----------------------- | ---------------------- |
| GET    | `/`                     | API information        |
| GET    | `/health`               | Health check           |
| POST   | `/experiments`          | Submit experiment      |
| GET    | `/experiments/{run_id}` | Get experiment         |
| GET    | `/experiments`          | List experiments       |
| POST   | `/batches`              | Submit batch           |
| GET    | `/batches/{batch_id}`   | Get batch status       |
| POST   | `/query`                | Natural language query |

### Example Usage

```python
from agent.api.client import LRETClient

client = LRETClient()

# Run a single experiment
result = client.run_experiment(
    n_qubits=10,
    depth=20,
    backend="lret",
    noise_level=0.01
)
print(f"Fidelity: {result['metrics']['fidelity']}")

# Run parameter sweep
from agent.batch.sweep import ParameterSweep

configs = ParameterSweep.grid_sweep(
    {"mode": "hybrid"},
    {"n_qubits": [8, 10, 12], "depth": [20, 30]}
)

batch = client.run_batch(configs)
for result in batch["results"]:
    print(f"Config: {result['config']}, Time: {result['metrics']['simulation_time_seconds']}")
```

---

## Phase 11-13 Summary

**Completed Features:**

- ✅ Session management with persistence
- ✅ Experiment tracking and lineage
- ✅ Batch execution (sequential, threaded, multiprocess)
- ✅ Parameter sweep utilities
- ✅ REST API with FastAPI
- ✅ Python API client
- ✅ Background task processing

---

_Continue to [agent5.md](agent5.md) for Phases 14-16: Visualization, Cloud Integration, and ML Optimization._
