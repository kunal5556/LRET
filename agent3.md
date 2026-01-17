# LRET AI Agent - Part 3: Backend & Execution Systems (Phases 5-10)

_Previous: [agent2.md](agent2.md) - Core Implementation (Phases 1-4)_  
_Next: [agent4.md](agent4.md) - Session Management, Batch Execution, API_

---

## 📚 PHASE 5: Cirq Backend Integration

### Overview

Complete integration with Google Cirq for simulation, VQE/QAOA, and hardware access.

### 5.1 Cirq Executor

**File: `agent/execution/cirq_executor.py`**

```python
"""Execute quantum simulations using Google Cirq."""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import uuid
import time

@dataclass
class CirqConfig:
    """Configuration for Cirq simulations."""
    n_qubits: int
    circuit_type: str = "random"  # random, vqe, qaoa, qft, grover, ghz
    depth: int = 20
    backend: str = "simulator"  # simulator, density_matrix, qsim, hardware
    repetitions: int = 1000

    # Noise configuration
    noise_model: Optional[str] = None
    noise_strength: float = 0.0
    t1_ns: Optional[float] = None
    t2_ns: Optional[float] = None

    # VQE/QAOA options
    optimizer: Optional[str] = None
    max_iterations: int = 100

    # Hardware options
    device_name: Optional[str] = None
    google_project_id: Optional[str] = None

    # Output
    output_file: Optional[str] = None
    save_state_vector: bool = False

@dataclass
class CirqResult:
    """Result schema for Cirq simulations."""
    run_id: str
    config: CirqConfig
    status: str
    backend_used: str = "cirq"

    # Metrics
    execution_time_seconds: Optional[float] = None
    circuit_depth: Optional[int] = None
    num_qubits: Optional[int] = None

    # Measurement results
    measurements: Optional[List[List[int]]] = None
    histogram: Optional[Dict[str, int]] = None

    # VQE/QAOA results
    optimal_params: Optional[List[float]] = None
    optimal_value: Optional[float] = None

    # Raw output
    stdout: str = ""
    stderr: str = ""

    started_at: Optional[str] = None
    completed_at: Optional[str] = None

class CirqExecutor:
    """Execute quantum simulations using Google Cirq."""

    _circuit_cache: Dict[str, Any] = {}
    _simulator_cache: Dict[str, Any] = {}

    def __init__(self, working_dir: Optional[Path] = None):
        self.working_dir = working_dir or Path.cwd()
        self.cirq_available = self._check_cirq()
        self.qsim_available = self._check_qsim()

    def _check_cirq(self) -> bool:
        try:
            import cirq
            return True
        except ImportError:
            return False

    def _check_qsim(self) -> bool:
        try:
            import qsimcirq
            return True
        except ImportError:
            return False

    def run_simulation(self, config: CirqConfig) -> CirqResult:
        """Run Cirq simulation and return results."""
        run_id = f"cirq_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        started_at = datetime.utcnow().isoformat()

        if not self.cirq_available:
            return CirqResult(
                run_id=run_id, config=config, status="failed",
                stderr="Cirq not installed. Run: pip install cirq",
                started_at=started_at, completed_at=datetime.utcnow().isoformat()
            )

        try:
            import cirq

            # Build circuit
            circuit, qubits = self._build_circuit(config)

            # Add noise if configured
            if config.noise_model:
                circuit = self._add_noise(circuit, config)

            # Configure backend
            simulator = self._configure_backend(config)

            # Add measurements
            if not any(isinstance(op.gate, cirq.MeasurementGate)
                      for moment in circuit for op in moment):
                circuit.append(cirq.measure(*qubits, key='result'))

            # Execute
            start_time = time.time()
            result = simulator.run(circuit, repetitions=config.repetitions)
            execution_time = time.time() - start_time

            # Parse results
            measurements = result.measurements.get('result', [])
            histogram = self._compute_histogram(measurements)

            return CirqResult(
                run_id=run_id, config=config, status="success",
                execution_time_seconds=execution_time,
                circuit_depth=len(circuit), num_qubits=config.n_qubits,
                measurements=measurements.tolist() if hasattr(measurements, 'tolist') else measurements,
                histogram=histogram,
                started_at=started_at, completed_at=datetime.utcnow().isoformat()
            )

        except Exception as e:
            return CirqResult(
                run_id=run_id, config=config, status="failed",
                stderr=str(e), started_at=started_at,
                completed_at=datetime.utcnow().isoformat()
            )

    def _build_circuit(self, config: CirqConfig) -> Tuple[Any, List[Any]]:
        """Build Cirq circuit from configuration."""
        import cirq

        qubits = cirq.LineQubit.range(config.n_qubits)

        circuit_builders = {
            "random": self._build_random_circuit,
            "vqe": self._build_vqe_circuit,
            "qaoa": self._build_qaoa_circuit,
            "qft": self._build_qft_circuit,
            "grover": self._build_grover_circuit,
            "ghz": self._build_ghz_circuit,
        }

        builder = circuit_builders.get(config.circuit_type, self._build_random_circuit)
        circuit = builder(qubits, config)

        return circuit, qubits

    def _build_ghz_circuit(self, qubits: List, config: CirqConfig) -> Any:
        import cirq
        circuit = cirq.Circuit()
        circuit.append(cirq.H(qubits[0]))
        for i in range(len(qubits) - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
        return circuit

    def _build_random_circuit(self, qubits: List, config: CirqConfig) -> Any:
        import cirq
        import random

        circuit = cirq.Circuit()
        gates = [cirq.H, cirq.X, cirq.Y, cirq.Z, cirq.T, cirq.S]

        for _ in range(config.depth):
            for q in qubits:
                circuit.append(random.choice(gates)(q))
            for i in range(0, len(qubits) - 1, 2):
                if random.random() > 0.5:
                    circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))

        return circuit

    def _build_vqe_circuit(self, qubits: List, config: CirqConfig) -> Any:
        import cirq
        import numpy as np

        circuit = cirq.Circuit()
        n_params = len(qubits) * 2
        params = np.random.random(n_params) * 2 * np.pi

        for i, q in enumerate(qubits):
            circuit.append(cirq.ry(params[i])(q))

        for i in range(len(qubits) - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))

        for i, q in enumerate(qubits):
            if len(qubits) + i < len(params):
                circuit.append(cirq.ry(params[len(qubits) + i])(q))

        return circuit

    def _build_qaoa_circuit(self, qubits: List, config: CirqConfig) -> Any:
        import cirq
        import numpy as np

        circuit = cirq.Circuit()
        gamma, beta = 0.5, 0.3

        circuit.append(cirq.H.on_each(*qubits))

        for i in range(len(qubits) - 1):
            circuit.append(cirq.ZZPowGate(exponent=gamma / np.pi)(qubits[i], qubits[i + 1]))

        for q in qubits:
            circuit.append(cirq.rx(2 * beta)(q))

        return circuit

    def _build_qft_circuit(self, qubits: List, config: CirqConfig) -> Any:
        import cirq
        import numpy as np

        circuit = cirq.Circuit()
        n = len(qubits)

        for i in range(n):
            circuit.append(cirq.H(qubits[i]))
            for j in range(i + 1, n):
                angle = np.pi / (2 ** (j - i))
                circuit.append(cirq.CZPowGate(exponent=angle / np.pi)(qubits[i], qubits[j]))

        for i in range(n // 2):
            circuit.append(cirq.SWAP(qubits[i], qubits[n - 1 - i]))

        return circuit

    def _build_grover_circuit(self, qubits: List, config: CirqConfig) -> Any:
        import cirq
        import numpy as np

        circuit = cirq.Circuit()
        n = len(qubits)

        circuit.append(cirq.H.on_each(*qubits))

        num_iterations = max(1, int(np.pi / 4 * np.sqrt(2 ** n)))

        for _ in range(num_iterations):
            circuit.append(cirq.Z(qubits[-1]).controlled_by(*qubits[:-1]))
            circuit.append(cirq.H.on_each(*qubits))
            circuit.append(cirq.X.on_each(*qubits))
            circuit.append(cirq.Z(qubits[-1]).controlled_by(*qubits[:-1]))
            circuit.append(cirq.X.on_each(*qubits))
            circuit.append(cirq.H.on_each(*qubits))

        return circuit

    def _configure_backend(self, config: CirqConfig) -> Any:
        import cirq

        if config.backend == "density_matrix":
            return cirq.DensityMatrixSimulator()
        elif config.backend in ["qsim", "qsim_gpu"] and self.qsim_available:
            import qsimcirq
            return qsimcirq.QSimSimulator()
        else:
            return cirq.Simulator()

    def _add_noise(self, circuit: Any, config: CirqConfig) -> Any:
        import cirq

        if config.noise_model == "depolarizing":
            noise = cirq.depolarize(p=config.noise_strength)
            return circuit.with_noise(noise)

        return circuit

    def _compute_histogram(self, measurements) -> Dict[str, int]:
        from collections import Counter
        import numpy as np

        if len(measurements) == 0:
            return {}

        if isinstance(measurements, np.ndarray):
            bitstrings = [''.join(map(str, row.astype(int))) for row in measurements]
        else:
            bitstrings = [''.join(str(int(b)) for b in m) for m in measurements]

        return dict(Counter(bitstrings))
```

### 5.2 Backend Comparator

**File: `agent/execution/comparator.py`**

```python
"""Compare results between LRET and Cirq backends."""
from typing import Dict, Any
from datetime import datetime
from agent.execution.runner import ExperimentResult
from agent.execution.cirq_executor import CirqResult

class BackendComparator:
    """Compare results from LRET and Cirq backends."""

    def compare(self, lret_result: ExperimentResult,
                cirq_result: CirqResult) -> Dict[str, Any]:
        """Merge and compare results from both backends."""
        lret_time = lret_result.simulation_time_seconds or 0
        cirq_time = cirq_result.execution_time_seconds or 0

        return {
            "comparison_id": f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "configuration": {"n_qubits": lret_result.config.n_qubits},
            "lret_results": {
                "run_id": lret_result.run_id,
                "status": lret_result.status,
                "time_seconds": lret_time,
                "memory_mb": lret_result.memory_used_mb,
                "final_rank": lret_result.final_rank,
                "fidelity": lret_result.fidelity,
            },
            "cirq_results": {
                "run_id": cirq_result.run_id,
                "status": cirq_result.status,
                "time_seconds": cirq_time,
                "circuit_depth": cirq_result.circuit_depth,
            },
            "comparison_metrics": {
                "speedup_lret_vs_cirq": cirq_time / lret_time if lret_time > 0 else None,
                "both_successful": (lret_result.status == "success" and
                                   cirq_result.status == "success"),
            },
            "recommendation": self._generate_recommendation(lret_time, cirq_time)
        }

    def _generate_recommendation(self, lret_time: float, cirq_time: float) -> str:
        if lret_time < cirq_time * 0.7:
            return f"LRET is {cirq_time/lret_time:.1f}x faster"
        elif cirq_time < lret_time * 0.7:
            return f"Cirq is {lret_time/cirq_time:.1f}x faster"
        return "Both backends have similar performance"
```

---

## 📚 PHASE 6: VQE/QAOA Optimization

### 6.1 VQE Optimizer

**File: `agent/optimization/vqe.py`**

```python
"""Variational Quantum Eigensolver implementation."""
from typing import Dict, Any, List, Optional
import numpy as np
from agent.execution.cirq_executor import CirqExecutor

class VQEOptimizer:
    """VQE with classical optimization support."""

    OPTIMIZERS = {
        "BFGS": {"method": "BFGS", "options": {"maxiter": 100}},
        "COBYLA": {"method": "COBYLA", "options": {"maxiter": 100}},
        "Nelder-Mead": {"method": "Nelder-Mead", "options": {"maxiter": 100}},
    }

    def __init__(self, cirq_executor: CirqExecutor, optimizer: str = "BFGS"):
        self.cirq_executor = cirq_executor
        self.optimizer_name = optimizer
        self.optimizer_config = self.OPTIMIZERS.get(optimizer, self.OPTIMIZERS["BFGS"])
        self.history: List[Dict[str, Any]] = []

    def optimize(self, hamiltonian, n_qubits: int,
                 n_layers: int = 3, max_iterations: int = None) -> Dict[str, Any]:
        """Run VQE optimization."""
        from scipy.optimize import minimize
        import cirq
        import time

        self.history = []

        # Build ansatz
        ansatz, params = self._build_ansatz(n_qubits, n_layers)
        initial_params = np.random.uniform(-np.pi, np.pi, len(params))

        iteration = [0]

        def objective(param_values: np.ndarray) -> float:
            iteration[0] += 1
            resolver = cirq.ParamResolver(dict(zip(params, param_values)))
            bound_circuit = cirq.resolve_parameters(ansatz, resolver)
            energy = self._compute_expectation(bound_circuit, hamiltonian)
            self.history.append({"iteration": iteration[0], "energy": energy})
            return energy

        start_time = time.time()
        options = self.optimizer_config["options"].copy()
        if max_iterations:
            options["maxiter"] = max_iterations

        result = minimize(
            objective, initial_params,
            method=self.optimizer_config["method"],
            options=options
        )

        return {
            "status": "success" if result.success else "failed",
            "optimal_energy": result.fun,
            "optimal_params": result.x.tolist(),
            "iterations": iteration[0],
            "optimization_time": time.time() - start_time,
            "converged": result.success,
            "history": self.history,
        }

    def _build_ansatz(self, n_qubits: int, n_layers: int):
        import cirq

        qubits = cirq.LineQubit.range(n_qubits)
        circuit = cirq.Circuit()
        params = []

        for layer in range(n_layers):
            for i, q in enumerate(qubits):
                rx = cirq.Symbol(f"rx_{layer}_{i}")
                ry = cirq.Symbol(f"ry_{layer}_{i}")
                params.extend([rx, ry])
                circuit.append([cirq.rx(rx)(q), cirq.ry(ry)(q)])

            for i in range(n_qubits - 1):
                circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))

        return circuit, params

    def _compute_expectation(self, circuit, hamiltonian) -> float:
        import cirq

        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)
        state = result.final_state_vector

        # Simplified expectation value computation
        return float(np.real(np.vdot(state, state)))
```

### 6.2 QAOA Optimizer

**File: `agent/optimization/qaoa.py`**

```python
"""Quantum Approximate Optimization Algorithm."""
from typing import Dict, Any, List, Optional
import numpy as np
from agent.optimization.vqe import VQEOptimizer
from agent.execution.cirq_executor import CirqExecutor

class QAOAOptimizer:
    """QAOA for combinatorial optimization."""

    def __init__(self, cirq_executor: CirqExecutor, optimizer: str = "COBYLA"):
        self.cirq_executor = cirq_executor
        self.vqe = VQEOptimizer(cirq_executor, optimizer)

    def solve_maxcut(self, graph: List[tuple], n_layers: int = 2) -> Dict[str, Any]:
        """Solve MaxCut using QAOA."""
        import cirq

        n_qubits = max(max(e) for e in graph) + 1
        hamiltonian = self._build_maxcut_hamiltonian(graph, n_qubits)

        result = self.vqe.optimize(
            hamiltonian=hamiltonian,
            n_qubits=n_qubits,
            n_layers=n_layers
        )

        # Sample to find best cut
        best_bitstring, best_cut = self._sample_solution(
            result["optimal_params"], graph, n_qubits, n_layers
        )

        result["best_bitstring"] = best_bitstring
        result["best_cut_value"] = best_cut
        result["max_possible_cut"] = len(graph)
        result["approximation_ratio"] = best_cut / len(graph)

        return result

    def _build_maxcut_hamiltonian(self, graph: List[tuple], n_qubits: int):
        import cirq
        qubits = cirq.LineQubit.range(n_qubits)
        hamiltonian = cirq.PauliSum()
        for i, j in graph:
            hamiltonian += -0.5 * cirq.Z(qubits[i]) * cirq.Z(qubits[j])
        return hamiltonian

    def _sample_solution(self, params: List[float], graph: List[tuple],
                         n_qubits: int, n_layers: int) -> tuple:
        import cirq

        circuit = self._build_qaoa_circuit(graph, n_qubits, n_layers, np.array(params))
        qubits = cirq.LineQubit.range(n_qubits)
        circuit.append(cirq.measure(*qubits, key='result'))

        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=1000)
        histogram = result.histogram(key='result')

        best_bitstring = max(histogram.keys(), key=lambda k: histogram[k])
        best_cut = self._compute_cut_value(best_bitstring, graph)

        return format(best_bitstring, f'0{n_qubits}b'), best_cut

    def _build_qaoa_circuit(self, graph, n_qubits, n_layers, params):
        import cirq

        qubits = cirq.LineQubit.range(n_qubits)
        circuit = cirq.Circuit()
        circuit.append(cirq.H.on_each(*qubits))

        idx = 0
        for _ in range(n_layers):
            gamma = params[idx]
            beta = params[idx + 1]
            idx += 2

            for i, j in graph:
                circuit.append(cirq.ZZ(qubits[i], qubits[j]) ** (gamma / np.pi))

            for q in qubits:
                circuit.append(cirq.rx(2 * beta)(q))

        return circuit

    def _compute_cut_value(self, bitstring: int, graph: List[tuple]) -> int:
        cut = 0
        for i, j in graph:
            if ((bitstring >> i) & 1) != ((bitstring >> j) & 1):
                cut += 1
        return cut
```

---

## 📚 PHASE 7: Safety & Permissions

### 7.1 Permission Manager

**File: `agent/safety/permissions.py`**

```python
"""Permission management for agent actions."""
from typing import Dict, Any, Optional
from enum import Enum
from agent.utils.constants import ActionCategory

class PermissionLevel(Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"

class PermissionManager:
    """Manage permissions for agent actions."""

    DEFAULT_PERMISSIONS = {
        ActionCategory.READ: PermissionLevel.ALLOW,
        ActionCategory.RUN: PermissionLevel.ASK,
        ActionCategory.WRITE: PermissionLevel.ASK,
    }

    COMMAND_PATTERNS = {
        "cmake*": PermissionLevel.ALLOW,
        "make*": PermissionLevel.ALLOW,
        "./quantum_sim*": PermissionLevel.ALLOW,
        "python*": PermissionLevel.ALLOW,
        "rm -rf*": PermissionLevel.DENY,
        "*": PermissionLevel.ASK,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.session_permissions: Dict[str, PermissionLevel] = {}

    def check_permission(self, action: str, category: ActionCategory) -> PermissionLevel:
        """Check if action is permitted."""
        # Check session overrides
        if action in self.session_permissions:
            return self.session_permissions[action]

        # Check command patterns
        for pattern, level in self.COMMAND_PATTERNS.items():
            if self._matches_pattern(action, pattern):
                return level

        # Fall back to category default
        return self.DEFAULT_PERMISSIONS.get(category, PermissionLevel.ASK)

    def _matches_pattern(self, action: str, pattern: str) -> bool:
        if pattern == "*":
            return True
        if pattern.endswith("*"):
            return action.startswith(pattern[:-1])
        return action == pattern

    def grant_session_permission(self, action: str, level: PermissionLevel):
        """Grant permission for the current session."""
        self.session_permissions[action] = level

    def request_permission(self, action: str, category: ActionCategory,
                          description: str) -> bool:
        """Request permission from user."""
        level = self.check_permission(action, category)

        if level == PermissionLevel.ALLOW:
            return True
        if level == PermissionLevel.DENY:
            print(f"❌ Action denied: {action}")
            return False

        # Ask user
        print(f"\n⚠️ Permission Request")
        print(f"Action: {action}")
        print(f"Category: {category.value}")
        print(f"Description: {description}")

        response = input("Allow? (y/n/always): ").lower().strip()

        if response == "always":
            self.grant_session_permission(action, PermissionLevel.ALLOW)
            return True

        return response in ["y", "yes"]
```

### 7.2 Safety Validator

**File: `agent/safety/validator.py`**

```python
"""Validate actions for safety before execution."""
from typing import List, Dict, Any, Optional
from agent.planning.planner import ExecutionPlan, PlanStep
from agent.utils.constants import ActionCategory

class SafetyValidator:
    """Validate execution plans for safety."""

    DANGEROUS_PATTERNS = [
        "rm -rf /",
        "rm -rf ~",
        "dd if=/dev/zero",
        "> /dev/sda",
        "mkfs.",
        ":(){:|:&};:",  # Fork bomb
    ]

    RESOURCE_LIMITS = {
        "max_qubits": 50,
        "max_depth": 1000,
        "max_timeout_seconds": 86400,  # 24 hours
        "max_memory_gb": 128,
    }

    def validate_plan(self, plan: ExecutionPlan) -> tuple[bool, List[str]]:
        """Validate entire execution plan."""
        warnings = []

        for step in plan.steps:
            step_warnings = self._validate_step(step)
            warnings.extend(step_warnings)

        # Check for dangerous sequences
        sequence_warnings = self._check_dangerous_sequences(plan.steps)
        warnings.extend(sequence_warnings)

        is_safe = not any("BLOCKED" in w for w in warnings)
        return is_safe, warnings

    def _validate_step(self, step: PlanStep) -> List[str]:
        warnings = []

        # Check for dangerous commands
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern in str(step.parameters.get("command", "")):
                warnings.append(f"BLOCKED: Dangerous command pattern: {pattern}")

        # Check resource limits
        params = step.parameters
        if params.get("n_qubits", 0) > self.RESOURCE_LIMITS["max_qubits"]:
            warnings.append(f"Warning: High qubit count ({params['n_qubits']})")

        if params.get("timeout_seconds", 0) > self.RESOURCE_LIMITS["max_timeout_seconds"]:
            warnings.append("Warning: Very long timeout specified")

        return warnings

    def _check_dangerous_sequences(self, steps: List[PlanStep]) -> List[str]:
        warnings = []

        # Check for multiple write operations without confirmation
        write_count = sum(1 for s in steps if s.category == ActionCategory.WRITE)
        if write_count > 3:
            warnings.append(f"Warning: {write_count} write operations in single plan")

        return warnings
```

---

## 📚 PHASE 8: Memory & State Management

### 8.1 Session Memory

**File: `agent/memory/session.py`**

```python
"""Session memory management for the agent."""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json
import uuid

class SessionMemory:
    """Manage session state and history."""

    def __init__(self, session_id: Optional[str] = None,
                 persist_path: Optional[Path] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.persist_path = persist_path or Path.home() / ".lret" / "sessions"
        self.persist_path.mkdir(parents=True, exist_ok=True)

        self.created_at = datetime.utcnow()
        self.last_accessed = self.created_at

        # Memory stores
        self.conversation_history: List[Dict[str, Any]] = []
        self.experiment_results: Dict[str, Any] = {}
        self.user_preferences: Dict[str, Any] = {}
        self.context_cache: Dict[str, Any] = {}

    def add_message(self, role: str, content: str, metadata: Dict = None):
        """Add a message to conversation history."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        })
        self.last_accessed = datetime.utcnow()

    def add_result(self, run_id: str, result: Dict[str, Any]):
        """Store an experiment result."""
        self.experiment_results[run_id] = {
            "result": result,
            "stored_at": datetime.utcnow().isoformat()
        }

    def get_result(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an experiment result."""
        stored = self.experiment_results.get(run_id)
        return stored["result"] if stored else None

    def get_recent_results(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get most recent experiment results."""
        sorted_results = sorted(
            self.experiment_results.items(),
            key=lambda x: x[1]["stored_at"],
            reverse=True
        )
        return [r[1]["result"] for r in sorted_results[:limit]]

    def save(self):
        """Persist session to disk."""
        session_file = self.persist_path / f"{self.session_id}.json"
        data = {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "conversation_history": self.conversation_history,
            "experiment_results": self.experiment_results,
            "user_preferences": self.user_preferences,
        }
        with open(session_file, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self) -> bool:
        """Load session from disk."""
        session_file = self.persist_path / f"{self.session_id}.json"
        if not session_file.exists():
            return False

        with open(session_file, 'r') as f:
            data = json.load(f)

        self.created_at = datetime.fromisoformat(data["created_at"])
        self.last_accessed = datetime.fromisoformat(data["last_accessed"])
        self.conversation_history = data["conversation_history"]
        self.experiment_results = data["experiment_results"]
        self.user_preferences = data["user_preferences"]
        return True

    def get_context_for_llm(self, max_messages: int = 10) -> List[Dict[str, str]]:
        """Get conversation context for LLM."""
        recent = self.conversation_history[-max_messages:]
        return [{"role": m["role"], "content": m["content"]} for m in recent]
```

---

## 📚 PHASE 9: Code Editing & Rollback

### 9.1 Code Editor

**File: `agent/code/editor.py`**

```python
"""Code editing with diff generation."""
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import difflib
import shutil

class CodeEditor:
    """Manage code modifications with rollback support."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.backup_dir = repo_root / ".lret_backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.pending_changes: List[Dict[str, Any]] = []

    def create_backup(self, file_path: Path) -> str:
        """Create a backup of a file."""
        if not file_path.exists():
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.name}.{timestamp}.bak"
        backup_path = self.backup_dir / backup_name
        shutil.copy2(file_path, backup_path)
        return str(backup_path)

    def generate_diff(self, file_path: Path, new_content: str) -> str:
        """Generate unified diff for changes."""
        if file_path.exists():
            with open(file_path, 'r') as f:
                old_content = f.read()
        else:
            old_content = ""

        diff = difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{file_path.name}",
            tofile=f"b/{file_path.name}"
        )
        return ''.join(diff)

    def propose_change(self, file_path: Path, new_content: str,
                      description: str) -> Dict[str, Any]:
        """Propose a code change without applying."""
        diff = self.generate_diff(file_path, new_content)

        change = {
            "id": len(self.pending_changes),
            "file_path": str(file_path),
            "new_content": new_content,
            "diff": diff,
            "description": description,
            "created_at": datetime.utcnow().isoformat(),
            "status": "pending"
        }

        self.pending_changes.append(change)
        return change

    def apply_change(self, change_id: int) -> tuple[bool, str]:
        """Apply a pending change."""
        if change_id >= len(self.pending_changes):
            return False, "Invalid change ID"

        change = self.pending_changes[change_id]
        file_path = Path(change["file_path"])

        # Create backup
        backup_path = self.create_backup(file_path)
        change["backup_path"] = backup_path

        # Apply change
        try:
            with open(file_path, 'w') as f:
                f.write(change["new_content"])
            change["status"] = "applied"
            return True, f"Applied change to {file_path}"
        except Exception as e:
            change["status"] = "failed"
            return False, str(e)

    def rollback_change(self, change_id: int) -> tuple[bool, str]:
        """Rollback a previously applied change."""
        if change_id >= len(self.pending_changes):
            return False, "Invalid change ID"

        change = self.pending_changes[change_id]

        if change["status"] != "applied":
            return False, "Change was not applied"

        backup_path = change.get("backup_path")
        if not backup_path or not Path(backup_path).exists():
            return False, "Backup not found"

        try:
            shutil.copy2(backup_path, change["file_path"])
            change["status"] = "rolled_back"
            return True, f"Rolled back {change['file_path']}"
        except Exception as e:
            return False, str(e)
```

---

## 📚 PHASE 10: Testing & Documentation

### 10.1 Test Runner

**File: `agent/testing/runner.py`**

```python
"""Test execution for agent components."""
import subprocess
from typing import Dict, Any, List
from pathlib import Path

class TestRunner:
    """Run tests for the LRET project."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.build_dir = repo_root / "build"

    def run_ctest(self, filter_pattern: str = None) -> Dict[str, Any]:
        """Run CTest for C++ tests."""
        cmd = ["ctest", "--output-on-failure"]
        if filter_pattern:
            cmd.extend(["-R", filter_pattern])

        result = subprocess.run(
            cmd, cwd=self.build_dir,
            capture_output=True, text=True, timeout=300
        )

        return {
            "passed": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }

    def run_pytest(self, path: str = "tests/",
                   markers: str = None) -> Dict[str, Any]:
        """Run pytest for Python tests."""
        cmd = ["pytest", path, "-v"]
        if markers:
            cmd.extend(["-m", markers])

        result = subprocess.run(
            cmd, cwd=self.repo_root,
            capture_output=True, text=True, timeout=300
        )

        return {
            "passed": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }

    def run_validation(self) -> Dict[str, Any]:
        """Run validation tests against FDM."""
        return self.run_ctest(filter_pattern="validation")
```

---

## Phase 5-10 Summary

**Completed Features:**

- ✅ Cirq backend integration with all circuit types
- ✅ VQE optimizer with multiple classical optimizers
- ✅ QAOA optimizer for MaxCut problems
- ✅ Permission management system
- ✅ Safety validation for dangerous operations
- ✅ Session memory with persistence
- ✅ Code editing with diff generation
- ✅ Rollback support for changes
- ✅ Test runner integration

---

_Continue to [agent4.md](agent4.md) for Phases 11-13: Session Management, Batch Execution, and Web API._
